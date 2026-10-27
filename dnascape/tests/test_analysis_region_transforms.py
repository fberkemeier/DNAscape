"""Formula/scaling, masks and exported Analysis recipes share one pipeline."""
from dataclasses import replace
import io

import numpy as np
import pytest

from dnascape.workflows.analysis import AnalysisDataset, SmoothingSettings, analyse_profiles
from dnascape.workflows.analysis_transforms import analysis_values, transform_profiles, analysis_mask_artifact, scale_analysis_values
from dnascape.workflows.filtering import DataFilterSettings
from dnascape.workflows.uploads import inspect_upload


def profile(values, name="A", identity=1, form="x"):
    payload = io.BytesIO()
    np.savez(payload, values=np.asarray(values), data_form={"x": "space", "t": "time", "xt": "spacetime"}[form])
    return dict(id=(identity, name), name=name, form=form, values=np.asarray(values), source=inspect_upload(name + ".npz", payload.getvalue()))


def test_formula_then_smoothing_scaling_then_region_transform_are_dataset_specific():
    first, second = profile([1., 5., 3.]), profile([10., 20., 30.], "B", 2)
    data = dict(enabled=True, target=(1, "A"), formulas={"1:A": "source * 2"},
                values={"1:A": dict(smoothing=SmoothingSettings("moving_average", 3), scaling="interval", minimum=0., maximum=10., invert=True)})
    region = dict(settings=DataFilterSettings(enabled=True, selection="threshold", comparison="above", threshold=5, action="replace", value=7), target=(1, "A"))
    result = transform_profiles([first, second], data_transforms=data, region_transforms=region)
    np.testing.assert_allclose(result[0]["values"], [7, 7, 0])
    np.testing.assert_equal(result[1]["values"], second["values"])
    np.testing.assert_equal(first["values"], [1, 5, 3])
    assert result[0]["transformed"] and not result[1]["transformed"]


def test_combine_has_one_profile_and_region_target_can_select_it():
    first, second = profile([1., 2., 3.]), profile([4., 5., 6.], "B", 2)
    result = transform_profiles([first, second], data_transforms=dict(enabled=True, combine=True, combined_formula="source1 + source2 * 2"),
        region_transforms=dict(settings=DataFilterSettings(enabled=True, selection="all", action="multiply", value=2), target=(1, "__combined__")))
    assert len(result) == 1 and result[0]["id"] == (1, "__combined__")
    np.testing.assert_equal(result[0]["values"], [18, 24, 30])
    with pytest.raises(ValueError, match="matching shapes"):
        transform_profiles([first, profile([1., 2.], "B", 2)], data_transforms=dict(enabled=True, combine=True))


@pytest.mark.parametrize("form", ["x", "t", "xt"])
def test_mask_only_preserves_profiles_and_pair_indices_across_dependences(form):
    a, b = np.array([-2., 8., -1., 3.]), np.array([2., 0., 1., 50.])
    if form == "t":
        a, b = a[None, :], b[None, :]
    elif form == "xt":
        a, b = np.column_stack([a, a]), np.column_stack([b, b])
    originals = [profile(a, form=form), profile(b, "B", 2, form)]
    settings = DataFilterSettings(enabled=True, selection="threshold", threshold=0., action="multiply", value=2.)
    result = transform_profiles(originals, region_transforms=dict(settings=settings, target=(1, "A"), analysis_on_mask_only=True))
    assert np.asarray(result[0]["values"]).shape == a.shape
    np.testing.assert_equal(result[0]["values"], np.where(a < 0, a*2, a))
    report = analyse_profiles([AnalysisDataset(item["name"], analysis_values(item), item["form"]) for item in result], methods=["pearson"])
    assert report.matrices[0].sample_counts[0, 1] == (4 if form == "xt" else 2)
    assert report.matrices[0].coefficients[0, 1] == pytest.approx(-1.)
    np.testing.assert_equal(originals[0]["values"], a)
    inactive = transform_profiles(originals, region_transforms=dict(settings=replace(settings, enabled=False), analysis_on_mask_only=True))
    np.testing.assert_equal(analysis_values(inactive[0]), a)


def test_empty_and_inverted_masks_and_native_mask_exports():
    original = profile([-1., 2., -3., 4.])
    settings = DataFilterSettings(enabled=True, selection="threshold", threshold=0., action="multiply", value=1.)
    result = transform_profiles([original], region_transforms=dict(settings=settings, analysis_on_mask_only=True))[0]
    positions = analysis_mask_artifact(result, filename="mask").data
    assert positions == b"0\n2\n"
    assert analysis_mask_artifact(result, filename="mask", layout="intervals").data == b"0\t1\n2\t3\n"
    inverted = transform_profiles([original], region_transforms=dict(settings=replace(settings, complement=True), analysis_on_mask_only=True))[0]
    np.testing.assert_equal(analysis_values(inverted), [np.nan, 2, np.nan, 4])
    empty = transform_profiles([original], region_transforms=dict(settings=replace(settings, threshold=-10), analysis_on_mask_only=True))[0]
    assert np.isnan(analysis_values(empty)).all()
    assert analysis_mask_artifact(empty, filename="empty").data == b""


@pytest.mark.parametrize("mode,expected", [
    ("none", [1, 2, 3]), ("interval", [0, .5, 1]), ("multiplier", [2, 4, 6]),
    ("center", [-1, 0, 1]), ("zscore", [-np.sqrt(1.5), 0, np.sqrt(1.5)]), ("maxabs", [1/3, 2/3, 1]),
])
def test_scaling_uses_data_manager_conventions_and_preserves_nan(mode, expected):
    np.testing.assert_allclose(scale_analysis_values([1., 2., np.nan, 3.], dict(scaling=mode, multiplier=2)),
                               [expected[0], expected[1], np.nan, expected[2]])


def test_mask_exports_use_genomic_coordinates_and_merge_intervals():
    source = inspect_upload("a.tsv", b"position_bp\tA\n1000\t-1\n2000\t-2\n3000\t3\n")
    item = dict(id=(1,"A"), name="A", form="x", values=np.array([-1., -2., 3.]), source=source)
    result = transform_profiles([item], region_transforms=dict(settings=DataFilterSettings(enabled=True)))[0]
    assert analysis_mask_artifact(result, filename="mask").data == b"1000\n2000\n"
    assert analysis_mask_artifact(result, filename="mask", layout="intervals").data == b"1000\t3000\n"
