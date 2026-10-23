"""Analysis transformations edit committed profiles before correlations."""

from dataclasses import replace
from concurrent.futures import CancelledError
import io

import numpy as np
import pytest

from app.pages import plotting as page
from dnascape.workflows.analysis import SmoothingSettings
from dnascape.workflows.filtering import DataFilterSettings
from dnascape.workflows.modulation import ProfileModulationSettings
from dnascape.workflows.uploads import inspect_upload


def source(number, payload, name="signal.csv"):
    state = page._ProfileSourceState(number, inspect_upload(name, payload, expected_form="auto"))
    page._set_default_selection(state)
    return state


def prepare(sources, data_filter, *, target=None, keep=False, methods=None, smoothing=None, cancel=lambda: None):
    return page._prepare_analysis_profiles(sources, smoothing or {}, methods, cancel,
                                         page._TransformationSettings(data_filter, target, keep))


def test_interpolation_uses_coordinates_and_correlations_use_transformed_values():
    first = source(1, b"position,A\n0,1\n1,nan\n3,7\n6,13\n")
    second = source(2, b"position,B\n0,1\n1,3\n3,7\n6,13\n", "reference.csv")
    settings = DataFilterSettings(enabled=True, selection="nan", action="interpolate")
    selected, report = prepare([first, second], settings, target=(1, "A"), methods=["pearson"])
    np.testing.assert_equal(selected[0].values, [1, 3, 7, 13])
    assert selected[0].transformed and not selected[1].transformed
    assert report.matrices[0].sample_counts[0, 1] == 4
    assert report.matrices[0].coefficients[0, 1] == pytest.approx(1)
    np.testing.assert_equal(page._materialize_source(first, missing_policy="ignore")[0].values, [1, np.nan, 7, 13])


def test_keep_originals_has_stable_distinct_ids_and_never_compounds_transforms():
    original = source(1, b"A\n1\n2\n3\n")
    settings = DataFilterSettings(enabled=True, selection="all", action="multiply", value=3)
    first, _ = prepare([original], settings, keep=True)
    second, _ = prepare([original], settings, keep=True)
    assert [item.series_id for item in first] == [item.series_id for item in second]
    assert len({item.series_id for item in first}) == 2
    assert [item.name for item in first] == ["A", "A (transformed)"]
    np.testing.assert_equal(first[0].values, [1, 2, 3])
    np.testing.assert_equal(first[1].values, [3, 6, 9])
    np.testing.assert_equal(second[1].values, first[1].values)


def test_modulation_is_checked_only_for_targeted_profiles_and_preserves_other_shapes():
    first = source(1, b"A\n1\n2\n3\n")
    second = source(2, b"B\n10\n20\n", "second.csv")
    settings = DataFilterSettings(enabled=True, selection="all", action="modulate",
                                  modulation=ProfileModulationSettings(operation="formula", combination="sum", profile_multiplier=-2, values=np.ones(3)))
    selected, _ = prepare([first, second], settings, target=(1, "A"))
    np.testing.assert_equal(selected[0].values, [-1, 0, 1])
    np.testing.assert_equal(selected[1].values, [10, 20])
    with pytest.raises(ValueError, match="does not match"):
        prepare([first, second], settings)
    with pytest.raises(ValueError, match="available dataset"):
        prepare([first], settings, target=(9, "removed"))


def test_transformations_follow_smoothing_and_cancel_without_mutating_inputs():
    first = source(1, b"A\n1\n5\n3\n")
    settings = DataFilterSettings(enabled=True, selection="all", action="multiply", value=2)
    result, _ = prepare([first], settings, smoothing={(1, "A"): SmoothingSettings("moving_average", window=3)})
    np.testing.assert_equal(result[0].values, [6, 6, 8])
    disabled, _ = prepare([first], replace(settings, enabled=False))
    np.testing.assert_equal(disabled[0].values, [1, 5, 3])
    assert not disabled[0].transformed
    def stop():
        raise CancelledError()
    with pytest.raises(CancelledError):
        prepare([first], settings, cancel=stop)


@pytest.mark.parametrize("values,expected,form", [
    ([1, np.nan, 5], [1, 3, 5], "x"),
    ([[1, np.nan, 5]], [[1, 3, 5]], "t"),
    ([[1, 2], [np.nan, np.nan], [5, 8]], [[1, 2], [3, 5], [5, 8]], "xt"),
])
def test_interpolation_preserves_space_time_and_matrix_axes(values, expected, form):
    payload = io.BytesIO()
    np.savez_compressed(payload, values=np.asarray(values))
    original = source(1, payload.getvalue(), "source.npz")
    selected, _ = prepare([original], DataFilterSettings(enabled=True, selection="nan", action="interpolate"))
    assert selected[0].form == form
    np.testing.assert_equal(selected[0].values, expected)
    np.testing.assert_equal(page._materialize_source(original, missing_policy="ignore")[0].values, values)
