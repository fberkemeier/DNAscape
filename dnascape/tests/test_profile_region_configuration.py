"""Reusable numeric exports retain genomic configuration before transforms and filters."""

import io
from dataclasses import replace

import numpy as np
import pytest

from dnascape.workflows.conversion import (
    DataConversionRequest,
    convert_data,
    filter_positions_artifact,
    prepare_data,
)
from dnascape.workflows.filtering import DataFilterSettings, apply_data_filter
from dnascape.workflows.modulation import ModulationProfile, ProfileModulationSettings
from dnascape.workflows.uploads import inspect_upload


@pytest.mark.parametrize("format", ["tsv", "txt", "csv", "npz"])
def test_native_exports_crop_rebin_then_transform_filter_and_export_positions(format):
    buffer = io.BytesIO()
    if format == "npz":
        np.savez(buffer, values=np.arange(1., 7.), data_form="space")
        payload = buffer.getvalue()
    else:
        payload = ("signal\n" if format != "txt" else "").encode() + b"1\n2\n3\n4\n5\n6\n"
    request = DataConversionRequest(
        inspect_upload(f"native.{format}", payload), "x", profile_binning=True,
        start=1000, end=5000, bin_size=2000, target_format="npz",
        transform_formula="source * 2",
        data_filter=DataFilterSettings(enabled=True, selection="threshold", comparison="above", threshold=8, value=0),
    )
    prepared = prepare_data(request)
    np.testing.assert_equal(prepared.values, [5, 0])
    np.testing.assert_equal(prepared.coordinates["position_kb"], [1, 3])
    result = convert_data(request)
    np.testing.assert_equal(result.preview.x, [0, 2])
    with np.load(io.BytesIO(result.artifact.data)) as archive:
        np.testing.assert_equal(archive["values"], [5, 0])
    assert filter_positions_artifact(result, output_name="filter").data.strip() == b"3000"


def test_supplied_positions_and_chromosome_headers_override_implicit_source_grid():
    inspection = inspect_upload("chromosomes.tsv", b"position_bp\tchr1\tchr2\n10000\t1\t10\n11000\t3\t30\n12000\t5\t50\n13000\t7\t70\n")
    request = DataConversionRequest(inspection, "x", value_column="chr2", profile_binning=True,
                                    start=11000, end=14000, bin_size=2000, source_start=500000)
    result = prepare_data(request)
    np.testing.assert_equal(result.values, [40, 70])
    np.testing.assert_equal(result.coordinates["position_kb"], [11, 13])
    assert result.labels == ("chr2",)
    with pytest.raises(ValueError, match="beyond"):
        prepare_data(replace(request, end=15000))


def test_matrix_partial_bins_and_finer_resolution_keep_columns_independent():
    buffer = io.BytesIO()
    np.savez(buffer, values=np.array([[2., 20.], [6., 60.]]), data_form="spacetime")
    request = DataConversionRequest(inspect_upload("matrix.npz", buffer.getvalue()), "xt",
                                    profile_binning=True, start=500, end=2000, bin_size=500)
    result = prepare_data(request)
    np.testing.assert_equal(result.values, [[2, 20], [6, 60], [6, 60]])
    np.testing.assert_equal(result.coordinates["position_kb"], [.5, 1, 1.5])


@pytest.mark.parametrize("shape,profile,expected", [
    ((4,), [10, 20], [10, 40, 3, 4]),
    ((2,), [10, 20, 30], [10, 40]),
    ((4, 1), [[10, 20]], [[10], [40], [3], [4]]),
    ((2, 3), [[10, 20], [30, 40], [50, 60]], [[10, 40, 3], [120, 200, 6]]),
])
def test_modulation_mismatch_preserves_unmatched_source_and_warns(shape, profile, expected):
    buffer = io.BytesIO()
    np.savez(buffer, values=np.arange(1., np.prod(shape) + 1).reshape(shape))
    settings = DataFilterSettings(enabled=True, selection="all", action="modulate",
        modulation=ProfileModulationSettings(operation="expression", values=np.asarray(profile), align_from_start=True))
    result = convert_data(DataConversionRequest(inspect_upload("source.npz", buffer.getvalue()),
                                                "xt" if len(shape) == 2 and min(shape) > 1 else "x", target_format="npz", data_filter=settings))
    with np.load(io.BytesIO(result.artifact.data)) as archive:
        np.testing.assert_equal(archive["values"], np.asarray(expected).reshape(archive["values"].shape))
    assert len(result.warnings) == 1
    assert "first sample" in result.warnings[0]


def test_modulation_coordinate_mismatch_warns_even_when_lengths_match():
    source = inspect_upload("source.tsv", b"position_kb\tsignal\n10\t2\n20\t3\n")
    settings = DataFilterSettings(enabled=True, selection="all", action="modulate",
        modulation=ProfileModulationSettings(operation="expression", align_from_start=True,
            profiles={None: ModulationProfile(np.array([4., 5.]), np.array([0., 1.]))}))
    result = convert_data(DataConversionRequest(source, "x", value_column="signal", data_filter=settings))
    np.testing.assert_equal(result.preview.values, [8, 15])
    assert len(result.warnings) == 1


def test_row_chromosomes_are_selected_before_coordinates_are_validated_or_rebinned():
    source = inspect_upload("regions.csv", b"chromosome,position_bp,signal\nchr1,0,1\nchr1,1000,3\nchr2,10000,20\nchr2,11000,40\n")
    result = prepare_data(DataConversionRequest(source, "x", value_column="signal", chromosome="chr2",
                                                profile_binning=True, start=10000, end=12000, bin_size=2000))
    np.testing.assert_equal(result.values, [30])
    np.testing.assert_equal(result.coordinates["position_kb"], [10])


def test_mismatch_formula_reductions_use_the_overlap_without_nan_padding():
    warnings = []
    settings = DataFilterSettings(enabled=True, selection="all", action="modulate", _warning_sink=warnings.append,
        modulation=ProfileModulationSettings(operation="expression", expression="source * profile / np.mean(profile)",
                                             values=np.array([1., 3.]), align_from_start=True))
    np.testing.assert_equal(apply_data_filter([2, 4, 8], settings), [1, 6, 8])
    assert len(warnings) == 1
