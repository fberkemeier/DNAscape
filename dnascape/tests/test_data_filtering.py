"""Filters edit converted bins without moving coordinates or modifying uploads."""

import gzip
import io
from concurrent.futures import CancelledError
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from dnascape.workflows.conversion import DataConversionRequest, convert_data, prepare_data, transcode_conversion_artifact
from dnascape.workflows.filtering import (
    DataFilterSettings, FilterProfile, apply_data_filter, read_filter_profile, resolve_filter,
)
from dnascape.workflows.uploads import inspect_upload


def settings(**kwargs):
    return DataFilterSettings(enabled=True, **kwargs)


def test_defaults_are_inert_and_enabled_defaults_replace_only_negative_bins():
    values = np.array([3., -2., np.nan, -1., 0.])
    disabled = DataFilterSettings(threshold="unfinished", action="unknown")
    np.testing.assert_equal(apply_data_filter(values, disabled), values)
    np.testing.assert_equal(apply_data_filter(values, settings()), [3, 0, np.nan, 0, 0])
    np.testing.assert_equal(values, [3, -2, np.nan, -1, 0])
    np.testing.assert_equal(apply_data_filter(values, settings(selection="profile")), values)
    np.testing.assert_equal(apply_data_filter(values, settings(selection="profile", value=None)), values)


@pytest.mark.parametrize("options,expected", [
    (dict(selection="threshold", comparison="above", threshold=5), [0, 4, 5, 99, 99]),
    (dict(selection="between", lower=4, upper=8), [0, 99, 99, 99, 10]),
    (dict(selection="minimum", comparison="below", threshold=5), [99, 99, 5, 8, 10]),
    (dict(selection="maximum", comparison="above", threshold=3), [0, 4, 5, 99, 99]),
    (dict(selection="all"), [99] * 5),
])
def test_signal_selection_uses_the_unedited_converted_values(options, expected):
    np.testing.assert_equal(apply_data_filter([0, 4, 5, 8, 10], settings(value=99, **options)), expected)


def test_means_are_per_contiguous_region_and_per_time_slice():
    values = np.array([[10, 100], [-2, -20], [-4, -40], [20, 200], [-8, -80], [30, 300]])
    result = apply_data_filter(values, settings(action="mean"))
    np.testing.assert_equal(result, [[10, 100], [-3, -30], [-3, -30], [20, 200], [-8, -80], [30, 300]])


@pytest.mark.parametrize("method,expected", [
    ("linear", [10, 15, 20, 30]), ("nearest", [10, 10, 10, 30]),
    ("previous", [10, 10, 10, 30]), ("next", [10, 30, 30, 30]),
    ("pchip", [10, 15, 20, 30]),
])
def test_interpolation_bridges_region_endpoints_using_actual_coordinates(method, expected):
    result = apply_data_filter([10, -2, -4, 30], settings(action="interpolate", interpolation=method), positions=[0, 1, 2, 4])
    np.testing.assert_allclose(result, expected)


def test_edge_regions_use_available_endpoints_and_whole_regions_report_no_anchor():
    np.testing.assert_equal(apply_data_filter([-1, 10, -2, 30, -3], settings(action="interpolate")), [10, 10, 20, 30, 30])
    with pytest.raises(ValueError, match="outside the selected regions"):
        apply_data_filter([1, 2], settings(selection="all", action="interpolate"))


@pytest.mark.parametrize("action,reference,expected", [
    ("minimum", "outside", [4, 4, 4, 8]), ("minimum", "all", [4, -2, -2, 8]),
    ("maximum", "outside", [4, 8, 8, 8]), ("maximum", "all", [4, 8, 8, 8]),
    ("add", "outside", [4, 0, 1, 8]), ("multiply", "outside", [4, -4, -2, 8]),
])
def test_reference_and_arithmetic_operations(action, reference, expected):
    np.testing.assert_equal(apply_data_filter([4, -2, -1, 8], settings(action=action, reference=reference, value=2)), expected)


def test_nan_is_an_explicit_selection_and_replacement_not_a_dropped_bin():
    np.testing.assert_equal(apply_data_filter([1, np.nan, 3], settings(selection="nan", value=7)), [1, 7, 3])
    np.testing.assert_equal(apply_data_filter([1, -2, 3], settings(action="nan")), [1, np.nan, 3])
    with pytest.raises(ValueError, match="no finite values to average"):
        apply_data_filter([1, np.nan, 3], settings(selection="nan", action="mean"))
    with pytest.raises(ValueError, match="No finite reference"):
        apply_data_filter([1, 2], settings(selection="all", action="minimum"))


def test_noise_has_requested_scale_reproducibility_and_independent_chromosomes():
    values = np.ones((20_000, 2))
    config = settings(selection="all", action="noise", standard_deviation=2, seed=42)
    first = apply_data_filter(values, config, chromosome="chr1")
    np.testing.assert_equal(first, apply_data_filter(values, config, chromosome="chr1"))
    assert not np.array_equal(first, apply_data_filter(values, config, chromosome="chr2"))
    assert not np.array_equal(first[:, 0], first[:, 1])
    assert abs((first - values).mean()) < .04
    assert (first - values).std() == pytest.approx(2, rel=.02)


def test_uploaded_masks_are_bin_aligned_and_never_silently_resampled():
    upload = inspect_upload("mask.tsv", b"position_kb\tmask\n0\t0\n1\t1\n2\t0\n3\t1\n")
    config = resolve_filter(settings(selection="profile", upload=upload, value=7))
    np.testing.assert_equal(apply_data_filter([1, 2, 3, 4], config), [1, 7, 3, 7])
    with pytest.raises(ValueError, match="one value for each converted bin"):
        apply_data_filter([1, 2, 3], config)


@pytest.mark.parametrize("format_name", ["tsv.gz", "npz", "npy"])
@pytest.mark.parametrize("layout", ["mask", "positions", "intervals"])
def test_uploaded_filter_formats_select_the_same_converted_bins(format_name, layout):
    values = np.array([0., 1., 0.]) if layout == "mask" else np.array([1000.]) if layout == "positions" else np.array([[1000., 2000.]])
    stream = io.BytesIO()
    if format_name == "npz":
        np.savez(stream, values=values)
        payload = stream.getvalue()
    elif format_name == "npy":
        np.save(stream, values)
        payload = stream.getvalue()
    else:
        text = "start\tend\n1000\t2000\n" if layout == "intervals" else "mask\n0\n1\n0\n" if layout == "mask" else "position_bp\n1000\n"
        payload = gzip.compress(text.encode())
    config = settings(selection="profile", upload=inspect_upload(f"filter.{format_name}", payload, expected_form="regions"), profile_layout=layout, value=7)
    np.testing.assert_equal(apply_data_filter([1, 2, 3], config, positions=[.5, 1.5, 2.5], intervals=np.array([[0, 1], [1, 2], [2, 3]])), [1, 7, 3])


@pytest.mark.parametrize("layout,values,expected", [
    ("positions", [999, 1000, 3000], [7, 7, 3]),
    ("intervals", [[1000, 2000], [1500, 2500]], [1, 7, 7]),
])
def test_genomic_positions_and_overlapping_intervals_use_half_open_converted_bins(layout, values, expected):
    config = settings(selection="profile", profile=FilterProfile(np.array(values), layout, "bp"), value=7)
    result = apply_data_filter([1, 2, 3], config, positions=[.5, 1.5, 2.5], intervals=np.array([[0, 1], [1, 2], [2, 3]]))
    np.testing.assert_equal(result, expected)


def test_sparse_positions_do_not_select_neighbouring_samples_on_an_unbinned_grid():
    profile = FilterProfile(np.array([1000., 1500.]), "positions", "bp")
    config = settings(selection="profile", profile=profile, value=7)
    np.testing.assert_equal(apply_data_filter([1, 2, 3], config, positions=[0, 1, 2]), [1, 7, 3])
    with pytest.raises(ValueError, match="no genomic coordinates"):
        apply_data_filter([1, 2, 3], config)
    np.testing.assert_equal(apply_data_filter([1, 2, 3], replace(config, profile=FilterProfile(np.array([1.]), "positions", "index"))), [1, 7, 3])


def test_bed_filter_respects_chromosomes_and_does_not_cross_boundaries():
    upload = inspect_upload("regions.bed", b"chr1\t1000\t2000\n")
    config = settings(selection="profile", upload=upload, profile_layout="intervals", action="add", value=10)
    source = inspect_upload("signal.bedgraph", b"chr1\t0\t1000\t1\nchr1\t1000\t2000\t2\nchr2\t0\t2000\t3\n")
    request = DataConversionRequest(inspection=source, input_form="track", bin_size=1000, chromosome_scope="all", data_filter=config, target_format="npz")
    prepared = prepare_data(request)
    np.testing.assert_equal(prepared.profile_values["chr1"], [1, 12])
    np.testing.assert_equal(prepared.profile_values["chr2"], [3, 3])
    with np.load(io.BytesIO(convert_data(request).artifact.data)) as archive:
        np.testing.assert_equal(archive["chr1"], [1, 12])
        np.testing.assert_equal(archive["chr2"], [3, 3])
    assert dict(prepared.metadata)["filter_action"] == "add"


@pytest.mark.parametrize("format_name", ["tsv", "csv", "txt", "npz"])
@pytest.mark.parametrize("compressed", [False, True])
def test_configure_precedes_filter_and_every_export_and_preview_uses_filtered_values(format_name, compressed):
    upload = inspect_upload("signal.csv", b"position,value\n0,-2\n1,1\n2,3\n")
    request = DataConversionRequest(inspection=upload, input_form="x", target_format=format_name, gzip_output=compressed,
                                    scaling_mode="multiplier", scale=-2, data_filter=settings(action="nan"))
    result = convert_data(request)
    payload = gzip.decompress(result.artifact.data) if compressed else result.artifact.data
    if format_name == "npz":
        with np.load(io.BytesIO(payload)) as archive:
            values = archive["values"].copy()
    else:
        table = pd.read_csv(io.BytesIO(payload), sep="," if format_name == "csv" else r"\s+", header=None if format_name == "txt" else 0, skip_blank_lines=False)
        values = table.iloc[:, -1].to_numpy(dtype=float)
    np.testing.assert_equal(values, [4, np.nan, np.nan])
    np.testing.assert_equal(prepare_data(request).values, [4, np.nan, np.nan])
    np.testing.assert_equal(result.preview.values, [4, np.nan, np.nan])
    assert upload.content == b"position,value\n0,-2\n1,1\n2,3\n"


@pytest.mark.parametrize("format_name", ["tsv", "csv", "txt", "npz"])
@pytest.mark.parametrize("all_missing", [False, True])
def test_generated_file_format_changes_preserve_filter_created_nan_bins(format_name, all_missing):
    result = convert_data(DataConversionRequest(
        inspection=inspect_upload("source.tsv", b"value\n1\n-2\n3\n"), input_form="x",
        data_filter=settings(selection="all" if all_missing else "threshold", action="nan"),
    ))
    artifact = transcode_conversion_artifact(result, output_name="filtered", target_format=format_name, gzip_output=True)
    payload = gzip.decompress(artifact.data)
    if format_name == "npz":
        with np.load(io.BytesIO(payload)) as archive:
            values = archive["values"].copy()
    else:
        values = pd.read_csv(io.BytesIO(payload), sep="," if format_name == "csv" else r"\s+",
                             header=None if format_name == "txt" else 0, skip_blank_lines=False).iloc[:, -1].to_numpy()
    np.testing.assert_equal(values, [np.nan] * 3 if all_missing else [1, np.nan, 3])


def test_track_filter_selects_after_binning_scaling_and_configure_missing_handling():
    upload = inspect_upload("signal.bedgraph", b"chr1\t0\t500\t-4\nchr1\t500\t1000\t8\nchr1\t1000\t2000\t-2\n")
    request = DataConversionRequest(inspection=upload, input_form="track", bin_size=1000, scaling_mode="multiplier", scale=2, data_filter=settings(value=7))
    prepared = prepare_data(request)
    np.testing.assert_equal(prepared.values, [4, 7])
    np.testing.assert_equal(prepared.coordinates["position_kb"], [.5, 1.5])


def test_timing_extraction_and_scaling_both_precede_filtering():
    fractions = np.zeros((2, 16))
    fractions[np.arange(2), [2, 10]] = 100
    rows = [["chr1", "chr1"], [0, 1000], [1000, 2000], *fractions.T]
    payload = gzip.compress(("\n".join("\t".join(map(str, row)) for row in rows) + "\n").encode())
    request = DataConversionRequest(
        inspection=inspect_upload("timing.mat.gz", payload), input_form="x", chromosome="chr1",
        repliseq_timing_method="mean", scaling_mode="multiplier", scale=2, target_format="npz",
        data_filter=settings(comparison="above", threshold=200, value=7),
    )
    np.testing.assert_allclose(prepare_data(request).values, [187.5, 7])
    with np.load(io.BytesIO(convert_data(request).artifact.data)) as archive:
        np.testing.assert_allclose(archive["values"], [187.5, 7])


def test_labelled_filter_aligns_to_a_chromosome_specific_table_profile():
    request = DataConversionRequest(
        inspection=inspect_upload("source.csv", b"chrom,position_kb,value\nchr1,0.5,1\nchr1,1.5,2\nchr1,2.5,3\n"),
        input_form="x", data_filter=settings(selection="profile", value=7,
            upload=inspect_upload("regions.bed", b"chr1\t1000\t2000\n"), profile_layout="intervals"),
    )
    np.testing.assert_equal(prepare_data(request).values, [1, 7, 3])


@pytest.mark.parametrize("form", ["x", "t", "xt"])
def test_shapes_are_preserved_and_nan_selection_follows_configure_missing_values(form):
    values = np.array([1., np.nan, 3.])
    if form == "t":
        values = values.reshape(1, -1)
    elif form == "xt":
        values = np.column_stack((values, values * 10))
    buffer = io.BytesIO()
    np.savez(buffer, values=values)
    request = DataConversionRequest(inspection=inspect_upload("source.npz", buffer.getvalue()), input_form=form,
                                    data_filter=settings(selection="nan", value=7))
    np.testing.assert_equal(prepare_data(request).values, np.nan_to_num(values))
    kept = prepare_data(replace(request, missing_policy="ignore")).values
    np.testing.assert_equal(kept, np.where(np.isnan(values), 7, values))
    assert kept.shape == values.shape


@pytest.mark.parametrize("options,message", [
    (dict(selection="between", lower=5, upper=1), "Lower filter threshold"),
    (dict(selection="maximum", threshold=-1), "non-negative"),
    (dict(action="noise", standard_deviation=-1), "standard deviation"),
    (dict(action="noise", seed=1.5), "Random seed"),
    (dict(action="replace", value=float("nan")), "finite number"),
])
def test_invalid_active_settings_have_clear_errors(options, message):
    with pytest.raises(ValueError, match=message):
        apply_data_filter([1, -2], settings(**options))


def test_overflow_and_cancellation_are_not_silently_ignored():
    with pytest.raises(ValueError, match="infinite values"):
        apply_data_filter([1e300], settings(selection="all", action="multiply", value=1e300))
    def cancelled():
        raise CancelledError()
    with pytest.raises(CancelledError):
        apply_data_filter([1, -2], settings(), cancel=cancelled)
