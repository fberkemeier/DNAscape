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
    DataFilterSettings, FilterProfile, FilterSelection, apply_data_filter,
    capture_filter_selection, read_filter_profile, resolve_filter,
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


@pytest.mark.parametrize("options,selected_indices", [
    ({"selection": "threshold", "threshold": 5}, [2, 3, 4, 5]),
    ({"selection": "between", "lower": 4, "upper": 8}, [0, 4, 5]),
    ({"selection": "between", "lower": 4, "upper": 8, "between_mode": "outside"}, [1, 2, 3, 5]),
    ({"selection": "minimum", "threshold": 5}, [2, 3, 4, 5]),
    ({"selection": "maximum", "comparison": "above", "threshold": 3}, [0, 1, 2, 5]),
    ({"selection": "nan"}, [0, 1, 2, 3, 4]),
    ({"selection": "profile", "profile": FilterProfile(np.array([1, 4]), layout="positions", units="index")}, [0, 2, 3, 5]),
    ({"selection": "profile", "profile": FilterProfile(np.array([[1, 3]]), layout="intervals", units="index")}, [0, 3, 4, 5]),
    ({"selection": "all"}, []),
])
def test_complement_inverts_every_selection_for_edits_preview_and_export(options, selected_indices):
    values = np.array([0., 4., 5., 8., 10., np.nan])
    mask = np.zeros(values.size, dtype=bool)
    mask[selected_indices] = True
    captured = []
    config = settings(complement=True, value=99, _position_sink=lambda selected, *_args, **_kwargs: captured.append(selected.copy()), **options)
    expected = values.copy()
    expected[mask] = 99
    np.testing.assert_equal(apply_data_filter(values, config), expected)
    np.testing.assert_equal(captured[0], mask)
    preview = capture_filter_selection(values, config)
    np.testing.assert_array_equal(np.unpackbits(np.frombuffer(preview.packed, dtype=np.uint8))[:values.size], mask)
    np.testing.assert_equal(values, [0, 4, 5, 8, 10, np.nan])


def test_complement_is_off_by_default_and_missing_profile_stays_inert():
    assert DataFilterSettings().complement is False
    values = np.array([1., -2., 3.])
    np.testing.assert_equal(apply_data_filter(values, DataFilterSettings(complement=True)), values)
    np.testing.assert_equal(apply_data_filter(values, settings(selection="profile", complement=True)), values)
    assert capture_filter_selection(values, settings(selection="profile", complement=True)) is None


@pytest.mark.parametrize("options,expected", [
    (dict(selection="threshold", comparison="above", threshold=5), [0, 4, 5, 99, 99]),
    (dict(selection="between", lower=4, upper=8), [0, 99, 99, 99, 10]),
    (dict(selection="minimum", comparison="below", threshold=5), [99, 99, 5, 8, 10]),
    (dict(selection="maximum", comparison="above", threshold=3), [0, 4, 5, 99, 99]),
    (dict(selection="all"), [99] * 5),
])
def test_signal_selection_uses_the_unedited_converted_values(options, expected):
    np.testing.assert_equal(apply_data_filter([0, 4, 5, 8, 10], settings(value=99, **options)), expected)


@pytest.mark.parametrize("mode,expected", [
    ("within", [0, 99, 99, 99, 10, np.nan]),
    ("outside", [99, 4, 5, 8, 99, np.nan]),
])
def test_between_thresholds_includes_boundaries_within_and_preserves_nan(mode, expected):
    config = settings(selection="between", lower=4, upper=8, between_mode=mode, value=99)
    np.testing.assert_equal(apply_data_filter([0, 4, 5, 8, 10, np.nan], config), expected)
    result = convert_data(DataConversionRequest(
        inspection=inspect_upload("signal.tsv", b"value\n0\n4\n5\n8\n10\n"), input_form="x",
        data_filter=config,
    ))
    assert dict(result.metadata)["filter_between_mode"] == mode
    with pytest.raises(ValueError, match="threshold range"):
        apply_data_filter([1], replace(config, between_mode="invalid"))


@pytest.mark.parametrize("enabled", [False, True])
def test_filter_preview_uses_post_configure_pre_edit_values_independently_of_apply(enabled):
    result = convert_data(DataConversionRequest(
        inspection=inspect_upload("signal.tsv", b"value\n-2\n1\n3\n"), input_form="x",
        scaling_mode="multiplier", scale=-2, data_filter=DataFilterSettings(enabled=enabled),
    ))
    assert result.preview.filter_selection.ranges() == [(.5, 2.5)]
    np.testing.assert_equal(result.preview.values, [4, 0, 0] if enabled else [4, -2, -6])


def test_preview_selection_ignores_unfinished_operation_and_inactive_selection_fields():
    selection = capture_filter_selection([1, -2, -3, 4], DataFilterSettings(action="modulate"))
    assert selection.ranges() == [(.5, 2.5)]
    assert capture_filter_selection([1, 2], DataFilterSettings(threshold=None)) is None
    assert capture_filter_selection([1, 2], DataFilterSettings(selection="profile")) is None
    assert FilterSelection.from_mask([]).ranges() == []
    assert FilterSelection.from_mask([True]).ranges() == [(-.5, .5)]


def test_preview_selection_is_packed_and_zoom_recovers_narrow_native_regions():
    mask = np.zeros(100_000, dtype=bool)
    mask[12_345] = True
    mask[80_000:80_003] = True
    selection = FilterSelection.from_mask(mask)
    assert len(selection.packed) == 12_500
    assert selection.ranges((12_340, 12_350)) == [(12_344.5, 12_345.5)]
    assert selection.ranges((79_998, 80_005)) == [(79_999.5, 80_002.5)]
    assert selection.ranges((100_001, 100_010)) == []
    alternating = FilterSelection.from_mask(np.arange(100_000) % 2 == 0)
    assert len(alternating.ranges()) <= 750
    assert len(alternating.ranges((1000, 1010))) > 1


@pytest.mark.parametrize("enabled", [False, True])
def test_track_preview_shading_uses_relative_positions_first_chromosome_and_partial_bins(enabled):
    source = inspect_upload("track.bedgraph", b"chr1\t10000\t11000\t1\nchr1\t11000\t12100\t-2\nchr2\t0\t2000\t-3\n")
    result = convert_data(DataConversionRequest(
        inspection=source, input_form="track", chromosome="chr1", start=10000, end=12100,
        bin_size=1000, data_filter=DataFilterSettings(enabled=enabled),
    ))
    np.testing.assert_allclose(result.preview.x, [0, 1, 1.55])
    np.testing.assert_allclose(result.preview.filter_selection.ranges(), [(.5, 1.825)])
    result = convert_data(DataConversionRequest(
        inspection=source, input_form="track", chromosome_scope="all", bin_size=1000,
        data_filter=DataFilterSettings(enabled=enabled), target_format="npz",
    ))
    assert result.preview.filter_selection.length == 13


def test_preview_shading_selects_only_the_displayed_time_slice():
    stream = io.BytesIO()
    np.savez(stream, values=np.array([[1, -1], [-2, 2], [3, -3]]))
    result = convert_data(DataConversionRequest(
        inspection=inspect_upload("matrix.npz", stream.getvalue()), input_form="xt",
        data_filter=DataFilterSettings(enabled=True),
    ))
    assert result.preview.filter_selection.ranges() == [(.5, 1.5)]


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
    result = convert_data(request)
    assert result.preview.filter_selection.ranges() == [(.5, 1.5)]
    with np.load(io.BytesIO(result.artifact.data)) as archive:
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
