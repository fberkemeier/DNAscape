"""Tests for the Data Manager's bounded ``rdata`` service."""

from __future__ import annotations

import gzip
import io
from concurrent.futures import CancelledError
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

import app.bigwig_service as bigwig_service
import app.data_conversion_service as conversion_service
from app.data_conversion_service import (
    DATA_EXPORT_FORMATS,
    DEFAULT_TARGET_FORMAT,
    DataConversionRequest,
    convert_data,
    inspect_data_source,
    inspect_track,
    prepare_data,
)
from app.mapping_uploads import inspect_upload, materialize_upload
from app.export_settings import preview_point_limit


def _npz_values(data: bytes) -> np.ndarray:
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        return archive["values"]


def test_profile_conversion_preserves_space_dependence_and_writes_gzipped_npz() -> None:
    inspection = inspect_upload(
        "signal.csv",
        b"position,signal\n0,1\n1,3\n2,2\n",
    )

    result = convert_data(
        DataConversionRequest(
            inspection=inspection,
            input_form="x",
            target_format="npz",
            output_name="spatial profile",
            gzip_output=True,
            value_column="signal",
        )
    )

    assert result.artifact.filename == "spatial_profile.npz.gz"
    assert result.artifact.media_type == "application/gzip"
    assert result.data_form == "space"
    assert result.shape == (3,)
    with np.load(io.BytesIO(gzip.decompress(result.artifact.data)), allow_pickle=False) as archive:
        assert str(archive["data_form"]) == "space"
        np.testing.assert_allclose(archive["values"], np.array([1.0, 3.0, 2.0]))


def test_matrix_conversion_preserves_space_time_shape_and_labels() -> None:
    inspection = inspect_upload(
        "field.csv",
        b"position,t0,t1\n0,0.1,0.2\n1,0.3,0.4\n",
    )

    result = convert_data(
        DataConversionRequest(
            inspection=inspection,
            input_form="xt",
            target_format="npz",
            gzip_output=False,
        )
    )

    assert result.data_form == "spacetime"
    assert result.shape == (2, 2)
    assert result.labels == ("t0", "t1")
    np.testing.assert_allclose(
        _npz_values(result.artifact.data),
        np.array([[0.1, 0.2], [0.3, 0.4]]),
    )


def test_time_profile_conversion_preserves_original_column_headers() -> None:
    inspection = inspect_upload(
        "time.tsv",
        b"minute_0\tminute_5\tminute_10\n0.1\t0.3\t0.8\n",
    )

    result = convert_data(
        DataConversionRequest(
            inspection=inspection,
            input_form="t",
            target_format="tsv",
        )
    )

    assert result.data_form == "time"
    assert result.labels == ("minute_0", "minute_5", "minute_10")
    assert result.artifact.data.decode("utf-8").splitlines()[0] == (
        "minute_0\tminute_5\tminute_10"
    )


def test_npz_space_profile_prefers_its_column_label_over_the_array_key() -> None:
    buffer = io.BytesIO()
    np.savez(
        buffer,
        values=np.array([0.2, 0.4, 0.8]),
        columns=np.array(["replicated_fraction"]),
        data_form=np.array("space"),
    )
    inspection = inspect_upload("profile.npz", buffer.getvalue())

    result = convert_data(
        DataConversionRequest(
            inspection=inspection,
            input_form="x",
            array_key="values",
            target_format="tsv",
        )
    )

    assert result.labels == ("replicated_fraction",)
    assert result.artifact.data.decode("utf-8").splitlines()[0] == "replicated_fraction"


def test_profile_custom_rescale_can_reverse_range_without_reordering_samples() -> None:
    inspection = inspect_upload("signal.tsv", b"signal\n1\n2\n3\n")
    request = DataConversionRequest(
        inspection=inspection,
        input_form="x",
        target_format="npz",
        value_column="signal",
        scaling_mode="interval",
        rescale_min=-1,
        rescale_max=1,
        rescale_invert=True,
    )

    prepared = prepare_data(request)
    result = convert_data(request)

    np.testing.assert_allclose(prepared.values, np.array([1.0, 0.0, -1.0]))
    np.testing.assert_allclose(
        _npz_values(result.artifact.data),
        prepared.values,
    )
    metadata = dict(result.metadata)
    assert metadata["value_scaling"] == "interval"
    assert metadata["target_minimum"] == "-1.0"
    assert metadata["target_maximum"] == "1.0"
    assert metadata["interval_inverted"] == "true"


def test_threshold_filter_masks_samples_and_smoothing_does_not_restore_them() -> None:
    inspection = inspect_upload("signal.tsv", b"signal\n1\n100\n3\n5\n")
    request = DataConversionRequest(
        inspection=inspection,
        input_form="x",
        target_format="npz",
        value_column="signal",
        missing_policy="ignore",
        threshold_enabled=True,
        threshold_max=10,
        smooth_window=3,
    )

    prepared = prepare_data(request)
    result = convert_data(request)

    assert prepared.shape == (4,)
    np.testing.assert_allclose(
        prepared.values,
        np.array([1.0, np.nan, 4.0, 4.0]),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        _npz_values(result.artifact.data),
        prepared.values,
        equal_nan=True,
    )
    assert result.preview is not None
    assert result.preview.total_count == 4
    assert result.preview.missing_count == 1
    np.testing.assert_allclose(result.preview.values, prepared.values, equal_nan=True)
    metadata = dict(result.metadata)
    assert metadata["threshold_enabled"] == "true"
    assert metadata["threshold_min"] == "none"
    assert metadata["threshold_max"] == "10.0"
    assert metadata["threshold_action"] == "mask"


def test_time_and_spacetime_polishing_follow_dnascape_axes() -> None:
    time_stream = io.BytesIO()
    np.save(time_stream, np.array([[1.0, 2.0, 3.0, 4.0, 5.0]]), allow_pickle=False)
    time_inspection = inspect_upload("time.npy", time_stream.getvalue())
    time_prepared = prepare_data(
        DataConversionRequest(
            inspection=time_inspection,
            input_form="t",
            smooth_window=3,
        )
    )
    np.testing.assert_allclose(time_prepared.values, [[1.5, 2.0, 3.0, 4.0, 4.5]])

    matrix_inspection = inspect_upload(
        "field.tsv",
        b"position\tt0\tt1\n0\t0\t10\n1\t5\t20\n2\t10\t30\n",
    )
    matrix_prepared = prepare_data(
        DataConversionRequest(
            inspection=matrix_inspection,
            input_form="xt",
            scaling_mode="interval",
            rescale_min=0,
            rescale_max=1,
        )
    )
    np.testing.assert_allclose(
        matrix_prepared.values,
        np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]),
    )


def test_constant_profile_custom_rescale_uses_target_midpoint() -> None:
    inspection = inspect_upload("constant.tsv", b"signal\n7\n7\n7\n")

    prepared = prepare_data(
        DataConversionRequest(
            inspection=inspection,
            input_form="x",
            value_column="signal",
            scaling_mode="interval",
            rescale_min=-2,
            rescale_max=4,
            rescale_invert=True,
        )
    )

    np.testing.assert_allclose(prepared.values, np.array([1.0, 1.0, 1.0]))


def test_explicit_scaling_mode_makes_inactive_legacy_fields_inert() -> None:
    inspection = inspect_upload("signal.tsv", b"signal\n1\n2\n3\n")

    prepared = prepare_data(
        DataConversionRequest(
            inspection=inspection,
            input_form="x",
            value_column="signal",
            scaling_mode="none",
            scale=float("inf"),
            offset=float("inf"),
            signal_transform="invert",
            rescale_min=5,
            rescale_max=1,
        )
    )

    np.testing.assert_allclose(prepared.values, np.array([1.0, 2.0, 3.0]))
    metadata = dict(prepared.metadata)
    assert metadata["value_scaling"] == "none"
    assert "scale_multiplier" not in metadata
    assert "target_minimum" not in metadata
    assert "legacy_offset" not in metadata
    assert "legacy_signal_transform" not in metadata


@pytest.mark.parametrize("form", ("x", "t", "xt", "track"))
@pytest.mark.parametrize("mode", ("none", "interval", "multiplier", "center", "zscore", "maxabs"))
def test_invert_is_independent_of_scaling_and_preserves_missing_positions(form, mode) -> None:
    values = np.array([2., np.nan, 3., 8.])
    extra = {}
    if form == "track":
        inspection = inspect_upload(
            "signal.bedgraph", b"chr1\t0\t10\t2\nchr1\t20\t30\t3\nchr1\t30\t40\t8\n",
        )
        extra = dict(bin_size=10, interpolation="none", track_missing="nan")
    else:
        if form == "t":
            values = values[None, :]
        elif form == "xt":
            values = np.column_stack((values, [12., np.nan, 18., 40.]))
        stream = io.BytesIO()
        np.save(stream, values, allow_pickle=False)
        inspection = inspect_upload("signal.npy", stream.getvalue())
    request = DataConversionRequest(
        inspection=inspection, input_form=form, missing_policy="ignore",
        scaling_mode=mode, rescale_invert=True, scale=2.5,
        rescale_min=-2, rescale_max=4, **extra,
    )
    expected = values.copy()
    columns = range(values.shape[1]) if form == "xt" else (None,)
    for column in columns:
        original = values[:, column] if column is not None else values
        inverted = np.nanmin(original) + (np.nanmax(original) - original)
        if mode == "interval":
            inverted = -2 + (inverted - np.nanmin(inverted)) / np.ptp(inverted[np.isfinite(inverted)]) * 6
        elif mode == "multiplier":
            inverted *= 2.5
        elif mode == "center":
            inverted -= np.nanmean(inverted)
        elif mode == "zscore":
            inverted = (inverted - np.nanmean(inverted)) / np.nanstd(inverted)
        elif mode == "maxabs":
            inverted /= np.nanmax(np.abs(inverted))
        if column is None:
            expected = inverted
        else:
            expected[:, column] = inverted
    prepared = prepare_data(request)
    np.testing.assert_allclose(prepared.values, expected, equal_nan=True)
    assert dict(prepared.metadata)["values_inverted"] == "true"
    original_coordinates = prepare_data(replace(request, rescale_invert=False)).coordinates
    assert prepared.coordinates.keys() == original_coordinates.keys()
    for key, coordinates in prepared.coordinates.items():
        np.testing.assert_equal(coordinates, original_coordinates[key])


def test_new_scaling_mode_cannot_overlap_legacy_normalization() -> None:
    inspection = inspect_upload("signal.tsv", b"signal\n1\n2\n")

    with pytest.raises(ValueError, match="cannot be combined with normalize"):
        prepare_data(
            DataConversionRequest(
                inspection=inspection,
                input_form="x",
                value_column="signal",
                scaling_mode="multiplier",
                scale=2,
                normalize="zscore",
            )
        )


def test_legacy_custom_rescale_request_maps_to_target_interval() -> None:
    inspection = inspect_upload("signal.tsv", b"signal\n2\n4\n6\n")

    prepared = prepare_data(
        DataConversionRequest(
            inspection=inspection,
            input_form="x",
            value_column="signal",
            normalize="rescale",
            rescale_min=10,
            rescale_max=20,
        )
    )

    np.testing.assert_allclose(prepared.values, np.array([10.0, 15.0, 20.0]))
    assert dict(prepared.metadata)["value_scaling"] == "interval"


def test_legacy_minmax_ignores_inactive_target_interval_fields() -> None:
    inspection = inspect_upload("signal.tsv", b"signal\n2\n4\n6\n")

    prepared = prepare_data(
        DataConversionRequest(
            inspection=inspection,
            input_form="x",
            value_column="signal",
            normalize="minmax",
            rescale_min=10,
            rescale_max=20,
            rescale_invert=True,
        )
    )

    np.testing.assert_allclose(prepared.values, np.array([0.0, 0.5, 1.0]))


def test_fully_thresholded_profile_is_safe_to_preview_and_rescale() -> None:
    inspection = inspect_upload("signal.tsv", b"signal\n1\n2\n")
    request = DataConversionRequest(
        inspection=inspection,
        input_form="x",
        target_format="npz",
        value_column="signal",
        missing_policy="ignore",
        threshold_enabled=True,
        threshold_min=10,
        scaling_mode="interval",
        rescale_min=-1,
        rescale_max=1,
    )

    result = convert_data(request)

    values = _npz_values(result.artifact.data)
    assert values.shape == (2,)
    assert np.isnan(values).all()
    assert result.preview is not None
    assert result.preview.finite_count == 0
    assert result.preview.missing_count == 2
    assert result.preview.minimum is None
    assert result.preview.maximum is None


def test_full_table_conversion_preserves_columns() -> None:
    inspection = inspect_upload(
        "table.csv",
        b"sample,signal\na,1.25\nb,2.5\n",
    )

    result = convert_data(
        DataConversionRequest(
            inspection=inspection,
            input_form="table",
            target_format="tsv",
            gzip_output=False,
            decimal_places=2,
        )
    )

    assert result.data_form == "table"
    assert result.artifact.data.decode("utf-8") == "sample\tsignal\na\t1.25\nb\t2.50\n"


def test_profile_conversion_can_preserve_missing_values() -> None:
    inspection = inspect_upload(
        "signal.csv",
        b"position,signal\n0,1\n1,\n2,3\n",
    )

    result = convert_data(
        DataConversionRequest(
            inspection=inspection,
            input_form="x",
            target_format="npz",
            gzip_output=False,
            value_column="signal",
            missing_policy="ignore",
        )
    )

    values = _npz_values(result.artifact.data)
    np.testing.assert_allclose(values[[0, 2]], np.array([1.0, 3.0]))
    assert np.isnan(values[1])


def test_bedgraph_inspection_and_conversion_use_real_track_bounds() -> None:
    inspection = inspect_upload(
        "signal.bedgraph",
        b"chr2\t0\t5\t4\nchr1\t0\t10\t1\nchr1\t20\t30\t3\n",
    )

    track = inspect_track(inspection)
    result = convert_data(
        DataConversionRequest(
            inspection=inspection,
            input_form="track",
            target_format="tsv",
            gzip_output=False,
            chromosome="chr1",
            start=0,
            end=30,
            bin_size=10,
        )
    )

    assert tuple((item.chromosome, item.start, item.end) for item in track.regions) == (
        ("chr1", 0, 30),
        ("chr2", 0, 5),
    )
    assert result.data_form == "space"
    assert result.labels == ("chr1",)
    assert result.artifact.data.startswith(b"chr1\n")
    assert b"start_bp" not in result.artifact.data


def test_conversion_reports_rdata_load_prepare_write_and_completion_progress() -> None:
    inspection = inspect_upload(
        "signal.csv",
        b"position,signal\n0,1\n1,3\n2,2\n",
    )
    events: list[dict[str, object]] = []

    result = convert_data(
        DataConversionRequest(
            inspection=inspection,
            input_form="x",
            target_format="npz",
            gzip_output=False,
            value_column="signal",
        ),
        events.append,
    )

    assert result.data_form == "space"
    assert [event["stage"] for event in events] == [
        "load",
        "load",
        "prepare",
        "prepare",
        "write",
        "write",
        "complete",
    ]
    assert events[-1]["progress"] == 1.0


def test_plain_conversion_checks_cancellation_from_rdata_stage_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inspection = inspect_upload("signal.tsv", b"signal\n1\n2\n3\n")
    stopping = False

    def cancel() -> None:
        if stopping:
            raise CancelledError("Conversion stopped.")

    def fake_rdata(
        _source: object,
        target: str,
        output_path: Path,
        **kwargs: object,
    ) -> object:
        nonlocal stopping
        assert target == "tsv"
        assert output_path.suffix == ".tsv"
        stopping = True
        report = kwargs["progress"]
        assert callable(report)
        report(
            {
                "route": "tsv->tsv",
                "stage": "load",
                "status": "done",
                "progress": 0.3,
            }
        )
        raise AssertionError("The cancellation callback should interrupt the stage event.")

    monkeypatch.setattr(conversion_service, "rdata", fake_rdata)

    with pytest.raises(CancelledError, match="Conversion stopped"):
        convert_data(
            DataConversionRequest(
                inspection=inspection,
                input_form="x",
                value_column="signal",
            ),
            cancel=cancel,
        )


def test_bigwig_conversion_checks_cancellation_from_chunk_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inspection = inspect_upload("signal.bw", b"&\xfc\x8f\x88fake-bigwig")
    stopping = False

    monkeypatch.setattr(
        conversion_service,
        "inspect_track",
        lambda _inspection, _cancel=None: conversion_service.TrackInspection(
            source_format="bigwig",
            regions=(conversion_service.TrackRegion("chr1", 0, 20),),
            backend="test",
        ),
    )

    def cancel() -> None:
        if stopping:
            raise CancelledError("Conversion stopped.")

    def fake_read_bigwig_bins(_path: Path, **kwargs: object) -> np.ndarray:
        nonlocal stopping
        stopping = True
        report = kwargs["progress"]
        assert callable(report)
        report(0.5)
        raise AssertionError("The cancellation callback should interrupt the BigWig read.")

    monkeypatch.setattr(conversion_service, "read_bigwig_bins", fake_read_bigwig_bins)

    with pytest.raises(CancelledError, match="Conversion stopped"):
        convert_data(
            DataConversionRequest(
                inspection=inspection,
                input_form="track",
                chromosome="chr1",
                start=0,
                end=20,
                bin_size=10,
            ),
            cancel=cancel,
        )


def test_bedgraph_conversion_checks_cancellation_after_track_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inspection = inspect_upload("signal.bedgraph", b"chr1\t0\t10\t1\n")
    stopping = False

    def cancel() -> None:
        if stopping:
            raise CancelledError("Conversion stopped.")

    def fake_rtrack(_source: Path, **_kwargs: object) -> object:
        nonlocal stopping
        stopping = True
        return {"table": pd.DataFrame({"chrom": ["chr1"], "value": [1.0]})}

    monkeypatch.setattr(conversion_service, "rtrack", fake_rtrack)

    with pytest.raises(CancelledError, match="Conversion stopped"):
        convert_data(
            DataConversionRequest(
                inspection=inspection,
                input_form="track",
                chromosome="chr1",
                start=0,
                end=10,
                bin_size=10,
            ),
            cancel=cancel,
        )


def test_text_export_is_the_uncompressed_default_and_numpy_formats_are_last() -> None:
    inspection = inspect_upload("signal.tsv", b"signal\n1\n2\n3\n")
    request = DataConversionRequest(
        inspection=inspection,
        input_form="x",
        value_column="signal",
    )

    result = convert_data(request)

    assert request.target_format == DEFAULT_TARGET_FORMAT == "tsv"
    assert request.gzip_output is False
    assert tuple(key for key, _label in DATA_EXPORT_FORMATS) == (
        "tsv",
        "txt",
        "csv",
        "npz",
    )
    assert result.artifact.filename == "rdata.tsv"
    assert result.artifact.data == b"signal\n1.0\n2.0\n3.0\n"
    assert result.preview is not None
    np.testing.assert_allclose(result.preview.values, np.array([1.0, 2.0, 3.0]))
    assert (result.preview.minimum, result.preview.maximum) == (1.0, 3.0)


def test_bedgraph_header_is_detected_and_track_pipeline_preserves_plot_coordinates() -> None:
    inspection = inspect_upload(
        "signal.bedgraph",
        b"chrom\tstart\tend\tsignal\nchr1\t0\t10\t0\nchr1\t20\t30\t10\n",
    )
    request = DataConversionRequest(
        inspection=inspection,
        input_form="track",
        chromosome="chr1",
        start=0,
        end=30,
        bin_size=10,
        interpolation="linear",
        normalize="minmax",
        value_name="relative_signal",
    )

    source = inspect_data_source(inspection)
    prepared = prepare_data(request)
    result = convert_data(request)

    assert source.source_format == "bedgraph"
    assert source.chromosomes == (conversion_service.TrackRegion("chr1", 0, 30, interval_count=2),)
    np.testing.assert_allclose(prepared.values, np.array([0.0, 0.5, 1.0]))
    np.testing.assert_allclose(prepared.coordinates["position_kb"], np.array([0.005, 0.015, 0.025]))
    assert prepared.values.flags.writeable is False
    rows = result.artifact.data.decode("utf-8").splitlines()
    assert rows == ["chr1", "0.0", "0.5", "1.0"]
    assert result.preview is not None
    assert result.preview.x_label == "Position Kb"
    np.testing.assert_allclose(result.preview.x, np.array([0.0, 0.01, 0.02]))


def test_ucsc_bedgraph_directives_survive_the_full_track_pipeline() -> None:
    inspection = inspect_upload(
        "sns_signal.bedGraph.gz",
        gzip.compress(
            b'\xef\xbb\xbftrack type=bedGraph name="SNS signal" visibility=full\n'
            b"chr1\t0\t10\t24\n"
            b"  browser position chr1:1-20\n"
            b"  # interleaved metadata\n"
            b"chr1\t10\t20\t4\n"
        ),
    )
    request = DataConversionRequest(
        inspection=inspection,
        input_form="track",
        chromosome="chr1",
        start=0,
        end=20,
        bin_size=10,
        aggregation="sum",
    )

    track = inspect_track(inspection)
    prepared = prepare_data(request)

    assert track.regions == (
        conversion_service.TrackRegion("chr1", 0, 20, interval_count=2),
    )
    np.testing.assert_allclose(prepared.values, np.array([240.0, 40.0]))


def test_sns_enrichment_maximum_keeps_peak_heights_and_zero_background(
    sns_peak_payload: bytes,
) -> None:
    inspection = inspect_upload("K562_Peaks_FDR15.bed.gz", gzip.compress(sns_peak_payload))
    request = DataConversionRequest(
        inspection=inspection,
        input_form="track",
        chromosome="chr1",
        start=0,
        end=2300,
        bin_size=1000,
        aggregation="max",
        value_column="enrichment_ratio",
        biological_role="initiation_rate",
    )
    prepared = prepare_data(request)
    np.testing.assert_array_equal(prepared.values, [10, 0, 20])
    mean = prepare_data(replace(request, aggregation="mean"))
    np.testing.assert_allclose(mean.values, [3.6, 0, 20])
    result = convert_data(request)
    reloaded = inspect_upload(result.artifact.filename, result.artifact.data, expected_form="x")
    np.testing.assert_array_equal(materialize_upload(reloaded), [10, 0, 20])
    with pytest.raises(ValueError, match="not a selectable numeric signal column"):
        prepare_data(replace(request, value_column="thick_start"))


def test_track_preview_uses_zero_based_bin_starts_without_changing_export_coordinates() -> None:
    inspection = inspect_upload("track.bedGraph", b"chr1\t5000\t8000\t2\n")
    request = DataConversionRequest(
        inspection=inspection, input_form="track", target_format="npz",
        chromosome="chr1", start=5000, end=8000, bin_size=1000,
    )
    prepared = prepare_data(request)
    result = convert_data(request)
    np.testing.assert_array_equal(prepared.coordinates["position_kb"], [5.5, 6.5, 7.5])
    assert result.preview is not None
    np.testing.assert_array_equal(result.preview.x, [0, 1, 2])
    np.testing.assert_array_equal(_npz_values(result.artifact.data), prepared.values)


def test_sns_preview_keeps_isolated_peaks_at_their_zero_based_bin_positions() -> None:
    inspection = inspect_upload(
        "sns_peaks.bed",
        b"chr1\t137000\t137100\t20\nchr1\t1999000\t2000000\t1\n",
    )
    result = convert_data(DataConversionRequest(
        inspection=inspection, input_form="track", target_format="npz",
        chromosome="chr1", start=0, end=2000000, bin_size=1000, aggregation="max",
        biological_role="initiation_rate",
    ))
    assert result.preview is not None
    preview = result.preview
    assert preview.total_count == 2000
    assert preview.values.size <= preview_point_limit()
    assert preview.maximum == 20
    assert np.max(preview.values) == 20
    assert preview.x[np.argmax(preview.values)] == 137.0
    values = _npz_values(result.artifact.data)
    assert values.shape == (2000,)
    assert values[137] == 20
    assert np.count_nonzero(values) == 2
    indices = preview.x.astype(int)
    np.testing.assert_array_equal(preview.values, values[indices])
    assert np.all(np.diff(indices) > 0)
    # The continuous-track preview remains unchanged for Repli-seq BigWig.
    continuous = conversion_service._preview(values, preserve_peaks=False)
    assert continuous is not None
    np.testing.assert_array_equal(continuous.values, values[np.linspace(0, 1999, preview_point_limit(), dtype=int)])


def test_coordinate_tsv_round_trips_as_a_mapping_profile() -> None:
    inspection = inspect_upload(
        "signal.bedgraph",
        b"chr1\t0\t10\t2\nchr1\t10\t20\t4\nchr1\t20\t30\t6\n",
    )
    converted = convert_data(
        DataConversionRequest(
            inspection=inspection,
            input_form="track",
            target_format="tsv",
            chromosome="chr1",
            start=0,
            end=30,
            bin_size=10,
            value_name="initiation_rate",
        )
    )

    mapping_input = inspect_upload(
        converted.artifact.filename,
        converted.artifact.data,
        expected_form="x",
    )
    values = materialize_upload(mapping_input, column="chr1")

    assert mapping_input.source_format == "tsv"
    assert "chr1" in mapping_input.numeric_columns
    np.testing.assert_allclose(values, np.array([2.0, 4.0, 6.0]))


@pytest.mark.parametrize(("target_format", "separator"), (("tsv", "\t"), ("csv", ",")))
def test_all_chromosome_text_export_is_wide_labelled_and_nan_padded(
    target_format: str,
    separator: str,
) -> None:
    inspection = inspect_upload(
        "signal.bedgraph",
        (
            b"chr2\t0\t10\t100\nchr2\t10\t20\t200\n"
            b"chr1\t0\t10\t0\nchr1\t10\t20\t10\nchr1\t20\t30\t20\n"
        ),
    )
    request = DataConversionRequest(
        inspection=inspection,
        input_form="track",
        chromosome_scope="all",
        target_format=target_format,
        bin_size=10,
        normalize="minmax",
        value_name="replication_timing",
    )

    prepared = prepare_data(request)
    result = convert_data(request)
    frame = pd.read_csv(io.BytesIO(result.artifact.data), sep=separator)

    assert list(frame.columns) == ["chr1", "chr2"]
    assert not {"chrom", "start_bp", "end_bp", "position_bp", "position_kb"} & set(frame.columns)
    np.testing.assert_allclose(frame["chr1"], np.array([0.0, 0.5, 1.0]))
    np.testing.assert_allclose(frame["chr2"].iloc[:2], np.array([0.0, 1.0]))
    assert np.isnan(frame["chr2"].iloc[2])
    # Normalization and every other preprocessing step are applied within,
    # never across, chromosome boundaries.
    np.testing.assert_allclose(prepared.profile_values["chr1"], np.array([0.0, 0.5, 1.0]))
    np.testing.assert_allclose(prepared.profile_values["chr2"], np.array([0.0, 1.0]))
    assert result.labels == ("chr1", "chr2")
    assert result.shape == (3, 2)
    assert result.preview is not None
    assert result.preview.value_label == "replication_timing"
    assert result.preview.source_shape == (3,)
    np.testing.assert_allclose(result.preview.values, np.array([0.0, 0.5, 1.0]))

    mapping_input = inspect_upload(
        result.artifact.filename,
        result.artifact.data,
        expected_form="x",
    )
    assert mapping_input.numeric_columns == ("chr1", "chr2")
    np.testing.assert_allclose(
        materialize_upload(mapping_input, column="chr2", missing_policy="ignore"),
        np.array([0.0, 1.0]),
    )
    mapping_input = inspect_upload(
        result.artifact.filename,
        result.artifact.data,
        expected_form="x",
    )
    np.testing.assert_allclose(
        materialize_upload(mapping_input, column="chr2"),
        np.array([0.0, 1.0]),
    )


def test_all_chromosome_npz_stores_native_length_named_profiles() -> None:
    inspection = inspect_upload(
        "signal.bedgraph",
        b"chr1\t0\t10\t1\nchr1\t10\t20\t2\nchr2\t0\t10\t3\n",
    )

    result = convert_data(
        DataConversionRequest(
            inspection=inspection,
            input_form="track",
            chromosome_scope="all",
            target_format="npz",
            bin_size=10,
        )
    )

    with np.load(io.BytesIO(result.artifact.data), allow_pickle=False) as archive:
        assert set(archive.files) == {
            "chr1",
            "chr2",
            "data_form",
            "semantic_form",
            "region_names",
        }
        np.testing.assert_allclose(archive["chr1"], np.array([1.0, 2.0]))
        np.testing.assert_allclose(archive["chr2"], np.array([3.0]))
        assert str(archive["data_form"]) == "space"
        assert str(archive["semantic_form"]) == "chromosome_profiles"
        np.testing.assert_array_equal(archive["region_names"], np.array(["chr1", "chr2"]))

    mapping_input = inspect_upload(
        result.artifact.filename,
        result.artifact.data,
        expected_form="x",
    )
    assert mapping_input.array_keys == ("chr1", "chr2")
    assert mapping_input.default_array_key == "chr1"
    np.testing.assert_allclose(
        materialize_upload(mapping_input, array_key="chr2"),
        np.array([3.0]),
    )
    mapping_input = inspect_upload(
        result.artifact.filename,
        result.artifact.data,
        expected_form="x",
    )
    assert mapping_input.region_names == ("chr1", "chr2")
    np.testing.assert_allclose(
        materialize_upload(mapping_input, array_key="chr2"),
        np.array([3.0]),
    )


@pytest.mark.parametrize("target_format", ("txt",))
def test_all_chromosome_scope_rejects_formats_that_lose_chromosome_labels(
    target_format: str,
) -> None:
    inspection = inspect_upload(
        "signal.bedgraph",
        b"chr1\t0\t10\t1\nchr2\t0\t10\t2\n",
    )

    with pytest.raises(ValueError, match="requires a labelled format"):
        convert_data(
            DataConversionRequest(
                inspection=inspection,
                input_form="track",
                chromosome_scope="all",
                target_format=target_format,
            )
        )


def test_single_chromosome_npz_preserves_the_chromosome_label() -> None:
    inspection = inspect_upload("signal.bedgraph", b"chr1\t0\t10\t1\nchr1\t10\t20\t2\n")

    result = convert_data(
        DataConversionRequest(
            inspection=inspection,
            input_form="track",
            target_format="npz",
            chromosome="chr1",
            bin_size=10,
        )
    )

    with np.load(io.BytesIO(result.artifact.data), allow_pickle=False) as archive:
        np.testing.assert_allclose(archive["values"], np.array([1.0, 2.0]))
        np.testing.assert_array_equal(archive["columns"], np.array(["chr1"]))


def test_track_default_origin_keeps_leading_bins_and_fills_them_with_zero() -> None:
    inspection = inspect_upload(
        "signal.bedgraph",
        b"chr1\t10\t20\t2\nchr1\t20\t30\t4\n",
    )

    track = inspect_track(inspection)
    prepared = prepare_data(
        DataConversionRequest(
            inspection=inspection,
            input_form="track",
            chromosome="chr1",
            bin_size=10,
        )
    )

    assert track.regions[0].start == 0
    assert prepared.values[0] == 0
    np.testing.assert_allclose(prepared.values[1:], np.array([2.0, 4.0]))
    np.testing.assert_allclose(prepared.coordinates["position_kb"], [0.005, 0.015, 0.025])


def test_all_chromosome_padded_size_is_preflighted_before_preparation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inspection = inspect_upload(
        "signal.bedgraph",
        b"chr1\t0\t30\t1\nchr2\t0\t20\t2\n",
    )
    monkeypatch.setattr(conversion_service, "MAX_ARRAY_ELEMENTS", 5)

    with pytest.raises(ValueError, match="6 padded cells"):
        prepare_data(
            DataConversionRequest(
                inspection=inspection,
                input_form="track",
                chromosome_scope="all",
                target_format="tsv",
                bin_size=10,
            )
        )


def test_all_chromosome_npz_member_limit_is_preflighted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inspection = inspect_upload(
        "signal.bedgraph",
        b"chr1\t0\t10\t1\nchr2\t0\t10\t2\n",
    )
    monkeypatch.setattr(conversion_service, "MAX_ARCHIVE_MEMBERS", 4)

    with pytest.raises(ValueError, match="supports at most 1"):
        convert_data(
            DataConversionRequest(
                inspection=inspection,
                input_form="track",
                chromosome_scope="all",
                target_format="npz",
            )
        )


def test_track_drop_policy_is_rejected_to_preserve_uniform_genomic_spacing() -> None:
    inspection = inspect_upload(
        "signal.bedgraph",
        b"chr1\t0\t10\t1\nchr1\t20\t30\t3\n",
    )

    with pytest.raises(ValueError, match="track missing-value policy"):
        prepare_data(
            DataConversionRequest(
                inspection=inspection,
                input_form="track",
                chromosome="chr1",
                start=0,
                end=30,
                bin_size=10,
                track_missing="drop",
            )
        )


def test_track_smooth_and_zscore_are_explicit_in_provenance() -> None:
    inspection = inspect_upload(
        "repliseq.bedgraph",
        b"chr1\t0\t10\t0\nchr1\t10\t20\t2\nchr1\t20\t30\t4\n",
    )

    prepared = prepare_data(
        DataConversionRequest(
            inspection=inspection,
            input_form="track",
            chromosome="chr1",
            start=0,
            end=30,
            bin_size=10,
            smooth_window=3,
            scaling_mode="zscore",
            value_name="relative_signal",
        )
    )

    np.testing.assert_allclose(
        prepared.values,
        np.array([-1.224744871391589, 0.0, 1.224744871391589]),
    )
    metadata = dict(prepared.metadata)
    assert metadata["value_scaling"] == "zscore"
    assert "legacy_offset" not in metadata


def test_track_multiplier_is_an_alternative_value_scaling_mode() -> None:
    inspection = inspect_upload(
        "signal.bedgraph",
        b"chr1\t0\t10\t0\nchr1\t10\t20\t2\nchr1\t20\t30\t4\n",
    )

    prepared = prepare_data(
        DataConversionRequest(
            inspection=inspection,
            input_form="track",
            chromosome="chr1",
            start=0,
            end=30,
            bin_size=10,
            scaling_mode="multiplier",
            scale=2.5,
        )
    )

    np.testing.assert_allclose(prepared.values, np.array([0.0, 5.0, 10.0]))
    metadata = dict(prepared.metadata)
    assert metadata["value_scaling"] == "multiplier"
    assert metadata["scale_multiplier"] == "2.5"
    assert "legacy_offset" not in metadata


def test_track_threshold_and_custom_rescale_preserve_genomic_bins() -> None:
    inspection = inspect_upload(
        "signal.bedgraph",
        b"chr1\t0\t10\t0\nchr1\t10\t20\t10\nchr1\t20\t30\t100\n",
    )

    prepared = prepare_data(
        DataConversionRequest(
            inspection=inspection,
            input_form="track",
            chromosome="chr1",
            start=0,
            end=30,
            bin_size=10,
            threshold_enabled=True,
            threshold_max=10,
            scaling_mode="interval",
            rescale_min=2,
            rescale_max=4,
            rescale_invert=True,
        )
    )

    assert prepared.shape == (3,)
    np.testing.assert_allclose(prepared.values, np.array([4.0, 2.0, 0.0]))
    np.testing.assert_allclose(prepared.coordinates["position_kb"], [0.005, 0.015, 0.025])


def test_track_smoothing_preserves_unfilled_source_gaps() -> None:
    inspection = inspect_upload(
        "signal.bedgraph",
        b"chr1\t0\t10\t1\nchr1\t20\t30\t3\n",
    )

    prepared = prepare_data(
        DataConversionRequest(
            inspection=inspection,
            input_form="track",
            chromosome="chr1",
            start=0,
            end=30,
            bin_size=10,
            track_missing="nan",
            interpolation="none",
            smooth_window=3,
        )
    )

    np.testing.assert_allclose(prepared.values, np.array([1.0, np.nan, 3.0]), equal_nan=True)


@pytest.mark.parametrize(("missing", "fill_value"), (("zero", 0.0), ("fill", 7.0)))
@pytest.mark.parametrize(
    ("scaling", "extra", "scaled_values"),
    (
        ("none", {}, [10.0, 20.0]),
        ("multiplier", {"scale": 3.0}, [30.0, 60.0]),
        ("center", {}, [-5.0, 5.0]),
        ("zscore", {}, [-1.0, 1.0]),
        ("maxabs", {}, [0.5, 1.0]),
        ("interval", {"rescale_min": 2.0, "rescale_max": 4.0}, [2.0, 4.0]),
        (
            "interval",
            {"rescale_min": 2.0, "rescale_max": 4.0, "rescale_invert": True},
            [4.0, 2.0],
        ),
    ),
)
def test_track_fills_are_written_after_smoothing_and_every_scaling_mode(
    missing: str,
    fill_value: float,
    scaling: str,
    extra: dict[str, Any],
    scaled_values: list[float],
) -> None:
    inspection = inspect_upload("signal.bedgraph", b"chr1\t0\t10\t10\nchr1\t20\t30\t20\n")
    prepared = prepare_data(
        DataConversionRequest(
            inspection=inspection,
            input_form="track",
            bin_size=10,
            track_missing=missing,
            fill_value=fill_value,
            smooth_window=3,
            scaling_mode=scaling,
            **extra,
        )
    )
    np.testing.assert_allclose(prepared.values, [scaled_values[0], fill_value, scaled_values[1]])
    np.testing.assert_allclose(prepared.coordinates["position_kb"], [0.005, 0.015, 0.025])


@pytest.mark.parametrize("missing", ("min", "max"))
@pytest.mark.parametrize(
    ("scaling", "extra", "minimum_result", "maximum_result"),
    (
        ("none", {}, [10, 10, 20], [10, 20, 20]),
        ("multiplier", {"scale": 3}, [30, 30, 60], [30, 60, 60]),
        ("multiplier", {"scale": -2}, [-20, -20, -40], [-20, -40, -40]),
        ("center", {}, [-10 / 3, -10 / 3, 20 / 3], [-20 / 3, 10 / 3, 10 / 3]),
        (
            "zscore", {},
            [-np.sqrt(0.5), -np.sqrt(0.5), np.sqrt(2)],
            [-np.sqrt(2), np.sqrt(0.5), np.sqrt(0.5)],
        ),
        ("maxabs", {}, [0.5, 0.5, 1], [0.5, 1, 1]),
        ("interval", {"rescale_min": 2, "rescale_max": 4}, [2, 2, 4], [2, 4, 4]),
        (
            "interval", {"rescale_min": 2, "rescale_max": 4, "rescale_invert": True},
            [4, 4, 2], [4, 2, 2],
        ),
    ),
)
def test_minimum_and_maximum_fills_participate_in_every_scaling_mode(
    missing: str,
    scaling: str,
    extra: dict[str, Any],
    minimum_result: list[float],
    maximum_result: list[float],
) -> None:
    inspection = inspect_upload("signal.bedgraph", b"chr1\t0\t10\t10\nchr1\t20\t30\t20\n")
    prepared = prepare_data(DataConversionRequest(
        inspection=inspection, input_form="track", bin_size=10,
        track_missing=missing, smooth_window=3, scaling_mode=scaling, **extra,
    ))
    np.testing.assert_allclose(prepared.values, minimum_result if missing == "min" else maximum_result)
    np.testing.assert_allclose(prepared.coordinates["position_kb"], [0.005, 0.015, 0.025])


@pytest.mark.parametrize(
    ("missing", "expected"),
    (("min", [0, 0, 0.5, 1, 0]), ("max", [1, 0, 0.5, 1, 1])),
)
def test_extreme_fills_use_values_after_thresholding_and_smoothing(
    missing: str, expected: list[float],
) -> None:
    inspection = inspect_upload(
        "signal.bedgraph",
        b"chr1\t0\t10\t0\nchr1\t10\t20\t10\nchr1\t20\t30\t30\n"
        b"chr1\t30\t40\t50\nchr1\t40\t50\t100\n",
    )
    prepared = prepare_data(DataConversionRequest(
        inspection=inspection, input_form="track", bin_size=10, track_missing=missing,
        threshold_enabled=True, threshold_min=10, threshold_max=50, smooth_window=3,
        scaling_mode="interval", rescale_min=0, rescale_max=1,
    ))
    np.testing.assert_allclose(prepared.values, expected)


@pytest.mark.parametrize(
    ("missing", "expected"),
    (("min", [[10, 100], [10, 100], [20, 300]]), ("max", [[10, 100], [20, 300], [20, 300]])),
)
def test_extreme_fills_are_calculated_independently_for_each_chromosome(
    missing: str, expected: list[list[float]],
) -> None:
    inspection = inspect_upload(
        "signal.bedgraph",
        b"chr1\t0\t10\t10\nchr1\t20\t30\t20\nchr2\t0\t10\t100\nchr2\t20\t30\t300\n",
    )
    prepared = prepare_data(DataConversionRequest(
        inspection=inspection, input_form="track", chromosome_scope="all",
        bin_size=10, track_missing=missing, scaling_mode="none",
    ))
    np.testing.assert_array_equal(prepared.values, expected)


@pytest.mark.parametrize("missing", ("min", "max"))
@pytest.mark.parametrize("threshold_all", (False, True))
def test_extreme_fills_report_an_error_when_no_finite_values_remain(
    missing: str, threshold_all: bool,
) -> None:
    inspection = inspect_upload("signal.bedgraph", b"chr1\t0\t10\t10\nchr1\t20\t30\t20\n")
    request = DataConversionRequest(
        inspection=inspection, input_form="track", chromosome="chr1", bin_size=10,
        start=0 if threshold_all else 10, end=30 if threshold_all else 20,
        track_missing=missing, threshold_enabled=threshold_all, threshold_min=100,
        scaling_mode="interval", rescale_min=2, rescale_max=4,
    )
    with pytest.raises(ValueError, match="Cannot fill empty values.*no finite values remain"):
        convert_data(request)


def test_plain_profile_zero_fill_is_written_after_scaling_without_losing_rows() -> None:
    request = DataConversionRequest(
        inspection=inspect_upload("profile.tsv", b'chr1\n10\n""\n20\n'),
        input_form="x",
        missing_policy="zero",
        scaling_mode="interval",
        rescale_min=2.0,
        rescale_max=4.0,
    )
    np.testing.assert_allclose(prepare_data(request).values, [2.0, 0.0, 4.0])
    result = convert_data(request)
    frame = pd.read_csv(io.BytesIO(result.artifact.data), sep="\t")
    np.testing.assert_allclose(frame["chr1"], [2.0, 0.0, 4.0])


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"threshold_enabled": True}, "requires a lower or upper"),
        (
            {"threshold_enabled": True, "threshold_min": 5, "threshold_max": 4},
            "Lower threshold cannot exceed",
        ),
        (
            {"scaling_mode": "interval", "rescale_min": 1, "rescale_max": 1},
            "Target maximum must be greater",
        ),
        (
            {"threshold_enabled": True, "threshold_min": float("inf")},
            "Lower threshold must be a finite number",
        ),
        ({"normalize": "minmax", "input_form": "table"}, "Data polishing is available"),
        ({"smooth_window": 3, "input_form": "scalar"}, "Data polishing is available"),
    ),
)
def test_polishing_options_are_validated(changes: dict[str, object], message: str) -> None:
    inspection = inspect_upload("signal.tsv", b"signal\n1\n2\n")
    request = DataConversionRequest(
        inspection=inspection,
        input_form="x",
        value_column="signal",
    )

    with pytest.raises(ValueError, match=message):
        prepare_data(replace(request, **changes))


def test_disabled_threshold_ignores_stale_hidden_bounds() -> None:
    inspection = inspect_upload("signal.tsv", b"signal\n1\n2\n")

    prepared = prepare_data(
        DataConversionRequest(
            inspection=inspection,
            input_form="x",
            value_column="signal",
            threshold_enabled=False,
            threshold_min=float("inf"),
            threshold_max=float("-inf"),
        )
    )

    np.testing.assert_allclose(prepared.values, np.array([1.0, 2.0]))
    assert dict(prepared.metadata)["threshold_enabled"] == "false"


def test_track_bin_limit_is_checked_before_rasterization(monkeypatch: pytest.MonkeyPatch) -> None:
    inspection = inspect_upload(
        "signal.bedgraph",
        b"chr1\t0\t30\t1\n",
    )
    monkeypatch.setattr(conversion_service, "_MAX_TRACK_BINS", 2)

    with pytest.raises(ValueError, match="creates 3 bins"):
        prepare_data(
            DataConversionRequest(
                inspection=inspection,
                input_form="track",
                chromosome="chr1",
                start=0,
                end=30,
                bin_size=10,
            )
        )


def test_track_memory_estimate_is_checked_before_dataframe_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inspection = inspect_upload("signal.bedgraph", b"chr1\t0\t750000\t1\n")

    def unexpected_rasterization(*_args: object, **_kwargs: object) -> None:
        pytest.fail("Memory validation must run before rasterizing the track.")

    monkeypatch.setattr(conversion_service, "rtrack", unexpected_rasterization)

    with pytest.raises(ValueError, match="Conversion memory limit.*Settings"):
        prepare_data(
            DataConversionRequest(
                inspection=inspection,
                input_form="track",
                chromosome="chr1",
                start=0,
                end=750000,
                bin_size=1,
            )
        )


@pytest.mark.parametrize("target_format", ("tsv", "npz"))
def test_raising_memory_budget_allows_all_chromosome_bigwig_conversion(
    monkeypatch: pytest.MonkeyPatch,
    target_format: str,
) -> None:
    inspection = inspect_upload("signal.bw", b"&\xfc\x8f\x88fake-bigwig")
    monkeypatch.setattr(
        conversion_service,
        "inspect_track",
        lambda _inspection, _cancel=None: conversion_service.TrackInspection(
            source_format="bigwig",
            regions=(
                conversion_service.TrackRegion("chr1", 0, 30),
                conversion_service.TrackRegion("chr2", 0, 20),
            ),
            backend="test",
        ),
    )
    # Exercise a large estimate using tiny profiles to keep the regression fast.
    monkeypatch.setattr(conversion_service, "_TRACK_ESTIMATED_BYTES_PER_BIN", 40 * 1024**2)
    read_chromosomes: list[str] = []

    def read_bins(_path: Path, **kwargs: Any) -> np.ndarray:
        chromosome = kwargs["chromosome"]
        read_chromosomes.append(chromosome)
        return np.array([1.0, 2.0, 3.0] if chromosome == "chr1" else [4.0, 5.0])

    monkeypatch.setattr(conversion_service, "read_bigwig_bins", read_bins)
    request = DataConversionRequest(
        inspection=inspection,
        input_form="track",
        chromosome_scope="all",
        target_format=target_format,
        bin_size=10,
    )
    with pytest.raises(ValueError, match="conversion memory limit is 64 MB"):
        convert_data(request)
    assert read_chromosomes == []

    result = convert_data(replace(request, memory_limit_mb=512))
    assert read_chromosomes == ["chr1", "chr2"]
    assert result.labels == ("chr1", "chr2")
    if target_format == "tsv":
        frame = pd.read_csv(io.BytesIO(result.artifact.data), sep="\t")
        np.testing.assert_allclose(frame["chr1"], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(frame["chr2"], [4.0, 5.0, np.nan])
    else:
        with np.load(io.BytesIO(result.artifact.data), allow_pickle=False) as archive:
            np.testing.assert_allclose(archive["chr1"], [1.0, 2.0, 3.0])
            np.testing.assert_allclose(archive["chr2"], [4.0, 5.0])


@pytest.mark.parametrize("memory_limit_mb", (0, 63, 4097, 64.5, float("nan"), float("inf"), True))
def test_conversion_memory_budget_rejects_invalid_values(memory_limit_mb: Any) -> None:
    inspection = inspect_upload("signal.bedgraph", b"chr1\t0\t10\t1\n")
    with pytest.raises(ValueError, match="Conversion memory limit"):
        prepare_data(
            DataConversionRequest(
                inspection=inspection,
                input_form="track",
                memory_limit_mb=memory_limit_mb,
            )
        )


def test_higher_memory_budget_scales_track_array_and_download_guards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Tiny guards expose secondary limits without allocating large test arrays.
    monkeypatch.setattr(conversion_service, "MAX_ARRAY_ELEMENTS", 5)
    monkeypatch.setattr(conversion_service, "_MAX_TRACK_BINS", 2)
    monkeypatch.setattr(conversion_service, "_MAX_CONVERTED_BYTES", 10)
    inspection = inspect_upload(
        "signal.bedgraph", b"chr1\t0\t30\t1\nchr2\t0\t20\t2\n"
    )
    request = DataConversionRequest(
        inspection=inspection,
        input_form="track",
        chromosome_scope="all",
        bin_size=10,
    )
    with pytest.raises(ValueError, match="creates 3 bins"):
        convert_data(request)
    result = convert_data(replace(request, memory_limit_mb=512))
    frame = pd.read_csv(io.BytesIO(result.artifact.data), sep="\t")
    np.testing.assert_allclose(frame["chr1"], [1.0, 1.0, 1.0])
    np.testing.assert_allclose(frame["chr2"], [2.0, 2.0, np.nan])
    assert len(result.artifact.data) > 10


def test_provenance_is_retained_without_url_credentials_or_query_tokens() -> None:
    inspection = inspect_upload("signal.tsv", b"signal\n1\n2\n")
    request = DataConversionRequest(
        inspection=inspection,
        input_form="x",
        value_column="signal",
        genome_build="GRCh38",
        units="relative signal",
        source_url="https://www.encodeproject.org/files/ENCFF000ABC/@@download/file.bigWig?token=x",
        biological_role="Repli-seq relative timing signal",
        semantic_note="Larger values indicate earlier replication in the source track.",
    )

    result = convert_data(request)

    metadata = dict(result.metadata)
    assert metadata["genome_build"] == "GRCh38"
    assert metadata["units"] == "relative signal"
    assert metadata["source_url"] == (
        "https://www.encodeproject.org/files/ENCFF000ABC/@@download/file.bigWig"
    )
    assert "token" not in metadata["source_url"]
    with pytest.raises(ValueError, match="username or password"):
        convert_data(replace(request, source_url="https://user:secret@example.org/file.bw"))


class _FakeGwseqReader:
    type = "bigwig"

    def __init__(self, path: str) -> None:
        self.path = path
        self.closed = False
        assert not open(path, "rb").read(2) == b"\x1f\x8b"

    def chroms(self) -> dict[str, int]:
        return {"chr1": 25, "chr2": 12}

    def info(self) -> dict[str, int]:
        return {"version": 4, "bases_covered": 20}

    def quantify(
        self,
        _chromosomes: np.ndarray,
        *,
        starts: np.ndarray,
        ends: np.ndarray,
        reduce: str,
        def_value: float,
        zoom: int,
    ) -> np.ndarray:
        assert zoom == -1
        assert reduce == "mean"
        assert np.all(ends > starts)
        mapping = {0: 1.0, 10: np.nan, 20: 3.0}
        return np.array([mapping[int(start)] for start in starts], dtype=float)

    def close(self) -> None:
        self.closed = True


def test_bigwig_without_optional_reader_explains_how_to_import_tracks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def missing_reader(name: str) -> None:
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(bigwig_service.importlib, "import_module", missing_reader)

    with pytest.raises(bigwig_service.BigWigDependencyError) as exc_info:
        bigwig_service.inspect_bigwig(tmp_path / "signal.bw")

    message = str(exc_info.value)
    assert "3.10–3.13" in message
    assert "python -m pip install -r app/requirements.txt" in message
    assert "bedGraph" in message
    assert "gwseq-io-pp" not in message


def test_gwseq_bigwig_fallback_inspects_bins_transforms_and_exports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = SimpleNamespace(open=lambda path: _FakeGwseqReader(path))
    monkeypatch.setattr(bigwig_service, "_load_backend", lambda: ("gwseq_io_pp", module))
    inspection = inspect_upload(
        "repliseq.bw.gz",
        gzip.compress(b"&\xfc\x8f\x88fake-bigwig"),
    )
    request = DataConversionRequest(
        inspection=inspection,
        input_form="track",
        chromosome="chr1",
        start=0,
        end=25,
        bin_size=10,
        interpolation="linear",
        signal_transform="invert",
        scale=2,
        offset=1,
        normalize="minmax",
        value_name="relative_signal",
    )
    events: list[dict[str, object]] = []

    track = inspect_track(inspection)
    prepared = prepare_data(request)
    result = convert_data(request, events.append)

    assert track.backend == "gwseq_io_pp"
    assert tuple((item.chromosome, item.end) for item in track.regions) == (
        ("chr1", 25),
        ("chr2", 12),
    )
    np.testing.assert_allclose(prepared.values, np.array([1.0, 0.5, 0.0]))
    assert dict(result.metadata)["backend"] == "gwseq_io_pp"
    stages = [event["stage"] for event in events]
    assert stages[0] == "load"
    assert stages[-5:] == ["prepare", "prepare", "write", "write", "complete"]
    assert any(event["status"] == "running" for event in events)
    progress_values = [float(str(event["progress"])) for event in events]
    assert progress_values == sorted(progress_values)
    assert result.artifact.data.startswith(b"chr1\n")


def test_gwseq_fallback_rejects_known_incorrect_min_max_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = SimpleNamespace(open=lambda path: _FakeGwseqReader(path))
    monkeypatch.setattr(bigwig_service, "_load_backend", lambda: ("gwseq_io_pp", module))
    source = tmp_path / "signal.bw"
    source.write_bytes(b"&\xfc\x8f\x88fake-bigwig")

    with pytest.raises(ValueError, match="cannot calculate them reliably"):
        bigwig_service.read_bigwig_bins(
            source,
            chromosome="chr1",
            start=0,
            end=20,
            bin_size=10,
            aggregation="min",
        )


def test_extensionless_bigwig_magic_and_manual_bedgraph_hint_are_supported() -> None:
    bigwig = inspect_upload("download", b"&\xfc\x8f\x88payload")
    bedgraph = inspect_upload(
        "download",
        b"chr1\t0\t10\t1\n",
        source_format="bedgraph",
    )

    assert bigwig.source_format == "bigwig"
    assert bedgraph.source_format == "bedgraph"
    with pytest.raises(ValueError, match="not a valid BigWig"):
        inspect_upload("download", b"not a bigwig", source_format="bigwig")


def test_result_preview_is_evenly_bounded_to_configured_point_budget() -> None:
    values = np.arange(2_000, dtype=float)
    stream = io.BytesIO()
    np.save(stream, values, allow_pickle=False)
    inspection = inspect_upload("profile.npy", stream.getvalue())

    result = convert_data(
        DataConversionRequest(
            inspection=inspection,
            input_form="x",
            target_format="npz",
        )
    )

    assert result.preview is not None
    assert result.preview.values.shape == (preview_point_limit(),)
    assert result.preview.total_count == 2_000
    assert result.preview.finite_count == 2_000
    assert result.preview.missing_count == 0
    assert result.preview.values[0] == 0
    assert result.preview.values[-1] == 1_999


def test_invalid_track_transform_and_region_are_rejected() -> None:
    inspection = inspect_upload("signal.bedgraph", b"chr1\t0\t10\t1\n")

    with pytest.raises(ValueError, match="signal transform"):
        prepare_data(
            DataConversionRequest(
                inspection=inspection,
                input_form="track",
                signal_transform="absolute timing",
            )
        )
    with pytest.raises(ValueError, match="available bound"):
        prepare_data(
            DataConversionRequest(
                inspection=inspection,
                input_form="track",
                chromosome="chr1",
                start=0,
                end=11,
            )
        )
