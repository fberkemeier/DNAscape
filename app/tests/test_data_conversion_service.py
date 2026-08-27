"""Tests for the Data Manager's bounded ``rdata`` service."""

from __future__ import annotations

import gzip
import io
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

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
            target_format="npy",
            gzip_output=False,
        )
    )

    assert result.data_form == "spacetime"
    assert result.shape == (2, 2)
    assert result.labels == ("t0", "t1")
    np.testing.assert_allclose(
        np.load(io.BytesIO(result.artifact.data), allow_pickle=False),
        np.array([[0.1, 0.2], [0.3, 0.4]]),
    )


def test_profile_custom_rescale_can_reverse_range_without_reordering_samples() -> None:
    inspection = inspect_upload("signal.tsv", b"signal\n1\n2\n3\n")
    request = DataConversionRequest(
        inspection=inspection,
        input_form="x",
        target_format="npy",
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
        np.load(io.BytesIO(result.artifact.data), allow_pickle=False),
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
        target_format="npy",
        value_column="signal",
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
        np.load(io.BytesIO(result.artifact.data), allow_pickle=False),
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
            rescale_invert=True,
        )
    )

    np.testing.assert_allclose(prepared.values, np.array([1.0, 2.0, 3.0]))
    metadata = dict(prepared.metadata)
    assert metadata["value_scaling"] == "none"
    assert "scale_multiplier" not in metadata
    assert "target_minimum" not in metadata
    assert "legacy_offset" not in metadata
    assert "legacy_signal_transform" not in metadata


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
        target_format="npy",
        value_column="signal",
        threshold_enabled=True,
        threshold_min=10,
        scaling_mode="interval",
        rescale_min=-1,
        rescale_max=1,
    )

    result = convert_data(request)

    values = np.load(io.BytesIO(result.artifact.data), allow_pickle=False)
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
            target_format="npy",
            gzip_output=False,
            value_column="signal",
        )
    )

    values = np.load(io.BytesIO(result.artifact.data), allow_pickle=False)
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
            target_format="npy",
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
        "csv",
        "txt",
        "npy",
        "npv",
        "npz",
    )
    assert result.artifact.filename == "dnascape_data.tsv"
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
    np.testing.assert_allclose(result.preview.x, np.array([0.005, 0.015, 0.025]))


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
            value_name="firing_rate",
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


@pytest.mark.parametrize("target_format", ("txt", "npy", "npv"))
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


def test_track_default_origin_includes_missing_leading_chromosome_bins() -> None:
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
    assert np.isnan(prepared.values[0])
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
    np.testing.assert_allclose(prepared.values, np.array([4.0, 2.0, np.nan]), equal_nan=True)
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
    inspection = inspect_upload("signal.bedgraph", b"chr1\t0\t100\t1\n")
    monkeypatch.setattr(conversion_service, "MAX_DECOMPRESSED_BYTES", 200)

    with pytest.raises(ValueError, match="would require about"):
        prepare_data(
            DataConversionRequest(
                inspection=inspection,
                input_form="track",
                chromosome="chr1",
                start=0,
                end=100,
                bin_size=10,
            )
        )


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


def test_result_preview_is_evenly_bounded_to_five_hundred_points() -> None:
    values = np.arange(2_000, dtype=float)
    stream = io.BytesIO()
    np.save(stream, values, allow_pickle=False)
    inspection = inspect_upload("profile.npy", stream.getvalue())

    result = convert_data(
        DataConversionRequest(
            inspection=inspection,
            input_form="x",
            target_format="npy",
        )
    )

    assert result.preview is not None
    assert result.preview.values.shape == (500,)
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
