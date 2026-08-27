"""Tests for the Data Manager's bounded ``rdata`` service."""

from __future__ import annotations

import gzip
import io

import numpy as np

from app.data_conversion_service import DataConversionRequest, convert_data, inspect_track
from app.mapping_uploads import inspect_upload


def test_profile_conversion_changes_space_to_time_and_writes_gzipped_npz() -> None:
    inspection = inspect_upload(
        "signal.csv",
        b"position,signal\n0,1\n1,3\n2,2\n",
    )

    result = convert_data(
        DataConversionRequest(
            inspection=inspection,
            input_form="x",
            output_form="time",
            target_format="npz",
            output_name="timing profile",
            gzip_output=True,
            value_column="signal",
        )
    )

    assert result.artifact.filename == "timing_profile.npz.gz"
    assert result.artifact.media_type == "application/gzip"
    assert result.data_form == "time"
    assert result.shape == (1, 3)
    with np.load(io.BytesIO(gzip.decompress(result.artifact.data)), allow_pickle=False) as archive:
        assert str(archive["data_form"]) == "time"
        np.testing.assert_allclose(archive["values"], np.array([[1.0, 3.0, 2.0]]))


def test_matrix_conversion_preserves_space_time_shape_and_labels() -> None:
    inspection = inspect_upload(
        "field.csv",
        b"position,t0,t1\n0,0.1,0.2\n1,0.3,0.4\n",
    )

    result = convert_data(
        DataConversionRequest(
            inspection=inspection,
            input_form="xt",
            output_form="spacetime",
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


def test_full_table_conversion_preserves_columns() -> None:
    inspection = inspect_upload(
        "table.csv",
        b"sample,signal\na,1.25\nb,2.5\n",
    )

    result = convert_data(
        DataConversionRequest(
            inspection=inspection,
            input_form="table",
            output_form="table",
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
            output_form="space",
            target_format="npy",
            gzip_output=False,
            value_column="signal",
            missing_policy="ignore",
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
            output_form="space",
            target_format="tsv",
            gzip_output=False,
            chromosome="chr1",
            start=0,
            end=30,
            bin_size=10,
            track_missing="zero",
        )
    )

    assert tuple((item.chromosome, item.start, item.end) for item in track.regions) == (
        ("chr1", 0, 30),
        ("chr2", 0, 5),
    )
    assert result.data_form == "space"
    assert result.artifact.data.startswith(b"chrom\tstart_bp\tend_bp")
