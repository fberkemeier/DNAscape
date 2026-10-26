"""Filter downloads retain the full, pre-edit selection and native coordinates."""

import gzip
import io

import numpy as np
import pytest

from dnascape.workflows.conversion import DataConversionRequest, convert_data, filter_positions_artifact
from dnascape.workflows.filtering import DataFilterSettings, read_filter_profile, detect_filter_profile
from dnascape.workflows.export_settings import preview_point_limit
from dnascape.workflows.uploads import inspect_upload


@pytest.mark.parametrize("format_name", ["tsv", "txt", "csv", "npz"])
@pytest.mark.parametrize("compressed", [False, True])
def test_positions_export_uses_full_pre_edit_selection_in_all_raw_formats(format_name, compressed):
    values = np.ones(10_003)
    values[::17] = -1
    result = convert_data(DataConversionRequest(
        inspection=inspect_upload("source.tsv", ("signal\n" + "\n".join(map(str, values))).encode()),
        input_form="x", data_filter=DataFilterSettings(enabled=True),
    ))
    assert len(result.preview.values) == preview_point_limit()
    assert np.all(result.preview.values >= 0)
    artifact = filter_positions_artifact(result, output_name="selected", target_format=format_name, gzip_output=compressed)
    payload = gzip.decompress(artifact.data) if compressed else artifact.data
    if format_name == "npz":
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            assert archive.files == ["positions"]
            actual = archive["positions"]
            assert actual.ndim == 1 and actual.dtype.kind in "iu"
    else:
        assert all(line.isdigit() for line in payload.decode().splitlines())
        actual = np.loadtxt(io.BytesIO(payload), dtype=np.int64, ndmin=1)
    np.testing.assert_equal(actual, np.arange(0, len(values), 17))
    profile = read_filter_profile(inspect_upload(artifact.filename, artifact.data), layout="positions", units="index")
    np.testing.assert_equal(profile.values, actual)
    assert artifact.filename == f"selected.{format_name}" + (".gz" if compressed else "")


def test_track_position_exports_are_headerless_integer_base_pairs_and_reusable():
    result = convert_data(DataConversionRequest(
        inspection=inspect_upload("track.bedgraph", b"chr1\t10000\t11000\t1\nchr1\t11000\t12100\t-2\nchr2\t0\t2000\t-3\n"),
        input_form="track", chromosome_scope="all", bin_size=1000,
        data_filter=DataFilterSettings(enabled=True),
    ))
    artifact = filter_positions_artifact(result, output_name="filter")
    assert artifact.data == b"500\n1500\n11500\n12050\n"
    profile = read_filter_profile(inspect_upload(artifact.filename, artifact.data), layout="positions", units="bp")
    np.testing.assert_equal(profile.values, [500, 1500, 11500, 12050])
    assert profile.chromosomes is None


def test_matrix_filter_exports_unique_positions_and_do_not_leak_between_results():
    stream = io.BytesIO()
    np.savez(stream, values=[[1, -1], [-2, 2], [3, -3]])
    request = DataConversionRequest(inspection=inspect_upload("matrix.npz", stream.getvalue()), input_form="xt", data_filter=DataFilterSettings(enabled=True))
    artifact = filter_positions_artifact(convert_data(request), output_name="filter")
    assert artifact.data == b"0\n1\n2\n"
    disabled = convert_data(DataConversionRequest(inspection=request.inspection, input_form="xt"))
    assert not disabled.filter_positions
    with pytest.raises(ValueError, match="no applied filter"):
        filter_positions_artifact(disabled, output_name="filter")


def test_integer_position_stays_inside_each_single_base_bin():
    result = convert_data(DataConversionRequest(
        inspection=inspect_upload("track.bedgraph", b"chr1\t0\t3\t-1\n"),
        input_form="track", bin_size=1, data_filter=DataFilterSettings(enabled=True),
    ))
    assert filter_positions_artifact(result, output_name="filter").data == b"0\n1\n2\n"


@pytest.mark.parametrize("format_name", ["tsv", "txt", "csv", "npz"])
@pytest.mark.parametrize("compressed", [False, True])
def test_interval_exports_merge_touching_regions_and_preserve_partial_ends(format_name, compressed):
    inspection = inspect_upload(
        "track.bedgraph", b"chr1\t0\t2000\t-1\nchr1\t2000\t3000\t1\nchr1\t3000\t3500\t-1\n",
    )
    result = convert_data(DataConversionRequest(
        inspection=inspection, input_form="track", bin_size=1000,
        data_filter=DataFilterSettings(enabled=True),
    ))
    artifact = filter_positions_artifact(
        result, output_name="regions", target_format=format_name, gzip_output=compressed, layout="intervals",
    )
    payload = gzip.decompress(artifact.data) if compressed else artifact.data
    if format_name == "npz":
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            assert archive.files == ["intervals"]
            actual = archive["intervals"]
            assert actual.dtype.kind in "iu"
    else:
        actual = np.loadtxt(io.BytesIO(payload), delimiter="," if format_name == "csv" else "\t", dtype=int)
        assert "start" not in payload.decode()
    np.testing.assert_equal(actual, [[0, 2000], [3000, 3500]])
    detected = detect_filter_profile(inspect_upload(artifact.filename, artifact.data), units="bp")
    assert detected.layout == "intervals"
    reused = convert_data(DataConversionRequest(
        inspection=inspection, input_form="track", bin_size=1000,
        data_filter=DataFilterSettings(enabled=True, selection="profile", profile=detected, action="replace", value=7),
    ))
    np.testing.assert_equal(reused.preview.values, [7, 7, 1, 7])


def test_automatic_coordinate_uploads_and_plain_signal_resolution():
    for name, payload, layout, values in (
        ("positions.tsv", b"1\n3\n", "positions", [1, 3]),
        ("regions.csv", b"start,end\n1,3\n", "intervals", [[1, 3]]),
    ):
        detected = detect_filter_profile(inspect_upload(name, payload, expected_form="regions"))
        assert detected.layout == layout
        assert detected.units == "kb"
        np.testing.assert_equal(detected.values, values)
    stream = io.BytesIO()
    np.savez(stream, regions=[[1, 2], [3, 4]])
    detected = detect_filter_profile(inspect_upload("regions.npz", stream.getvalue()))
    assert detected.layout == "intervals"
    for array, expected in (([[1, 2]], "intervals"), ([1, 2], "positions")):
        single = io.BytesIO()
        np.savez(single, selection=array)
        assert detect_filter_profile(inspect_upload("selection.npz", single.getvalue())).layout == expected
    result = convert_data(DataConversionRequest(
        inspection=inspect_upload("signal.tsv", b"signal\n1\n2\n3\n4\n"),
        input_form="x", bin_size=1000,
        data_filter=DataFilterSettings(
            enabled=True, selection="profile", profile=detected, action="replace", value=9,
        ),
    ))
    np.testing.assert_equal(result.preview.values, [1, 9, 3, 9])
    with pytest.raises(ValueError, match="one column"):
        detect_filter_profile(inspect_upload("matrix.tsv", b"1\t2\t3\n4\t5\t6\n"))
    with pytest.raises(ValueError, match="larger ends"):
        detect_filter_profile(inspect_upload("backwards.tsv", b"3\t1\n", expected_form="regions"))


def test_matrix_interval_export_unions_repeated_time_selections():
    stream = io.BytesIO()
    np.savez(stream, values=[[1, -1], [-2, 2], [3, -3]])
    result = convert_data(DataConversionRequest(
        inspection=inspect_upload("matrix.npz", stream.getvalue()), input_form="xt",
        data_filter=DataFilterSettings(enabled=True),
    ))
    assert filter_positions_artifact(result, output_name="filter", layout="intervals").data == b"0\t3\n"
