"""Filter downloads retain the full, pre-edit selection and native coordinates."""

import gzip
import io

import numpy as np
import pytest

from dnascape.workflows.conversion import DataConversionRequest, convert_data, filter_positions_artifact
from dnascape.workflows.filtering import DataFilterSettings, read_filter_profile
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
