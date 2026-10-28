"""Exact BigWig bins and interpreter-specific optional readers."""

import base64
import gzip
import re
import subprocess
import sys
import zlib
from concurrent.futures import CancelledError
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from packaging.requirements import Requirement

from dnascape.workflows import bigwig


@pytest.mark.parametrize("version,name", [
    ((3, 10), "pybigtools"), ((3, 11), "pybigtools"),
    ((3, 12), "pybigtools"), ((3, 13), "pybigtools"),
    ((3, 14), "figwig"), ((3, 15), "figwig"),
])
def test_reader_selection_matches_dependency_markers(monkeypatch, version, name):
    calls = []
    module = object()
    monkeypatch.setattr(bigwig.sys, "version_info", version)
    monkeypatch.setattr(bigwig.importlib, "import_module", lambda n: calls.append(n) or module)
    assert bigwig._load_backend() == (name, module)
    assert calls == [name]
    project = (Path(__file__).parents[2] / "pyproject.toml").read_text()
    readers = [Requirement(value) for value in re.findall(r'"((?:pybigtools|figwig)[^"]+)"', project)]
    selected = [reader.name for reader in readers
                if reader.marker.evaluate({"python_version": ".".join(map(str, version))})]
    assert selected == [name, name]  # Both app and test installation extras.


@pytest.fixture
def small_bigwig(tmp_path):
    # A real little-endian BigWig, written with pybigtools: chr1 has measured
    # zero/negative values, gaps and a non-divisible end; chr2 has a singleton.
    payload = (
        "eJxT+9PfwcLAwLCPkQEMIqA0GxMDBjCAyoVgSo0YMBeFx3EASCg8EwtyYGBw8GMqOOxQcnC"
        "v86Fnqo7MUBUVc5IbmDLZWBJSElgvnxFYx9zR1mFiUcDoKbfCSuAsf2CRg0ximzKDJzPTRaD"
        "SlAaGMrDapksKHw40curd3yDPIJzNVAqUS2po8APKsTjNmPHwG6cfg4Qk08WJPacqGICxAop"
        "BDiBmZkAFjECR5IwiQxBbGoiBbCNQJLIDcVpOYgkokjuB+MGaDBWQOej6YfLw1MCCMBcZgM"
        "xOgKYObbjdCBpkXzdUQAbJbGQ7lkPlxaHiasCUCQCT8j0U"
    )
    path = tmp_path / "signal.bw"
    path.write_bytes(zlib.decompress(base64.b64decode(payload)))
    return path


def _reference(values, start, end, width, aggregation):
    result = []
    for first in range(start, end, width):
        section = values[first:min(first + width, end)]
        valid = section[~np.isnan(section)]
        if aggregation == "coverage":
            result.append(len(valid) / len(section))
        elif not len(valid):
            result.append(np.nan)
        else:
            result.append({"mean": np.mean, "sum": np.sum, "min": np.min, "max": np.max,
                           "std": lambda x: np.std(x, ddof=1) if len(x) > 1 else 0.0}[aggregation](valid))
    return np.array(result)


@pytest.mark.parametrize("aggregation", ["mean", "sum", "min", "max", "std", "coverage"])
@pytest.mark.parametrize("start,end,width", [(0, 27, 1), (0, 27, 10), (2, 26, 4), (0, 27, 64)])
def test_installed_reader_matches_exact_per_base_reference(
    monkeypatch, small_bigwig, aggregation, start, end, width,
):
    pytest.importorskip("figwig" if sys.version_info >= (3, 14) else "pybigtools")
    # Exercise both chunks of bins and chunks within one bin on real readers.
    monkeypatch.setattr(bigwig, "_FIGWIG_CHUNK_BASES", 7)
    monkeypatch.setattr(bigwig, "_FIGWIG_CHUNK_BINS", 2)
    metadata = bigwig.inspect_bigwig(small_bigwig)
    assert metadata.chromosomes == (("chr1", 27), ("chr2", 7), ("flat", 137))
    values = np.full(27, np.nan)
    for lo, hi, value in [(0, 3, 2), (3, 5, 4), (10, 12, 0), (12, 15, -3), (23, 27, 8)]:
        values[lo:hi] = value
    progress = []
    result = bigwig.read_bigwig_bins(
        small_bigwig, chromosome="chr1", start=start, end=end,
        bin_size=width, aggregation=aggregation, progress=progress.append,
    )
    np.testing.assert_allclose(result, _reference(values, start, end, width, aggregation),
                               rtol=1e-12, atol=1e-12, equal_nan=True)
    assert progress[-1] == 1
    assert progress == sorted(progress)


@pytest.mark.parametrize("aggregation", ["mean", "sum", "min", "max", "std", "coverage"])
def test_figwig_bounds_temporary_reads_and_preserves_numerical_precision(monkeypatch, aggregation):
    signal = np.array([1e8, 1e8 + 8, np.nan, 1e8 - 8, 0, -2, np.nan] * 13, dtype=np.float32)
    reads = []

    def read(chromosome, starts, *, width, missing, keep_nan, n_jobs):
        assert chromosome == "chr1" and np.isnan(missing) and keep_nan
        assert width <= 7 and n_jobs == 2
        reads.append((starts[0], width))
        return signal[starts[0]:starts[0] + width][None, :]

    monkeypatch.setattr(bigwig, "_FIGWIG_CHUNK_BASES", 7)
    handle = SimpleNamespace(read=read)
    for width in (1, 3, 13):
        end = len(signal) // width * width
        progress = []
        actual = bigwig._query_figwig(handle, "chr1", 0, end, end // width,
                                      aggregation, progress.append)
        expected = _reference(signal.astype(float), 0, end, width, aggregation)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-8, equal_nan=True)
        assert progress[0] == 0 and progress[-1] == 1
    assert len(reads) > 1


def test_figwig_cancellation_propagates_between_reads(monkeypatch):
    reads = []
    monkeypatch.setattr(bigwig, "_FIGWIG_CHUNK_BASES", 4)

    def read(*args, width, **kwargs):
        reads.append(width)
        return np.ones((1, width))

    def progress(fraction):
        if fraction >= .25:
            raise CancelledError("cancelled")

    with pytest.raises(CancelledError, match="cancelled"):
        bigwig._query_figwig(SimpleNamespace(read=read), "chr1", 0, 16, 1, "mean", progress)
    assert reads == [4]


def test_figwig_import_is_deferred_until_a_reader_is_needed():
    # The module itself must not pull in Numba/LLVM during app startup.
    subprocess.run([sys.executable, "-B", "-c", "import sys; import dnascape.workflows.bigwig; "
                    "assert not {'figwig', 'numba', 'llvmlite'}.intersection(sys.modules)"], check=True)


def test_figwig_rejects_nonfinite_values_instead_of_hiding_them_in_a_summary():
    handle = SimpleNamespace(read=lambda *args, **kwargs: np.array([[np.inf, 0.0]]))
    with pytest.raises(ValueError, match="infinite values"):
        bigwig._query_figwig(handle, "chr1", 0, 2, 1, "std")


def test_installed_reader_rejects_invalid_regions(small_bigwig):
    pytest.importorskip("figwig" if sys.version_info >= (3, 14) else "pybigtools")
    with pytest.raises(KeyError, match="not found"):
        bigwig.read_bigwig_bins(small_bigwig, chromosome="missing", start=0, end=1,
                                bin_size=1, aggregation="mean")
    with pytest.raises(ValueError, match="outside"):
        bigwig.read_bigwig_bins(small_bigwig, chromosome="chr2", start=0, end=8,
                                bin_size=1, aggregation="mean")


def test_constant_noninteger_signal_has_zero_sd_with_either_reader(small_bigwig):
    pytest.importorskip("figwig" if sys.version_info >= (3, 14) else "pybigtools")
    values = bigwig.read_bigwig_bins(small_bigwig, chromosome="flat", start=0, end=137,
                                     bin_size=137, aggregation="std")
    np.testing.assert_array_equal(values, [0.0])


def test_public_reader_propagates_conversion_cancellation(monkeypatch, small_bigwig):
    pytest.importorskip("figwig" if sys.version_info >= (3, 14) else "pybigtools")
    monkeypatch.setattr(bigwig, "_FIGWIG_CHUNK_BASES", 7)

    def stop(fraction):
        if fraction > 0:
            raise CancelledError("Conversion stopped.")

    with pytest.raises(CancelledError, match="Conversion stopped"):
        bigwig.read_bigwig_bins(small_bigwig, chromosome="chr1", start=0, end=27,
                                bin_size=10, aggregation="std", progress=stop)


@pytest.mark.parametrize("target", ["tsv", "txt", "csv", "npz"])
def test_actual_reader_prepares_and_exports_compressed_uploads(small_bigwig, target):
    pytest.importorskip("figwig" if sys.version_info >= (3, 14) else "pybigtools")
    from dnascape.workflows.conversion import DataConversionRequest, convert_data, prepare_data
    from dnascape.workflows.uploads import inspect_upload

    upload = inspect_upload("signal.bw.gz", gzip.compress(small_bigwig.read_bytes()))
    request = DataConversionRequest(inspection=upload, input_form="track", chromosome="chr1",
                                    start=0, end=27, bin_size=10, target_format=target)
    prepared = prepare_data(request)
    np.testing.assert_allclose(prepared.values, [2.8, -1.8, 8.0])
    events = []
    result = convert_data(request, events.append)
    assert result.shape == (3,)
    assert result.artifact.data
    assert result.target_format == target
    assert dict(result.metadata)["backend"] == ("figwig" if sys.version_info >= (3, 14) else "pybigtools")
    assert events[-1]["stage"] == "complete"
