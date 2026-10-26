"""Histogram files use the same native statistics as the optional guide."""

import csv
import gzip
import io
import lzma
import zipfile

import numpy as np
import pytest

from dnascape.workflows.exports import build_histogram_statistics_export
from dnascape.workflows.preview import PreviewLine, build_preview_plot_options


def histogram(values, **overrides):
    arguments = dict(
        plot_type="Histogram", title="Profile", x_label="Value", y_label="Count",
        color="#b03412", palette="viridis", histogram_bins=5, density=False,
        show_legend=False, show_colorbar=False, log_y=False, invert_y=False,
        y_limits=None, series_label="Signal", values=values, x_axis=np.arange(len(values)),
    )
    return build_preview_plot_options(**(arguments | overrides))


def test_statistics_use_all_native_values_and_survive_hidden_guides_and_kde():
    values = np.r_[np.linspace(1, 9, 2501), 1000, np.nan, np.inf]
    for density in (False, True):
        options = histogram(values, density=density, show_histogram_statistics=False)
        row, = options["dnascapeHistogramStatistics"]
        finite = values[np.isfinite(values)]
        assert row == dict(dataset="Signal", mean=np.mean(finite), median=np.median(finite),
                           standard_deviation=np.std(finite))
        assert options["graphic"] == []


def test_combined_statistics_follow_each_histogram_domain_and_dataset_name():
    values = np.asarray([-2, 1, 2, 4, 9, np.nan])
    lines = (
        PreviewLine("First", np.arange(6.), values, "#b03412"),
        PreviewLine("Second", np.arange(6.), values * 2, "#2596be"),
    )
    options = histogram(values, lines=lines, log_x=True, histogram_x_limits=(1, 5),
                        histogram_grouped=True, show_histogram_statistics=True)
    rows = options["dnascapeHistogramStatistics"]
    assert [row["dataset"] for row in rows] == ["First", "Second"]
    for row, expected in zip(rows, ([1., 2., 4.], [2., 4.])):
        assert row["mean"] == np.mean(expected)
        assert row["median"] == np.median(expected)
        assert row["standard_deviation"] == np.std(expected)
    assert "2.3333" in options["graphic"][0]["style"]["text"]


@pytest.mark.parametrize("file_format", ["tsv", "txt", "csv", "npz"])
@pytest.mark.parametrize("compression", ["none", "gzip", "zip", "xz"])
def test_statistics_files_roundtrip_with_unicode_names_and_full_precision(file_format, compression):
    rows = [
        dict(dataset="Signal α, first", mean=1 / 3, median=2., standard_deviation=np.sqrt(2)),
        dict(dataset="Second", mean=None, median=None, standard_deviation=None),
    ]
    artifact = build_histogram_statistics_export(
        rows, file_format, filename="../statistics", compression=compression,
    )
    assert "/" not in artifact.filename and "\\" not in artifact.filename
    payload = artifact.data
    if compression == "gzip":
        payload = gzip.decompress(payload)
    elif compression == "xz":
        payload = lzma.decompress(payload)
    elif compression == "zip":
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            assert len(archive.namelist()) == 1
            payload = archive.read(archive.namelist()[0])
    if file_format == "npz":
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            assert archive.files == ["dataset", "mean", "median", "standard_deviation"]
            assert archive["dataset"].tolist() == [row["dataset"] for row in rows]
            assert archive["mean"][0] == 1 / 3
            assert archive["standard_deviation"][0] == np.sqrt(2)
            assert np.isnan(archive["mean"][1])
    else:
        decoded = list(csv.DictReader(io.StringIO(payload.decode()), delimiter="," if file_format == "csv" else "\t"))
        assert decoded[0]["dataset"] == rows[0]["dataset"]
        assert float(decoded[0]["mean"]) == 1 / 3
        assert float(decoded[0]["standard_deviation"]) == np.sqrt(2)
        assert decoded[1]["mean"] == "nan"


def test_statistics_reject_removed_formats_and_empty_exports():
    with pytest.raises(ValueError):
        build_histogram_statistics_export([dict(dataset="Signal")], "npy")
    with pytest.raises(ValueError, match="No histogram statistics"):
        build_histogram_statistics_export([], "tsv")
