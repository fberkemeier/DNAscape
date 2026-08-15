"""Regression tests for lazy, framework-independent simulation exports."""

from __future__ import annotations

import gzip
import io

import numpy as np
import pytest

import app.simulation_exports as simulation_exports
from app.simulation_exports import (
    build_plot_export,
    build_raw_export,
    safe_export_filename,
    select_result_output,
)
from app.simulation_service import ConstantSimulationResult


def _result() -> ConstantSimulationResult:
    return ConstantSimulationResult(
        position_kb=np.array([10.0, 11.0, 12.0]),
        replication_timing=np.array([18.0, 14.0, 20.0]),
        elapsed_seconds=0.1,
        replicates=2,
        geometry="linear",
        time_grid=np.array([0.0, 5.0]),
        outputs={
            "replication_timing": np.array([18.0, 14.0, 20.0]),
            "replicated_fraction": np.array([0.0, 0.75]),
            "completion_time": np.array([18.0, 21.0]),
            # DNAscape's canonical detailed shape is (position, time).
            "replicated_fraction_xt": np.array(
                [
                    [0.0, 0.8],
                    [0.0, 0.7],
                    [0.0, 0.75],
                ]
            ),
            "custom_time_output": np.array([3.0, 4.0]),
        },
        detailed_outputs=True,
    )


def test_output_selection_infers_space_time_replicate_and_xt_axes() -> None:
    result = _result()

    space = select_result_output(result, "replication_timing")
    assert space.axis_kind == "space"
    np.testing.assert_array_equal(space.x, result.position_kb)

    time = select_result_output(result, "replicated_fraction")
    assert time.axis_kind == "time"
    np.testing.assert_array_equal(time.x, result.time_grid)

    replicate = select_result_output(result, "completion_time")
    assert replicate.axis_kind == "replicate"
    np.testing.assert_array_equal(replicate.x, [1.0, 2.0])

    space_time = select_result_output(result, "replicated_fraction_xt")
    assert space_time.axis_kind == "xt"
    assert space_time.values.shape == (2, 3)
    np.testing.assert_array_equal(space_time.values, result.outputs["replicated_fraction_xt"].T)
    np.testing.assert_array_equal(space_time.x, result.position_kb)
    np.testing.assert_array_equal(space_time.y, result.time_grid)


def test_explicit_axis_kind_resolves_unknown_output_ambiguity() -> None:
    result = _result()
    with pytest.raises(ValueError, match="unambiguous axis"):
        select_result_output(result, "custom_time_output")

    selected = select_result_output(result, "custom_time_output", axis_kind="time")
    assert selected.axis_kind == "time"
    np.testing.assert_array_equal(selected.x, result.time_grid)


@pytest.mark.parametrize(
    ("file_format", "signature", "media_type"),
    (
        ("png", b"\x89PNG\r\n\x1a\n", "image/png"),
        ("svg", b"<?xml", "image/svg+xml"),
        ("pdf", b"%PDF", "application/pdf"),
    ),
)
def test_plot_exports_are_download_ready(
    file_format: str,
    signature: bytes,
    media_type: str,
) -> None:
    artifact = build_plot_export(
        _result(),
        "replication_timing",
        file_format,
        filename="../plots/My result.csv.gz",
        title="Constant-input simulation",
        x_label="Genomic position (kb)",
        y_label="Replication time (min)",
        color="#b03412",
        invert_y=True,
        y_limits=(10.0, 24.0),
        figure_size=(4.0, 3.0),
        dpi=100,
    )

    assert artifact.data.startswith(signature)
    assert artifact.filename == f"My_result.{file_format}"
    assert artifact.media_type == media_type


def test_xt_plot_export_accepts_palette_and_normalizes_orientation() -> None:
    artifact = build_plot_export(
        _result(),
        "replicated_fraction_xt",
        "png",
        cmap="magma",
        figure_size=(4.0, 3.0),
        dpi=100,
    )
    assert artifact.data.startswith(b"\x89PNG")
    assert len(artifact.data) > 1_000


def test_histogram_mode_and_distribution_outputs_use_histogram_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from matplotlib.axes import Axes

    calls: list[tuple[int, bool]] = []
    original_hist = Axes.hist

    def recording_hist(self, values, *args, **kwargs):
        calls.append((kwargs["bins"], kwargs["density"]))
        return original_hist(self, values, *args, **kwargs)

    monkeypatch.setattr(Axes, "hist", recording_hist)
    build_plot_export(
        _result(),
        "replication_timing",
        "svg",
        view_mode="Histogram",
        histogram_bins=7,
        density=True,
        figure_size=(4.0, 3.0),
        dpi=100,
    )
    assert calls == [(7, True)]

    calls.clear()
    build_plot_export(
        _result(),
        "completion_time",
        "svg",
        view_mode="Single view",
        histogram_bins=5,
        figure_size=(4.0, 3.0),
        dpi=100,
    )
    assert calls == [(5, False)]


def test_matplotlib_objects_are_not_imported_into_module_namespace() -> None:
    assert "matplotlib" not in simulation_exports.__dict__
    assert "Figure" not in simulation_exports.__dict__
    assert "FigureCanvasAgg" not in simulation_exports.__dict__
    assert "grid" not in build_plot_export.__code__.co_names


@pytest.mark.parametrize(
    ("file_format", "delimiter", "media_type"),
    (
        ("csv", ",", "text/csv; charset=utf-8"),
        ("tsv", "\t", "text/tab-separated-values; charset=utf-8"),
        ("txt", " ", "text/plain; charset=utf-8"),
    ),
)
def test_text_raw_exports_include_axis_and_value_columns(
    file_format: str,
    delimiter: str,
    media_type: str,
) -> None:
    artifact = build_raw_export(
        _result(),
        "replication_timing",
        file_format,
        filename="simulation result",
    )
    lines = artifact.data.decode("utf-8").splitlines()
    assert lines[0] == delimiter.join(("position_kb", "replication_timing"))
    assert lines[1] == delimiter.join(("10", "18"))
    assert artifact.filename == f"simulation_result.{file_format}"
    assert artifact.media_type == media_type


def test_xt_text_export_is_long_form_with_position_and_time() -> None:
    artifact = build_raw_export(_result(), "replicated_fraction_xt", "csv")
    lines = artifact.data.decode("utf-8").splitlines()
    assert lines[0] == "position_kb,time_min,replicated_fraction_xt"
    assert lines[1:4] == ["10,0,0", "11,0,0", "12,0,0"]
    assert lines[4:] == ["10,5,0.8", "11,5,0.7", "12,5,0.75"]


def test_npy_and_npz_exports_round_trip_without_pickle() -> None:
    result = _result()
    npy = build_raw_export(result, "replication_timing", "npy")
    np.testing.assert_array_equal(
        np.load(io.BytesIO(npy.data), allow_pickle=False),
        result.replication_timing,
    )

    npz = build_raw_export(result, "replicated_fraction_xt", "npz")
    with np.load(io.BytesIO(npz.data), allow_pickle=False) as archive:
        np.testing.assert_array_equal(
            archive["values"],
            result.outputs["replicated_fraction_xt"].T,
        )
        np.testing.assert_array_equal(archive["position_kb"], result.position_kb)
        np.testing.assert_array_equal(archive["time_min"], result.time_grid)


def test_optional_gzip_wraps_raw_bytes_and_suffix() -> None:
    plain = build_raw_export(_result(), "replication_timing", "csv", filename="result")
    compressed = build_raw_export(
        _result(),
        "replication_timing",
        ".CSV",
        filename="result.csv",
        gzip_output=True,
    )
    assert gzip.decompress(compressed.data) == plain.data
    assert compressed.filename == "result.csv.gz"
    assert compressed.media_type == "application/gzip"


@pytest.mark.parametrize(
    ("filename", "extension", "gzip_output", "expected"),
    (
        (r"../../folder\\plot final.PDF", "png", False, "plot_final.png"),
        ("CON", "csv", False, "_CON.csv"),
        ("data.npy.gz", "npz", True, "data.npz.gz"),
        ("   ", "svg", False, "dnascape_simulation.svg"),
    ),
)
def test_safe_filename_removes_paths_reserved_names_and_stale_suffixes(
    filename: str,
    extension: str,
    gzip_output: bool,
    expected: str,
) -> None:
    assert safe_export_filename(filename, extension, gzip_output=gzip_output) == expected


def test_raw_export_limits_recommend_safe_binary_fallbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(simulation_exports, "MAX_TEXT_EXPORT_VALUES", 2)
    with pytest.raises(ValueError, match="Use NPY or NPZ instead"):
        build_raw_export(_result(), "replication_timing", "csv")

    monkeypatch.setattr(simulation_exports, "MAX_BINARY_EXPORT_BYTES", 1)
    with pytest.raises(ValueError, match="Binary export would contain"):
        build_raw_export(_result(), "replication_timing", "npy")


def test_invalid_formats_plot_bounds_and_missing_outputs_are_clear() -> None:
    result = _result()
    with pytest.raises(ValueError, match="Unsupported export format"):
        build_raw_export(result, "replication_timing", "xlsx")
    with pytest.raises(ValueError, match="Unsupported export format"):
        build_plot_export(result, "replication_timing", "jpg")
    with pytest.raises(ValueError, match="pixel limit"):
        build_plot_export(
            result,
            "replication_timing",
            "png",
            figure_size=(20.0, 20.0),
            dpi=600,
        )
    with pytest.raises(ValueError, match="View mode must be"):
        build_plot_export(result, "replication_timing", "png", view_mode="Scatter")
    with pytest.raises(ValueError, match="Histogram bins"):
        build_plot_export(result, "replication_timing", "png", histogram_bins=0)
    with pytest.raises(ValueError, match="is not available"):
        build_raw_export(result, "missing", "csv")
