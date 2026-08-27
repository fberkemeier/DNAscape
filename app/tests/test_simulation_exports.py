"""Regression tests for lazy, framework-independent simulation exports."""

from __future__ import annotations

import base64
import gzip
import io
import lzma
import zipfile
from pathlib import Path

import numpy as np
import pytest

import app.simulation_exports as simulation_exports
from app.simulation_exports import (
    build_chart_capture_export,
    build_plot_export,
    build_raw_export,
    safe_export_filename,
    select_result_output,
)
from app.simulation_service import ConstantSimulationResult
from dnascape.io import rload


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
            "origin_efficiency": np.array([0.2, 0.6, 0.2]),
            "fork_directionality": np.array([-0.5, 0.0, 0.5]),
            "replicated_fraction": np.array([0.0, 0.75]),
            "total_forks": np.array([4.0, 7.0]),
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


def _png_data_uri() -> tuple[str, bytes]:
    from PIL import Image

    output = io.BytesIO()
    Image.new("RGB", (3, 2), "white").save(output, format="PNG")
    payload = output.getvalue()
    return f"data:image/png;base64,{base64.b64encode(payload).decode('ascii')}", payload


@pytest.mark.parametrize(
    ("file_format", "signature", "media_type"),
    (
        ("png", b"\x89PNG\r\n\x1a\n", "image/png"),
        ("svg", b"<?xml", "image/svg+xml"),
        ("pdf", b"%PDF", "application/pdf"),
    ),
)
def test_chart_capture_exports_package_the_exact_preview_image(
    file_format: str,
    signature: bytes,
    media_type: str,
) -> None:
    data_uri, png_payload = _png_data_uri()
    artifact = build_chart_capture_export(
        data_uri,
        file_format,
        filename="visible preview",
        dpi=150,
    )

    assert artifact.data.startswith(signature)
    assert artifact.filename == f"visible_preview.{file_format}"
    assert artifact.media_type == media_type
    if file_format == "png":
        assert artifact.data == png_payload
    elif file_format == "svg":
        assert base64.b64encode(png_payload) in artifact.data
        assert b'width="3" height="2"' in artifact.data


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


def test_automatic_y_limits_follow_the_selected_data_and_view() -> None:
    from app.pages.simulation import _automatic_y_limits, _default_axis_labels

    result = _result()
    line_limits = _automatic_y_limits(
        result,
        output_label="Replication timing",
        view_mode="Default",
        histogram_bins=60,
        density=False,
        log_y=False,
    )
    assert line_limits[0] < float(np.min(result.replication_timing))
    assert line_limits[1] > float(np.max(result.replication_timing))

    histogram_limits = _automatic_y_limits(
        result,
        output_label="Replication timing",
        view_mode="Histogram",
        histogram_bins=8,
        density=False,
        log_y=False,
    )
    assert histogram_limits[0] == 0.0
    assert histogram_limits[1] > 0.0

    log_limits = _automatic_y_limits(
        result,
        output_label="Replicated fraction",
        view_mode="Default",
        histogram_bins=60,
        density=False,
        log_y=True,
    )
    assert 0.0 < log_limits[0] < log_limits[1]

    assert _default_axis_labels(
        output_label="Replication timing",
        view_mode="Histogram",
        density=False,
    ) == ("Replication timing", "Count")
    assert _default_axis_labels(
        output_label="S-phase duration",
        view_mode="Default",
        density=True,
    ) == ("Duration (min)", "Density")


def test_output_defaults_use_scientific_domains_and_clean_dynamic_ticks() -> None:
    from app.pages.simulation import (
        _automatic_y_limits,
        _default_y_limits,
        _result_plot_options,
    )

    result = _result()
    for output_label, expected in (
        ("Fork directionality", (-1.0, 1.0)),
        ("Origin efficiency", (0.0, 1.0)),
        ("Replicated fraction", (0.0, 1.0)),
    ):
        assert _default_y_limits(output_label) == expected
        assert (
            _automatic_y_limits(
                result,
                output_label=output_label,
                view_mode="Default",
                histogram_bins=60,
                density=False,
                log_y=False,
            )
            == expected
        )

    assert _default_y_limits("Replication timing") is None
    automatic_limits = _automatic_y_limits(
        result,
        output_label="Replication timing",
        view_mode="Default",
        histogram_bins=60,
        density=False,
        log_y=False,
    )
    assert 0.0 < automatic_limits[0] < float(np.min(result.replication_timing))
    assert automatic_limits[1] > float(np.max(result.replication_timing))
    assert all(float(value).is_integer() for value in automatic_limits)

    automatic_options = _result_plot_options(
        result,
        output_label="Replication timing",
        title="Replication timing",
        series_label="Mean",
        x_axis_label="Position (kb)",
        y_axis_label="Replication timing",
        color="#2c7f77",
        palette="viridis",
        view_mode="Default",
        histogram_bins=60,
        density=False,
        show_legend=True,
        show_colorbar=True,
        log_y=False,
        invert_y=False,
        y_limits=automatic_limits,
    )
    assert automatic_options["yAxis"]["min"] == automatic_limits[0]
    assert automatic_options["yAxis"]["max"] == automatic_limits[1]

    manual_options = _result_plot_options(
        result,
        output_label="Replication timing",
        title="Replication timing",
        series_label="Mean",
        x_axis_label="Position (kb)",
        y_axis_label="Replication timing",
        color="#2c7f77",
        palette="viridis",
        view_mode="Default",
        histogram_bins=60,
        density=False,
        show_legend=True,
        show_colorbar=True,
        log_y=False,
        invert_y=False,
        y_limits=(4.0, 12.0),
    )
    assert manual_options["yAxis"]["min"] == 4.0
    assert manual_options["yAxis"]["max"] == 12.0


def test_fixed_quantity_domains_do_not_leak_into_histogram_log_or_xt_axes() -> None:
    from app.pages.simulation import _automatic_y_limits, _default_y_limits

    result = _result()

    assert _default_y_limits("Fork directionality", view_mode="Histogram") is None
    histogram_limits = _automatic_y_limits(
        result,
        output_label="Fork directionality",
        view_mode="Histogram",
        histogram_bins=3,
        density=False,
        log_y=False,
    )
    assert histogram_limits[0] == 0.0
    assert histogram_limits[1] > 0.0
    assert histogram_limits != (-1.0, 1.0)

    assert _default_y_limits("Origin efficiency", log_y=True) is None
    log_limits = _automatic_y_limits(
        result,
        output_label="Origin efficiency",
        view_mode="Default",
        histogram_bins=60,
        density=False,
        log_y=True,
    )
    assert 0.0 < log_limits[0] < log_limits[1]

    assert _default_y_limits("Replicated fraction (space-time)") is None
    xt_limits = _automatic_y_limits(
        result,
        output_label="Replicated fraction (space-time)",
        view_mode="Default",
        histogram_bins=60,
        density=False,
        log_y=False,
    )
    assert xt_limits[0] <= float(np.min(result.time_grid))
    assert xt_limits[1] >= float(np.max(result.time_grid))


def test_nonnegative_count_outputs_auto_floor_at_zero_except_on_log_axes() -> None:
    from app.pages.simulation import _automatic_y_limits

    result = _result()
    linear_limits = _automatic_y_limits(
        result,
        output_label="Total forks",
        view_mode="Default",
        histogram_bins=60,
        density=False,
        log_y=False,
    )
    assert linear_limits[0] == 0.0
    assert linear_limits[1] > float(np.max(result.outputs["total_forks"]))

    log_limits = _automatic_y_limits(
        result,
        output_label="Total forks",
        view_mode="Default",
        histogram_bins=60,
        density=False,
        log_y=True,
    )
    assert 0.0 < log_limits[0] < log_limits[1]


@pytest.mark.parametrize(
    ("output_label", "view_mode", "series_type"),
    (
        ("Replication timing", "Default", "line"),
        ("S-phase duration", "Default", "bar"),
        ("Replicated fraction (space-time)", "Default", "custom"),
    ),
)
def test_default_view_resolves_by_output_and_inversion_keeps_x_axis_at_bottom(
    output_label: str,
    view_mode: str,
    series_type: str,
) -> None:
    from app.pages.simulation import _result_plot_options

    options = _result_plot_options(
        _result(),
        output_label=output_label,
        title=output_label,
        series_label="Mean",
        x_axis_label="",
        y_axis_label="",
        color="#2c7f77",
        palette="viridis",
        view_mode=view_mode,
        histogram_bins=8,
        density=False,
        show_legend=True,
        show_colorbar=True,
        log_y=False,
        invert_y=True,
        y_limits=None,
    )

    assert options["series"][0]["type"] == series_type
    assert options["yAxis"]["inverse"] is True
    assert options["xAxis"]["position"] == "bottom"
    if output_label == "Replicated fraction (space-time)":
        assert options["xAxis"]["name"] == "Position (kb)"
        assert options["yAxis"]["name"] == "Time (min)"
        assert options["series"][0]["silent"] is True
        assert options["tooltip"]["show"] is False


def test_time_profiles_are_static_and_keep_rounded_numeric_axes() -> None:
    from app.pages.simulation import _result_plot_options

    options = _result_plot_options(
        _result(),
        output_label="Replicated fraction",
        title="Replicated fraction",
        series_label="Mean",
        x_axis_label="",
        y_axis_label="",
        color="#2c7f77",
        palette="viridis",
        view_mode="Default",
        histogram_bins=8,
        density=False,
        show_legend=True,
        show_colorbar=True,
        log_y=False,
        invert_y=False,
        y_limits=None,
    )

    assert options["series"][0]["type"] == "line"
    assert options["series"][0]["silent"] is True
    assert options["tooltip"]["show"] is False
    assert ":formatter" in options["xAxis"]["axisLabel"]
    assert ":formatter" in options["yAxis"]["axisLabel"]


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


def test_plot_export_preserves_preview_titles_labels_and_series_name() -> None:
    artifact = build_plot_export(
        _result(),
        "replication_timing",
        "svg",
        title="Preview parity title",
        series_label="Preview series",
        x_label="Preview X axis",
        y_label="Preview Y axis",
        show_legend=True,
        log_y=True,
        y_limits=(10.0, 24.0),
        figure_size=(4.0, 3.0),
        dpi=100,
    )

    svg = artifact.data.decode("utf-8")
    for visible_text in (
        "Preview parity title",
        "Preview series",
        "Preview X axis",
        "Preview Y axis",
    ):
        assert visible_text in svg


def test_xt_plot_export_respects_preview_colorbar_visibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from matplotlib.figure import Figure

    colorbar_calls = 0
    original_colorbar = Figure.colorbar

    def recording_colorbar(self, *args, **kwargs):
        nonlocal colorbar_calls
        colorbar_calls += 1
        return original_colorbar(self, *args, **kwargs)

    monkeypatch.setattr(Figure, "colorbar", recording_colorbar)
    build_plot_export(
        _result(),
        "replicated_fraction_xt",
        "svg",
        show_colorbar=False,
        figure_size=(4.0, 3.0),
        dpi=100,
    )
    assert colorbar_calls == 0

    build_plot_export(
        _result(),
        "replicated_fraction_xt",
        "svg",
        show_colorbar=True,
        figure_size=(4.0, 3.0),
        dpi=100,
    )
    assert colorbar_calls == 1


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
        view_mode="Default",
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
    ),
)
def test_tabular_raw_exports_follow_rdata_space_profile_layout(
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
    assert lines[0] == "replication_timing"
    assert lines[1:] == ["18", "14", "20"]
    assert artifact.filename == f"simulation_result.{file_format}"
    assert artifact.media_type == media_type


def test_txt_raw_export_is_headerless_numeric_rdata_payload() -> None:
    artifact = build_raw_export(_result(), "replication_timing", "txt")
    assert artifact.media_type == "text/plain; charset=utf-8"
    np.testing.assert_array_equal(
        np.loadtxt(io.BytesIO(artifact.data)),
        _result().replication_timing,
    )


def test_xt_text_export_uses_rdata_space_by_time_layout() -> None:
    artifact = build_raw_export(_result(), "replicated_fraction_xt", "csv")
    lines = artifact.data.decode("utf-8").splitlines()
    assert lines == ["t0,t1", "0,0.8", "0,0.7", "0,0.75"]


@pytest.mark.parametrize("file_format", ("npy", "npv"))
def test_numpy_vector_exports_round_trip_without_pickle(file_format: str) -> None:
    result = _result()
    npy = build_raw_export(result, "replication_timing", file_format)
    np.testing.assert_array_equal(
        np.load(io.BytesIO(npy.data), allow_pickle=False),
        result.replication_timing,
    )


def test_npz_export_is_self_describing_and_uses_dnascape_axis_order() -> None:
    result = _result()
    npz = build_raw_export(result, "replicated_fraction_xt", "npz")
    with np.load(io.BytesIO(npz.data), allow_pickle=False) as archive:
        np.testing.assert_array_equal(
            archive["values"],
            result.outputs["replicated_fraction_xt"],
        )
        assert str(archive["data_form"]) == "spacetime"
        np.testing.assert_array_equal(archive["columns"], np.array(["t0", "t1"]))
        np.testing.assert_array_equal(archive["position_kb"], result.position_kb)
        np.testing.assert_array_equal(archive["time_min"], result.time_grid)


@pytest.mark.parametrize("file_format", ("csv", "tsv", "txt", "npy", "npv", "npz"))
def test_raw_exports_round_trip_through_rdata(
    file_format: str,
    tmp_path: Path,
) -> None:
    artifact = build_raw_export(
        _result(),
        "replication_timing",
        file_format,
        filename="replication_timing",
    )
    path = tmp_path / artifact.filename
    path.write_bytes(artifact.data)
    if file_format in {"csv", "tsv"}:
        loaded = rload(path, value_column="replication_timing")
    else:
        loaded = rload(path)
    np.testing.assert_array_equal(np.asarray(loaded).reshape(-1), _result().replication_timing)


def test_space_time_npz_round_trips_with_dnascape_data_form(tmp_path: Path) -> None:
    artifact = build_raw_export(
        _result(),
        "replicated_fraction_xt",
        "npz",
        filename="replicated_fraction",
    )
    path = tmp_path / artifact.filename
    path.write_bytes(artifact.data)
    loaded = rload(path, data_form="auto", return_metadata=True)
    assert loaded["data_form"] == "spacetime"
    np.testing.assert_array_equal(loaded["values"], _result().outputs["replicated_fraction_xt"])


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


def test_global_compression_choices_wrap_one_named_raw_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plain = build_raw_export(_result(), "replication_timing", "csv", filename="result")

    zipped = build_raw_export(
        _result(),
        "replication_timing",
        "csv",
        filename="result",
        compression="zip",
    )
    assert zipped.filename == "result.zip"
    with zipfile.ZipFile(io.BytesIO(zipped.data)) as archive:
        assert archive.namelist() == ["result.csv"]
        assert archive.read("result.csv") == plain.data

    xz = build_raw_export(
        _result(),
        "replication_timing",
        "csv",
        filename="result",
        compression="xz",
    )
    assert xz.filename == "result.csv.xz"
    assert lzma.decompress(xz.data) == plain.data

    calls: list[tuple[bytes, str]] = []

    def fake_7z(data: bytes, member_name: str) -> bytes:
        calls.append((data, member_name))
        return b"7z archive"

    monkeypatch.setattr(simulation_exports, "_seven_zip_bytes", fake_7z)
    seven_zip = build_raw_export(
        _result(),
        "replication_timing",
        "csv",
        filename="result",
        compression="7z",
    )
    assert calls == [(plain.data, "result.csv")]
    assert seven_zip.filename == "result.7z"
    assert seven_zip.media_type == "application/x-7z-compressed"


def test_gzipped_npz_export_round_trips_through_rdata(tmp_path: Path) -> None:
    artifact = build_raw_export(
        _result(),
        "replicated_fraction_xt",
        "npz",
        filename="replicated_fraction",
        compression="gzip",
    )
    path = tmp_path / artifact.filename
    path.write_bytes(artifact.data)

    loaded = rload(path, return_metadata=True)
    assert loaded["data_form"] == "spacetime"
    np.testing.assert_array_equal(
        loaded["values"],
        _result().outputs["replicated_fraction_xt"],
    )


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
