"""Pure, bounded serializers for cached simulation results.

The NiceGUI layer can pass an :class:`ExportArtifact` directly to
``ui.download(artifact.data, artifact.filename, artifact.media_type)``.  Heavy
plotting imports are intentionally deferred until a plot is actually exported.
"""

from __future__ import annotations

import base64
import binascii
import csv
import gzip
import importlib
import io
import lzma
import math
import re
import shutil
import subprocess
import tempfile
import unicodedata
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import unquote_to_bytes
from xml.etree import ElementTree

import numpy as np

from dnascape.workflows.export_settings import DATA_FILE_FORMAT_OPTIONS, normalize_compression
from dnascape.workflows.preview import _kde_data, has_finite_values
from dnascape.workflows.result_naming import RSIM_FILE_PREFIX, output_file_token
from dnascape.workflows.simulation import ConstantSimulationResult

AxisKind = Literal["space", "time", "replicate", "sample", "xt"]

PLOT_FORMATS = frozenset({"png", "svg", "pdf"})
RAW_FORMATS = frozenset(DATA_FILE_FORMAT_OPTIONS)
EXPORT_FORMATS = PLOT_FORMATS | RAW_FORMATS

MAX_PLOT_SERIES_POINTS = 20_000
MAX_PLOT_XT_ROWS = 1_000
MAX_PLOT_XT_COLUMNS = 2_000
MAX_PLOT_PIXELS = 25_000_000
MAX_TEXT_EXPORT_VALUES = 1_000_000
MAX_BINARY_EXPORT_BYTES = 128 * 1024 * 1024
MAX_CHART_CAPTURE_BYTES = 64 * 1024 * 1024

_SPACE_OUTPUTS = {
    "replication_timing",
    "origin_efficiency",
    "fork_directionality",
}
_TIME_OUTPUTS = {
    "replicated_fraction",
    "total_forks",
    "active_forks",
    "stalled_forks",
    "firing_factors",
}
_REPLICATE_OUTPUTS = {"completion_time", "s_phase_duration", "num_origins"}
_SAMPLE_OUTPUTS = {"inter_origin_distances"}
_DISTRIBUTION_OUTPUTS = _REPLICATE_OUTPUTS | _SAMPLE_OUTPUTS
_XT_OUTPUTS = {
    "replicated_fraction_xt",
    "initiation_events_xt",
    "coalescence_events_xt",
    "right_moving_fork_density_xt",
    "left_moving_fork_density_xt",
}
_OUTPUT_LABELS = {
    "replication_timing": "Replication timing",
    "completion_time": "Completion time",
    "s_phase_duration": "S-phase duration",
    "num_origins": "Origins per replicate",
    "inter_origin_distances": "Inter-origin distance",
    "origin_efficiency": "Origin efficiency",
    "fork_directionality": "Fork directionality",
    "replicated_fraction": "Replicated fraction",
    "total_forks": "Total forks",
    "active_forks": "Active forks",
    "stalled_forks": "Stalled forks",
    "firing_factors": "Firing factors",
    "replicated_fraction_xt": "Replicated fraction",
    "initiation_events_xt": "Initiation events",
    "coalescence_events_xt": "Coalescence events",
    "right_moving_fork_density_xt": "Right-moving fork density",
    "left_moving_fork_density_xt": "Left-moving fork density",
}
_Y_LABELS = {
    "replication_timing": "Time (min)",
    "completion_time": "Completion time (min)",
    "s_phase_duration": "Duration (min)",
    "num_origins": "Origins",
    "inter_origin_distances": "Distance (kb)",
    "origin_efficiency": "Probability",
    "fork_directionality": "Right - left fraction",
    "replicated_fraction": "Fraction",
    "total_forks": "Forks",
    "active_forks": "Forks",
    "stalled_forks": "Forks",
    "firing_factors": "Available factors",
}
_WINDOWS_RESERVED_STEMS = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{number}" for number in range(1, 10)),
    *(f"LPT{number}" for number in range(1, 10)),
}
_SVG_PATH_TOKEN_RE = re.compile(r"[A-Za-z]|[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
_MIN_METADATA_FREE_RECTANGLES_TO_COMPACT = 32


@dataclass(frozen=True)
class ExportArtifact:
    """Ready-to-download bytes and their browser-facing metadata."""

    data: bytes
    filename: str
    media_type: str


@dataclass(frozen=True)
class SelectedOutput:
    """A validated output with explicit plotting/export axes.

    Space-time arrays are normalized to ``(time, position)`` regardless of
    whether the scientific result stores them as ``(position, time)``.
    """

    key: str
    label: str
    axis_kind: AxisKind
    values: np.ndarray
    x: np.ndarray
    y: np.ndarray | None
    x_label: str
    y_label: str


def build_chart_capture_export(
    data_uri: str,
    file_format: str,
    *,
    filename: str | None = f"{RSIM_FILE_PREFIX}_plot",
    dpi: int = 150,
) -> ExportArtifact:
    """Package an exact ECharts preview capture without rasterizing vectors.

    PNG is captured from the visible canvas renderer. SVG and PDF must arrive
    from the preview's export-only SVG renderer; raster captures are rejected
    for those formats so a bitmap can never be hidden inside a vector wrapper.
    """

    normalized_format = normalize_export_format(file_format, allowed=PLOT_FORMATS)
    mime_type, payload = _decode_chart_data_uri(data_uri)
    if len(payload) > MAX_CHART_CAPTURE_BYTES:
        raise ValueError(
            f"Captured plot is {len(payload) / 1024**2:.1f} MiB; the local limit is "
            f"{MAX_CHART_CAPTURE_BYTES / 1024**2:.0f} MiB. Reduce the export DPI."
        )

    if normalized_format == "pdf":
        if mime_type != "image/svg+xml":
            raise ValueError("PDF export requires a vector SVG capture of the preview chart.")
        payload = _svg_capture_to_pdf(payload)
        media_type = "application/pdf"
    elif normalized_format == "svg":
        if mime_type != "image/svg+xml":
            raise ValueError("SVG export requires a vector SVG capture of the preview chart.")
        payload = _validated_vector_svg(payload)
        media_type = "image/svg+xml"
    else:
        if mime_type != "image/png":
            raise ValueError(f"The preview returned {mime_type!r}, but 'image/png' was expected.")
        media_type = "image/png"

    return ExportArtifact(
        data=payload,
        filename=safe_export_filename(filename, normalized_format),
        media_type=media_type,
    )


def normalize_export_format(file_format: str, *, allowed: frozenset[str]) -> str:
    """Normalize and validate a user-facing export format."""

    normalized = str(file_format).strip().lower().lstrip(".")
    if normalized not in allowed:
        choices = ", ".join(sorted(allowed))
        raise ValueError(f"Unsupported export format {file_format!r}; choose {choices}.")
    return normalized


def safe_export_filename(
    filename: str | None,
    extension: str,
    *,
    default_stem: str = RSIM_FILE_PREFIX,
    gzip_output: bool = False,
) -> str:
    """Return a path-free, cross-platform filename with one correct suffix."""

    normalized_extension = normalize_export_format(extension, allowed=EXPORT_FORMATS)
    candidate = str(filename or "").strip()
    candidate = re.split(r"[\\/]", candidate)[-1]
    candidate = re.sub(r"(?i)\.(?:gz|zip|7z|xz)$", "", candidate)
    # Legacy source suffixes still need replacing when exporting to NPZ/text.
    known_extensions = EXPORT_FORMATS | {"npy", "npv"}
    candidate = re.sub(
        rf"(?i)\.({'|'.join(sorted(known_extensions))})$",
        "",
        candidate,
    )
    candidate = unicodedata.normalize("NFKD", candidate).encode("ascii", "ignore").decode("ascii")
    candidate = re.sub(r"[^A-Za-z0-9._-]+", "_", candidate)
    candidate = re.sub(r"_+", "_", candidate).strip(" ._-")
    if not candidate:
        candidate = default_stem
    if candidate.upper() in _WINDOWS_RESERVED_STEMS:
        candidate = f"_{candidate}"
    suffix = f".{normalized_extension}{'.gz' if gzip_output else ''}"
    max_stem_length = max(1, 180 - len(suffix))
    candidate = candidate[:max_stem_length].rstrip(" .") or default_stem
    return f"{candidate}{suffix}"


def _seven_zip_bytes(data: bytes, member_name: str) -> bytes:
    """Create a one-member 7z archive with py7zr or a local 7-Zip binary."""

    try:
        py7zr = importlib.import_module("py7zr")
    except ImportError:
        executable = shutil.which("7z") or shutil.which("7zz")
        if executable is None:
            windows_7z = Path(r"C:\Program Files\7-Zip\7z.exe")
            executable = str(windows_7z) if windows_7z.is_file() else None
        if executable is None:
            raise RuntimeError(
                "7-Zip compression requires py7zr or a local 7z executable. "
                "Install the DNAscape app dependencies and try again."
            ) from None
        with tempfile.TemporaryDirectory(prefix="dnascape-export-") as temporary:
            directory = Path(temporary)
            member_path = directory / member_name
            member_path.write_bytes(data)
            archive_path = directory / "export.7z"
            completed = subprocess.run(
                [
                    executable,
                    "a",
                    "-t7z",
                    "-mx=5",
                    "-bd",
                    "-y",
                    str(archive_path),
                    member_path.name,
                ],
                cwd=directory,
                capture_output=True,
                check=False,
                timeout=30,
            )
            if completed.returncode != 0 or not archive_path.is_file():
                raise RuntimeError("7-Zip could not create the raw-data archive.") from None
            return archive_path.read_bytes()

    archive = io.BytesIO()
    with py7zr.SevenZipFile(archive, mode="w") as output:
        output.writestr(data, member_name)
    return archive.getvalue()


def compressed_export_artifact(
    data: bytes,
    *,
    filename: str,
    media_type: str,
    compression: str,
) -> ExportArtifact:
    """Wrap one raw-data file in the selected deterministic compression format."""

    normalized = normalize_compression(compression)
    if normalized == "none":
        return ExportArtifact(data=data, filename=filename, media_type=media_type)
    if normalized == "gzip":
        return ExportArtifact(
            data=gzip.compress(data, compresslevel=6, mtime=0),
            filename=f"{filename}.gz",
            media_type="application/gzip",
        )
    stem = filename.rsplit(".", 1)[0]
    if normalized == "zip":
        archive = io.BytesIO()
        with zipfile.ZipFile(
            archive,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
        ) as output:
            output.writestr(filename, data)
        return ExportArtifact(
            data=archive.getvalue(),
            filename=f"{stem}.zip",
            media_type="application/zip",
        )
    if normalized == "7z":
        return ExportArtifact(
            data=_seven_zip_bytes(data, filename),
            filename=f"{stem}.7z",
            media_type="application/x-7z-compressed",
        )
    if normalized == "xz":
        return ExportArtifact(
            data=lzma.compress(data, format=lzma.FORMAT_XZ, preset=6),
            filename=f"{filename}.xz",
            media_type="application/x-xz",
        )
    raise AssertionError(f"Unhandled compression format {normalized!r}.")


def _raw_compression(compression: str, gzip_output: bool | None) -> str:
    """Keep the former gzip flag compatible while compression becomes configurable."""

    if gzip_output is not None:
        return "gzip" if gzip_output else "none"
    return normalize_compression(compression)


def select_result_output(
    result: ConstantSimulationResult,
    output_key: str,
    *,
    axis_kind: AxisKind | None = None,
) -> SelectedOutput:
    """Select one numeric result and infer or validate its scientific axes."""

    key = str(output_key).strip()
    from .preview_inputs import INPUT_PREFIX, input_preview_result, input_profile_label
    result = input_preview_result(result, key)
    value = result.outputs.get(key)
    if value is None and key == "replication_timing":
        value = result.replication_timing
    if value is None:
        raise ValueError(f"Simulation output {key!r} is not available.")
    try:
        values = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Simulation output {key!r} must be numeric.") from exc
    if values.ndim not in {1, 2} or values.size == 0:
        raise ValueError("Only non-empty one-dimensional or space-time outputs can be exported.")

    position = _validated_axis(result.position_kb, "position")
    time = _validated_axis(result.time_grid, "time", allow_empty=True)
    input_kind = {"x": "space", "t": "time", "xt": "xt"}.get(
        getattr(result, "input_forms", {}).get(key)
    )
    inferred_kind = axis_kind or input_kind or _infer_axis_kind(
        key,
        values,
        position_size=position.size,
        time_size=time.size,
        replicates=int(result.replicates),
    )
    label = input_profile_label(key) if key.startswith(INPUT_PREFIX) else _OUTPUT_LABELS.get(key, _humanize(key))

    if inferred_kind == "xt":
        if values.ndim != 2 or time.size == 0:
            raise ValueError("Space-time output requires a two-dimensional array and a time grid.")
        if values.shape == (position.size, time.size):
            normalized_values = values.T
        elif values.shape == (time.size, position.size):
            normalized_values = values
        else:
            raise ValueError(
                "Space-time output shape must match position x time or time x position."
            )
        return SelectedOutput(
            key=key,
            label=label,
            axis_kind="xt",
            values=normalized_values,
            x=position,
            y=time,
            x_label="Position (kb)",
            y_label="Time (min)",
        )

    if values.ndim != 1:
        raise ValueError("A non-space-time output must be one-dimensional.")
    if inferred_kind == "space":
        axis, x_label = position, "Position (kb)"
    elif inferred_kind == "time":
        axis, x_label = time, "Time (min)"
    elif inferred_kind == "replicate":
        axis = np.arange(1, int(result.replicates) + 1, dtype=float)
        x_label = "Replicate"
    elif inferred_kind == "sample":
        # Inter-origin distances are pooled over realizations, so their count
        # generally differs from the ensemble size. Give each measured interval
        # a stable one-based index for line views and self-describing exports.
        axis = np.arange(1, values.size + 1, dtype=float)
        x_label = "Interval"
    else:  # defensive guard for future AxisKind extensions
        raise ValueError(f"Unsupported axis kind: {inferred_kind!r}.")
    if values.size != axis.size:
        raise ValueError(
            f"Output has {values.size:,} values but its {inferred_kind} axis has {axis.size:,}."
        )
    return SelectedOutput(
        key=key,
        label=label,
        axis_kind=inferred_kind,
        values=values,
        x=axis,
        y=None,
        x_label=x_label,
        y_label=_Y_LABELS.get(key, label),
    )


def build_plot_export(
    result: ConstantSimulationResult,
    output_key: str,
    file_format: str,
    *,
    filename: str | None = None,
    title: str | None = None,
    series_label: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    color: str = "#b03412",
    cmap: str = "viridis",
    invert_y: bool = False,
    y_limits: tuple[float, float] | None = None,
    view_mode: str = "Default",
    histogram_bins: int = 60,
    density: bool = False,
    show_legend: bool = True,
    show_colorbar: bool = True,
    log_y: bool = False,
    axis_kind: AxisKind | None = None,
    figure_size: tuple[float, float] = (8.0, 4.5),
    dpi: int = 150,
) -> ExportArtifact:
    """Lazily render a selected 1D or space-time output as PNG, SVG, or PDF."""

    normalized_format = normalize_export_format(file_format, allowed=PLOT_FORMATS)
    selected = select_result_output(result, output_key, axis_kind=axis_kind)
    width, height, normalized_dpi = _validated_canvas(figure_size, dpi)
    normalized_limits = _validated_limits(y_limits)
    normalized_view = str(view_mode).strip().casefold()
    if normalized_view not in {"default", "histogram", "time array"}:
        raise ValueError("View mode must be Default, Histogram, or Time array.")
    use_histogram = selected.axis_kind != "xt" and (
        selected.key in _DISTRIBUTION_OUTPUTS or normalized_view == "histogram"
    )
    normalized_bins = _validated_histogram_bins(histogram_bins)
    if not has_finite_values(selected.values):
        raise ValueError("The selected output does not contain any finite values to plot.")

    # Matplotlib is deliberately absent from module-level imports. Opening an
    # export menu therefore has no plotting import/startup cost.
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(width, height), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots()
    plot_title = str(title).strip() if title is not None else selected.label
    axes.set_title(plot_title or selected.label)

    if selected.axis_kind == "xt":
        assert selected.y is not None
        # Match plotf and the notebook figures: rows are time, columns are
        # genomic position, and imshow receives the canonical display matrix.
        default_x_label = selected.x_label
        default_y_label = selected.y_label
        time_indices = _sample_indices(selected.y.size, MAX_PLOT_XT_ROWS)
        position_indices = _sample_indices(selected.x.size, MAX_PLOT_XT_COLUMNS)
        image_values = selected.values[np.ix_(time_indices, position_indices)]
        image_extent = _image_extent(selected.x, selected.y)
        image = axes.imshow(
            image_values,
            aspect="auto",
            origin="lower",
            extent=image_extent,
            cmap=cmap,
            interpolation="nearest",
            rasterized=True,
        )
        if show_colorbar:
            figure.colorbar(image, ax=axes, label=selected.label)
    elif use_histogram:
        default_x_label = selected.y_label
        default_y_label = "Density" if density else "Count"
        finite_values = selected.values[np.isfinite(selected.values)]
        label = str(series_label).strip() if series_label is not None else selected.label
        if density:
            kde_x, kde_y = _kde_data(finite_values)
            axes.plot(kde_x, kde_y, color=color, label=label)
        else:
            axes.hist(
                finite_values,
                bins=normalized_bins,
                density=False,
                color=color,
                alpha=0.86,
                edgecolor="none",
                label=label,
            )
    else:
        default_x_label = selected.x_label
        default_y_label = selected.y_label
        indices = _sample_indices(selected.values.size, MAX_PLOT_SERIES_POINTS)
        axes.plot(
            selected.x[indices],
            selected.values[indices],
            color=color,
            linewidth=1.8,
            label=str(series_label).strip() if series_label is not None else "Mean",
        )
    axes.set_xlabel(str(x_label).strip() if x_label is not None else default_x_label)
    axes.set_ylabel(str(y_label).strip() if y_label is not None else default_y_label)

    if normalized_limits is not None:
        axes.set_ylim(*normalized_limits)
    if log_y and selected.axis_kind != "xt":
        axes.set_yscale("log")
    if invert_y:
        axes.invert_yaxis()
    if show_legend and selected.axis_kind != "xt":
        axes.legend()

    buffer = io.BytesIO()
    try:
        figure.savefig(buffer, format=normalized_format, dpi=normalized_dpi)
    finally:
        figure.clear()
    data = buffer.getvalue()
    media_type = {
        "png": "image/png",
        "svg": "image/svg+xml",
        "pdf": "application/pdf",
    }[normalized_format]
    default_name = f"{RSIM_FILE_PREFIX}_{output_file_token(output_key)}_plot"
    return ExportArtifact(
        data=data,
        filename=safe_export_filename(
            filename,
            normalized_format,
            default_stem=default_name,
        ),
        media_type=media_type,
    )


def build_raw_export(
    result: ConstantSimulationResult,
    output_key: str,
    file_format: str,
    *,
    filename: str | None = None,
    compression: str = "none",
    gzip_output: bool | None = None,
    axis_kind: AxisKind | None = None,
) -> ExportArtifact:
    """Serialize one selected output as text or self-describing NPZ bytes."""

    normalized_format = normalize_export_format(file_format, allowed=RAW_FORMATS)
    selected = select_result_output(result, output_key, axis_kind=axis_kind)
    if normalized_format in {"csv", "tsv", "txt"}:
        if selected.values.size > MAX_TEXT_EXPORT_VALUES:
            raise ValueError(
                f"Text export contains {selected.values.size:,} values; the interactive "
                f"limit is {MAX_TEXT_EXPORT_VALUES:,}. Use NPZ instead."
            )
        data = _text_bytes(selected, normalized_format)
    else:
        axis_bytes = selected.x.nbytes + (0 if selected.y is None else selected.y.nbytes)
        estimated_bytes = selected.values.nbytes + axis_bytes
        if estimated_bytes > MAX_BINARY_EXPORT_BYTES:
            raise ValueError(
                f"Binary export would contain at least {estimated_bytes / 1024**2:.1f} MiB; "
                f"the local limit is {MAX_BINARY_EXPORT_BYTES / 1024**2:.0f} MiB."
            )
        data = _binary_bytes(selected)

    default_name = f"{RSIM_FILE_PREFIX}_{output_file_token(output_key)}"
    inner_filename = safe_export_filename(
        filename,
        normalized_format,
        default_stem=default_name,
    )
    return compressed_export_artifact(
        data=data,
        filename=inner_filename,
        media_type={
            "csv": "text/csv; charset=utf-8",
            "tsv": "text/tab-separated-values; charset=utf-8",
            "txt": "text/plain; charset=utf-8",
            "npz": "application/octet-stream",
        }[normalized_format],
        compression=_raw_compression(compression, gzip_output),
    )


def _decode_chart_data_uri(data_uri: str) -> tuple[str, bytes]:
    """Decode a bounded image data URI returned by ECharts."""

    if not isinstance(data_uri, str) or not data_uri.startswith("data:"):
        raise ValueError("The preview did not return a downloadable chart image.")
    try:
        header, encoded = data_uri.split(",", 1)
        metadata = header[5:].split(";")
        mime_type = metadata[0].casefold()
        if "base64" in (item.casefold() for item in metadata[1:]):
            payload = base64.b64decode(encoded, validate=True)
        else:
            payload = unquote_to_bytes(encoded)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("The preview returned an invalid chart image.") from exc
    if not payload:
        raise ValueError("The preview returned an empty chart image.")
    return mime_type, payload


def _validated_vector_svg_root(svg_data: bytes) -> ElementTree.Element:
    """Parse a chart SVG with vector axes and optional embedded PNG heatmaps."""

    try:
        root = ElementTree.fromstring(svg_data)
    except ElementTree.ParseError as exc:
        raise ValueError("The preview returned an invalid SVG capture.") from exc
    if root.tag.rsplit("}", 1)[-1].casefold() != "svg":
        raise ValueError("The preview capture is not an SVG document.")
    raster_elements = {"canvas", "foreignobject", "video"}
    for element in root.iter():
        if element.tag.rsplit("}", 1)[-1].casefold() in raster_elements:
            raise ValueError("The SVG preview contains an embedded raster surface.")
        if element.tag.rsplit("}", 1)[-1].casefold() == "image":
            href = next(
                (value for key, value in element.attrib.items() if key.rsplit("}", 1)[-1] == "href"),
                "",
            )
            if not href.startswith("data:image/png;base64,"):
                raise ValueError("Heatmap SVG images must contain embedded PNG data.")
            mime, payload = _decode_chart_data_uri(href)
            if mime != "image/png" or not payload.startswith(b"\x89PNG\r\n\x1a\n"):
                raise ValueError("The heatmap SVG contains an invalid PNG image.")
    return root


def _validated_vector_svg(svg_data: bytes) -> bytes:
    """Validate an SVG chart, retaining its raster heatmap and vector axes."""

    _validated_vector_svg_root(svg_data)
    return svg_data


def _is_axis_aligned_rectangle_path(path_data: str) -> bool:
    """Recognize the compact rectangle paths emitted by ECharts/ZRender."""

    tokens = _SVG_PATH_TOKEN_RE.findall(path_data)
    if len(tokens) != 13 or tokens[0].casefold() != "m" or tokens[-1].casefold() != "z":
        return False
    commands = tuple(tokens[index] for index in (3, 6, 9))
    if any(command.casefold() != "l" for command in commands):
        return False
    try:
        values = tuple(float(tokens[index]) for index in (1, 2, 4, 5, 7, 8, 10, 11))
    except ValueError:
        return False
    if not all(math.isfinite(value) for value in values):
        return False

    x0, y0, first_x, first_y, second_x, second_y, third_x, third_y = values

    def close(left: float, right: float) -> bool:
        return math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-9)

    if commands[0].islower():
        horizontal_first = (
            not close(first_x, 0.0)
            and close(first_y, 0.0)
            and close(second_x, 0.0)
            and not close(second_y, 0.0)
            and close(third_x, -first_x)
            and close(third_y, 0.0)
        )
        vertical_first = (
            close(first_x, 0.0)
            and not close(first_y, 0.0)
            and not close(second_x, 0.0)
            and close(second_y, 0.0)
            and close(third_x, 0.0)
            and close(third_y, -first_y)
        )
        return horizontal_first or vertical_first

    horizontal_first = (
        not close(first_x, x0)
        and close(first_y, y0)
        and close(second_x, first_x)
        and not close(second_y, first_y)
        and close(third_x, x0)
        and close(third_y, second_y)
    )
    vertical_first = (
        close(first_x, x0)
        and not close(first_y, y0)
        and not close(second_x, first_x)
        and close(second_y, first_y)
        and close(third_x, second_x)
        and close(third_y, y0)
    )
    return horizontal_first or vertical_first


def _compacted_echarts_svg_for_pdf(svg_data: bytes) -> bytes:
    """Remove interaction metadata and combine safe per-datum vector paths.

    ECharts emits histograms and other custom series as one SVG ``path`` per
    datum.  Thousands of otherwise identical rectangles make ``svglib`` spend
    most of the export time repeatedly parsing attributes.  PDF has no hover
    interaction, so its ``ecmeta_*`` attributes can be removed. Opaque
    histogram and heatmap rectangles with the same appearance can then share
    one compound path; other paths are combined only when adjacent. Their
    geometry, colour, and vector nature stay unchanged. Transparent paths are
    deliberately left separate because combining overlapping translucent
    shapes could change their appearance.
    """

    root = _validated_vector_svg_root(svg_data)
    changed = False
    for parent in root.iter():
        children = list(parent)
        metadata_free_rectangles = sum(
            child.tag.rsplit("}", 1)[-1].casefold() == "path"
            and not any(key.casefold().startswith("ecmeta_") for key in child.attrib)
            and _is_axis_aligned_rectangle_path(str(child.attrib.get("d", "")))
            for child in children
        )
        compact_metadata_free_rectangles = (
            metadata_free_rectangles >= _MIN_METADATA_FREE_RECTANGLES_TO_COMPACT
        )
        previous: ElementTree.Element | None = None
        previous_signature: tuple[object, ...] | None = None
        rectangle_paths: dict[tuple[object, ...], ElementTree.Element] = {}
        for child in children:
            attributes = child.attrib
            metadata = {
                key: value
                for key, value in attributes.items()
                if key.rsplit("}", 1)[-1].casefold().startswith("ecmeta_")
            }
            series_index = metadata.get("ecmeta_series_index")
            is_path = child.tag.rsplit("}", 1)[-1].casefold() == "path"
            rectangle_path = is_path and _is_axis_aligned_rectangle_path(
                str(attributes.get("d", ""))
            )
            is_datum_path = (
                is_path
                and "ecmeta_data_index" in metadata
                and series_index is not None
                and bool(attributes.get("d"))
            )
            rectangle_path = rectangle_path and (is_datum_path or compact_metadata_free_rectangles)
            for key in metadata:
                del attributes[key]
                changed = True

            style = str(attributes.get("style", "")).casefold()
            paint = " ".join(
                str(attributes.get(name, "")).casefold() for name in ("fill", "stroke")
            )
            has_effects = any(
                name in attributes
                for name in (
                    "id",
                    "filter",
                    "mask",
                    "marker-start",
                    "marker-mid",
                    "marker-end",
                    "opacity",
                    "fill-opacity",
                    "stroke-opacity",
                )
            )
            opaque_fill = (
                str(attributes.get("fill", "")).casefold() not in {"", "none"}
                and "rgba(" not in paint
                and "opacity" not in style
                and not has_effects
            )
            if (not is_datum_path and not rectangle_path) or not opaque_fill:
                previous = None
                previous_signature = None
                continue

            signature: tuple[object, ...] = (
                series_index,
                tuple(sorted((key, value) for key, value in attributes.items() if key != "d")),
            )
            rectangle_target = rectangle_paths.get(signature) if rectangle_path else None
            if rectangle_target is not None:
                rectangle_target.set(
                    "d",
                    f"{rectangle_target.get('d', '')} {attributes['d']}",
                )
                parent.remove(child)
                changed = True
                previous = None
                previous_signature = None
            elif rectangle_path:
                # Histogram bars and heatmap cells do not overlap within a
                # series. Group equal-colour rectangles even when neighbouring
                # values use another colour; this removes the worst svglib
                # parsing cost on continuous heatmaps.
                rectangle_paths[signature] = child
                previous = None
                previous_signature = None
            elif previous is not None and signature == previous_signature:
                previous.set("d", f"{previous.get('d', '')} {attributes['d']}")
                parent.remove(child)
                changed = True
            else:
                previous = child
                previous_signature = signature

    if not changed:
        return svg_data
    return ElementTree.tostring(root, encoding="utf-8", xml_declaration=True)


def _svg_capture_to_pdf(svg_data: bytes) -> bytes:
    """Convert a chart SVG, preserving vector labels and embedded heatmaps."""

    vector_svg = _compacted_echarts_svg_for_pdf(svg_data)
    try:
        from reportlab.graphics import renderPDF
        from reportlab.lib.utils import ImageReader
        from reportlab.pdfgen.canvas import Canvas
        from svglib.svglib import svg2rlg
    except ImportError as exc:
        raise RuntimeError(
            "Vector PDF export requires the DNAscape app export dependencies. "
            "Install them with `pip install -e .[app]`."
        ) from exc

    try:
        class HeatmapCanvas(Canvas):
            def drawInlineImage(self, image, x, y, width=None, height=None, **kwargs):
                # The default inline-image renderer discards PNG alpha. Keep
                # missing heatmap cells transparent while drawing all axes and
                # labels through the normal vector renderer.
                return self.drawImage(
                    ImageReader(image), x, y, width, height, mask="auto", **kwargs,
                )

        drawing = svg2rlg(io.BytesIO(vector_svg))
        if drawing is None:
            raise ValueError("The preview SVG could not be converted to PDF.")
        # SVG/CSS pixels are defined as 1/96 inch; ReportLab coordinates are
        # points (1/72 inch). Preserve the browser chart's physical dimensions.
        scale = 72.0 / 96.0
        drawing.scale(scale, scale)
        drawing.width *= scale
        drawing.height *= scale
        pdf_data = bytes(renderPDF.drawToString(drawing, canvasMaker=HeatmapCanvas))
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError("The vector preview could not be converted to PDF.") from exc
    if not pdf_data.startswith(b"%PDF"):
        raise ValueError("The vector PDF renderer returned an invalid document.")
    has_heatmap = any(
        element.tag.rsplit("}", 1)[-1].casefold() == "image"
        for element in _validated_vector_svg_root(vector_svg).iter()
    )
    if not has_heatmap and b"/Subtype /Image" in pdf_data:
        raise ValueError("The PDF renderer rasterized part of the preview unexpectedly.")
    return pdf_data


def _infer_axis_kind(
    key: str,
    values: np.ndarray,
    *,
    position_size: int,
    time_size: int,
    replicates: int,
) -> AxisKind:
    if key in _XT_OUTPUTS:
        return "xt"
    if key in _SPACE_OUTPUTS:
        return "space"
    if key in _TIME_OUTPUTS:
        return "time"
    if key in _REPLICATE_OUTPUTS:
        return "replicate"
    if key in _SAMPLE_OUTPUTS:
        return "sample"
    if values.ndim == 2:
        if time_size and set(values.shape) == {position_size, time_size}:
            return "xt"
        raise ValueError(
            f"Cannot infer axes for two-dimensional output {key!r}; pass axis_kind='xt'."
        )
    matches: list[AxisKind] = []
    if values.size == position_size:
        matches.append("space")
    if time_size and values.size == time_size:
        matches.append("time")
    if values.size == replicates:
        matches.append("replicate")
    if len(matches) != 1:
        raise ValueError(
            f"Cannot infer an unambiguous axis for output {key!r}; pass axis_kind explicitly."
        )
    return matches[0]


def _validated_axis(
    value: np.ndarray,
    label: str,
    *,
    allow_empty: bool = False,
) -> np.ndarray:
    try:
        axis = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"The {label} axis must be numeric.") from exc
    if axis.size == 0 and allow_empty:
        return axis
    if axis.size == 0 or not np.all(np.isfinite(axis)):
        raise ValueError(f"The {label} axis must be non-empty and finite.")
    return axis


def _validated_canvas(
    figure_size: tuple[float, float],
    dpi: int,
) -> tuple[float, float, int]:
    try:
        width, height = (float(value) for value in figure_size)
        normalized_dpi = int(dpi)
    except (TypeError, ValueError) as exc:
        raise ValueError("Figure size and DPI must be numeric.") from exc
    if not all(math.isfinite(value) and 1 <= value <= 20 for value in (width, height)):
        raise ValueError("Figure width and height must be between 1 and 20 inches.")
    if normalized_dpi != dpi or not 72 <= normalized_dpi <= 600:
        raise ValueError("DPI must be a whole number between 72 and 600.")
    if width * normalized_dpi * height * normalized_dpi > MAX_PLOT_PIXELS:
        raise ValueError("Figure dimensions and DPI exceed the local pixel limit.")
    return width, height, normalized_dpi


def _validated_limits(
    limits: tuple[float, float] | None,
) -> tuple[float, float] | None:
    if limits is None:
        return None
    try:
        lower, upper = (float(value) for value in limits)
    except (TypeError, ValueError) as exc:
        raise ValueError("Y limits must contain two numbers.") from exc
    if not all(math.isfinite(value) for value in (lower, upper)) or upper <= lower:
        raise ValueError("Y maximum must be finite and greater than Y minimum.")
    return lower, upper


def _validated_histogram_bins(value: int) -> int:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Histogram bins must be a whole number between 1 and 2,000.") from exc
    if not math.isfinite(numeric) or not numeric.is_integer() or not 1 <= numeric <= 2_000:
        raise ValueError("Histogram bins must be a whole number between 1 and 2,000.")
    return int(numeric)


def _sample_indices(size: int, limit: int) -> np.ndarray:
    if size <= limit:
        return np.arange(size)
    return np.unique(np.linspace(0, size - 1, limit, dtype=int))


def _image_extent(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    def bounds(axis: np.ndarray) -> tuple[float, float]:
        lower, upper = float(np.min(axis)), float(np.max(axis))
        if lower == upper:
            padding = max(abs(lower) * 0.05, 0.5)
            return lower - padding, upper + padding
        return lower, upper

    x_min, x_max = bounds(x)
    y_min, y_max = bounds(y)
    return x_min, x_max, y_min, y_max


def _text_bytes(selected: SelectedOutput, file_format: str) -> bytes:
    values, _data_form, columns = _rdata_payload(selected)
    if file_format == "txt":
        stream = io.StringIO(newline="")
        np.savetxt(stream, values, fmt="%.18e")
        return stream.getvalue().encode("utf-8")

    delimiter = {"csv": ",", "tsv": "\t"}[file_format]
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, delimiter=delimiter, lineterminator="\n")
    if values.ndim == 1:
        writer.writerow((_column_name(selected.key),))
        writer.writerows((_number_text(value),) for value in values)
    else:
        assert columns is not None
        writer.writerow(columns)
        writer.writerows(tuple(_number_text(value) for value in row) for row in values)
    return stream.getvalue().encode("utf-8")


def _binary_bytes(selected: SelectedOutput) -> bytes:
    stream = io.BytesIO()
    values, data_form, columns = _rdata_payload(selected)
    data_form_array = np.asarray(data_form)
    if selected.axis_kind == "xt":
        assert selected.y is not None
        assert columns is not None
        np.savez(
            stream,
            values=values,
            data_form=data_form_array,
            columns=np.asarray(columns),
            position_kb=selected.x,
            time_min=selected.y,
        )
    elif selected.axis_kind == "space":
        np.savez(
            stream,
            values=values,
            data_form=data_form_array,
            position_kb=selected.x,
        )
    elif selected.axis_kind == "time":
        assert columns is not None
        np.savez(
            stream,
            values=values,
            data_form=data_form_array,
            columns=np.asarray(columns),
            time_min=selected.x,
        )
    elif selected.axis_kind == "sample":
        np.savez(
            stream,
            values=values,
            data_form=data_form_array,
            sample=selected.x,
        )
    else:
        np.savez(
            stream,
            values=values,
            data_form=data_form_array,
            replicate=selected.x,
        )
    return stream.getvalue()


def _rdata_payload(selected: SelectedOutput) -> tuple[np.ndarray, str, tuple[str, ...] | None]:
    """Return values in the same shape and metadata convention as ``rsave``."""

    if selected.axis_kind == "xt":
        assert selected.y is not None
        values = selected.values.T  # DNAscape convention is (space, time).
        columns = tuple(f"t{index}" for index in range(selected.y.size))
        return values, "spacetime", columns
    if selected.axis_kind == "time":
        values = selected.values.reshape(1, -1)
        columns = tuple(f"t{index}" for index in range(selected.values.size))
        return values, "time", columns
    return selected.values, "space", None


def _number_text(value: float) -> str:
    text = repr(float(value))
    return text[:-2] if text.endswith(".0") else text


def _column_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_") or "value"


def _humanize(value: str) -> str:
    return value.replace("_xt", "").replace("_", " ").strip().title() or "Simulation output"
