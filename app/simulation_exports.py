"""Pure, bounded serializers for cached simulation results.

The NiceGUI layer can pass an :class:`ExportArtifact` directly to
``ui.download(artifact.data, artifact.filename, artifact.media_type)``.  Heavy
plotting imports are intentionally deferred until a plot is actually exported.
"""

from __future__ import annotations

import csv
import gzip
import io
import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Literal

import numpy as np

from app.simulation_service import ConstantSimulationResult

AxisKind = Literal["space", "time", "replicate", "xt"]

PLOT_FORMATS = frozenset({"png", "svg", "pdf"})
RAW_FORMATS = frozenset({"csv", "tsv", "txt", "npy", "npz"})
EXPORT_FORMATS = PLOT_FORMATS | RAW_FORMATS

MAX_PLOT_SERIES_POINTS = 20_000
MAX_PLOT_XT_ROWS = 1_000
MAX_PLOT_XT_COLUMNS = 2_000
MAX_PLOT_PIXELS = 25_000_000
MAX_TEXT_EXPORT_VALUES = 1_000_000
MAX_BINARY_EXPORT_BYTES = 128 * 1024 * 1024

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
_REPLICATE_OUTPUTS = {"completion_time", "num_origins"}
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
    "num_origins": "Origins per replicate",
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
    "num_origins": "Origins",
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
    default_stem: str = "dnascape_simulation",
    gzip_output: bool = False,
) -> str:
    """Return a path-free, cross-platform filename with one correct suffix."""

    normalized_extension = normalize_export_format(extension, allowed=EXPORT_FORMATS)
    candidate = str(filename or "").strip()
    candidate = re.split(r"[\\/]", candidate)[-1]
    candidate = re.sub(r"(?i)\.gz$", "", candidate)
    candidate = re.sub(
        rf"(?i)\.({'|'.join(sorted(EXPORT_FORMATS))})$",
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


def select_result_output(
    result: ConstantSimulationResult,
    output_key: str,
    *,
    axis_kind: AxisKind | None = None,
) -> SelectedOutput:
    """Select one numeric result and infer or validate its scientific axes."""

    key = str(output_key).strip()
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
    inferred_kind = axis_kind or _infer_axis_kind(
        key,
        values,
        position_size=position.size,
        time_size=time.size,
        replicates=int(result.replicates),
    )
    label = _OUTPUT_LABELS.get(key, _humanize(key))

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
    filename: str | None = "dnascape_simulation_plot",
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    color: str = "#2c7f77",
    cmap: str = "viridis",
    invert_y: bool = False,
    y_limits: tuple[float, float] | None = None,
    view_mode: str = "Single view",
    histogram_bins: int = 60,
    density: bool = False,
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
    if normalized_view not in {"single view", "multiple view", "histogram", "time array"}:
        raise ValueError("View mode must be Single view, Multiple view, Histogram, or Time array.")
    use_histogram = selected.axis_kind != "xt" and (
        selected.key in _REPLICATE_OUTPUTS or normalized_view == "histogram"
    )
    normalized_bins = _validated_histogram_bins(histogram_bins)
    if not np.any(np.isfinite(selected.values)):
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
        default_x_label = selected.x_label
        default_y_label = selected.y_label
        x_indices = _sample_indices(selected.x.size, MAX_PLOT_XT_COLUMNS)
        y_indices = _sample_indices(selected.y.size, MAX_PLOT_XT_ROWS)
        image_values = selected.values[np.ix_(y_indices, x_indices)]
        image = axes.imshow(
            image_values,
            aspect="auto",
            origin="lower",
            extent=_image_extent(selected.x[x_indices], selected.y[y_indices]),
            cmap=cmap,
            interpolation="nearest",
        )
        figure.colorbar(image, ax=axes, label=selected.label)
    elif use_histogram:
        default_x_label = selected.y_label
        default_y_label = "Density" if density else "Count"
        finite_values = selected.values[np.isfinite(selected.values)]
        axes.hist(
            finite_values,
            bins=normalized_bins,
            density=bool(density),
            color=color,
            alpha=0.86,
            edgecolor="none",
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
        )
    axes.set_xlabel(str(x_label).strip() if x_label is not None else default_x_label)
    axes.set_ylabel(str(y_label).strip() if y_label is not None else default_y_label)

    if normalized_limits is not None:
        axes.set_ylim(*normalized_limits)
    if invert_y:
        axes.invert_yaxis()

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
    return ExportArtifact(
        data=data,
        filename=safe_export_filename(filename, normalized_format),
        media_type=media_type,
    )


def build_raw_export(
    result: ConstantSimulationResult,
    output_key: str,
    file_format: str,
    *,
    filename: str | None = "dnascape_simulation",
    gzip_output: bool = False,
    axis_kind: AxisKind | None = None,
) -> ExportArtifact:
    """Serialize one selected output as text, NPY, or self-describing NPZ bytes."""

    normalized_format = normalize_export_format(file_format, allowed=RAW_FORMATS)
    selected = select_result_output(result, output_key, axis_kind=axis_kind)
    if normalized_format in {"csv", "tsv", "txt"}:
        if selected.values.size > MAX_TEXT_EXPORT_VALUES:
            raise ValueError(
                f"Text export contains {selected.values.size:,} values; the interactive "
                f"limit is {MAX_TEXT_EXPORT_VALUES:,}. Use NPY or NPZ instead."
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
        data = _binary_bytes(selected, normalized_format)

    if gzip_output:
        data = gzip.compress(data, compresslevel=6, mtime=0)
    media_type = (
        "application/gzip"
        if gzip_output
        else {
            "csv": "text/csv; charset=utf-8",
            "tsv": "text/tab-separated-values; charset=utf-8",
            "txt": "text/plain; charset=utf-8",
            "npy": "application/octet-stream",
            "npz": "application/octet-stream",
        }[normalized_format]
    )
    return ExportArtifact(
        data=data,
        filename=safe_export_filename(
            filename,
            normalized_format,
            gzip_output=gzip_output,
        ),
        media_type=media_type,
    )


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
        if axis.size == 1:
            return float(axis[0] - 0.5), float(axis[0] + 0.5)
        return float(axis[0]), float(axis[-1])

    x_min, x_max = bounds(x)
    y_min, y_max = bounds(y)
    return x_min, x_max, y_min, y_max


def _text_bytes(selected: SelectedOutput, file_format: str) -> bytes:
    delimiter = {"csv": ",", "tsv": "\t", "txt": " "}[file_format]
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, delimiter=delimiter, lineterminator="\n")
    value_name = _column_name(selected.key)
    if selected.axis_kind == "xt":
        assert selected.y is not None
        writer.writerow(("position_kb", "time_min", value_name))
        for time_index, time_value in enumerate(selected.y):
            row = selected.values[time_index]
            for position_value, output_value in zip(selected.x, row):
                writer.writerow(
                    (
                        _number_text(position_value),
                        _number_text(time_value),
                        _number_text(output_value),
                    )
                )
    else:
        axis_name = {
            "space": "position_kb",
            "time": "time_min",
            "replicate": "replicate",
        }[selected.axis_kind]
        writer.writerow((axis_name, value_name))
        writer.writerows(
            (_number_text(axis_value), _number_text(output_value))
            for axis_value, output_value in zip(selected.x, selected.values)
        )
    return stream.getvalue().encode("utf-8")


def _binary_bytes(selected: SelectedOutput, file_format: str) -> bytes:
    stream = io.BytesIO()
    if file_format == "npy":
        np.save(stream, selected.values, allow_pickle=False)
        return stream.getvalue()

    if selected.axis_kind == "xt":
        assert selected.y is not None
        np.savez_compressed(
            stream,
            values=selected.values,
            position_kb=selected.x,
            time_min=selected.y,
        )
        return stream.getvalue()
    if selected.axis_kind == "space":
        np.savez_compressed(stream, values=selected.values, position_kb=selected.x)
    elif selected.axis_kind == "time":
        np.savez_compressed(stream, values=selected.values, time_min=selected.x)
    else:
        np.savez_compressed(stream, values=selected.values, replicate=selected.x)
    return stream.getvalue()


def _number_text(value: float) -> str:
    text = repr(float(value))
    return text[:-2] if text.endswith(".0") else text


def _column_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_") or "value"


def _humanize(value: str) -> str:
    return value.replace("_xt", "").replace("_", " ").strip().title() or "Simulation output"
