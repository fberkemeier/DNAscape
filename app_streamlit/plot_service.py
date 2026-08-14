"""Bounded, side-effect-free plot construction for the Streamlit app."""

from __future__ import annotations

from typing import Any, Literal, cast

import numpy as np

from app_streamlit.preview import preview_indices, preview_matrix, preview_series

DEFAULT_LINE_POINTS = 5_000
DEFAULT_HEATMAP_ROWS = 900
DEFAULT_HEATMAP_COLUMNS = 420
DEFAULT_HISTOGRAM_SAMPLES = 250_000
MAX_HISTOGRAM_BINS = 10_000
_VALID_NUMERIC_KINDS = frozenset("biuf")
LegendLocation = Literal[
    "best",
    "upper right",
    "upper left",
    "lower left",
    "lower right",
    "right",
    "center left",
    "center right",
    "lower center",
    "upper center",
    "center",
]
_LEGEND_LOCATIONS: tuple[LegendLocation, ...] = (
    "best",
    "upper right",
    "upper left",
    "lower left",
    "lower right",
    "right",
    "center left",
    "center right",
    "lower center",
    "upper center",
    "center",
)


def _numeric_array(values: Any, label: str) -> np.ndarray:
    """Return a real numeric array or fail before Matplotlib allocates a figure."""

    array = np.asarray(values)
    if array.dtype.kind not in _VALID_NUMERIC_KINDS:
        raise ValueError(f"{label} must contain real numeric values.")
    return array


def _all_finite(values: np.ndarray, *, chunk_size: int = 1_000_000) -> bool:
    """Check finiteness without allocating a full-size temporary mask."""

    flat = values.reshape(-1)
    for start in range(0, flat.size, chunk_size):
        if not np.all(np.isfinite(flat[start : start + chunk_size])):
            return False
    return True


def _has_finite(values: np.ndarray, *, chunk_size: int = 1_000_000) -> bool:
    """Find at least one finite value with bounded temporary memory."""

    flat = values.reshape(-1)
    for start in range(0, flat.size, chunk_size):
        if np.any(np.isfinite(flat[start : start + chunk_size])):
            return True
    return False


def _validated_limits(
    limits: tuple[float, float] | None,
    label: str,
    *,
    positive: bool = False,
) -> tuple[float, float] | None:
    if limits is None:
        return None
    try:
        low, high = (float(value) for value in limits)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} limits must contain two numbers.") from exc
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        raise ValueError(f"{label} limits must be finite and satisfy minimum < maximum.")
    if positive and low <= 0:
        raise ValueError(f"{label} limits must be positive when using a log scale.")
    return low, high


def _axis_extent(values: np.ndarray, label: str) -> tuple[float, float]:
    """Convert coordinate centers to stable image edges, including one-cell axes."""

    axis = np.asarray(values, dtype=float).reshape(-1)
    if not _all_finite(axis):
        raise ValueError(f"{label} coordinates must be finite.")
    if axis.size == 1:
        center = float(axis[0])
        return center - 0.5, center + 0.5
    deltas = np.diff(axis)
    if not (np.all(deltas > 0) or np.all(deltas < 0)):
        raise ValueError(f"{label} coordinates must be strictly monotonic.")
    return (
        float(axis[0] - deltas[0] / 2.0),
        float(axis[-1] + deltas[-1] / 2.0),
    )


def _new_figure(figsize: tuple[float, float]):
    """Create an Agg-backed figure without importing pyplot or global state."""

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=figsize, constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.add_subplot(1, 1, 1)
    return figure, axes


def line_figure(
    x_values: Any,
    y_values: Any,
    *,
    title: str = "",
    x_label: str = "Position (kb)",
    y_label: str = "Value",
    color: str = "#2c7f77",
    line_width: float = 1.8,
    label: str | None = None,
    legend_location: str = "best",
    show_grid: bool = True,
    x_limits: tuple[float, float] | None = None,
    y_limits: tuple[float, float] | None = None,
    log_y: bool = False,
    invert_y: bool = False,
    figsize: tuple[float, float] = (9.2, 4.5),
    max_points: int = DEFAULT_LINE_POINTS,
    full_resolution: bool = False,
):
    """Build a line figure, uniformly bounding interactive previews by default."""

    x_array = _numeric_array(x_values, "Line x coordinates").reshape(-1)
    y_array = _numeric_array(y_values, "Line values").reshape(-1)
    if x_array.size == 0 or y_array.size == 0:
        raise ValueError("Line plots require at least one x/y value pair.")
    if x_array.size != y_array.size:
        raise ValueError("Line x coordinates and values must have the same length.")
    if not _all_finite(x_array):
        raise ValueError("Line x coordinates must be finite.")
    if not _has_finite(y_array):
        raise ValueError("Line plots require at least one finite y value.")
    try:
        line_width = float(line_width)
    except (TypeError, ValueError) as exc:
        raise ValueError("Line width must be a positive finite number.") from exc
    if not np.isfinite(line_width) or line_width <= 0:
        raise ValueError("Line width must be a positive finite number.")
    if legend_location not in _LEGEND_LOCATIONS:
        raise ValueError(f"Unsupported legend location: {legend_location!r}.")
    resolved_legend_location = cast(LegendLocation, legend_location)
    x_limits = _validated_limits(x_limits, "X")
    y_limits = _validated_limits(y_limits, "Y", positive=bool(log_y))
    if log_y and not np.any(np.isfinite(y_array) & (y_array > 0)):
        raise ValueError("Log-scale line plots require at least one positive y value.")
    if full_resolution:
        x_plot, y_plot = x_array, y_array
    else:
        x_plot, y_plot = preview_series(x_array, y_array, max_points=max_points)

    figure, axes = _new_figure(figsize)
    try:
        axes.plot(x_plot, y_plot, color=color, linewidth=line_width, label=label)
        axes.set(title=title, xlabel=x_label, ylabel=y_label)
        if x_limits is not None:
            axes.set_xlim(*x_limits)
        if y_limits is not None:
            axes.set_ylim(*y_limits)
        if log_y:
            axes.set_yscale("log")
        if invert_y:
            axes.invert_yaxis()
        if label:
            axes.legend(loc=resolved_legend_location)
        axes.grid(bool(show_grid), alpha=0.22, linewidth=0.7)
    except Exception:
        figure.clear()
        raise
    return figure, axes


def heatmap_figure(
    values: Any,
    *,
    title: str = "",
    x_label: str = "Time",
    y_label: str = "Position",
    colorbar_label: str = "Value",
    x_values: Any | None = None,
    y_values: Any | None = None,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    max_rows: int = DEFAULT_HEATMAP_ROWS,
    max_columns: int = DEFAULT_HEATMAP_COLUMNS,
    full_resolution: bool = False,
    show_colorbar: bool = True,
    invert_y: bool = False,
    figsize: tuple[float, float] = (9.2, 5.2),
):
    """Build a space-time heatmap with bounded preview dimensions by default."""

    matrix = _numeric_array(values, "Heatmap data")
    if matrix.ndim != 2 or 0 in matrix.shape:
        raise ValueError("Heatmaps require a non-empty two-dimensional array.")
    if not _has_finite(matrix):
        raise ValueError("Heatmaps require at least one finite value.")
    if vmin is not None:
        vmin = float(vmin)
        if not np.isfinite(vmin):
            raise ValueError("Heatmap minimum must be finite.")
    if vmax is not None:
        vmax = float(vmax)
        if not np.isfinite(vmax):
            raise ValueError("Heatmap maximum must be finite.")
    if vmin is not None and vmax is not None and vmax <= vmin:
        raise ValueError("Heatmap maximum must be greater than its minimum.")
    if full_resolution:
        plot_matrix = matrix
        row_indices = np.arange(matrix.shape[0], dtype=int)
        column_indices = np.arange(matrix.shape[1], dtype=int)
    else:
        plot_matrix, row_indices, column_indices = preview_matrix(
            matrix,
            max_rows=max_rows,
            max_columns=max_columns,
        )

    extent = None
    if x_values is not None or y_values is not None:
        x_axis = (
            _numeric_array(x_values, "Heatmap x coordinates").reshape(-1)
            if x_values is not None
            else np.arange(matrix.shape[1])
        )
        y_axis = (
            _numeric_array(y_values, "Heatmap y coordinates").reshape(-1)
            if y_values is not None
            else np.arange(matrix.shape[0])
        )
        if x_axis.size != matrix.shape[1] or y_axis.size != matrix.shape[0]:
            raise ValueError("Heatmap coordinate axes must match matrix columns and rows.")
        x_plot = x_axis[column_indices]
        y_plot = y_axis[row_indices]
        x_left, x_right = _axis_extent(x_plot, "Heatmap x")
        y_bottom, y_top = _axis_extent(y_plot, "Heatmap y")
        extent = (x_left, x_right, y_bottom, y_top)

    figure, axes = _new_figure(figsize)
    try:
        image = axes.imshow(
            plot_matrix,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            extent=extent,
        )
        axes.set(title=title, xlabel=x_label, ylabel=y_label)
        if invert_y:
            axes.invert_yaxis()
        if show_colorbar:
            colorbar = figure.colorbar(image, ax=axes, pad=0.02)
            colorbar.set_label(colorbar_label)
    except Exception:
        figure.clear()
        raise
    return figure, axes


def histogram_figure(
    values: Any,
    *,
    bins: int = 60,
    title: str = "",
    x_label: str = "Value",
    y_label: str = "Count",
    color: str = "#2c7f77",
    max_samples: int = DEFAULT_HISTOGRAM_SAMPLES,
    full_resolution: bool = False,
):
    """Build a finite-value histogram, sampling only interactive previews."""

    flat = _numeric_array(values, "Histogram data").reshape(-1)
    if flat.size == 0:
        raise ValueError("Histograms require at least one value.")
    try:
        bins = int(bins)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Histogram bins must be a positive integer.") from exc
    if bins < 1 or bins > MAX_HISTOGRAM_BINS:
        raise ValueError(
            f"Histogram bins must be between 1 and {MAX_HISTOGRAM_BINS:,}."
        )
    candidates = flat if full_resolution else flat[preview_indices(flat.size, max_samples)]
    finite = candidates[np.isfinite(candidates)]
    if finite.size == 0:
        raise ValueError("The selected data contain no finite values to plot.")

    figure, axes = _new_figure((8.2, 4.5))
    try:
        axes.hist(finite, bins=bins, color=color, alpha=0.86)
        axes.set(title=title, xlabel=x_label, ylabel=y_label)
        axes.grid(False)
    except Exception:
        figure.clear()
        raise
    return figure, axes


def release_figure(figure: Any) -> None:
    """Release artists held by a locally constructed figure."""

    figure.clear()
