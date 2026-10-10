"""Headless ECharts options shared by the app and downloadable workflows.

This module contains the same data preparation, limits, sampling and chart
specification used by the interactive previews. It does not import NiceGUI.
"""

from __future__ import annotations

import base64
import io
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from html import escape
from typing import Any

import numpy as np

PALETTES: dict[str, list[str]] = {
    "viridis": ["#440154", "#31688e", "#35b779", "#fde725"],
    "magma": ["#000004", "#721f81", "#f1605d", "#fcfdbf"],
    "plasma": ["#0d0887", "#9c179e", "#ed7953", "#f0f921"],
    "cividis": ["#00204c", "#575d6d", "#a59c74", "#ffee99"],
    "coolwarm": ["#3b4cc0", "#b9d0e5", "#f5c4ad", "#b40426"],
}

MAX_HEATMAP_ROWS = 300
MAX_HEATMAP_COLUMNS = 720
MAX_HEATMAP_CELLS = MAX_HEATMAP_ROWS * MAX_HEATMAP_COLUMNS
MAX_HEATMAP_TOOLTIP_CELLS = 2_400
MAX_HEATMAP_WORK_CELLS = 262_144
# Dense overlays remain fast and readable without a per-point hover surface.
# Six curves is the largest combined view where the tooltip still helps more
# than it obscures the data.
MAX_INTERACTIVE_CURVES = 6

FIXED_QUANTITY_Y_LIMITS: dict[str, tuple[float, float]] = {
    "origin_efficiency": (0.0, 1.0),
    "fork_directionality": (-1.0, 1.0),
    "replicated_fraction": (0.0, 1.0),
}

# ECharts' ``containLabel`` changes the location of the actual axis after the
# grid rectangle has been calculated.  Combining it with a dynamic ``left``
# gutter therefore reserves the tick-label width twice and makes the plot jump
# horizontally as labels change.  We size the complete left gutter ourselves
# instead: tick labels, their margin, the rotated title, and a small outer
# breathing space are all deterministic.
_PLOT_GRID_LEFT = 30
_PLOT_GRID_RIGHT = 24
_PLOT_GRID_TOP = 24
_PLOT_GRID_BOTTOM = 58
_HEATMAP_GRID_RIGHT = 86
_X_AXIS_NAME_GAP = 28
_Y_AXIS_NAME_GAP_MIN = 28
_Y_AXIS_NAME_GAP_MAX = 82
_AXIS_LABEL_FONT_SIZE = 10
_AXIS_LABEL_MARGIN = 7
_AXIS_NAME_FONT_SIZE = 11
_Y_AXIS_TITLE_TICK_CLEARANCE = 5
_PLOT_OUTER_GUTTER = 8
_AXIS_MARGIN_FRACTION = 0.04

# Keep hover text consistent even when ECharts would otherwise expose the full
# binary representation of a floating-point value.
_TOOLTIP_VALUE_FORMATTER = (
    "value => { const n = Number(value); if (!Number.isFinite(n)) return value; "
    "return n !== 0 && Math.abs(n) < 0.01 ? n.toExponential(2) : n.toFixed(2); }"
)
_TOOLTIP_POINTER_FORMATTER = (
    "params => { const n = Number(params.value); "
    "return Number.isFinite(n) ? n.toFixed(2) : params.value; }"
)

CURVE_TYPES = {"solid": "Solid", "dashed": "Dashed", "dotted": "Dotted", "dashdot": "Dash-dot"}


def curve_dash_pattern(line_type: str) -> str | list[int]:
    """Translate a named curve pattern to the canvas and SVG renderers."""
    if line_type not in CURVE_TYPES:
        raise ValueError(f"Unknown curve type {line_type!r}.")
    return [8, 4, 2, 4] if line_type == "dashdot" else line_type


@dataclass(frozen=True)
class PreviewLine:
    """One named line supplied to the shared preview renderer."""

    name: str
    x: np.ndarray
    y: np.ndarray
    color: str
    opacity: float = 1.0
    width: float | None = None
    show_in_legend: bool = True
    z: int = 3
    line_type: str | None = None


@dataclass(frozen=True)
class PreviewBand:
    """One filled interval supplied to the shared line renderer."""

    name: str
    x: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    color: str
    opacity: float = 0.16
    show_in_legend: bool = True


def _finite_number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def _positive_finite(value: Any) -> float | None:
    numeric = _finite_number(value)
    return numeric if numeric is not None and numeric > 0 else None


def _histogram_bin_count(value: Any) -> int | None:
    numeric = _positive_finite(value)
    if numeric is None or not numeric.is_integer():
        return None
    count = int(numeric)
    return count if 1 <= count <= 2_000 else None


def histogram_domain(
    values: Any,
    *,
    log_x: bool = False,
    x_limits: tuple[float, float] | None = None,
) -> tuple[np.ndarray, tuple[float, float]] | None:
    """Return finite histogram values and the exact domain used for binning."""

    finite = finite_values(values)
    if log_x:
        finite = finite[finite > 0]
    if finite.size == 0:
        return None
    if x_limits is None:
        lower = float(np.min(finite))
        upper = float(np.max(finite))
    else:
        lower = float(x_limits[0])
        upper = float(x_limits[1])
        if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
            raise ValueError("Histogram X maximum must be greater than X minimum.")
        if log_x and lower <= 0:
            raise ValueError("Histogram X limits must be positive on a logarithmic axis.")
        finite = finite[(finite >= lower) & (finite <= upper)]
    if upper == lower:
        if log_x:
            factor = math.sqrt(10.0)
            lower, upper = lower / factor, upper * factor
        else:
            # This is NumPy's native constant-data expansion.
            lower, upper = lower - 0.5, upper + 0.5
    return finite, (lower, upper)


def _histogram_span(
    values: Any,
    *,
    log_x: bool = False,
    x_limits: tuple[float, float] | None = None,
) -> float | None:
    domain = histogram_domain(values, log_x=log_x, x_limits=x_limits)
    return None if domain is None else domain[1][1] - domain[1][0]


def approximate_histogram_bin_size(
    values: Any,
    bins: Any,
    *,
    log_x: bool = False,
    x_limits: tuple[float, float] | None = None,
) -> float | None:
    """Return the data-derived bin width shown in the shared controls."""

    count = _histogram_bin_count(bins)
    span = _histogram_span(values, log_x=log_x, x_limits=x_limits)
    if count is None or span is None:
        return None
    return span / count


def approximate_histogram_bin_count(
    values: Any,
    bin_size: Any,
    *,
    log_x: bool = False,
    x_limits: tuple[float, float] | None = None,
) -> int | None:
    """Return a bounded bin count for the requested bin width."""

    width = _positive_finite(bin_size)
    span = _histogram_span(values, log_x=log_x, x_limits=x_limits)
    if width is None or span is None:
        return None
    count = int(math.ceil(span / width - 1e-12))
    return max(1, min(2_000, count))


def _histogram_edges(
    histogram_range: tuple[float, float],
    *,
    bins: int,
    bin_size: float | None,
) -> np.ndarray:
    """Build safe histogram edges while preserving arbitrary valid widths."""

    lower, upper = histogram_range
    span = upper - lower
    width = _positive_finite(bin_size)
    if width is None:
        count = max(1, min(2_000, int(bins)))
        return np.linspace(lower, upper, count + 1, dtype=float)
    requested_count = max(1, int(math.ceil(span / width - 1e-12)))
    if requested_count > 2_000:
        return np.linspace(lower, upper, 2_001, dtype=float)
    edges = lower + np.arange(requested_count + 1, dtype=float) * width
    edges[-1] = upper
    return edges


def _histogram_data(
    values: Any,
    *,
    bins: int,
    bin_size: float | None,
    density: bool,
    log_x: bool,
    x_limits: tuple[float, float] | None,
    boundary: str = "left",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return plotted values, counts, and edges for one histogram."""

    domain = histogram_domain(values, log_x=log_x, x_limits=x_limits)
    if domain is None:
        raise ValueError("The selected output contains no values for this histogram.")
    finite, histogram_range = domain
    edges = _histogram_edges(histogram_range, bins=bins, bin_size=bin_size)
    normalized_boundary = str(boundary or "left").strip().casefold()
    if normalized_boundary not in {"left", "right"}:
        raise ValueError("Histogram bin boundaries must be 'left' or 'right'.")
    if finite.size and normalized_boundary == "right":
        # Negating and reversing converts NumPy's native [a, b) ownership to
        # (a, b] without losing either outer endpoint. This remains correct for
        # non-uniform edges, including a short final bin from a custom width.
        counts = np.histogram(-finite, bins=-edges[::-1], density=False)[0][::-1].astype(float)
    elif finite.size:
        counts = np.histogram(finite, bins=edges, density=False)[0].astype(float)
    else:
        counts = np.zeros(edges.size - 1, dtype=float)
    if density and counts.sum() > 0:
        counts /= counts.sum() * np.diff(edges)
    return finite, counts, edges


def finite_values(values: Any) -> np.ndarray:
    """Return a flattened finite float view of arbitrary numeric data."""

    flattened = np.asarray(values, dtype=float).reshape(-1)
    return flattened[np.isfinite(flattened)]


def has_finite_values(values: Any) -> bool:
    """Check a large array without allocating a mask for its full extent."""

    matrix = np.asarray(values)
    for start in range(0, matrix.size, MAX_HEATMAP_WORK_CELLS):
        block = np.asarray(matrix.flat[start : start + MAX_HEATMAP_WORK_CELLS], dtype=float)
        if np.isfinite(block).any():
            return True
    return False


def _validate_log_axis(values: Any, label: str, *, bounds: bool = False) -> None:
    finite = finite_values(values)
    if finite.size == 0 or not np.any(finite > 0) or (bounds and np.any(finite <= 0)):
        raise ValueError(f"{label} requires positive values.")


def nice_linear_limits(
    lower: float,
    upper: float,
    *,
    floor_at_zero: bool = False,
    intervals: int = 6,
) -> tuple[float, float]:
    """Return exact data bounds with a small scale-relative outer margin.

    Tick spacing is deliberately handled separately.  This prevents a value
    axis from growing to the next convenient tick (for example, 12 for data
    ending at 10.2) while retaining readable interior ticks.  ``intervals`` is
    accepted for backward compatibility with callers of the earlier helper.
    """

    del intervals
    if not math.isfinite(lower) or not math.isfinite(upper):
        raise ValueError("Plot axes require finite limits.")
    if upper < lower:
        lower, upper = upper, lower
    span = upper - lower
    if span > 0:
        padding = span * _AXIS_MARGIN_FRACTION
    else:
        # Constant data has no range from which to derive a margin.  Use its
        # order of magnitude, with a modest unit-scale fallback around zero.
        padding = max(abs(lower), 1.0) * _AXIS_MARGIN_FRACTION
    padded_lower = lower - padding
    if floor_at_zero and lower >= 0:
        padded_lower = 0.0
    elif lower == 0.0 and upper > 0.0:
        # Positions and other naturally zero-based axes should not acquire a
        # small, scientifically meaningless negative coordinate.
        padded_lower = 0.0
    return padded_lower, upper + padding


def _logarithmic_data_limits(lower: float, upper: float) -> tuple[float, float]:
    """Pad a positive domain symmetrically in log space."""

    if not math.isfinite(lower) or not math.isfinite(upper) or lower <= 0 or upper <= 0:
        raise ValueError("Log axes require positive finite limits.")
    if upper < lower:
        lower, upper = upper, lower
    log_lower = math.log10(lower)
    log_upper = math.log10(upper)
    log_span = log_upper - log_lower
    padding = (
        log_span * _AXIS_MARGIN_FRACTION
        if log_span > 0
        else max(abs(log_lower), 1.0) * _AXIS_MARGIN_FRACTION
    )
    return 10.0 ** (log_lower - padding), 10.0 ** (log_upper + padding)


def _nice_tick_step(span: float, intervals: int = 6) -> float:
    """Return a compact 1/2/2.5/5 tick step for a positive numeric span."""

    if not math.isfinite(span) or span <= 0:
        return 1.0
    rough_step = span / max(1, intervals)
    exponent = math.floor(math.log10(rough_step))
    magnitude = 10.0**exponent
    fraction = rough_step / magnitude
    nice_fraction = next(
        candidate for candidate in (1.0, 2.0, 2.5, 5.0, 10.0) if fraction <= candidate
    )
    return nice_fraction * magnitude


def _nice_axis_limits(
    lower: float,
    upper: float,
    *,
    floor_at_zero: bool = False,
    intervals: int = 6,
) -> tuple[float, float]:
    """Expand a numeric domain to boundaries aligned with readable ticks."""

    if not math.isfinite(lower) or not math.isfinite(upper):
        raise ValueError("Plot axes require finite limits.")
    if upper <= lower:
        padding = max(abs(lower) * 0.05, 0.5)
        lower -= padding
        upper += padding
    step = _nice_tick_step(upper - lower, intervals)
    nice_lower = 0.0 if floor_at_zero and lower >= 0 else math.floor(lower / step) * step
    nice_upper = math.ceil(upper / step) * step
    if math.isclose(nice_lower, nice_upper):
        nice_upper = nice_lower + step
    if math.isclose(nice_lower, 0.0, abs_tol=step * 1e-12):
        nice_lower = 0.0
    return nice_lower, nice_upper


_TICK_FORMATTER = (
    "value => { const n = Number(value); if (!Number.isFinite(n)) return value; "
    "if (Math.abs(n) < 1e-12) return '0'; const a = Math.abs(n); "
    "if (a >= 1e5 || a < 1e-3) return n.toExponential(2)"
    ".replace(/\\.0+e/, 'e').replace(/(\\.[0-9]*?)0+e/, '$1e'); "
    "return Number(n.toPrecision(5)).toString(); }"
)


def _axis_label_options(
    *,
    hide_padded_minimum: bool = False,
    hide_padded_maximum: bool = False,
) -> dict[str, Any]:
    options: dict[str, Any] = {
        "color": "#746d68",
        "fontSize": _AXIS_LABEL_FONT_SIZE,
        "lineHeight": 12,
        "margin": _AXIS_LABEL_MARGIN,
        "hideOverlap": True,
        # NiceGUI updates an existing ECharts instance with merge semantics.
        # Explicit defaults prevent custom ticks and endpoint visibility from
        # leaking into a different plot type after a live view change.
        "customValues": None,
        "showMinLabel": not hide_padded_minimum,
        "showMaxLabel": not hide_padded_maximum,
        ":formatter": _TICK_FORMATTER,
    }
    return options


def _axis_name_options() -> dict[str, Any]:
    """Return the shared, legible axis-title style."""

    return {
        "color": "#625c57",
        "fontSize": _AXIS_NAME_FONT_SIZE,
        "fontWeight": 400,
    }


def _axis_tick_candidates(lower: float, upper: float, *, log_axis: bool) -> list[float]:
    """Return representative displayed ticks for layout estimation."""

    if log_axis:
        if lower <= 0 or upper <= 0:
            return [lower, upper]
        first = math.ceil(math.log10(lower))
        last = math.floor(math.log10(upper))
        candidates = [10.0**exponent for exponent in range(first, last + 1)]
        return candidates or [lower, upper]
    step = _nice_tick_step(upper - lower)
    tolerance = step * 1e-9
    # The axes in this module have explicit minima. With an explicit
    # ``interval`` ECharts advances ticks from that minimum, rather than from
    # the next round multiple of the interval. Starting at a padded minimum
    # such as 17.16 therefore yields labels 22.16, 27.16, ...; estimating only
    # the round labels 20, 25, ... substantially under-reserves title space.
    candidates: list[float] = []
    tick = lower
    while tick <= upper + tolerance and len(candidates) < 20:
        candidates.append(tick)
        tick += step
    return candidates or [lower, upper]


def _aligned_axis_ticks(lower: float, upper: float, *, log_axis: bool) -> list[float]:
    """Return clean tick values inside a data domain, including aligned endpoints.

    Plot limits include a small, scale-relative margin.  Letting ECharts derive
    ticks from those explicit limits anchors every tick to the padded minimum;
    a domain starting at ``17.16`` consequently produces ``22.16, 27.16, ...``.
    ECharts 6 supports explicit tick and label values, so keep the exact padded
    coordinate domain while deriving readable ticks from the data itself.
    """

    if not math.isfinite(lower) or not math.isfinite(upper):
        return []
    if upper < lower:
        lower, upper = upper, lower
    if log_axis:
        if lower <= 0 or upper <= 0:
            return []
        tolerance = 1e-12
        first = math.ceil(math.log10(lower) - tolerance)
        last = math.floor(math.log10(upper) + tolerance)
        return [10.0**exponent for exponent in range(first, last + 1)]

    span = upper - lower
    if math.isclose(span, 0.0):
        return [lower]
    step = _nice_tick_step(span)
    tolerance = step * 1e-9
    first = math.ceil((lower - tolerance) / step) * step
    ticks: list[float] = []
    index = 0
    while index < 128:
        tick = first + index * step
        if tick > upper + tolerance:
            break
        if tick >= lower - tolerance:
            clean_tick = 0.0 if math.isclose(tick, 0.0, abs_tol=tolerance) else tick
            ticks.append(float(f"{clean_tick:.12g}"))
        index += 1
    return ticks


def _use_endpoint_safe_ticks(
    axis_options: dict[str, Any],
    *,
    data_lower: float,
    data_upper: float,
    log_axis: bool,
    tick_lower: float | None = None,
    tick_upper: float | None = None,
) -> None:
    """Install aligned ticks without turning padded limits into grid lines.

    ``showMinLine`` and ``showMaxLine`` refer to the first and last *custom
    tick* in ECharts 6, not to the padded coordinate limits. Every value in
    ``ticks`` is therefore a genuine displayed tick and must retain its grid
    line, including an aligned data endpoint such as a histogram count of 12.
    Artificial padded boundaries are absent from ``customValues`` already, so
    enabling these lines cannot add a grid line at a non-ticked boundary.
    """

    # ``None`` is deliberate: omitting a formerly set interval would leave it
    # behind when NiceGUI calls ECharts with ``notMerge=False``. Clear it even
    # for a narrow logarithmic domain which has no whole-decade custom tick.
    axis_options["interval"] = None
    ticks = _aligned_axis_ticks(
        data_lower if tick_lower is None else tick_lower,
        data_upper if tick_upper is None else tick_upper,
        log_axis=log_axis,
    )
    if not ticks:
        return
    axis_options["axisTick"] = {
        **axis_options.get("axisTick", {}),
        "customValues": ticks,
    }
    axis_label = {**axis_options.get("axisLabel", {})}
    # Custom values never contain the artificial padded boundaries, so the
    # first and last meaningful labels must not inherit the old padding guards.
    axis_label["showMinLabel"] = True
    axis_label["showMaxLabel"] = True
    axis_label["customValues"] = ticks
    axis_options["axisLabel"] = axis_label
    axis_options["splitLine"] = {
        **axis_options.get("splitLine", {}),
        # ECharts resolves split-line coordinates from the splitLine model;
        # axisTick/axisLabel customValues do not implicitly carry over.
        "customValues": ticks,
        "showMinLine": True,
        "showMaxLine": True,
    }


def _y_axis_name_gap(
    lower: float,
    upper: float,
    *,
    log_axis: bool = False,
    tick_values: Sequence[float] | None = None,
) -> int:
    """Reserve enough space between the axis and its rotated title.

    ECharts measures ``nameGap`` from the axis line to the nearest edge of the
    title, not from its tick labels. The widest formatted tick therefore has
    to be part of this value; the title's own width belongs in the outer grid
    margin rather than in the space between the title and ticks.
    """

    candidates = (
        list(tick_values)
        if tick_values is not None
        else _axis_tick_candidates(lower, upper, log_axis=log_axis)
    )
    labels = [_compact_tick(value) for value in candidates]
    widest = max((len(label) for label in labels), default=1)
    # At the explicit 10 px axis-label font, six pixels per glyph is a safe
    # upper estimate for tabular digits and scientific notation. Include the
    # label margin and a visible clearance.
    estimated = int(
        math.ceil(
            widest * 6.0
            + _AXIS_LABEL_MARGIN
            + _Y_AXIS_TITLE_TICK_CLEARANCE
        )
    )
    return max(_Y_AXIS_NAME_GAP_MIN, min(_Y_AXIS_NAME_GAP_MAX, estimated))


def _cartesian_grid(
    *,
    heatmap_colorbar: bool = False,
    y_name_gap: int = _Y_AXIS_NAME_GAP_MIN,
    y_name: str = "",
    bottom_legend: bool = False,
) -> dict[str, Any]:
    """Return deterministic outer chart margins shared by every plot type."""

    if y_name:
        # Each line becomes another title-width after the 90-degree rotation.
        # Keep its complete text box and an antialiasing gutter inside the
        # canvas instead of relying on overflow outside the chart surface.
        title_width = _AXIS_NAME_FONT_SIZE * len(y_name.split("\n"))
        left = math.ceil(y_name_gap + title_width + _PLOT_OUTER_GUTTER)
    else:
        # ``y_name_gap`` also encodes the estimated tick width. Remove the
        # title-specific allowance when there is no visible Y-axis name.
        left = math.ceil(
            y_name_gap
            - _Y_AXIS_TITLE_TICK_CLEARANCE
            + _PLOT_OUTER_GUTTER
        )

    return {
        "left": max(_PLOT_GRID_LEFT, left),
        "right": _HEATMAP_GRID_RIGHT if heatmap_colorbar else _PLOT_GRID_RIGHT,
        "top": _PLOT_GRID_TOP,
        "bottom": _PLOT_GRID_BOTTOM + (48 if bottom_legend else 0),
        # With explicit pixel geometry, asking ECharts to contain the labels
        # would add their width a second time and create the large left pad.
        "containLabel": False,
        # ECharts 6 otherwise adjusts this explicit geometry a second time
        # to contain axis names. Keep the same pixel spacing in every panel.
        "outerBoundsMode": "none",
    }


def build_legend_options(
    names: Sequence[str],
    *,
    grid: dict[str, Any],
    position: str = "Top right",
    show: bool = True,
    interactive: bool = True,
    item_width: int = 20,
    item_height: int = 3,
) -> dict[str, Any]:
    """Anchor a scrollable legend inside the selected corner of the axes."""

    normalized_position = str(position or "Top right").strip().casefold()
    vertical_edge = "bottom" if normalized_position.startswith("bottom") else "top"
    horizontal_edge = "left" if normalized_position.endswith("left") else "right"
    return {
        "show": bool(show),
        "type": "scroll",
        "selectedMode": bool(interactive),
        "orient": "vertical",
        # ECharts offsets refer to the chart container. Include the axis
        # gutter so bottom legends stay above the X axis and its labels.
        vertical_edge: int(grid["bottom"]) + 8 if vertical_edge == "bottom" else 24,
        horizontal_edge: (
            max(84, int(grid["left"]) + 8)
            if horizontal_edge == "left"
            else int(grid["right"]) + 4
        ),
        "itemWidth": item_width,
        "itemHeight": item_height,
        "backgroundColor": "rgba(255,255,255,0.84)",
        "borderColor": "#ddd7d1",
        "borderWidth": 1,
        "borderRadius": 4,
        "padding": [4, 6],
        "textStyle": {"color": "#625c57", "fontSize": 11},
        "data": list(names),
    }


def _data_axis_bounds(
    values: np.ndarray,
    *,
    log_axis: bool = False,
) -> tuple[float, float]:
    """Return exact finite data bounds, matching ``plothm`` image extents."""

    finite = np.asarray(values, dtype=float).reshape(-1)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("Plot axes require at least one finite coordinate.")
    lower = float(np.min(finite))
    upper = float(np.max(finite))
    # A small, nonzero interval still has physical coordinates. In particular,
    # relative closeness at a large chromosome offset must not turn a narrow
    # selection into a synthetic singleton with a wide padded extent.
    if lower == upper:
        if log_axis:
            if lower <= 0:
                raise ValueError("Log axes require positive finite coordinates.")
            factor = 10.0**0.1
            return lower / factor, upper * factor
        padding = max(abs(lower) * 0.05, 0.5)
        return lower - padding, upper + padding
    return lower, upper


def _compact_tick(value: float) -> str:
    magnitude = abs(float(value))
    if magnitude >= 100_000 or (magnitude and magnitude < 0.001):
        return f"{value:.2e}".replace(".00e", "e").replace(".0e", "e")
    return f"{value:.5g}"


def _limits_match(
    actual: tuple[float, float],
    expected: tuple[float, float],
) -> bool:
    """Whether an axis interval is the data-derived padded interval."""

    return all(
        math.isclose(left, right, rel_tol=1e-9, abs_tol=max(abs(right), 1.0) * 1e-12)
        for left, right in zip(actual, expected)
    )


def _automatic_linear_limits(
    limits: tuple[float, float],
    lower: float,
    upper: float,
) -> bool:
    """Recognize an automatic margin while preserving explicit user limits."""

    return _limits_match(limits, nice_linear_limits(lower, upper)) or _limits_match(
        limits,
        nice_linear_limits(lower, upper, floor_at_zero=True),
    )


def _fixed_scientific_limits(limits: tuple[float, float]) -> bool:
    return any(_limits_match(limits, fixed) for fixed in set(FIXED_QUANTITY_Y_LIMITS.values()))


def _whole_numbers(values: Any | None) -> bool:
    if values is None:
        return False
    finite = finite_values(values)
    return bool(finite.size and np.all(np.abs(finite - np.round(finite)) < 1e-9))


def _hover_formatter(*, integer_only: bool) -> str:
    return (
        "value => { const n = Number(value); if (!Number.isFinite(n)) return value; "
        "if (n !== 0 && Math.abs(n) < 0.01) return n.toExponential(2); "
        f"return {str(integer_only).lower()} && Number.isInteger(n) ? String(n) : n.toFixed(2); }}"
    )


def _visual_axis_endpoint(values: Any) -> float | None:
    """Include a round zero-based bin boundary without extending arbitrary data."""
    axis = np.unique(np.asarray(values, dtype=float).reshape(-1))
    axis = axis[np.isfinite(axis)]
    if axis.size < 2 or not math.isclose(float(axis[0]), 0, abs_tol=1e-9):
        return None
    if math.isclose(float(axis[-1]) / 100, round(float(axis[-1]) / 100), abs_tol=1e-9):
        return None
    steps = np.diff(axis)
    if np.allclose(steps, steps[0], rtol=1e-8, atol=1e-9):
        step = float(steps[0])
    elif _whole_numbers(axis):
        # Downsampled previews still retain a common underlying integer grid.
        step = float(np.gcd.reduce(np.rint(steps).astype(np.int64)))
    else:
        return None
    end = float(axis[-1]) + step
    return end if end >= 100 and math.isclose(end / 100, round(end / 100), abs_tol=1e-9) else None


def _tooltip_options(
    *, interactive: bool, trigger: str, x_values: Any = None, values: Any = None,
) -> dict[str, Any]:
    """Return consistently rounded hover values for every interactive plot."""

    return {
        "show": bool(interactive),
        "trigger": trigger,
        "confine": True,
        ":valueFormatter": _hover_formatter(integer_only=_whole_numbers(values)),
        "axisPointer": {
            "type": "line",
            "label": {
                "show": bool(interactive),
                ":formatter": "params => (" + _hover_formatter(integer_only=_whole_numbers(x_values)) + ")(params.value)",
            },
        },
    }


def limit_curve_hover(
    options: dict[str, Any],
    *,
    maximum: int = MAX_INTERACTIVE_CURVES,
) -> dict[str, Any]:
    """Disable dense curve hover details while retaining legend selection."""

    curves = [
        item
        for item in options.get("series", ())
        if isinstance(item, dict) and item.get("type") in {"line", "scatter"}
    ]
    if len(curves) <= max(0, int(maximum)):
        return options
    tooltip = dict(options.get("tooltip", {}))
    tooltip["show"] = False
    axis_pointer = dict(tooltip.get("axisPointer", {}))
    pointer_label = dict(axis_pointer.get("label", {}))
    pointer_label["show"] = False
    axis_pointer["label"] = pointer_label
    tooltip["axisPointer"] = axis_pointer
    options["tooltip"] = tooltip
    for curve in curves:
        curve["silent"] = True
        emphasis = dict(curve.get("emphasis", {}))
        emphasis["disabled"] = True
        curve["emphasis"] = emphasis
    return options


def _histogram_statistics_graphic(
    values: np.ndarray,
    *,
    position: str,
) -> list[dict[str, Any]]:
    """Return a compact, legend-like histogram summary overlay."""

    normalized_position = str(position or "Top right").strip().casefold()
    placement: dict[str, Any] = {
        "top": 24 if normalized_position.startswith("top") else None,
        "bottom": 8 if normalized_position.startswith("bottom") else None,
        "left": 84 if normalized_position.endswith("left") else None,
        "right": 28 if normalized_position.endswith("right") else None,
    }
    placement = {key: value for key, value in placement.items() if value is not None}
    summary = (
        "No values in range"
        if values.size == 0
        else (
            f"Mean       {_compact_tick(float(np.mean(values)))}\n"
            f"Median    {_compact_tick(float(np.median(values)))}\n"
            f"Std dev   {_compact_tick(float(np.std(values)))}"
        )
    )
    return [
        {
            "type": "text",
            **placement,
            # Keep the summary above custom histogram rectangles, axis
            # pointers, and any later series added by a caller.
            "z": 10_000,
            "z2": 10_000,
            "zlevel": 20,
            "silent": True,
            "style": {
                "text": summary,
                "fill": "#625c57",
                "font": "11px sans-serif",
                "lineHeight": 16,
                "backgroundColor": "rgba(255,255,255,0.88)",
                "borderColor": "#ddd7d1",
                "borderWidth": 1,
                "borderRadius": 4,
                "padding": [5, 7],
            },
        }
    ]


def _palette_rgb(palette: str) -> np.ndarray:
    colours = PALETTES.get(palette, PALETTES["viridis"])
    return np.asarray(
        [tuple(int(colour[index : index + 2], 16) for index in (1, 3, 5)) for colour in colours],
        dtype=float,
    )


def _heatmap_data_uri(
    matrix: np.ndarray,
    *,
    palette: str,
    z_minimum: float,
    z_maximum: float,
    invert_y: bool,
    invert_x: bool = False,
    invert_z: bool = False,
) -> str:
    """Encode a bounded heatmap raster using the same nearest-image model as plotf."""

    from PIL import Image

    finite = np.isfinite(matrix)
    scale = np.zeros(matrix.shape, dtype=float)
    np.subtract(matrix, z_minimum, out=scale, where=finite)
    scale /= z_maximum - z_minimum
    np.clip(scale, 0.0, 1.0, out=scale)
    anchors = _palette_rgb(palette)
    if invert_z:
        anchors = anchors[::-1]
    positions = scale * (anchors.shape[0] - 1)
    lower = np.floor(positions).astype(int)
    upper = np.minimum(lower + 1, anchors.shape[0] - 1)
    weight = (positions - lower)[..., None]
    rgb = anchors[lower] * (1.0 - weight) + anchors[upper] * weight
    rgba = np.empty((*matrix.shape, 4), dtype=np.uint8)
    rgba[..., :3] = np.rint(rgb).astype(np.uint8)
    rgba[..., 3] = np.where(finite, 255, 0).astype(np.uint8)
    if not invert_y:
        rgba = np.flipud(rgba)
    if invert_x:
        rgba = np.fliplr(rgba)
    output = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(output, format="PNG", compress_level=3)
    payload = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:image/png;base64,{payload}"


def automatic_y_limits(
    values: Sequence[Any],
    *,
    plot_type: str,
    histogram_bins: int,
    density: bool,
    log_y: bool,
    log_x: bool = False,
    histogram_bin_size: float | None = None,
    histogram_x_limits: tuple[float, float] | None = None,
    histogram_boundary: str = "left",
    y_axis: Any | None = None,
    fixed_limits: tuple[float, float] | None = None,
    floor_at_zero: bool = False,
) -> tuple[float, float]:
    """Calculate editable Y limits consistently for every preview page."""

    if fixed_limits is not None and plot_type == "Line" and not log_y:
        return fixed_limits
    if plot_type == "Heatmap":
        finite = finite_values(y_axis)
    else:
        finite_parts = [finite_values(value) for value in values]
        finite_parts = [part for part in finite_parts if part.size]
        finite = np.concatenate(finite_parts) if finite_parts else np.asarray([], dtype=float)
        if plot_type == "Histogram" and finite.size:
            _, counts, _ = _histogram_data(
                finite,
                bins=histogram_bins,
                bin_size=histogram_bin_size,
                density=bool(density),
                log_x=log_x,
                x_limits=histogram_x_limits,
                boundary=histogram_boundary,
            )
            finite = finite_values(counts)
    if log_y and plot_type != "Heatmap":
        finite = finite[finite > 0]
    if finite.size == 0:
        raise ValueError("The selected output has no finite values for automatic Y limits.")

    lower = float(np.min(finite))
    upper = float(np.max(finite))
    if plot_type == "Heatmap":
        if log_y:
            finite = finite[finite > 0]
            if finite.size == 0:
                raise ValueError("Log Y requires positive values.")
        # Images already occupy their complete coordinate extent. Line-style
        # margins expose empty strips beyond the last heatmap row/column.
        return _data_axis_bounds(finite, log_axis=log_y)
    if log_y:
        return _logarithmic_data_limits(lower, upper)
    return nice_linear_limits(
        lower,
        upper,
        floor_at_zero=plot_type == "Histogram" or floor_at_zero,
    )


def _bounded_indices(size: int, limit: int) -> np.ndarray:
    if size <= limit:
        return np.arange(size)
    return np.unique(np.linspace(0, size - 1, limit, dtype=int))


def _line_points(x: Any, y: Any, *, limit: int = 2_500) -> list[list[float | None]]:
    axis = np.asarray(x, dtype=float).reshape(-1)
    values = np.asarray(y, dtype=float).reshape(-1)
    if axis.size != values.size:
        raise ValueError("The selected output has no matching axis.")
    indices = _bounded_indices(values.size, limit)
    return [
        [
            float(axis[index]),
            float(values[index]) if np.isfinite(values[index]) else None,
        ]
        for index in indices
    ]


def _band_segments(
    band: PreviewBand,
    *,
    log_y: bool,
    limit: int = 2_500,
) -> list[list[float]]:
    """Return bounded adjacent polygons for one variable-width filled band."""

    axis = np.asarray(band.x, dtype=float).reshape(-1)
    lower = np.asarray(band.lower, dtype=float).reshape(-1)
    upper = np.asarray(band.upper, dtype=float).reshape(-1)
    if axis.size != lower.size or axis.size != upper.size:
        raise ValueError("The filled interval has no matching axis.")
    if axis.size < 2:
        return []
    indices = _bounded_indices(axis.size, limit)
    axis = axis[indices]
    lower = lower[indices]
    upper = upper[indices]
    segments: list[list[float]] = []
    for index in range(axis.size - 1):
        values = np.asarray(
            [
                axis[index],
                lower[index],
                upper[index],
                axis[index + 1],
                lower[index + 1],
                upper[index + 1],
            ],
            dtype=float,
        )
        if not np.isfinite(values).all():
            continue
        if log_y and np.any(values[[1, 2, 4, 5]] <= 0):
            continue
        segments.append(values.tolist())
    return segments


def _sample_heatmap(
    values: Any,
    x_axis: Any,
    y_axis: Any,
    *,
    limit: int = MAX_HEATMAP_CELLS,
    log_x: bool = False,
    log_y: bool = False,
    log_z: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Adaptively average a heatmap into a bounded raster using small tiles."""
    matrix = np.asarray(values)
    x_values = np.asarray(x_axis, dtype=float).reshape(-1)
    y_values = np.asarray(y_axis, dtype=float).reshape(-1)
    if matrix.ndim == 1 and y_values.size == 1 and matrix.size == x_values.size:
        matrix = matrix.reshape(1, -1)
    if matrix.shape == (y_values.size, x_values.size):
        pass
    elif matrix.shape == (x_values.size, y_values.size):
        matrix = matrix.T
    else:
        raise ValueError("The selected heatmap output has no matching axes.")

    if log_x:
        _validate_log_axis(x_values, "Log X")
        positive_x = x_values > 0
        positions = np.flatnonzero(positive_x)
        selection = (
            slice(positions[0], positions[-1] + 1)
            if np.all(np.diff(positions) == 1)
            else positive_x
        )
        x_values = x_values[selection]
        matrix = matrix[:, selection]
    if log_y:
        _validate_log_axis(y_values, "Log Y")
        positive_y = y_values > 0
        positions = np.flatnonzero(positive_y)
        selection = (
            slice(positions[0], positions[-1] + 1)
            if np.all(np.diff(positions) == 1)
            else positive_y
        )
        y_values = y_values[selection]
        matrix = matrix[selection, :]

    rows, columns = matrix.shape
    if not rows or not columns or limit < 1:
        raise ValueError("The selected heatmap has no displayable cells.")
    sampled_rows = min(rows, MAX_HEATMAP_ROWS)
    sampled_columns = min(columns, MAX_HEATMAP_COLUMNS)
    if sampled_rows * sampled_columns > limit:
        scale = math.sqrt(limit / (sampled_rows * sampled_columns))
        sampled_rows = max(1, min(int(sampled_rows * scale), limit))
        sampled_columns = max(1, min(int(sampled_columns * scale), limit // sampled_rows))

    y_edges = np.linspace(0, rows, sampled_rows + 1, dtype=int)
    x_edges = np.linspace(0, columns, sampled_columns + 1, dtype=int)
    sums = np.zeros((sampled_rows, sampled_columns), dtype=float)
    counts = np.zeros(sums.shape, dtype=np.int64)
    data_minimum, data_maximum = math.inf, -math.inf
    # Reduce bounded tiles, never allocating masks or float/log copies of the
    # full mapping matrix. Broadcast time-only fields remain cheap views too.
    for column_start in range(0, columns, MAX_HEATMAP_WORK_CELLS):
        column_stop = min(columns, column_start + MAX_HEATMAP_WORK_CELLS)
        width = column_stop - column_start
        local_starts = np.r_[
            0, x_edges[(x_edges > column_start) & (x_edges < column_stop)] - column_start,
        ]
        output_columns = np.searchsorted(x_edges, column_start + local_starts, side="right") - 1
        row_chunk = max(1, MAX_HEATMAP_WORK_CELLS // width)
        for output_row, (row_start, row_stop) in enumerate(zip(y_edges[:-1], y_edges[1:], strict=True)):
            for start in range(int(row_start), int(row_stop), row_chunk):
                block = np.asarray(
                    matrix[start : min(start + row_chunk, row_stop), column_start:column_stop],
                    dtype=float,
                )
                finite = np.isfinite(block)
                if log_z:
                    finite &= block > 0
                    transformed = np.zeros(block.shape, dtype=float)
                    np.log10(block, out=transformed, where=finite)
                else:
                    transformed = np.where(finite, block, 0.0)
                finite_values_block = transformed[finite]
                if finite_values_block.size:
                    data_minimum = min(data_minimum, float(np.min(finite_values_block)))
                    data_maximum = max(data_maximum, float(np.max(finite_values_block)))
                sums[output_row, output_columns] += np.add.reduceat(transformed.sum(axis=0), local_starts)
                counts[output_row, output_columns] += np.add.reduceat(finite.sum(axis=0), local_starts)
    if not np.isfinite(data_minimum):
        raise ValueError(
            "Log Z requires at least one positive heatmap value."
            if log_z else "The selected output contains no finite values."
        )
    if log_z:
        z_minimum, z_maximum = float(math.floor(data_minimum)), float(math.ceil(data_maximum))
        if z_minimum == z_maximum:
            z_maximum += 1.0
    else:
        z_minimum, z_maximum = _nice_axis_limits(
            data_minimum, data_maximum, floor_at_zero=data_minimum >= 0, intervals=5,
        )
    sampled = np.full(sums.shape, np.nan, dtype=float)
    np.divide(sums, counts, out=sampled, where=counts > 0)

    return x_values, y_values, sampled, z_minimum, z_maximum


def _heatmap_tooltip_data(
    matrix: np.ndarray,
    *,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
    log_z: bool,
    limit: int = MAX_HEATMAP_TOOLTIP_CELLS,
    x_coordinates: Any = None,
    y_coordinates: Any = None,
) -> list[list[float]]:
    """Build a bounded transparent hit grid over the heatmap raster.

    The rendered heatmap remains a compact image.  This much smaller custom
    layer provides representative X, Y, and value hover data without sending
    hundreds of thousands of DOM-facing ECharts items to the browser.
    """

    rows, columns = matrix.shape
    scale = min(1.0, math.sqrt(limit / max(1, rows * columns)))
    sampled_rows = max(1, min(rows, int(math.floor(rows * scale))))
    sampled_columns = max(1, min(columns, int(math.floor(columns * scale))))
    while sampled_rows * sampled_columns > limit:
        sampled_columns -= 1

    row_edges = np.linspace(0, rows, sampled_rows + 1, dtype=int)
    column_edges = np.linspace(0, columns, sampled_columns + 1, dtype=int)
    x_edges = np.linspace(x_limits[0], x_limits[1], sampled_columns + 1)
    y_edges = np.linspace(y_limits[0], y_limits[1], sampled_rows + 1)

    def representative_coordinates(coordinates: Any, edges: np.ndarray, size: int) -> np.ndarray | None:
        if coordinates is None:
            return None
        axis = np.asarray(coordinates, dtype=float).reshape(-1)
        centers = (edges[:-1] + edges[1:] - 1) / 2
        indices = np.rint((centers + .5) * axis.size / size - .5).astype(int)
        return axis[np.clip(indices, 0, axis.size - 1)]

    x_centers = representative_coordinates(x_coordinates, column_edges, columns)
    y_centers = representative_coordinates(y_coordinates, row_edges, rows)
    data: list[list[float]] = []
    for row in range(sampled_rows):
        for column in range(sampled_columns):
            block = matrix[
                row_edges[row] : row_edges[row + 1],
                column_edges[column] : column_edges[column + 1],
            ]
            finite = finite_values(block)
            if finite.size == 0:
                continue
            value = float(np.mean(finite))
            if log_z:
                value = 10.0**value
            x_lower = float(x_edges[column])
            x_upper = float(x_edges[column + 1])
            y_lower = float(y_edges[row])
            y_upper = float(y_edges[row + 1])
            data.append(
                [
                    float(x_centers[column]) if x_centers is not None else (x_lower + x_upper) / 2.0,
                    float(y_centers[row]) if y_centers is not None else (y_lower + y_upper) / 2.0,
                    value,
                    x_lower,
                    y_lower,
                    x_upper,
                    y_upper,
                ]
            )
    return data


def _plot_axis_limits(
    limits: tuple[float, float] | None,
    automatic: tuple[float, float] | None,
    *,
    label: str,
    log_axis: bool = False,
) -> tuple[float, float] | None:
    if limits is None:
        return automatic
    lower, upper = map(float, limits)
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        raise ValueError(f"{label} maximum must be greater than its finite minimum.")
    if log_axis:
        _validate_log_axis((lower, upper), f"Log {label}", bounds=True)
    return lower, upper


def build_preview_plot_options(
    *,
    plot_type: str,
    title: str,
    x_label: str,
    y_label: str,
    color: str,
    palette: str,
    histogram_bins: int,
    density: bool,
    show_legend: bool,
    show_colorbar: bool,
    log_y: bool,
    invert_y: bool,
    y_limits: tuple[float, float] | None,
    series_label: str,
    values: Any | None = None,
    x_axis: Any | None = None,
    y_axis: Any | None = None,
    lines: Sequence[PreviewLine] = (),
    bands: Sequence[PreviewBand] = (),
    line_width: float = 2.0,
    show_markers: bool = False,
    legend_position: str = "Top right",
    show_grid: bool = True,
    log_z: bool = False,
    interactive: bool = True,
    log_x: bool = False,
    invert_x: bool = False,
    invert_z: bool = False,
    show_histogram_statistics: bool = False,
    histogram_statistics_position: str = "Top right",
    histogram_bin_size: float | None = None,
    histogram_x_limits: tuple[float, float] | None = None,
    histogram_touching: bool = True,
    histogram_boundary: str = "left",
    x_limits: tuple[float, float] | None = None,
    z_limits: tuple[float, float] | None = None,
    aspect_ratio: float | None = None,
    line_type: str = "solid",
    secondary_y_label: str | None = None,
    secondary_y_limits: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Build bounded ECharts options for a line, histogram, boxplot, or heatmap."""

    title_options = {"show": False, "text": title}
    if aspect_ratio is not None:
        aspect_ratio = _positive_finite(aspect_ratio)
        if aspect_ratio is None:
            raise ValueError("Aspect ratio must be finite and positive.")
    axis_line = {"onZero": False, "lineStyle": {"color": "#a69e98"}}
    axis_tick = {
        "show": True,
        "inside": False,
        "length": 4,
        "customValues": None,
        "lineStyle": {"color": "#a69e98"},
    }
    split_line = {
        "show": bool(show_grid),
        "showMinLine": True,
        "showMaxLine": True,
        # Clear custom values when a live chart changes to a categorical axis.
        "customValues": None,
        "lineStyle": {"color": "#e8e3de", "type": "dashed", "width": 1},
    }
    if plot_type == "Heatmap":
        if values is None or x_axis is None or y_axis is None:
            raise ValueError("Heatmap preview requires values and both axes.")
        x_values, y_values, sampled, z_minimum, z_maximum = _sample_heatmap(
            values,
            x_axis,
            y_axis,
            log_x=log_x,
            log_y=log_y,
            log_z=log_z,
        )
        if z_limits is not None:
            z_minimum, z_maximum = _plot_axis_limits(z_limits, None, label="Z", log_axis=log_z)
            if log_z:
                z_minimum, z_maximum = math.log10(z_minimum), math.log10(z_maximum)
        image_uri = _heatmap_data_uri(
            sampled,
            palette=palette,
            z_minimum=z_minimum,
            z_maximum=z_maximum,
            invert_y=bool(invert_y),
            invert_x=bool(invert_x),
            invert_z=bool(invert_z),
        )
        x_data_minimum, x_data_maximum = _data_axis_bounds(x_values, log_axis=log_x)
        visual_end = _visual_axis_endpoint(x_axis) if not log_x and x_limits is None else None
        y_data_minimum, y_data_maximum = _data_axis_bounds(y_values, log_axis=log_y)
        x_minimum, x_maximum = _plot_axis_limits(
            x_limits, (x_data_minimum, visual_end or x_data_maximum), label="X", log_axis=log_x,
        )
        y_minimum, y_maximum = _plot_axis_limits(
            y_limits, (y_data_minimum, y_data_maximum), label="Y", log_axis=log_y,
        )
        if log_y:
            _validate_log_axis((y_minimum, y_maximum), "Log Y", bounds=True)
        heatmap_tick_lower = float(y_minimum)
        heatmap_tick_upper = float(y_maximum)
        heatmap_y_ticks = _aligned_axis_ticks(
            heatmap_tick_lower,
            heatmap_tick_upper,
            log_axis=log_y,
        )
        y_name_gap = _y_axis_name_gap(
            y_minimum,
            y_maximum,
            log_axis=log_y,
            tick_values=heatmap_y_ticks,
        )
        heatmap_y_axis: dict[str, Any] = {
            "z": 5,
            "type": "log" if log_y else "value",
            "min": y_minimum,
            "max": y_maximum,
            "splitNumber": 6,
            "name": y_label,
            "nameLocation": "middle",
            "nameGap": y_name_gap,
            "nameMoveOverlap": False,
            "nameTextStyle": _axis_name_options(),
            "inverse": bool(invert_y),
            "axisTick": axis_tick,
            "axisLabel": _axis_label_options(),
            "axisLine": axis_line,
            "splitLine": split_line,
        }
        if not log_y:
            heatmap_y_axis["interval"] = _nice_tick_step(y_maximum - y_minimum)
        _use_endpoint_safe_ticks(
            heatmap_y_axis,
            data_lower=y_data_minimum,
            data_upper=y_data_maximum,
            log_axis=log_y,
            tick_lower=heatmap_tick_lower,
            tick_upper=heatmap_tick_upper,
        )
        colorbar_label = (series_label or title).strip() or "Value"
        if log_z:
            colorbar_label = f"{colorbar_label} (log scale)"
        display_minimum = 10.0**z_minimum if log_z else z_minimum
        display_maximum = 10.0**z_maximum if log_z else z_maximum
        visual_map: dict[str, Any] = {
            "show": bool(show_colorbar),
            "min": z_minimum,
            "max": z_maximum,
            "calculable": False,
            "hoverLink": False,
            "realtime": False,
            "silent": True,
            "orient": "vertical",
            # Keep a compact scale immediately beside the raster. Endpoint
            # labels remain horizontal above and below the vertical bar.
            "right": 24,
            "top": "middle",
            "itemWidth": 12,
            "itemHeight": 156,
            "inRange": {
                "color": list(reversed(PALETTES.get(palette, PALETTES["viridis"])))
                if invert_z
                else PALETTES.get(palette, PALETTES["viridis"])
            },
            "textStyle": {
                "color": "#746d68",
                "fontSize": 10,
                "align": "center",
            },
            "text": [
                _compact_tick(display_maximum),
                _compact_tick(display_minimum),
            ],
            "textGap": 5,
            "padding": 0,
            # The raster is coloured in Python; do not recolour the transparent
            # hover grid which is appended as a second series below.
            "seriesIndex": 0,
        }
        if log_z:
            visual_map[":formatter"] = (
                "value => { const z = Math.pow(10, value); "
                "return z < 0.01 || z >= 1000 ? z.toExponential(1) "
                ": Number(z.toPrecision(3)).toString(); }"
            )
        image_literal = json.dumps(image_uri)
        render_item = (
            "(params, api) => { "
            "const a = api.coord([api.value(0), api.value(1)]); "
            "const b = api.coord([api.value(2), api.value(3)]); "
            "const x = Math.min(a[0], b[0]); const y = Math.min(a[1], b[1]); "
            "return {type: 'image', silent: true, style: {"
            f"image: {image_literal}, "
            "x: x, y: y, width: Math.abs(b[0] - a[0]), "
            "height: Math.abs(b[1] - a[1]), imageSmoothing: false}}; }"
        )
        tooltip_render_item = (
            "(params, api) => { "
            "const a = api.coord([api.value(3), api.value(4)]); "
            "const b = api.coord([api.value(5), api.value(6)]); "
            "return {type: 'rect', shape: {x: Math.min(a[0], b[0]), "
            "y: Math.min(a[1], b[1]), width: Math.abs(b[0] - a[0]), "
            "height: Math.abs(b[1] - a[1])}, "
            "style: {fill: 'rgba(0,0,0,0.001)', stroke: null}}; }"
        )
        tooltip_series = (
            [
                {
                    "name": series_label or title,
                    "type": "custom",
                    "coordinateSystem": "cartesian2d",
                    "dimensions": [
                        x_label or "X",
                        y_label or "Y",
                        series_label or "Value",
                        "X lower",
                        "Y lower",
                        "X upper",
                        "Y upper",
                    ],
                    "encode": {"x": 0, "y": 1, "tooltip": [0, 1, 2]},
                    ":renderItem": tooltip_render_item,
                    "silent": False,
                    "clip": True,
                    "emphasis": {"disabled": True},
                    "data": _heatmap_tooltip_data(
                        sampled,
                        x_limits=(x_data_minimum, x_data_maximum),
                        y_limits=(y_data_minimum, y_data_maximum),
                        log_z=log_z,
                        x_coordinates=x_values,
                        y_coordinates=y_values,
                    ),
                }
            ]
            if interactive
            else []
        )
        heatmap_x_axis: dict[str, Any] = {
            "z": 5,
            "type": "log" if log_x else "value",
            "min": x_minimum,
            "max": x_maximum,
            **({} if log_x else {"interval": _nice_tick_step(x_maximum - x_minimum)}),
            "splitNumber": 6,
            "position": "bottom",
            "inverse": bool(invert_x),
            "name": x_label,
            "nameLocation": "middle",
            "nameGap": _X_AXIS_NAME_GAP,
            "nameTextStyle": _axis_name_options(),
            "axisLine": {"onZero": False, **axis_line},
            "axisTick": axis_tick,
            "axisLabel": _axis_label_options(),
            "splitLine": split_line,
        }
        _use_endpoint_safe_ticks(
            heatmap_x_axis,
            data_lower=x_data_minimum,
            data_upper=x_data_maximum,
            log_axis=log_x,
            tick_lower=x_minimum,
            tick_upper=x_maximum,
        )
        tooltip = _tooltip_options(interactive=interactive, trigger="item", x_values=x_values, values=sampled)
        tooltip[":formatter"] = (
            "params => { const v = params.value; "
            f"const fx = ({_hover_formatter(integer_only=_whole_numbers(x_values))}); "
            f"const fy = ({_hover_formatter(integer_only=_whole_numbers(y_values))}); "
            f"const fz = ({_hover_formatter(integer_only=not log_z and _whole_numbers(sampled))}); "
            f"return {json.dumps(escape(x_label or 'X') + ': ')} + fx(v[0]) + '<br>' + "
            f"{json.dumps(escape(y_label or 'Y') + ': ')} + fy(v[1]) + '<br>' + "
            f"(params.marker || '') + {json.dumps(escape(series_label or title or 'Value') + ': ')} + fz(v[2]); }}"
        )
        return {
            "animation": False,
            "dnascapeAspectRatio": aspect_ratio,
            "title": title_options,
            "tooltip": tooltip,
            "graphic": (
                [
                    {
                        "type": "text",
                        "right": 1 + max(12, len(_compact_tick(display_maximum)) * 6, len(_compact_tick(display_minimum)) * 6) / 2,
                        "top": "middle",
                        "rotation": math.pi / 2,
                        "z": 10_000,
                        "z2": 10_000,
                        "silent": True,
                        "style": {
                            "text": colorbar_label,
                            "fill": "#625c57",
                            "font": "11px sans-serif",
                            "align": "center",
                            "verticalAlign": "middle",
                        },
                    }
                ]
                if show_colorbar
                else []
            ),
            "grid": _cartesian_grid(
                heatmap_colorbar=bool(show_colorbar),
                y_name_gap=y_name_gap,
                y_name=y_label,
            ),
            "xAxis": heatmap_x_axis,
            "yAxis": heatmap_y_axis,
            "visualMap": visual_map,
            "series": [
                {
                    "name": series_label or title,
                    "type": "custom",
                    "z": 0,
                    "coordinateSystem": "cartesian2d",
                    "data": [
                        [
                            x_data_minimum,
                            y_data_minimum,
                            x_data_maximum,
                            y_data_maximum,
                        ]
                    ],
                    "encode": {"x": [0, 2], "y": [1, 3]},
                    ":renderItem": render_item,
                    "silent": True,
                    "clip": True,
                    "emphasis": {"disabled": True},
                },
                *tooltip_series,
            ],
        }

    if plot_type == "Histogram":
        finite = finite_values(values)
        if finite.size == 0:
            raise ValueError("The selected output contains no finite values.")
        if log_x:
            _validate_log_axis(finite, "Log X")
        finite, counts, edges = _histogram_data(
            finite,
            bins=histogram_bins,
            bin_size=histogram_bin_size,
            density=bool(density),
            log_x=log_x,
            x_limits=histogram_x_limits,
            boundary=histogram_boundary,
        )
        histogram_axis_limits = (
            (float(edges[0]), float(edges[-1]))
            if histogram_x_limits is not None
            else (
                _logarithmic_data_limits(float(edges[0]), float(edges[-1]))
                if log_x
                else nice_linear_limits(float(edges[0]), float(edges[-1]))
            )
        )
        histogram_y_axis: dict[str, Any] = {
            "type": "log" if log_y else "value",
            "splitNumber": 6,
            "name": y_label or ("Density" if density else "Count"),
            "nameLocation": "middle",
            "nameTextStyle": _axis_name_options(),
            "inverse": bool(invert_y),
            "axisTick": axis_tick,
            "axisLine": axis_line,
            "splitLine": split_line,
        }
        if y_limits is not None:
            histogram_y_axis["min"], histogram_y_axis["max"] = y_limits
        elif not log_y:
            histogram_y_axis["min"], histogram_y_axis["max"] = nice_linear_limits(
                0.0,
                float(np.max(counts)),
                floor_at_zero=True,
            )
        elif counts.size:
            positive_counts = counts[counts > 0]
            _validate_log_axis(positive_counts, "Log Y")
            histogram_y_axis["min"], histogram_y_axis["max"] = _logarithmic_data_limits(
                float(np.min(positive_counts)),
                float(np.max(positive_counts)),
            )
        if not log_y and "min" in histogram_y_axis and "max" in histogram_y_axis:
            histogram_y_axis["interval"] = _nice_tick_step(
                float(histogram_y_axis["max"]) - float(histogram_y_axis["min"])
            )
        if log_y:
            positive_counts = counts[counts > 0]
            _validate_log_axis(positive_counts, "Log Y")
            if y_limits is not None:
                _validate_log_axis(y_limits, "Log Y", bounds=True)
            bar_baseline = (
                float(y_limits[0]) if y_limits is not None else float(np.min(positive_counts)) * 0.8
            )
        else:
            bar_baseline = 0.0
        y_lower = float(histogram_y_axis["min"])
        y_upper = float(histogram_y_axis["max"])
        automatic_y_margin = y_limits is None or (
            not log_y
            and not _fixed_scientific_limits((y_lower, y_upper))
            and _automatic_linear_limits(
                (y_lower, y_upper),
                0.0,
                float(np.max(counts)),
            )
        )
        histogram_y_axis["axisLabel"] = _axis_label_options(
            hide_padded_minimum=automatic_y_margin and y_lower < 0.0,
            hide_padded_maximum=automatic_y_margin and y_upper > float(np.max(counts)),
        )
        if not log_y:
            histogram_y_axis["interval"] = _nice_tick_step(
                float(np.max(counts)) if automatic_y_margin else y_upper - y_lower
            )
        histogram_y_data_lower = float(np.min(counts[counts > 0])) if log_y else 0.0
        histogram_y_data_upper = float(np.max(counts))
        histogram_tick_lower = histogram_y_data_lower if automatic_y_margin else y_lower
        histogram_tick_upper = histogram_y_data_upper if automatic_y_margin else y_upper
        y_name_gap = _y_axis_name_gap(
            y_lower,
            y_upper,
            log_axis=log_y,
            tick_values=_aligned_axis_ticks(
                histogram_tick_lower,
                histogram_tick_upper,
                log_axis=log_y,
            ),
        )
        histogram_y_axis["nameGap"] = y_name_gap
        histogram_y_axis["nameMoveOverlap"] = False
        _use_endpoint_safe_ticks(
            histogram_y_axis,
            data_lower=histogram_y_data_lower,
            data_upper=histogram_y_data_upper,
            log_axis=log_y,
            tick_lower=histogram_tick_lower,
            tick_upper=histogram_tick_upper,
        )
        histogram_x_axis: dict[str, Any] = {
            "type": "log" if log_x else "value",
            "min": histogram_axis_limits[0],
            "max": histogram_axis_limits[1],
            "splitNumber": 6,
            "position": "bottom",
            "inverse": bool(invert_x),
            "name": x_label,
            "nameLocation": "middle",
            "nameGap": _X_AXIS_NAME_GAP,
            "nameTextStyle": _axis_name_options(),
            "axisLine": {"onZero": False, **axis_line},
            "axisTick": axis_tick,
            "axisLabel": _axis_label_options(
                hide_padded_minimum=(
                    histogram_x_limits is None and histogram_axis_limits[0] < float(edges[0])
                ),
                hide_padded_maximum=(
                    histogram_x_limits is None and histogram_axis_limits[1] > float(edges[-1])
                ),
            ),
            "splitLine": split_line,
        }
        if not log_x:
            histogram_x_axis["interval"] = _nice_tick_step(float(edges[-1]) - float(edges[0]))
        _use_endpoint_safe_ticks(
            histogram_x_axis,
            data_lower=float(edges[0]),
            data_upper=float(edges[-1]),
            log_axis=log_x,
        )
        gap_fraction = 0.0 if histogram_touching else 0.14
        histogram_render_item = (
            "(params, api) => { "
            "const topLeft = api.coord([api.value(0), api.value(2)]); "
            "const baseLeft = api.coord([api.value(0), api.value(3)]); "
            "const baseRight = api.coord([api.value(1), api.value(3)]); "
            "const x0 = Math.min(baseLeft[0], baseRight[0]); "
            "const x1 = Math.max(baseLeft[0], baseRight[0]); "
            "const y0 = Math.min(topLeft[1], baseLeft[1]); "
            "const y1 = Math.max(topLeft[1], baseLeft[1]); "
            f"const gap = Math.min(4, (x1 - x0) * {gap_fraction:.8g}); "
            "const left = Math.max(params.coordSys.x, x0 + gap / 2); "
            "const right = Math.min(params.coordSys.x + params.coordSys.width, x1 - gap / 2); "
            "const top = Math.max(params.coordSys.y, y0); "
            "const bottom = Math.min(params.coordSys.y + params.coordSys.height, y1); "
            "if (!Number.isFinite(left + right + top + bottom) || right <= left || bottom <= top) "
            "return null; "
            "return {type: 'rect', shape: {x: left, y: top, width: right - left, "
            "height: bottom - top}, style: api.style()}; }"
        )
        histogram_data = [
            [float(left), float(right), float(value), bar_baseline]
            for left, right, value in zip(edges[:-1], edges[1:], counts)
            if not log_y or value > 0
        ]
        bottom_statistics = bool(show_histogram_statistics) and str(
            histogram_statistics_position or ""
        ).strip().casefold().startswith("bottom")
        return {
            "animation": False,
            "dnascapeAspectRatio": aspect_ratio,
            "title": title_options,
            "graphic": (
                _histogram_statistics_graphic(
                    finite,
                    position=histogram_statistics_position,
                )
                if show_histogram_statistics
                else []
            ),
            "tooltip": _tooltip_options(interactive=interactive, trigger="axis", x_values=edges, values=counts),
            "grid": _cartesian_grid(
                y_name_gap=y_name_gap,
                y_name=histogram_y_axis["name"],
                bottom_legend=bottom_statistics,
            ),
            "xAxis": histogram_x_axis,
            "yAxis": histogram_y_axis,
            "series": [
                {
                    "name": series_label or title,
                    "type": "custom",
                    "coordinateSystem": "cartesian2d",
                    "dimensions": ["Bin start", "Bin end", y_label or "Count", "Baseline"],
                    "encode": {"x": [0, 1], "y": 2, "tooltip": [0, 1, 2]},
                    "silent": not interactive,
                    "itemStyle": {"color": color},
                    "emphasis": {"disabled": not interactive},
                    ":renderItem": histogram_render_item,
                    "clip": True,
                    "data": histogram_data,
                }
            ],
        }

    if plot_type == "Boxplot":
        finite = finite_values(values)
        if log_y:
            _validate_log_axis(finite, "Log Y")
            finite = finite[finite > 0]
        if finite.size < 2:
            raise ValueError("Boxplot view requires at least two finite values.")
        q1, median, q3 = np.quantile(finite, (0.25, 0.5, 0.75))
        iqr = float(q3 - q1)
        lower_values = finite[finite >= q1 - 1.5 * iqr]
        upper_values = finite[finite <= q3 + 1.5 * iqr]
        lower = float(np.min(lower_values)) if lower_values.size else float(np.min(finite))
        upper = float(np.max(upper_values)) if upper_values.size else float(np.max(finite))
        box_y_axis: dict[str, Any] = {
            "type": "log" if log_y else "value",
            "splitNumber": 6,
            "name": y_label,
            "nameLocation": "middle",
            "nameTextStyle": _axis_name_options(),
            "inverse": bool(invert_y),
            "axisTick": axis_tick,
            "axisLine": axis_line,
            "splitLine": split_line,
        }
        if y_limits is not None:
            box_y_axis["min"], box_y_axis["max"] = y_limits
        elif not log_y:
            box_y_axis["min"], box_y_axis["max"] = nice_linear_limits(
                float(np.min(finite)),
                float(np.max(finite)),
            )
        else:
            box_y_axis["min"], box_y_axis["max"] = _logarithmic_data_limits(
                float(np.min(finite)),
                float(np.max(finite)),
            )
        if not log_y and "min" in box_y_axis and "max" in box_y_axis:
            box_y_axis["interval"] = _nice_tick_step(
                float(box_y_axis["max"]) - float(box_y_axis["min"])
            )
        if log_y and y_limits is not None:
            _validate_log_axis(y_limits, "Log Y", bounds=True)
        y_lower = float(box_y_axis["min"])
        y_upper = float(box_y_axis["max"])
        automatic_box_limits = y_limits is None or (
            not log_y
            and not _fixed_scientific_limits((y_lower, y_upper))
            and _automatic_linear_limits(
                (y_lower, y_upper),
                float(np.min(finite)),
                float(np.max(finite)),
            )
        )
        box_y_axis["axisLabel"] = _axis_label_options(
            hide_padded_minimum=automatic_box_limits and y_lower < float(np.min(finite)),
            hide_padded_maximum=automatic_box_limits and y_upper > float(np.max(finite)),
        )
        if not log_y:
            box_y_axis["interval"] = _nice_tick_step(
                float(np.max(finite) - np.min(finite))
                if automatic_box_limits
                else y_upper - y_lower
            )
        box_y_data_lower = float(np.min(finite))
        box_y_data_upper = float(np.max(finite))
        box_tick_lower = box_y_data_lower if automatic_box_limits else y_lower
        box_tick_upper = box_y_data_upper if automatic_box_limits else y_upper
        y_name_gap = _y_axis_name_gap(
            y_lower,
            y_upper,
            log_axis=log_y,
            tick_values=_aligned_axis_ticks(
                box_tick_lower,
                box_tick_upper,
                log_axis=log_y,
            ),
        )
        box_y_axis["nameGap"] = y_name_gap
        box_y_axis["nameMoveOverlap"] = False
        _use_endpoint_safe_ticks(
            box_y_axis,
            data_lower=box_y_data_lower,
            data_upper=box_y_data_upper,
            log_axis=log_y,
            tick_lower=box_tick_lower,
            tick_upper=box_tick_upper,
        )
        box_x_limits = _plot_axis_limits(x_limits, None, label="X")
        return {
            "animation": False,
            "dnascapeAspectRatio": aspect_ratio,
            "title": title_options,
            "tooltip": _tooltip_options(interactive=interactive, trigger="item", values=[lower, q1, median, q3, upper]),
            "grid": _cartesian_grid(
                y_name_gap=y_name_gap,
                y_name=y_label,
            ),
            "xAxis": {
                "type": "category",
                "data": [series_label or title],
                **({"min": box_x_limits[0], "max": box_x_limits[1]} if box_x_limits else {}),
                "inverse": bool(invert_x),
                "name": x_label,
                "nameLocation": "middle",
                "nameGap": _X_AXIS_NAME_GAP,
                "nameTextStyle": _axis_name_options(),
                "axisLabel": _axis_label_options(),
                "axisLine": axis_line,
                "axisTick": axis_tick,
                "splitLine": split_line,
            },
            "yAxis": box_y_axis,
            "series": [
                {
                    "name": series_label or title,
                    "type": "boxplot",
                    "silent": not interactive,
                    "data": [[lower, float(q1), float(median), float(q3), upper]],
                    "emphasis": {"disabled": not interactive},
                    "itemStyle": {
                        "color": color,
                        "borderColor": color,
                        "opacity": 0.78,
                    },
                }
            ],
        }

    if plot_type != "Line":
        raise ValueError(f"Unknown preview plot type {plot_type!r}.")
    dual_y_axes = secondary_y_label is not None
    if dual_y_axes and (len(lines) != 2 or bands):
        raise ValueError("Two Y axes require exactly two curves without filled bands.")
    if not lines:
        if values is None or x_axis is None:
            raise ValueError("Line preview requires values and a matching axis.")
        lines = (
            PreviewLine(
                name=series_label or title,
                x=np.asarray(x_axis),
                y=np.asarray(values),
                color=color,
            ),
        )
    line_options: list[dict[str, Any]] = []
    all_x_values: list[np.ndarray] = []
    all_y_values: list[np.ndarray] = []
    band_render_item = (
        "(params, api) => { const points = ["
        "api.coord([api.value(0), api.value(1)]), "
        "api.coord([api.value(3), api.value(4)]), "
        "api.coord([api.value(3), api.value(5)]), "
        "api.coord([api.value(0), api.value(2)])]; "
        "return {type: 'polygon', shape: {points: points}, style: api.style()}; }"
    )
    for band in bands:
        all_x_values.append(finite_values(band.x))
        all_y_values.extend((finite_values(band.lower), finite_values(band.upper)))
        segments = _band_segments(band, log_y=log_y)
        if not segments:
            continue
        line_options.append(
            {
                "type": "custom",
                "name": band.name,
                "coordinateSystem": "cartesian2d",
                "silent": True,
                "clip": True,
                "z": 1,
                "itemStyle": {"color": band.color, "opacity": band.opacity},
                "emphasis": {"disabled": True},
                ":renderItem": band_render_item,
                "data": segments,
            }
        )
    for index, line in enumerate(lines):
        flattened = np.asarray(line.y, dtype=float).reshape(-1)
        line_x = np.asarray(line.x, dtype=float).reshape(-1)
        scalar = flattened.size == 1
        all_x_values.append(finite_values(line_x))
        if not dual_y_axes or index == 0:
            all_y_values.append(finite_values(flattened))
        line_options.append(
            {
                "type": "scatter" if scalar else "line",
                "name": line.name,
                # ECharts substitutes a circle in line legends for "none".
                # Hidden line markers retain a plain line legend instead.
                "symbol": "circle" if scalar else "line",
                "showSymbol": bool(scalar),
                "symbolSize": 12 if scalar else 0,
                "silent": not interactive,
                "connectNulls": False,
                "sampling": "lttb",
                "z": line.z,
                "lineStyle": {
                    "width": line.width if line.width is not None else line_width,
                    "color": line.color,
                    "opacity": line.opacity,
                    "type": curve_dash_pattern(line.line_type or line_type),
                },
                **({"yAxisIndex": index} if dual_y_axes else {}),
                "itemStyle": {"color": line.color, "opacity": line.opacity},
                "emphasis": {"disabled": not interactive},
                "data": _line_points(line.x, flattened),
            }
        )
    finite_x_parts = [part for part in all_x_values if part.size]
    finite_x = np.concatenate(finite_x_parts) if finite_x_parts else np.asarray([], dtype=float)
    if finite_x.size == 0:
        raise ValueError("The selected output has no finite horizontal coordinates.")
    if log_x:
        _validate_log_axis(finite_x, "Log X")
    x_data_lower = float(np.min(finite_x[finite_x > 0])) if log_x else float(np.min(finite_x))
    x_data_upper = float(np.max(finite_x))
    automatic_x_limits = (
        _logarithmic_data_limits(x_data_lower, x_data_upper)
        if log_x
        else nice_linear_limits(x_data_lower, x_data_upper)
    )
    custom_x_limits = x_limits is not None
    visual_end = _visual_axis_endpoint(finite_x) if not log_x and not custom_x_limits else None
    x_limits = _plot_axis_limits(x_limits, automatic_x_limits, label="X", log_axis=log_x)
    finite_y_parts = [part for part in all_y_values if part.size]
    finite_y = np.concatenate(finite_y_parts) if finite_y_parts else np.asarray([], dtype=float)
    if finite_y.size == 0:
        raise ValueError("The selected output has no finite vertical values.")
    y_axis_options: dict[str, Any] = {
        "type": "log" if log_y else "value",
        "splitNumber": 6,
        "name": y_label,
        "nameLocation": "middle",
        "nameTextStyle": _axis_name_options(),
        "inverse": bool(invert_y),
        "axisTick": axis_tick,
        "axisLine": axis_line,
        "splitLine": split_line,
    }
    if y_limits is not None:
        y_axis_options["min"], y_axis_options["max"] = y_limits
    elif not log_y:
        if finite_y.size:
            y_axis_options["min"], y_axis_options["max"] = nice_linear_limits(
                float(np.min(finite_y)),
                float(np.max(finite_y)),
            )
    elif finite_y.size:
        positive_y = finite_y[finite_y > 0]
        _validate_log_axis(positive_y, "Log Y")
        y_axis_options["min"], y_axis_options["max"] = _logarithmic_data_limits(
            float(np.min(positive_y)),
            float(np.max(positive_y)),
        )
    if not log_y and "min" in y_axis_options and "max" in y_axis_options:
        y_axis_options["interval"] = _nice_tick_step(
            float(y_axis_options["max"]) - float(y_axis_options["min"])
        )
    if log_y:
        _validate_log_axis(finite_y, "Log Y")
        if y_limits is not None:
            _validate_log_axis(y_limits, "Log Y", bounds=True)
    y_lower = float(y_axis_options["min"])
    y_upper = float(y_axis_options["max"])
    automatic_line_limits = y_limits is None or (
        not log_y
        and not _fixed_scientific_limits((y_lower, y_upper))
        and finite_y.size > 0
        and _automatic_linear_limits(
            (y_lower, y_upper),
            float(np.min(finite_y)),
            float(np.max(finite_y)),
        )
    )
    y_axis_options["axisLabel"] = _axis_label_options(
        hide_padded_minimum=automatic_line_limits and y_lower < float(np.min(finite_y)),
        hide_padded_maximum=automatic_line_limits and y_upper > float(np.max(finite_y)),
    )
    if not log_y:
        y_axis_options["interval"] = _nice_tick_step(
            float(np.max(finite_y) - np.min(finite_y))
            if automatic_line_limits
            else y_upper - y_lower
        )
    line_y_tick_values = finite_y[finite_y > 0] if log_y else finite_y
    line_y_data_lower = float(np.min(line_y_tick_values))
    line_y_data_upper = float(np.max(line_y_tick_values))
    line_tick_lower = line_y_data_lower if automatic_line_limits else y_lower
    line_tick_upper = line_y_data_upper if automatic_line_limits else y_upper
    y_name_gap = _y_axis_name_gap(
        y_lower,
        y_upper,
        log_axis=log_y,
        tick_values=_aligned_axis_ticks(
            line_tick_lower,
            line_tick_upper,
            log_axis=log_y,
        ),
    )
    y_axis_options["nameGap"] = y_name_gap
    # ECharts 6 can move middle names farther from their ticks even with
    # explicit grid margins. Our gap already includes the formatted ticks;
    # moving it again clips the title at the left edge of the canvas.
    y_axis_options["nameMoveOverlap"] = False
    _use_endpoint_safe_ticks(
        y_axis_options,
        data_lower=line_y_data_lower,
        data_upper=line_y_data_upper,
        log_axis=log_y,
        tick_lower=line_tick_lower,
        tick_upper=line_tick_upper,
    )
    legend_names: list[str] = []
    for item in (
        *(line.name for line in lines if line.show_in_legend),
        *(band.name for band in bands if band.show_in_legend),
    ):
        if item not in legend_names:
            legend_names.append(item)
    grid = _cartesian_grid(y_name_gap=y_name_gap, y_name=y_label)
    rendered_y_axes: dict[str, Any] | list[dict[str, Any]] = y_axis_options
    if dual_y_axes:
        second = build_preview_plot_options(
            plot_type="Line", title=title, x_label=x_label, y_label=secondary_y_label,
            color=lines[1].color, palette=palette, histogram_bins=histogram_bins,
            density=False, show_legend=False, show_colorbar=False, log_y=log_y,
            invert_y=invert_y, y_limits=secondary_y_limits, lines=(lines[1],),
            series_label=lines[1].name,
            line_width=line_width, line_type=line_type, show_grid=False,
            interactive=interactive, log_x=log_x,
        )
        right_axis = second["yAxis"]
        right_axis["position"] = "right"
        y_axis_options["position"] = "left"
        for axis, line in zip((y_axis_options, right_axis), lines, strict=True):
            axis["axisLine"] = {"onZero": False, "lineStyle": {"color": line.color}}
            axis["nameTextStyle"] = {**axis["nameTextStyle"], "color": line.color}
        grid["right"] = second["grid"]["left"]
        rendered_y_axes = [y_axis_options, right_axis]
    legend_options = build_legend_options(
        legend_names,
        grid=grid,
        position=legend_position,
        show=show_legend,
        interactive=interactive,
    )
    x_axis_options: dict[str, Any] = {
        "type": "log" if log_x else "value",
        "min": x_limits[0],
        "max": x_limits[1],
        **({} if log_x else {"interval": _nice_tick_step(
            x_limits[1] - x_limits[0] if custom_x_limits else x_data_upper - x_data_lower,
        )}),
        "splitNumber": 6,
        "position": "bottom",
        "inverse": bool(invert_x),
        "name": x_label,
        "nameLocation": "middle",
        "nameGap": _X_AXIS_NAME_GAP,
        "nameTextStyle": _axis_name_options(),
        "axisLine": {"onZero": False, **axis_line},
        "axisTick": axis_tick,
        "axisLabel": _axis_label_options(
            hide_padded_minimum=not custom_x_limits and x_limits[0] < x_data_lower,
            hide_padded_maximum=not custom_x_limits and x_limits[1] > x_data_upper,
        ),
        "splitLine": split_line,
    }
    _use_endpoint_safe_ticks(
        x_axis_options,
        data_lower=x_data_lower,
        data_upper=x_data_upper,
        tick_lower=x_limits[0] if custom_x_limits else None,
        tick_upper=x_limits[1] if custom_x_limits else visual_end,
        log_axis=log_x,
    )
    return limit_curve_hover({
        "animation": False,
        "dnascapeAspectRatio": aspect_ratio,
        "title": title_options,
        "tooltip": {
            **_tooltip_options(
                interactive=interactive,
                trigger=("item" if all(np.asarray(line.y).size == 1 for line in lines) else "axis"),
                x_values=finite_x,
                values=np.concatenate([finite_values(line.y) for line in lines]) if dual_y_axes else finite_y,
            ),
        },
        "legend": legend_options,
        "grid": grid,
        "xAxis": x_axis_options,
        "yAxis": rendered_y_axes,
        "series": line_options,
    })
