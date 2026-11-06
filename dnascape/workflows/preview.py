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
from functools import wraps
from typing import Any

import numpy as np

from dnascape.workflows.export_settings import preview_point_limit

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
CURVE_TYPES = {"solid": "Solid", "dashed": "Dashed", "dotted": "Dotted", "dashdot": "Dash-dot"}


def curve_dash_pattern(line_type: str) -> str | list[int]:
    """Translate a named curve pattern to the canvas and SVG renderers."""
    if line_type not in CURVE_TYPES:
        raise ValueError(f"Unknown curve type {line_type!r}.")
    return [8, 4, 2, 4] if line_type == "dashdot" else line_type


@dataclass(frozen=True)
class PreviewLine:
    """One named profile and its curve or distribution presentation."""

    name: str
    x: np.ndarray
    y: np.ndarray
    color: str
    opacity: float = 1.0
    width: float | None = None
    show_in_legend: bool = True
    z: int = 3
    line_type: str | None = None
    box_position: float | None = None
    box_label: str | None = None


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
            upper *= factor
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
    """Start at the smallest positive sample and pad the upper log bound."""

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
    return lower, 10.0 ** (log_upper + padding)


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

    Some axes include a small, scale-relative margin. Letting ECharts derive
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


def style_curve_legend(legend: dict[str, Any], series: Sequence[dict[str, Any]]) -> None:
    """Keep curve legend dashes at one width and hide their marker symbols."""
    line_names = {item.get("name") for item in series if item.get("type") == "line"}
    entries = []
    for item in legend.get("data", ()):
        name = item.get("name") if isinstance(item, dict) else item
        if name in line_names:
            item = dict(item) if isinstance(item, dict) else {"name": name}
            item["itemStyle"] = {**item.get("itemStyle", {}), "opacity": 0}
            item["lineStyle"] = {**item.get("lineStyle", {}), "width": 1.8, "opacity": 1}
        entries.append(item)
    legend["data"] = entries


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
        vertical_edge: int(grid["bottom"] if vertical_edge == "bottom" else grid["top"]) + 8,
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
    if log_axis:
        finite = finite[finite > 0]
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
            return lower, upper * factor
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


def _tooltip_options(
    *, interactive: bool, trigger: str, x_values: Any = None, values: Any = None,
) -> dict[str, Any]:
    """Return consistently rounded hover values for every interactive plot."""

    return {
        "show": bool(interactive),
        "trigger": trigger,
        "confine": True,
        "borderColor": "#ddd7d1",
        "triggerOn": "mousemove|click" if interactive else "none",
        ":valueFormatter": _hover_formatter(integer_only=_whole_numbers(values)),
        ":formatter": (
            "params => { const rows=Array.isArray(params)?params:[params]; if(!rows.length)return ''; "
            f"const fy=({_hover_formatter(integer_only=_whole_numbers(values))}); "
            "return rows.map(p=>{const v=p.value; "
            "if(Array.isArray(v)&&(p.seriesType==='boxplot'||(p.seriesType==='custom'&&v.length===6))){const values=v.slice(-5); "
            "return (p.marker||'')+['Min','Q1','Median','Q3','Max'].map((name,i)=>name+': <b>'+fy(values[i])+'</b>').join('<br>');} "
            "const y=Array.isArray(v)?(p.seriesType==='custom'?v[2]:v[1]):v; "
            "return (p.marker||'')+'<b>'+fy(y)+'</b>';}).join('<br>'); }"
        ),
        "axisPointer": {
            "show": "auto" if interactive else False,
            "type": "line",
            "axis": "x",
            "snap": True,
            "lineStyle": {"color": "#b8b0aa", "width": 1, "type": "dashed"},
            "label": {
                "show": bool(interactive),
                "backgroundColor": "#b03412",
                "color": "#ffffff",
                "fontSize": 11,
                "lineHeight": 14,
                "padding": [3, 6],
                "borderRadius": 3,
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


def _palette_rgb(palette: str) -> np.ndarray:
    colours = PALETTES.get(palette, PALETTES["viridis"])
    return np.asarray(
        [tuple(int(colour[index : index + 2], 16) for index in (1, 3, 5)) for colour in colours],
        dtype=float,
    )


def _heatmap_value_style(
    *, palette: str, z_minimum: float, z_maximum: float, log_z: bool, invert_z: bool,
) -> str:
    """Colour a hover readout with the same clipped RGB mapping as the raster."""

    anchors = _palette_rgb(palette)
    if invert_z:
        anchors = anchors[::-1]
    return (
        "value => { const z = Number(value); "
        "if (!Number.isFinite(z)" + (" || z <= 0" if log_z else "") + ") "
        "return {background: '#746d68', foreground: '#ffffff'}; "
        f"const anchors = {json.dumps(anchors.astype(int).tolist())}; "
        f"const scaled = {'Math.log10(z)' if log_z else 'z'}; "
        f"const position = Math.max(0, Math.min(1, (scaled - {z_minimum!r}) / "
        f"({z_maximum!r} - {z_minimum!r}))) * (anchors.length - 1); "
        "const lower = Math.floor(position), upper = Math.min(lower + 1, anchors.length - 1); "
        "const weight = position - lower; "
        # Match NumPy's round-to-even raster conversion at exact half values.
        "const round = value => {const lower = Math.floor(value); "
        "return value - lower === 0.5 ? lower + lower % 2 : Math.round(value);}; "
        "const rgb = anchors[lower].map((channel, i) => "
        "round(channel * (1 - weight) + anchors[upper][i] * weight)); "
        "const linear = rgb.map(channel => {const c = channel / 255; "
        "return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);}); "
        "const luminance = 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]; "
        "return {background: 'rgb(' + rgb.join(',') + ')', "
        "foreground: luminance > 0.179 ? '#000000' : '#ffffff'}; }"
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


def _kde_data(
    values: Any, *, log_x: bool = False, x_limits: tuple[float, float] | None = None,
    bandwidth_scale: float = 1.0,
    evaluation_limits: tuple[float, float] | None = None,
    point_limit: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Bounded Gaussian KDE with Scott bandwidth and a constant-data fallback."""

    from scipy.stats import gaussian_kde

    bandwidth_scale = _positive_finite(bandwidth_scale)
    if bandwidth_scale is None:
        raise ValueError("KDE smoothness must be finite and greater than zero.")

    samples = finite_values(values)
    if log_x:
        samples = samples[samples > 0]
    if not samples.size:
        raise ValueError("KDE requires finite values, including positive values for Log X.")
    if samples.size > 20_000:
        samples = samples[np.linspace(0, samples.size - 1, 20_000, dtype=int)]
    spread = float(np.std(samples, ddof=1)) if samples.size > 1 else 0.0
    constant = samples.size < 2 or spread == 0.0
    bandwidth = (
        max(abs(float(samples[0])) * 0.05, (x_limits[1] - x_limits[0]) * 0.05 if x_limits else 0.05)
        if constant else spread * samples.size ** (-0.2)
    ) * bandwidth_scale
    lower = float(np.min(samples)) - 3 * bandwidth
    upper = float(np.max(samples)) + 3 * bandwidth
    if log_x:
        lower = max(lower, float(np.min(samples)) * 0.5)
    domain = evaluation_limits if evaluation_limits is not None else x_limits
    if domain is not None:
        lower, upper = _plot_axis_limits(domain, None, label="X", log_axis=log_x)
    count = preview_point_limit() if point_limit is None else point_limit
    grid = np.geomspace(lower, upper, count) if log_x else np.linspace(lower, upper, count)
    if constant:
        curve = np.exp(-0.5 * ((grid - samples[0]) / bandwidth) ** 2) / (
            bandwidth * np.sqrt(2 * np.pi)
        )
    else:
        curve = gaussian_kde(samples, bw_method=lambda estimator: estimator.scotts_factor() * bandwidth_scale)(grid)
    return grid, curve


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
    kde_bandwidth: float = 1.0,
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
            heights = []
            for part in finite_parts:
                if density:
                    _, counts = _kde_data(part, log_x=log_x, x_limits=histogram_x_limits, bandwidth_scale=kde_bandwidth)
                else:
                    _, counts, _ = _histogram_data(
                        part, bins=histogram_bins, bin_size=histogram_bin_size,
                        density=False, log_x=log_x, x_limits=histogram_x_limits,
                        boundary=histogram_boundary,
                    )
                heights.append(counts)
            finite = finite_values(np.concatenate(heights))
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


class DetailedPreviewData(list):
    """JSON-safe display data with a server-only, bounded refinement callback."""

    def __init__(self, values: Any, refine: Any, *, coverage: Any = None) -> None:
        super().__init__(values)
        self.refine = refine
        self.coverage = coverage


def buffered_preview_limits(
    limits: tuple[float, float] | None, bounds: tuple[float, float], *, log: bool = False,
) -> tuple[float, float] | None:
    """Prefetch one visible width on either side, within the native domain."""
    if limits is None:
        return None
    lower, upper = limits
    if log and lower > 0:
        span = math.log(upper) - math.log(lower)
        return (math.exp(max(math.log(bounds[0]) if bounds[0] > 0 else -math.inf, math.log(lower) - span)),
                math.exp(min(math.log(bounds[1]), math.log(upper) + span)))
    span = upper - lower
    return max(bounds[0], lower - span), min(bounds[1], upper + span)


def _visible_slice(axis: np.ndarray, limits: tuple[float, float] | None) -> slice:
    if limits is None:
        return slice(0, axis.size)
    ascending = axis[-1] >= axis[0]
    ordered = all(np.all((axis[start + 1:stop + 1] >= axis[start:stop]) if ascending else
                         (axis[start + 1:stop + 1] <= axis[start:stop]))
                  for start in range(0, max(0, axis.size - 1), 65536)
                  for stop in (min(axis.size - 1, start + 65536),))
    if ordered:
        sorted_axis = axis if ascending else axis[::-1]
        start, stop = np.searchsorted(sorted_axis, limits[0], side="left"), np.searchsorted(sorted_axis, limits[1], side="right")
        if not ascending:
            start, stop = axis.size - stop, axis.size - start
        return slice(max(0, int(start) - 1), min(axis.size, max(int(stop), int(start) + 1) + 1))
    first, last = axis.size, -1
    for start in range(0, axis.size, 65536):
        block = axis[start:start + 65536]
        inside = np.flatnonzero((block >= limits[0]) & (block <= limits[1]))
        if inside.size:
            first, last = min(first, start + int(inside[0])), start + int(inside[-1])
    return slice(max(0, first - 1), min(axis.size, last + 2)) if last >= 0 else slice(0, min(axis.size, 2))


def _visible_indices(axis: np.ndarray, limits: tuple[float, float] | None, limit: int, *, log: bool = False) -> np.ndarray:
    selection = _visible_slice(axis, limits)
    detail = selection.start + _bounded_indices(selection.stop - selection.start, limit)
    if limits is None:
        return detail
    bounds = (float(np.nanmin(axis)), float(np.nanmax(axis)))
    buffer = _visible_slice(axis, buffered_preview_limits(limits, bounds, log=log))
    # Each neighbouring window gets its own bounded budget. The visible
    # window retains the full configured point count, including narrow peaks.
    before = buffer.start + _bounded_indices(max(0, selection.start - buffer.start), limit)
    after = selection.stop + _bounded_indices(max(0, buffer.stop - selection.stop), limit)
    detail = np.union1d(detail, np.union1d(before, after))
    # Keep a bounded overview outside the detailed tile. Changing the browser
    # viewport can then reveal existing data immediately, before refinement.
    overview = _bounded_indices(axis.size, limit)
    outside = overview[(overview < buffer.start) | (overview >= buffer.stop)]
    return np.union1d(outside, detail)


def retain_curve_overview(overview: Sequence[Sequence[Any]], detail: Sequence[Sequence[Any]]) -> list:
    """Join a streamed/KDE detail tile to its original coarse profile."""
    if not detail:
        return list(overview)
    lower = min(row[0] for row in detail)
    upper = max(row[0] for row in detail)
    before = [row for row in overview if row[0] < lower]
    after = [row for row in overview if row[0] > upper]
    if len(overview) > 1 and overview[-1][0] < overview[0][0]:
        return after + list(detail) + before
    return before + list(detail) + after


def _line_points(x: Any, y: Any, *, limit: int | None = None, view: tuple[float, float] | None = None, log_x: bool = False) -> list[list[float | None]]:
    axis = np.asarray(x, dtype=float).reshape(-1)
    values = np.asarray(y, dtype=float).reshape(-1)
    if axis.size != values.size:
        raise ValueError("The selected output has no matching axis.")
    indices = _visible_indices(axis, view, preview_point_limit() if limit is None else limit, log=log_x)
    points = [
        [
            float(axis[index]),
            float(values[index]) if np.isfinite(values[index]) else None,
        ]
        for index in indices
    ]
    return DetailedPreviewData(points, lambda x_limits, _y_limits: {
        "data": _line_points(axis, values, limit=limit, view=x_limits, log_x=log_x),
    }, coverage={"x": buffered_preview_limits(view, (float(np.nanmin(axis)), float(np.nanmax(axis))), log=log_x)})


def _band_segments(
    band: PreviewBand,
    *,
    log_y: bool,
    log_x: bool = False,
    limit: int | None = None,
    view: tuple[float, float] | None = None,
) -> list[list[float]]:
    """Return bounded adjacent polygons for one variable-width filled band."""

    axis = np.asarray(band.x, dtype=float).reshape(-1)
    lower = np.asarray(band.lower, dtype=float).reshape(-1)
    upper = np.asarray(band.upper, dtype=float).reshape(-1)
    if axis.size != lower.size or axis.size != upper.size:
        raise ValueError("The filled interval has no matching axis.")
    if axis.size < 2:
        return []
    full_axis, full_lower, full_upper = axis, lower, upper
    indices = _visible_indices(axis, view, preview_point_limit() if limit is None else limit, log=log_x)
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
    return DetailedPreviewData(segments, lambda x_limits, _y_limits: {
        "data": _band_segments(PreviewBand(band.name, full_axis, full_lower, full_upper, band.color, opacity=band.opacity), log_y=log_y, log_x=log_x, limit=limit, view=x_limits),
    }, coverage={"x": buffered_preview_limits(view, (float(np.nanmin(full_axis)), float(np.nanmax(full_axis))), log=log_x)})


def _sample_heatmap(
    values: Any,
    x_axis: Any,
    y_axis: Any,
    *,
    limit: int = MAX_HEATMAP_CELLS,
    log_x: bool = False,
    log_y: bool = False,
    log_z: bool = False,
    scale_to_preview: bool = False,
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
    sampled = np.full(sums.shape, np.nan, dtype=float)
    np.divide(sums, counts, out=sampled, where=counts > 0)
    if scale_to_preview:
        # Density overviews average native cells. Isolated native peaks can
        # dwarf every displayed pixel and wash out the entire overview. Use
        # the same bounded raster we colour, retaining its scale on zoom.
        finite_sample = sampled[np.isfinite(sampled)]
        data_minimum = float(np.min(finite_sample))
        data_maximum = float(np.max(finite_sample))
    if log_z:
        z_minimum, z_maximum = float(data_minimum), float(math.ceil(data_maximum))
        if scale_to_preview and data_minimum == data_maximum:
            z_minimum -= 1.0
        if z_minimum == z_maximum:
            z_maximum += 1.0
    else:
        if scale_to_preview and data_minimum >= 0:
            # Anchor densities at zero, including small constant fields.
            # Otherwise generic constant-domain padding can swamp the signal.
            data_minimum = 0.0
            if data_maximum == 0:
                data_maximum = 1.0
        z_minimum, z_maximum = _nice_axis_limits(
            data_minimum, data_maximum, floor_at_zero=data_minimum >= 0, intervals=5,
        )

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


def _centered_box_renderer(gap: float, width: float) -> str:
    """Place each box on its own numeric coordinate, without series grouping offsets."""
    return (
        "(params,api)=>{const x=api.value(0);const p=[1,2,3,4,5].map(i=>api.coord([x,api.value(i)]));"
        f"const next=api.coord([x+{float(gap) / 2!r},api.value(3)]);"
        "const half=Math.min(24,Math.max(4,Math.abs(next[0]-p[2][0])*0.55));"
        "const cx=p[2][0],color=api.visual('color');"
        f"const style={{stroke:color,lineWidth:{width!r},fill:'none'}};"
        "const line=(x1,y1,x2,y2)=>({type:'line',shape:{x1,y1,x2,y2},style});"
        "return {type:'group',children:["
        "{type:'rect',shape:{x:cx-half,y:Math.min(p[1][1],p[3][1]),width:2*half,height:Math.abs(p[1][1]-p[3][1])},"
        "style:{...api.style(),stroke:color,lineWidth:style.lineWidth}},"
        "line(cx,p[0][1],cx,p[1][1]),line(cx,p[3][1],cx,p[4][1]),"
        "line(cx-half*.6,p[0][1],cx+half*.6,p[0][1]),"
        "line(cx-half*.6,p[4][1],cx+half*.6,p[4][1]),"
        "line(cx-half,p[2][1],cx+half,p[2][1])]};}"
    )


def _combined_preview_options(
    lines: Sequence[PreviewLine], arguments: dict[str, Any],
) -> dict[str, Any]:
    """Compose separately prepared curves or distributions on shared or independent axes."""

    dual = arguments["secondary_y_label"] is not None
    independent_x = dual and arguments["secondary_x_label"] is not None
    if not lines or (dual and len(lines) != 2):
        raise ValueError("Two Y axes require exactly two datasets.")
    plot_type = arguments["plot_type"]
    if plot_type not in {"Line", "Histogram", "Boxplot"}:
        raise ValueError("Separate axes require curves, histograms, or boxplots.")
    box_positions = [float(line.box_position if line.box_position is not None else index + 1)
                     for index, line in enumerate(lines)] if plot_type == "Boxplot" else []
    if box_positions and (not np.all(np.isfinite(box_positions)) or len(set(box_positions)) != len(box_positions)):
        raise ValueError("Choose a distinct, finite X position for each box.")
    if box_positions and (arguments["log_x"] or arguments["secondary_log_x"]) and min(box_positions) <= 0:
        raise ValueError("Log X requires positive box positions.")
    box_gap = min(np.diff(sorted(box_positions))) if len(box_positions) > 1 else 1.0
    box_bounds = [min(box_positions) - box_gap / 2, max(box_positions) + box_gap / 2] if box_positions else []
    if box_positions and (arguments["log_x"] or arguments["secondary_log_x"]):
        box_bounds[0] = max(box_bounds[0], min(box_positions) / 2)
    shared_range = arguments["histogram_x_limits"]
    if plot_type == "Histogram" and not independent_x and shared_range is None:
        domain = histogram_domain(
            np.concatenate([finite_values(line.y) for line in lines]),
            log_x=arguments["log_x"],
        )
        shared_range = domain[1] if domain is not None else None
    parts = []
    for index, line in enumerate(lines):
        settings = {key: value for key, value in arguments.items() if not key.startswith("secondary_")}
        settings.update(
            values=line.y, x_axis=line.x, color=line.color, series_label=line.name,
            show_legend=False, line_width=line.width or arguments["line_width"],
            line_type=line.line_type or arguments["line_type"],
        )
        if plot_type == "Line":
            settings["lines"] = (line,)
        if plot_type == "Histogram":
            settings["histogram_x_limits"] = shared_range
        if index == 1 and dual:
            settings["y_label"] = arguments["secondary_y_label"]
            settings["y_limits"] = arguments["secondary_y_limits"]
            for flag in ("log_y", "invert_y", "log_x", "invert_x"):
                value = arguments[f"secondary_{flag}"]
                if value is not None:
                    settings[flag] = value
            if independent_x:
                settings["x_label"] = arguments["secondary_x_label"]
                settings["x_limits"] = arguments["secondary_x_limits"]
                if plot_type == "Histogram":
                    settings["histogram_x_limits"] = arguments["secondary_x_limits"]
                    # Separate X scales use the same bin count, with widths
                    # derived from each dataset's own range.
                    settings["histogram_bin_size"] = None
        part = build_preview_plot_options(**settings)
        if plot_type == "Boxplot":
            axis = part["xAxis"]
            axis.pop("data", None)
            axis.update(
                type="log" if settings["log_x"] else "value",
                min=box_bounds[0], max=box_bounds[1],
                position="bottom", inverse=bool(settings["invert_x"]),
            )
            part["dnascapeDefaultAxes"]["x"] = list(box_bounds)
            if settings["x_limits"] is not None:
                axis["min"], axis["max"] = _plot_axis_limits(
                    settings["x_limits"], None, label="X", log_axis=settings["log_x"],
                )
            axis["axisLabel"]["customValues"] = box_positions
            axis["dnascapeBoxPositions"] = box_positions
            labels = [line.box_label if line.box_label is not None else _compact_tick(position)
                      for line, position in zip(lines, box_positions)]
            axis["axisLabel"][":formatter"] = (
                f"value=>{{const positions={json.dumps(box_positions)},labels={json.dumps(labels)};"
                "const i=positions.findIndex(p=>Math.abs(p-Number(value))<=1e-9*Math.max(1,Math.abs(p)));"
                "return i<0?'':labels[i];}"
            )
            axis["axisTick"]["customValues"] = box_positions
            axis["splitLine"]["customValues"] = box_positions
            series = part["series"][0]
            series["data"] = [[box_positions[index], *series["data"][0]]]
            series["dimensions"] = ["position", "min", "Q1", "median", "Q3", "max"]
            series["encode"] = {"x": 0, "y": [1, 2, 3, 4, 5], "tooltip": [1, 2, 3, 4, 5]}
            series.update(type="custom", coordinateSystem="cartesian2d", clip=True)
            series[":renderItem"] = _centered_box_renderer(box_gap, float(line.width or arguments["line_width"]))
        parts.append(part)
    options = parts[0]
    options["series"] = [series for part in parts for series in part["series"]]
    for index, series in enumerate(options["series"]):
        series["z"] = 100 - index
        if plot_type == "Histogram":
            series["itemStyle"]["opacity"] = lines[index].opacity
            if arguments["density"]:
                series["lineStyle"]["opacity"] = lines[index].opacity
        if dual:
            series["yAxisIndex"] = index
            if independent_x:
                series["xAxisIndex"] = index
    for axis_name, independent in (("yAxis", dual), ("xAxis", independent_x)):
        axes = [part[axis_name] for part in parts]
        key = axis_name[0]
        defaults = [part["dnascapeDefaultAxes"][key] for part in parts]
        if independent:
            options["dnascapeDefaultAxes"][key] = defaults[0]
            options["dnascapeDefaultAxes"][f"secondary_{key}"] = defaults[1]
            for index, axis in enumerate(axes):
                axis["position"] = ("left" if index == 0 else "right") if axis_name == "yAxis" else (
                    "bottom" if index == 0 else "top"
                )
                axis["axisLine"] = {**axis["axisLine"], "show": True, "onZero": False,
                                    "lineStyle": {**axis["axisLine"].get("lineStyle", {}), "color": lines[index].color}}
                axis["nameTextStyle"] = {**axis["nameTextStyle"], "color": lines[index].color}
            options[axis_name] = axes
        else:
            options["dnascapeDefaultAxes"][key] = [
                min(item[0] for item in defaults), max(item[1] for item in defaults),
            ]
            axis = axes[0]
            axis["min"] = min(float(item["min"]) for item in axes)
            axis["max"] = max(float(item["max"]) for item in axes)
            if axis["type"] == "value":
                axis["interval"] = _nice_tick_step(axis["max"] - axis["min"])
            if plot_type != "Boxplot" or axis_name == "yAxis":
                _use_endpoint_safe_ticks(
                    axis, data_lower=axis["min"], data_upper=axis["max"],
                    tick_lower=axis["min"], tick_upper=axis["max"], log_axis=axis["type"] == "log",
                )
    if dual:
        options["grid"]["right"] = parts[1]["grid"]["left"]
    if independent_x:
        options["grid"]["top"] = max(options["grid"]["top"], options["grid"]["bottom"])
    options["legend"] = build_legend_options(
        [line.name for line in lines], grid=options["grid"],
        position=arguments["legend_position"], show=arguments["show_legend"],
        interactive=arguments["interactive"],
    )
    style_curve_legend(options["legend"], options["series"])
    plotted = [np.asarray(series["data"], dtype=float) for series in options["series"]]
    if plot_type == "Boxplot":
        hover_x = [row[:, 0] for row in plotted]
        hover_y = [row[:, 1:].reshape(-1) for row in plotted]
    elif plot_type == "Histogram" and not arguments["density"]:
        hover_x = [row[:, :2].reshape(-1) for row in plotted]
        hover_y = [row[:, 2] for row in plotted]
    else:
        hover_x = [row[:, 0] for row in plotted]
        hover_y = [row[:, 1] for row in plotted]
    options["tooltip"] = _tooltip_options(
        interactive=arguments["interactive"], trigger="item" if plot_type == "Boxplot" else "axis",
        x_values=np.concatenate(hover_x), values=np.concatenate(hover_y),
    )
    limit_curve_hover(options)
    return options


def set_heatmap_colorbar_label(options: dict[str, Any], label: str | None = None) -> None:
    """Edit the colour-scale title without replacing its numeric endpoints.

    A blank field restores the quantity name (including any supplied units).
    The logarithmic-scale annotation belongs to the scale and stays visible
    when its title is edited.
    """
    metadata = options.get("dnascapeColorbar", {})
    for item in options.get("graphic", []):
        if item.get("id") != "dn-colorbar-label":
            continue
        title = str(label or metadata.get("label", "")).strip()
        if not title:
            title = str(metadata.get("label", ""))
        if metadata.get("log") and not title.endswith("(log scale)"):
            title += " (log scale)"
        item["style"]["text"] = title


def _build_preview_plot_options(
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
    show_grid: bool = False,
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
    kde_bandwidth: float = 1.0,
    histogram_grouped: bool = False,
    histogram_boundary: str = "left",
    x_limits: tuple[float, float] | None = None,
    z_limits: tuple[float, float] | None = None,
    heatmap_scale_to_preview: bool = False,
    aspect_ratio: float | None = None,
    line_type: str = "solid",
    secondary_y_label: str | None = None,
    secondary_y_limits: tuple[float, float] | None = None,
    secondary_x_label: str | None = None,
    secondary_x_limits: tuple[float, float] | None = None,
    secondary_log_x: bool | None = None,
    secondary_invert_x: bool | None = None,
    secondary_log_y: bool | None = None,
    secondary_invert_y: bool | None = None,
) -> dict[str, Any]:
    """Build bounded ECharts options for a line, histogram, boxplot, or heatmap."""

    if secondary_y_label is not None or (
        plot_type in {"Histogram", "Boxplot"} and len(lines) > 1
    ):
        arguments = dict(locals())
        arguments.pop("lines")
        arguments.pop("bands")
        if bands:
            raise ValueError("Separate axes and distribution plots do not support filled bands.")
        return _combined_preview_options(lines, arguments)
    if values is None and len(lines) == 1 and plot_type in {"Histogram", "Boxplot"}:
        # A single named profile is the same valid input as multiple profiles.
        # Do not require callers to duplicate its samples in ``values``.
        values = lines[0].y
        series_label = series_label or lines[0].name
        color = lines[0].color
    title_options = {"show": False, "text": title}
    if aspect_ratio is not None:
        aspect_ratio = _positive_finite(aspect_ratio)
        if aspect_ratio is None:
            raise ValueError("Aspect ratio must be finite and positive.")
    axis_line = {"show": True, "onZero": False,
                 "lineStyle": {"color": "#a69e98", "width": 1.2, "type": "solid", "opacity": 1}}
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
            scale_to_preview=heatmap_scale_to_preview,
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
        y_data_minimum, y_data_maximum = _data_axis_bounds(y_values, log_axis=log_y)
        x_minimum, x_maximum = _plot_axis_limits(
            x_limits, (x_data_minimum, x_data_maximum), label="X", log_axis=log_x,
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
        colorbar_quantity = (series_label or title).strip() or "Value"
        colorbar_label = colorbar_quantity
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
        hover_data = _heatmap_tooltip_data(
            sampled, x_limits=(x_data_minimum, x_data_maximum),
            y_limits=(y_data_minimum, y_data_maximum), log_z=log_z,
            x_coordinates=x_values, y_coordinates=y_values,
        )
        initial_raster = [[x_data_minimum, y_data_minimum, x_data_maximum, y_data_maximum]]
        source_x = np.asarray(x_axis, dtype=float).reshape(-1)
        source_y = np.asarray(y_axis, dtype=float).reshape(-1)
        source_matrix = np.asarray(values)
        if source_matrix.shape != (source_y.size, source_x.size):
            source_matrix = source_matrix.reshape(source_x.size, source_y.size).T

        def refine_heatmap(x_view: Any, y_view: Any) -> dict[str, Any]:
            def selection(axis: np.ndarray, view: Any) -> slice:
                return _visible_slice(axis, view)

            x_buffer = buffered_preview_limits(x_view, (float(np.min(source_x)), float(np.max(source_x))), log=log_x)
            y_buffer = buffered_preview_limits(y_view, (float(np.min(source_y)), float(np.max(source_y))), log=log_y)
            xs, ys = selection(source_x, x_buffer), selection(source_y, y_buffer)
            if xs.start == 0 and xs.stop == source_x.size and ys.start == 0 and ys.stop == source_y.size:
                return {"data": DetailedPreviewData(initial_raster, refine_heatmap), ":renderItem": render_item, "hoverData": hover_data}
            crop_x, crop_y, crop, _, _ = _sample_heatmap(
                source_matrix[ys, xs], source_x[xs], source_y[ys], limit=MAX_HEATMAP_CELLS * 9,
                log_x=log_x, log_y=log_y, log_z=log_z,
            )
            xmin, xmax = _data_axis_bounds(crop_x, log_axis=log_x)
            ymin, ymax = _data_axis_bounds(crop_y, log_axis=log_y)
            image = _heatmap_data_uri(crop, palette=palette, z_minimum=z_minimum, z_maximum=z_maximum,
                                      invert_x=invert_x, invert_y=invert_y, invert_z=invert_z)
            # The overview remains underneath the tile, clipped out of its
            # rectangle so genuinely missing fine cells stay transparent.
            layered_render_item = (
                "(params, api) => { "
                f"const base = ({render_item})(params, api); "
                f"const a = api.coord([{xmin!r}, {ymin!r}]), b = api.coord([{xmax!r}, {ymax!r}]); "
                "const tile = {x: Math.min(a[0],b[0]), y: Math.min(a[1],b[1]), "
                "width: Math.abs(b[0]-a[0]), height: Math.abs(b[1]-a[1])}; "
                "const s = base.style, right = tile.x+tile.width, bottom = tile.y+tile.height; "
                "const pieces = ["
                "{x:s.x,y:s.y,width:s.width,height:tile.y-s.y},"
                "{x:s.x,y:bottom,width:s.width,height:s.y+s.height-bottom},"
                "{x:s.x,y:tile.y,width:tile.x-s.x,height:tile.height},"
                "{x:right,y:tile.y,width:s.x+s.width-right,height:tile.height}]; "
                "return {type:'group',children:[...pieces.filter(r=>r.width>0&&r.height>0).map(r=>"
                "({...base,clipPath:{type:'rect',shape:r}})),"
                f"{{type:'image',silent:true,style:{{...tile,image:{json.dumps(image)},imageSmoothing:false}}}}]}}; }}"
            )
            detail_hover = _heatmap_tooltip_data(crop, x_limits=(xmin, xmax), y_limits=(ymin, ymax),
                                               log_z=log_z, x_coordinates=crop_x, y_coordinates=crop_y)
            overview_hover = [row for row in hover_data
                              if row[5] < xmin or row[3] > xmax or row[6] < ymin or row[4] > ymax]
            return {
                "data": DetailedPreviewData(initial_raster, refine_heatmap, coverage={"x": x_buffer, "y": y_buffer}),
                ":renderItem": layered_render_item,
                "hoverData": overview_hover + detail_hover,
            }
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
                    "data": hover_data,
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
            "nameMoveOverlap": False,
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
        tooltip["axisPointer"].update(
            type="cross", axis="auto", snap=False,
            crossStyle={"color": "#b8b0aa", "width": 1, "type": "dashed"},
        )
        tooltip["axisPointer"]["label"][":formatter"] = (
            "params => (params.axisDimension === 'y' ? ("
            + _hover_formatter(integer_only=_whole_numbers(y_values)) + ") : ("
            + _hover_formatter(integer_only=_whole_numbers(x_values)) + "))(params.value)"
        )
        tooltip[":formatter"] = (
            "params => { const v = params.value; "
            f"const fz = ({_hover_formatter(integer_only=not log_z and _whole_numbers(sampled))}); "
            f"const colour = ({_heatmap_value_style(palette=palette, z_minimum=z_minimum, z_maximum=z_maximum, log_z=log_z, invert_z=invert_z)})(v[2]); "
            "return '<span style=\"background:' + colour.background + ';color:' + colour.foreground + "
            "';padding:1px 4px;border-radius:3px;font-weight:600\">' + fz(v[2]) + '</span>'; }"
        )
        return {
            "animation": False,
            "dnascapeAspectRatio": aspect_ratio,
            "dnascapePlotType": "Heatmap",
            "dnascapeColorbar": {"label": colorbar_quantity, "log": bool(log_z)},
            "dnascapeDefaultAxes": {
                "x": [x_data_minimum, x_data_maximum],
                "y": [y_data_minimum, y_data_maximum],
            },
            "title": title_options,
            "tooltip": tooltip,
            "graphic": (
                [
                    {
                        "id": "dn-colorbar-label",
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
                    "data": DetailedPreviewData(initial_raster, refine_heatmap),
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
        if density:
            kde_x, kde_y = _kde_data(finite, log_x=log_x, x_limits=histogram_x_limits, bandwidth_scale=kde_bandwidth)
            options = build_preview_plot_options(
                plot_type="Line", title=title, x_label=x_label, y_label=y_label,
                color=color, palette=palette, histogram_bins=histogram_bins, density=False,
                show_legend=show_legend, show_colorbar=False, log_y=log_y,
                invert_y=invert_y, y_limits=y_limits, series_label=series_label,
                lines=(PreviewLine(series_label, kde_x, kde_y, color, width=line_width, line_type=line_type),),
                line_width=line_width, line_type=line_type, x_limits=histogram_x_limits,
                log_x=log_x, invert_x=invert_x, show_grid=show_grid,
                interactive=interactive, aspect_ratio=aspect_ratio, legend_position=legend_position,
            )
            options["dnascapePlotType"] = "Histogram"
            kde_samples = finite if finite.size <= 20_000 else finite[np.linspace(0, finite.size - 1, 20_000, dtype=int)]
            kde_overview = list(options["series"][0]["data"])

            def refine_kde(x_view: Any, _y_view: Any) -> dict[str, Any]:
                buffer = buffered_preview_limits(x_view, tuple(options["dnascapeDefaultAxes"]["x"]), log=log_x)
                count = preview_point_limit() * (3 if x_view is not None else 1)
                detail_x, detail_y = _kde_data(kde_samples, log_x=log_x,
                                              x_limits=histogram_x_limits, bandwidth_scale=kde_bandwidth,
                                              evaluation_limits=buffer, point_limit=count)
                detail = [[float(x), float(y)] for x, y in zip(detail_x, detail_y, strict=True)]
                return {"data": DetailedPreviewData(retain_curve_overview(kde_overview, detail), refine_kde, coverage={"x": buffer})}

            options["series"][0]["data"] = DetailedPreviewData(options["series"][0]["data"], refine_kde)
            return options
        # Retain the full sample for reset domains and their counts. The
        # selected window may legitimately contain no samples at all.
        _, counts, edges = _histogram_data(
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
            "name": y_label,
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
            "nameMoveOverlap": False,
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
        default_domain = histogram_domain(finite, log_x=log_x)[1]
        default_counts = counts
        if histogram_x_limits is not None and not _limits_match(histogram_x_limits, default_domain):
            _, default_counts, _ = _histogram_data(
                finite, bins=histogram_bins, bin_size=histogram_bin_size,
                density=False, log_x=log_x, x_limits=default_domain, boundary=histogram_boundary,
            )
            # Shared bin intervals can concentrate one dataset in a single
            # slot. Navigation must include the counts actually on screen.
            default_counts = np.concatenate((default_counts, counts))
        default_y = (
            _logarithmic_data_limits(
                float(np.min(default_counts[default_counts > 0])), float(np.max(default_counts)),
            ) if log_y else nice_linear_limits(0.0, float(np.max(default_counts)), floor_at_zero=True)
        )
        return {
            "animation": False,
            "dnascapeAspectRatio": aspect_ratio,
            "dnascapePlotType": "Histogram",
            "dnascapeDefaultAxes": {
                "x": list(default_domain if histogram_x_limits is not None else (
                    _logarithmic_data_limits(*default_domain) if log_x else nice_linear_limits(*default_domain)
                )),
                "y": list(default_y),
            },
            "title": title_options,
            "tooltip": _tooltip_options(interactive=interactive, trigger="axis", x_values=edges, values=counts),
            "grid": _cartesian_grid(
                y_name_gap=y_name_gap,
                y_name=histogram_y_axis["name"],
                bottom_legend=False,
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
            "dnascapePlotType": "Boxplot",
            "dnascapeDefaultAxes": {
                "x": [-0.5, 0.5],
                "y": list(
                    _logarithmic_data_limits(float(np.min(finite)), float(np.max(finite)))
                    if log_y else nice_linear_limits(float(np.min(finite)), float(np.max(finite)))
                ),
            },
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
                "nameMoveOverlap": False,
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
        segments = _band_segments(band, log_y=log_y, log_x=log_x)
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
        all_y_values.append(finite_values(flattened))
        line_options.append(
            {
                "type": "scatter" if scalar else "line",
                "name": line.name,
                # Hidden symbols appear only at the hovered curve samples.
                "symbol": "circle",
                "showSymbol": bool(scalar),
                "symbolSize": 12 if scalar else 6,
                "silent": not interactive,
                "connectNulls": False,
                # The server already bounds both overview and detail. Another
                # sampling pass could discard native peaks inside the tile.
                "sampling": None,
                "z": line.z,
                "lineStyle": {
                    "width": line.width if line.width is not None else line_width,
                    "color": line.color,
                    "opacity": line.opacity,
                    "type": curve_dash_pattern(line.line_type or line_type),
                },
                "itemStyle": {"color": line.color, "opacity": line.opacity},
                "emphasis": {"disabled": not interactive, "scale": False},
                "data": _line_points(line.x, flattened, log_x=log_x),
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
    # Keep the horizontal extent at the native coordinates. Tick placement is
    # independent, so a convenient tick must never extend the data domain.
    automatic_x_limits = (x_data_lower, x_data_upper)
    if x_data_lower == x_data_upper:
        automatic_x_limits = (
            _logarithmic_data_limits(x_data_lower, x_data_upper) if log_x
            else nice_linear_limits(x_data_lower, x_data_upper)
        )
    custom_x_limits = x_limits is not None
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
    legend_options = build_legend_options(
        legend_names,
        grid=grid,
        position=legend_position,
        show=show_legend,
        interactive=interactive,
    )
    style_curve_legend(legend_options, line_options)
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
        "nameMoveOverlap": False,
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
        tick_upper=x_limits[1] if custom_x_limits else None,
        log_axis=log_x,
    )
    return limit_curve_hover({
        "animation": False,
        "dnascapeAspectRatio": aspect_ratio,
        "dnascapePlotType": "Line",
        "dnascapeDefaultAxes": {
            "x": list(automatic_x_limits),
            "y": list(
                _logarithmic_data_limits(line_y_data_lower, line_y_data_upper) if log_y else
                nice_linear_limits(line_y_data_lower, line_y_data_upper)
            ),
        },
        "title": title_options,
        "tooltip": {
            **_tooltip_options(
                interactive=interactive,
                trigger=("item" if all(np.asarray(line.y).size == 1 for line in lines) else "axis"),
                x_values=finite_x,
                values=finite_y,
            ),
        },
        "legend": legend_options,
        "grid": grid,
        "xAxis": x_axis_options,
        "yAxis": y_axis_options,
        "series": line_options,
    })


def histogram_guide_options(
    options: dict[str, Any], datasets: Sequence[tuple[str, str, Any]], *,
    legend: bool, statistics: bool, position: str = "Top right",
) -> None:
    """Use a full legend, or a selectable statistics panel when requested."""
    summaries = []
    for name, _color, values in datasets:
        finite = finite_values(values)
        summaries.append(dict(
            dataset=str(name),
            mean=float(np.mean(finite)) if finite.size else None,
            median=float(np.median(finite)) if finite.size else None,
            standard_deviation=float(np.std(finite)) if finite.size else None,
        ))
    # Keep full precision independently of the rounded, optional guide panel.
    options["dnascapeHistogramStatistics"] = summaries
    if not statistics:
        options["legend"] = build_legend_options(
            [str(name) for name, _, _ in datasets], grid=options.get("grid", {}),
            position=position, show=legend, interactive=bool(options.get("tooltip", {}).get("show")),
        )
        options["graphic"] = []
        options["dnascapeHistogramGuides"] = dict(choices=[], selected=0, visible=False, selectable=False)
        return
    options["legend"] = {**options.get("legend", {}), "show": False}
    choices = []
    grid = options.get("grid", {})
    panel_width = min(240, max(160, max((len(str(name)[:28]) for name, _, _ in datasets), default=0) * 6 + 50))
    for (name, color, _values), summary in zip(datasets, summaries):
        heading = legend or len(datasets) > 1
        title = str(name).replace("{", "(").replace("}", ")").replace("|", " ")
        title = title[:27] + "…" if len(title) > 28 else title
        lines = ([f"{{dot|●}} {{name|{title}}}" + ("  ▾" if len(datasets) > 1 else "")] if heading else [])
        if statistics:
            lines.extend([
                f"Mean       {_compact_tick(summary['mean'])}" if summary["mean"] is not None else "No finite values",
                f"Median    {_compact_tick(summary['median'])}" if summary["median"] is not None else "",
                f"Std dev   {_compact_tick(summary['standard_deviation'])}" if summary["standard_deviation"] is not None else "",
            ])
        placement = {
            "top" if position.startswith("Top") else "bottom": grid.get("top" if position.startswith("Top") else "bottom", 24) + 8,
            "left" if position.endswith("left") else "right": grid.get("left" if position.endswith("left") else "right", 28) + 8,
        }
        graphic = dict(
            id="dn-histogram-guide", type="text", **placement,
            z=10000, z2=10000, zlevel=20, silent=len(datasets) <= 1,
            style=dict(
                text="\n".join(lines), fill="#625c57", font="11px sans-serif", lineHeight=17,
                backgroundColor="rgba(255,255,255,0.94)", borderColor="#ddd7d1", borderWidth=1,
                borderRadius=5, padding=[7, 9],
                width=panel_width - 18, overflow="truncate",
                rich=dict(dot=dict(fill=color), name=dict(fill="#625c57", fontWeight=500)),
            ),
        )
        if len(datasets) > 1:
            graphic[":onclick"] = "event => window.dnascapeHistogramMenu?.(event)"
        choices.append(dict(name=str(name), color=color, graphic=graphic))
    visible = bool(choices and (legend or statistics))
    options["graphic"] = [choices[0]["graphic"]] if visible else []
    options["dnascapeHistogramGuides"] = dict(
        choices=choices, selected=0, visible=visible, selectable=len(datasets) > 1,
        width=panel_width,
        height=14 + 17 * ((1 if legend or len(datasets) > 1 else 0) + (3 if statistics else 0)),
        position=position,
    )


def arrange_histogram_bins(options: dict[str, Any], dataset_count: int) -> None:
    for index, series in enumerate(options.get("series", [])):
        renderer = series.get(":renderItem", "")
        series[":renderItem"] = renderer.replace(
            "const x0 = Math.min(baseLeft[0], baseRight[0]); const x1 = Math.max(baseLeft[0], baseRight[0]);",
            "const slot0=Math.min(baseLeft[0],baseRight[0]),slot1=Math.max(baseLeft[0],baseRight[0]); "
            f"const x0=slot0+(slot1-slot0)*{index}/{dataset_count},x1=slot0+(slot1-slot0)*{index + 1}/{dataset_count};",
        )


@wraps(_build_preview_plot_options)
def build_preview_plot_options(**kwargs: Any) -> dict[str, Any]:
    options = _build_preview_plot_options(**kwargs)
    if kwargs.get("plot_type") == "Histogram":
        if not kwargs.get("density"):
            # Rectangle hover supplies one count. The shared browser runtime
            # places its guide at the rendered bar's centre and labels its
            # full bin interval, including grouped and logarithmic histograms.
            options["tooltip"]["trigger"] = "item"
            options["tooltip"]["axisPointer"].update(show=False, type="none")
            options["tooltip"]["axisPointer"]["label"]["show"] = False
            options["axisPointer"] = {"show": False, "triggerEmphasis": False}
        lines = kwargs.get("lines", ())
        datasets = (
            [(line.name, line.color, line.y) for line in lines]
            if lines else [(kwargs.get("series_label") or kwargs.get("title", "Values"), kwargs.get("color", "#b03412"), kwargs.get("values", []))]
        )
        filtered = []
        for index, (name, color, values) in enumerate(datasets):
            finite = finite_values(values)
            logarithmic = kwargs.get("secondary_log_x") if index == 1 and kwargs.get("secondary_x_label") is not None else kwargs.get("log_x")
            if logarithmic:
                finite = finite[finite > 0]
            domain = kwargs.get("secondary_x_limits") if index == 1 and kwargs.get("secondary_x_label") is not None else kwargs.get("histogram_x_limits")
            if domain is not None and not kwargs.get("density"):
                finite = finite[(finite >= domain[0]) & (finite <= domain[1])]
            filtered.append((name, color, finite))
        datasets = filtered
        histogram_guide_options(
            options, datasets, legend=kwargs.get("show_legend", False),
            statistics=kwargs.get("show_histogram_statistics", False),
            position=kwargs.get("legend_position", "Top right"),
        )
        for index, series in enumerate(options.get("series", [])):
            series["dnascapeDataset"] = index
        if kwargs.get("histogram_grouped") and not kwargs.get("density") and len(datasets) > 1:
            arrange_histogram_bins(options, len(datasets))
    return options
