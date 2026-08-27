"""Shared preview plotting for Mapping, Simulation, and Plotting workflows."""

from __future__ import annotations

import base64
import io
import json
import math
import textwrap
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from nicegui import ui

from app.pages.common import expansion, plot_toolbar

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

FIXED_QUANTITY_Y_LIMITS: dict[str, tuple[float, float]] = {
    "origin_efficiency": (0.0, 1.0),
    "fork_directionality": (-1.0, 1.0),
    "replicated_fraction": (0.0, 1.0),
}


@dataclass(frozen=True)
class PreviewLine:
    """One named line supplied to the shared preview renderer."""

    name: str
    x: np.ndarray
    y: np.ndarray
    color: str


@dataclass(frozen=True)
class PreviewSettingsControls:
    """Shared controls used by Mapping and Simulation preview settings."""

    view: Any
    title: Any
    series_label: Any
    x_axis_label: Any
    y_axis_label: Any
    primary_color: Any
    secondary_color: Any
    heatmap_palette: Any
    log_z: Any
    line_width: Any
    histogram_bins: Any
    show_legend: Any
    legend_position: Any
    show_grid: Any
    density: Any
    colorbar: Any
    log_y: Any
    invert_y: Any
    y_minimum: Any
    y_maximum: Any
    reset_y_limits: Any


def _mark(control: Any, markers: Mapping[str, str], key: str) -> Any:
    marker = markers.get(key)
    return control.mark(marker) if marker else control


def preview_settings_controls(
    *,
    initial_title: str,
    initial_series_label: str,
    initial_x_label: str,
    initial_y_label: str,
    on_reset_y_limits: Callable[[], Any],
    markers: Mapping[str, str] | None = None,
) -> PreviewSettingsControls:
    """Render the common, thematically grouped preview settings panel."""

    markers = {} if markers is None else markers
    with expansion(
        "Plotting settings",
        icon="palette",
        extra_classes="dn-preview-settings-section",
    ):
        settings_grid = ui.element("div").classes("dn-preview-settings-grid")
        _mark(settings_grid, markers, "container")
        with settings_grid:
            with ui.column().classes("dn-preview-setting-group"):
                ui.label("Plot").classes("dn-setting-group-title")
                view = _mark(
                    ui.select(
                        ["Default", "Histogram"],
                        value="Default",
                        label="View",
                    )
                    .props("dense options-dense")
                    .classes("dn-field"),
                    markers,
                    "view",
                )
                title = _mark(
                    ui.input(label="Title", value=initial_title).props("dense").classes("dn-field"),
                    markers,
                    "title",
                )
                series_label = _mark(
                    ui.input(label="Series label", value=initial_series_label)
                    .props("dense")
                    .classes("dn-field"),
                    markers,
                    "series_label",
                )
                histogram_bins = _mark(
                    ui.number(
                        label="Histogram bins",
                        value=60,
                        min=1,
                        max=2000,
                        step=1,
                    )
                    .props("dense")
                    .classes("dn-field dn-field-compact"),
                    markers,
                    "histogram_bins",
                )
                density = _mark(
                    ui.checkbox("Density", value=False),
                    markers,
                    "density",
                )

            with ui.column().classes("dn-preview-setting-group"):
                ui.label("Style").classes("dn-setting-group-title")
                with ui.element("div").classes("dn-preview-colour-grid"):
                    primary_color = _mark(
                        ui.input(label="Curve color", value="#2c7f77")
                        .props("type=color dense")
                        .classes("dn-field"),
                        markers,
                        "primary_color",
                    )
                    secondary_color = _mark(
                        ui.input(label="Second curve color", value="#b03412")
                        .props("type=color dense")
                        .classes("dn-field"),
                        markers,
                        "secondary_color",
                    )
                secondary_color.visible = False
                heatmap_palette = _mark(
                    ui.select(
                        ["viridis", "magma", "plasma", "cividis", "coolwarm"],
                        value="viridis",
                        label="Heatmap palette",
                    )
                    .props("dense options-dense")
                    .classes("dn-field"),
                    markers,
                    "heatmap_palette",
                )
                log_z = _mark(
                    ui.checkbox("Log Z", value=False),
                    markers,
                    "log_z",
                )
                line_width = _mark(
                    ui.number(
                        label="Curve thickness",
                        value=1.8,
                        min=0.2,
                        max=8.0,
                        step=0.1,
                    )
                    .props("dense")
                    .classes("dn-field dn-field-compact"),
                    markers,
                    "line_width",
                )

            with ui.column().classes("dn-preview-setting-group"):
                ui.label("Axes").classes("dn-setting-group-title")
                x_axis_label = _mark(
                    ui.input(label="X-axis label", value=initial_x_label)
                    .props("dense")
                    .classes("dn-field"),
                    markers,
                    "x_axis_label",
                )
                y_axis_label = _mark(
                    ui.input(label="Y-axis label", value=initial_y_label)
                    .props("dense")
                    .classes("dn-field"),
                    markers,
                    "y_axis_label",
                )
                with ui.element("div").classes("dn-inline-actions"):
                    log_y = _mark(ui.checkbox("Log Y", value=False), markers, "log_y")
                    invert_y = _mark(
                        ui.checkbox("Invert Y", value=False),
                        markers,
                        "invert_y",
                    )
                with ui.element("div").classes("dn-y-limit-editor"):
                    y_minimum = _mark(
                        ui.number(label="Y min", value=None, step=0.1)
                        .props("dense")
                        .classes("dn-field dn-field-compact"),
                        markers,
                        "y_minimum",
                    )
                    y_maximum = _mark(
                        ui.number(label="Y max", value=None, step=0.1)
                        .props("dense")
                        .classes("dn-field dn-field-compact"),
                        markers,
                        "y_maximum",
                    )
                    reset_y_limits = _mark(
                        ui.button(icon="restart_alt", on_click=on_reset_y_limits)
                        .props('flat round dense aria-label="Reset Y limits to data range"')
                        .classes("dn-y-limit-reset"),
                        markers,
                        "reset_y_limits",
                    )
                    reset_y_limits.tooltip("Reset to the data-derived range")

            with ui.column().classes("dn-preview-setting-group"):
                ui.label("Guides").classes("dn-setting-group-title")
                with ui.element("div").classes("dn-inline-actions"):
                    show_legend = _mark(
                        ui.checkbox("Legend", value=True),
                        markers,
                        "show_legend",
                    )
                    show_grid = _mark(
                        ui.checkbox("Grid", value=True),
                        markers,
                        "show_grid",
                    )
                    colorbar = _mark(
                        ui.checkbox("Colorbar", value=True),
                        markers,
                        "colorbar",
                    )
                legend_position = _mark(
                    ui.select(
                        ["Top right", "Top left", "Bottom right", "Bottom left"],
                        value="Top right",
                        label="Legend position",
                    )
                    .props("dense options-dense")
                    .classes("dn-field"),
                    markers,
                    "legend_position",
                )

    return PreviewSettingsControls(
        view=view,
        title=title,
        series_label=series_label,
        x_axis_label=x_axis_label,
        y_axis_label=y_axis_label,
        primary_color=primary_color,
        secondary_color=secondary_color,
        heatmap_palette=heatmap_palette,
        log_z=log_z,
        line_width=line_width,
        histogram_bins=histogram_bins,
        show_legend=show_legend,
        legend_position=legend_position,
        show_grid=show_grid,
        density=density,
        colorbar=colorbar,
        log_y=log_y,
        invert_y=invert_y,
        y_minimum=y_minimum,
        y_maximum=y_maximum,
        reset_y_limits=reset_y_limits,
    )


def sync_preview_settings_visibility(
    controls: PreviewSettingsControls,
    *,
    plot_type: str,
    multiple_curves: bool = False,
    allow_histogram: bool = True,
    allow_boxplot: bool = False,
) -> None:
    """Show only controls that apply to the current plot representation."""

    heatmap = plot_type == "Heatmap"
    histogram = plot_type == "Histogram"
    boxplot = plot_type == "Boxplot"
    line = plot_type == "Line"
    view_options = ["Default"]
    if allow_histogram:
        view_options.append("Histogram")
    if allow_boxplot:
        view_options.append("Boxplot")
    if list(controls.view.options) != view_options:
        controls.view.options = view_options
        if controls.view.value not in view_options:
            controls.view.value = "Default"
        controls.view.update()
    controls.heatmap_palette.visible = heatmap
    controls.log_z.visible = heatmap
    controls.colorbar.visible = heatmap
    controls.histogram_bins.visible = histogram
    controls.density.visible = histogram
    controls.primary_color.visible = not heatmap
    controls.secondary_color.visible = line and multiple_curves
    controls.line_width.visible = line
    controls.show_legend.visible = line
    controls.legend_position.visible = line and bool(controls.show_legend.value)
    primary_label = (
        "Bar color"
        if histogram
        else "Box color"
        if boxplot
        else "First curve color"
        if multiple_curves
        else "Curve color"
    )
    if controls.primary_color._props.get("label") != primary_label:
        controls.primary_color._props["label"] = primary_label
        controls.primary_color.update()


def finite_values(values: Any) -> np.ndarray:
    """Return a flattened finite float view of arbitrary numeric data."""

    flattened = np.asarray(values, dtype=float).reshape(-1)
    return flattened[np.isfinite(flattened)]


def nice_linear_limits(
    lower: float,
    upper: float,
    *,
    floor_at_zero: bool = False,
    intervals: int = 6,
) -> tuple[float, float]:
    """Round a finite range outwards to compact, human-readable tick values."""

    span = upper - lower
    padding = span * 0.05 if span > 0 else max(abs(lower) * 0.05, 0.5)
    padded_lower = 0.0 if floor_at_zero else lower - padding
    padded_upper = upper + padding
    padded_span = padded_upper - padded_lower
    if padded_span <= 0 or not math.isfinite(padded_span):
        return padded_lower, padded_upper

    rough_step = padded_span / max(1, intervals)
    exponent = math.floor(math.log10(rough_step))
    magnitude = 10.0**exponent
    fraction = rough_step / magnitude
    nice_fraction = next(
        candidate for candidate in (1.0, 2.0, 2.5, 5.0, 10.0) if fraction <= candidate
    )
    step = nice_fraction * magnitude
    nice_lower = 0.0 if floor_at_zero else math.floor(padded_lower / step) * step
    nice_upper = math.ceil(padded_upper / step) * step
    if math.isclose(nice_lower, 0.0, abs_tol=step * 1e-12):
        nice_lower = 0.0
    return nice_lower, nice_upper


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


def _axis_label_options() -> dict[str, Any]:
    return {
        "color": "#746d68",
        "hideOverlap": True,
        ":formatter": _TICK_FORMATTER,
    }


def _sampled_axis_bounds(values: np.ndarray) -> tuple[float, float]:
    """Return plotf's lower coordinate and one-sample-past upper coordinate."""

    finite = np.asarray(values, dtype=float).reshape(-1)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("Plot axes require at least one finite coordinate.")
    lower = float(np.min(finite))
    if finite.size == 1:
        upper = lower + 1.0
    else:
        ordered = np.sort(np.unique(finite))
        steps = np.diff(ordered)
        positive_steps = steps[np.isfinite(steps) & (steps > 0)]
        step = float(np.median(positive_steps)) if positive_steps.size else 1.0
        upper = float(np.max(finite)) + step
    return lower, upper


def _axis_extent(values: np.ndarray) -> tuple[float, float]:
    """Match plotf's sampled-axis extent while keeping rounded outer ticks."""

    lower, upper = _sampled_axis_bounds(values)
    return _nice_axis_limits(lower, upper)


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
) -> str:
    """Encode a bounded heatmap raster using the same nearest-image model as plotf."""

    from PIL import Image

    finite = np.isfinite(matrix)
    scale = np.zeros(matrix.shape, dtype=float)
    np.subtract(matrix, z_minimum, out=scale, where=finite)
    scale /= z_maximum - z_minimum
    np.clip(scale, 0.0, 1.0, out=scale)
    anchors = _palette_rgb(palette)
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
            counts, _ = np.histogram(
                finite,
                bins=max(1, min(2_000, int(histogram_bins))),
                density=bool(density),
            )
            finite = finite_values(counts)
    if log_y and plot_type != "Heatmap":
        finite = finite[finite > 0]
    if finite.size == 0:
        raise ValueError("The selected output has no finite values for automatic Y limits.")

    lower = float(np.min(finite))
    upper = float(np.max(finite))
    if plot_type == "Heatmap":
        return _axis_extent(finite)
    if log_y:
        log_lower = 10.0 ** math.floor(math.log10(lower))
        log_upper = 10.0 ** math.ceil(math.log10(upper))
        if math.isclose(log_lower, log_upper):
            log_lower /= 10.0
            log_upper *= 10.0
        return log_lower, log_upper
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


def _sample_heatmap(
    values: Any,
    x_axis: Any,
    y_axis: Any,
    *,
    limit: int = MAX_HEATMAP_CELLS,
    log_z: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    matrix = np.asarray(values, dtype=float)
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

    if log_z:
        positive = np.isfinite(matrix) & (matrix > 0)
        if not np.any(positive):
            raise ValueError("Log Z requires at least one positive heatmap value.")
        logarithmic = np.full(matrix.shape, np.nan, dtype=float)
        np.log10(matrix, out=logarithmic, where=positive)
        matrix = logarithmic

    z_values = finite_values(matrix)
    if z_values.size == 0:
        raise ValueError("The selected output contains no finite values.")
    if log_z:
        z_minimum = float(math.floor(float(np.min(z_values))))
        z_maximum = float(math.ceil(float(np.max(z_values))))
        if math.isclose(z_minimum, z_maximum):
            z_maximum = z_minimum + 1.0
    else:
        z_minimum, z_maximum = _nice_axis_limits(
            float(np.min(z_values)),
            float(np.max(z_values)),
            floor_at_zero=bool(np.min(z_values) >= 0),
            intervals=5,
        )

    rows, columns = matrix.shape
    if (
        rows <= MAX_HEATMAP_ROWS
        and columns <= MAX_HEATMAP_COLUMNS
        and rows * columns <= limit
    ):
        return x_values, y_values, matrix, z_minimum, z_maximum

    sampled_rows = min(rows, MAX_HEATMAP_ROWS)
    sampled_columns = min(columns, MAX_HEATMAP_COLUMNS)
    if sampled_rows * sampled_columns > limit:
        sampled_columns = max(1, min(sampled_columns, limit // sampled_rows))

    y_edges = np.linspace(0, rows, sampled_rows + 1, dtype=int)
    x_edges = np.linspace(0, columns, sampled_columns + 1, dtype=int)
    y_starts = y_edges[:-1]
    x_starts = x_edges[:-1]
    finite = np.isfinite(matrix)
    weighted = np.where(finite, matrix, 0.0)
    row_sums = np.add.reduceat(weighted, y_starts, axis=0)
    row_counts = np.add.reduceat(finite.astype(np.int64), y_starts, axis=0)
    sums = np.add.reduceat(row_sums, x_starts, axis=1)
    counts = np.add.reduceat(row_counts, x_starts, axis=1)
    sampled = np.full(sums.shape, np.nan, dtype=float)
    np.divide(sums, counts, out=sampled, where=counts > 0)

    return x_values, y_values, sampled, z_minimum, z_maximum


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
    line_width: float = 2.0,
    show_markers: bool = False,
    legend_position: str = "Top right",
    show_grid: bool = True,
    log_z: bool = False,
    interactive: bool = True,
) -> dict[str, Any]:
    """Build bounded ECharts options for a line, histogram, boxplot, or heatmap."""

    title_options = {
        "text": title,
        "left": "center",
        "textStyle": {"color": "#332f2c", "fontSize": 15, "fontWeight": 600},
    }
    axis_line = {"lineStyle": {"color": "#a69e98"}}
    split_line = {
        "show": bool(show_grid),
        "lineStyle": {"color": "#e8e3de", "type": "dashed", "width": 1},
    }
    if plot_type == "Heatmap":
        if values is None or x_axis is None or y_axis is None:
            raise ValueError("Heatmap preview requires values and both axes.")
        x_values, y_values, sampled, z_minimum, z_maximum = _sample_heatmap(
            values,
            x_axis,
            y_axis,
            log_z=log_z,
        )
        image_uri = _heatmap_data_uri(
            sampled,
            palette=palette,
            z_minimum=z_minimum,
            z_maximum=z_maximum,
            invert_y=bool(invert_y),
        )
        x_data_minimum, x_data_maximum = _sampled_axis_bounds(x_values)
        y_data_minimum, y_data_maximum = _sampled_axis_bounds(y_values)
        x_minimum, x_maximum = _axis_extent(x_values)
        if y_limits is None:
            y_minimum, y_maximum = _axis_extent(y_values)
        else:
            y_minimum, y_maximum = y_limits
        heatmap_y_axis: dict[str, Any] = {
            "type": "value",
            "min": y_minimum,
            "max": y_maximum,
            "interval": _nice_tick_step(y_maximum - y_minimum),
            "splitNumber": 6,
            "name": y_label,
            "nameLocation": "middle",
            "nameGap": 38,
            "inverse": bool(invert_y),
            "axisTick": {"show": False},
            "axisLabel": _axis_label_options(),
            "axisLine": axis_line,
            "splitLine": split_line,
        }
        colorbar_label = textwrap.fill((series_label or title).strip() or "Value", width=20)
        if log_z:
            colorbar_label = f"{colorbar_label}\n(log scale)"
        visual_map: dict[str, Any] = {
            "show": bool(show_colorbar),
            "min": z_minimum,
            "max": z_maximum,
            "calculable": False,
            "hoverLink": False,
            "realtime": False,
            "silent": True,
            "orient": "vertical",
            "right": 12,
            "top": "middle",
            "itemWidth": 12,
            "itemHeight": 180,
            "inRange": {"color": PALETTES.get(palette, PALETTES["viridis"])},
            "textStyle": {"color": "#746d68", "fontSize": 10},
            "text": [colorbar_label, ""],
            "textGap": 7,
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
        return {
            "animation": False,
            "title": title_options,
            "tooltip": {"show": False},
            "grid": {
                "left": 14,
                "right": 122 if show_colorbar else 18,
                "top": 54,
                "bottom": 14,
                "containLabel": True,
            },
            "xAxis": {
                "type": "value",
                "min": x_minimum,
                "max": x_maximum,
                "interval": _nice_tick_step(x_maximum - x_minimum),
                "splitNumber": 6,
                "position": "bottom",
                "name": x_label,
                "nameLocation": "middle",
                "nameGap": 28,
                "axisLine": {"onZero": False, **axis_line},
                "axisTick": {"show": False},
                "axisLabel": _axis_label_options(),
                "splitLine": split_line,
            },
            "yAxis": heatmap_y_axis,
            "visualMap": visual_map,
            "series": [
                {
                    "name": series_label or title,
                    "type": "custom",
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
                }
            ],
        }

    if plot_type == "Histogram":
        finite = finite_values(values)
        if finite.size == 0:
            raise ValueError("The selected output contains no finite values.")
        counts, edges = np.histogram(
            finite,
            bins=max(1, min(2_000, int(histogram_bins))),
            density=bool(density),
        )
        centers = (edges[:-1] + edges[1:]) / 2
        histogram_x_limits = _nice_axis_limits(float(edges[0]), float(edges[-1]))
        histogram_y_axis: dict[str, Any] = {
            "type": "log" if log_y else "value",
            "splitNumber": 6,
            "name": y_label or ("Density" if density else "Count"),
            "nameLocation": "middle",
            "nameGap": 36,
            "inverse": bool(invert_y),
            "axisLabel": _axis_label_options(),
            "axisLine": axis_line,
            "splitLine": split_line,
        }
        if y_limits is not None:
            histogram_y_axis["min"], histogram_y_axis["max"] = y_limits
        elif not log_y:
            histogram_y_axis["min"], histogram_y_axis["max"] = _nice_axis_limits(
                0.0,
                float(np.max(counts)),
                floor_at_zero=True,
            )
        if not log_y and "min" in histogram_y_axis and "max" in histogram_y_axis:
            histogram_y_axis["interval"] = _nice_tick_step(
                float(histogram_y_axis["max"]) - float(histogram_y_axis["min"])
            )
        return {
            "animation": False,
            "title": title_options,
            "tooltip": {
                "show": bool(interactive),
                "trigger": "axis",
                "confine": True,
            },
            "grid": {
                "left": 14,
                "right": 18,
                "top": 58,
                "bottom": 14,
                "containLabel": True,
            },
            "xAxis": {
                "type": "value",
                "min": histogram_x_limits[0],
                "max": histogram_x_limits[1],
                "interval": _nice_tick_step(histogram_x_limits[1] - histogram_x_limits[0]),
                "splitNumber": 6,
                "position": "bottom",
                "name": x_label,
                "nameLocation": "middle",
                "nameGap": 28,
                "axisLine": {"onZero": False, **axis_line},
                "axisLabel": _axis_label_options(),
                "splitLine": split_line,
            },
            "yAxis": histogram_y_axis,
            "series": [
                {
                    "name": series_label or title,
                    "type": "bar",
                    "silent": not interactive,
                    "itemStyle": {"color": color},
                    "emphasis": {"disabled": not interactive},
                    "data": [
                        [float(center), float(value)]
                        for center, value in zip(centers, counts)
                    ],
                }
            ],
        }

    if plot_type == "Boxplot":
        finite = finite_values(values)
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
            "nameGap": 36,
            "inverse": bool(invert_y),
            "axisLabel": _axis_label_options(),
            "axisLine": axis_line,
            "splitLine": split_line,
        }
        if y_limits is not None:
            box_y_axis["min"], box_y_axis["max"] = y_limits
        elif not log_y:
            box_y_axis["min"], box_y_axis["max"] = _nice_axis_limits(
                float(np.min(finite)),
                float(np.max(finite)),
            )
        if not log_y and "min" in box_y_axis and "max" in box_y_axis:
            box_y_axis["interval"] = _nice_tick_step(
                float(box_y_axis["max"]) - float(box_y_axis["min"])
            )
        return {
            "animation": False,
            "title": title_options,
            "tooltip": {
                "show": bool(interactive),
                "trigger": "item",
                "confine": True,
            },
            "grid": {
                "left": 14,
                "right": 18,
                "top": 58,
                "bottom": 14,
                "containLabel": True,
            },
            "xAxis": {
                "type": "category",
                "data": [series_label or title],
                "name": x_label,
                "nameLocation": "middle",
                "nameGap": 28,
                "axisLabel": _axis_label_options(),
                "axisLine": axis_line,
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
    for line in lines:
        flattened = np.asarray(line.y, dtype=float).reshape(-1)
        line_x = np.asarray(line.x, dtype=float).reshape(-1)
        scalar = flattened.size == 1
        all_x_values.append(finite_values(line_x))
        all_y_values.append(finite_values(flattened))
        line_options.append(
            {
                "type": "scatter" if scalar else "line",
                "name": line.name,
                "symbol": "circle" if scalar else "none",
                "showSymbol": bool(scalar),
                "symbolSize": 12 if scalar else 0,
                "silent": not interactive,
                "connectNulls": False,
                "sampling": "lttb",
                "lineStyle": {"width": line_width, "color": line.color},
                "itemStyle": {"color": line.color},
                "emphasis": {"disabled": not interactive},
                "data": _line_points(line.x, flattened),
            }
        )
    finite_x_parts = [part for part in all_x_values if part.size]
    finite_x = np.concatenate(finite_x_parts) if finite_x_parts else np.asarray([], dtype=float)
    if finite_x.size == 0:
        raise ValueError("The selected output has no finite horizontal coordinates.")
    x_limits = _nice_axis_limits(float(np.min(finite_x)), float(np.max(finite_x)))
    y_axis_options: dict[str, Any] = {
        "type": "log" if log_y else "value",
        "splitNumber": 6,
        "name": y_label,
        "nameLocation": "middle",
        "nameGap": 36,
        "inverse": bool(invert_y),
        "axisLabel": _axis_label_options(),
        "axisLine": axis_line,
        "splitLine": split_line,
    }
    if y_limits is not None:
        y_axis_options["min"], y_axis_options["max"] = y_limits
    elif not log_y:
        finite_y_parts = [part for part in all_y_values if part.size]
        finite_y = (
            np.concatenate(finite_y_parts) if finite_y_parts else np.asarray([], dtype=float)
        )
        if finite_y.size:
            y_axis_options["min"], y_axis_options["max"] = _nice_axis_limits(
                float(np.min(finite_y)),
                float(np.max(finite_y)),
            )
    if not log_y and "min" in y_axis_options and "max" in y_axis_options:
        y_axis_options["interval"] = _nice_tick_step(
            float(y_axis_options["max"]) - float(y_axis_options["min"])
        )
    normalized_legend = str(legend_position or "Top right").strip().casefold()
    legend_options: dict[str, Any] = {
        "show": bool(show_legend),
        "selectedMode": bool(interactive),
        "orient": "vertical",
        "top": 64 if normalized_legend.startswith("top") else None,
        "bottom": 62 if normalized_legend.startswith("bottom") else None,
        "left": 84 if normalized_legend.endswith("left") else None,
        "right": 28 if normalized_legend.endswith("right") else None,
        "itemWidth": 20,
        "itemHeight": 3,
        "backgroundColor": "rgba(255,255,255,0.84)",
        "borderColor": "#ddd7d1",
        "borderWidth": 1,
        "borderRadius": 4,
        "padding": [4, 6],
        "textStyle": {"color": "#625c57", "fontSize": 11},
        "data": [line.name for line in lines],
    }
    legend_options = {key: value for key, value in legend_options.items() if value is not None}
    return {
        "animation": False,
        "title": title_options,
        "tooltip": {
            "show": bool(interactive),
            "trigger": "item" if all(np.asarray(line.y).size == 1 for line in lines) else "axis",
            "confine": True,
            "axisPointer": {"type": "line"},
        },
        "legend": legend_options,
        "grid": {
            "left": 14,
            "right": 18,
            "top": 58,
            "bottom": 14,
            "containLabel": True,
        },
        "xAxis": {
            "type": "value",
            "min": x_limits[0],
            "max": x_limits[1],
            "interval": _nice_tick_step(x_limits[1] - x_limits[0]),
            "splitNumber": 6,
            "position": "bottom",
            "name": x_label,
            "nameLocation": "middle",
            "nameGap": 28,
            "axisLine": {"onZero": False, **axis_line},
            "axisLabel": _axis_label_options(),
            "splitLine": split_line,
        },
        "yAxis": y_axis_options,
        "series": line_options,
    }


def render_preview_chart(
    options: dict[str, Any],
    *,
    title: str,
    marker: str,
    on_download: Callable[[str], Any],
) -> Any:
    """Render the shared preview chart, enlargement dialog, and export toolbar."""

    with ui.dialog() as zoom_dialog, ui.card().classes("dn-plot-dialog"):
        with ui.row().classes("dn-plot-dialog-header"):
            ui.label(title).classes("dn-panel-title")
            ui.space()
            ui.button(icon="close", on_click=zoom_dialog.close).props(
                'flat round dense aria-label="Close enlarged plot"'
            )
        ui.echart(options, renderer="canvas").classes("dn-chart-wrap-large").style(
            "width: 100%; min-height: 0; height: 100%;"
        )
    with ui.element("div").classes("dn-plot-shell"):
        chart = (
            ui.echart(options, renderer="canvas").style("width: 100%; height: 360px;").mark(marker)
        )
        plot_toolbar(on_zoom=zoom_dialog.open, on_download=on_download)
    return chart
