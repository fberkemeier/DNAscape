"""Shared preview plotting for Mapping, Simulation, and Plotting workflows."""

from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from nicegui import run, ui

from app.export_settings import get_export_settings
from app.field_hints import field_hint
from app.icons import RESET_ICON
from app.notifications import notify
from dnascape.workflows.circular import DEFAULT_OUTER_RADIUS, DEFAULT_INNER_RADIUS

from dnascape.workflows.preview import (
    CURVE_TYPES,
    PALETTES as PALETTES,
    MAX_HEATMAP_ROWS as MAX_HEATMAP_ROWS,
    MAX_HEATMAP_COLUMNS as MAX_HEATMAP_COLUMNS,
    MAX_HEATMAP_CELLS as MAX_HEATMAP_CELLS,
    MAX_HEATMAP_TOOLTIP_CELLS as MAX_HEATMAP_TOOLTIP_CELLS,
    MAX_INTERACTIVE_CURVES as MAX_INTERACTIVE_CURVES,
    FIXED_QUANTITY_Y_LIMITS as FIXED_QUANTITY_Y_LIMITS,
    _PLOT_GRID_LEFT as _PLOT_GRID_LEFT,
    _PLOT_GRID_RIGHT as _PLOT_GRID_RIGHT,
    _PLOT_GRID_TOP as _PLOT_GRID_TOP,
    _PLOT_GRID_BOTTOM as _PLOT_GRID_BOTTOM,
    _HEATMAP_GRID_RIGHT as _HEATMAP_GRID_RIGHT,
    _X_AXIS_NAME_GAP as _X_AXIS_NAME_GAP,
    _Y_AXIS_NAME_GAP_MIN as _Y_AXIS_NAME_GAP_MIN,
    _Y_AXIS_NAME_GAP_MAX as _Y_AXIS_NAME_GAP_MAX,
    _AXIS_LABEL_FONT_SIZE as _AXIS_LABEL_FONT_SIZE,
    _AXIS_LABEL_MARGIN as _AXIS_LABEL_MARGIN,
    _AXIS_NAME_FONT_SIZE as _AXIS_NAME_FONT_SIZE,
    _Y_AXIS_TITLE_TICK_CLEARANCE as _Y_AXIS_TITLE_TICK_CLEARANCE,
    _PLOT_OUTER_GUTTER as _PLOT_OUTER_GUTTER,
    _AXIS_MARGIN_FRACTION as _AXIS_MARGIN_FRACTION,
    PreviewLine as PreviewLine,
    PreviewBand as PreviewBand,
    _finite_number as _finite_number,
    _positive_finite as _positive_finite,
    _histogram_bin_count as _histogram_bin_count,
    histogram_domain as histogram_domain,
    _histogram_span as _histogram_span,
    approximate_histogram_bin_size as approximate_histogram_bin_size,
    approximate_histogram_bin_count as approximate_histogram_bin_count,
    _histogram_edges as _histogram_edges,
    _histogram_data as _histogram_data,
    finite_values as finite_values,
    has_finite_values as has_finite_values,
    _validate_log_axis as _validate_log_axis,
    nice_linear_limits as nice_linear_limits,
    _logarithmic_data_limits as _logarithmic_data_limits,
    _nice_tick_step as _nice_tick_step,
    _nice_axis_limits as _nice_axis_limits,
    _TICK_FORMATTER as _TICK_FORMATTER,
    _axis_label_options as _axis_label_options,
    _axis_name_options as _axis_name_options,
    _axis_tick_candidates as _axis_tick_candidates,
    _aligned_axis_ticks as _aligned_axis_ticks,
    _use_endpoint_safe_ticks as _use_endpoint_safe_ticks,
    _y_axis_name_gap as _y_axis_name_gap,
    _cartesian_grid as _cartesian_grid,
    _data_axis_bounds as _data_axis_bounds,
    _compact_tick as _compact_tick,
    _limits_match as _limits_match,
    _automatic_linear_limits as _automatic_linear_limits,
    _fixed_scientific_limits as _fixed_scientific_limits,
    _tooltip_options as _tooltip_options,
    limit_curve_hover as limit_curve_hover,
    _palette_rgb as _palette_rgb,
    _heatmap_data_uri as _heatmap_data_uri,
    automatic_y_limits as automatic_y_limits,
    _bounded_indices as _bounded_indices,
    _line_points as _line_points,
    _band_segments as _band_segments,
    _sample_heatmap as _sample_heatmap,
    _heatmap_tooltip_data as _heatmap_tooltip_data,
    build_preview_plot_options as build_preview_plot_options,
    set_heatmap_colorbar_label,
)


@dataclass
class _PlotTitle:
    """Internal result metadata; titles are not editable plot settings."""
    value: str

    def update(self) -> None:
        pass


@dataclass(frozen=True)
class PreviewSettingsControls:
    """Shared controls used by Mapping, Simulation, and Plotting previews."""

    view: Any
    title: Any
    x_axis_label: Any
    y_axis_label: Any
    primary_color: Any
    primary_color_surface: Any
    secondary_color: Any
    secondary_color_surface: Any
    heatmap_palette: Any
    log_x: Any
    invert_x: Any
    log_z: Any
    invert_z: Any
    x_axis_options: Any
    y_axis_options: Any
    z_axis_options: Any
    line_width: Any
    histogram_bins: Any
    histogram_bin_size: Any
    histogram_touching: Any
    histogram_boundary: Any
    kde_bandwidth: Any
    histogram_grouped: Any
    show_hover: Any
    ensemble_profile_options: Any
    show_individual_profiles: Any
    individual_profile_count: Any
    show_standard_deviation: Any
    histogram_x_limit_editor: Any
    histogram_x_minimum: Any
    histogram_x_maximum: Any
    reset_histogram_x_limits: Any
    show_histogram_statistics: Any
    histogram_statistics_position: Any
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
    x_limit_editor: Any = None
    x_minimum: Any = None
    x_maximum: Any = None
    reset_x_limits: Any = None
    z_limit_editor: Any = None
    z_minimum: Any = None
    z_maximum: Any = None
    reset_z_limits: Any = None
    aspect_ratio: Any = None
    reset_aspect_ratio: Any = None
    line_type: Any = None
    show_fill: Any = None
    fill_side: Any = None
    fill_options: Any = None
    reverse_order: Any = None
    circular_combine: Any = None
    circular_inner_radius: Any = None
    circular_options: Any = None
    circular_outer_radius: Any = None
    reset_circular_radii: Any = None
    _resetting_circular_radii: bool = field(default=False, init=False, repr=False)
    z_axis_label: Any = None
    surface: Any = None
    _sync_navigation_controls: bool = field(default=True, init=False, repr=False)
    _stacked_navigation: dict = field(default_factory=dict, init=False, repr=False)
    _cartesian_aspect_ratio: float = field(default=2.5, init=False, repr=False)
    legend_name: Any = None
    show_filter: Any = None
    filter_guide_options: Any = None
    filter_name: Any = None
    filter_guides: Mapping[str, Any] = field(default_factory=dict)
    filter_guide_names: Mapping[str, Any] = field(default_factory=dict)
    export_histogram_statistics: Any = None
    histogram_statistics_options: Any = None
    dual_y_axes: Any = None
    secondary_y_axis_label: Any = None
    secondary_y_limit_editor: Any = None
    secondary_y_minimum: Any = None
    secondary_y_maximum: Any = None
    reset_secondary_y_limits: Any = None
    x_axis_caption: Any = None
    y_axis_caption: Any = None
    secondary_x_axis_label: Any = None
    secondary_x_limit_editor: Any = None
    secondary_x_minimum: Any = None
    secondary_x_maximum: Any = None
    reset_secondary_x_limits: Any = None
    secondary_x_axis_options: Any = None
    secondary_y_axis_options: Any = None
    secondary_log_x: Any = None
    secondary_invert_x: Any = None
    secondary_log_y: Any = None
    secondary_invert_y: Any = None
    _resetting_limits: bool = field(default=False, init=False, repr=False)
    _automatic_axes: set[str] = field(default_factory=lambda: {"x", "z"}, init=False, repr=False)
    _log_limit_error: tuple | None = field(default=None, init=False, repr=False)
    _navigation: _PreviewNavigation = field(default_factory=lambda: _PreviewNavigation(), init=False, repr=False)
    _histogram_dataset: int = field(default=0, init=False, repr=False)
    _statistics_export: Any = field(default=None, init=False, repr=False)


@dataclass
class _PreviewNavigation:
    """Viewport state survives presentation changes without rebinning data."""

    context: str = ""
    ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    source_limits: dict[str, tuple[float, float] | None] = field(default_factory=dict)
    axes: dict[str, dict[str, Any]] = field(default_factory=dict)
    title_gaps: dict[str, tuple[str, float]] = field(default_factory=dict)
    token: str = field(default_factory=lambda: uuid4().hex)
    sequence: int = 0

    def clear(self, axis: str | None = None) -> None:
        self.token = uuid4().hex
        self.sequence = 0
        if axis is None:
            self.ranges.clear()
            self.source_limits.clear()
        else:
            self.ranges.pop(axis, None)
            self.source_limits.pop(axis, None)


class _CircularRefinementWindow(tuple):
    """Keep a seam-crossing viewport as two native-coordinate detail reads."""

    def __new__(cls, arguments, parts):
        instance = super().__new__(cls, arguments)
        instance.parts = parts
        return instance


def refresh_statistics_exports(controls: PreviewSettingsControls) -> None:
    exporter = getattr(controls, "_statistics_export", None)
    if exporter is not None:
        exporter.refresh()


def _navigation_axes(options: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    defaults = options.get("dnascapeDefaultAxes", {})
    result = {}
    for dimension in ("x", "y"):
        specifications = options.get(f"{dimension}Axis", [])
        if isinstance(specifications, Mapping):
            specifications = [specifications]
        for index, spec in enumerate(specifications[:2]):
            if spec.get("type") == "category":
                continue
            key = dimension if index == 0 else f"secondary_{dimension}"
            bounds = defaults.get(key, (spec.get("min"), spec.get("max")))
            if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
                continue
            lower, upper = (_finite_number(value) for value in bounds)
            logarithmic = spec.get("type") == "log"
            if lower is None or upper is None or lower >= upper or (logarithmic and lower <= 0):
                continue
            result[key] = dict(min=lower, max=upper, log=logarithmic, inverse=bool(spec.get("inverse")), index=index, dimension=dimension)
    return result


def _clamp_navigation_range(
    values: Any, axis: Mapping[str, Any],
) -> tuple[float, float] | None:
    if not isinstance(values, (list, tuple)) or len(values) != 2:
        return None
    lower, upper = (_finite_number(value) for value in values)
    if lower is None or upper is None or lower >= upper or (axis["log"] and lower <= 0):
        return None
    transform = math.log if axis["log"] else float
    inverse = math.exp if axis["log"] else float
    floor, ceiling = transform(axis["min"]), transform(axis["max"])
    start, end = transform(lower), transform(upper)
    span = min(end - start, ceiling - floor)
    if axis.get("circular") and span < (ceiling - floor) * (1 - 1e-9):
        start = floor + (start - floor) % (ceiling - floor)
    else:
        start = max(floor, min(start, ceiling - span))
    return inverse(start), inverse(start + span)


def _set_navigation_axis(options: dict[str, Any], key: str, limits: tuple[float, float]) -> None:
    dimension = key[-1]
    specifications = options[f"{dimension}Axis"]
    multiple = isinstance(specifications, list)
    index = 1 if key.startswith("secondary_") else 0
    spec = dict(specifications[index] if multiple else specifications)
    spec.update(min=limits[0], max=limits[1], interval=None)
    axis_line = spec.get("axisLine", {})
    spec["axisLine"] = {**axis_line, "show": True, "onZero": False,
                        "lineStyle": {"width": 1.2, **axis_line.get("lineStyle", {}), "type": "solid", "opacity": 1}}
    for part in ("axisLabel", "axisTick", "splitLine"):
        spec[part] = {**spec.get(part, {}), "customValues": None}
    spec["axisLabel"].update(showMinLabel=True, showMaxLabel=True)
    if dimension == "x" and "dnascapeBoxPositions" in spec:
        ticks = [value for value in spec["dnascapeBoxPositions"] if limits[0] <= value <= limits[1]]
        for part in ("axisLabel", "axisTick", "splitLine"):
            spec[part]["customValues"] = ticks
    elif dimension == "x" and spec.get("type") == "value":
        _use_endpoint_safe_ticks(
            spec, data_lower=limits[0], data_upper=limits[1], log_axis=False,
        )
    if multiple:
        specifications = list(specifications)
        specifications[index] = spec
        options[f"{dimension}Axis"] = specifications
    else:
        options[f"{dimension}Axis"] = spec


@dataclass
class PreviewChartHandle:
    """One visible canvas chart paired with a lazy export-only SVG renderer.

    Canvas remains the most responsive renderer for large interactive previews.
    ECharts can only return genuine SVG from an instance created with its SVG
    renderer, however, so vector downloads are routed to an invisible twin. Its
    options are populated only when a download is requested, avoiding the cost
    of maintaining two full charts during live interaction.
    """

    canvas: Any
    vector: Any
    settings: PreviewSettingsControls | None = None
    zoom: Any = None
    zoom_card: Any = None
    zoom_dialog: Any = None
    shell: Any = None
    source_options: dict[str, Any] | None = field(default=None, repr=False)
    _navigation: _PreviewNavigation = field(default_factory=_PreviewNavigation, init=False, repr=False)
    _histogram_dataset: int = field(default=0, init=False, repr=False)
    _revision: int = field(default=0, init=False, repr=False)
    _had_histogram_guide: bool = field(default=False, init=False, repr=False)
    _vector_revision: int = field(default=-1, init=False, repr=False)
    _vector_capture_revision: int = field(default=-1, init=False, repr=False)
    _vector_capture_key: str | None = field(default=None, init=False, repr=False)
    _vector_capture: str | None = field(default=None, init=False, repr=False)
    _refining: bool = field(default=False, init=False, repr=False)
    _refinement_request: int = field(default=0, init=False, repr=False)
    _refinement_idle: asyncio.Event = field(default_factory=asyncio.Event, init=False, repr=False)
    _vector_capture_lock: asyncio.Lock = field(
        default_factory=asyncio.Lock,
        init=False,
        repr=False,
    )

    @property
    def options(self) -> dict[str, Any]:
        """Expose the visible chart options for existing live-update callers."""

        # NiceGUI wraps nested lists in ObservableList, dropping refinement
        # callbacks. Keep the native sources in a separate server-side dict.
        return self.source_options if self.source_options is not None else self.canvas.options

    def _sync_canvas_options(self) -> None:
        if self.canvas.options is not self.options:
            self.canvas.options.clear()
            self.canvas.options.update(self.options)

    def update(self) -> None:
        """Update the interactive chart; the vector twin stays lazy."""

        self._apply_navigation()
        self._refine_data()
        self._apply_presentation()
        self._apply_curve_fill()
        self._apply_guides()
        self._position_toolbar()
        if self.settings is not None:
            if self.settings._sync_navigation_controls:
                sync_preview_axis_limits(self.settings, self.options)
            self._apply_aspect_ratio()
        self._sync_canvas_options()
        self.canvas.update()
        if self.settings is not None:
            refresh_statistics_exports(self.settings)
        if self.zoom is not None and self.zoom_dialog is not None and self.zoom_dialog.value:
            self._sync_zoom()
        self.invalidate_export()

    def invalidate_export(self) -> None:
        """Discard SVG captures after a data, style, or browser theme change."""

        self._revision += 1
        self._vector_capture = None
        self._vector_capture_revision = -1
        self._vector_capture_key = None

    def attach_settings(self, controls: PreviewSettingsControls) -> None:
        self.settings = controls
        self._navigation = controls._navigation
        self._apply_navigation()
        self._refine_data()
        self._apply_presentation()
        self._apply_curve_fill()
        self._apply_guides()
        self._position_toolbar()
        if controls._sync_navigation_controls:
            sync_preview_axis_limits(controls, self.options)
        self._apply_aspect_ratio()
        self._sync_canvas_options()
        refresh_statistics_exports(controls)

    def _apply_curve_fill(self) -> None:
        """Fill each curve to the displayed edge, including inverted/log axes.

        ECharts resolves start/end against the current axis limits, so the fill
        follows browser gestures and vector exports without extra data series.
        """
        control = getattr(self.settings, "show_fill", None)
        if control is None:
            return
        enabled = bool(control.value)
        above = self.settings.fill_side.value == "above"
        axes = self.options.get("yAxis", {})
        axes = axes if isinstance(axes, list) else [axes]
        for series in self.options.get("series", []):
            if series.get("type") != "line":
                continue
            if not enabled:
                series["areaStyle"] = None
                continue
            axis = axes[min(int(series.get("yAxisIndex", 0)), len(axes) - 1)]
            style = series.get("lineStyle", {})
            series["areaStyle"] = {
                "color": style.get("color", series.get("itemStyle", {}).get("color")),
                "opacity": 0.15 * float(style.get("opacity", 1.0)),
                "origin": "end" if above != bool(axis.get("inverse")) else "start",
            }

    def _apply_presentation(self) -> None:
        if self.settings is None:
            return
        heatmap = self.options.get("dnascapePlotType") == "Heatmap"
        if not heatmap:
            reverse = bool(getattr(self.settings.reverse_order, "value", False))
            series = self.options.get("series", [])
            for index, item in enumerate(series):
                item.setdefault("dnascapePresentationOrder", index)
                item.setdefault("dnascapePresentationZ", item.get("z", 2))
            self.options["series"] = sorted(series, key=lambda item: (
                item.get("type") != "custom", (-1 if reverse else 1)*item["dnascapePresentationOrder"],
            ))
            for kind in {item.get("type") for item in series}:
                layer = [item for item in self.options["series"] if item.get("type") == kind and not item.get("silent")]
                for item, z in zip(layer, sorted((item["dnascapePresentationZ"] for item in layer), reverse=True)):
                    item["z"] = z
            legend = self.options.get("legend", {})
            if isinstance(legend, dict) and isinstance(legend.get("data"), list):
                self.options.setdefault("dnascapeLegendOrder", list(legend["data"]))
                legend["data"] = list(reversed(self.options["dnascapeLegendOrder"])) if reverse else list(self.options["dnascapeLegendOrder"])
            if self.settings.view.value == "Circular":
                inner = _finite_number(self.settings.circular_inner_radius.value)
                outer = _finite_number(self.settings.circular_outer_radius.value)
                if inner is None or outer is None or not 0 <= inner < outer <= 100:
                    return
                from dnascape.workflows.circular import circular_preview_options
                circular_preview_options(self.options,
                    layout="Overlay" if self.settings.circular_combine.value else "Rings",
                    outer_radius=outer, inner_radius=inner)
                self.settings.circular_combine.set_visibility(self.options.get("dnascapeCircularDatasetCount", 1) > 1)
        else:
            label = str(getattr(self.settings.z_axis_label, "value", "") or "")
            set_heatmap_colorbar_label(self.options, label)
            default = self.options.get("dnascapeColorbar", {}).get("label", "")
            if self.settings.z_axis_label._props.get("placeholder") != default:
                self.settings.z_axis_label._props["placeholder"] = default
                self.settings.z_axis_label.update()

    def _position_toolbar(self) -> None:
        if self.shell is None:
            return
        grid = self.options.get("grid", {})
        if isinstance(grid, list):
            grid = grid[0] if grid else {}
        right = _finite_number(grid.get("right"))
        bottom = _finite_number(grid.get("bottom"))
        self.shell.style(
            f"--dn-plot-grid-right: {max(0, right if right is not None else 24)}px; "
            f"--dn-plot-grid-bottom: {max(0, bottom if bottom is not None else 58)}px;"
        )

    def _apply_guides(self) -> None:
        hover = bool(self.options.get("tooltip", {}).get("show")) and (
            self.settings is None or bool(self.settings.show_hover.value)
        )
        self.options["axisPointer"] = {
            **self.options.get("axisPointer", {}),
            "show": "auto" if hover else False,
            "triggerOn": "mousemove|click" if hover else "none",
        }
        if not hover:
            tooltip = dict(self.options.get("tooltip", {}))
            tooltip["show"] = False
            tooltip["triggerOn"] = "none"
            pointer = tooltip.get("axisPointer", {})
            tooltip["axisPointer"] = {**pointer, "show": False, "type": "none",
                                      "label": {**pointer.get("label", {}), "show": False}}
            self.options["tooltip"] = tooltip
            for axis in self.options.get("angleAxis", []):
                axis["axisPointer"] = {**axis.get("axisPointer", {}), "show": False}
            for series in self.options.get("series", []):
                series["emphasis"] = {**series.get("emphasis", {}), "disabled": True}
                if series.get("type") == "line":
                    series["symbolSize"] = 0
        guide = self.options.get("dnascapeHistogramGuides")
        if guide and guide.get("visible"):
            self._had_histogram_guide = True
        elif self._had_histogram_guide:
            # ECharts merges graphic elements by id; an empty list leaves the
            # previous statistics panel on screen, including in the zoom view.
            self.options["graphic"] = [
                item for item in self.options.get("graphic", []) if item.get("id") != "dn-histogram-guide"
            ] + [{"id": "dn-histogram-guide", "$action": "remove"}]
        if self.options.get("dnascapeCircular"):
            self.options["axisPointer"] = {"show": False, "triggerOn": "none"}
            self.options["tooltip"]["triggerOn"] = "none"
        if hasattr(self.canvas, "classes"):
            if guide and guide.get("visible"):
                self.canvas.classes(add="dn-preview-guide-visible")
            else:
                self.canvas.classes(remove="dn-preview-guide-visible")
        if not guide or not guide.get("choices"):
            return
        selected = self.settings._histogram_dataset if self.settings is not None else self._histogram_dataset
        selected = max(0, min(int(selected), len(guide["choices"]) - 1))
        guide["selected"] = selected
        graphic = [item for item in self.options.get("graphic", []) if item.get("id") != "dn-histogram-guide"]
        self.options["graphic"] = graphic + ([guide["choices"][selected]["graphic"]] if guide["visible"] else [])

    def select_histogram_dataset(self, payload: Any) -> None:
        if not isinstance(payload, Mapping) or not isinstance(payload.get("index"), int):
            return
        selected = payload["index"]
        guide = self.options.get("dnascapeHistogramGuides", {})
        if not guide.get("selectable"):
            return
        if not 0 <= selected < len(guide.get("choices", [])):
            return
        self._histogram_dataset = selected
        if self.settings is not None:
            object.__setattr__(self.settings, "_histogram_dataset", selected)
        self.update()

    def _refinement_plan(self, *, force: bool = False) -> list:
        jobs = []
        def circular_window(arguments, key):
            axis = self._navigation.axes.get(key, {})
            if axis.get("circular") and arguments[0] is not None:
                from dnascape.workflows.circular import circular_window_parts
                parts = circular_window_parts(axis, arguments[0])
                if len(parts) > 1:
                    return _CircularRefinementWindow(arguments, parts)
            return arguments
        def visible_window(key: str) -> tuple[float, float] | None:
            if key in self._navigation.ranges:
                return self._navigation.ranges[key]
            axis = self._navigation.axes.get(key)
            if axis is None:
                return None
            spec = self.options[f"{axis['dimension']}Axis"]
            if isinstance(spec, list):
                spec = spec[axis["index"]]
            current = (_finite_number(spec.get("min")), _finite_number(spec.get("max")))
            if all(value is not None for value in current) and any(
                not math.isclose(current[index], axis[endpoint], rel_tol=1e-10, abs_tol=1e-12)
                for index, endpoint in enumerate(("min", "max"))
            ):
                return current
            return None

        for index, series in enumerate(self.options.get("series", [])):
            x_key = "secondary_x" if series.get("xAxisIndex") == 1 else "x"
            y_key = "secondary_y" if series.get("yAxisIndex") == 1 else "y"
            areas = series.get("dnascapeCircularMaskSource", series.get("markArea", {}).get("data"))
            if callable(getattr(areas, "refine", None)):
                area_window = ((visible_window(x_key), visible_window(y_key))
                               if getattr(areas, 'supports_y', False) else visible_window(x_key))
                if areas.window != area_window:
                    arguments = area_window if getattr(areas, 'supports_y', False) else (area_window,)
                    jobs.append((index, "areas", areas, circular_window(arguments, x_key)))
            data = series.get("dnascapeCircularSourceData", series.get("data"))
            if not callable(getattr(data, "refine", None)):
                continue
            y_key = "secondary_y" if series.get("yAxisIndex") == 1 else "y"
            window = (visible_window(x_key), visible_window(y_key))
            if not force and window == (None, None) and not getattr(data, "window", None):
                continue
            if not force and getattr(data, "window", None) == window:
                continue
            if not force and self._reusable_detail(data, window, (x_key, y_key)):
                continue
            jobs.append((index, "data", data, circular_window(window, x_key)))
        return jobs

    def _reusable_detail(self, data: Any, window: tuple, keys: tuple) -> bool:
        previous, coverage = getattr(data, "window", None), getattr(data, "coverage", None)
        if not previous or not coverage:
            return False
        for view, old_view, key, dimension in zip(window, previous, keys, ("x", "y"), strict=True):
            if view == old_view or dimension not in coverage:
                continue
            bounds = coverage[dimension]
            if view is None or old_view is None or bounds is None:
                return False
            axis = self._navigation.axes[key]
            transform = math.log if axis["log"] else float
            lower, upper = map(transform, view)
            span = upper - lower
            # Refine on a meaningful zoom-in, or before the prefetched tile
            # runs out. Small pans and wheel steps reuse the existing data.
            if span < (transform(old_view[1]) - transform(old_view[0])) * .85:
                return False
            if lower - span * .25 < transform(bounds[0]) and bounds[0] > axis["min"]:
                return False
            if upper + span * .25 > transform(bounds[1]) and bounds[1] < axis["max"]:
                return False
        return True

    @staticmethod
    def _compute_refinements(jobs: list) -> list:
        from dnascape.workflows.preview import DetailedPreviewData

        results = []
        for job in jobs:
            _index, kind, source, arguments = job
            parts = getattr(arguments, "parts", None)
            if parts is None:
                results.append((job, source.refine(*arguments)))
                continue
            pieces = [source.refine(part, *arguments[1:]) for part in parts]

            def merge(values):
                unique = {json.dumps(item, sort_keys=True): item for value in values for item in value}
                items = list(unique.values())
                if kind == "data":
                    if items and len(items[0]) == 6:
                        # Rebuild one uncertainty strip from its native nodes;
                        # overlapping coarse/detail quads would darken the band.
                        nodes = {item[offset]: item[offset+1:offset+3]
                                 for item in items for offset in (0, 3)}
                        coordinates = sorted(nodes)
                        items = [[a, *nodes[a], b, *nodes[b]] for a, b in zip(coordinates, coordinates[1:])]
                    else:
                        items.sort(key=lambda item: item[0] if item[0] is not None else -math.inf)
                merged = DetailedPreviewData(items, source.refine)
                if hasattr(source, "supports_y"):
                    merged.supports_y = source.supports_y
                return merged

            if kind == "areas":
                result = merge(pieces)
            else:
                result = dict(pieces[0])
                result["data"] = merge([piece["data"] for piece in pieces])
                if "hoverData" in result:
                    result["hoverData"] = merge([piece.get("hoverData", []) for piece in pieces])
            results.append((job, result))
        return results

    def _install_refinements(self, results: list) -> bool:
        changed = False
        for (index, kind, source, window), result in results:
            if index >= len(self.options.get("series", [])):
                continue
            series = self.options["series"][index]
            current = series.get("dnascapeCircularMaskSource", series.get("markArea", {}).get("data")) if kind == "areas" else series.get("dnascapeCircularSourceData", series.get("data"))
            if current is not source:
                continue
            changed = True
            if kind == "areas":
                if "dnascapeCircularMaskSource" in series:
                    series["dnascapeCircularMaskSource"] = result
                else:
                    series["markArea"]["data"] = result
                continue
            refined = dict(result)
            hover = refined.pop("hoverData", None)
            series.update(refined)
            if "dnascapeCircularSourceData" in series:
                series["dnascapeCircularSourceData"] = series["data"]
            if hasattr(series["data"], "refine"):
                series["data"].window = window if any(value is not None for value in window) else None
            if hover is not None and index + 1 < len(self.options["series"]):
                self.options["series"][index + 1]["data"] = hover
        return changed

    def _refine_data(self, *, force: bool = False) -> None:
        self._install_refinements(self._compute_refinements(self._refinement_plan(force=force)))

    def _apply_navigation(self) -> None:
        axes = _navigation_axes(self.options)
        if not axes:
            return
        circular = (self.settings.view.value == "Circular" if self.settings is not None
                    else bool(self.options.get("dnascapeCircular")))
        if circular:
            for axis in axes.values():
                if axis["dimension"] == "x":
                    axis["circular"] = True
        context = json.dumps([
            self.options.get("dnascapePlotType"),
            circular,
            {key: {name: value for name, value in axis.items() if name != "inverse"} for key, axis in axes.items()},
        ], sort_keys=True)
        if self._navigation.context != context:
            self._navigation.clear()
            self._navigation.title_gaps.clear()
            self._navigation.context = context
            self._navigation.axes = {key: dict(axis) for key, axis in axes.items()}
            # Workflows supply zero floors and fixed scientific ranges, such
            # as fractions from 0 to 1. Preserve those defaults through redraws
            # whose numeric axis inputs now describe a zoomed view.
            for key, axis in self._navigation.axes.items():
                if axis["dimension"] != "y" or axis["log"]:
                    continue
                spec = self.options["yAxis"]
                if isinstance(spec, list):
                    spec = spec[axis["index"]]
                provided = (_finite_number(spec.get("min")), _finite_number(spec.get("max")))
                if all(value is not None for value in provided) and _fixed_scientific_limits(provided) and (
                    axis["min"] >= provided[0] - (provided[1] - provided[0]) * 0.1
                    and axis["max"] <= provided[1] + (provided[1] - provided[0]) * 0.1
                ):
                    axis["min"], axis["max"] = provided
                elif spec.get("min") == 0 and math.isclose(spec.get("max", 0), axis["max"], rel_tol=1e-9):
                    axis["min"] = min(0.0, axis["min"])
        for key, axis in axes.items():
            self._navigation.axes[key]["inverse"] = axis["inverse"]
        axes = self._navigation.axes
        self.options["dnascapeNavigation"] = dict(
            context=context, axes=axes, token=self._navigation.token, sequence=self._navigation.sequence,
            polar=circular,
            heatmap=self.options.get("dnascapePlotType") == "Heatmap",
        )
        for key, values in tuple(self._navigation.ranges.items()):
            if key in axes and (limits := _clamp_navigation_range(values, axes[key])) is not None:
                _set_navigation_axis(self.options, key, limits)
        self._stabilize_title_spacing()
        self._sync_navigation_fields()

    def _stabilize_title_spacing(self) -> None:
        """Retain collision-driven spacing through redraws and vector exports."""
        grid = self.options.get("grid", {})
        grid = grid[0] if isinstance(grid, list) and grid else grid
        specifications = self.options.get("yAxis", [])
        specifications = specifications if isinstance(specifications, list) else [specifications]
        for index, spec in enumerate(specifications):
            key = "secondary_y" if index else "y"
            name = spec.get("name", "")
            gap = _finite_number(spec.get("nameGap"))
            if not name or gap is None:
                self._navigation.title_gaps.pop(key, None)
                continue
            previous = self._navigation.title_gaps.get(key)
            retained = max(gap, previous[1]) if previous and previous[0] == name else gap
            self._navigation.title_gaps[key] = (name, retained)
            if retained > gap:
                side = "right" if spec.get("position") == "right" else "left"
                grid[side] = float(grid.get(side, 24)) + retained - gap
                spec["nameGap"] = retained

    def _sync_navigation_fields(self) -> None:
        controls = self.settings
        if controls is None or not controls._sync_navigation_controls:
            return
        object.__setattr__(controls, "_resetting_limits", True)
        try:
            for key, values in self._navigation.ranges.items():
                field_axis = "histogram_x" if key == "x" and controls.view.value == "Histogram" else key
                for endpoint, value in zip(("minimum", "maximum"), values, strict=True):
                    control = getattr(controls, f"{field_axis}_{endpoint}", None)
                    if control is not None:
                        control.set_value(float(f"{value:.12g}"))
        finally:
            object.__setattr__(controls, "_resetting_limits", False)

    def _commit_navigation(self, payload: Any) -> bool:
        """Commit a small browser viewport event, keeping all profile data intact."""
        if not isinstance(payload, Mapping) or payload.get("context") != self._navigation.context:
            return False
        if payload.get("token", self._navigation.token) != self._navigation.token:
            return False
        sequence = payload.get("sequence")
        if sequence is not None and (
            not isinstance(sequence, int) or isinstance(sequence, bool) or sequence <= self._navigation.sequence
        ):
            return False
        axes = self._navigation.axes
        requested = payload.get("ranges")
        if not isinstance(requested, Mapping):
            return False
        if sequence is not None:
            self._navigation.sequence = sequence
        for key, values in requested.items():
            if key not in axes or (limits := _clamp_navigation_range(values, axes[key])) is None:
                continue
            if self.settings is not None and key not in self._navigation.source_limits:
                self._navigation.source_limits[key] = (
                    histogram_x_limits_from_controls(self.settings)
                    if key == "x" and self.settings.view.value == "Histogram" else
                    axis_limits_from_controls(self.settings, key)
                )
            self._navigation.ranges[key] = limits
        title_gaps = payload.get("titleGaps", {})
        if isinstance(title_gaps, Mapping):
            specifications = self.options.get("yAxis", [])
            specifications = specifications if isinstance(specifications, list) else [specifications]
            for key, raw_gap in title_gaps.items():
                axis = axes.get(key)
                gap = _finite_number(raw_gap)
                if axis is None or axis["dimension"] != "y" or gap is None or not 0 <= gap <= 128:
                    continue
                name = specifications[axis["index"]].get("name", "")
                previous = self._navigation.title_gaps.get(key)
                self._navigation.title_gaps[key] = (
                    name, max(gap, previous[1]) if previous and previous[0] == name else gap,
                )
        return True

    def accept_navigation(self, payload: Any) -> None:
        """Commit synchronously for callers already running outside a UI gesture."""
        if self._commit_navigation(payload):
            self.update()

    async def accept_navigation_async(self, payload: Any) -> None:
        """Coalesce native-data reads in one worker without blocking the UI."""
        if not self._commit_navigation(payload):
            return
        self._apply_navigation()
        if self.options.get("dnascapeCircular"):
            # Even a fully sampled profile needs its polar axes and clipped
            # series updated before an SVG export or a later style redraw.
            self._apply_presentation()
            self._apply_curve_fill()
            self._apply_guides()
            self._sync_canvas_options()
        if self.settings is not None and self.settings._sync_navigation_controls:
            sync_preview_axis_limits(self.settings, self.options)
        self.invalidate_export()
        self._refinement_request += 1
        if self._refining:
            return
        self._refining = True
        self._refinement_idle.clear()
        try:
            while not getattr(self.canvas, "is_deleted", False):
                request, token = self._refinement_request, self._navigation.token
                jobs = self._refinement_plan()
                if not jobs:
                    break
                results = await run.io_bound(self._compute_refinements, jobs)
                if getattr(self.canvas, "is_deleted", False):
                    break
                if request != self._refinement_request:
                    continue
                if token != self._navigation.token:
                    break
                if results is not None and self._install_refinements(results):
                    self.update()
                break
        finally:
            self._refining = False
            self._refinement_idle.set()

    def _sync_zoom(self) -> None:
        if self.zoom.options is not self.options:
            self.zoom.options.clear()
            self.zoom.options.update(self.options)
        self.zoom.update()

    def open_zoom(self) -> None:
        self._sync_zoom()
        self.zoom_dialog.open()

    def _apply_aspect_ratio(self) -> None:
        control = self.settings.aspect_ratio if self.settings is not None else None
        ratio = (_positive_finite(control.value) or 2.5) if control is not None else None
        if self.options.get("dnascapeCircular"):
            ratio = 1.
        elif self.options.get("dnascapePlotType") == "Heatmap" and self.settings is not None and self.settings.view.value == "Circular":
            ratio = self.settings._cartesian_aspect_ratio
        if self.shell is not None:
            self.shell.style("max-width: 30rem; margin-inline: auto;" if self.options.get("dnascapeCircular")
                             else "max-width: none; margin-inline: 0;")
        self.options["dnascapeAspectRatio"] = ratio
        size = "height: 360px; aspect-ratio: auto; min-height: 0;"
        if ratio is not None:
            size = f"height: auto; aspect-ratio: {ratio}; min-height: 0 !important; flex: none;"
        for chart in (self.canvas, self.vector):
            if hasattr(chart, "style"):
                chart.style("width: 100%; " + size)
        if hasattr(self.vector, "style"):
            self.vector.style("bottom: auto;")
        if self.zoom is not None:
            self.zoom.style(
                "width: 100%; " + size + " flex: none;" if ratio is not None
                else "width: 100%; height: 100%; min-height: 0; aspect-ratio: auto; flex: 1 1 auto;"
            )
        if self.zoom_card is not None:
            self.zoom_card.style("width: min(94vw, 42rem);" if self.options.get("dnascapeCircular")
                                 else "width: min(94vw, 88rem);")
            self.zoom_card.style(
                "height: auto; max-height: 90dvh !important; overflow: auto;" if ratio is not None
                else "height: min(90dvh, 56rem); max-height: none !important; overflow: hidden;"
            )

    async def run_chart_method(
        self,
        name: str,
        *args: Any,
        timeout: float = 1,
    ) -> Any:
        arguments = args[0] if args and isinstance(args[0], Mapping) else {}
        if name.lstrip(":") == "getDataURL" and self._refining:
            await self._refinement_idle.wait()
        vector_capture = (
            name.lstrip(":") == "getDataURL" and str(arguments.get("type", "")).casefold() == "svg"
        )
        capture_key = (
            json.dumps(arguments, sort_keys=True, separators=(",", ":"), default=str)
            if vector_capture
            else None
        )
        if not vector_capture:
            return await self.canvas.run_chart_method(name, *args, timeout=timeout)

        # A double-click (or near-simultaneous SVG and PDF request) must not
        # make the browser build and transfer the same potentially large SVG
        # twice. Recheck the cache inside the lock because another request may
        # have populated it while this coroutine was waiting.
        async with self._vector_capture_lock:
            if (
                self._vector_capture is not None
                and self._vector_capture_revision == self._revision
                and self._vector_capture_key == capture_key
            ):
                return self._vector_capture
            capture_revision = self._revision
            if self._vector_revision != self._revision:
                # The client update serializes the options before ECharts sees
                # them, so recursively copying every plotted point here only
                # duplicates a potentially very large payload in Python. A
                # shallow top-level copy keeps the two component dictionaries
                # independent without copying immutable nested plot data.
                self.vector.options.clear()
                self.vector.options.update(self.options)
                self.vector.update()
                self._vector_revision = self._revision
            result = await self.vector.run_chart_method(name, *args, timeout=timeout)
            if isinstance(result, str) and self._revision == capture_revision:
                # SVG and PDF share the same genuine-vector browser capture.
                # Keep it until the visible chart changes so switching format
                # avoids another render and websocket transfer.
                self._vector_capture = result
                self._vector_capture_revision = capture_revision
                self._vector_capture_key = capture_key
            return result


def histogram_x_limits_from_controls(
    controls: PreviewSettingsControls,
    *,
    log_x: bool = False,
) -> tuple[float, float] | None:
    """Return a complete, valid histogram X interval from shared controls."""

    source = controls._navigation.source_limits.get("x") if "x" in controls._navigation.ranges else None
    lower = _finite_number(source[0] if source is not None else controls.histogram_x_minimum.value)
    upper = _finite_number(source[1] if source is not None else controls.histogram_x_maximum.value)
    if lower is None or upper is None or upper <= lower:
        return None
    if log_x and lower <= 0:
        return None
    return lower, upper


def axis_limits_from_controls(
    controls: PreviewSettingsControls, axis: str,
) -> tuple[float, float] | None:
    """Read an optional complete axis interval from the shared controls."""
    if axis in controls._navigation.ranges and axis in controls._navigation.source_limits:
        return controls._navigation.source_limits[axis]
    if axis in getattr(controls, "_automatic_axes", ()):
        return None
    lower_control = getattr(controls, f"{axis}_minimum", None)
    upper_control = getattr(controls, f"{axis}_maximum", None)
    if lower_control is None or upper_control is None:
        return None
    lower, upper = _finite_number(lower_control.value), _finite_number(upper_control.value)
    return (lower, upper) if lower is not None and upper is not None and upper > lower else None


def sync_preview_axis_limits(controls: PreviewSettingsControls, options: Mapping[str, Any]) -> None:
    """Display the actual automatic bounds without turning them into overrides."""
    object.__setattr__(controls, "_resetting_limits", True)
    try:
        for axis in controls._automatic_axes:
            specification = options.get("xAxis" if axis in {"x", "secondary_x"} else "visualMap", {})
            if isinstance(specification, list):
                index = 1 if axis == "secondary_x" else 0
                specification = specification[index] if len(specification) > index else {}
            elif axis == "secondary_x":
                continue
            if not specification or (axis == "x" and specification.get("type") == "category"):
                continue
            bounds = [_finite_number(specification.get(endpoint)) for endpoint in ("min", "max")]
            if any(value is None for value in bounds):
                continue
            if axis == "z" and bool(controls.log_z.value):
                bounds = [10.0 ** value for value in bounds]
            for suffix, value in zip(("minimum", "maximum"), bounds, strict=True):
                control = getattr(controls, f"{axis}_{suffix}")
                displayed = float(f"{value:.12g}")
                if control.value != displayed:
                    control.value = displayed
                    control.update()
    finally:
        object.__setattr__(controls, "_resetting_limits", False)


def _reset_preview_axis(
    controls: PreviewSettingsControls, axis: str, callback: Callable[[], Any],
) -> None:
    # Suppress intermediate redraws while clearing both endpoints. The frozen
    # control collection owns this internal state; its UI fields stay mutable.
    controls._navigation.clear(axis)
    object.__setattr__(controls, "_resetting_limits", True)
    try:
        if axis in {"x", "z", "secondary_x"}:
            controls._automatic_axes.add(axis)
        for suffix in ("minimum", "maximum"):
            control = getattr(controls, f"{axis}_{suffix}")
            control.value = None
            control.update()
    finally:
        object.__setattr__(controls, "_resetting_limits", False)
    callback()


def _clear_preview_navigation(controls: PreviewSettingsControls) -> None:
    """Restore the unzoomed inputs before changing representation or binning."""
    object.__setattr__(controls, "_resetting_limits", True)
    try:
        for axis, values in controls._navigation.source_limits.items():
            field_axis = "histogram_x" if axis == "x" and controls.view.value == "Histogram" else axis
            for endpoint, value in zip(("minimum", "maximum"), values or (None, None), strict=True):
                control = getattr(controls, f"{field_axis}_{endpoint}", None)
                if control is not None:
                    control.set_value(value)
        controls._navigation.clear()
    finally:
        object.__setattr__(controls, "_resetting_limits", False)


def _report_log_minimum(controls: PreviewSettingsControls, axis: str, minimum: float) -> None:
    error = (axis, minimum)
    if controls._log_limit_error != error:
        notify(f"Log {axis.upper()} minimum must be greater than zero.", type="negative")
    object.__setattr__(controls, "_log_limit_error", error)


def plotting_settings_ready(
    controls: PreviewSettingsControls,
    *,
    plot_type: str,
    allow_empty_y_limits: bool = True,
) -> bool:
    """Whether live plotting fields contain a complete numeric edit.

    Number inputs temporarily become ``None`` while their text is being
    replaced. Treat that state as an unfinished edit so the existing preview
    remains visible and no transient error notification is emitted.
    """

    error = None
    for axis in ("x", "y", *(("z",) if plot_type == "Heatmap" else ())):
        if not getattr(controls, f"log_{axis}").value:
            continue
        if axis in controls._automatic_axes and not (axis == "x" and plot_type == "Histogram"):
            continue
        prefix = "histogram_x" if axis == "x" and plot_type == "Histogram" else axis
        minimum = _finite_number(getattr(controls, f"{prefix}_minimum").value)
        if minimum is not None and minimum <= 0:
            error = (axis, minimum)
            break
    if error is not None:
        _report_log_minimum(controls, *error)
        return False
    object.__setattr__(controls, "_log_limit_error", None)
    if controls.view.value == 'Circular':
        inner = _finite_number(controls.circular_inner_radius.value)
        outer = _finite_number(controls.circular_outer_radius.value)
        if inner is None or outer is None or not 0 <= inner < outer <= 100:
            return False
    if plot_type == "Histogram" and controls.density.value and _positive_finite(controls.kde_bandwidth.value) is None:
        return False
    if plot_type == "Histogram" and (
        _histogram_bin_count(controls.histogram_bins.value) is None
        or _positive_finite(controls.histogram_bin_size.value) is None
        or histogram_x_limits_from_controls(
            controls,
            log_x=bool(controls.log_x.value),
        )
        is None
    ):
        return False
    if plot_type == "Line" and _positive_finite(controls.line_width.value) is None:
        return False
    if controls.aspect_ratio is not None and controls.aspect_ratio.value is not None:
        if _positive_finite(controls.aspect_ratio.value) is None:
            return False
    for axis in (("x", "z") if plot_type == "Heatmap" else ("x",) if plot_type != "Histogram" else ()):
        if axis in getattr(controls, "_automatic_axes", ()):
            continue
        lower = getattr(controls, f"{axis}_minimum", None)
        upper = getattr(controls, f"{axis}_maximum", None)
        if lower is not None and upper is not None and (lower.value is not None or upper.value is not None):
            limits = axis_limits_from_controls(controls, axis)
            if limits is None or (bool(getattr(controls, f"log_{axis}").value) and limits[0] <= 0):
                return False
    y_minimum = controls.y_minimum.value
    y_maximum = controls.y_maximum.value
    if y_minimum is None and y_maximum is None:
        if not allow_empty_y_limits:
            return False
    if (y_minimum is None) != (y_maximum is None):
        return False
    if y_minimum is not None and (
        _finite_number(y_minimum) is None or _finite_number(y_maximum) is None
    ):
        return False
    if y_minimum is not None and float(y_maximum) <= float(y_minimum):
        return False
    dual_y_axes = getattr(controls, "dual_y_axes", None)
    if dual_y_axes is not None and dual_y_axes.visible and dual_y_axes.value:
        for axis in ("secondary_y", "secondary_x"):
            lower, upper = getattr(controls, f"{axis}_minimum").value, getattr(controls, f"{axis}_maximum").value
            if (lower is None) != (upper is None):
                return False
            if lower is not None and (
                _finite_number(lower) is None or _finite_number(upper) is None or float(upper) <= float(lower)
            ):
                return False
    return True


class HistogramBinSynchronizer:
    """Keep histogram range, exact width, and bounded count controls in sync."""

    def __init__(
        self,
        controls: PreviewSettingsControls,
        *,
        values: Callable[[], Any | None],
        on_valid_change: Callable[[], Any],
        log_x: Callable[[], bool] | None = None,
    ) -> None:
        self.controls = controls
        self._values = values
        self._on_valid_change = on_valid_change
        self._log_x = log_x or (lambda: False)
        self._updating = False
        self._x_limits_automatic = True
        self._basis = "bins"
        controls.histogram_bins.on_value_change(self._bins_changed)
        controls.histogram_bin_size.on_value_change(self._bin_size_changed)
        controls.histogram_x_minimum.on_value_change(self._x_limits_changed)
        controls.histogram_x_maximum.on_value_change(self._x_limits_changed)
        controls.reset_histogram_x_limits.on_click(self.reset_x_limits)

    def _x_limits(self) -> tuple[float, float] | None:
        return histogram_x_limits_from_controls(
            self.controls,
            log_x=bool(self._log_x()),
        )

    def _sync_automatic_x_limits(self, values: Any) -> bool:
        if not self._x_limits_automatic:
            minimum = _finite_number(self.controls.histogram_x_minimum.value)
            if self.controls.view.value == "Histogram" and self._log_x() and minimum is not None and minimum <= 0:
                _report_log_minimum(self.controls, "x", minimum)
            return self._x_limits() is not None
        domain = histogram_domain(values, log_x=bool(self._log_x()))
        if domain is None:
            return False
        lower, upper = domain[1]
        self._updating = True
        try:
            self.controls.histogram_x_minimum.value = float(f"{lower:.12g}")
            self.controls.histogram_x_maximum.value = float(f"{upper:.12g}")
            self.controls.histogram_x_minimum.update()
            self.controls.histogram_x_maximum.update()
        finally:
            self._updating = False
        return True

    def sync_from_bins(self, *, redraw: bool = False) -> bool:
        """Refresh displayed bin size from the current count and data."""

        if self._updating or _histogram_bin_count(self.controls.histogram_bins.value) is None:
            return False
        values = self._values()
        if values is None or not self._sync_automatic_x_limits(values):
            return False
        width = approximate_histogram_bin_size(
            values,
            self.controls.histogram_bins.value,
            log_x=bool(self._log_x()),
            x_limits=self._x_limits(),
        )
        if width is None:
            return False
        self._updating = True
        try:
            self.controls.histogram_bin_size.value = width
            self.controls.histogram_bin_size.update()
        finally:
            self._updating = False
        if redraw:
            self._on_valid_change()
        return True

    def sync_from_size(self, *, redraw: bool = False) -> bool:
        """Refresh bin count from the entered width and active X interval."""

        if self._updating:
            return False
        width = _positive_finite(self.controls.histogram_bin_size.value)
        values = self._values()
        if width is None or values is None or not self._sync_automatic_x_limits(values):
            return False
        count = approximate_histogram_bin_count(
            values,
            width,
            log_x=bool(self._log_x()),
            x_limits=self._x_limits(),
        )
        if count is None:
            return False
        span = _histogram_span(
            values,
            log_x=bool(self._log_x()),
            x_limits=self._x_limits(),
        )
        if span is None:
            return False
        safe_width = span / count if span / width > 2_000 else width
        self._updating = True
        try:
            self.controls.histogram_bin_size.value = safe_width
            self.controls.histogram_bins.value = count
            self.controls.histogram_bin_size.update()
            self.controls.histogram_bins.update()
        finally:
            self._updating = False
        if redraw:
            self._on_valid_change()
        return True

    def sync_to_data(self, *, redraw: bool = False) -> bool:
        """Reconcile the pair after the plotted output or domain changes."""

        if self._basis == "size":
            return self.sync_from_size(redraw=redraw)
        return self.sync_from_bins(redraw=redraw)

    def _bins_changed(self) -> None:
        if not self._updating:
            _clear_preview_navigation(self.controls)
            self._basis = "bins"
            self.sync_from_bins(redraw=True)

    def _bin_size_changed(self) -> None:
        if not self._updating:
            _clear_preview_navigation(self.controls)
            self._basis = "size"
            self.sync_from_size(redraw=True)

    def _x_limits_changed(self) -> None:
        if not self._updating and not self.controls._resetting_limits:
            self.controls._navigation.clear("x")
            self._x_limits_automatic = False
            self._basis = "size"
            self.sync_from_size(redraw=True)

    def reset_x_limits(self) -> None:
        """Restore the data-derived X interval and preserve the chosen width."""

        self.controls._navigation.clear("x")
        self._x_limits_automatic = True
        self._basis = "size"
        self.sync_from_size(redraw=True)


def _mark(control: Any, markers: Mapping[str, str], key: str) -> Any:
    marker = markers.get(key)
    return control.mark(marker) if marker else control


def _axis_limit_editor(axis: str, markers: Mapping[str, str]) -> tuple[Any, Any, Any, Any]:
    """Create a matching pair of limits and an always-available reset button."""

    axis_label = {"secondary_y": "Y2", "secondary_x": "X2"}.get(axis, axis.upper())
    with _mark(
        ui.element("div").classes("dn-y-limit-editor"), markers, f"{axis}_limit_editor",
    ) as editor:
        minimum, maximum = (
            _mark(
                ui.number(label=f"{axis_label} {suffix}", value=None, **({"step": 0.1} if axis in {"y", "secondary_y"} else {}))
                .props('dense inputmode=decimal placeholder="Auto"')
                .classes("dn-field dn-field-compact"),
                markers,
                f"{axis}_{endpoint}",
            )
            for suffix, endpoint in (("min", "minimum"), ("max", "maximum"))
        )
        reset = _mark(
            ui.button(icon=RESET_ICON, color=None)
            .props(f'flat round dense aria-label="Reset {axis_label} limits to data range"')
            .classes("dn-reset-button dn-y-limit-reset"),
            markers,
            f"reset_{axis}_limits",
        )
        reset.tooltip("Reset to the data-derived colour range" if axis == "z" else "Reset to the data-derived range")
    return editor, minimum, maximum, reset


_PREVIEW_SETTING_MARKER_SUFFIXES: dict[str, str] = {
    "surface": "plot-settings",
    "container": "preview-settings",
    "view": "view",
    "x_axis_label": "x-axis-label",
    "y_axis_label": "y-axis-label",
    "z_axis_label": "z-axis-label",
    "reverse_order": "reverse-order",
    "circular_options": "circular-options",
    "circular_combine": "circular-combine",
    "circular_inner_radius": "circular-inner-radius",
    "circular_outer_radius": "circular-outer-radius",
    "reset_circular_radii": "reset-circular-radii",
    "primary_color": "primary-color",
    "secondary_color": "secondary-color",
    "heatmap_palette": "heatmap-palette",
    "log_x": "log-x",
    "invert_x": "invert-x",
    "log_y": "log-y",
    "invert_y": "invert-y",
    "log_z": "log-z",
    "invert_z": "invert-z",
    "x_axis_options": "x-axis-options",
    "y_axis_options": "y-axis-options",
    "x_axis_caption": "x-axis-caption",
    "y_axis_caption": "y-axis-caption",
    "secondary_x_axis_options": "x2-axis-options",
    "secondary_y_axis_options": "y2-axis-options",
    "z_axis_options": "z-axis-options",
    "line_width": "line-width",
    "line_type": "curve-type",
    "show_fill": "fill",
    "fill_side": "fill-side",
    "fill_options": "fill-options",
    "dual_y_axes": "dual-y-axes",
    "secondary_y_axis_label": "y2-axis-label",
    "secondary_x_axis_label": "x2-axis-label",
    "secondary_x_limit_editor": "x2-limit-editor",
    "secondary_x_minimum": "x2-minimum",
    "secondary_x_maximum": "x2-maximum",
    "reset_secondary_x_limits": "reset-x2-limits",
    "secondary_log_x": "log-x2",
    "secondary_invert_x": "invert-x2",
    "secondary_log_y": "log-y2",
    "secondary_invert_y": "invert-y2",
    "secondary_y_limit_editor": "y2-limit-editor",
    "secondary_y_minimum": "y2-minimum",
    "secondary_y_maximum": "y2-maximum",
    "reset_secondary_y_limits": "reset-y2-limits",
    "aspect_ratio": "aspect-ratio",
    "reset_aspect_ratio": "reset-aspect-ratio",
    "histogram_bins": "histogram-bins",
    "histogram_bin_size": "histogram-bin-size",
    "histogram_touching": "histogram-touching",
    "histogram_boundary": "histogram-boundary",
    "kde_bandwidth": "kde-smoothness",
    "histogram_grouped": "side-by-side-bins",
    "show_hover": "show-hover",
    "ensemble_profile_options": "ensemble-profile-options",
    "show_individual_profiles": "show-individual-profiles",
    "individual_profile_count": "individual-profile-count",
    "show_standard_deviation": "show-standard-deviation",
    "histogram_x_limit_editor": "histogram-x-limit-editor",
    "histogram_x_minimum": "histogram-x-minimum",
    "histogram_x_maximum": "histogram-x-maximum",
    "reset_histogram_x_limits": "reset-histogram-x-limits",
    "show_histogram_statistics": "show-histogram-statistics",
    "export_histogram_statistics": "export-histogram-statistics",
    "histogram_statistics_options": "histogram-statistics-options",
    "histogram_statistics_position": "histogram-statistics-position",
    "show_legend": "show-legend",
    "legend_name": "legend-name",
    "show_filter": "show-filter",
    "filter_guide_options": "filter-guide-options",
    "filter_name": "filter-name",
    "legend_position": "legend-position",
    "show_grid": "show-grid",
    "density": "density",
    "colorbar": "colorbar",
    "y_minimum": "y-minimum",
    "y_maximum": "y-maximum",
    "reset_y_limits": "reset-y-limits",
    **{key: key.replace("_", "-") for key in (
        "x_limit_editor", "x_minimum", "x_maximum", "reset_x_limits",
        "z_limit_editor", "z_minimum", "z_maximum", "reset_z_limits",
    )},
}


def preview_settings_markers(
    prefix: str,
    *,
    qualifier: str | None = None,
    overrides: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return a complete, consistently named marker map for plot controls.

    Mapping, Simulation, and Plotting render the same shared settings factory.
    Keeping its marker inventory here prevents one page from silently omitting
    a common control from interaction tests or accessibility tooling.  Legacy
    public test markers can be retained with the small ``overrides`` mapping.
    """

    normalized_prefix = str(prefix).strip().strip("-")
    if not normalized_prefix:
        raise ValueError("A plotting-settings marker prefix is required.")
    markers = {
        key: f"{normalized_prefix}-{suffix}"
        for key, suffix in _PREVIEW_SETTING_MARKER_SUFFIXES.items()
    }
    if overrides:
        unknown = set(overrides).difference(markers)
        if unknown:
            names = ", ".join(sorted(unknown))
            raise KeyError(f"Unknown preview-settings marker keys: {names}")
        markers.update({key: str(value) for key, value in overrides.items()})
    normalized_qualifier = str(qualifier or "").strip().strip("-")
    if normalized_qualifier:
        markers = {
            key: f"{marker}-{normalized_qualifier}"
            for key, marker in markers.items()
        }
    return markers


def compact_color_input(
    label: str,
    value: str,
    *,
    marker: str | None = None,
) -> tuple[Any, Any]:
    """Render a labelled color swatch which opens NiceGUI's picker on click."""

    with ui.element("div").classes("dn-compact-color-control") as surface:
        ui.label(label).classes("dn-compact-color-label")
        control = (
            ui.color_input(value=value, preview=True)
            .props(f'dense readonly borderless aria-label="{label}"')
            .classes("dn-compact-color-input")
        )
        if marker:
            control.mark(marker)

        def sync_brand_swatch() -> None:
            control.button.classes(
                add="dn-brand-color-swatch" if str(control.value).casefold() == "#b03412" else None,
                remove=None if str(control.value).casefold() == "#b03412" else "dn-brand-color-swatch",
            )

        control.on_value_change(sync_brand_swatch)

        def sync_picker_color() -> None:
            if str(control.value).casefold() == "#b03412":
                control.picker.set_color(get_export_settings().theme_color)

        control.button.on_click(sync_picker_color)
        sync_brand_swatch()
    return surface, control


def preview_settings_controls(
    *,
    initial_title: str,
    initial_x_label: str,
    initial_y_label: str,
    on_reset_y_limits: Callable[[], Any],
    markers: Mapping[str, str] | None = None,
    allow_dual_y_axes: bool = False,
    on_reset_secondary_y_limits: Callable[[], Any] | None = None,
    allow_filter: bool = False,
    allow_legend_name: bool = False,
    filter_labels: Mapping[str, str] | None = None,
) -> PreviewSettingsControls:
    """Render the common, thematically grouped preview settings panel."""

    from app.pages.common import expansion

    markers = {} if markers is None else markers
    legend_name = show_filter = filter_name = filter_guide_options = None
    filter_guides, filter_guide_names = {}, {}
    with expansion(
        "Plot settings",
        icon="palette",
        extra_classes="dn-preview-settings-section",
    ):
        settings_surface = ui.context.slot.parent.parent_slot.parent
        _mark(settings_surface, markers, "surface")
        settings_grid = ui.element("div").classes(
            "dn-preview-settings-grid dn-preview-settings-main-grid"
        )
        _mark(settings_grid, markers, "container")
        with settings_grid:
            with ui.column().classes("dn-preview-setting-group"):
                ui.label("Style").classes("dn-setting-group-title")
                with ui.element("div").classes("dn-preview-view-grid"):
                    view = _mark(
                        ui.select(
                            ["Default", "Circular", "Histogram"],
                            value="Default",
                            label="View",
                        )
                        .props("dense options-dense")
                        .classes("dn-field"),
                        markers,
                        "view",
                    )
                    with ui.element("div").classes("dn-aspect-ratio-editor"):
                        aspect_ratio = _mark(
                            ui.number(label="Aspect ratio", value=2.5, min=0.2, max=8, step=0.1)
                            .props('dense placeholder="Default"').classes("dn-field dn-field-compact"),
                            markers, "aspect_ratio",
                        )
                        with aspect_ratio.add_slot("append"):
                            reset_aspect_ratio = _mark(
                                ui.button(icon=RESET_ICON, color=None).props(
                                    'flat round dense aria-label="Reset aspect ratio to default"'
                                ).classes("dn-reset-button dn-y-limit-reset"), markers, "reset_aspect_ratio",
                            )
                        reset_aspect_ratio.tooltip("Reset to the default aspect ratio")
                title = _PlotTitle(initial_title)
                with ui.column().classes("w-full").mark(markers.get("circular_options", "circular-options")) as circular_options:
                    with ui.element("div").classes("dn-preview-histogram-grid"):
                        circular_outer_radius = _mark(
                            ui.number(label="Outer radius (%)", value=DEFAULT_OUTER_RADIUS, min=1, max=100, step=1)
                            .props("dense").classes("dn-field dn-field-compact"), markers, "circular_outer_radius",
                        )
                        circular_inner_radius = _mark(
                            ui.number(label="Inner radius (%)", value=DEFAULT_INNER_RADIUS, min=0, max=99, step=1)
                            .props("dense").classes("dn-field dn-field-compact"), markers, "circular_inner_radius",
                        )
                        with circular_inner_radius.add_slot('append'):
                            reset_circular_radii = _mark(
                                ui.button(icon=RESET_ICON, color=None).props(
                                    'flat round dense aria-label="Reset both circular radii to defaults"'
                                ).classes('dn-reset-button dn-y-limit-reset'), markers, 'reset_circular_radii',
                            )
                        reset_circular_radii.tooltip('Reset both radii')
                    circular_combine = _mark(ui.checkbox("Combine", value=False), markers, "circular_combine")
                    field_hint(circular_combine, "Overlay datasets on the same circular track.")
                circular_options.visible = False
                circular_outer_radius.visible = False
                circular_inner_radius.visible = False
                circular_combine.visible = False
                with ui.element("div").classes("dn-preview-histogram-grid"):
                    histogram_bins = _mark(
                        ui.number(
                            label="Number of bins",
                            value=60,
                            min=1,
                            max=2000,
                            precision=0,
                            step=1,
                        )
                        .props("dense")
                        .classes("dn-field dn-field-compact"),
                        markers,
                        "histogram_bins",
                    )
                    histogram_bin_size = _mark(
                        ui.number(
                            label="Bin size",
                            value=None,
                        )
                        .props("dense inputmode=decimal")
                        .classes("dn-field dn-field-compact"),
                        markers,
                        "histogram_bin_size",
                    )
                histogram_boundary = _mark(
                    ui.select(
                        {
                            "left": "Left ≤ value < right",
                            "right": "Left < value ≤ right",
                        },
                        value="left",
                        label="Bin boundaries",
                    )
                    .props("dense options-dense")
                    .classes("dn-field"),
                    markers,
                    "histogram_boundary",
                )
                with _mark(
                    ui.column().classes("dn-preview-ensemble-options"),
                    markers,
                    "ensemble_profile_options",
                ) as ensemble_profile_options:
                    ui.label("Ensemble").classes("dn-setting-subgroup-title")
                    with ui.element("div").classes("dn-inline-actions"):
                        show_individual_profiles = _mark(
                            ui.checkbox("Individual profiles", value=False),
                            markers,
                            "show_individual_profiles",
                        )
                        show_standard_deviation = _mark(
                            ui.checkbox("Mean ± standard deviation", value=False),
                            markers,
                            "show_standard_deviation",
                        )
                    individual_profile_count = _mark(
                        ui.number(
                            label="Profiles shown",
                            value=10,
                            min=1,
                            max=50,
                            precision=0,
                            step=1,
                        )
                        .props("dense")
                        .classes("dn-field dn-field-compact"),
                        markers,
                        "individual_profile_count",
                    )

                with ui.element("div").classes("dn-preview-colour-grid"):
                    primary_color_surface, primary_color = compact_color_input(
                        "Color",
                        "#b03412",
                        marker=markers.get("primary_color"),
                    )
                    line_width = _mark(
                        ui.number(
                            label="Thickness",
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
                    line_type = _mark(
                        ui.select(CURVE_TYPES, value="solid", label="Type")
                        .props("dense options-dense").classes("dn-field"), markers, "line_type",
                    )
                    secondary_color_surface, secondary_color = compact_color_input(
                        "Second color",
                        "#2596be",
                        marker=markers.get("secondary_color"),
                    )
                    secondary_color_surface.classes("dn-preview-secondary-color")
                secondary_color_surface.visible = False
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
                with _mark(
                    ui.element("div").classes("dn-preview-fill-controls"), markers, "fill_options",
                ) as fill_options:
                    show_fill = _mark(ui.checkbox("Fill", value=False), markers, "show_fill")
                    fill_side = _mark(
                        ui.select({"below": "Below", "above": "Above"}, value="below", label="Side")
                        .props("dense options-dense").classes("dn-field"), markers, "fill_side",
                    )
                reverse_order = _mark(ui.checkbox("Reverse order", value=False), markers, "reverse_order")
                with ui.element("div").classes("dn-kde-controls"):
                    density = _mark(
                        ui.checkbox("KDE", value=False),
                        markers,
                        "density",
                    )
                    kde_bandwidth = _mark(
                        ui.number("Smoothness", value=1.0, min=0.05, max=20, step=0.1)
                        .props("dense inputmode=decimal").classes("dn-field dn-field-compact"),
                        markers, "kde_bandwidth",
                    )
                with ui.element("div").classes("dn-inline-actions"):
                    histogram_touching = _mark(
                        ui.checkbox("Touching bins", value=True),
                        markers,
                        "histogram_touching",
                    )
                    histogram_grouped = _mark(
                        ui.checkbox("Side by side", value=False), markers, "histogram_grouped",
                    )

            with ui.column().classes("dn-preview-setting-group"):
                ui.label("Axes").classes("dn-setting-group-title")
                x_axis_label = _mark(
                    ui.input(label="X-axis label", value=initial_x_label)
                    .props("dense clearable")
                    .classes("dn-field"),
                    markers,
                    "x_axis_label",
                )
                y_axis_label = _mark(
                    ui.input(label="Y-axis label", value=initial_y_label)
                    .props("dense clearable")
                    .classes("dn-field"),
                    markers,
                    "y_axis_label",
                )
                z_axis_label = _mark(
                    ui.input(label="Z-axis label", value="").props("dense").classes("dn-field"),
                    markers, "z_axis_label",
                )
                z_axis_label.visible = False
                with ui.element("div").classes("dn-axis-option-grid"):
                    with _mark(ui.element("div"), markers, "x_axis_options").classes("dn-axis-option-row") as x_axis_options:
                        x_axis_caption = _mark(ui.label("X"), markers, "x_axis_caption").classes("dn-axis-option-label")
                        log_x = _mark(ui.checkbox("Log", value=False), markers, "log_x")
                        invert_x = _mark(
                            ui.checkbox("Invert", value=False),
                            markers,
                            "invert_x",
                        )
                    with _mark(ui.element("div"), markers, "y_axis_options").classes("dn-axis-option-row") as y_axis_options:
                        y_axis_caption = _mark(ui.label("Y"), markers, "y_axis_caption").classes("dn-axis-option-label")
                        log_y = _mark(ui.checkbox("Log", value=False), markers, "log_y")
                        invert_y = _mark(
                            ui.checkbox("Invert", value=False),
                            markers,
                            "invert_y",
                        )
                    with _mark(ui.element("div"), markers, "z_axis_options").classes("dn-axis-option-row") as z_axis_options:
                        ui.label("Z").classes("dn-axis-option-label")
                        log_z = _mark(ui.checkbox("Log", value=False), markers, "log_z")
                        invert_z = _mark(
                            ui.checkbox("Invert", value=False),
                            markers,
                            "invert_z",
                        )
                with ui.element("div").classes("dn-y-limit-editor") as histogram_x_limit_editor:
                    histogram_x_minimum = _mark(
                        ui.number(label="X min", value=None)
                        .props("dense inputmode=decimal")
                        .classes("dn-field dn-field-compact"),
                        markers,
                        "histogram_x_minimum",
                    )
                    histogram_x_maximum = _mark(
                        ui.number(label="X max", value=None)
                        .props("dense inputmode=decimal")
                        .classes("dn-field dn-field-compact"),
                        markers,
                        "histogram_x_maximum",
                    )
                    reset_histogram_x_limits = _mark(
                        ui.button(icon=RESET_ICON, color=None)
                        .props('flat round dense aria-label="Reset histogram X limits"')
                        .classes("dn-reset-button dn-y-limit-reset"),
                        markers,
                        "reset_histogram_x_limits",
                    )
                    reset_histogram_x_limits.tooltip("Reset to the data range")
                x_limit_editor, x_minimum, x_maximum, reset_x_limits = _axis_limit_editor("x", markers)
                _, y_minimum, y_maximum, reset_y_limits = _axis_limit_editor("y", markers)
                z_limit_editor, z_minimum, z_maximum, reset_z_limits = _axis_limit_editor("z", markers)
                dual_y_axes = secondary_y_axis_label = secondary_y_limit_editor = None
                secondary_y_minimum = secondary_y_maximum = reset_secondary_y_limits = None
                secondary_x_axis_label = secondary_x_limit_editor = None
                secondary_x_minimum = secondary_x_maximum = reset_secondary_x_limits = None
                secondary_x_axis_options = secondary_y_axis_options = None
                secondary_log_x = secondary_invert_x = secondary_log_y = secondary_invert_y = None
                if allow_dual_y_axes:
                    dual_y_axes = _mark(ui.switch("Two Y axes", value=False), markers, "dual_y_axes")
                    dual_y_axes.visible = False
                    secondary_y_axis_label = _mark(
                        ui.input(label="Y2 label", value="Value").props("dense clearable").classes("dn-field"),
                        markers, "secondary_y_axis_label",
                    )
                    secondary_y_limit_editor, secondary_y_minimum, secondary_y_maximum, reset_secondary_y_limits = _axis_limit_editor("secondary_y", markers)
                    secondary_y_axis_label.visible = secondary_y_limit_editor.visible = False
                    secondary_x_axis_label = _mark(
                        ui.input(label="X2 label", value=initial_x_label).props("dense clearable").classes("dn-field"),
                        markers, "secondary_x_axis_label",
                    )
                    secondary_x_limit_editor, secondary_x_minimum, secondary_x_maximum, reset_secondary_x_limits = _axis_limit_editor("secondary_x", markers)
                    secondary_x_axis_label.visible = secondary_x_limit_editor.visible = False
                    with _mark(ui.element("div"), markers, "secondary_x_axis_options").classes("dn-axis-option-row") as secondary_x_axis_options:
                        ui.label("X2").classes("dn-axis-option-label")
                        secondary_log_x = _mark(ui.checkbox("Log", value=False), markers, "secondary_log_x")
                        secondary_invert_x = _mark(ui.checkbox("Invert", value=False), markers, "secondary_invert_x")
                    with _mark(ui.element("div"), markers, "secondary_y_axis_options").classes("dn-axis-option-row") as secondary_y_axis_options:
                        ui.label("Y2").classes("dn-axis-option-label")
                        secondary_log_y = _mark(ui.checkbox("Log", value=False), markers, "secondary_log_y")
                        secondary_invert_y = _mark(ui.checkbox("Invert", value=False), markers, "secondary_invert_y")
                    secondary_x_axis_options.visible = secondary_y_axis_options.visible = False

            with ui.column().classes("dn-preview-setting-group"):
                ui.label("Guides").classes("dn-setting-group-title")
                with ui.element("div").classes("dn-inline-actions"):
                    show_legend = _mark(
                        ui.checkbox("Legend", value=True),
                        markers,
                        "show_legend",
                    )
                    with ui.element("div").classes("dn-guide-pair"):
                        show_grid = _mark(ui.checkbox("Grid", value=False), markers, "show_grid")
                        show_hover = _mark(ui.checkbox("Hover", value=True), markers, "show_hover")
                    colorbar = _mark(
                        ui.checkbox("Colorbar", value=True), markers, "colorbar",
                    )
                with _mark(ui.element("div"), markers, "histogram_statistics_options").classes("dn-inline-actions") as histogram_statistics_options:
                    show_histogram_statistics = _mark(
                        ui.checkbox("Statistics", value=True), markers, "show_histogram_statistics",
                    )
                    export_histogram_statistics = _mark(
                        ui.checkbox("Export statistics", value=False), markers, "export_histogram_statistics",
                    )
                if allow_filter:
                    with ui.element("div").classes("dn-inline-actions") as filter_guide_options:
                        _mark(filter_guide_options, markers, "filter_guide_options")
                        show_filter = _mark(ui.checkbox("Mask", value=False), markers, "show_filter")
                if allow_legend_name:
                    legend_name = _mark(
                        ui.input("Legend name", value="").props('dense placeholder="Automatic"').classes("dn-field"),
                        markers, "legend_name",
                    )
                if allow_filter:
                    filter_name = _mark(
                        ui.input("Mask name", value="Mask").props("dense").classes("dn-field"),
                        markers, "filter_name",
                    )
                if filter_labels:
                    with ui.row().classes('dn-inline-actions') as filter_guide_options:
                        for key, label in filter_labels.items():
                            prefix = markers.get('container', 'preview-settings')
                            filter_guides[key] = ui.checkbox(label, value=False).mark(f'{prefix}-{key.replace("_", "-")}-filter')
                    for key, label in filter_labels.items():
                        filter_guide_names[key] = ui.input(f'{label} name', value=label).props('dense').classes('dn-field').mark(f'{prefix}-{key.replace("_", "-")}-filter-name')
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
                histogram_statistics_position = _mark(
                    ui.select(
                        ["Top right", "Top left", "Bottom right", "Bottom left"],
                        value="Top right",
                        label="Statistics position",
                    )
                    .props("dense options-dense")
                    .classes("dn-field"),
                    markers,
                    "histogram_statistics_position",
                )

    controls = PreviewSettingsControls(
        view=view,
        title=title,
        x_axis_label=x_axis_label,
        y_axis_label=y_axis_label,
        primary_color=primary_color,
        primary_color_surface=primary_color_surface,
        secondary_color=secondary_color,
        secondary_color_surface=secondary_color_surface,
        heatmap_palette=heatmap_palette,
        log_x=log_x,
        invert_x=invert_x,
        log_z=log_z,
        invert_z=invert_z,
        x_axis_options=x_axis_options,
        y_axis_options=y_axis_options,
        z_axis_options=z_axis_options,
        line_width=line_width,
        histogram_bins=histogram_bins,
        histogram_bin_size=histogram_bin_size,
        histogram_touching=histogram_touching,
        histogram_boundary=histogram_boundary,
        kde_bandwidth=kde_bandwidth,
        histogram_grouped=histogram_grouped,
        show_hover=show_hover,
        ensemble_profile_options=ensemble_profile_options,
        show_individual_profiles=show_individual_profiles,
        individual_profile_count=individual_profile_count,
        show_standard_deviation=show_standard_deviation,
        histogram_x_limit_editor=histogram_x_limit_editor,
        histogram_x_minimum=histogram_x_minimum,
        histogram_x_maximum=histogram_x_maximum,
        reset_histogram_x_limits=reset_histogram_x_limits,
        show_histogram_statistics=show_histogram_statistics,
        export_histogram_statistics=export_histogram_statistics,
        histogram_statistics_options=histogram_statistics_options,
        histogram_statistics_position=histogram_statistics_position,
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
        x_limit_editor=x_limit_editor,
        x_minimum=x_minimum,
        x_maximum=x_maximum,
        reset_x_limits=reset_x_limits,
        z_limit_editor=z_limit_editor,
        z_minimum=z_minimum,
        z_maximum=z_maximum,
        reset_z_limits=reset_z_limits,
        aspect_ratio=aspect_ratio,
        reset_aspect_ratio=reset_aspect_ratio,
        line_type=line_type,
        show_fill=show_fill,
        fill_side=fill_side,
        fill_options=fill_options,
        reverse_order=reverse_order,
        circular_combine=circular_combine,
        circular_inner_radius=circular_inner_radius,
        circular_options=circular_options,
        circular_outer_radius=circular_outer_radius,
        reset_circular_radii=reset_circular_radii,
        z_axis_label=z_axis_label,
        surface=settings_surface,
        legend_name=legend_name,
        show_filter=show_filter,
        filter_guide_options=filter_guide_options,
        filter_name=filter_name,
        filter_guides=filter_guides,
        filter_guide_names=filter_guide_names,
        dual_y_axes=dual_y_axes,
        secondary_y_axis_label=secondary_y_axis_label,
        secondary_y_limit_editor=secondary_y_limit_editor,
        secondary_y_minimum=secondary_y_minimum,
        secondary_y_maximum=secondary_y_maximum,
        reset_secondary_y_limits=reset_secondary_y_limits,
        x_axis_caption=x_axis_caption,
        y_axis_caption=y_axis_caption,
        secondary_x_axis_label=secondary_x_axis_label,
        secondary_x_limit_editor=secondary_x_limit_editor,
        secondary_x_minimum=secondary_x_minimum,
        secondary_x_maximum=secondary_x_maximum,
        reset_secondary_x_limits=reset_secondary_x_limits,
        secondary_x_axis_options=secondary_x_axis_options,
        secondary_y_axis_options=secondary_y_axis_options,
        secondary_log_x=secondary_log_x,
        secondary_invert_x=secondary_invert_x,
        secondary_log_y=secondary_log_y,
        secondary_invert_y=secondary_invert_y,
    )
    reset_y_limits.on_click(lambda: _reset_preview_axis(controls, "y", on_reset_y_limits))
    if reset_secondary_y_limits is not None:
        controls._automatic_axes.add("secondary_x")
        reset_secondary_y_limits.on_click(lambda: _reset_preview_axis(
            controls, "secondary_y", on_reset_secondary_y_limits or (lambda: None),
        ))
    return controls


def bind_preview_settings_updates(
    controls: PreviewSettingsControls,
    *,
    on_redraw: Callable[[], Any],
    on_view_change: Callable[[], Any],
    on_density_change: Callable[[], Any],
    on_histogram_range_change: Callable[[], Any],
    on_log_x_change: Callable[[], Any],
    on_y_limits_change: Callable[[], Any],
    on_visibility_change: Callable[[], Any],
    on_line_type_change: Callable[[], Any] | None = None,
    default_aspect_ratio: float | None = 2.5,
) -> None:
    """Wire the common plotting settings with identical live behavior.

    The callbacks which derive axes or available representations remain owned
    by each workflow, but every shared field redraws and refreshes visibility
    in the same order. Context-specific controls can be wired by the caller
    after this helper.
    """

    for control in (
        controls.x_axis_label,
        controls.y_axis_label,
        controls.primary_color,
        controls.secondary_color,
        controls.heatmap_palette,
        controls.invert_x,
        controls.log_z,
        controls.invert_z,
        controls.line_width,
        controls.aspect_ratio,
        controls.fill_side,
        controls.reverse_order,
        controls.circular_combine,
        controls.z_axis_label,
        controls.histogram_touching,
        controls.histogram_grouped,
        controls.show_hover,
        controls.histogram_statistics_position,
        controls.legend_position,
        controls.show_grid,
        controls.colorbar,
        controls.invert_y,
    ):
        control.on_value_change(on_redraw)
    for control in (controls.legend_name, controls.filter_name):
        if control is not None:
            control.on_value_change(on_redraw)
    for control in controls.filter_guide_names.values():
        control.on_value_change(on_redraw)
    def change_domain(callback: Callable[[], Any]) -> None:
        _clear_preview_navigation(controls)
        callback()

    controls.histogram_boundary.on_value_change(lambda: change_domain(on_histogram_range_change))
    controls.line_type.on_value_change(on_line_type_change or on_redraw)
    controls.reset_aspect_ratio.on_click(lambda: controls.aspect_ratio.set_value(default_aspect_ratio))
    def circular_radii_changed() -> None:
        if controls._resetting_circular_radii:
            return
        inner = _finite_number(controls.circular_inner_radius.value)
        outer = _finite_number(controls.circular_outer_radius.value)
        if inner is not None and outer is not None and 0 <= inner < outer <= 100:
            on_redraw()

    def reset_circular_radii() -> None:
        object.__setattr__(controls, '_resetting_circular_radii', True)
        try:
            controls.circular_outer_radius.set_value(DEFAULT_OUTER_RADIUS)
            controls.circular_inner_radius.set_value(DEFAULT_INNER_RADIUS)
        finally:
            object.__setattr__(controls, '_resetting_circular_radii', False)
        on_redraw()

    controls.circular_inner_radius.on_value_change(circular_radii_changed)
    controls.circular_outer_radius.on_value_change(circular_radii_changed)
    controls.reset_circular_radii.on_click(reset_circular_radii)
    def refresh_visibility_and_plot() -> None:
        # One event should produce one redraw.  This matters for a dense
        # heatmap, where duplicate browser updates are visibly expensive.
        on_visibility_change()
        on_redraw()

    controls.show_histogram_statistics.on_value_change(refresh_visibility_and_plot)
    controls.export_histogram_statistics.on_value_change(lambda: refresh_statistics_exports(controls))
    controls.show_legend.on_value_change(refresh_visibility_and_plot)
    controls.show_fill.on_value_change(refresh_visibility_and_plot)
    if controls.show_filter is not None:
        controls.show_filter.on_value_change(refresh_visibility_and_plot)
    for control in controls.filter_guides.values():
        control.on_value_change(refresh_visibility_and_plot)
    controls.view.on_value_change(lambda: change_domain(on_view_change))
    controls.density.on_value_change(lambda: change_domain(on_density_change))
    controls.kde_bandwidth.on_value_change(lambda: change_domain(on_histogram_range_change))
    controls.log_x.on_value_change(lambda: change_domain(on_log_x_change))
    controls.log_y.on_value_change(lambda: change_domain(on_histogram_range_change))
    for axis in ("x", "z", *(("secondary_x",) if controls.secondary_x_minimum is not None else ())):
        def limits_changed(_event: Any, selected: str = axis) -> None:
            if controls._resetting_limits:
                return
            controls._navigation.clear(selected)
            minimum = getattr(controls, f"{selected}_minimum").value
            maximum = getattr(controls, f"{selected}_maximum").value
            if minimum is None and maximum is None:
                controls._automatic_axes.add(selected)
            else:
                controls._automatic_axes.discard(selected)
            on_redraw()

        for suffix in ("minimum", "maximum"):
            getattr(controls, f"{axis}_{suffix}").on_value_change(limits_changed)
        getattr(controls, f"reset_{axis}_limits").on_click(
            lambda _event, selected=axis: _reset_preview_axis(controls, selected, on_redraw),
        )
    def y_limits_changed() -> None:
        if not controls._resetting_limits:
            controls._navigation.clear("y")
            on_y_limits_change()

    controls.y_minimum.on_value_change(y_limits_changed)
    controls.y_maximum.on_value_change(y_limits_changed)
    if controls.secondary_y_minimum is not None:
        def secondary_y_edited() -> None:
            if not controls._resetting_limits:
                controls._navigation.clear("secondary_y")

        controls.secondary_y_minimum.on_value_change(secondary_y_edited)
        controls.secondary_y_maximum.on_value_change(secondary_y_edited)


def sync_preview_settings_visibility(
    controls: PreviewSettingsControls,
    *,
    plot_type: str,
    multiple_curves: bool = False,
    allow_histogram: bool = True,
    allow_boxplot: bool = False,
    allow_circular: bool = True,
    allow_individual_profiles: bool = False,
    allow_standard_deviation: bool = False,
    filter_available: bool = False,
    available_filters: tuple[str, ...] = (),
) -> None:
    """Show only controls that apply to the current plot representation."""

    heatmap = plot_type == "Heatmap"
    histogram = plot_type == "Histogram"
    boxplot = plot_type == "Boxplot"
    line = plot_type in {"Line", "Circular"}
    view_options = ["Default"]
    if not heatmap and allow_circular:
        view_options.append("Circular")
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
    controls.z_axis_label.visible = heatmap
    controls.reverse_order.visible = not heatmap
    circular = not heatmap and controls.view.value == "Circular"
    controls.circular_options.visible = circular
    controls.circular_combine.visible = circular and multiple_curves
    controls.circular_inner_radius.visible = circular
    controls.circular_outer_radius.visible = circular
    if circular and controls.aspect_ratio.enabled:
        object.__setattr__(controls, "_cartesian_aspect_ratio", _positive_finite(controls.aspect_ratio.value) or 2.5)
        controls.aspect_ratio.disable()
        controls.reset_aspect_ratio.disable()
        controls.aspect_ratio.set_value(1.)
    elif not circular and not controls.aspect_ratio.enabled:
        controls.aspect_ratio.enable()
        controls.reset_aspect_ratio.enable()
        controls.aspect_ratio.set_value(controls._cartesian_aspect_ratio)
    controls.z_axis_options.visible = heatmap
    controls.x_axis_options.visible = not boxplot
    controls.y_axis_options.visible = True
    controls.colorbar.visible = heatmap
    kde = histogram and bool(controls.density.value)
    controls.histogram_bins.visible = histogram and not kde
    controls.histogram_bin_size.visible = histogram and not kde
    controls.histogram_touching.visible = histogram and not kde
    controls.histogram_boundary.visible = histogram and not kde
    controls.kde_bandwidth.visible = kde
    controls.histogram_grouped.visible = histogram and not kde and multiple_curves
    controls.histogram_x_limit_editor.visible = histogram
    controls.x_limit_editor.visible = not histogram
    controls.z_limit_editor.visible = heatmap
    controls.density.visible = histogram
    controls.show_histogram_statistics.visible = histogram
    controls.export_histogram_statistics.visible = histogram
    controls.histogram_statistics_options.visible = histogram
    controls.histogram_statistics_position.visible = False
    individual_profiles = line and allow_individual_profiles
    standard_deviation = line and allow_standard_deviation
    ensemble_profiles = individual_profiles or standard_deviation
    controls.ensemble_profile_options.visible = ensemble_profiles
    controls.show_individual_profiles.visible = individual_profiles
    controls.show_standard_deviation.visible = standard_deviation
    controls.individual_profile_count.visible = individual_profiles and bool(
        controls.show_individual_profiles.value
    )
    controls.primary_color_surface.visible = not heatmap
    controls.secondary_color_surface.visible = line and multiple_curves
    controls.line_width.visible = line or kde
    controls.line_type.visible = line or kde
    controls.fill_options.visible = line or kde
    controls.fill_side.visible = (line or kde) and bool(controls.show_fill.value)
    legend_available = line or histogram or (boxplot and multiple_curves) or (heatmap and (filter_available or bool(available_filters)))
    controls.show_legend.visible = legend_available
    if controls.legend_name is not None:
        controls.legend_name.visible = legend_available and bool(controls.show_legend.value)
    if controls.show_filter is not None:
        controls.show_filter.visible = (line or heatmap) and filter_available
        controls.show_filter.set_enabled(filter_available)
        if getattr(controls, "filter_guide_options", None) is not None:
            controls.filter_guide_options.visible = (line or heatmap) and filter_available
        controls.filter_name.visible = (
            (line or heatmap) and filter_available and bool(controls.show_filter.value) and bool(controls.show_legend.value)
        )
    for key, control in controls.filter_guides.items():
        available = (line or heatmap) and key in available_filters
        control.set_visibility(available)
        control.set_enabled(available)
        controls.filter_guide_names[key].set_visibility(available and bool(control.value) and bool(controls.show_legend.value))
    if controls.filter_guides and controls.filter_guide_options is not None:
        controls.filter_guide_options.set_visibility((line or heatmap) and any(key in available_filters for key in controls.filter_guides))
    controls.legend_position.visible = legend_available and bool(
        controls.show_legend.value or (histogram and controls.show_histogram_statistics.value)
    )
    controls.legend_position.props(f'label="{"Panel position" if histogram and controls.show_histogram_statistics.value else "Legend position"}"')
    refresh_statistics_exports(controls)


def render_preview_chart(
    options: dict[str, Any],
    *,
    title: str,
    marker: str,
    on_download: Callable[[str], Any],
) -> PreviewChartHandle:
    """Render a responsive preview plus its exact, export-only SVG twin."""

    from app.pages.common import plot_toolbar

    with ui.dialog() as zoom_dialog, ui.card().classes("dn-plot-dialog") as zoom_card:
        with ui.row().classes("dn-plot-dialog-header"):
            ui.label(title).classes("dn-panel-title")
            ui.space()
            ui.button(icon="close", on_click=zoom_dialog.close).props(
                'flat round dense aria-label="Close enlarged plot"'
            )
        zoom_chart = ui.echart(dict(options), renderer="canvas").classes("dn-chart-wrap-large dn-plot-navigation").style(
            "width: 100%; min-height: 0; height: 100%;"
        )
    with ui.element("div").classes("dn-plot-shell") as shell:
        chart = (
            ui.echart(options, renderer="canvas").classes("dn-plot-navigation").style("width: 100%; height: 360px;").mark(marker)
        )
        vector_chart = (
            ui.echart({"animation": False}, renderer="svg")
            .props('aria-hidden="true"')
            .classes("dn-chart-vector-export")
            .style(
                "position: absolute; inset: 0; width: 100%; height: 360px; "
                "opacity: 0; pointer-events: none;"
            )
        )
        plot_toolbar(on_zoom=lambda: handle.open_zoom(), on_download=on_download)
    handle = PreviewChartHandle(canvas=chart, vector=vector_chart, zoom=zoom_chart, zoom_card=zoom_card,
                                zoom_dialog=zoom_dialog, shell=shell, source_options=options)
    chart._dn_preview_handle = handle
    handle._apply_navigation()
    handle._position_toolbar()
    handle._sync_canvas_options()
    zoom_chart.options.update(chart.options)
    for surface in (chart, zoom_chart):
        surface.on("plotViewport", lambda event: handle.accept_navigation_async(event.args))
        surface.on("plotDataset", lambda event: handle.select_histogram_dataset(event.args))
    chart.on("brandTheme", handle.invalidate_export)
    return handle


def refresh_preview_points() -> None:
    """Apply the global display budget to existing charts without losing viewports."""
    for element in list(ui.context.client.elements.values()):
        handle = getattr(element, "_dn_preview_handle", None)
        if handle is not None and not element.is_deleted and handle.options.get("dnascapePlotType") in {"Line", "Histogram"}:
            handle._refine_data(force=True)
            handle.update()
