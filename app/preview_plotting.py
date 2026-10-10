"""Shared preview plotting for Mapping, Simulation, and Plotting workflows."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from nicegui import ui

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
    _TOOLTIP_VALUE_FORMATTER as _TOOLTIP_VALUE_FORMATTER,
    _TOOLTIP_POINTER_FORMATTER as _TOOLTIP_POINTER_FORMATTER,
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
    _histogram_statistics_graphic as _histogram_statistics_graphic,
    _palette_rgb as _palette_rgb,
    _heatmap_data_uri as _heatmap_data_uri,
    automatic_y_limits as automatic_y_limits,
    _bounded_indices as _bounded_indices,
    _line_points as _line_points,
    _band_segments as _band_segments,
    _sample_heatmap as _sample_heatmap,
    _heatmap_tooltip_data as _heatmap_tooltip_data,
    build_preview_plot_options as build_preview_plot_options,
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
    dual_y_axes: Any = None
    secondary_y_axis_label: Any = None
    secondary_y_limit_editor: Any = None
    secondary_y_minimum: Any = None
    secondary_y_maximum: Any = None
    reset_secondary_y_limits: Any = None
    _resetting_limits: bool = field(default=False, init=False, repr=False)
    _automatic_axes: set[str] = field(default_factory=lambda: {"x", "z"}, init=False, repr=False)


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
    _revision: int = field(default=0, init=False, repr=False)
    _vector_revision: int = field(default=-1, init=False, repr=False)
    _vector_capture_revision: int = field(default=-1, init=False, repr=False)
    _vector_capture_key: str | None = field(default=None, init=False, repr=False)
    _vector_capture: str | None = field(default=None, init=False, repr=False)
    _vector_capture_lock: asyncio.Lock = field(
        default_factory=asyncio.Lock,
        init=False,
        repr=False,
    )

    @property
    def options(self) -> dict[str, Any]:
        """Expose the visible chart options for existing live-update callers."""

        return self.canvas.options

    def update(self) -> None:
        """Update the interactive chart; the vector twin stays lazy."""

        if self.settings is not None:
            sync_preview_axis_limits(self.settings, self.options)
            self._apply_aspect_ratio()
        self.canvas.update()
        self.invalidate_export()

    def invalidate_export(self) -> None:
        """Discard SVG captures after a data, style, or browser theme change."""

        self._revision += 1
        self._vector_capture = None
        self._vector_capture_revision = -1
        self._vector_capture_key = None

    def attach_settings(self, controls: PreviewSettingsControls) -> None:
        self.settings = controls
        sync_preview_axis_limits(controls, self.options)
        self._apply_aspect_ratio()

    def _apply_aspect_ratio(self) -> None:
        control = self.settings.aspect_ratio if self.settings is not None else None
        ratio = _positive_finite(control.value) if control is not None else None
        self.options["dnascapeAspectRatio"] = ratio
        size = "height: 360px; aspect-ratio: auto; min-height: 0;"
        if ratio is not None:
            size = f"height: auto; aspect-ratio: {ratio}; min-height: 240px;"
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
                self.vector.options.update(self.canvas.options)
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

    lower = _finite_number(controls.histogram_x_minimum.value)
    upper = _finite_number(controls.histogram_x_maximum.value)
    if lower is None or upper is None or upper <= lower:
        return None
    if log_x and lower <= 0:
        return None
    return lower, upper


def axis_limits_from_controls(
    controls: PreviewSettingsControls, axis: str,
) -> tuple[float, float] | None:
    """Read an optional complete axis interval from the shared controls."""
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
            specification = options.get("xAxis" if axis == "x" else "visualMap", {})
            if isinstance(specification, list):
                specification = next(iter(specification), {})
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
    object.__setattr__(controls, "_resetting_limits", True)
    try:
        if axis in {"x", "z"}:
            controls._automatic_axes.add(axis)
        for suffix in ("minimum", "maximum"):
            control = getattr(controls, f"{axis}_{suffix}")
            control.value = None
            control.update()
    finally:
        object.__setattr__(controls, "_resetting_limits", False)
    callback()


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
        lower, upper = controls.secondary_y_minimum.value, controls.secondary_y_maximum.value
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
            self._basis = "bins"
            self.sync_from_bins(redraw=True)

    def _bin_size_changed(self) -> None:
        if not self._updating:
            self._basis = "size"
            self.sync_from_size(redraw=True)

    def _x_limits_changed(self) -> None:
        if not self._updating:
            self._x_limits_automatic = False
            self._basis = "size"
            self.sync_from_size(redraw=True)

    def reset_x_limits(self) -> None:
        """Restore the data-derived X interval and preserve the chosen width."""

        self._x_limits_automatic = True
        self._basis = "size"
        self.sync_from_size(redraw=True)


def _mark(control: Any, markers: Mapping[str, str], key: str) -> Any:
    marker = markers.get(key)
    return control.mark(marker) if marker else control


def _axis_limit_editor(axis: str, markers: Mapping[str, str]) -> tuple[Any, Any, Any, Any]:
    """Create a matching pair of limits and an always-available reset button."""

    axis_label = "Y2" if axis == "secondary_y" else axis.upper()
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
            ui.button(icon="restart_alt")
            .props(f'flat round dense aria-label="Reset {axis_label} limits to data range"')
            .classes("dn-y-limit-reset"),
            markers,
            f"reset_{axis}_limits",
        )
        reset.tooltip("Reset to the data-derived colour range" if axis == "z" else "Reset to the data-derived range")
    return editor, minimum, maximum, reset


_PREVIEW_SETTING_MARKER_SUFFIXES: dict[str, str] = {
    "container": "preview-settings",
    "view": "view",
    "x_axis_label": "x-axis-label",
    "y_axis_label": "y-axis-label",
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
    "z_axis_options": "z-axis-options",
    "line_width": "line-width",
    "line_type": "curve-type",
    "dual_y_axes": "dual-y-axes",
    "secondary_y_axis_label": "y2-axis-label",
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
    "ensemble_profile_options": "ensemble-profile-options",
    "show_individual_profiles": "show-individual-profiles",
    "individual_profile_count": "individual-profile-count",
    "show_standard_deviation": "show-standard-deviation",
    "histogram_x_limit_editor": "histogram-x-limit-editor",
    "histogram_x_minimum": "histogram-x-minimum",
    "histogram_x_maximum": "histogram-x-maximum",
    "reset_histogram_x_limits": "reset-histogram-x-limits",
    "show_histogram_statistics": "show-histogram-statistics",
    "histogram_statistics_position": "histogram-statistics-position",
    "show_legend": "show-legend",
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
) -> PreviewSettingsControls:
    """Render the common, thematically grouped preview settings panel."""

    from app.pages.common import expansion

    markers = {} if markers is None else markers
    with expansion(
        "Plotting settings",
        icon="palette",
        extra_classes="dn-preview-settings-section",
    ):
        settings_grid = ui.element("div").classes(
            "dn-preview-settings-grid dn-preview-settings-main-grid"
        )
        _mark(settings_grid, markers, "container")
        with settings_grid:
            with ui.column().classes("dn-preview-setting-group"):
                ui.label("Style").classes("dn-setting-group-title")
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
                title = _PlotTitle(initial_title)
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
                    histogram_bin_size.tooltip(
                        "Any positive width is accepted; at most 2,000 bins are drawn."
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
                histogram_boundary.tooltip(
                    "Choose which adjacent bin owns an exact internal edge; "
                    "the outer range endpoints are always included."
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
                    individual_profile_count.tooltip("Show up to 50 retained simulation profiles.")

                with ui.element("div").classes("dn-preview-colour-grid"):
                    primary_color_surface, primary_color = compact_color_input(
                        "Color",
                        "#b03412",
                        marker=markers.get("primary_color"),
                    )
                    secondary_color_surface, secondary_color = compact_color_input(
                        "Second color",
                        "#2596be",
                        marker=markers.get("secondary_color"),
                    )
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
                line_type = _mark(
                    ui.select(CURVE_TYPES, value="solid", label="Curve type")
                    .props("dense options-dense").classes("dn-field"), markers, "line_type",
                )
                line_type.tooltip("Default pattern; each curve can use its own.")
                with ui.element("div").classes("dn-aspect-ratio-editor"):
                    aspect_ratio = _mark(
                        ui.number(label="Aspect ratio", value=None, min=0.25, max=5, step=0.1)
                        .props('dense placeholder="Default"').classes("dn-field dn-field-compact"),
                        markers, "aspect_ratio",
                    )
                    aspect_ratio.tooltip(
                        "Width divided by height. Default keeps the current preview size; "
                        "short plots retain enough height for the axes and colorbar."
                    )
                    reset_aspect_ratio = _mark(
                        ui.button(icon="restart_alt").props(
                            'flat round dense aria-label="Reset aspect ratio to default"'
                        ).classes("dn-y-limit-reset"), markers, "reset_aspect_ratio",
                    )
                    reset_aspect_ratio.tooltip("Reset to the default aspect ratio")
                with ui.element("div").classes("dn-inline-actions"):
                    density = _mark(
                        ui.checkbox("Density", value=False),
                        markers,
                        "density",
                    )
                    histogram_touching = _mark(
                        ui.checkbox("Touching bins", value=True),
                        markers,
                        "histogram_touching",
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
                with ui.element("div").classes("dn-axis-option-grid"):
                    with ui.element("div").classes("dn-axis-option-row") as x_axis_options:
                        ui.label("X").classes("dn-axis-option-label")
                        log_x = _mark(ui.checkbox("Log", value=False), markers, "log_x")
                        invert_x = _mark(
                            ui.checkbox("Invert", value=False),
                            markers,
                            "invert_x",
                        )
                    with ui.element("div").classes("dn-axis-option-row") as y_axis_options:
                        ui.label("Y").classes("dn-axis-option-label")
                        log_y = _mark(ui.checkbox("Log", value=False), markers, "log_y")
                        invert_y = _mark(
                            ui.checkbox("Invert", value=False),
                            markers,
                            "invert_y",
                        )
                    with ui.element("div").classes("dn-axis-option-row") as z_axis_options:
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
                        ui.button(icon="restart_alt")
                        .props('flat round dense aria-label="Reset histogram X limits"')
                        .classes("dn-y-limit-reset"),
                        markers,
                        "reset_histogram_x_limits",
                    )
                    reset_histogram_x_limits.tooltip("Reset to the data range")
                x_limit_editor, x_minimum, x_maximum, reset_x_limits = _axis_limit_editor("x", markers)
                _, y_minimum, y_maximum, reset_y_limits = _axis_limit_editor("y", markers)
                z_limit_editor, z_minimum, z_maximum, reset_z_limits = _axis_limit_editor("z", markers)
                dual_y_axes = secondary_y_axis_label = secondary_y_limit_editor = None
                secondary_y_minimum = secondary_y_maximum = reset_secondary_y_limits = None
                if allow_dual_y_axes:
                    dual_y_axes = _mark(ui.switch("Two Y axes", value=False), markers, "dual_y_axes")
                    dual_y_axes.tooltip("One Y axis for each of the two curves in a combined plot.")
                    dual_y_axes.visible = False
                    secondary_y_axis_label = _mark(
                        ui.input(label="Y2 label", value="Value").props("dense").classes("dn-field"),
                        markers, "secondary_y_axis_label",
                    )
                    secondary_y_limit_editor, secondary_y_minimum, secondary_y_maximum, reset_secondary_y_limits = _axis_limit_editor("secondary_y", markers)
                    secondary_y_axis_label.visible = secondary_y_limit_editor.visible = False

            with ui.column().classes("dn-preview-setting-group"):
                ui.label("Guides").classes("dn-setting-group-title")
                with ui.element("div").classes("dn-inline-actions"):
                    show_legend = _mark(
                        ui.checkbox("Legend", value=True),
                        markers,
                        "show_legend",
                    )
                    show_histogram_statistics = _mark(
                        ui.checkbox("Summary statistics", value=True),
                        markers,
                        "show_histogram_statistics",
                    )
                    show_histogram_statistics.tooltip(
                        "Show the mean, median, and standard deviation on the histogram."
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
        ensemble_profile_options=ensemble_profile_options,
        show_individual_profiles=show_individual_profiles,
        individual_profile_count=individual_profile_count,
        show_standard_deviation=show_standard_deviation,
        histogram_x_limit_editor=histogram_x_limit_editor,
        histogram_x_minimum=histogram_x_minimum,
        histogram_x_maximum=histogram_x_maximum,
        reset_histogram_x_limits=reset_histogram_x_limits,
        show_histogram_statistics=show_histogram_statistics,
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
        dual_y_axes=dual_y_axes,
        secondary_y_axis_label=secondary_y_axis_label,
        secondary_y_limit_editor=secondary_y_limit_editor,
        secondary_y_minimum=secondary_y_minimum,
        secondary_y_maximum=secondary_y_maximum,
        reset_secondary_y_limits=reset_secondary_y_limits,
    )
    reset_y_limits.on_click(lambda: _reset_preview_axis(controls, "y", on_reset_y_limits))
    if reset_secondary_y_limits is not None:
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
        controls.histogram_touching,
        controls.histogram_statistics_position,
        controls.legend_position,
        controls.show_grid,
        controls.colorbar,
        controls.invert_y,
    ):
        control.on_value_change(on_redraw)
    controls.histogram_boundary.on_value_change(on_histogram_range_change)
    controls.line_type.on_value_change(on_line_type_change or on_redraw)
    controls.reset_aspect_ratio.on_click(lambda: controls.aspect_ratio.set_value(None))
    def refresh_visibility_and_plot() -> None:
        # One event should produce one redraw.  This matters for a dense
        # heatmap, where duplicate browser updates are visibly expensive.
        on_visibility_change()
        on_redraw()

    controls.show_histogram_statistics.on_value_change(refresh_visibility_and_plot)
    controls.show_legend.on_value_change(refresh_visibility_and_plot)
    controls.view.on_value_change(on_view_change)
    controls.density.on_value_change(on_density_change)
    controls.log_x.on_value_change(on_log_x_change)
    controls.log_y.on_value_change(on_histogram_range_change)
    for axis in ("x", "z"):
        def limits_changed(_event: Any, selected: str = axis) -> None:
            if controls._resetting_limits:
                return
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
    controls.y_minimum.on_value_change(lambda: None if controls._resetting_limits else on_y_limits_change())
    controls.y_maximum.on_value_change(lambda: None if controls._resetting_limits else on_y_limits_change())


def sync_preview_settings_visibility(
    controls: PreviewSettingsControls,
    *,
    plot_type: str,
    multiple_curves: bool = False,
    allow_histogram: bool = True,
    allow_boxplot: bool = False,
    allow_individual_profiles: bool = False,
    allow_standard_deviation: bool = False,
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
    controls.z_axis_options.visible = heatmap
    controls.x_axis_options.visible = not boxplot
    controls.y_axis_options.visible = True
    controls.colorbar.visible = heatmap
    controls.histogram_bins.visible = histogram
    controls.histogram_bin_size.visible = histogram
    controls.histogram_touching.visible = histogram
    controls.histogram_boundary.visible = histogram
    controls.histogram_x_limit_editor.visible = histogram
    controls.x_limit_editor.visible = not histogram
    controls.z_limit_editor.visible = heatmap
    controls.density.visible = histogram
    controls.show_histogram_statistics.visible = histogram
    controls.histogram_statistics_position.visible = histogram and bool(
        controls.show_histogram_statistics.value
    )
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
    controls.line_width.visible = line
    controls.line_type.visible = line
    legend_available = line or (histogram and multiple_curves)
    controls.show_legend.visible = legend_available
    controls.legend_position.visible = legend_available and bool(controls.show_legend.value)


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
        zoom_chart = ui.echart(options, renderer="canvas").classes("dn-chart-wrap-large").style(
            "width: 100%; min-height: 0; height: 100%;"
        )
    with ui.element("div").classes("dn-plot-shell"):
        chart = (
            ui.echart(options, renderer="canvas").style("width: 100%; height: 360px;").mark(marker)
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
        plot_toolbar(on_zoom=zoom_dialog.open, on_download=on_download)
    handle = PreviewChartHandle(canvas=chart, vector=vector_chart, zoom=zoom_chart, zoom_card=zoom_card)
    chart.on("brandTheme", handle.invalidate_export)
    return handle
