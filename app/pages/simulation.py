"""Interactive NiceGUI simulation workflow."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Callable
from dataclasses import dataclass
from threading import Lock
from typing import Any

import numpy as np
from nicegui import run, ui

from app.pages.common import (
    DATA_ICON,
    RawOutputControls,
    action_dock,
    expansion,
    page_intro,
    panel,
    panel_title,
    plot_toolbar,
    preview_plot,
    raw_output_controls,
    source_input,
    stat_strip,
    two_column_fields,
)
from app.simulation_exports import build_plot_export, build_raw_export
from app.simulation_service import (
    ConstantSimulationRequest,
    ConstantSimulationResult,
    run_constant_simulation,
)


@dataclass(frozen=True)
class _OutputSpec:
    key: str
    x_label: str
    y_label: str
    kind: str


_BASE_OUTPUTS = {
    "Replication timing": _OutputSpec(
        "replication_timing", "Position (kb)", "Replication timing", "space"
    ),
    "Origin efficiency": _OutputSpec(
        "origin_efficiency", "Position (kb)", "Origin efficiency", "space"
    ),
    "Fork directionality": _OutputSpec(
        "fork_directionality", "Position (kb)", "Fork directionality", "space"
    ),
}
_DETAILED_OUTPUTS = {
    "Replicated fraction": _OutputSpec(
        "replicated_fraction", "Time (min)", "Replicated fraction", "time"
    ),
    "Total forks": _OutputSpec("total_forks", "Time (min)", "Total forks", "time"),
    "Active forks": _OutputSpec("active_forks", "Time (min)", "Active forks", "time"),
    "Stalled forks": _OutputSpec(
        "stalled_forks", "Time (min)", "Stalled forks", "time"
    ),
    "Firing factors": _OutputSpec(
        "firing_factors", "Time (min)", "Available factors", "time"
    ),
    "S-phase duration": _OutputSpec(
        "completion_time", "Replicate", "Duration (min)", "distribution"
    ),
    "Origins per replicate": _OutputSpec(
        "num_origins", "Replicate", "Origins", "distribution"
    ),
    "Replicated fraction (space-time)": _OutputSpec(
        "replicated_fraction_xt", "Time (min)", "Position (kb)", "xt"
    ),
    "Initiation events (space-time)": _OutputSpec(
        "initiation_events_xt", "Time (min)", "Position (kb)", "xt"
    ),
    "Coalescence events (space-time)": _OutputSpec(
        "coalescence_events_xt", "Time (min)", "Position (kb)", "xt"
    ),
    "Right fork density (space-time)": _OutputSpec(
        "right_moving_fork_density_xt", "Time (min)", "Position (kb)", "xt"
    ),
    "Left fork density (space-time)": _OutputSpec(
        "left_moving_fork_density_xt", "Time (min)", "Position (kb)", "xt"
    ),
}
_OUTPUT_SPECS = {**_BASE_OUTPUTS, **_DETAILED_OUTPUTS}


class _ProgressState:
    """Thread-safe hand-off from DNAscape's worker callback to the UI loop."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._value = 0.0
        self._message = "Waiting to run"

    def report(self, value: float, message: str) -> None:
        with self._lock:
            self._value = max(0.0, min(1.0, float(value)))
            self._message = str(message)

    def snapshot(self) -> tuple[float, str]:
        with self._lock:
            return self._value, self._message


def _number(control: Any, label: str) -> float:
    """Read a NiceGUI numeric control with a useful validation message."""

    if control.value is None:
        raise ValueError(f"{label} is required.")
    try:
        return float(control.value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a number.") from exc


def _integer(control: Any, label: str) -> int:
    value = _number(control, label)
    if not np.isfinite(value) or not value.is_integer():
        raise ValueError(f"{label} must be a whole number.")
    return int(value)


def _sample_series(
    x: np.ndarray,
    y: np.ndarray,
    *,
    limit: int = 2_500,
) -> list[list[float | None]]:
    """Bound browser payload size while retaining both ends of the series."""

    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    if x.size != y.size:
        raise ValueError("The selected simulation output has no matching axis.")
    if x.size > limit:
        indices = np.linspace(0, x.size - 1, limit, dtype=int)
        x = x[indices]
        y = y[indices]
    return [
        [float(x_value), float(y_value) if np.isfinite(y_value) else None]
        for x_value, y_value in zip(x, y)
    ]


def _sample_heatmap(
    values: np.ndarray,
    position_kb: np.ndarray,
    time_grid: np.ndarray,
    *,
    limit: int = 40_000,
) -> tuple[list[str], list[str], list[list[float | int | None]]]:
    """Bound a position-by-time matrix before sending it to the browser."""

    matrix = np.asarray(values, dtype=float)
    if matrix.shape == (time_grid.size, position_kb.size):
        matrix = matrix.T
    if matrix.shape != (position_kb.size, time_grid.size):
        raise ValueError("The selected space-time output has no matching axes.")
    max_axis = max(1, int(math.sqrt(limit)))
    position_indices = np.linspace(
        0, max(0, position_kb.size - 1), min(position_kb.size, max_axis), dtype=int
    )
    time_indices = np.linspace(
        0, max(0, time_grid.size - 1), min(time_grid.size, max_axis), dtype=int
    )
    sampled = matrix[np.ix_(position_indices, time_indices)]
    data: list[list[float | int | None]] = []
    for position_index, row in enumerate(sampled):
        for time_index, value in enumerate(row):
            data.append(
                [
                    time_index,
                    position_index,
                    float(value) if np.isfinite(value) else None,
                ]
            )
    return (
        [f"{value:g}" for value in time_grid[time_indices]],
        [f"{value:g}" for value in position_kb[position_indices]],
        data,
    )


def _result_plot_options(
    result: ConstantSimulationResult,
    *,
    output_label: str,
    title: str,
    series_label: str,
    x_axis_label: str,
    y_axis_label: str,
    color: str,
    palette: str,
    view_mode: str,
    histogram_bins: int,
    density: bool,
    show_legend: bool,
    show_colorbar: bool,
    log_y: bool,
    invert_y: bool,
    y_limits: tuple[float, float] | None,
) -> dict[str, Any]:
    """Build bounded ECharts options for any supported simulation output."""

    spec = _OUTPUT_SPECS[output_label]
    values = result.outputs.get(spec.key)
    if values is None:
        if spec.key == "firing_factors":
            raise ValueError("Enable a finite firing-factor pool to plot firing factors.")
        if output_label in _DETAILED_OUTPUTS:
            raise ValueError(
                "Run the simulation again with detailed outputs enabled to plot this result."
            )
        raise ValueError(f"DNAscape did not return {output_label.lower()}.")
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).any():
        raise ValueError(
            "The selected output has no finite values. Try a positive firing rate or a "
            "longer replication-time cap."
        )

    title_options = {
        "text": title.strip() or output_label,
        "left": "center",
        "textStyle": {"fontSize": 15, "fontWeight": 500},
    }
    if spec.kind == "xt":
        x_categories, y_categories, data = _sample_heatmap(
            values, result.position_kb, result.time_grid
        )
        finite = values[np.isfinite(values)]
        palettes = {
            "viridis": ["#440154", "#31688e", "#35b779", "#fde725"],
            "magma": ["#000004", "#721f81", "#f1605d", "#fcfdbf"],
            "plasma": ["#0d0887", "#9c179e", "#ed7953", "#f0f921"],
            "cividis": ["#00204c", "#575d6d", "#a59c74", "#ffee99"],
            "coolwarm": ["#3b4cc0", "#b9d0e5", "#f5c4ad", "#b40426"],
        }
        return {
            "animation": False,
            "title": title_options,
            "tooltip": {"position": "top"},
            "grid": {"left": 74, "right": 34, "top": 54, "bottom": 76},
            "xAxis": {
                "type": "category",
                "data": x_categories,
                "name": x_axis_label.strip() or spec.x_label,
                "nameLocation": "middle",
                "nameGap": 38,
                "splitArea": {"show": False},
                "axisLabel": {"hideOverlap": True},
            },
            "yAxis": {
                "type": "category",
                "data": y_categories,
                "name": y_axis_label.strip() or spec.y_label,
                "nameLocation": "middle",
                "nameGap": 56,
                "inverse": bool(invert_y),
                "splitArea": {"show": False},
                "axisLabel": {"hideOverlap": True},
            },
            "visualMap": {
                "show": bool(show_colorbar),
                "min": float(np.min(finite)),
                "max": float(np.max(finite)),
                "calculable": False,
                "orient": "horizontal",
                "left": "center",
                "bottom": 4,
                "inRange": {"color": palettes.get(palette, palettes["viridis"])},
            },
            "series": [
                {
                    "name": output_label,
                    "type": "heatmap",
                    "data": data,
                    "progressive": 4_000,
                    "emphasis": {"itemStyle": {"shadowBlur": 5}},
                }
            ],
        }

    flattened = values.reshape(-1)
    histogram = spec.kind == "distribution" or view_mode == "Histogram"
    if histogram:
        finite = flattened[np.isfinite(flattened)]
        counts, edges = np.histogram(
            finite,
            bins=max(1, min(2_000, int(histogram_bins))),
            density=bool(density),
        )
        centers = (edges[:-1] + edges[1:]) / 2
        return {
            "animation": False,
            "title": title_options,
            "tooltip": {"trigger": "axis"},
            "grid": {"left": 70, "right": 24, "top": 58, "bottom": 58},
            "xAxis": {
                "type": "category",
                "data": [f"{value:g}" for value in centers],
                "name": x_axis_label.strip() or spec.y_label,
                "nameLocation": "middle",
                "nameGap": 36,
                "axisLabel": {"hideOverlap": True},
                "splitLine": {"show": False},
            },
            "yAxis": {
                "type": "log" if log_y else "value",
                "name": y_axis_label.strip() or ("Density" if density else "Count"),
                "nameLocation": "middle",
                "nameGap": 50,
                "inverse": bool(invert_y),
                "splitLine": {"show": False},
            },
            "series": [
                {
                    "name": series_label.strip() or output_label,
                    "type": "bar",
                    "itemStyle": {"color": color},
                    "data": [float(value) for value in counts],
                }
            ],
        }

    axis = result.position_kb if spec.kind == "space" else result.time_grid
    data = _sample_series(axis, flattened)
    y_axis: dict[str, Any] = {
        "type": "log" if log_y else "value",
        "name": y_axis_label.strip() or spec.y_label,
        "nameLocation": "middle",
        "nameGap": 52,
        "inverse": bool(invert_y),
        "splitLine": {"show": False},
    }
    if y_limits is not None:
        y_axis["min"], y_axis["max"] = y_limits
    return {
        "animation": False,
        "title": title_options,
        "tooltip": {"trigger": "axis"},
        "legend": {"show": bool(show_legend), "top": 28},
        "grid": {"left": 70, "right": 24, "top": 62, "bottom": 58},
        "xAxis": {
            "type": "value",
            "name": x_axis_label.strip() or spec.x_label,
            "nameLocation": "middle",
            "nameGap": 34,
            "splitLine": {"show": False},
        },
        "yAxis": y_axis,
        "series": [
            {
                "name": series_label.strip() or "Mean",
                "type": "line",
                "showSymbol": False,
                "connectNulls": False,
                "sampling": "lttb",
                "lineStyle": {"width": 2, "color": color},
                "itemStyle": {"color": color},
                "data": data,
            }
        ],
    }


def _render_result_plot(
    result: ConstantSimulationResult,
    *,
    output_label: str,
    title: str,
    series_label: str,
    x_axis_label: str,
    y_axis_label: str,
    color: str,
    palette: str,
    view_mode: str,
    histogram_bins: int,
    density: bool,
    show_legend: bool,
    show_colorbar: bool,
    log_y: bool,
    invert_y: bool,
    y_limits: tuple[float, float] | None,
    on_download: Callable[[str], Any],
) -> None:
    """Render a bounded ECharts preview with shared zoom/download tools."""

    options = _result_plot_options(
        result,
        output_label=output_label,
        title=title,
        series_label=series_label,
        x_axis_label=x_axis_label,
        y_axis_label=y_axis_label,
        color=color,
        palette=palette,
        view_mode=view_mode,
        histogram_bins=histogram_bins,
        density=density,
        show_legend=show_legend,
        show_colorbar=show_colorbar,
        log_y=log_y,
        invert_y=invert_y,
        y_limits=y_limits,
    )
    with ui.dialog() as zoom_dialog, ui.card().classes("dn-plot-dialog"):
        with ui.row().classes("dn-plot-dialog-header"):
            ui.label(title.strip() or output_label).classes("dn-panel-title")
            ui.space()
            ui.button(icon="close", on_click=zoom_dialog.close).props(
                'flat round dense aria-label="Close enlarged plot"'
            )
        ui.echart(options).classes("dn-chart-wrap-large").style(
            "width: 100%; min-height: 0; height: 100%;"
        )
    with ui.element("div").classes("dn-plot-shell"):
        ui.echart(options).style("width: 100%; height: 360px;").mark(
            "simulation-result-plot"
        )
        plot_toolbar(on_zoom=zoom_dialog.open, on_download=on_download)


def render(_navigate: Callable[[str], None]) -> None:
    """Render the local simulation controls and live preview."""

    page_intro(
        "Simulation",
        "Simulate DNA replication",
        "Simulate stochastic replication dynamics, vary kinetic assumptions, and preview DNAscape-ready outputs.",
    )
    with ui.element("div").classes("dn-workspace"):
        with ui.column().classes("dn-control-stack"):
            with panel(extra_classes="dn-control-card"):
                panel_title("Controls")
                with expansion(
                    "Data",
                    icon=DATA_ICON,
                    expanded=True,
                    extra_classes="dn-data-section",
                ):
                    with ui.element("div").classes("dn-source-grid dn-data-fields"):
                        firing_source = source_input(
                            "Firing rate",
                            default_mode="Constant",
                            constant=0.01,
                            units="min⁻¹ per bin",
                        )
                        speed_source = source_input(
                            "Fork speed",
                            default_mode="Constant",
                            constant=1.4,
                            units="kb/min",
                        )
                    ui.label(
                        "This first live integration runs Constant inputs. Profile uploads remain available for the next step."
                    ).classes("dn-muted")

                with expansion("Region", icon="straighten"):
                    with two_column_fields():
                        resolution_x = (
                            ui.number(
                                label="Resolution (kb/bin)", value=1.0, min=0.000001, step=0.1
                            )
                            .props("dense")
                            .classes("dn-field")
                        )
                        resolution_t = (
                            ui.number(
                                label="Time step (min/bin)", value=5.0, min=0.000001, step=1.0
                            )
                            .props("dense")
                            .classes("dn-field")
                        )
                        region_start = (
                            ui.number(label="Start (kb)", value=0.0, min=0.0, step=1.0)
                            .props("dense")
                            .classes("dn-field")
                        )
                        region_end = (
                            ui.number(
                                label="End (kb, exclusive)", value=1000.0, min=0.000001, step=1.0
                            )
                            .props("dense")
                            .classes("dn-field")
                        )

                with expansion("Simulation settings", icon="tune"):
                    with two_column_fields():
                        replicates = (
                            ui.number(label="Replicates", value=20, min=1, step=1)
                            .props("dense")
                            .classes("dn-field")
                        )
                        geometry = (
                            ui.select(["linear", "periodic"], value="linear", label="Geometry")
                            .props("dense options-dense")
                            .classes("dn-field")
                        )
                        time_window = (
                            ui.number(
                                label="Time window (min)", value=300.0, min=0.000001, step=10.0
                            )
                            .props("dense")
                            .classes("dn-field")
                        )
                        max_replication_time = (
                            ui.number(
                                label="Max replication time (min)",
                                value=1200.0,
                                min=0.000001,
                                step=50.0,
                            )
                            .props("dense")
                            .classes("dn-field")
                        )
                        missing_values = (
                            ui.select(
                                ["error", "interpolate", "zero"],
                                value="error",
                                label="Missing values",
                            )
                            .props("dense options-dense")
                            .classes("dn-field")
                        )
                    track_detailed_outputs = (
                        ui.checkbox("Track detailed outputs", value=False)
                        .mark("simulation-track-detailed-outputs")
                    )
                    ui.label(
                        "Adds time-series and space-time statistics; keep this off for the fastest run."
                    ).classes("dn-muted")
                    fixed_seed = ui.checkbox("Fixed seed", value=True)
                    seed = (
                        ui.number(label="Seed", value=1, min=0, step=1)
                        .props("dense")
                        .classes("dn-field")
                        .mark("simulation-seed")
                    )
                    seed.bind_visibility_from(fixed_seed, "value")
                    with two_column_fields():
                        stall_rate = (
                            ui.number(label="Stall rate", value=0.0, min=0.0, max=1.0, step=0.001)
                            .props("dense")
                            .classes("dn-field")
                        )
                    non_recovering = ui.checkbox("Non-recovering stalls", value=True)
                    stall_duration = (
                        ui.number(label="Stall duration (min)", value=10.0, min=0.000001, step=1.0)
                        .props("dense")
                        .classes("dn-field")
                        .mark("simulation-stall-duration")
                    )
                    stall_duration.bind_visibility_from(
                        non_recovering,
                        "value",
                        backward=lambda value: not value,
                    )
                    finite_pool = ui.checkbox("Finite firing-factor pool", value=False)
                    with (
                        ui.column()
                        .classes("w-full gap-2")
                        .mark("simulation-pool-fields") as pool_fields
                    ):
                        with two_column_fields():
                            pool_size = (
                                ui.number(label="Pool size", value=100, min=2, step=1)
                                .props("dense")
                                .classes("dn-field")
                            )
                            recycle_rate = (
                                ui.number(label="Recycle rate", value=0.0, min=0.0, step=0.001)
                                .props("dense")
                                .classes("dn-field")
                            )
                        factor_coupling = ui.checkbox(
                            "Couple firing factors to active forks", value=False
                        )
                    pool_fields.bind_visibility_from(finite_pool, "value")

                with expansion("Plotting settings", icon="palette"):
                    with two_column_fields():
                        output = (
                            ui.select(
                                list(_BASE_OUTPUTS),
                                value="Replication timing",
                                label="Output",
                            )
                            .props("dense options-dense")
                            .classes("dn-field")
                            .mark("simulation-output")
                        )
                        view = (
                            ui.select(
                                ["Single view", "Multiple view", "Histogram", "Time array"],
                                value="Single view",
                                label="View",
                            )
                            .props("dense options-dense")
                            .classes("dn-field")
                            .mark("simulation-view")
                        )
                        title = (
                            ui.input(label="Title", value="Simulation output")
                            .props("dense")
                            .classes("dn-field")
                        )
                        series_label = (
                            ui.input(label="Label", value="Mean").props("dense").classes("dn-field")
                        )
                        x_axis_label = (
                            ui.input(label="X-axis label", value="Position (kb)")
                            .props("dense")
                            .classes("dn-field")
                        )
                        y_axis_label = (
                            ui.input(label="Y-axis label", value="Replication timing")
                            .props("dense")
                            .classes("dn-field")
                        )
                        line_or_bar_color = (
                            ui.input(label="Line / bar color", value="#2c7f77")
                            .props("type=color dense")
                            .classes("dn-field")
                            .mark("simulation-line-or-bar-color")
                        )
                        heatmap_palette = (
                            ui.select(
                                ["viridis", "magma", "plasma", "cividis", "coolwarm"],
                                value="viridis",
                                label="Heatmap palette",
                            )
                            .props("dense options-dense")
                            .classes("dn-field")
                            .mark("simulation-heatmap-palette")
                        )
                        histogram_bins = (
                            ui.number(label="Histogram bins", value=60, min=1, max=2000, step=1)
                            .props("dense")
                            .classes("dn-field")
                            .mark("simulation-histogram-bins")
                        )
                        figure_width = (
                            ui.number(label="Width (in)", value=8.0, min=1.0, step=0.5)
                            .props("dense")
                            .classes("dn-field")
                        )
                        figure_height = (
                            ui.number(label="Height (in)", value=4.5, min=1.0, step=0.5)
                            .props("dense")
                            .classes("dn-field")
                        )
                    with ui.element("div").classes("dn-inline-actions"):
                        show_legend = ui.checkbox("Legend", value=True)
                        density = ui.checkbox("Density", value=False).mark("simulation-density")
                        colorbar = ui.checkbox("Colorbar", value=True).mark("simulation-colorbar")
                        log_y = ui.checkbox("Log Y", value=False)
                        invert_y = ui.checkbox("Invert Y", value=False)
                        set_y_limits = ui.checkbox("Y limits", value=False)
                    legend_position = (
                        ui.select(
                            ["best", "upper right", "upper left", "lower right", "lower left"],
                            value="best",
                            label="Legend position",
                        )
                        .props("dense options-dense")
                        .classes("dn-field")
                    )
                    legend_position.bind_visibility_from(show_legend, "value")
                    histogram_bins.bind_visibility_from(
                        view, "value", backward=lambda value: value == "Histogram"
                    )
                    density.bind_visibility_from(
                        view, "value", backward=lambda value: value == "Histogram"
                    )
                    heatmap_palette.bind_visibility_from(
                        view, "value", backward=lambda value: value == "Time array"
                    )
                    colorbar.bind_visibility_from(
                        view, "value", backward=lambda value: value == "Time array"
                    )
                    line_or_bar_color.bind_visibility_from(
                        view, "value", backward=lambda value: value != "Time array"
                    )
                    with two_column_fields() as y_limit_fields:
                        y_minimum = (
                            ui.number(label="Y minimum", value=0.0, step=0.1)
                            .props("dense")
                            .classes("dn-field")
                        )
                        y_maximum = (
                            ui.number(label="Y maximum", value=1.0, step=0.1)
                            .props("dense")
                            .classes("dn-field")
                        )
                    y_limit_fields.bind_visibility_from(set_y_limits, "value")

                    def sync_output_options() -> None:
                        labels = list(_BASE_OUTPUTS)
                        if track_detailed_outputs.value:
                            labels.extend(_DETAILED_OUTPUTS)
                        output.options = labels
                        if output.value not in labels:
                            output.value = "Replication timing"
                        output.update()

                    def sync_plot_labels() -> None:
                        selected = str(output.value or "Replication timing")
                        spec = _OUTPUT_SPECS.get(selected, _BASE_OUTPUTS["Replication timing"])
                        title.value = selected
                        x_axis_label.value = spec.x_label
                        y_axis_label.value = spec.y_label
                        if spec.kind == "xt":
                            view.value = "Time array"
                        elif spec.kind == "distribution":
                            view.value = "Histogram"
                        elif view.value in {"Time array", "Histogram"}:
                            view.value = "Single view"
                        for control in (title, x_axis_label, y_axis_label, view):
                            control.update()
                        if latest_result is not None:
                            render_latest_result()

                    track_detailed_outputs.on_value_change(sync_output_options)
                    output.on_value_change(sync_plot_labels)

                running = False
                latest_result: ConstantSimulationResult | None = None

                def current_y_limits() -> tuple[float, float] | None:
                    if not set_y_limits.value:
                        return None
                    limits = (
                        _number(y_minimum, "Y minimum"),
                        _number(y_maximum, "Y maximum"),
                    )
                    if limits[1] <= limits[0]:
                        raise ValueError("Y maximum must be greater than Y minimum.")
                    return limits

                async def download_plot(file_format: str) -> None:
                    if latest_result is None:
                        ui.notify("Run a simulation before downloading a plot.", type="warning")
                        return
                    try:
                        selected_label = str(output.value or "Replication timing")
                        spec = _OUTPUT_SPECS[selected_label]
                        artifact = await run.io_bound(
                            build_plot_export,
                            latest_result,
                            spec.key,
                            file_format,
                            filename=f"dnascape_{spec.key}_plot",
                            title=str(title.value or selected_label),
                            x_label=str(x_axis_label.value or spec.x_label),
                            y_label=str(y_axis_label.value or spec.y_label),
                            color=str(line_or_bar_color.value or "#2c7f77"),
                            cmap=str(heatmap_palette.value or "viridis"),
                            invert_y=bool(invert_y.value),
                            y_limits=current_y_limits(),
                            view_mode=str(view.value or "Single view"),
                            histogram_bins=_integer(histogram_bins, "Histogram bins"),
                            density=bool(density.value),
                            figure_size=(
                                _number(figure_width, "Figure width"),
                                _number(figure_height, "Figure height"),
                            ),
                        )
                        if artifact is None:
                            raise RuntimeError("Plot export stopped before returning data.")
                    except (RuntimeError, TypeError, ValueError) as exc:
                        ui.notify(str(exc), type="negative")
                        return
                    ui.download(artifact.data, artifact.filename, artifact.media_type)

                async def download_raw(controls: RawOutputControls) -> None:
                    if latest_result is None:
                        ui.notify("Run a simulation before downloading raw data.", type="warning")
                        return
                    try:
                        selected_label = str(output.value or "Replication timing")
                        artifact = await run.io_bound(
                            build_raw_export,
                            latest_result,
                            _OUTPUT_SPECS[selected_label].key,
                            str(controls.file_type.value or "NPZ"),
                            filename=str(controls.file_name.value or "dnascape_simulation"),
                            gzip_output=bool(controls.gzip_output.value),
                        )
                        if artifact is None:
                            raise RuntimeError("Raw export stopped before returning data.")
                    except (RuntimeError, TypeError, ValueError) as exc:
                        ui.notify(str(exc), type="negative")
                        return
                    ui.download(artifact.data, artifact.filename, artifact.media_type)

                def render_latest_result() -> None:
                    if latest_result is None:
                        return
                    result_slot.clear()
                    with result_slot:
                        try:
                            _render_result_plot(
                                latest_result,
                                output_label=str(output.value or "Replication timing"),
                                title=str(title.value or ""),
                                series_label=str(series_label.value or ""),
                                x_axis_label=str(x_axis_label.value or ""),
                                y_axis_label=str(y_axis_label.value or ""),
                                color=str(line_or_bar_color.value or "#2c7f77"),
                                palette=str(heatmap_palette.value or "viridis"),
                                view_mode=str(view.value or "Single view"),
                                histogram_bins=_integer(histogram_bins, "Histogram bins"),
                                density=bool(density.value),
                                show_legend=bool(show_legend.value),
                                show_colorbar=bool(colorbar.value),
                                log_y=bool(log_y.value),
                                invert_y=bool(invert_y.value),
                                y_limits=current_y_limits(),
                                on_download=download_plot,
                            )
                        except (RuntimeError, TypeError, ValueError) as exc:
                            with ui.element("div").classes("dn-empty-state"):
                                ui.icon("info_outline").classes("dn-empty-icon")
                                ui.label("Output unavailable").classes("dn-empty-title")
                                ui.label(str(exc)).classes("dn-muted")

                async def simulate() -> None:
                    nonlocal latest_result, running
                    if running:
                        ui.notify("A simulation is already running.", type="warning")
                        return
                    error.visible = False
                    if (
                        firing_source.mode.value != "Constant"
                        or speed_source.mode.value != "Constant"
                    ):
                        message = (
                            "This first integration supports Constant inputs only. "
                            "Choose Constant for both firing rate and fork speed."
                        )
                        error.text = message
                        error.visible = True
                        status.text = "Choose Constant inputs to run"
                        status_stats.text = "Input needed"
                        ui.notify(message, type="warning")
                        return

                    try:
                        request = ConstantSimulationRequest(
                            firing_rate=_number(firing_source.constant, "Firing rate"),
                            fork_speed=_number(speed_source.constant, "Fork speed"),
                            resolution_x=_number(resolution_x, "Spatial resolution"),
                            resolution_t=_number(resolution_t, "Time step"),
                            region_start_kb=_number(region_start, "Region start"),
                            region_end_kb=_number(region_end, "Region end"),
                            replicates=_integer(replicates, "Replicates"),
                            geometry=str(geometry.value),
                            time_window=_number(time_window, "Time window"),
                            max_replication_time=_number(
                                max_replication_time, "Maximum replication time"
                            ),
                            missing_values=str(missing_values.value),
                            seed=_integer(seed, "Seed") if fixed_seed.value else None,
                            stall_rate=_number(stall_rate, "Stall rate"),
                            stall_duration=(
                                math.inf
                                if non_recovering.value
                                else _number(stall_duration, "Stall duration")
                            ),
                            firing_factors=(
                                float(_integer(pool_size, "Firing-factor pool"))
                                if finite_pool.value
                                else None
                            ),
                            firing_factor_recycle=(
                                _number(recycle_rate, "Firing-factor recycle rate")
                                if finite_pool.value
                                else 0.0
                            ),
                            firing_factor_fork_coupling=(
                                bool(factor_coupling.value) if finite_pool.value else False
                            ),
                            track_detailed_outputs=bool(track_detailed_outputs.value),
                        )
                    except (TypeError, ValueError) as exc:
                        error.text = str(exc)
                        error.visible = True
                        status.text = "Check the settings"
                        status_stats.text = "Invalid input"
                        return

                    running = True
                    simulate_button.disable()
                    progress_state = _ProgressState()
                    progress.visible = True
                    progress.value = 0.02
                    status.text = "Preparing simulation..."
                    status_stats.text = "Running"
                    error.visible = False
                    latest_result = None
                    result_slot.clear()
                    with result_slot:
                        ui.label("Simulation is running in the background...").classes("dn-muted")
                    try:
                        worker = asyncio.create_task(
                            run.io_bound(
                                run_constant_simulation,
                                request,
                                progress_state.report,
                            )
                        )
                        while not worker.done():
                            value, message = progress_state.snapshot()
                            progress.value = value
                            status.text = message
                            await asyncio.sleep(0.08)
                        result = await worker
                        if result is None:
                            raise RuntimeError(
                                "The simulation worker stopped before returning data."
                            )
                        progress.value = 1.0
                        status.text = f"Complete in {result.elapsed_seconds:.2f} s"
                        status_stats.text = "Complete"
                        replicate_stats.text = f"{result.replicates:,}"
                        region_stats.text = (
                            f"{request.region_start_kb:g}–{request.region_end_kb:g} kb"
                        )
                        geometry_stats.text = result.geometry.title()
                        latest_result = result
                        render_latest_result()
                    except Exception as exc:  # keep malformed local jobs inside the preview
                        progress.visible = False
                        status.text = "Simulation could not be completed"
                        status_stats.text = "Error"
                        error.text = str(exc) or type(exc).__name__
                        error.visible = True
                        result_slot.clear()
                        with result_slot:
                            preview_plot(
                                "Mean replication timing",
                                kind="simulation",
                                caption="Correct the settings and run again.",
                            )
                    finally:
                        running = False
                        simulate_button.enable()

                with action_dock():
                    simulate_button = (
                        ui.button("Simulate", icon="play_arrow", on_click=simulate)
                        .props("no-caps")
                        .classes("dn-action-primary")
                        .mark("action-simulate")
                    )

        with ui.column().classes("dn-preview-stack"):
            with panel(extra_classes="dn-preview-card"):
                panel_title("Preview")
                stats = stat_strip(
                    (
                        ("Replicates", "20"),
                        ("Region", "0–1,000 kb"),
                        ("Geometry", "Linear"),
                        ("Status", "Not run"),
                    )
                )
                replicate_stats = stats["Replicates"]
                region_stats = stats["Region"]
                geometry_stats = stats["Geometry"]
                status_stats = stats["Status"]
                progress = (
                    ui.linear_progress(value=0.0)
                    .props("rounded instant-feedback")
                    .classes("dn-simulation-progress")
                    .mark("simulation-progress")
                )
                progress.visible = False
                status = (
                    ui.label("Ready for constant inputs")
                    .classes("dn-muted")
                    .mark("simulation-status")
                )
                error = ui.label("").classes("text-negative").mark("simulation-error")
                error.visible = False
                with (
                    ui.element("div")
                    .classes("w-full")
                    .mark("simulation-result-container") as result_slot
                ):
                    preview_plot(
                        "Mean replication timing",
                        kind="simulation",
                        caption="Run the constant-input ensemble to replace this preview.",
                    )
                raw_output_controls(
                    file_name="dnascape_simulation",
                    default_type="NPZ",
                    on_download=download_raw,
                )
