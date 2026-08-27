"""Interactive NiceGUI simulation workflow."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Callable
from concurrent.futures import CancelledError
from dataclasses import dataclass, replace
from threading import Event, Lock
from typing import Any

import numpy as np
from nicegui import run, ui

from app.export_settings import get_export_settings
from app.mapping_uploads import (
    UploadInspection,
    inspect_upload_event,
    materialize_upload,
)
from app.pages.common import (
    DATA_ICON,
    RawOutputControls,
    action_dock,
    data_upload,
    expansion,
    page_intro,
    panel,
    panel_title,
    preview_plot,
    raw_output_controls,
    set_data_upload_locked,
    stat_strip,
    two_column_fields,
)
from app.preview_plotting import (
    FIXED_QUANTITY_Y_LIMITS,
    HistogramBinSynchronizer,
    PreviewBand,
    PreviewLine,
    automatic_y_limits,
    build_preview_plot_options,
    histogram_x_limits_from_controls,
    plotting_settings_ready,
    preview_settings_controls,
    render_preview_chart,
    sync_preview_settings_visibility,
)
from app.session_cache import (
    cache_artifact,
    cache_upload,
    cache_workflow,
    consume_cached_workflow,
    current_session_cache,
)
from app.simulation_exports import (
    build_chart_capture_export,
    build_raw_export,
    select_result_output,
)
from app.simulation_service import (
    MAX_REPLICATES,
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
_DISTRIBUTION_OUTPUTS = {
    "S-phase duration": _OutputSpec(
        "s_phase_duration", "Replicate", "Duration (min)", "distribution"
    ),
    "Origins per replicate": _OutputSpec("num_origins", "Replicate", "Origins", "distribution"),
    "Inter-origin distance": _OutputSpec(
        "inter_origin_distances",
        "Sample",
        "Distance (kb)",
        "distribution",
    ),
}
_TIME_OUTPUTS = {
    "Replicated fraction": _OutputSpec(
        "replicated_fraction", "Time (min)", "Replicated fraction", "time"
    ),
    "Total forks": _OutputSpec("total_forks", "Time (min)", "Total forks", "time"),
    "Active forks": _OutputSpec("active_forks", "Time (min)", "Active forks", "time"),
    "Stalled forks": _OutputSpec("stalled_forks", "Time (min)", "Stalled forks", "time"),
    "Firing factors": _OutputSpec("firing_factors", "Time (min)", "Available factors", "time"),
}
_SPACETIME_OUTPUTS = {
    "Replicated fraction (space-time)": _OutputSpec(
        "replicated_fraction_xt", "Position (kb)", "Time (min)", "xt"
    ),
    "Initiation events (space-time)": _OutputSpec(
        "initiation_events_xt", "Position (kb)", "Time (min)", "xt"
    ),
    "Coalescence events (space-time)": _OutputSpec(
        "coalescence_events_xt", "Position (kb)", "Time (min)", "xt"
    ),
    "Right fork density (space-time)": _OutputSpec(
        "right_moving_fork_density_xt", "Position (kb)", "Time (min)", "xt"
    ),
    "Left fork density (space-time)": _OutputSpec(
        "left_moving_fork_density_xt", "Position (kb)", "Time (min)", "xt"
    ),
}
_STANDARD_OUTPUTS = {**_BASE_OUTPUTS, **_DISTRIBUTION_OUTPUTS}
_OUTPUT_SPECS = {
    **_STANDARD_OUTPUTS,
    **_TIME_OUTPUTS,
    **_SPACETIME_OUTPUTS,
}

# Some simulation quantities have a meaningful physical domain which should not
# shrink to the particular values observed in one run. Other outputs retain an
# automatic, data-dependent range chosen by ECharts.
_NONNEGATIVE_LINE_OUTPUTS = {
    "Total forks",
    "Active forks",
    "Stalled forks",
    "Firing factors",
}

_SOURCE_ACCEPT = ".csv,.tsv,.txt,.npy,.npv,.npz,.gz"
_SOURCE_DEPENDENCE = {
    "x": "Space",
    "t": "Time",
    "xt": "Space-time",
}

_MAX_PREVIEW_PROFILES = 50


@dataclass(frozen=True)
class _SimulationSourceControls:
    """Live controls and retained upload state for one rsim kinetic field."""

    mode: Any
    form: Any
    constant: Any
    uploader: Any
    inspections: list[UploadInspection]
    selection: dict[str, str | None]


def _render_simulation_source(
    label: str,
    *,
    constant: float,
    units: str,
    on_profile_change: Callable[[], Any] | None = None,
) -> _SimulationSourceControls:
    """Render one scalar-or-upload rsim source using Mapping's upload surface."""

    marker = "-".join(label.casefold().replace("/", " ").split())
    inspections: list[UploadInspection] = []
    selection: dict[str, str | None] = {"column": None, "array_key": None}
    profile_changed = on_profile_change or (lambda: None)

    with ui.element("div").classes("dn-source-compact").mark(f"source-{marker}"):
        with ui.element("div").classes("dn-source-head"):
            mode = (
                ui.select(
                    {"scalar": "Constant", "profile": "Profile"},
                    value="scalar",
                    label="Input type",
                )
                .props("dense options-dense")
                .classes("dn-field dn-source-mode")
                .mark(f"source-{marker}-mode")
            )
            form = (
                ui.select(
                    {"x": "Space", "t": "Time", "xt": "Space-time"},
                    value="x",
                )
                .props('dense options-dense aria-label="Detected profile dependence"')
                .classes("dn-simulation-exact-form")
                .mark(f"source-{marker}-form")
            )
            form.set_visibility(False)
        with ui.element("div").classes("dn-source-fields") as constant_fields:
            constant_input = (
                ui.number(
                    label=f"Constant value ({units})",
                    value=constant,
                    min=0,
                    step=0.001 if constant < 0.1 else 0.1,
                )
                .props("dense")
                .classes("dn-field")
                .mark(f"source-{marker}-constant")
            )

        with (
            ui.element("div")
            .classes("dn-source-fields dn-source-profile-fields")
            .mark(f"source-{marker}-profile")
        ) as upload_fields:

            def sync_selection() -> None:
                if not inspections:
                    selection.update(column=None, array_key=None)
                    return
                inspection = inspections[0]
                row_vector = bool(
                    str(form.value) == "t"
                    and inspection.shape
                    and inspection.shape[0] == 1
                    and len(inspection.numeric_columns) > 1
                )
                if str(form.value) in {"x", "t"} and inspection.numeric_columns and not row_vector:
                    if selection["column"] not in inspection.numeric_columns:
                        selection["column"] = inspection.default_column
                else:
                    selection["column"] = None
                if inspection.array_keys:
                    if selection["array_key"] not in inspection.array_keys:
                        selection["array_key"] = inspection.default_array_key
                else:
                    selection["array_key"] = None

            @ui.refreshable
            def render_selection() -> None:
                sync_selection()
                if not inspections:
                    return
                inspection = inspections[0]
                matrix_columns = (
                    inspection.inferred_form == "xt" and len(inspection.numeric_columns) > 1
                )
                row_vector = bool(
                    str(form.value) == "t"
                    and inspection.shape
                    and inspection.shape[0] == 1
                    and len(inspection.numeric_columns) > 1
                )
                if matrix_columns:
                    column_options: Any = {
                        "__all_columns__": "All columns (space-time)",
                        **{column: column for column in inspection.numeric_columns},
                    }
                    column_value = selection["column"] or "__all_columns__"
                    column_label = "Profile"
                elif (
                    str(form.value) in {"x", "t"} and inspection.numeric_columns and not row_vector
                ):
                    column_options = list(inspection.numeric_columns)
                    column_value = selection["column"]
                    column_label = "Value column"
                else:
                    column_options = None
                    column_value = None
                    column_label = "Value column"
                if column_options is not None:
                    column_control = (
                        ui.select(
                            column_options,
                            value=column_value,
                            label=column_label,
                        )
                        .props("dense options-dense")
                        .classes("dn-field dn-simulation-source-select")
                        .mark(f"source-{marker}-value-column")
                    )
                    if len(inspection.numeric_columns) == 1 and not matrix_columns:
                        column_control.disable()

                    def store_column() -> None:
                        selected = (
                            None if column_control.value is None else str(column_control.value)
                        )
                        if matrix_columns and selected == "__all_columns__":
                            selection["column"] = None
                            form.set_value("xt")
                        else:
                            selection["column"] = selected
                            if matrix_columns:
                                form.set_value("x")
                        render_summary.refresh()
                        profile_changed()

                    column_control.on_value_change(store_column)
                if inspection.array_keys:
                    array_control = (
                        ui.select(
                            list(inspection.array_keys),
                            value=selection["array_key"],
                            label="Array",
                        )
                        .props("dense options-dense")
                        .classes("dn-field dn-simulation-source-select")
                        .mark(f"source-{marker}-array-key")
                    )
                    if len(inspection.array_keys) == 1:
                        array_control.disable()

                    def store_array() -> None:
                        selection["array_key"] = (
                            None if array_control.value is None else str(array_control.value)
                        )
                        render_summary.refresh()
                        profile_changed()

                    array_control.on_value_change(store_array)

            async def uploaded(event: Any) -> bool:
                try:
                    inspection = await inspect_upload_event(
                        event,
                        expected_form="auto",
                    )
                except (OSError, TypeError, ValueError) as exc:
                    ui.notify(str(exc), type="negative")
                    return False
                detected_form = str(inspection.inferred_form)
                if detected_form not in _SOURCE_DEPENDENCE:
                    ui.notify(
                        f"{label} must be a space, time, or space-time profile.",
                        type="negative",
                    )
                    return False
                inspections[:] = [inspection]
                form.set_value(detected_form)
                sync_selection()
                render_selection.refresh()
                render_summary.refresh()
                cache_upload(inspection, source="Simulation", navigate_to="simulation")
                profile_changed()
                return True

            def cleared(_event: Any) -> None:
                inspections.clear()
                selection.update(column=None, array_key=None)
                render_selection.refresh()
                render_summary.refresh()
                profile_changed()

            uploader = data_upload(
                accept=_SOURCE_ACCEPT,
                accepted_formats="CSV, TSV, TXT, NPY, NPV, NPZ, GZ",
                marker=f"source-{marker}-upload",
                show_accepted_formats=False,
                on_upload=uploaded,
                on_clear=cleared,
                max_files=1,
                hide_when_loaded=True,
            )

            @ui.refreshable
            def render_summary() -> None:
                if not inspections:
                    return
                inspection = inspections[0]
                shape = inspection.shape
                try:
                    selected = _materialize_simulation_source(
                        _SimulationSourceControls(
                            mode=mode,
                            form=form,
                            constant=constant_input,
                            uploader=uploader,
                            inspections=inspections,
                            selection=selection,
                        ),
                        label=label,
                        missing_policy="zero",
                    )
                    shape = tuple(np.asarray(selected).shape)
                except (OSError, TypeError, ValueError):
                    pass
                size = " x ".join(str(value) for value in shape) if shape else "1"
                dependence = _SOURCE_DEPENDENCE.get(str(form.value), "Unknown")
                with (
                    ui.element("div")
                    .classes("dn-mapping-detected-data dn-profile-metadata")
                    .mark(f"source-{marker}-detected-data")
                ):
                    ui.label(f"Dependence: {dependence}").classes(
                        "dn-mapping-detected-summary dn-mapping-detected-dependence"
                    )
                    ui.label(f"Array size: {size}").classes("dn-mapping-detected-summary")

            render_summary()
            render_selection()

        constant_fields.bind_visibility_from(
            mode,
            "value",
            backward=lambda value: value == "scalar",
        )
        upload_fields.bind_visibility_from(
            mode,
            "value",
            backward=lambda value: value == "profile",
        )

        def mode_changed() -> None:
            selection["column"] = None
            render_selection.refresh()
            render_summary.refresh()
            profile_changed()

        mode.on_value_change(mode_changed)

    return _SimulationSourceControls(
        mode=mode,
        form=form,
        constant=constant_input,
        uploader=uploader,
        inspections=inspections,
        selection=selection,
    )


def _materialize_simulation_source(
    controls: _SimulationSourceControls,
    *,
    label: str,
    missing_policy: str,
) -> float | np.ndarray:
    """Build one rsim field from its scalar control or retained upload."""

    if str(controls.mode.value) == "scalar":
        return _number(controls.constant, label)
    form = str(controls.form.value)
    if form not in {"x", "t", "xt"}:
        raise ValueError(f"Choose a valid input type for {label.casefold()}.")
    if not controls.inspections:
        raise ValueError(f"Upload {label.casefold()} data.")
    inspection = replace(controls.inspections[0], expected_form=form)
    return materialize_upload(
        inspection,
        column=controls.selection["column"] if form in {"x", "t"} else None,
        array_key=controls.selection["array_key"],
        missing_policy=missing_policy,
    )


def _uploaded_spatial_size(
    controls: _SimulationSourceControls,
    value: float | np.ndarray,
) -> int | None:
    """Return the spatial axis length encoded by an uploaded rsim field."""

    if str(controls.mode.value) == "scalar":
        return None
    form = str(controls.form.value)
    array = np.asarray(value)
    if form == "x":
        return int(array.size)
    if form == "xt":
        return int(array.shape[0])
    return None


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


def _bounded_profile_count(value: Any, available: int) -> int | None:
    """Return a complete profile-count edit without emitting transient errors."""

    if value in (None, "") or available < 1:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric) or not numeric.is_integer() or numeric < 1:
        return None
    return min(int(numeric), min(_MAX_PREVIEW_PROFILES, int(available)))


def _replication_profile_arrays(
    result: ConstantSimulationResult,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return validated optional ensemble arrays from a detailed result."""

    profiles = getattr(result, "replication_time_profiles", None)
    standard_deviation = getattr(result, "replication_timing_std", None)
    profile_array = None if profiles is None else np.asarray(profiles, dtype=float)
    if profile_array is not None and (
        profile_array.ndim != 2 or profile_array.shape[1] != result.position_kb.size
    ):
        profile_array = None
    deviation_array = (
        None
        if standard_deviation is None
        else np.asarray(standard_deviation, dtype=float).reshape(-1)
    )
    if deviation_array is not None and deviation_array.size != result.position_kb.size:
        deviation_array = None
    return profile_array, deviation_array


def _automatic_y_limits(
    result: ConstantSimulationResult,
    *,
    output_label: str,
    view_mode: str,
    histogram_bins: int,
    density: bool,
    log_y: bool,
    log_x: bool = False,
    histogram_bin_size: float | None = None,
    histogram_x_limits: tuple[float, float] | None = None,
    histogram_boundary: str = "left",
    show_individual_profiles: bool = False,
    individual_profile_count: int = 10,
    show_standard_deviation: bool = False,
) -> tuple[float, float]:
    """Return stable, editable plot bounds derived from the selected result."""

    spec = _OUTPUT_SPECS[output_label]
    selected = select_result_output(result, spec.key)
    plot_type = (
        "Heatmap"
        if spec.kind == "xt"
        else "Boxplot"
        if view_mode == "Boxplot"
        else "Histogram"
        if spec.kind == "distribution" or view_mode == "Histogram"
        else "Line"
    )
    plotted_values: list[Any] = [selected.values]
    if output_label == "Replication timing" and plot_type == "Line":
        profiles, standard_deviation = _replication_profile_arrays(result)
        if profiles is not None and show_individual_profiles:
            count = min(
                max(1, int(individual_profile_count)),
                profiles.shape[0],
                _MAX_PREVIEW_PROFILES,
            )
            plotted_values.extend(profiles[:count])
        if standard_deviation is not None and show_standard_deviation:
            mean = np.asarray(selected.values, dtype=float).reshape(-1)
            plotted_values.extend((mean - standard_deviation, mean + standard_deviation))
    limits = automatic_y_limits(
        plotted_values,
        plot_type=plot_type,
        histogram_bins=histogram_bins,
        density=density,
        log_y=log_y,
        log_x=log_x,
        histogram_bin_size=histogram_bin_size,
        histogram_x_limits=histogram_x_limits,
        histogram_boundary=histogram_boundary,
        y_axis=result.time_grid if spec.kind == "xt" else None,
        fixed_limits=_default_y_limits(
            output_label,
            view_mode=view_mode,
            log_y=log_y,
        ),
        floor_at_zero=output_label in _NONNEGATIVE_LINE_OUTPUTS,
    )
    if spec.kind != "xt":
        return limits

    # Detailed fields are tracked over the requested simulation window, which
    # can extend far beyond the biologically informative part of S phase. Use
    # the latest finite ensemble-mean replication time as the visible cutoff
    # for every space-time output. Never extend beyond the retained time grid:
    # doing so would introduce an empty band in a truncated result.
    mean_replication_times = np.asarray(result.replication_timing, dtype=float).reshape(-1)
    mean_replication_times = mean_replication_times[np.isfinite(mean_replication_times)]
    tracked_times = np.asarray(result.time_grid, dtype=float).reshape(-1)
    tracked_times = tracked_times[np.isfinite(tracked_times)]
    if log_y:
        mean_replication_times = mean_replication_times[mean_replication_times > 0]
        tracked_times = tracked_times[tracked_times > 0]
    if mean_replication_times.size == 0 or tracked_times.size == 0:
        return limits
    y_maximum = min(
        float(np.max(mean_replication_times)),
        float(np.max(tracked_times)),
    )
    if y_maximum <= limits[0]:
        return limits
    return limits[0], y_maximum


def _default_y_limits(
    output_label: str,
    *,
    view_mode: str = "Default",
    log_y: bool = False,
) -> tuple[float, float] | None:
    """Return a fixed physical Y domain when the active view has one.

    Histogram Y values are frequencies rather than the selected quantity, and
    logarithmic axes cannot use the zero/negative lower bounds in these domains.
    Those cases therefore continue to use a data-derived automatic range.
    """

    spec = _OUTPUT_SPECS[output_label]
    if spec.kind in {"distribution", "xt"} or view_mode in {"Histogram", "Boxplot"} or log_y:
        return None
    return FIXED_QUANTITY_Y_LIMITS.get(spec.key)


def _default_axis_labels(
    *,
    output_label: str,
    view_mode: str,
    density: bool,
) -> tuple[str, str]:
    """Return labels for the quantity and frequency axes in the active view."""

    spec = _OUTPUT_SPECS[output_label]
    if view_mode == "Boxplot":
        return "", spec.y_label
    if spec.kind == "distribution" or view_mode == "Histogram":
        return spec.y_label, "Density" if density else "Count"
    return spec.x_label, spec.y_label


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
    legend_position: str = "Top right",
    line_width: float = 1.8,
    show_markers: bool = False,
    show_grid: bool = True,
    log_z: bool = False,
    log_x: bool = False,
    invert_x: bool = False,
    invert_z: bool = False,
    show_histogram_statistics: bool = False,
    histogram_statistics_position: str = "Top right",
    histogram_bin_size: float | None = None,
    histogram_x_limits: tuple[float, float] | None = None,
    histogram_touching: bool = True,
    histogram_boundary: str = "left",
    show_individual_profiles: bool = False,
    individual_profile_count: int = 10,
    show_standard_deviation: bool = False,
) -> dict[str, Any]:
    """Build bounded ECharts options for any supported simulation output."""

    spec = _OUTPUT_SPECS[output_label]
    values = result.outputs.get(spec.key)
    if values is None:
        if spec.key == "firing_factors":
            raise ValueError("Enable a finite firing-factor pool to plot firing factors.")
        if output_label in _TIME_OUTPUTS:
            raise ValueError(
                "Run the simulation again with time courses enabled to plot this result."
            )
        if output_label in _SPACETIME_OUTPUTS:
            raise ValueError(
                "Run the simulation again with space-time outputs enabled to plot this result."
            )
        raise ValueError(f"DNAscape did not return {output_label.lower()}.")
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).any():
        if spec.key == "inter_origin_distances":
            raise ValueError(
                "No inter-origin distances were recorded. At least two origins must "
                "fire in a realization."
            )
        raise ValueError(
            "The selected output has no finite values. Try a positive firing rate or a "
            "longer replication-time cap."
        )

    plot_type = (
        "Heatmap"
        if spec.kind == "xt"
        else "Boxplot"
        if view_mode == "Boxplot"
        else "Histogram"
        if spec.kind == "distribution" or view_mode == "Histogram"
        else "Line"
    )
    axis = result.position_kb if spec.kind == "space" else result.time_grid
    lines: tuple[PreviewLine, ...] = ()
    bands: tuple[PreviewBand, ...] = ()
    ensemble_active = False
    if output_label == "Replication timing" and plot_type == "Line":
        profiles, standard_deviation = _replication_profile_arrays(result)
        individual_lines: tuple[PreviewLine, ...] = ()
        if profiles is not None and show_individual_profiles:
            count = min(
                max(1, int(individual_profile_count)),
                profiles.shape[0],
                _MAX_PREVIEW_PROFILES,
            )
            individual_lines = tuple(
                PreviewLine(
                    name="Individual simulations",
                    x=result.position_kb,
                    y=profile,
                    color=color,
                    opacity=0.22,
                    width=max(0.55, float(line_width) * 0.55),
                    show_in_legend=index == 0,
                    z=2,
                )
                for index, profile in enumerate(profiles[:count])
            )
            ensemble_active = bool(individual_lines)
        mean = np.asarray(values, dtype=float).reshape(-1)
        if standard_deviation is not None and show_standard_deviation:
            bands = (
                PreviewBand(
                    name="Mean ± SD",
                    x=result.position_kb,
                    lower=mean - standard_deviation,
                    upper=mean + standard_deviation,
                    color=color,
                    opacity=0.16,
                ),
            )
            ensemble_active = True
        if ensemble_active:
            lines = (
                *individual_lines,
                PreviewLine("Mean", result.position_kb, mean, color, z=4),
            )
    return build_preview_plot_options(
        plot_type=plot_type,
        title=title.strip() or output_label,
        x_label=x_axis_label.strip()
        or (
            spec.y_label
            if plot_type == "Histogram"
            else ""
            if plot_type == "Boxplot"
            else spec.x_label
        ),
        y_label=y_axis_label.strip() or ("Density" if density else "Count")
        if plot_type == "Histogram"
        else y_axis_label.strip() or spec.y_label,
        color=color,
        palette=palette,
        histogram_bins=histogram_bins,
        density=density,
        show_legend=show_legend,
        show_colorbar=show_colorbar,
        log_y=log_y,
        log_z=log_z,
        invert_y=invert_y,
        y_limits=y_limits,
        series_label=(
            output_label
            if plot_type == "Heatmap"
            else series_label.strip() or ("Mean" if plot_type == "Line" else output_label)
        ),
        values=values.T if spec.kind == "xt" else values,
        x_axis=result.position_kb if spec.kind == "xt" else axis,
        y_axis=result.time_grid if spec.kind == "xt" else None,
        lines=lines,
        bands=bands,
        legend_position=legend_position,
        line_width=line_width,
        show_markers=show_markers,
        show_grid=show_grid,
        interactive=spec.kind not in {"time", "xt"} and not ensemble_active,
        log_x=log_x,
        invert_x=invert_x,
        invert_z=invert_z,
        show_histogram_statistics=show_histogram_statistics,
        histogram_statistics_position=histogram_statistics_position,
        histogram_bin_size=histogram_bin_size,
        histogram_x_limits=histogram_x_limits,
        histogram_touching=histogram_touching,
        histogram_boundary=histogram_boundary,
    )


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
    legend_position: str,
    show_colorbar: bool,
    log_y: bool,
    log_z: bool,
    invert_y: bool,
    y_limits: tuple[float, float] | None,
    on_download: Callable[[str], Any],
    line_width: float = 1.8,
    show_markers: bool = False,
    show_grid: bool = True,
    log_x: bool = False,
    invert_x: bool = False,
    invert_z: bool = False,
    show_histogram_statistics: bool = False,
    histogram_statistics_position: str = "Top right",
    histogram_bin_size: float | None = None,
    histogram_x_limits: tuple[float, float] | None = None,
    histogram_touching: bool = True,
    histogram_boundary: str = "left",
    show_individual_profiles: bool = False,
    individual_profile_count: int = 10,
    show_standard_deviation: bool = False,
) -> Any:
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
        legend_position=legend_position,
        show_colorbar=show_colorbar,
        log_y=log_y,
        log_z=log_z,
        invert_y=invert_y,
        y_limits=y_limits,
        line_width=line_width,
        show_markers=show_markers,
        show_grid=show_grid,
        log_x=log_x,
        invert_x=invert_x,
        invert_z=invert_z,
        show_histogram_statistics=show_histogram_statistics,
        histogram_statistics_position=histogram_statistics_position,
        histogram_bin_size=histogram_bin_size,
        histogram_x_limits=histogram_x_limits,
        histogram_touching=histogram_touching,
        histogram_boundary=histogram_boundary,
        show_individual_profiles=show_individual_profiles,
        individual_profile_count=individual_profile_count,
        show_standard_deviation=show_standard_deviation,
    )
    return render_preview_chart(
        options,
        title=title.strip() or output_label,
        marker="simulation-result-plot",
        on_download=on_download,
    )


def render(_navigate: Callable[[str], None]) -> None:
    """Render the local simulation controls and live preview."""

    running = False
    active_cancel: Event | None = None
    restored_entry = consume_cached_workflow("simulation")
    restored_result = restored_entry.payload if restored_entry is not None else None
    latest_result: ConstantSimulationResult | None = (
        restored_result if isinstance(restored_result, ConstantSimulationResult) else None
    )
    result_chart: Any | None = None
    y_limits_automatic = True
    updating_y_limits = False
    source_region_refresh: dict[str, Callable[[], Any]] = {"call": lambda: None}

    page_intro(
        "Simulation",
        "Simulate DNA replication",
        "Run stochastic replication ensembles and inspect their outputs.",
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
                    with ui.element("div").classes("dn-simulation-data-grid"):
                        with ui.column().classes("dn-setting-group"):
                            ui.label("Firing rate").classes("dn-setting-group-title")
                            firing_source = _render_simulation_source(
                                "Firing rate",
                                constant=0.01,
                                units="min⁻¹ per bin",
                                on_profile_change=lambda: source_region_refresh["call"](),
                            )
                        with ui.column().classes("dn-setting-group"):
                            ui.label("Fork speed").classes("dn-setting-group-title")
                            speed_source = _render_simulation_source(
                                "Fork speed",
                                constant=1.4,
                                units="kb/min",
                                on_profile_change=lambda: source_region_refresh["call"](),
                            )
                with expansion("Region", icon="straighten"):
                    with two_column_fields():
                        resolution_x = (
                            ui.number(
                                label="Resolution (kb/bin)", value=1.0, min=0.000001, step=0.1
                            )
                            .props("dense")
                            .classes("dn-field")
                            .mark("simulation-resolution-x")
                        )
                        resolution_t = (
                            ui.number(
                                label="Time step (min/bin)", value=5.0, min=0.000001, step=1.0
                            )
                            .props("dense")
                            .classes("dn-field")
                            .mark("simulation-resolution-t")
                        )
                        region_start = (
                            ui.number(label="Start (kb)", value=0.0, min=0.0, step=1.0)
                            .props("dense")
                            .classes("dn-field")
                            .mark("simulation-region-start")
                        )
                        region_end = (
                            ui.number(
                                label="End (kb, exclusive)", value=1000.0, min=0.000001, step=1.0
                            )
                            .props("dense")
                            .classes("dn-field")
                            .mark("simulation-region-end")
                        )

                    region_sync_state: dict[str, tuple[int, ...] | None] = {"mismatch": None}

                    def sync_region_to_profiles() -> None:
                        sizes: set[int] = set()
                        for source, source_label in (
                            (firing_source, "Firing rate"),
                            (speed_source, "Fork speed"),
                        ):
                            if str(source.mode.value) != "profile" or not source.inspections:
                                continue
                            try:
                                value = _materialize_simulation_source(
                                    source,
                                    label=source_label,
                                    missing_policy="zero",
                                )
                            except (OSError, TypeError, ValueError):
                                continue
                            size = _uploaded_spatial_size(source, value)
                            if size is not None:
                                sizes.add(size)
                        if len(sizes) > 1:
                            mismatch = tuple(sorted(sizes))
                            if region_sync_state["mismatch"] != mismatch:
                                ui.notify(
                                    "Uploaded spatial profiles have different lengths: "
                                    + ", ".join(f"{size:,} bins" for size in mismatch)
                                    + ". Choose matching arrays before simulating.",
                                    type="negative",
                                )
                            region_sync_state["mismatch"] = mismatch
                            return
                        region_sync_state["mismatch"] = None
                        if not sizes:
                            return
                        raw_resolution = resolution_x.value
                        if raw_resolution is None:
                            return
                        try:
                            spatial_resolution = float(raw_resolution)
                        except (TypeError, ValueError):
                            return
                        if not math.isfinite(spatial_resolution) or spatial_resolution <= 0:
                            return
                        region_start.set_value(0.0)
                        region_end.set_value(sizes.pop() * spatial_resolution)

                    source_region_refresh["call"] = sync_region_to_profiles
                    resolution_x.on_value_change(sync_region_to_profiles)
                    sync_region_to_profiles()

                with expansion("Simulation settings", icon="tune"):
                    with ui.element("div").classes("dn-simulation-settings-grid"):
                        with ui.column().classes("dn-setting-group"):
                            ui.label("Ensemble").classes("dn-setting-group-title")
                            replicates = (
                                ui.number(
                                    label="Replicates",
                                    value=20,
                                    min=1,
                                    max=MAX_REPLICATES,
                                    step=1,
                                )
                                .props("dense")
                                .classes("dn-field dn-field-compact")
                                .mark("simulation-replicates")
                            )
                            geometry = (
                                ui.select(
                                    ["linear", "periodic"],
                                    value="linear",
                                    label="Geometry",
                                )
                                .props("dense options-dense")
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
                            with ui.element("div").classes("dn-setting-inline"):
                                fixed_seed = ui.checkbox("Fixed seed", value=False)
                                seed = (
                                    ui.number(label="Seed", value=1, min=0, step=1)
                                    .props("dense")
                                    .classes("dn-field dn-field-xs")
                                    .mark("simulation-seed")
                                )
                            seed.bind_visibility_from(fixed_seed, "value")

                        with ui.column().classes("dn-setting-group"):
                            ui.label("Timing & outputs").classes("dn-setting-group-title")
                            time_window = (
                                ui.number(
                                    label="Time window (min)",
                                    value=300.0,
                                    min=0.000001,
                                    step=10.0,
                                )
                                .props("dense")
                                .classes("dn-field dn-field-compact")
                            )
                            max_replication_time = (
                                ui.number(
                                    label="Maximum time (min)",
                                    value=1200.0,
                                    min=0.000001,
                                    step=50.0,
                                )
                                .props("dense")
                                .classes("dn-field dn-field-compact")
                            )
                            track_time_outputs = ui.checkbox(
                                "Track time courses",
                                value=bool(
                                    latest_result is not None
                                    and getattr(latest_result, "time_outputs", False)
                                ),
                            ).mark("simulation-track-time-outputs")
                            track_time_outputs.tooltip(
                                "Retain replicated fraction, fork counts, and firing-factor "
                                "availability across time."
                            )
                            track_detailed_outputs = ui.checkbox(
                                "Track space-time outputs",
                                value=bool(
                                    latest_result is not None and latest_result.detailed_outputs
                                ),
                            ).mark("simulation-track-detailed-outputs")
                            track_detailed_outputs.tooltip(
                                "Retain position-by-time heatmaps; these require more memory."
                            )

                        with ui.column().classes("dn-setting-group dn-fork-behaviour-group"):
                            ui.label("Fork behaviour").classes("dn-setting-group-title")
                            with ui.element("div").classes("dn-fork-behaviour-grid"):
                                with (
                                    ui.element("div")
                                    .classes("dn-setting-subgroup")
                                    .mark("simulation-stalling-settings")
                                ):
                                    ui.label("Stalling").classes("dn-setting-subgroup-title")
                                    with ui.element("div").classes("dn-setting-inline"):
                                        stall_rate = (
                                            ui.number(
                                                label="Stall rate (stalls/kb)",
                                                value=0.0,
                                                min=0.0,
                                                step=0.001,
                                            )
                                            .props("dense")
                                            .classes("dn-field dn-field-compact")
                                            .mark("simulation-stall-rate-per-kb")
                                        )
                                        non_recovering = ui.checkbox("Non-recovering", value=True)
                                    stall_duration = (
                                        ui.number(
                                            label="Recovery time (min)",
                                            value=10.0,
                                            min=0.000001,
                                            step=1.0,
                                        )
                                        .props("dense")
                                        .classes("dn-field dn-field-compact")
                                        .mark("simulation-stall-duration")
                                    )
                                    stall_duration.bind_visibility_from(
                                        non_recovering,
                                        "value",
                                        backward=lambda value: not value,
                                    )
                                    stall_probability = (
                                        ui.label()
                                        .classes("dn-settings-help")
                                        .mark("simulation-stall-probability")
                                    )

                                    def sync_stall_probability() -> None:
                                        try:
                                            rate_value = stall_rate.value
                                            bin_width_value = resolution_x.value
                                            if rate_value is None or bin_width_value is None:
                                                raise ValueError
                                            rate = float(rate_value)
                                            bin_width = float(bin_width_value)
                                            if (
                                                not math.isfinite(rate)
                                                or rate < 0.0
                                                or not math.isfinite(bin_width)
                                                or bin_width <= 0.0
                                            ):
                                                raise ValueError
                                        except (TypeError, ValueError):
                                            stall_probability.text = (
                                                "Converted automatically using the spatial resolution."
                                            )
                                            return
                                        probability = -math.expm1(-(rate * bin_width))
                                        stall_probability.text = (
                                            f"At {bin_width:g} kb/bin: {probability:.1%} "
                                            "probability per directed bin."
                                        )

                                    stall_rate.on_value_change(sync_stall_probability)
                                    resolution_x.on_value_change(sync_stall_probability)
                                    sync_stall_probability()

                                with (
                                    ui.element("div")
                                    .classes("dn-setting-subgroup")
                                    .mark("simulation-factor-pool-settings")
                                ):
                                    ui.label("Firing-factor pool").classes(
                                        "dn-setting-subgroup-title"
                                    )
                                    with ui.element("div").classes("dn-setting-inline"):
                                        finite_pool = ui.checkbox("Finite pool", value=False).mark(
                                            "simulation-finite-pool"
                                        )
                                        pool_size = (
                                            ui.number(label="Pool size", value=100, min=2, step=1)
                                            .props("dense")
                                            .classes("dn-field dn-field-xs")
                                        )
                                    pool_size.bind_visibility_from(finite_pool, "value")
                                    with (
                                        ui.element("div")
                                        .classes("dn-setting-inline")
                                        .mark("simulation-pool-fields") as pool_fields
                                    ):
                                        recycle_rate = (
                                            ui.number(
                                                label="Recycle rate",
                                                value=0.0,
                                                min=0.0,
                                                step=0.001,
                                            )
                                            .props("dense")
                                            .classes("dn-field dn-field-compact")
                                        )
                                        factor_coupling = ui.checkbox("Fork coupling", value=False)
                                    pool_fields.bind_visibility_from(finite_pool, "value")

                def current_y_limits() -> tuple[float, float] | None:
                    if y_minimum.value is None or y_maximum.value is None:
                        return None
                    limits = (
                        _number(y_minimum, "Y minimum"),
                        _number(y_maximum, "Y maximum"),
                    )
                    if limits[1] <= limits[0]:
                        raise ValueError("Y maximum must be greater than Y minimum.")
                    return limits

                def active_plot_type() -> str:
                    selected = str(output.value or "Replication timing")
                    spec = _OUTPUT_SPECS.get(selected, _BASE_OUTPUTS["Replication timing"])
                    requested_view = str(view.value or "Default")
                    if spec.kind == "xt" and requested_view == "Default":
                        return "Heatmap"
                    if requested_view == "Boxplot":
                        return "Boxplot"
                    if spec.kind == "distribution" or requested_view == "Histogram":
                        return "Histogram"
                    return "Line"

                def populate_automatic_y_limits(*, redraw: bool = True) -> None:
                    """Restore the selected output's data-derived Y range."""

                    nonlocal updating_y_limits, y_limits_automatic
                    y_limits_automatic = True
                    if latest_result is None:
                        return
                    if not plotting_settings_ready(
                        plot_settings,
                        plot_type=active_plot_type(),
                        allow_empty_y_limits=y_limits_automatic,
                    ):
                        return
                    profile_count = selected_profile_count()
                    if (
                        bool(show_individual_profiles.value)
                        and available_profile_count() > 0
                        and profile_count is None
                    ):
                        return
                    try:
                        limits = _automatic_y_limits(
                            latest_result,
                            output_label=str(output.value or "Replication timing"),
                            view_mode=str(view.value or "Default"),
                            histogram_bins=_integer(histogram_bins, "Histogram bins"),
                            density=bool(density.value),
                            log_y=bool(log_y.value),
                            log_x=bool(log_x.value),
                            histogram_bin_size=(
                                float(plot_settings.histogram_bin_size.value)
                                if plot_settings.histogram_bin_size.value is not None
                                else None
                            ),
                            histogram_x_limits=histogram_x_limits_from_controls(
                                plot_settings,
                                log_x=bool(log_x.value),
                            ),
                            histogram_boundary=str(
                                plot_settings.histogram_boundary.value or "left"
                            ),
                            show_individual_profiles=bool(
                                show_individual_profiles.value
                            ),
                            individual_profile_count=profile_count or 1,
                            show_standard_deviation=bool(
                                show_standard_deviation.value
                            ),
                        )
                    except ValueError:
                        limits = None
                    updating_y_limits = True
                    try:
                        y_minimum.value = None if limits is None else float(f"{limits[0]:.10g}")
                        y_maximum.value = None if limits is None else float(f"{limits[1]:.10g}")
                        y_minimum.update()
                        y_maximum.update()
                    finally:
                        updating_y_limits = False
                    if redraw:
                        render_latest_result()

                def use_custom_y_limits() -> None:
                    nonlocal y_limits_automatic
                    if updating_y_limits:
                        return
                    y_limits_automatic = False
                    render_latest_result()

                def refresh_axis_range() -> None:
                    if y_limits_automatic:
                        populate_automatic_y_limits()
                    else:
                        render_latest_result()

                async def download_plot(file_format: str) -> None:
                    if latest_result is None or result_chart is None:
                        ui.notify("Run a simulation before downloading a plot.", type="warning")
                        return
                    try:
                        selected_label = str(output.value or "Replication timing")
                        spec = _OUTPUT_SPECS[selected_label]
                        normalized_dpi = int(get_export_settings().plot_dpi)
                        pixel_ratio = max(1.0, min(4.0, normalized_dpi / 96.0))
                        capture_format = (
                            "png" if str(file_format).casefold() == "png" else "svg"
                        )
                        capture_options: dict[str, Any] = {"type": capture_format}
                        if capture_format == "png":
                            capture_options["pixelRatio"] = pixel_ratio
                        data_uri = await result_chart.run_chart_method(
                            "getDataURL",
                            capture_options,
                            timeout=10,
                        )
                        artifact = await run.io_bound(
                            build_chart_capture_export,
                            data_uri,
                            file_format,
                            filename=f"dnascape_{spec.key}_plot",
                            dpi=normalized_dpi,
                        )
                        if artifact is None or not artifact.data:
                            raise RuntimeError("Plot export stopped before returning data.")
                    except (RuntimeError, TimeoutError, TypeError, ValueError) as exc:
                        ui.notify(str(exc), type="negative")
                        return
                    cache_artifact(
                        artifact,
                        source="Simulation",
                        kind="plot",
                        detail=selected_label,
                        navigate_to="simulation",
                    )
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
                            str(controls.file_type.value or "NPY"),
                            filename=str(
                                controls.file_name.value
                                or f"dnascape_simulation_{_OUTPUT_SPECS[selected_label].key}"
                            ),
                            compression=(
                                get_export_settings().raw_compression
                                if controls.compress.value
                                else "none"
                            ),
                        )
                        if artifact is None or not artifact.data:
                            raise RuntimeError("Raw export stopped before returning data.")
                    except Exception as exc:
                        ui.notify(str(exc), type="negative")
                        return
                    cache_artifact(
                        artifact,
                        source="Simulation",
                        kind="raw",
                        detail=selected_label,
                        navigate_to="simulation",
                    )
                    ui.download(artifact.data, artifact.filename, artifact.media_type)

                def render_latest_result() -> None:
                    nonlocal result_chart
                    if latest_result is None:
                        return
                    if not plotting_settings_ready(
                        plot_settings,
                        plot_type=active_plot_type(),
                        allow_empty_y_limits=y_limits_automatic,
                    ):
                        return
                    profile_count = selected_profile_count()
                    if (
                        bool(show_individual_profiles.value)
                        and available_profile_count() > 0
                        and profile_count is None
                    ):
                        return
                    result_chart = None
                    result_slot.clear()
                    with result_slot:
                        try:
                            result_chart = _render_result_plot(
                                latest_result,
                                output_label=str(output.value or "Replication timing"),
                                title=str(title.value or ""),
                                series_label=str(output.value or "Replication timing"),
                                x_axis_label=str(x_axis_label.value or ""),
                                y_axis_label=str(y_axis_label.value or ""),
                                color=str(line_or_bar_color.value or "#2c7f77"),
                                palette=str(heatmap_palette.value or "viridis"),
                                view_mode=str(view.value or "Default"),
                                histogram_bins=_integer(histogram_bins, "Histogram bins"),
                                density=bool(density.value),
                                show_legend=bool(show_legend.value),
                                legend_position=str(legend_position.value or "Top right"),
                                show_histogram_statistics=bool(
                                    show_histogram_statistics.value
                                ),
                                histogram_statistics_position=str(
                                    histogram_statistics_position.value or "Top right"
                                ),
                                histogram_bin_size=(
                                    float(plot_settings.histogram_bin_size.value)
                                    if plot_settings.histogram_bin_size.value is not None
                                    else None
                                ),
                                histogram_x_limits=histogram_x_limits_from_controls(
                                    plot_settings,
                                    log_x=bool(log_x.value),
                                ),
                                histogram_touching=bool(
                                    plot_settings.histogram_touching.value
                                ),
                                histogram_boundary=str(
                                    plot_settings.histogram_boundary.value or "left"
                                ),
                                show_individual_profiles=bool(
                                    show_individual_profiles.value
                                ),
                                individual_profile_count=profile_count or 1,
                                show_standard_deviation=bool(
                                    show_standard_deviation.value
                                ),
                                line_width=float(_number(line_width, "Curve thickness")),
                                show_markers=False,
                                show_grid=bool(show_grid.value),
                                show_colorbar=bool(colorbar.value),
                                log_y=bool(log_y.value),
                                log_z=bool(log_z.value),
                                invert_y=bool(invert_y.value),
                                log_x=bool(log_x.value),
                                invert_x=bool(invert_x.value),
                                invert_z=bool(invert_z.value),
                                # Automatic fields contain outward-rounded bounds
                                # (or a fixed physical domain); applying the same
                                # values keeps the editor and preview in agreement.
                                y_limits=current_y_limits(),
                                on_download=download_plot,
                            )
                        except (RuntimeError, TypeError, ValueError) as exc:
                            result_chart = None
                            ui.notify(str(exc), type="negative")
                            preview_plot(
                                (
                                    "No inter-origin distances were recorded; at least two "
                                    "origins must fire in a realization."
                                    if _OUTPUT_SPECS[
                                        str(output.value or "Replication timing")
                                    ].key
                                    == "inter_origin_distances"
                                    else "Output unavailable"
                                ),
                                kind="simulation",
                            )

                async def simulate() -> None:
                    nonlocal active_cancel, latest_result, result_chart, running
                    if running:
                        if active_cancel is not None and not active_cancel.is_set():
                            active_cancel.set()
                            status.text = "Stopping simulation..."
                            simulate_button.disable()
                        return

                    cancel_event = Event()

                    def check_cancelled() -> None:
                        if cancel_event.is_set():
                            raise CancelledError("Simulation stopped.")

                    frozen_controls = [
                        control
                        for control in (
                            firing_source.mode,
                            firing_source.constant,
                            speed_source.mode,
                            speed_source.constant,
                            resolution_x,
                            resolution_t,
                            region_start,
                            region_end,
                            replicates,
                            geometry,
                            missing_values,
                            fixed_seed,
                            seed,
                            time_window,
                            max_replication_time,
                            track_time_outputs,
                            track_detailed_outputs,
                            stall_rate,
                            non_recovering,
                            stall_duration,
                            finite_pool,
                            pool_size,
                            recycle_rate,
                            factor_coupling,
                        )
                        if bool(getattr(control, "enabled", False))
                    ]

                    def release_simulation_controls() -> None:
                        nonlocal active_cancel, running
                        running = False
                        active_cancel = None
                        set_data_upload_locked(firing_source.uploader, False)
                        set_data_upload_locked(speed_source.uploader, False)
                        for control in frozen_controls:
                            control.enable()
                        simulate_button.set_text("Simulate")
                        simulate_button.set_icon("play_arrow")
                        simulate_button.enable()

                    running = True
                    active_cancel = cancel_event
                    set_data_upload_locked(firing_source.uploader, True)
                    set_data_upload_locked(speed_source.uploader, True)
                    for control in frozen_controls:
                        control.disable()
                    simulate_button.set_text("Stop")
                    simulate_button.set_icon("stop")
                    progress.visible = True
                    progress.value = 0.01
                    status.text = "Reading inputs..."

                    try:
                        missing_policy = str(missing_values.value)
                        preparation = asyncio.create_task(
                            run.io_bound(
                                lambda: (
                                    _materialize_simulation_source(
                                        firing_source,
                                        label="Firing rate",
                                        missing_policy=missing_policy,
                                    ),
                                    _materialize_simulation_source(
                                        speed_source,
                                        label="Fork speed",
                                        missing_policy=missing_policy,
                                    ),
                                )
                            )
                        )
                        while not preparation.done():
                            if cancel_event.is_set():
                                preparation.cancel()
                                try:
                                    await preparation
                                except asyncio.CancelledError:
                                    pass
                                check_cancelled()
                            await asyncio.sleep(0.04)
                        source_values = await preparation
                        check_cancelled()
                        if source_values is None:
                            raise RuntimeError(
                                "Input preparation stopped before returning simulation data."
                            )
                        firing_rate, fork_speed = source_values
                        resolution_x_value = _number(resolution_x, "Spatial resolution")
                        region_start_value = _number(region_start, "Region start")
                        spatial_sizes = {
                            size
                            for size in (
                                _uploaded_spatial_size(firing_source, firing_rate),
                                _uploaded_spatial_size(speed_source, fork_speed),
                            )
                            if size is not None
                        }
                        if len(spatial_sizes) > 1:
                            sizes = ", ".join(f"{size:,}" for size in sorted(spatial_sizes))
                            raise ValueError(
                                f"Firing-rate and fork-speed spatial axes disagree ({sizes} bins)."
                            )
                        if spatial_sizes:
                            region_end_value = region_start_value + (
                                spatial_sizes.pop() * resolution_x_value
                            )
                            region_end.value = region_end_value
                            region_end.update()
                        else:
                            region_end_value = _number(region_end, "Region end")
                        request = ConstantSimulationRequest(
                            firing_rate=firing_rate,
                            fork_speed=fork_speed,
                            resolution_x=resolution_x_value,
                            resolution_t=_number(resolution_t, "Time step"),
                            region_start_kb=region_start_value,
                            region_end_kb=region_end_value,
                            replicates=_integer(replicates, "Replicates"),
                            geometry=str(geometry.value),
                            time_window=_number(time_window, "Time window"),
                            max_replication_time=_number(
                                max_replication_time, "Maximum replication time"
                            ),
                            missing_values=str(missing_values.value),
                            seed=_integer(seed, "Seed") if fixed_seed.value else None,
                            stall_rate_per_kb=_number(
                                stall_rate,
                                "Stall rate (stalls/kb)",
                            ),
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
                            track_time_outputs=bool(track_time_outputs.value),
                            track_detailed_outputs=bool(track_detailed_outputs.value),
                        )
                        check_cancelled()
                    except CancelledError:
                        progress.visible = False
                        status.text = "Simulation stopped"
                        stats_container.visible = False
                        result_slot.clear()
                        with result_slot:
                            preview_plot("Simulation stopped", kind="simulation")
                        release_simulation_controls()
                        return
                    except Exception as exc:
                        ui.notify(str(exc), type="negative")
                        status.text = "Check the settings"
                        progress.visible = False
                        release_simulation_controls()
                        return

                    progress_state = _ProgressState()
                    progress.value = 0.02
                    status.text = "Preparing simulation..."
                    latest_result = None
                    result_chart = None
                    sync_view_controls()
                    stats_container.visible = False
                    result_slot.clear()
                    with result_slot:
                        ui.label("Simulation is running in the background...").classes("dn-muted")
                    try:
                        worker = asyncio.create_task(
                            run.io_bound(
                                run_constant_simulation,
                                request,
                                progress_state.report,
                                check_cancelled,
                            )
                        )
                        while not worker.done():
                            if cancel_event.is_set():
                                status.text = "Stopping simulation..."
                            else:
                                value, message = progress_state.snapshot()
                                progress.value = value
                                status.text = message
                            await asyncio.sleep(0.08)
                        result = await worker
                        if result is None:
                            check_cancelled()
                            raise RuntimeError(
                                "The simulation worker stopped before returning data."
                            )
                        progress.value = 1.0
                        status.text = f"Complete in {result.elapsed_seconds:.2f} s"
                        replicate_stats.text = f"{result.replicates:,}"
                        resolution_stats.text = (
                            f"{request.resolution_x:g} kb/bin · {request.resolution_t:g} min/bin"
                        )
                        region_stats.text = (
                            f"{request.region_start_kb:g}–{request.region_end_kb:g} kb"
                        )
                        geometry_stats.text = result.geometry.title()
                        latest_result = result
                        sync_output_options()
                        if result.inter_origin_distances_truncated:
                            retained = result.outputs["inter_origin_distances"].size
                            ui.notify(
                                "Inter-origin distance output retained "
                                f"{retained:,} of {result.inter_origin_distance_count:,} "
                                "intervals for local preview and download.",
                                type="warning",
                            )
                        histogram_bin_sync.sync_to_data(redraw=False)
                        sync_view_controls()
                        populate_automatic_y_limits(redraw=False)
                        stats_container.visible = True
                        render_latest_result()
                        cache_workflow(
                            result,
                            name="Simulation result",
                            source="Simulation",
                            detail=(
                                f"{result.replicates:,} replicates | "
                                f"{request.region_start_kb:g}-{request.region_end_kb:g} kb | "
                                f"{result.geometry.title()}"
                            ),
                            navigate_to="simulation",
                        )
                    except CancelledError:
                        progress.visible = False
                        status.text = "Simulation stopped"
                        stats_container.visible = False
                        latest_result = None
                        result_slot.clear()
                        with result_slot:
                            preview_plot(
                                "Simulation stopped",
                                kind="simulation",
                            )
                    except Exception as exc:  # keep malformed local jobs inside the preview
                        progress.visible = False
                        status.text = "Simulation could not be completed"
                        stats_container.visible = False
                        ui.notify(str(exc) or type(exc).__name__, type="negative")
                        result_slot.clear()
                        with result_slot:
                            preview_plot(
                                "No result",
                                kind="simulation",
                            )
                    finally:
                        release_simulation_controls()

                with action_dock():
                    simulate_button = (
                        ui.button("Simulate", icon="play_arrow", on_click=simulate)
                        .props("no-caps")
                        .classes("dn-action-primary")
                        .mark("action-simulate")
                    )

        with ui.column().classes("dn-preview-stack"):
            with panel(extra_classes="dn-preview-card"):
                with (
                    ui.element("div")
                    .classes("dn-preview-heading-row")
                    .mark("simulation-preview-header-row")
                ):
                    panel_title("Preview")
                with (
                    ui.element("div")
                    .classes("dn-stat-grid dn-simulation-summary-grid")
                    .mark("simulation-preview-output-bar")
                ):
                    with (
                        ui.element("div")
                        .classes("dn-stat dn-simulation-output-stat")
                        .mark("simulation-output-stat")
                    ):
                        ui.label("Output").classes("dn-stat-label")
                        output = (
                            ui.select(
                                [
                                    *list(_STANDARD_OUTPUTS),
                                    *(
                                        [
                                            label
                                            for label, spec in _TIME_OUTPUTS.items()
                                            if spec.key in latest_result.outputs
                                        ]
                                        if latest_result is not None
                                        and getattr(latest_result, "time_outputs", False)
                                        else []
                                    ),
                                    *(
                                        [
                                            label
                                            for label, spec in _SPACETIME_OUTPUTS.items()
                                            if spec.key in latest_result.outputs
                                        ]
                                        if latest_result is not None
                                        and latest_result.detailed_outputs
                                        else []
                                    ),
                                ],
                                value="Replication timing",
                            )
                            .props('dense borderless options-dense aria-label="Preview output"')
                            .classes("dn-field dn-simulation-output-select")
                            .mark("simulation-output")
                        )
                    with (
                        ui.element("div")
                        .classes("dn-simulation-run-stats")
                        .mark("simulation-stats") as stats_container
                    ):
                        stats = stat_strip(
                            (
                                ("Replicates", ""),
                                ("Resolution", ""),
                                ("Region", ""),
                                ("Geometry", ""),
                            )
                        )
                replicate_stats = stats["Replicates"]
                resolution_stats = stats["Resolution"]
                region_stats = stats["Region"]
                geometry_stats = stats["Geometry"]
                stats_container.visible = False
                progress = (
                    ui.linear_progress(value=0.0, show_value=False)
                    .props("rounded instant-feedback")
                    .classes("dn-simulation-progress")
                    .mark("simulation-progress")
                )
                progress.visible = False
                status = ui.label("Ready").classes("dn-muted").mark("simulation-status")
                with (
                    ui.element("div")
                    .classes("w-full")
                    .mark("simulation-result-container") as result_slot
                ):
                    preview_plot(
                        "No result yet",
                        kind="simulation",
                    )

                plot_settings = preview_settings_controls(
                    initial_title="Replication timing",
                    initial_x_label="Position (kb)",
                    initial_y_label="Replication timing",
                    on_reset_y_limits=lambda: populate_automatic_y_limits(),
                    markers={
                        "container": "simulation-preview-settings",
                        "view": "simulation-view",
                        "title": "simulation-preview-title",
                        "x_axis_label": "simulation-x-axis-label",
                        "y_axis_label": "simulation-y-axis-label",
                        "primary_color": "simulation-line-or-bar-color",
                        "secondary_color": "simulation-second-curve-color",
                        "line_width": "simulation-line-width",
                        "legend_position": "simulation-legend-position",
                        "show_grid": "simulation-show-grid",
                        "heatmap_palette": "simulation-heatmap-palette",
                        "log_x": "simulation-log-x",
                        "invert_x": "simulation-invert-x",
                        "log_y": "simulation-log-y",
                        "invert_y": "simulation-invert-y",
                        "log_z": "simulation-log-z",
                        "invert_z": "simulation-invert-z",
                        "histogram_bins": "simulation-histogram-bins",
                        "histogram_bin_size": "simulation-histogram-bin-size",
                        "histogram_touching": "simulation-histogram-touching",
                        "histogram_boundary": "simulation-histogram-boundary",
                        "ensemble_profile_options": "simulation-ensemble-profile-options",
                        "show_individual_profiles": "simulation-show-individual-profiles",
                        "individual_profile_count": "simulation-individual-profile-count",
                        "show_standard_deviation": "simulation-show-standard-deviation",
                        "histogram_x_minimum": "simulation-histogram-x-minimum",
                        "histogram_x_maximum": "simulation-histogram-x-maximum",
                        "reset_histogram_x_limits": "simulation-reset-histogram-x-limits",
                        "show_histogram_statistics": "simulation-show-histogram-statistics",
                        "histogram_statistics_position": (
                            "simulation-histogram-statistics-position"
                        ),
                        "density": "simulation-density",
                        "colorbar": "simulation-colorbar",
                        "y_minimum": "simulation-y-minimum",
                        "y_maximum": "simulation-y-maximum",
                        "reset_y_limits": "simulation-reset-y-limits",
                    },
                )
                view = plot_settings.view
                title = plot_settings.title
                x_axis_label = plot_settings.x_axis_label
                y_axis_label = plot_settings.y_axis_label
                line_or_bar_color = plot_settings.primary_color
                heatmap_palette = plot_settings.heatmap_palette
                log_x = plot_settings.log_x
                invert_x = plot_settings.invert_x
                log_z = plot_settings.log_z
                invert_z = plot_settings.invert_z
                line_width = plot_settings.line_width
                histogram_bins = plot_settings.histogram_bins
                histogram_touching = plot_settings.histogram_touching
                histogram_boundary = plot_settings.histogram_boundary
                show_individual_profiles = plot_settings.show_individual_profiles
                individual_profile_count = plot_settings.individual_profile_count
                show_standard_deviation = plot_settings.show_standard_deviation
                show_histogram_statistics = plot_settings.show_histogram_statistics
                histogram_statistics_position = plot_settings.histogram_statistics_position
                show_legend = plot_settings.show_legend
                legend_position = plot_settings.legend_position
                show_grid = plot_settings.show_grid
                density = plot_settings.density
                colorbar = plot_settings.colorbar
                log_y = plot_settings.log_y
                invert_y = plot_settings.invert_y
                y_minimum = plot_settings.y_minimum
                y_maximum = plot_settings.y_maximum
                raw_controls = raw_output_controls(
                    file_name="dnascape_simulation_replication_timing",
                    default_type="NPY",
                    on_download=download_raw,
                )

                def available_profile_count() -> int:
                    if latest_result is None:
                        return 0
                    profiles, _standard_deviation = _replication_profile_arrays(latest_result)
                    return 0 if profiles is None else int(profiles.shape[0])

                def standard_deviation_available() -> bool:
                    if latest_result is None:
                        return False
                    _profiles, standard_deviation = _replication_profile_arrays(latest_result)
                    return standard_deviation is not None

                def selected_profile_count() -> int | None:
                    return _bounded_profile_count(
                        individual_profile_count.value,
                        available_profile_count(),
                    )

                def sync_view_controls() -> None:
                    """Show controls for the representation rendered by Default."""

                    selected = str(output.value or "Replication timing")
                    spec = _OUTPUT_SPECS.get(
                        selected,
                        _BASE_OUTPUTS["Replication timing"],
                    )
                    requested_view = str(view.value or "Default")
                    available_profiles = available_profile_count()
                    allow_individual_profiles = (
                        selected == "Replication timing"
                        and requested_view == "Default"
                        and available_profiles > 0
                    )
                    allow_standard_deviation = (
                        selected == "Replication timing"
                        and requested_view == "Default"
                        and standard_deviation_available()
                    )
                    if allow_individual_profiles:
                        maximum = min(_MAX_PREVIEW_PROFILES, available_profiles)
                        individual_profile_count._props["max"] = maximum
                        current_count = _bounded_profile_count(
                            individual_profile_count.value,
                            maximum,
                        )
                        if current_count is not None and (
                            individual_profile_count.value != current_count
                        ):
                            individual_profile_count.value = current_count
                        individual_profile_count.update()
                    if requested_view == "Default":
                        time_array = spec.kind == "xt"
                        histogram = spec.kind == "distribution"
                        boxplot = False
                    else:
                        time_array = False
                        histogram = requested_view == "Histogram"
                        boxplot = requested_view == "Boxplot"
                    sync_preview_settings_visibility(
                        plot_settings,
                        plot_type=(
                            "Heatmap"
                            if time_array
                            else "Boxplot"
                            if boxplot
                            else "Histogram"
                            if histogram
                            else "Line"
                        ),
                        allow_histogram=spec.kind != "xt",
                        allow_boxplot=spec.kind == "distribution",
                        allow_individual_profiles=allow_individual_profiles,
                        allow_standard_deviation=allow_standard_deviation,
                    )

                def sync_output_options() -> None:
                    """Expose only outputs retained by the latest simulation."""

                    labels = list(_STANDARD_OUTPUTS)
                    if latest_result is None:
                        if track_time_outputs.value:
                            labels.extend(_TIME_OUTPUTS)
                        if track_detailed_outputs.value:
                            labels.extend(_SPACETIME_OUTPUTS)
                    else:
                        labels.extend(
                            label
                            for label, spec in _TIME_OUTPUTS.items()
                            if spec.key in latest_result.outputs
                        )
                        labels.extend(
                            label
                            for label, spec in _SPACETIME_OUTPUTS.items()
                            if spec.key in latest_result.outputs
                        )
                    output.options = labels
                    selection_changed = output.value not in labels
                    if output.value not in labels:
                        output.value = "Replication timing"
                    output.update()
                    if selection_changed:
                        sync_plot_labels()
                    else:
                        sync_view_controls()

                def sync_plot_labels() -> None:
                    """Switch the existing result locally without starting a new run."""

                    selected = str(output.value or "Replication timing")
                    spec = _OUTPUT_SPECS.get(
                        selected,
                        _BASE_OUTPUTS["Replication timing"],
                    )
                    title.value = selected
                    raw_controls.file_name.value = f"dnascape_simulation_{spec.key}"
                    # "Default" resolves to the natural representation for the
                    # selected quantity: line, histogram, or space-time heatmap.
                    view.value = "Default"
                    sync_view_controls()
                    default_x_label, default_y_label = _default_axis_labels(
                        output_label=selected,
                        view_mode=str(view.value or "Default"),
                        density=bool(density.value),
                    )
                    x_axis_label.value = default_x_label
                    y_axis_label.value = default_y_label
                    for control in (
                        title,
                        x_axis_label,
                        y_axis_label,
                        view,
                        raw_controls.file_name,
                    ):
                        control.update()
                    if latest_result is not None:
                        histogram_bin_sync.sync_to_data(redraw=False)
                        populate_automatic_y_limits(redraw=False)
                        render_latest_result()

                def sync_view_axis_labels() -> None:
                    """Keep histogram quantity/count labels consistent with its view."""

                    sync_view_controls()
                    selected = str(output.value or "Replication timing")
                    default_x_label, default_y_label = _default_axis_labels(
                        output_label=selected,
                        view_mode=str(view.value or "Default"),
                        density=bool(density.value),
                    )
                    x_axis_label.value = default_x_label
                    y_axis_label.value = default_y_label
                    x_axis_label.update()
                    y_axis_label.update()
                    histogram_bin_sync.sync_to_data(redraw=False)
                    refresh_axis_range()

                def histogram_values() -> Any | None:
                    if latest_result is None:
                        return None
                    selected = str(output.value or "Replication timing")
                    spec = _OUTPUT_SPECS.get(selected, _BASE_OUTPUTS["Replication timing"])
                    return latest_result.outputs.get(spec.key)

                histogram_bin_sync = HistogramBinSynchronizer(
                    plot_settings,
                    values=histogram_values,
                    on_valid_change=refresh_axis_range,
                    log_x=lambda: bool(log_x.value),
                )

                def log_x_changed() -> None:
                    histogram_bin_sync.sync_to_data(redraw=False)
                    render_latest_result()

                def ensemble_display_changed() -> None:
                    sync_view_controls()
                    refresh_axis_range()

                def individual_profile_count_changed() -> None:
                    if selected_profile_count() is not None:
                        refresh_axis_range()

                track_time_outputs.on_value_change(sync_output_options)
                track_detailed_outputs.on_value_change(sync_output_options)
                output.on_value_change(sync_plot_labels)
                for plot_control in (
                    title,
                    x_axis_label,
                    y_axis_label,
                    line_or_bar_color,
                    line_width,
                    heatmap_palette,
                    invert_x,
                    log_z,
                    invert_z,
                    show_histogram_statistics,
                    histogram_statistics_position,
                    show_legend,
                    legend_position,
                    show_grid,
                    colorbar,
                    invert_y,
                    histogram_touching,
                ):
                    plot_control.on_value_change(render_latest_result)
                histogram_boundary.on_value_change(refresh_axis_range)
                show_individual_profiles.on_value_change(ensemble_display_changed)
                show_standard_deviation.on_value_change(ensemble_display_changed)
                individual_profile_count.on_value_change(individual_profile_count_changed)
                show_histogram_statistics.on_value_change(sync_view_controls)
                show_legend.on_value_change(sync_view_controls)
                view.on_value_change(sync_view_axis_labels)
                density.on_value_change(sync_view_axis_labels)
                log_x.on_value_change(log_x_changed)
                log_y.on_value_change(refresh_axis_range)
                y_minimum.on_value_change(use_custom_y_limits)
                y_maximum.on_value_change(use_custom_y_limits)
                sync_view_controls()

                if latest_result is not None:
                    histogram_bin_sync.sync_to_data(redraw=False)
                    position_steps = np.diff(latest_result.position_kb)
                    time_steps = np.diff(latest_result.time_grid)
                    position_resolution = (
                        float(np.median(position_steps)) if position_steps.size else 0.0
                    )
                    time_resolution = float(np.median(time_steps)) if time_steps.size else 0.0
                    replicate_stats.text = f"{latest_result.replicates:,}"
                    resolution_stats.text = (
                        f"{position_resolution:g} kb/bin | {time_resolution:g} min/bin"
                    )
                    region_stats.text = (
                        f"{latest_result.position_kb[0]:g}-{latest_result.position_kb[-1]:g} kb"
                    )
                    geometry_stats.text = latest_result.geometry.title()
                    for value in stats.values():
                        value.update()
                    stats_container.visible = True
                    status.text = "Restored from Cached Files"
                    populate_automatic_y_limits(redraw=False)
                    render_latest_result()

                def clear_simulation_output() -> None:
                    nonlocal latest_result, result_chart
                    latest_result = None
                    result_chart = None
                    progress.visible = False
                    stats_container.visible = False
                    status.text = "Ready"
                    sync_view_controls()
                    result_slot.clear()
                    with result_slot:
                        preview_plot("No result yet", kind="simulation")

                current_session_cache().register_output_clearer(
                    "simulation",
                    clear_simulation_output,
                )
