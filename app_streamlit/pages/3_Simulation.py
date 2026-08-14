"""Reproducible, explicitly executed Simulation workflow for DNAscape."""

from __future__ import annotations

import math
import time
from dataclasses import replace
from functools import partial
from typing import Any, Mapping

import numpy as np
import streamlit as st

from app_streamlit.app_shell import (
    render_control_panel_marker,
    render_page_intro,
    safe_name_stem,
)
from app_streamlit.exports import (
    array_export_target,
    figure_export_target,
    gzip_payload,
    serialize_array,
    serialize_figure,
)
from app_streamlit.jobs import (
    BackgroundJob,
    JobContext,
    any_background_job_running,
    cancel_background_job,
    poll_background_job,
    start_background_job,
)
from app_streamlit.plot_service import heatmap_figure, histogram_figure, line_figure, release_figure
from app_streamlit.preview import array_summary, memory_label, preview_series
from app_streamlit.preview_ui import render_image_preview
from app_streamlit.science_inputs import prepare_profile_field, validate_aligned_positions
from app_streamlit.science_ui import FieldInput, render_field_input, spatial_axis_for_input
from app_streamlit.state import (
    clear_workflow_result,
    get_dataset_summary,
    get_workflow_result,
    initialized_widget_key,
    persistent_widget_key,
    scientific_signature,
    set_dataset_summary,
    set_workflow_result,
    workflow_result_is_stale,
)
from app_streamlit.workflow_models import (
    SimulationConfig,
    half_open_positions,
    interval_bin_count,
    validate_positive_fields,
)
from app_streamlit.workflow_ui import available_data_views, view_axis, view_series

PAGE_NAMESPACE = "simulation"
JOB_KEY = "simulation_background_job_v2"
PLOT_KEY = "simulation_plot_preview_v2"
RAW_EXPORT_KEY = "simulation_raw_export_v2"
FIGURE_EXPORT_KEY = "simulation_figure_export_v2"

SPACE_OUTPUTS = {
    "replication_timing",
    "origin_efficiency",
    "fork_directionality",
}
TIME_OUTPUTS = {
    "replicated_fraction",
    "total_forks",
    "active_forks",
    "stalled_forks",
    "firing_factors",
}
DISTRIBUTION_OUTPUTS = {
    "completion_time",
    "num_origins",
    "inter_origin_distances",
}
XT_OUTPUTS = {
    "replicated_fraction_xt",
    "initiation_events_xt",
    "coalescence_events_xt",
    "right_moving_fork_density_xt",
    "left_moving_fork_density_xt",
}
OUTPUT_UNITS = {
    "replication_timing": "min",
    "origin_efficiency": "probability",
    "fork_directionality": "right - left fraction",
    "completion_time": "min",
    "num_origins": "origins",
    "inter_origin_distances": "kb",
    "rep_times_per_sim": "min",
    "replicated_fraction": "fraction",
    "replicated_fraction_xt": "fraction",
    "initiation_events_xt": "mean events/bin/time bin",
    "coalescence_events_xt": "mean events/bin/time bin",
    "right_moving_fork_density_xt": "forks/bin",
    "left_moving_fork_density_xt": "forks/bin",
    "total_forks": "forks",
    "active_forks": "forks",
    "stalled_forks": "forks",
    "firing_factors": "available factors",
}

# Labels, defaults, and view choices intentionally mirror origin/main.  The
# values still resolve to the optimized package output keys used below.
HEATMAP_PALETTES = ("viridis", "plasma", "magma", "cividis", "inferno")
LEGEND_LOCATIONS = {
    "Lower right": "lower right",
    "Lower left": "lower left",
    "Upper right": "upper right",
    "Upper left": "upper left",
    "Best": "best",
}
SIMULATION_OUTPUTS = {
    "Replication timing": {
        "key": "replication_timing", "kind": "space",
        "ytitle": "Replication timing", "color": "#1f77b4",
    },
    "Origin efficiency": {
        "key": "origin_efficiency", "kind": "space",
        "ytitle": "Origin efficiency", "color": "#b03412",
    },
    "Fork directionality": {
        "key": "fork_directionality", "kind": "space",
        "ytitle": "Fork directionality", "color": "#9467bd",
    },
    "Replicated fraction": {
        "key": "replicated_fraction", "kind": "time",
        "ytitle": "Replicated fraction", "color": "#2ca02c",
    },
    "Total forks": {
        "key": "total_forks", "kind": "time",
        "ytitle": "Total forks", "color": "#ff7f0e",
    },
    "Active forks": {
        "key": "active_forks", "kind": "time",
        "ytitle": "Active forks", "color": "#1f77b4",
    },
    "Stalled forks": {
        "key": "stalled_forks", "kind": "time",
        "ytitle": "Stalled forks", "color": "#8c564b",
    },
    "Firing factors": {
        "key": "firing_factors", "kind": "time",
        "ytitle": "Firing factors", "color": "#7f7f7f",
    },
    "Replicated fraction (space-time)": {
        "key": "replicated_fraction_xt", "kind": "xt",
        "label": "Replicated fraction", "cmap": "viridis",
    },
    "Initiation events (space-time)": {
        "key": "initiation_events_xt", "kind": "xt",
        "label": "Initiation events", "cmap": "magma",
    },
    "Coalescence events (space-time)": {
        "key": "coalescence_events_xt", "kind": "xt",
        "label": "Coalescence events", "cmap": "plasma",
    },
    "Right fork density (space-time)": {
        "key": "right_moving_fork_density_xt", "kind": "xt",
        "label": "Right-moving fork density", "cmap": "cividis",
    },
    "Left fork density (space-time)": {
        "key": "left_moving_fork_density_xt", "kind": "xt",
        "label": "Left-moving fork density", "cmap": "cividis",
    },
}
SIMULATION_OUTPUTS_BY_KEY = {
    str(spec["key"]): {"title": title, "label": str(spec.get("label", title)), **spec}
    for title, spec in SIMULATION_OUTPUTS.items()
}
SIMULATION_VIEW_OPTIONS = {
    "space": ("Single view", "Multiple view", "Histogram"),
    "time": ("Single view", "Multiple view", "Histogram"),
    "xt": ("Time array", "Multiple view", "Histogram"),
}


def _axis_from_field(field: FieldInput) -> np.ndarray | None:
    for name, scale in (("position_kb", 1.0), ("position_bp", 0.001), ("start_bp", 0.001)):
        axis = field.axes.get(name)
        if axis is not None:
            return np.asarray(axis, dtype=float).reshape(-1) * scale
    return None


def _inferred_resolution(sources: Mapping[str, FieldInput | None]) -> float:
    for field in sources.values():
        if field is None:
            continue
        axis = _axis_from_field(field)
        if axis is not None and axis.size > 1:
            steps = np.diff(axis)
            if np.all(np.isfinite(steps)) and np.all(steps > 0):
                return float(np.median(steps))
    return 1.0


def _region_defaults(
    sources: Mapping[str, FieldInput | None], resolution_kb: float
) -> tuple[float, float]:
    for field in sources.values():
        if field is None or field.data_form not in {"space", "spacetime", "table"}:
            continue
        axis = _axis_from_field(field)
        if axis is not None and axis.size:
            return float(axis[0]), float(axis[-1] + resolution_kb)
        values = np.asarray(field.values)
        if values.ndim:
            return 0.0, float(values.shape[0] * resolution_kb)
    return 0.0, 1_000.0


def _render_sources() -> dict[str, FieldInput | None]:
    columns = st.columns(2, gap="small")
    with columns[0]:
        firing = render_field_input(
            f"{PAGE_NAMESPACE}_firing_rate",
            "firing_rate",
            "Firing rate",
            allow_constant=True,
            constant_default=0.01,
            constant_min=0.0,
            constant_step=0.001,
            units="min^-1 per bin",
            default_mode="Profile",
            help_text=(
                "Values are per-bin hazards in min^-1. A density in kb^-1 min^-1 must be "
                "multiplied by the bin width before simulation."
            ),
        )
    with columns[1]:
        speed = render_field_input(
            f"{PAGE_NAMESPACE}_fork_speed",
            "fork_speed",
            "Fork speed",
            allow_constant=True,
            constant_default=1.4,
            constant_min=1e-9,
            constant_step=0.1,
            units="kb/min",
            default_mode="Constant",
            help_text="Scalar, spatial, time-only (1, T), and space-time (n, T) fields are supported.",
        )
    return {"firing_rate": firing, "fork_speed": speed}


def _render_configuration(
    sources: Mapping[str, FieldInput | None],
) -> tuple[SimulationConfig, str, bool]:
    resolution_key = persistent_widget_key(PAGE_NAMESPACE, "resolution_x")
    source_token = scientific_signature(
        tuple(
            sorted(
                (name, source.source_id)
                for name, source in sources.items()
                if source is not None
            )
        )
    )
    source_token_key = f"{PAGE_NAMESPACE}_region_source_token"
    source_changed = st.session_state.get(source_token_key) != source_token
    if resolution_key not in st.session_state or source_changed:
        st.session_state[resolution_key] = _inferred_resolution(sources)
    resolution_guess = float(st.session_state[resolution_key])
    default_start, default_end = _region_defaults(sources, resolution_guess)
    start_key = persistent_widget_key(PAGE_NAMESPACE, "region_start")
    end_key = persistent_widget_key(PAGE_NAMESPACE, "region_end")
    if source_changed or start_key not in st.session_state:
        st.session_state[start_key] = default_start
    if source_changed or end_key not in st.session_state:
        st.session_state[end_key] = default_end
    st.session_state[source_token_key] = source_token

    with st.expander("Region", expanded=False):
        region_top = st.columns(2, gap="small")
        resolution_x = float(region_top[0].number_input(
            "Resolution", min_value=1e-6, step=0.1, key=resolution_key,
            help="Spatial resolution in kb/bin. Input firing values are per bin.",
        ))
        resolution_t = float(region_top[1].number_input(
            "Time step", min_value=1e-6, step=1.0,
            key=initialized_widget_key(PAGE_NAMESPACE, "resolution_t", 5.0),
            help="Temporal resolution in min/bin for time-dependent outputs.",
        ))
        region = st.columns(2, gap="small")
        region_start = float(region[0].number_input(
            "Start (kb)", min_value=0.0, step=resolution_x, key=start_key,
        ))
        region_end = float(region[1].number_input(
            "End (kb)", min_value=resolution_x, step=resolution_x, key=end_key,
            help="The end coordinate is exclusive.",
        ))

    with st.expander("Simulation settings", expanded=False):
        setup = st.columns(2, gap="small")
        realizations = int(setup[0].number_input(
            "Replicates", min_value=1, step=1,
            key=initialized_widget_key(PAGE_NAMESPACE, "realizations", 20),
        ))
        geometry = setup[1].selectbox(
            "Geometry", ("linear", "periodic"),
            key=persistent_widget_key(PAGE_NAMESPACE, "geometry"),
            help="Periodic geometry includes the wrap distance in inter-origin distances.",
        )

        timing = st.columns(2, gap="small")
        time_max = float(timing[0].number_input(
            "Time window", min_value=resolution_t, step=10.0,
            key=initialized_widget_key(PAGE_NAMESPACE, "time_max", 300.0),
            help="Upper bound of the summary time grid in minutes.",
        ))
        max_rep_time = float(timing[1].number_input(
            "Max replication time", min_value=resolution_t, step=50.0,
            key=initialized_widget_key(
                PAGE_NAMESPACE, "max_rep_time", 1200.0
            ),
            help="Loci still unresolved at this cap are censored, not known completion times.",
        ))

        options = st.columns(2, gap="small")
        missing = options[0].selectbox(
            "Missing values", ("error", "interpolate", "zero"),
            key=persistent_widget_key(PAGE_NAMESPACE, "missing"),
        )
        fixed_seed = options[1].checkbox(
            "Fixed seed",
            key=initialized_widget_key(PAGE_NAMESPACE, "fixed_seed", True),
        )
        seed = None
        if fixed_seed:
            seed = int(st.number_input(
                "Seed", min_value=0, step=1,
                key=initialized_widget_key(PAGE_NAMESPACE, "seed", 1),
            ))
        else:
            st.warning("No seed produces a deliberately non-reproducible ensemble.")

        stall = st.columns(2, gap="small")
        stall_rate = float(stall[0].number_input(
            "Stall rate", min_value=0.0, max_value=1.0,
            format="%.6g",
            key=initialized_widget_key(PAGE_NAMESPACE, "stall_rate", 0.0),
            help="Probability per bin crossing; its physical effect changes with resolution.",
        ))
        permanent_stall = stall[1].checkbox(
            "Non-recovering stalls",
            key=initialized_widget_key(
                PAGE_NAMESPACE, "permanent_stall", True
            ),
        )
        stall_duration = math.inf
        if not permanent_stall:
            stall_duration = float(st.number_input(
                "Stall duration", min_value=1e-6,
                key=initialized_widget_key(
                    PAGE_NAMESPACE, "stall_duration", 10.0
                ),
                help="Mean recovery time in minutes for transient stalls.",
            ))

        firing_factors = None
        recycle = 0.0
        coupling = False
        finite_pool = st.checkbox(
            "Finite firing-factor pool",
            key=initialized_widget_key(PAGE_NAMESPACE, "finite_pool", False),
        )
        if finite_pool:
            factor_columns = st.columns(2, gap="small")
            firing_factors = float(factor_columns[0].number_input(
                "Pool size", min_value=2.0, step=1.0,
                key=initialized_widget_key(
                    PAGE_NAMESPACE, "firing_factors", 100.0
                ),
            ))
            recycle = float(factor_columns[1].number_input(
                "Recycle rate", min_value=0.0, format="%.6g",
                key=initialized_widget_key(
                    PAGE_NAMESPACE, "factor_recycle", 0.0
                ),
            ))
            coupling = st.checkbox(
                "Couple to active forks",
                key=initialized_widget_key(
                    PAGE_NAMESPACE, "factor_coupling", False
                ),
            )

        # Output-specific summaries are requested automatically from the
        # plotting selection, as in the online interface.  Large optional
        # intermediates remain disabled unless the chosen view needs them.
        time_stats = False
        spacetime_stats = False
        retain_rep_times = False
        retain_iod = False
        dens_block_n = 4096

    config = SimulationConfig(
        resolution_x=resolution_x,
        resolution_t=resolution_t,
        region_start_kb=region_start,
        region_end_kb=region_end,
        realizations=realizations,
        geometry=geometry,
        time_max=time_max,
        max_rep_time=max_rep_time,
        stall_rate=stall_rate,
        stall_duration=stall_duration,
        firing_factors=firing_factors,
        firing_factor_recycle=recycle,
        firing_factor_fork_coupling=coupling,
        seed=seed,
        time_stats=time_stats,
        spacetime_stats=spacetime_stats,
        retain_rep_times_per_sim=retain_rep_times,
        retain_inter_origin_distances=retain_iod,
        dens_block_n=dens_block_n,
    )
    active_jobs = [
        st.session_state.get("mapping_background_job_v2"),
        st.session_state.get("simulation_background_job_v2"),
    ]
    execution_busy = any_background_job_running(*active_jobs)
    if execution_busy:
        st.caption(
            "A scientific job is already running in this session. Wait for it to finish "
            "before starting another large calculation."
        )
    return config, str(missing), execution_busy


def _prepare_sources(
    sources: Mapping[str, FieldInput],
    config: SimulationConfig,
    missing: str,
) -> tuple[dict[str, Any], np.ndarray, int, list[str]]:
    prepared: dict[str, Any] = {}
    spatial_fields = []
    warnings: list[str] = []
    for feature, source in sources.items():
        if source.data_form == "scalar":
            prepared[feature] = float(source.values)
            continue
        field = prepare_profile_field(
            source.values,
            source.data_form,
            resolution_kb=config.resolution_x,
            region_start_kb=config.region_start_kb,
            region_end_kb=config.region_end_kb,
            position_kb=spatial_axis_for_input(source, config.resolution_x),
            missing=missing,
            name="Firing rate" if feature == "firing_rate" else "Fork speed",
            expected_time_points=None,
        )
        prepared[feature] = field.values
        spatial_fields.append(field)
        if field.field_kind in {"time", "spacetime"}:
            horizon = (field.values.shape[-1] - 1) * config.resolution_t
            if horizon < config.max_rep_time:
                if feature == "firing_rate":
                    warnings.append(
                        f"Firing rate ends at {horizon:g} min. Its cumulative hazard stops "
                        "there, so remaining loci may never fire."
                    )
                else:
                    warnings.append(
                        f"Fork speed ends at {horizon:g} min and is clamped to its final "
                        "sample for later simulation times."
                    )

    position = validate_aligned_positions(*spatial_fields)
    n_bins = int(position.size) if position is not None else interval_bin_count(
        config.region_start_kb, config.region_end_kb, config.resolution_x
    )
    if position is None:
        position = half_open_positions(n_bins, config.resolution_x, config.region_start_kb)
    validate_positive_fields(prepared)
    return prepared, position, n_bins, warnings


def _readonly_array(value: Any) -> np.ndarray:
    array = np.asarray(value)
    try:
        array.setflags(write=False)
    except ValueError:
        pass
    return array


def _execute_simulation(
    source: Mapping[str, Any],
    assumptions: Mapping[str, Any],
    position_kb: np.ndarray,
    config_metadata: Mapping[str, Any],
    *,
    job_context: JobContext,
) -> dict[str, Any]:
    from dnascape.simulation import rsim
    from dnascape.utils import progress_context

    started = time.perf_counter()
    run_assumptions = dict(assumptions)
    run_assumptions["verbose"] = True
    run_assumptions["print_every"] = 1
    job_context.report(0.02, "Preparing simulation...")
    with progress_context(
        job_context.report,
        job_context.raise_if_cancelled,
        suppress_output=True,
    ):
        raw = rsim(source=dict(source), assumptions=run_assumptions)
    job_context.report(0.995, "Collecting results...")
    names = [
        "replication_timing",
        "completion_time",
        "origin_efficiency",
        "fork_directionality",
        "rep_times_per_sim",
        "inter_origin_distances",
        "replicated_fraction",
        "replicated_fraction_xt",
        "initiation_events_xt",
        "coalescence_events_xt",
        "right_moving_fork_density_xt",
        "left_moving_fork_density_xt",
        "total_forks",
        "active_forks",
        "stalled_forks",
        "firing_factors",
    ]
    outputs: dict[str, np.ndarray] = {}
    for name in names:
        value = raw.get(name)
        if value is not None:
            outputs[name] = _readonly_array(value)
    origins = raw.get("num_origins")
    if origins is not None:
        outputs["num_origins"] = _readonly_array(origins)
    time_grid = raw.get("time_grid")
    if time_grid is not None and (
        assumptions.get("time_statsQ") or assumptions.get("time_stats_xtQ")
    ):
        outputs["time_grid"] = _readonly_array(time_grid)
    outputs["position_kb"] = _readonly_array(position_kb)
    output_forms = {
        name: (
            "space"
            if name in SPACE_OUTPUTS
            else "time"
            if name in TIME_OUTPUTS
            else "distribution"
            if name in DISTRIBUTION_OUTPUTS
            else "spacetime"
            if name in XT_OUTPUTS
            else "ensemble_space"
            if name == "rep_times_per_sim"
            else "array"
        )
        for name in outputs
        if name not in {"position_kb", "time_grid"}
    }
    return {
        "outputs": outputs,
        "elapsed_seconds": float(time.perf_counter() - started),
        "configuration": dict(config_metadata),
        "seed": assumptions.get("seed"),
        "realizations": assumptions.get("sim_number"),
        "output_forms": output_forms,
    }


def _consume_job(job: BackgroundJob) -> None:
    if job.status == "complete" and not job.consumed:
        payload = job.result
        if not isinstance(payload, Mapping):
            job.status = "error"
            job.error_type = "InvalidJobResult"
            job.error_message = "Simulation returned an invalid result."
            job.result = None
            return
        set_workflow_result(
            "simulation", payload, job.input_signature,
            metadata={
                "realizations": payload.get("realizations"),
                "seed": payload.get("seed"),
                "elapsed_seconds": payload.get("elapsed_seconds"),
            },
        )
        job.consumed = True
        job.result = None
        st.session_state.pop(PLOT_KEY, None)
        st.session_state.pop(RAW_EXPORT_KEY, None)
        st.session_state.pop(FIGURE_EXPORT_KEY, None)


@st.fragment(run_every=0.4)
def _render_active_job(job_id: str) -> None:
    """Poll one running simulation without rerunning the full page."""

    active = st.session_state.get(JOB_KEY)
    if not isinstance(active, BackgroundJob) or active.job_id != job_id:
        return
    poll_background_job(active)
    if not active.running:
        _consume_job(active)
        st.rerun()
    st.progress(
        active.progress_value,
        text=f"{active.progress_text}  Elapsed {active.elapsed_seconds:.1f} s",
    )
    if st.button(
        "Cancel",
        key=f"simulation_cancel_{active.job_id}",
        disabled=active.cancellation_requested,
        width="stretch",
    ):
        cancel_background_job(active)


def _render_job_status() -> None:
    job = st.session_state.get(JOB_KEY)
    if not isinstance(job, BackgroundJob):
        return
    poll_background_job(job)
    _consume_job(job)
    if job.running:
        _render_active_job(job.job_id)
    elif job.status == "error":
        st.error(f"{job.label} failed ({job.error_type}): {job.error_message}")
    elif job.status == "cancelled":
        st.info("Simulation cancelled.")


def _numeric_outputs(payload: Mapping[str, Any]) -> dict[str, np.ndarray]:
    outputs = payload.get("outputs", {})
    axes = {"position_kb", "time_grid"}
    if not isinstance(outputs, Mapping):
        return {}
    return {
        str(name): value for name, value in outputs.items()
        if name not in axes and isinstance(value, np.ndarray) and value.dtype.kind in "biufc"
    }


def _axes_for_output(payload: Mapping[str, Any], name: str) -> dict[str, np.ndarray]:
    outputs = payload.get("outputs", {})
    if not isinstance(outputs, Mapping):
        return {}
    axes: dict[str, np.ndarray] = {}
    if name in SPACE_OUTPUTS or name == "rep_times_per_sim" or name in XT_OUTPUTS:
        position = outputs.get("position_kb")
        if isinstance(position, np.ndarray):
            axes["position_kb"] = position
    if name in TIME_OUTPUTS or name in XT_OUTPUTS:
        time_grid = outputs.get("time_grid")
        if isinstance(time_grid, np.ndarray):
            axes["time_grid"] = time_grid
    return axes


def _histogram_bin_count(values: np.ndarray, bin_size: float) -> int:
    """Convert a physical bin width into a safe bounded bin count."""

    if not np.isfinite(bin_size) or bin_size <= 0:
        raise ValueError("Histogram bin size must be finite and greater than zero.")
    finite = np.asarray(values).reshape(-1)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("The selected output contains no finite values to plot.")
    span = float(np.max(finite) - np.min(finite))
    if span <= 0:
        return 1
    return max(1, min(4096, int(math.ceil(span / float(bin_size)))))


def _build_result_figure(
    payload: Mapping[str, Any],
    settings: Mapping[str, Any],
    *,
    full_resolution: bool = False,
):
    output_name = str(settings.get("output") or "")
    title = str(settings.get("title") or output_name.replace("_", " ").title())
    color = str(settings.get("color") or "#2c7f77")
    cmap = str(settings.get("cmap") or "viridis")
    overlay_id = str(settings.get("overlay_id") or "")
    view_mode = str(settings.get("view") or "Single view")
    figsize_value = settings.get("figsize", (9.2, 4.5))
    try:
        figsize = (float(figsize_value[0]), float(figsize_value[1]))
    except (IndexError, TypeError, ValueError):
        figsize = (9.2, 4.5)
    if not all(np.isfinite(value) and value > 0 for value in figsize):
        raise ValueError("Figure width and height must be finite and positive.")

    arrays = _numeric_outputs(payload)
    if output_name not in arrays:
        raise ValueError("The selected simulation output is not available.")
    values = arrays[output_name]
    axes = _axes_for_output(payload, output_name)
    units = OUTPUT_UNITS.get(output_name, "value")
    if output_name in DISTRIBUTION_OUTPUTS or view_mode == "Histogram":
        bins = _histogram_bin_count(
            values, float(settings.get("bin_size") or 1.0)
        )
        return histogram_figure(
            values,
            bins=bins,
            title=title,
            x_label=str(settings.get("x_label") or units),
            y_label=str(settings.get("y_label") or "Frequency"),
            color=color,
            full_resolution=full_resolution,
        )[0]
    if output_name == "rep_times_per_sim":
        return heatmap_figure(
            values,
            title=title,
            x_label="Position (kb)",
            y_label="Realization",
            colorbar_label="Replication time (min)",
            x_values=axes.get("position_kb"),
            y_values=np.arange(values.shape[0]),
            cmap=cmap,
            show_colorbar=bool(settings.get("show_colorbar", True)),
            invert_y=bool(settings.get("invert_y", False)),
            figsize=figsize,
            full_resolution=full_resolution,
        )[0]
    if output_name in XT_OUTPUTS:
        return heatmap_figure(
            values,
            title=title,
            x_label="Time (min)",
            y_label="Position (kb)",
            colorbar_label=units,
            x_values=axes.get("time_grid"),
            y_values=axes.get("position_kb"),
            cmap=cmap,
            show_colorbar=bool(settings.get("show_colorbar", True)),
            invert_y=bool(settings.get("invert_y", False)),
            figsize=figsize,
            full_resolution=full_resolution,
        )[0]

    y = np.asarray(values).reshape(-1)
    if output_name in TIME_OUTPUTS:
        x = axes.get("time_grid")
        x_label = "Time (min)"
    else:
        x = axes.get("position_kb")
        x_label = "Position (kb)"
    if x is None or x.size != y.size:
        x = np.arange(y.size)
        x_label = "Index"
    figure, plot_axes = line_figure(
        x,
        y,
        title=title,
        x_label=str(settings.get("x_label") or x_label),
        y_label=str(settings.get("y_label") or units),
        color=color,
        label=(
            str(settings.get("label") or output_name.replace("_", " "))
            if settings.get("show_legend")
            else None
        ),
        legend_location=str(settings.get("legend_location") or "best"),
        y_limits=settings.get("y_limits"),
        log_y=bool(settings.get("log_y", False)),
        invert_y=bool(settings.get("invert_y", False)),
        figsize=figsize,
        full_resolution=full_resolution,
    )
    if overlay_id and output_name in SPACE_OUTPUTS:
        views = {view.view_id: view for view in available_data_views(include_workflows=False)}
        overlay = views.get(overlay_id)
        if overlay is not None:
            other = view_series(overlay)
            other_x = view_axis(overlay, other.size)
            if other.size != y.size or not np.allclose(other_x, x):
                raise ValueError(
                    "Comparison overlays must have the same length and physical coordinate "
                    "axis as the simulated result. Align them on Data Management first."
                )
            if full_resolution:
                x_plot, y_plot = other_x, other
            else:
                x_plot, y_plot = preview_series(other_x, other, max_points=5000)
            plot_axes.plot(
                x_plot, y_plot, color="#d97757", linewidth=1.4, alpha=0.85,
                label=overlay.name,
            )
            plot_axes.legend(loc="best")
    return figure


def _configured_output_names(config: SimulationConfig, stored) -> tuple[str, ...]:
    """Return result outputs followed by outputs requested for the next run."""

    actual = list(_numeric_outputs(stored.payload)) if stored is not None else []
    requested = [
        "replication_timing",
        "origin_efficiency",
        "fork_directionality",
        "completion_time",
        "num_origins",
    ]
    if config.retain_rep_times_per_sim:
        requested.append("rep_times_per_sim")
    if config.retain_inter_origin_distances:
        requested.append("inter_origin_distances")
    if config.time_stats:
        requested.extend(
            (
                "replicated_fraction",
                "total_forks",
                "active_forks",
                "stalled_forks",
            )
        )
        if config.firing_factors is not None:
            requested.append("firing_factors")
    if config.spacetime_stats:
        requested.extend(sorted(XT_OUTPUTS))
    return tuple(dict.fromkeys(actual + requested))


def _config_for_requested_output(
    config: SimulationConfig, settings: Mapping[str, Any]
) -> SimulationConfig:
    """Request only the optional rsim arrays needed by the selected preview."""

    output_name = str(settings.get("output") or "")
    return replace(
        config,
        time_stats=output_name in TIME_OUTPUTS,
        spacetime_stats=output_name in XT_OUTPUTS,
        retain_rep_times_per_sim=output_name == "rep_times_per_sim",
        retain_inter_origin_distances=output_name == "inter_origin_distances",
    )


def _store_plot_preview(stored: Any, settings: Mapping[str, Any]) -> None:
    """Build and retain one bounded preview for the current simulation result."""

    output_name = str(settings.get("output") or "")
    figure = None
    try:
        figure = _build_result_figure(
            stored.payload,
            settings,
            full_resolution=False,
        )
        preview = serialize_figure(figure, "png", dpi=115)
    finally:
        if figure is not None:
            release_figure(figure)
    signature = str(
        settings.get("signature")
        or scientific_signature(stored.input_signature, dict(settings))
    )
    st.session_state[PLOT_KEY] = {
        "signature": signature,
        "result_signature": stored.input_signature,
        "payload": preview.payload,
        "output": output_name,
        "settings": dict(settings),
    }
    st.session_state.pop(FIGURE_EXPORT_KEY, None)


def _simulation_figure_download_bytes(
    payload: Mapping[str, Any],
    settings: Mapping[str, Any],
    figure_format: str,
    full_resolution: bool,
) -> bytes:
    """Build a requested figure only when its download button is clicked."""

    figure = None
    try:
        figure = _build_result_figure(
            payload,
            settings,
            full_resolution=full_resolution,
        )
        return serialize_figure(figure, figure_format, dpi=220).payload
    finally:
        if figure is not None:
            release_figure(figure)


def _render_plotting_settings(
    stored,
    config: SimulationConfig,
) -> dict[str, Any]:
    """Render the plotting form used by the online Simulation page."""

    arrays = _numeric_outputs(stored.payload) if stored is not None else {}
    with st.expander("Plotting settings", expanded=False):
        selector_cols = st.columns([1.2, 0.8], gap="small")
        output_key = persistent_widget_key(PAGE_NAMESPACE, "preview_output")
        output_labels = tuple(SIMULATION_OUTPUTS)
        if st.session_state.get(output_key) not in output_labels:
            st.session_state[output_key] = output_labels[0]
        output_label = selector_cols[0].selectbox(
            "Output", output_labels, key=output_key
        )
        output_spec = SIMULATION_OUTPUTS[output_label]
        output_name = str(output_spec["key"])
        output_kind = str(output_spec["kind"])

        view_options = SIMULATION_VIEW_OPTIONS[output_kind]
        view_key = persistent_widget_key(PAGE_NAMESPACE, "preview_view")
        if st.session_state.get(view_key) not in view_options:
            st.session_state[view_key] = view_options[0]
        preview_mode = selector_cols[1].selectbox(
            "View", view_options, key=view_key
        )
        is_histogram = preview_mode == "Histogram"
        is_heatmap = output_kind == "xt" and not is_histogram

        if is_histogram:
            default_x = str(output_spec.get("label", output_label))
            default_y = "Frequency"
        elif output_kind == "space":
            default_x, default_y = "Genomic position (kb)", str(output_spec["ytitle"])
        elif output_kind == "time":
            default_x, default_y = "S-phase time (min)", str(output_spec["ytitle"])
        else:
            default_x, default_y = "Genomic position (kb)", "S-phase time (min)"

        widget_suffix = f"{output_name}_{preview_mode.lower().replace(' ', '_')}"
        text_cols = st.columns([1.15, 0.85], gap="small")
        title = text_cols[0].text_input(
            "Title",
            placeholder="Auto",
            key=initialized_widget_key(
                PAGE_NAMESPACE, f"title_{widget_suffix}", ""
            ),
        )
        label = text_cols[1].text_input(
            "Label",
            placeholder=str(output_spec.get("label", output_label)),
            key=initialized_widget_key(
                PAGE_NAMESPACE, f"label_{widget_suffix}", ""
            ),
        )
        axis_cols = st.columns(2, gap="small")
        x_label = axis_cols[0].text_input(
            "x-axis",
            key=initialized_widget_key(
                PAGE_NAMESPACE, f"x_axis_{widget_suffix}", default_x
            ),
        )
        y_label = axis_cols[1].text_input(
            "y-axis",
            key=initialized_widget_key(
                PAGE_NAMESPACE, f"y_axis_{widget_suffix}", default_y
            ),
        )

        color = str(output_spec.get("color", "#1f77b4"))
        cmap = str(output_spec.get("cmap", HEATMAP_PALETTES[0]))
        show_legend = False
        legend_location_label = "Best"
        show_colorbar = False
        invert_y = False
        log_y = False
        use_y_limits = False
        density = False
        bin_size = 5.0 if output_kind != "time" else 2.0

        if is_histogram:
            histogram_cols = st.columns([0.78, 0.74, 0.98], gap="small")
            show_legend = histogram_cols[0].checkbox(
                "Legend",
                key=initialized_widget_key(
                    PAGE_NAMESPACE, f"legend_{widget_suffix}", False
                ),
            )
            density = histogram_cols[1].checkbox(
                "Density",
                key=initialized_widget_key(
                    PAGE_NAMESPACE, f"density_{widget_suffix}", False
                ),
            )
            bin_size = float(histogram_cols[2].number_input(
                "Bin size", min_value=0.1,
                step=0.5 if output_kind != "time" else 0.2, format="%.2f",
                key=initialized_widget_key(
                    PAGE_NAMESPACE, f"bin_size_{widget_suffix}", bin_size
                ),
            ))
        elif is_heatmap:
            style_cols = st.columns([0.78, 1.22], gap="small")
            show_colorbar = style_cols[0].checkbox(
                "Colorbar",
                key=initialized_widget_key(
                    PAGE_NAMESPACE, f"colorbar_{widget_suffix}", True
                ),
            )
            palette_key = persistent_widget_key(PAGE_NAMESPACE, f"palette_{widget_suffix}")
            if st.session_state.get(palette_key) not in HEATMAP_PALETTES:
                st.session_state[palette_key] = cmap
            cmap = style_cols[1].selectbox("Palette", HEATMAP_PALETTES, key=palette_key)
            scale_cols = st.columns(3, gap="small")
            invert_y = scale_cols[0].checkbox(
                "Invert y-axis",
                key=initialized_widget_key(
                    PAGE_NAMESPACE, f"invert_{widget_suffix}", False
                ),
            )
            log_y = scale_cols[1].checkbox(
                "Log y-axis",
                key=initialized_widget_key(
                    PAGE_NAMESPACE, f"log_{widget_suffix}", False
                ),
            )
            use_y_limits = scale_cols[2].checkbox(
                "Y limits",
                key=initialized_widget_key(
                    PAGE_NAMESPACE, f"limits_{widget_suffix}", False
                ),
            )
        else:
            style_cols = st.columns([0.72, 1.0, 0.92], gap="small")
            show_legend = style_cols[0].checkbox(
                "Legend",
                key=initialized_widget_key(
                    PAGE_NAMESPACE, f"legend_{widget_suffix}", True
                ),
            )
            location_key = persistent_widget_key(PAGE_NAMESPACE, f"location_{widget_suffix}")
            if st.session_state.get(location_key) not in LEGEND_LOCATIONS:
                st.session_state[location_key] = "Lower right"
            legend_location_label = style_cols[1].selectbox(
                "Position", tuple(LEGEND_LOCATIONS), key=location_key,
                disabled=not show_legend,
            )
            color = style_cols[2].color_picker(
                "Color",
                key=initialized_widget_key(
                    PAGE_NAMESPACE, f"color_{widget_suffix}", color
                ),
            )
            scale_cols = st.columns(3, gap="small")
            invert_y = scale_cols[0].checkbox(
                "Invert y-axis",
                key=initialized_widget_key(
                    PAGE_NAMESPACE, f"invert_{widget_suffix}", False
                ),
            )
            log_y = scale_cols[1].checkbox(
                "Log y-axis",
                key=initialized_widget_key(
                    PAGE_NAMESPACE, f"log_{widget_suffix}", False
                ),
            )
            use_y_limits = scale_cols[2].checkbox(
                "Y limits",
                key=initialized_widget_key(
                    PAGE_NAMESPACE, f"limits_{widget_suffix}", False
                ),
            )

        default_height = 3.6 if is_histogram else (4.6 if preview_mode == "Multiple view" else (4.1 if is_heatmap else 3.8))
        size_cols = st.columns(2, gap="small")
        figure_width = float(size_cols[0].number_input(
            "Width", min_value=3.0, max_value=16.0, step=0.5,
            format="%.1f",
            key=initialized_widget_key(
                PAGE_NAMESPACE, f"width_{widget_suffix}", 9.5
            ),
        ))
        figure_height = float(size_cols[1].number_input(
            "Height", min_value=2.0, max_value=12.0, step=0.2,
            format="%.1f",
            key=initialized_widget_key(
                PAGE_NAMESPACE, f"height_{widget_suffix}", default_height
            ),
        ))
        y_min, y_max = 0.0, 1.0
        if use_y_limits:
            y_cols = st.columns(2, gap="small")
            y_min = float(y_cols[0].number_input(
                "Min", step=1.0 if output_kind != "time" else 0.1,
                key=initialized_widget_key(
                    PAGE_NAMESPACE, f"y_min_{widget_suffix}", 0.0
                ),
            ))
            y_max = float(y_cols[1].number_input(
                "Max", step=1.0 if output_kind != "time" else 0.1,
                key=initialized_widget_key(
                    PAGE_NAMESPACE, f"y_max_{widget_suffix}", 1.0
                ),
            ))
            if y_max <= y_min:
                st.error("The y-axis maximum must be greater than the minimum.")

        can_plot = stored is not None and output_name in arrays
        update_requested = st.button(
            "Update preview",
            type="primary",
            width="stretch",
            disabled=not can_plot,
            key="simulation_update_plot",
            help="Apply plot-only settings without rerunning the simulation.",
        )

    settings = {
        "output": output_name,
        "output_label": output_label,
        "output_kind": output_kind,
        "view": preview_mode,
        "title": title.strip() or output_label,
        "label": label.strip() or str(output_spec.get("label", output_label)),
        "x_label": x_label,
        "y_label": y_label,
        "color": color,
        "cmap": cmap,
        "show_legend": show_legend,
        "legend_location": LEGEND_LOCATIONS[legend_location_label],
        "show_colorbar": show_colorbar,
        "invert_y": invert_y,
        "log_y": log_y,
        "y_limits": (y_min, y_max) if use_y_limits and y_max > y_min else None,
        "density": density,
        "bin_size": bin_size,
        "figsize": (figure_width, figure_height),
    }
    result_token = stored.input_signature if stored is not None else "pending"
    settings["signature"] = scientific_signature(result_token, settings)
    plot = st.session_state.get(PLOT_KEY)
    missing_current_preview = bool(
        stored is not None
        and can_plot
        and (
            not isinstance(plot, dict)
            or plot.get("result_signature") != stored.input_signature
        )
    )
    if stored is not None and can_plot and (update_requested or missing_current_preview):
        try:
            with st.spinner("Rendering plot..."):
                _store_plot_preview(stored, settings)
            plot = st.session_state.get(PLOT_KEY)
        except (ValueError, TypeError, OSError, MemoryError) as exc:
            st.error(str(exc))
    elif (
        can_plot
        and isinstance(plot, dict)
        and plot.get("result_signature") == result_token
        and plot.get("signature") != settings["signature"]
    ):
        st.caption("Plot settings changed - select Update preview to apply them.")
    settings["available"] = can_plot
    return settings


def _simulation_raw_download_bytes(
    values: np.ndarray,
    output_format: str,
    axes: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
    compress: bool,
) -> bytes:
    """Serialize raw simulation output only when Download is clicked."""

    exported = serialize_array(
        values,
        output_format,
        axes=axes,
        metadata=metadata,
    )
    if compress:
        exported = gzip_payload(exported.payload, exported.extension)
    return exported.payload


def _render_raw_output(stored, output_name: str) -> None:
    payload = stored.payload
    arrays = _numeric_outputs(payload)
    values = arrays[output_name]
    with st.expander("Raw output", expanded=False):
        st.markdown(
            '<p class="small-muted">Download the currently previewed rsim output in a DNAscape-ready format.</p>',
            unsafe_allow_html=True,
        )
        st.caption("Export")
        raw_columns = st.columns([0.46, 0.18, 0.12, 0.24], gap="small")
        raw_name = raw_columns[0].text_input(
            "File name",
            key=persistent_widget_key(PAGE_NAMESPACE, "raw_file_name"),
            placeholder="Name (auto)",
            label_visibility="collapsed",
        )
        raw_format_options = ("CSV", "TSV", "TXT", "Array")
        raw_format_key = persistent_widget_key(PAGE_NAMESPACE, "raw_format")
        if st.session_state.get(raw_format_key) not in raw_format_options:
            st.session_state[raw_format_key] = "CSV"
        raw_format_label = raw_columns[1].selectbox(
            "File type", ("CSV", "TSV", "TXT", "Array"),
            key=raw_format_key,
            label_visibility="collapsed",
        )
        compress = raw_columns[2].checkbox(
            "GZ",
            key=initialized_widget_key(PAGE_NAMESPACE, "raw_gzip", False),
        )
        raw_format = "npy" if raw_format_label == "Array" else raw_format_label.lower()
        extension, mime = array_export_target(
            raw_format,
            compressed=bool(compress),
        )
        axes = _axes_for_output(payload, output_name)
        metadata = {
            "feature": output_name,
            "seed": payload.get("seed"),
            "realizations": payload.get("realizations"),
        }
        default_stem = safe_name_stem("dnascape", "simulation", output_name, "raw")
        file_stem = safe_name_stem(raw_name) if raw_name.strip() else default_stem
        raw_columns[3].download_button(
            "Download",
            data=partial(
                _simulation_raw_download_bytes,
                values,
                raw_format,
                axes,
                metadata,
                bool(compress),
            ),
            file_name=f"{file_stem}.{extension}",
            mime=mime,
            on_click="ignore",
            width="stretch",
        )


def _render_compact_stats(stored, output_name: str) -> None:
    payload = stored.payload
    arrays = _numeric_outputs(payload)
    values = arrays[output_name]
    retained_bytes = sum(value.nbytes for value in arrays.values())
    seed = "random" if payload.get("seed") is None else str(int(payload["seed"]))
    ensemble_parts = [
        f"{int(payload.get('realizations') or 0):,} runs",
        f"seed {seed}",
    ]
    completion = arrays.get("completion_time")
    if completion is not None and completion.size:
        finite = np.asarray(completion)[np.isfinite(completion)]
        if finite.size:
            ensemble_parts.append(
                f"completion {float(np.mean(finite)):.2f} +/- {float(np.std(finite)):.2f} min"
            )
        cap = float(payload.get("configuration", {}).get("max_rep_time", math.inf))
        censored = int(np.count_nonzero(completion >= cap)) if np.isfinite(cap) else 0
        if censored:
            ensemble_parts.append(f"{censored} capped")
    origins = arrays.get("num_origins")
    if origins is not None and origins.size:
        finite_origins = np.asarray(origins)[np.isfinite(origins)]
        if finite_origins.size:
            ensemble_parts.append(
                f"origins {float(np.mean(finite_origins)):.2f} +/- {float(np.std(finite_origins)):.2f}"
            )
    ensemble_parts.extend(
        (
            f"{float(payload.get('elapsed_seconds', 0.0)):.2f} s",
            f"{memory_label(retained_bytes)} retained",
        )
    )
    st.caption("**Stats** - " + " - ".join(ensemble_parts))

    summary_key = f"workflow:simulation:{stored.input_signature}:{output_name}"
    summary = get_dataset_summary(summary_key)
    if summary is None:
        summary = set_dataset_summary(summary_key, array_summary(values))
    shape = " x ".join(map(str, values.shape)) or "scalar"
    feature = output_name.replace("_", " ").title()
    details = (
        f" - min {summary.minimum:.4g} - mean {summary.mean:.4g} - max {summary.maximum:.4g}"
        if summary.finite_count else ""
    )
    st.caption(
        f"{feature} - {shape} - {OUTPUT_UNITS.get(output_name, 'value')}{details}"
    )


def _clear_simulation_output() -> None:
    """Clear the retained simulation and its derived preview/export payloads."""

    clear_workflow_result("simulation")
    for key in (PLOT_KEY, RAW_EXPORT_KEY, FIGURE_EXPORT_KEY):
        st.session_state.pop(key, None)


def _render_result(stored, current_signature: str | None, plot_settings: Mapping[str, Any] | None) -> None:
    payload = stored.payload
    arrays = _numeric_outputs(payload)
    if not arrays:
        st.warning("The stored simulation contains no numerical summaries.")
        return
    if current_signature and workflow_result_is_stale("simulation", current_signature):
        st.warning("Inputs or simulation settings changed. Rerun to update this result.")

    plot = st.session_state.get(PLOT_KEY)
    if not (
        isinstance(plot, dict)
        and plot.get("result_signature") == stored.input_signature
    ):
        preferred = str((plot_settings or {}).get("output") or "")
        output_name = preferred if preferred in arrays else next(iter(arrays))
        fallback_settings = dict(plot_settings or {})
        fallback_settings["output"] = output_name
        fallback_settings["title"] = str(
            fallback_settings.get("title")
            or output_name.replace("_", " ").title()
        )
        fallback_settings.pop("signature", None)
        fallback_settings["signature"] = scientific_signature(
            stored.input_signature, fallback_settings
        )
        try:
            with st.spinner("Rendering plot..."):
                _store_plot_preview(stored, fallback_settings)
            plot = st.session_state.get(PLOT_KEY)
        except (ValueError, TypeError, OSError, MemoryError) as exc:
            st.error(f"Could not render the simulation plot: {exc}")

    if isinstance(plot, dict) and plot.get("result_signature") == stored.input_signature:
        output_name = str(plot.get("output"))
        render_image_preview(
            plot["payload"],
            file_name=safe_name_stem("dnascape", "simulation", output_name) + ".png",
            shell_key="simulation-preview-shell",
            clear_key="simulation-preview-clear",
            on_clear=_clear_simulation_output,
            alt=f"Simulation preview: {output_name.replace('_', ' ')}",
            downloads={},
        )
        settings_value = plot.get("settings")
        export_settings = (
            dict(settings_value)
            if isinstance(settings_value, Mapping)
            else dict(plot_settings or {})
        )
        export_settings["output"] = output_name
        export_stem = safe_name_stem("dnascape", "simulation", output_name)
        export_columns = st.columns(3, gap="small")
        for column, image_format in zip(export_columns, ("png", "svg", "pdf")):
            column.download_button(
                image_format.upper(),
                data=partial(
                    _simulation_figure_download_bytes,
                    payload,
                    export_settings,
                    image_format,
                    True,
                ),
                file_name=f"{export_stem}.{image_format}",
                mime=figure_export_target(image_format)[1],
                key=f"simulation_export_{image_format}_{stored.input_signature[:12]}",
                on_click="ignore",
                width="stretch",
                help="Build a full-resolution export only when clicked.",
            )
    else:
        output_name = str((plot_settings or {}).get("output") or next(iter(arrays)))

    if output_name not in arrays:
        output_name = next(iter(arrays))
    _render_raw_output(stored, output_name)
    _render_compact_stats(stored, output_name)


def render_simulation_workspace() -> None:
    stored = get_workflow_result("simulation")
    left, right = st.columns([0.35, 0.65], gap="medium")
    with left:
        control_stack = st.container(key="dnascape-control-stack")

    with control_stack:
        with st.container(
            border=True,
            height="stretch",
            key="dnascape-control-card",
        ):
            with st.expander("Data", expanded=True):
                sources = _render_sources()
            config, missing, execution_busy = _render_configuration(sources)
            plot_settings = _render_plotting_settings(stored, config)
            config = _config_for_requested_output(config, plot_settings)
            source_ids = {
                name: source.source_id for name, source in sources.items() if source is not None
            }
            signature = scientific_signature(
                "simulation-v2", source_ids, config, missing
            ) if len(source_ids) == 2 else None

            try:
                config.validate()
                estimate = config.estimate()
                st.caption(
                    f"Ensemble: {estimate.realizations:,} runs - {estimate.spatial_bins:,} bins"
                    + (f" - {estimate.time_points:,} time points" if estimate.time_points else "")
                    + f" - {estimate.memory_mib:.1f} MiB retained"
                )
                if estimate.expensive or estimate.spatial_bins * estimate.realizations >= 10_000_000:
                    st.warning("This ensemble may take substantial time or memory.")
                if estimate.impractical:
                    st.error("The requested retained output exceeds the app's 4 GiB safety estimate.")
            except ValueError as exc:
                estimate = None
                st.error(str(exc))

        job = st.session_state.get(JOB_KEY)
        running = isinstance(job, BackgroundJob) and job.running
        with st.container(key="dnascape-control-action"):
            submitted = st.button(
                "Simulate",
                type="primary",
                width="stretch",
                disabled=(
                    any(source is None for source in sources.values()) or execution_busy
                ),
            )

        if submitted:
            if running:
                st.warning("A simulation is already running.")
            elif estimate is not None and estimate.impractical:
                st.error("Reduce the requested ensemble or retained outputs before running.")
            else:
                try:
                    complete_sources = {
                        name: source for name, source in sources.items() if source is not None
                    }
                    config.validate()
                    prepared, position, _n_bins, preparation_warnings = _prepare_sources(
                        complete_sources, config, missing
                    )
                    for warning in preparation_warnings:
                        st.warning(warning)
                    assumptions = config.assumptions(include_interval=True)
                    assumptions["interval_kb"] = (
                        (
                            float(position[0]),
                            float(position[-1] + config.resolution_x),
                        )
                        if any(
                            source.data_form in {"space", "spacetime"}
                            for source in complete_sources.values()
                        )
                        else (config.region_start_kb, config.region_end_kb)
                    )
                    assert signature is not None
                    config_metadata = {
                        "resolution_x": config.resolution_x,
                        "resolution_t": config.resolution_t,
                        "region_start_kb": config.region_start_kb,
                        "region_end_kb": config.region_end_kb,
                        "max_rep_time": config.max_rep_time,
                        "time_max": config.time_max,
                        "geometry": config.geometry,
                    }
                    st.session_state[JOB_KEY] = start_background_job(
                        "Simulation", signature, _execute_simulation,
                        prepared, assumptions, position, config_metadata,
                    )
                    st.rerun()
                except (ValueError, TypeError, KeyError, RuntimeError, MemoryError) as exc:
                    st.error(str(exc))

    with right:
        with st.container(border=True, key="dnascape-preview-card"):
            st.markdown(
                '<div class="plotting-panel-title">Preview</div>',
                unsafe_allow_html=True,
            )
            _render_job_status()
            stored = get_workflow_result("simulation")
            if stored is None:
                st.caption("No simulation output.")
            else:
                _render_result(stored, signature, plot_settings)


def render_page() -> None:
    """Render Simulation inside the persistent DNAscape shell."""

    render_page_intro(
        "Simulation",
        "Simulate DNA replication",
        "Simulate stochastic replication dynamics, vary kinetic assumptions, and preview DNAscape-ready outputs.",
    )
    render_control_panel_marker("simulation-page-marker")
    render_simulation_workspace()
