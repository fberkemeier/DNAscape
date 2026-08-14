"""Explicit, route-aware Mapping workflow for the DNAscape app."""

from __future__ import annotations

import time
from dataclasses import dataclass
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
from app_streamlit.plot_service import heatmap_figure, line_figure, release_figure
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
    MappingConfig,
    build_time_grid,
    half_open_positions,
    interval_bin_count,
    validate_positive_fields,
)
from app_streamlit.workflow_ui import available_data_views, view_axis, view_series

PAGE_NAMESPACE = "mapping"
JOB_KEY = "mapping_background_job_v2"
PLOT_KEY = "mapping_plot_preview_v2"
RAW_EXPORT_KEY = "mapping_raw_export_v2"
FIGURE_EXPORT_KEY = "mapping_figure_export_v2"


@dataclass(frozen=True)
class MappingRoute:
    route_id: str
    label: str
    source_features: tuple[str, ...]
    target: str
    family: str
    target_forms: tuple[str, ...]
    description: str


ROUTES = (
    MappingRoute(
        "fit_firing", "Infer firing rate from timing", ("replication_timing", "fork_speed"),
        "firing_rate", "fit_firing", ("constant", "space", "time", "spacetime"),
        "Estimate an initiation hazard compatible with observed replication timing and fork speed.",
    ),
    MappingRoute(
        "fit_speed", "Infer fork speed from timing", ("replication_timing", "firing_rate"),
        "fork_speed", "fit_speed", ("constant", "space", "time", "spacetime"),
        "Estimate fork speed compatible with observed timing and firing.",
    ),
    MappingRoute(
        "predict_timing", "Predict replication timing", ("firing_rate", "fork_speed"),
        "replication_timing", "predict", ("space",),
        "Use deterministic propagation for scalar/spatial fields and simulation for time-dependent fields.",
    ),
    MappingRoute(
        "timing_to_direction", "Calculate fork directionality", ("replication_timing", "fork_speed"),
        "fork_directionality", "direction", ("space",),
        "Calculate right-minus-left fork directionality from timing and scalar/spatial fork speed.",
    ),
    MappingRoute(
        "direction_to_timing", "Reconstruct timing from directionality", ("fork_directionality", "fork_speed"),
        "replication_timing", "reconstruct", ("space",),
        "Integrate directionality using a constant fork speed and optional timing anchor.",
    ),
    MappingRoute(
        "fraction", "Map replicated fraction", ("firing_rate", "fork_speed"),
        "replicated_fraction", "mean_field", ("time",),
        "Compute the genome-wide replicated fraction on an explicit time grid.",
    ),
    MappingRoute(
        "fraction_xt", "Map replicated fraction in space-time", ("firing_rate", "fork_speed"),
        "replicated_fraction_xt", "mean_field", ("spacetime",),
        "Compute the replicated fraction for every genomic bin and time point.",
    ),
    MappingRoute(
        "initiation", "Map initiation events", ("firing_rate", "fork_speed"),
        "initiation_events", "mean_field", ("space",),
        "Compute time-integrated mean-field initiation events per genomic bin.",
    ),
    MappingRoute(
        "initiation_xt", "Map initiation events in space-time", ("firing_rate", "fork_speed"),
        "initiation_events_xt", "mean_field", ("spacetime",),
        "Compute ensemble-mean initiation counts per genomic and time bin.",
    ),
    MappingRoute(
        "coalescence", "Map coalescence events", ("firing_rate", "fork_speed"),
        "coalescence_events", "mean_field", ("space",),
        "Compute time-integrated fork-coalescence events per genomic bin.",
    ),
    MappingRoute(
        "coalescence_xt", "Map coalescence events in space-time", ("firing_rate", "fork_speed"),
        "coalescence_events_xt", "mean_field", ("spacetime",),
        "Compute ensemble-mean coalescence counts per genomic and time bin.",
    ),
    MappingRoute(
        "right_density_xt", "Map right-moving fork density", ("firing_rate", "fork_speed"),
        "right_fork_density_xt", "mean_field", ("spacetime",),
        "Compute the mean-field right-moving fork density in space-time.",
    ),
    MappingRoute(
        "left_density_xt", "Map left-moving fork density", ("firing_rate", "fork_speed"),
        "left_fork_density_xt", "mean_field", ("spacetime",),
        "Compute the mean-field left-moving fork density in space-time.",
    ),
    MappingRoute(
        "total_density_xt", "Map total fork density", ("firing_rate", "fork_speed"),
        "fork_density_xt", "mean_field", ("spacetime",),
        "Compute the total active-fork density in space-time.",
    ),
)
ROUTES_BY_ID = {route.route_id: route for route in ROUTES}

FEATURE_LABELS = {
    "replication_timing": "Replication timing",
    "firing_rate": "Firing rate",
    "fork_speed": "Fork speed",
    "fork_directionality": "Fork directionality",
}
FEATURE_UNITS = {
    "replication_timing": "min",
    "firing_rate": "kb^-1 min^-1",
    "fork_speed": "kb/min",
    "fork_directionality": "dimensionless",
}
OUTPUT_UNITS = {
    "replication_timing": "min",
    "firing_rate": "kb^-1 min^-1",
    "fork_speed": "kb/min",
    "fork_directionality": "right - left fraction",
    "replicated_fraction": "fraction",
    "replicated_fraction_xt": "fraction",
    "initiation_events": "events/kb",
    "initiation_events_xt": "events kb^-1 min^-1",
    "coalescence_events": "events/kb",
    "coalescence_events_xt": "events kb^-1 min^-1",
    "right_fork_density_xt": "forks/kb",
    "left_fork_density_xt": "forks/kb",
    "fork_density_xt": "forks/kb",
}


def _route_selector() -> MappingRoute:
    key = persistent_widget_key(PAGE_NAMESPACE, "route")
    options = [route.route_id for route in ROUTES]
    if st.session_state.get(key) not in options:
        st.session_state[key] = options[0]
    route_id = st.selectbox(
        "Mapping operation",
        options,
        format_func=lambda value: ROUTES_BY_ID[value].label,
        key=key,
    )
    route = ROUTES_BY_ID[route_id]
    st.caption(route.description)
    return route


def _render_sources(route: MappingRoute) -> dict[str, FieldInput | None]:
    sources: dict[str, FieldInput | None] = {}
    for feature in route.source_features:
        label = FEATURE_LABELS[feature]
        allow_constant = feature in {"firing_rate", "fork_speed"}
        allow_profile = not (
            route.family == "reconstruct" and feature == "fork_speed"
        )
        default = 1.4 if feature == "fork_speed" else 0.001
        sources[feature] = render_field_input(
            f"{PAGE_NAMESPACE}_{feature}",
            feature,
            label,
            allow_constant=allow_constant,
            allow_profile=allow_profile,
            constant_default=default,
            constant_min=1e-9 if feature == "fork_speed" else 0.0,
            constant_step=0.1 if feature == "fork_speed" else 0.0001,
            units=FEATURE_UNITS[feature],
            default_mode="Constant" if feature == "fork_speed" else "Profile",
            help_text=(
                "A one-dimensional vector is interpreted as a spatial profile. "
                "Time-only fields must be prepared as a (1, T) matrix and space-time "
                "fields as (n, T)."
            ),
        )
    return sources


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


def _render_configuration(
    route: MappingRoute,
    sources: Mapping[str, FieldInput | None],
    action_container: Any | None = None,
) -> tuple[bool, MappingConfig, dict[str, Any], str]:
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

    with st.container():
        region_panel = st.expander("Region", expanded=False)
        resolution_columns = region_panel.columns(2, gap="small")
        resolution_x = float(resolution_columns[0].number_input(
            "Spatial resolution (kb)", min_value=1e-6, step=0.1, key=resolution_key,
            help="Left-edge genomic coordinates use a half-open [start, end) interval.",
        ))
        missing = resolution_columns[1].selectbox(
            "Missing values", ("error", "interpolate", "zero"),
            key=persistent_widget_key(PAGE_NAMESPACE, "missing"),
            help="Missing genomic bins are never dropped because that would change physical distances.",
        )
        interval_columns = region_panel.columns(2, gap="small")
        region_start = float(interval_columns[0].number_input(
            "Region start (kb)", min_value=0.0, step=resolution_x, key=start_key,
        ))
        region_end = float(interval_columns[1].number_input(
            "Region end (kb, exclusive)", min_value=resolution_x, step=resolution_x, key=end_key,
        ))

        mapping_panel = st.expander("Mapping settings", expanded=False)
        target_form = route.target_forms[0]
        if len(route.target_forms) > 1:
            target_form = mapping_panel.selectbox(
                "Target form", route.target_forms,
                key=persistent_widget_key(PAGE_NAMESPACE, f"{route.route_id}_target_form"),
                help="Space-time matrices use rows for genomic bins and columns for time points.",
            )

        time_dependent_source = any(
            field is not None and field.data_form in {"time", "spacetime"}
            for field in sources.values()
        )
        requires_explicit_grid = route.family == "mean_field" or (
            route.family == "predict" and time_dependent_source
        )
        fixed_temporal_sampling = (
            route.family == "fit_firing"
            and target_form == "space"
            and time_dependent_source
        )
        optional_grid = route.family in {"fit_firing", "fit_speed"} and (
            target_form in {"time", "spacetime"}
            or (time_dependent_source and not fixed_temporal_sampling)
        )
        resolution_t = 5.0
        time_max: float | None = None
        use_custom_time = requires_explicit_grid
        if requires_explicit_grid or optional_grid:
            if optional_grid:
                use_custom_time = region_panel.checkbox(
                    "Override the package-derived fit time grid",
                    key=initialized_widget_key(
                        PAGE_NAMESPACE, f"{route.route_id}_custom_time", False
                    ),
                    help="Leave off to preserve DNAscape's route-specific automatic fit horizon.",
                )
            time_columns = region_panel.columns(2, gap="small")
            resolution_t = float(time_columns[0].number_input(
                "Time step (min)", min_value=1e-6, step=1.0,
                key=initialized_widget_key(
                    PAGE_NAMESPACE, f"{route.route_id}_resolution_t", 5.0
                ),
            ))
            if use_custom_time:
                time_max = float(time_columns[1].number_input(
                    "Time window (min)", min_value=resolution_t, step=10.0,
                    key=initialized_widget_key(
                        PAGE_NAMESPACE, f"{route.route_id}_time_max", 300.0
                    ),
                ))
            else:
                time_columns[1].caption("Automatic time horizon")
        elif fixed_temporal_sampling:
            region_panel.caption(
                "Spatial firing inference from a time-dependent speed field uses the "
                "package route's fixed 5-minute temporal sampling."
            )

        periodic_allowed = route.family in {"predict", "direction", "reconstruct"} or (
            route.family == "mean_field"
            and route.route_id in {
                "coalescence", "coalescence_xt", "right_density_xt",
                "left_density_xt", "total_density_xt",
            }
        )
        periodic = False
        if periodic_allowed:
            periodic = mapping_panel.checkbox(
                "Periodic / circular boundary",
                key=initialized_widget_key(
                    PAGE_NAMESPACE, f"{route.route_id}_periodic", False
                ),
                help="Off uses linear chromosome boundaries.",
            )

        advanced: dict[str, Any] = {}
        with mapping_panel:
            if route.family in {"fit_firing", "fit_speed"}:
                advanced["maxiter"] = int(st.number_input(
                    "Fit iterations", min_value=0, step=1,
                    key=initialized_widget_key(
                        PAGE_NAMESPACE, f"{route.route_id}_maxiter", 10
                    ),
                    help="Zero returns initialization only; it is not a completed fit.",
                ))
                override_step = st.checkbox(
                    "Override route-specific fit step",
                    key=initialized_widget_key(
                        PAGE_NAMESPACE, f"{route.route_id}_override_step", False
                    ),
                )
                if override_step:
                    advanced["fit_step"] = float(st.number_input(
                        "Fit step", min_value=1e-9, format="%.6g",
                        key=initialized_widget_key(
                            PAGE_NAMESPACE, f"{route.route_id}_fit_step", 0.1
                        ),
                    ))
                use_error = st.checkbox(
                    "Use MSE early-stop threshold",
                    key=initialized_widget_key(
                        PAGE_NAMESPACE, f"{route.route_id}_use_error", False
                    ),
                )
                if use_error:
                    advanced["error"] = float(st.number_input(
                        "MSE threshold", min_value=0.0, format="%.6g",
                        key=initialized_widget_key(
                            PAGE_NAMESPACE, f"{route.route_id}_error", 0.001
                        ),
                    ))
                use_neighbourhood = st.checkbox(
                    "Override fitting neighbourhood",
                    key=initialized_widget_key(
                        PAGE_NAMESPACE, f"{route.route_id}_use_neighbourhood", False
                    ),
                )
                if use_neighbourhood:
                    advanced["neighbourhood_range"] = float(st.number_input(
                        "Neighbourhood range (kb)", min_value=resolution_x,
                        key=initialized_widget_key(
                            PAGE_NAMESPACE, f"{route.route_id}_neighbourhood", 2000.0
                        ),
                    ))
                if route.family == "fit_firing":
                    advanced["err_threshold"] = float(st.number_input(
                        "Firing floor exponent", min_value=0.0, step=1.0,
                        key=initialized_widget_key(
                            PAGE_NAMESPACE, "firing_floor_exponent", 15.0
                        ),
                        help="The numerical floor is 10 raised to minus this value.",
                    ))
                if target_form in {"time", "spacetime"} and route.family == "fit_firing":
                    advanced["target_shape"] = st.selectbox(
                        "Temporal firing shape", ("weibull", "free"),
                        key=persistent_widget_key(PAGE_NAMESPACE, "firing_target_shape"),
                    )
                    if advanced["target_shape"] == "weibull":
                        wb = st.columns(3, gap="small")
                        advanced["weibull_shape"] = float(wb[0].number_input(
                            "Weibull shape", min_value=1e-6,
                            key=initialized_widget_key(
                                PAGE_NAMESPACE, "weibull_shape", 2.0
                            ),
                        ))
                        advanced["weibull_delay"] = float(wb[1].number_input(
                            "Weibull delay (min)", min_value=0.0,
                            key=initialized_widget_key(
                                PAGE_NAMESPACE, "weibull_delay", 0.0
                            ),
                        ))
                        advanced["fit_weibull_shape"] = wb[2].checkbox(
                            "Fit shape",
                            key=initialized_widget_key(
                                PAGE_NAMESPACE, "fit_weibull_shape", False
                            ),
                        )
                    regularization = st.columns(2, gap="small")
                    advanced["alpha_smooth"] = float(regularization[0].number_input(
                        "Temporal smoothness", min_value=0.0, format="%.6g",
                        key=initialized_widget_key(
                            PAGE_NAMESPACE, "alpha_smooth", 0.01
                        ),
                    ))
                    advanced["alpha_prior"] = float(regularization[1].number_input(
                        "Temporal prior", min_value=0.0, format="%.6g",
                        key=initialized_widget_key(
                            PAGE_NAMESPACE, "alpha_prior", 0.0
                        ),
                    ))
                if route.family == "fit_speed":
                    advanced["target_shape"] = "best"
                    advanced["initial_fork_speed"] = float(st.number_input(
                        "Initial fork speed (kb/min)", min_value=1e-9,
                        key=initialized_widget_key(
                            PAGE_NAMESPACE, "initial_fork_speed", 1.4
                        ),
                    ))
                if target_form == "spacetime":
                    update_mode = st.selectbox(
                        "Space-time update",
                        ("factorized", "mixed"),
                        key=persistent_widget_key(PAGE_NAMESPACE, f"{route.route_id}_spacetime_update"),
                        help="Mixed requests cellwise updates; factorized alternates spatial and temporal phases.",
                    )
                    advanced["spacetime_update"] = update_mode
                    if update_mode == "factorized":
                        order_options = (
                            ("crossed", "space_then_time", "time_then_space")
                            if route.family == "fit_speed"
                            else ("space_then_time", "time_then_space")
                        )
                        advanced["spacetime_order"] = st.selectbox(
                            "Factorized phase order",
                            order_options,
                            key=persistent_widget_key(PAGE_NAMESPACE, f"{route.route_id}_spacetime_order"),
                        )
            if route.family in {"direction", "reconstruct", "fit_speed"}:
                if st.checkbox(
                    "Set timing anchor T0",
                    key=initialized_widget_key(
                        PAGE_NAMESPACE, f"{route.route_id}_use_t0", False
                    ),
                ):
                    advanced["T0"] = float(st.number_input(
                        "T0 (min)", min_value=0.0,
                        key=initialized_widget_key(
                            PAGE_NAMESPACE, f"{route.route_id}_t0", 0.0
                        ),
                    ))
            if route.family in {"direction", "reconstruct"}:
                if st.checkbox(
                    "Smooth timing before differentiation/integration",
                    key=initialized_widget_key(
                        PAGE_NAMESPACE, f"{route.route_id}_use_smooth", False
                    ),
                ):
                    advanced["smoothw"] = int(st.number_input(
                        "Smoothing window (bins)", min_value=1, step=1,
                        key=initialized_widget_key(
                            PAGE_NAMESPACE, f"{route.route_id}_smoothw", 5
                        ),
                    ))
            if route.route_id in {
                "coalescence", "coalescence_xt", "right_density_xt",
                "left_density_xt", "total_density_xt",
            }:
                advanced["cfl"] = float(st.number_input(
                    "CFL factor", min_value=0.01, max_value=1.0,
                    key=initialized_widget_key(
                        PAGE_NAMESPACE, f"{route.route_id}_cfl", 0.8
                    ),
                    help="Stability factor for light-cone propagation.",
                ))

        automatic_horizon = None
        if time_max is None and target_form in {"time", "spacetime"}:
            timing_source = sources.get("replication_timing")
            observed_max = 0.0
            if timing_source is not None:
                timing_values = np.asarray(timing_source.values, dtype=float)
                if timing_values.size and np.any(np.isfinite(timing_values)):
                    observed_max = float(np.nanmax(timing_values))
            automatic_horizon = max(
                1200.0,
                1.5 * observed_max,
                observed_max + 10.0 * resolution_t,
            )

        config = MappingConfig(
            target=route.target,
            target_form=target_form,
            route_id=route.route_id,
            resolution_x=resolution_x,
            resolution_t=resolution_t,
            region_start_kb=region_start,
            region_end_kb=region_end,
            periodic=periodic,
            maxiter=int(advanced.get("maxiter", 10)),
            fit_step=advanced.get("fit_step"),
            time_max=time_max,
            automatic_time_horizon_min=automatic_horizon,
            target_shape=advanced.get("target_shape"),
            weibull_shape=float(advanced.get("weibull_shape", 2.0)),
            weibull_delay=float(advanced.get("weibull_delay", 0.0)),
            fit_weibull_shape=bool(advanced.get("fit_weibull_shape", False)),
            cfl=advanced.get("cfl"),
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
        if action_container is None:
            action_container = st.container(key="dnascape-control-action")
        with action_container:
            submitted = st.button(
                "Map",
                type="primary",
                key="mapping_run_button",
                width="stretch",
                disabled=(
                    any(source is None for source in sources.values()) or execution_busy
                ),
            )
    return submitted, config, advanced, str(missing)


def build_mapping_assumptions(
    route: MappingRoute,
    config: MappingConfig,
    advanced: Mapping[str, Any],
    source_kinds: Mapping[str, str],
    n_bins: int,
    interval_kb: tuple[float, float],
) -> dict[str, Any]:
    """Build only kwargs supported by the selected rmap route family."""

    assumptions: dict[str, Any] = {"resolution": config.resolution_x, "verbose": False}
    time_grid = (
        build_time_grid(config.time_max, config.resolution_t)
        if config.time_max is not None
        else None
    )
    if route.family in {"fit_firing", "fit_speed"}:
        assumptions.update(target_form=config.target_form, maxiter=config.maxiter)
        allowed = {"fit_step", "error", "neighbourhood_range", "T0"}
        if route.family == "fit_firing":
            allowed |= {
                "err_threshold", "target_shape", "weibull_shape", "weibull_delay",
                "fit_weibull_shape", "alpha_smooth", "alpha_prior", "spacetime_order",
                "spacetime_update",
            }
        else:
            allowed |= {
                "target_shape", "initial_fork_speed", "spacetime_order",
                "spacetime_update",
            }
        assumptions.update({key: value for key, value in advanced.items() if key in allowed})
        time_dependent = any(
            kind in {"time", "spacetime"} for kind in source_kinds.values()
        )
        fixed_temporal_sampling = (
            route.family == "fit_firing"
            and config.target_form == "space"
            and time_dependent
        )
        if not fixed_temporal_sampling and (
            config.target_form in {"time", "spacetime"} or time_dependent
        ):
            assumptions["resolution_time"] = config.resolution_t
            if time_grid is not None:
                assumptions["time_grid"] = time_grid
        assumptions["n_bins"] = int(n_bins)
    elif route.family == "mean_field":
        assumptions.update(
            resolution_time=config.resolution_t,
            time_grid=time_grid,
            max_time=config.time_max,
            interval_kb=interval_kb,
            n_bins=int(n_bins),
            perQ=config.periodic,
        )
        if "cfl" in advanced:
            assumptions["cfl"] = float(advanced["cfl"])
    elif route.family == "predict":
        assumptions.update(
            perQ=config.periodic,
            interval_kb=interval_kb,
            n_bins=int(n_bins),
        )
        if any(kind in {"time", "spacetime"} for kind in source_kinds.values()):
            assumptions.update(
                resolution_time=config.resolution_t,
                time_grid=time_grid,
                sim_number=1,
                seed=1,
                retain_rep_times_per_sim=False,
                retain_inter_origin_distances=False,
            )
    else:
        assumptions["perQ"] = config.periodic
        assumptions.update({key: advanced[key] for key in ("T0", "smoothw") if key in advanced})
    return assumptions


def _prepare_sources(
    sources: Mapping[str, FieldInput],
    config: MappingConfig,
    missing: str,
) -> tuple[dict[str, Any], dict[str, str], np.ndarray, int]:
    time_grid = (
        build_time_grid(config.time_max, config.resolution_t)
        if config.time_max is not None
        else None
    )
    prepared: dict[str, Any] = {}
    fields = []
    kinds: dict[str, str] = {}
    for feature, source in sources.items():
        if source.data_form == "scalar":
            prepared[feature] = float(source.values)
            kinds[feature] = "scalar"
            continue
        if feature in {"replication_timing", "fork_directionality"} and source.data_form != "space":
            raise ValueError(f"{FEATURE_LABELS[feature]} must be a one-dimensional spatial profile.")
        field = prepare_profile_field(
            source.values,
            source.data_form,
            resolution_kb=config.resolution_x,
            region_start_kb=config.region_start_kb,
            region_end_kb=config.region_end_kb,
            position_kb=spatial_axis_for_input(source, config.resolution_x),
            missing=missing,
            name=FEATURE_LABELS[feature],
            expected_time_points=time_grid.size if time_grid is not None else None,
        )
        prepared[feature] = field.values
        kinds[feature] = field.field_kind
        fields.append(field)

    position = validate_aligned_positions(*fields)
    n_bins = int(position.size) if position is not None else interval_bin_count(
        config.region_start_kb, config.region_end_kb, config.resolution_x
    )
    if position is None:
        position = half_open_positions(n_bins, config.resolution_x, config.region_start_kb)
    validate_positive_fields(prepared)
    if "fork_directionality" in prepared:
        directionality = np.asarray(prepared["fork_directionality"], dtype=float)
        if np.any(np.abs(directionality) > 1 + 1e-9):
            raise ValueError("Fork directionality must lie between -1 and 1.")
    if "replication_timing" in prepared and np.any(
        np.asarray(prepared["replication_timing"], dtype=float) < 0
    ):
        raise ValueError("Replication timing must be non-negative.")
    return prepared, kinds, position, n_bins


def _readonly_array(value: Any) -> np.ndarray:
    array = np.asarray(value)
    try:
        array.setflags(write=False)
    except ValueError:
        pass
    return array


def _execute_mapping(
    source: Mapping[str, Any],
    target: str,
    assumptions: Mapping[str, Any],
    position_kb: np.ndarray,
    resolution_t: float,
    target_form: str,
    route_label: str,
    *,
    job_context: JobContext,
) -> dict[str, Any]:
    from dnascape.routing import rmap
    from dnascape.utils import progress_context

    started = time.perf_counter()
    job_context.report(0.03, "Preparing mapping...")
    execution_assumptions = dict(assumptions)
    execution_assumptions["verbose"] = True
    with progress_context(
        job_context.report,
        job_context.raise_if_cancelled,
        suppress_output=True,
    ):
        info = rmap(
            source=dict(source), target=target, assumptions=execution_assumptions,
            return_info=True,
        )
    job_context.raise_if_cancelled()
    job_context.report(0.96, "Preparing results...")
    values = info["values"]
    outputs: dict[str, Any] = {}
    if isinstance(values, Mapping):
        for name, value in values.items():
            if value is not None and np.asarray(value).dtype.kind in "biufc":
                outputs[str(name)] = _readonly_array(value)
    else:
        outputs[target] = _readonly_array(values)
    outputs["position_kb"] = _readonly_array(position_kb)

    time_grid = info.get("time_grid")
    if time_grid is None:
        time_grid = assumptions.get("time_grid")
    if time_grid is None and target_form in {"time", "spacetime"}:
        numerical = next((value for value in outputs.values() if isinstance(value, np.ndarray)), None)
        if numerical is not None and numerical.ndim:
            time_size = numerical.shape[-1]
            time_grid = np.arange(time_size, dtype=float) * float(resolution_t)
    if time_grid is not None:
        outputs["time_grid"] = _readonly_array(time_grid)

    output_forms: dict[str, str] = {}
    for name, value in outputs.items():
        if name in {"position_kb", "time_grid"}:
            continue
        if name == target:
            output_forms[name] = target_form
        elif value.ndim == 0:
            output_forms[name] = "scalar"
        elif name.endswith("_xt") or value.ndim == 2 and value.shape[0] > 1:
            output_forms[name] = "spacetime"
        elif value.ndim == 2 and value.shape[0] == 1:
            output_forms[name] = "time"
        else:
            output_forms[name] = "space"

    diagnostics = {
        key: value for key, value in info.items()
        if key not in {"values", "source", "assumptions"}
        and (value is None or isinstance(value, (str, int, float, bool)))
    }
    return {
        "outputs": outputs,
        "target": target,
        "target_form": target_form,
        "method": str(info.get("method", "rmap")),
        "route_label": route_label,
        "elapsed_seconds": float(time.perf_counter() - started),
        "diagnostics": diagnostics,
        "output_forms": output_forms,
    }


def _consume_job(job: BackgroundJob) -> bool:
    if job.status == "complete" and not job.consumed:
        payload = job.result
        if not isinstance(payload, Mapping):
            job.status = "error"
            job.error_type = "InvalidJobResult"
            job.error_message = "Mapping returned an invalid result."
            job.result = None
            return False
        set_workflow_result(
            "mapping", payload, job.input_signature,
            metadata={
                "route": payload.get("route_label", job.label),
                "method": payload.get("method", "rmap"),
                "elapsed_seconds": payload.get("elapsed_seconds"),
            },
        )
        job.consumed = True
        job.result = None
        st.session_state.pop(PLOT_KEY, None)
        st.session_state.pop(RAW_EXPORT_KEY, None)
        st.session_state.pop(FIGURE_EXPORT_KEY, None)
        return True
    return False


@st.fragment(run_every=0.5)
def _render_running_job() -> None:
    job = st.session_state.get(JOB_KEY)
    if not isinstance(job, BackgroundJob) or not job.running:
        return
    poll_background_job(job)
    if not job.running:
        _consume_job(job)
        st.rerun()

    st.progress(job.progress_value, text=job.progress_text)
    status_columns = st.columns([0.7, 0.3], vertical_alignment="center")
    status_columns[0].caption(f"Elapsed {job.elapsed_seconds:.1f} s")
    if status_columns[1].button(
        "Cancel",
        key="mapping_cancel_job",
        width="stretch",
        disabled=job.cancellation_requested,
    ):
        cancel_background_job(job)
        st.rerun()


def _render_job_status() -> None:
    job = st.session_state.get(JOB_KEY)
    if not isinstance(job, BackgroundJob):
        return
    poll_background_job(job)
    if _consume_job(job):
        st.rerun()
    if job.running:
        _render_running_job()
    elif job.status == "error":
        st.error(f"{job.label} failed ({job.error_type}): {job.error_message}")
    elif job.status == "cancelled":
        st.info("Mapping cancelled.")


def _numeric_outputs(payload: Mapping[str, Any]) -> dict[str, np.ndarray]:
    container = payload.get("outputs", {})
    if not isinstance(container, Mapping):
        return {}
    axes = {"position_kb", "time_grid", "time_min", "position_bp", "start_bp"}
    return {
        str(name): value for name, value in container.items()
        if name not in axes and isinstance(value, np.ndarray) and value.dtype.kind in "biufc"
    }


def _axes_for_output(payload: Mapping[str, Any], values: np.ndarray) -> dict[str, np.ndarray]:
    outputs = payload.get("outputs", {})
    axes: dict[str, np.ndarray] = {}
    if isinstance(outputs, Mapping):
        for name in ("position_kb", "time_grid"):
            axis = outputs.get(name)
            if isinstance(axis, np.ndarray) and axis.size in values.shape:
                axes[name] = axis
    return axes


def _build_result_figure(
    payload: Mapping[str, Any], output_name: str, title: str, color: str, cmap: str,
    overlay_id: str = "",
    *,
    full_resolution: bool = False,
):
    values = _numeric_outputs(payload)[output_name]
    axes = _axes_for_output(payload, values)
    squeezed = np.squeeze(values)
    units = OUTPUT_UNITS.get(output_name, "value")
    if squeezed.ndim <= 1:
        y = squeezed.reshape(-1)
        time_axis = axes.get("time_grid")
        position_axis = axes.get("position_kb")
        if (
            time_axis is not None
            and time_axis.size == y.size
            and (
                output_name == "replicated_fraction"
                or payload.get("output_forms", {}).get(output_name) == "time"
            )
        ):
            x, x_label = time_axis, "Time (min)"
        else:
            x = position_axis if position_axis is not None and position_axis.size == y.size else np.arange(y.size)
            x_label = "Position (kb)" if position_axis is not None else "Index"
        figure, plot_axes = line_figure(
            x, y, title=title, x_label=x_label, y_label=units, color=color,
            label=output_name.replace("_", " "),
            full_resolution=full_resolution,
        )
        if overlay_id:
            views = {view.view_id: view for view in available_data_views(include_workflows=False)}
            overlay = views.get(overlay_id)
            if overlay is not None:
                other = view_series(overlay)
                other_x = view_axis(overlay, other.size)
                if other.size != y.size or not np.allclose(other_x, x):
                    raise ValueError(
                        "Observed-data overlays must have the same length and physical coordinate "
                        "axis as the mapped result. Align them on Data Management first."
                    )
                if full_resolution:
                    x_plot, y_plot = other_x, other
                else:
                    x_plot, y_plot = preview_series(other_x, other, max_points=5000)
                plot_axes.plot(x_plot, y_plot, color="#d97757", linewidth=1.4, alpha=0.85, label=overlay.name)
                plot_axes.legend(loc="best")
        return figure
    if squeezed.ndim == 2:
        return heatmap_figure(
            squeezed,
            title=title,
            x_label="Time (min)",
            y_label="Position (kb)",
            colorbar_label=units,
            x_values=axes.get("time_grid"),
            y_values=axes.get("position_kb"),
            cmap=cmap,
            full_resolution=full_resolution,
        )[0]
    raise ValueError("The selected mapping output cannot be plotted directly.")


def _selected_output(payload: Mapping[str, Any]) -> str:
    arrays = _numeric_outputs(payload)
    if not arrays:
        raise ValueError("The mapping result contains no numerical output.")
    output_names = list(arrays)
    output_key = persistent_widget_key(PAGE_NAMESPACE, "result_output")
    selected = st.session_state.get(output_key)
    if selected not in output_names:
        preferred = payload.get("target")
        selected = preferred if preferred in arrays else output_names[0]
        st.session_state[output_key] = selected
    return str(selected)


def _plot_preferences(
    payload: Mapping[str, Any], output_name: str | None = None
) -> tuple[str, str, str, str, str]:
    output_name = output_name or _selected_output(payload)
    title_key = persistent_widget_key(PAGE_NAMESPACE, f"plot_title_{output_name}")
    color_key = persistent_widget_key(PAGE_NAMESPACE, "plot_color")
    cmap_key = persistent_widget_key(PAGE_NAMESPACE, "plot_cmap")
    overlay_key = persistent_widget_key(PAGE_NAMESPACE, f"plot_overlay_{output_name}")
    return (
        output_name,
        str(st.session_state.get(title_key, output_name.replace("_", " ").title())),
        str(st.session_state.get(color_key, "#2c7f77")),
        str(st.session_state.get(cmap_key, "viridis")),
        str(st.session_state.get(overlay_key, "")),
    )


def _plot_signature(stored, settings: tuple[str, str, str, str, str]) -> str:
    return scientific_signature(stored.input_signature, *settings)


def _store_plot_preview(stored, settings: tuple[str, str, str, str, str]) -> None:
    output_name, title, color, cmap, overlay_id = settings
    figure = None
    try:
        figure = _build_result_figure(
            stored.payload,
            output_name,
            title,
            color,
            cmap,
            overlay_id,
            full_resolution=False,
        )
        preview = serialize_figure(figure, "png", dpi=115)
    finally:
        if figure is not None:
            release_figure(figure)
    st.session_state[PLOT_KEY] = {
        "signature": _plot_signature(stored, settings),
        "result_signature": stored.input_signature,
        "payload": preview.payload,
        "output": output_name,
        "settings": settings,
    }


def _mapping_figure_download_bytes(
    payload: Mapping[str, Any],
    output_name: str,
    title: str,
    color: str,
    cmap: str,
    overlay_id: str,
    figure_format: str,
    full_resolution: bool,
) -> bytes:
    """Build a requested mapping figure only when Download is clicked."""

    figure = None
    try:
        figure = _build_result_figure(
            payload,
            output_name,
            title,
            color,
            cmap,
            overlay_id,
            full_resolution=full_resolution,
        )
        return serialize_figure(figure, figure_format, dpi=220).payload
    finally:
        if figure is not None:
            release_figure(figure)


def _render_plotting_settings(
    stored,
    target: str,
    target_form: str,
) -> dict[str, Any]:
    """Render plot controls before or after a mapping result exists."""

    arrays = _numeric_outputs(stored.payload) if stored is not None else {}
    output_names = list(arrays) if arrays else [target]
    with st.expander("Plotting settings", expanded=False):
        output_key = persistent_widget_key(PAGE_NAMESPACE, "result_output")
        if st.session_state.get(output_key) not in output_names:
            st.session_state[output_key] = target
        output_name = st.selectbox(
            "Feature",
            output_names,
            format_func=lambda value: value.replace("_", " ").title(),
            key=output_key,
        )
        title_key = persistent_widget_key(
            PAGE_NAMESPACE, f"plot_title_{output_name}"
        )
        if title_key not in st.session_state:
            st.session_state[title_key] = output_name.replace("_", " ").title()
        title = st.text_input("Title", key=title_key)
        style_columns = st.columns(2, gap="small")
        color = style_columns[0].color_picker(
            "Line color",
            key=initialized_widget_key(
                PAGE_NAMESPACE, "plot_color", "#2c7f77"
            ),
        )
        cmap = style_columns[1].selectbox(
            "Heatmap palette",
            ("viridis", "magma", "plasma", "cividis"),
            key=persistent_widget_key(PAGE_NAMESPACE, "plot_cmap"),
            disabled=(
                stored.payload.get("output_forms", {}).get(output_name, target_form)
                != "spacetime"
                if stored is not None
                else target_form != "spacetime"
            ),
        )
        overlay_id = ""
        output_form = (
            stored.payload.get("output_forms", {}).get(output_name, target_form)
            if stored is not None
            else target_form
        )
        values = arrays.get(output_name)
        if output_form == "space" and (
            values is None or np.squeeze(values).ndim <= 1
        ):
            views = available_data_views(include_workflows=False)
            ids = [""] + [view.view_id for view in views]
            by_id = {view.view_id: view for view in views}
            overlay_key = persistent_widget_key(
                PAGE_NAMESPACE, f"plot_overlay_{output_name}"
            )
            if st.session_state.get(overlay_key) not in ids:
                st.session_state[overlay_key] = ""
            overlay_id = st.selectbox(
                "Observed-data overlay",
                ids,
                format_func=lambda value: "None" if not value else by_id[value].name,
                key=overlay_key,
            )

        settings = (output_name, title, color, cmap, overlay_id)
        can_plot = stored is not None and output_name in arrays
        if st.button(
            "Update plot",
            key="mapping_update_plot",
            type="primary",
            width="stretch",
            disabled=not can_plot,
            help="Apply these settings without rerunning mapping.",
        ):
            try:
                with st.spinner("Updating preview..."):
                    _store_plot_preview(stored, settings)
            except (ValueError, TypeError, OSError, MemoryError) as exc:
                st.error(str(exc))

        plot_signature = (
            _plot_signature(stored, settings)
            if stored is not None
            else scientific_signature("pending", *settings)
        )
        plot = st.session_state.get(PLOT_KEY)
        settings_changed = bool(
            can_plot
            and isinstance(plot, dict)
            and plot.get("signature") != plot_signature
        )
        if settings_changed:
            st.caption("Settings changed - select Update plot to apply them.")

    return {
        "output": output_name,
        "title": title,
        "color": color,
        "cmap": cmap,
        "overlay_id": overlay_id,
        "signature": plot_signature,
        "settings_changed": settings_changed,
    }


def _compact_result_summary(
    stored, output_name: str, values: np.ndarray
) -> None:
    summary_key = f"workflow:mapping:{stored.input_signature}:{output_name}"
    summary = get_dataset_summary(summary_key)
    if summary is None:
        summary = set_dataset_summary(summary_key, array_summary(values))

    payload = stored.payload
    shape = " x ".join(map(str, values.shape)) or "scalar"
    details = [
        f"{output_name.replace('_', ' ').title()} ({OUTPUT_UNITS.get(output_name, 'value')})",
        f"{shape}",
        memory_label(values.nbytes),
        f"{float(payload.get('elapsed_seconds', 0.0)):.2f} s",
    ]
    statistics: list[str] = []
    if summary.finite_count:
        statistics.extend(
            (
                f"mean {summary.mean:.4g}",
                f"range {summary.minimum:.4g}–{summary.maximum:.4g}",
            )
        )

        if output_name.startswith(("initiation_events", "coalescence_events")):
            factor = 1.0
            axes = _axes_for_output(payload, values)
            position = axes.get("position_kb")
            if position is not None and position.size > 1:
                factor *= float(np.median(np.diff(position)))
            time_grid = axes.get("time_grid")
            if values.ndim > 1 and time_grid is not None and time_grid.size > 1:
                factor *= float(np.median(np.diff(time_grid)))
            total = float(np.nansum(values)) * factor
            event = "origins" if output_name.startswith("initiation") else "coalescences"
            statistics.append(f"expected {event} {total:.4g}")
        elif output_name.startswith("replicated_fraction"):
            final_fraction = (
                float(np.nanmean(values[..., -1]))
                if values.ndim > 1
                else float(np.ravel(values)[-1])
            )
            statistics.append(f"final fraction {final_fraction:.4g}")
        elif "fork_density" in output_name:
            statistics.append(f"peak density {summary.maximum:.4g}")

    st.caption("Summary · " + " · ".join(details + statistics))


def _clear_mapping_output() -> None:
    clear_workflow_result("mapping")
    st.session_state.pop(PLOT_KEY, None)
    st.session_state.pop(RAW_EXPORT_KEY, None)
    st.session_state.pop(FIGURE_EXPORT_KEY, None)


def _mapping_raw_download_bytes(
    values: np.ndarray,
    output_format: str,
    axes: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
    compress: bool,
) -> bytes:
    """Serialize mapped values only when Download is clicked."""

    exported = serialize_array(
        values,
        output_format,
        axes=axes,
        metadata=metadata,
    )
    if compress:
        exported = gzip_payload(exported.payload, exported.extension)
    return exported.payload


def _render_result(
    stored,
    current_signature: str | None,
    plot_settings: Mapping[str, Any] | None,
) -> None:
    payload = stored.payload
    arrays = _numeric_outputs(payload)
    if not arrays:
        st.warning("The stored mapping result contains no numerical output.")
        return
    if current_signature and workflow_result_is_stale("mapping", current_signature):
        st.warning("Mapping inputs changed. Select Map to update this result.")

    preferred_output = str((plot_settings or {}).get("output") or "")
    if preferred_output not in arrays:
        payload_target = str(payload.get("target") or "")
        preferred_output = payload_target if payload_target in arrays else next(iter(arrays))
    settings = (
        preferred_output,
        str(
            (plot_settings or {}).get("title")
            or preferred_output.replace("_", " ").title()
        ),
        str((plot_settings or {}).get("color") or "#2c7f77"),
        str((plot_settings or {}).get("cmap") or "viridis"),
        str((plot_settings or {}).get("overlay_id") or ""),
    )
    output_name, title, color, cmap, overlay_id = settings
    values = arrays[output_name]
    plot_signature = _plot_signature(stored, settings)
    plot = st.session_state.get(PLOT_KEY)
    if not isinstance(plot, dict) or (
        plot.get("result_signature") != stored.input_signature
    ):
        try:
            with st.spinner("Rendering preview..."):
                _store_plot_preview(stored, settings)
            plot = st.session_state.get(PLOT_KEY)
        except (ValueError, TypeError, OSError, MemoryError) as exc:
            st.error(f"Could not render the mapping preview: {exc}")
            plot = None

    settings_changed = isinstance(plot, dict) and plot.get("signature") != plot_signature
    if isinstance(plot, dict):
        displayed_output = str(plot.get("output", output_name))
        if displayed_output in arrays:
            output_name = displayed_output
            values = arrays[output_name]
        if settings_changed:
            st.caption("Plot settings changed - select Update plot.")
        render_image_preview(
            plot["payload"],
            file_name=safe_name_stem("dnascape", "mapping", output_name) + ".png",
            shell_key="map-preview-shell",
            clear_key="map-preview-clear",
            on_clear=_clear_mapping_output,
            alt=f"{output_name.replace('_', ' ')} mapping preview",
            downloads={},
        )
        stored_settings = plot.get("settings")
        export_settings = (
            stored_settings
            if isinstance(stored_settings, tuple) and len(stored_settings) == 5
            else settings
        )
        (
            export_output,
            export_title,
            export_color,
            export_cmap,
            export_overlay,
        ) = tuple(str(value) for value in export_settings)
        export_stem = safe_name_stem("dnascape", "mapping", export_output)
        export_columns = st.columns(3, gap="small")
        for column, image_format in zip(export_columns, ("png", "svg", "pdf")):
            column.download_button(
                image_format.upper(),
                data=partial(
                    _mapping_figure_download_bytes,
                    payload,
                    export_output,
                    export_title,
                    export_color,
                    export_cmap,
                    export_overlay,
                    image_format,
                    True,
                ),
                file_name=f"{export_stem}.{image_format}",
                mime=figure_export_target(image_format)[1],
                key=f"mapping_export_{image_format}_{stored.input_signature[:12]}",
                on_click="ignore",
                width="stretch",
                help="Build a full-resolution export only when clicked.",
            )

    with st.expander("Raw output", expanded=False):
        st.markdown(
            '<p class="small-muted">Download the mapped output in a DNAscape-ready format.</p>',
            unsafe_allow_html=True,
        )
        st.caption("Export")
        export_columns = st.columns([0.46, 0.18, 0.12, 0.24], gap="small")
        raw_name = export_columns[0].text_input(
            "File name",
            key=persistent_widget_key(PAGE_NAMESPACE, "raw_file_name"),
            placeholder="Name (auto)",
            label_visibility="collapsed",
        )
        raw_format_options = ("CSV", "TSV", "TXT", "Array")
        raw_format_key = persistent_widget_key(PAGE_NAMESPACE, "raw_format")
        if st.session_state.get(raw_format_key) not in raw_format_options:
            st.session_state[raw_format_key] = "CSV"
        raw_format_label = export_columns[1].selectbox(
            "File type", ("CSV", "TSV", "TXT", "Array"),
            key=raw_format_key,
            label_visibility="collapsed",
        )
        compress = export_columns[2].checkbox(
            "GZ",
            key=initialized_widget_key(PAGE_NAMESPACE, "raw_gzip", False),
        )
        raw_format = "npy" if raw_format_label == "Array" else raw_format_label.lower()
        extension, mime = array_export_target(
            raw_format,
            compressed=bool(compress),
        )
        export_values = values.reshape(1) if values.ndim == 0 else values
        default_stem = safe_name_stem("dnascape", "mapping", output_name, "raw")
        file_stem = safe_name_stem(raw_name) if raw_name.strip() else default_stem
        export_columns[3].download_button(
            "Download",
            data=partial(
                _mapping_raw_download_bytes,
                export_values,
                raw_format,
                _axes_for_output(payload, values),
                {"feature": output_name, "method": payload.get("method")},
                bool(compress),
            ),
            file_name=f"{file_stem}.{extension}",
            mime=mime,
            on_click="ignore",
            width="stretch",
        )

    _compact_result_summary(stored, output_name, values)


def render_mapping_workspace() -> None:
    controls, preview = st.columns([0.35, 0.65], gap="medium")
    with controls:
        control_stack = st.container(key="dnascape-control-stack")

    with control_stack:
        control_card = st.container(
            border=True,
            height="stretch",
            key="dnascape-control-card",
        )
        control_action = st.container(key="dnascape-control-action")

    with control_card:
        with st.expander("Data", expanded=True):
            route = _route_selector()
            sources = _render_sources(route)
        submitted, config, advanced, missing = _render_configuration(
            route,
            sources,
            action_container=control_action,
        )
        plot_settings = _render_plotting_settings(
            get_workflow_result("mapping"),
            route.target,
            config.target_form,
        )

        source_ids = {
            name: source.source_id for name, source in sources.items() if source is not None
        }
        signature = scientific_signature(
            "mapping-v2", route.route_id, source_ids, config, advanced, missing
        ) if len(source_ids) == len(route.source_features) else None
        try:
            config.validate()
            estimate = config.estimate()
            st.caption(
                f"Grid: {estimate.spatial_bins:,} bins"
                + (f" × {estimate.time_points:,} time points" if estimate.time_points else "")
                + f" · about {estimate.memory_mib:.1f} MiB"
            )
            if estimate.expensive:
                st.warning("This mapping may be expensive. Narrow the interval or coarsen the grid if appropriate.")
            if estimate.impractical:
                st.error("The requested output exceeds the app's 4 GiB safety estimate.")
        except ValueError as exc:
            estimate = None
            st.error(str(exc))

        job = st.session_state.get(JOB_KEY)
        running = isinstance(job, BackgroundJob) and job.running
        if submitted:
            if running:
                st.warning("A mapping job is already running.")
            elif estimate is not None and estimate.impractical:
                st.error("Reduce the requested grid before running mapping.")
            else:
                try:
                    complete_sources = {
                        name: source for name, source in sources.items() if source is not None
                    }
                    config.validate()
                    prepared, kinds, position, n_bins = _prepare_sources(
                        complete_sources, config, missing
                    )
                    if route.family == "reconstruct" and kinds.get("fork_speed") != "scalar":
                        raise ValueError("Timing reconstruction currently supports a constant fork speed.")
                    if route.family == "direction" and kinds.get("fork_speed") not in {"scalar", "space"}:
                        raise ValueError("Directionality mapping supports scalar or spatial fork speed.")
                    effective_interval = (
                        (float(position[0]), float(position[-1] + config.resolution_x))
                        if any(kind in {"space", "spacetime"} for kind in kinds.values())
                        else (config.region_start_kb, config.region_end_kb)
                    )
                    assumptions = build_mapping_assumptions(
                        route,
                        config,
                        advanced,
                        kinds,
                        n_bins,
                        effective_interval,
                    )
                    assert signature is not None
                    st.session_state[JOB_KEY] = start_background_job(
                        route.label, signature, _execute_mapping,
                        prepared, route.target, assumptions, position,
                        config.resolution_t, config.target_form, route.label,
                    )
                    st.rerun()
                except (ValueError, TypeError, KeyError, RuntimeError, MemoryError) as exc:
                    st.error(str(exc))

    with preview:
        with st.container(border=True, key="dnascape-preview-card"):
            st.markdown(
                '<div class="plotting-panel-title">Preview</div>',
                unsafe_allow_html=True,
            )
            _render_job_status()
            stored = get_workflow_result("mapping")
            if stored is None:
                st.caption("Choose the inputs and select Map.")
            else:
                _render_result(stored, signature, plot_settings)


def render_page() -> None:
    """Render Mapping inside the persistent DNAscape shell."""

    render_page_intro(
        "Mapping",
        "Map replication kinetic landscapes",
        "Choose a target, set a precise mapping route, and preview mapped DNAscape outputs.",
    )
    render_control_panel_marker("mapping-page-marker")
    render_mapping_workspace()
