"""Mapping defaults, applied to the active scientific branch only.

Keep these values explicit: a global iteration cap silently changes unrelated
solvers and makes the displayed settings disagree with their accuracy budget.
Benchmark scripts and tuning reports are kept locally outside version control.
"""

from dataclasses import replace

DEFAULT_MAPPING_MAX_TIME = 1200.0


def mapping_defaults(route, settings, shape=None):
    overrides: dict[str, float | int] = {
        item.key: 100 for item in settings
        if item.key.endswith("sim_number") and item.has_default
    }
    if {"time_grid", "max_time"} <= {item.key for item in settings}:
        # Forward fields need a finite clock even when every input is static.
        # Leave explicit grids optional; the workflow derives their default
        # from this limit and the chosen temporal resolution.
        overrides["max_time"] = DEFAULT_MAPPING_MAX_TIME
    if route.target == "firing_rate" and shape == "weibull":
        overrides.update(
            maxiter=20 if route.output_dependence == "spacetime" else 10,
            alpha_smooth=0.1,
        )
    elif route.method in {
        "timing_and_speed_to_firing_rate", "timing_to_firing_constant_fork_speed",
    }:
        # code_fit.py used the homogeneous timing law's squared-ratio update.
        # Keep ten steps; the page restores 0.6 when stalling is enabled.
        overrides.update(maxiter=10, fit_step=2.0)
    elif route.target == "firing_rate" and shape == "free":
        if route.output_dependence in {"time", "spacetime"}:
            overrides.update(temporal_maxiter=30)
        if route.output_dependence == "spacetime":
            overrides.update(space_maxiter=30)
    return tuple(
        replace(item, default=overrides[item.key], has_default=True)
        if item.key in overrides else item
        for item in settings
    )
