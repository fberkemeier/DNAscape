"""Coarse initialization followed by native-resolution kinetic fitting."""

import time
import inspect
import numpy as np


def adaptive_fit(observed, supplied, options):
    from ._kinetic_fit import fit_kinetic, _FORMS, _schedule, _integer

    observed = np.asarray(observed, dtype=float)
    fraction = options["observation"] not in {"timing", "replication_timing"}
    if fraction and np.any(np.isfinite(observed) & ((observed < 0.0) | (observed > 1.0))):
        raise ValueError("Replicated fraction must be between zero and one.")
    n = len(observed)
    coarse_n = min(n, max(1024, int(np.ceil(n / 4))))
    edges = np.linspace(0.0, n, coarse_n + 1)
    dx = float(options["resolution"])
    coarse_dx = n * dx / coarse_n
    form = _FORMS[str(options["target_form"]).lower()]
    space = options["space_maxiter"]
    temporal = options["temporal_maxiter"]
    if options["maxiter"] is not None:
        space = temporal = options["maxiter"]
    space, temporal = _integer(space, "Spatial iterations"), _integer(temporal, "Time iterations")
    warm_space = max(0, space - 1) if form in {"scalar", "x", "xt"} else 0
    warm_time = max(0, temporal - 1) if form in {"t", "xt"} else 0
    order = str(options["spacetime_order"]).replace("-", "_")
    if order in {"space_then_time", "space_time"}:
        warm_time = 0
    if order in {"time_then_space", "time_space"}:
        warm_space = 0
    if not warm_space and not warm_time:
        return None

    def rebin(values, harmonic=False):
        a = np.asarray(values, dtype=float)
        if a.size == 1 or a.shape[0] != n:
            return values
        if harmonic:
            a = 1.0 / a
        flat = a.reshape(n, -1)
        prefix = np.vstack((np.zeros(flat.shape[1]), np.cumsum(flat, axis=0)))
        columns = [np.interp(edges, np.arange(n + 1), column) for column in prefix.T]
        reduced = np.diff(np.asarray(columns).T, axis=0) / np.diff(edges)[:, None]
        reduced = reduced.reshape((coarse_n,) + a.shape[1:])
        return 1.0 / reduced if harmonic else reduced

    accepted = set(inspect.signature(fit_kinetic).parameters) - {
        "observed",
        "supplied",
        "compatibility",
    }
    original = {key: value for key, value in options.items() if key in accepted}
    original.update(options.get("compatibility", {}))
    original.update(
        adaptive_fitting=False, maxiter=None, _started=options.get("_started") or time.time()
    )
    total = len(_schedule(form, space, temporal, order))
    original["_progress_budget"] = options.get("_progress_budget") or total
    anchor = float(original.get("T0") or 0.0)
    if options["observation"] in {"timing", "replication_timing"}:
        reference = float(np.nanmedian(observed - anchor))
    else:
        clock = original.get("time_grid")
        if clock is None:
            clock = np.arange(observed.shape[-1]) * float(options["resolution_time"])
        finite = np.isfinite(observed)
        counts = finite.sum(axis=0)
        means = np.divide(
            np.where(finite, observed, 0.0).sum(axis=0),
            counts,
            out=np.full(observed.shape[-1], np.nan),
            where=counts > 0,
        )
        survival = np.where(np.isfinite(means), 1.0 - means, 0.0)
        reference = max(float(np.trapezoid(survival, clock)), float(options["resolution_time"]))
    original["weibull_reference"] = original.get("weibull_reference") or reference
    warm = dict(
        original,
        resolution=coarse_dx,
        n_bins=coarse_n,
        space_maxiter=warm_space,
        temporal_maxiter=warm_time,
        _warmup=True,
        adaptive_fitting=coarse_n >= 2048,
    )
    for key in ("weibull_delay", "initial_initiation_rate", "initial_fork_speed"):
        if warm.get(key) is not None:
            warm[key] = rebin(warm[key])
    valid = np.isfinite(observed)
    observation_weights = options.get("observation_weights")
    fine_weights = (valid.astype(float) if observation_weights is None else
                    np.where(valid, np.broadcast_to(observation_weights, observed.shape), 0.))
    mass, weights = rebin(np.where(valid, observed, 0.0)*fine_weights), rebin(fine_weights)
    coarse_observed = np.divide(mass, weights, out=np.full_like(mass, np.nan), where=weights > 0.0)
    if observation_weights is not None:
        warm["observation_weights"] = weights
    if fraction:
        # Weighted prefix differences can round a saturated bin to 1+ulp.
        # Only re-binned valid probabilities are clipped; raw invalid data
        # were rejected above, so this cannot conceal invalid observations.
        coarse_observed = np.clip(coarse_observed, 0.0, 1.0)
    tracker = original["_progress_tracker"]
    tracker.reserve(n, total - warm_space - warm_time)
    try:
        coarse = fit_kinetic(
            coarse_observed,
            rebin(supplied, harmonic=options["target"] == "initiation_rate"),
            **warm,
        )
    finally:
        tracker.release()
    coarse_info = coarse._rmap_info
    used_space = coarse_info["space_iterations"]
    used_time = coarse_info["time_iterations"]
    initial = dict(
        original,
        space_maxiter=space - used_space,
        temporal_maxiter=temporal - used_time,
        _warm_start=coarse_info["_parameter_state"],
        _progress_offset=options.get("_progress_offset", 0) + used_space + used_time,
    )
    result = fit_kinetic(observed, supplied, **initial)
    info = result._rmap_info
    info["coarse_fit_history"] = coarse_info["fit_history"]
    info["coarse_resolution"] = coarse_dx
    info["native_resolution"] = dx
    info["adaptive_fitting"] = True
    info["phase_history"] = [
        dict(resolution=coarse_dx, **item) if "resolution" not in item else item
        for item in coarse_info["phase_history"]
    ] + [dict(item, resolution=dx) for item in info["phase_history"]]
    for key in (
        "space_iterations",
        "time_iterations",
        "iterations",
        "iterations_completed",
        "space_iterations_completed",
        "temporal_iterations_completed",
        "objective_evaluations",
    ):
        info[key] += coarse_info[key]
    info["iteration_budget"] = info["fit_progress_total"] = total
    return result
