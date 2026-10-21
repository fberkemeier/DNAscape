"""Fitting algorithms and fitting-oriented helpers."""

import math
import os
import time

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.optimize import least_squares, minimize
from scipy.signal import fftconvolve
from scipy.special import gamma

from ._spatial_firing import timing_mse_gradient
from ._periodic_firing import PeriodicFiringSmoother
from .constants import mapt
from .io import _resolve_data_path, savecsv
from .mapping import (
    _map_firing_timing_legacy_linear,
    _normalise_resolution_x,
    map_firing_timing,
    map_replicated_fraction_xt,
)
from .simulation import _resolve_stall_probability, rsim
from .utils import (
    _fit_progress_enabled,
    _report_progress,
    check_cancelled,
    etaferr,
    mean_squared_error,
    smoothf,
)

# Dense timing diagnostics create several arrays over the complete position-time
# grid. Keep exact recomputation bounded while retaining exact optimisation on
# the already subsampled fitting grid.
_FULL_TIMING_DIAGNOSTIC_CELL_CAP = 1_000_000
_FULL_TIMING_DIAGNOSTIC_WORK_CAP = 10_000_000


def _periodic_speed_inverse(
    source, speed_form, resolution, resolution_x, resolution_time, time_grid,
    T0, initial_fork_speed, speed_bounds, maxiter, verbose,
):
    """Use the deterministic constrained inverse for periodic speed targets."""
    from .inverse import map_timing_and_firing_to_speed

    return map_timing_and_firing_to_speed(
        source, target_form=speed_form,
        resolution=_normalise_resolution_x(resolution, resolution_x=resolution_x),
        resolution_time=resolution_time, time_grid=time_grid, T0=T0,
        initial_fork_speed=initial_fork_speed, speed_bounds=speed_bounds,
        maxiter=20 if maxiter is None else max(1, int(maxiter)), perQ=True,
        verbose=verbose,
    )


class _FitConverged(Exception):
    """Internal signal used to stop optimiser-backed fits early."""


def _normalise_error_threshold(error):
    if error is None:
        return None
    value = float(error)
    if not np.isfinite(value) or value < 0:
        raise ValueError("error must be None or a finite value >= 0.")
    return value


def _error_reached(value, error):
    return error is not None and np.isfinite(value) and value <= error


def _use_exact_full_timing_diagnostic(n_positions, n_time):
    """Return whether a complete position-time diagnostic grid is affordable."""
    n_positions = int(n_positions)
    n_time = int(n_time)
    cells = n_positions * n_time
    time_quadratic_work = cells * n_time
    return (
        cells <= int(_FULL_TIMING_DIAGNOSTIC_CELL_CAP)
        and time_quadratic_work <= int(_FULL_TIMING_DIAGNOSTIC_WORK_CAP)
    )


def _expand_sampled_timing_diagnostic(predicted, fit_idx, n_positions):
    """Expand a retained sampled timing prediction to every genomic position.

    Values between retained positions are linearly interpolated. Positions
    outside the fitted span use the nearest finite endpoint, which avoids
    inventing unstable edge slopes while guaranteeing a finite diagnostic.
    """
    retained = np.asarray(predicted, dtype=float).reshape(-1)
    indices = np.asarray(fit_idx, dtype=np.int64).reshape(-1)
    n_positions = int(n_positions)
    if retained.size != indices.size:
        raise ValueError("sampled timing prediction and fit indices must have matching lengths.")
    valid = np.isfinite(retained) & (indices >= 0) & (indices < n_positions)
    if not np.any(valid):
        raise ValueError("sampled timing prediction must contain at least one finite value.")
    indices = indices[valid]
    retained = retained[valid]
    order = np.argsort(indices, kind="stable")
    indices = indices[order]
    retained = retained[order]
    unique_indices, first = np.unique(indices, return_index=True)
    retained = retained[first]
    full_indices = np.arange(n_positions, dtype=float)
    return np.interp(
        full_indices,
        unique_indices.astype(float),
        retained,
        left=float(retained[0]),
        right=float(retained[-1]),
    )


def select_profile_region(
    profile,
    raw_resolution_kb=1.0,
    interval_kb=None,
    output_resolution_kb=None,
    resample_method="sample",
    input_resolution_kb=None,
):
    """Select a genomic interval from a regular spatial profile.

    Parameters
    ----------
    profile : array-like
        One-dimensional profile sampled at a regular genomic spacing.
    raw_resolution_kb : float, default 1.0
        Genomic spacing, in kb, between neighbouring values in ``profile``.
    interval_kb : tuple, optional
        ``(start_kb, end_kb)`` interval to select from the raw profile.
    output_resolution_kb : float, optional
        Spacing, in kb, of the returned profile. Defaults to
        ``raw_resolution_kb``.
    resample_method : {"sample", "nearest", "linear", "interpolate", "mean", "average"}
        Rule used to move from the raw profile to the requested output
        positions. The default, ``"sample"``, picks the nearest raw value.
    input_resolution_kb : float, optional
        Deprecated alias for ``raw_resolution_kb``.

    Returns
    -------
    selected : ndarray
        Profile values over the selected interval at ``output_resolution_kb``.
    positions_kb : ndarray
        Genomic positions, in kb, corresponding to ``selected``.
    """
    arr = np.asarray(profile)
    if arr.ndim != 1:
        raise ValueError("profile must be a one-dimensional array.")
    if arr.size == 0:
        raise ValueError("profile must contain at least one value.")

    if input_resolution_kb is not None:
        raw_resolution_kb = input_resolution_kb

    raw_resolution_kb = float(raw_resolution_kb)
    if not np.isfinite(raw_resolution_kb) or raw_resolution_kb <= 0:
        raise ValueError("raw_resolution_kb must be a finite positive value.")

    if output_resolution_kb is None:
        output_resolution_kb = raw_resolution_kb
    output_resolution_kb = float(output_resolution_kb)
    if not np.isfinite(output_resolution_kb) or output_resolution_kb <= 0:
        raise ValueError("output_resolution_kb must be a finite positive value.")

    method = str(resample_method).strip().lower().replace("-", "_")
    method_aliases = {
        "sample": "sample",
        "nearest": "sample",
        "pick": "sample",
        "linear": "linear",
        "interpolate": "linear",
        "interp": "linear",
        "mean": "mean",
        "average": "mean",
        "avg": "mean",
    }
    if method not in method_aliases:
        raise ValueError(
            "resample_method must be one of 'sample', 'linear', or 'mean'."
        )
    method = method_aliases[method]
    if method == "mean" and output_resolution_kb < raw_resolution_kb:
        raise ValueError(
            "resample_method='mean' requires output_resolution_kb >= raw_resolution_kb. "
            "Use resample_method='sample' or 'linear' for finer output positions."
        )

    genome_end_kb = arr.size * raw_resolution_kb
    if interval_kb is None:
        start_kb, end_kb = 0.0, genome_end_kb
    else:
        if len(interval_kb) != 2:
            raise ValueError("interval_kb must be a (start_kb, end_kb) pair.")
        start_kb, end_kb = map(float, interval_kb)

    if not (np.isfinite(start_kb) and np.isfinite(end_kb)):
        raise ValueError("interval_kb values must be finite.")
    if start_kb < 0 or end_kb <= start_kb:
        raise ValueError("interval_kb must satisfy 0 <= start_kb < end_kb.")
    if end_kb > genome_end_kb + 1e-9:
        raise ValueError("interval_kb extends beyond the profile length.")

    positions_kb = np.arange(start_kb, end_kb, output_resolution_kb, dtype=float)
    positions_kb = positions_kb[positions_kb < end_kb - 1e-12]
    if positions_kb.size == 0:
        return np.asarray([], dtype=arr.dtype), positions_kb

    if method == "sample":
        indices = np.floor(positions_kb / raw_resolution_kb + 0.5).astype(int)
        indices = np.clip(indices, 0, arr.size - 1)
        selected = arr[indices]
    elif method == "linear":
        raw_positions_kb = np.arange(arr.size, dtype=float) * raw_resolution_kb
        selected = np.interp(positions_kb, raw_positions_kb, arr.astype(float))
    else:
        selected = []
        for start in positions_kb:
            stop = min(start + output_resolution_kb, end_kb)
            start_idx = int(np.ceil(start / raw_resolution_kb))
            stop_idx = int(np.floor((stop - 1e-12) / raw_resolution_kb)) + 1
            start_idx = max(start_idx, 0)
            stop_idx = min(stop_idx, arr.size)
            if stop_idx <= start_idx:
                nearest = int(
                    np.clip(
                        np.floor(start / raw_resolution_kb + 0.5),
                        0,
                        arr.size - 1,
                    )
                )
                selected.append(arr[nearest])
            else:
                selected.append(np.mean(arr[start_idx:stop_idx]))
        selected = np.asarray(selected)

    return selected, positions_kb


def _uniform_speed_for_simulation(fork_speed, time_grid, resolution_time):
    """Resample an explicitly timed speed field onto rsim's uniform clock."""
    step = float(resolution_time)
    if not np.isfinite(step) or step <= 0:
        raise ValueError("resolution_time must be finite and positive.")
    speed = np.asarray(fork_speed, dtype=float)
    if time_grid is None:
        return fork_speed, None
    axis = np.asarray(time_grid, dtype=float)
    if (axis.ndim != 1 or len(axis) < 2 or not np.all(np.isfinite(axis))
            or axis[0] != 0 or np.any(np.diff(axis) <= 0)):
        raise ValueError("time_grid must begin at zero and be finite and strictly increasing.")
    uniform = np.arange(int(np.ceil(axis[-1] / step)) + 1, dtype=float) * step
    if speed.ndim == 2 and speed.shape[1] > 1:
        if speed.shape[1] != len(axis):
            raise ValueError("Dynamic fork_speed columns must match time_grid.")
        speed = np.stack([np.interp(uniform, axis, row) for row in speed])
        return speed, uniform
    return fork_speed, uniform


def map_timing_firing(
    observed_timing,
    fork_speed=1.4,
    fork_speeds=None,
    stall_rate=0.,
    tau=np.inf,
    resolution=1.0,
    resolution_x=None,
    neighbourhood_range=2000,
    fit_step=0.6,
    maxiter=10,
    sim_number=100,
    err_threshold=15.0,
    deviation_threshold=None,
    error=None,
    space_fit_step=None,
    space_maxiter=None,
    verbose=True,
    print_every=1,
    perQ=False,
    chr_number=1,
    save_errQ=False,
    stall_rate_per_kb=None,
    refine_edge_kb=1000,
    resolution_time=5.0,
    time_grid=None,
    refine_seed=None,
    stabilize=True,
):
    """Infer a spatial firing-rate profile from replication timing.

    ``fork_speed`` is the canonical argument name. ``fork_speeds`` is accepted
    as an older compatibility alias. For a non-periodic fitted sub-interval,
    ``refine_edge_kb`` sets the maximum reflected context used to stabilize a
    narrow endpoint update taper. Every bin remains refinable; pass zero when
    the supplied endpoints are true chromosome ends.

    With constant speed and linear geometry, ``stabilize`` safeguards the
    original local update. If it raises MSE, the remaining iterations use the
    exact gradient of the same forward model. Pass False to reproduce the
    historical, potentially worsening sequence of ratio updates.

    With periodic geometry, it also regularizes log-rate curvature around
    the complete ring before every forward prediction, including the seed.
    Observed timing is never smoothed. ``stabilize=False`` retains the legacy
    unconstrained periodic ratio updates.
    """
    resolution = _normalise_resolution_x(resolution, resolution_x=resolution_x)
    stall_rate = _resolve_stall_probability(
        stall_rate,
        stall_rate_per_kb,
        resolution,
    )
    if space_fit_step is not None:
        fit_step = space_fit_step
    if space_maxiter is not None:
        maxiter = space_maxiter
    error = _normalise_error_threshold(error)

    mse_list = []
    fit_history = []
    used_descent = False

    T = np.asarray(observed_timing, dtype=float)
    if T.ndim != 1:
        raise ValueError("observed_timing must be a one-dimensional array.")
    if T.size == 0:
        raise ValueError("observed_timing must contain at least one value.")
    if np.any(~np.isfinite(T)) or np.any(T <= 0):
        raise ValueError("observed_timing must contain finite positive values.")
    n = len(T)

    if fork_speeds is not None:
        fork_speed = fork_speeds
    fork_speed, _ = _uniform_speed_for_simulation(fork_speed, time_grid, resolution_time)

    fork_speed_arr = np.asarray(fork_speed, dtype=float)
    variable_fork_speed = fork_speed_arr.size > 1
    fork_speed_initial = (
        float(np.nanmean(fork_speed_arr))
        if variable_fork_speed
        else float(fork_speed_arr.reshape(-1)[0])
    )
    if not np.isfinite(fork_speed_initial) or fork_speed_initial <= 0:
        raise ValueError("fork_speed must contain finite positive values.")
    fork_speed_per_bin = fork_speed_initial / resolution

    # initial guess for firing rate
    x0 = (math.pi / (4.0 * fork_speed_per_bin)) * np.maximum(T, 1e-80) ** (-2)
    floor = 10.0 ** (-err_threshold)
    periodic_smoother = (
        PeriodicFiringSmoother(T, fork_speed_initial, resolution)
        if perQ and stabilize else None
    )
    if periodic_smoother is not None:
        x0 = periodic_smoother(x0, floor=floor)
    x = x0

    fit_mask, update_weights, edge_bins, taper_bins = _spatial_firing_edge_controls(
        n,
        resolution,
        observed_timing=T,
        fork_speed=fork_speed_initial,
        neighbourhood_range=neighbourhood_range,
        perQ=perQ,
        refine_edge_kb=refine_edge_kb,
    )
    historical_linear = not perQ and not variable_fork_speed and stall_rate == 0.0

    def update_firing_rate(observed, predicted, current):
        t_safe = np.maximum(observed, 1e-80)
        p_safe = np.maximum(predicted, 1e-80)
        ratio = p_safe / t_safe

        # Missing flanking origins make a finite sub-interval's endpoint
        # updates less certain.  Bound each single update and taper only the
        # narrowest endpoint bins instead of freezing a broad edge band at the
        # smooth analytical seed.  Every position therefore remains
        # refinable, while repeated boundary error cannot become a terminal
        # rate spike.
        if edge_bins:
            edge = ~fit_mask
            ratio[edge] = np.clip(ratio[edge], 0.5, 2.0)
        updated = current * ratio ** (fit_step * update_weights)
        updated = np.maximum(updated, floor)

        if deviation_threshold is not None:
            mse = np.abs(predicted - observed)
            close = mse <= deviation_threshold
            updated[close] = current[close]

        return periodic_smoother(updated, floor=floor) if periodic_smoother is not None else updated

    # initial prediction
    y = _map_spatial_firing_timing_with_context(
        x,
        fork_speed=fork_speed_initial,
        resolution=resolution,
        neighbourhood_range=neighbourhood_range,
        perQ=perQ,
        edge_bins=edge_bins,
        historical_linear=historical_linear,
    )
    mse = mean_squared_error(T[fit_mask], y[fit_mask])
    best_x = np.asarray(x, dtype=float).copy()
    best_mse = mse
    best_timing_fit = np.asarray(y, dtype=float).copy()

    if stall_rate > 0. or variable_fork_speed:
        maxiter0 = 0
    else:
        maxiter0 = maxiter

    print_every = max(1, int(print_every))
    start_time = time.time()

    for it in range(1, maxiter0 + 1):
        check_cancelled()
        candidate_x = update_firing_rate(T, y, x)
        candidate_y = _map_spatial_firing_timing_with_context(
            candidate_x,
            fork_speed=fork_speed_initial,
            resolution=resolution,
            neighbourhood_range=neighbourhood_range,
            perQ=perQ,
            edge_bins=edge_bins,
            historical_linear=historical_linear,
        )
        candidate_mse = mean_squared_error(T[fit_mask], candidate_y[fit_mask])
        if (
            stabilize and historical_linear and fit_step > 0
            and candidate_mse > best_mse + 1e-10 * max(1.0, best_mse)
        ):
            used_descent = True
            # The ratio rule is not a gradient: changing an origin changes
            # nearby timing as well. Preserve successful historical updates,
            # but spend the remaining iteration budget on a coupled descent
            # when that rule starts worsening the actual timing objective.
            best_x, best_mse, best_timing_fit, descent_history = (
                _refine_spatial_firing_descent(
                    best_x, T, best_timing_fit, best_mse,
                    fork_speed=fork_speed_initial, resolution=resolution,
                    neighbourhood_range=neighbourhood_range, fit_mask=fit_mask,
                    edge_bins=edge_bins, update_weights=update_weights,
                    fit_step=fit_step, floor=floor,
                    deviation_threshold=deviation_threshold,
                    first_iteration=it, maxiter=maxiter0, start_time=start_time,
                    verbose=verbose, print_every=print_every, error=error,
                )
            )
            fit_history.extend(descent_history)
            mse_list.extend(descent_history)
            break
        x, y, mse = candidate_x, candidate_y, candidate_mse
        if mse < best_mse:
            best_mse = mse
            best_x = np.asarray(x, dtype=float).copy()
            best_timing_fit = np.asarray(y, dtype=float).copy()

        fit_history.append(best_mse if stabilize and historical_linear else mse)

        done = it == maxiter0 or _error_reached(mse, error)
        if _fit_progress_enabled(verbose) and (it % print_every == 0 or done):
            reported_mse = fit_history[-1]
            etaferr(it=it, maxiter=maxiter0, start_time=start_time, err=reported_mse, done=done)
            mse_list.append(reported_mse)
        if done and it != maxiter0:
            break
            

    # Stalling models
    if stall_rate > 0. or variable_fork_speed:

        alpha = fit_step
        iterations = maxiter
    
        rate_min = 1e-9
        rate_max = None
        T_target = np.asarray(observed_timing, float)
        
        rates = np.clip(np.asarray(x, float), rate_min, None)
        #eff_f = 0.17
        #rates = (math.pi / (4.0 * 10 * fork_speed_per_bin)) * np.maximum(T, 1e-80) ** (-2)
        
        if rate_max is None:
            pos = rates[fit_mask & (rates > 0)]
            rate_max = (10.0 * np.median(pos)) if pos.size else 1.0
        if periodic_smoother is not None:
            rates = np.clip(periodic_smoother(rates, floor=rate_min), rate_min, rate_max)
    
        hist_mse = []
        T_pred = None
        best_rates = rates.copy()
        best_mse = np.inf
        start_time = time.time()
    
        for k in range(iterations):
            check_cancelled()
            simulation_rates, simulation_speed, prediction_slice = (
                _pad_spatial_firing_inputs(rates, fork_speed, edge_bins=edge_bins)
            )
            simres = rsim(
                firing_rate=simulation_rates,
                sim_number=sim_number,
                fork_speed=simulation_speed,
                stall_rate=stall_rate,
                tau=tau,
                resolution_space=resolution,
                resolution_time=resolution_time,
                seed=None if refine_seed is None else int(refine_seed) + k,
                verbose=False,
                perQ=perQ,
                retain_rep_times_per_sim=False,
                retain_inter_origin_distances=False,
            )
            T_pred = np.asarray(simres['replication_timing'], dtype=float)[prediction_slice]
    
            valid = (
                np.isfinite(T_pred)
                & np.isfinite(T_target)
                & (T_pred > 0)
                & (T_target > 0)
            )
            if not np.any(valid):
                break
            score_valid = valid & fit_mask
            if not np.any(score_valid):
                score_valid = valid
            residual = T_pred[score_valid] - T_target[score_valid]
            mse = float(np.mean(residual**2))
            hist_mse.append(mse)
            fit_history.append(mse)
            if mse < best_mse:
                best_mse = mse
                best_rates = rates.copy()
                best_timing_fit = np.asarray(T_pred, dtype=float).copy()

            eps = 1e-12
            ratio = np.ones_like(rates)
            ratio[valid] = (T_pred[valid] + eps) / (T_target[valid] + eps)
            if edge_bins:
                edge = ~fit_mask
                ratio[edge & valid] = np.clip(ratio[edge & valid], 0.5, 2.0)
            rates[valid] *= np.power(
                ratio[valid],
                alpha * update_weights[valid],
            )
            np.clip(rates, rate_min, rate_max, out=rates)
            if periodic_smoother is not None:
                rates = np.clip(periodic_smoother(rates, floor=rate_min), rate_min, rate_max)
    
            done = (k + 1) == iterations or _error_reached(mse, error)
            if _fit_progress_enabled(verbose) and ((k + 1) % print_every == 0 or done):
                etaferr(it=k+1, maxiter=iterations, start_time=start_time, err=mse, done=done)
                mse_list.append(mse)
            if done and (k + 1) != iterations:
                break

        x = best_rates
    else:
        x = best_x

    #np.savetxt(f"data/fit_error/mse_chr{chr_number}.txt", mse_list, fmt="%.6f")

    if save_errQ:
        savecsv(arr=mse_list, chr_number=chr_number, stitle="fit_error", outpath="fit_error")
        
    return _RmapInfoArray(
        x,
        {
            "replication_timing_fit": best_timing_fit,
            "fit_mse": None if not np.isfinite(best_mse) else float(best_mse),
            "fit_mse_description": (
                "Mean squared timing error over the refinement-supported region."
            ),
            "fit_edge_bins": edge_bins,
            "fit_taper_bins": taper_bins,
            "fit_history": np.asarray(fit_history, dtype=float),
            "fit_used_descent": used_descent,
            "fit_periodic_smoothing_kb": (
                periodic_smoother.length_kb if periodic_smoother is not None else 0.
            ),
        },
    )


def _refine_spatial_firing_descent(
    initial_rates, observed, initial_prediction, initial_mse, *, fork_speed,
    resolution, neighbourhood_range, fit_mask, edge_bins, update_weights,
    fit_step, floor, deviation_threshold, first_iteration, maxiter, start_time,
    verbose, print_every, error,
):
    """Safeguard a failing local update using the original map's adjoint."""
    origin = np.log(np.maximum(initial_rates, floor))
    scale = float(fit_step) * np.asarray(update_weights, dtype=float)
    if deviation_threshold is not None:
        scale = scale.copy()
        scale[np.abs(initial_prediction - observed) <= deviation_threshold] = 0.0
    active = np.flatnonzero(scale > 0)
    best_rates = np.asarray(initial_rates).copy()
    best_prediction = np.asarray(initial_prediction).copy()
    best_mse = float(initial_mse)
    history = []
    if not active.size:
        return best_rates, best_mse, best_prediction, history

    def objective(coordinates):
        nonlocal best_rates, best_prediction, best_mse
        log_rates = origin.copy()
        log_rates[active] += scale[active] * coordinates
        rates = np.exp(log_rates)
        mse, gradient, prediction = timing_mse_gradient(
            rates, observed, fork_speed=fork_speed, resolution=resolution,
            neighbourhood_range=neighbourhood_range, fit_mask=fit_mask,
            edge_bins=edge_bins,
        )
        if mse < best_mse:
            best_rates, best_prediction, best_mse = rates.copy(), prediction.copy(), mse
        return mse, gradient[active] * rates[active] * scale[active]

    def accepted(_coordinates):
        history.append(best_mse)
        iteration = first_iteration + len(history) - 1
        done = iteration == maxiter or _error_reached(best_mse, error)
        if _fit_progress_enabled(verbose) and (iteration % print_every == 0 or done):
            etaferr(it=iteration, maxiter=maxiter, start_time=start_time, err=best_mse, done=done)
        if done and iteration != maxiter:
            raise _FitConverged

    lower = (math.log(floor) - origin[active]) / scale[active]
    # This numerical ceiling is far above instantaneous initiation at the
    # selected grid step; it prevents overflow without constraining useful fits.
    upper = (math.log(1e6 * fork_speed / resolution) - origin[active]) / scale[active]
    try:
        minimize(
            objective, np.zeros(active.size), jac=True, method="L-BFGS-B",
            bounds=list(zip(lower, upper)), callback=accepted,
            options={"maxiter": maxiter - first_iteration + 1, "maxls": 8,
                     "ftol": 0.0, "gtol": 0.0},
        )
    except _FitConverged:
        pass
    if not history or history[-1] != best_mse:
        # A better trial may be retained even if the line search stopped before
        # accepting it. Keep history, reported MSE and the returned rates aligned.
        history.append(best_mse)
    if _fit_progress_enabled(verbose) and first_iteration + len(history) - 1 < maxiter:
        etaferr(it=first_iteration + len(history) - 1, maxiter=maxiter,
                start_time=start_time, err=best_mse, done=True)
    return best_rates, best_mse, best_prediction, history


def _time_axis_from_timing(relative_timing, resolution_time=5.0, time_grid=None):
    if time_grid is not None:
        time_axis = np.asarray(time_grid, dtype=float)
        if time_axis.ndim != 1 or time_axis.size == 0:
            raise ValueError("time_grid must be a one-dimensional array.")
        if np.any(~np.isfinite(time_axis)):
            raise ValueError("time_grid must contain finite values.")
        if time_axis.size > 1 and np.any(np.diff(time_axis) <= 0):
            raise ValueError("time_grid must be strictly increasing.")
        return time_axis

    resolution_time = float(resolution_time)
    if not np.isfinite(resolution_time) or resolution_time <= 0:
        raise ValueError("resolution_t must be a finite positive value.")

    finite = relative_timing[np.isfinite(relative_timing)]
    max_time = float(np.nanmax(finite)) if finite.size else 0.0
    n_time = int(np.floor(max_time / resolution_time)) + 1
    return np.arange(max(1, n_time), dtype=float) * resolution_time


def _survival_time_axis_from_timing(
    relative_timing,
    resolution_time=5.0,
    time_grid=None,
    min_max_time=1200.0,
    tail_multiplier=1.5,
):
    if time_grid is not None:
        return _time_axis_from_timing(
            relative_timing,
            resolution_time=resolution_time,
            time_grid=time_grid,
        )

    resolution_time = float(resolution_time)
    if not np.isfinite(resolution_time) or resolution_time <= 0:
        raise ValueError("resolution_t must be a finite positive value.")

    finite = np.asarray(relative_timing, dtype=float)
    finite = finite[np.isfinite(finite)]
    max_time = float(np.nanmax(finite)) if finite.size else 0.0
    horizon = max(
        float(min_max_time),
        max_time * float(tail_multiplier),
        max_time + 10.0 * resolution_time,
    )
    n_time = int(np.floor(horizon / resolution_time)) + 1
    return np.arange(max(1, n_time), dtype=float) * resolution_time


def _temporal_profile_from_spatial_fit(
    observed_timing,
    firing_rate_space,
    resolution_time=5.0,
    time_grid=None,
    T0=None,
    temporal_smoothw=None,
    floor=0.0,
):
    T = np.asarray(observed_timing, dtype=float)
    rates = np.asarray(firing_rate_space, dtype=float)
    if T.ndim != 1 or rates.ndim != 1 or T.size != rates.size:
        raise ValueError("observed_timing and firing_rate_space must be matching one-dimensional arrays.")

    finite = np.isfinite(T) & np.isfinite(rates)
    if not np.any(finite):
        raise ValueError("observed_timing and firing_rate_space must contain finite overlapping values.")

    if T0 is None:
        T0 = float(np.nanmin(T[finite]))
    T0 = float(T0)
    if not np.isfinite(T0):
        raise ValueError("T0 must be finite when supplied.")

    relative_timing = np.maximum(T - T0, 0.0)
    time_axis = _time_axis_from_timing(
        relative_timing,
        resolution_time=resolution_time,
        time_grid=time_grid,
    )
    if time_axis.size == 1:
        indices = np.zeros(T.size, dtype=int)
    else:
        indices = np.searchsorted(time_axis, relative_timing, side="left")
        prev = np.clip(indices - 1, 0, time_axis.size - 1)
        indices = np.clip(indices, 0, time_axis.size - 1)
        use_prev = np.abs(relative_timing - time_axis[prev]) <= np.abs(relative_timing - time_axis[indices])
        indices[use_prev] = prev[use_prev]

    sums = np.zeros(time_axis.size, dtype=float)
    counts = np.zeros(time_axis.size, dtype=int)
    for idx, rate, ok in zip(indices, rates, finite):
        if ok:
            sums[idx] += rate
            counts[idx] += 1

    temporal = np.full(time_axis.size, np.nan, dtype=float)
    occupied = counts > 0
    temporal[occupied] = sums[occupied] / counts[occupied]

    if np.any(occupied):
        x = np.arange(time_axis.size)
        temporal = np.interp(x, x[occupied], temporal[occupied])
    else:
        temporal.fill(float(np.nanmedian(rates[finite])))

    if temporal_smoothw is not None:
        temporal = smoothf(temporal, window=temporal_smoothw)

    floor = float(floor)
    if not np.isfinite(floor) or floor < 0:
        raise ValueError("floor must be finite and >= 0.")
    temporal = np.maximum(temporal, floor)
    return temporal, time_axis


_SPATIAL_FIRING_FIT_OPTIONS = {
    "fork_speeds",
    "stall_rate",
    "stall_rate_per_kb",
    "tau",
    "neighbourhood_range",
    "fit_step",
    "maxiter",
    "error",
    "sim_number",
    "err_threshold",
    "deviation_threshold",
    "space_fit_step",
    "space_maxiter",
    "verbose",
    "print_every",
    "perQ",
    "refine_edge_kb",
    "chr_number",
    "save_errQ",
}


def _spatial_firing_fit_kwargs(kwargs):
    return {key: value for key, value in kwargs.items() if key in _SPATIAL_FIRING_FIT_OPTIONS}


def _timing_bin_indices(timing, n_time, resolution_time=5.0, T0=None):
    timing = np.asarray(timing, dtype=float)
    if T0 is None:
        finite = timing[np.isfinite(timing)]
        T0 = float(np.nanmin(finite)) if finite.size else 0.0
    T0 = float(T0)
    if not np.isfinite(T0):
        raise ValueError("T0 must be finite when supplied.")
    relative = np.maximum(timing - T0, 0.0)
    bins = np.floor(relative / float(resolution_time) + 0.5).astype(int)
    return np.clip(bins, 0, max(0, int(n_time) - 1))


def _refinement_crop_mask(n, resolution, perQ=False, refine_edge_kb=1000):
    mask = np.ones(int(n), dtype=bool)
    if perQ or refine_edge_kb is None:
        return mask
    edge_bins = int(float(refine_edge_kb) / float(resolution))
    if edge_bins > 0 and 2 * edge_bins < n:
        mask[:edge_bins] = False
        mask[-edge_bins:] = False
    return mask


def _spatial_firing_edge_controls(
    n,
    resolution,
    *,
    observed_timing,
    fork_speed,
    neighbourhood_range,
    perQ=False,
    refine_edge_kb=1000,
):
    """Build scoring and update controls for uncertain linear endpoints.

    ``refine_edge_kb`` is an upper bound on the amount of endpoint context to
    protect, rather than a band whose rates are held fixed.  The effective
    context is also bounded by the forward-map neighbourhood, an estimate of
    two characteristic fork-travel distances, and ten percent of the supplied
    profile.  This keeps the safeguard local on short mapping intervals.

    All bins retain a positive update weight.  Only the innermost 2.5 percent
    at either endpoint receives a raised-cosine taper, and the rest of the
    context band is updated normally.  The context band is omitted from the
    reported fit score because its flanking source data remain inferred.
    """
    n = int(n)
    if n <= 0:
        raise ValueError("n must be a positive integer.")
    resolution = float(resolution)
    if not np.isfinite(resolution) or resolution <= 0:
        raise ValueError("resolution must be a finite positive value.")

    mask = np.ones(n, dtype=bool)
    weights = np.ones(n, dtype=float)
    if perQ or refine_edge_kb is None:
        return mask, weights, 0, 0
    edge_kb = float(refine_edge_kb)
    if not np.isfinite(edge_kb) or edge_kb < 0:
        raise ValueError("refine_edge_kb must be None or a finite value >= 0.")
    requested_bins = int(edge_kb / resolution)
    if requested_bins <= 0 or n < 4:
        return mask, weights, 0, 0

    timing = np.asarray(observed_timing, dtype=float)
    positive_timing = timing[np.isfinite(timing) & (timing > 0)]
    characteristic_time = (
        float(np.nanpercentile(positive_timing, 90))
        if positive_timing.size
        else resolution / float(fork_speed)
    )
    travel_bins = max(
        1,
        int(np.ceil(2.0 * float(fork_speed) * characteristic_time / resolution)),
    )
    neighbourhood_bins = max(1, int(float(neighbourhood_range) / resolution))
    short_profile_cap = max(1, int(np.ceil(0.1 * n)))
    edge_bins = min(
        requested_bins,
        travel_bins,
        neighbourhood_bins,
        short_profile_cap,
    )
    mask[:edge_bins] = False
    mask[-edge_bins:] = False

    taper_bins = min(edge_bins, max(1, int(np.ceil(0.025 * n))))
    if taper_bins == 1:
        taper = np.asarray([0.25], dtype=float)
    else:
        phase = np.linspace(0.0, 1.0, taper_bins)
        taper = 0.25 + 0.75 * np.sin(0.5 * math.pi * phase) ** 2
    weights[:taper_bins] = taper
    weights[-taper_bins:] = taper[::-1]
    return mask, weights, edge_bins, taper_bins


def _pad_spatial_firing_inputs(firing_rate, fork_speed, *, edge_bins):
    """Reflect spatial axes while preserving every public ``rsim`` field form."""
    rates = np.asarray(firing_rate, dtype=float).reshape(-1)
    edge_bins = int(edge_bins)
    if edge_bins <= 0:
        return rates, fork_speed, slice(None)

    pad_mode = "reflect" if rates.size > 1 else "edge"
    padded_rates = np.pad(rates, (edge_bins, edge_bins), mode=pad_mode)

    speed = np.asarray(fork_speed, dtype=float)
    if speed.size == 1:
        padded_speed = float(speed.reshape(-1)[0])
    elif speed.ndim == 1:
        if speed.size != rates.size:
            raise ValueError("fork_speed must match the length of firing_rate.")
        speed_pad_mode = "reflect" if speed.size > 1 else "edge"
        padded_speed = np.pad(
            speed,
            (edge_bins, edge_bins),
            mode=speed_pad_mode,
        )
    elif speed.ndim == 2 and speed.shape[0] == 1 and speed.shape[1] > 1:
        # Public (1, T) fields are time-only; the firing-rate profile supplies
        # the padded spatial length to rsim, so no speed padding is needed.
        padded_speed = speed
    elif speed.ndim == 2 and speed.shape[0] == rates.size:
        # Public spatial columns (n, 1) and space-time fields (n, T) both keep
        # space on axis zero.  Reflect only that axis and leave time untouched.
        speed_pad_mode = "reflect" if speed.shape[0] > 1 else "edge"
        padded_speed = np.pad(
            speed,
            ((edge_bins, edge_bins), (0, 0)),
            mode=speed_pad_mode,
        )
    else:
        raise ValueError(
            "fork_speed must be scalar, (n,), (1,T), (n,1), or (n,T), "
            "with spatial length matching firing_rate."
        )
    return padded_rates, padded_speed, slice(edge_bins, edge_bins + rates.size)


def _map_spatial_firing_timing_with_context(
    firing_rate,
    *,
    fork_speed,
    resolution,
    neighbourhood_range,
    perQ,
    edge_bins,
    historical_linear=False,
):
    """Forward-map a fit candidate with reflected context when appropriate."""
    if historical_linear:
        rates, speed, output_slice = _pad_spatial_firing_inputs(
            firing_rate, fork_speed, edge_bins=edge_bins,
        )
        return _map_firing_timing_legacy_linear(
            rates, fork_speed=speed, resolution=resolution,
            neighbourhood_range=neighbourhood_range,
        )[output_slice]
    if perQ or not edge_bins:
        return map_firing_timing(
            firing_rate,
            fork_speed=fork_speed,
            resolution=resolution,
            neighbourhood_range=neighbourhood_range,
            perQ=perQ,
        )

    padded_rates, padded_speed, output_slice = _pad_spatial_firing_inputs(
        firing_rate,
        fork_speed,
        edge_bins=edge_bins,
    )
    prediction = map_firing_timing(
        padded_rates,
        fork_speed=padded_speed,
        resolution=resolution,
        neighbourhood_range=neighbourhood_range,
        perQ=False,
    )
    return np.asarray(prediction, dtype=float)[output_slice]


def _mean_ratio_by_bin(ratio, bins, n_time, mask):
    updates = np.ones(int(n_time), dtype=float)
    for idx in range(int(n_time)):
        here = mask & (bins == idx)
        if np.any(here):
            updates[idx] = float(np.nanmean(ratio[here]))
    return updates


def _canonical_spacetime_update(value):
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "separable": "factorized",
        "factorised": "factorized",
        "factorized": "factorized",
        "row_column": "factorized",
        "mixed": "mixed",
        "cellwise": "mixed",
        "cell_wise": "mixed",
        "local": "mixed",
    }
    if text not in aliases:
        raise ValueError("spacetime_update must be 'factorized' or 'mixed'.")
    return aliases[text]


def _canonical_spacetime_order(value):
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "crossed": "crossed",
        "cross": "crossed",
        "alternating": "crossed",
        "alternate": "crossed",
        "mixed": "mixed",
        "joint": "mixed",
        "simultaneous": "mixed",
        "coupled": "mixed",
        "space_then_time": "space_then_time",
        "space_time": "space_then_time",
        "space_first": "space_then_time",
        "x_then_t": "space_then_time",
        "x_t": "space_then_time",
        "time_then_space": "time_then_space",
        "timing_then_space": "time_then_space",
        "timing_first": "time_then_space",
        "timing_space": "time_then_space",
        "time_space": "time_then_space",
        "time_first": "time_then_space",
        "t_then_x": "time_then_space",
        "t_x": "time_then_space",
    }
    if text not in aliases:
        raise ValueError(
            "spacetime_order must be 'crossed', 'mixed', 'space_then_time', or 'time_then_space'."
        )
    return aliases[text]


def _canonical_target_shape(value):
    if value is None:
        return "free"
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "default": "free",
        "free": "free",
        "unconstrained": "free",
        "nonparametric": "free",
        "non_parametric": "free",
        "current": "free",
        "weibull": "weibull",
        "weibull_hazard": "weibull",
    }
    if text not in aliases:
        raise ValueError("target_shape must be 'free' or 'weibull'.")
    return aliases[text]


def _canonical_speed_target_shape(value):
    if value is None:
        return "best"
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "default": "best",
        "best": "best",
        "hybrid": "best",
        "recommended": "best",
    }
    if text not in aliases:
        raise ValueError("fork-speed target_shape must be 'best'.")
    return aliases[text]


def _canonical_weibull_target_form(value):
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "t": "time",
        "time": "time",
        "temporal": "time",
        "row_vector": "time",
        "transposed_array": "time",
        "xt": "spacetime",
        "x_t": "spacetime",
        "space_time": "spacetime",
        "spacetime": "spacetime",
        "matrix": "spacetime",
    }
    return aliases.get(text, text)


def _resolve_spacetime_phase_controls(
    fit_step=None,
    maxiter=None,
    sim_number=None,
    space_fit_step=None,
    space_maxiter=None,
    temporal_fit_step=None,
    temporal_maxiter=None,
    temporal_sim_number=None,
    default_space_fit_step=0.6,
    default_space_maxiter=10,
    default_temporal_fit_step=0.35,
    default_temporal_maxiter=10,
):
    if fit_step is not None:
        if space_fit_step is None:
            space_fit_step = fit_step
        if temporal_fit_step is None:
            temporal_fit_step = fit_step
    if maxiter is not None:
        space_maxiter = maxiter
        temporal_maxiter = maxiter
    if space_fit_step is None:
        space_fit_step = default_space_fit_step
    if temporal_fit_step is None:
        temporal_fit_step = default_temporal_fit_step
    if space_maxiter is None:
        space_maxiter = default_space_maxiter
    if temporal_maxiter is None:
        temporal_maxiter = default_temporal_maxiter
    if temporal_sim_number is None:
        temporal_sim_number = 100 if sim_number is None else sim_number
    return (
        float(space_fit_step),
        int(space_maxiter),
        float(temporal_fit_step),
        int(temporal_maxiter),
        int(temporal_sim_number),
    )


class _RmapInfoArray(np.ndarray):
    def __new__(cls, array, info=None):
        obj = np.asarray(array, dtype=float).view(cls)
        obj._rmap_info = dict(info or {})
        return obj

    def __array_finalize__(self, obj):
        self._rmap_info = getattr(obj, "_rmap_info", {})


class _RmapInfoFloat(float):
    def __new__(cls, value, info=None):
        obj = super().__new__(cls, value)
        obj._rmap_info = dict(info or {})
        return obj


def _trapz_weights(time_axis):
    t = np.asarray(time_axis, dtype=float)
    if t.ndim != 1 or t.size == 0:
        raise ValueError("time_grid must be a non-empty one-dimensional array.")
    if t.size == 1:
        return np.ones(1, dtype=float)
    if np.any(~np.isfinite(t)) or np.any(np.diff(t) <= 0):
        raise ValueError("time_grid must contain finite, strictly increasing values.")
    weights = np.empty(t.size, dtype=float)
    weights[0] = 0.5 * (t[1] - t[0])
    weights[-1] = 0.5 * (t[-1] - t[-2])
    if t.size > 2:
        weights[1:-1] = 0.5 * (t[2:] - t[:-2])
    return weights


def _symmetric_convolution(values, kernel):
    kernel = np.asarray(kernel, dtype=float)
    full_kernel = np.concatenate((kernel[:0:-1], kernel))
    out = fftconvolve(np.asarray(values, dtype=float), full_kernel, mode="same")
    if out.size != values.size:
        start = (out.size - values.size) // 2
        out = out[start:start + values.size]
    return out


def _weibull_time_axis_for_timing(
    target_timing,
    resolution_time=5.0,
    time_grid=None,
    tmax_factor=3.0,
):
    if time_grid is not None:
        return _time_axis_from_timing(
            np.asarray(target_timing, dtype=float),
            resolution_time=resolution_time,
            time_grid=time_grid,
        )
    resolution_time = float(resolution_time)
    if not np.isfinite(resolution_time) or resolution_time <= 0:
        raise ValueError("resolution_t must be finite and positive.")
    finite = np.asarray(target_timing, dtype=float)
    finite = finite[np.isfinite(finite) & (finite > 0)]
    max_timing = float(np.nanmax(finite)) if finite.size else resolution_time
    max_time = max(max_timing * float(tmax_factor), max_timing + 5.0 * resolution_time)
    return np.arange(0.0, max_time + resolution_time, resolution_time)


def _weibull_delay_profile(delay, n):
    arr = np.asarray(delay, dtype=float)
    if arr.size == 1:
        value = float(arr.reshape(-1)[0])
        if not np.isfinite(value) or value < 0:
            raise ValueError("weibull_delay must be finite and >= 0.")
        return np.full(int(n), value, dtype=float)
    arr = arr.reshape(-1)
    if arr.size != int(n):
        raise ValueError("weibull_delay must be scalar or match observed_timing.")
    if np.any(~np.isfinite(arr)) or np.any(arr < 0):
        raise ValueError("weibull_delay must contain finite values >= 0.")
    return arr.astype(float, copy=False)


def _weibull_alpha_initial(target_timing, fork_speed, shape=2.0, floor=1e-12):
    T = np.asarray(target_timing, dtype=float)
    speed_values = np.asarray(fork_speed, dtype=float).reshape(-1)
    speed = float(np.nanmedian(speed_values[np.isfinite(speed_values) & (speed_values > 0)]))
    if not np.isfinite(speed) or speed <= 0:
        raise ValueError("fork_speed must contain finite positive values.")
    k = float(shape)
    T_safe = np.maximum(T, 1e-9)
    coeff = (k + 1.0) / (2.0 * speed)
    alpha = coeff * np.power(gamma(1.0 + 1.0 / (k + 1.0)) / T_safe, k + 1.0)
    finite_positive = alpha[np.isfinite(alpha) & (alpha > 0)]
    fill = float(np.nanmedian(finite_positive)) if finite_positive.size else 1e-9
    alpha = np.where(np.isfinite(alpha) & (alpha > 0), alpha, fill)
    return np.maximum(alpha, float(floor))


def _weibull_travel_backend(fork_speed, n, resolution, delay):
    speed = np.asarray(fork_speed, dtype=float)
    if speed.size == 1:
        v = float(speed.reshape(-1)[0])
        if not np.isfinite(v) or v <= 0:
            raise ValueError("fork_speed must be finite and positive.")
        delay_is_uniform = np.allclose(delay, delay[0])
        if delay_is_uniform:
            distances = np.arange(int(n), dtype=float) * float(resolution) / v
            return "constant_fft", v, distances
        return "characteristic_cones", v, None

    # Preserve the row-array convention for v(t). Squeezing a temporal row
    # into a vector would silently treat it as v(x) when X and T coincide.
    if speed.ndim == 2 and speed.shape[1] == 1:
        speed = speed[:, 0]
    if speed.ndim == 1:
        if speed.size != int(n):
            raise ValueError("fork_speed must be scalar or match observed_timing.")
        if np.any(~np.isfinite(speed)) or np.any(speed <= 0):
            raise ValueError("fork_speed must contain finite positive values.")
        return "characteristic_cones", speed.copy(), None
    if speed.ndim == 2 and speed.shape[0] in {1, int(n)}:
        if np.any(~np.isfinite(speed)) or np.any(speed <= 0):
            raise ValueError("fork_speed must contain finite positive values.")
        return "temporal_fft" if speed.shape[0] == 1 else "characteristic_cones", speed, None
    raise ValueError("fork_speed must be scalar, spatial, temporal, or space-time dependent.")


def _weibull_survival_fields(
    alpha,
    fork_speed,
    shape=2.0,
    resolution=1.0,
    resolution_time=5.0,
    time_grid=None,
    delay=0.0,
    rate_floor=1e-12,
    perQ=False,
):
    alpha = np.asarray(alpha, dtype=float).reshape(-1)
    if alpha.size == 0:
        raise ValueError("alpha must be a non-empty spatial profile.")
    if np.any(~np.isfinite(alpha)) or np.any(alpha <= 0):
        raise ValueError("alpha must contain finite positive values.")
    k = float(shape)
    if not np.isfinite(k) or k <= 0:
        raise ValueError("weibull_shape must be finite and positive.")

    t = np.asarray(time_grid, dtype=float)
    if t.ndim != 1 or t.size == 0:
        raise ValueError("time_grid must be a non-empty one-dimensional array.")
    rate_floor = float(rate_floor)
    if not np.isfinite(rate_floor) or rate_floor < 0:
        raise ValueError("rate_floor must be finite and >= 0.")
    weights = _trapz_weights(t)
    delay = _weibull_delay_profile(delay, alpha.size)
    speed_used = np.asarray(fork_speed, dtype=float)
    if speed_used.size == 1 and np.all(delay == delay[0]) and not perQ:
        mode, speed_used, travel = _weibull_travel_backend(fork_speed, alpha.size, resolution, delay)
        A = np.zeros((t.size, alpha.size), dtype=float)
        offset = float(delay[0])
        for idx, time_value in enumerate(t):
            lag = np.maximum(time_value - offset - travel, 0.0)
            kernel = np.power(lag, k)
            A[idx] = float(resolution) * _symmetric_convolution(alpha, kernel)
    else:
        operator = _make_weibull_cone(fork_speed, t, delay, resolution, resolution_time, perQ)
        A, _ = operator.volumes(alpha, k)
        mode = operator.backend

    A = np.maximum(A, 0.0)
    survival = np.exp(-np.clip(A, 0.0, 700.0))
    predicted_timing = weights @ survival
    firing_rate = _weibull_hazard_samples(alpha, k, t, delay, rate_floor)
    scale = np.power(alpha, -1.0 / k)
    return {
        "alpha": alpha,
        "lambda": scale,
        "firing_rate": firing_rate.T,
        "predicted_timing": predicted_timing,
        "survival": survival.T,
        "extended_volume": A.T,
        "fork_speed": speed_used,
        "weibull_shape": k,
        "weibull_delay": delay,
        "time_grid": t,
        "forward_backend": mode,
    }


def _make_weibull_cone(fork_speed, time_grid, delay, resolution, resolution_time, perQ):
    from .lightcone import WeibullCone
    from .mapping import _expand_space_time_field

    speed = _expand_space_time_field(
        fork_speed, len(time_grid), len(delay), time_grid, resolution_time,
        "fork_speed", positive=True,
    )
    return WeibullCone(speed, time_grid, delay, resolution, periodic=perQ)


def _weibull_hazard_samples(alpha, shape, time_grid, delay, floor=0.0):
    """Finite hazard samples, with an interval average at a singular onset."""
    t = np.asarray(time_grid, dtype=float)
    alpha = np.asarray(alpha, dtype=float).reshape(-1)
    delays = np.broadcast_to(np.asarray(delay, dtype=float).reshape(-1), alpha.shape)
    age = t[:, None] - delays[None, :]
    amplitudes = np.broadcast_to(alpha, age.shape)
    hazard = np.zeros_like(age)
    positive = age > 0
    hazard[positive] = shape * amplitudes[positive] * age[positive] ** (shape - 1.0)
    activation = age == 0
    if shape == 1:
        hazard[activation] = amplitudes[activation]
    elif shape < 1:
        dt = np.min(np.diff(t)) if t.size > 1 else 1.0
        hazard[activation] = amplitudes[activation] * dt ** (shape - 1.0)
    return np.maximum(hazard, float(floor))


def _weibull_cone_loss_grad(
    params, operator, target_timing, fit_mask, shape, time_grid, offset,
    variable_alpha, z0, alpha_smooth, alpha_prior, floor, fit_shape,
):
    params = np.asarray(params, dtype=float)
    z = params[:-1] if fit_shape else params
    k = float(np.exp(params[-1])) if fit_shape else float(shape)
    alpha = np.exp(z) if variable_alpha else np.full(target_timing.size, np.exp(z[0]))
    volume, shape_volume = operator.volumes(alpha, k, shape_gradient=fit_shape)
    survival = np.exp(-np.clip(volume, 0, 700))
    weights = _trapz_weights(time_grid)
    predicted = weights @ survival
    residual = np.where(fit_mask, predicted - target_timing, 0.0)
    data_loss = 0.5 * float(residual @ residual)
    adjoint_weight = -weights[:, None] * survival * residual[None, :]
    grad_alpha = operator.alpha_adjoint(adjoint_weight, k)
    grad = grad_alpha * alpha if variable_alpha else np.array([grad_alpha @ alpha])
    regularisation = 0.0
    if variable_alpha and alpha_smooth > 0 and z.size > 1:
        difference = np.diff(z)
        regularisation += 0.5 * alpha_smooth * float(difference @ difference)
        grad[:-1] -= alpha_smooth * difference
        grad[1:] += alpha_smooth * difference
        if operator.periodic:
            seam = z[0] - z[-1]
            regularisation += 0.5 * alpha_smooth * seam**2
            grad[0] += alpha_smooth * seam
            grad[-1] -= alpha_smooth * seam
    if alpha_prior > 0:
        difference = z - z0[:z.size]
        regularisation += 0.5 * alpha_prior * float(difference @ difference)
        grad += alpha_prior * difference
    if fit_shape:
        grad = np.append(grad, k * np.sum(adjoint_weight * shape_volume))
    hazard = _weibull_hazard_samples(alpha, k, time_grid, operator.delay, floor)
    return data_loss + regularisation, grad, {
        "alpha": alpha, "lambda": alpha ** (-1.0 / k),
        "firing_rate": np.maximum(hazard, floor).T,
        "predicted_timing": predicted + offset,
        "residual": predicted - target_timing,
        "survival": survival.T, "extended_volume": volume.T,
        "fork_speed": operator.speed.T, "weibull_shape": k,
        "weibull_delay": operator.delay, "time_grid": np.asarray(time_grid),
        "objective": data_loss + regularisation, "data_loss": data_loss,
        "regularisation": regularisation, "target_shape": "weibull",
        "forward_backend": operator.backend,
    }


def _weibull_constant_speed_loss_grad(
    z,
    target_timing,
    fit_mask,
    fork_speed,
    shape,
    resolution,
    time_grid,
    delay,
    offset=0.0,
    variable_alpha=True,
    z0=None,
    alpha_smooth=0.0,
    alpha_prior=0.0,
    floor=1e-12,
    fit_shape=False,
):
    params = np.asarray(z, dtype=float)
    if fit_shape:
        z = params[:-1]
        shape = float(np.exp(params[-1]))
    else:
        z = params
        shape = float(shape)
    n = target_timing.size
    if variable_alpha:
        alpha = np.exp(z)
    else:
        alpha = np.full(n, float(np.exp(z[0])), dtype=float)

    speed = float(np.asarray(fork_speed, dtype=float).reshape(-1)[0])
    delay_value = float(np.asarray(delay, dtype=float).reshape(-1)[0])
    t = np.asarray(time_grid, dtype=float)
    weights = _trapz_weights(t)
    distances = np.arange(n, dtype=float) * float(resolution) / speed

    A = np.zeros((t.size, n), dtype=float)
    survival = np.zeros_like(A)
    predicted_relative = np.zeros(n, dtype=float)
    predicted_shape_grad = np.zeros(n, dtype=float) if fit_shape else None

    for idx, time_value in enumerate(t):
        lag = np.maximum(time_value - delay_value - distances, 0.0)
        kernel = np.power(lag, shape)
        A[idx] = float(resolution) * _symmetric_convolution(alpha, kernel)
        survival[idx] = np.exp(-np.clip(A[idx], 0.0, 700.0))
        predicted_relative += weights[idx] * survival[idx]
        if fit_shape:
            kernel_log = np.zeros_like(kernel)
            positive = lag > 0
            kernel_log[positive] = kernel[positive] * np.log(lag[positive])
            A_shape_grad = float(resolution) * _symmetric_convolution(alpha, kernel_log)
            predicted_shape_grad += -weights[idx] * survival[idx] * A_shape_grad

    residual = predicted_relative - target_timing
    residual_weighted = np.zeros_like(residual)
    residual_weighted[fit_mask] = residual[fit_mask]
    data_loss = 0.5 * float(np.dot(residual_weighted, residual_weighted))

    grad_alpha = np.zeros(n, dtype=float)
    for idx, time_value in enumerate(t):
        lag = np.maximum(time_value - delay_value - distances, 0.0)
        kernel = np.power(lag, shape)
        q = -weights[idx] * residual_weighted * survival[idx]
        grad_alpha += float(resolution) * _symmetric_convolution(q, kernel)

    if variable_alpha:
        grad = grad_alpha * alpha
    else:
        grad = np.array([float(np.dot(grad_alpha, alpha))], dtype=float)

    regularisation = 0.0
    if variable_alpha and alpha_smooth > 0 and z.size > 1:
        dz = np.diff(z)
        regularisation += 0.5 * alpha_smooth * float(np.dot(dz, dz))
        smooth_grad = np.zeros_like(z)
        smooth_grad[:-1] -= dz
        smooth_grad[1:] += dz
        grad += alpha_smooth * smooth_grad
    if alpha_prior > 0 and z0 is not None:
        dz0 = z - z0[:z.size]
        regularisation += 0.5 * alpha_prior * float(np.dot(dz0, dz0))
        grad += alpha_prior * dz0
    if fit_shape:
        grad_shape = float(np.dot(residual_weighted, predicted_shape_grad))
        grad = np.concatenate((grad, [grad_shape * shape]))

    firing_rate = _weibull_hazard_samples(alpha, shape, t, [delay_value], floor)
    fields = {
        "alpha": alpha,
        "lambda": np.power(alpha, -1.0 / shape),
        "firing_rate": firing_rate.T,
        "predicted_timing": predicted_relative + offset,
        "residual": predicted_relative + offset - (target_timing + offset),
        "survival": survival.T,
        "extended_volume": A.T,
        "fork_speed": speed,
        "weibull_shape": shape,
        "weibull_delay": np.full(n, delay_value, dtype=float),
        "time_grid": t,
        "objective": data_loss + regularisation,
        "data_loss": data_loss,
        "regularisation": regularisation,
        "target_shape": "weibull",
    }
    return data_loss + regularisation, grad, fields


def _fit_weibull_survival_firing(
    observed_timing,
    fork_speed,
    resolution=1.0,
    resolution_time=5.0,
    time_grid=None,
    T0=None,
    target_form="spacetime",
    weibull_shape=2.0,
    weibull_delay=0.0,
    fit_weibull_shape=False,
    alpha_smooth=1e-2,
    alpha_prior=0.0,
    maxiter=10,
    refine_edge_kb=1000,
    refine_verbose=None,
    perQ=False,
    floor=1e-12,
    error=None,
    stall_rate=0.0,
    stall_rate_per_kb=None,
    tau=np.inf,
    sim_number=100,
    refine_seed=None,
):
    error = _normalise_error_threshold(error)
    if _resolve_stall_probability(stall_rate, stall_rate_per_kb, resolution) > 0:
        from .weibull_stochastic import fit_stochastic_weibull

        return fit_stochastic_weibull(
            observed_timing, fork_speed, resolution=resolution, resolution_time=resolution_time,
            time_grid=time_grid, T0=T0, target_form=target_form,
            weibull_shape=weibull_shape, weibull_delay=weibull_delay,
            fit_weibull_shape=fit_weibull_shape, alpha_smooth=alpha_smooth,
            alpha_prior=alpha_prior, maxiter=maxiter, refine_edge_kb=refine_edge_kb,
            refine_verbose=refine_verbose, perQ=perQ, floor=floor, error=error,
            stall_rate=stall_rate, stall_rate_per_kb=stall_rate_per_kb, tau=tau,
            sim_number=sim_number, refine_seed=refine_seed,
        )
    T_obs = np.asarray(observed_timing, dtype=float)
    if T_obs.ndim != 1 or T_obs.size == 0:
        raise ValueError("observed_timing must be a non-empty one-dimensional array.")

    offset = 0.0 if T0 is None else float(T0)
    if not np.isfinite(offset):
        raise ValueError("T0 must be finite when supplied.")
    target_timing = T_obs - offset
    valid = np.isfinite(T_obs) & np.isfinite(target_timing) & (target_timing > 0)
    crop = _refinement_crop_mask(T_obs.size, resolution, perQ=perQ, refine_edge_kb=refine_edge_kb)
    fit_mask = valid & crop
    if not np.any(fit_mask):
        raise ValueError("observed_timing must contain finite positive values after edge filtering.")

    target_form = _canonical_weibull_target_form(target_form)
    if target_form not in {"time", "spacetime"}:
        raise ValueError("Weibull target_shape is implemented for target_form='time' or 'spacetime'.")

    k0 = float(weibull_shape)
    if not np.isfinite(k0) or k0 <= 0:
        raise ValueError("weibull_shape must be finite and positive.")
    t = _weibull_time_axis_for_timing(
        target_timing[fit_mask],
        resolution_time=resolution_time,
        time_grid=time_grid,
    )
    delay = _weibull_delay_profile(weibull_delay, T_obs.size)

    alpha_init = _weibull_alpha_initial(
        np.where(valid, target_timing, np.nanmedian(target_timing[fit_mask])),
        fork_speed,
        shape=k0,
        floor=floor,
    )
    if target_form == "time":
        alpha0 = np.full(T_obs.size, float(np.nanmedian(alpha_init[fit_mask])), dtype=float)
        variable_alpha = False
    else:
        alpha0 = alpha_init
        variable_alpha = True

    z0 = np.log(np.maximum(alpha0 if variable_alpha else alpha0[:1], float(floor)))
    if fit_weibull_shape:
        params0 = np.concatenate([z0, [math.log(k0)]])
    else:
        params0 = z0

    smooth = float(alpha_smooth)
    prior = float(alpha_prior)
    if smooth < 0 or prior < 0:
        raise ValueError("alpha_smooth and alpha_prior must be >= 0.")
    verbose = bool(refine_verbose) if refine_verbose is not None else False
    start_time = time.time()
    calls = 0
    best = {"loss": np.inf, "fields": None}
    maxiter = int(maxiter)
    n_fit = max(1, int(np.count_nonzero(fit_mask)))

    def unpack(params):
        if fit_weibull_shape:
            z = params[:-1]
            k = float(np.clip(np.exp(params[-1]), 0.25, 8.0))
        else:
            z = params
            k = k0
        if variable_alpha:
            alpha = np.exp(z)
        else:
            alpha = np.full(T_obs.size, float(np.exp(z[0])), dtype=float)
        return z, alpha, k

    def objective(params):
        nonlocal calls
        params = np.asarray(params, dtype=float)
        if np.any(~np.isfinite(params)):
            return 1e30
        z, alpha, k = unpack(params)
        try:
            fields = _weibull_survival_fields(
                alpha,
                fork_speed=fork_speed,
                shape=k,
                resolution=resolution,
                resolution_time=resolution_time,
                time_grid=t,
                delay=delay,
                rate_floor=floor,
            )
        except NotImplementedError:
            # Unsupported speed fields cannot become valid optimizer steps.
            # Preserve the backend explanation instead of masking it as a
            # numerical loss and eventually trying to return empty fit data.
            raise
        except Exception:
            return 1e30
        predicted = fields["predicted_timing"] + offset
        residual = predicted - T_obs
        data_loss = 0.5 * float(np.dot(residual[fit_mask], residual[fit_mask]))
        mse = mean_squared_error(T_obs[fit_mask], predicted[fit_mask])
        reg = 0.0
        if variable_alpha and smooth > 0 and z.size > 1:
            dz = np.diff(z)
            reg += 0.5 * smooth * float(np.dot(dz, dz))
        if prior > 0:
            dz0 = z - z0[:z.size]
            reg += 0.5 * prior * float(np.dot(dz0, dz0))
        loss = data_loss + reg
        calls += 1
        if loss < best["loss"]:
            fields["predicted_timing"] = predicted
            fields["residual"] = residual
            fields["objective"] = loss
            fields["data_loss"] = data_loss
            fields["mse"] = mse
            fields["regularisation"] = reg
            fields["target_form"] = target_form
            fields["target_shape"] = "weibull"
            best["loss"] = loss
            best["fields"] = fields
        maxiter_display = max(1, maxiter)
        done = calls >= maxiter_display or _error_reached(mse, error)
        if _fit_progress_enabled(verbose) and calls <= maxiter_display:
            etaferr(
                it=maxiter_display if done else min(calls, maxiter_display),
                maxiter=maxiter_display,
                start_time=start_time,
                err=mse,
                done=done,
            )
        if _error_reached(mse, error):
            raise _FitConverged
        return loss

    bounds = [(math.log(float(floor)), None)] * z0.size
    if fit_weibull_shape:
        bounds.append((math.log(0.25), math.log(8.0)))

    speed_arr = np.asarray(fork_speed, dtype=float)
    analytic_constant_speed = (
        speed_arr.size == 1
        and np.allclose(delay, delay[0])
        and not perQ
    )

    if analytic_constant_speed:
        def objective_grad(params):
            nonlocal calls
            loss, grad, fields = _weibull_constant_speed_loss_grad(
                params,
                target_timing=target_timing,
                fit_mask=fit_mask,
                fork_speed=fork_speed,
                shape=k0,
                resolution=resolution,
                time_grid=t,
                delay=delay,
                offset=offset,
                variable_alpha=variable_alpha,
                z0=z0,
                alpha_smooth=smooth,
                alpha_prior=prior,
                floor=floor,
                fit_shape=fit_weibull_shape,
            )
            fields["target_form"] = target_form
            data_mse = 2.0 * fields["data_loss"] / n_fit
            fields["mse"] = data_mse
            calls += 1
            if loss < best["loss"]:
                best["loss"] = loss
                best["fields"] = fields
            maxiter_display = max(1, maxiter)
            done = calls >= maxiter_display or _error_reached(data_mse, error)
            if _fit_progress_enabled(verbose) and calls <= maxiter_display:
                etaferr(
                    it=maxiter_display if done else min(calls, maxiter_display),
                    maxiter=maxiter_display,
                    start_time=start_time,
                    err=data_mse,
                    done=done,
                )
            if _error_reached(data_mse, error):
                raise _FitConverged
            return loss, grad

        runner = lambda: minimize(
            objective_grad,
            params0,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={
                "maxiter": maxiter,
                "maxfun": max(maxiter * 10, maxiter + 20),
                "ftol": 1e-8,
                "maxls": 20,
            },
        )
    else:
        operator = _make_weibull_cone(
            fork_speed, t, delay, resolution, resolution_time, perQ,
        )

        def objective_cone_grad(params):
            nonlocal calls
            check_cancelled()
            loss, gradient, fields = _weibull_cone_loss_grad(
                params, operator, target_timing, fit_mask, k0, t, offset,
                variable_alpha, z0, smooth, prior, floor, fit_weibull_shape,
            )
            fields["target_form"] = target_form
            mse = 2.0 * fields["data_loss"] / n_fit
            fields["mse"] = mse
            calls += 1
            if loss < best["loss"]:
                best["loss"] = loss
                best["fields"] = fields
            if _fit_progress_enabled(verbose) and calls <= max(1, maxiter):
                etaferr(
                    it=calls, maxiter=max(1, maxiter), start_time=start_time,
                    err=mse, done=calls >= maxiter or _error_reached(mse, error),
                )
            if _error_reached(mse, error):
                raise _FitConverged
            return loss, gradient

        runner = lambda: minimize(
            objective_cone_grad,
            params0,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={
                "maxiter": maxiter,
                "maxfun": max(maxiter * 10, maxiter + 20),
                "ftol": 1e-8,
                "maxls": 20,
            },
        )

    try:
        result = runner()
    except _FitConverged:
        class _EarlyStopResult:
            success = True
            message = "Reached requested error threshold."

        result = _EarlyStopResult()

    maxiter_display = max(1, maxiter)
    if _fit_progress_enabled(verbose) and 0 < calls < maxiter_display and best["fields"] is not None:
        etaferr(
            it=maxiter_display,
            maxiter=maxiter_display,
            start_time=start_time,
            err=best["fields"].get("mse", np.nan),
            done=True,
        )

    fields = dict(best["fields"])
    fields["optimisation_success"] = bool(result.success)
    fields["optimisation_message"] = str(result.message)
    values = fields["firing_rate"]
    if target_form == "time":
        values = np.nanmean(values, axis=0, keepdims=True)
    info_array_cls = globals().get("_RmapInfoArray")
    if info_array_cls is None:
        class _LocalRmapInfoArray(np.ndarray):
            def __new__(cls, array, info=None):
                obj = np.asarray(array, dtype=float).view(cls)
                obj._rmap_info = dict(info or {})
                return obj

            def __array_finalize__(self, obj):
                self._rmap_info = getattr(obj, "_rmap_info", {})

        info_array_cls = _LocalRmapInfoArray
    return info_array_cls(values, fields)


def _mixed_spacetime_correction(ratio, bins, n_time, mask, kernel_t=1.0):
    correction = np.ones((int(n_time), ratio.size), dtype=float)
    columns = np.where(mask)[0]
    if columns.size == 0:
        return correction

    centers = np.asarray(bins[columns], dtype=float)
    values = np.asarray(ratio[columns], dtype=float)
    kernel_t = 0.0 if kernel_t is None else float(kernel_t)
    if not np.isfinite(kernel_t) or kernel_t < 0:
        raise ValueError("kernel_xt_t must be finite and >= 0.")

    if kernel_t == 0:
        correction[bins[columns], columns] = values
        return correction

    time_idx = np.arange(int(n_time), dtype=float)[:, None]
    weights = np.exp(-0.5 * ((time_idx - centers[None, :]) / kernel_t) ** 2)
    correction[:, columns] = 1.0 + weights * (values[None, :] - 1.0)
    return np.maximum(correction, 1e-12)


def _refine_temporal_firing(
    firing_rate_time,
    observed_timing,
    fork_speed,
    resolution,
    resolution_time,
    T0=None,
    temporal_fit_step=0.35,
    temporal_maxiter=10,
    temporal_sim_number=100,
    refine_ratio_bounds=(0.25, 4.0),
    refine_edge_kb=1000,
    refine_seed=None,
    refine_verbose=None,
    perQ=False,
    floor=0.0,
    error=None,
    stall_rate=0.0,
    stall_rate_per_kb=None,
    tau=np.inf,
):
    rates = np.asarray(firing_rate_time, dtype=float).reshape(1, -1).copy()
    error = _normalise_error_threshold(error)
    maxiter = int(temporal_maxiter)
    if maxiter <= 0:
        return rates

    T = np.asarray(observed_timing, dtype=float)
    n = T.size
    n_time = rates.shape[1]
    bins = _timing_bin_indices(T, n_time, resolution_time=resolution_time, T0=T0)
    crop = _refinement_crop_mask(n, resolution, perQ=perQ, refine_edge_kb=refine_edge_kb)
    low, high = map(float, refine_ratio_bounds)
    if not (np.isfinite(low) and np.isfinite(high) and 0 < low < high):
        raise ValueError("refine_ratio_bounds must be a positive (low, high) pair.")

    fit_step = float(temporal_fit_step)
    if not np.isfinite(fit_step) or fit_step < 0:
        raise ValueError("fit_step_t must be finite and >= 0.")

    verbose = bool(refine_verbose) if refine_verbose is not None else False
    best_rates = rates.copy()
    best_err = np.inf
    best_prediction = None
    history = []
    start_time = time.time()
    for it in range(1, maxiter + 1):
        seed = None if refine_seed is None else int(refine_seed) + it - 1
        T_pred = rsim(
            firing_rate=rates,
            fork_speed=fork_speed,
            sim_number=temporal_sim_number,
            resolution_space=resolution,
            resolution_time=resolution_time,
            n_bins=n,
            perQ=perQ,
            stall_rate=stall_rate,
            stall_rate_per_kb=stall_rate_per_kb,
            tau=tau,
            seed=seed,
            verbose=False,
            retain_rep_times_per_sim=False,
            retain_inter_origin_distances=False,
        )["replication_timing"]
        valid = crop & np.isfinite(T) & np.isfinite(T_pred) & (T > 0) & (T_pred > 0)
        if not np.any(valid):
            break
        err = mean_squared_error(T[valid], T_pred[valid])
        history.append(float(err))
        if err < best_err:
            best_err = err
            best_rates = rates.copy()
            best_prediction = np.asarray(T_pred, dtype=float).copy()
        ratio = np.ones(n, dtype=float)
        ratio[valid] = np.clip(T_pred[valid] / T[valid], low, high)
        updates = _mean_ratio_by_bin(ratio, bins, n_time, valid)
        rates[0] *= updates ** fit_step
        np.maximum(rates, floor, out=rates)

        done = it == maxiter or _error_reached(err, error)
        if _fit_progress_enabled(verbose):
            etaferr(it=it, maxiter=maxiter, start_time=start_time, err=err, done=done)
        if done and it != maxiter:
            break

    if best_prediction is None:
        return best_rates
    residual = best_prediction - T
    return _RmapInfoArray(best_rates, {
        "replication_timing_fit": best_prediction,
        "residual": residual,
        "fit_mse": float(np.mean(residual[np.isfinite(residual)]**2)),
        "fit_history": np.asarray(history),
        "fit_objective": float(best_err),
    })


def _refine_spacetime_firing(
    firing_rate_spacetime,
    observed_timing,
    fork_speed,
    resolution,
    resolution_time,
    T0=None,
    spacetime_fit_step=0.25,
    spacetime_space_fit_step=None,
    spacetime_time_fit_step=None,
    refine_maxiter=10,
    spacetime_sim_number=100,
    spacetime_update="factorized",
    spacetime_kernel_t=1.0,
    refine_ratio_bounds=(0.25, 4.0),
    refine_edge_kb=1000,
    refine_seed=None,
    refine_verbose=None,
    perQ=False,
    floor=0.0,
    error=None,
    stall_rate=0.0,
    stall_rate_per_kb=None,
    tau=np.inf,
):
    rates = np.asarray(firing_rate_spacetime, dtype=float).copy()
    error = _normalise_error_threshold(error)
    maxiter = int(refine_maxiter)
    if maxiter <= 0:
        return rates

    T = np.asarray(observed_timing, dtype=float)
    n = T.size
    if rates.ndim != 2 or rates.shape[1] != n:
        raise ValueError("firing_rate_spacetime must have internal shape (time, positions).")

    periodic_smoother = PeriodicFiringSmoother(T, fork_speed, resolution) if perQ else None
    if periodic_smoother is not None:
        rates = periodic_smoother(rates, axis=1, floor=floor)

    n_time = rates.shape[0]
    bins = _timing_bin_indices(T, n_time, resolution_time=resolution_time, T0=T0)
    crop = _refinement_crop_mask(n, resolution, perQ=perQ, refine_edge_kb=refine_edge_kb)
    low, high = map(float, refine_ratio_bounds)
    if not (np.isfinite(low) and np.isfinite(high) and 0 < low < high):
        raise ValueError("refine_ratio_bounds must be a positive (low, high) pair.")

    base_step = float(spacetime_fit_step)
    space_step = base_step if spacetime_space_fit_step is None else float(spacetime_space_fit_step)
    time_step = base_step if spacetime_time_fit_step is None else float(spacetime_time_fit_step)
    if not np.isfinite(space_step) or space_step < 0:
        raise ValueError("fit_step_xt_x must be finite and >= 0.")
    if not np.isfinite(time_step) or time_step < 0:
        raise ValueError("fit_step_xt_t must be finite and >= 0.")
    update_mode = _canonical_spacetime_update(spacetime_update)

    verbose = bool(refine_verbose) if refine_verbose is not None else False
    best_rates = rates.copy()
    best_err = np.inf
    best_prediction = None
    history = []
    start_time = time.time()
    for it in range(1, maxiter + 1):
        seed = None if refine_seed is None else int(refine_seed) + it - 1
        T_pred = rsim(
            firing_rate=rates.T,
            fork_speed=fork_speed,
            sim_number=spacetime_sim_number,
            resolution_space=resolution,
            resolution_time=resolution_time,
            n_bins=n,
            perQ=perQ,
            stall_rate=stall_rate,
            stall_rate_per_kb=stall_rate_per_kb,
            tau=tau,
            seed=seed,
            verbose=False,
            retain_rep_times_per_sim=False,
            retain_inter_origin_distances=False,
        )["replication_timing"]
        valid = crop & np.isfinite(T) & np.isfinite(T_pred) & (T > 0) & (T_pred > 0)
        if not np.any(valid):
            break
        err = mean_squared_error(T[valid], T_pred[valid])
        history.append(float(err))
        if err < best_err:
            best_err = err
            best_rates = rates.copy()
            best_prediction = np.asarray(T_pred, dtype=float).copy()
        ratio = np.ones(n, dtype=float)
        ratio[valid] = np.clip(T_pred[valid] / T[valid], low, high)
        if update_mode == "mixed":
            correction = _mixed_spacetime_correction(
                ratio,
                bins,
                n_time,
                valid,
                kernel_t=spacetime_kernel_t,
            )
            rates *= correction ** base_step
        else:
            rates[:, valid] *= ratio[valid][None, :] ** space_step
            time_updates = _mean_ratio_by_bin(ratio, bins, n_time, valid)
            rates *= time_updates[:, None] ** time_step
        np.maximum(rates, floor, out=rates)
        if periodic_smoother is not None:
            rates = periodic_smoother(rates, axis=1, floor=floor)

        done = it == maxiter or _error_reached(err, error)
        if _fit_progress_enabled(verbose):
            etaferr(it=it, maxiter=maxiter, start_time=start_time, err=err, done=done)
        if done and it != maxiter:
            break

    if best_prediction is None:
        return best_rates
    residual = best_prediction - T
    return _RmapInfoArray(best_rates, {
        "replication_timing_fit": best_prediction,
        "residual": residual,
        "fit_mse": float(np.mean(residual[np.isfinite(residual)]**2)),
        "fit_history": np.asarray(history),
        "fit_objective": float(best_err),
        "fit_periodic_smoothing_kb": (
            periodic_smoother.length_kb if periodic_smoother is not None else 0.
        ),
    })


def map_timing_firing_time(
    observed_timing,
    fork_speed=1.4,
    resolution=1.0,
    resolution_x=None,
    resolution_time=5.0,
    time_grid=None,
    T0=None,
    target_shape="weibull",
    weibull_shape=2.0,
    weibull_delay=0.0,
    fit_weibull_shape=False,
    alpha_smooth=1e-2,
    alpha_prior=0.0,
    temporal_smoothw=None,
    space_fit_step=None,
    space_maxiter=None,
    temporal_fit_step=0.35,
    temporal_maxiter=10,
    temporal_sim_number=100,
    refine_ratio_bounds=(0.25, 4.0),
    refine_edge_kb=1000,
    refine_seed=None,
    refine_verbose=None,
    error=None,
    floor=0.0,
    **kwargs,
):
    """Infer a time-only firing profile from timing and constant fork speed.

    The route first fits the implemented spatial profile ``I(x)`` and then
    projects that fit onto the timing coordinate. The returned row vector has
    shape ``(1, n_time)`` so it is interpreted as ``I(t)`` by ``rsim``.
    """
    resolved_resolution = _normalise_resolution_x(
        resolution,
        resolution_x=resolution_x,
    )
    stall_rate = kwargs.get("stall_rate", 0.0)
    stall_rate_per_kb = kwargs.get("stall_rate_per_kb")
    tau = kwargs.get("tau", np.inf)
    _resolve_stall_probability(
        stall_rate,
        stall_rate_per_kb,
        resolved_resolution,
    )
    target_shape = _canonical_target_shape(target_shape)
    if target_shape == "weibull":
        if refine_verbose is None:
            refine_verbose = bool(kwargs.get("verbose", False))
        return _fit_weibull_survival_firing(
            observed_timing,
            fork_speed=fork_speed,
            resolution=resolved_resolution,
            resolution_time=resolution_time,
            time_grid=time_grid,
            T0=T0,
            target_form="time",
            weibull_shape=weibull_shape,
            weibull_delay=weibull_delay,
            fit_weibull_shape=fit_weibull_shape,
            alpha_smooth=alpha_smooth,
            alpha_prior=alpha_prior,
            maxiter=temporal_maxiter,
            refine_edge_kb=refine_edge_kb,
            refine_verbose=refine_verbose,
            perQ=kwargs.get("perQ", False),
            floor=max(float(floor), 1e-12),
            error=error,
            stall_rate=stall_rate,
            stall_rate_per_kb=stall_rate_per_kb,
            tau=tau,
            sim_number=kwargs.get("sim_number", temporal_sim_number),
            refine_seed=refine_seed,
        )

    fit_kwargs = _spatial_firing_fit_kwargs(kwargs)
    fit_kwargs["refine_edge_kb"] = refine_edge_kb
    if space_fit_step is not None:
        fit_kwargs["fit_step"] = space_fit_step
    if space_maxiter is not None:
        fit_kwargs["maxiter"] = space_maxiter
    if error is not None:
        fit_kwargs["error"] = error

    fit_kwargs.update(resolution_time=resolution_time, time_grid=time_grid, refine_seed=refine_seed)
    firing_rate_space = map_timing_firing(
        observed_timing,
        fork_speed=fork_speed,
        resolution=resolution,
        resolution_x=resolution_x,
        **fit_kwargs,
    )
    temporal, time_axis = _temporal_profile_from_spatial_fit(
        observed_timing,
        firing_rate_space,
        resolution_time=resolution_time,
        time_grid=time_grid,
        T0=T0,
        temporal_smoothw=temporal_smoothw,
        floor=floor,
    )
    if refine_verbose is None:
        refine_verbose = bool(fit_kwargs.get("verbose", False))
    if refine_verbose:
        print("\nRefining time-dependent firing rate...")
    return _refine_temporal_firing(
        temporal.reshape(1, -1),
        observed_timing,
        fork_speed=fork_speed,
        resolution=resolved_resolution,
        resolution_time=resolution_time,
        T0=T0,
        temporal_fit_step=temporal_fit_step,
        temporal_maxiter=temporal_maxiter,
        temporal_sim_number=temporal_sim_number,
        refine_ratio_bounds=refine_ratio_bounds,
        refine_edge_kb=refine_edge_kb,
        refine_seed=refine_seed,
        refine_verbose=refine_verbose,
        perQ=fit_kwargs.get("perQ", False),
        floor=floor,
        error=error,
        stall_rate=stall_rate,
        stall_rate_per_kb=stall_rate_per_kb,
        tau=tau,
    )


def map_timing_firing_spacetime(
    observed_timing,
    fork_speed=1.4,
    resolution=1.0,
    resolution_x=None,
    resolution_time=5.0,
    time_grid=None,
    T0=None,
    target_shape="weibull",
    weibull_shape=2.0,
    weibull_delay=0.0,
    fit_weibull_shape=False,
    alpha_smooth=1e-2,
    alpha_prior=0.0,
    temporal_smoothw=None,
    space_fit_step=None,
    space_maxiter=None,
    temporal_fit_step=None,
    temporal_maxiter=None,
    temporal_sim_number=None,
    spacetime_fit_step=None,
    spacetime_space_fit_step=None,
    spacetime_time_fit_step=None,
    spacetime_sim_number=None,
    spacetime_update="factorized",
    spacetime_order="space_then_time",
    spacetime_kernel_t=1.0,
    refine_ratio_bounds=(0.25, 4.0),
    refine_edge_kb=1000,
    refine_seed=None,
    refine_verbose=None,
    error=None,
    floor=0.0,
    **kwargs,
):
    """Infer a space-time firing field from timing and constant fork speed.

    ``spacetime_update='factorized'`` fits an x profile, builds a temporal
    projection, and refines the x and t components in ``spacetime_order``.
    ``spacetime_update='mixed'`` refines local x,t corrections directly.
    """
    resolved_resolution = _normalise_resolution_x(
        resolution,
        resolution_x=resolution_x,
    )
    stall_rate = kwargs.get("stall_rate", 0.0)
    stall_rate_per_kb = kwargs.get("stall_rate_per_kb")
    tau = kwargs.get("tau", np.inf)
    _resolve_stall_probability(
        stall_rate,
        stall_rate_per_kb,
        resolved_resolution,
    )
    target_shape = _canonical_target_shape(target_shape)
    if target_shape == "weibull":
        global_maxiter = kwargs.get("maxiter", None)
        if global_maxiter is not None:
            space_maxiter = global_maxiter
            temporal_maxiter = global_maxiter
        if refine_verbose is None:
            refine_verbose = bool(kwargs.get("verbose", False))
        phase_maxiter = max(
            10 if space_maxiter is None else int(space_maxiter),
            10 if temporal_maxiter is None else int(temporal_maxiter),
        )
        return _fit_weibull_survival_firing(
            observed_timing,
            fork_speed=fork_speed,
            resolution=resolved_resolution,
            resolution_time=resolution_time,
            time_grid=time_grid,
            T0=T0,
            target_form="spacetime",
            weibull_shape=weibull_shape,
            weibull_delay=weibull_delay,
            fit_weibull_shape=fit_weibull_shape,
            alpha_smooth=alpha_smooth,
            alpha_prior=alpha_prior,
            maxiter=phase_maxiter,
            refine_edge_kb=refine_edge_kb,
            refine_verbose=refine_verbose,
            perQ=kwargs.get("perQ", False),
            floor=max(float(floor), 1e-12),
            error=error,
            stall_rate=stall_rate,
            stall_rate_per_kb=stall_rate_per_kb,
            tau=tau,
            sim_number=kwargs.get("sim_number", spacetime_sim_number or temporal_sim_number or 100),
            refine_seed=refine_seed,
        )

    update_mode = _canonical_spacetime_update(spacetime_update)
    order = _canonical_spacetime_order(spacetime_order)
    fit_kwargs = _spatial_firing_fit_kwargs(kwargs)
    fit_kwargs["refine_edge_kb"] = refine_edge_kb
    global_fit_step = fit_kwargs.pop("fit_step", None)
    global_maxiter = fit_kwargs.pop("maxiter", None)
    global_sim_number = fit_kwargs.pop("sim_number", None)
    (
        space_fit_step,
        space_maxiter,
        temporal_fit_step,
        temporal_maxiter,
        temporal_sim_number,
    ) = _resolve_spacetime_phase_controls(
        fit_step=global_fit_step,
        maxiter=global_maxiter,
        sim_number=global_sim_number,
        space_fit_step=space_fit_step,
        space_maxiter=space_maxiter,
        temporal_fit_step=temporal_fit_step,
        temporal_maxiter=temporal_maxiter,
        temporal_sim_number=temporal_sim_number,
        default_space_fit_step=0.6,
        default_space_maxiter=10,
    )
    if spacetime_sim_number is None:
        spacetime_sim_number = temporal_sim_number
    if spacetime_fit_step is None:
        spacetime_fit_step = 0.5 * (space_fit_step + temporal_fit_step)

    initial_space_maxiter = (
        space_maxiter
        if update_mode == "factorized" and order == "space_then_time"
        else 0
    )
    fit_kwargs["fit_step"] = space_fit_step
    fit_kwargs["maxiter"] = initial_space_maxiter
    if error is not None:
        fit_kwargs["error"] = error
    if global_sim_number is not None:
        fit_kwargs["sim_number"] = global_sim_number

    fit_kwargs.update(resolution_time=resolution_time, time_grid=time_grid, refine_seed=refine_seed)
    firing_rate_space = map_timing_firing(
        observed_timing,
        fork_speed=fork_speed,
        resolution=resolution,
        resolution_x=resolution_x,
        **fit_kwargs,
    )
    temporal, time_axis = _temporal_profile_from_spatial_fit(
        observed_timing,
        firing_rate_space,
        resolution_time=resolution_time,
        time_grid=time_grid,
        T0=T0,
        temporal_smoothw=temporal_smoothw,
        floor=floor,
    )

    spatial = np.asarray(firing_rate_space, dtype=float)
    scale = float(np.nanmean(spatial[spatial > 0])) if np.any(spatial > 0) else 1.0
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    firing_rate_spacetime = np.outer(temporal, spatial / scale)
    if refine_verbose is None:
        refine_verbose = bool(fit_kwargs.get("verbose", False))

    if update_mode == "factorized":
        def refine_factorized_phase(field, phase):
            is_space = phase == "space"
            phase_maxiter = space_maxiter if is_space else temporal_maxiter
            phase_step = space_fit_step if is_space else temporal_fit_step
            if phase_maxiter <= 0:
                return field
            if refine_verbose:
                label = "space" if is_space else "time"
                print(f"\nRefining factorized {label} component of firing rate...")
            return _refine_spacetime_firing(
                field,
                observed_timing,
                fork_speed=fork_speed,
                resolution=resolved_resolution,
                resolution_time=resolution_time,
                T0=T0,
                spacetime_fit_step=phase_step,
                spacetime_space_fit_step=phase_step if is_space else 0.0,
                spacetime_time_fit_step=0.0 if is_space else phase_step,
                refine_maxiter=phase_maxiter,
                spacetime_sim_number=temporal_sim_number,
                spacetime_update="factorized",
                spacetime_kernel_t=spacetime_kernel_t,
                refine_ratio_bounds=refine_ratio_bounds,
                refine_edge_kb=refine_edge_kb,
                refine_seed=refine_seed,
                refine_verbose=refine_verbose,
                perQ=fit_kwargs.get("perQ", False),
                floor=floor,
                error=error,
                stall_rate=stall_rate,
                stall_rate_per_kb=stall_rate_per_kb,
                tau=tau,
            )

        if order == "time_then_space":
            firing_rate_spacetime = refine_factorized_phase(firing_rate_spacetime, "time")
            firing_rate_spacetime = refine_factorized_phase(firing_rate_spacetime, "space")
        else:
            firing_rate_spacetime = refine_factorized_phase(firing_rate_spacetime, "time")
        return firing_rate_spacetime.T

    if refine_verbose:
        print("\nRefining mixed space-time firing rate...")
    return _refine_spacetime_firing(
        firing_rate_spacetime,
        observed_timing,
        fork_speed=fork_speed,
        resolution=resolved_resolution,
        resolution_time=resolution_time,
        T0=T0,
        spacetime_fit_step=spacetime_fit_step,
        spacetime_space_fit_step=spacetime_space_fit_step,
        spacetime_time_fit_step=spacetime_time_fit_step,
        refine_maxiter=max(space_maxiter, temporal_maxiter),
        spacetime_sim_number=spacetime_sim_number,
        spacetime_update=spacetime_update,
        spacetime_kernel_t=spacetime_kernel_t,
        refine_ratio_bounds=refine_ratio_bounds,
        refine_edge_kb=refine_edge_kb,
        refine_seed=refine_seed,
        refine_verbose=refine_verbose,
        perQ=fit_kwargs.get("perQ", False),
        floor=floor,
        error=error,
        stall_rate=stall_rate,
        stall_rate_per_kb=stall_rate_per_kb,
        tau=tau,
    ).T


def map_timing_constant_firing(
    observed_timing,
    fork_speed=1.4,
    resolution=1.0,
    resolution_x=None,
    resolution_time=5.0,
    neighbourhood_range=2000,
    fit_step=0.6,
    maxiter=10,
    sim_number=100,
    err_threshold=15.0,
    rate_bounds=None,
    error=None,
    refine_seed=None,
    perQ=False,
    n_bins=None,
    verbose=True,
    print_every=1,
    time_grid=None,
):
    """Infer one constant firing rate from timing with supplied fork speed."""
    resolution = _normalise_resolution_x(resolution, resolution_x=resolution_x)
    error = _normalise_error_threshold(error)
    T = np.asarray(observed_timing, dtype=float)
    if T.ndim != 1:
        raise ValueError("observed_timing must be a one-dimensional array.")
    if T.size == 0:
        raise ValueError("observed_timing must contain at least one value.")
    if n_bins is not None and int(n_bins) != T.size:
        raise ValueError("n_bins must match the length of observed_timing.")

    fork_speed, _ = _uniform_speed_for_simulation(fork_speed, time_grid, resolution_time)
    fork_speed_arr = np.asarray(fork_speed, dtype=float)
    variable_fork_speed = fork_speed_arr.size != 1
    if variable_fork_speed:
        if np.any(~np.isfinite(fork_speed_arr)) or np.any(fork_speed_arr <= 0):
            raise ValueError("fork_speed must contain finite positive values.")
        fork_speed_reference = float(np.nanmedian(fork_speed_arr[fork_speed_arr > 0]))
    else:
        fork_speed = float(fork_speed_arr.reshape(-1)[0])
        if not np.isfinite(fork_speed) or fork_speed <= 0:
            raise ValueError("fork_speed must be a finite positive scalar.")
        fork_speed_reference = fork_speed

    lm = int(1000 / resolution)
    if perQ:
        crop = slice(None)
    else:
        crop = slice(lm, -lm) if 2 * lm < T.size else slice(None)

    v_per_bin = fork_speed_reference / resolution
    finite_positive_timing = T[crop][np.isfinite(T[crop]) & (T[crop] > 0)]
    if finite_positive_timing.size:
        profile_guess = (math.pi / (4.0 * v_per_bin)) * finite_positive_timing ** (-2)
        rate_guess = float(np.nanmedian(profile_guess))
    else:
        rate_guess = 1e-6
    if not np.isfinite(rate_guess) or rate_guess <= 0:
        rate_guess = 1e-6

    floor = 10.0 ** (-err_threshold)
    if rate_bounds is None:
        low = max(rate_guess * 1e-4, floor)
        high = max(rate_guess * 1e4, low * 10.0)
    else:
        if len(rate_bounds) != 2:
            raise ValueError("rate_bounds must be a (min_rate, max_rate) pair.")
        low, high = map(float, rate_bounds)
        if not (np.isfinite(low) and np.isfinite(high)) or low <= 0 or high <= low:
            raise ValueError("rate_bounds must satisfy 0 < min_rate < max_rate.")

    rate = float(np.clip(rate_guess, low, high))

    maxiter = int(maxiter)
    print_every = max(1, int(print_every))
    best_rate = rate
    best_mse = np.inf
    best_timing_fit = None
    start_time = time.time()

    for it in range(1, maxiter + 1):
        check_cancelled()
        if variable_fork_speed:
            seed = None if refine_seed is None else int(refine_seed) + it - 1
            predicted_full = rsim(
                firing_rate=rate,
                fork_speed=fork_speed,
                sim_number=sim_number,
                resolution_space=resolution,
                resolution_time=resolution_time,
                n_bins=T.size,
                perQ=perQ,
                seed=seed,
                verbose=False,
                retain_rep_times_per_sim=False,
                retain_inter_origin_distances=False,
            )["replication_timing"]
        else:
            predicted_full = map_firing_timing(
                rate,
                fork_speed=fork_speed,
                resolution=resolution,
                neighbourhood_range=neighbourhood_range,
                perQ=perQ,
                n_bins=T.size,
            )
        observed = T[crop]
        predicted = predicted_full[crop]
        ok = (
            np.isfinite(observed)
            & np.isfinite(predicted)
            & (observed > 0)
            & (predicted > 0)
        )
        if not np.any(ok):
            break
        mse = mean_squared_error(observed[ok], predicted[ok])
        if mse < best_mse:
            best_mse = mse
            best_rate = rate
            best_timing_fit = np.asarray(predicted_full, dtype=float).copy()
        ratio = float(np.nanmedian(predicted[ok] / observed[ok]))
        if not np.isfinite(ratio) or ratio <= 0:
            break
        rate *= ratio ** fit_step
        rate = float(np.clip(rate, low, high))
        done = it == maxiter or _error_reached(mse, error)
        if _fit_progress_enabled(verbose) and (it % print_every == 0 or done):
            etaferr(it=it, maxiter=maxiter, start_time=start_time, err=mse, done=done)
        if done and it != maxiter:
            break

    info = (
        {}
        if best_timing_fit is None
        else {"replication_timing_fit": best_timing_fit}
    )
    return _RmapInfoFloat(best_rate, info)


def map_timing_firing_constant_speed(
    source,
    resolution=1.0,
    resolution_x=None,
    resolution_time=5.0,
    time_grid=None,
    T0=None,
    neighbourhood_range=2000,
    fit_step=0.1,
    maxiter=10,
    speed_bounds=None,
    initial_fork_speed=1.4,
    target_shape="best",
    error=None,
    refine_edge_kb=1000,
    perQ=False,
    n_bins=None,
    verbose=True,
    print_every=1,
):
    """Infer one constant fork speed from timing and firing rate."""
    if perQ:
        return _periodic_speed_inverse(
            source, "constant", resolution, resolution_x, resolution_time, time_grid,
            T0, initial_fork_speed, speed_bounds, maxiter, verbose,
        )
    resolution = _normalise_resolution_x(resolution, resolution_x=resolution_x)
    error = _normalise_error_threshold(error)
    target_shape = _canonical_speed_target_shape(target_shape)
    if not isinstance(source, dict):
        raise ValueError("source must contain replication_timing and firing_rate.")
    if "replication_timing" not in source or "firing_rate" not in source:
        raise ValueError("source must contain replication_timing and firing_rate.")

    T = np.asarray(source["replication_timing"], dtype=float)
    if T.ndim != 1:
        raise ValueError("replication_timing must be a one-dimensional array.")
    if T.size == 0:
        raise ValueError("replication_timing must contain at least one value.")
    if n_bins is not None and int(n_bins) != T.size:
        raise ValueError("n_bins must match the length of replication_timing.")

    del neighbourhood_range
    return _fit_constant_speed_with_general_firing(
        T,
        source["firing_rate"],
        resolution=resolution,
        resolution_time=resolution_time,
        time_grid=time_grid,
        T0=T0,
        fit_step=fit_step,
        maxiter=maxiter,
        speed_bounds=speed_bounds,
        initial_fork_speed=initial_fork_speed,
        refine_edge_kb=refine_edge_kb,
        perQ=perQ,
        error=error,
        verbose=verbose,
        print_every=print_every,
    )


def map_timing_firing_spatial_speed(
    source,
    resolution=1.0,
    resolution_x=None,
    resolution_time=5.0,
    time_grid=None,
    T0=None,
    neighbourhood_range=2000,
    fit_step=0.4,
    maxiter=10,
    space_fit_step=None,
    space_maxiter=None,
    speed_bounds=None,
    initial_fork_speed=1.4,
    target_shape="best",
    weibull_shape=2.0,
    weibull_delay=0.0,
    error=None,
    smoothw=100,
    refine_edge_kb=1000,
    perQ=False,
    n_bins=None,
    verbose=True,
    print_every=1,
):
    """Fit effective spatial speed by iterating the deterministic timing map.

    A late prediction increases local speed; an early prediction decreases it.
    Smooth the log-speed correction, rather than repeatedly flattening the
    accumulated profile. Accept only updates that improve timing MSE, falling
    back to a local, damped correction when smoothing prevents improvement.
    The best evaluated profile is retained. Firing uses the legacy per-bin
    hazard convention here; the public rmap wrapper converts rate densities.

    For a linear subregion, ``refine_edge_kb`` limits reflected endpoint
    context and a narrow update taper, as in spatial firing-rate fitting.
    Every supplied position remains refinable. Pass zero for true chromosome
    ends; periodic fits already have flanking context and are unaffected.
    """
    del resolution_time, time_grid
    resolution = _normalise_resolution_x(resolution, resolution_x=resolution_x)
    target_shape = _canonical_speed_target_shape(target_shape)
    error = _normalise_error_threshold(error)
    if space_fit_step is not None:
        fit_step = space_fit_step
    if space_maxiter is not None:
        maxiter = space_maxiter
    if not isinstance(source, dict):
        raise ValueError("source must contain replication_timing and firing_rate.")
    if "replication_timing" not in source or "firing_rate" not in source:
        raise ValueError("source must contain replication_timing and firing_rate.")

    T = np.asarray(source["replication_timing"], dtype=float)
    if T.ndim != 1:
        raise ValueError("replication_timing must be a one-dimensional array.")
    if T.size == 0:
        raise ValueError("replication_timing must contain at least one value.")
    if n_bins is not None and int(n_bins) != T.size:
        raise ValueError("n_bins must match the length of replication_timing.")

    anchor = 0.0 if T0 is None else float(T0)
    if not np.isfinite(anchor):
        raise ValueError("T0 must be finite.")
    relative_timing = T - anchor
    valid = np.isfinite(relative_timing) & (relative_timing > 0)
    if not np.any(valid):
        raise ValueError("replication_timing must contain finite values greater than T0.")
    fit_step = float(fit_step)
    if not np.isfinite(fit_step) or fit_step <= 0:
        raise ValueError("fit_step must be finite and positive.")
    if int(maxiter) != maxiter or maxiter < 0:
        raise ValueError("maxiter must be a nonnegative integer.")
    if not np.isfinite(neighbourhood_range) or neighbourhood_range <= 0:
        raise ValueError("neighbourhood_range must be finite and positive.")

    firing_rate = np.asarray(source["firing_rate"], dtype=float)
    if firing_rate.size != 1:
        if firing_rate.ndim == 2 and firing_rate.shape[1] == 1:
            firing_rate = firing_rate.reshape(-1)
        if firing_rate.ndim != 1 or firing_rate.size != T.size:
            raise ValueError("firing_rate must be scalar, (n,), or (n,1) for spatial speed fitting.")
        if np.any(~np.isfinite(firing_rate)) or np.any(firing_rate < 0):
            raise ValueError("firing_rate must contain finite values >= 0.")
    else:
        firing_rate = float(firing_rate.reshape(-1)[0])
        if not np.isfinite(firing_rate) or firing_rate < 0:
            raise ValueError("firing_rate scalar must be finite and >= 0.")
    if not np.any(np.asarray(firing_rate) > 0):
        raise ValueError("A positive firing rate is required to infer fork speed.")

    initial = np.asarray(initial_fork_speed, dtype=float)
    if initial.size == 1:
        value = float(initial.reshape(-1)[0])
        if not np.isfinite(value) or value <= 0:
            value = 1.4
        speed = np.full(T.size, value, dtype=float)
    elif initial.ndim == 1 and initial.size == T.size:
        speed = initial.astype(float, copy=True)
    elif initial.ndim == 2 and initial.shape[1] == 1 and initial.shape[0] == T.size:
        speed = initial.reshape(-1).astype(float, copy=True)
    else:
        raise ValueError("initial_fork_speed must be scalar, (n,), or (n,1).")
    if np.any(~np.isfinite(speed)) or np.any(speed <= 0):
        raise ValueError("initial_fork_speed must contain finite positive values.")

    reference_speed = float(np.nanmedian(speed))
    if target_shape == "best" and speed_bounds is None:
        low, high = _direct_speed_shape_bounds([reference_speed], speed_bounds=None)
    elif speed_bounds is None:
        low = max(reference_speed * 1e-3, 1e-12)
        high = max(reference_speed * 1e3, low * 10.0)
    else:
        if len(speed_bounds) != 2:
            raise ValueError("speed_bounds must be a (min_speed, max_speed) pair.")
        low, high = map(float, speed_bounds)
        if not (np.isfinite(low) and np.isfinite(high)) or low <= 0 or high <= low:
            raise ValueError("speed_bounds must satisfy 0 < min_speed < max_speed.")
    speed = np.clip(speed, low, high)

    fit_mask, update_weights, edge_bins, taper_bins = _spatial_firing_edge_controls(
        T.size,
        resolution,
        observed_timing=relative_timing,
        fork_speed=reference_speed,
        neighbourhood_range=neighbourhood_range,
        perQ=perQ,
        refine_edge_kb=refine_edge_kb,
    )
    # Reflection needs an explicit spatial firing profile even when the
    # supplied firing rate is constant. Keep the unprotected path unchanged.
    context_firing = (
        np.full(T.size, firing_rate, dtype=float)
        if edge_bins and np.asarray(firing_rate).size == 1 else firing_rate
    )

    maxiter = int(maxiter)
    print_every = max(1, int(print_every))
    best_speed = speed.copy()
    best_mse = np.inf
    best_timing_fit = None
    history = []
    converged = False
    convergence_message = (
        "Initialization only; optimization disabled by maxiter=0."
        if maxiter == 0 else "Maximum spatial-speed updates reached."
    )
    start_time = time.time()
    predicted = None
    evaluations = 0

    def predict(candidate):
        nonlocal evaluations
        check_cancelled()
        values = _map_spatial_firing_timing_with_context(
            context_firing,
            fork_speed=candidate,
            resolution=resolution,
            neighbourhood_range=neighbourhood_range,
            perQ=perQ,
            edge_bins=edge_bins,
        )
        evaluations += 1
        check_cancelled()
        return values

    # Evaluate the initial profile and every update, including the last one.
    # Fit all supplied positions; score the core where flanking data are known.
    for it in range(maxiter + 1):
        _report_progress(
            .04 + .90 * it / max(1, maxiter),
            f"Fitting fork speed... iteration {it}/{maxiter}",
        )
        check_cancelled()
        if predicted is None:
            predicted = predict(speed)
        check_cancelled()
        ok = valid & np.isfinite(predicted) & (predicted > 0)
        if not np.any(ok):
            raise ValueError("The supplied firing rate produced no finite timing predictions.")
        score_ok = ok & fit_mask
        if not np.any(score_ok):
            score_ok = ok
        mse = mean_squared_error(relative_timing[score_ok], predicted[score_ok])
        history.append(mse)
        if mse < best_mse:
            best_mse = mse
            best_speed = speed.copy()
            best_timing_fit = predicted + anchor
        converged = _error_reached(mse, error)
        done = it == maxiter or converged
        if _fit_progress_enabled(verbose) and (it % print_every == 0 or done):
            etaferr(
                it=it, maxiter=max(1, maxiter), start_time=start_time,
                err=mse, done=done, label="Fitting fork speed",
            )
        if done:
            if converged:
                convergence_message = "Timing MSE threshold reached."
            break
        ratio = np.ones(T.size, dtype=float)
        ratio[ok] = predicted[ok] / relative_timing[ok]
        if edge_bins:
            ratio[~fit_mask] = np.clip(ratio[~fit_mask], .5, 2.)
        correction = np.log(ratio)
        directions = []
        if smoothw is not None and int(smoothw) > 1:
            smoothed = uniform_filter1d(
                correction, size=min(int(smoothw), T.size),
                mode="wrap" if perQ else "nearest",
            )
            directions.append((smoothed, 1))
        directions.append((correction, 8))
        accepted = False
        for direction, attempts in directions:
            for attempt in range(attempts):
                # Clipping in log space avoids overflow for very small times.
                candidate = np.exp(np.clip(
                    np.log(speed) + fit_step * .5 ** attempt * direction * update_weights,
                    np.log(low), np.log(high),
                ))
                candidate = np.clip(candidate, low, high)
                if np.array_equal(candidate, speed):
                    continue
                candidate_prediction = predict(candidate)
                candidate_ok = valid & np.isfinite(candidate_prediction) & (candidate_prediction > 0)
                if not np.array_equal(candidate_ok, ok):
                    continue
                candidate_mse = mean_squared_error(
                    relative_timing[score_ok], candidate_prediction[score_ok],
                )
                if candidate_mse < mse:
                    speed = candidate
                    predicted = candidate_prediction
                    accepted = True
                    break
            if accepted:
                break
        if not accepted:
            converged = True
            convergence_message = "No improving spatial-speed update within the supplied bounds."
            break

    _report_progress(.94, "Fork-speed fitting complete")

    if target_shape == "exponential":
        return _project_exponential_speed_shape(
            best_speed,
            "space",
            resolution=resolution,
            speed_bounds=(low, high),
        )
    if target_shape == "weibull":
        return _project_weibull_speed_shape(
            best_speed,
            "space",
            resolution=resolution,
            speed_bounds=(low, high),
            weibull_shape=weibull_shape,
            weibull_delay=weibull_delay,
        )
    if target_shape == "best":
        info = {
            "target_shape": "best",
            "fork_speed_shape": "site-specific v(x)",
            "fork_slowness_shape": "site-specific beta(x)",
            "fork_speed_bounds": (float(low), float(high)),
            "fit_mse": None if not np.isfinite(best_mse) else float(best_mse),
            "fit_history": np.asarray(history),
            "fit_edge_bins": edge_bins,
            "fit_taper_bins": taper_bins,
            "converged": converged,
            "convergence_message": convergence_message,
            "optimizer_evaluations": evaluations,
            "forward_model": "deterministic local effective-speed timing map",
            "speed_form": "space",
        }
        timing_fit = (
            best_timing_fit
            if best_timing_fit is not None
            else anchor + predict(best_speed)
        )
        info["replication_timing_fit"] = timing_fit
        info["residual"] = timing_fit - T
        return _RmapInfoArray(best_speed, info=info)
    return best_speed


def _validate_timing_and_firing_for_speed(source):
    if not isinstance(source, dict):
        raise ValueError("source must contain replication_timing and firing_rate.")
    if "replication_timing" not in source or "firing_rate" not in source:
        raise ValueError("source must contain replication_timing and firing_rate.")
    T = np.asarray(source["replication_timing"], dtype=float)
    if T.ndim != 1:
        raise ValueError("replication_timing must be a one-dimensional array.")
    if T.size == 0:
        raise ValueError("replication_timing must contain at least one value.")
    firing_rate = np.asarray(source["firing_rate"], dtype=float)
    return T, firing_rate


def _spatial_firing_for_speed_fit(firing_rate, n):
    arr = np.asarray(firing_rate, dtype=float)
    if arr.size == 1:
        value = float(arr.reshape(-1)[0])
        if not np.isfinite(value) or value < 0:
            raise ValueError("firing_rate scalar must be finite and >= 0.")
        return value
    if arr.ndim == 1:
        if arr.size != n:
            raise ValueError("firing_rate must match the length of replication_timing.")
        if np.any(~np.isfinite(arr)) or np.any(arr < 0):
            raise ValueError("firing_rate must contain finite values >= 0.")
        return arr.astype(float, copy=False)
    if arr.ndim == 2 and arr.shape[1] == 1 and arr.shape[0] == n:
        arr = arr.reshape(-1)
        if np.any(~np.isfinite(arr)) or np.any(arr < 0):
            raise ValueError("firing_rate must contain finite values >= 0.")
        return arr
    raise ValueError("firing_rate must be scalar, (n,), or (n,1) for spatial speed fitting.")


def _spacetime_firing_for_speed_fit(firing_rate, n):
    arr = np.asarray(firing_rate, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != n or arr.shape[1] <= 1:
        raise ValueError("firing_rate must have shape (positions, time) for v(x,t) fitting.")
    if np.any(~np.isfinite(arr)) or np.any(arr < 0):
        raise ValueError("firing_rate must contain finite values >= 0.")
    return arr.T.astype(float, copy=False)


def _speed_bounds_from_profile(profile, speed_bounds=None):
    profile = np.asarray(profile, dtype=float)
    finite_positive = profile[np.isfinite(profile) & (profile > 0)]
    reference = float(np.nanmedian(finite_positive)) if finite_positive.size else 1.4
    if speed_bounds is None:
        low = max(reference * 1e-3, 1e-12)
        high = max(reference * 1e3, low * 10.0)
    else:
        if len(speed_bounds) != 2:
            raise ValueError("speed_bounds must be a (min_speed, max_speed) pair.")
        low, high = map(float, speed_bounds)
        if not (np.isfinite(low) and np.isfinite(high)) or low <= 0 or high <= low:
            raise ValueError("speed_bounds must satisfy 0 < min_speed < max_speed.")
    return low, high


def _direct_speed_shape_bounds(profile, speed_bounds=None):
    if speed_bounds is not None:
        return _speed_bounds_from_profile(profile, speed_bounds=speed_bounds)

    profile = np.asarray(profile, dtype=float)
    finite_positive = profile[np.isfinite(profile) & (profile > 0)]
    reference = float(np.nanmedian(finite_positive)) if finite_positive.size else 1.4
    low = max(reference * 0.05, 0.01)
    high = min(max(reference * 5.0, low * 10.0), 10.0)
    if high <= low:
        high = low * 10.0
    return low, high


def _optimizer_eps_from_fit_step(fit_step):
    fit_step = float(fit_step)
    if not np.isfinite(fit_step) or fit_step < 0:
        raise ValueError("fit_step must be finite and >= 0.")
    return min(max(fit_step, 1e-6), 1e-2)


def _normalised_fit_axis(values):
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size <= 1:
        return np.zeros(values.size, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size <= 1:
        return np.zeros(values.size, dtype=float)
    vmin = float(np.nanmin(finite))
    vmax = float(np.nanmax(finite))
    span = vmax - vmin
    if not np.isfinite(span) or span <= 0:
        return np.zeros(values.size, dtype=float)
    midpoint = 0.5 * (vmin + vmax)
    return (values - midpoint) / span


def _project_exponential_speed_shape(
    speed_field,
    target_form,
    resolution=1.0,
    time_axis=None,
    speed_bounds=None,
):
    """Project a free speed estimate onto v0 * exp(a_x x + a_t t)."""
    form = _canonical_weibull_target_form(target_form)
    speed = np.asarray(speed_field, dtype=float)

    if form == "space":
        values = speed.reshape(-1)
        x_axis = np.arange(values.size, dtype=float) * float(resolution)
        x_norm = _normalised_fit_axis(x_axis)
        design = np.column_stack((np.ones(values.size), x_norm))
        out_shape = speed.shape
    elif form == "time":
        values = speed.reshape(-1)
        if time_axis is None:
            time_axis = np.arange(values.size, dtype=float)
        t_norm = _normalised_fit_axis(time_axis)
        design = np.column_stack((np.ones(values.size), t_norm))
        out_shape = speed.shape
    elif form == "spacetime":
        if speed.ndim != 2:
            raise ValueError("space-time fork speed must be a two-dimensional field.")
        n_time, n_positions = speed.shape
        if time_axis is None:
            time_axis = np.arange(n_time, dtype=float)
        t_norm = _normalised_fit_axis(time_axis)
        x_axis = np.arange(n_positions, dtype=float) * float(resolution)
        x_norm = _normalised_fit_axis(x_axis)
        tt, xx = np.meshgrid(t_norm, x_norm, indexing="ij")
        design = np.column_stack((
            np.ones(speed.size),
            xx.reshape(-1),
            tt.reshape(-1),
        ))
        values = speed.reshape(-1)
        out_shape = speed.shape
    else:
        raise ValueError("exponential fork-speed target_shape is implemented for v(x), v(t), and v(x,t).")

    ok = np.isfinite(values) & (values > 0) & np.all(np.isfinite(design), axis=1)
    if not np.any(ok):
        raise ValueError("fork-speed field must contain at least one finite positive value.")

    coeffs, *_ = np.linalg.lstsq(design[ok], np.log(values[ok]), rcond=None)
    fitted = np.exp(design @ coeffs).reshape(out_shape)
    low, high = _speed_bounds_from_profile(values[ok], speed_bounds=speed_bounds)
    fitted = np.clip(fitted, low, high)

    info = {
        "target_shape": "exponential",
        "fork_speed_v0": float(np.exp(coeffs[0])),
        "fork_speed_amplitude_t": float(coeffs[-1]),
        "fork_speed_amplitude_x": 0.0,
    }
    if form == "space":
        info["fork_speed_amplitude_x"] = float(coeffs[1])
        info["fork_speed_amplitude_t"] = 0.0
    if form == "spacetime":
        info["fork_speed_amplitude_x"] = float(coeffs[1])
    return _RmapInfoArray(fitted, info=info)


def _weibull_speed_time_basis(time_axis, shape=2.0, delay=0.0):
    time_axis = np.asarray(time_axis, dtype=float).reshape(-1)
    if time_axis.size == 0 or np.any(~np.isfinite(time_axis)):
        raise ValueError("time_grid must contain finite values for Weibull fork-speed targets.")
    if time_axis.size > 1 and np.any(np.diff(time_axis) <= 0):
        raise ValueError("time_grid must be strictly increasing for Weibull fork-speed targets.")
    k = float(shape)
    if not np.isfinite(k) or k <= 0:
        raise ValueError("weibull_shape must be finite and positive.")
    d = float(np.asarray(delay, dtype=float).reshape(-1)[0])
    if not np.isfinite(d):
        raise ValueError("weibull_delay must be finite for fork-speed targets.")
    if time_axis.size > 1:
        dt = float(np.nanmedian(np.diff(time_axis)))
    else:
        dt = 1.0
    if not np.isfinite(dt) or dt <= 0:
        dt = 1.0
    elapsed = np.maximum(time_axis - d, 0.0) + 0.5 * dt
    basis = k * np.power(np.maximum(elapsed, 1e-12), k - 1.0)
    scale = float(np.nanmedian(basis[np.isfinite(basis) & (basis > 0)]))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    return basis / scale


def _project_weibull_speed_shape(
    speed_field,
    target_form,
    resolution=1.0,
    time_axis=None,
    speed_bounds=None,
    weibull_shape=2.0,
    weibull_delay=0.0,
):
    """Project fork speed through a Weibull-shaped slowness profile.

    The fitted object is movement time per kb, s = 1 / v. For v(x,t),
    s(x,t) = beta(x) * h_k(t), where h_k is a normalised Weibull hazard
    shape and beta(x) is fitted independently at each genomic position.
    """
    form = _canonical_weibull_target_form(target_form)
    speed = np.asarray(speed_field, dtype=float)
    low, high = _speed_bounds_from_profile(speed, speed_bounds=speed_bounds)

    if np.any(~np.isfinite(speed)) or np.any(speed <= 0):
        raise ValueError("fork-speed field must contain finite positive values.")

    if form == "space":
        fitted = np.clip(speed.astype(float, copy=True), low, high)
        beta = 1.0 / np.maximum(fitted, 1e-12)
        info = {
            "target_shape": "weibull",
            "fork_speed_shape": "site-specific v(x)",
            "fork_slowness_shape": "site-specific beta(x)",
            "weibull_shape": float(weibull_shape),
            "weibull_delay": float(np.asarray(weibull_delay, dtype=float).reshape(-1)[0]),
            "fork_slowness_beta": beta,
        }
        return _RmapInfoArray(fitted, info=info)

    if form == "time":
        values = speed.reshape(-1)
        if time_axis is None:
            time_axis = np.arange(values.size, dtype=float)
        basis = _weibull_speed_time_basis(time_axis, shape=weibull_shape, delay=weibull_delay)
        slowness = 1.0 / np.maximum(values, 1e-12)
        ok = np.isfinite(slowness) & np.isfinite(basis) & (basis > 0)
        beta = float(np.dot(slowness[ok], basis[ok]) / max(np.dot(basis[ok], basis[ok]), 1e-30))
        beta = max(beta, 1e-12)
        fitted = 1.0 / np.maximum(beta * basis, 1e-12)
        fitted = np.clip(fitted, low, high).reshape(speed.shape)
        info = {
            "target_shape": "weibull",
            "fork_speed_shape": "1 / (beta * h_k(t))",
            "fork_slowness_shape": "beta * h_k(t)",
            "weibull_shape": float(weibull_shape),
            "weibull_delay": float(np.asarray(weibull_delay, dtype=float).reshape(-1)[0]),
            "fork_slowness_beta": beta,
        }
        return _RmapInfoArray(fitted, info=info)

    if form == "spacetime":
        if speed.ndim != 2:
            raise ValueError("space-time fork speed must be a two-dimensional field.")
        n_time, n_positions = speed.shape
        if time_axis is None:
            time_axis = np.arange(n_time, dtype=float)
        basis = _weibull_speed_time_basis(time_axis, shape=weibull_shape, delay=weibull_delay)
        if basis.size != n_time:
            raise ValueError("time_grid must match the time axis of the speed field.")
        slowness = 1.0 / np.maximum(speed, 1e-12)
        denom = max(float(np.dot(basis, basis)), 1e-30)
        beta = np.nansum(slowness * basis[:, None], axis=0) / denom
        beta = np.maximum(np.where(np.isfinite(beta), beta, np.nanmedian(beta)), 1e-12)
        fitted_slowness = beta[None, :] * basis[:, None]
        fitted = 1.0 / np.maximum(fitted_slowness, 1e-12)
        fitted = np.clip(fitted, low, high)
        info = {
            "target_shape": "weibull",
            "fork_speed_shape": "1 / (beta(x) * h_k(t))",
            "fork_slowness_shape": "beta(x) * h_k(t)",
            "weibull_shape": float(weibull_shape),
            "weibull_delay": float(np.asarray(weibull_delay, dtype=float).reshape(-1)[0]),
            "fork_slowness_beta": beta,
        }
        return _RmapInfoArray(fitted, info=info)

    raise ValueError("Weibull fork-speed target_shape is implemented for v(x), v(t), and v(x,t).")


def _retag_speed_shape(values, target_shape, **info_updates):
    info = dict(getattr(values, "_rmap_info", {}) or {})
    info.update(info_updates)
    info["target_shape"] = target_shape
    return _RmapInfoArray(np.asarray(values, dtype=float), info=info)


def _project_best_spacetime_speed_shape(speed_field, time_axis=None, speed_bounds=None):
    speed = np.asarray(speed_field, dtype=float)
    if speed.ndim != 2:
        raise ValueError("space-time fork speed must be a two-dimensional field.")
    low, high = _speed_bounds_from_profile(speed, speed_bounds=speed_bounds)
    spatial = np.nanmedian(speed, axis=0)
    positive = np.isfinite(spatial) & (spatial > 0)
    reference = float(np.nanmedian(spatial[positive])) if np.any(positive) else _initial_positive_speed_value(speed)
    spatial = np.where(positive, spatial, reference)
    spatial = np.clip(spatial, low, high)
    scale = float(np.nanmedian(spatial[spatial > 0])) if np.any(spatial > 0) else reference
    if not np.isfinite(scale) or scale <= 0:
        scale = reference
    denom = np.maximum(spatial[None, :] / max(scale, 1e-12), 1e-12)
    temporal = np.nanmedian(speed / denom, axis=1)
    temporal = np.where(np.isfinite(temporal) & (temporal > 0), temporal, scale)
    fitted = np.outer(temporal, spatial / max(scale, 1e-12))
    fitted = np.clip(fitted, low, high)
    info = {
        "target_shape": "best",
        "fork_speed_shape": "factorized v_x(x) * q(t)",
        "fork_speed_component_x": "site-specific v_x(x)",
        "fork_speed_component_t": "smooth multiplicative q(t)",
        "fork_speed_v0": float(scale),
        "fork_speed_temporal_factor": temporal / max(scale, 1e-12),
        "fork_speed_bounds": (float(low), float(high)),
    }
    return _RmapInfoArray(fitted, info=info)


def _speed_shape_fit_grid(n_positions, max_positions):
    """Retain edge context in the forward grid; crop only the scored values."""
    stride = max(1, int(math.ceil(n_positions / max_positions)))
    indices = np.arange(0, n_positions, stride, dtype=int)
    if indices[-1] != n_positions - 1:
        indices = np.append(indices, n_positions - 1)
    return indices, stride


def _spacetime_speed_timing_fit_info(
    speed_spacetime,
    observed_timing,
    firing_rate,
    resolution,
    resolution_time,
    time_axis,
    T0=None,
    refine_edge_kb=1000,
    perQ=False,
):
    if perQ:
        return {}
    speed = np.asarray(speed_spacetime, dtype=float)
    T = np.asarray(observed_timing, dtype=float)
    time_axis = np.asarray(time_axis, dtype=float)
    if speed.ndim != 2 or speed.shape[1] != T.size:
        return {}

    try:
        offset = 0.0 if T0 is None else float(T0)
        crop = _refinement_crop_mask(T.size, resolution, perQ=perQ, refine_edge_kb=refine_edge_kb)
        interpolated = not _use_exact_full_timing_diagnostic(T.size, time_axis.size)
        if not interpolated:
            try:
                fraction = map_replicated_fraction_xt(
                    firing_rate, fork_speed=speed.T, resolution=resolution,
                    time_grid=time_axis, n_bins=T.size,
                )
                predicted_full = np.trapezoid(1.0 - fraction, time_axis, axis=1) + offset
                scored = crop & np.isfinite(T) & np.isfinite(predicted_full)
                fit_mse = mean_squared_error(T[scored], predicted_full[scored])
            except (MemoryError, ValueError):
                interpolated = True
        if interpolated:
            fit_idx, stride = _speed_shape_fit_grid(T.size, 1500)
            fraction_fit = map_replicated_fraction_xt(
                _subset_firing_positions(firing_rate, fit_idx, T.size),
                fork_speed=speed.T[fit_idx],
                resolution=float(resolution) * stride,
                resolution_time=resolution_time,
                time_grid=time_axis,
                n_bins=fit_idx.size,
                perQ=False,
            )
            predicted_fit = np.trapezoid(
                np.clip(1.0 - np.asarray(fraction_fit, dtype=float), 0.0, 1.0),
                time_axis,
                axis=1,
            ) + offset
            scored = crop[fit_idx] & np.isfinite(T[fit_idx]) & np.isfinite(predicted_fit)
            fit_mse = mean_squared_error(T[fit_idx][scored], predicted_fit[scored])
            predicted_full = _expand_sampled_timing_diagnostic(predicted_fit, fit_idx, T.size)
    except Exception:
        return {}

    if not np.isfinite(fit_mse):
        return {}
    info = {
        "fit_mse": float(fit_mse),
        "fit_mse_description": (
            "Deterministic timing MSE on scored positions for the returned v(x,t); "
            "it is not the stochastic rsim MSE."
        ),
        "replication_timing_fit_interpolated": interpolated,
    }
    if predicted_full is not None:
        info["replication_timing_fit"] = predicted_full
        info["residual"] = predicted_full - T
    return info


def _fit_exponential_speed_shape(
    observed_timing,
    firing_rate,
    target_form,
    resolution=1.0,
    resolution_time=5.0,
    time_axis=None,
    T0=None,
    initial_fork_speed=1.4,
    speed_bounds=None,
    fit_step=0.25,
    maxiter=10,
    sim_number=100,
    refine_edge_kb=1000,
    refine_seed=None,
    refine_verbose=None,
    perQ=False,
    error=None,
):
    """Fit a low-dimensional exponential fork-slowness surface.

    The fitted positive field is represented as
    v(x,t) = v0 * exp(b_x X + b_t T), equivalently
    s(x,t) = 1 / v(x,t) = s0 * exp(-b_x X - b_t T).
    Timing is scored deterministically by integrating survival from the
    analytical replicated-fraction map, not by stochastic simulation.
    """
    form = _canonical_weibull_target_form(target_form)
    if form not in {"time", "spacetime"}:
        raise ValueError("direct exponential fork-speed fitting is implemented for v(t) and v(x,t).")
    if perQ:
        raise NotImplementedError("direct exponential fork-speed fitting currently supports linear geometry only.")

    T = np.asarray(observed_timing, dtype=float)
    if T.ndim != 1 or T.size == 0:
        raise ValueError("replication_timing must be a non-empty one-dimensional array.")

    offset = 0.0 if T0 is None else float(T0)
    if not np.isfinite(offset):
        raise ValueError("T0 must be finite when supplied.")
    target_timing = T - offset
    if time_axis is None:
        time_axis = _survival_time_axis_from_timing(
            np.maximum(target_timing, 0.0),
            resolution_time=resolution_time,
        )
    time_axis = np.asarray(time_axis, dtype=float)
    if time_axis.ndim != 1 or time_axis.size == 0:
        raise ValueError("time_grid must be a non-empty one-dimensional array.")

    n_positions = T.size
    x_axis = np.arange(n_positions, dtype=float) * float(resolution)
    t_norm = _normalised_fit_axis(time_axis)
    x_norm = _normalised_fit_axis(x_axis)

    reference_speed = _initial_positive_speed_value(initial_fork_speed)
    low, high = _direct_speed_shape_bounds([reference_speed], speed_bounds=speed_bounds)
    error = _normalise_error_threshold(error)
    finite_diff_step = _optimizer_eps_from_fit_step(fit_step) * 1e-3
    maxiter = int(maxiter)

    firing = np.asarray(firing_rate, dtype=float)
    if form == "spacetime":
        if firing.ndim != 2 or firing.shape[0] != n_positions:
            raise ValueError("firing_rate must have shape (positions, time) for direct v(x,t) fitting.")
    elif form == "time" and firing.ndim == 2 and firing.shape[0] == n_positions:
        pass
    elif form == "time" and firing.ndim == 2 and firing.shape[1] == n_positions:
        firing = firing.T
    elif firing.ndim == 2 and firing.shape[0] == 1:
        pass
    elif firing.ndim == 1 and firing.size == n_positions:
        pass
    elif firing.size == 1:
        pass
    else:
        raise ValueError("firing_rate must be scalar, x-dependent, time-dependent, or space-time dependent.")

    crop = _refinement_crop_mask(n_positions, resolution, perQ=perQ, refine_edge_kb=refine_edge_kb)
    valid_full = crop & np.isfinite(T) & np.isfinite(target_timing) & (target_timing > 0)
    if not np.any(valid_full):
        raise ValueError("replication_timing must contain finite positive values after edge filtering.")

    fit_idx, stride = _speed_shape_fit_grid(n_positions, 1500)
    score_mask = valid_full[fit_idx]
    T_fit = target_timing[fit_idx]
    resolution_fit = float(resolution) * float(stride)
    x_fit_axis = x_axis[fit_idx]
    x_fit_norm = _normalised_fit_axis(x_fit_axis)

    if firing.ndim == 2 and firing.shape[0] == n_positions:
        firing_fit = firing[fit_idx]
    elif firing.ndim == 1 and firing.size == n_positions:
        firing_fit = firing[fit_idx]
    else:
        firing_fit = firing

    def make_speed_internal(values, full=True, mse=None, predicted=None):
        values = np.asarray(values, dtype=float)
        log_v0 = float(np.clip(values[0], math.log(low), math.log(high)))
        b_t = float(np.clip(values[-1], -8.0, 8.0))
        if form == "time":
            b_x = 0.0
            speed = np.exp(log_v0 + b_t * t_norm)[None, :]
        else:
            b_x = float(np.clip(values[1], -8.0, 8.0))
            x_values = x_norm if full else x_fit_norm
            speed = np.exp(log_v0 + b_x * x_values[None, :] + b_t * t_norm[:, None])
        speed = np.clip(speed, low, high)
        info = {
            "target_shape": "exponential",
            "speed_form": form,
            "time_grid": time_axis.copy(),
            "speed_shape": "v0 * exp(b_x * X + b_t * T)",
            "slowness_shape": "s0 * exp(-b_x * X - b_t * T)",
            "fork_speed_v0": float(np.exp(log_v0)),
            "fork_speed_amplitude_x": b_x,
            "fork_speed_amplitude_t": b_t,
            "fork_slowness_s0": float(np.exp(-log_v0)),
            "fork_slowness_amplitude_x": -b_x,
            "fork_slowness_amplitude_t": -b_t,
            "fork_speed_bounds": (float(low), float(high)),
            "fit_mse": None if mse is None or not np.isfinite(mse) else float(mse),
        }
        if predicted is not None:
            info["replication_timing_fit"] = predicted
            info["residual"] = predicted - T
        return _RmapInfoArray(speed, info=info)

    def predict_timing(values, *, full=False):
        speed = make_speed_internal(values, full=full)
        fork_speed = speed if form == "time" else speed.T
        current_firing = firing if full else firing_fit
        current_resolution = float(resolution) if full else resolution_fit
        current_n = n_positions if full else fit_idx.size
        fraction = map_replicated_fraction_xt(
            current_firing,
            fork_speed=fork_speed,
            resolution=current_resolution,
            resolution_time=resolution_time,
            time_grid=time_axis,
            n_bins=current_n,
            perQ=False,
        )
        survival = np.clip(1.0 - np.asarray(fraction, dtype=float), 0.0, 1.0)
        predicted = np.trapezoid(survival, time_axis, axis=1)
        return predicted

    best = {"loss": np.inf, "mse": np.inf, "params": None, "predicted": None}
    verbose = bool(refine_verbose) if refine_verbose is not None else False
    start_time = time.time()
    calls = 0
    iterations = 0
    history = []
    maxiter_display = max(1, maxiter)

    params0 = np.zeros(2 if form == "time" else 3, dtype=float)
    params0[0] = math.log(np.clip(reference_speed, low, high))
    bounds = [(math.log(low), math.log(high))]
    if form == "spacetime":
        bounds.append((-8.0, 8.0))
    bounds.append((-8.0, 8.0))

    def objective(values):
        nonlocal calls
        check_cancelled()
        values = np.asarray(values, dtype=float)
        if np.any(~np.isfinite(values)):
            return 1e30
        try:
            predicted = predict_timing(values)
        except Exception:
            return 1e30
        valid = score_mask & np.isfinite(predicted)
        if not np.any(valid):
            mse = np.inf
            loss = 1e30
        else:
            residual = predicted[valid] - T_fit[valid]
            mse = mean_squared_error(T_fit[valid], predicted[valid])
            loss = 0.5 * float(np.dot(residual, residual))
        calls += 1
        if loss < best["loss"]:
            best["loss"] = loss
            best["mse"] = mse
            best["params"] = values.copy()
            best["predicted"] = predicted.copy()
        if calls == 1 and _fit_progress_enabled(verbose):
            etaferr(it=0, maxiter=maxiter_display, start_time=start_time, err=best["mse"], done=False)
        if _error_reached(mse, error):
            raise _FitConverged
        return loss

    def iteration_complete(_parameters):
        nonlocal iterations
        iterations += 1
        history.append(float(best["mse"]))
        if _fit_progress_enabled(verbose):
            etaferr(
                it=iterations, maxiter=maxiter_display, start_time=start_time,
                err=best["mse"], done=iterations >= maxiter_display,
            )

    def objective_gradient(values):
        check_cancelled()
        values = np.asarray(values, dtype=float)
        predicted = predict_timing(values)
        columns = []
        for index in range(values.size):
            delta = np.zeros_like(values)
            delta[index] = finite_diff_step
            columns.append((predict_timing(values + delta) - predict_timing(values - delta)) / (2 * finite_diff_step))
        jacobian = np.column_stack(columns)
        valid = score_mask & np.isfinite(predicted) & np.all(np.isfinite(jacobian), axis=1)
        # Differentiate the forward prediction before applying the timing
        # offset: differencing the squared loss amplifies rounding in T0.
        return jacobian[valid].T @ (predicted[valid] - T_fit[valid])

    try:
        if maxiter <= 0:
            objective(params0)
        else:
            minimize(
                objective,
                params0,
                method="L-BFGS-B",
                bounds=bounds,
                jac=objective_gradient,
                callback=iteration_complete,
                options={
                    "maxiter": maxiter,
                    "maxfun": (maxiter + 1) * (params0.size + 1) * 10,
                    "ftol": 1e-9,
                    "maxls": 10,
                },
            )
    except _FitConverged:
        pass

    if _fit_progress_enabled(verbose) and iterations < maxiter_display:
        etaferr(
            it=maxiter_display, maxiter=maxiter_display, start_time=start_time,
            err=best["mse"], done=True,
        )

    params = best["params"] if best["params"] is not None else params0
    full_speed = make_speed_internal(params, full=True, mse=best["mse"])
    interpolate_diagnostic = not _use_exact_full_timing_diagnostic(
        n_positions,
        time_axis.size,
    )
    if not interpolate_diagnostic:
        try:
            full_predicted = predict_timing(params, full=True)
        except Exception:
            interpolate_diagnostic = True
    if interpolate_diagnostic:
        retained_prediction = best["predicted"]
        if retained_prediction is None:
            retained_prediction = predict_timing(params, full=False)
        full_predicted = _expand_sampled_timing_diagnostic(
            retained_prediction,
            fit_idx,
            n_positions,
        )
    full_predicted = full_predicted + offset
    full_speed._rmap_info["replication_timing_fit"] = full_predicted
    full_speed._rmap_info["residual"] = full_predicted - T
    full_speed._rmap_info["replication_timing_fit_interpolated"] = interpolate_diagnostic
    full_speed._rmap_info["fit_history"] = np.asarray(history or [best["mse"]], dtype=float)
    return full_speed


def _fit_weibull_speed_shape(
    observed_timing,
    firing_rate,
    target_form,
    resolution=1.0,
    resolution_time=5.0,
    time_axis=None,
    T0=None,
    initial_fork_speed=1.4,
    speed_bounds=None,
    fit_step=0.25,
    maxiter=10,
    refine_edge_kb=1000,
    refine_verbose=None,
    perQ=False,
    error=None,
    weibull_shape=2.0,
    weibull_delay=0.0,
):
    """Fit a Weibull-shaped fork-slowness profile for v(t)."""
    form = _canonical_weibull_target_form(target_form)
    if form != "time":
        raise ValueError("direct Weibull fork-speed fitting is implemented for v(t).")
    if perQ:
        raise NotImplementedError("direct Weibull fork-speed fitting currently supports linear geometry only.")

    T = np.asarray(observed_timing, dtype=float)
    if T.ndim != 1 or T.size == 0:
        raise ValueError("replication_timing must be a non-empty one-dimensional array.")

    offset = 0.0 if T0 is None else float(T0)
    if not np.isfinite(offset):
        raise ValueError("T0 must be finite when supplied.")
    target_timing = T - offset
    if time_axis is None:
        time_axis = _survival_time_axis_from_timing(
            np.maximum(target_timing, 0.0),
            resolution_time=resolution_time,
        )
    time_axis = np.asarray(time_axis, dtype=float)
    if time_axis.ndim != 1 or time_axis.size == 0:
        raise ValueError("time_grid must be a non-empty one-dimensional array.")

    firing = np.asarray(firing_rate, dtype=float)
    n_positions = T.size
    if firing.ndim == 1 and firing.size == n_positions:
        pass
    elif firing.ndim == 2 and firing.shape == (n_positions, 1):
        firing = firing.reshape(-1)
    elif firing.ndim == 2 and firing.shape[0] == 1:
        pass
    elif firing.size == 1:
        pass
    else:
        raise ValueError("firing_rate must be scalar, x-dependent, or time-dependent for v(t) fitting.")

    crop = _refinement_crop_mask(n_positions, resolution, perQ=perQ, refine_edge_kb=refine_edge_kb)
    valid_full = crop & np.isfinite(T) & np.isfinite(target_timing) & (target_timing > 0)
    if not np.any(valid_full):
        raise ValueError("replication_timing must contain finite positive values after edge filtering.")

    max_fit_positions = 1500
    valid_idx = np.flatnonzero(valid_full)
    stride = max(1, int(math.ceil(valid_idx.size / max_fit_positions)))
    fit_idx = valid_idx[::stride]
    if fit_idx[-1] != valid_idx[-1]:
        fit_idx = np.concatenate([fit_idx, valid_idx[-1:]])
    resolution_fit = float(resolution) * float(stride)
    T_fit = T[fit_idx]

    if firing.ndim == 1 and firing.size == n_positions:
        firing_fit = firing[fit_idx]
    elif firing.ndim == 2 and firing.shape[0] == n_positions:
        firing_fit = firing[fit_idx]
    else:
        firing_fit = firing

    reference_speed = _initial_positive_speed_value(initial_fork_speed)
    low, high = _direct_speed_shape_bounds([reference_speed], speed_bounds=speed_bounds)
    error = _normalise_error_threshold(error)
    finite_diff_step = _optimizer_eps_from_fit_step(fit_step)

    basis = _weibull_speed_time_basis(
        time_axis,
        shape=weibull_shape,
        delay=weibull_delay,
    )
    log_beta0 = math.log(max(1.0 / reference_speed, 1e-12))
    bounds = [(math.log(max(1.0 / high, 1e-30)), math.log(max(1.0 / low, 1e-30)))]

    def make_speed_internal(values, mse=None, predicted=None):
        log_beta = float(np.clip(np.asarray(values, dtype=float).reshape(-1)[0], bounds[0][0], bounds[0][1]))
        beta = float(np.exp(log_beta))
        speed = 1.0 / np.maximum(beta * basis, 1e-12)
        speed = np.clip(speed, low, high).reshape(1, -1)
        info = {
            "target_shape": "weibull",
            "fork_speed_shape": "1 / (beta * h_k(t))",
            "fork_slowness_shape": "beta * h_k(t)",
            "weibull_shape": float(weibull_shape),
            "weibull_delay": float(np.asarray(weibull_delay, dtype=float).reshape(-1)[0]),
            "fork_slowness_beta": beta,
            "fork_speed_bounds": (float(low), float(high)),
            "fit_mse": None if mse is None or not np.isfinite(mse) else float(mse),
        }
        if predicted is not None:
            info["replication_timing_fit"] = predicted
            info["residual"] = predicted - T
        return _RmapInfoArray(speed, info=info)

    def predict_timing(values, full=False):
        speed = make_speed_internal(values)
        current_firing = firing if full else firing_fit
        n_current = n_positions if full else fit_idx.size
        res_current = float(resolution) if full else resolution_fit
        fraction = map_replicated_fraction_xt(
            current_firing,
            fork_speed=speed,
            resolution=res_current,
            resolution_time=resolution_time,
            time_grid=time_axis,
            n_bins=n_current,
            perQ=False,
        )
        survival = np.clip(1.0 - np.asarray(fraction, dtype=float), 0.0, 1.0)
        return np.trapezoid(survival, time_axis, axis=1) + offset

    best = {"loss": np.inf, "mse": np.inf, "params": None}
    verbose = bool(refine_verbose) if refine_verbose is not None else False
    start_time = time.time()
    calls = 0
    maxiter = int(maxiter)
    maxiter_display = max(1, maxiter)
    stop_at_call_budget = maxiter > 0

    def objective(values):
        nonlocal calls
        try:
            predicted = predict_timing(values, full=False)
        except Exception:
            return 1e30
        valid = np.isfinite(T_fit) & np.isfinite(predicted)
        if not np.any(valid):
            mse = np.inf
            loss = 1e30
        else:
            residual = predicted[valid] - T_fit[valid]
            mse = mean_squared_error(T_fit[valid], predicted[valid])
            loss = 0.5 * float(np.dot(residual, residual))
        calls += 1
        if loss < best["loss"]:
            best["loss"] = loss
            best["mse"] = mse
            best["params"] = np.asarray(values, dtype=float).copy()
        done = calls >= maxiter_display or _error_reached(mse, error)
        if _fit_progress_enabled(verbose) and calls <= maxiter_display:
            etaferr(
                it=maxiter_display if done else min(calls, maxiter_display),
                maxiter=maxiter_display,
                start_time=start_time,
                err=best["mse"] if np.isfinite(best["mse"]) else mse,
                done=done,
            )
        if (stop_at_call_budget and calls >= maxiter_display) or _error_reached(mse, error):
            raise _FitConverged
        return loss

    params0 = np.array([log_beta0], dtype=float)
    if maxiter <= 0:
        objective(params0)
    else:
        try:
            minimize(
                objective,
                params0,
                method="L-BFGS-B",
                bounds=bounds,
                options={
                    "maxiter": maxiter,
                    "maxfun": maxiter,
                    "ftol": 1e-9,
                    "eps": finite_diff_step,
                    "maxls": 10,
                },
            )
        except _FitConverged:
            pass

    params = best["params"] if best["params"] is not None else params0
    predicted_full = None
    try:
        predicted_fit = predict_timing(params, full=False)
        predicted_full = np.full(n_positions, np.nan, dtype=float)
        predicted_full[fit_idx] = predicted_fit
    except Exception:
        pass
    return make_speed_internal(params, mse=best["mse"], predicted=predicted_full)


def _space_time_firing_for_separable_speed(firing_rate, n_positions, n_time):
    arr = np.asarray(firing_rate, dtype=float)
    if arr.size == 1:
        value = float(arr.reshape(-1)[0])
        return np.full((n_time, n_positions), value, dtype=float)
    if arr.ndim == 1 and arr.size == n_positions:
        return np.tile(arr.reshape(1, -1), (n_time, 1))
    if arr.ndim == 2 and arr.shape == (n_positions, n_time):
        return arr.T.astype(float, copy=False)
    if arr.ndim == 2 and arr.shape == (1, n_time):
        return np.repeat(arr.reshape(n_time, 1), n_positions, axis=1)
    if arr.ndim == 2 and arr.shape == (n_time, n_positions):
        return arr.astype(float, copy=False)
    raise ValueError("firing_rate shape is not compatible with separable fork-speed fitting.")


def _cumtrapz_rows_local(values, axis):
    values = np.asarray(values, dtype=float)
    axis = np.asarray(axis, dtype=float)
    out = np.zeros_like(values, dtype=float)
    if axis.size > 1:
        widths = np.diff(axis)
        out[:, 1:] = np.cumsum(0.5 * (values[:, :-1] + values[:, 1:]) * widths[None, :], axis=1)
    return out


def _integrate_cdf_between_axis(cdf_row, axis, left, right):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    a = np.clip(np.minimum(left, right), axis[0], axis[-1])
    b = np.clip(np.maximum(left, right), axis[0], axis[-1])
    return np.maximum(np.interp(b, axis, cdf_row) - np.interp(a, axis, cdf_row), 0.0)


def _predict_timing_separable_speed(
    firing_rate,
    spatial_speed,
    temporal_factor,
    observed_timing,
    time_axis,
    resolution,
    offset=0.0,
    exact=False,
):
    spatial_speed = np.asarray(spatial_speed, dtype=float).reshape(-1)
    temporal_factor = np.asarray(temporal_factor, dtype=float).reshape(-1)
    time_axis = np.asarray(time_axis, dtype=float).reshape(-1)
    n_positions = spatial_speed.size
    n_time = time_axis.size
    if n_positions == 0 or n_time == 0:
        raise ValueError("spatial_speed and time_axis must be non-empty.")
    if temporal_factor.size != n_time:
        raise ValueError("temporal_factor must match time_axis.")
    if np.any(~np.isfinite(spatial_speed)) or np.any(spatial_speed <= 0):
        raise ValueError("spatial_speed must be finite and positive.")
    if np.any(~np.isfinite(temporal_factor)) or np.any(temporal_factor <= 0):
        raise ValueError("temporal_factor must be finite and positive.")

    if exact:
        fraction = map_replicated_fraction_xt(
            firing_rate, fork_speed=np.outer(spatial_speed, temporal_factor),
            resolution=resolution, time_grid=time_axis, n_bins=n_positions,
        )
        return np.trapezoid(1.0 - fraction, time_axis, axis=1) + float(offset)

    I = _space_time_firing_for_separable_speed(firing_rate, n_positions, n_time)
    if np.any(~np.isfinite(I)) or np.any(I < 0):
        raise ValueError("firing_rate must contain finite values >= 0.")

    x = np.arange(n_positions, dtype=float) * float(resolution)
    phi = np.zeros(n_positions, dtype=float)
    if n_positions > 1:
        inv_speed = 1.0 / np.maximum(spatial_speed, 1e-12)
        dx = np.diff(x)
        phi[1:] = np.cumsum(0.5 * (inv_speed[:-1] + inv_speed[1:]) * dx)

    C = np.zeros(n_time, dtype=float)
    if n_time > 1:
        dt = np.diff(time_axis)
        C[1:] = np.cumsum(0.5 * (temporal_factor[:-1] + temporal_factor[1:]) * dt)

    I_mid = 0.5 * (I[:-1] + I[1:]) if n_time > 1 else np.zeros((0, n_positions), dtype=float)
    cdf = _cumtrapz_rows_local(I_mid, x)
    A = np.zeros((n_time, n_positions), dtype=float)
    for n in range(1, n_time):
        total = np.zeros(n_positions, dtype=float)
        for k in range(n):
            dt = time_axis[k + 1] - time_axis[k]
            radius = max(C[n] - 0.5 * (C[k] + C[k + 1]), 0.0)
            left_phi = np.clip(phi - radius, phi[0], phi[-1])
            right_phi = np.clip(phi + radius, phi[0], phi[-1])
            left_x = np.interp(left_phi, phi, x)
            right_x = np.interp(right_phi, phi, x)
            total += dt * _integrate_cdf_between_axis(cdf[k], x, left_x, right_x)
        A[n] = total
    survival = np.exp(-np.maximum(A, 0.0))
    return np.trapezoid(survival, time_axis, axis=0) + float(offset)


def _time_axis_for_speed_fit(T, firing_rate, resolution_time=5.0, time_grid=None, T0=None):
    if time_grid is not None:
        axis = np.asarray(time_grid, dtype=float)
        if axis.ndim != 1 or axis.size == 0:
            raise ValueError("time_grid must be a non-empty one-dimensional array.")
        return axis
    resolution_time = float(resolution_time)
    if not np.isfinite(resolution_time) or resolution_time <= 0:
        raise ValueError("resolution_t must be a finite positive value.")
    n_positions = np.asarray(T, dtype=float).size
    arr = np.asarray(firing_rate, dtype=float)
    if arr.ndim == 2:
        if arr.shape[0] == n_positions and arr.shape[1] > 1:
            return np.arange(arr.shape[1], dtype=float) * resolution_time
        if arr.shape[0] == 1 and arr.shape[1] > 1:
            return np.arange(arr.shape[1], dtype=float) * resolution_time
        if arr.shape[1] == n_positions and arr.shape[0] > 1:
            return np.arange(arr.shape[0], dtype=float) * resolution_time
    offset = 0.0 if T0 is None else float(T0)
    if not np.isfinite(offset):
        raise ValueError("T0 must be finite when supplied.")
    return _time_axis_from_timing(
        np.maximum(np.asarray(T, dtype=float) - offset, 0.0),
        resolution_time=resolution_time,
    )


def _subset_firing_positions(firing_rate, fit_idx, n_positions):
    arr = np.asarray(firing_rate, dtype=float)
    if arr.size == 1:
        return float(arr.reshape(-1)[0])
    if arr.ndim == 1:
        return arr[fit_idx] if arr.size == n_positions else arr
    if arr.ndim == 2:
        if arr.shape[0] == 1:
            return arr
        if arr.shape[1] == 1 and arr.shape[0] == n_positions:
            return arr[fit_idx]
        if arr.shape[0] == n_positions:
            return arr[fit_idx]
        if arr.shape[1] == n_positions:
            return arr[:, fit_idx].T
    return arr


def _speed_fit_indices(T, resolution, perQ=False, refine_edge_kb=1000, max_fit_positions=800):
    T = np.asarray(T, dtype=float)
    crop = _refinement_crop_mask(T.size, resolution, perQ=perQ, refine_edge_kb=refine_edge_kb)
    valid_idx = np.flatnonzero(crop & np.isfinite(T) & (T > 0))
    if valid_idx.size == 0:
        raise ValueError("replication_timing must contain finite positive values after edge filtering.")
    stride = max(1, int(math.ceil(valid_idx.size / int(max_fit_positions))))
    fit_idx = valid_idx[::stride]
    if fit_idx[-1] != valid_idx[-1]:
        fit_idx = np.concatenate([fit_idx, valid_idx[-1:]])
    return fit_idx, stride


def _fit_constant_speed_with_general_firing(
    observed_timing,
    firing_rate,
    resolution=1.0,
    resolution_time=5.0,
    time_grid=None,
    T0=None,
    fit_step=0.6,
    maxiter=10,
    speed_bounds=None,
    initial_fork_speed=1.4,
    refine_edge_kb=1000,
    perQ=False,
    error=None,
    verbose=True,
    print_every=1,
):
    if perQ:
        raise NotImplementedError("deterministic fork-speed fitting currently supports linear geometry only.")
    T = np.asarray(observed_timing, dtype=float)
    offset = 0.0 if T0 is None else float(T0)
    if not np.isfinite(offset):
        raise ValueError("T0 must be finite when supplied.")
    relative_timing = T - offset
    fit_idx, stride = _speed_fit_indices(
        relative_timing,
        resolution,
        perQ=perQ,
        refine_edge_kb=refine_edge_kb,
    )
    T_fit = T[fit_idx]
    firing_fit = _subset_firing_positions(firing_rate, fit_idx, T.size)
    time_axis = _time_axis_for_speed_fit(
        T,
        firing_rate,
        resolution_time=resolution_time,
        time_grid=time_grid,
        T0=T0,
    )
    resolution_fit = float(resolution) * float(stride)

    speed = _initial_positive_speed_value(initial_fork_speed)
    low, high = _direct_speed_shape_bounds([speed], speed_bounds=speed_bounds)
    speed = float(np.clip(speed, low, high))
    maxiter = int(maxiter)
    fit_step = float(fit_step)
    if not np.isfinite(fit_step) or fit_step < 0:
        raise ValueError("fit_step must be finite and >= 0.")
    error = _normalise_error_threshold(error)
    print_every = max(1, int(print_every))
    best_speed = speed
    best_mse = np.inf
    best_timing_fit = None
    start_time = time.time()

    for it in range(1, max(1, maxiter) + 1):
        predicted = _predict_timing_separable_speed(
            firing_fit,
            np.full(fit_idx.size, speed, dtype=float),
            np.ones(time_axis.size, dtype=float),
            T_fit,
            time_axis,
            resolution_fit,
            offset=offset,
        )
        valid = np.isfinite(T_fit) & np.isfinite(predicted) & (T_fit > 0) & (predicted > 0)
        if not np.any(valid):
            break
        mse = mean_squared_error(T_fit[valid], predicted[valid])
        if mse < best_mse:
            best_mse = mse
            best_speed = speed
            best_timing_fit = predicted.copy()
        done = it == max(1, maxiter) or _error_reached(mse, error)
        if _fit_progress_enabled(verbose) and (it % print_every == 0 or done):
            etaferr(it=it, maxiter=max(1, maxiter), start_time=start_time, err=mse, done=done)
        if done or maxiter <= 0:
            break
        ratio = float(np.nanmedian(predicted[valid] / T_fit[valid]))
        if not np.isfinite(ratio) or ratio <= 0:
            break
        speed = float(np.clip(speed * ratio ** fit_step, low, high))

    info = {
        "target_shape": "best",
        "fork_speed_shape": "constant v0",
        "fit_mse": None if not np.isfinite(best_mse) else float(best_mse),
        "fit_mse_description": (
            "Deterministic timing MSE on cropped/subsampled positions using the "
            "supplied firing rate."
        ),
    }
    interpolate_diagnostic = not _use_exact_full_timing_diagnostic(T.size, time_axis.size)
    if not interpolate_diagnostic:
        try:
            predicted_full = _predict_timing_separable_speed(
                firing_rate,
                np.full(T.size, best_speed, dtype=float),
                np.ones(time_axis.size, dtype=float),
                T,
                time_axis,
                resolution,
                offset=offset,
            )
        except Exception:
            interpolate_diagnostic = True
    if interpolate_diagnostic:
        if best_timing_fit is None:
            best_timing_fit = _predict_timing_separable_speed(
                firing_fit,
                np.full(fit_idx.size, best_speed, dtype=float),
                np.ones(time_axis.size, dtype=float),
                T_fit,
                time_axis,
                resolution_fit,
                offset=offset,
            )
        predicted_full = _expand_sampled_timing_diagnostic(
            best_timing_fit,
            fit_idx,
            T.size,
        )
    info["replication_timing_fit"] = predicted_full
    info["residual"] = predicted_full - T
    info["replication_timing_fit_interpolated"] = interpolate_diagnostic
    return _RmapInfoFloat(best_speed, info)


def _fit_spatial_speed_with_general_firing(
    observed_timing,
    firing_rate,
    resolution=1.0,
    resolution_time=5.0,
    time_grid=None,
    T0=None,
    fit_step=0.4,
    maxiter=10,
    speed_bounds=None,
    initial_fork_speed=1.4,
    smoothw=100,
    refine_edge_kb=1000,
    perQ=False,
    error=None,
    verbose=True,
    print_every=1,
):
    if perQ:
        raise NotImplementedError("deterministic fork-speed fitting currently supports linear geometry only.")
    T = np.asarray(observed_timing, dtype=float)
    offset = 0.0 if T0 is None else float(T0)
    if not np.isfinite(offset):
        raise ValueError("T0 must be finite when supplied.")
    relative_timing = T - offset
    fit_idx, stride = _speed_fit_indices(
        relative_timing,
        resolution,
        perQ=perQ,
        refine_edge_kb=refine_edge_kb,
    )
    T_fit = T[fit_idx]
    firing_fit = _subset_firing_positions(firing_rate, fit_idx, T.size)
    time_axis = _time_axis_for_speed_fit(
        T,
        firing_rate,
        resolution_time=resolution_time,
        time_grid=time_grid,
        T0=T0,
    )
    resolution_fit = float(resolution) * float(stride)

    speed_full = _initial_spatial_speed_profile(initial_fork_speed, T.size)
    reference = _initial_positive_speed_value(speed_full)
    low, high = _direct_speed_shape_bounds([reference], speed_bounds=speed_bounds)
    speed = np.clip(speed_full[fit_idx].astype(float, copy=True), low, high)
    maxiter = int(maxiter)
    fit_step = float(fit_step)
    if not np.isfinite(fit_step) or fit_step < 0:
        raise ValueError("fit_step must be finite and >= 0.")
    error = _normalise_error_threshold(error)
    print_every = max(1, int(print_every))
    smooth_window = smoothw
    if smooth_window is not None:
        smooth_window = max(3, min(int(smooth_window), speed.size))
    best_speed = speed.copy()
    best_mse = np.inf
    best_timing_fit = None
    start_time = time.time()

    for it in range(1, max(1, maxiter) + 1):
        predicted = _predict_timing_separable_speed(
            firing_fit,
            speed,
            np.ones(time_axis.size, dtype=float),
            T_fit,
            time_axis,
            resolution_fit,
            offset=offset,
        )
        valid = np.isfinite(T_fit) & np.isfinite(predicted) & (T_fit > 0) & (predicted > 0)
        if not np.any(valid):
            break
        mse = mean_squared_error(T_fit[valid], predicted[valid])
        if mse < best_mse:
            best_mse = mse
            best_speed = speed.copy()
            best_timing_fit = predicted.copy()
        done = it == max(1, maxiter) or _error_reached(mse, error)
        if _fit_progress_enabled(verbose) and (it % print_every == 0 or done):
            etaferr(it=it, maxiter=max(1, maxiter), start_time=start_time, err=mse, done=done)
        if done or maxiter <= 0:
            break
        ratio = np.ones_like(speed)
        ratio[valid] = np.clip(predicted[valid] / T_fit[valid], 0.25, 4.0)
        speed = np.clip(speed * ratio ** fit_step, low, high)
        if smooth_window is not None and smooth_window > 2 and speed.size > 2:
            speed = np.clip(smoothf(speed, window=smooth_window), low, high)

    x_full = np.arange(T.size, dtype=float) * float(resolution)
    x_fit = fit_idx.astype(float) * float(resolution)
    speed_full = np.interp(x_full, x_fit, best_speed, left=best_speed[0], right=best_speed[-1])
    speed_full = np.clip(speed_full, low, high)
    info = {
        "target_shape": "best",
        "fork_speed_shape": "site-specific v(x)",
        "fork_speed_bounds": (float(low), float(high)),
        "fit_mse": None if not np.isfinite(best_mse) else float(best_mse),
        "fit_mse_description": (
            "Deterministic timing MSE on cropped/subsampled positions using the supplied time-dependent firing rate."
        ),
    }
    interpolate_diagnostic = not _use_exact_full_timing_diagnostic(T.size, time_axis.size)
    if not interpolate_diagnostic:
        try:
            predicted_full = _predict_timing_separable_speed(
                firing_rate,
                speed_full,
                np.ones(time_axis.size, dtype=float),
                T,
                time_axis,
                resolution,
                offset=offset,
            )
        except Exception:
            interpolate_diagnostic = True
    if interpolate_diagnostic:
        if best_timing_fit is None:
            best_timing_fit = _predict_timing_separable_speed(
                firing_fit,
                best_speed,
                np.ones(time_axis.size, dtype=float),
                T_fit,
                time_axis,
                resolution_fit,
                offset=offset,
            )
        predicted_full = _expand_sampled_timing_diagnostic(
            best_timing_fit,
            fit_idx,
            T.size,
        )
    info["replication_timing_fit"] = predicted_full
    info["residual"] = predicted_full - T
    info["replication_timing_fit_interpolated"] = interpolate_diagnostic
    return _RmapInfoArray(speed_full, info=info)


def _initial_temporal_factor_profile(initial_fork_speed, n_time, spatial_scale):
    temporal_speed = _initial_temporal_speed_profile(initial_fork_speed, n_time, default=spatial_scale)
    return np.maximum(temporal_speed / max(float(spatial_scale), 1e-12), 1e-12)


def _fit_spatial_component_for_separable_speed(
    observed_timing,
    firing_rate,
    temporal_factor,
    resolution=1.0,
    resolution_time=5.0,
    time_grid=None,
    T0=None,
    fit_step=0.4,
    maxiter=10,
    speed_bounds=None,
    initial_fork_speed=1.4,
    smoothw=100,
    refine_edge_kb=1000,
    perQ=False,
    error=None,
    verbose=True,
    print_every=1,
):
    if perQ:
        raise NotImplementedError("factorized fork-speed fitting currently supports linear geometry only.")
    T = np.asarray(observed_timing, dtype=float)
    offset = 0.0 if T0 is None else float(T0)
    if not np.isfinite(offset):
        raise ValueError("T0 must be finite when supplied.")
    fit_idx, stride = _speed_fit_indices(
        T - offset,
        resolution,
        perQ=perQ,
        refine_edge_kb=refine_edge_kb,
    )
    T_fit = T[fit_idx]
    firing_fit = _subset_firing_positions(firing_rate, fit_idx, T.size)
    time_axis = _time_axis_for_speed_fit(
        T,
        firing_rate,
        resolution_time=resolution_time,
        time_grid=time_grid,
        T0=T0,
    )
    q = np.asarray(temporal_factor, dtype=float).reshape(-1)
    if q.size != time_axis.size:
        raise ValueError("temporal fork-speed factor must match time_grid.")
    if np.any(~np.isfinite(q)) or np.any(q <= 0):
        raise ValueError("temporal fork-speed factor must contain finite positive values.")

    resolution_fit = float(resolution) * float(stride)
    spatial_full = _initial_spatial_speed_profile(initial_fork_speed, T.size)
    reference = _initial_positive_speed_value(spatial_full)
    low, high = _direct_speed_shape_bounds([reference], speed_bounds=speed_bounds)
    speed = np.clip(spatial_full[fit_idx].astype(float, copy=True), low, high)
    maxiter = int(maxiter)
    fit_step = float(fit_step)
    if not np.isfinite(fit_step) or fit_step < 0:
        raise ValueError("fit_step_x must be finite and >= 0.")
    error = _normalise_error_threshold(error)
    print_every = max(1, int(print_every))
    smooth_window = smoothw
    if smooth_window is not None:
        smooth_window = max(3, min(int(smooth_window), speed.size))
    best_speed = speed.copy()
    best_mse = np.inf
    best_predicted = None
    start_time = time.time()
    maxiter_display = max(1, maxiter)

    for it in range(1, maxiter_display + 1):
        predicted = _predict_timing_separable_speed(
            firing_fit,
            speed,
            q,
            T_fit,
            time_axis,
            resolution_fit,
            offset=offset,
        )
        valid = np.isfinite(T_fit) & np.isfinite(predicted) & (T_fit > 0) & (predicted > 0)
        if not np.any(valid):
            break
        mse = mean_squared_error(T_fit[valid], predicted[valid])
        if mse < best_mse:
            best_mse = mse
            best_speed = speed.copy()
            best_predicted = predicted.copy()
        done = it == maxiter_display or _error_reached(mse, error)
        if _fit_progress_enabled(verbose) and (it % print_every == 0 or done):
            etaferr(it=it, maxiter=maxiter_display, start_time=start_time, err=mse, done=done)
        if done or maxiter <= 0:
            break
        ratio = np.ones_like(speed)
        ratio[valid] = np.clip(predicted[valid] / T_fit[valid], 0.25, 4.0)
        speed = np.clip(speed * ratio ** fit_step, low, high)
        if smooth_window is not None and smooth_window > 2 and speed.size > 2:
            speed = np.clip(smoothf(speed, window=smooth_window), low, high)

    x_full = np.arange(T.size, dtype=float) * float(resolution)
    x_fit = fit_idx.astype(float) * float(resolution)
    speed_full = np.interp(x_full, x_fit, best_speed, left=best_speed[0], right=best_speed[-1])
    speed_full = np.clip(speed_full, low, high)
    predicted_full = np.full(T.size, np.nan, dtype=float)
    if best_predicted is not None:
        predicted_full[fit_idx] = best_predicted
    info = {
        "target_shape": "best",
        "fork_speed_component_x": "site-specific v_x(x) conditioned on current q(t)",
        "fork_speed_bounds": (float(low), float(high)),
        "fit_mse": None if not np.isfinite(best_mse) else float(best_mse),
    }
    if best_predicted is not None:
        info["replication_timing_fit"] = predicted_full
        info["residual"] = predicted_full - T
    return _RmapInfoArray(speed_full, info=info)


def _fit_temporal_factor_for_separable_speed(
    observed_timing,
    firing_rate,
    spatial_speed,
    resolution=1.0,
    resolution_time=5.0,
    time_grid=None,
    T0=None,
    initial_temporal_factor=None,
    fit_step=0.35,
    maxiter=10,
    speed_bounds=None,
    temporal_smoothw=None,
    refine_ratio_bounds=(0.25, 4.0),
    refine_edge_kb=1000,
    perQ=False,
    error=None,
    verbose=True,
    print_every=1,
):
    if perQ:
        raise NotImplementedError("factorized fork-speed fitting currently supports linear geometry only.")
    T = np.asarray(observed_timing, dtype=float)
    spatial = np.asarray(spatial_speed, dtype=float).reshape(-1)
    if spatial.size != T.size:
        raise ValueError("spatial fork-speed component must match replication_timing length.")
    if np.any(~np.isfinite(spatial)) or np.any(spatial <= 0):
        raise ValueError("spatial fork-speed component must contain finite positive values.")
    offset = 0.0 if T0 is None else float(T0)
    if not np.isfinite(offset):
        raise ValueError("T0 must be finite when supplied.")
    fit_idx, stride = _speed_fit_indices(
        T - offset,
        resolution,
        perQ=perQ,
        refine_edge_kb=refine_edge_kb,
    )
    T_fit = T[fit_idx]
    spatial_fit = spatial[fit_idx]
    firing_fit = _subset_firing_positions(firing_rate, fit_idx, T.size)
    time_axis = _time_axis_for_speed_fit(
        T,
        firing_rate,
        resolution_time=resolution_time,
        time_grid=time_grid,
        T0=T0,
    )
    scale = _initial_positive_speed_value(spatial)
    if initial_temporal_factor is None:
        q = np.ones(time_axis.size, dtype=float)
    else:
        q = np.asarray(initial_temporal_factor, dtype=float).reshape(-1)
        if q.size != time_axis.size:
            raise ValueError("initial temporal fork-speed factor must match time_grid.")
    if np.any(~np.isfinite(q)) or np.any(q <= 0):
        q = np.ones(time_axis.size, dtype=float)

    low, high = _direct_speed_shape_bounds(spatial, speed_bounds=speed_bounds)
    q_low = max(low / max(float(np.nanmax(spatial)), 1e-12), 1e-12)
    q_high = high / max(float(np.nanmin(spatial)), 1e-12)
    if not np.isfinite(q_high) or q_high <= q_low:
        q_low, q_high = 1e-3, 1e3
    q = np.clip(q, q_low, q_high)
    if refine_ratio_bounds is not None:
        low_ratio, high_ratio = map(float, refine_ratio_bounds)
        if not (np.isfinite(low_ratio) and np.isfinite(high_ratio) and 0 < low_ratio < high_ratio):
            raise ValueError("refine_ratio_bounds must be a positive (low, high) pair.")
    fit_step = float(fit_step)
    if not np.isfinite(fit_step) or fit_step < 0:
        raise ValueError("fit_step_t must be finite and >= 0.")
    maxiter = int(maxiter)
    error = _normalise_error_threshold(error)
    print_every = max(1, int(print_every))
    start_time = time.time()
    maxiter_display = max(1, maxiter)
    resolution_fit = float(resolution) * float(stride)

    t_norm = _normalised_fit_axis(time_axis)
    log_q = np.log(q)
    finite = np.isfinite(log_q) & np.isfinite(t_norm)
    if np.count_nonzero(finite) >= 2 and np.nanstd(t_norm[finite]) > 0:
        slope0, intercept0 = np.polyfit(t_norm[finite], log_q[finite], 1)
    else:
        intercept0 = float(np.nanmedian(log_q[finite])) if np.any(finite) else 0.0
        slope0 = 0.0
    if not np.isfinite(intercept0):
        intercept0 = 0.0
    if not np.isfinite(slope0):
        slope0 = 0.0

    log_low = float(np.log(q_low))
    log_high = float(np.log(q_high))
    if log_high <= log_low:
        log_low, log_high = -8.0, 8.0
    params0 = np.array(
        [
            np.clip(intercept0, log_low, log_high),
            np.clip(slope0, -8.0, 8.0),
        ],
        dtype=float,
    )
    bounds = [(log_low, log_high), (-8.0, 8.0)]
    finite_diff_step = _optimizer_eps_from_fit_step(fit_step)
    best = {
        "params": params0.copy(),
        "q": np.clip(np.exp(params0[0] + params0[1] * t_norm), q_low, q_high),
        "mse": np.inf,
        "predicted": None,
        "calls": 0,
    }

    def make_q(params):
        values = np.exp(float(params[0]) + float(params[1]) * t_norm)
        values = np.clip(values, q_low, q_high)
        if temporal_smoothw is not None and values.size > 2:
            values = np.clip(smoothf(values, window=temporal_smoothw), q_low, q_high)
        return values

    def objective(params):
        q_candidate = make_q(params)
        predicted = _predict_timing_separable_speed(
            firing_fit,
            spatial_fit,
            q_candidate,
            T_fit,
            time_axis,
            resolution_fit,
            offset=offset,
        )
        valid = np.isfinite(T_fit) & np.isfinite(predicted) & (T_fit > 0) & (predicted > 0)
        if not np.any(valid):
            mse = np.inf
            loss = np.inf
        else:
            residual = predicted[valid] - T_fit[valid]
            mse = mean_squared_error(T_fit[valid], predicted[valid])
            loss = 0.5 * float(np.dot(residual, residual))
        best["calls"] += 1
        if mse < best["mse"]:
            best["params"] = np.asarray(params, dtype=float).copy()
            best["q"] = q_candidate.copy()
            best["mse"] = float(mse)
            best["predicted"] = predicted.copy()
        done = best["calls"] >= maxiter_display or _error_reached(mse, error)
        if _fit_progress_enabled(verbose) and best["calls"] <= maxiter_display:
            etaferr(
                it=maxiter_display if done else min(best["calls"], maxiter_display),
                maxiter=maxiter_display,
                start_time=start_time,
                err=best["mse"] if np.isfinite(best["mse"]) else mse,
                done=done,
            )
        if maxiter <= 0 or done:
            raise _FitConverged
        return loss

    if maxiter <= 0:
        try:
            objective(params0)
        except _FitConverged:
            pass
    else:
        try:
            minimize(
                objective,
                params0,
                method="L-BFGS-B",
                bounds=bounds,
                options={
                    "maxiter": maxiter,
                    "maxfun": maxiter,
                    "ftol": 1e-9,
                    "eps": finite_diff_step,
                    "maxls": 10,
                },
            )
        except _FitConverged:
            pass

    best_q = best["q"]
    best_mse = best["mse"]
    best_predicted = best["predicted"]
    params = best["params"]
    if _fit_progress_enabled(verbose) and 0 < best["calls"] < maxiter_display and best_predicted is not None:
        etaferr(
            it=maxiter_display,
            maxiter=maxiter_display,
            start_time=start_time,
            err=best_mse,
            done=True,
        )

    predicted_full = np.full(T.size, np.nan, dtype=float)
    if best_predicted is not None:
        predicted_full[fit_idx] = best_predicted
    info = {
        "target_shape": "best",
        "fork_speed_component_t": "smooth q(t) = exp(a + b T) conditioned on current v_x(x)",
        "fork_speed_v0": float(scale),
        "fork_speed_q0": float(np.exp(params[0])),
        "fork_speed_q_amplitude_t": float(params[1]),
        "fork_speed_bounds": (float(low), float(high)),
        "fit_mse": None if not np.isfinite(best_mse) else float(best_mse),
    }
    if best_predicted is not None:
        info["replication_timing_fit"] = predicted_full
        info["residual"] = predicted_full - T
    return _RmapInfoArray(best_q, info=info)


def _fit_linear_increasing_temporal_factor_for_separable_speed(
    observed_timing,
    firing_rate,
    spatial_speed,
    resolution=1.0,
    resolution_time=5.0,
    time_grid=None,
    T0=None,
    initial_temporal_factor=None,
    fit_step=0.003,
    maxiter=10,
    speed_bounds=None,
    refine_ratio_bounds=(0.25, 4.0),
    refine_edge_kb=1000,
    perQ=False,
    error=None,
    verbose=True,
    print_every=1,
):
    if perQ:
        raise NotImplementedError("factorized fork-speed fitting currently supports linear geometry only.")
    T = np.asarray(observed_timing, dtype=float)
    spatial = np.asarray(spatial_speed, dtype=float).reshape(-1)
    if spatial.size != T.size:
        raise ValueError("spatial fork-speed component must match replication_timing length.")
    if np.any(~np.isfinite(spatial)) or np.any(spatial <= 0):
        raise ValueError("spatial fork-speed component must contain finite positive values.")
    offset = 0.0 if T0 is None else float(T0)
    if not np.isfinite(offset):
        raise ValueError("T0 must be finite when supplied.")

    fit_idx, stride = _speed_fit_indices(
        T - offset,
        resolution,
        perQ=perQ,
        refine_edge_kb=refine_edge_kb,
    )
    T_fit = T[fit_idx]
    spatial_fit = spatial[fit_idx]
    firing_fit = _subset_firing_positions(firing_rate, fit_idx, T.size)
    time_axis = _time_axis_for_speed_fit(
        T,
        firing_rate,
        resolution_time=resolution_time,
        time_grid=time_grid,
        T0=T0,
    )
    if initial_temporal_factor is None:
        baseline_q = np.ones(time_axis.size, dtype=float)
    else:
        baseline_q = np.asarray(initial_temporal_factor, dtype=float).reshape(-1)
        if baseline_q.size != time_axis.size:
            raise ValueError("initial temporal fork-speed factor must match time_grid.")
        if np.any(~np.isfinite(baseline_q)) or np.any(baseline_q <= 0):
            baseline_q = np.ones(time_axis.size, dtype=float)

    low, high = _direct_speed_shape_bounds(spatial, speed_bounds=speed_bounds)
    q_low = max(low / max(float(np.nanmax(spatial)), 1e-12), 1e-12)
    q_high = high / max(float(np.nanmin(spatial)), 1e-12)
    if not np.isfinite(q_high) or q_high <= q_low:
        q_low, q_high = 1e-3, 1e3
    if refine_ratio_bounds is not None:
        low_ratio, high_ratio = map(float, refine_ratio_bounds)
        if not (np.isfinite(low_ratio) and np.isfinite(high_ratio) and 0 < low_ratio < high_ratio):
            raise ValueError("refine_ratio_bounds must be a positive (low, high) pair.")
        q_low = max(q_low, low_ratio)
        q_high = min(q_high, high_ratio)
        if q_high <= q_low:
            q_low, q_high = low_ratio, high_ratio

    baseline_q = np.clip(baseline_q, q_low, q_high)
    fit_step = float(fit_step)
    if not np.isfinite(fit_step) or fit_step < 0:
        raise ValueError("fit_step_t must be finite and >= 0.")
    maxiter = int(maxiter)
    error = _normalise_error_threshold(error)
    print_every = max(1, int(print_every))
    maxiter_display = max(1, maxiter)
    resolution_fit = float(resolution) * float(stride)

    if time_axis.size <= 1:
        t01 = np.zeros(time_axis.size, dtype=float)
    else:
        span = float(time_axis[-1] - time_axis[0])
        t01 = np.zeros(time_axis.size, dtype=float) if span <= 0 else (time_axis - time_axis[0]) / span

    max_linear_increase = max(fit_step * maxiter_display, 0.0)
    max_linear_increase = min(max_linear_increase, max(q_high - 1.0, 0.0))
    if maxiter <= 0 or max_linear_increase <= 0:
        slopes = np.array([0.0], dtype=float)
    else:
        slopes = np.linspace(0.0, max_linear_increase, maxiter_display)

    def timing_mse(q):
        predicted = _predict_timing_separable_speed(
            firing_fit,
            spatial_fit,
            q,
            T_fit,
            time_axis,
            resolution_fit,
            offset=offset,
        )
        valid = np.isfinite(T_fit) & np.isfinite(predicted) & (T_fit > 0) & (predicted > 0)
        if not np.any(valid):
            return np.inf, predicted
        return mean_squared_error(T_fit[valid], predicted[valid]), predicted

    baseline_mse, baseline_predicted = timing_mse(baseline_q)
    best_q = baseline_q.copy()
    best_mse = baseline_mse
    best_predicted = baseline_predicted
    best_slope = 0.0
    accepted = False
    start_time = time.time()

    for i, slope in enumerate(slopes, start=1):
        candidate_q = np.clip(1.0 + float(slope) * t01, q_low, q_high)
        mse, predicted = timing_mse(candidate_q)
        if np.isfinite(mse) and (not np.isfinite(best_mse) or mse < best_mse):
            best_q = candidate_q
            best_mse = float(mse)
            best_predicted = predicted
            best_slope = float(slope)
            accepted = best_slope > 0
        done = i == slopes.size or _error_reached(best_mse, error)
        if _fit_progress_enabled(verbose) and (i % print_every == 0 or done):
            etaferr(it=i, maxiter=slopes.size, start_time=start_time, err=best_mse, done=done)
        if done and i != slopes.size:
            break

    predicted_full = np.full(T.size, np.nan, dtype=float)
    if best_predicted is not None:
        predicted_full[fit_idx] = best_predicted
    info = {
        "target_shape": "best",
        "fork_speed_component_t": "guarded linear increasing q(t) conditioned on current v_x(x)",
        "fork_speed_v0": float(_initial_positive_speed_value(spatial)),
        "fork_speed_q_slope_t": float(best_slope),
        "fork_speed_temporal_accepted": bool(accepted),
        "fork_speed_temporal_baseline_mse": None if not np.isfinite(baseline_mse) else float(baseline_mse),
        "fork_speed_bounds": (float(low), float(high)),
        "fit_mse": None if not np.isfinite(best_mse) else float(best_mse),
    }
    if best_predicted is not None:
        info["replication_timing_fit"] = predicted_full
        info["residual"] = predicted_full - T
    return _RmapInfoArray(best_q, info=info)


def _fit_mixed_intercept_slope_spacetime_speed(
    observed_timing,
    firing_rate,
    spatial_firing_rate=None,
    resolution=1.0,
    resolution_time=5.0,
    time_grid=None,
    T0=None,
    neighbourhood_range=2000,
    space_fit_step=0.4,
    space_maxiter=10,
    temporal_fit_step=0.003,
    temporal_maxiter=10,
    speed_bounds=None,
    initial_fork_speed=1.4,
    smoothw=100,
    refine_ratio_bounds=(0.25, 4.0),
    refine_edge_kb=1000,
    perQ=False,
    error=None,
    verbose=True,
    print_every=1,
):
    if perQ:
        raise NotImplementedError("mixed fork-speed fitting currently supports linear geometry only.")
    T = np.asarray(observed_timing, dtype=float)
    time_axis = np.asarray(time_grid, dtype=float)
    if time_axis.ndim != 1 or time_axis.size == 0:
        raise ValueError("time_grid must be a non-empty one-dimensional array.")
    offset = 0.0 if T0 is None else float(T0)
    if not np.isfinite(offset):
        raise ValueError("T0 must be finite when supplied.")

    if spatial_firing_rate is None:
        firing_xt = _space_time_firing_for_separable_speed(firing_rate, T.size, time_axis.size)
        spatial_firing = np.nanmean(firing_xt, axis=0)
    else:
        spatial_firing = _spatial_firing_for_speed_fit(spatial_firing_rate, T.size)

    spatial_seed = map_timing_firing_spatial_speed(
        {"replication_timing": T, "firing_rate": spatial_firing},
        resolution=resolution,
        neighbourhood_range=neighbourhood_range,
        fit_step=space_fit_step,
        maxiter=space_maxiter,
        speed_bounds=speed_bounds,
        initial_fork_speed=initial_fork_speed,
        target_shape="best",
        smoothw=smoothw,
        error=error,
        refine_edge_kb=refine_edge_kb,
        perQ=perQ,
        n_bins=T.size,
        verbose=verbose,
        print_every=print_every,
    )
    spatial_info = dict(getattr(spatial_seed, "_rmap_info", {}) or {})
    speed_space = np.asarray(spatial_seed, dtype=float).reshape(-1)
    if speed_space.size != T.size:
        raise ValueError("spatial fork-speed seed must match replication_timing length.")
    if np.any(~np.isfinite(speed_space)) or np.any(speed_space <= 0):
        raise ValueError("spatial fork-speed seed must contain finite positive values.")

    low, high = _direct_speed_shape_bounds(speed_space, speed_bounds=speed_bounds)
    speed_space = np.clip(speed_space, low, high)
    fit_idx, stride = _speed_fit_indices(
        T - offset,
        resolution,
        perQ=perQ,
        refine_edge_kb=refine_edge_kb,
    )
    T_fit = T[fit_idx]
    firing_fit = _subset_firing_positions(firing_rate, fit_idx, T.size)
    resolution_fit = float(resolution) * float(stride)

    q_low = max(low / max(float(np.nanmax(speed_space)), 1e-12), 1e-12)
    q_high = high / max(float(np.nanmin(speed_space)), 1e-12)
    if not np.isfinite(q_high) or q_high <= q_low:
        q_low, q_high = 1e-3, 1e3
    if refine_ratio_bounds is not None:
        low_ratio, high_ratio = map(float, refine_ratio_bounds)
        if not (np.isfinite(low_ratio) and np.isfinite(high_ratio) and 0 < low_ratio < high_ratio):
            raise ValueError("refine_ratio_bounds must be a positive (low, high) pair.")
    else:
        low_ratio, high_ratio = 0.25, 4.0

    if time_axis.size <= 1:
        t01 = np.zeros(time_axis.size, dtype=float)
    else:
        span = float(time_axis[-1] - time_axis[0])
        t01 = np.zeros(time_axis.size, dtype=float) if span <= 0 else (time_axis - time_axis[0]) / span

    space_step = float(space_fit_step)
    slope_step = float(temporal_fit_step)
    if not np.isfinite(space_step) or space_step < 0:
        raise ValueError("fit_step_x must be finite and >= 0.")
    if not np.isfinite(slope_step) or slope_step < 0:
        raise ValueError("fit_step_t must be finite and >= 0.")
    maxiter = max(1, int(max(space_maxiter, temporal_maxiter)))
    error = _normalise_error_threshold(error)
    print_every = max(1, int(print_every))
    smooth_window = smoothw
    if smooth_window is not None:
        smooth_window = max(3, min(int(smooth_window), speed_space.size))

    max_slope = max(q_high - 1.0, 0.0)
    slope = 0.0

    def q_from_slope(value):
        return np.clip(1.0 + float(value) * t01, q_low, q_high)

    def timing_mse(space, slope_value):
        q = q_from_slope(slope_value)
        predicted = _predict_timing_separable_speed(
            firing_fit,
            np.asarray(space, dtype=float).reshape(-1)[fit_idx],
            q,
            T_fit,
            time_axis,
            resolution_fit,
            offset=offset,
        )
        valid = np.isfinite(T_fit) & np.isfinite(predicted) & (T_fit > 0) & (predicted > 0)
        if not np.any(valid):
            return np.inf, predicted
        return mean_squared_error(T_fit[valid], predicted[valid]), predicted

    current_mse, current_predicted = timing_mse(speed_space, slope)
    best_space = speed_space.copy()
    best_slope = slope
    best_mse = current_mse
    best_predicted = current_predicted
    accepted_temporal = False
    no_improvement = 0
    start_time = time.time()

    x_full = np.arange(T.size, dtype=float) * float(resolution)
    x_fit = fit_idx.astype(float) * float(resolution)

    for it in range(1, maxiter + 1):
        q = q_from_slope(slope)
        predicted = _predict_timing_separable_speed(
            firing_fit,
            speed_space[fit_idx],
            q,
            T_fit,
            time_axis,
            resolution_fit,
            offset=offset,
        )
        valid = np.isfinite(T_fit) & np.isfinite(predicted) & (T_fit > 0) & (predicted > 0)
        ratio_fit = np.ones(fit_idx.size, dtype=float)
        if np.any(valid):
            ratio_fit[valid] = np.clip(predicted[valid] / T_fit[valid], low_ratio, high_ratio)
        ratio_full = np.interp(x_full, x_fit, ratio_fit, left=ratio_fit[0], right=ratio_fit[-1])
        space_candidate = np.clip(speed_space * ratio_full ** space_step, low, high)
        if smooth_window is not None and smooth_window > 2 and space_candidate.size > 2:
            space_candidate = np.clip(smoothf(space_candidate, window=smooth_window), low, high)
        slope_candidate = min(max_slope, slope + slope_step)

        candidates = [
            ("current", speed_space, slope),
            ("space", space_candidate, slope),
            ("time", speed_space, slope_candidate),
            ("mixed", space_candidate, slope_candidate),
        ]
        chosen_space = speed_space
        chosen_slope = slope
        chosen_mse = current_mse
        chosen_predicted = current_predicted
        for _name, candidate_space, candidate_slope in candidates[1:]:
            mse, candidate_predicted = timing_mse(candidate_space, candidate_slope)
            if np.isfinite(mse) and (not np.isfinite(chosen_mse) or mse < chosen_mse):
                chosen_space = np.asarray(candidate_space, dtype=float).copy()
                chosen_slope = float(candidate_slope)
                chosen_mse = float(mse)
                chosen_predicted = candidate_predicted

        if np.isfinite(chosen_mse) and (not np.isfinite(current_mse) or chosen_mse < current_mse):
            speed_space = chosen_space
            slope = chosen_slope
            current_mse = chosen_mse
            current_predicted = chosen_predicted
            no_improvement = 0
            if current_mse < best_mse or not np.isfinite(best_mse):
                best_space = speed_space.copy()
                best_slope = float(slope)
                best_mse = float(current_mse)
                best_predicted = current_predicted
                accepted_temporal = best_slope > 0
        else:
            no_improvement += 1
            space_step *= 0.5
            slope_step *= 0.5

        done = it == maxiter or _error_reached(best_mse, error) or no_improvement >= 3
        if _fit_progress_enabled(verbose) and (it % print_every == 0 or done):
            etaferr(it=it, maxiter=maxiter, start_time=start_time, err=best_mse, done=done)
        if done and it != maxiter:
            break

    q_best = q_from_slope(best_slope)
    speed_spacetime = np.outer(q_best, best_space)
    speed_spacetime = np.clip(speed_spacetime, low, high)
    predicted_full = np.full(T.size, np.nan, dtype=float)
    if best_predicted is not None:
        predicted_full[fit_idx] = best_predicted
    temporal_info = {
        "target_shape": "best",
        "spacetime_order": "mixed",
        "fork_speed_shape": "mixed factorized v_x(x) * (1 + slope * t)",
        "fork_speed_component_x": "iterative site-specific v_x(x)",
        "fork_speed_component_t": "joint linear increasing q(t)",
        "fork_speed_q_slope_t": float(best_slope),
        "fork_speed_temporal_accepted": bool(accepted_temporal),
        "fork_speed_bounds": (float(low), float(high)),
        "fit_mse": None if not np.isfinite(best_mse) else float(best_mse),
    }
    if best_predicted is not None:
        temporal_info["replication_timing_fit"] = predicted_full
        temporal_info["residual"] = predicted_full - T
    spatial_info = dict(spatial_info)
    spatial_info["fork_speed_component_x"] = "initial spatial v_x(x) seed"
    return speed_spacetime, (low, high), spatial_info, temporal_info


def _fit_crossed_spacetime_speed(
    observed_timing,
    firing_rate,
    resolution=1.0,
    resolution_time=5.0,
    time_grid=None,
    T0=None,
    space_fit_step=0.4,
    space_maxiter=10,
    temporal_fit_step=0.003,
    temporal_maxiter=10,
    speed_bounds=None,
    initial_fork_speed=1.4,
    smoothw=100,
    refine_ratio_bounds=(0.25, 4.0),
    refine_edge_kb=1000,
    perQ=False,
    error=None,
    verbose=True,
    print_every=1,
    start_phase="space",
):
    if perQ:
        raise NotImplementedError("crossed fork-speed fitting currently supports linear geometry only.")
    T = np.asarray(observed_timing, dtype=float)
    if T.ndim != 1 or T.size == 0:
        raise ValueError("replication_timing must be a non-empty one-dimensional array.")
    time_axis = np.asarray(time_grid, dtype=float)
    if time_axis.ndim != 1 or time_axis.size == 0:
        raise ValueError("time_grid must be a non-empty one-dimensional array.")
    offset = 0.0 if T0 is None else float(T0)
    if not np.isfinite(offset):
        raise ValueError("T0 must be finite when supplied.")

    base_speed = _initial_positive_speed_value(initial_fork_speed)
    low, high = _direct_speed_shape_bounds([base_speed], speed_bounds=speed_bounds)
    speed_space = np.full(T.size, np.clip(base_speed, low, high), dtype=float)
    temporal_factor = np.ones(time_axis.size, dtype=float)

    fit_idx, stride = _speed_shape_fit_grid(T.size, 800)
    T_fit = T[fit_idx]
    crop = _refinement_crop_mask(T.size, resolution, perQ=perQ, refine_edge_kb=refine_edge_kb)
    score_mask = crop[fit_idx] & np.isfinite(T_fit) & (T_fit > offset)
    if not np.any(score_mask):
        raise ValueError("replication_timing must contain finite positive values after edge filtering.")
    firing_fit = _subset_firing_positions(firing_rate, fit_idx, T.size)
    resolution_fit = float(resolution) * float(stride)
    x_full = np.arange(T.size, dtype=float) * float(resolution)
    x_fit = fit_idx.astype(float) * float(resolution)

    if refine_ratio_bounds is None:
        low_ratio, high_ratio = 0.25, 4.0
    else:
        low_ratio, high_ratio = map(float, refine_ratio_bounds)
        if not (np.isfinite(low_ratio) and np.isfinite(high_ratio) and 0 < low_ratio < high_ratio):
            raise ValueError("refine_ratio_bounds must be a positive (low, high) pair.")

    space_step = float(space_fit_step)
    time_step = float(temporal_fit_step)
    if not np.isfinite(space_step) or space_step < 0:
        raise ValueError("fit_step_x must be finite and >= 0.")
    if not np.isfinite(time_step) or time_step < 0:
        raise ValueError("fit_step_t must be finite and >= 0.")
    remaining = {"space": max(0, int(space_maxiter)), "time": max(0, int(temporal_maxiter))}
    maxiter = sum(remaining.values())
    error = _normalise_error_threshold(error)
    print_every = max(1, int(print_every))
    smooth_window = smoothw
    if smooth_window is not None:
        smooth_window = max(3, min(int(smooth_window), speed_space.size))

    if time_axis.size <= 1:
        t01 = np.zeros(time_axis.size, dtype=float)
    else:
        span = float(time_axis[-1] - time_axis[0])
        t01 = np.zeros(time_axis.size, dtype=float) if span <= 0 else (time_axis - time_axis[0]) / span

    def q_bounds(space):
        space = np.asarray(space, dtype=float).reshape(-1)
        q_low = max(low / max(float(np.nanmin(space)), 1e-12), 1e-12)
        q_high = high / max(float(np.nanmax(space)), 1e-12)
        if not np.isfinite(q_high):
            return 1e-6, 1e6
        return q_low, max(q_low, q_high)

    def space_bounds(q):
        q = np.asarray(q, dtype=float).reshape(-1)
        s_low = max(low / max(float(np.nanmin(q)), 1e-12), 1e-12)
        s_high = high / max(float(np.nanmax(q)), 1e-12)
        if not np.isfinite(s_high):
            return low, high
        return s_low, max(s_low, s_high)

    def score(space, q):
        check_cancelled()
        predicted = _predict_timing_separable_speed(
            firing_fit,
            np.asarray(space, dtype=float).reshape(-1)[fit_idx],
            np.asarray(q, dtype=float).reshape(-1),
            T_fit,
            time_axis,
            resolution_fit,
            offset=offset,
            exact=True,
        )
        valid = score_mask & np.isfinite(predicted) & (predicted > offset)
        if not np.any(valid):
            return np.inf, predicted
        return mean_squared_error(T_fit[valid], predicted[valid]), predicted

    def spatial_candidate(space, q, predicted):
        valid = score_mask & np.isfinite(predicted) & (predicted > offset)
        ratio_fit = np.ones(fit_idx.size, dtype=float)
        if np.any(valid):
            ratio_fit[valid] = np.clip((predicted[valid] - offset) / (T_fit[valid] - offset), low_ratio, high_ratio)
        ratio_full = np.interp(x_full, x_fit[valid], ratio_fit[valid])
        s_low, s_high = space_bounds(q)
        candidate = np.clip(np.asarray(space, dtype=float) * ratio_full ** space_step, s_low, s_high)
        if smooth_window is not None and smooth_window > 2 and candidate.size > 2:
            # ``smoothf`` uses zero-padded convolution. Dividing by the
            # smoothed support removes edge attenuation, which otherwise
            # dominates short regions when the default window spans them.
            smoothed = smoothf(candidate, window=smooth_window)
            support = smoothf(np.ones_like(candidate), window=smooth_window)
            candidate = np.divide(
                smoothed,
                support,
                out=candidate.copy(),
                where=support > 0,
            )
            candidate = np.clip(candidate, s_low, s_high)
        return candidate

    def temporal_candidates(space, q):
        q_low, q_high = q_bounds(space)
        baseline = np.asarray(q, dtype=float).reshape(-1)
        if time_step == 0 or time_axis.size <= 1:
            return [np.clip(baseline, q_low, q_high)]
        best_candidate = {"q": baseline, "mse": np.inf}

        def objective(parameters):
            candidate = np.clip(
                baseline * np.exp(parameters[0] + parameters[1] * t01), q_low, q_high,
            )
            mse, predicted = score(space, candidate)
            if np.isfinite(mse) and mse < best_candidate["mse"]:
                best_candidate.update(q=candidate, mse=float(mse))
            valid = score_mask
            return np.nan_to_num(predicted[valid] - T_fit[valid], nan=1e15, posinf=1e15, neginf=-1e15)

        # A bounded trust-region update determines the useful amplitude;
        # the old fixed 0.003 increment barely changed q(t) in short runs.
        least_squares(
            objective, np.zeros(2), bounds=(-8., 8.), max_nfev=3,
            diff_step=_optimizer_eps_from_fit_step(time_step) * 1e-3,
        )
        return [best_candidate["q"]]

    current_mse, current_predicted = score(speed_space, temporal_factor)
    best_space = speed_space.copy()
    best_q = temporal_factor.copy()
    best_mse = current_mse
    best_predicted = current_predicted
    phase = "time" if str(start_phase) == "time" else "space"
    history = []
    accepted_space = 0
    accepted_time = 0
    rejected_space = 0
    rejected_time = 0
    start_time = time.time()

    for it in range(1, maxiter + 1):
        if remaining[phase] == 0:
            phase = "time" if phase == "space" else "space"
        remaining[phase] -= 1
        accepted = False
        candidate_mse = np.inf
        candidate_predicted = None
        candidate_space = speed_space
        candidate_q = temporal_factor

        if phase == "space":
            candidate_space = spatial_candidate(speed_space, temporal_factor, current_predicted)
            candidate_mse, candidate_predicted = score(candidate_space, temporal_factor)
        else:
            for q_candidate in temporal_candidates(speed_space, temporal_factor):
                mse, predicted = score(speed_space, q_candidate)
                if np.isfinite(mse) and mse < candidate_mse:
                    candidate_mse = float(mse)
                    candidate_predicted = predicted
                    candidate_q = q_candidate

        if np.isfinite(candidate_mse) and (
            not np.isfinite(current_mse) or candidate_mse < current_mse
        ):
            speed_space = np.asarray(candidate_space, dtype=float).copy()
            temporal_factor = np.asarray(candidate_q, dtype=float).copy()
            current_mse = float(candidate_mse)
            current_predicted = candidate_predicted
            accepted = True
            if phase == "space":
                accepted_space += 1
            else:
                accepted_time += 1
            if not np.isfinite(best_mse) or current_mse < best_mse:
                best_space = speed_space.copy()
                best_q = temporal_factor.copy()
                best_mse = float(current_mse)
                best_predicted = current_predicted
        else:
            if phase == "space":
                rejected_space += 1
            else:
                rejected_time += 1

        history.append(
            {
                "iteration": int(it),
                "phase": phase,
                "accepted": bool(accepted),
                "mse": None if not np.isfinite(current_mse) else float(current_mse),
                "candidate_mse": None if not np.isfinite(candidate_mse) else float(candidate_mse),
            }
        )

        done = it == maxiter or _error_reached(best_mse, error)
        if _fit_progress_enabled(verbose) and (it % print_every == 0 or done):
            etaferr(it=it, maxiter=maxiter, start_time=start_time, err=best_mse, done=done)
        if done and it != maxiter:
            break
        phase = "time" if phase == "space" else "space"

    speed_spacetime = np.outer(best_q, best_space)
    speed_spacetime = np.clip(speed_spacetime, low, high)
    predicted_full = np.full(T.size, np.nan, dtype=float)
    if best_predicted is not None:
        predicted_full[fit_idx] = best_predicted
    temporal_info = {
        "target_shape": "best",
        "spacetime_order": "crossed",
        "fork_speed_shape": "crossed factorized v_x(x) * q(t)",
        "fork_speed_component_x": "crossed one-step site-specific v_x(x) updates",
        "fork_speed_component_t": "bounded temporal q(t) amplitude and slope updates",
        "fork_speed_v0": float(base_speed),
        "fork_speed_temporal_factor": best_q.copy(),
        "fork_speed_crossed_space_steps": int(accepted_space),
        "fork_speed_crossed_time_steps": int(accepted_time),
        "fork_speed_crossed_rejected_space_steps": int(rejected_space),
        "fork_speed_crossed_rejected_time_steps": int(rejected_time),
        "fork_speed_bounds": (float(low), float(high)),
        "fit_mse": None if not np.isfinite(best_mse) else float(best_mse),
        "fit_history": history,
    }
    if best_predicted is not None:
        temporal_info["replication_timing_fit"] = predicted_full
        temporal_info["residual"] = predicted_full - T
    spatial_info = {
        "target_shape": "best",
        "fork_speed_component_x": "crossed one-step v_x(x) updates from constant speed",
        "fork_speed_v0": float(base_speed),
        "fork_speed_bounds": (float(low), float(high)),
        "fit_mse": None if not np.isfinite(best_mse) else float(best_mse),
    }
    return speed_spacetime, (low, high), spatial_info, temporal_info


def _fit_best_spacetime_speed_shape(
    observed_timing,
    firing_rate,
    spatial_speed=None,
    resolution=1.0,
    resolution_time=5.0,
    time_axis=None,
    T0=None,
    initial_fork_speed=1.4,
    speed_bounds=None,
    fit_step=0.25,
    maxiter=10,
    refine_edge_kb=1000,
    refine_seed=None,
    refine_verbose=None,
    perQ=False,
    error=None,
):
    """Fit a separable v(x,t) against the full I(x,t) timing objective."""
    if perQ:
        raise NotImplementedError("best fork-speed shape currently supports linear geometry only.")

    T = np.asarray(observed_timing, dtype=float)
    if T.ndim != 1 or T.size == 0:
        raise ValueError("replication_timing must be a non-empty one-dimensional array.")

    offset = 0.0 if T0 is None else float(T0)
    if not np.isfinite(offset):
        raise ValueError("T0 must be finite when supplied.")
    target_timing = T - offset
    if time_axis is None:
        time_axis = _survival_time_axis_from_timing(
            np.maximum(target_timing, 0.0),
            resolution_time=resolution_time,
        )
    time_axis = np.asarray(time_axis, dtype=float)
    if time_axis.ndim != 1 or time_axis.size == 0:
        raise ValueError("time_grid must be a non-empty one-dimensional array.")

    n_positions = T.size
    firing_raw = np.asarray(firing_rate, dtype=float)
    if firing_raw.size == 1:
        firing = float(firing_raw.reshape(-1)[0])
    elif firing_raw.ndim == 1 and firing_raw.size == n_positions:
        firing = firing_raw.astype(float, copy=False)
    elif firing_raw.ndim == 2 and firing_raw.shape == (n_positions, time_axis.size):
        firing = firing_raw.astype(float, copy=False)
    elif firing_raw.ndim == 2 and firing_raw.shape == (time_axis.size, n_positions):
        firing = firing_raw.T.astype(float, copy=False)
    else:
        raise ValueError("firing_rate must be scalar, x-dependent, or have shape (positions, time).")
    if np.any(~np.isfinite(firing)) or np.any(np.asarray(firing) < 0):
        raise ValueError("firing_rate must contain finite values >= 0.")

    crop = _refinement_crop_mask(n_positions, resolution, perQ=perQ, refine_edge_kb=refine_edge_kb)
    valid_full = crop & np.isfinite(T) & np.isfinite(target_timing) & (target_timing > 0)
    if not np.any(valid_full):
        raise ValueError("replication_timing must contain finite positive values after edge filtering.")

    max_fit_positions = 1200
    valid_idx = np.flatnonzero(valid_full)
    stride = max(1, int(math.ceil(valid_idx.size / max_fit_positions)))
    fit_idx = valid_idx[::stride]
    if fit_idx[-1] != valid_idx[-1]:
        fit_idx = np.concatenate([fit_idx, valid_idx[-1:]])

    T_fit = T[fit_idx]
    resolution_fit = float(resolution) * float(stride)

    if isinstance(firing, np.ndarray) and firing.ndim == 2:
        firing_fit = firing[fit_idx]
    elif isinstance(firing, np.ndarray) and firing.ndim == 1:
        firing_fit = firing[fit_idx]
    else:
        firing_fit = firing

    reference_speed = _initial_positive_speed_value(initial_fork_speed)
    low, high = _direct_speed_shape_bounds([reference_speed], speed_bounds=speed_bounds)
    error = _normalise_error_threshold(error)

    initial = np.asarray(initial_fork_speed, dtype=float)
    if spatial_speed is not None:
        spatial_seed = np.asarray(spatial_speed, dtype=float).reshape(-1)
        if spatial_seed.size != n_positions:
            raise ValueError("spatial_speed must match replication_timing when supplied.")
    elif initial.size == 1:
        relative_timing = np.asarray(target_timing, dtype=float)
        finite_positive_timing = relative_timing[np.isfinite(relative_timing) & (relative_timing > 0)]
        timing_reference = (
            float(np.nanmedian(finite_positive_timing))
            if finite_positive_timing.size
            else max(float(resolution_time), 1.0)
        )
        timing_factor = np.ones(n_positions, dtype=float)
        valid_timing = np.isfinite(relative_timing) & (relative_timing > 0)
        timing_factor[valid_timing] = np.sqrt(
            np.clip(timing_reference / np.maximum(relative_timing[valid_timing], 1e-12), 0.1, 10.0)
        )
        spatial_seed = reference_speed * timing_factor
    elif initial.ndim == 1 and initial.size == n_positions:
        spatial_seed = initial.astype(float, copy=False)
    else:
        raise ValueError("initial_fork_speed must be scalar or match replication_timing.")
    if np.any(~np.isfinite(spatial_seed)) or np.any(spatial_seed <= 0):
        raise ValueError("spatial_speed seed must be finite and positive.")
    spatial_seed = np.clip(spatial_seed, low, high)
    speed_fit = spatial_seed[fit_idx].astype(float, copy=True)
    temporal_factor = np.ones(time_axis.size, dtype=float)

    def predict_timing(spatial_component, temporal, current_firing=firing_fit):
        spatial_component = np.asarray(spatial_component, dtype=float).reshape(-1)
        temporal = np.asarray(temporal, dtype=float).reshape(-1)
        n_current = spatial_component.size
        n_time = time_axis.size
        if temporal.size != n_time:
            raise ValueError("temporal fork-speed factor must match time_grid.")

        firing_arr = np.asarray(current_firing, dtype=float)
        if firing_arr.size == 1:
            I = np.full((n_time, n_current), float(firing_arr.reshape(-1)[0]), dtype=float)
        elif firing_arr.ndim == 1 and firing_arr.size == n_current:
            I = np.tile(firing_arr.reshape(1, -1), (n_time, 1))
        elif firing_arr.ndim == 2 and firing_arr.shape == (n_current, n_time):
            I = firing_arr.T.astype(float, copy=False)
        elif firing_arr.ndim == 2 and firing_arr.shape == (n_time, n_current):
            I = firing_arr.astype(float, copy=False)
        elif firing_arr.ndim == 2 and firing_arr.shape == (1, n_time):
            I = np.repeat(firing_arr.reshape(n_time, 1), n_current, axis=1)
        else:
            raise ValueError("firing_rate shape is not compatible with separable fork-speed fitting.")
        if np.any(~np.isfinite(I)) or np.any(I < 0):
            raise ValueError("firing_rate must contain finite values >= 0.")

        x = np.arange(n_current, dtype=float) * float(resolution_fit)
        phi = np.zeros(n_current, dtype=float)
        if n_current > 1:
            inv_speed = 1.0 / np.maximum(spatial_component, 1e-12)
            dx = np.diff(x)
            phi[1:] = np.cumsum(0.5 * (inv_speed[:-1] + inv_speed[1:]) * dx)

        C = np.zeros(n_time, dtype=float)
        if n_time > 1:
            dt = np.diff(time_axis)
            C[1:] = np.cumsum(0.5 * (temporal[:-1] + temporal[1:]) * dt)

        I_mid = 0.5 * (I[:-1] + I[1:]) if n_time > 1 else np.zeros((0, n_current), dtype=float)
        cdf = np.zeros_like(I_mid, dtype=float)
        if n_current > 1:
            dx = np.diff(x)
            cdf[:, 1:] = np.cumsum(0.5 * (I_mid[:, :-1] + I_mid[:, 1:]) * dx[None, :], axis=1)

        A = np.zeros((n_time, n_current), dtype=float)
        for n in range(1, n_time):
            total = np.zeros(n_current, dtype=float)
            for k in range(n):
                dt = time_axis[k + 1] - time_axis[k]
                radius = max(C[n] - 0.5 * (C[k] + C[k + 1]), 0.0)
                left_phi = np.clip(phi - radius, phi[0], phi[-1])
                right_phi = np.clip(phi + radius, phi[0], phi[-1])
                left_x = np.interp(left_phi, phi, x)
                right_x = np.interp(right_phi, phi, x)
                a = np.clip(np.minimum(left_x, right_x), x[0], x[-1])
                b = np.clip(np.maximum(left_x, right_x), x[0], x[-1])
                total += dt * np.maximum(np.interp(b, x, cdf[k]) - np.interp(a, x, cdf[k]), 0.0)
            A[n] = total
        survival = np.exp(-np.maximum(A, 0.0))
        return np.trapezoid(survival, time_axis, axis=0) + offset

    def score(spatial_component, temporal):
        predicted = predict_timing(spatial_component, temporal)
        valid = np.isfinite(T_fit) & np.isfinite(predicted)
        if not np.any(valid):
            return np.inf, predicted
        return mean_squared_error(T_fit[valid], predicted[valid]), predicted

    verbose = bool(refine_verbose) if refine_verbose is not None else False
    start_time = time.time()
    maxiter = int(maxiter)
    maxiter_display = max(1, maxiter)
    fit_step = float(fit_step)
    if not np.isfinite(fit_step) or fit_step < 0:
        raise ValueError("fit_step must be finite and >= 0.")
    ratio_low, ratio_high = 0.25, 4.0
    smooth_window = max(3, min(int(round(1000.0 / max(resolution_fit, 1e-12))), speed_fit.size))
    best_mse, best_predicted = score(speed_fit, temporal_factor)
    if not np.isfinite(best_mse):
        raise RuntimeError(
            "The best v(x,t) fork-speed fit could not evaluate a finite initial timing prediction. "
            "Check that firing_rate has shape (positions, time) and that time_grid matches its time dimension."
        )
    best_speed_fit = speed_fit.copy()

    if maxiter <= 0:
        if _fit_progress_enabled(verbose):
            etaferr(it=1, maxiter=1, start_time=start_time, err=best_mse, done=True)
    else:
        for it in range(1, maxiter + 1):
            mse, predicted = score(speed_fit, temporal_factor)
            if np.isfinite(mse) and mse < best_mse:
                best_mse = mse
                best_predicted = predicted.copy()
                best_speed_fit = speed_fit.copy()
            valid = np.isfinite(T_fit) & np.isfinite(predicted) & (T_fit > 0) & (predicted > 0)
            if not np.any(valid):
                break
            ratio = np.ones_like(speed_fit)
            ratio[valid] = np.clip(predicted[valid] / T_fit[valid], ratio_low, ratio_high)
            speed_fit = np.clip(speed_fit * ratio ** fit_step, low, high)
            if smooth_window > 2 and speed_fit.size > 2:
                speed_fit = smoothf(speed_fit, window=smooth_window)
                speed_fit = np.clip(speed_fit, low, high)
            done = it == maxiter or _error_reached(best_mse, error)
            if _fit_progress_enabled(verbose):
                etaferr(
                    it=it,
                    maxiter=maxiter_display,
                    start_time=start_time,
                    err=best_mse,
                    done=done,
                )
            if done and it != maxiter:
                break

    x_full = np.arange(n_positions, dtype=float) * float(resolution)
    x_fit = fit_idx.astype(float) * float(resolution)
    speed_full = np.interp(x_full, x_fit, best_speed_fit, left=best_speed_fit[0], right=best_speed_fit[-1])
    speed_full = np.clip(speed_full, low, high)
    scale = float(np.nanmedian(speed_full[np.isfinite(speed_full) & (speed_full > 0)]))
    if not np.isfinite(scale) or scale <= 0:
        scale = reference_speed
    temporal_best = np.ones(time_axis.size, dtype=float)
    temporal_candidate_mse = None
    try:
        temporal_profile, _ = _temporal_profile_from_spatial_fit(
            T,
            speed_full,
            resolution_time=resolution_time,
            time_grid=time_axis,
            T0=T0,
            temporal_smoothw=None,
            floor=0.0,
        )
        speed_time = _project_exponential_speed_shape(
            temporal_profile.reshape(1, -1),
            "time",
            time_axis=time_axis,
            speed_bounds=(low, high),
        )
        temporal_candidate = np.asarray(speed_time, dtype=float).reshape(-1) / scale
        candidate_mse, candidate_predicted = score(best_speed_fit, temporal_candidate)
        temporal_candidate_mse = candidate_mse
        if np.isfinite(candidate_mse) and candidate_mse <= best_mse:
            temporal_best = temporal_candidate
            best_mse = candidate_mse
            best_predicted = candidate_predicted.copy()
    except Exception:
        temporal_candidate_mse = None

    speed = temporal_best[:, None] * speed_full[None, :]
    speed = np.clip(speed, low, high)
    predicted_full = np.full(n_positions, np.nan, dtype=float)
    predicted_full[fit_idx] = best_predicted
    info = {
        "target_shape": "best",
        "fork_speed_shape": "site-specific v(x) times exponential v(t)",
        "fork_speed_component_x": "site-specific v(x)",
        "fork_speed_component_t": "v0 * exp(b_t * T)",
        "fork_speed_v0": scale,
        "fork_speed_amplitude_t": 0.0 if np.allclose(temporal_best, 1.0) else None,
        "fork_speed_bounds": (float(low), float(high)),
        "temporal_candidate_mse": None
        if temporal_candidate_mse is None or not np.isfinite(temporal_candidate_mse)
        else float(temporal_candidate_mse),
        "fit_mse": float(best_mse),
        "fit_mse_description": (
            "Deterministic timing MSE on cropped/subsampled positions using the full I(x,t) "
            "and the returned v(x,t); it is not the stochastic rsim MSE."
        ),
        "replication_timing_fit": predicted_full,
        "residual": predicted_full - T,
    }
    if verbose:
        print(f"[rmap]  final deterministic MSE: {best_mse:.4e}")
    return _RmapInfoArray(speed, info=info)


def _firing_bounds_from_profile(profile, firing_bounds=None, floor=0.0):
    profile = np.asarray(profile, dtype=float)
    finite_positive = profile[np.isfinite(profile) & (profile > 0)]
    reference = float(np.nanmedian(finite_positive)) if finite_positive.size else 1e-6
    floor = float(floor)
    if not np.isfinite(floor) or floor < 0:
        raise ValueError("firing_floor must be finite and >= 0.")
    if firing_bounds is None:
        low = max(reference * 1e-4, floor)
        high = max(reference * 1e4, low * 10.0, 1e-12)
    else:
        if len(firing_bounds) != 2:
            raise ValueError("firing_bounds must be a (min_rate, max_rate) pair.")
        low, high = map(float, firing_bounds)
        if not (np.isfinite(low) and np.isfinite(high)) or low < 0 or high <= low:
            raise ValueError("firing_bounds must satisfy 0 <= min_rate < max_rate.")
        low = max(low, floor)
    return low, high


def _speed_fit_bin_hazard(firing_rate, resolution, is_density):
    """Convert public density only at a legacy local-map/simulation boundary."""
    if not is_density:
        return firing_rate
    return np.asarray(firing_rate, dtype=float) * float(resolution)


def _refine_temporal_speed(
    fork_speed_time,
    observed_timing,
    firing_rate,
    resolution,
    resolution_time,
    T0=None,
    temporal_fit_step=0.003,
    temporal_maxiter=10,
    temporal_sim_number=100,
    speed_bounds=None,
    refine_ratio_bounds=(0.25, 4.0),
    refine_edge_kb=1000,
    refine_seed=None,
    refine_verbose=None,
    perQ=False,
    error=None,
    time_grid=None,
):
    speed = np.asarray(fork_speed_time, dtype=float).reshape(1, -1).copy()
    error = _normalise_error_threshold(error)
    maxiter = int(temporal_maxiter)
    if maxiter <= 0:
        return speed

    T = np.asarray(observed_timing, dtype=float)
    offset = 0.0 if T0 is None else float(T0)
    if not np.isfinite(offset):
        raise ValueError("T0 must be finite when supplied.")
    relative_timing = T - offset
    n = T.size
    n_time = speed.shape[1]
    if time_grid is None:
        bins = _timing_bin_indices(T, n_time, resolution_time=resolution_time, T0=offset)
    else:
        axis = np.asarray(time_grid, dtype=float)
        if axis.shape != (n_time,):
            raise ValueError("time_grid must match the temporal fork-speed samples.")
        bins = np.searchsorted(0.5 * (axis[:-1] + axis[1:]), relative_timing)
    simulation_firing, _ = _uniform_speed_for_simulation(
        firing_rate, time_grid, resolution_time,
    )
    crop = _refinement_crop_mask(n, resolution, perQ=perQ, refine_edge_kb=refine_edge_kb)
    low_speed, high_speed = _speed_bounds_from_profile(speed, speed_bounds=speed_bounds)
    low_ratio, high_ratio = map(float, refine_ratio_bounds)
    if not (np.isfinite(low_ratio) and np.isfinite(high_ratio) and 0 < low_ratio < high_ratio):
        raise ValueError("refine_ratio_bounds must be a positive (low, high) pair.")

    fit_step = float(temporal_fit_step)
    if not np.isfinite(fit_step) or fit_step < 0:
        raise ValueError("fit_step_t must be finite and >= 0.")

    verbose = bool(refine_verbose) if refine_verbose is not None else False
    best_speed = speed.copy()
    best_err = np.inf
    best_prediction = None
    history = []
    start_time = time.time()
    for it in range(1, maxiter + 1):
        check_cancelled()
        simulation_speed, _ = _uniform_speed_for_simulation(
            speed, time_grid, resolution_time,
        )
        seed = None if refine_seed is None else int(refine_seed) + it - 1
        T_pred = rsim(
            firing_rate=simulation_firing,
            fork_speed=simulation_speed,
            sim_number=temporal_sim_number,
            resolution_space=resolution,
            resolution_time=resolution_time,
            n_bins=n,
            perQ=perQ,
            seed=seed,
            verbose=False,
            retain_rep_times_per_sim=False,
            retain_inter_origin_distances=False,
        )["replication_timing"]
        valid = crop & np.isfinite(relative_timing) & np.isfinite(T_pred) & (relative_timing > 0) & (T_pred > 0)
        if not np.any(valid):
            break
        err = mean_squared_error(relative_timing[valid], T_pred[valid])
        history.append(float(err))
        if err < best_err:
            best_err = err
            best_speed = speed.copy()
            best_prediction = np.asarray(T_pred, dtype=float).copy() + offset
        ratio = np.ones(n, dtype=float)
        ratio[valid] = np.clip(T_pred[valid] / relative_timing[valid], low_ratio, high_ratio)
        updates = _mean_ratio_by_bin(ratio, bins, n_time, valid)
        speed[0] *= updates ** fit_step
        np.clip(speed, low_speed, high_speed, out=speed)

        done = it == maxiter or _error_reached(err, error)
        if _fit_progress_enabled(verbose):
            etaferr(it=it, maxiter=maxiter, start_time=start_time, err=err, done=done)
        if done and it != maxiter:
            break

    if best_prediction is None:
        return best_speed
    residual = best_prediction - T
    return _RmapInfoArray(best_speed, {
        "replication_timing_fit": best_prediction,
        "residual": residual,
        "fit_mse": float(best_err),
        "fit_history": np.asarray(history),
        "fit_mse_description": "Best simulated timing MSE on the scored positions.",
    })


def _refine_spacetime_speed(
    fork_speed_spacetime,
    observed_timing,
    firing_rate,
    resolution,
    resolution_time,
    T0=None,
    spacetime_fit_step=0.25,
    spacetime_space_fit_step=None,
    spacetime_time_fit_step=None,
    refine_maxiter=10,
    spacetime_sim_number=100,
    spacetime_update="factorized",
    spacetime_kernel_t=1.0,
    speed_bounds=None,
    refine_ratio_bounds=(0.25, 4.0),
    refine_edge_kb=1000,
    refine_seed=None,
    refine_verbose=None,
    perQ=False,
    error=None,
    time_grid=None,
):
    speed = np.asarray(fork_speed_spacetime, dtype=float).copy()
    error = _normalise_error_threshold(error)
    maxiter = int(refine_maxiter)
    if maxiter <= 0:
        return speed

    T = np.asarray(observed_timing, dtype=float)
    offset = 0.0 if T0 is None else float(T0)
    if not np.isfinite(offset):
        raise ValueError("T0 must be finite when supplied.")
    relative_timing = T - offset
    n = T.size
    if speed.ndim != 2 or speed.shape[1] != n:
        raise ValueError("fork_speed_spacetime must have internal shape (time, positions).")

    n_time = speed.shape[0]
    if time_grid is None:
        bins = _timing_bin_indices(T, n_time, resolution_time=resolution_time, T0=offset)
    else:
        axis = np.asarray(time_grid, dtype=float)
        if axis.shape != (n_time,):
            raise ValueError("time_grid must match the temporal fork-speed samples.")
        bins = np.searchsorted(0.5 * (axis[:-1] + axis[1:]), relative_timing)
    simulation_firing, _ = _uniform_speed_for_simulation(
        firing_rate, time_grid, resolution_time,
    )
    crop = _refinement_crop_mask(n, resolution, perQ=perQ, refine_edge_kb=refine_edge_kb)
    low_speed, high_speed = _speed_bounds_from_profile(speed, speed_bounds=speed_bounds)
    low_ratio, high_ratio = map(float, refine_ratio_bounds)
    if not (np.isfinite(low_ratio) and np.isfinite(high_ratio) and 0 < low_ratio < high_ratio):
        raise ValueError("refine_ratio_bounds must be a positive (low, high) pair.")

    base_step = float(spacetime_fit_step)
    space_step = base_step if spacetime_space_fit_step is None else float(spacetime_space_fit_step)
    time_step = base_step if spacetime_time_fit_step is None else float(spacetime_time_fit_step)
    if not np.isfinite(space_step) or space_step < 0:
        raise ValueError("fit_step_xt_x must be finite and >= 0.")
    if not np.isfinite(time_step) or time_step < 0:
        raise ValueError("fit_step_xt_t must be finite and >= 0.")
    update_mode = _canonical_spacetime_update(spacetime_update)

    verbose = bool(refine_verbose) if refine_verbose is not None else False
    best_speed = speed.copy()
    best_err = np.inf
    best_prediction = None
    history = []
    start_time = time.time()
    for it in range(1, maxiter + 1):
        check_cancelled()
        simulation_speed, _ = _uniform_speed_for_simulation(
            speed.T, time_grid, resolution_time,
        )
        seed = None if refine_seed is None else int(refine_seed) + it - 1
        T_pred = rsim(
            firing_rate=simulation_firing,
            fork_speed=simulation_speed,
            sim_number=spacetime_sim_number,
            resolution_space=resolution,
            resolution_time=resolution_time,
            n_bins=n,
            perQ=perQ,
            seed=seed,
            verbose=False,
            retain_rep_times_per_sim=False,
            retain_inter_origin_distances=False,
        )["replication_timing"]
        valid = crop & np.isfinite(relative_timing) & np.isfinite(T_pred) & (relative_timing > 0) & (T_pred > 0)
        if not np.any(valid):
            break
        err = mean_squared_error(relative_timing[valid], T_pred[valid])
        history.append(float(err))
        if err < best_err:
            best_err = err
            best_speed = speed.copy()
            best_prediction = np.asarray(T_pred, dtype=float).copy() + offset
        ratio = np.ones(n, dtype=float)
        ratio[valid] = np.clip(T_pred[valid] / relative_timing[valid], low_ratio, high_ratio)
        if update_mode == "mixed":
            correction = _mixed_spacetime_correction(
                ratio,
                bins,
                n_time,
                valid,
                kernel_t=spacetime_kernel_t,
            )
            speed *= correction ** base_step
        else:
            speed[:, valid] *= ratio[valid][None, :] ** space_step
            time_updates = _mean_ratio_by_bin(ratio, bins, n_time, valid)
            speed *= time_updates[:, None] ** time_step
        np.clip(speed, low_speed, high_speed, out=speed)

        done = it == maxiter or _error_reached(err, error)
        if _fit_progress_enabled(verbose):
            etaferr(it=it, maxiter=maxiter, start_time=start_time, err=err, done=done)
        if done and it != maxiter:
            break

    if best_prediction is None:
        return best_speed
    residual = best_prediction - T
    return _RmapInfoArray(best_speed, {
        "replication_timing_fit": best_prediction,
        "residual": residual,
        "fit_mse": float(best_err),
        "fit_history": np.asarray(history),
        "fit_mse_description": "Best simulated timing MSE on the scored positions.",
    })


def map_timing_firing_temporal_speed(
    source,
    resolution=1.0,
    resolution_x=None,
    resolution_time=5.0,
    time_grid=None,
    T0=None,
    target_shape="best",
    temporal_smoothw=None,
    neighbourhood_range=2000,
    fit_step=0.4,
    maxiter=10,
    space_fit_step=None,
    space_maxiter=None,
    temporal_fit_step=0.003,
    temporal_maxiter=10,
    temporal_sim_number=100,
    speed_bounds=None,
    initial_fork_speed=1.4,
    weibull_shape=2.0,
    weibull_delay=0.0,
    smoothw=100,
    perQ=False,
    n_bins=None,
    refine_ratio_bounds=(0.25, 4.0),
    refine_edge_kb=1000,
    refine_seed=None,
    refine_verbose=None,
    error=None,
    verbose=True,
    print_every=1,
    _firing_rate_is_density=False,
):
    """Infer a time-only fork-speed profile from timing and spatial firing."""
    if perQ:
        return _periodic_speed_inverse(
            source, "time", resolution, resolution_x, resolution_time, time_grid,
            T0, initial_fork_speed, speed_bounds, temporal_maxiter, verbose,
        )
    resolution = _normalise_resolution_x(resolution, resolution_x=resolution_x)
    target_shape = _canonical_speed_target_shape(target_shape)
    T, firing_rate = _validate_timing_and_firing_for_speed(source)
    if n_bins is not None and int(n_bins) != T.size:
        raise ValueError("n_bins must match the length of replication_timing.")
    firing_rate = _spatial_firing_for_speed_fit(firing_rate, T.size)
    if refine_verbose is None:
        refine_verbose = bool(verbose)

    if target_shape in {"exponential", "best"}:
        speed_time = _fit_exponential_speed_shape(
            T,
            firing_rate=firing_rate,
            target_form="time",
            resolution=resolution,
            resolution_time=resolution_time,
            time_axis=time_grid,
            T0=T0,
            initial_fork_speed=initial_fork_speed,
            speed_bounds=speed_bounds,
            fit_step=temporal_fit_step,
            maxiter=temporal_maxiter,
            refine_edge_kb=refine_edge_kb,
            refine_seed=refine_seed,
            refine_verbose=refine_verbose,
            perQ=perQ,
            error=error,
        )
        if target_shape == "best":
            return _retag_speed_shape(
                speed_time,
                "best",
                fork_speed_shape="exponential v(t)",
                fork_speed_component_t="v0 * exp(b_t * T)",
            )
        return speed_time
    if target_shape == "weibull":
        return _fit_weibull_speed_shape(
            T,
            firing_rate=firing_rate,
            target_form="time",
            resolution=resolution,
            resolution_time=resolution_time,
            time_axis=time_grid,
            T0=T0,
            initial_fork_speed=initial_fork_speed,
            speed_bounds=speed_bounds,
            fit_step=temporal_fit_step,
            maxiter=temporal_maxiter,
            refine_edge_kb=refine_edge_kb,
            refine_verbose=refine_verbose,
            perQ=perQ,
            error=error,
            weibull_shape=weibull_shape,
            weibull_delay=weibull_delay,
        )

    speed_space = map_timing_firing_spatial_speed(
        {"replication_timing": T, "firing_rate": _speed_fit_bin_hazard(
            firing_rate, resolution, _firing_rate_is_density)},
        resolution=resolution,
        T0=T0,
        neighbourhood_range=neighbourhood_range,
        fit_step=fit_step,
        maxiter=maxiter,
        space_fit_step=space_fit_step,
        space_maxiter=space_maxiter,
        speed_bounds=speed_bounds,
        initial_fork_speed=initial_fork_speed,
        target_shape="best" if target_shape == "best" else "free",
        weibull_shape=weibull_shape,
        weibull_delay=weibull_delay,
        smoothw=smoothw,
        error=error,
        refine_edge_kb=refine_edge_kb,
        perQ=perQ,
        n_bins=n_bins,
        verbose=verbose,
        print_every=print_every,
    )
    temporal, time_axis = _temporal_profile_from_spatial_fit(
        T,
        speed_space,
        resolution_time=resolution_time,
        time_grid=time_grid,
        T0=T0,
        temporal_smoothw=temporal_smoothw,
        floor=0.0,
    )
    low, high = _speed_bounds_from_profile(speed_space, speed_bounds=speed_bounds)
    temporal = np.clip(temporal, low, high)
    if refine_verbose:
        print("\nRefining time-dependent fork speed...")
    speed_time = _refine_temporal_speed(
        temporal.reshape(1, -1),
        T,
        firing_rate=_speed_fit_bin_hazard(firing_rate, resolution, _firing_rate_is_density),
        resolution=resolution,
        resolution_time=resolution_time,
        time_grid=time_axis,
        T0=T0,
        temporal_fit_step=temporal_fit_step,
        temporal_maxiter=temporal_maxiter,
        temporal_sim_number=temporal_sim_number,
        speed_bounds=(low, high),
        refine_ratio_bounds=refine_ratio_bounds,
        refine_edge_kb=refine_edge_kb,
        refine_seed=refine_seed,
        refine_verbose=refine_verbose,
        perQ=perQ,
        error=error,
    )
    if target_shape == "exponential":
        speed_time = _project_exponential_speed_shape(
            speed_time,
            "time",
            time_axis=time_axis,
            speed_bounds=(low, high),
        )
    if target_shape == "weibull":
        speed_time = _project_weibull_speed_shape(
            speed_time,
            "time",
            time_axis=time_axis,
            speed_bounds=(low, high),
            weibull_shape=weibull_shape,
            weibull_delay=weibull_delay,
        )
    return speed_time


def _firing_kind_for_speed_components(firing_rate, n_positions):
    arr = np.asarray(firing_rate, dtype=float)
    if arr.size == 1:
        return "scalar"
    if arr.ndim == 1 and arr.size == n_positions:
        return "x"
    if arr.ndim == 2:
        if arr.shape == (n_positions, 1):
            return "x"
        if arr.shape[0] == 1 and arr.shape[1] > 1:
            return "t"
        if arr.shape[0] == n_positions and arr.shape[1] > 1:
            return "xt"
    raise ValueError("firing_rate must be scalar, x-dependent, t-dependent, or x,t-dependent.")


def _combine_speed_components(speed_space, speed_time, speed_bounds=None):
    spatial = np.asarray(speed_space, dtype=float).reshape(-1)
    temporal = np.asarray(speed_time, dtype=float).reshape(-1)
    if spatial.size == 0 or temporal.size == 0:
        raise ValueError("space and time fork-speed components must be non-empty.")
    if np.any(~np.isfinite(spatial)) or np.any(spatial <= 0):
        raise ValueError("spatial fork-speed component must contain finite positive values.")
    if np.any(~np.isfinite(temporal)) or np.any(temporal <= 0):
        raise ValueError("temporal fork-speed component must contain finite positive values.")
    low, high = _speed_bounds_from_profile(
        np.concatenate([spatial, temporal]),
        speed_bounds=speed_bounds,
    )
    scale = float(np.nanmedian(spatial[spatial > 0])) if np.any(spatial > 0) else 1.0
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.nanmedian(temporal[temporal > 0])) if np.any(temporal > 0) else 1.0
    field = np.outer(temporal, spatial / max(scale, 1e-12))
    return np.clip(field, low, high), (low, high)


def _project_spacetime_speed_target_shape(
    speed_spacetime,
    target_shape,
    resolution,
    time_axis,
    speed_bounds=None,
    weibull_shape=2.0,
    weibull_delay=0.0,
):
    if target_shape == "exponential":
        return _project_exponential_speed_shape(
            speed_spacetime,
            "spacetime",
            resolution=resolution,
            time_axis=time_axis,
            speed_bounds=speed_bounds,
        )
    if target_shape == "weibull":
        return _project_weibull_speed_shape(
            speed_spacetime,
            "spacetime",
            resolution=resolution,
            time_axis=time_axis,
            speed_bounds=speed_bounds,
            weibull_shape=weibull_shape,
            weibull_delay=weibull_delay,
        )
    if target_shape == "best":
        return _project_best_spacetime_speed_shape(
            speed_spacetime,
            time_axis=time_axis,
            speed_bounds=speed_bounds,
        )
    return speed_spacetime


def _fit_spacetime_speed_components(
    source,
    source_kind,
    resolution,
    resolution_time,
    time_axis,
    T0,
    target_shape,
    temporal_smoothw,
    neighbourhood_range,
    space_fit_step,
    space_maxiter,
    temporal_fit_step,
    temporal_maxiter,
    temporal_sim_number,
    speed_bounds,
    initial_fork_speed,
    weibull_shape,
    weibull_delay,
    smoothw,
    perQ,
    n_bins,
    refine_ratio_bounds,
    refine_edge_kb,
    refine_seed,
    refine_verbose,
    error,
    verbose,
    print_every,
    order,
    spatial_firing_rate=None,
    _firing_rate_is_density=False,
):
    missing_helpers = [
        name for name in (
            "_initial_temporal_factor_profile",
            "_fit_spatial_component_for_separable_speed",
            "_fit_temporal_factor_for_separable_speed",
            "_fit_linear_increasing_temporal_factor_for_separable_speed",
            "_fit_crossed_spacetime_speed",
            "_fit_mixed_intercept_slope_spacetime_speed",
        )
        if name not in globals()
    ]
    if missing_helpers:
        import importlib
        import sys

        module = sys.modules.get(__name__)
        if module is not None:
            importlib.reload(module)

    T, firing_rate = _validate_timing_and_firing_for_speed(source)
    time_axis = np.asarray(time_axis, dtype=float)
    if time_axis.ndim != 1 or time_axis.size == 0:
        raise ValueError("time_grid must be a non-empty one-dimensional array.")
    direct_spatial_firing = None
    if spatial_firing_rate is not None:
        direct_spatial_firing = _spatial_firing_for_speed_fit(spatial_firing_rate, T.size)
    elif str(source_kind) in {"scalar", "x"}:
        direct_spatial_firing = _spatial_firing_for_speed_fit(firing_rate, T.size)

    speed_space = _initial_spatial_speed_profile(initial_fork_speed, T.size)
    scale = _initial_positive_speed_value(speed_space)
    temporal_factor = _initial_temporal_factor_profile(
        initial_fork_speed,
        time_axis.size,
        scale,
    )
    spatial_info = {}
    temporal_info = {}

    source_kind = str(source_kind)
    del target_shape, temporal_sim_number
    del weibull_shape, weibull_delay, n_bins, refine_seed, refine_verbose

    def unit_temporal_factor(values):
        arr = np.asarray(values, dtype=float).reshape(-1)
        return arr.size == time_axis.size and np.allclose(arr, 1.0, rtol=1e-8, atol=1e-12)

    def constant_spatial_component(values):
        arr = np.asarray(values, dtype=float).reshape(-1)
        if arr.size == 0 or np.any(~np.isfinite(arr)) or np.any(arr <= 0):
            return False
        return float(np.nanmax(arr) - np.nanmin(arr)) <= max(float(np.nanmedian(arr)) * 1e-8, 1e-12)

    if order in {"crossed", "mixed"}:
        return _fit_crossed_spacetime_speed(
            T,
            firing_rate,
            resolution=resolution,
            resolution_time=resolution_time,
            time_grid=time_axis,
            T0=T0,
            space_fit_step=space_fit_step,
            space_maxiter=space_maxiter,
            temporal_fit_step=temporal_fit_step,
            temporal_maxiter=temporal_maxiter,
            speed_bounds=speed_bounds,
            initial_fork_speed=initial_fork_speed,
            smoothw=smoothw,
            refine_ratio_bounds=refine_ratio_bounds,
            refine_edge_kb=refine_edge_kb,
            perQ=perQ,
            error=error,
            verbose=verbose,
            print_every=print_every,
            start_phase="time" if order == "time_then_space" else "space",
        )

    def fit_space(current_space, current_time_factor):
        if direct_spatial_firing is not None and unit_temporal_factor(current_time_factor):
            return map_timing_firing_spatial_speed(
                {"replication_timing": T, "firing_rate": _speed_fit_bin_hazard(
                    direct_spatial_firing, resolution, _firing_rate_is_density)},
                resolution=resolution,
                T0=T0,
                neighbourhood_range=neighbourhood_range,
                fit_step=space_fit_step,
                maxiter=space_maxiter,
                speed_bounds=speed_bounds,
                initial_fork_speed=current_space,
                target_shape="best",
                smoothw=smoothw,
                error=error,
                refine_edge_kb=refine_edge_kb,
                perQ=perQ,
                n_bins=T.size,
                verbose=verbose,
                print_every=print_every,
            )
        return _fit_spatial_component_for_separable_speed(
            T,
            firing_rate,
            current_time_factor,
            resolution=resolution,
            resolution_time=resolution_time,
            time_grid=time_axis,
            T0=T0,
            fit_step=space_fit_step,
            maxiter=space_maxiter,
            speed_bounds=speed_bounds,
            initial_fork_speed=current_space,
            smoothw=smoothw,
            refine_edge_kb=refine_edge_kb,
            perQ=perQ,
            error=error,
            verbose=verbose,
            print_every=print_every,
        )

    def fit_time(current_space, current_time_factor):
        if source_kind in {"scalar", "x"} and constant_spatial_component(current_space):
            speed_time = map_timing_firing_temporal_speed(
                {"replication_timing": T, "firing_rate": firing_rate},
                resolution=resolution,
                _firing_rate_is_density=_firing_rate_is_density,
                resolution_time=resolution_time,
                time_grid=time_axis,
                T0=T0,
                target_shape="best",
                temporal_smoothw=temporal_smoothw,
                neighbourhood_range=neighbourhood_range,
                temporal_fit_step=temporal_fit_step,
                temporal_maxiter=temporal_maxiter,
                speed_bounds=speed_bounds,
                initial_fork_speed=_initial_positive_speed_value(current_space),
                smoothw=smoothw,
                perQ=perQ,
                n_bins=T.size,
                refine_ratio_bounds=refine_ratio_bounds,
                refine_edge_kb=refine_edge_kb,
                error=error,
                verbose=verbose,
                print_every=print_every,
            )
            scale = _initial_positive_speed_value(current_space)
            q = np.asarray(speed_time, dtype=float).reshape(-1) / max(scale, 1e-12)
            info = dict(getattr(speed_time, "_rmap_info", {}) or {})
            info["target_shape"] = "best"
            info["fork_speed_component_t"] = "v0 * exp(b_t * T) conditioned on constant v_x"
            info["fork_speed_v0"] = float(scale)
            return _RmapInfoArray(q, info=info)
        return _fit_linear_increasing_temporal_factor_for_separable_speed(
            T,
            firing_rate,
            current_space,
            resolution=resolution,
            resolution_time=resolution_time,
            time_grid=time_axis,
            T0=T0,
            initial_temporal_factor=current_time_factor,
            fit_step=temporal_fit_step,
            maxiter=temporal_maxiter,
            speed_bounds=speed_bounds,
            refine_ratio_bounds=refine_ratio_bounds,
            refine_edge_kb=refine_edge_kb,
            error=error,
            perQ=perQ,
            verbose=verbose,
            print_every=print_every,
        )

    phases = ("time", "space") if order == "time_then_space" else ("space", "time")
    for phase in phases:
        if phase == "space":
            speed_space = fit_space(speed_space, temporal_factor)
            spatial_info = dict(getattr(speed_space, "_rmap_info", {}) or {})
        else:
            temporal_factor = fit_time(speed_space, temporal_factor)
            temporal_info = dict(getattr(temporal_factor, "_rmap_info", {}) or {})

    scale = _initial_positive_speed_value(speed_space)
    speed_time = np.asarray(temporal_factor, dtype=float).reshape(-1) * max(scale, 1e-12)
    speed_spacetime, bounds = _combine_speed_components(
        speed_space,
        speed_time,
        speed_bounds=speed_bounds,
    )
    return speed_spacetime, bounds, spatial_info, temporal_info


def map_timing_firing_spacetime_speed(
    source,
    resolution=1.0,
    resolution_x=None,
    resolution_time=5.0,
    time_grid=None,
    T0=None,
    target_shape="best",
    temporal_smoothw=None,
    neighbourhood_range=2000,
    fit_step=None,
    maxiter=None,
    sim_number=None,
    space_fit_step=None,
    space_maxiter=None,
    temporal_fit_step=None,
    temporal_maxiter=None,
    temporal_sim_number=None,
    spacetime_fit_step=None,
    spacetime_space_fit_step=None,
    spacetime_time_fit_step=None,
    spacetime_sim_number=None,
    spacetime_update="factorized",
    spacetime_order="crossed",
    spacetime_kernel_t=1.0,
    speed_bounds=None,
    initial_fork_speed=1.4,
    weibull_shape=2.0,
    weibull_delay=0.0,
    smoothw=100,
    perQ=False,
    n_bins=None,
    refine_ratio_bounds=(0.25, 4.0),
    refine_edge_kb=1000,
    refine_seed=None,
    refine_verbose=None,
    error=None,
    verbose=True,
    print_every=1,
    _firing_rate_is_density=False,
):
    """Infer a space-time fork-speed field from timing and I(x,t)."""
    if perQ:
        return _periodic_speed_inverse(
            source, "spacetime", resolution, resolution_x, resolution_time, time_grid,
            T0, initial_fork_speed, speed_bounds,
            max((value for value in (maxiter, space_maxiter, temporal_maxiter) if value is not None), default=20),
            verbose,
        )
    resolution = _normalise_resolution_x(resolution, resolution_x=resolution_x)
    target_shape = _canonical_speed_target_shape(target_shape)
    update_mode = _canonical_spacetime_update(spacetime_update)
    order = _canonical_spacetime_order(spacetime_order)
    (
        space_fit_step,
        space_maxiter,
        temporal_fit_step,
        temporal_maxiter,
        temporal_sim_number,
    ) = _resolve_spacetime_phase_controls(
        fit_step=fit_step,
        maxiter=maxiter,
        sim_number=sim_number,
        space_fit_step=space_fit_step,
        space_maxiter=space_maxiter,
        temporal_fit_step=temporal_fit_step,
        temporal_maxiter=temporal_maxiter,
        temporal_sim_number=temporal_sim_number,
        default_space_fit_step=0.4,
        default_space_maxiter=10,
        default_temporal_fit_step=0.003,
    )
    if spacetime_space_fit_step is not None:
        space_fit_step = float(spacetime_space_fit_step)
    if spacetime_time_fit_step is not None:
        temporal_fit_step = float(spacetime_time_fit_step)
    if spacetime_sim_number is None:
        spacetime_sim_number = temporal_sim_number
    if spacetime_fit_step is None:
        spacetime_fit_step = 0.5 * (space_fit_step + temporal_fit_step)
    T, firing_rate = _validate_timing_and_firing_for_speed(source)
    if n_bins is not None and int(n_bins) != T.size:
        raise ValueError("n_bins must match the length of replication_timing.")
    firing_rate_xt = _spacetime_firing_for_speed_fit(firing_rate, T.size)
    firing_rate_space = np.nanmean(firing_rate_xt, axis=0)

    if time_grid is None:
        time_grid = np.arange(firing_rate_xt.shape[0], dtype=float) * float(resolution_time)

    if update_mode == "factorized":
        speed_spacetime, bounds, spatial_info, temporal_info = _fit_spacetime_speed_components(
            {"replication_timing": T, "firing_rate": firing_rate},
            "xt",
            resolution=resolution,
            resolution_time=resolution_time,
            time_axis=time_grid,
            T0=T0,
            target_shape=target_shape,
            temporal_smoothw=temporal_smoothw,
            neighbourhood_range=neighbourhood_range,
            space_fit_step=space_fit_step,
            space_maxiter=space_maxiter,
            temporal_fit_step=temporal_fit_step,
            temporal_maxiter=temporal_maxiter,
            temporal_sim_number=temporal_sim_number,
            speed_bounds=speed_bounds,
            initial_fork_speed=initial_fork_speed,
            weibull_shape=weibull_shape,
            weibull_delay=weibull_delay,
            smoothw=smoothw,
            perQ=perQ,
            n_bins=n_bins,
            refine_ratio_bounds=refine_ratio_bounds,
            refine_edge_kb=refine_edge_kb,
            refine_seed=refine_seed,
            refine_verbose=refine_verbose,
            error=error,
            verbose=verbose,
            print_every=print_every,
            order=order,
            _firing_rate_is_density=_firing_rate_is_density,
            spatial_firing_rate=firing_rate_space,
        )
        speed_spacetime = _project_spacetime_speed_target_shape(
            speed_spacetime,
            target_shape,
            resolution=resolution,
            time_axis=time_grid,
            speed_bounds=bounds,
            weibull_shape=weibull_shape,
            weibull_delay=weibull_delay,
        )
        if target_shape == "best":
            info = dict(getattr(speed_spacetime, "_rmap_info", {}) or {})
            info["time_grid"] = np.asarray(time_grid, dtype=float).copy()
            info["speed_form"] = "spacetime"
            if "fit_mse" in spatial_info:
                info["spatial_component_mse"] = spatial_info["fit_mse"]
            if "fit_mse" in temporal_info:
                info["temporal_component_mse"] = temporal_info["fit_mse"]
            for key in (
                "spacetime_order",
                "fork_speed_shape",
                "fork_speed_component_x",
                "fork_speed_component_t",
                "fork_speed_q_slope_t",
                "fork_speed_temporal_accepted",
                "fork_speed_temporal_baseline_mse",
                "fork_speed_temporal_factor",
                "fork_speed_crossed_space_steps",
                "fork_speed_crossed_time_steps",
                "fork_speed_crossed_rejected_space_steps",
                "fork_speed_crossed_rejected_time_steps",
                "fit_history",
            ):
                if key in temporal_info:
                    info[key] = temporal_info[key]
            info.update(
                _spacetime_speed_timing_fit_info(
                    speed_spacetime,
                    T,
                    firing_rate_xt.T,
                    resolution,
                    resolution_time,
                    time_grid,
                    T0=T0,
                    refine_edge_kb=refine_edge_kb,
                    perQ=perQ,
                )
            )
            if verbose and "fit_mse" in info:
                print(f"[rmap]  final deterministic MSE: {info['fit_mse']:.4e}")
            return _RmapInfoArray(np.asarray(speed_spacetime, dtype=float), info=info).T
        return speed_spacetime.T

    speed_space = map_timing_firing_spatial_speed(
        {"replication_timing": T, "firing_rate": _speed_fit_bin_hazard(
            firing_rate_space, resolution, _firing_rate_is_density)},
        resolution=resolution,
        T0=T0,
        neighbourhood_range=neighbourhood_range,
        fit_step=space_fit_step,
        maxiter=0,
        speed_bounds=speed_bounds,
        initial_fork_speed=initial_fork_speed,
        target_shape="best" if target_shape == "best" else "free",
        weibull_shape=weibull_shape,
        weibull_delay=weibull_delay,
        smoothw=smoothw,
        error=error,
        refine_edge_kb=refine_edge_kb,
        perQ=perQ,
        n_bins=n_bins,
        verbose=verbose,
        print_every=print_every,
    )
    spatial_seed_info = dict(getattr(speed_space, "_rmap_info", {}) or {})
    spatial_seed_mse = spatial_seed_info.get("fit_mse")
    temporal, _ = _temporal_profile_from_spatial_fit(
        T,
        speed_space,
        resolution_time=resolution_time,
        time_grid=time_grid,
        T0=T0,
        temporal_smoothw=temporal_smoothw,
        floor=0.0,
    )
    scale = float(np.nanmean(speed_space[speed_space > 0])) if np.any(speed_space > 0) else 1.0
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    speed_spacetime = np.outer(temporal, speed_space / scale)
    low, high = _speed_bounds_from_profile(speed_space, speed_bounds=speed_bounds)
    speed_spacetime = np.clip(speed_spacetime, low, high)
    if refine_verbose is None:
        refine_verbose = bool(verbose)
    if target_shape == "best":
        speed_spacetime = _project_best_spacetime_speed_shape(
            speed_spacetime,
            time_axis=time_grid,
            speed_bounds=(low, high),
        )

    if update_mode == "factorized":
        def refine_factorized_phase(field, phase):
            is_space = phase == "space"
            phase_maxiter = space_maxiter if is_space else temporal_maxiter
            phase_step = space_fit_step if is_space else temporal_fit_step
            if phase_maxiter <= 0:
                return field
            if refine_verbose:
                label = "space" if is_space else "time"
                print(f"\nRefining factorized {label} component of fork speed...")
            return _refine_spacetime_speed(
                field,
                T,
                firing_rate=_speed_fit_bin_hazard(firing_rate_xt.T, resolution, _firing_rate_is_density),
                resolution=resolution,
                resolution_time=resolution_time,
                time_grid=time_grid,
                T0=T0,
                spacetime_fit_step=phase_step,
                spacetime_space_fit_step=phase_step if is_space else 0.0,
                spacetime_time_fit_step=0.0 if is_space else phase_step,
                refine_maxiter=phase_maxiter,
                spacetime_sim_number=temporal_sim_number,
                spacetime_update="factorized",
                spacetime_kernel_t=spacetime_kernel_t,
                speed_bounds=(low, high),
                refine_ratio_bounds=refine_ratio_bounds,
                refine_edge_kb=refine_edge_kb,
                refine_seed=refine_seed,
                refine_verbose=refine_verbose,
                perQ=perQ,
                error=error,
            )

        if order == "time_then_space":
            speed_spacetime = refine_factorized_phase(speed_spacetime, "time")
            speed_spacetime = refine_factorized_phase(speed_spacetime, "space")
        else:
            speed_spacetime = refine_factorized_phase(speed_spacetime, "space")
            speed_spacetime = refine_factorized_phase(speed_spacetime, "time")
        if target_shape == "exponential":
            speed_spacetime = _project_exponential_speed_shape(
                speed_spacetime,
                "spacetime",
                resolution=resolution,
                time_axis=time_grid,
                speed_bounds=(low, high),
            )
        if target_shape == "weibull":
            speed_spacetime = _project_weibull_speed_shape(
                speed_spacetime,
                "spacetime",
                resolution=resolution,
                time_axis=time_grid,
                speed_bounds=(low, high),
                weibull_shape=weibull_shape,
                weibull_delay=weibull_delay,
            )
        if target_shape == "best":
            speed_spacetime = _project_best_spacetime_speed_shape(
                speed_spacetime,
                time_axis=time_grid,
                speed_bounds=(low, high),
            )
            info = dict(getattr(speed_spacetime, "_rmap_info", {}) or {})
            info["time_grid"] = np.asarray(time_grid, dtype=float).copy()
            info["speed_form"] = "spacetime"
            if spatial_seed_mse is not None and np.isfinite(spatial_seed_mse):
                info["spatial_seed_mse"] = float(spatial_seed_mse)
            info.update(
                _spacetime_speed_timing_fit_info(
                    speed_spacetime,
                    T,
                    firing_rate_xt.T,
                    resolution,
                    resolution_time,
                    time_grid,
                    T0=T0,
                    refine_edge_kb=refine_edge_kb,
                    perQ=perQ,
                )
            )
            if verbose and "fit_mse" in info:
                print(f"[rmap]  final deterministic MSE: {info['fit_mse']:.4e}")
            return _RmapInfoArray(np.asarray(speed_spacetime, dtype=float), info=info).T
        return speed_spacetime.T

    if refine_verbose:
        print("\nRefining mixed space-time fork speed...")
    speed_spacetime = _refine_spacetime_speed(
        speed_spacetime,
        T,
        firing_rate=_speed_fit_bin_hazard(firing_rate_xt.T, resolution, _firing_rate_is_density),
        resolution=resolution,
        resolution_time=resolution_time,
        time_grid=time_grid,
        T0=T0,
        spacetime_fit_step=spacetime_fit_step,
        spacetime_space_fit_step=spacetime_space_fit_step,
        spacetime_time_fit_step=spacetime_time_fit_step,
        refine_maxiter=max(space_maxiter, temporal_maxiter),
        spacetime_sim_number=spacetime_sim_number,
        spacetime_update=spacetime_update,
        spacetime_kernel_t=spacetime_kernel_t,
        speed_bounds=(low, high),
        refine_ratio_bounds=refine_ratio_bounds,
        refine_edge_kb=refine_edge_kb,
        refine_seed=refine_seed,
        refine_verbose=refine_verbose,
        perQ=perQ,
        error=error,
    )
    if target_shape == "exponential":
        speed_spacetime = _project_exponential_speed_shape(
            speed_spacetime,
            "spacetime",
            resolution=resolution,
            time_axis=time_grid,
            speed_bounds=(low, high),
        )
    if target_shape == "weibull":
        speed_spacetime = _project_weibull_speed_shape(
            speed_spacetime,
            "spacetime",
            resolution=resolution,
            time_axis=time_grid,
            speed_bounds=(low, high),
            weibull_shape=weibull_shape,
            weibull_delay=weibull_delay,
        )
    if target_shape == "best":
        speed_spacetime = _project_best_spacetime_speed_shape(
            speed_spacetime,
            time_axis=time_grid,
            speed_bounds=(low, high),
        )
        info = dict(getattr(speed_spacetime, "_rmap_info", {}) or {})
        info["time_grid"] = np.asarray(time_grid, dtype=float).copy()
        info["speed_form"] = "spacetime"
        if spatial_seed_mse is not None and np.isfinite(spatial_seed_mse):
            info["spatial_seed_mse"] = float(spatial_seed_mse)
        info.update(
            _spacetime_speed_timing_fit_info(
                speed_spacetime,
                T,
                firing_rate_xt.T,
                resolution,
                resolution_time,
                time_grid,
                T0=T0,
                refine_edge_kb=refine_edge_kb,
                perQ=perQ,
            )
        )
        if verbose and "fit_mse" in info:
            print(f"[rmap]  final deterministic MSE: {info['fit_mse']:.4e}")
        return _RmapInfoArray(np.asarray(speed_spacetime, dtype=float), info=info).T
    return speed_spacetime.T


def _expand_firing_to_spacetime(field, n_time, n_positions, floor=0.0):
    arr = np.asarray(field, dtype=float)
    if arr.size == 1:
        value = float(arr.reshape(-1)[0])
        if not np.isfinite(value) or value < 0:
            raise ValueError("initial_firing_rate scalar must be finite and >= 0.")
        return np.full((n_time, n_positions), max(value, floor), dtype=float)
    if arr.ndim == 1:
        if arr.size != n_positions:
            raise ValueError("initial_firing_rate (n,) must match replication_timing length.")
        if np.any(~np.isfinite(arr)) or np.any(arr < 0):
            raise ValueError("initial_firing_rate must contain finite values >= 0.")
        return np.tile(np.maximum(arr, floor), (n_time, 1))
    if arr.ndim == 2:
        if arr.shape == (1, n_time):
            values_t = arr.reshape(-1)
            if np.any(~np.isfinite(values_t)) or np.any(values_t < 0):
                raise ValueError("initial_firing_rate (1,T) must contain finite values >= 0.")
            return np.tile(np.maximum(values_t, floor)[:, None], (1, n_positions))
        if arr.shape == (n_positions, 1):
            values_x = arr.reshape(-1)
            if np.any(~np.isfinite(values_x)) or np.any(values_x < 0):
                raise ValueError("initial_firing_rate (n,1) must contain finite values >= 0.")
            return np.tile(np.maximum(values_x, floor), (n_time, 1))
        if arr.shape == (n_positions, n_time):
            if np.any(~np.isfinite(arr)) or np.any(arr < 0):
                raise ValueError("initial_firing_rate (n,T) must contain finite values >= 0.")
            return np.maximum(arr.T.astype(float, copy=True), floor)
    raise ValueError("initial_firing_rate must be scalar, (n,), (1,T), (n,1), or (n,T).")


def _expand_speed_to_spacetime(field, n_time, n_positions):
    arr = np.asarray(field, dtype=float)
    if arr.size == 1:
        value = float(arr.reshape(-1)[0])
        if not np.isfinite(value) or value <= 0:
            raise ValueError("initial_fork_speed scalar must be finite and > 0.")
        return np.full((n_time, n_positions), value, dtype=float)
    if arr.ndim == 1:
        if arr.size != n_positions:
            raise ValueError("initial_fork_speed (n,) must match replication_timing length.")
        if np.any(~np.isfinite(arr)) or np.any(arr <= 0):
            raise ValueError("initial_fork_speed must contain finite positive values.")
        return np.tile(arr, (n_time, 1))
    if arr.ndim == 2:
        if arr.shape == (1, n_time):
            values_t = arr.reshape(-1)
            if np.any(~np.isfinite(values_t)) or np.any(values_t <= 0):
                raise ValueError("initial_fork_speed (1,T) must contain finite positive values.")
            return np.tile(values_t[:, None], (1, n_positions))
        if arr.shape == (n_positions, 1):
            values_x = arr.reshape(-1)
            if np.any(~np.isfinite(values_x)) or np.any(values_x <= 0):
                raise ValueError("initial_fork_speed (n,1) must contain finite positive values.")
            return np.tile(values_x, (n_time, 1))
        if arr.shape == (n_positions, n_time):
            if np.any(~np.isfinite(arr)) or np.any(arr <= 0):
                raise ValueError("initial_fork_speed (n,T) must contain finite positive values.")
            return arr.T.astype(float, copy=True)
    raise ValueError("initial_fork_speed must be scalar, (n,), (1,T), (n,1), or (n,T).")


def _time_axis_for_timing(T, resolution_time=5.0, time_grid=None, T0=None):
    finite = np.isfinite(T)
    if T0 is None:
        T0 = float(np.nanmin(T[finite])) if np.any(finite) else 0.0
    T0 = float(T0)
    if not np.isfinite(T0):
        raise ValueError("T0 must be finite when supplied.")
    relative_timing = np.maximum(np.asarray(T, dtype=float) - T0, 0.0)
    return _time_axis_from_timing(
        relative_timing,
        resolution_time=resolution_time,
        time_grid=time_grid,
    )


def _initial_positive_speed_value(initial_fork_speed, default=1.4):
    arr = np.asarray(initial_fork_speed, dtype=float)
    positive = arr[np.isfinite(arr) & (arr > 0)]
    if positive.size:
        return float(np.nanmedian(positive))
    return float(default)


def _initial_spatial_speed_profile(initial_fork_speed, n_positions, default=1.4):
    arr = np.asarray(initial_fork_speed, dtype=float)
    if arr.size == 1:
        return np.full(n_positions, _initial_positive_speed_value(arr, default=default), dtype=float)
    if arr.ndim == 1:
        if arr.size != n_positions:
            raise ValueError("initial_fork_speed (n,) must match replication_timing length.")
        if np.any(~np.isfinite(arr)) or np.any(arr <= 0):
            raise ValueError("initial_fork_speed must contain finite positive values.")
        return arr.astype(float, copy=True)
    if arr.ndim == 2:
        if arr.shape == (n_positions, 1):
            values = arr.reshape(-1)
            if np.any(~np.isfinite(values)) or np.any(values <= 0):
                raise ValueError("initial_fork_speed (n,1) must contain finite positive values.")
            return values.astype(float, copy=True)
        if arr.shape[0] == n_positions and arr.shape[1] > 1:
            values = np.nanmean(arr, axis=1)
            if np.any(~np.isfinite(values)) or np.any(values <= 0):
                raise ValueError("initial_fork_speed (n,T) must contain finite positive values.")
            return values.astype(float, copy=True)
        if arr.shape[0] == 1:
            value = _initial_positive_speed_value(arr, default=default)
            return np.full(n_positions, value, dtype=float)
    raise ValueError("initial_fork_speed must be scalar, (n,), (n,1), (1,T), or (n,T).")


def _initial_temporal_speed_profile(initial_fork_speed, n_time, default=1.4):
    arr = np.asarray(initial_fork_speed, dtype=float)
    if arr.size == 1:
        return np.full(n_time, _initial_positive_speed_value(arr, default=default), dtype=float)
    if arr.ndim == 1:
        value = _initial_positive_speed_value(arr, default=default)
        return np.full(n_time, value, dtype=float)
    if arr.ndim == 2:
        if arr.shape == (1, n_time):
            values = arr.reshape(-1)
            if np.any(~np.isfinite(values)) or np.any(values <= 0):
                raise ValueError("initial_fork_speed (1,T) must contain finite positive values.")
            return values.astype(float, copy=True)
        if arr.shape[1] == n_time and arr.shape[0] > 1:
            values = np.nanmean(arr, axis=0)
            if np.any(~np.isfinite(values)) or np.any(values <= 0):
                raise ValueError("initial_fork_speed (n,T) must contain finite positive values.")
            return values.astype(float, copy=True)
        value = _initial_positive_speed_value(arr, default=default)
        return np.full(n_time, value, dtype=float)
    raise ValueError("initial_fork_speed must be scalar, (n,), (1,T), (n,1), or (n,T).")


def _fit_crop(n, resolution, perQ=False):
    lm = int(1000 / resolution)
    if perQ:
        return slice(None)
    return slice(lm, -lm) if 2 * lm < n else slice(None)


def _fill_nonperiodic_profile_edges(profile, crop, perQ=False):
    if perQ or crop == slice(None):
        return profile
    n = profile.size
    start = 0 if crop.start is None else int(crop.start)
    stop = n if crop.stop is None else int(crop.stop)
    if stop < 0:
        stop = n + stop
    start = max(0, min(start, n))
    stop = max(0, min(stop, n))
    if stop <= start:
        return profile
    if start > 0:
        profile[:start] = profile[start]
    if stop < n:
        profile[stop:] = profile[stop - 1]
    return profile


def map_timing_firing_constant_speed_simulated(
    source,
    resolution=1.0,
    resolution_x=None,
    resolution_time=5.0,
    time_grid=None,
    T0=None,
    fit_step=0.6,
    maxiter=10,
    sim_number=100,
    speed_bounds=None,
    initial_fork_speed=1.4,
    target_shape="best",
    error=None,
    refine_edge_kb=1000,
    refine_seed=None,
    perQ=False,
    n_bins=None,
    verbose=True,
    print_every=1,
):
    """Infer one fork speed from timing and time-dependent firing-rate inputs."""
    if perQ:
        return _periodic_speed_inverse(
            source, "constant", resolution, resolution_x, resolution_time, time_grid,
            T0, initial_fork_speed, speed_bounds, maxiter, verbose,
        )
    resolution = _normalise_resolution_x(resolution, resolution_x=resolution_x)
    error = _normalise_error_threshold(error)
    target_shape = _canonical_speed_target_shape(target_shape)
    T, firing_rate = _validate_timing_and_firing_for_speed(source)
    if n_bins is not None and int(n_bins) != T.size:
        raise ValueError("n_bins must match the length of replication_timing.")

    if target_shape == "best":
        return _fit_constant_speed_with_general_firing(
            T,
            firing_rate,
            resolution=resolution,
            resolution_time=resolution_time,
            time_grid=time_grid,
            T0=T0,
            fit_step=fit_step,
            maxiter=maxiter,
            speed_bounds=speed_bounds,
            initial_fork_speed=initial_fork_speed,
            refine_edge_kb=refine_edge_kb,
            perQ=perQ,
            error=error,
            verbose=verbose,
            print_every=print_every,
        )

    speed = _initial_positive_speed_value(initial_fork_speed)
    low, high = _speed_bounds_from_profile([speed], speed_bounds=speed_bounds)
    speed = float(np.clip(speed, low, high))
    crop = _fit_crop(T.size, resolution, perQ=perQ)
    maxiter = int(maxiter)
    print_every = max(1, int(print_every))
    best_speed = speed
    best_mse = np.inf
    start_time = time.time()

    for it in range(1, maxiter + 1):
        seed = None if refine_seed is None else int(refine_seed) + it - 1
        predicted = rsim(
            firing_rate=firing_rate,
            fork_speed=speed,
            sim_number=sim_number,
            resolution_space=resolution,
            resolution_time=resolution_time,
            n_bins=T.size,
            perQ=perQ,
            seed=seed,
            verbose=False,
            retain_rep_times_per_sim=False,
            retain_inter_origin_distances=False,
        )["replication_timing"]
        observed = T[crop]
        predicted = predicted[crop]
        ok = (
            np.isfinite(observed)
            & np.isfinite(predicted)
            & (observed > 0)
            & (predicted > 0)
        )
        if not np.any(ok):
            break
        mse = mean_squared_error(observed[ok], predicted[ok])
        if mse < best_mse:
            best_mse = mse
            best_speed = speed
        ratio = float(np.nanmedian(predicted[ok] / observed[ok]))
        if not np.isfinite(ratio) or ratio <= 0:
            break
        speed = float(np.clip(speed * ratio ** fit_step, low, high))
        done = it == maxiter or _error_reached(mse, error)
        if _fit_progress_enabled(verbose) and (it % print_every == 0 or done):
            etaferr(it=it, maxiter=maxiter, start_time=start_time, err=mse, done=done)
        if done and it != maxiter:
            break

    return best_speed


def map_timing_firing_spatial_speed_simulated(
    source,
    resolution=1.0,
    resolution_x=None,
    resolution_time=5.0,
    time_grid=None,
    T0=None,
    fit_step=0.4,
    maxiter=10,
    space_fit_step=None,
    space_maxiter=None,
    sim_number=100,
    speed_bounds=None,
    initial_fork_speed=1.4,
    target_shape="best",
    weibull_shape=2.0,
    weibull_delay=0.0,
    error=None,
    smoothw=100,
    refine_edge_kb=1000,
    refine_seed=None,
    perQ=False,
    n_bins=None,
    verbose=True,
    print_every=1,
):
    """Infer a spatial fork-speed profile using stochastic timing refinement."""
    if perQ:
        return _periodic_speed_inverse(
            source, "space", resolution, resolution_x, resolution_time, time_grid,
            T0, initial_fork_speed, speed_bounds,
            maxiter if space_maxiter is None else space_maxiter, verbose,
        )
    resolution = _normalise_resolution_x(resolution, resolution_x=resolution_x)
    target_shape = _canonical_speed_target_shape(target_shape)
    error = _normalise_error_threshold(error)
    if space_fit_step is not None:
        fit_step = space_fit_step
    if space_maxiter is not None:
        maxiter = space_maxiter
    T, firing_rate = _validate_timing_and_firing_for_speed(source)
    if n_bins is not None and int(n_bins) != T.size:
        raise ValueError("n_bins must match the length of replication_timing.")

    if target_shape == "best":
        return _fit_spatial_speed_with_general_firing(
            T,
            firing_rate,
            resolution=resolution,
            resolution_time=resolution_time,
            time_grid=time_grid,
            T0=T0,
            fit_step=fit_step,
            maxiter=maxiter,
            speed_bounds=speed_bounds,
            initial_fork_speed=initial_fork_speed,
            smoothw=smoothw,
            refine_edge_kb=refine_edge_kb,
            perQ=perQ,
            error=error,
            verbose=verbose,
            print_every=print_every,
        )

    speed = _initial_spatial_speed_profile(initial_fork_speed, T.size)
    if target_shape == "best" and speed_bounds is None:
        low, high = _direct_speed_shape_bounds(speed, speed_bounds=None)
    else:
        low, high = _speed_bounds_from_profile(speed, speed_bounds=speed_bounds)
    speed = np.clip(speed, low, high)
    crop = _fit_crop(T.size, resolution, perQ=perQ)
    speed = _fill_nonperiodic_profile_edges(speed, crop, perQ=perQ)
    maxiter = int(maxiter)
    print_every = max(1, int(print_every))
    best_speed = speed.copy()
    best_mse = np.inf
    start_time = time.time()

    for it in range(1, maxiter + 1):
        seed = None if refine_seed is None else int(refine_seed) + it - 1
        predicted = rsim(
            firing_rate=firing_rate,
            fork_speed=speed,
            sim_number=sim_number,
            resolution_space=resolution,
            resolution_time=resolution_time,
            n_bins=T.size,
            perQ=perQ,
            seed=seed,
            verbose=False,
            retain_rep_times_per_sim=False,
            retain_inter_origin_distances=False,
        )["replication_timing"]
        observed = T[crop]
        predicted_crop = predicted[crop]
        ok_crop = (
            np.isfinite(observed)
            & np.isfinite(predicted_crop)
            & (observed > 0)
            & (predicted_crop > 0)
        )
        if not np.any(ok_crop):
            break
        mse = mean_squared_error(observed[ok_crop], predicted_crop[ok_crop])
        if mse < best_mse:
            best_mse = mse
            best_speed = speed.copy()
        ratio = np.ones(T.size, dtype=float)
        ratio_view = ratio[crop]
        ratio_view[ok_crop] = predicted_crop[ok_crop] / observed[ok_crop]
        speed *= ratio ** fit_step
        if smoothw is not None:
            speed = smoothf(speed, smoothw)
        speed = np.clip(speed, low, high)
        speed = _fill_nonperiodic_profile_edges(speed, crop, perQ=perQ)
        done = it == maxiter or _error_reached(mse, error)
        if _fit_progress_enabled(verbose) and (it % print_every == 0 or done):
            etaferr(it=it, maxiter=maxiter, start_time=start_time, err=mse, done=done)
        if done and it != maxiter:
            break

    if target_shape == "exponential":
        return _project_exponential_speed_shape(
            best_speed,
            "space",
            resolution=resolution,
            speed_bounds=(low, high),
        )
    if target_shape == "weibull":
        return _project_weibull_speed_shape(
            best_speed,
            "space",
            resolution=resolution,
            speed_bounds=(low, high),
            weibull_shape=weibull_shape,
            weibull_delay=weibull_delay,
        )
    if target_shape == "best":
        return _RmapInfoArray(
            best_speed,
            info={
                "target_shape": "best",
                "fork_speed_shape": "site-specific v(x)",
                "fork_speed_bounds": (float(low), float(high)),
                "fit_mse": None if not np.isfinite(best_mse) else float(best_mse),
            },
        )
    return best_speed


def map_timing_firing_temporal_speed_simulated(
    source,
    resolution=1.0,
    resolution_x=None,
    resolution_time=5.0,
    time_grid=None,
    T0=None,
    target_shape="best",
    weibull_shape=2.0,
    weibull_delay=0.0,
    temporal_smoothw=None,
    fit_step=None,
    maxiter=None,
    temporal_fit_step=None,
    temporal_maxiter=None,
    temporal_sim_number=100,
    speed_bounds=None,
    initial_fork_speed=1.4,
    refine_ratio_bounds=(0.25, 4.0),
    refine_edge_kb=1000,
    refine_seed=None,
    refine_verbose=None,
    error=None,
    perQ=False,
    n_bins=None,
    verbose=True,
    print_every=1,
    _firing_rate_is_density=False,
    **kwargs,
):
    """Infer a time-only fork-speed profile from timing and general firing."""
    if perQ:
        return _periodic_speed_inverse(
            source, "time", resolution, resolution_x, resolution_time, time_grid,
            T0, initial_fork_speed, speed_bounds,
            maxiter if temporal_maxiter is None else temporal_maxiter, verbose,
        )
    if refine_verbose is None:
        refine_verbose = bool(verbose)
    del print_every, kwargs
    resolution = _normalise_resolution_x(resolution, resolution_x=resolution_x)
    target_shape = _canonical_speed_target_shape(target_shape)
    if temporal_fit_step is None:
        temporal_fit_step = 0.003 if fit_step is None else fit_step
    if temporal_maxiter is None:
        temporal_maxiter = 10 if maxiter is None else maxiter
    T, firing_rate = _validate_timing_and_firing_for_speed(source)
    if n_bins is not None and int(n_bins) != T.size:
        raise ValueError("n_bins must match the length of replication_timing.")
    time_axis = _time_axis_for_speed_fit(
        T,
        firing_rate,
        resolution_time=resolution_time,
        time_grid=time_grid,
        T0=T0,
    )
    if target_shape == "best":
        speed_time = _fit_exponential_speed_shape(
            T,
            firing_rate=firing_rate,
            target_form="time",
            resolution=resolution,
            resolution_time=resolution_time,
            time_axis=time_axis,
            T0=T0,
            initial_fork_speed=initial_fork_speed,
            speed_bounds=speed_bounds,
            fit_step=temporal_fit_step,
            maxiter=temporal_maxiter,
            refine_edge_kb=refine_edge_kb,
            refine_seed=refine_seed,
            refine_verbose=refine_verbose,
            perQ=perQ,
            error=error,
        )
        return _retag_speed_shape(
            speed_time,
            "best",
            fork_speed_shape="exponential v(t)",
            fork_speed_component_t="v0 * exp(b_t * T)",
        )
    speed_time = _initial_temporal_speed_profile(initial_fork_speed, time_axis.size)
    if temporal_smoothw is not None:
        speed_time = smoothf(speed_time, temporal_smoothw)
    low, high = _speed_bounds_from_profile(speed_time, speed_bounds=speed_bounds)
    speed_time = np.clip(speed_time, low, high)
    if refine_verbose:
        print("\nRefining time-dependent fork speed...")
    speed_time = _refine_temporal_speed(
        speed_time.reshape(1, -1),
        T,
        firing_rate=_speed_fit_bin_hazard(firing_rate, resolution, _firing_rate_is_density),
        resolution=resolution,
        resolution_time=resolution_time,
        time_grid=time_axis,
        T0=T0,
        temporal_fit_step=temporal_fit_step,
        temporal_maxiter=temporal_maxiter,
        temporal_sim_number=temporal_sim_number,
        speed_bounds=(low, high),
        refine_ratio_bounds=refine_ratio_bounds,
        refine_edge_kb=refine_edge_kb,
        refine_seed=refine_seed,
        refine_verbose=refine_verbose,
        perQ=perQ,
        error=error,
    )
    if target_shape == "exponential":
        speed_time = _project_exponential_speed_shape(
            speed_time,
            "time",
            time_axis=time_axis,
            speed_bounds=(low, high),
        )
    if target_shape == "best":
        speed_time = _project_exponential_speed_shape(
            speed_time,
            "time",
            time_axis=time_axis,
            speed_bounds=(low, high),
        )
        speed_time = _retag_speed_shape(
            speed_time,
            "best",
            fork_speed_shape="exponential v(t)",
            fork_speed_component_t="v0 * exp(b_t * T)",
        )
    if target_shape == "weibull":
        speed_time = _project_weibull_speed_shape(
            speed_time,
            "time",
            time_axis=time_axis,
            speed_bounds=(low, high),
            weibull_shape=weibull_shape,
            weibull_delay=weibull_delay,
        )
    return speed_time


def map_timing_firing_spacetime_speed_simulated(
    source,
    resolution=1.0,
    resolution_x=None,
    resolution_time=5.0,
    time_grid=None,
    T0=None,
    target_shape="best",
    weibull_shape=2.0,
    weibull_delay=0.0,
    neighbourhood_range=2000,
    fit_step=None,
    maxiter=None,
    space_fit_step=None,
    space_maxiter=None,
    temporal_smoothw=None,
    temporal_fit_step=None,
    temporal_maxiter=None,
    temporal_sim_number=None,
    spacetime_fit_step=None,
    spacetime_space_fit_step=None,
    spacetime_time_fit_step=None,
    spacetime_sim_number=None,
    spacetime_update="factorized",
    spacetime_order="crossed",
    spacetime_kernel_t=1.0,
    speed_bounds=None,
    initial_fork_speed=1.4,
    smoothw=100,
    refine_ratio_bounds=(0.25, 4.0),
    refine_edge_kb=1000,
    refine_seed=None,
    refine_verbose=None,
    error=None,
    perQ=False,
    n_bins=None,
    verbose=True,
    print_every=1,
    _firing_rate_is_density=False,
    **kwargs,
):
    """Infer a space-time fork-speed field from timing and general firing."""
    if perQ:
        return _periodic_speed_inverse(
            source, "spacetime", resolution, resolution_x, resolution_time, time_grid,
            T0, initial_fork_speed, speed_bounds,
            max((value for value in (maxiter, space_maxiter, temporal_maxiter) if value is not None), default=20),
            verbose,
        )
    if refine_verbose is None:
        refine_verbose = bool(verbose)
    sim_number = kwargs.pop("sim_number", None)
    del kwargs
    resolution = _normalise_resolution_x(resolution, resolution_x=resolution_x)
    target_shape = _canonical_speed_target_shape(target_shape)
    update_mode = _canonical_spacetime_update(spacetime_update)
    order = _canonical_spacetime_order(spacetime_order)
    (
        space_fit_step,
        space_maxiter,
        temporal_fit_step,
        temporal_maxiter,
        temporal_sim_number,
    ) = _resolve_spacetime_phase_controls(
        fit_step=fit_step,
        maxiter=maxiter,
        sim_number=sim_number,
        space_fit_step=space_fit_step,
        space_maxiter=space_maxiter,
        temporal_fit_step=temporal_fit_step,
        temporal_maxiter=temporal_maxiter,
        temporal_sim_number=temporal_sim_number,
        default_space_fit_step=0.4,
        default_space_maxiter=10,
        default_temporal_fit_step=0.003,
    )
    if spacetime_space_fit_step is not None:
        space_fit_step = float(spacetime_space_fit_step)
    if spacetime_time_fit_step is not None:
        temporal_fit_step = float(spacetime_time_fit_step)
    if spacetime_sim_number is None:
        spacetime_sim_number = temporal_sim_number
    if spacetime_fit_step is None:
        spacetime_fit_step = 0.5 * (space_fit_step + temporal_fit_step)
    T, firing_rate = _validate_timing_and_firing_for_speed(source)
    if n_bins is not None and int(n_bins) != T.size:
        raise ValueError("n_bins must match the length of replication_timing.")
    time_axis = _time_axis_for_speed_fit(
        T,
        firing_rate,
        resolution_time=resolution_time,
        time_grid=time_grid,
        T0=T0,
    )
    source_kind = _firing_kind_for_speed_components(firing_rate, T.size)
    if update_mode == "factorized":
        speed_spacetime, bounds, spatial_info, temporal_info = _fit_spacetime_speed_components(
            {"replication_timing": T, "firing_rate": firing_rate},
            source_kind,
            resolution=resolution,
            resolution_time=resolution_time,
            time_axis=time_axis,
            T0=T0,
            target_shape=target_shape,
            temporal_smoothw=temporal_smoothw,
            neighbourhood_range=neighbourhood_range,
            space_fit_step=space_fit_step,
            space_maxiter=space_maxiter,
            temporal_fit_step=temporal_fit_step,
            temporal_maxiter=temporal_maxiter,
            temporal_sim_number=temporal_sim_number,
            speed_bounds=speed_bounds,
            initial_fork_speed=initial_fork_speed,
            weibull_shape=weibull_shape,
            weibull_delay=weibull_delay,
            smoothw=smoothw,
            perQ=perQ,
            n_bins=n_bins,
            refine_ratio_bounds=refine_ratio_bounds,
            refine_edge_kb=refine_edge_kb,
            refine_seed=refine_seed,
            refine_verbose=refine_verbose,
            error=error,
            verbose=verbose,
            print_every=print_every,
            order=order,
            _firing_rate_is_density=_firing_rate_is_density,
        )
        speed_spacetime = _project_spacetime_speed_target_shape(
            speed_spacetime,
            target_shape,
            resolution=resolution,
            time_axis=time_axis,
            speed_bounds=bounds,
            weibull_shape=weibull_shape,
            weibull_delay=weibull_delay,
        )
        if target_shape == "best":
            info = dict(getattr(speed_spacetime, "_rmap_info", {}) or {})
            info["time_grid"] = np.asarray(time_axis, dtype=float).copy()
            info["speed_form"] = "spacetime"
            if "fit_mse" in spatial_info:
                info["spatial_component_mse"] = spatial_info["fit_mse"]
            if "fit_mse" in temporal_info:
                info["temporal_component_mse"] = temporal_info["fit_mse"]
            for key in (
                "spacetime_order",
                "fork_speed_shape",
                "fork_speed_component_x",
                "fork_speed_component_t",
                "fork_speed_q_slope_t",
                "fork_speed_temporal_accepted",
                "fork_speed_temporal_baseline_mse",
                "fork_speed_temporal_factor",
                "fork_speed_crossed_space_steps",
                "fork_speed_crossed_time_steps",
                "fork_speed_crossed_rejected_space_steps",
                "fork_speed_crossed_rejected_time_steps",
                "fit_history",
            ):
                if key in temporal_info:
                    info[key] = temporal_info[key]
            info.update(
                _spacetime_speed_timing_fit_info(
                    speed_spacetime,
                    T,
                    firing_rate,
                    resolution,
                    resolution_time,
                    time_axis,
                    T0=T0,
                    refine_edge_kb=refine_edge_kb,
                    perQ=perQ,
                )
            )
            return _RmapInfoArray(np.asarray(speed_spacetime, dtype=float), info=info).T
        return speed_spacetime.T

    speed_spacetime = _expand_speed_to_spacetime(initial_fork_speed, time_axis.size, T.size)
    low, high = _speed_bounds_from_profile(speed_spacetime, speed_bounds=speed_bounds)
    speed_spacetime = np.clip(speed_spacetime, low, high)
    if update_mode == "factorized":
        def refine_factorized_phase(field, phase):
            is_space = phase == "space"
            phase_maxiter = space_maxiter if is_space else temporal_maxiter
            phase_step = space_fit_step if is_space else temporal_fit_step
            if phase_maxiter <= 0:
                return field
            if refine_verbose:
                label = "space" if is_space else "time"
                print(f"\nRefining factorized {label} component of fork speed...")
            return _refine_spacetime_speed(
                field,
                T,
                firing_rate=_speed_fit_bin_hazard(firing_rate, resolution, _firing_rate_is_density),
                resolution=resolution,
                resolution_time=resolution_time,
                time_grid=time_axis,
                T0=T0,
                spacetime_fit_step=phase_step,
                spacetime_space_fit_step=phase_step if is_space else 0.0,
                spacetime_time_fit_step=0.0 if is_space else phase_step,
                refine_maxiter=phase_maxiter,
                spacetime_sim_number=temporal_sim_number,
                spacetime_update="factorized",
                spacetime_kernel_t=spacetime_kernel_t,
                speed_bounds=(low, high),
                refine_ratio_bounds=refine_ratio_bounds,
                refine_edge_kb=refine_edge_kb,
                refine_seed=refine_seed,
                refine_verbose=refine_verbose,
                perQ=perQ,
                error=error,
            )

        phases = (
            ("time", "space")
            if order == "time_then_space"
            else ("space", "time")
        )
        for phase in phases:
            speed_spacetime = refine_factorized_phase(speed_spacetime, phase)
        if target_shape == "exponential":
            speed_spacetime = _project_exponential_speed_shape(
                speed_spacetime,
                "spacetime",
                resolution=resolution,
                time_axis=time_axis,
                speed_bounds=(low, high),
            )
        if target_shape == "best":
            speed_spacetime = _project_best_spacetime_speed_shape(
                speed_spacetime,
                time_axis=time_axis,
                speed_bounds=(low, high),
            )
            info = dict(getattr(speed_spacetime, "_rmap_info", {}) or {})
            info["time_grid"] = np.asarray(time_axis, dtype=float).copy()
            info["speed_form"] = "spacetime"
            info.update(
                _spacetime_speed_timing_fit_info(
                    speed_spacetime,
                    T,
                    firing_rate,
                    resolution,
                    resolution_time,
                    time_axis,
                    T0=T0,
                    refine_edge_kb=refine_edge_kb,
                    perQ=perQ,
                )
            )
            return _RmapInfoArray(np.asarray(speed_spacetime, dtype=float), info=info).T
        if target_shape == "weibull":
            speed_spacetime = _project_weibull_speed_shape(
                speed_spacetime,
                "spacetime",
                resolution=resolution,
                time_axis=time_axis,
                speed_bounds=(low, high),
                weibull_shape=weibull_shape,
                weibull_delay=weibull_delay,
            )
        return speed_spacetime.T

    if refine_verbose:
        print("\nRefining mixed space-time fork speed...")
    speed_spacetime = _refine_spacetime_speed(
        speed_spacetime,
        T,
        firing_rate=_speed_fit_bin_hazard(firing_rate, resolution, _firing_rate_is_density),
        resolution=resolution,
        resolution_time=resolution_time,
        time_grid=time_axis,
        T0=T0,
        spacetime_fit_step=spacetime_fit_step,
        spacetime_space_fit_step=spacetime_space_fit_step,
        spacetime_time_fit_step=spacetime_time_fit_step,
        refine_maxiter=max(space_maxiter, temporal_maxiter),
        spacetime_sim_number=spacetime_sim_number,
        spacetime_update=spacetime_update,
        spacetime_kernel_t=spacetime_kernel_t,
        speed_bounds=(low, high),
        refine_ratio_bounds=refine_ratio_bounds,
        refine_edge_kb=refine_edge_kb,
        refine_seed=refine_seed,
        refine_verbose=refine_verbose,
        perQ=perQ,
        error=error,
    )
    if target_shape == "exponential":
        speed_spacetime = _project_exponential_speed_shape(
            speed_spacetime,
            "spacetime",
            resolution=resolution,
            time_axis=time_axis,
            speed_bounds=(low, high),
        )
    if target_shape == "best":
        speed_spacetime = _project_best_spacetime_speed_shape(
            speed_spacetime,
            time_axis=time_axis,
            speed_bounds=(low, high),
        )
        info = dict(getattr(speed_spacetime, "_rmap_info", {}) or {})
        info["time_grid"] = np.asarray(time_axis, dtype=float).copy()
        info["speed_form"] = "spacetime"
        info.update(
            _spacetime_speed_timing_fit_info(
                speed_spacetime,
                T,
                firing_rate,
                resolution,
                resolution_time,
                time_axis,
                T0=T0,
                refine_edge_kb=refine_edge_kb,
                perQ=perQ,
            )
        )
        return _RmapInfoArray(np.asarray(speed_spacetime, dtype=float), info=info).T
    if target_shape == "weibull":
        speed_spacetime = _project_weibull_speed_shape(
            speed_spacetime,
            "spacetime",
            resolution=resolution,
            time_axis=time_axis,
            speed_bounds=(low, high),
            weibull_shape=weibull_shape,
            weibull_delay=weibull_delay,
        )
    return speed_spacetime.T


def __getattr__(name):
    if name in {"rmap", "DataTypeMapper", "MappingSpec"}:
        from . import routing
        return getattr(routing, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def rfit(
    source="firing_rate",
    target="replication_timing",
    data=None,
    assumptions=None,
    method="auto",
    cell_line="H1",
    chr_number=1,
    fork_speed=1.4,
    resolution=1.0,
    resolution_x=None,
    saveQ=False,
    return_info=False,
    source_feature=None,
    d1=None,
    d2=None,
    **kwargs
):
    """Fit or map one replication feature to another.

    Prefer the same terms used by ``rmap``: ``source``, ``target``, ``data``,
    ``assumptions``, and ``method``. ``source`` may be a dictionary when a
    route needs multiple supplied features. If ``source`` is not a string or
    mapping, it is treated as the older data-array argument. The older
    ``source_feature``, ``d1``, and ``d2`` names are still accepted as aliases.
    """
    from .routing import _canonical_assumption, rmap

    if source_feature is not None:
        source = source_feature
    if d1 is not None:
        source = d1
    if d2 is not None:
        target = d2
    if data is None and not isinstance(source, (str, dict)):
        data = source
        source = "firing_rate"

    target = rmap.canonical(target)

    source_payload = None
    source_features = set()
    if isinstance(source, dict):
        if data is not None:
            raise ValueError("Pass either source={feature: data} or source='feature' with data=..., not both.")
        if not source:
            raise ValueError("rfit source dictionaries must contain at least one feature.")
        source_payload = {rmap.canonical(key): value for key, value in source.items()}
        missing_values = [key for key, value in source_payload.items() if value is None]
        if missing_values:
            raise ValueError(
                "rfit source dictionaries must provide data for every feature: "
                + ", ".join(missing_values)
            )
        source_features = set(source_payload)
    else:
        source_feature = rmap.canonical(source)
        source_features = {source_feature}

    if data is None and source_payload is None:
        source_name = mapt[source_feature]
        data = np.loadtxt(
            _resolve_data_path(f"{source_name}/{source_name}_{cell_line}_chr{chr_number}.txt"),
            dtype=float,
        )
    if source_payload is None:
        source_payload = source_feature

    merged_assumptions = {}
    if assumptions:
        merged_assumptions.update(
            {_canonical_assumption(key): value for key, value in assumptions.items()}
        )
    if "fork_speed" not in merged_assumptions and "fork_speed" not in source_features:
        merged_assumptions["fork_speed"] = fork_speed
    if resolution_x is not None:
        resolution = resolution_x
    if "resolution" not in merged_assumptions:
        merged_assumptions["resolution"] = resolution

    data2 = rmap(
        source=source_payload,
        target=target,
        data=None if isinstance(source_payload, dict) else data,
        assumptions=merged_assumptions,
        method=method,
        return_info=return_info,
        **kwargs,
    )

    if saveQ:
        values = data2["values"] if return_info else data2
        target_name = mapt[target]
        out = _resolve_data_path(
            f"{target_name}/{target_name}_{cell_line}_chr{chr_number}.txt",
            for_write=True,
        )
        os.makedirs(out.parent, exist_ok=True)
        np.savetxt(out, values)

    return data2
