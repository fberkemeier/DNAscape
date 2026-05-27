"""Fitting algorithms and fitting-oriented helpers."""

import os
import math
import time

import numpy as np

from .constants import mapt
from .io import _resolve_data_path, savecsv
from .mapping import _normalise_resolution_x, map_firing_timing
from .simulation import rsim
from .utils import etaferr, mean_squared_error, smoothf

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
    space_fit_step=None,
    space_maxiter=None,
    verbose=True,
    print_every=1,
    perQ=False,
    chr_number=1,
    save_errQ=False
):
    """Infer a spatial firing-rate profile from replication timing.

    ``fork_speed`` is the canonical argument name. ``fork_speeds`` is accepted
    as an older compatibility alias.
    """
    resolution = _normalise_resolution_x(resolution, resolution_x=resolution_x)
    if space_fit_step is not None:
        fit_step = space_fit_step
    if space_maxiter is not None:
        maxiter = space_maxiter

    mse_list = []

    T = np.asarray(observed_timing, dtype=float)
    n = len(T)

    if fork_speeds is not None:
        fork_speed = fork_speeds

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
    x = x0

    lm = int(1000 / resolution)
    if perQ:
        crop = slice(None)
    else:
        crop = slice(lm, -lm) if 2 * lm < n else slice(None)
    floor = 10.0 ** (-err_threshold)

    def update_firing_rate(observed, predicted, current):
        t_safe = np.maximum(observed, 1e-80)
        p_safe = np.maximum(predicted, 1e-80)
        ratio  = p_safe / t_safe

        updated = current * (ratio ** fit_step)
        updated = np.maximum(updated, floor)

        if deviation_threshold is not None:
            mse = np.abs(predicted - observed)
            close = mse <= deviation_threshold
            updated[close] = current[close]

        return updated

    # initial prediction
    y = map_firing_timing(
        x,
        fork_speed=fork_speed_initial,
        resolution=resolution,
        neighbourhood_range=neighbourhood_range,
        perQ=perQ,
    )
    mse = mean_squared_error(T[crop], y[crop])
    best_x = np.asarray(x, dtype=float).copy()
    best_mse = mse

    if stall_rate > 0. or variable_fork_speed:
        maxiter0 = 0
    else:
        maxiter0 = maxiter

    print_every = max(1, int(print_every))
    start_time = time.time()

    for it in range(1, maxiter0 + 1):
        x = update_firing_rate(T, y, x)
        y = map_firing_timing(
            x,
            fork_speed=fork_speed_initial,
            resolution=resolution,
            neighbourhood_range=neighbourhood_range,
            perQ=perQ
        )
        mse = mean_squared_error(T[crop], y[crop])
        if mse < best_mse:
            best_mse = mse
            best_x = np.asarray(x, dtype=float).copy()

        if verbose and (it % print_every == 0 or it == maxiter0):
            etaferr(it=it, maxiter=maxiter0, start_time=start_time, err=mse)
            mse_list.append(mse)
            

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
            pos = rates[rates > 0]
            rate_max = (10.0 * np.median(pos)) if pos.size else 1.0
    
        hist_mse = []
        T_pred = None
        best_rates = rates.copy()
        best_mse = np.inf
        start_time = time.time()
    
        for k in range(iterations):
            simres = rsim(
                firing_rate=rates,
                sim_number=sim_number,
                fork_speed=fork_speed,
                stall_rate=stall_rate,
                tau=tau,
                resolution_space=resolution,
                verbose=False,
                perQ=perQ
            )
            T_pred = simres['replication_timing']
    
            mse = T_pred - T_target
            mse = float(np.mean(mse**2))
            hist_mse.append(mse)
            if mse < best_mse:
                best_mse = mse
                best_rates = rates.copy()

            eps = 1e-12
            ratio = (T_pred + eps) / (T_target + eps)
            rates *= np.power(ratio, alpha)
            np.clip(rates, rate_min, rate_max, out=rates)
    
            if verbose and ((k + 1) % print_every == 0 or (k + 1) == iterations):
                etaferr(it=k+1, maxiter=iterations, start_time=start_time, err=mse)
                mse_list.append(mse)

        x = best_rates
    else:
        x = best_x

    #np.savetxt(f"data/fit_error/mse_chr{chr_number}.txt", mse_list, fmt="%.6f")

    if save_errQ:
        savecsv(arr=mse_list, chr_number=chr_number, stitle=f'fit_error', outpath="fit_error")
        
    return x


def _time_axis_from_timing(relative_timing, resolution_time=1.0, time_grid=None):
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


def _temporal_profile_from_spatial_fit(
    observed_timing,
    firing_rate_space,
    resolution_time=1.0,
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
    "tau",
    "neighbourhood_range",
    "fit_step",
    "maxiter",
    "sim_number",
    "err_threshold",
    "deviation_threshold",
    "space_fit_step",
    "space_maxiter",
    "verbose",
    "print_every",
    "perQ",
    "chr_number",
    "save_errQ",
}


def _spatial_firing_fit_kwargs(kwargs):
    return {key: value for key, value in kwargs.items() if key in _SPATIAL_FIRING_FIT_OPTIONS}


def _timing_bin_indices(timing, n_time, resolution_time=1.0, T0=None):
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
            "spacetime_order must be 'space_then_time' or 'time_then_space'."
        )
    return aliases[text]


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
    default_temporal_maxiter=4,
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
    temporal_maxiter=4,
    temporal_sim_number=100,
    refine_ratio_bounds=(0.25, 4.0),
    refine_edge_kb=1000,
    refine_seed=None,
    refine_verbose=None,
    perQ=False,
    floor=0.0,
):
    rates = np.asarray(firing_rate_time, dtype=float).reshape(1, -1).copy()
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
            seed=seed,
            verbose=False,
        )["replication_timing"]
        valid = crop & np.isfinite(T) & np.isfinite(T_pred) & (T > 0) & (T_pred > 0)
        if not np.any(valid):
            break
        err = mean_squared_error(T[valid], T_pred[valid])
        if err < best_err:
            best_err = err
            best_rates = rates.copy()
        ratio = np.ones(n, dtype=float)
        ratio[valid] = np.clip(T_pred[valid] / T[valid], low, high)
        updates = _mean_ratio_by_bin(ratio, bins, n_time, valid)
        rates[0] *= updates ** fit_step
        np.maximum(rates, floor, out=rates)

        if verbose:
            etaferr(it=it, maxiter=maxiter, start_time=start_time, err=err)

    return best_rates


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
    refine_maxiter=4,
    spacetime_sim_number=100,
    spacetime_update="factorized",
    spacetime_kernel_t=1.0,
    refine_ratio_bounds=(0.25, 4.0),
    refine_edge_kb=1000,
    refine_seed=None,
    refine_verbose=None,
    perQ=False,
    floor=0.0,
):
    rates = np.asarray(firing_rate_spacetime, dtype=float).copy()
    maxiter = int(refine_maxiter)
    if maxiter <= 0:
        return rates

    T = np.asarray(observed_timing, dtype=float)
    n = T.size
    if rates.ndim != 2 or rates.shape[1] != n:
        raise ValueError("firing_rate_spacetime must have shape (time, positions).")

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
    start_time = time.time()
    for it in range(1, maxiter + 1):
        seed = None if refine_seed is None else int(refine_seed) + it - 1
        T_pred = rsim(
            firing_rate=rates,
            fork_speed=fork_speed,
            sim_number=spacetime_sim_number,
            resolution_space=resolution,
            resolution_time=resolution_time,
            n_bins=n,
            perQ=perQ,
            seed=seed,
            verbose=False,
        )["replication_timing"]
        valid = crop & np.isfinite(T) & np.isfinite(T_pred) & (T > 0) & (T_pred > 0)
        if not np.any(valid):
            break
        err = mean_squared_error(T[valid], T_pred[valid])
        if err < best_err:
            best_err = err
            best_rates = rates.copy()
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

        if verbose:
            etaferr(it=it, maxiter=maxiter, start_time=start_time, err=err)

    return best_rates


def map_timing_firing_time(
    observed_timing,
    fork_speed=1.4,
    resolution=1.0,
    resolution_x=None,
    resolution_time=1.0,
    time_grid=None,
    T0=None,
    temporal_smoothw=None,
    space_fit_step=None,
    space_maxiter=None,
    temporal_fit_step=0.35,
    temporal_maxiter=4,
    temporal_sim_number=100,
    refine_ratio_bounds=(0.25, 4.0),
    refine_edge_kb=1000,
    refine_seed=None,
    refine_verbose=None,
    floor=0.0,
    **kwargs,
):
    """Infer a time-only firing profile from timing and constant fork speed.

    The route first fits the implemented spatial profile ``I(x)`` and then
    projects that fit onto the timing coordinate. The returned row vector has
    shape ``(1, n_time)`` so it is interpreted as ``I(t)`` by ``rsim``.
    """
    fit_kwargs = _spatial_firing_fit_kwargs(kwargs)
    if space_fit_step is not None:
        fit_kwargs["fit_step"] = space_fit_step
    if space_maxiter is not None:
        fit_kwargs["maxiter"] = space_maxiter

    firing_rate_space = map_timing_firing(
        observed_timing,
        fork_speed=fork_speed,
        resolution=resolution,
        resolution_x=resolution_x,
        **fit_kwargs,
    )
    temporal, _ = _temporal_profile_from_spatial_fit(
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
        resolution=_normalise_resolution_x(resolution, resolution_x=resolution_x),
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
    )


def map_timing_firing_spacetime(
    observed_timing,
    fork_speed=1.4,
    resolution=1.0,
    resolution_x=None,
    resolution_time=1.0,
    time_grid=None,
    T0=None,
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
    floor=0.0,
    **kwargs,
):
    """Infer a space-time firing field from timing and constant fork speed.

    ``spacetime_update='factorized'`` fits an x profile, builds a temporal
    projection, and refines the x and t components in ``spacetime_order``.
    ``spacetime_update='mixed'`` refines local x,t corrections directly.
    """
    update_mode = _canonical_spacetime_update(spacetime_update)
    order = _canonical_spacetime_order(spacetime_order)
    fit_kwargs = _spatial_firing_fit_kwargs(kwargs)
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
    if global_sim_number is not None:
        fit_kwargs["sim_number"] = global_sim_number

    firing_rate_space = map_timing_firing(
        observed_timing,
        fork_speed=fork_speed,
        resolution=resolution,
        resolution_x=resolution_x,
        **fit_kwargs,
    )
    temporal, _ = _temporal_profile_from_spatial_fit(
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
                resolution=_normalise_resolution_x(resolution, resolution_x=resolution_x),
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
            )

        if order == "time_then_space":
            firing_rate_spacetime = refine_factorized_phase(firing_rate_spacetime, "time")
            firing_rate_spacetime = refine_factorized_phase(firing_rate_spacetime, "space")
        else:
            firing_rate_spacetime = refine_factorized_phase(firing_rate_spacetime, "time")
        return firing_rate_spacetime

    if refine_verbose:
        print("\nRefining mixed space-time firing rate...")
    return _refine_spacetime_firing(
        firing_rate_spacetime,
        observed_timing,
        fork_speed=fork_speed,
        resolution=_normalise_resolution_x(resolution, resolution_x=resolution_x),
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
    )


def map_timing_constant_firing(
    observed_timing,
    fork_speed=1.4,
    resolution=1.0,
    resolution_x=None,
    resolution_time=1.0,
    neighbourhood_range=2000,
    fit_step=0.6,
    maxiter=10,
    sim_number=100,
    err_threshold=15.0,
    rate_bounds=None,
    refine_seed=None,
    perQ=False,
    n_bins=None,
    verbose=True,
    print_every=1,
):
    """Infer one constant firing rate from timing with supplied fork speed."""
    resolution = _normalise_resolution_x(resolution, resolution_x=resolution_x)
    T = np.asarray(observed_timing, dtype=float)
    if T.ndim != 1:
        raise ValueError("observed_timing must be a one-dimensional array.")
    if T.size == 0:
        raise ValueError("observed_timing must contain at least one value.")
    if n_bins is not None and int(n_bins) != T.size:
        raise ValueError("n_bins must match the length of observed_timing.")

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
    start_time = time.time()

    for it in range(1, maxiter + 1):
        if variable_fork_speed:
            seed = None if refine_seed is None else int(refine_seed) + it - 1
            predicted = rsim(
                firing_rate=rate,
                fork_speed=fork_speed,
                sim_number=sim_number,
                resolution_space=resolution,
                resolution_time=resolution_time,
                n_bins=T.size,
                perQ=perQ,
                seed=seed,
                verbose=False,
            )["replication_timing"]
        else:
            predicted = map_firing_timing(
                rate,
                fork_speed=fork_speed,
                resolution=resolution,
                neighbourhood_range=neighbourhood_range,
                perQ=perQ,
                n_bins=T.size,
            )
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
            best_rate = rate
        ratio = float(np.nanmedian(predicted[ok] / observed[ok]))
        if not np.isfinite(ratio) or ratio <= 0:
            break
        rate *= ratio ** fit_step
        rate = float(np.clip(rate, low, high))
        if verbose and (it % print_every == 0 or it == maxiter):
            etaferr(it=it, maxiter=maxiter, start_time=start_time, err=mse)

    return best_rate


def map_timing_firing_constant_speed(
    source,
    resolution=1.0,
    resolution_x=None,
    neighbourhood_range=2000,
    fit_step=0.6,
    maxiter=10,
    speed_bounds=None,
    initial_fork_speed=1.4,
    perQ=False,
    n_bins=None,
    verbose=True,
    print_every=1,
):
    """Infer one constant fork speed from timing and firing rate."""
    resolution = _normalise_resolution_x(resolution, resolution_x=resolution_x)
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

    speed = float(initial_fork_speed)
    if not np.isfinite(speed) or speed <= 0:
        speed = 1.4

    if speed_bounds is None:
        low = max(speed * 1e-3, 1e-12)
        high = max(speed * 1e3, low * 10.0)
    else:
        if len(speed_bounds) != 2:
            raise ValueError("speed_bounds must be a (min_speed, max_speed) pair.")
        low, high = map(float, speed_bounds)
        if not (np.isfinite(low) and np.isfinite(high)) or low <= 0 or high <= low:
            raise ValueError("speed_bounds must satisfy 0 < min_speed < max_speed.")
    speed = float(np.clip(speed, low, high))

    lm = int(1000 / resolution)
    if perQ:
        crop = slice(None)
    else:
        crop = slice(lm, -lm) if 2 * lm < T.size else slice(None)

    firing_rate = source["firing_rate"]
    maxiter = int(maxiter)
    print_every = max(1, int(print_every))
    best_speed = speed
    best_mse = np.inf
    start_time = time.time()

    for it in range(1, maxiter + 1):
        predicted = map_firing_timing(
            firing_rate,
            fork_speed=speed,
            resolution=resolution,
            neighbourhood_range=neighbourhood_range,
            perQ=perQ,
            n_bins=T.size,
        )
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
        speed *= ratio ** fit_step
        speed = float(np.clip(speed, low, high))
        if verbose and (it % print_every == 0 or it == maxiter):
            etaferr(it=it, maxiter=maxiter, start_time=start_time, err=mse)

    return best_speed


def map_timing_firing_spatial_speed(
    source,
    resolution=1.0,
    resolution_x=None,
    neighbourhood_range=2000,
    fit_step=0.4,
    maxiter=40,
    space_fit_step=None,
    space_maxiter=None,
    speed_bounds=None,
    initial_fork_speed=1.4,
    smoothw=100,
    perQ=False,
    n_bins=None,
    verbose=True,
    print_every=1,
):
    """Infer a spatial effective fork-speed profile from timing and firing."""
    resolution = _normalise_resolution_x(resolution, resolution_x=resolution_x)
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
    if speed_bounds is None:
        low = max(reference_speed * 1e-3, 1e-12)
        high = max(reference_speed * 1e3, low * 10.0)
    else:
        if len(speed_bounds) != 2:
            raise ValueError("speed_bounds must be a (min_speed, max_speed) pair.")
        low, high = map(float, speed_bounds)
        if not (np.isfinite(low) and np.isfinite(high)) or low <= 0 or high <= low:
            raise ValueError("speed_bounds must satisfy 0 < min_speed < max_speed.")
    speed = np.clip(speed, low, high)

    lm = int(1000 / resolution)
    if perQ:
        crop = slice(None)
    else:
        crop = slice(lm, -lm) if 2 * lm < T.size else slice(None)

    def fill_nonperiodic_edges(profile):
        if perQ or crop == slice(None):
            return profile
        start = 0 if crop.start is None else int(crop.start)
        stop = T.size if crop.stop is None else int(crop.stop)
        if stop < 0:
            stop = T.size + stop
        start = max(0, min(start, T.size))
        stop = max(0, min(stop, T.size))
        if stop <= start:
            return profile
        if start > 0:
            profile[:start] = profile[start]
        if stop < T.size:
            profile[stop:] = profile[stop - 1]
        return profile

    speed = fill_nonperiodic_edges(speed)

    maxiter = int(maxiter)
    print_every = max(1, int(print_every))
    best_speed = speed.copy()
    best_mse = np.inf
    start_time = time.time()

    for it in range(1, maxiter + 1):
        predicted = map_firing_timing(
            firing_rate,
            fork_speed=speed,
            resolution=resolution,
            neighbourhood_range=neighbourhood_range,
            perQ=perQ,
            n_bins=T.size,
        )
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
        speed = fill_nonperiodic_edges(speed)

        if verbose and (it % print_every == 0 or it == maxiter):
            etaferr(it=it, maxiter=maxiter, start_time=start_time, err=mse)

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
    if arr.ndim != 2 or arr.shape[0] == 1 or arr.shape[1] != n:
        raise ValueError("firing_rate must have shape (time, positions) for v(x,t) fitting.")
    if np.any(~np.isfinite(arr)) or np.any(arr < 0):
        raise ValueError("firing_rate must contain finite values >= 0.")
    return arr.astype(float, copy=False)


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


def _refine_temporal_speed(
    fork_speed_time,
    observed_timing,
    firing_rate,
    resolution,
    resolution_time,
    T0=None,
    temporal_fit_step=0.35,
    temporal_maxiter=4,
    temporal_sim_number=100,
    speed_bounds=None,
    refine_ratio_bounds=(0.25, 4.0),
    refine_edge_kb=1000,
    refine_seed=None,
    refine_verbose=None,
    perQ=False,
):
    speed = np.asarray(fork_speed_time, dtype=float).reshape(1, -1).copy()
    maxiter = int(temporal_maxiter)
    if maxiter <= 0:
        return speed

    T = np.asarray(observed_timing, dtype=float)
    n = T.size
    n_time = speed.shape[1]
    bins = _timing_bin_indices(T, n_time, resolution_time=resolution_time, T0=T0)
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
    start_time = time.time()
    for it in range(1, maxiter + 1):
        seed = None if refine_seed is None else int(refine_seed) + it - 1
        T_pred = rsim(
            firing_rate=firing_rate,
            fork_speed=speed,
            sim_number=temporal_sim_number,
            resolution_space=resolution,
            resolution_time=resolution_time,
            n_bins=n,
            perQ=perQ,
            seed=seed,
            verbose=False,
        )["replication_timing"]
        valid = crop & np.isfinite(T) & np.isfinite(T_pred) & (T > 0) & (T_pred > 0)
        if not np.any(valid):
            break
        err = mean_squared_error(T[valid], T_pred[valid])
        if err < best_err:
            best_err = err
            best_speed = speed.copy()
        ratio = np.ones(n, dtype=float)
        ratio[valid] = np.clip(T_pred[valid] / T[valid], low_ratio, high_ratio)
        updates = _mean_ratio_by_bin(ratio, bins, n_time, valid)
        speed[0] *= updates ** fit_step
        np.clip(speed, low_speed, high_speed, out=speed)

        if verbose:
            etaferr(it=it, maxiter=maxiter, start_time=start_time, err=err)

    return best_speed


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
    refine_maxiter=4,
    spacetime_sim_number=100,
    spacetime_update="factorized",
    spacetime_kernel_t=1.0,
    speed_bounds=None,
    refine_ratio_bounds=(0.25, 4.0),
    refine_edge_kb=1000,
    refine_seed=None,
    refine_verbose=None,
    perQ=False,
):
    speed = np.asarray(fork_speed_spacetime, dtype=float).copy()
    maxiter = int(refine_maxiter)
    if maxiter <= 0:
        return speed

    T = np.asarray(observed_timing, dtype=float)
    n = T.size
    if speed.ndim != 2 or speed.shape[1] != n:
        raise ValueError("fork_speed_spacetime must have shape (time, positions).")

    n_time = speed.shape[0]
    bins = _timing_bin_indices(T, n_time, resolution_time=resolution_time, T0=T0)
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
    start_time = time.time()
    for it in range(1, maxiter + 1):
        seed = None if refine_seed is None else int(refine_seed) + it - 1
        T_pred = rsim(
            firing_rate=firing_rate,
            fork_speed=speed,
            sim_number=spacetime_sim_number,
            resolution_space=resolution,
            resolution_time=resolution_time,
            n_bins=n,
            perQ=perQ,
            seed=seed,
            verbose=False,
        )["replication_timing"]
        valid = crop & np.isfinite(T) & np.isfinite(T_pred) & (T > 0) & (T_pred > 0)
        if not np.any(valid):
            break
        err = mean_squared_error(T[valid], T_pred[valid])
        if err < best_err:
            best_err = err
            best_speed = speed.copy()
        ratio = np.ones(n, dtype=float)
        ratio[valid] = np.clip(T_pred[valid] / T[valid], low_ratio, high_ratio)
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

        if verbose:
            etaferr(it=it, maxiter=maxiter, start_time=start_time, err=err)

    return best_speed


def map_timing_firing_temporal_speed(
    source,
    resolution=1.0,
    resolution_x=None,
    resolution_time=1.0,
    time_grid=None,
    T0=None,
    temporal_smoothw=None,
    neighbourhood_range=2000,
    fit_step=0.4,
    maxiter=40,
    space_fit_step=None,
    space_maxiter=None,
    temporal_fit_step=0.35,
    temporal_maxiter=4,
    temporal_sim_number=100,
    speed_bounds=None,
    initial_fork_speed=1.4,
    smoothw=100,
    perQ=False,
    n_bins=None,
    refine_ratio_bounds=(0.25, 4.0),
    refine_edge_kb=1000,
    refine_seed=None,
    refine_verbose=None,
    verbose=True,
    print_every=1,
):
    """Infer a time-only fork-speed profile from timing and spatial firing."""
    resolution = _normalise_resolution_x(resolution, resolution_x=resolution_x)
    T, firing_rate = _validate_timing_and_firing_for_speed(source)
    if n_bins is not None and int(n_bins) != T.size:
        raise ValueError("n_bins must match the length of replication_timing.")
    firing_rate = _spatial_firing_for_speed_fit(firing_rate, T.size)

    speed_space = map_timing_firing_spatial_speed(
        {"replication_timing": T, "firing_rate": firing_rate},
        resolution=resolution,
        neighbourhood_range=neighbourhood_range,
        fit_step=fit_step,
        maxiter=maxiter,
        space_fit_step=space_fit_step,
        space_maxiter=space_maxiter,
        speed_bounds=speed_bounds,
        initial_fork_speed=initial_fork_speed,
        smoothw=smoothw,
        perQ=perQ,
        n_bins=n_bins,
        verbose=verbose,
        print_every=print_every,
    )
    temporal, _ = _temporal_profile_from_spatial_fit(
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
    if refine_verbose is None:
        refine_verbose = bool(verbose)
    if refine_verbose:
        print("\nRefining time-dependent fork speed...")
    return _refine_temporal_speed(
        temporal.reshape(1, -1),
        T,
        firing_rate=firing_rate,
        resolution=resolution,
        resolution_time=resolution_time,
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
    )


def map_timing_firing_spacetime_speed(
    source,
    resolution=1.0,
    resolution_x=None,
    resolution_time=1.0,
    time_grid=None,
    T0=None,
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
    spacetime_order="space_then_time",
    spacetime_kernel_t=1.0,
    speed_bounds=None,
    initial_fork_speed=1.4,
    smoothw=100,
    perQ=False,
    n_bins=None,
    refine_ratio_bounds=(0.25, 4.0),
    refine_edge_kb=1000,
    refine_seed=None,
    refine_verbose=None,
    verbose=True,
    print_every=1,
):
    """Infer a space-time fork-speed field from timing and I(x,t)."""
    resolution = _normalise_resolution_x(resolution, resolution_x=resolution_x)
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
        default_space_maxiter=40,
    )
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

    speed_space = map_timing_firing_spatial_speed(
        {"replication_timing": T, "firing_rate": firing_rate_space},
        resolution=resolution,
        neighbourhood_range=neighbourhood_range,
        fit_step=space_fit_step,
        maxiter=space_maxiter
        if update_mode == "factorized" and order == "space_then_time"
        else 0,
        speed_bounds=speed_bounds,
        initial_fork_speed=initial_fork_speed,
        smoothw=smoothw,
        perQ=perQ,
        n_bins=n_bins,
        verbose=verbose,
        print_every=print_every,
    )
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
                firing_rate=firing_rate_xt,
                resolution=resolution,
                resolution_time=resolution_time,
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
            )

        if order == "time_then_space":
            speed_spacetime = refine_factorized_phase(speed_spacetime, "time")
            speed_spacetime = refine_factorized_phase(speed_spacetime, "space")
        else:
            speed_spacetime = refine_factorized_phase(speed_spacetime, "time")
        return speed_spacetime

    if refine_verbose:
        print("\nRefining mixed space-time fork speed...")
    return _refine_spacetime_speed(
        speed_spacetime,
        T,
        firing_rate=firing_rate_xt,
        resolution=resolution,
        resolution_time=resolution_time,
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
    )


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
        if arr.shape == (n_time, n_positions):
            if np.any(~np.isfinite(arr)) or np.any(arr < 0):
                raise ValueError("initial_firing_rate (T,n) must contain finite values >= 0.")
            return np.maximum(arr.astype(float, copy=True), floor)
    raise ValueError("initial_firing_rate must be scalar, (n,), (1,T), (n,1), or (T,n).")


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
        if arr.shape == (n_time, n_positions):
            if np.any(~np.isfinite(arr)) or np.any(arr <= 0):
                raise ValueError("initial_fork_speed (T,n) must contain finite positive values.")
            return arr.astype(float, copy=True)
    raise ValueError("initial_fork_speed must be scalar, (n,), (1,T), (n,1), or (T,n).")


def _time_axis_for_timing(T, resolution_time=1.0, time_grid=None, T0=None):
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
        if arr.shape[1] == n_positions and arr.shape[0] > 1:
            values = np.nanmean(arr, axis=0)
            if np.any(~np.isfinite(values)) or np.any(values <= 0):
                raise ValueError("initial_fork_speed (T,n) must contain finite positive values.")
            return values.astype(float, copy=True)
        if arr.shape[0] == 1:
            value = _initial_positive_speed_value(arr, default=default)
            return np.full(n_positions, value, dtype=float)
    raise ValueError("initial_fork_speed must be scalar, (n,), (n,1), (1,T), or (T,n).")


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
        if arr.shape[0] == n_time and arr.shape[1] > 1:
            values = np.nanmean(arr, axis=1)
            if np.any(~np.isfinite(values)) or np.any(values <= 0):
                raise ValueError("initial_fork_speed (T,n) must contain finite positive values.")
            return values.astype(float, copy=True)
        value = _initial_positive_speed_value(arr, default=default)
        return np.full(n_time, value, dtype=float)
    raise ValueError("initial_fork_speed must be scalar, (n,), (1,T), (n,1), or (T,n).")


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
    resolution_time=1.0,
    fit_step=0.6,
    maxiter=10,
    sim_number=100,
    speed_bounds=None,
    initial_fork_speed=1.4,
    refine_seed=None,
    perQ=False,
    n_bins=None,
    verbose=True,
    print_every=1,
):
    """Infer one fork speed from timing and time-dependent firing-rate inputs."""
    resolution = _normalise_resolution_x(resolution, resolution_x=resolution_x)
    T, firing_rate = _validate_timing_and_firing_for_speed(source)
    if n_bins is not None and int(n_bins) != T.size:
        raise ValueError("n_bins must match the length of replication_timing.")

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
        if verbose and (it % print_every == 0 or it == maxiter):
            etaferr(it=it, maxiter=maxiter, start_time=start_time, err=mse)

    return best_speed


def map_timing_firing_spatial_speed_simulated(
    source,
    resolution=1.0,
    resolution_x=None,
    resolution_time=1.0,
    fit_step=0.4,
    maxiter=40,
    space_fit_step=None,
    space_maxiter=None,
    sim_number=100,
    speed_bounds=None,
    initial_fork_speed=1.4,
    smoothw=100,
    refine_seed=None,
    perQ=False,
    n_bins=None,
    verbose=True,
    print_every=1,
):
    """Infer a spatial fork-speed profile using stochastic timing refinement."""
    resolution = _normalise_resolution_x(resolution, resolution_x=resolution_x)
    if space_fit_step is not None:
        fit_step = space_fit_step
    if space_maxiter is not None:
        maxiter = space_maxiter
    T, firing_rate = _validate_timing_and_firing_for_speed(source)
    if n_bins is not None and int(n_bins) != T.size:
        raise ValueError("n_bins must match the length of replication_timing.")

    speed = _initial_spatial_speed_profile(initial_fork_speed, T.size)
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
        if verbose and (it % print_every == 0 or it == maxiter):
            etaferr(it=it, maxiter=maxiter, start_time=start_time, err=mse)

    return best_speed


def map_timing_firing_temporal_speed_simulated(
    source,
    resolution=1.0,
    resolution_x=None,
    resolution_time=1.0,
    time_grid=None,
    T0=None,
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
    perQ=False,
    n_bins=None,
    verbose=True,
    print_every=1,
    **kwargs,
):
    """Infer a time-only fork-speed profile from timing and general firing."""
    if refine_verbose is None:
        refine_verbose = bool(verbose)
    del print_every, kwargs
    resolution = _normalise_resolution_x(resolution, resolution_x=resolution_x)
    if temporal_fit_step is None:
        temporal_fit_step = 0.35 if fit_step is None else fit_step
    if temporal_maxiter is None:
        temporal_maxiter = 4 if maxiter is None else maxiter
    T, firing_rate = _validate_timing_and_firing_for_speed(source)
    if n_bins is not None and int(n_bins) != T.size:
        raise ValueError("n_bins must match the length of replication_timing.")
    time_axis = _time_axis_for_timing(T, resolution_time=resolution_time, time_grid=time_grid, T0=T0)
    speed_time = _initial_temporal_speed_profile(initial_fork_speed, time_axis.size)
    if temporal_smoothw is not None:
        speed_time = smoothf(speed_time, temporal_smoothw)
    low, high = _speed_bounds_from_profile(speed_time, speed_bounds=speed_bounds)
    speed_time = np.clip(speed_time, low, high)
    if refine_verbose:
        print("\nRefining time-dependent fork speed...")
    return _refine_temporal_speed(
        speed_time.reshape(1, -1),
        T,
        firing_rate=firing_rate,
        resolution=resolution,
        resolution_time=resolution_time,
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
    )


def map_timing_firing_spacetime_speed_simulated(
    source,
    resolution=1.0,
    resolution_x=None,
    resolution_time=1.0,
    time_grid=None,
    T0=None,
    fit_step=None,
    maxiter=None,
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
    speed_bounds=None,
    initial_fork_speed=1.4,
    refine_ratio_bounds=(0.25, 4.0),
    refine_edge_kb=1000,
    refine_seed=None,
    refine_verbose=None,
    perQ=False,
    n_bins=None,
    verbose=True,
    print_every=1,
    **kwargs,
):
    """Infer a space-time fork-speed field from timing and general firing."""
    if refine_verbose is None:
        refine_verbose = bool(verbose)
    sim_number = kwargs.pop("sim_number", None)
    del print_every, kwargs
    resolution = _normalise_resolution_x(resolution, resolution_x=resolution_x)
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
        default_space_fit_step=0.25,
        default_space_maxiter=4,
    )
    if spacetime_sim_number is None:
        spacetime_sim_number = temporal_sim_number
    if spacetime_fit_step is None:
        spacetime_fit_step = 0.5 * (space_fit_step + temporal_fit_step)
    T, firing_rate = _validate_timing_and_firing_for_speed(source)
    if n_bins is not None and int(n_bins) != T.size:
        raise ValueError("n_bins must match the length of replication_timing.")
    time_axis = _time_axis_for_timing(T, resolution_time=resolution_time, time_grid=time_grid, T0=T0)
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
                firing_rate=firing_rate,
                resolution=resolution,
                resolution_time=resolution_time,
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
            )

        phases = (
            ("time", "space")
            if order == "time_then_space"
            else ("space", "time")
        )
        for phase in phases:
            speed_spacetime = refine_factorized_phase(speed_spacetime, phase)
        return speed_spacetime

    if refine_verbose:
        print("\nRefining mixed space-time fork speed...")
    return _refine_spacetime_speed(
        speed_spacetime,
        T,
        firing_rate=firing_rate,
        resolution=resolution,
        resolution_time=resolution_time,
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
    )


def _maybe_smooth_positive(values, window, floor):
    if window is None:
        return values
    window = int(window)
    if window <= 1:
        return values
    return np.maximum(smoothf(values, window=window), floor)


def map_timing_joint_spacetime_kinetics(
    observed_timing,
    resolution=1.0,
    resolution_x=None,
    resolution_time=1.0,
    time_grid=None,
    T0=None,
    initial_firing_rate=None,
    initial_fork_speed=1.4,
    initial_kinetics=None,
    target_form="spacetime",
    space_init_fit_step=0.6,
    space_init_maxiter=10,
    space_init_sim_number=100,
    joint_maxiter=6,
    joint_sim_number=100,
    firing_space_fit_step=0.15,
    firing_time_fit_step=0.1,
    speed_space_fit_step=0.15,
    speed_time_fit_step=0.1,
    firing_bounds=None,
    speed_bounds=None,
    firing_floor=1e-12,
    refine_ratio_bounds=(0.25, 4.0),
    refine_edge_kb=1000,
    refine_seed=None,
    spatial_smoothw=100,
    temporal_smoothw=None,
    perQ=False,
    verbose=True,
    print_every=1,
):
    """Jointly fit space-time firing-rate and fork-speed fields from timing.

    The update is simultaneous: each iteration simulates timing from the
    current ``I(x,t)`` and ``v(x,t)`` fields, then applies smooth spatial and
    temporal residual corrections to both fields. This is intentionally a
    high-dimensional effective fit, so the solution should be interpreted as
    one compatible kinetic decomposition rather than a unique biological
    answer.
    """
    form = str(target_form).strip().lower().replace(" ", "")
    if form not in {"spacetime", "space-time", "space_time", "xt", "x,t", "(x,t)"}:
        raise ValueError("joint kinetic fitting currently requires target_form='spacetime'.")

    resolution = _normalise_resolution_x(resolution, resolution_x=resolution_x)
    resolution_time = float(resolution_time)
    if not np.isfinite(resolution_time) or resolution_time <= 0:
        raise ValueError("resolution_t must be a finite positive value.")

    T = np.asarray(observed_timing, dtype=float)
    if T.ndim != 1:
        raise ValueError("observed_timing must be a one-dimensional array.")
    if T.size == 0:
        raise ValueError("observed_timing must contain at least one value.")

    finite = np.isfinite(T)
    if not np.any(finite):
        raise ValueError("observed_timing must contain at least one finite value.")
    if T0 is None:
        T0 = float(np.nanmin(T[finite]))
    relative_timing = np.maximum(T - float(T0), 0.0)
    time_axis = _time_axis_from_timing(
        relative_timing,
        resolution_time=resolution_time,
        time_grid=time_grid,
    )
    n_time = int(time_axis.size)
    n_positions = int(T.size)

    if initial_kinetics is not None:
        if not isinstance(initial_kinetics, dict):
            raise ValueError("initial_kinetics must be a dictionary when supplied.")
        if initial_firing_rate is None and "firing_rate" in initial_kinetics:
            initial_firing_rate = initial_kinetics["firing_rate"]
        if "fork_speed" in initial_kinetics:
            initial_fork_speed = initial_kinetics["fork_speed"]

    initial_speed_arr = np.asarray(initial_fork_speed, dtype=float)
    initial_speed_positive = initial_speed_arr[
        np.isfinite(initial_speed_arr) & (initial_speed_arr > 0)
    ]
    initial_speed_reference = (
        float(np.nanmedian(initial_speed_positive))
        if initial_speed_positive.size
        else 1.4
    )

    firing_floor = float(firing_floor)
    if not np.isfinite(firing_floor) or firing_floor < 0:
        raise ValueError("firing_floor must be finite and >= 0.")

    if initial_firing_rate is None:
        positive_timing = T[finite & (T > 0)]
        timing_floor = (
            max(float(np.nanpercentile(positive_timing, 5)) * 0.1, resolution_time * 0.5, 1e-6)
            if positive_timing.size
            else max(resolution_time * 0.5, 1e-6)
        )
        init_timing = np.where(finite & (T > timing_floor), T, timing_floor)
        firing_space = map_timing_firing(
            init_timing,
            fork_speed=initial_speed_reference,
            resolution=resolution,
            fit_step=space_init_fit_step,
            maxiter=space_init_maxiter,
            sim_number=space_init_sim_number,
            verbose=False,
            perQ=perQ,
        )
        firing_space = _maybe_smooth_positive(firing_space, spatial_smoothw, firing_floor)
        firing_time, _ = _temporal_profile_from_spatial_fit(
            T,
            firing_space,
            resolution_time=resolution_time,
            time_grid=time_axis,
            T0=T0,
            temporal_smoothw=temporal_smoothw,
            floor=firing_floor,
        )
        firing_scale = float(np.nanmean(firing_space[firing_space > 0])) if np.any(firing_space > 0) else 1.0
        if not np.isfinite(firing_scale) or firing_scale <= 0:
            firing_scale = 1.0
        firing_rate = np.outer(firing_time, firing_space / firing_scale)
    else:
        firing_rate = _expand_firing_to_spacetime(
            initial_firing_rate,
            n_time,
            n_positions,
            floor=firing_floor,
        )

    fork_speed = _expand_speed_to_spacetime(initial_fork_speed, n_time, n_positions)

    low_firing, high_firing = _firing_bounds_from_profile(
        firing_rate,
        firing_bounds=firing_bounds,
        floor=firing_floor,
    )
    low_speed, high_speed = _speed_bounds_from_profile(
        fork_speed,
        speed_bounds=speed_bounds,
    )
    np.clip(firing_rate, low_firing, high_firing, out=firing_rate)
    np.clip(fork_speed, low_speed, high_speed, out=fork_speed)

    low_ratio, high_ratio = map(float, refine_ratio_bounds)
    if not (np.isfinite(low_ratio) and np.isfinite(high_ratio) and 0 < low_ratio < high_ratio):
        raise ValueError("refine_ratio_bounds must be a positive (low, high) pair.")

    steps = {
        "firing_space_fit_step": firing_space_fit_step,
        "firing_time_fit_step": firing_time_fit_step,
        "speed_space_fit_step": speed_space_fit_step,
        "speed_time_fit_step": speed_time_fit_step,
    }
    for name, value in steps.items():
        value = float(value)
        if not np.isfinite(value) or value < 0:
            display = {
                "firing_space_fit_step": "fit_step_firing_x",
                "firing_time_fit_step": "fit_step_firing_t",
                "speed_space_fit_step": "fit_step_speed_x",
                "speed_time_fit_step": "fit_step_speed_t",
            }.get(name, name)
            raise ValueError(f"{display} must be finite and >= 0.")
        steps[name] = value

    maxiter = int(joint_maxiter)
    sim_number = int(joint_sim_number)
    if maxiter < 0:
        raise ValueError("maxiter_joint must be >= 0.")
    if sim_number <= 0:
        raise ValueError("joint_sim_number must be a positive integer.")

    bins = _timing_bin_indices(T, n_time, resolution_time=resolution_time, T0=T0)
    crop = _refinement_crop_mask(
        n_positions,
        resolution,
        perQ=perQ,
        refine_edge_kb=refine_edge_kb,
    )
    fit_history = []
    T_pred = None
    best_firing_rate = firing_rate.copy()
    best_fork_speed = fork_speed.copy()
    best_timing_fit = None
    best_err = np.inf
    print_every = max(1, int(print_every))
    start_time = time.time()

    for it in range(1, maxiter + 1):
        seed = None if refine_seed is None else int(refine_seed) + it - 1
        T_pred = rsim(
            firing_rate=firing_rate,
            fork_speed=fork_speed,
            sim_number=sim_number,
            resolution_space=resolution,
            resolution_time=resolution_time,
            n_bins=n_positions,
            perQ=perQ,
            seed=seed,
            verbose=False,
        )["replication_timing"]
        valid = crop & np.isfinite(T) & np.isfinite(T_pred) & (T > 0) & (T_pred > 0)
        if not np.any(valid):
            break

        err = mean_squared_error(T[valid], T_pred[valid])
        fit_history.append(err)
        if err < best_err:
            best_err = err
            best_firing_rate = firing_rate.copy()
            best_fork_speed = fork_speed.copy()
            best_timing_fit = T_pred.copy()

        ratio = np.ones(n_positions, dtype=float)
        ratio[valid] = np.clip(T_pred[valid] / T[valid], low_ratio, high_ratio)
        spatial_update = _maybe_smooth_positive(ratio, spatial_smoothw, 1e-12)
        temporal_update = _mean_ratio_by_bin(ratio, bins, n_time, valid)
        temporal_update = _maybe_smooth_positive(temporal_update, temporal_smoothw, 1e-12)

        firing_rate[:, valid] *= spatial_update[valid][None, :] ** steps["firing_space_fit_step"]
        fork_speed[:, valid] *= spatial_update[valid][None, :] ** steps["speed_space_fit_step"]
        firing_rate *= temporal_update[:, None] ** steps["firing_time_fit_step"]
        fork_speed *= temporal_update[:, None] ** steps["speed_time_fit_step"]

        np.clip(firing_rate, low_firing, high_firing, out=firing_rate)
        np.clip(fork_speed, low_speed, high_speed, out=fork_speed)

        if verbose and (it % print_every == 0 or it == maxiter):
            etaferr(it=it, maxiter=maxiter, start_time=start_time, err=err)

    return {
        "firing_rate": best_firing_rate,
        "fork_speed": best_fork_speed,
        "replication_timing_fit": best_timing_fit,
        "fit_history": np.asarray(fit_history, dtype=float),
        "time_grid": time_axis,
    }



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
