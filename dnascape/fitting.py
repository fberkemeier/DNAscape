"""Fitting algorithms and fitting-oriented helpers."""

import math
import os
import time

import numpy as np
from scipy.optimize import minimize
from scipy.signal import fftconvolve
from scipy.special import gamma

from .constants import mapt
from .io import _resolve_data_path, savecsv
from .mapping import (
    _normalise_resolution_x,
    map_firing_timing,
    map_replicated_fraction_xt,
)
from .simulation import rsim
from .utils import check_cancelled, etaferr, mean_squared_error, smoothf


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
    error=None,
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
    error = _normalise_error_threshold(error)

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
    best_timing_fit = np.asarray(y, dtype=float).copy()

    if stall_rate > 0. or variable_fork_speed:
        maxiter0 = 0
    else:
        maxiter0 = maxiter

    print_every = max(1, int(print_every))
    start_time = time.time()

    for it in range(1, maxiter0 + 1):
        check_cancelled()
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
            best_timing_fit = np.asarray(y, dtype=float).copy()

        done = it == maxiter0 or _error_reached(mse, error)
        if verbose and (it % print_every == 0 or done):
            etaferr(it=it, maxiter=maxiter0, start_time=start_time, err=mse, done=done)
            mse_list.append(mse)
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
            pos = rates[rates > 0]
            rate_max = (10.0 * np.median(pos)) if pos.size else 1.0
    
        hist_mse = []
        T_pred = None
        best_rates = rates.copy()
        best_mse = np.inf
        start_time = time.time()
    
        for k in range(iterations):
            check_cancelled()
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
                best_timing_fit = np.asarray(T_pred, dtype=float).copy()

            eps = 1e-12
            ratio = (T_pred + eps) / (T_target + eps)
            rates *= np.power(ratio, alpha)
            np.clip(rates, rate_min, rate_max, out=rates)
    
            done = (k + 1) == iterations or _error_reached(mse, error)
            if verbose and ((k + 1) % print_every == 0 or done):
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
        {"replication_timing_fit": best_timing_fit},
    )


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
        if int(n) > 1500:
            raise NotImplementedError(
                "x-dependent Weibull delays currently use the pairwise backend; "
                "use a smaller interval or a scalar weibull_delay."
            )
        x = np.arange(int(n), dtype=float) * float(resolution)
        return "pairwise", v, np.abs(x[:, None] - x[None, :]) / v

    speed = np.squeeze(speed)
    if speed.ndim == 1:
        if speed.size != int(n):
            raise ValueError("fork_speed must be scalar or match observed_timing.")
        if np.any(~np.isfinite(speed)) or np.any(speed <= 0):
            raise ValueError("fork_speed must contain finite positive values.")
        if int(n) > 1500:
            raise NotImplementedError(
                "Weibull fitting with v(x) currently uses a pairwise travel-time "
                "backend; use a smaller interval for this route."
            )
        inv_v = 1.0 / speed
        phi = np.zeros(int(n), dtype=float)
        if int(n) > 1:
            phi[1:] = np.cumsum(0.5 * (inv_v[1:] + inv_v[:-1]) * float(resolution))
        return "pairwise", speed.copy(), np.abs(phi[:, None] - phi[None, :])

    raise NotImplementedError(
        "Weibull fitting currently implements constant fork speed, with a "
        "pairwise prototype for v(x). v(t) and v(x,t) need the planned "
        "cumulative-distance or transport backends."
    )


def _weibull_survival_fields(
    alpha,
    fork_speed,
    shape=2.0,
    resolution=1.0,
    resolution_time=5.0,
    time_grid=None,
    delay=0.0,
    rate_floor=1e-12,
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
    mode, speed_used, travel = _weibull_travel_backend(
        fork_speed,
        alpha.size,
        resolution,
        delay,
    )

    A = np.zeros((t.size, alpha.size), dtype=float)
    if mode == "constant_fft":
        offset = float(delay[0])
        for idx, time_value in enumerate(t):
            lag = np.maximum(time_value - offset - travel, 0.0)
            kernel = np.power(lag, k)
            A[idx] = float(resolution) * _symmetric_convolution(alpha, kernel)
    else:
        source_delay = delay[None, :]
        for idx, time_value in enumerate(t):
            power = np.power(np.maximum(time_value - source_delay - travel, 0.0), k)
            A[idx] = float(resolution) * power.dot(alpha)

    A = np.maximum(A, 0.0)
    survival = np.exp(-np.clip(A, 0.0, 700.0))
    predicted_timing = weights @ survival
    firing_rate = (
        k
        * alpha[None, :]
        * np.power(np.maximum(t[:, None] - delay[None, :], 0.0), k - 1.0)
    )
    if rate_floor > 0:
        firing_rate = np.maximum(firing_rate, rate_floor)
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

    firing_rate = (
        shape
        * alpha[None, :]
        * np.power(np.maximum(t[:, None] - delay_value, 0.0), shape - 1.0)
    )
    firing_rate = np.maximum(firing_rate, float(floor))
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
):
    error = _normalise_error_threshold(error)
    if perQ:
        raise NotImplementedError("Weibull survival fitting currently supports linear geometry only.")
    T_obs = np.asarray(observed_timing, dtype=float)
    if T_obs.ndim != 1 or T_obs.size == 0:
        raise ValueError("observed_timing must be a non-empty one-dimensional array.")

    offset = 0.0 if T0 is None else float(T0)
    if not np.isfinite(offset):
        raise ValueError("T0 must be finite when supplied.")
    target_timing = T_obs - offset
    valid = np.isfinite(T_obs) & np.isfinite(target_timing) & (target_timing > 0)
    crop = _refinement_crop_mask(T_obs.size, resolution, perQ=False, refine_edge_kb=refine_edge_kb)
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
        if verbose and calls <= maxiter_display:
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
            if verbose and calls <= maxiter_display:
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
        runner = lambda: minimize(
            objective,
            params0,
            method="L-BFGS-B",
            bounds=bounds,
            options={
                "maxiter": maxiter,
                "maxfun": max(maxiter * 4, maxiter + 1),
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
    if verbose and 0 < calls < maxiter_display and best["fields"] is not None:
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

        done = it == maxiter or _error_reached(err, error)
        if verbose:
            etaferr(it=it, maxiter=maxiter, start_time=start_time, err=err, done=done)
        if done and it != maxiter:
            break

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
            firing_rate=rates.T,
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

        done = it == maxiter or _error_reached(err, error)
        if verbose:
            etaferr(it=it, maxiter=maxiter, start_time=start_time, err=err, done=done)
        if done and it != maxiter:
            break

    return best_rates


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
    target_shape = _canonical_target_shape(target_shape)
    if target_shape == "weibull":
        if refine_verbose is None:
            refine_verbose = bool(kwargs.get("verbose", False))
        return _fit_weibull_survival_firing(
            observed_timing,
            fork_speed=fork_speed,
            resolution=_normalise_resolution_x(resolution, resolution_x=resolution_x),
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
        )

    fit_kwargs = _spatial_firing_fit_kwargs(kwargs)
    if space_fit_step is not None:
        fit_kwargs["fit_step"] = space_fit_step
    if space_maxiter is not None:
        fit_kwargs["maxiter"] = space_maxiter
    if error is not None:
        fit_kwargs["error"] = error

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
        error=error,
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
            resolution=_normalise_resolution_x(resolution, resolution_x=resolution_x),
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
        )

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
    if error is not None:
        fit_kwargs["error"] = error
    if global_sim_number is not None:
        fit_kwargs["sim_number"] = global_sim_number

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
                error=error,
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
        error=error,
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
        if verbose and (it % print_every == 0 or done):
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
    """Infer a spatial effective fork-speed profile from timing and firing."""
    del resolution_time, time_grid, T0, refine_edge_kb
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

        done = it == maxiter or _error_reached(mse, error)
        if verbose and (it % print_every == 0 or done):
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
        info = {
            "target_shape": "best",
            "fork_speed_shape": "site-specific v(x)",
            "fork_slowness_shape": "site-specific beta(x)",
            "fork_speed_bounds": (float(low), float(high)),
            "fit_mse": None if not np.isfinite(best_mse) else float(best_mse),
        }
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

    def subset_field(field, idx):
        arr = np.asarray(field, dtype=float)
        if arr.size == 1:
            return float(arr.reshape(-1)[0])
        if arr.ndim == 1:
            return arr[idx] if arr.size == T.size else arr
        if arr.ndim == 2:
            if arr.shape[0] == T.size:
                return arr[idx]
            if arr.shape[1] == T.size:
                return arr[:, idx].T
            return arr
        return arr

    fit_mse = None
    predicted_full = None
    try:
        offset = 0.0 if T0 is None else float(T0)
        crop = _refinement_crop_mask(T.size, resolution, perQ=perQ, refine_edge_kb=refine_edge_kb)
        valid_idx = np.flatnonzero(crop & np.isfinite(T))
        if valid_idx.size:
            stride = max(1, int(math.ceil(valid_idx.size / 1500)))
            fit_idx = valid_idx[::stride]
            if fit_idx[-1] != valid_idx[-1]:
                fit_idx = np.concatenate([fit_idx, valid_idx[-1:]])
            resolution_fit = float(resolution) * float(stride)
            fraction_fit = map_replicated_fraction_xt(
                subset_field(firing_rate, fit_idx),
                fork_speed=speed.T[fit_idx],
                resolution=resolution_fit,
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
            ok = np.isfinite(T[fit_idx]) & np.isfinite(predicted_fit)
            if np.any(ok):
                fit_mse = mean_squared_error(T[fit_idx][ok], predicted_fit[ok])
                predicted_full = np.full(T.size, np.nan, dtype=float)
                predicted_full[fit_idx] = predicted_fit
    except Exception:
        return {}

    if fit_mse is None or not np.isfinite(fit_mse):
        return {}
    info = {
        "fit_mse": float(fit_mse),
        "fit_mse_description": (
            "Deterministic timing MSE on cropped/subsampled positions for the final hybrid v(x,t); "
            "it is not the stochastic rsim MSE."
        ),
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
    finite_diff_step = _optimizer_eps_from_fit_step(fit_step)
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

    max_fit_positions = 1500
    valid_idx = np.flatnonzero(valid_full)
    stride = max(1, int(math.ceil(valid_idx.size / max_fit_positions)))
    fit_idx = valid_idx[::stride]
    if fit_idx[-1] != valid_idx[-1]:
        fit_idx = np.concatenate([fit_idx, valid_idx[-1:]])
    T_fit = T[fit_idx]
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

    def predict_timing(values):
        speed = make_speed_internal(values, full=False)
        fork_speed = speed if form == "time" else speed.T
        fraction = map_replicated_fraction_xt(
            firing_fit,
            fork_speed=fork_speed,
            resolution=resolution_fit,
            resolution_time=resolution_time,
            time_grid=time_axis,
            n_bins=fit_idx.size,
            perQ=False,
        )
        survival = np.clip(1.0 - np.asarray(fraction, dtype=float), 0.0, 1.0)
        predicted = np.trapezoid(survival, time_axis, axis=1) + offset
        return predicted

    best = {"loss": np.inf, "mse": np.inf, "params": None, "predicted": None}
    verbose = bool(refine_verbose) if refine_verbose is not None else False
    start_time = time.time()
    calls = 0
    maxiter_display = max(1, maxiter)
    stop_at_call_budget = maxiter > 0

    params0 = np.zeros(2 if form == "time" else 3, dtype=float)
    params0[0] = math.log(np.clip(reference_speed, low, high))
    bounds = [(math.log(low), math.log(high))]
    if form == "spacetime":
        bounds.append((-8.0, 8.0))
    bounds.append((-8.0, 8.0))

    def objective(values):
        nonlocal calls
        values = np.asarray(values, dtype=float)
        if np.any(~np.isfinite(values)):
            return 1e30
        try:
            predicted = predict_timing(values)
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
            best["params"] = values.copy()
            best["predicted"] = predicted.copy()
        done = calls >= maxiter_display or _error_reached(mse, error)
        if verbose and calls <= maxiter_display:
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
    full_speed = make_speed_internal(params, full=True, mse=best["mse"])
    try:
        full_predicted = predict_timing(params)
        predicted_full = np.full(n_positions, np.nan, dtype=float)
        predicted_full[fit_idx] = full_predicted
        full_speed.info["replication_timing_fit"] = predicted_full
        full_speed.info["residual"] = predicted_full - T
    except Exception:
        pass
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
        if verbose and calls <= maxiter_display:
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
        done = it == max(1, maxiter) or _error_reached(mse, error)
        if verbose and (it % print_every == 0 or done):
            etaferr(it=it, maxiter=max(1, maxiter), start_time=start_time, err=mse, done=done)
        if done or maxiter <= 0:
            break
        ratio = float(np.nanmedian(predicted[valid] / T_fit[valid]))
        if not np.isfinite(ratio) or ratio <= 0:
            break
        speed = float(np.clip(speed * ratio ** fit_step, low, high))

    return best_speed


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
        done = it == max(1, maxiter) or _error_reached(mse, error)
        if verbose and (it % print_every == 0 or done):
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
        if verbose and (it % print_every == 0 or done):
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
        if verbose and best["calls"] <= maxiter_display:
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
    if verbose and 0 < best["calls"] < maxiter_display and best_predicted is not None:
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
        if verbose and (i % print_every == 0 or done):
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
        if verbose and (it % print_every == 0 or done):
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

    fit_idx, stride = _speed_fit_indices(
        T - offset,
        resolution,
        perQ=perQ,
        refine_edge_kb=refine_edge_kb,
    )
    T_fit = T[fit_idx]
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
    maxiter = max(0, int(max(space_maxiter, temporal_maxiter)))
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
        q_low = max(low / max(float(np.nanmax(space)), 1e-12), 1e-12)
        q_high = high / max(float(np.nanmin(space)), 1e-12)
        if not np.isfinite(q_high) or q_high <= q_low:
            return 1e-6, 1e6
        return q_low, q_high

    def space_bounds(q):
        q = np.asarray(q, dtype=float).reshape(-1)
        s_low = max(low / max(float(np.nanmax(q)), 1e-12), 1e-12)
        s_high = high / max(float(np.nanmin(q)), 1e-12)
        if not np.isfinite(s_high) or s_high <= s_low:
            return low, high
        return s_low, s_high

    def score(space, q):
        predicted = _predict_timing_separable_speed(
            firing_fit,
            np.asarray(space, dtype=float).reshape(-1)[fit_idx],
            np.asarray(q, dtype=float).reshape(-1),
            T_fit,
            time_axis,
            resolution_fit,
            offset=offset,
        )
        valid = np.isfinite(T_fit) & np.isfinite(predicted) & (T_fit > 0) & (predicted > 0)
        if not np.any(valid):
            return np.inf, predicted
        return mean_squared_error(T_fit[valid], predicted[valid]), predicted

    def spatial_candidate(space, q, predicted):
        valid = np.isfinite(T_fit) & np.isfinite(predicted) & (T_fit > 0) & (predicted > 0)
        ratio_fit = np.ones(fit_idx.size, dtype=float)
        if np.any(valid):
            ratio_fit[valid] = np.clip(predicted[valid] / T_fit[valid], low_ratio, high_ratio)
        ratio_full = np.interp(x_full, x_fit, ratio_fit, left=ratio_fit[0], right=ratio_fit[-1])
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
        candidates = []
        if time_step > 0:
            candidates.append(np.clip(baseline * (1.0 + time_step * t01), q_low, q_high))
            candidates.append(np.clip(baseline * np.maximum(1.0 - time_step * t01, 1e-6), q_low, q_high))
        else:
            candidates.append(np.clip(baseline, q_low, q_high))
        return candidates

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
        if verbose and (it % print_every == 0 or done):
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
        "fork_speed_component_t": "crossed one-step temporal q(t) slope updates",
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
        if verbose:
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
            if verbose:
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
):
    speed = np.asarray(fork_speed_time, dtype=float).reshape(1, -1).copy()
    error = _normalise_error_threshold(error)
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

        done = it == maxiter or _error_reached(err, error)
        if verbose:
            etaferr(it=it, maxiter=maxiter, start_time=start_time, err=err, done=done)
        if done and it != maxiter:
            break

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
):
    speed = np.asarray(fork_speed_spacetime, dtype=float).copy()
    error = _normalise_error_threshold(error)
    maxiter = int(refine_maxiter)
    if maxiter <= 0:
        return speed

    T = np.asarray(observed_timing, dtype=float)
    n = T.size
    if speed.ndim != 2 or speed.shape[1] != n:
        raise ValueError("fork_speed_spacetime must have internal shape (time, positions).")

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
            fork_speed=speed.T,
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

        done = it == maxiter or _error_reached(err, error)
        if verbose:
            etaferr(it=it, maxiter=maxiter, start_time=start_time, err=err, done=done)
        if done and it != maxiter:
            break

    return best_speed


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
):
    """Infer a time-only fork-speed profile from timing and spatial firing."""
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
        {"replication_timing": T, "firing_rate": firing_rate},
        resolution=resolution,
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
                {"replication_timing": T, "firing_rate": direct_spatial_firing},
                resolution=resolution,
                neighbourhood_range=neighbourhood_range,
                fit_step=space_fit_step,
                maxiter=space_maxiter,
                speed_bounds=speed_bounds,
                initial_fork_speed=current_space,
                target_shape="best",
                smoothw=smoothw,
                error=error,
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
):
    """Infer a space-time fork-speed field from timing and I(x,t)."""
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
        {"replication_timing": T, "firing_rate": firing_rate_space},
        resolution=resolution,
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
                firing_rate=firing_rate_xt.T,
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
        firing_rate=firing_rate_xt.T,
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
        if verbose and (it % print_every == 0 or done):
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
        if verbose and (it % print_every == 0 or done):
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
    **kwargs,
):
    """Infer a time-only fork-speed profile from timing and general firing."""
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
    **kwargs,
):
    """Infer a space-time fork-speed field from timing and general firing."""
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
    resolution_time=5.0,
    time_grid=None,
    T0=None,
    initial_firing_rate=None,
    initial_fork_speed=1.4,
    initial_kinetics=None,
    target_form="spacetime",
    space_init_fit_step=0.6,
    space_init_maxiter=10,
    space_init_sim_number=100,
    joint_maxiter=10,
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
    error=None,
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
    error = _normalise_error_threshold(error)
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
            firing_rate=firing_rate.T,
            fork_speed=fork_speed.T,
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

        done = it == maxiter or _error_reached(err, error)
        if verbose and (it % print_every == 0 or done):
            etaferr(it=it, maxiter=maxiter, start_time=start_time, err=err, done=done)
        if done and it != maxiter:
            break

    return {
        "firing_rate": best_firing_rate.T,
        "fork_speed": best_fork_speed.T,
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
