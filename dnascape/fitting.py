"""Fitting algorithms and fitting-oriented helpers."""

import os
import math
import time

import numpy as np

from .constants import mapt
from .io import _resolve_data_path, savecsv
from .mapping import map_firing_timing
from .simulation import rsim
from .utils import etaferr, mean_squared_error

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
    neighbourhood_range=2000,
    fit_step=0.6,
    maxiter=10,
    sim_number=1000,
    err_threshold=15.0,
    deviation_threshold=None,
    verbose=False,
    perQ=False,
    chr_number=1,
    save_errQ=False
):
    """Infer a spatial firing-rate profile from replication timing.

    ``fork_speed`` is the canonical argument name. ``fork_speeds`` is accepted
    as an older compatibility alias.
    """

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

    if stall_rate > 0. or variable_fork_speed:
        maxiter0 = 0
    else:
        maxiter0 = maxiter

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

        if verbose:
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
    
            eps = 1e-12
            ratio = (T_pred + eps) / (T_target + eps)
            rates *= np.power(ratio, alpha)
            np.clip(rates, rate_min, rate_max, out=rates)
    
            mse = T_pred - T_target
            mse = float(np.mean(mse**2))
            hist_mse.append(mse)
    
            if verbose:
                etaferr(it=k+1, maxiter=iterations, start_time=start_time, err=mse)
                mse_list.append(mse)

        x = rates

    #np.savetxt(f"data/fit_error/mse_chr{chr_number}.txt", mse_list, fmt="%.6f")

    if save_errQ:
        savecsv(arr=mse_list, chr_number=chr_number, stitle=f'fit_error', outpath="fit_error")
        
    return x



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
    saveQ=False,
    return_info=False,
    source_feature=None,
    d1=None,
    d2=None,
    **kwargs
):
    """Fit or map one replication feature to another.

    Prefer the same terms used by ``rmap``: ``source``, ``target``, ``data``,
    ``assumptions``, and ``method``. If ``source`` is not a string or mapping,
    it is treated as the older data-array argument. The older
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
    if isinstance(source, dict):
        if data is not None:
            raise ValueError("Pass either source={feature: data} or source='feature' with data=..., not both.")
        if len(source) != 1:
            raise ValueError("rfit source dictionaries must contain exactly one feature.")
        source_key, data = next(iter(source.items()))
        source_feature = rmap.canonical(source_key)
        source_payload = {source_feature: data}
    else:
        source_feature = rmap.canonical(source)

    if data is None:
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
    if "fork_speed" not in merged_assumptions:
        merged_assumptions["fork_speed"] = fork_speed
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
