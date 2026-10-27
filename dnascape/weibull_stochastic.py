"""Weibull-constrained refinement using the established stochastic fork model."""

import time

import numpy as np

from ._fit_bounds import fit_bounds, initial_rate
from .mapping import _expand_space_time_field
from .simulation import rsim
from .utils import check_cancelled, etaferr, measured_work


def _simulation_speed(fork_speed, output_t, simulation_t, n, dt):
    """Retain scalar/x/t fields instead of expanding them to a dense matrix."""
    check_cancelled()
    raw = np.asarray(fork_speed, dtype=float)
    if np.any(~np.isfinite(raw)) or np.any(raw <= 0):
        raise ValueError("fork_speed must contain finite positive values.")
    if raw.size == 1:
        return float(raw.reshape(-1)[0])
    if raw.ndim == 1 and raw.size == n:
        return raw
    if raw.ndim == 2 and raw.shape == (n, 1):
        return raw[:, 0]
    if raw.ndim == 2 and raw.shape[0] == 1:
        temporal = _expand_space_time_field(raw, output_t.size, 1, output_t, dt, "fork_speed", positive=True)
        return np.interp(simulation_t, output_t, temporal[:, 0])[None, :]
    temporal = _expand_space_time_field(raw, output_t.size, n, output_t, dt, "fork_speed", positive=True)
    result = np.empty((n, simulation_t.size))
    for index in range(n):
        if index % 128 == 0:
            check_cancelled()
        result[index] = np.interp(simulation_t, output_t, temporal[:, index])
    return result


def _interval_hazard(alpha, shape, times, delay, dt, *, temporal=False):
    check_cancelled()
    if temporal and np.all(delay == delay[0]):
        alpha, delay = alpha[:1], delay[:1]
    hazard = np.empty((alpha.size, times.size))
    columns = max(1, 1_000_000 // max(1, times.size))
    for start in range(0, alpha.size, columns):
        check_cancelled()
        stop = start + columns
        age_lo = np.maximum(times[:, None] - delay[None, start:stop], 0.)
        age_hi = np.maximum(times[:, None] + dt - delay[None, start:stop], 0.)
        hazard[start:stop] = (alpha[start:stop] * (age_hi**shape - age_lo**shape) / dt).T
    return hazard


def _spatial_log_roughness(log_alpha, *, periodic):
    """Mean neighbour penalty with equal weights over a line or full ring."""
    if log_alpha.size <= 1:
        return 0.0
    differences = np.roll(log_alpha, -1) - log_alpha if periodic else np.diff(log_alpha)
    return float(np.mean(differences**2))


def fit_stochastic_weibull(
    observed_timing, fork_speed, *, resolution, resolution_time, time_grid,
    T0, target_form, weibull_shape, weibull_delay, fit_weibull_shape,
    alpha_smooth, alpha_prior, maxiter, refine_edge_kb, refine_verbose, perQ,
    floor, error, stall_rate, stall_rate_per_kb, tau, sim_number, refine_seed,
    rate_bounds=None, initial_initiation_rate=None,
):
    from .fitting import (
        _refinement_crop_mask,
        _RmapInfoArray,
        _weibull_alpha_initial,
        _weibull_delay_profile,
        _weibull_time_axis_for_timing,
    )

    observed = np.asarray(observed_timing, dtype=float)
    if observed.ndim != 1 or observed.size == 0:
        raise ValueError("observed_timing must be a non-empty spatial vector.")
    offset = 0.0 if T0 is None else float(T0)
    relative = observed - offset
    mask = np.isfinite(relative) & (relative > 0)
    mask &= _refinement_crop_mask(observed.size, resolution, perQ, refine_edge_kb)
    if not np.any(mask):
        raise ValueError("observed_timing needs finite positive values after edge filtering.")
    if int(sim_number) < 1 or int(maxiter) < 0:
        raise ValueError("sim_number must be positive and maxiter non-negative.")
    k = float(weibull_shape)
    if not np.isfinite(k) or k <= 0:
        raise ValueError("weibull_shape must be finite and positive.")
    if alpha_smooth < 0 or alpha_prior < 0:
        raise ValueError("alpha_smooth and alpha_prior must be non-negative.")
    delay = _weibull_delay_profile(weibull_delay, observed.size)
    output_t = _weibull_time_axis_for_timing(relative[mask], resolution_time, time_grid)
    dt = float(resolution_time)
    # rsim proposes no new initiations beyond its supplied hazard horizon.
    # Cover six observed mean durations to make that numerical cutoff explicit.
    horizon = max(float(output_t[-1]), float(np.max(relative[mask])) * 6.0, float(np.max(delay)) + dt)
    simulation_t = np.arange(0.0, horizon + dt, dt)
    speed = _simulation_speed(fork_speed, output_t, simulation_t, observed.size, dt)
    initial_alpha = _weibull_alpha_initial(
        np.where(mask, relative, np.median(relative[mask])), fork_speed, k, floor,
    )
    temporal = target_form in {"time", "t"}
    if temporal:
        initial_alpha[:] = np.median(initial_alpha[mask])
    low, high = fit_bounds(rate_bounds, (floor, np.exp(100)), label="Initiation-amplitude bounds")
    initial_alpha = initial_rate(initial_initiation_rate, initial_alpha, low, high)
    z_initial = np.log(np.maximum(initial_alpha, floor))
    z = z_initial.copy()
    seed = 0 if refine_seed is None else int(refine_seed)
    started = time.time()
    history = []

    def evaluate(log_alpha, shape, diagnostics=False):
        check_cancelled()
        alpha = np.exp(np.clip(log_alpha, np.log(low), np.log(high)))
        # Exact interval-averaged hazard avoids the k<1 activation singularity.
        hazard = _interval_hazard(alpha, shape, simulation_t, delay, dt, temporal=temporal)
        simulation = rsim(
            # Weibull amplitudes describe initiation per kb. The legacy
            # simulator accepts initiation hazard per spatial bin.
            initiation_rate=hazard * float(resolution), fork_speed=speed, n_bins=observed.size,
            resolution_space=resolution, resolution_time=dt, sim_number=int(sim_number),
            perQ=perQ, stall_rate=stall_rate, stall_rate_per_kb=stall_rate_per_kb, tau=tau,
            seed=seed, verbose=False, time_statsQ=False, time_stats_xtQ=diagnostics,
            time_grid=output_t, retain_rep_times_per_sim=False, retain_inter_origin_distances=False,
        )
        prediction = np.asarray(simulation["replication_timing"], dtype=float)
        if np.any(~np.isfinite(prediction[mask])):
            raise ValueError("The supplied stalling model did not yield finite replication times.")
        residual = prediction[mask] - relative[mask]
        mse = float(np.mean(residual**2))
        regularisation = alpha_smooth * _spatial_log_roughness(log_alpha, periodic=perQ) if not temporal else 0.0
        regularisation += alpha_prior * float(np.mean((log_alpha - z_initial)**2))
        return mse + regularisation, mse, prediction, simulation

    with measured_work(maxiter + 1):
        best_loss, best_mse, prediction, _ = evaluate(z, k)
    history.append(best_mse)
    for iteration in range(int(maxiter)):
        if error is not None and best_mse <= error:
            break
        ratio = np.ones(observed.size)
        ratio[mask] = np.clip(prediction[mask] / relative[mask], 0.25, 4.0)
        correction = (k + 1.0) * np.log(ratio)
        if temporal:
            correction[:] = np.mean(correction[mask])
        elif alpha_smooth > 0 and z.size > 2:
            padded = np.pad(correction, 1, mode="wrap" if perQ else "edge")
            correction = 0.25 * padded[:-2] + 0.5 * padded[1:-1] + 0.25 * padded[2:]
        accepted = False
        # Common random numbers and monotone acceptance prevent stochastic
        # differences between ensembles from masquerading as fitting gains.
        for step in (0.5, 0.25, 0.125):
            proposed_z = np.clip(z + step * correction, np.log(low), np.log(high))
            loss, mse, proposed_prediction, _ = evaluate(proposed_z, k)
            if loss < best_loss:
                z, best_loss, best_mse, prediction = proposed_z, loss, mse, proposed_prediction
                accepted = True
                break
        if fit_weibull_shape:
            for factor in (np.exp(-0.03), np.exp(0.03)):
                proposed_k = float(np.clip(k * factor, 0.25, 8.0))
                # Preserve the hazard near the observed mean while changing
                # shape so amplitude and shape do not move in lockstep.
                proposed_z = np.clip(z + (k - proposed_k) * np.log(np.median(relative[mask])), np.log(low), np.log(high))
                loss, mse, proposed_prediction, _ = evaluate(proposed_z, proposed_k)
                if loss < best_loss:
                    z, k = proposed_z, proposed_k
                    best_loss, best_mse, prediction = loss, mse, proposed_prediction
                    accepted = True
        history.append(best_mse)
        if refine_verbose:
            etaferr(it=iteration + 1, maxiter=max(1, int(maxiter)), start_time=started,
                    err=best_mse, done=iteration + 1 == maxiter or not accepted)
        if not accepted:
            break
    diagnostics = observed.size * output_t.size <= 1_000_000
    _, best_mse, prediction, simulation = evaluate(z, k, diagnostics=diagnostics)
    alpha = np.exp(z)
    hazard = np.maximum(_interval_hazard(alpha, k, output_t, delay, dt, temporal=temporal), floor)
    values = np.mean(hazard, axis=0, keepdims=True) if temporal else hazard
    info = {
        "alpha": alpha, "lambda": alpha ** (-1.0 / k), "weibull_shape": k,
        "weibull_delay": delay, "time_grid": output_t, "target_shape": "weibull",
        "target_form": target_form, "predicted_timing": prediction + offset,
        "residual": prediction + offset - observed, "mse": best_mse,
        "fit_mse": best_mse, "fit_history": np.asarray(history),
        "forward_backend": "stochastic_weibull", "simulation_hazard_horizon": simulation_t[-1] + dt,
        "sim_number": int(sim_number), "refine_seed": seed,
    }
    if diagnostics:
        info["survival"] = 1.0 - np.asarray(simulation["replicated_fraction_xt"])
    else:
        info["spacetime_diagnostics_omitted"] = True
    return _RmapInfoArray(values, info)
