"""Bounded timing-domain inversion of the directionality transport identity."""

from __future__ import annotations

import time

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse.linalg import LinearOperator

from .utils import _progress_update, check_cancelled


def integrated_directionality(increments, *, periodic=False, centered=False):
    """Apply the same discrete integral as ``map_directionality_timing``."""
    values = np.cumsum(increments, axis=0)
    if periodic:
        values -= np.arange(len(values)) * np.sum(increments) / len(values)
    if centered:
        values -= np.mean(values)
    return values


def fit_spatial_directionality_speed(
    timing,
    directionality,
    *,
    resolution,
    periodic,
    reference_time,
    reference_speed,
    bounds,
    regularization,
):
    """Fit site speeds with cumulative residuals and an analytic O(n) Jacobian.

    Independent timing errors remain independent observations: no numerical
    derivative of timing is taken. First differences of site speed select a
    smooth solution where the transport equation provides little information.
    A robust second-difference noise estimate strengthens this penalty for
    noisy tracks without interpreting isolated sharp features as global noise.
    """
    count = timing.size
    centered = periodic or reference_time is None
    observed = timing - (np.mean(timing) if centered else reference_time)
    forcing = resolution * directionality
    reference_steps = forcing / reference_speed
    differences = np.diff(timing, n=2)
    noise = (
        (
            float(np.median(np.abs(differences - np.median(differences))))
            / (0.6744897501960817 * np.sqrt(6.0))
        )
        if differences.size
        else 0.0
    )
    step_scale = float(np.sqrt(np.mean(reference_steps**2)))
    signal_scale = max(float(np.std(timing)), step_scale, np.finfo(float).eps)
    penalty = regularization * max(step_scale, np.sqrt(noise * signal_scale)) / reference_speed
    smooth_count = count if periodic else count - 1
    ramp = np.arange(count) / count
    max_evaluations = 100
    evaluations = 0
    started = time.perf_counter()

    def integrate(values):
        return integrated_directionality(values, periodic=periodic, centered=centered)

    def integrate_transpose(values):
        if centered:
            values = values - np.mean(values)
        result = np.cumsum(values[::-1])[::-1]
        if periodic:
            result -= np.dot(ramp, values)
        return result

    def edges(speed):
        result = 0.5 * (speed + np.roll(speed, 1))
        if not periodic:
            result[0] = speed[0]
        return result

    def edge_transpose(values):
        result = 0.5 * (values + np.roll(values, -1))
        if not periodic:
            result[0] += 0.5 * values[0]
            result[-1] -= 0.5 * values[0]
        return result

    def smooth(values):
        return values - np.roll(values, 1) if periodic else np.diff(values)

    def smooth_transpose(values):
        if periodic:
            return values - np.roll(values, -1)
        result = np.zeros(count)
        result[:-1] -= values
        result[1:] += values
        return result

    def residual(speed):
        nonlocal evaluations
        check_cancelled()
        timing_residual = integrate(forcing / edges(speed)) - observed
        evaluations += 1
        elapsed = time.perf_counter() - started
        _progress_update(
            evaluations, max_evaluations,
            elapsed * max(0, max_evaluations - evaluations) / evaluations,
            label="Fitting fork speed", err=float(np.mean(timing_residual ** 2)),
        )
        return (
            np.concatenate((timing_residual, penalty * smooth(speed)))
            / signal_scale
        )

    def jacobian(speed):
        weights = -forcing / edges(speed) ** 2

        def matvec(vector):
            vector = np.asarray(vector).reshape(-1)
            return (
                np.concatenate((integrate(weights * edges(vector)), penalty * smooth(vector)))
                / signal_scale
            )

        def rmatvec(vector):
            vector = np.asarray(vector).reshape(-1)
            return (
                edge_transpose(weights * integrate_transpose(vector[:count]))
                + penalty * smooth_transpose(vector[count:])
            ) / signal_scale

        return LinearOperator(
            (count + smooth_count, count), matvec=matvec, rmatvec=rmatvec, dtype=float
        )

    initial = np.full(count, np.clip(reference_speed, *bounds))
    solution = least_squares(
        residual,
        initial,
        jac=jacobian,
        bounds=bounds,
        method="trf",
        tr_solver="lsmr",
        max_nfev=max_evaluations,
        ftol=1e-6,
        xtol=1e-6,
        gtol=1e-6,
        tr_options={"atol": 1e-6, "btol": 1e-6},
    )
    if not solution.success or np.any(~np.isfinite(solution.x)):
        raise RuntimeError("Fork-speed reconstruction did not converge to a finite profile.")
    final_residual = integrate(forcing / edges(solution.x)) - observed
    _progress_update(
        evaluations, evaluations, 0., label="Fitting fork speed",
        err=float(np.mean(final_residual ** 2)),
    )
    return np.asarray(solution.x, dtype=float)
