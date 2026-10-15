"""Replication timing from equally spaced high-resolution Repli-seq fractions."""

from __future__ import annotations

import warnings
from collections.abc import Callable

import numpy as np


def replication_timing_from_fractions(
    fractions: np.ndarray,
    *,
    s_phase_duration_min: float = 600.0,
    method: str = "sigmoid",
    progress: Callable[[float], None] | None = None,
    cancel: Callable[[], None] | None = None,
) -> np.ndarray:
    """Return minutes per genomic bin from sixteen chronological fractions.

    ``sigmoid`` estimates T50 by least-squares fitting a logistic cumulative
    distribution, as in Zhao, Sasaki and Gilbert (2020). Fraction boundaries
    are equally spaced over S phase; the cumulative signal starts at zero.
    ``mean`` instead weights each interval midpoint by its observed signal.
    Incomplete or zero-signal bins remain NaN, and a failed fit remains NaN.
    """

    values = np.asarray(fractions, dtype=float)
    if values.ndim != 2 or values.shape[1] != 16:
        raise ValueError("Replication timing requires sixteen columns in S1–S16 order.")
    duration = float(s_phase_duration_min)
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("S-phase duration must be a positive finite number of minutes.")
    if method not in {"sigmoid", "mean"}:
        raise ValueError("Choose sigmoid or mean replication timing.")
    if np.any(values[np.isfinite(values)] < 0) or np.any(np.isinf(values)):
        raise ValueError("Repli-seq fraction signals must be non-negative and finite or missing.")
    if cancel is not None:
        cancel()
    timing = np.full(values.shape[0], np.nan)
    complete = np.isfinite(values).all(axis=1)
    totals = np.sum(values, axis=1)
    valid = np.flatnonzero(complete & (totals > 0))
    if method == "mean":
        midpoints = (np.arange(16) + 0.5) / 16
        timing[valid] = values[valid] @ midpoints / totals[valid] * duration
    else:
        from scipy.optimize import OptimizeWarning, curve_fit
        from scipy.special import expit

        boundaries = np.arange(17, dtype=float)

        def logistic(t, midpoint, width):
            return expit((t - midpoint) / width)

        def jacobian(t, midpoint, width):
            scaled = (t - midpoint) / width
            probability = expit(scaled)
            gradient = probability * (1 - probability)
            return np.column_stack((-gradient / width, -gradient * scaled / width))

        for count, index in enumerate(valid):
            if count % 32 == 0:
                if cancel is not None:
                    cancel()
                if progress is not None:
                    progress(count / max(1, valid.size))
            cumulative = np.r_[0.0, np.cumsum(values[index]) / totals[index]]
            midpoint = float(np.interp(0.5, cumulative, boundaries))
            width = max(float(np.interp(0.75, cumulative, boundaries) -
                              np.interp(0.25, cumulative, boundaries)) / (2 * np.log(3)), 0.1)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", OptimizeWarning)
                    parameters, _ = curve_fit(
                        logistic, boundaries, cumulative,
                        p0=(midpoint, min(width, 16)), jac=jacobian,
                        bounds=([0, 0.01], [16, 16]), max_nfev=160,
                    )
                timing[index] = parameters[0] / 16 * duration
            except (RuntimeError, ValueError, FloatingPointError):
                continue
    if cancel is not None:
        cancel()
    if progress is not None:
        progress(1.0)
    return timing
