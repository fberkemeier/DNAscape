"""Adjoint of the original constant-speed, linear firing-to-timing map.

The inverse uses this only when the historical local ratio update increases
MSE. The forward model and per-bin firing units remain unchanged. Both passes
use O(positions) memory, without a dense Jacobian or a time-position array.
"""

import numpy as np
from scipy.ndimage import uniform_filter1d

from .utils import check_cancelled, fast_shift_add


def timing_mse_gradient(
    rates, observed, *, fork_speed, resolution, neighbourhood_range,
    fit_mask, edge_bins,
):
    """Return MSE, its rate gradient and timing, including reflected context.

    At distance k the legacy integrator adds
    exp(-A_previous) * (1 - exp(-dt * U_k)) / U_k,
    where U_k is the sum of rates within k bins. Reverse differentiation of
    that recurrence includes all neighboring origins. Reflection is pulled
    back through the same index map used by the forward prediction.
    """
    rates = np.asarray(rates, dtype=float)
    indexes = np.arange(rates.size)
    edge_bins = int(edge_bins)
    if edge_bins:
        indexes = np.pad(indexes, edge_bins, mode="reflect")
    padded = rates[indexes]
    output_slice = slice(edge_bins, edge_bins + rates.size)
    dt = float(resolution) / float(fork_speed)
    radius = max(1, int(float(neighbourhood_range) / float(resolution)))
    neighbourhood = padded.copy()
    exponent = np.zeros_like(padded)
    survival = np.ones_like(padded)
    timing = np.zeros_like(padded)

    for distance in range(radius + 1):
        if distance % 32 == 0:
            check_cancelled()
        if distance:
            fast_shift_add(neighbourhood, padded, distance, perQ=False)
            fast_shift_add(neighbourhood, padded, -distance, perQ=False)
        next_exponent = exponent + dt * neighbourhood
        next_survival = np.exp(-next_exponent)
        increment = (survival - next_survival) / np.clip(neighbourhood, 1e-80, None)
        increment[neighbourhood == 0] = survival[neighbourhood == 0] * dt
        timing += increment
        exponent, survival = next_exponent, next_survival

    fitted = timing[output_slice].copy()
    residual = fitted - observed
    mse = float(np.mean(residual[fit_mask] ** 2))
    weights = np.zeros_like(padded)
    weights[output_slice] = 2.0 * residual * fit_mask / np.count_nonzero(fit_mask)
    exponent_adjoint = np.zeros_like(padded)
    gradient = np.zeros_like(padded)

    for distance in range(radius, -1, -1):
        if distance % 32 == 0:
            check_cancelled()
        q = dt * neighbourhood
        previous_exponent = np.maximum(exponent - q, 0.0)
        previous_survival = np.exp(-previous_exponent)
        term = np.divide(previous_survival - survival, neighbourhood,
                         out=previous_survival * dt, where=neighbourhood > 0)
        derivative = np.divide(dt * survival - term, neighbourhood,
                               out=np.zeros_like(term), where=neighbourhood > 0)
        # Avoid subtraction of nearly equal terms in the U derivative.
        small = q < 1e-4
        qs = q[small]
        derivative[small] = -previous_survival[small] * dt**2 * (
            0.5 - qs / 3.0 + qs**2 / 8.0 - qs**3 / 30.0
        )
        neighbourhood_adjoint = weights * derivative + exponent_adjoint * dt
        exponent_adjoint -= weights * term
        # The transpose of a centered box sum is the same box sum. Constant
        # padding preserves linear boundaries; periodic wrap is not used.
        width = 2 * distance + 1
        gradient += width * uniform_filter1d(
            neighbourhood_adjoint, size=width, mode="constant",
        )
        if 0 < distance < padded.size:
            neighbourhood[distance:] -= padded[:-distance]
            neighbourhood[:-distance] -= padded[distance:]
            np.maximum(neighbourhood, padded, out=neighbourhood)
        exponent, survival = previous_exponent, previous_survival

    gradient = np.bincount(indexes, weights=gradient, minlength=rates.size)
    return mse, gradient, fitted
