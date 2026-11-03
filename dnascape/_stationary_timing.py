"""Bounded-time-block stationary timing and its exact discrete adjoint."""

import numpy as np

from ._hazard_static import StationaryHazard
from ._resources import ensure_capacity, working_bytes
from .utils import check_cancelled, report_measured_work


def stationary_timing(rate, speed, clock, weights, dx, periodic, radius,
                      observed, valid, offset, scale_loss, with_speed=False):
    """Integrate the selected clock without keeping its full space-time field.

    The second pass recomputes each block for the adjoint. It uses the same
    centred-cell integrals, quadrature and boundaries as StationaryHazard.
    """
    n = len(rate)
    ensure_capacity(n * 512, "Mapping the selected spatial region")
    # Include two hazard buffers, survival, adjoint and NumPy temporaries.
    block = max(1, min(64, working_bytes(16 * 1024**2) // max(1, n * 96) - 1))
    ensure_capacity((block + 1) * n * 96, "Mapping the selected time samples")
    prediction = np.full(n, weights[0] + offset)
    sensitivity = np.zeros(n)
    final = np.ones(n)

    def blocks():
        for first in range(1, len(clock), block):
            check_cancelled()
            last = min(first + block, len(clock))
            selected = slice(first, last)
            operator = StationaryHazard(speed, np.r_[0., clock[selected]], dx,
                                        periodic, radius, cache_bytes=0)
            hazard = operator.forward(rate, record=True)
            survival = np.exp(-hazard[1:])
            yield selected, operator, hazard, survival

    for selected, _operator, hazard, survival in blocks():
        prediction += weights[selected] @ survival
        sensitivity += np.einsum("t,tx,tx->x", weights[selected], hazard[1:], survival)
        final = survival[-1].copy()
        report_measured_work(.5 * selected.stop / len(clock))
    residual = prediction - observed
    q = np.where(valid, residual, 0.) * (2. / (int(valid.sum()) * scale_loss))
    gradient = np.zeros(n)
    for selected, operator, hazard, survival in blocks():
        derivative = np.zeros_like(hazard)
        derivative[1:] = -weights[selected, None] * survival * q[None, :]
        rate_gradient, speed_gradient = operator.backward(derivative, with_speed=with_speed)
        gradient += speed_gradient if with_speed else rate_gradient
        report_measured_work(.5 + .5 * selected.stop / len(clock))
    return prediction, residual, sensitivity, final, gradient[None, :]
