"""Deterministic kinetic inference, including explicitly restricted joint fits."""

import numpy as np
from scipy.optimize import minimize

from ._kinetic_fit import fit_kinetic
from ._timing_quadrature import StationaryTiming
from .utils import check_cancelled


def fit_joint_static(timing, initiation_template, speed_template, *,
                     speed_prior, resolution=1., perQ=False,
                     neighbourhood_range=2000., observation_weights=None,
                     scale_bounds=(.1, 10.), maxiter=30, quadrature_tolerance=1e-5):
    """Fit two positive template multipliers with an independent speed prior.

    Both spatial template shapes stay fixed. ``speed_prior=(scale, log_sd)``
    anchors the speed multiplier through a Gaussian prior on its logarithm.
    This restriction is deliberate: timing alone cannot in general identify
    both kinetic fields. The returned MSE measures timing fit, not evidence
    of parameter recovery. Templates use density/kb/min and speed kb/min.
    """
    timing = np.asarray(timing, float)
    if timing.ndim != 1 or not len(timing):
        raise ValueError("Timing must be a spatial vector.")
    valid = np.isfinite(timing)
    if not np.any(valid) or np.any(timing[valid] <= 0.):
        raise ValueError("Supply finite positive mean replication times.")
    rate = np.broadcast_to(np.asarray(initiation_template, float), timing.shape).copy()
    speed = np.broadcast_to(np.asarray(speed_template, float), timing.shape).copy()
    if np.any(~np.isfinite(rate)) or np.any(rate <= 0.) or np.any(~np.isfinite(speed)) or np.any(speed <= 0.):
        raise ValueError("Joint-fit templates must be finite and positive.")
    center, deviation = map(float, speed_prior)
    lower, upper = map(float, scale_bounds)
    if not (np.isfinite([center, deviation, lower, upper]).all()
            and center > 0. and deviation > 0. and 0. < lower < upper):
        raise ValueError("Use positive finite prior parameters and increasing positive scale bounds.")
    weights = (np.ones(timing.shape) if observation_weights is None else
               np.broadcast_to(np.asarray(observation_weights, float), timing.shape))
    if np.any(~np.isfinite(weights)) or np.any(weights < 0.):
        raise ValueError("Observation weights must be finite and non-negative.")
    weights = np.where(valid, weights, 0.)
    if not weights.sum():
        raise ValueError("At least one observation must have positive weight.")
    weights = weights/weights.sum()
    scale = max(float(np.median(timing[valid]))**2, 1.)
    history = []
    cached = [None, None]

    def objective(parameters):
        check_cancelled()
        if cached[0] is not None and np.array_equal(parameters, cached[0]):
            return cached[1]
        a, b = np.exp(parameters)
        # Every allowed source has arrived by this horizon. H is then affine,
        # so the remaining exponential timing tail is integrated exactly.
        cap = min(float(neighbourhood_range), len(timing)*resolution/(2. if perQ else 1.))
        horizon = max(6.*float(np.max(timing[valid])), cap/float(np.min(speed*b)))
        operator = StationaryTiming(speed*b, horizon, resolution=resolution,
                                     periodic=perQ, radius=neighbourhood_range,
                                     tolerance=quadrature_tolerance)
        prediction = operator.predict(rate*a)
        residual = np.where(valid, prediction-timing, 0.)
        mse = float(weights@(residual**2))
        gi, gv = operator.backward_timing(2.*weights*residual/scale)
        prior = (parameters[1]-np.log(center))/deviation
        gradient = np.array([gi@(rate*a), gv@(speed*b)+prior/deviation])
        value = mse/scale+.5*prior**2, gradient, prediction, mse
        cached[:] = [parameters.copy(), value]
        return value

    def evaluate(parameters):
        value, gradient, _, _ = objective(parameters)
        return value, gradient

    def accepted(parameters):
        history.append(objective(parameters)[3])

    result = minimize(evaluate, np.log([1., np.clip(center, lower, upper)]), jac=True,
                      method="L-BFGS-B", bounds=[(np.log(lower), np.log(upper))]*2,
                      callback=accepted, options={"maxiter": int(maxiter), "ftol": 1e-12, "gtol": 1e-8})
    loss, _, prediction, mse = objective(result.x)
    a, b = np.exp(result.x)
    return dict(initiation_rate=rate*a, fork_speed=speed*b,
                initiation_scale=float(a), speed_scale=float(b),
                replication_timing_fit=prediction, fit_mse=mse, loss=loss,
                iterations=int(result.nit), success=bool(result.success),
                message=str(result.message), fit_history=np.asarray(history),
                restriction="fixed spatial shapes; independent log-speed prior")


__all__ = ["fit_kinetic", "fit_joint_static"]
