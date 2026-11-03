"""Exact cell/time-hat cone convolution for spatially constant fork speed.

Linear temporal interpolation of the initiation samples is integrated exactly,
including the half hat at time zero. Spatial samples own centred constant cells.
The radius clips the convolution kernel, not the observed genomic interval.
"""

import numpy as np
from scipy.fft import irfftn, next_fast_len, rfftn

from .utils import check_cancelled, report_measured_work


def _hat_primitive(q, dt, negative_half=False):
    """Primitive and value of the integrated temporal nodal hat."""
    u = np.clip(q + dt, 0., dt)
    primitive = u**3 / (6. * dt)
    value = u*u / (2. * dt)
    if negative_half:
        primitive += .5 * dt * np.maximum(q, 0.)
    else:
        u = np.clip(q, 0., dt)
        primitive += .5 * dt * u + .5 * u*u - u**3 / (6. * dt)
        primitive += dt * np.maximum(q - dt, 0.)
        value += u - .5 * u*u / dt
    return primitive, value


class ConstantSpeedConvolution:
    backend = "exact_cone_convolution"

    def __init__(self, speed_tx, time_grid, resolution=1., periodic=False, radius=2000.):
        self.speed = np.asarray(speed_tx)
        self.t = np.asarray(time_grid, dtype=float)
        self.n = self.speed.shape[1]
        self.dx = float(resolution)
        self.v = float(self.speed[0, 0])
        self.periodic = bool(periodic)
        self.radius = min(float(radius), .5 * self.n * self.dx) if periodic else float(radius)
        self.dt = float(self.t[1] - self.t[0])
        reach = min(self.radius, self.v * (self.t[-1] + self.dt))
        self.width = min(self.n // 2 if periodic else self.n - 1,
                         int(np.ceil(reach / self.dx + .5)))
        self.shape = (next_fast_len(2 * len(self.t) - 1),
                      self.n if periodic else next_fast_len(self.n + self.width))
        from ._resources import ensure_capacity
        ensure_capacity(self.shape[0] * self.shape[1] * 128, "Allocating the Mapping transforms")
        self._kernels = None
        self._rate = None
        self._last_spectrum = None
        self.diagnostics = {"backend": self.backend, "convolution_shape": self.shape,
                            "spatial_discretization": "centred piecewise-constant cells"}

    @classmethod
    def applicable(cls, speed, clock, *, max_fft_cells=24_000_000):
        if len(clock) < 2 or not np.allclose(np.diff(clock), clock[1] - clock[0], rtol=1e-11, atol=1e-12):
            return False
        if not np.all(speed == speed[0, 0]):
            return False
        # A conservative bound before allocating the transforms. Large grids
        # use the streaming transport calculation instead.
        return next_fast_len(2 * len(clock) - 1) * next_fast_len(2 * speed.shape[1]) <= max_fft_cells

    def _kernel(self, negative_half=False, derivative=False):
        check_cancelled()
        distance = np.arange(self.width + 1, dtype=float) * self.dx
        a = np.maximum(0., distance - .5 * self.dx)
        b = np.minimum(self.radius, distance + .5 * self.dx)
        active = b > a
        age = self.t[:, None]
        qa, qb = age - a / self.v, age - b / self.v
        pa, va = _hat_primitive(qa, self.dt, negative_half)
        pb, vb = _hat_primitive(qb, self.dt, negative_half)
        if derivative:
            weights = pa - pb + va * (a / self.v) - vb * (b / self.v)
        else:
            weights = self.v * (pa - pb)
        saturated = qb >= (0. if negative_half else self.dt)
        weights = np.where(saturated, 0. if derivative else
                           (.5 if negative_half else 1.) * self.dt * (b - a), weights)
        weights[:, ~active] = 0.
        weights[:, 0] *= 2.
        kernel = np.zeros((len(self.t), self.shape[1]))
        kernel[:, :self.width + 1] += weights
        if self.width:
            kernel[:, -np.arange(1, self.width + 1)] += weights[:, 1:]
        return rfftn(kernel, s=self.shape)

    def _prepare(self):
        if self._kernels is None:
            self._kernels = self._kernel(), self._kernel(negative_half=True)

    def forward(self, rate_tx, record=False):
        check_cancelled()
        report_measured_work(.05)
        self._prepare()
        report_measured_work(.3)
        check_cancelled()
        rate = np.asarray(rate_tx, dtype=float)
        source = rfftn(rate, s=self.shape)
        initial = np.fft.rfft(rate[0], n=self.shape[1])
        spectrum = source * self._kernels[0] - initial[None, :] * self._kernels[1]
        report_measured_work(.7)
        check_cancelled()
        result = irfftn(spectrum, s=self.shape)[:len(self.t), :self.n]
        result[0] = 0.
        if record:
            self._rate = rate.copy()
            self._last_spectrum = source, initial
        report_measured_work(1.)
        return np.maximum(result, 0.)

    def backward(self, dH, with_speed=True):
        if self._rate is None:
            raise RuntimeError("Record a forward prediction before its adjoint.")
        check_cancelled()
        gradient = np.array(dH, dtype=float, copy=True)
        gradient[0] = 0.
        spectrum = rfftn(gradient, s=self.shape)
        rate_gradient = irfftn(spectrum * self._kernels[0].conj(), s=self.shape)[:len(self.t), :self.n].copy()
        correction = irfftn(spectrum * self._kernels[1].conj(), s=self.shape)[0, :self.n]
        rate_gradient[0] -= correction
        speed_gradient = np.zeros_like(self._rate) if with_speed else np.broadcast_to(0., self._rate.shape)
        if with_speed:
            source, initial = self._last_spectrum
            derivative = irfftn(source * self._kernel(derivative=True)
                         - initial[None, :] * self._kernel(negative_half=True, derivative=True),
                         s=self.shape)[:len(self.t), :self.n]
            speed_gradient[0, 0] = np.sum(gradient * derivative)
        return rate_gradient, speed_gradient
