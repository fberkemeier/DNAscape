"""Cell-exact convolution for stationary initiation and time-only speed.

An origin at distance r can fire until sigma=D^-1(D(t)-r). Integrating
sigma over a source cell requires only J(s)=integral_0^s tau*v(tau) d tau.
Each observation time therefore needs one spatial convolution, not a
space-time transport solve. The reverse pass differentiates the same kernels.
"""

import numpy as np
from scipy.fft import irfft, next_fast_len, rfft

from .utils import check_cancelled, report_measured_work


class StationarySourceHazard:
    backend = "stationary_source_convolution"

    def __init__(self, speed, clock, resolution=1.0, periodic=False, radius=2000.0,
                 *, differentiate_speed=True):
        self.speed = np.asarray(speed)
        self.t = np.array(clock, dtype=float, copy=True)
        if (
            self.speed.ndim != 2
            or self.t.ndim != 1
            or len(self.t) < 2
            or self.speed.shape != (len(self.t), self.speed.shape[1])
            or not self.speed.shape[1]
            or self.t[0] != 0
            or np.any(~np.isfinite(self.t))
            or np.any(np.diff(self.t) <= 0)
        ):
            raise ValueError("Speed must match a strictly increasing time grid starting at zero.")
        if (
            np.any(~np.isfinite(self.speed[:, 0]))
            or np.any(self.speed[:, 0] <= 0)
            or self.speed.strides[1] != 0
            and np.any(self.speed != self.speed[:, :1])
        ):
            raise ValueError("Speed must be positive and uniform in space.")
        self.speed = np.broadcast_to(self.speed[:, :1].copy(), self.speed.shape)
        self.dt = np.diff(self.t)
        self.n = self.speed.shape[1]
        self.dx = float(resolution)
        if not np.isfinite(self.dx) or self.dx <= 0 or np.isnan(radius) or radius <= 0:
            raise ValueError("Spatial resolution and influence radius must be positive.")
        self.periodic = bool(periodic)
        self.velocity = self.speed[:, 0]
        self.travel = np.r_[
            0.0, np.cumsum(0.5 * (self.velocity[:-1] + self.velocity[1:]) * self.dt)
        ]
        self.cap = min(float(radius), self.n * self.dx / 2 if periodic else np.inf)
        reach = min(self.cap, float(self.travel[-1]))
        self.width = min(
            self.n // 2 if periodic else self.n - 1, int(np.ceil(reach / self.dx + 0.5))
        )
        self.length = self.n if periodic else next_fast_len(self.n + self.width)
        self._spectrum = None
        self._state = None
        self.differentiate_speed = differentiate_speed

    @staticmethod
    def applicable(speed, clock, *, max_fft_cells=24_000_000):
        return len(clock) > 1 and len(clock) * next_fast_len(2 * speed.shape[1]) <= max_fft_cells

    def _primitive(self, query):
        z = np.maximum(query, 0.0)
        index = np.clip(np.searchsorted(self.travel, z, side="right") - 1, 0, len(self.dt) - 1)
        offset = z - self.travel[index]
        vl, vr = self.velocity[index], self.velocity[index + 1]
        slope = (vr - vl) / self.dt[index]
        root = np.sqrt(np.maximum(vl * vl + 2 * slope * offset, 0.0))
        s = np.divide(2 * offset, vl + root, out=np.zeros_like(offset), where=offset > 0.0)
        s = np.clip(s, 0.0, self.dt[index])
        a1 = s * s / (2 * self.dt[index])
        a0 = s - a1
        c1 = s**3 / (3 * self.dt[index])
        c0 = 0.5 * s * s - c1
        value = self.J[index] + self.t[index] * (vl * a0 + vr * a1) + vl * c0 + vr * c1
        return value, (index, s, a0, a1, c0, c1, query > 0.0)

    def _prepare_primitives(self):
        if hasattr(self, "J"):
            return
        interval = self.t[:-1] * 0.5 * self.dt * (self.velocity[:-1] + self.velocity[1:])
        interval += self.dt**2 * (self.velocity[:-1] / 6 + self.velocity[1:] / 3)
        self.J = np.r_[0.0, np.cumsum(interval)]

    def _kernel_block(self, selected):
        self._prepare_primitives()
        distance = np.arange(self.width + 1) * self.dx
        a = np.maximum(0.0, distance - 0.5 * self.dx)
        b = np.minimum(self.cap, distance + 0.5 * self.dx)
        travel = self.travel[selected, None]
        if np.all(self.velocity == self.velocity[0]):
            # The same primitive J(D^-1(z)) = z^2/(2v), with no search or
            # derivative records when the prescribed speed is constant.
            left = np.maximum(travel - a, 0.)
            right = np.maximum(travel - b, 0.)
            np.square(left, out=left)
            np.square(right, out=right)
            left /= 2. * self.velocity[0]
            right /= 2. * self.velocity[0]
        else:
            left, _ = self._primitive(travel - a)
            right, _ = self._primitive(travel - b)
        ring = np.maximum(left - right, 0.)
        ring[:, 0] *= 2.
        ring[:, b <= a] = 0.
        kernel = np.zeros((len(ring), self.length))
        kernel[:, :self.width + 1] = ring
        if self.width:
            kernel[:, (-np.arange(1, self.width + 1)) % self.length] += ring[:, 1:]
        return rfft(kernel, axis=1)

    def _kernel(self):
        if self._spectrum is not None:
            return
        check_cancelled()
        self._prepare_primitives()
        from ._resources import ensure_capacity
        temporary_times = len(self.t) if self.differentiate_speed else min(16, len(self.t))
        spectrum_bytes = len(self.t) * (self.length // 2 + 1) * 16
        ensure_capacity(spectrum_bytes + temporary_times * (self.length * 24 + (self.width + 1) * 160),
                        "Building the mapping convolution")
        # Construct the kernel in time blocks when speed derivatives are not
        # required, avoiding permanent XT derivative tables for initiation fits.
        if not self.differentiate_speed:
            self._spectrum = np.empty((len(self.t), self.length // 2 + 1), dtype=complex)
            for first in range(0, len(self.t), 16):
                check_cancelled()
                selected = slice(first, first + 16)
                self._spectrum[selected] = self._kernel_block(selected)
            self._kernel_state = None
            return
        distance = np.arange(self.width + 1) * self.dx
        a = np.maximum(0.0, distance - 0.5 * self.dx)
        b = np.minimum(self.cap, distance + 0.5 * self.dx)
        left, lr = self._primitive(self.travel[:, None] - a)
        right, rr = self._primitive(self.travel[:, None] - b)
        factor = np.ones(len(distance))
        factor[0] = 2.0
        ring = np.maximum(left - right, 0.0) * factor
        ring[:, b <= a] = 0.0
        check_cancelled()
        kernel = np.zeros((len(self.t), self.length))
        kernel[:, : self.width + 1] = ring
        if self.width:
            kernel[:, (-np.arange(1, self.width + 1)) % self.length] += ring[:, 1:]
        self._spectrum = rfft(kernel, axis=1)
        check_cancelled()
        self._kernel_state = lr, rr, factor, ring > 0.0

    def forward(self, rate, record=False):
        check_cancelled()
        self._kernel()
        report_measured_work(0.3)
        rate = np.asarray(rate, dtype=float)
        if rate.shape != self.speed.shape or rate.strides[0] != 0 and np.any(rate != rate[:1]):
            raise ValueError("Initiation must be stationary in time and match the speed grid.")
        source = rate[0]
        if source.shape != (self.n,) or np.any(~np.isfinite(source)) or np.any(source < 0):
            raise ValueError("Stationary initiation must be a non-negative spatial vector.")
        spectrum = rfft(source, n=self.length)
        result = irfft(self._spectrum * spectrum[None, :], n=self.length, axis=1)[:, : self.n]
        check_cancelled()
        self._state = (spectrum, result > 0.0) if record else None
        report_measured_work(1.0)
        return np.maximum(result, 0.0)

    def backward(self, gradient, with_speed=True, *, with_rate=True):
        if with_speed and not self.differentiate_speed:
            raise ValueError("Speed derivatives were not requested for this operator.")
        if self._state is None:
            raise RuntimeError("Record a prediction before differentiating it.")
        check_cancelled()
        source, active = self._state
        g = rfft(np.asarray(gradient) * active, n=self.length, axis=1)
        shape = self.speed.shape
        rate_gradient = np.zeros(shape) if with_rate else np.broadcast_to(0.0, shape)
        if with_rate:
            rate_gradient[0] = irfft(np.sum(g * np.conj(self._spectrum), axis=0), n=self.length)[
                : self.n
            ]
        speed_gradient = np.zeros(shape) if with_speed else np.broadcast_to(0.0, shape)
        if not with_speed:
            return rate_gradient, speed_gradient
        kg = irfft(g * np.conj(source)[None, :], n=self.length, axis=1)
        check_cancelled()
        ring = kg[:, : self.width + 1].copy()
        if self.width:
            ring[:, 1:] += kg[:, (-np.arange(1, self.width + 1)) % self.length]
        lr, rr, factor, positive = self._kernel_state
        ring *= factor * positive
        gd = np.zeros(len(self.t))
        gj = np.zeros(len(self.t))
        gv = np.zeros(len(self.t))
        for sign, record in ((1.0, lr), (-1.0, rr)):
            index, s, a0, a1, c0, c1, active = record
            q = sign * ring * active
            sigma = self.t[index] + s
            gd += np.sum(q * sigma, axis=1)
            gd += np.bincount(index.ravel(), weights=(-q * sigma).ravel(), minlength=len(self.t))
            gj += np.bincount(index.ravel(), weights=q.ravel(), minlength=len(self.t))
            gv += np.bincount(
                index.ravel(),
                weights=(q * ((self.t[index] - sigma) * a0 + c0)).ravel(),
                minlength=len(self.t),
            )
            gv += np.bincount(
                (index + 1).ravel(),
                weights=(q * ((self.t[index] - sigma) * a1 + c1)).ravel(),
                minlength=len(self.t),
            )
        d = np.cumsum(gd[:0:-1])[::-1]
        j = np.cumsum(gj[:0:-1])[::-1]
        gv[:-1] += 0.5 * self.dt * d + j * (0.5 * self.t[:-1] * self.dt + self.dt**2 / 6)
        gv[1:] += 0.5 * self.dt * d + j * (0.5 * self.t[:-1] * self.dt + self.dt**2 / 3)
        speed_gradient[:, 0] = gv
        return rate_gradient, speed_gradient
