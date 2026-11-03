"""Stationary KJMA cones in the fork travel-time coordinate.

With y(x)=integral dx/v(x), the cone kernel is (t-|y(x)-y(xi)|)+.
Cell integrals of I and I*y give its value and its discrete adjoint in O(XT),
without characteristic stepping, temporal interpolation or a spatial CFL.
Both inputs are constant inside centered spatial cells.
"""

from __future__ import annotations

import numpy as np

from .utils import check_cancelled, report_measured_work


class StationaryHazard:
    backend = "stationary_travel_primitive"

    def __init__(self, speed, clock, resolution=1.0, periodic=False, radius=2000.0,
                 *, cache_bytes=64 * 1024**2):
        self.speed = np.array(speed, dtype=float, copy=True)
        self.t = np.asarray(clock, dtype=float)
        self.dx = float(resolution)
        self.periodic = bool(periodic)
        self.n = self.speed.size
        self.length = self.n * self.dx
        self.cap = min(float(radius), self.length / 2 if periodic else np.inf)
        if (
            self.speed.ndim != 1
            or not self.n
            or np.any(~np.isfinite(self.speed))
            or np.any(self.speed <= 0)
            or not np.isfinite(self.dx)
            or self.dx <= 0
            or np.isnan(self.cap)
            or self.cap <= 0
        ):
            raise ValueError("Stationary speed must be a positive spatial vector.")
        if (
            self.t.ndim != 1
            or not self.t.size
            or self.t[0] != 0
            or np.any(~np.isfinite(self.t))
            or np.any(np.diff(self.t) <= 0)
        ):
            raise ValueError("Time grid must start at zero and increase strictly.")
        self.slow = 1 / self.speed
        self.uniform_speed = bool(np.all(self.speed == self.speed[0]))
        self.y = np.r_[0.0, np.cumsum(self.dx * self.slow)]
        self.centers = (np.arange(self.n) + 0.5) * self.dx
        self.yc = self.y[:-1] + 0.5 * self.dx * self.slow
        self.recorded = False
        self._geometry = {}
        self._geometry_bytes = 0
        self._cache_bytes = cache_bytes

    def _inverse(self, query):
        if self.uniform_speed:
            position = query * self.speed[0]
            return position if self.periodic else np.clip(position, 0.0, self.length)
        turns = np.floor(query / self.y[-1]) if self.periodic else 0.0
        local = query - turns * self.y[-1] if self.periodic else np.clip(query, 0.0, self.y[-1])
        cell = np.clip(np.searchsorted(self.y, local, side="right") - 1, 0, self.n - 1)
        return turns * self.length + cell * self.dx + (local - self.y[cell]) * self.speed[cell]

    def _prefix(self, position, key=None):
        geometry = self._geometry.get(key) if key is not None else None
        if geometry is None:
            position = position() if callable(position) else position
            turns = np.floor(position / self.length) if self.periodic else 0.0
            local = (
                position - turns * self.length
                if self.periodic
                else np.clip(position, 0.0, self.length)
            )
            cell = np.clip((local / self.dx).astype(np.int32), 0, self.n - 1)
            h = local - cell * self.dx
            qpart = self.y[cell] * h + 0.5 * self.slow[cell] * h * h
            geometry = cell, h, turns, qpart
            cost = sum(a.nbytes for a in geometry if isinstance(a, np.ndarray)) + 256
            if key is not None and self._geometry_bytes + cost <= self._cache_bytes:
                self._geometry[key] = geometry
                self._geometry_bytes += cost
        cell, h, turns, qpart = geometry
        mass = self.mass[cell] + h * self.rate[cell]
        moment = self.moment[cell] + qpart * self.rate[cell]
        total_moment = (
            moment
            + turns * self.moment[-1]
            + 0.5 * turns * (turns - 1) * self.y[-1] * self.mass[-1]
            + turns * self.y[-1] * mass
        )
        return mass + turns * self.mass[-1], total_moment, (cell, h, turns, mass, qpart)

    def _slice(self, time):
        left = self._prefix(
            lambda: np.maximum(self.centers - self.cap, self._inverse(self.yc - time)), (-1, time)
        )
        right = self._prefix(
            lambda: np.minimum(self.centers + self.cap, self._inverse(self.yc + time)), (1, time)
        )
        center = self.center_prefix
        distance = (
            self.yc * (2 * center[0] - left[0] - right[0]) + left[1] + right[1] - 2 * center[1]
        )
        return time * (right[0] - left[0]) - distance, left, right

    def forward(self, rate, record=False):
        from ._resources import ensure_capacity
        ensure_capacity(len(self.t) * self.n * 64, "Evaluating replication timing")
        self.rate = np.array(rate, dtype=float, copy=True)
        if self.rate.shape != (self.n,) or np.any(~np.isfinite(self.rate)) or np.any(self.rate < 0):
            raise ValueError("Stationary initiation must be a non-negative spatial vector.")
        self.mass = np.r_[0.0, np.cumsum(self.rate * self.dx)]
        self.moment = np.r_[
            0.0, np.cumsum(self.rate * (self.y[:-1] * self.dx + 0.5 * self.slow * self.dx**2))
        ]
        self.center_prefix = self._prefix(self.centers)
        hazard = np.zeros((len(self.t), self.n))
        for k, time in enumerate(self.t[1:], 1):
            check_cancelled()
            hazard[k] = np.maximum(self._slice(time)[0], 0.0)
            report_measured_work(k / max(1, len(self.t) - 1))
        self.recorded = bool(record)
        return hazard

    def backward(self, gradient, with_speed=True):
        if not self.recorded:
            raise RuntimeError("Record a prediction before differentiating it.")
        gradient = np.asarray(gradient)
        if gradient.shape != (len(self.t), self.n):
            raise ValueError("Hazard gradient must match (time, position).")
        gm = np.zeros(self.n + 1)
        gq = np.zeros(self.n + 1)
        gi = np.zeros(self.n)
        gy = np.zeros(self.n + 1)
        gw = np.zeros(self.n)
        receiver = np.zeros(self.n)

        def accumulate(mass_gradient, moment_gradient, record):
            cell, h, turns, mass, qpart = record
            lm = mass_gradient + turns * self.y[-1] * moment_gradient
            gm[:] += np.bincount(cell, weights=lm, minlength=self.n + 1)
            gq[:] += np.bincount(cell, weights=moment_gradient, minlength=self.n + 1)
            gm[-1] += np.sum(
                turns * mass_gradient + 0.5 * turns * (turns - 1) * self.y[-1] * moment_gradient
            )
            gq[-1] += np.sum(turns * moment_gradient)
            gi[:] += np.bincount(cell, weights=h * lm + qpart * moment_gradient, minlength=self.n)
            if with_speed:
                gy[:-1] += np.bincount(
                    cell, weights=self.rate[cell] * h * moment_gradient, minlength=self.n
                )
                gw[:] += np.bincount(
                    cell, weights=0.5 * self.rate[cell] * h * h * moment_gradient, minlength=self.n
                )
                gy[-1] += np.sum(
                    (0.5 * turns * (turns - 1) * self.mass[-1] + turns * mass) * moment_gradient
                )

        center_mass = np.zeros(self.n)
        center_moment = np.zeros(self.n)
        for k, time in enumerate(self.t[1:], 1):
            check_cancelled()
            hazard, left, right = self._slice(time)
            g = gradient[k] * (hazard > 0)
            accumulate(g * (-time + self.yc), -g, left[2])
            accumulate(g * (time + self.yc), -g, right[2])
            center_mass -= 2 * g * self.yc
            center_moment += 2 * g
            if with_speed:
                receiver += g * (left[0] + right[0] - 2 * self.center_prefix[0])
        accumulate(center_mass, center_moment, self.center_prefix[2])
        qm = np.cumsum(gq[:0:-1])[::-1]
        gi += self.dx * np.cumsum(gm[:0:-1])[::-1]
        gi += qm * (self.y[:-1] * self.dx + 0.5 * self.slow * self.dx**2)
        if with_speed:
            gy[:-1] += qm * self.rate * self.dx + receiver
            gw += 0.5 * qm * self.rate * self.dx**2 + 0.5 * self.dx * receiver
            gw += self.dx * np.cumsum(gy[:0:-1])[::-1]
            return gi, -gw / self.speed**2
        return gi, None
