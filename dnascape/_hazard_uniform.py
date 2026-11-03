"""Primitive-only cones when both kinetic fields are uniform in space.

The travel clock D=int(v), and primitives A=int(I), B=int(I*D), reduce a
one-sided cone to d*A(sigma)+D(t)*(A(t)-A(sigma))-B(t)+B(sigma).
The crossing-time derivative cancels exactly. Linear input samples, including
their within-interval quadratic travel clocks, are integrated analytically.
"""

import numpy as np

from .utils import check_cancelled, report_measured_work


class UniformKineticsHazard:
    backend = "uniform_kinetic_primitives"

    def __init__(
        self, speed_tx, time_grid, resolution=1.0, periodic=False, radius=2000.0, length=None,
        *, compress=False,
    ):
        self.speed = np.asarray(speed_tx)
        self.t = np.asarray(time_grid, dtype=float)
        self.dt = np.diff(self.t)
        self.n = self.speed.shape[1]
        self.dx = float(resolution)
        self.radius = float(radius)
        if periodic:
            self.distances = np.array(
                [min(self.radius, (self.n * self.dx if length is None else length) / 2.0)]
            )
            self.left = self.right = np.zeros(self.n, dtype=int)
        else:
            # All boundaries beyond the furthest possible travel distance
            # have the same unrestricted cone. Compress those histories too.
            reach = min(
                self.radius, float(np.sum(0.5 * (self.speed[:-1, 0] + self.speed[1:, 0]) * self.dt))
            )
            left = np.minimum(reach, (np.arange(self.n) + 0.5) * self.dx)
            self.distances, indices = np.unique(np.r_[left, left[::-1]], return_inverse=True)
            self.left, self.right = indices[: self.n], indices[self.n :]
        self.groups = None
        if compress:
            # Uniform inputs give the same history at reflected positions and
            # throughout the interior beyond the cone's reach. Evaluate each
            # distinct history once; the fitter sums its observation weights.
            self.groups = np.minimum(self.left, self.right)
            count = int(self.groups.max()) + 1
            self.left, self.right = self.left[:count], self.right[:count]
        self._state = None

    def _moments(self, index, s):
        dt = self.dt[index]
        a1 = s * s / (2.0 * dt)
        a0 = s - a1
        c1 = s**3 / (3.0 * dt)
        c0 = 0.5 * s * s - c1
        e1 = s**4 / (8.0 * dt)
        e0 = s**3 / 6.0 - e1
        return a0, a1, c0, c1, e0, e1

    def _partial(self, index, s, rate, velocity, travel):
        a0, a1, c0, c1, e0, e1 = self._moments(index, s)
        il, ir = rate[index], rate[index + 1]
        vl, vr = velocity[index], velocity[index + 1]
        a = il * a0 + ir * a1
        b = (
            travel[index] * a
            + vl * (il * c0 + ir * c1)
            + (vr - vl) / self.dt[index] * (il * e0 + ir * e1)
        )
        return a, b

    def forward(self, rate_tx, record=False):
        check_cancelled()
        if len(self.t) == 1:
            if record:
                self._state = ("single",)
            return np.zeros((1, len(self.left)), dtype=float)
        rate = np.asarray(rate_tx)[:, 0]
        velocity = self.speed[:, 0]
        travel = np.r_[0.0, np.cumsum(0.5 * (velocity[:-1] + velocity[1:]) * self.dt)]
        indices = np.arange(len(self.dt))
        a, b = self._partial(indices, self.dt, rate, velocity, travel)
        A, B = np.r_[0.0, np.cumsum(a)], np.r_[0.0, np.cumsum(b)]
        report_measured_work(0.25)
        check_cancelled()
        z = np.maximum(travel[:, None] - self.distances, 0.0)
        index = np.clip(np.searchsorted(travel, z, side="right") - 1, 0, len(self.dt) - 1)
        offset = z - travel[index]
        slope = (velocity[index + 1] - velocity[index]) / self.dt[index]
        root = np.sqrt(np.maximum(velocity[index] ** 2 + 2.0 * slope * offset, 0.0))
        s = np.divide(
            2.0 * offset, velocity[index] + root, out=np.zeros_like(offset), where=offset != 0.0
        )
        s = np.clip(s, 0.0, self.dt[index])
        pa, pb = self._partial(index, s, rate, velocity, travel)
        As, Bs = A[index] + pa, B[index] + pb
        report_measured_work(0.75)
        check_cancelled()
        G = self.distances * As + travel[:, None] * (A[:, None] - As) - B[:, None] + Bs
        result = G[:, self.left] + G[:, self.right]
        result[0] = 0.0
        if record:
            self._state = rate, velocity, travel, A, B, index, s, As
        report_measured_work(1.0)
        return np.maximum(result, 0.0)

    def backward(self, dH, with_speed=True):
        if self._state is None:
            raise RuntimeError("Record a prediction before differentiating it.")
        check_cancelled()
        if len(self._state) == 1:
            shape = (1, 1) if self.groups is not None else self.speed.shape
            return np.zeros(shape), np.zeros(shape)
        rate, velocity, travel, A, B, index, s, As = self._state
        g = np.asarray(dH)
        q = np.stack(
            [
                np.bincount(self.left, row, minlength=len(self.distances))
                + np.bincount(self.right, row, minlength=len(self.distances))
                for row in g
            ]
        )
        q[0] = 0.0
        ga = np.sum(q * travel[:, None], axis=1)
        gb = -q.sum(axis=1)
        gd = np.sum(q * (A[:, None] - As), axis=1)
        gia, gib = q * (self.distances - travel[:, None]), q
        ga += np.bincount(index.ravel(), gia.ravel(), minlength=len(self.t))
        gb += np.bincount(index.ravel(), gib.ravel(), minlength=len(self.t))
        gi, gv = np.zeros(len(self.t)), np.zeros(len(self.t))

        def partial_pullback(indices, duration, wa, wb):
            a0, a1, c0, c1, e0, e1 = self._moments(indices, duration)
            il, ir = rate[indices], rate[indices + 1]
            vl, vr = velocity[indices], velocity[indices + 1]
            slope = (vr - vl) / self.dt[indices]
            gi[:] += np.bincount(
                indices.ravel(),
                (wa * a0 + wb * (travel[indices] * a0 + vl * c0 + slope * e0)).ravel(),
                minlength=len(self.t),
            )
            gi[:] += np.bincount(
                (indices + 1).ravel(),
                (wa * a1 + wb * (travel[indices] * a1 + vl * c1 + slope * e1)).ravel(),
                minlength=len(self.t),
            )
            if with_speed:
                gd[:] += np.bincount(
                    indices.ravel(), (wb * (il * a0 + ir * a1)).ravel(), minlength=len(self.t)
                )
                extra = (il * e0 + ir * e1) / self.dt[indices]
                gv[:] += np.bincount(
                    indices.ravel(), (wb * (il * c0 + ir * c1 - extra)).ravel(), minlength=len(self.t)
                )
                gv[:] += np.bincount(
                    (indices + 1).ravel(), (wb * extra).ravel(), minlength=len(self.t)
                )

        partial_pullback(index, s, gia, gib)
        interval_a = np.cumsum(ga[:0:-1])[::-1]
        interval_b = np.cumsum(gb[:0:-1])[::-1]
        partial_pullback(np.arange(len(self.dt)), self.dt, interval_a, interval_b)
        if with_speed:
            interval_d = 0.5 * self.dt * np.cumsum(gd[:0:-1])[::-1]
            gv[:-1] += interval_d
            gv[1:] += interval_d
        shape = (len(self.t), 1) if self.groups is not None else self.speed.shape
        rate_gradient = np.zeros(shape)
        rate_gradient[:, 0] = gi
        speed_gradient = np.zeros(shape) if with_speed else np.broadcast_to(0.0, shape)
        if with_speed:
            speed_gradient[:, 0] = gv
        return rate_gradient, speed_gradient
