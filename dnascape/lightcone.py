"""Bounded-memory replication cones and their exact discrete adjoints.

The Poisson survival law is ``S(x,t) = exp(-integral_cone I)``. On a circle
the union of the two backwards cones covers the genome at most once. Sources
are piecewise constant over spatial cells; characteristics use midpoint
integration and the source integral uses the temporal midpoint rule. The
adjoint differentiates this same discretization, without one solve per bin.
"""

import numpy as np
from scipy.signal import fftconvolve

from .utils import check_cancelled


class ReplicationCone:
    """Linear operator from interval-integrated cell rates to cone volume."""

    def __init__(self, speed_tx, time_grid, resolution=1.0, periodic=False, cache_bytes=64 * 1024**2):
        self.speed = np.asarray(speed_tx, dtype=float)
        self.t = np.asarray(time_grid, dtype=float)
        self.dx = float(resolution)
        self.periodic = bool(periodic)
        if self.speed.ndim != 2 or self.speed.shape[0] != self.t.size:
            raise ValueError("speed_tx must have one row per time point.")
        if self.t.ndim != 1 or self.t.size < 1 or np.any(~np.isfinite(self.t)):
            raise ValueError("time_grid must be a finite non-empty vector.")
        if np.any(np.diff(self.t) <= 0) or self.t[0] != 0:
            raise ValueError("Replication cones require an increasing time grid starting at zero.")
        if not np.isfinite(self.dx) or self.dx <= 0:
            raise ValueError("resolution must be finite and positive.")
        if np.any(~np.isfinite(self.speed)) or np.any(self.speed <= 0):
            raise ValueError("fork_speed must contain finite positive values.")
        self.n = self.speed.shape[1]
        self.x = np.arange(self.n, dtype=float) * self.dx
        self.length = self.n * self.dx
        self.uniform_space = bool(np.all(self.speed == self.speed[:, :1]))
        self.static_speed = bool(np.all(self.speed == self.speed[:1]))
        if self.uniform_space:
            self.distance = np.concatenate((
                [0.0], np.cumsum(0.5 * (self.speed[:-1, 0] + self.speed[1:, 0]) * np.diff(self.t))
            ))
        elif self.static_speed:
            inv_v = 1.0 / self.speed[0]
            self.travel = np.concatenate((
                [0.0], np.cumsum(0.5 * (inv_v[:-1] + inv_v[1:]) * self.dx)
            ))
            self.travel_length = self.travel[-1] + 0.5 * self.dx * (inv_v[-1] + inv_v[0])
        # Cached geometries use O(X T^2) space only when explicitly affordable.
        # Larger domains stream one output-time slice in O(X T) memory.
        projected = self.t.size * (self.t.size - 1) * self.n * 4
        self._geometry = {} if projected <= int(cache_bytes) else None

    def _positions_from_travel(self, travel):
        if self.periodic:
            turns = np.floor(travel / self.travel_length)
            phase = travel - turns * self.travel_length
            return np.interp(
                phase, np.append(self.travel, self.travel_length),
                np.append(self.x, self.length),
            ) + turns * self.length
        return np.interp(
            travel,
            np.concatenate(([-0.5 * self.dx / self.speed[0, 0]], self.travel,
                            [self.travel[-1] + 0.5 * self.dx / self.speed[0, -1]])),
            np.concatenate(([-0.5 * self.dx], self.x, [self.length - 0.5 * self.dx])),
        )

    def _velocity(self, positions, interval, weight):
        values = (1.0 - weight) * self.speed[interval] + weight * self.speed[interval + 1]
        if self.periodic:
            return np.interp(
                np.mod(positions, self.length), np.append(self.x, self.length),
                np.append(values, values[0]),
            )
        return np.interp(positions, self.x, values)

    def _advance(self, positions, interval, upper_weight, lower_weight, direction):
        duration = (upper_weight - lower_weight) * (self.t[interval + 1] - self.t[interval])
        max_speed = np.max(self.speed[interval:interval + 2])
        count = max(1, int(np.ceil(duration * max_speed / (2.0 * self.dx))))
        result = positions.copy()
        for index in range(count):
            hi = upper_weight + (lower_weight - upper_weight) * index / count
            lo = upper_weight + (lower_weight - upper_weight) * (index + 1) / count
            dt = duration / count
            first = self._velocity(result, interval, hi)
            mid = result + direction * 0.5 * dt * first
            result += direction * dt * self._velocity(mid, interval, 0.5 * (hi + lo))
            if not self.periodic:
                result = np.clip(result, -0.5 * self.dx, self.length - 0.5 * self.dx)
        return result

    def geometry(self, output_index):
        if self._geometry is not None and output_index in self._geometry:
            return self._geometry[output_index]
        left = np.empty((output_index, self.n), dtype=float)
        right = np.empty_like(left)
        if self.uniform_space:
            # Integral of linearly interpolated speed at each interval midpoint.
            dt = np.diff(self.t[:output_index + 1])
            midpoint_distance = self.distance[:output_index] + dt * (
                3.0 * self.speed[:output_index, 0] + self.speed[1:output_index + 1, 0]
            ) / 8.0
            radius = self.distance[output_index] - midpoint_distance
            left[:] = self.x[None, :] - radius[:, None]
            right[:] = self.x[None, :] + radius[:, None]
        elif self.static_speed:
            age = self.t[output_index] - 0.5 * (
                self.t[:output_index] + self.t[1:output_index + 1]
            )
            left[:] = self._positions_from_travel(self.travel[None, :] - age[:, None])
            right[:] = self._positions_from_travel(self.travel[None, :] + age[:, None])
        else:
            position_left = self.x.copy()
            position_right = self.x.copy()
            for interval in range(output_index - 1, -1, -1):
                position_left = self._advance(position_left, interval, 1.0, 0.5, -1.0)
                position_right = self._advance(position_right, interval, 1.0, 0.5, 1.0)
                left[interval] = position_left
                right[interval] = position_right
                position_left = self._advance(position_left, interval, 0.5, 0.0, -1.0)
                position_right = self._advance(position_right, interval, 0.5, 0.0, 1.0)
        # Prefix coordinates are measured from the left edge of cell zero.
        left = (left + 0.5 * self.dx) / self.dx
        right = (right + 0.5 * self.dx) / self.dx
        if self.periodic:
            covered = right - left >= self.n
            left[covered] = 0.0
            right[covered] = float(self.n)
        else:
            np.clip(left, 0, self.n, out=left)
            np.clip(right, 0, self.n, out=right)
        if self._geometry is not None:
            left = left.astype(np.float32)
            right = right.astype(np.float32)
            self._geometry[output_index] = (left, right)
        return left, right

    def _coordinates(self, position):
        if self.periodic:
            turns = np.floor(position / self.n)
            local = position - turns * self.n
        else:
            turns = np.zeros_like(position)
            local = position
        lower = np.minimum(np.floor(local).astype(np.intp), self.n - 1)
        fraction = local - lower
        return lower, fraction, turns

    def apply(self, interval_rates):
        rates = np.asarray(interval_rates, dtype=float)
        if rates.shape != (self.t.size - 1, self.n):
            raise ValueError("Interval rates must have shape (T - 1, X).")
        prefix = np.pad(np.cumsum(rates * self.dx, axis=1), ((0, 0), (1, 0)))
        volume = np.zeros((self.t.size, self.n), dtype=float)
        for output_index in range(1, self.t.size):
            check_cancelled()
            left, right = self.geometry(output_index)
            current = prefix[:output_index]
            mass = np.zeros_like(left, dtype=float)
            for position, sign in ((right, 1.0), (left, -1.0)):
                lower, fraction, turns = self._coordinates(position)
                low = np.take_along_axis(current, lower, axis=1)
                high = np.take_along_axis(current, lower + 1, axis=1)
                mass += sign * (low + fraction * (high - low) + turns * current[:, -1:])
            volume[output_index] = np.sum(mass, axis=0)
        return volume

    def adjoint(self, weights):
        """Transpose of ``apply`` under the Euclidean inner product."""

        weights = np.asarray(weights, dtype=float)
        if weights.shape != self.speed.shape:
            raise ValueError("Adjoint weights must have shape (T, X).")
        prefix_gradient = np.zeros((self.t.size - 1, self.n + 1))
        for output_index in range(1, self.t.size):
            check_cancelled()
            left, right = self.geometry(output_index)
            offsets = np.arange(output_index, dtype=np.intp)[:, None] * (self.n + 1)
            size = output_index * (self.n + 1)
            for position, sign in ((right, 1.0), (left, -1.0)):
                lower, fraction, turns = self._coordinates(position)
                q = np.broadcast_to(sign * weights[output_index], lower.shape)
                contributions = np.bincount(
                    (lower + offsets).ravel(), (q * (1.0 - fraction)).ravel(), minlength=size,
                ) + np.bincount(
                    (lower + 1 + offsets).ravel(), (q * fraction).ravel(), minlength=size,
                )
                prefix_gradient[:output_index] += contributions.reshape(output_index, self.n + 1)
                prefix_gradient[:output_index, -1] += np.sum(q * turns, axis=1)
        return self.dx * np.cumsum(prefix_gradient[:, :0:-1], axis=1)[:, ::-1]


class WeibullCone:
    """Reusable survival/gradient operator for a fixed speed field and grid."""

    def __init__(self, speed_tx, time_grid, delay, resolution=1.0, periodic=False):
        self.speed = np.asarray(speed_tx, dtype=float)
        self.t = np.asarray(time_grid, dtype=float)
        self.delay = np.asarray(delay, dtype=float).reshape(-1)
        self._output_slice = slice(None)
        if self.t.size and self.t[0] > 0:
            self.speed = np.vstack((self.speed[:1], self.speed))
            self.t = np.insert(self.t, 0, 0.0)
            self._output_slice = slice(1, None)
        self.dx = float(resolution)
        self.periodic = bool(periodic)
        self.n = self.delay.size
        self.cone = ReplicationCone(self.speed, self.t, self.dx, periodic)
        self.convolution = self.cone.uniform_space and bool(np.all(self.delay == self.delay[0]))
        self.backend = "periodic_fft" if self.convolution and periodic else (
            "temporal_fft" if self.convolution else "characteristic_cones"
        )
        self._kernel_cache = {}

    def _convolve(self, values, kernel):
        if self.periodic:
            return np.fft.irfft(np.fft.rfft(values) * np.fft.rfft(kernel), n=self.n)
        return fftconvolve(values, np.concatenate((kernel[:0:-1], kernel)), mode="same")

    def _kernels(self, shape):
        if shape not in self._kernel_cache:
            distances = np.arange(self.n, dtype=float) * self.dx
            if self.periodic:
                distances = np.minimum(distances, self.n * self.dx - distances)
            available = self.cone.distance[:, None] - distances[None, :]
            latest = np.interp(np.maximum(available, 0.0), self.cone.distance, self.t)
            ages = np.maximum(latest - self.delay[0], 0.0)
            ages[available < 0] = 0.0
            kernel = ages**shape
            derivative = np.zeros_like(kernel)
            positive = ages > 0
            derivative[positive] = kernel[positive] * np.log(ages[positive])
            # Shape optimization should not accumulate one grid per proposal.
            self._kernel_cache.clear()
            self._kernel_cache[shape] = (kernel, derivative)
        return self._kernel_cache[shape]

    def volumes(self, alpha, shape, shape_gradient=False):
        if self.convolution:
            kernel, derivative = self._kernels(shape)
            volume = np.array([self.dx * self._convolve(alpha, row) for row in kernel])
            derivative_volume = (
                np.array([self.dx * self._convolve(alpha, row) for row in derivative])
                if shape_gradient else None
            )
        else:
            age = np.maximum(self.t[:, None] - self.delay[None, :], 0.0)
            cumulative = age**shape
            integrated_hazard = np.diff(cumulative, axis=0)
            volume = self.cone.apply(integrated_hazard * alpha)
            derivative_volume = None
            if shape_gradient:
                derivative = np.zeros_like(cumulative)
                positive = age > 0
                derivative[positive] = cumulative[positive] * np.log(age[positive])
                derivative_volume = self.cone.apply(np.diff(derivative, axis=0) * alpha)
        return np.maximum(volume[self._output_slice], 0.0), (
            derivative_volume[self._output_slice] if derivative_volume is not None else None
        )

    def alpha_adjoint(self, weights, shape):
        if self._output_slice.start == 1:
            weights = np.vstack((np.zeros((1, self.n)), weights))
        if self.convolution:
            kernel, _ = self._kernels(shape)
            return sum(self.dx * self._convolve(row, k) for row, k in zip(weights, kernel))
        age = np.maximum(self.t[:, None] - self.delay[None, :], 0.0)
        integrated_hazard = np.diff(age**shape, axis=0)
        return np.sum(self.cone.adjoint(weights) * integrated_hazard, axis=0)
