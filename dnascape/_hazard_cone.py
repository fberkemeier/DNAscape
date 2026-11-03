"""Finite-influence replication cones and their discrete reverse derivatives.

Spatial samples own cells of width ``resolution`` centred on each sample.
Initiation is constant within each cell; speed is interpolated linearly between
cell centres, and characteristics use explicit midpoint steps. Temporal source
integrals use the trapezoid rule. The transpose differentiates precisely these
operations, including interpolation of the speed and moving cone boundaries.

A finite influence radius explicitly excludes more distant potential origins;
it is a model approximation when the physical cone extends beyond that radius.
On a circle every source cell is counted at most once. A radius covering half
the circumference already includes the entire circular genomic neighbourhood.
"""

from __future__ import annotations

import numpy as np

from .utils import check_cancelled, report_measured_work


class ConeHazard:
    """Map time-by-space initiation samples to cumulative replication hazard.

    Only the latest recorded source is retained. Speed derivatives recompute
    the characteristic tape for one output time at a time. Fixed-speed source
    geometry is cached only within a 64 MB budget and otherwise streamed,
    avoiding an unbounded ``O(X T**2)`` geometry or derivative cache.
    """

    backend = "characteristic_cones"
    _CACHE_BYTES = 64 * 1024**2

    def __init__(self, speed_tx, time_grid, resolution=1.0, periodic=False, radius=2000.0):
        self.speed = np.array(speed_tx, dtype=float, copy=True)
        self.t = np.array(time_grid, dtype=float, copy=True)
        self.dx = float(resolution)
        self.periodic = bool(periodic)
        self.radius = np.inf if radius is None else float(radius)
        if self.t.ndim != 1 or not self.t.size or np.any(~np.isfinite(self.t)):
            raise ValueError("time_grid must be a finite non-empty vector.")
        if self.t[0] != 0 or np.any(np.diff(self.t) <= 0):
            raise ValueError("Replication cones require increasing times starting at zero.")
        if self.speed.ndim != 2 or self.speed.shape[0] != self.t.size or not self.speed.shape[1]:
            raise ValueError("speed_tx must have shape (number of times, number of positions).")
        if np.any(~np.isfinite(self.speed)) or np.any(self.speed <= 0):
            raise ValueError("Fork speed must contain finite positive values.")
        if not np.isfinite(self.dx) or self.dx <= 0:
            raise ValueError("resolution must be finite and positive.")
        if np.isnan(self.radius) or self.radius <= 0:
            raise ValueError("radius must be positive.")
        self.n = self.speed.shape[1]
        self.x = np.arange(self.n, dtype=float) * self.dx
        self.length = self.n * self.dx
        self.dt = np.diff(self.t)
        self.uniform_space = bool(np.all(self.speed == self.speed[:, :1]))
        if self.uniform_space:
            self._distance = np.r_[0., np.cumsum(.5 * (self.speed[:-1, 0] + self.speed[1:, 0]) * self.dt)]
            self._midpoint_distance = self._distance[:-1] + self.dt * (3. * self.speed[:-1, 0] + self.speed[1:, 0]) / 8.
        self._radius = self.radius
        if self.periodic and self.radius >= 0.5 * self.length:
            self._radius = np.inf
        self._mask_lower = self.x - self._radius
        self._mask_upper = self.x + self._radius
        if self.periodic:
            # Follow the complete circular cone, even when its fastest path
            # reaches an in-mask source by travelling the longer way around.
            self._lower = np.full(self.n, -np.inf)
            self._upper = np.full(self.n, np.inf)
        else:
            self._lower = np.maximum(self._mask_lower, -0.5 * self.dx)
            self._upper = np.minimum(self._mask_upper, self.length - 0.5 * self.dx)
        self._interval_speed = np.maximum(np.max(self.speed[:-1], axis=1), np.max(self.speed[1:], axis=1))
        self._recorded = None
        geometry_bytes = self.t.size * (self.t.size - 1) // 2 * (17 * self.n + 512)
        self._geometry_cache = {} if geometry_bytes <= self._CACHE_BYTES else None
        self._geometry_bytes = geometry_bytes if self._geometry_cache is not None else 0
        self._source_operator = None
        self._source_attempted = False
        self._forward_calls = 0
        self.diagnostics = {
            "backend": self.backend,
            "influence_radius": self.radius,
            "finite_influence_approximation": bool(np.isfinite(self._radius)),
            "spatial_discretization": "centred piecewise-constant cells",
            "cached_geometry": self._geometry_cache is not None,
            "source_operator_bytes": 0,
        }

    def _velocity(self, positions, interval, weight, *, record=False):
        coordinate = positions / self.dx
        if self.periodic:
            local = np.mod(coordinate, self.n)
            # A tiny negative coordinate can round to exactly n after mod.
            # Keep it on the final interpolation segment, whose upper node is
            # node zero, rather than attempting to index speed[n].
            lower = np.minimum(np.floor(local).astype(np.intp), self.n - 1)
            fraction = local - lower
            upper = (lower + 1) % self.n
            active = 1.0
        else:
            local = np.clip(coordinate, 0, self.n - 1)
            lower = np.floor(local).astype(np.intp)
            upper = np.minimum(lower + 1, self.n - 1)
            fraction = local - lower
            active = (coordinate >= 0) & (coordinate <= self.n - 1)
        row = (1.0 - weight) * self.speed[interval] + weight * self.speed[interval + 1]
        result = (1.0 - fraction) * row[lower] + fraction * row[upper]
        tape = None
        if record:
            tape = (lower, upper, fraction, weight, active * (row[upper] - row[lower]) / self.dx)
        return result, tape

    def _velocity_backward(self, gradient, tape, interval, speed_gradient):
        lower, upper, fraction, weight, slope = tape
        spatial = np.bincount(lower, gradient * (1.0 - fraction), minlength=self.n)
        spatial += np.bincount(upper, gradient * fraction, minlength=self.n)
        speed_gradient[interval] += (1.0 - weight) * spatial
        speed_gradient[interval + 1] += weight * spatial
        return gradient * slope

    def _advance(self, positions, interval, hi, lo, direction, *, record=False):
        duration = (hi - lo) * self.dt[interval]
        count = max(1, int(np.ceil(duration * self._interval_speed[interval] / (2.0 * self.dx))))
        step = duration / count
        result = positions
        tape = [] if record else None
        for index in range(count):
            check_cancelled()
            upper_weight = hi + (lo - hi) * index / count
            lower_weight = hi + (lo - hi) * (index + 1) / count
            previous = result
            first, _ = self._velocity(previous, interval, upper_weight)
            middle = result + direction * 0.5 * step * first
            second, _ = self._velocity(middle, interval, 0.5 * (upper_weight + lower_weight))
            raw = result + direction * step * second
            result = np.clip(raw, self._lower, self._upper)
            if record:
                # Retain one position vector per step, rather than the eight
                # interpolation-index/weight vectors used by its derivative.
                tape.append((previous, direction * step, upper_weight, lower_weight))
        return result, tape

    def _advance_backward(self, gradient, tape, interval, speed_gradient):
        for positions, step, upper_weight, lower_weight in reversed(tape):
            first_value, first = self._velocity(positions, interval, upper_weight, record=True)
            middle = positions + 0.5 * step * first_value
            second_value, second = self._velocity(middle, interval, 0.5 * (upper_weight + lower_weight), record=True)
            raw = positions + step * second_value
            active = (raw > self._lower) & (raw < self._upper)
            raw_gradient = gradient * active
            middle_gradient = self._velocity_backward(step * raw_gradient, second, interval, speed_gradient)
            gradient = raw_gradient + middle_gradient
            gradient += self._velocity_backward(0.5 * step * middle_gradient, first, interval, speed_gradient)
        return gradient

    def _prefix_coordinates(self, positions):
        coordinate = (positions + 0.5 * self.dx) / self.dx
        if self.periodic:
            turns = np.floor(coordinate / self.n)
            local = coordinate - turns * self.n
        else:
            turns = np.zeros_like(coordinate)
            local = np.clip(coordinate, 0.0, self.n)
        lower = np.minimum(np.floor(local).astype(np.intp), self.n - 1)
        return lower, local - lower, turns

    def _interval_integral(self, prefix, left, right):
        result = np.zeros(self.n)
        for positions, sign in ((right, 1.0), (left, -1.0)):
            lower, fraction, turns = self._prefix_coordinates(positions)
            result += sign * (prefix[lower] + fraction * (prefix[lower + 1] - prefix[lower]) + turns * prefix[-1])
        return result

    def _masked_segments(self, left, right, covered):
        for turn in (-1, 0, 1):
            mask_left = self._mask_lower + turn * self.length
            mask_right = self._mask_upper + turn * self.length
            low = np.maximum(left, mask_left)
            high = np.minimum(right, mask_right)
            active = (high > low) & ~covered
            if np.any(active):
                yield low, high, active, left > mask_left, right < mask_right

    def _integral(self, prefix, left, right):
        covered = (right - left >= self.length) if self.periodic else np.zeros(self.n, dtype=bool)
        if self.periodic and np.isfinite(self._radius):
            result = np.zeros(self.n)
            for low, high, active, _, _ in self._masked_segments(left, right, covered):
                result += np.where(active, self._interval_integral(prefix, low, high), 0.)
            if np.any(covered):
                result += np.where(covered, self._interval_integral(prefix, self._mask_lower, self._mask_upper), 0.)
        else:
            result = self._interval_integral(prefix, left, right)
            result[covered] = prefix[-1]
        return result, covered

    def _interval_backward(self, prefix, left, right, weights, prefix_gradient):
        boundary_gradients = []
        for positions, sign in ((left, -1.0), (right, 1.0)):
            lower, fraction, turns = self._prefix_coordinates(positions)
            weight = sign * weights
            prefix_gradient += np.bincount(lower, weight * (1.0 - fraction), minlength=self.n + 1)
            prefix_gradient += np.bincount(lower + 1, weight * fraction, minlength=self.n + 1)
            prefix_gradient[-1] += np.dot(weight, turns)
            boundary_gradients.append(weight * (prefix[lower + 1] - prefix[lower]) / self.dx)
        return boundary_gradients

    def _integral_backward(self, prefix, left, right, weights, covered):
        prefix_gradient = np.zeros(self.n + 1)
        if self.periodic and np.isfinite(self._radius):
            dleft = np.zeros(self.n)
            dright = np.zeros(self.n)
            for low, high, active, left_active, right_active in self._masked_segments(left, right, covered):
                low_gradient, high_gradient = self._interval_backward(prefix, low, high, weights * active, prefix_gradient)
                dleft += low_gradient * left_active
                dright += high_gradient * right_active
            if np.any(covered):
                self._interval_backward(prefix, self._mask_lower, self._mask_upper, weights * covered, prefix_gradient)
        else:
            dleft, dright = self._interval_backward(prefix, left, right, np.where(covered, 0., weights), prefix_gradient)
            prefix_gradient[-1] += np.sum(weights[covered])
        source_gradient = self.dx * np.cumsum(prefix_gradient[:0:-1])[::-1]
        return source_gradient, dleft, dright

    def _slice(self, index, prefix, *, record=False):
        if not record:
            geometry, full_tail = self._geometry(index)
            hazard = np.zeros(self.n)
            if full_tail >= 0:
                hazard += np.sum(prefix[:full_tail + 1, -1])
            for interval, left, right, _ in geometry:
                check_cancelled()
                hazard += self._integral(prefix[interval], left, right)[0]
            return hazard, None, full_tail
        left = self.x.copy()
        right = self.x.copy()
        hazard = np.zeros(self.n)
        tape = [] if record else None
        full_tail = -1
        for interval in range(index - 1, -1, -1):
            check_cancelled()
            left_mid, left_hi = self._advance(left, interval, 1.0, 0.5, -1.0, record=record)
            right_mid, right_hi = self._advance(right, interval, 1.0, 0.5, 1.0, record=record)
            mass, covered = self._integral(prefix[interval], left_mid, right_mid)
            if np.all(covered) and not np.isfinite(self._radius):
                hazard += np.sum(prefix[:interval + 1, -1])
                full_tail = interval
                break
            hazard += mass
            left, left_lo = self._advance(left_mid, interval, 0.5, 0.0, -1.0, record=record)
            right, right_lo = self._advance(right_mid, interval, 0.5, 0.0, 1.0, record=record)
            if record:
                tape.append((interval, left_mid, right_mid, covered, left_hi, right_hi, left_lo, right_lo))
        return hazard, tape, full_tail

    def _geometry(self, index):
        if self._geometry_cache is not None and index in self._geometry_cache:
            return self._geometry_cache[index]
        geometry = []
        full_tail = -1
        left = self.x.copy()
        right = self.x.copy()
        for interval in range(index - 1, -1, -1):
            check_cancelled()
            if self.uniform_space:
                distance = self._distance[index] - self._midpoint_distance[interval]
                left_mid = np.clip(self.x - distance, self._lower, self._upper)
                right_mid = np.clip(self.x + distance, self._lower, self._upper)
            else:
                left_mid, _ = self._advance(left, interval, 1., .5, -1.)
                right_mid, _ = self._advance(right, interval, 1., .5, 1.)
            covered = ((right_mid - left_mid >= self.length) if self.periodic
                       else np.zeros(self.n, dtype=bool))
            if np.all(covered) and not np.isfinite(self._radius):
                full_tail = interval
                break
            geometry.append((interval, left_mid, right_mid, covered))
            if not self.uniform_space:
                left, _ = self._advance(left_mid, interval, .5, 0., -1.)
                right, _ = self._advance(right_mid, interval, .5, 0., 1.)
        result = geometry, full_tail
        if self._geometry_cache is not None:
            self._geometry_cache[index] = result
        return result

    def _compile_source_operator(self):
        """Cache the fixed-speed source-prefix map within the memory budget.

        COO construction is grouped by receiver time, keeping the number of
        temporary arrays small. The conservative peak-memory allowance also
        covers concatenation and COO-to-CSR conversion, not only final CSR
        storage. Large problems continue using streamed cone integrals.
        """
        self._source_attempted = True
        count = self.t.size * (self.t.size - 1) // 2
        if not count or self._geometry_cache is None:
            return
        allowance = self._CACHE_BYTES - self._geometry_bytes
        index_type = np.int32 if max(self.t.size * self.n, (self.t.size - 1) * (self.n + 1)) < 2**31 else np.int64
        entry_peak = 40 if index_type is np.int32 else 56
        maximum_entries = max(0, allowance // entry_peak)
        if 4 * count * self.n > maximum_entries:
            return
        from scipy.sparse import coo_matrix

        data_blocks, row_blocks, column_blocks = [], [], []
        entries = 0
        for index in range(1, self.t.size):
            check_cancelled()
            data, rows, columns = [], [], []
            geometry, full_tail = self._geometry(index)

            def add_column(interval, receivers, column, coefficient):
                nonlocal entries
                active = coefficient != 0.
                if not np.any(active):
                    return
                receivers = receivers[active]
                coefficient = coefficient[active]
                column = np.broadcast_to(column, active.shape)[active]
                entries += coefficient.size
                data.append(np.asarray(coefficient, dtype=float))
                rows.append(np.asarray(index * self.n + receivers, dtype=index_type))
                columns.append(np.asarray(interval * (self.n + 1) + column, dtype=index_type))

            def add_boundary(interval, positions, sign, active):
                receivers = np.flatnonzero(active)
                if not receivers.size:
                    return
                lower, fraction, turns = self._prefix_coordinates(positions[receivers])
                add_column(interval, receivers, lower, sign * (1. - fraction))
                add_column(interval, receivers, lower + 1, sign * fraction)
                add_column(interval, receivers, np.full(receivers.size, self.n), sign * turns)

            def add_interval(interval, left, right, active):
                add_boundary(interval, left, -1., active)
                add_boundary(interval, right, 1., active)

            for interval, left, right, covered in geometry:
                check_cancelled()
                if self.periodic and np.isfinite(self._radius):
                    for low, high, active, _, _ in self._masked_segments(left, right, covered):
                        add_interval(interval, low, high, active)
                    if np.any(covered):
                        add_interval(interval, self._mask_lower, self._mask_upper, covered)
                else:
                    add_interval(interval, left, right, ~covered)
                    receivers = np.flatnonzero(covered)
                    if receivers.size:
                        add_column(interval, receivers, np.full(receivers.size, self.n), np.ones(receivers.size))
                if entries > maximum_entries:
                    return
            if full_tail >= 0:
                receivers = np.arange(self.n)
                for interval in range(full_tail + 1):
                    check_cancelled()
                    add_column(interval, receivers, np.full(self.n, self.n), np.ones(self.n))
                    if entries > maximum_entries:
                        return
            if data:
                data_blocks.append(np.concatenate(data))
                row_blocks.append(np.concatenate(rows))
                column_blocks.append(np.concatenate(columns))
        if not entries:
            return
        values = np.concatenate(data_blocks)
        row = np.concatenate(row_blocks)
        column = np.concatenate(column_blocks)
        del data_blocks, row_blocks, column_blocks
        data.clear()
        rows.clear()
        columns.clear()
        check_cancelled()
        operator = coo_matrix((values, (row, column)), shape=(self.t.size * self.n, (self.t.size - 1) * (self.n + 1))).tocsr()
        operator.eliminate_zeros()
        actual = operator.data.nbytes + operator.indices.nbytes + operator.indptr.nbytes
        if actual + self._geometry_bytes <= self._CACHE_BYTES:
            self._source_operator = operator
            self.diagnostics["source_operator_bytes"] = actual

    def forward(self, rate_tx, record=False):
        """Return hazard for nonnegative initiation-rate density samples."""
        rate = np.asarray(rate_tx, dtype=float)
        if rate.shape != self.speed.shape or np.any(~np.isfinite(rate)) or np.any(rate < 0):
            raise ValueError("rate_tx must match speed_tx and contain finite nonnegative values.")
        integrated = 0.5 * (rate[:-1] + rate[1:]) * self.dt[:, None]
        prefix = np.pad(np.cumsum(integrated * self.dx, axis=1), ((0, 0), (1, 0)))
        self._forward_calls += 1
        # A fork-speed proposal generally uses this operator only once. Avoid
        # building an initiation-only CSR that its full speed adjoint cannot
        # reuse; repeated source evaluations or a source-only reverse pass
        # establish that the fixed geometry will actually be reused.
        if self._forward_calls > 1 and not self._source_attempted:
            self._compile_source_operator()
        if self._source_operator is not None:
            check_cancelled()
            hazard = np.asarray(self._source_operator @ prefix.ravel()).reshape(rate.shape)
        else:
            hazard = np.zeros_like(rate)
            for index in range(1, self.t.size):
                check_cancelled()
                report_measured_work(index * (index - 1) / max(1, self.t.size * (self.t.size - 1)))
                hazard[index] = self._slice(index, prefix)[0]
        if record:
            self._recorded = prefix
        return hazard

    def backward(self, dH, with_speed=True):
        """Differentiate the latest recorded prediction with respect to I,v.

        Set ``with_speed=False`` for fixed-speed initiation fitting; the
        source adjoint then reuses geometry and returns a zero speed gradient.
        """
        if self._recorded is None:
            raise RuntimeError("Call forward(rate_tx, record=True) before backward.")
        weights = np.asarray(dH, dtype=float)
        if weights.shape != self.speed.shape or np.any(~np.isfinite(weights)):
            raise ValueError("dH must match speed_tx and contain finite values.")
        prefix = self._recorded
        if not with_speed and not self._source_attempted:
            self._compile_source_operator()
        if not with_speed and self._source_operator is not None:
            check_cancelled()
            prefix_gradient = np.asarray(self._source_operator.T @ weights.ravel()).reshape(prefix.shape)
            integrated_gradient = self.dx * np.cumsum(prefix_gradient[:, :0:-1], axis=1)[:, ::-1]
            rate_gradient = np.zeros_like(self.speed)
            rate_gradient[:-1] += .5 * self.dt[:, None] * integrated_gradient
            rate_gradient[1:] += .5 * self.dt[:, None] * integrated_gradient
            return rate_gradient, np.zeros_like(self.speed)
        integrated_gradient = np.zeros((self.t.size - 1, self.n))
        speed_gradient = np.zeros_like(self.speed)
        full_gradient = np.zeros(self.t.size - 1)
        for index in range(1, self.t.size):
            check_cancelled()
            report_measured_work(index * (index - 1) / max(1, self.t.size * (self.t.size - 1)))
            if not np.any(weights[index]):
                continue
            if with_speed:
                _, tape, full_tail = self._slice(index, prefix, record=True)
            else:
                geometry, full_tail = self._geometry(index)
            if full_tail >= 0:
                full_gradient[:full_tail + 1] += np.sum(weights[index])
            if not with_speed:
                for interval, left, right, covered in geometry:
                    check_cancelled()
                    integrated_gradient[interval] += self._integral_backward(prefix[interval], left, right, weights[index], covered)[0]
                continue
            left_gradient = np.zeros(self.n)
            right_gradient = np.zeros(self.n)
            for interval, left, right, covered, left_hi, right_hi, left_lo, right_lo in reversed(tape):
                source, dleft, dright = self._integral_backward(prefix[interval], left, right, weights[index], covered)
                integrated_gradient[interval] += source
                left_gradient = self._advance_backward(left_gradient, left_lo, interval, speed_gradient) + dleft
                right_gradient = self._advance_backward(right_gradient, right_lo, interval, speed_gradient) + dright
                left_gradient = self._advance_backward(left_gradient, left_hi, interval, speed_gradient)
                right_gradient = self._advance_backward(right_gradient, right_hi, interval, speed_gradient)
            # Release this receiver-time tape before constructing the next.
            del tape
        integrated_gradient += self.dx * full_gradient[:, None]
        rate_gradient = np.zeros_like(self.speed)
        rate_gradient[:-1] += 0.5 * self.dt[:, None] * integrated_gradient
        rate_gradient[1:] += 0.5 * self.dt[:, None] * integrated_gradient
        return rate_gradient, speed_gradient
