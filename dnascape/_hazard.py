"""Linear-time KJMA hazards with derivatives of the discretized prediction.

The cumulative spatial source ``F(x,t)=integral_{left}^x I(y,t)dy`` turns the
two backwards-cone integrals into advection equations::

    (P_plus)_t + v (P_plus)_x = F
    (P_minus)_t - v (P_minus)_x = F
    H = P_minus - P_plus.

Midpoint semi-Lagrangian steps evaluate these potentials in O(X T) work for
a fixed number of trajectory substeps. Sources are constant inside centered
spatial cells, matching the finite-radius characteristic-cone backend. The
reverse pass differentiates *these same discrete steps*, including spatial
interpolation, trajectory dependence on speed, and source accumulation.

When a cone reaches a radius cutoff or wraps a circle, the domain of source
integration is a capped union rather than a pair of transport potentials.
Those cases use :class:`._hazard_cone.ConeHazard` instead.
"""

from __future__ import annotations

import numpy as np

from .utils import check_cancelled, report_measured_work


def _weights(nodes, positions):
    """Local polynomial values and first derivatives at each position."""
    count = nodes.shape[1]
    if count == 2:
        width = nodes[:, 1] - nodes[:, 0]
        weight = (positions - nodes[:, 0]) / width
        return (np.column_stack((1. - weight, weight)),
                np.column_stack((-1. / width, 1. / width)))
    if count == 4:
        a, b, c, d = nodes.T
        za, zb, zc, zd = positions - a, positions - b, positions - c, positions - d
        da = (a - b) * (a - c) * (a - d)
        db = (b - a) * (b - c) * (b - d)
        dc = (c - a) * (c - b) * (c - d)
        dd = (d - a) * (d - b) * (d - c)
        weights = np.column_stack((zb * zc * zd / da, za * zc * zd / db,
                                   za * zb * zd / dc, za * zb * zc / dd))
        derivative = np.column_stack(((zc * zd + zb * zd + zb * zc) / da,
                                      (zc * zd + za * zd + za * zc) / db,
                                      (zb * zd + za * zd + za * zb) / dc,
                                      (zb * zc + za * zc + za * zb) / dd))
        return weights, derivative
    weights = np.ones_like(nodes)
    derivative = np.zeros_like(nodes)
    for column in range(count):
        for other in range(count):
            if other != column:
                weights[:, column] *= (
                    (positions - nodes[:, other]) / (nodes[:, column] - nodes[:, other])
                )
        for differentiated in range(count):
            if differentiated == column:
                continue
            term = 1. / (nodes[:, column] - nodes[:, differentiated])
            for other in range(count):
                if other not in (column, differentiated):
                    term = term * (
                        (positions - nodes[:, other]) / (nodes[:, column] - nodes[:, other])
                    )
            derivative[:, column] += term
    return weights, derivative


class StaticHazardOperator:
    """Exact centered-cell cones for stationary initiation and scalar speed.

    ``forward`` accepts a length-X initiation-density vector and returns a
    time-by-position hazard. ``backward`` returns that vector's gradient and
    one scalar speed gradient. Piecewise-constant source-cell integrals are
    analytic, including partial cells, finite influence radii and finite
    circles. This is the stationary/constant-speed shortcut of the same KJMA
    continuum model, rather than a nodal trapezoid approximation.
    """

    backend = "static_cell_primitive"

    def __init__(self, speed, time_grid, resolution=1., periodic=False, radius=2000.):
        self.speed = float(speed)
        self.t = np.array(time_grid, dtype=float, copy=True)
        self.dx = float(resolution)
        self.periodic = bool(periodic)
        self.radius = np.inf if radius is None else float(radius)
        if not np.isfinite(self.speed) or self.speed <= 0:
            raise ValueError("Fork speed must be finite and positive.")
        if (self.t.ndim != 1 or not self.t.size or self.t[0] != 0.
                or np.any(~np.isfinite(self.t)) or np.any(np.diff(self.t) <= 0)):
            raise ValueError("Time grid must start at zero and increase strictly.")
        if not np.isfinite(self.dx) or self.dx <= 0:
            raise ValueError("Spatial resolution must be finite and positive.")
        if np.isnan(self.radius) or self.radius <= 0:
            raise ValueError("Influence radius must be positive.")
        self._recorded = None
        self.diagnostics = {
            "backend": self.backend, "influence_radius": self.radius,
            "spatial_discretization": "centred piecewise-constant cells",
        }

    def _prepare(self, rate):
        self.n = rate.size
        self.length = self.n * self.dx
        self.centers = (np.arange(self.n) + .5) * self.dx
        self.mass = np.r_[0., np.cumsum(rate * self.dx)]
        self.moment = np.r_[0., np.cumsum(rate * self.dx * self.centers)]
        self.rate = rate

    def _prefix(self, positions):
        if self.periodic:
            turns = np.floor(positions / self.length)
            local = positions - turns * self.length
        else:
            turns = np.zeros_like(positions)
            local = np.clip(positions, 0., self.length)
        cell = np.minimum(np.floor(local / self.dx).astype(np.intp), self.n - 1)
        cell = np.maximum(cell, 0)
        h = local - cell * self.dx
        partial_moment = cell * self.dx * h + .5 * h**2
        local_mass = self.mass[cell] + h * self.rate[cell]
        local_moment = self.moment[cell] + partial_moment * self.rate[cell]
        mass = local_mass + turns * self.mass[-1]
        moment = (local_moment + turns * self.moment[-1]
                  + .5 * turns * (turns - 1.) * self.length * self.mass[-1]
                  + turns * self.length * local_mass)
        return mass, moment, (cell, h, partial_moment, turns)

    def _prefix_pullback(self, gmass, gmoment, record):
        cell, h, partial_moment, turns = record
        local_mass_gradient = gmass + turns * self.length * gmoment
        local_moment_gradient = gmoment
        mass_gradient = np.zeros(self.n + 1)
        moment_gradient = np.zeros(self.n + 1)
        np.add.at(mass_gradient, cell, local_mass_gradient)
        np.add.at(moment_gradient, cell, local_moment_gradient)
        mass_gradient[-1] += np.sum(turns * gmass + .5 * turns * (turns - 1.) * self.length * gmoment)
        moment_gradient[-1] += np.sum(turns * gmoment)
        result = np.zeros(self.n)
        np.add.at(result, cell, h * local_mass_gradient + partial_moment * local_moment_gradient)
        result += self.dx * np.cumsum(mass_gradient[:0:-1])[::-1]
        result += self.dx * self.centers * np.cumsum(moment_gradient[:0:-1])[::-1]
        return result

    def _slice(self, time):
        radius = min(self.speed * time, self.radius,
                     .5 * self.length if self.periodic else np.inf)
        left = self._prefix(self.centers - radius)
        center = self._prefix(self.centers)
        right = self._prefix(self.centers + radius)
        distance_moment = (2. * self.centers * center[0]
                           - self.centers * (left[0] + right[0])
                           + left[1] + right[1] - 2. * center[1])
        hazard = time * (right[0] - left[0]) - distance_moment / self.speed
        return hazard, distance_moment, left, center, right

    def forward(self, rate_x, record=False):
        rate = np.asarray(rate_x, dtype=float)
        if rate.ndim != 1 or not rate.size or np.any(~np.isfinite(rate)) or np.any(rate < 0):
            raise ValueError("Stationary initiation density must be a finite non-negative vector.")
        self._prepare(rate.copy())
        hazard = np.zeros((self.t.size, self.n))
        for k, time in enumerate(self.t[1:], 1):
            check_cancelled()
            report_measured_work(k / max(1, self.t.size - 1))
            hazard[k] = np.maximum(self._slice(time)[0], 0.)
        self._recorded = rate.copy() if record else None
        return hazard

    def backward(self, dH):
        if self._recorded is None:
            raise RuntimeError("Call forward(..., record=True) before backward().")
        gradient = np.asarray(dH, dtype=float)
        if gradient.shape != (self.t.size, self.n) or np.any(~np.isfinite(gradient)):
            raise ValueError("Hazard gradient must have shape (time, position) and be finite.")
        self._prepare(self._recorded)
        rate_gradient = np.zeros(self.n)
        speed_gradient = 0.
        for k, time in enumerate(self.t[1:], 1):
            check_cancelled()
            report_measured_work(k / max(1, self.t.size - 1))
            hazard, moment, left, center, right = self._slice(time)
            g = gradient[k] * (hazard > 0.)
            rate_gradient += self._prefix_pullback(
                g * (-time + self.centers / self.speed), -g / self.speed, left[2])
            rate_gradient += self._prefix_pullback(
                -2. * g * self.centers / self.speed, 2. * g / self.speed, center[2])
            rate_gradient += self._prefix_pullback(
                g * (time + self.centers / self.speed), -g / self.speed, right[2])
            # Moving-boundary terms vanish where r=v*t because the cone
            # kernel is zero there; all other radius caps are speed-free.
            speed_gradient += np.dot(g, moment) / self.speed**2
        return rate_gradient, float(speed_gradient)


class HazardOperator:
    """KJMA map from sampled initiation density and speed to cumulative hazard.

    Arrays use ``(time, position)`` order. A spatial cell has width
    ``resolution`` kb; rates are in kb^-1 min^-1 and speeds in kb min^-1.
    ``radius`` is the maximum source distance in kb (``None`` is unlimited).
    ``backend='auto'`` chooses transport only when every characteristic stays
    inside that radius and a periodic cone cannot cover the circle. Use
    ``backend='cones'`` to hold backend selection fixed during a fit whose
    speed proposals can cross one of these limits. ``backend='transport'``
    requires an uncapped geometry and raises otherwise.

    Cubic local interpolation is limited to its enclosing sample values to
    prevent overshoot at sharp peaks; ``interpolation``
    can be set to ``'linear'`` for a monotone, lower-order reference. A CFL
    bound of one controls midpoint trajectory accuracy, not stability.
    """

    _CHECKPOINT_BYTES = 64 * 1024**2
    _CACHE_BYTES = 64 * 1024**2

    def __init__(self, speed_tx, time_grid, resolution=1., periodic=False,
                 radius=2000., backend="auto", *, interpolation="cubic"):
        from ._resources import ensure_capacity, hosted, working_bytes
        ensure_capacity(np.size(speed_tx) * 192, "Evaluating the space-time replication model")
        if hosted():
            self._CACHE_BYTES = min(self._CACHE_BYTES, working_bytes(8 * 1024**2))
            self._CHECKPOINT_BYTES = min(self._CHECKPOINT_BYTES, working_bytes(4 * 1024**2))
        supplied_speed = np.asarray(speed_tx, dtype=float)
        if supplied_speed.ndim == 2 and supplied_speed.strides[0] == 0:
            self.speed = np.broadcast_to(supplied_speed[:1].copy(), supplied_speed.shape)
        elif supplied_speed.ndim == 2 and supplied_speed.strides[1] == 0:
            self.speed = np.broadcast_to(supplied_speed[:, :1].copy(), supplied_speed.shape)
        else:
            self.speed = np.array(supplied_speed, copy=True)
        self.t = np.array(time_grid, dtype=float, copy=True)
        self.dx = float(resolution)
        self.periodic = bool(periodic)
        self.radius = np.inf if radius is None else float(radius)
        if (self.speed.ndim != 2 or self.t.ndim != 1 or self.t.size < 1
                or self.speed.shape[0] != self.t.size or self.speed.shape[1] < 1):
            raise ValueError("Speed must have shape (time, position) matching the time grid.")
        if (np.any(~np.isfinite(self.t)) or self.t[0] != 0.
                or np.any(np.diff(self.t) <= 0)):
            raise ValueError("Time grid must be finite, start at zero and increase strictly.")
        if not np.isfinite(self.dx) or self.dx <= 0:
            raise ValueError("Spatial resolution must be finite and positive.")
        if np.any(~np.isfinite(self.speed)) or np.any(self.speed <= 0):
            raise ValueError("Fork speed must contain finite positive values.")
        if np.isnan(self.radius) or self.radius <= 0:
            raise ValueError("Influence radius must be positive.")
        if interpolation not in {"linear", "cubic"}:
            raise ValueError("Interpolation must be 'linear' or 'cubic'.")
        aliases = {"characteristic_cones": "cones", "cone": "cones",
                   "hazard_transport": "transport", "potential_transport": "transport"}
        requested = aliases.get(str(backend), str(backend))
        if requested not in {"auto", "transport", "cones"}:
            raise ValueError("Hazard backend must be 'auto', 'transport' or 'cones'.")
        self.n = self.speed.shape[1]
        self.x = np.arange(self.n, dtype=float) * self.dx
        self.length = self.n * self.dx
        self.interpolation = interpolation
        # A conservative trajectory-length bound, including all time slices.
        # Auto selection is fixed for the lifetime of this operator.
        self.travel_bound = float(np.sum(
            np.maximum(np.max(self.speed[:-1], axis=1),
                       np.max(self.speed[1:], axis=1)) * np.diff(self.t)))
        cap = min(self.radius, .5 * self.length) if self.periodic else self.radius
        safe = ((not self.periodic and self.radius >= self.length)
                or self.travel_bound <= cap * (1. - 1e-12))
        if requested == "transport" and not safe:
            raise ValueError("Transport requires cones to stay inside the influence radius and half circle.")
        self._delegate = None
        if requested == "cones" or (requested == "auto" and not safe):
            from ._hazard_cone import ConeHazard
            self._delegate = ConeHazard(self.speed, self.t, self.dx, self.periodic, self.radius)
            self.backend = getattr(self._delegate, "backend", "characteristic_cones")
        else:
            self.backend = "hazard_transport"
        self.diagnostics = {
            "backend": self.backend,
            "influence_radius": self.radius,
            "finite_influence_approximation": bool(self._delegate is not None and np.isfinite(self.radius)),
            "spatial_discretization": "centred piecewise-constant cells",
            "backend_selection_fixed": True,
        }
        if self._delegate is not None:
            self.diagnostics.update(getattr(self._delegate, "diagnostics", {}))
        self._tape = None
        self._active = None
        self._prefix = None
        self._record_stride = 1
        self._geometry_cache = {}
        self._geometry_bytes = 0

    def _cache_geometry(self, key, value):
        def size(item):
            if isinstance(item, np.ndarray):
                return item.nbytes + 128
            if isinstance(item, (tuple, list)):
                return 64 + 8 * len(item) + sum(size(child) for child in item)
            if isinstance(item, bytes):
                return len(item) + 64
            return 64
        # Include conservative Python-object/dictionary overhead as well as
        # the numerical arrays; views are deliberately over-counted.
        cost = 128 + size(key) + size(value)
        if self._geometry_bytes + cost <= self._CACHE_BYTES:
            self._geometry_cache[key] = value
            self._geometry_bytes += cost
            self.diagnostics["geometry_cache_bytes"] = self._geometry_bytes
        return value

    def _potential_geometry(self, positions, direction):
        key = ("potential", direction, positions.tobytes())
        cached = self._geometry_cache.get(key)
        if cached is not None:
            return cached
        if self.periodic:
            coordinate = positions / self.dx
            lower = np.floor(coordinate).astype(np.intp)
            count = min(4 if self.interpolation == "cubic" else 2, self.n)
            if count == 1:
                indices = np.zeros((self.n, 1), dtype=np.intp)
                weights = np.ones((self.n, 1))
                turns = coordinate[:, None] / self.n
                derivative = np.zeros_like(weights)
                geometry = (indices, weights, turns, derivative, None, None,
                            np.full(self.n, 1. / self.length))
                return self._cache_geometry(key, geometry)
            unwrapped = lower[:, None] + np.arange(count) - (1 if count == 4 else 0)
            indices = np.mod(unwrapped, self.n)
            turns = np.floor_divide(unwrapped, self.n).astype(float)
            weights, derivative = _weights(unwrapped.astype(float) * self.dx, positions)
            bracket_indices = indices[:, 1:3] if count == 4 else None
            bracket_turns = turns[:, 1:3] if count == 4 else None
        else:
            if direction > 0:
                coordinates = np.concatenate(([-.5 * self.dx], self.x))
                mapping = np.concatenate(([-1], np.arange(self.n)))
                external = np.zeros(self.n + 1)
                exterior = positions <= coordinates[0]
            else:
                coordinates = np.concatenate((self.x, [self.length - .5 * self.dx]))
                mapping = np.concatenate((np.arange(self.n), [-1]))
                external = np.zeros(self.n + 1)
                external[-1] = 1.
                exterior = positions >= coordinates[-1]
            clipped = np.clip(positions, coordinates[0], coordinates[-1])
            lower = np.searchsorted(coordinates, clipped, side="right") - 1
            count = min(4 if self.interpolation == "cubic" else 2, self.n + 1)
            start = np.clip(lower - (1 if count == 4 else 0), 0, self.n + 1 - count)
            selected = start[:, None] + np.arange(count)
            weights, derivative = _weights(coordinates[selected], clipped)
            indices, turns = mapping[selected], external[selected]
            derivative[exterior] = 0.
            if np.any(exterior):
                indices[exterior] = -1
                weights[exterior] = 0.
                turns[exterior] = 0.
                if direction < 0:
                    weights[exterior, 0] = turns[exterior, 0] = 1.
            if count == 4:
                bracket = np.clip(lower, 0, self.n - 1)[:, None] + np.arange(2)
                bracket_indices, bracket_turns = mapping[bracket], external[bracket]
            else:
                bracket_indices = bracket_turns = None
        return self._cache_geometry(key, (
            indices, weights, turns, derivative, bracket_indices, bracket_turns, None))

    def _potential_sample(self, values, positions, total, direction, *, with_position=True):
        """Potential interpolation and a compact exact reverse record."""
        indices, weights, turns, derivative, bracket_indices, bracket_turns, external_slope = self._potential_geometry(positions, direction)
        sampled = np.where(indices >= 0, values[np.maximum(indices, 0)], 0.) + turns * total
        result = np.sum(weights * sampled, axis=1)
        slope = np.sum(derivative * sampled, axis=1) if with_position else None
        if with_position and external_slope is not None:
            slope += external_slope * total
        if bracket_indices is not None:
            bracket = np.where(bracket_indices >= 0, values[np.maximum(bracket_indices, 0)], 0.) + bracket_turns * total
            lower_side = np.where(bracket[:, 0] <= bracket[:, 1], 0, 1)
            upper_side = 1 - lower_side
            rows = np.arange(self.n)
            chosen = np.where(result < bracket[rows, lower_side], lower_side,
                              np.where(result > bracket[rows, upper_side], upper_side, -1))
            limited = chosen >= 0
            if np.any(limited):
                result[limited] = bracket[rows[limited], chosen[limited]]
                indices, weights, turns = indices.copy(), weights.copy(), turns.copy()
                indices[limited] = -1
                weights[limited] = turns[limited] = 0.
                indices[limited, 0] = bracket_indices[rows[limited], chosen[limited]]
                weights[limited, 0] = 1.
                turns[limited, 0] = bracket_turns[rows[limited], chosen[limited]]
                if with_position:
                    slope[limited] = 0.
        return result, (indices, weights, turns, slope)

    def _potential_pullback(self, gradient, record, *, with_position=True):
        indices, weights, turns, slope = record
        contributions = gradient[:, None] * weights
        valid = indices >= 0
        result = np.bincount(indices[valid], contributions[valid], minlength=self.n)
        return result, float(np.sum(contributions * turns)), gradient * slope if with_position else None

    def _speed_sample(self, values, positions):
        coordinate = positions / self.dx
        if self.periodic:
            lower_raw = np.floor(coordinate).astype(np.intp)
            weight = coordinate - lower_raw
            lower = np.mod(lower_raw, self.n)
            upper = np.mod(lower_raw + 1, self.n)
            slope = (values[upper] - values[lower]) / self.dx
        else:
            clipped = np.clip(coordinate, 0., self.n - 1.)
            lower = np.minimum(np.floor(clipped).astype(np.intp), self.n - 1)
            upper = np.minimum(lower + 1, self.n - 1)
            weight = clipped - lower
            slope = (values[upper] - values[lower]) / self.dx
            slope[(coordinate <= 0) | (coordinate >= self.n - 1)] = 0.
        return ((1. - weight) * values[lower] + weight * values[upper],
                (lower, upper, weight, slope))

    def _speed_pullback(self, gradient, record):
        lower, upper, weight, slope = record
        result = np.zeros(self.n)
        np.add.at(result, lower, gradient * (1. - weight))
        np.add.at(result, upper, gradient * weight)
        return result, gradient * slope

    def _source_sample(self, prefix, positions, *, with_position=True):
        """Exact integral of the piecewise-constant centered-cell source."""
        key = ("source", positions.tobytes())
        geometry = self._geometry_cache.get(key)
        if geometry is None:
            coordinate = (positions + .5 * self.dx) / self.dx
            if self.periodic:
                lower_raw = np.floor(coordinate).astype(np.intp)
                turns = np.floor_divide(lower_raw, self.n).astype(float)
                lower = np.mod(lower_raw, self.n)
                weight = coordinate - lower_raw
                active = np.ones_like(weight)
            else:
                clipped = np.clip(coordinate, 0., self.n)
                lower = np.minimum(np.floor(clipped).astype(np.intp), self.n - 1)
                weight = clipped - lower
                turns = np.zeros_like(weight)
                active = ((coordinate > 0) & (coordinate < self.n)).astype(float)
            upper = lower + 1
            geometry = self._cache_geometry(key, (lower, upper, weight, turns, active))
        lower, upper, weight, turns, active = geometry
        value = ((1. - weight) * prefix[lower] + weight * prefix[upper]
                 + turns * prefix[-1])
        slope = active * (prefix[upper] - prefix[lower]) / self.dx if with_position else None
        return value, (lower, upper, weight, turns, slope)

    def _source_pullback(self, gradient, record, *, with_position=True):
        lower, upper, weight, turns, slope = record
        result = (np.bincount(lower, gradient * (1. - weight), minlength=self.n + 1)
                  + np.bincount(upper, gradient * weight, minlength=self.n + 1))
        result[-1] += np.dot(gradient, turns)
        return result, gradient * slope if with_position else None

    def _step_geometry(self, k, dt, high, middle):
        key = ("step", k, dt, high, middle)
        geometry = self._geometry_cache.get(key)
        if geometry is not None:
            return geometry
        high_speed = (1. - high) * self.speed[k - 1] + high * self.speed[k]
        mid_speed = (1. - middle) * self.speed[k - 1] + middle * self.speed[k]
        geometry = []
        for direction in (1., -1.):
            midpoint = self.x - .5 * direction * dt * high_speed
            velocity, velocity_record = self._speed_sample(mid_speed, midpoint)
            departure = self.x - direction * dt * velocity
            geometry.append((direction, midpoint, departure, velocity_record))
        return self._cache_geometry(key, tuple(geometry))

    def _advance_potentials(self, plus, minus, total, prefix, k, dt, high, middle):
        mid_prefix = (1. - middle) * prefix[k - 1] + middle * prefix[k]
        results = []
        for direction, midpoint, departure, _ in self._step_geometry(k, dt, high, middle):
            potential = plus if direction > 0 else minus
            previous, _ = self._potential_sample(potential, departure, total, direction, with_position=False)
            source, _ = self._source_sample(mid_prefix, midpoint, with_position=False)
            results.append(previous + dt * source)
        return results[0], results[1], total + dt * mid_prefix[-1]

    def _reverse_steps(self):
        """Replay large tapes in square-root checkpoint blocks."""
        if self._record_stride == 1:
            yield from reversed(self._tape)
            return
        for start in reversed(range(0, len(self._tape), self._record_stride)):
            check_cancelled()
            states = self._tape[start][-1]
            local = []
            end = min(start + self._record_stride, len(self._tape))
            for index in range(start, end):
                check_cancelled()
                k, output_step, dt, high, middle, _ = self._tape[index]
                local.append((k, output_step, dt, high, middle, states))
                if index + 1 < end:
                    states = self._advance_potentials(
                        *states, self._prefix, k, dt, high, middle)
            yield from reversed(local)

    def forward(self, rate_tx, record=False):
        """Evaluate H; retain the discrete reverse tape only when requested."""
        rate = np.asarray(rate_tx, dtype=float)
        if rate.shape != self.speed.shape:
            raise ValueError("Initiation density must have the same (time, position) shape as speed.")
        if np.any(~np.isfinite(rate)) or np.any(rate < 0):
            raise ValueError("Initiation density must contain finite non-negative values.")
        if self._delegate is not None:
            result = self._delegate.forward(rate, record=record)
            self.diagnostics.update(getattr(self._delegate, "diagnostics", {}))
            return result
        self._tape = [] if record else None
        prefix = np.pad(np.cumsum(rate * self.dx, axis=1), ((0, 0), (1, 0)))
        self._prefix = prefix if record else None
        substeps = np.maximum(1, np.ceil(
            np.maximum(np.max(self.speed[:-1], axis=1), np.max(self.speed[1:], axis=1))
            * np.diff(self.t) / self.dx).astype(np.intp))
        total_steps = int(np.sum(substeps))
        # Small problems avoid replay. Large problems retain O(sqrt(S))
        # checkpoints and replay equally sized blocks during reverse, instead
        # of storing two full X-vectors for all S trajectory steps.
        tape_bytes = total_steps * self.n * 2 * np.dtype(float).itemsize
        self._record_stride = (max(1, int(np.ceil(np.sqrt(total_steps))))
                               if record and tape_bytes > self._CHECKPOINT_BYTES else 1)
        self.diagnostics["adjoint_checkpoint_stride"] = self._record_stride
        minus, plus = np.zeros(self.n), np.zeros(self.n)
        total = 0.
        hazard = np.zeros_like(rate)
        active = np.zeros_like(rate, dtype=bool)
        for k in range(1, self.t.size):
            check_cancelled()
            report_measured_work((k - 1) / max(1, self.t.size - 1))
            interval = self.t[k] - self.t[k - 1]
            count = int(substeps[k - 1])
            dt = interval / count
            for substep in range(count):
                check_cancelled()
                high = (substep + 1.) / count
                midpoint_weight = (substep + .5) / count
                # Store states, rather than every interpolation weight. The
                # reverse pass recomputes local weights, reducing tape memory
                # from dozens of X-vectors per step to the two potentials.
                if record:
                    previous_states = ((plus, minus, total)
                                       if len(self._tape) % self._record_stride == 0 else None)
                plus, minus, total = self._advance_potentials(
                    plus, minus, total, prefix, k, dt, high, midpoint_weight)
                if record:
                    self._tape.append((k, substep == count - 1, dt, high, midpoint_weight,
                                       previous_states))
            raw = minus - plus
            hazard[k] = np.maximum(raw, 0.)
            active[k] = raw > 0.
        self._active = active if record else None
        return hazard

    def backward(self, dH, with_speed=True):
        """Return gradients (dI, dv) of a recorded forward map.

        ``with_speed=False`` skips coordinate/speed derivatives and returns a
        broadcast zero speed gradient, reducing both work and memory for an
        initiation-only fit with a fixed propagation field.
        """
        if self._delegate is not None:
            return self._delegate.backward(dH, with_speed=with_speed)
        if self._tape is None or self._active is None:
            raise RuntimeError("Call forward(..., record=True) before backward().")
        gradient = np.asarray(dH, dtype=float)
        if gradient.shape != self.speed.shape or np.any(~np.isfinite(gradient)):
            raise ValueError("Hazard gradient must be finite and have shape (time, position).")
        prefix_gradient = np.zeros((self.t.size, self.n + 1))
        speed_gradient = (np.zeros_like(self.speed) if with_speed else
                          np.broadcast_to(np.asarray(0.), self.speed.shape))
        gplus, gminus = np.zeros(self.n), np.zeros(self.n)
        gtotal = 0.
        for k, output_step, dt, high, middle, previous_states in self._reverse_steps():
            check_cancelled()
            if output_step:
                report_measured_work((self.t.size - 1 - k) / max(1, self.t.size - 1))
                g = gradient[k] * self._active[k]
                gplus -= g
                gminus += g
            # The exterior potential B advances by dt times the whole-domain
            # initiation mass, and is shared by both characteristic families.
            gprefix = np.zeros(self.n + 1)
            gprefix[-1] += dt * gtotal
            ghigh = np.zeros(self.n)
            gmid = np.zeros(self.n)
            previous_total = gtotal
            old_plus, old_minus, old_total = previous_states
            mid_prefix = (1. - middle) * self._prefix[k - 1] + middle * self._prefix[k]
            for direction, midpoint, departure, velocity_record in self._step_geometry(k, dt, high, middle):
                potential = old_plus if direction > 0 else old_minus
                _, potential_record = self._potential_sample(potential, departure, old_total, direction,
                                                             with_position=with_speed)
                _, source_record = self._source_sample(mid_prefix, midpoint, with_position=with_speed)
                gnew = gplus if direction > 0 else gminus
                gold, gb, departure_gradient = self._potential_pullback(gnew, potential_record,
                                                                       with_position=with_speed)
                previous_total += gb
                source_gradient, midpoint_gradient = self._source_pullback(dt * gnew, source_record,
                                                                           with_position=with_speed)
                gprefix += source_gradient
                if with_speed:
                    velocity_gradient, velocity_midpoint_gradient = self._speed_pullback(
                        -direction * dt * departure_gradient, velocity_record)
                    gmid += velocity_gradient
                    midpoint_gradient += velocity_midpoint_gradient
                    ghigh += -.5 * direction * dt * midpoint_gradient
                if direction > 0:
                    gplus = gold
                else:
                    gminus = gold
            prefix_gradient[k - 1] += (1. - middle) * gprefix
            prefix_gradient[k] += middle * gprefix
            if with_speed:
                speed_gradient[k - 1] += (1. - high) * ghigh + (1. - middle) * gmid
                speed_gradient[k] += high * ghigh + middle * gmid
            gtotal = previous_total
        rate_gradient = self.dx * np.cumsum(prefix_gradient[:, :0:-1], axis=1)[:, ::-1]
        return rate_gradient, speed_gradient

