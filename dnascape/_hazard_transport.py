"""Large-step characteristic transport with retarded finite-radius boundaries.

Two potentials give the unrestricted cone integral. Advected monotone labels
locate when each characteristic crossed x +/- R; subtracting its older
potential gives the capped cone without retracing its past. On a circle R is
at most half the circumference, so no origin is counted twice.
"""

import numpy as np

from ._hazard import HazardOperator
from .utils import check_cancelled, report_measured_work


class FastTransportHazard(HazardOperator):
    backend = "retarded_potential_transport"

    def __init__(
        self,
        speed_tx,
        time_grid,
        resolution=1.0,
        periodic=False,
        radius=2000.0,
        *,
        uniform_speed=False,
        differentiate_speed=True,
    ):
        # Geometry/interpolation routines are shared with the reference
        # transport, while finite-radius handling below replaces cone replay.
        super().__init__(speed_tx, time_grid, resolution, False, np.inf, "transport")
        self.periodic = bool(periodic)
        self.radius = float(radius)
        self.cap = min(self.radius, self.length / 2.0) if periodic else self.radius
        self.capped = self.travel_bound > self.cap and (periodic or self.cap < self.length)
        self.uniform_speed = bool(uniform_speed and np.all(self.speed == self.speed[:, :1]))
        self.differentiate_speed = bool(differentiate_speed)
        self.labelled = self.capped and not self.uniform_speed
        self.travel = np.r_[
            0.0, np.cumsum(0.5 * (self.speed[:-1, 0] + self.speed[1:, 0]) * np.diff(self.t))
        ]
        self.backend = "retarded_potential_transport"
        self.diagnostics.update(
            backend=self.backend,
            influence_radius=self.radius,
            radius_boundary_correction=self.capped,
        )
        self._cache_limit = self._CACHE_BYTES
        self._fields = None
        self._winner = None
        self._kinematic_cache = {}
        self._kinematic_bytes = 0
        self._travel_slice_index = None
        self._travel_slices = {}

    def _cache_kinematics(self, key, value):
        # Labels and radius crossings depend only on the prescribed speed.
        # Keep this bounded separately from interpolation geometry, which can
        # otherwise evict the crossings before an adjoint reuses them.
        def size(item):
            if isinstance(item, np.ndarray):
                return item.nbytes + 128
            if isinstance(item, (tuple, list)):
                return 64 + sum(size(child) for child in item)
            return 64

        cost = size(value) + 256
        if self._kinematic_bytes + cost <= self._CACHE_BYTES:
            self._kinematic_cache[key] = value
            self._kinematic_bytes += cost
        return value

    def _potential_sample(self, values, positions, total, direction, *, with_position=True):
        u = positions / self.dx
        raw = np.floor(u).astype(int)
        f = u - raw
        j = raw.copy()
        linear = np.full(self.n, self.n < 4)
        scale = np.full(self.n, 1.0 / self.dx)
        if self.periodic:
            turn = np.floor_divide(j, self.n)
            loc = j % self.n
            extended = np.r_[values[-1] - total, values, values[: min(2, self.n)] + total]
            if self.n == 1:
                extended = np.r_[
                    values[-1] - total, values, values[0] + total, values[0] + 2 * total
                ]
            a, b, c, d = (extended[np.minimum(loc + s, len(extended) - 1)] for s in (0, 1, 2, 3))
        else:
            left, right = u < 0.0, u >= self.n - 1
            j = np.clip(j, -1, self.n - 1)
            f[left] = np.clip(2.0 * (u[left] + 0.5), 0.0, 1.0)
            f[right] = np.clip(2.0 * (u[right] - (self.n - 1)), 0.0, 1.0)
            linear |= (j < 1) | (j > self.n - 3)
            scale[left | right] = 2.0 / self.dx
            scale[u <= -0.5] = 0.0
            scale[u >= self.n - 0.5] = 0.0
            a, b, c, d = (values[np.clip(j + s, 0, self.n - 1)] for s in (-1, 0, 1, 2))
            b = np.where(left, 0.0, b)
            c = np.where(right, total, c)
            turn = 0.0
        c1 = (-2.0 * a - 3.0 * b + 6.0 * c - d) / 6.0
        c2 = 0.5 * (a - 2.0 * b + c)
        c3 = (-a + 3.0 * b - 3.0 * c + d) / 6.0
        result = b + f * (c1 + f * (c2 + f * c3))
        result = np.where(linear, b + f * (c - b), result)
        slope = (
            np.where(linear, c - b, c1 + f * (2.0 * c2 + 3.0 * f * c3)) * scale
            if with_position
            else None
        )
        low, high = np.minimum(b, c), np.maximum(b, c)
        chosen = np.where(
            result < low,
            np.where(b <= c, 0, 1),
            np.where(result > high, np.where(b >= c, 0, 1), -1),
        )
        result = np.clip(result, low, high) + turn * total
        if with_position:
            slope[chosen >= 0] = 0.0
        record = (raw if self.periodic else j, f, linear, chosen, slope, self.periodic)
        return result, record

    def _potential_pullback(self, gradient, record, *, with_position=True):
        j, f, linear, chosen, slope, periodic = record
        weights = (
            -f * (f - 1.0) * (f - 2.0) / 6.0,
            (f + 1.0) * (f - 1.0) * (f - 2.0) / 2.0,
            -(f + 1.0) * f * (f - 2.0) / 2.0,
            (f + 1.0) * f * (f - 1.0) / 6.0,
        )
        result = np.zeros(self.n)
        total = 0.0
        for slot, (offset, w) in enumerate(zip((-1, 0, 1, 2), weights)):
            lw = 1.0 - f if slot == 1 else f if slot == 2 else np.zeros(self.n)
            w = np.where(linear, lw, w)
            w = np.where(chosen >= 0, (chosen == slot - 1).astype(float), w)
            index = j + offset
            value = gradient * w
            if periodic:
                result += np.bincount(index % self.n, value, minlength=self.n)
                total += np.dot(value, np.floor_divide(index, self.n))
            else:
                valid = (index >= 0) & (index < self.n)
                result += np.bincount(index[valid], value[valid], minlength=self.n)
                total += value[index >= self.n].sum()
        return result, float(total), gradient * slope if with_position else None

    def _shift(self, field, direction, *, adjoint=False):
        count = int(round(self.cap / self.dx))
        shift = int(direction) * count * (-1 if adjoint else 1)
        result = np.roll(field, shift, axis=1)
        if not self.periodic:
            if shift > 0:
                result[:, :shift] = 0.0
            elif shift < 0:
                result[:, shift:] = 0.0
        return result

    @property
    def _integer_cap(self):
        return abs(self.cap / self.dx - round(self.cap / self.dx)) < 1e-11

    def _label_sample(self, values, positions):
        # U(x+L)=U(x)+L. Interpolating U-x preserves this identity and the
        # exact uniform-speed characteristic even beyond a linear boundary.
        displacement = values - self.x
        value, record = self._speed_sample(displacement, positions)
        lower, upper, weight, slope = record
        return positions + value, (lower, upper, weight, 1.0 + slope)

    def _label_pullback(self, gradient, record):
        lower, upper, weight, slope = record
        values = np.bincount(lower, gradient * (1.0 - weight), minlength=self.n) + np.bincount(
            upper, gradient * weight, minlength=self.n
        )
        return values, gradient * slope

    def _steps(self):
        override = getattr(self, "_steps_override", None)
        if override is not None:
            return override
        # Positive travel coordinates cross even a steep slow/fast spatial
        # interface without a CFL restriction. Only large temporal changes
        # need an extra midpoint stage.
        change = np.max(
            np.abs(self.speed[1:] - self.speed[:-1]) / (self.speed[1:] + self.speed[:-1]), axis=1
        )
        return np.maximum(1, np.ceil(2.0 * change).astype(int))

    def _travel_slice(self, k, weight):
        # Both directions use the same instantaneous travel coordinate. Keep
        # just this time slice, rather than duplicating its arrays in every
        # characteristic record across the complete prediction clock.
        if self._travel_slice_index != k:
            self._travel_slices.clear()
            self._travel_slice_index = k
        cached = self._travel_slices.get(weight)
        if cached is None:
            velocity = (1.0 - weight) * self.speed[k - 1] + weight * self.speed[k]
            slowness = 1.0 / velocity
            extended = np.r_[slowness, slowness[0]] if self.periodic else slowness
            nodes = np.r_[0.0, np.cumsum(0.5 * self.dx * (extended[:-1] + extended[1:]))]
            cached = velocity, slowness, nodes
            self._travel_slices[weight] = cached
        return cached

    def _departure(self, k, dt, high, count, q, direction):
        key = ("travel", k, dt, high, count, q, direction)
        # Prescribed speeds reuse only positions across many initiation
        # proposals. Speed fits retain the reverse records directly, avoiding
        # the cost of decoding compact records during each reverse pass.
        cache = self._geometry_cache if self.differentiate_speed else self._kinematic_cache
        remember = self._cache_geometry if self.differentiate_speed else self._cache_kinematics
        cached = cache.get(key)
        if cached is not None:
            return cached
        weight = high - 0.5 * (1.0 - q) / count
        if self.uniform_speed:
            velocity = (1.0 - weight) * self.speed[k - 1, 0] + weight * self.speed[k, 0]
            multiplier = -direction * dt * (1.0 - q)
            return remember(key, (self.x + multiplier * velocity, (multiplier, weight)))
        velocity, slowness, nodes = self._travel_slice(k, weight)
        target = nodes[: self.n] - direction * dt * (1.0 - q)
        turns = np.floor(target / nodes[-1]) if self.periodic else np.zeros(self.n)
        local = target - turns * nodes[-1]
        lower = np.clip(np.searchsorted(nodes, local, side="right") - 1, 0, len(nodes) - 1)
        upper = np.minimum(lower + 1, len(nodes) - 1)
        li, ui = lower % self.n, upper % self.n
        outside = (local < 0.0) | (local > nodes[-1])
        ui = np.where(outside, li, ui)
        offset = local - nodes[lower]
        slope = (slowness[ui] - slowness[li]) / self.dx
        root = np.sqrt(np.maximum(slowness[li] ** 2 + 2.0 * slope * offset, 0.0))
        distance = 2.0 * offset / (slowness[li] + root)
        position = lower * self.dx + distance + turns * self.length
        if not self.differentiate_speed:
            # A prescribed speed needs only departure positions, never their
            # speed derivative. This fits substantially more fixed geometry
            # into the same bounded cache when estimating initiation.
            return self._cache_kinematics(key, (position, None))
        c1 = np.where(outside, 0.0, 0.5 * distance * distance / self.dx)
        record = (
            li, ui, distance - c1, c1,
            slowness[li] + slope * distance, turns if self.periodic else 0.0,
            velocity, weight,
        )
        return remember(key, (position, record))

    def _travel_pullback(self, gradient, record):
        if len(record) == 2:
            multiplier, weight = record
            result = np.zeros(self.n)
            result[0] = multiplier * np.sum(gradient)
            return result, weight
        lo, hi, c0, c1, slowness, turns, velocity, weight = record
        q = gradient / slowness
        count = self.n + 1 if self.periodic else self.n
        prefix = np.zeros(count)
        prefix[: self.n] += q
        prefix[: self.n] -= np.bincount(lo, q, minlength=self.n)
        if self.periodic:
            prefix[-1] -= np.dot(turns, q)
        gs = -np.bincount(lo, q * c0, minlength=self.n) - np.bincount(hi, q * c1, minlength=self.n)
        segments = 0.5 * self.dx * np.cumsum(prefix[:0:-1])[::-1]
        if self.periodic:
            gs += segments + np.roll(segments, 1)
        else:
            gs[:-1] += segments
            gs[1:] += segments
        return -gs / velocity**2, weight

    def _advance(self, states, prefix, k, dt, high, middle, count):
        plus, minus, total, uplus, uminus = states
        result, labels = [], []
        label_key = ("labels", k, dt, high, middle, count)
        cached_labels = self._kinematic_cache.get(label_key)
        gauss = (0.5 - np.sqrt(3.0) / 6.0, 0.5 + np.sqrt(3.0) / 6.0)
        for direction in (1.0, -1.0):
            departure, _ = self._departure(k, dt, high, count, 0.0, direction)
            previous = plus if direction > 0 else minus
            value, _ = self._potential_sample(
                previous, departure, total, direction, with_position=False
            )
            for q in gauss:
                r = 1.0 - q
                position, _ = self._departure(k, dt, high, count, q, direction)
                weight = high - r / count
                source = (
                    prefix[0] if prefix.strides[0] == 0
                    else (1.0 - weight) * prefix[k - 1] + weight * prefix[k]
                )
                sampled, _ = self._source_sample(source, position, with_position=False)
                value = value + 0.5 * dt * sampled
            result.append(value)
            if self.labelled and cached_labels is None:
                old = uplus if direction > 0 else uminus
                labels.append(self._label_sample(old, departure)[0])
        if self.labelled:
            labels = (
                cached_labels
                if cached_labels is not None
                else self._cache_kinematics(label_key, tuple(labels))
            )
        total += dt * ((1.0 - middle) * prefix[k - 1, -1] + middle * prefix[k, -1])
        return (
            result[0],
            result[1],
            total,
            labels[0] if labels else None,
            labels[1] if labels else None,
        )

    def _shift_fields(self, potential, cumulative, labels, direction):
        positions = self.x - direction * self.cap
        if self._integer_cap:
            shifted_d = self._shift(potential - cumulative, direction)
            shifted_u = np.empty((0,))
            if self.labelled:
                if self.periodic:
                    shifted_u = positions + self._shift(labels - self.x, direction)
                else:
                    indices = np.clip(
                        np.arange(self.n) - int(direction) * int(round(self.cap / self.dx)),
                        0,
                        self.n - 1,
                    )
                    shifted_u = positions + (labels - self.x)[:, indices]
            return shifted_u, shifted_d
        shifted_u, shifted_d = [], []
        for k in range(len(self.t)):
            if self.labelled:
                shifted_u.append(self._label_sample(labels[k], positions)[0])
            shifted_d.append(
                self._potential_sample(
                    potential[k] - cumulative[k], positions, 0.0, direction, with_position=False
                )[0]
            )
        return np.asarray(shifted_u), np.asarray(shifted_d)

    def _root(self, k, labels, shifted, direction):
        key = ("root", k, direction)
        cached = self._kinematic_cache.get(key)
        if cached is not None:
            return cached
        positions = self.x - direction * self.cap
        if self.uniform_speed:
            target = self.travel[k] - self.cap
            lo = int(
                np.clip(
                    np.searchsorted(self.travel[: k + 1], target, side="right") - 1,
                    0,
                    max(k - 1, 0),
                )
            )
            hi = min(lo + 1, k)
            denominator = self.travel[hi] - self.travel[lo]
            weight = (
                np.clip((target - self.travel[lo]) / denominator, 0.0, 1.0) if denominator else 0.0
            )
            active = np.full(self.n, target >= 0.0)
            if not self.periodic:
                active &= (positions >= -0.5 * self.dx) & (positions <= self.length - 0.5 * self.dx)
            return self._cache_kinematics(key, (lo, hi, weight, denominator, active))
        query = labels[k]
        active = -direction * query >= -direction * shifted[0]
        if not self.periodic:
            active &= (positions >= -0.5 * self.dx) & (positions <= self.length - 0.5 * self.dx)
        if not np.any(active):
            return self._cache_kinematics(key, (0, 0, 0.0, 0.0, active))
        lo, hi = np.zeros(self.n, dtype=int), np.full(self.n, k, dtype=int)
        rows = np.arange(self.n)
        for _ in range(max(1, int(k).bit_length())):
            mid = (lo + hi) // 2
            before = -direction * shifted[mid, rows] <= -direction * query
            lo = np.where(before, mid, lo)
            hi = np.where(before, hi, mid)
        low, high = shifted[lo, rows], shifted[hi, rows]
        denominator = high - low
        weight = np.divide(query - low, denominator, out=np.zeros(self.n), where=denominator != 0.0)
        return self._cache_kinematics(key, (lo, hi, np.clip(weight, 0.0, 1.0), denominator, active))

    def forward(self, rate_tx, record=False):
        check_cancelled()
        rate = np.asarray(rate_tx, dtype=float)
        if rate.shape != self.speed.shape or np.any(~np.isfinite(rate)) or np.any(rate < 0):
            raise ValueError(
                "Initiation density must be finite, non-negative and match the speed grid."
            )
        source = rate[:1] if rate.strides[0] == 0 else rate
        prefix = np.pad(np.cumsum(source, axis=1) * self.dx, ((0, 0), (1, 0)))
        prefix = np.broadcast_to(prefix, (len(self.t), self.n + 1))
        plus, minus = np.zeros_like(rate), np.zeros_like(rate)
        uplus = np.empty_like(rate) if self.labelled else None
        uminus = np.empty_like(rate) if self.labelled else None
        if self.labelled:
            uplus[0] = uminus[0] = self.x
        states = (
            plus[0],
            minus[0],
            0.0,
            self.x if self.labelled else None,
            self.x if self.labelled else None,
        )
        self._tape = [] if record else None
        for k, count in enumerate(self._steps(), 1):
            check_cancelled()
            dt = (self.t[k] - self.t[k - 1]) / count
            for step in range(count):
                high, middle = (step + 1.0) / count, (step + 0.5) / count
                if record:
                    self._tape.append((k, step == count - 1, dt, high, middle, count, states))
                states = self._advance(states, prefix, k, dt, high, middle, count)
            plus[k], minus[k] = states[:2]
            if self.labelled:
                uplus[k], uminus[k] = states[3:]
            report_measured_work(k / max(1, len(self.t) - 1))
        raw = minus - plus
        cumulative = None
        shifted = []
        if self.capped:
            if prefix.strides[0] == 0:
                centres = 0.5 * (prefix[0, :-1] + prefix[0, 1:])
                cumulative = self.t[:, None] * centres
            else:
                centres = 0.5 * (prefix[:, :-1] + prefix[:, 1:])
                cumulative = np.vstack(
                    (
                        np.zeros(self.n),
                        np.cumsum(
                            0.5 * (centres[:-1] + centres[1:]) * np.diff(self.t)[:, None], axis=0
                        ),
                    )
                )
            for direction, potential, labels in ((1.0, plus, uplus), (-1.0, minus, uminus)):
                su, sd = self._shift_fields(potential, cumulative, labels, direction)
                shifted.append((su, sd))
                for k in range(1, len(self.t)):
                    check_cancelled()
                    lo, hi, w, _, active = self._root(k, labels, su, direction)
                    if not np.any(active):
                        continue
                    rows = np.arange(self.n)
                    raw[k] += direction * np.where(
                        active, (1.0 - w) * sd[lo, rows] + w * sd[hi, rows], 0.0
                    )
        clipped = np.maximum(raw, 0.0)
        monotone = np.maximum.accumulate(clipped, axis=0)
        self.diagnostics["max_monotone_correction"] = float(np.max(monotone - clipped))
        self._winner = None
        if record and np.any(monotone != clipped):
            indices = np.arange(len(self.t), dtype=np.int32)[:, None]
            self._winner = np.maximum.accumulate(np.where(clipped >= monotone, indices, 0), axis=0)
        self._prefix = prefix if record else None
        self._fields = (plus, minus, uplus, uminus, cumulative, shifted) if record else None
        self._active = raw > 0.0 if record else None
        return monotone

    def backward(self, dH, with_speed=True, *, with_rate=True):
        if with_speed and not self.differentiate_speed:
            raise ValueError("This operator recorded prescribed speed; request only its initiation derivative.")
        if self._fields is None:
            raise RuntimeError("Record a forward prediction before its adjoint.")
        check_cancelled()
        plus, minus, uplus, uminus, cumulative, shifted = self._fields
        g = np.asarray(dH)
        if self._winner is not None:
            accumulated = np.zeros_like(g)
            rows = np.arange(self.n)
            for k in range(len(self.t)):
                np.add.at(accumulated, (self._winner[k], rows), g[k])
            g = accumulated
        g = g * self._active
        gp, gm = -g.copy(), g.copy()
        gu = [np.zeros_like(g), np.zeros_like(g)] if self.labelled else [None, None]
        gtravel = np.zeros(len(self.t))
        gc = np.zeros_like(g) if self.capped else None
        if self.capped:
            rows = np.arange(self.n)
            for j, (direction, potential, labels) in enumerate(
                ((1.0, plus, uplus), (-1.0, minus, uminus))
            ):
                su, sd = shifted[j]
                gsu, gsd = np.zeros_like(g), np.zeros_like(g)
                for k in range(1, len(self.t)):
                    check_cancelled()
                    lo, hi, w, denominator, active = self._root(k, labels, su, direction)
                    if not np.any(active):
                        continue
                    value = direction * g[k] * active
                    if self.uniform_speed:
                        gsd[lo] += value * (1.0 - w)
                        gsd[hi] += value * w
                    else:
                        np.add.at(gsd, (lo, rows), value * (1.0 - w))
                        np.add.at(gsd, (hi, rows), value * w)
                    if with_speed:
                        q = np.divide(
                            value * (sd[hi, rows] - sd[lo, rows]),
                            denominator,
                            out=np.zeros(self.n),
                            where=denominator != 0.0,
                        )
                        if self.uniform_speed:
                            total = q.sum()
                            gtravel[k] += total
                            gtravel[lo] -= total * (1.0 - w)
                            gtravel[hi] -= total * w
                        else:
                            gu[j][k] += q
                            np.add.at(gsu, (lo, rows), -q * (1.0 - w))
                            np.add.at(gsu, (hi, rows), -q * w)
                positions = self.x - direction * self.cap
                if self._integer_cap:
                    value = self._shift(gsd, direction, adjoint=True)
                    (gp if direction > 0 else gm)[:] += value
                    gc -= value
                    if with_speed and self.labelled:
                        gu[j] += self._shift(gsu, direction, adjoint=True)
                    continue
                for k in range(len(self.t)):
                    _, record = self._potential_sample(
                        potential[k] - cumulative[k], positions, 0.0, direction, with_position=False
                    )
                    value = self._potential_pullback(gsd[k], record, with_position=False)[0]
                    (gp if direction > 0 else gm)[k] += value
                    gc[k] -= value
                    if with_speed and self.labelled:
                        _, label_record = self._label_sample(labels[k], positions)
                        gu[j][k] += self._label_pullback(gsu[k], label_record)[0]
        prefix_gradient = np.zeros_like(self._prefix) if with_rate else None
        speed_gradient = (
            np.zeros_like(self.speed) if with_speed else np.broadcast_to(0.0, self.speed.shape)
        )
        gplus, gminus, gtotal = np.zeros(self.n), np.zeros(self.n), 0.0
        gup, gum = np.zeros(self.n), np.zeros(self.n)
        gauss = (0.5 - np.sqrt(3.0) / 6.0, 0.5 + np.sqrt(3.0) / 6.0)
        for k, output, dt, high, middle, count, previous in reversed(self._tape):
            check_cancelled()
            if output:
                gplus += gp[k]
                gminus += gm[k]
                if self.labelled and with_speed:
                    gup += gu[0][k]
                    gum += gu[1][k]
            old_plus, old_minus, old_total, old_up, old_um = previous
            if with_rate:
                prefix_gradient[k - 1, -1] += (1.0 - middle) * dt * gtotal
                prefix_gradient[k, -1] += middle * dt * gtotal
            previous_total = gtotal
            for direction in (1.0, -1.0):
                departure, travel_record = self._departure(k, dt, high, count, 0.0, direction)
                old = old_plus if direction > 0 else old_minus
                value = gplus if direction > 0 else gminus
                _, pr = self._potential_sample(
                    old, departure, old_total, direction, with_position=with_speed
                )
                gold, gb, gd = self._potential_pullback(value, pr, with_position=with_speed)
                previous_total += gb
                if self.labelled and with_speed:
                    _, ur = self._label_sample(old_up if direction > 0 else old_um, departure)
                    glu, gdu = self._label_pullback(gup if direction > 0 else gum, ur)
                    gd += gdu
                    if direction > 0:
                        gup = glu
                    else:
                        gum = glu
                if with_speed:
                    value_velocity, weight_velocity = self._travel_pullback(gd, travel_record)
                    speed_gradient[k - 1] += (1.0 - weight_velocity) * value_velocity
                    speed_gradient[k] += weight_velocity * value_velocity
                for q in gauss:
                    r = 1.0 - q
                    position, source_travel = self._departure(k, dt, high, count, q, direction)
                    weight = high - r / count
                    source = (
                        self._prefix[0] if self._prefix.strides[0] == 0
                        else (1.0 - weight) * self._prefix[k - 1] + weight * self._prefix[k]
                    )
                    _, sr = self._source_sample(source, position, with_position=with_speed)
                    if with_rate:
                        ps, gs = self._source_pullback(
                            0.5 * dt * value, sr, with_position=with_speed
                        )
                        prefix_gradient[k - 1] += (1.0 - weight) * ps
                        prefix_gradient[k] += weight * ps
                    else:
                        gs = 0.5 * dt * value * sr[-1] if with_speed else None
                    if with_speed:
                        value_velocity, weight_velocity = self._travel_pullback(gs, source_travel)
                        speed_gradient[k - 1] += (1.0 - weight_velocity) * value_velocity
                        speed_gradient[k] += weight_velocity * value_velocity
                if direction > 0:
                    gplus = gold
                else:
                    gminus = gold
            gtotal = previous_total
        if self.capped and with_rate:
            temporal = np.cumsum(gc[:0:-1], axis=0)[::-1] * (0.5 * np.diff(self.t))[:, None]
            centres = np.zeros_like(gc)
            centres[:-1] += temporal
            centres[1:] += temporal
            prefix_gradient[:, :-1] += 0.5 * centres
            prefix_gradient[:, 1:] += 0.5 * centres
        if self.capped and self.uniform_speed and with_speed:
            derivative = 0.5 * np.diff(self.t) * np.cumsum(gtravel[:0:-1])[::-1]
            speed_gradient[:-1, 0] += derivative
            speed_gradient[1:, 0] += derivative
        rate_gradient = (
            self.dx * np.cumsum(prefix_gradient[:, :0:-1], axis=1)[:, ::-1]
            if with_rate
            else np.broadcast_to(0.0, self.speed.shape)
        )
        return rate_gradient, speed_gradient
