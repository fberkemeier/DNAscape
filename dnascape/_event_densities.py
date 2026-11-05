"""Deterministic source/sink observables, isolated from the kinetic fit solvers.

With s=exp(-H) and g+/-=rho+/-/s, the common equations are
    g+/-_t +/- (v g+/-)_x = I, H_t = v(g+ + g-),
    n = I s, c = 2 v s g+ g-.
Stationary fields use exact travel-coordinate cone primitives. General fields
use conservative second-order spatial fluxes, SSP RK2 for sampled fields and
RK4 when accumulating event counts.
Counts are integrated independently of the displayed temporal sampling.
"""

import numpy as np
from numpy.polynomial.legendre import leggauss

from .mapping import _resolve_n_bins_from_fields
from .utils import check_cancelled, initial_prediction, report_measured_work

_NODES, _WEIGHTS = leggauss(16)
_DENSE_OUTPUT_VALUES = 2_000_000


class _Output:
    """Append temporal slices with bounded RAM and no final dense duplicate."""

    def __init__(self, n):
        self.n, self.count, self.capacity = n, 0, 16
        self.values = self.allocate(self.capacity)

    def allocate(self, capacity):
        if self.n * capacity > _DENSE_OUTPUT_VALUES:
            from ._kinetic_tiled import disk_array

            return disk_array((capacity, self.n))
        return np.empty((capacity, self.n))

    def append(self, values):
        if self.count == self.capacity:
            self.capacity *= 2
            expanded = self.allocate(self.capacity)
            block = max(1, _DENSE_OUTPUT_VALUES // self.n)
            for start in range(0, self.count, block):
                check_cancelled()
                stop = min(start + block, self.count)
                expanded[start:stop] = self.values[start:stop]
            self.values = expanded
        self.values[self.count] = values
        self.count += 1

    def result(self):
        return self.values[: self.count].T


class _Input:
    def __init__(self, values, n, dt, clock, name, positive=False):
        a = np.asarray(values, dtype=float)
        if np.any(~np.isfinite(a)) or np.any(a <= 0 if positive else a < 0):
            raise ValueError(
                f"{name} must contain finite {'positive' if positive else 'non-negative'} values."
            )
        if not a.size:
            raise ValueError(f"{name} cannot be empty.")
        if a.size == 1:
            self.values = a.reshape(1, 1)
        elif a.ndim == 1 and a.size == n:
            self.values = a[:, None]
        elif a.ndim == 2 and a.shape[0] in (1, n):
            self.values = a
        else:
            raise ValueError(
                f"{name} must be scalar, a spatial vector, a temporal row or a position-by-time matrix."
            )
        m = self.values.shape[1]
        self.clock = clock if clock is not None and len(clock) == m else np.arange(m) * dt
        self.static = m == 1 or np.all(self.values == self.values[:, :1])
        self.uniform = self.values.shape[0] == 1 or np.all(self.values == self.values[:1])

    def at(self, time):
        if self.static:
            return self.values[:, 0]
        k = int(
            np.clip(np.searchsorted(self.clock, time, side="right") - 1, 0, len(self.clock) - 2)
        )
        w = np.clip((time - self.clock[k]) / (self.clock[k + 1] - self.clock[k]), 0.0, 1.0)
        return (1.0 - w) * self.values[:, k] + w * self.values[:, k + 1]


def _conditional_transport(density, velocity, dx, periodic, direction):
    previous, following = np.roll(density, 1), np.roll(density, -1)
    if not periodic:
        previous[0] = (
            -density[0] if direction == 1 else 2 * density[0] - density[min(1, len(density) - 1)]
        )
        following[-1] = (
            -density[-1] if direction == -1 else 2 * density[-1] - density[max(0, len(density) - 2)]
        )
    backward, forward = density - previous, following - density
    slope = np.where(
        backward * forward > 0.0, np.sign(backward) * np.minimum(abs(backward), abs(forward)), 0.0
    )
    if direction == 1:
        face = 0.5 * (velocity + np.roll(velocity, -1))
        flux = face * np.maximum(density + 0.5 * slope, 0.0)
        if not periodic:
            flux[-1] = velocity[-1] * max(density[-1] + 0.5 * slope[-1], 0.0)
        incoming = np.roll(flux, 1)
        if not periodic:
            incoming[0] = 0.0
    else:
        face = 0.5 * (velocity + np.roll(velocity, 1))
        flux = face * np.maximum(density - 0.5 * slope, 0.0)
        if not periodic:
            flux[0] = velocity[0] * max(density[0] - 0.5 * slope[0], 0.0)
        incoming = np.roll(flux, -1)
        if not periodic:
            incoming[-1] = 0.0
    return (incoming - flux) / dx


class _Transport:
    def __init__(self, n, dx, periodic, floor, cfl, counts):
        from ._event_native import create_transport

        self.n, self.dx, self.periodic, self.floor, self.cfl = n, dx, periodic, floor, cfl
        self.state = np.zeros(5 * n)
        self.native = create_transport(n)
        self.method = "conditional_transport_rk4" if counts else "conditional_transport_ssprk2"
        self.counts = counts

    def advance(self, start, end, rate, speed):
        i0, i1, v0, v1 = rate.at(start), rate.at(end), speed.at(start), speed.at(end)
        vmax, imax = max(np.max(v0), np.max(v1)), max(np.max(i0), np.max(i1))
        # Transport stability and reaction accuracy are independent of the
        # display clock. No adaptive ODE controller revisits a step.
        step = (self.cfl if self.counts else min(self.cfl, 0.65)) * self.dx / vmax
        if self.counts:
            step = min(step, 0.02 / np.sqrt(max(imax * vmax, 1e-300)))
        steps = max(1, int(np.ceil((end - start) / step)))
        if self.native is not None:
            self.native.advance(
                self.state,
                end - start,
                steps,
                self.dx,
                self.periodic,
                self.counts,
                self.floor,
                i0,
                i1,
                v0,
                v1,
            )
            return
        n, h = self.n, (end - start) / steps

        def rhs(y, w):
            r, v = (1.0 - w) * i0 + w * i1, (1.0 - w) * v0 + w * v1
            s = np.exp(-np.maximum(y[:n], 0.0)) if self.counts else np.zeros(n)
            plus, minus = np.maximum(y[n : 2 * n], 0.0), np.maximum(y[2 * n : 3 * n], 0.0)
            velocity = np.broadcast_to(v, (n,))
            return np.concatenate(
                (
                    v * (plus + minus),
                    r + _conditional_transport(plus, velocity, self.dx, self.periodic, 1),
                    r + _conditional_transport(minus, velocity, self.dx, self.periodic, -1),
                    np.broadcast_to(r * s, (n,)),
                    np.where(s > self.floor, 2 * v * plus * minus * s, 0.0),
                )
            )

        for j in range(steps):
            if j % 32 == 0:
                check_cancelled()
            w = j / steps
            k1 = rhs(self.state, w)
            if not self.counts:
                k2 = rhs(self.state + h * k1, w + 1.0 / steps)
                self.state = np.maximum(self.state + 0.5 * h * (k1 + k2), 0.0)
                continue
            k2 = rhs(self.state + 0.5 * h * k1, w + 0.5 / steps)
            k3 = rhs(self.state + 0.5 * h * k2, w + 0.5 / steps)
            k4 = rhs(self.state + h * k3, w + 1.0 / steps)
            self.state = np.maximum(self.state + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4), 0.0)


class _Stationary:
    def __init__(self, n, dx, periodic, rate, speed, floor):
        from ._hazard_static import StationaryHazard

        self.rate = np.broadcast_to(rate.at(0.0), (n,))
        self.speed = np.broadcast_to(speed.at(0.0), (n,))
        self.operator = StationaryHazard(self.speed, [0.0], dx, periodic, np.inf, cache_bytes=0)
        # Periodic mean-field transport wraps conditional fork populations,
        # rather than discarding incoming forks at an influence-radius cap.
        self.operator.cap = np.inf
        self.operator.forward(self.rate)
        self.n, self.floor = n, floor
        self.state = np.zeros(5 * n)
        self.method = "stationary_cone_primitives"
        self.quadrature_span = 2.0 / np.sqrt(
            max(float(np.max(self.rate) * np.max(self.speed)), 1e-300)
        )

    def values(self, time):
        op = self.operator
        hazard, left, right = op._slice(time)
        plus = (op.center_prefix[0] - left[0]) / self.speed
        minus = (right[0] - op.center_prefix[0]) / self.speed
        s = np.exp(-np.maximum(hazard, 0.0))
        birth = self.rate * s
        death = np.where(s > self.floor, 2 * self.speed * plus * minus * s, 0.0)
        return hazard, plus, minus, birth, death

    def advance(self, start, end, rate, speed):
        left = start
        while self.counts and left < end:
            right = min(end, left + self.quadrature_span)
            half, middle = 0.5 * (right - left), 0.5 * (right + left)
            for node, weight in zip(_NODES, _WEIGHTS):
                check_cancelled()
                *_, birth, death = self.values(middle + half * node)
                self.state[3 * self.n : 4 * self.n] += half * weight * birth
                self.state[4 * self.n :] += half * weight * death
            left = right
            if np.max(np.exp(-np.maximum(self.values(right)[0], 0.0))) < 1e-14:
                break
        self.state[: 3 * self.n] = np.concatenate(self.values(end)[:3])


class _Uniform(_Stationary):
    """Closed temporal primitives, including both finite linear boundaries."""

    def __init__(self, n, dx, periodic, rate, speed, floor):
        self.n, self.floor, self.periodic = n, floor, periodic
        self.state = np.zeros(5 * n)
        self.method = "uniform_kinetic_primitives"
        self.quadrature_span = 2.0 / np.sqrt(
            max(float(np.max(rate.values) * np.max(speed.values)), 1e-300)
        )
        self.clock = np.unique(np.r_[0.0, rate.clock, speed.clock])
        self.r = np.asarray([rate.at(t)[0] for t in self.clock])
        self.v = np.asarray([speed.at(t)[0] for t in self.clock])
        steps = np.diff(self.clock)
        self.rs = np.r_[np.diff(self.r) / steps, 0.0]
        self.vs = np.r_[np.diff(self.v) / steps, 0.0]
        self.A = np.r_[0.0, np.cumsum(self.r[:-1] * steps + 0.5 * self.rs[:-1] * steps**2)]
        self.D = np.r_[0.0, np.cumsum(self.v[:-1] * steps + 0.5 * self.vs[:-1] * steps**2)]
        partial = (
            self.D[:-1] * (self.r[:-1] * steps + 0.5 * self.rs[:-1] * steps**2)
            + 0.5 * self.r[:-1] * self.v[:-1] * steps**2
            + (self.rs[:-1] * self.v[:-1] + 0.5 * self.r[:-1] * self.vs[:-1]) * steps**3 / 3.0
            + self.rs[:-1] * self.vs[:-1] * steps**4 / 8.0
        )
        self.B = np.r_[0.0, np.cumsum(partial)]
        self.distance = (np.arange(n) + 0.5) * dx

    def primitive(self, index, h):
        r, b, v, d = self.r[index], self.rs[index], self.v[index], self.vs[index]
        a = r * h + 0.5 * b * h * h
        moment = (
            self.D[index] * a
            + 0.5 * r * v * h * h
            + (b * v + 0.5 * r * d) * h**3 / 3.0
            + b * d * h**4 / 8.0
        )
        return self.A[index] + a, self.D[index] + v * h + 0.5 * d * h * h, self.B[index] + moment

    def values(self, time):
        k = min(int(np.searchsorted(self.clock, time, side="right") - 1), len(self.clock) - 1)
        a, travel, b = self.primitive(k, time - self.clock[k])
        if self.periodic:
            hazard = np.broadcast_to(max(0.0, 2 * (travel * a - b)), (self.n,))
            plus = minus = np.broadcast_to(a, (self.n,))
        else:
            # The interior has the same unrestricted history everywhere.
            # Only sites already reached by a boundary need an inverse clock.
            hazard = np.full(self.n, max(0.0, 2 * (travel * a - b)))
            plus, minus = np.full(self.n, a), np.full(self.n, a)
            count = int(np.searchsorted(self.distance, travel, side="left"))
            if count:
                distance = self.distance[:count]
                z = travel - distance
                index = np.clip(np.searchsorted(self.D, z, side="right") - 1, 0, len(self.D) - 1)
                offset = z - self.D[index]
                root = np.sqrt(np.maximum(self.v[index] ** 2 + 2 * self.vs[index] * offset, 0.0))
                h = 2 * offset / (self.v[index] + root)
                az, _, bz = self.primitive(index, h)
                correction = (distance - travel) * az + bz
                hazard[:count] += correction
                hazard[-count:] += correction[::-1]
                plus[:count] = np.maximum(a - az, 0.0)
                minus[-count:] = np.maximum(a - az, 0.0)[::-1]
            hazard = np.maximum(hazard, 0.0)
        s = np.exp(-hazard)
        h = time - self.clock[k]
        birth = (self.r[k] + self.rs[k] * h) * s
        death = np.where(s > self.floor, 2 * (self.v[k] + self.vs[k] * h) * s * plus * minus, 0.0)
        return hazard, plus, minus, birth, death


def _fields(state, n, rate, velocity, floor):
    s = np.exp(-np.maximum(state[:n], 0.0))
    right, left = s * state[n : 2 * n], s * state[2 * n : 3 * n]
    return {
        "replicated_fraction_xt": -np.expm1(-np.maximum(state[:n], 0.0)),
        "right_fork_density_xt": right,
        "left_fork_density_xt": left,
        "fork_density_xt": right + left,
        "initiation_events_xt": rate * s,
        "coalescence_events_xt": np.where(
            s > floor, 2 * velocity * s * state[n : 2 * n] * state[2 * n : 3 * n], 0.0
        ),
    }


def _median_crossing(solver, before, start, end, speed, n):
    from scipy.optimize import brentq

    if isinstance(solver, _Stationary):
        return brentq(
            lambda t: float(np.mean(-np.expm1(-np.maximum(solver.values(t)[0], 0.0)))) - 0.5,
            start,
            end,
        )
    h = end - start
    low, high = before[:n], solver.state[:n]
    d0 = h * speed.at(start) * (before[n : 2 * n] + before[2 * n : 3 * n])
    d1 = h * speed.at(end) * (solver.state[n : 2 * n] + solver.state[2 * n : 3 * n])

    def residual(w):
        hazard = (
            (2 * w**3 - 3 * w * w + 1) * low
            + (w**3 - 2 * w * w + w) * d0
            + (-2 * w**3 + 3 * w * w) * high
            + (w**3 - w * w) * d1
        )
        return float(np.mean(-np.expm1(-np.clip(hazard, low, high)))) - 0.5

    return start + h * brentq(residual, 0.0, 1.0)


@initial_prediction
def kinetic_event_densities(
    firing,
    speed,
    *,
    unreplicated_floor=1e-12,
    cfl=0.8,
    time_multiplier=None,
    outputs=None,
    _complete=False,
    **options,
):
    """Compute requested observables in one pass, without stochastic simulations.

    ``time_multiplier`` sets the horizon to a multiple (>=1) of regional median
    replication time: the first time mean(f(x,t)) reaches 0.5. Explicit
    ``max_time`` takes priority. Otherwise a supplied clock is respected, or
    integration continues to completion. ``_complete`` is for efficiencies.
    Temporal profiles keep their original clock and hold their last value.
    """
    dx, dt = float(options.get("resolution", 1.0)), float(options.get("resolution_time", 5.0))
    floor, cfl = float(unreplicated_floor), float(cfl)
    if not np.isfinite(dx) or dx <= 0 or not np.isfinite(dt) or dt <= 0:
        raise ValueError("Spatial and temporal resolutions must be finite and positive.")
    if not np.isfinite(floor) or not 0 <= floor < 1:
        raise ValueError("Unreplicated floor must be finite and between zero and one.")
    if not np.isfinite(cfl) or not 0 < cfl <= 1:
        raise ValueError("CFL must be finite and between zero and one.")
    if time_multiplier is not None:
        time_multiplier = float(time_multiplier)
        if not np.isfinite(time_multiplier) or time_multiplier < 1:
            raise ValueError("Maximum time multiplier must be finite and at least one.")
    clock = options.get("time_grid")
    if clock is not None:
        clock = np.asarray(clock, dtype=float)
        if (
            clock.ndim != 1
            or not clock.size
            or np.any(~np.isfinite(clock))
            or clock[0] < 0
            or np.any(np.diff(clock) <= 0)
        ):
            raise ValueError(
                "Time grid must be a finite increasing vector starting at or after zero."
            )
    n = _resolve_n_bins_from_fields(
        firing,
        speed,
        n_bins=options.get("n_bins"),
        resolution_x=dx,
        genome_length=options.get("genome_length"),
        interval_kb=options.get("interval_kb"),
    )
    if n is None:
        raise ValueError("Provide a spatial profile or a region length.")
    n = int(n)
    rate, velocity = (
        _Input(firing, n, dt, clock, "Initiation rate"),
        _Input(speed, n, dt, clock, "Fork speed", True),
    )
    periodic = bool(options.get("perQ", False))
    requested = (
        set(outputs)
        if outputs is not None
        else {
            "initiation_events_xt",
            "coalescence_events_xt",
            "initiation_events",
            "coalescence_events",
        }
    )
    counts = bool(requested & {"initiation_events", "coalescence_events"})
    if rate.uniform and velocity.uniform:
        solver = _Uniform(n, dx, periodic, rate, velocity, floor)
    elif rate.static and velocity.static:
        solver = _Stationary(n, dx, periodic, rate, velocity, floor)
    else:
        solver = _Transport(n, dx, periodic, floor, cfl, counts)
    solver.counts = counts
    maximum = options.get("max_time")
    if maximum is None:
        maximum = options.get("max_rep_time")
    if maximum is not None and (not np.isfinite(maximum) or maximum <= 0):
        raise ValueError("Maximum time must be finite and positive.")
    explicit_clock = (
        clock if maximum is None and time_multiplier is None and not _complete else None
    )
    complete = _complete or maximum is None and time_multiplier is None and explicit_clock is None
    if _complete:
        maximum = None
    keys = requested & {
        "replicated_fraction_xt",
        "right_fork_density_xt",
        "left_fork_density_xt",
        "fork_density_xt",
        "initiation_events_xt",
        "coalescence_events_xt",
    }
    stored = {key: _Output(n) for key in keys}
    times, previous, median = [], 0.0, None
    baseline = np.zeros(2 * n)
    support_end = max(float(rate.clock[-1]), float(velocity.clock[-1]))
    pulse_end = support_end + n * dx / float(np.min(velocity.values))
    knots = np.unique(np.r_[rate.clock, velocity.clock])
    index = 0
    while True:
        check_cancelled()
        if explicit_clock is not None:
            if index >= len(explicit_clock):
                break
            end = float(explicit_clock[index])
        else:
            end = index * dt
            if maximum is not None:
                end = min(end, float(maximum))
        stops = np.r_[knots[(knots > previous) & (knots < end)], end]
        before = solver.state.copy() if median is None else None
        start = previous
        for stop in stops:
            if stop > start:
                solver.advance(start, float(stop), rate, velocity)
            start = float(stop)
        fields = _fields(solver.state, n, rate.at(end), velocity.at(end), floor)
        fraction = float(np.mean(fields["replicated_fraction_xt"]))
        if median is None and fraction >= 0.5:
            median = _median_crossing(solver, before, previous, end, velocity, n)
            if time_multiplier is not None and maximum is None and not _complete:
                maximum = time_multiplier * median
                if maximum < end:
                    solver.state = before
                    solver.advance(previous, maximum, rate, velocity)
                    end = maximum
                    fields = _fields(solver.state, n, rate.at(end), velocity.at(end), floor)
        times.append(end)
        for key in keys:
            stored[key].append(fields[key])
        if index == 0:
            baseline = solver.state[3 * n :].copy()
        if explicit_clock is not None:
            report_measured_work(end / max(float(explicit_clock[-1]), 1e-30))
        elif maximum is not None:
            report_measured_work(end / maximum)
        else:
            report_measured_work(
                min(0.95, fraction * (0.1 if time_multiplier is not None else 0.9))
            )
        if explicit_clock is None:
            if maximum is not None and end >= maximum:
                break
            if complete and index and np.max(1.0 - fields["replicated_fraction_xt"]) < 1e-9:
                break
            if index and not np.any(rate.values):
                break
            if end >= pulse_end and not np.any(rate.values[:, -1]):
                if time_multiplier is not None and median is None:
                    raise ValueError(
                        "The replicated fraction never reaches 50%; specify an explicit maximum time."
                    )
                break
        from ._resources import hosted

        if len(times) >= 100_000 or (hosted() and n * (len(times) + 1) * len(keys) > 64_000_000):
            raise ValueError(
                "The requested event output is too large; use a coarser output resolution."
            )
        previous, index = end, index + 1
    result = {key: values.result() for key, values in stored.items()}
    if "initiation_events" in requested:
        result["initiation_events"] = solver.state[3 * n : 4 * n].copy() - baseline[:n]
    if "coalescence_events" in requested:
        result["coalescence_events"] = solver.state[4 * n :].copy() - baseline[n:]
    result.update(
        time_grid=np.asarray(times), median_replication_time=median, observable_method=solver.method
    )
    return result
