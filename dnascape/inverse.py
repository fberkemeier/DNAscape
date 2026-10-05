"""Constrained effective kinetic fits with deterministic mean-field sensitivities.

Timing alone does not identify two unrestricted kinetic fields. These fits
therefore use bounded positive interpolation bases, smoothness regularization,
and an explicit anchor for the mean log fork speed in joint fits. They return
one compatible effective model, together with its residual and convergence
diagnostics; they do not claim unique biological kinetic identification.
"""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.optimize import OptimizeResult, least_squares

from .mapping import _interp_time_rows
from .utils import _report_progress, check_cancelled, etaferr

_FORMS = {"scalar": "constant", "x": "space", "t": "time", "xt": "spacetime"}


def _form(value):
    value = _FORMS.get(str(value), str(value))
    if value not in {"constant", "space", "time", "spacetime"}:
        raise ValueError("Kinetic forms must be constant, space, time, or spacetime.")
    return value


def _linear_basis(size, knots):
    count = min(int(knots), size)
    if count < 1:
        raise ValueError("Basis knot counts must be positive integers.")
    if count == 1:
        return np.ones((size, 1))
    return np.stack(
        [np.interp(np.arange(size), np.linspace(0, size - 1, count), row) for row in np.eye(count)],
        axis=1,
    )


@dataclass
class _Field:
    form: str
    bx: np.ndarray
    bt: np.ndarray
    initial: np.ndarray
    bounds: tuple
    offset: int
    fixed: object = None

    @property
    def count(self):
        return 0 if self.fixed is not None else self.bx.shape[1] * self.bt.shape[1]

    def evaluate(self, parameters):
        if self.fixed is not None:
            return self.fixed
        coefficients = parameters[self.offset : self.offset + self.count].reshape(
            self.bt.shape[1], self.bx.shape[1]
        )
        # A static field has one temporal basis coefficient. Evaluate that
        # profile only once: materializing its repeated time slices can cost
        # gigabytes for chromosome-wide timing inputs.
        bt = self.bt[:1] if self.bt.shape[1] == 1 else self.bt
        bx = self.bx[:1] if self.bx.shape[1] == 1 else self.bx
        values = np.exp(bt @ coefficients @ bx.T)
        return np.broadcast_to(values, (self.bt.shape[0], self.bx.shape[0]))

    def derivative(self, values, index, n_parameters):
        derivative = np.zeros((values.shape[1], n_parameters))
        if self.count:
            basis = (self.bt[index][None, :, None] * self.bx[:, None, :]).reshape(
                values.shape[1], self.count
            )
            derivative[:, self.offset : self.offset + self.count] = values[index, :, None] * basis
        return derivative

    def output(self, values):
        if self.form == "constant":
            return float(values[0, 0])
        if self.form == "space":
            return values[0].copy()
        if self.form == "time":
            return values[:, 0].reshape(1, -1).copy()
        return values.T.copy()


def _mean_field_timing(
    firing, speed, time_grid, resolution, perQ, fields=None, parameters=None,
    *, progress=None, use_native=True,
):
    """Match mapping.map_fork_densities_xt while propagating exact tangents.

    Only current states and their parameter derivatives are retained. Memory
    grows with positions times basis coefficients, not positions times times.
    The integer CFL subdivision is fixed within each differentiable interval.
    """
    n = firing.shape[1]
    p = 0 if parameters is None else len(parameters)
    from ._inverse_native import create_transport

    native = create_transport(n, p) if use_native else None
    f, right, left, timing = (np.zeros(n) for _ in range(4))
    df, dright, dleft, dtiming = (np.zeros((n, p)) for _ in range(4))
    static_firing = firing.strides[0] == 0
    static_speed = speed.strides[0] == 0
    i_static = np.ascontiguousarray(firing[0]) if static_firing else None
    v_static = np.ascontiguousarray(speed[0]) if static_speed else None
    max_velocity = float(np.max(v_static)) if static_speed else None
    di_static = fields[0].derivative(firing, 0, p) if p and static_firing else None
    dv_static = fields[1].derivative(speed, 0, p) if p and static_speed else None
    last_update = perf_counter()
    for ti in range(1, len(time_grid)):
        check_cancelled()
        interval = time_grid[ti] - time_grid[ti - 1]
        substeps = max(
            1, int(np.ceil(
                (max_velocity if static_speed else np.max(speed[ti - 1 : ti + 1]))
                * interval / (0.8 * resolution)
            ))
        )
        dt = interval / substeps
        i0, i1 = (i_static, i_static) if static_firing else (firing[ti - 1], firing[ti])
        v0, v1 = (v_static, v_static) if static_speed else (speed[ti - 1], speed[ti])
        di0 = di1 = dv0 = dv1 = None
        if p:
            di0, di1 = (di_static, di_static) if static_firing else (
                fields[0].derivative(firing, index, p) for index in (ti - 1, ti)
            )
            dv0, dv1 = (dv_static, dv_static) if static_speed else (
                fields[1].derivative(speed, index, p) for index in (ti - 1, ti)
            )
        now = perf_counter()
        if progress is not None and now - last_update >= .2:
            progress((ti - 1) / (len(time_grid) - 1))
            last_update = now
        if native is not None:
            native.advance(
                interval, substeps, resolution, perQ, i0, i1, v0, v1,
                di0, di1, dv0, dv1, f, right, left, timing, df, dright, dleft, dtiming,
            )
            continue
        old_f, old_df = f.copy(), df.copy()
        for step in range(substeps):
            weight = (step + 0.5) / substeps
            initiation = i_static if static_firing else (1 - weight) * i0 + weight * i1
            velocity = v_static if static_speed else (1 - weight) * v0 + weight * v1
            if p:
                di = di_static if static_firing else (1 - weight) * di0 + weight * di1
                dv = dv_static if static_speed else (1 - weight) * dv0 + weight * dv1
            rf, lf = velocity * right, -velocity * left
            rin, lout = np.roll(rf, 1), np.roll(lf, -1)
            if not perQ:
                rin[0], lout[-1] = 0, 0
            rt = np.maximum(right - dt * (rf - rin) / resolution, 0)
            lt = np.maximum(left - dt * (lout - lf) / resolution, 0)
            u = np.maximum(1 - f, 0)
            source = initiation * u
            denominator = np.maximum(u, 1e-12)
            raw_sink = 2 * velocity * rt * lt / denominator
            max_sink = source + np.minimum(rt, lt) / dt
            sink = np.minimum(raw_sink, max_sink)
            rn = np.maximum(rt + dt * (source - sink), 0)
            ln = np.maximum(lt + dt * (source - sink), 0)
            total = 0.5 * (rt + rn + lt + ln)
            rate = velocity * total
            if p:
                drf = dv * right[:, None] + velocity[:, None] * dright
                dlf = -dv * left[:, None] - velocity[:, None] * dleft
                drin, dlout = np.roll(drf, 1, axis=0), np.roll(dlf, -1, axis=0)
                if not perQ:
                    drin[0], dlout[-1] = 0, 0
                drt = (dright - dt * (drf - drin) / resolution) * (rt > 0)[:, None]
                dlt = (dleft - dt * (dlout - dlf) / resolution) * (lt > 0)[:, None]
                du = -df * (u > 0)[:, None]
                dsource = di * u[:, None] + initiation[:, None] * du
                dsink = (
                    2
                    * (
                        dv * (rt * lt)[:, None]
                        + velocity[:, None] * (drt * lt[:, None] + rt[:, None] * dlt)
                    )
                    / denominator[:, None]
                    - (raw_sink / denominator)[:, None] * du * (u > 1e-12)[:, None]
                )
                dmax_sink = dsource + np.where((rt <= lt)[:, None], drt, dlt) / dt
                dsink = np.where((raw_sink <= max_sink)[:, None], dsink, dmax_sink)
                drn = (drt + dt * (dsource - dsink)) * (rn > 0)[:, None]
                dln = (dlt + dt * (dsource - dsink)) * (ln > 0)[:, None]
                drate = dv * total[:, None] + 0.5 * velocity[:, None] * (drt + drn + dlt + dln)
                df += np.where((dt * rate < u)[:, None], dt * drate, du)
            f = np.clip(f + np.minimum(dt * rate, u), 0, 1)
            completed = f >= 1 - 1e-12
            right, left = rn, ln
            right[completed], left[completed] = 0, 0
            if p:
                dright, dleft = drn, dln
                dright[completed], dleft[completed], df[completed] = 0, 0, 0
        timing += interval * (1 - 0.5 * (old_f + f))
        if p:
            dtiming -= interval * 0.5 * (old_df + df)
    return timing, dtiming, f


def _compact_space_time_field(field, n_time, n, t, resolution_time, name, positive=False):
    """Validate varying coordinates once, then expose the full field as a view."""
    values = np.asarray(field, dtype=float)
    if values.size == 1:
        base = values.reshape(1, 1).copy()
    elif values.ndim == 1 and values.size == n:
        base = values.reshape(1, n).copy()
    elif values.ndim == 2 and values.shape[0] == 1 and values.shape[1] > 1:
        base = _interp_time_rows(values.reshape(-1, 1), t, resolution_time)
    elif values.ndim == 2 and values.shape == (n, 1):
        base = values.reshape(1, n).copy()
    elif values.ndim == 2 and values.shape[0] == n:
        base = _interp_time_rows(values.T, t, resolution_time)
    else:
        raise ValueError(
            f"{name} must be scalar, x-dependent, time-dependent, or space-time dependent."
        )
    if np.any(~np.isfinite(base)) or np.any(base < 0):
        raise ValueError(f"{name} must contain finite values >= 0.")
    if positive and np.any(base <= 0):
        raise ValueError(f"{name} must contain finite positive values.")
    return np.broadcast_to(base, (n_time, n))


def _make_field(
    form, fixed, initial, bounds, n, t, resolution_time, spatial_knots, temporal_knots, offset, name
):
    bx = _linear_basis(n, spatial_knots if form in {"space", "spacetime"} else 1)
    bt = _linear_basis(len(t), temporal_knots if form in {"time", "spacetime"} else 1)
    if fixed is not None:
        expanded = _compact_space_time_field(
            fixed, len(t), n, t, resolution_time, name, positive=name == "fork_speed"
        )
        return _Field(form, bx, bt, np.zeros(0), bounds, offset, expanded)
    expanded = _compact_space_time_field(
        initial, len(t), n, t, resolution_time, name, positive=True
    )
    # Project only axes that vary in the initialization. Both interpolation
    # bases form a partition of unity, so a constant input projects exactly
    # onto equal coefficients without a large repeated least-squares problem.
    base = expanded[
        :1 if expanded.strides[0] == 0 else len(t),
        :1 if expanded.strides[1] == 0 else n,
    ]
    logs = np.log(np.clip(base, *bounds))
    if logs.shape[0] == 1:
        coefficients = np.broadcast_to(logs, (bt.shape[1], logs.shape[1]))
    elif bt.shape[1] == 1:
        coefficients = np.mean(logs, axis=0, keepdims=True)
    else:
        coefficients = np.linalg.lstsq(bt, logs, rcond=None)[0]
    if coefficients.shape[1] == 1:
        coefficients = np.broadcast_to(coefficients, (bt.shape[1], bx.shape[1])).copy()
    elif bx.shape[1] == 1:
        coefficients = np.mean(coefficients, axis=1, keepdims=True)
    else:
        coefficients = np.linalg.lstsq(bx, coefficients.T, rcond=None)[0].T
    coefficients = np.clip(coefficients, np.log(bounds[0]), np.log(bounds[1]))
    return _Field(form, bx, bt, coefficients.reshape(-1), bounds, offset)


def fit_timing_kinetics(
    observed_timing,
    *,
    firing_form="space",
    speed_form="constant",
    firing_rate=None,
    fork_speed=None,
    initial_firing_rate=None,
    initial_fork_speed=1.4,
    resolution=1.0,
    resolution_time=2.0,
    time_grid=None,
    T0=0.0,
    perQ=False,
    maxiter=20,
    spatial_knots=12,
    temporal_knots=4,
    regularization=0.001,
    speed_anchor=0.1,
    firing_bounds=None,
    speed_bounds=(0.2, 6.0),
    tolerance=1e-5,
    verbose=False,
):
    """Fit one or both kinetic fields to timing under explicit constraints.

    A supplied ``firing_rate`` or ``fork_speed`` is held fixed. Forms select
    the fitted field parameterizations, never inferred from ambiguous shapes.
    ``speed_anchor`` anchors the mean log speed to the initial value only when
    both fields are inferred. ``fit_mse`` always excludes regularization.
    Timing is the mean-field survival integral over the supplied/generated
    time grid; ``unreplicated_at_end`` reports a possibly truncated horizon.
    """
    observed = np.asarray(observed_timing, dtype=float)
    if observed.ndim != 1 or not observed.size or not np.any(np.isfinite(observed)):
        raise ValueError("observed_timing must be a non-empty vector with finite observations.")
    resolution, resolution_time = float(resolution), float(resolution_time)
    if (
        not np.isfinite(resolution)
        or resolution <= 0
        or not np.isfinite(resolution_time)
        or resolution_time <= 0
    ):
        raise ValueError("Spatial and temporal resolutions must be finite and positive.")
    for name, value in (("regularization", regularization), ("speed_anchor", speed_anchor)):
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative.")
    if (
        not np.isfinite(maxiter)
        or int(maxiter) != maxiter
        or maxiter < 0
        or not np.isfinite(tolerance)
        or tolerance <= 0
    ):
        raise ValueError("maxiter must be nonnegative and tolerance must be finite and positive.")
    anchor = 0.0 if T0 is None else float(T0)
    if not np.isfinite(anchor):
        raise ValueError("T0 must be finite.")
    valid = np.isfinite(observed)
    relative = observed[valid] - anchor
    if np.any(relative <= 0):
        raise ValueError("Observed timing must be later than T0.")
    if time_grid is None:
        supplied_lengths = {
            array.shape[1]
            for value in (firing_rate, fork_speed)
            if value is not None
            for array in (np.asarray(value),)
            if array.ndim == 2 and array.shape[1] > 1
        }
        if len(supplied_lengths) > 1:
            raise ValueError("Supplied temporal kinetic fields must use the same time grid.")
        if supplied_lengths:
            t = np.arange(supplied_lengths.pop(), dtype=float) * resolution_time
        else:
            horizon = max(4 * float(np.max(relative)), 10 * resolution_time)
            t = np.arange(0, horizon + resolution_time, resolution_time)
    else:
        t = np.asarray(time_grid, dtype=float)
    if (
        t.ndim != 1
        or t.size < 2
        or not np.all(np.isfinite(t))
        or t[0] != 0
        or np.any(np.diff(t) <= 0)
    ):
        raise ValueError("time_grid must begin at zero and be finite and strictly increasing.")
    firing_form, speed_form = _form(firing_form), _form(speed_form)
    speed_bounds = (0.2, 6.0) if speed_bounds is None else tuple(map(float, speed_bounds))
    firing_bounds = (1e-9, 1.0) if firing_bounds is None else tuple(map(float, firing_bounds))
    for name, bounds in (("firing_bounds", firing_bounds), ("speed_bounds", speed_bounds)):
        if len(bounds) != 2 or not np.all(np.isfinite(bounds)) or not 0 < bounds[0] < bounds[1]:
            raise ValueError(f"{name} must contain finite positive increasing bounds.")
    if initial_fork_speed is None:
        initial_fork_speed = 1.4
        supplied_firing = np.asarray(firing_rate) if firing_rate is not None else None
        if (
            speed_form == "space" and fork_speed is None
            and supplied_firing is not None and supplied_firing.size == 1
        ):
            rate = float(supplied_firing.reshape(-1)[0])
            if np.isfinite(rate) and rate > 0:
                # The local infinite-domain estimate is only an initializer.
                # Use it for predominantly low-bound estimates, or when every
                # estimate exceeds the upper bound. A partially high-speed
                # seed can raise the global CFL cost without resolving the
                # remaining slow regions; retain the established start there.
                # Also retain the established start for
                # ordinary, compatible profiles. The full finite-domain fit,
                # grid and objective remain unchanged in either case.
                with np.errstate(over="ignore", divide="ignore", under="ignore"):
                    local_speed = np.pi / (4.0 * rate * relative**2)
                below = local_speed < speed_bounds[0]
                above = local_speed > speed_bounds[1]
                if np.mean(below) >= .5 or np.all(above):
                    initial_fork_speed = np.interp(
                        np.arange(observed.size), np.flatnonzero(valid),
                        np.clip(local_speed, *speed_bounds),
                    )
    speed_reference = float(np.median(np.asarray(initial_fork_speed)))
    _report_progress(.04, "Preparing kinetic fields...")
    if initial_firing_rate is None:
        initial_firing_rate = np.pi / (4 * speed_reference * np.median(relative) ** 2)
    first = _make_field(
        firing_form,
        firing_rate,
        initial_firing_rate,
        firing_bounds,
        observed.size,
        t,
        resolution_time,
        spatial_knots,
        temporal_knots,
        0,
        "firing_rate",
    )
    second = _make_field(
        speed_form,
        fork_speed,
        initial_fork_speed,
        speed_bounds,
        observed.size,
        t,
        resolution_time,
        spatial_knots,
        temporal_knots,
        first.count,
        "fork_speed",
    )
    fields = (first, second)
    initial = np.concatenate([field.initial for field in fields])
    if not initial.size:
        raise ValueError("At least one kinetic field must be fitted.")
    lower = np.concatenate([np.full(field.count, np.log(field.bounds[0])) for field in fields])
    upper = np.concatenate([np.full(field.count, np.log(field.bounds[1])) for field in fields])
    penalty_rows = []
    for field in fields:
        if not field.count:
            continue
        indices = (
            np.arange(field.count).reshape(field.bt.shape[1], field.bx.shape[1]) + field.offset
        )
        pairs = []
        for axis in (0, 1):
            if indices.shape[axis] > 1:
                pairs.extend(
                    zip(
                        np.take(indices, range(indices.shape[axis] - 1), axis=axis).ravel(),
                        np.take(indices, range(1, indices.shape[axis]), axis=axis).ravel(),
                    )
                )
        for left, right in pairs:
            row = np.zeros(initial.size)
            row[left], row[right] = -1, 1
            penalty_rows.append(np.sqrt(regularization / max(len(pairs), 1)) * row)
    penalty = np.asarray(penalty_rows).reshape(-1, initial.size)
    anchor_row = np.zeros((0, initial.size))
    anchor_target = np.zeros(0)
    if first.count and second.count and speed_anchor:
        anchor_row = np.zeros((1, initial.size))
        anchor_row[0, second.offset :] = np.sqrt(speed_anchor) / second.count
        anchor_target = anchor_row @ initial
    scale = max(float(np.median(relative)), 1.0) * np.sqrt(np.count_nonzero(valid))
    cache = {}
    history = []

    def evaluate(parameters):
        check_cancelled()
        if "parameters" in cache and np.array_equal(parameters, cache["parameters"]):
            return cache["residual"], cache["jacobian"]
        evaluation = len(history) + 1
        label = "Fitting fork speed" if firing_rate is not None else "Fitting kinetics"

        def report_step(fraction):
            _report_progress(
                .04 + .90 * (evaluation - 1 + fraction) / max(int(maxiter), 1),
                f"{label}... evaluation {evaluation}/{max(int(maxiter), 1)} ({fraction:.0%})",
            )

        report_step(0.)
        firing, speed = (field.evaluate(parameters) for field in fields)
        prediction, jacobian, completion = _mean_field_timing(
            firing, speed, t, resolution, perQ, fields, parameters, progress=report_step,
        )
        prediction += anchor
        residual = prediction[valid] - observed[valid]
        mse = float(np.mean(residual**2))
        history.append(mse)
        if verbose:
            etaferr(len(history), int(maxiter), started, mse, label="rmap constrained kinetics")
        full_residual = np.concatenate(
            [residual / scale, penalty @ parameters, anchor_row @ parameters - anchor_target]
        )
        full_jacobian = np.vstack([jacobian[valid] / scale, penalty, anchor_row])
        cache.update(
            parameters=parameters.copy(),
            residual=full_residual,
            jacobian=full_jacobian,
            prediction=prediction,
            completion=completion,
        )
        return full_residual, full_jacobian

    import time

    started = time.time()
    if maxiter == 0:
        solution = OptimizeResult(
            x=initial,
            success=False,
            nfev=1,
            message="Initialization only; optimization disabled by maxiter=0.",
        )
    else:
        solution = least_squares(
            lambda parameters: evaluate(parameters)[0],
            initial,
            jac=lambda parameters: evaluate(parameters)[1],
            bounds=(lower, upper),
            max_nfev=int(maxiter),
            ftol=tolerance,
            xtol=tolerance,
            gtol=tolerance,
            method="trf",
        )
    evaluate(solution.x)
    firing, speed = (field.evaluate(solution.x) for field in fields)
    prediction = cache["prediction"]
    return {
        "firing_rate": first.output(firing),
        "fork_speed": second.output(speed),
        "firing_form": firing_form,
        "speed_form": speed_form,
        "time_grid": t,
        "replication_timing_fit": prediction,
        "residual": prediction - observed,
        "fit_mse": float(np.mean((prediction[valid] - observed[valid]) ** 2)),
        "fit_history": np.asarray(history),
        "converged": bool(solution.success),
        "convergence_message": str(solution.message),
        "optimizer_evaluations": int(solution.nfev),
        "fit_objective": float(np.dot(cache["residual"], cache["residual"])),
        "unreplicated_at_end": float(np.max(1 - cache["completion"])),
        "regularization": float(regularization),
        "speed_anchor": float(speed_anchor),
        "identifiability": "Regularized effective kinetics; timing alone does not uniquely identify both fields.",
        "forward_model": "deterministic mean-field fork transport",
    }


def map_timing_joint_kinetics(
    observed_timing,
    firing_form="space",
    speed_form="constant",
    resolution=1.0,
    resolution_time=2.0,
    time_grid=None,
    T0=0.0,
    initial_firing_rate=None,
    initial_fork_speed=None,
    initial_kinetics=None,
    firing_bounds=None,
    speed_bounds=(0.2, 6.0),
    spatial_knots=12,
    temporal_knots=4,
    regularization=0.001,
    speed_anchor=0.1,
    maxiter=20,
    tolerance=1e-5,
    perQ=False,
    verbose=False,
):
    """Joint timing inverse for every pair of constant/x/t/xt kinetic forms."""
    if initial_kinetics is not None:
        if not isinstance(initial_kinetics, dict):
            raise ValueError("initial_kinetics must be a dictionary when supplied.")
        if initial_firing_rate is None:
            initial_firing_rate = initial_kinetics.get("firing_rate")
        if initial_fork_speed is None:
            initial_fork_speed = initial_kinetics.get("fork_speed")
    if initial_fork_speed is None:
        initial_fork_speed = 1.4
    return fit_timing_kinetics(
        observed_timing,
        firing_form=firing_form,
        speed_form=speed_form,
        resolution=resolution,
        resolution_time=resolution_time,
        time_grid=time_grid,
        T0=T0,
        initial_firing_rate=initial_firing_rate,
        initial_fork_speed=initial_fork_speed,
        firing_bounds=firing_bounds,
        speed_bounds=speed_bounds,
        spatial_knots=spatial_knots,
        temporal_knots=temporal_knots,
        regularization=regularization,
        speed_anchor=speed_anchor,
        maxiter=maxiter,
        tolerance=tolerance,
        perQ=perQ,
        verbose=verbose,
    )


def map_timing_and_firing_to_speed(
    source,
    target_form="space",
    target_shape="best",
    resolution=1.0,
    resolution_time=2.0,
    time_grid=None,
    T0=0.0,
    initial_fork_speed=1.4,
    speed_bounds=(0.2, 6.0),
    spatial_knots=12,
    temporal_knots=4,
    regularization=0.01,
    maxiter=20,
    tolerance=1e-4,
    perQ=False,
    verbose=False,
):
    """Fork-speed inverse with a fixed supplied firing field, including periodic genomes."""
    from .fitting import _RmapInfoArray, _RmapInfoFloat

    if target_shape != "best":
        raise ValueError("The constrained fork-speed route supports target_shape='best'.")
    info = fit_timing_kinetics(
        source["replication_timing"],
        firing_rate=source["firing_rate"],
        speed_form=target_form,
        resolution=resolution,
        resolution_time=resolution_time,
        time_grid=time_grid,
        T0=T0,
        initial_fork_speed=initial_fork_speed,
        speed_bounds=speed_bounds,
        spatial_knots=spatial_knots,
        temporal_knots=temporal_knots,
        regularization=regularization,
        maxiter=maxiter,
        tolerance=tolerance,
        perQ=perQ,
        verbose=verbose,
    )
    values = info.pop("fork_speed")
    info.pop("firing_rate", None)
    info["target_shape"] = "best"
    info["fork_speed_shape"] = {
        "constant": "constant v0",
        "space": "bounded spatial interpolation v(x)",
        "time": "bounded temporal interpolation v(t)",
        "spacetime": "bounded tensor interpolation v(x,t)",
    }[_form(target_form)]
    return (
        _RmapInfoFloat(values, info)
        if np.asarray(values).ndim == 0
        else _RmapInfoArray(values, info)
    )


__all__ = ["fit_timing_kinetics", "map_timing_joint_kinetics", "map_timing_and_firing_to_speed"]
