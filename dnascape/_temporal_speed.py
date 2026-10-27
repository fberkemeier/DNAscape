"""Cached survival predictions and exact tangents for a spatially uniform v(t).

Only observation sites are sampled. Every native initiation bin contributes
to the cone integral, including origins between sampled observation sites.
"""

import numpy as np

from .inverse import _compact_space_time_field
from .utils import _fit_progress_enabled, check_cancelled, etaferr, initial_prediction, report_measured_work


class TemporalSpeedPrediction:
    def __init__(self, firing, n_positions, positions, time_grid, resolution, resolution_time):
        self.positions = np.asarray(positions, dtype=int)
        self.t = np.asarray(time_grid, dtype=float)
        self.dx = float(resolution)
        self.n = int(n_positions)
        if self.n < 2:
            raise ValueError("Temporal cone fitting requires at least two positions.")
        if self.t.ndim != 1 or not self.t.size or np.any(~np.isfinite(self.t)) or self.t[0] < 0 or np.any(np.diff(self.t) <= 0):
            raise ValueError("time_grid must be finite, nonnegative and increasing.")
        if not np.isfinite(self.dx) or self.dx <= 0:
            raise ValueError("Spatial resolution must be finite and positive.")
        self.leading = self.t[0] > 0
        rates = _compact_space_time_field(firing, len(self.t), self.n, self.t, resolution_time, "initiation_rate")
        self.static = rates.strides[0] == 0 or np.asarray(firing).ndim < 2 or np.asarray(firing).shape[1] == 1
        if self.leading:
            self.t = np.insert(self.t, 0, 0.)
            if not self.static:
                rates = np.vstack((rates[:1], rates))
        self.dt = np.diff(self.t)
        row_count = 1 if self.static else max(0, len(rates) - 1)
        self.cdf = np.zeros((row_count, self.n))
        for first in range(0, row_count, 8):
            check_cancelled()
            last = min(first + 8, row_count)
            midpoint = rates[:1] if self.static else .5 * (rates[first:last] + rates[first + 1:last + 1])
            increments = .5 * (midpoint[:, :-1] + midpoint[:, 1:]) * self.dx
            np.cumsum(increments, axis=1, out=self.cdf[first:last, 1:])
        self.x = self.positions * self.dx

    def _mass(self, coordinates, rows, derivative):
        local = np.clip(coordinates / self.dx, 0., self.n - 1.)
        index = np.minimum(local.astype(np.intp), self.n - 2)
        row = 0 if self.static else rows[:, None]
        first, second = self.cdf[row, index], self.cdf[row, index + 1]
        result = first + (second - first) * (local - index)
        if not derivative:
            return result, None
        slope = (second - first) / self.dx
        slope[(coordinates < 0.) | (coordinates > (self.n - 1) * self.dx)] = 0.
        return result, slope

    @initial_prediction
    def predict(self, speed, derivative=None):
        """Return timing, its parameter Jacobian, and terminal survival.

        Differentiate the same piecewise-linear source CDF and midpoint cones
        used by map_replicated_fraction_xt, rather than finite-differencing
        several complete predictions for each optimizer proposal.
        """
        speed = np.asarray(speed, dtype=float).reshape(-1)
        if self.leading:
            speed = np.insert(speed, 0, speed[0])
        if speed.size != self.t.size or np.any(~np.isfinite(speed)) or np.any(speed <= 0):
            raise ValueError("Fork speed must be positive and match the time grid.")
        p = 0 if derivative is None else np.asarray(derivative).shape[1]
        travel = np.r_[0., np.cumsum(.5 * (speed[:-1] + speed[1:]) * self.dt)]
        midpoint = .5 * (travel[:-1] + travel[1:])
        if p:
            derivative = np.asarray(derivative, dtype=float)
            if self.leading:
                derivative = np.vstack((derivative[:1], derivative))
            tangent = np.vstack((np.zeros((1, p)), np.cumsum(
                .5 * (derivative[:-1] + derivative[1:]) * self.dt[:, None], axis=0)))
            midpoint_tangent = .5 * (tangent[:-1] + tangent[1:])
        timing = np.zeros(self.x.size)
        jacobian = np.zeros((self.x.size, p))
        previous = np.ones(self.x.size)
        previous_tangent = np.zeros_like(jacobian)
        total_work = max(1, self.t.size * (self.t.size - 1))
        for end in range(1, self.t.size):
            volume = np.zeros(self.x.size)
            volume_tangent = np.zeros_like(jacobian)
            # Bound temporary arrays independently of the original X-by-T grid.
            for first in range(0, end, 32):
                check_cancelled()
                report_measured_work((end * (end - 1) + 2 * first) / total_work)
                rows = np.arange(first, min(first + 32, end))
                radius = travel[end] - midpoint[rows]
                right, right_slope = self._mass(self.x[None, :] + radius[:, None], rows, bool(p))
                left, left_slope = self._mass(self.x[None, :] - radius[:, None], rows, bool(p))
                volume += np.sum(self.dt[rows, None] * np.maximum(right - left, 0.), axis=0)
                if p:
                    radius_tangent = tangent[end] - midpoint_tangent[rows]
                    volume_tangent += np.einsum("km,kp,k->mp", right_slope + left_slope,
                                                 radius_tangent, self.dt[rows])
            survival = np.exp(-volume)
            survival_tangent = -survival[:, None] * volume_tangent
            if not (self.leading and end == 1):
                timing += .5 * self.dt[end - 1] * (previous + survival)
                jacobian += .5 * self.dt[end - 1] * (previous_tangent + survival_tangent)
            previous, previous_tangent = survival, survival_tangent
        return timing, jacobian, previous


def fit_temporal_speed(timing, firing, *, resolution=1., resolution_time=5., time_grid=None,
                       T0=None, initial_fork_speed=1.4, speed_bounds=None, fit_step=.003,
                       maxiter=10, refine_edge_kb=1000., error=None, verbose=False, print_every=1):
    """Fit the sequential temporal phase with the spatial component held constant.

    Keep its guarded linear proposal, then refine scale and temporal slope
    using cached cone integrals. Compare linear and exponential time profiles
    on the same timing objective; retain only improvements over the supplied
    initial guess. No spatial fit or simulation is performed.
    """
    import time
    from scipy.optimize import minimize
    from . import fitting as fits

    observed = np.asarray(timing, dtype=float)
    rate = np.asarray(firing, dtype=float)
    if not np.any(rate > 0):
        raise ValueError("A positive initiation rate is required to infer fork speed.")
    if observed.size < 2:
        return fits._fit_exponential_speed_shape(observed, firing, "time", resolution=resolution,
            resolution_time=resolution_time, time_axis=time_grid, T0=T0,
            initial_fork_speed=initial_fork_speed, speed_bounds=speed_bounds, fit_step=fit_step,
            maxiter=maxiter, refine_edge_kb=refine_edge_kb, error=error, refine_verbose=verbose)
    offset = 0. if T0 is None else float(T0)
    if not np.isfinite(offset):
        raise ValueError("T0 must be finite when supplied.")
    relative = observed - offset
    sites, _ = fits._speed_fit_indices(relative, resolution, refine_edge_kb=refine_edge_kb, max_fit_positions=1500)
    scale = fits._initial_positive_speed_value(initial_fork_speed)
    low, high = fits._direct_speed_shape_bounds([scale], speed_bounds=speed_bounds)
    scale = float(np.clip(scale, low, high))
    maxiter = max(0, int(maxiter))
    fit_step = float(fit_step)
    if not np.isfinite(fit_step) or fit_step < 0:
        raise ValueError("fit_step_t must be finite and >= 0.")
    error = fits._normalise_error_threshold(error)
    static = rate.ndim < 2 or rate.shape[1] == 1
    automatic = time_grid is None and static
    axis = (fits._survival_time_axis_from_timing(relative, resolution_time=resolution_time,
            min_max_time=0., tail_multiplier=4.) if automatic else
            fits._time_axis_for_speed_fit(observed, firing, resolution_time=resolution_time,
                                         time_grid=time_grid, T0=T0))
    axis = np.asarray(axis, dtype=float)
    time_scale = max(float(np.max(relative[sites])), float(resolution_time))
    initial = np.full(axis.size, scale)
    candidate_initial = np.asarray(initial_fork_speed, dtype=float) if initial_fork_speed is not None else np.asarray(scale)
    if candidate_initial.size == axis.size:
        initial = np.clip(candidate_initial.reshape(-1), low, high)
    models = {}

    def model(clock, positions=sites):
        key = (len(clock), positions is sites)
        if key not in models:
            check_cancelled()
            if (not static and len(clock) * observed.size > 64_000_000) or len(clock) > 100_000:
                raise ValueError("The computed time domain is too large. Increase the time resolution.")
            # Automatic extensions use static initiation, so each cached source
            # CDF remains O(X), rather than allocating another X-by-T field.
            models[key] = TemporalSpeedPrediction(firing, observed.size, positions, clock, resolution, resolution_time)
        return models[key]

    def run_speed(make_field, positions=sites):
        clock = axis
        for extension in range(9):
            speed, tangent = make_field(clock)
            predicted, jacobian, survival = model(clock, positions).predict(speed, tangent)
            if not automatic:
                break
            total_rate = model(clock, positions).cdf[0, -1]
            future_bound = (observed.size - 1) * resolution / low + 1. / max(total_rate, 1e-300) + 2. * resolution_time
            if np.max(survival) * future_bound <= 1e-8 * max(1., np.max(relative[sites])):
                break
            if extension == 8:
                raise ValueError("The timing prediction has not completed. Check initiation rate and speed bounds.")
            clock = np.arange(0., max(clock[-1] * 2., resolution_time) + resolution_time, resolution_time)
        return predicted + offset, jacobian, clock

    budget = max(1, 3 * maxiter + 1)
    evaluations = 0
    completed_work = 0
    history = []
    best_score = np.inf
    started = time.time()

    def publish(mse, *, advance=True):
        nonlocal evaluations, best_score, completed_work
        evaluations += 1
        best_score = min(best_score, float(mse))
        history.append(best_score)
        if advance:
            completed_work += 1
        if advance and _fit_progress_enabled(verbose) and completed_work % max(1, int(print_every)) == 0:
            etaferr(min(completed_work, budget - 1), budget, started, best_score, done=False, label="Mapping")

    def seed_prediction(q):
        return run_speed(lambda clock: (np.interp(clock, axis, np.clip(scale * q, low, high)), None))[0]

    seed = fits._fit_linear_increasing_temporal_factor_for_separable_speed(
        observed, firing, np.full(observed.size, scale), resolution=resolution,
        resolution_time=resolution_time, time_grid=axis, T0=offset,
        initial_temporal_factor=initial / scale, fit_step=fit_step, maxiter=maxiter,
        speed_bounds=(low, high), refine_edge_kb=refine_edge_kb, error=error,
        verbose=False, predict_timing=seed_prediction, on_evaluation=publish, prediction_budget=budget,
        fit_positions=sites)
    seed_speed = scale * np.asarray(seed).reshape(-1)
    best = dict(mse=float(seed._rmap_info["fit_mse"]), family="initial", params=None,
                speed=seed_speed, axis=axis)
    if seed._rmap_info["fork_speed_temporal_accepted"]:
        best["family"] = "linear"

    def speed_field(params, family, clock):
        normalized = (clock - axis[0]) / time_scale
        v0 = np.exp(params[0])
        if family == "linear":
            raw = v0 * (1. + params[1] * normalized)
            speed = np.clip(raw, low, high)
            tangent = np.column_stack((raw, v0 * normalized))
            tangent[(raw < low) | (raw > high)] = 0.
        else:
            log_speed = params[0] + params[1] * normalized
            speed = np.exp(np.clip(log_speed, np.log(low), np.log(high)))
            tangent = np.column_stack((speed, speed * normalized))
            tangent[(log_speed < np.log(low)) | (log_speed > np.log(high))] = 0.
        return speed, tangent

    def predict(params, family, positions=sites):
        return run_speed(lambda clock: speed_field(params, family, clock), positions)

    # Refining two temporal coefficients needs one prediction and its exact
    # Jacobian per proposal, rather than five extra finite-difference solves.
    if maxiter > 0 and fit_step > 0:
        for family in ("linear", "exponential"):
            if best["mse"] < 1e-12 or fits._error_reached(best["mse"], error):
                break
            normalized = (axis - axis[0]) / time_scale
            reference = seed_speed if best["params"] is None else speed_field(best["params"], best["family"], axis)[0]
            log_seed = np.log(np.maximum(reference, low))
            active = normalized <= 1.
            if axis.size > 1:
                dependent = reference / reference[0] - 1. if family == "linear" else log_seed - log_seed[0]
                slope = float(np.dot(normalized[active], dependent[active]) / max(np.dot(normalized[active], normalized[active]), 1e-30))
            else:
                slope = 0.
            initial_parameters = np.array([log_seed[0], np.clip(slope, -8., 8.)])
            cache = {}

            def evaluate(parameters):
                check_cancelled()
                if "params" in cache and np.array_equal(cache["params"], parameters):
                    return cache["residual"], cache["jacobian"]
                predicted, jacobian, clock = predict(parameters, family)
                residual = predicted - observed[sites]
                mse = float(np.mean(residual**2))
                if mse < best["mse"]:
                    best.update(mse=mse, family=family, params=parameters.copy(), axis=clock)
                cache.update(params=parameters.copy(), residual=residual, jacobian=jacobian)
                publish(mse, advance=False)
                return residual, jacobian

            def objective(parameters):
                result, jacobian = evaluate(parameters)
                if best["mse"] < 1e-12 or fits._error_reached(best["mse"], error):
                    raise fits._FitConverged
                return .5 * float(np.dot(result, result)), jacobian.T @ result

            def iteration_complete(_parameters):
                nonlocal completed_work
                completed_work += 1
                if _fit_progress_enabled(verbose):
                    etaferr(min(completed_work, budget - 1), budget, started, best_score, done=False, label="Mapping")

            try:
                minimize(objective, initial_parameters, jac=True, method="L-BFGS-B",
                    bounds=((np.log(low), np.log(high)), (-8., 8.)), callback=iteration_complete,
                    options={"maxiter": maxiter, "maxfun": 10 * (maxiter + 1), "maxls": 10, "ftol": 1e-10})
            except fits._FitConverged:
                break
    if best["params"] is None:
        initial_field = lambda clock: (np.interp(clock, axis, seed_speed), None)
        predicted, _, output_axis = run_speed(initial_field)
        speed = np.interp(output_axis, axis, seed_speed)
    else:
        predicted, _, output_axis = predict(best["params"], best["family"])
        speed = speed_field(best["params"], best["family"], output_axis)[0]
    interpolated = not fits._use_exact_full_timing_diagnostic(observed.size, len(output_axis))
    if interpolated:
        full_prediction = fits._expand_sampled_timing_diagnostic(predicted, sites, observed.size)
    elif best["params"] is None:
        full_prediction = run_speed(initial_field, np.arange(observed.size))[0]
    else:
        full_prediction = predict(best["params"], best["family"], np.arange(observed.size))[0]
    if _fit_progress_enabled(verbose):
        etaferr(budget, budget, started, best["mse"], done=True, label="Mapping")
    family = best["family"]
    if family == "initial" and np.all(speed == speed[0]):
        family = "constant"
    component = {"exponential": "v0 * exp(b_t * T)", "linear": "v0 * (1 + b_t * T)",
                 "constant": "v0", "initial": "initial v(t)"}[family]
    return fits._RmapInfoArray(speed[None, :], info=dict(
        target_shape="best", speed_form="time", time_grid=output_axis,
        fork_speed_shape=f"{family} v(t)", fork_speed_component_t=component,
        fork_speed_v0=float(speed[0]), fork_speed_time_origin=float(axis[0]), fork_speed_time_scale=time_scale,
        fork_speed_bounds=(low, high), fit_mse=best["mse"], fit_history=np.asarray(history),
        replication_timing_fit=full_prediction, residual=full_prediction - observed,
        replication_timing_fit_interpolated=interpolated, optimizer_evaluations=evaluations,
        fit_progress_total=budget,
        temporal_fit_method="guarded temporal component; spatial component held constant"))
