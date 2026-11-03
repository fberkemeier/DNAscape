"""One deterministic inverse problem for initiation rate and fork speed.

The fitted object is a positive kinetic field.  The observation operator is
either replicated fraction or its survival integral, and both use exactly the
same discretised cumulative replication hazard and its reverse derivative.
Spatial and temporal iterations count accepted/rejected block proposals, never
line-search evaluations.  Temporal fields use a shifted Weibull hazard basis;
they are effective kinetic models, rather than uniquely identified histories.
"""

from __future__ import annotations

import time
from copy import copy
from dataclasses import dataclass

import numpy as np
from scipy.special import gamma

from ._fit_bounds import fit_bounds
from ._fit_constraints import current_fit_constraint, fit_free_mask, has_fit_constraint, project_fit
from ._kinetic_tail import eventual_sources
from .inverse import _compact_space_time_field
from .utils import (
    _fit_progress_enabled,
    check_cancelled,
    etaferr,
    measured_work,
    measured_work_phase,
)

_FORMS = {
    "c": "scalar",
    "constant": "scalar",
    "scalar": "scalar",
    "x": "x",
    "space": "x",
    "array": "x",
    "t": "t",
    "time": "t",
    "transposed array": "t",
    "xt": "xt",
    "spacetime": "xt",
    "matrix": "xt",
}


def _integer(value, name):
    if not np.isfinite(value) or int(value) != value or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return int(value)


def _trapz_weights(clock):
    weights = np.zeros(len(clock))
    dt = np.diff(clock)
    weights[:-1] += 0.5 * dt
    weights[1:] += 0.5 * dt
    return weights


def _spatial_size(field):
    values = np.asarray(field)
    if values.ndim == 1 and values.size > 1:
        return values.size
    if values.ndim == 2 and values.shape[0] > 1:
        return values.shape[0]
    return None


def _uniform_space(field):
    values = np.asarray(field)
    return values.size == 1 or values.ndim == 2 and values.shape[0] == 1


def _supplied_on_clock(field, clock, native_clock, n, resolution_time, name, positive=False):
    """Hold the last supplied value after its native (possibly irregular) clock."""
    values = np.asarray(field, dtype=float)
    temporal = values.ndim == 2 and values.shape[1] > 1
    if not temporal:
        return _compact_space_time_field(
            field, len(clock), n, clock, resolution_time, name, positive=positive
        )
    rows = values.T
    if values.shape[0] not in {1, n}:
        raise ValueError(f"{name} must match the selected genomic region.")
    if np.any(~np.isfinite(values)) or np.any(values < 0) or positive and np.any(values <= 0):
        raise ValueError(
            f"{name} must contain finite {'positive' if positive else 'non-negative'} values."
        )
    source_time = (
        native_clock if len(native_clock) == len(rows) else np.arange(len(rows)) * resolution_time
    )
    if np.array_equal(clock, source_time):
        return np.broadcast_to(rows, (len(clock), n))
    lower = np.clip(np.searchsorted(source_time, clock, side="right") - 1, 0, len(rows) - 1)
    upper = np.minimum(lower + 1, len(rows) - 1)
    denominator = np.maximum(source_time[upper] - source_time[lower], 1e-300)
    weight = np.clip((clock - source_time[lower]) / denominator, 0.0, 1.0)
    interpolated = (1.0 - weight[:, None]) * rows[lower] + weight[:, None] * rows[upper]
    return np.broadcast_to(interpolated, (len(clock), n))


class _StaticHazardAdapter:
    """Expose the stationary-cell primitive through the full-field interface."""

    def __init__(self, speed, clock, resolution, periodic, radius):
        from ._hazard_static import StationaryHazard

        self.shape = speed.shape
        self.operator = StationaryHazard(speed[0], clock, resolution, periodic, radius)
        self.backend = self.operator.backend

    def forward(self, rate, record=False):
        return self.operator.forward(rate[0], record=record)

    def backward(self, weights, with_speed=True):
        rate_x, speed_x = self.operator.backward(weights, with_speed=with_speed)
        return rate_x[None, :], speed_x[None, :] if with_speed else None


@dataclass
class _KineticModel:
    target: str
    form: str
    n: int
    clock: np.ndarray
    reference: float
    delay: np.ndarray
    shape: float
    fit_shape: bool
    bounds: tuple
    projection: object = None
    onset: float = 1.0

    def __post_init__(self):
        self.temporal = self.form in {"t", "xt"}
        self.spatial = self.form in {"x", "xt"}
        self.spatial_count = self.n if self.spatial else (1 if self.form == "scalar" else 0)
        self.amplitude_index = self.spatial_count if self.temporal else None
        count = self.spatial_count + int(self.temporal)
        self.shape_index = count if self.temporal and self.fit_shape else None
        self.shape_count = (
            (self.n if self.form == "xt" else 1) if self.shape_index is not None else 0
        )
        self.shape_slice = slice(count, count + self.shape_count)
        count += self.shape_count
        self.baseline_index = count if self.temporal and self.target == "fork_speed" else None
        count += int(self.baseline_index is not None)
        self.count = count
        self.lower = np.full(count, -60.0)
        self.upper = np.full(count, 60.0)
        if self.shape_index is not None:
            self.lower[self.shape_slice], self.upper[self.shape_slice] = np.log(0.25), np.log(8.0)
        if self.baseline_index is not None:
            self.lower[self.baseline_index], self.upper[self.baseline_index] = -12.0, 12.0

    def basis(self, parameters, *, derivatives=True):
        shape = (
            np.asarray([self.shape])
            if self.shape_index is None
            else np.exp(parameters[self.shape_slice])
        )
        columns = max(self.delay.size, shape.size)
        age = np.broadcast_to(self.clock[:, None] - self.delay[None, :], (len(self.clock), columns))
        positive = age >= 0.0
        # A fixed physical onset cutoff makes shape derivatives finite at
        # t=delay, without changing the fitted curve when dt is changed.
        normalized = np.maximum(age, self.onset) / self.reference
        shapes = np.broadcast_to(shape[None, :], normalized.shape)
        basis = np.zeros_like(normalized)
        derivative = np.zeros_like(normalized) if derivatives else None
        if derivatives:
            basis[positive] = shapes[positive] * normalized[positive] ** (shapes[positive] - 1.0)
        else:
            # Prediction-only blocks need no masked copies of every XT
            # operand and no logarithms for a derivative they will not use.
            np.power(normalized, shapes - 1., out=basis)
            basis *= shapes
            np.copyto(basis, 0., where=~positive)
        if derivatives:
            derivative[positive] = basis[positive] * (
                1.0 + shapes[positive] * np.log(normalized[positive])
            )
        return basis, derivative

    def initial(self, initial, typical_timing, supplied):
        parameters = np.zeros(self.count)
        if self.shape_index is not None:
            parameters[self.shape_slice] = np.log(self.shape)
        if initial is None:
            if self.target == "fork_speed":
                scale = np.full(self.n, 1.4)
            else:
                supplied_speed = np.asarray(supplied)
                velocity = (
                    supplied_speed
                    if supplied_speed.ndim == 1 and supplied_speed.size == self.n
                    else np.median(supplied_speed, axis=1)
                    if supplied_speed.ndim == 2 and supplied_speed.shape[0] == self.n
                    else float(np.median(supplied_speed))
                )
                timing = np.broadcast_to(np.asarray(typical_timing), (self.n,))
                if self.temporal:
                    scale = (
                        (self.shape + 1.0) * self.reference ** (self.shape - 1.0) / (2.0 * velocity)
                    ) * (gamma(1.0 + 1.0 / (self.shape + 1.0)) / np.maximum(timing, 1e-9)) ** (
                        self.shape + 1.0
                    )
                else:
                    scale = np.pi / (4.0 * velocity * np.maximum(timing, 1e-9) ** 2)
        else:
            raw = np.asarray(initial, dtype=float)
            if np.any(~np.isfinite(raw)) or np.any(raw <= 0):
                raise ValueError("Initial fitted values must be finite and positive.")
            if raw.size == 1:
                scale = np.full(self.n, float(raw.reshape(-1)[0]))
            elif raw.ndim == 1 and raw.size == self.n:
                scale = raw.copy()
            elif raw.ndim == 2 and raw.shape[0] == self.n:
                scale = np.mean(raw, axis=1)
            elif raw.ndim == 2 and raw.shape[0] == 1:
                scale = np.full(self.n, float(np.mean(raw)))
            else:
                raise ValueError("The initial fitted profile does not match the genomic region.")
        lower_seed = max(self.bounds[0] * (1.0 + 1e-8), 1e-20)
        upper_seed = self.bounds[1] * (1.0 - 1e-8) if np.isfinite(self.bounds[1]) else np.inf
        scale = np.clip(scale, lower_seed, upper_seed)
        if self.target == "fork_speed" and self.temporal:
            # A positive basal speed plus a Weibull contribution permits both
            # the initial speed and the temporal trend to be fitted.
            parameters[self.baseline_index] = np.log(1.0)
            active = self.clock <= max(float(np.median(typical_timing)), self.clock[1])
            # The initial shape is shared, so its mean can be seeded without
            # expanding an entire chromosome's space-time basis.
            total, count = 0.0, 0
            width = max(1, 1_000_000 // max(1, int(active.sum())))
            for first in range(0, self.delay.size, width):
                age = self.clock[active, None] - self.delay[None, first:first + width]
                basis = self.shape * (np.maximum(age, self.onset) / self.reference) ** (self.shape - 1.)
                total += float(np.where(age >= 0., basis, 0.).sum())
                count += basis.size
            scale = scale / max(1.0 + total / count, 1e-12)
        scale = np.maximum(scale, 1e-20)
        logs = np.log(scale)
        if self.form == "xt":
            parameters[: self.n] = logs - np.mean(logs)
            parameters[self.amplitude_index] = np.mean(logs)
        elif self.form == "t":
            parameters[self.amplitude_index] = np.mean(logs)
        elif self.form == "x":
            parameters[: self.n] = logs
        else:
            parameters[0] = np.mean(logs)
        if self.shape_index is not None:
            parameters[self.shape_slice] = np.log(self.shape)
        return np.clip(parameters, self.lower, self.upper)

    def public(self, field):
        if self.form == "scalar":
            return np.asarray(field[0, 0])
        if self.form == "x":
            return np.asarray(field[0]).copy()
        if self.form == "t":
            return np.asarray(field[:, :1].T).copy()
        return np.asarray(field.T).copy()

    def _public_grid(self, values):
        values = np.asarray(values)
        if self.form == "scalar":
            return np.broadcast_to(values.reshape(1, 1), (len(self.clock), self.n))
        if self.form == "x":
            return np.broadcast_to(values.reshape(1, self.n), (len(self.clock), self.n))
        if self.form == "t":
            return np.broadcast_to(values.reshape(len(self.clock), 1), (len(self.clock), self.n))
        return values.T

    def evaluate(self, parameters, *, need_state=True):
        if self.form in {"x", "xt"}:
            logs = parameters[: self.n]
            if self.form == "xt":
                logs = logs - np.mean(logs)
        elif self.form == "scalar":
            logs = parameters[:1]
        else:
            logs = np.zeros(1)
        if self.temporal:
            logs = logs + parameters[self.amplitude_index]
        amplitude = np.exp(np.clip(logs, -60.0, 60.0))
        basis = basis_gradient = None
        if self.temporal:
            basis, basis_gradient = self.basis(parameters, derivatives=need_state)
            baseline = (
                np.exp(parameters[self.baseline_index]) if self.baseline_index is not None else 0.0
            )
            raw = amplitude[None, :] * (basis + baseline)
        else:
            raw = amplitude[None, :]
        bounded = np.clip(raw, *self.bounds)
        free = (raw >= self.bounds[0]) & (raw <= self.bounds[1])
        values = np.broadcast_to(bounded, (len(self.clock), self.n))
        if self.projection is not None or has_fit_constraint(self.target):
            public = self.public(values)
            if self.projection is None:
                projected = project_fit(public, self.target, self.form)
                free_public = fit_free_mask(public, self.target, self.form)
            else:
                projected = self.projection.project(public)
                free_public = ~self.projection.mask.reshape(np.asarray(public).shape)
            values = self._public_grid(projected)
            if free_public is not None:
                if not self.temporal:
                    free = free & np.asarray(free_public).reshape(1, -1)
                else:
                    free = np.broadcast_to(free, values.shape) & self._public_grid(free_public)
        return values, (raw, amplitude, basis, basis_gradient, free) if need_state else None

    def gradient(self, field_gradient, parameters, state):
        raw, amplitude, basis, basis_gradient, free = state
        if not self.temporal:
            # Stationary parameters share one value across all time samples.
            # Sum their adjoint first; do not expand static masks/amplitudes
            # into full space-time multiplication temporaries.
            local = np.sum(field_gradient, axis=0) * raw[0] * free[0]
            gradient = np.zeros(self.count)
            if self.form == "x":
                gradient[: self.n] = local
            else:
                gradient[0] = np.sum(local)
            return gradient
        q = field_gradient * np.broadcast_to(free, field_gradient.shape)
        local = np.sum(q * np.broadcast_to(raw, field_gradient.shape), axis=0)
        gradient = np.zeros(self.count)
        if self.form == "xt":
            gradient[: self.n] = local - np.mean(local)
        elif self.form == "x":
            gradient[: self.n] = local
        elif self.form == "scalar":
            gradient[0] = np.sum(local)
        if self.temporal:
            gradient[self.amplitude_index] = np.sum(local)
        if self.shape_index is not None:
            shape_gradient = np.sum(q * amplitude[None, :] * basis_gradient, axis=0)
            gradient[self.shape_slice] = (
                shape_gradient if self.shape_count > 1 else np.sum(shape_gradient)
            )
        if self.baseline_index is not None:
            gradient[self.baseline_index] = np.sum(
                q * amplitude[None, :] * np.exp(parameters[self.baseline_index])
            )
        return gradient

    def block(self, phase):
        if phase == "space":
            return np.arange(self.spatial_count)
        return np.arange(self.spatial_count, self.count)


def _schedule(form, space, temporal, order):
    if form in {"scalar", "x"}:
        return ["space"] * space
    if form == "t":
        return ["time"] * temporal
    order = str(order).replace("-", "_")
    if order in {"cross", "crossed", "alternating", "mixed", "joint"}:
        return [
            phase
            for index in range(max(space, temporal))
            for phase, count in (("space", space), ("time", temporal))
            if index < count
        ]
    if order in {"space_then_time", "space_time"}:
        return ["space"] * space + ["time"] * temporal
    if order in {"time_then_space", "time_space"}:
        return ["time"] * temporal + ["space"] * space
    raise ValueError("Fit order must be crossed, space_then_time, or time_then_space.")


def _lbfgs_direction(gradient, history):
    q = gradient.copy()
    alphas = []
    for s, y, rho in reversed(history):
        alpha = rho * np.dot(s, q)
        q -= alpha * y
        alphas.append(alpha)
    if history:
        s, y, _ = history[-1]
        scale = np.dot(s, y) / max(np.dot(y, y), 1e-30)
    else:
        scale = 1.0 / max(float(np.max(np.abs(q))), 1e-12)
    direction = scale * q
    for (s, y, rho), alpha in zip(history, reversed(alphas)):
        direction += s * (alpha - rho * np.dot(y, direction))
    return -direction


def fit_kinetic(
    observed,
    supplied,
    *,
    target,
    observation="timing",
    target_form="x",
    resolution=1.0,
    resolution_time=5.0,
    time_grid=None,
    n_bins=None,
    perQ=False,
    neighbourhood_range=2000.0,
    space_maxiter=10,
    temporal_maxiter=10,
    space_fit_step=None,
    temporal_fit_step=None,
    spacetime_order="crossed",
    rate_bounds=None,
    speed_bounds=None,
    initial_initiation_rate=None,
    initial_fork_speed=1.4,
    weibull_shape=2.0,
    weibull_delay=0.0,
    fit_weibull_shape=True,
    alpha_smooth=0.001,
    alpha_prior=0.0,
    regularization=None,
    T0=None,
    error=None,
    verbose=False,
    backend="auto",
    fit_projection=None,
    maxiter=None,
    fit_step=None,
    floor=0.0,
    err_threshold=None,
    max_time=None,
    max_rep_time=None,
    genome_length=None,
    interval_kb=None,
    refine_edge_kb=0.0,
    print_every=1,
    weibull_reference=None,
    weibull_onset=1.0,
    adaptive_fitting=True,
    relative_tolerance=0.001,
    _warm_start=None,
    _progress_offset=0,
    _progress_budget=None,
    _started=None,
    _warmup=False,
    _progress_tracker=None,
    **compatibility,
):
    """Fit one kinetic field to timing or replicated-fraction observations.

    ``supplied`` is the other kinetic field, in the usual public DNAscape
    scalar/x/t/xt representation.  Initiation is a density per kb per minute;
    speed is in kb per minute.  Iteration counts refer to block updates, with
    rejected line searches leaving the best field untouched.  Extra legacy
    numerical/simulation controls are accepted by adapters but do not change
    this deterministic no-stalling model.
    """
    check_cancelled()
    if _progress_tracker is None:
        from ._kinetic_progress import FitProgress

        progress_form = _FORMS.get(str(target_form).lower(), "x")
        progress_space = _integer(
            maxiter if maxiter is not None else space_maxiter, "Spatial iterations"
        )
        progress_time = _integer(
            maxiter if maxiter is not None else temporal_maxiter, "Time iterations"
        )
        _progress_tracker = FitProgress(
            len(_schedule(progress_form, progress_space, progress_time, spacetime_order))
        )
    if (
        adaptive_fitting
        and target in {"initiation_rate", "fork_speed"}
        and str(target_form).lower() in _FORMS
        and not alpha_prior
        and observation in {"timing", "replication_timing", "fraction", "replicated_fraction_xt"}
        and np.asarray(observed).ndim in {1, 2}
        and len(observed) >= 2048
        and (str(target_form) in {"x", "space", "xt", "spacetime"} or _spatial_size(supplied))
        and fit_projection is None
        and not has_fit_constraint(target)
        and backend == "auto"
    ):
        from ._kinetic_multigrid import adaptive_fit

        adaptive = adaptive_fit(observed, supplied, dict(locals()))
        if adaptive is not None:
            return adaptive
    del compatibility
    from ._hazard import HazardOperator
    from ._hazard_fft import ConstantSpeedConvolution
    from ._hazard_stationary_source import StationarySourceHazard
    from ._hazard_transport import FastTransportHazard
    from ._hazard_uniform import UniformKineticsHazard
    from .fitting import _RmapInfoArray, _RmapInfoFloat

    if target not in {"initiation_rate", "fork_speed"}:
        raise ValueError("The fitted target must be initiation rate or fork speed.")
    form = _FORMS.get(str(target_form).lower())
    if form is None:
        raise ValueError("Target dependence must be constant, space, time, or spacetime.")
    if observation in {"replication_timing", "timing"}:
        observation = "timing"
    elif observation in {"replicated_fraction", "replicated_fraction_xt", "fraction"}:
        observation = "fraction"
    else:
        raise ValueError("Observations must be timing or replicated fraction.")
    dx, dt = float(resolution), float(resolution_time)
    relative_tolerance = float(relative_tolerance)
    if not np.isfinite(relative_tolerance) or relative_tolerance < 0:
        raise ValueError("Fit tolerance must be finite and non-negative.")
    radius = float(neighbourhood_range)
    if not np.isfinite(dx) or dx <= 0 or not np.isfinite(dt) or dt <= 0:
        raise ValueError("Spatial and temporal resolutions must be finite and positive.")
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("Influence radius must be finite and positive.")
    if not isinstance(perQ, (bool, np.bool_)):
        raise ValueError("Periodic must be true or false.")
    if maxiter is not None:
        space_maxiter = temporal_maxiter = maxiter
    space_maxiter = _integer(space_maxiter, "Spatial iterations")
    temporal_maxiter = _integer(temporal_maxiter, "Time iterations")
    space_step = float(
        fit_step
        if space_fit_step is None and fit_step is not None
        else 1.0
        if space_fit_step is None
        else space_fit_step
    )
    time_step = float(
        fit_step
        if temporal_fit_step is None and fit_step is not None
        else 1.0
        if temporal_fit_step is None
        else temporal_fit_step
    )
    if not np.isfinite(space_step) or space_step < 0 or not np.isfinite(time_step) or time_step < 0:
        raise ValueError("Fit steps must be finite and non-negative.")
    smooth = float(alpha_smooth if regularization is None else regularization)
    prior = float(alpha_prior)
    if not np.isfinite(smooth) or smooth < 0 or not np.isfinite(prior) or prior < 0:
        raise ValueError("Smoothness and prior strengths must be finite and non-negative.")
    if error is not None and (not np.isfinite(error) or error < 0):
        raise ValueError("MSE threshold must be finite and non-negative.")
    shape = float(weibull_shape)
    if not np.isfinite(shape) or not 0.25 <= shape <= 8.0:
        raise ValueError("Weibull shape must be between 0.25 and 8.")
    onset = float(weibull_onset)
    if not np.isfinite(onset) or onset <= 0:
        raise ValueError("Weibull onset smoothing time must be finite and positive.")
    offset = 0.0 if T0 is None else float(T0)
    if not np.isfinite(offset):
        raise ValueError("T0 must be finite.")
    raw_observed = np.asarray(observed, dtype=float)
    if raw_observed.size == 0 or raw_observed.ndim > 2 or not np.any(np.isfinite(raw_observed)):
        raise ValueError("Supply a non-empty numeric observation with finite values.")
    mean_fraction = observation == "fraction" and (
        raw_observed.ndim == 1 or raw_observed.shape[0] == 1
    )
    if observation == "timing":
        if raw_observed.ndim != 1:
            raise ValueError("Replication timing must be a spatial vector.")
        n = raw_observed.size
        valid = np.isfinite(raw_observed)
        if np.any(raw_observed[valid] <= offset):
            raise ValueError("Replication timing must be later than T0.")
        typical = np.where(valid, raw_observed - offset, np.nanmedian(raw_observed[valid] - offset))
    else:
        valid = np.isfinite(raw_observed)
        if np.any((raw_observed[valid] < 0.0) | (raw_observed[valid] > 1.0)):
            raise ValueError("Replicated fraction must be between zero and one.")
        n = None if mean_fraction else raw_observed.shape[0]
        typical = None
    supplied_n = _spatial_size(supplied)
    n = n or supplied_n or n_bins
    if n is None and interval_kb is not None:
        n = max(1, int(np.ceil((float(interval_kb[1]) - float(interval_kb[0])) / dx)))
    if n is None and genome_length is not None:
        n = max(1, int(np.ceil(float(genome_length) / dx)))
    if n is None:
        raise ValueError("A spatial profile or genomic region is required for this fit.")
    n = int(n)
    if (
        n < 1
        or n_bins is not None
        and int(n_bins) != n
        or supplied_n is not None
        and supplied_n != n
    ):
        raise ValueError(
            "Spatial inputs and the selected region must have the same number of points."
        )
    original_n = n
    supplied_array = np.asarray(supplied, dtype=float)
    if (not supplied_array.size or np.any(~np.isfinite(supplied_array))
            or np.any(supplied_array < 0)
            or target == "initiation_rate" and np.any(supplied_array <= 0)):
        name = "Fork speed" if target == "initiation_rate" else "Initiation rate"
        domain = "positive" if target == "initiation_rate" else "non-negative"
        raise ValueError(f"{name} must contain finite {domain} values.")
    if target == "fork_speed" and not np.any(supplied_array > 0):
        raise ValueError("A positive initiation rate is required to infer fork speed.")
    if time_grid is None:
        if observation == "fraction":
            clock = np.arange(raw_observed.shape[-1], dtype=float) * dt
        else:
            maximum = max_time if max_time is not None else max_rep_time
            horizon_factor = 3.5 if form in {"t", "xt"} and shape >= 1.5 else 6.0
            # Later supplied samples do not require later prediction samples:
            # interpolate from their native clock and extend only if survival
            # actually remains. Long input files must not inflate every fit.
            maximum = (
                float(maximum)
                if maximum is not None
                else max(horizon_factor * float(np.max(typical)), 10.0 * dt)
            )
            if not np.isfinite(maximum) or maximum <= 0:
                raise ValueError("Maximum time must be finite and positive.")
            intervals = maximum / dt
            if not np.isfinite(intervals) or intervals > 99_999:
                raise ValueError(
                    "The fit time grid exceeds 100,000 samples; check the timing units or time resolution."
                )
            clock = np.arange(int(np.ceil(intervals)) + 1) * dt
    else:
        clock = np.asarray(time_grid, dtype=float)
    if (
        clock.ndim != 1
        or clock.size < 2
        or np.any(~np.isfinite(clock))
        or clock[0] != 0
        or np.any(np.diff(clock) <= 0)
    ):
        raise ValueError("Time grid must start at zero and be finite and strictly increasing.")
    if observation == "fraction" and clock.size != raw_observed.shape[-1]:
        raise ValueError("The time grid must match the replicated-fraction observations.")
    if observation == "timing" and clock[-1] <= float(np.max(typical)):
        raise ValueError("The time grid must extend beyond the observed replication timing.")
    native_clock = clock.copy()
    delay = np.asarray(weibull_delay, dtype=float).reshape(-1)
    if delay.size not in {1, n} or np.any(~np.isfinite(delay)) or np.any(delay < 0):
        raise ValueError(
            "Weibull delay must be non-negative and scalar or match the genomic region."
        )
    compact = bool(
        perQ
        and form in {"scalar", "t"}
        and _uniform_space(supplied)
        and delay.size == 1
        and fit_projection is None
        and not has_fit_constraint(target)
    )
    # Uniform periodic histories have one prediction regardless of chromosome
    # length. Retain the physical circumference in the compact cone operator.
    if compact:
        n = 1
    if clock.size > 100_000:
        raise ValueError("The fit time grid exceeds 100,000 samples; check the timing units or time resolution.")
    if observation == "timing" and target == "fork_speed":
        # The prediction clock may end before the upload. Completion still
        # depends on the upload's eventual continuation, not that shorter
        # clock's last positive sample (a finite pulse has an infinite mean).
        final_rate = supplied_array[:, -1] if supplied_array.ndim == 2 else supplied_array
        final_rate = np.broadcast_to(final_rate.reshape(-1), (n,))
        if not np.all(eventual_sources(final_rate, dx, radius, perQ)):
            raise ValueError(
                "The supplied profile allows no continuing initiation within the influence radius of some positions; a finite mean replication time cannot be fitted."
            )
    if observation == "fraction":
        fraction_matrix = raw_observed.reshape(1, -1) if mean_fraction else raw_observed
        finite_fraction = np.isfinite(fraction_matrix)
        counts = finite_fraction.sum(axis=0)
        means = np.divide(
            np.where(finite_fraction, fraction_matrix, 0.0).sum(axis=0),
            counts,
            out=np.full(clock.size, np.nan),
            where=counts > 0,
        )
        survival = np.where(np.isfinite(means), 1.0 - means, 0.0)
        rough_timing = float(_trapz_weights(clock) @ survival)
        typical = np.full(n, max(rough_timing, dt))
    elif compact:
        typical = np.asarray([np.mean(typical)])
    reference = (
        float(weibull_reference) if weibull_reference is not None else float(np.median(typical))
    )
    if not np.isfinite(reference) or reference <= 0:
        raise ValueError("Weibull reference time must be finite and positive.")
    if not np.isfinite(float(floor)) or float(floor) < 0:
        raise ValueError("Initiation-rate floor must be finite and non-negative.")
    if err_threshold is not None and (
        not np.isfinite(float(err_threshold)) or float(err_threshold) < 0
    ):
        raise ValueError("Numerical cutoff exponent must be finite and non-negative.")
    numerical_floor = max(
        float(floor),
        0.0 if err_threshold is None else 10.0 ** np.clip(-float(err_threshold), -300.0, 300.0),
    )
    bounds = (
        fit_bounds(
            rate_bounds,
            (max(numerical_floor, 1e-15), np.inf),
            label="Initiation-rate bounds",
            allow_zero=True,
        )
        if target == "initiation_rate"
        else fit_bounds(speed_bounds, (0.01, 10.0), label="Fork-speed bounds")
    )
    model = _KineticModel(
        target,
        form,
        n,
        clock,
        reference,
        delay,
        shape,
        bool(fit_weibull_shape),
        bounds,
        fit_projection,
        onset,
    )
    initial_value = initial_initiation_rate if target == "initiation_rate" else initial_fork_speed
    seed_supplied = supplied
    if (_warm_start is None and target == "initiation_rate"
            and supplied_array.ndim == 2 and supplied_array.shape[1] > 1):
        relevant_clock = clock[clock <= max(float(np.median(typical)), dt)]
        seed_supplied = _supplied_on_clock(
            supplied, relevant_clock, native_clock, n, dt, "fork_speed", positive=True
        ).T
    parameters = (
        model.initial(initial_value, typical, seed_supplied)
        if _warm_start is None else np.zeros(model.count)
    )
    if _warm_start is not None:
        state = _warm_start
        native_x = (np.arange(n) + 0.5) / n
        old_amplitude = np.asarray(state["log_amplitude"]).reshape(-1)
        old_x = (np.arange(len(old_amplitude)) + 0.5) / len(old_amplitude)
        amplitude = np.interp(native_x, old_x, old_amplitude)
        if form == "xt":
            parameters[:n] = amplitude - amplitude.mean()
            parameters[model.amplitude_index] = amplitude.mean()
        elif form == "x":
            parameters[:n] = amplitude
        elif form == "t":
            parameters[model.amplitude_index] = amplitude.mean()
        else:
            parameters[0] = amplitude.mean()
        if model.shape_index is not None:
            old_shape = np.asarray(state["log_shape"]).reshape(-1)
            shape_x = (np.arange(len(old_shape)) + 0.5) / len(old_shape)
            parameters[model.shape_slice] = (
                np.interp(native_x, shape_x, old_shape)
                if model.shape_count > 1
                else old_shape.mean()
            )
        if model.baseline_index is not None:
            parameters[model.baseline_index] = state["log_baseline"]
        parameters = np.clip(parameters, model.lower, model.upper)
    initial_parameters = parameters.copy()
    schedule = _schedule(form, space_maxiter, temporal_maxiter, spacetime_order)
    budget = len(schedule)
    supplied_name = "fork_speed" if target == "initiation_rate" else "initiation_rate"
    # Large spatial grids are evaluated below in overlapping influence tiles.
    # Do not interpolate the supplied field into a full XT temporary first.
    fixed = None
    valid_observed = np.isfinite(raw_observed)
    if observation == "timing" and refine_edge_kb and not perQ:
        edge = min(int(float(refine_edge_kb) / dx), max(0, (original_n - 1) // 2))
        if edge:
            valid_observed[:edge] = valid_observed[-edge:] = False
    if not valid_observed.any():
        raise ValueError("No finite observations remain in the selected region.")
    observation_count = int(valid_observed.sum())
    weights = _trapz_weights(clock)
    history = []
    phase_history = []
    calls = 0
    cached_operator = None
    scalar_response = None
    enforce_completion = False
    completion_tolerance = 1e-8
    started = time.time() if _started is None else _started
    scale_loss = (
        max(float(np.nanmedian(raw_observed[valid_observed] - offset)) ** 2, 1.0)
        if observation == "timing"
        else 1.0
    )
    static_primitive = bool(
        backend == "auto"
        and not compact
        and (
            target == "initiation_rate"
            and form in {"scalar", "x"}
            and (supplied_array.ndim < 2 or supplied_array.shape[1] == 1)
            or target == "fork_speed"
            and form in {"scalar", "x"}
            and (supplied_array.ndim < 2 or supplied_array.shape[1] == 1)
        )
    )
    scalar_linear = bool(
        target == "initiation_rate"
        and form == "scalar"
        and fit_projection is None
        and not has_fit_constraint(target)
    )
    uniform_kinetics = bool(
        form in {"scalar", "t"}
        and _uniform_space(supplied)
        and delay.size == 1
        and fit_projection is None
        and not has_fit_constraint(target)
    )
    stationary_source = bool(
        backend == "auto"
        and (
            target == "initiation_rate"
            and form in {"scalar", "x"}
            and _uniform_space(supplied)
            or target == "fork_speed"
            and form in {"scalar", "t"}
            and (supplied_array.ndim < 2 or supplied_array.shape[1] == 1)
        )
    )

    from ._kinetic_tiled import DENSE_FIT_VALUES, TiledKineticFit
    from ._resources import ensure_capacity, hosted, working_bytes

    tiled = None
    bounded = None
    # Keep the stationary primitive only for spatially varying prescribed
    # speed; uniform speed retains the much faster spatial convolution.
    streamed = (hosted() and static_primitive and not stationary_source
                and observation == "timing")

    def configure_grid():
        nonlocal tiled, fixed, bounded
        from ._kinetic_bounded import (
            ConeConvolutionTiming,
            StationaryConvolutionTiming,
            UniformTiming,
        )

        bounded = None
        if hosted() and observation == "timing" and backend == "auto" and not compact:
            args = (model, raw_observed, valid_observed, offset, scale_loss, dx, perQ, radius)
            constant_supplied = bool(np.all(supplied_array == supplied_array.reshape(-1)[0]))
            uniform = (form in {"scalar", "t"} and constant_supplied and delay.size == 1
                       and fit_projection is None and not has_fit_constraint(target))
            if uniform_kinetics or uniform:
                bounded = UniformTiming(*args, supplied=supplied, native_clock=native_clock, dt=dt)
            elif (target == "initiation_rate" and
                  (stationary_source or constant_supplied and form in {"scalar", "x"})):
                speed = (np.broadcast_to(supplied_array.reshape(-1)[0], (len(clock), n))
                         if constant_supplied else _supplied_on_clock(
                             supplied, clock, native_clock, n, dt, supplied_name, positive=True))
                bounded = StationaryConvolutionTiming(*args, speed=speed)
            elif stationary_source:
                rate = _supplied_on_clock(supplied, clock, native_clock, n, dt,
                                          supplied_name)
                bounded = StationaryConvolutionTiming(*args, rate=rate)
            elif (target == "initiation_rate" and form in {"t", "xt"}
                  and constant_supplied and fit_projection is None
                  and not has_fit_constraint(target)
                  and np.allclose(np.diff(clock), clock[1], rtol=1e-11, atol=1e-12)):
                bounded = ConeConvolutionTiming(*args, speed=float(supplied_array.reshape(-1)[0]), dt=clock[1])
        if bounded is not None:
            tiled, fixed = None, None
            return
        dense_limit = DENSE_FIT_VALUES
        if hosted():
            dense_limit = min(dense_limit, working_bytes(128 * 1024**2) // 192)
        if not streamed and not compact and not uniform_kinetics and n * len(clock) > dense_limit:
            tiled = TiledKineticFit(
                model, supplied, native_clock, dt, dx, perQ, radius,
                observation, raw_observed, valid_observed, mean_fraction, offset, scale_loss,
                static_primitive, stationary_source, backend,
            )
            fixed = None
        else:
            tiled = None
            if not streamed:
                ensure_capacity((1 if compact else n) * len(clock) * 192,
                                "Mapping the selected space-time grid")
            fixed = _supplied_on_clock(
                supplied, clock, native_clock, n, dt, supplied_name,
                positive=target == "initiation_rate",
            )

    configure_grid()

    def field_derivative(operator, d_hazard):
        options = {"with_speed": target == "fork_speed"}
        if isinstance(operator, (FastTransportHazard, StationarySourceHazard)):
            options["with_rate"] = target == "initiation_rate"
        rate_gradient, speed_gradient = operator.backward(d_hazard / scale_loss, **options)
        return rate_gradient if target == "initiation_rate" else speed_gradient

    def evaluate_dense(p, need_gradient=True):
        nonlocal calls, cached_operator, scalar_response
        check_cancelled()
        fitted, model_state = model.evaluate(p)
        rate, speed = (fitted, fixed) if target == "initiation_rate" else (fixed, fitted)
        if observation == "timing" and not np.all(eventual_sources(rate[-1], dx, radius, perQ)):
            if enforce_completion:
                calls += 1
                return {"loss": np.inf, "mse": np.inf, "gradient": np.zeros(model.count)}
            raise ValueError(
                "The kinetics or region transforms allow no continuing initiation within the influence radius of some positions; a finite mean replication time cannot be fitted."
            )
        if cached_operator is None or target == "fork_speed":
            if backend == "auto" and (compact or uniform_kinetics):
                operator = UniformKineticsHazard(
                    speed, clock, dx, perQ, radius, length=original_n * dx,
                    compress=not compact,
                )
            elif stationary_source and StationarySourceHazard.applicable(speed, clock):
                operator = StationarySourceHazard(speed, clock, dx, perQ, radius,
                                                 differentiate_speed=target == "fork_speed")
            elif static_primitive:
                operator = _StaticHazardAdapter(speed, clock, dx, perQ, radius)
            elif (
                backend == "auto"
                and target == "initiation_rate"
                and ConstantSpeedConvolution.applicable(speed, clock)
            ):
                operator = ConstantSpeedConvolution(speed, clock, dx, perQ, radius)
            elif backend == "auto":
                operator = FastTransportHazard(
                    speed,
                    clock,
                    dx,
                    perQ,
                    radius,
                    uniform_speed=target == "initiation_rate" or form in {"scalar", "t"},
                    differentiate_speed=target == "fork_speed",
                )
            else:
                operator = HazardOperator(
                    speed, clock, resolution=dx, periodic=perQ, radius=radius, backend=backend
                )
            if target == "initiation_rate":
                cached_operator = operator
        else:
            operator = cached_operator
        with measured_work_phase(0.0, 0.5):
            if scalar_linear:
                if scalar_response is None:
                    scalar_response = operator.forward(
                        np.broadcast_to(np.asarray(1.0), rate.shape), record=False
                    )
                hazard = scalar_response * float(rate[0, 0])
            else:
                hazard = operator.forward(rate, record=need_gradient or budget > 0)
        survival_field = np.exp(-np.maximum(hazard, 0.0))
        groups = getattr(operator, "groups", None)
        if (
            observation == "timing"
            and enforce_completion
            and np.max(survival_field[-1]) > completion_tolerance
        ):
            # A slower proposal must not appear to fit better by moving its
            # missing survival tail outside the fixed fitting clock.
            calls += 1
            return {"loss": np.inf, "mse": np.inf, "gradient": np.zeros(model.count)}
        if observation == "timing":
            predicted_small = weights @ survival_field + offset
            predicted = (
                np.broadcast_to(predicted_small, (original_n,)) if compact else predicted_small
            )
            if groups is not None:
                predicted = predicted_small[groups]
            residual = predicted - raw_observed
            q = np.where(valid_observed, residual, 0.0)
            mse = float(np.sum(q * q) / observation_count)
            q = np.asarray([q.sum()]) if compact else q
            if groups is not None:
                q = np.bincount(groups, q, minlength=hazard.shape[1])
            d_hazard = -weights[:, None] * survival_field * q[None, :] * (2.0 / observation_count)
        else:
            fraction = -np.expm1(-np.maximum(hazard, 0.0))
            predicted_small = fraction.mean(axis=1) if mean_fraction else fraction.T
            if mean_fraction and groups is not None:
                counts = np.bincount(groups, minlength=hazard.shape[1])
                predicted_small = (fraction @ counts) / n
            predicted = (
                np.broadcast_to(predicted_small, raw_observed.shape)
                if compact and not mean_fraction
                else predicted_small.reshape(raw_observed.shape)
                if mean_fraction
                else predicted_small
            )
            if groups is not None and not mean_fraction:
                predicted = predicted_small[groups]
            residual = predicted - raw_observed
            q = np.where(valid_observed, residual, 0.0) * (2.0 / observation_count)
            mse = float(np.sum(np.where(valid_observed, residual, 0.0) ** 2) / observation_count)
            if mean_fraction:
                if groups is None:
                    q_hazard = np.broadcast_to(q.reshape(-1, 1) / n, hazard.shape)
                else:
                    counts = np.bincount(groups, minlength=hazard.shape[1])
                    q_hazard = q.reshape(-1, 1) * counts[None, :] / n
            elif compact:
                q_hazard = q.sum(axis=0)[:, None]
            else:
                q_hazard = q.T
                if groups is not None:
                    q_hazard = np.stack([
                        np.bincount(groups, row, minlength=hazard.shape[1]) for row in q_hazard
                    ])
            d_hazard = q_hazard * survival_field
        if need_gradient:
            with measured_work_phase(0.5, 1.0):
                if scalar_response is not None:
                    derivative = np.asarray([[np.sum(d_hazard * scalar_response) / scale_loss]])
                else:
                    derivative = field_derivative(operator, d_hazard)
            gradient = model.gradient(derivative, p, model_state)
        else:
            # Rejected line-search proposals need no adjoint. Keep the last
            # forward tape so an accepted proposal can be differentiated
            # immediately, without evaluating its prediction a second time.
            gradient = np.zeros(model.count)
        calls += 1
        return regularize({
            "loss": mse / scale_loss,
            "mse": mse,
            "gradient": gradient,
            "fitted": fitted,
            "hazard": hazard,
            "survival": survival_field,
            "prediction": predicted,
            "residual": residual,
            "backend": getattr(operator, "backend", getattr(operator, "name", backend)),
            "positive_rate": bool(np.any(rate > 0.0)),
            "spatial_groups": groups,
            "gradient_state": None
            if need_gradient
            else (operator, d_hazard, p.copy(), model_state),
        }, p)

    def regularize(result, p):
        gradient = result["gradient"]
        penalty = 0.0
        if smooth and model.spatial_count > 1:
            z = p[: model.spatial_count]
            difference = np.diff(z)
            # The mean penalty is independent of chromosome sampling density.
            coefficient = smooth / max(1, difference.size + int(perQ))
            penalty += 0.5 * coefficient * np.dot(difference, difference)
            gradient[: model.spatial_count - 1] -= coefficient * difference
            gradient[1 : model.spatial_count] += coefficient * difference
            if perQ:
                seam = z[0] - z[-1]
                penalty += 0.5 * coefficient * seam**2
                gradient[0] += coefficient * seam
                gradient[model.spatial_count - 1] -= coefficient * seam
        if smooth and model.shape_count > 1:
            z = p[model.shape_slice]
            difference = np.diff(z)
            coefficient = smooth / max(1, difference.size + int(perQ))
            penalty += 0.5 * coefficient * np.dot(difference, difference)
            shape_gradient = gradient[model.shape_slice]
            shape_gradient[:-1] -= coefficient * difference
            shape_gradient[1:] += coefficient * difference
            if perQ:
                seam = z[0] - z[-1]
                penalty += 0.5 * coefficient * seam**2
                shape_gradient[0] += coefficient * seam
                shape_gradient[-1] -= coefficient * seam
        if prior:
            difference = p - initial_parameters
            penalty += 0.5 * prior * np.mean(difference**2)
            gradient += prior * difference / len(p)
        result["loss"] += penalty
        return result

    def evaluate(p, need_gradient=True):
        nonlocal calls
        if bounded is not None:
            calls += 1
            if hasattr(bounded, "source"):
                rate = bounded.source(p)
                if not np.all(eventual_sources(rate[-1], dx, radius, perQ)):
                    if enforce_completion:
                        return {"loss": np.inf, "mse": np.inf, "gradient": np.zeros(model.count)}
                    raise ValueError("The kinetics allow no continuing initiation near some positions.")
            result = bounded.evaluate(p, need_gradient)
            if enforce_completion and np.max(result["survival"][-1]) > completion_tolerance:
                return {"loss": np.inf, "mse": np.inf, "gradient": np.zeros(model.count)}
            return regularize(result, p)
        if streamed:
            from ._stationary_timing import stationary_timing
            fitted, state = model.evaluate(p)
            rate, speed = (fitted, fixed) if target == "initiation_rate" else (fixed, fitted)
            if not np.all(eventual_sources(rate[0], dx, radius, perQ)):
                if enforce_completion:
                    return {"loss": np.inf, "mse": np.inf, "gradient": np.zeros(model.count)}
                raise ValueError("The kinetics allow no continuing initiation near some positions.")
            with measured_work_phase(0., 1.):
                predicted, residual, sensitivity, final, derivative = stationary_timing(
                    rate[0], speed[0], clock, weights, dx, perQ, radius,
                    raw_observed, valid_observed, offset, scale_loss,
                    with_speed=target == "fork_speed",
                    need_gradient=need_gradient,
                )
            calls += 1
            if enforce_completion and np.max(final) > completion_tolerance:
                return {"loss": np.inf, "mse": np.inf, "gradient": np.zeros(model.count)}
            mse = float(np.sum(np.where(valid_observed, residual, 0.) ** 2) / observation_count)
            return regularize(dict(loss=mse / scale_loss, mse=mse,
                                   gradient=(model.gradient(derivative, p, state) if need_gradient
                                             else np.zeros(model.count)), fitted=fitted,
                                   hazard=None, survival=final[None, :], prediction=predicted,
                                   residual=residual, sensitivity=sensitivity,
                                   positive_rate=bool(np.any(rate[0] > 0)), spatial_groups=None,
                                   backend="streamed_stationary_travel_primitive",
                                   streamed_parameters=None if need_gradient else p.copy(),
                                   gradient_state=None), p)
        if tiled is None:
            return evaluate_dense(p, need_gradient)
        calls += 1
        result = tiled.evaluate(p, need_gradient, enforce_completion, completion_tolerance)
        return regularize(result, p)

    def complete_gradient(result):
        if "bounded_state" in result:
            bounded.complete(result)
            return
        streamed_parameters = result.pop("streamed_parameters", None)
        if streamed_parameters is not None:
            from ._stationary_timing import stationary_timing
            fitted, model_state = model.evaluate(streamed_parameters)
            rate, speed = (fitted, fixed) if target == "initiation_rate" else (fixed, fitted)
            derivative = stationary_timing(
                rate[0], speed[0], clock, weights, dx, perQ, radius,
                raw_observed, valid_observed, offset, scale_loss,
                with_speed=target == "fork_speed", gradient_only=result["residual"],
            )[-1]
            result["gradient"] += model.gradient(derivative, streamed_parameters, model_state)
            return
        tiled_parameters = result.pop("tiled_parameters", None)
        if tiled_parameters is not None:
            # Recompute one tile's tape at a time for accepted proposals;
            # keeping every tile's tape would undo the memory bound.
            evaluated = evaluate(tiled_parameters, need_gradient=True)
            result["gradient"] = evaluated["gradient"]
            return
        state = result.pop("gradient_state", None)
        if state is None:
            return
        operator, d_hazard, p, model_state = state
        if scalar_response is not None:
            derivative = np.asarray([[np.sum(d_hazard * scalar_response) / scale_loss]])
        else:
            derivative = field_derivative(operator, d_hazard)
        result["gradient"] += model.gradient(derivative, p, model_state)

    from ._kinetic_progress import FitStage

    progress_stage = FitStage(_progress_tracker, n)
    prediction_work = _progress_tracker.estimate(1.0, n, budget) + 1
    if observation == "timing":
        actual_projection = model.projection
        active_projection = actual_projection or current_fit_constraint(target)

        def planning_projection():
            if active_projection is None or active_projection.form != form:
                return None
            # References and cancellation functions remain shared; only the
            # mutable frozen selection belongs to this planning copy.
            clone = copy(active_projection)
            for name in ("mask", "frozen", "positions"):
                value = getattr(active_projection, name, None)
                if hasattr(clone, name):
                    setattr(clone, name, None if value is None else value.copy())
            return clone

        for extension in range(7):
            model.projection = planning_projection()
            with measured_work(prediction_work):
                current = evaluate(parameters, need_gradient=False)
            if np.max(current["survival"][-1]) <= completion_tolerance:
                break
            if not current["positive_rate"]:
                raise ValueError(
                    "The supplied kinetics or region transform allow no initiation; a finite mean replication time cannot be fitted."
                )
            if extension == 6:
                raise ValueError(
                    "Replication does not complete on the expanded time grid. Check the supplied initiation profile, influence radius and region transforms."
                )
            check_cancelled()
            count = max(1, int(np.ceil(clock[-1] / dt)))
            extended_size = clock.size + count
            if extended_size > 100_000:
                raise ValueError(
                    "Replication has not completed on the available fit grid; increase its resolution or check the supplied kinetics."
                )
            extended = np.r_[clock, clock[-1] + np.arange(1, count + 1, dtype=float) * dt]
            # Release the old kernel/objective before budgeting its replacement.
            current = None
            bounded = None
            cached_operator = None
            clock = extended
            model.clock = clock
            weights = _trapz_weights(clock)
            configure_grid()
            cached_operator = None
            scalar_response = None
        model.projection = actual_projection
        enforce_completion = True
        if active_projection is not None:
            # Freeze the real mask only after the full fitting clock is known.
            current = evaluate(parameters, need_gradient=False)
            if not np.isfinite(current["loss"]):
                raise ValueError(
                    "The region transform prevents completion on the fitting clock; use a compatible transform or longer time domain."
                )
    else:
        with measured_work(prediction_work):
            current = evaluate(parameters, need_gradient=False)
    progress_stage.ready()
    if _fit_progress_enabled(verbose):
        etaferr(
            _progress_offset,
            _progress_tracker.total,
            started,
            current["mse"],
            done=False,
            label="Mapping",
            eta_seconds=progress_stage.eta(budget),
        )
    history.append(current["mse"])
    phase_memories = {"space": [], "time": []}
    phase_indices = {"space": None, "time": None}
    phase_scales = {
        key: float((_warm_start or {}).get("phase_scales", {}).get(key, 1.0))
        for key in ("space", "time")
    }
    completed = {"space": 0, "time": 0}
    stalled_phases = {"space": 0, "time": 0}
    for phase in schedule:
        check_cancelled()
        if relative_tolerance and stalled_phases[phase] >= 2:
            continue
        if error is not None and current["mse"] <= error:
            break
        spatial_active = None
        if relative_tolerance > 0.0:
            tolerance = (
                relative_tolerance * np.maximum(np.abs(raw_observed - offset), dt)
                if observation == "timing"
                else relative_tolerance
            )
            if error is not None:
                tolerance = np.minimum(tolerance, np.sqrt(error))
            outside = valid_observed & (np.abs(current["residual"]) > tolerance)
            if not np.any(outside):
                break
            if form in {"x", "xt"}:
                from scipy.ndimage import maximum_filter1d

                bad = (
                    outside
                    if observation == "timing"
                    else (np.any(outside, axis=1) if not mean_fraction else np.ones(n, dtype=bool))
                )
                halo = min(n, int(np.ceil(radius / dx)))
                spatial_active = maximum_filter1d(
                    bad.astype(np.uint8), size=2 * halo + 1, mode="wrap" if perQ else "constant"
                ).astype(bool)
        indices = model.block(phase)
        if spatial_active is not None:
            if phase == "space":
                indices = indices[spatial_active]
            elif model.shape_count > 1:
                keep = np.ones(len(indices), dtype=bool)
                shapes = (indices >= model.shape_slice.start) & (indices < model.shape_slice.stop)
                keep[shapes] = spatial_active[indices[shapes] - model.shape_slice.start]
                indices = indices[keep]
        if phase_indices[phase] is not None and not np.array_equal(phase_indices[phase], indices):
            phase_memories[phase].clear()
        phase_indices[phase] = indices.copy()
        step = space_step if phase == "space" else time_step
        if observation == "timing":
            local_error = np.abs(current["residual"][valid_observed]) / np.maximum(
                raw_observed[valid_observed] - offset, dt
            )
            relative_error = (
                np.sqrt(np.mean(local_error**2)) if phase == "time" else np.max(local_error)
            )
            step = min(step, max(0.002, 3.0 * relative_error))
        partial_space = form == "xt" and phase == "space" and len(indices) < model.spatial_count
        if indices.size and step > 0:
            complete_gradient(current)
        gradient = current["gradient"][indices].copy()
        if partial_space:
            gradient += current["gradient"][model.amplitude_index] / n
        metric = np.ones(len(indices))
        if phase == "time" and model.shape_count > 1:
            metric[(indices >= model.shape_slice.start) & (indices < model.shape_slice.stop)] = (
                np.sqrt(n)
            )
        accepted = False
        old_parameters = parameters.copy()
        old = current
        if indices.size and step > 0 and np.max(np.abs(gradient)) > 1e-14:
            direction = None
            if target == "initiation_rate" and form == "x" and observation == "timing":
                # H is linear in initiation. Its exact response to a common
                # log-amplitude change supplies a local Newton scale without
                # another prediction. The adjoint and line search still
                # require descent of the complete, coupled objective.
                sensitivity = current.get("sensitivity")
                if sensitivity is None:
                    sensitivity = np.einsum(
                        "t,tx,tx->x", weights, current["hazard"], current["survival"]
                    )
                local_direction = current["residual"][indices] / np.maximum(
                    sensitivity[indices], 1e-12
                )
                local_direction[gradient == 0.0] = 0.0
                if np.all(np.isfinite(local_direction)) and np.dot(local_direction, gradient) < 0:
                    direction = local_direction
            if target == "fork_speed" and form == "xt" and phase == "space" and observation == "timing":
                # T scales as v**(-1/2) in the homogeneous stationary limit.
                # Use that local scale to propose spatial amplitude updates,
                # avoiding normalization by a few large gradient entries.
                # This is a preconditioner, not an independent point fit:
                # the exact coupled gradient must agree with the direction,
                # and the complete objective/MSE still governs acceptance.
                local_direction = 2.0 * current["residual"][indices] / np.maximum(
                    current["prediction"][indices] - offset, dt
                )
                local_direction[gradient == 0.0] = 0.0
                if np.all(np.isfinite(local_direction)) and np.dot(local_direction, gradient) < 0:
                    direction = local_direction
            if direction is None:
                direction = metric * _lbfgs_direction(metric * gradient, phase_memories[phase])
            if np.dot(direction, gradient) >= 0:
                direction = -gradient / max(float(np.max(np.abs(gradient))), 1e-12)
            if form == "xt" and phase == "space" and len(indices) == model.spatial_count:
                direction -= direction.mean()
            magnitude = max(float(np.max(np.abs(direction))), 1e-30)
            direction *= min(1.0, step / magnitude)
            for trial in range(8):
                check_cancelled()
                candidate = parameters.copy()
                candidate[indices] = np.clip(
                    parameters[indices] + direction * phase_scales[phase] * 0.5**trial,
                    model.lower[indices],
                    model.upper[indices],
                )
                if partial_space:
                    candidate[model.amplitude_index] += np.mean(candidate[:n] - parameters[:n])
                if np.array_equal(candidate, parameters):
                    break
                evaluated = evaluate(candidate, need_gradient=False)
                displacement = candidate[indices] - parameters[indices]
                armijo = current["loss"] + 1e-4 * float(np.dot(gradient, displacement))
                if (
                    np.isfinite(evaluated["loss"])
                    and evaluated["loss"] <= armijo + 1e-14
                    and evaluated["mse"] <= current["mse"] + 1e-12
                ):
                    complete_gradient(evaluated)
                    parameters, current, accepted = candidate, evaluated, True
                    phase_scales[phase] = min(1.0, phase_scales[phase] * 0.5**trial * 1.5)
                    break
        if accepted:
            s = parameters[indices] - old_parameters[indices]
            y = current["gradient"][indices] - old["gradient"][indices]
            if partial_space:
                y += (
                    current["gradient"][model.amplitude_index]
                    - old["gradient"][model.amplitude_index]
                ) / n
            s = s / metric
            y = y * metric
            curvature = float(np.dot(s, y))
            if curvature > 1e-12 * max(float(np.linalg.norm(s) * np.linalg.norm(y)), 1e-30):
                phase_memories[phase].append((s, y, 1.0 / curvature))
                phase_memories[phase] = phase_memories[phase][-5:]
        else:
            phase_memories[phase].clear()
            phase_scales[phase] = max(1e-6, 0.25 * phase_scales[phase])
        gain = (old["mse"] - current["mse"]) / max(old["mse"], 1e-30)
        if gain <= min(0.001, 0.1 * relative_tolerance):
            stalled_phases[phase] += 1
        else:
            # A useful update in either coupled phase can make the other
            # phase useful again. Flat/rejected proposals cannot keep a
            # plateau alive by repeatedly shrinking a redundant line search.
            stalled_phases = {"space": 0, "time": 0}
        completed[phase] += 1
        history.append(current["mse"])
        phase_history.append(
            {
                "phase": phase,
                "iteration": completed[phase],
                "accepted": accepted,
                "mse": current["mse"],
            }
        )
        iteration = sum(completed.values())
        progress_stage.update()
        if _fit_progress_enabled(verbose) and iteration % max(1, int(print_every)) == 0:
            etaferr(
                iteration + _progress_offset,
                _progress_tracker.total,
                started,
                current["mse"],
                done=False,
                label="Mapping",
                eta_seconds=progress_stage.eta(budget - iteration),
            )
    if _fit_progress_enabled(verbose):
        etaferr(
            sum(completed.values()) + _progress_offset,
            _progress_tracker.total,
            started,
            current["mse"],
            done=not _warmup,
            label="Mapping",
            eta_seconds=progress_stage.eta(0),
        )
    fitted = (bounded.public(parameters) if bounded is not None else
              tiled.public(parameters) if tiled is not None else model.public(current["fitted"]))
    final_survival = current["survival"][-1]
    info = {
        "fit_mse": current["mse"],
        "residual": current["residual"],
        "time_grid": clock,
        "fit_history": np.asarray(history),
        "phase_history": phase_history,
        "target_shape": "weibull" if model.temporal else "positive_field",
        "target_form": {"scalar": "constant", "x": "space", "t": "time", "xt": "spacetime"}[form],
        "forward_backend": current["backend"],
        "fitting_backend": "hazard_adjoint_lbfgs",
        "space_iterations": completed["space"],
        "time_iterations": completed["time"],
        "iterations": sum(completed.values()),
        "iteration_budget": budget,
        "objective_evaluations": calls,
        "iterations_completed": sum(completed.values()),
        "space_iterations_completed": completed["space"],
        "temporal_iterations_completed": completed["time"],
        "fit_progress_total": budget,
        "spacetime_order": spacetime_order,
        "neighbourhood_range": radius,
        "unreplicated_at_end": float(np.max(final_survival)),
        "observation": observation,
        "weibull_reference": reference,
        "tiled_fitting": tiled is not None,
        "disk_backed_output": isinstance(fitted, np.memmap),
        "native_resolution": dx,
    }
    log_amplitude = (
        parameters[:n] - parameters[:n].mean() + parameters[model.amplitude_index]
        if form == "xt"
        else parameters[:n]
        if form == "x"
        else np.asarray([parameters[model.amplitude_index if form == "t" else 0]])
    )
    info["_parameter_state"] = {
        "log_amplitude": log_amplitude.copy(),
        "log_shape": parameters[model.shape_slice].copy()
        if model.shape_index is not None
        else np.asarray([np.log(shape)]),
        "log_baseline": parameters[model.baseline_index]
        if model.baseline_index is not None
        else None,
        "phase_scales": phase_scales.copy(),
    }
    if model.temporal:
        if model.shape_index is None:
            info["weibull_shape"] = shape
        else:
            shapes = np.exp(parameters[model.shape_slice])
            info["weibull_shape"] = shapes.copy() if model.shape_count > 1 else float(shapes[0])
        info["weibull_delay"] = delay.copy()
        info["weibull_onset_cutoff_min"] = onset
        info["alpha"] = (
            np.exp(parameters[:n] - np.mean(parameters[:n]) + parameters[model.amplitude_index])
            if form == "xt"
            else np.exp(parameters[model.amplitude_index])
        )
        if model.baseline_index is not None:
            info["weibull_baseline"] = float(np.exp(parameters[model.baseline_index]))
    if observation == "timing":
        info["replication_timing_fit"] = current["prediction"]
        if clock.size * n <= 1_000_000 and current["survival"].shape[0] == clock.size:
            groups = current["spatial_groups"]
            info["survival"] = (
                current["survival"].T if groups is None else current["survival"][:, groups].T
            )
    else:
        info["replicated_fraction_fit" if mean_fraction else "replicated_fraction_xt_fit"] = (
            current["prediction"]
        )
        timing_prediction = current.get("timing_prediction")
        if timing_prediction is None:
            timing_prediction = (weights @ current["survival"]) + offset
        if current["spatial_groups"] is not None:
            timing_prediction = timing_prediction[current["spatial_groups"]]
        timing_prediction = (
            np.broadcast_to(timing_prediction, (original_n,)) if compact else timing_prediction
        )
        final_rate = current.get("final_rate")
        if final_rate is None:
            final_rate = current["fitted"][-1] if target == "initiation_rate" else fixed[-1]
        restricted = float(np.max(final_survival)) > completion_tolerance or not np.all(
            eventual_sources(final_rate, dx, radius, perQ)
        )
        info["timing_is_restricted_mean"] = restricted
        info["restricted_mean_timing" if restricted else "replication_timing_fit"] = (
            timing_prediction
        )
    if np.asarray(fitted).ndim == 0:
        return _RmapInfoFloat(float(fitted), info)
    return _RmapInfoArray(fitted, info)
