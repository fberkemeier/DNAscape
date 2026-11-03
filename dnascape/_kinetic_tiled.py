"""Native-resolution kinetic objectives with bounded spatial working sets.

Tiles share one parameter vector, objective and adjoint. Only their receiving
cells contribute to the loss; overlapping influence halos supply origins and
fork trajectories. This is not a set of independently fitted subregions.
"""

import numpy as np
import tempfile

from ._fit_constraints import current_fit_constraint, fit_constraint_context
from ._kinetic_tail import eventual_sources
from .utils import check_cancelled, measured_work_phase


# Keep the existing fast dense path for ordinary regional fits. The tiled
# path starts before the old 64-million-value allocation guard is reached.
DENSE_FIT_VALUES = 32_000_000
TILE_VALUES = 4_000_000


def disk_array(shape, dtype=float, order="C"):
    """A normal numeric array whose backing file lives only as long as its map.

    TemporaryFile is unlinked immediately on POSIX and delete-on-close on
    Windows. The mapping owns the remaining OS handle, including on cancel.
    """
    with tempfile.TemporaryFile(prefix="dnascape-fit-") as handle:
        return np.memmap(handle, dtype=dtype, mode="w+", shape=shape, order=order)


class TiledKineticFit:
    def __init__(self, model, supplied, native_clock, dt, dx, periodic, radius,
                 observation, observed, valid, mean_fraction, offset, scale_loss,
                 static, stationary, backend, *, quadrature=False, quadrature_tolerance=None,
                 observation_weights=None):
        self.model, self.supplied, self.native_clock = model, supplied, native_clock
        self.dt, self.dx, self.periodic = dt, dx, periodic
        self.radius = min(radius, model.n * dx / 2) if periodic else radius
        self.observation, self.observed, self.valid = observation, observed, valid
        self.mean_fraction, self.offset, self.scale_loss = mean_fraction, offset, scale_loss
        self.static, self.stationary, self.backend = static, stationary, backend
        self.count = int(valid.sum())
        self.observation_weights = observation_weights
        self.projection = None
        self.halo = int(np.ceil(self.radius / self.dx)) + 8
        self.steps = None
        self.cached_operator = None
        self.cached_geometry = None
        self.quadrature = quadrature
        self.typical = (np.where(valid, observed-offset, np.nanmedian(observed-offset))
                        if observation == "timing" else None)
        self.quadrature_tolerance = (quadrature_tolerance if quadrature_tolerance is not None
                                    else 1e-6*np.maximum(self.typical, dt)
                                    if self.typical is not None else None)

    def tiles(self):
        n = self.model.n
        # Extra cells keep the interpolation stencil away from artificial
        # boundaries as well as retaining the full physical influence halo.
        halo = min(n, self.halo)
        # Do not let a long clock leave a tiny core inside a large halo:
        # that would repeatedly solve almost the same neighbourhood.
        core = min(16384, max(256, halo, TILE_VALUES // len(self.model.clock) - 2 * halo))
        if self.quadrature:
            core = 32768
        if core + 2 * halo >= n:
            yield np.arange(n), slice(0, n), slice(0, n), self.periodic
            return
        for first in range(0, n, core):
            check_cancelled()
            stop = min(n, first + core)
            left, right = first - halo, stop + halo
            if not self.periodic:
                left, right = max(0, left), min(n, right)
            indices = np.arange(left, right) % n
            yield indices, slice(first - left, stop - left), slice(first, stop), False

    def local_model(self, p, indices):
        from ._kinetic_fit import _KineticModel

        parent = self.model
        delay = parent.delay if parent.delay.size == 1 else parent.delay[indices]
        local = _KineticModel(parent.target, parent.form, len(indices), parent.clock,
                              parent.reference, delay, parent.shape, parent.fit_shape,
                              parent.bounds, onset=parent.onset)
        q = np.zeros(local.count)
        if parent.form in {"x", "xt"}:
            q[:local.n] = p[indices]
        elif parent.form == "scalar":
            q[0] = p[0]
        if parent.temporal:
            q[local.amplitude_index] = p[parent.amplitude_index]
            if parent.form == "xt":
                q[local.amplitude_index] += q[:local.n].mean() - p[:parent.n].mean()
        if parent.shape_index is not None:
            q[local.shape_slice] = (p[parent.shape_slice][indices]
                                   if parent.shape_count > 1 else p[parent.shape_slice])
        if parent.baseline_index is not None:
            q[local.baseline_index] = p[parent.baseline_index]
        return local, q

    def fields(self, p, indices, *, project=True):
        from ._kinetic_fit import _supplied_on_clock

        local, q = self.local_model(p, indices)
        # The worker's constraint belongs to the entire selected region.
        # Apply its already-frozen values below, using global coordinates.
        with fit_constraint_context(None):
            fitted, state = local.evaluate(q)
        if project and self.projection is not None:
            projection = self.projection
            mask, frozen = projection.mask, projection.frozen
            if local.form in {"x", "xt"}:
                mask, frozen = mask[indices].T, frozen[indices].T
            elif local.form == "t":
                mask, frozen = mask.reshape(-1, 1), frozen.reshape(-1, 1)
            else:
                mask, frozen = mask.reshape(1, 1), frozen.reshape(1, 1)
            fitted = np.where(mask, frozen, fitted)
            state = (*state[:-1], np.broadcast_to(state[-1], fitted.shape) & ~mask)
        supplied = np.asarray(self.supplied)
        if supplied.ndim and supplied.shape[0] == self.model.n:
            supplied = supplied[indices]
        fixed = _supplied_on_clock(
            supplied, self.model.clock, self.native_clock, local.n, self.dt,
            "fork_speed" if local.target == "initiation_rate" else "initiation_rate",
            positive=local.target == "initiation_rate",
        )
        rate, speed = ((fitted, fixed) if local.target == "initiation_rate" else (fixed, fitted))
        return local, q, fitted, state, rate, speed

    def public(self, p, *, project=True, order="C"):
        form, n, nt = self.model.form, self.model.n, len(self.model.clock)
        if form in {"scalar", "t"}:
            indices = np.arange(min(n, 1))
            local, _, fitted, *_ = self.fields(p, indices, project=project)
            return local.public(fitted)
        result = disk_array((n, nt), order=order) if form == "xt" else np.empty(n)
        for indices, core, destination, _ in self.tiles():
            local, _, fitted, *_ = self.fields(p, indices, project=project)
            result[destination] = (fitted[:, core].T if form == "xt" else fitted[0, core])
        return result

    def prepare_projection(self, p):
        projection = self.model.projection or current_fit_constraint(self.model.target)
        if projection is None or projection.form != self.model.form:
            self.projection = None
            return
        if projection.frozen is None:
            # Region transforms keep their global selection, interpolation
            # anchors, statistics and single reproducible noise draw.
            raw = self.public(p, project=False, order="F")
            if self.model.form == "xt" and hasattr(projection, "prepare_tiled"):
                projection.prepare_tiled(raw)
            else:
                projection.project(raw)
        self.projection = projection

    def operator(self, speed, periodic):
        from ._hazard import HazardOperator
        from ._hazard_fft import ConstantSpeedConvolution
        from ._hazard_transport import FastTransportHazard
        from ._hazard_stationary_source import StationarySourceHazard
        from ._kinetic_fit import _StaticHazardAdapter, _uniform_space

        args = (speed, self.model.clock, self.dx, periodic, self.radius)
        cacheable = (
            self.model.target == "initiation_rate" and _uniform_space(self.supplied)
            or self.model.target == "fork_speed" and self.stationary
            and self.model.form in {"scalar", "t"} and self.model.delay.size == 1
        )
        key = (speed.shape, periodic, speed[:, 0].tobytes())
        if cacheable and self.cached_geometry == key:
            return self.cached_operator

        def remember(operator):
            # Uniform speed gives the same translated geometry in every
            # equally sized tile. Its sampled values also identify changes
            # between fitted-speed proposals. Retain just one kernel, never
            # a chromosome's collection of tapes.
            if cacheable:
                self.cached_operator, self.cached_geometry = operator, key
            return operator

        if self.stationary:
            candidate = StationarySourceHazard(*args)
            if len(self.model.clock) * candidate.length <= 24_000_000:
                return remember(candidate)
        if self.static:
            return _StaticHazardAdapter(*args)
        if self.backend == "auto" and self.model.target == "initiation_rate":
            if self.constant_convolution_fits(speed.shape[1], periodic):
                return remember(ConstantSpeedConvolution(*args))
            if ConstantSpeedConvolution.applicable(speed, self.model.clock):
                return remember(ConstantSpeedConvolution(*args))
        if self.backend == "auto":
            operator = FastTransportHazard(
                *args, uniform_speed=self.model.target == "initiation_rate" or self.model.form in {"scalar", "t"},
                differentiate_speed=self.model.target == "fork_speed",
            )
            operator._steps_override = self.steps
            return remember(operator)
        return HazardOperator(speed, self.model.clock, resolution=self.dx,
                              periodic=periodic, radius=self.radius, backend=self.backend)

    def constant_convolution_fits(self, n, periodic=False):
        from scipy.fft import next_fast_len

        supplied = np.asarray(self.supplied)
        clock = self.model.clock
        if (supplied.size != 1 or self.model.target != "initiation_rate"
                or not np.allclose(np.diff(clock), clock[1] - clock[0], rtol=1e-11, atol=1e-12)):
            return False
        reach = min(self.radius, float(supplied.reshape(-1)[0]) * (clock[-1] + clock[1]))
        width = min(n // 2 if periodic else n - 1, int(np.ceil(reach / self.dx + .5)))
        length = n if periodic else next_fast_len(n + width)
        return next_fast_len(2 * len(clock) - 1) * length <= 24_000_000

    def prepare_geometry(self, p):
        from scipy.fft import next_fast_len

        maximum_tile = max(len(indices) for indices, *_ in self.tiles())
        stationary_size = len(self.model.clock) * next_fast_len(
            maximum_tile + min(maximum_tile - 1, int(np.ceil(self.radius / self.dx + .5))))
        if (self.static or self.stationary and stationary_size <= 24_000_000
                or self.constant_convolution_fits(maximum_tile)):
            # Cell-exact stationary/constant-speed convolutions already have
            # strictly finite spatial support and need only the radius halo.
            return
        # The retarded-potential shortcut subtracts unrestricted histories.
        # Preserve their numerical domain of dependence too, before that
        # subtraction, so no artificial boundary enters a receiving cell.
        # Use one global temporal stepping plan on every tile.
        nt = len(self.model.clock)
        maximum, change = np.zeros(nt), np.zeros(nt - 1)
        for first in range(0, self.model.n, 16384):
            check_cancelled()
            indices = np.arange(first, min(self.model.n, first + 16384))
            *_, speed = self.fields(p, indices)
            maximum = np.maximum(maximum, np.max(speed, axis=1))
            change = np.maximum(change, np.max(
                np.abs(speed[1:] - speed[:-1]) / (speed[1:] + speed[:-1]), axis=1))
        self.steps = np.maximum(1, np.ceil(2. * change).astype(int))
        reach = float(np.maximum(maximum[:-1], maximum[1:]) @ np.diff(self.model.clock))
        self.halo = int(np.ceil(max(self.radius, reach) / self.dx)) + 2 * int(self.steps.sum()) + 8

    def add_gradient(self, gradient, local, q, state, derivative, indices):
        g = local.gradient(derivative, q, state)
        parent = self.model
        if parent.form == "xt":
            # Undo the tile's mean-centering before accumulating overlaps;
            # enforce the single global identifiability constraint afterward.
            np.add.at(gradient, indices, g[:local.n] + g[local.amplitude_index] / local.n)
        elif parent.form == "x":
            np.add.at(gradient, indices, g[:local.n])
        elif parent.form == "scalar":
            gradient[0] += g[0]
        if parent.temporal:
            gradient[parent.amplitude_index] += g[local.amplitude_index]
        if parent.shape_index is not None:
            if parent.shape_count > 1:
                np.add.at(gradient, parent.shape_slice.start + indices, g[local.shape_slice])
            else:
                gradient[parent.shape_slice] += g[local.shape_slice]
        if parent.baseline_index is not None:
            gradient[parent.baseline_index] += g[local.baseline_index]

    def evaluate(self, p, need_gradient, enforce_completion, completion_tolerance):
        from ._kinetic_fit import _trapz_weights

        self.prepare_projection(p)
        self.prepare_geometry(p)
        n = self.model.n
        weights = _trapz_weights(self.model.clock)
        prediction = np.zeros_like(self.observed)
        timing = np.empty(n)
        final_survival, final_rate, sensitivity = np.empty(n), np.empty(n), np.empty(n)
        gradient = np.zeros(self.model.count)
        tail_bounds, quadrature_errors = np.zeros(n), np.zeros(n)
        # A mean-fraction observation couples all receivers. Obtain that one
        # temporal residual first, then apply its adjoint in a second pass.
        passes = 2 if need_gradient and self.mean_fraction else 1
        tile_count = sum(1 for _ in self.tiles())
        for iteration in range(passes):
            for index, (indices, core, destination, periodic) in enumerate(self.tiles()):
                check_cancelled()
                local, q, fitted, state, rate, speed = self.fields(p, indices)
                if self.quadrature:
                    from ._timing_quadrature import StationaryTiming

                    receivers = np.arange(len(indices))[core]
                    operator = StationaryTiming(
                        speed[0], self.model.clock[-1]*self.typical[destination]/np.max(self.typical),
                        resolution=self.dx, periodic=periodic, radius=self.radius,
                        receivers=receivers, tolerance=self.quadrature_tolerance[destination],
                    )
                    predicted = operator.predict(rate[0], record=need_gradient) + self.offset
                    prediction[destination] = timing[destination] = predicted
                    final_survival[destination] = operator.final_survival
                    final_rate[destination] = rate[0, core]
                    sensitivity[destination] = operator.sensitivity
                    tail_bounds[destination] = operator.tail_bound
                    quadrature_errors[destination] = operator.quadrature_error
                    backend_name = "tiled_" + operator.backend
                    if need_gradient:
                        residual = np.where(self.valid[destination], predicted-self.observed[destination], 0.)
                        if self.observation_weights is not None:
                            residual = residual*self.observation_weights[destination]
                        gi, gv = operator.backward_timing(
                            residual*(2./(self.count*self.scale_loss)),
                            with_speed=self.model.target == "fork_speed")
                        derivative = (gi if self.model.target == "initiation_rate" else gv)[None, :]
                        self.add_gradient(gradient, local, q, state, derivative, indices)
                    del operator
                    continue
                operator = self.operator(speed, periodic)
                backend_name = "tiled_" + getattr(operator, "backend", "hazard")
                differentiate = need_gradient and (not self.mean_fraction or iteration == 1)
                start = (iteration * tile_count + index) / (passes * tile_count)
                width = 1. / (passes * tile_count)
                middle = start + width * (.5 if differentiate else 1.)
                with measured_work_phase(start, middle):
                    hazard = operator.forward(rate, record=differentiate)
                survival = np.exp(-np.maximum(hazard, 0.))
                current = survival[:, core]
                if iteration == 0:
                    timing[destination] = weights @ current + self.offset
                    final_survival[destination] = current[-1]
                    final_rate[destination] = rate[-1, core]
                    sensitivity[destination] = np.einsum("t,tx,tx->x", weights, hazard[:, core], current)
                    if self.observation == "timing":
                        from ._kinetic_tail import kinetic_timing_bound

                        supplied = np.asarray(self.supplied)
                        if supplied.ndim and supplied.shape[0] == n:
                            supplied = supplied[indices]
                        tail_bounds[destination] = kinetic_timing_bound(
                            local, q, state, rate, speed, survival[-1], supplied,
                            self.native_clock, self.dt, self.dx, self.radius, periodic)[core]
                        prediction[destination] = timing[destination]
                    elif self.mean_fraction:
                        prediction.reshape(-1)[:] += (-np.expm1(-np.maximum(hazard[:, core], 0.))).sum(axis=1) / n
                    else:
                        prediction[destination] = (hazard[:, core].T if self.observation == "hazard"
                                                   else -np.expm1(-np.maximum(hazard[:, core].T, 0.)))
                if not differentiate:
                    continue
                if self.observation == "timing":
                    residual = np.where(self.valid[destination], prediction[destination] - self.observed[destination], 0.)
                    if self.observation_weights is not None:
                        residual = residual*self.observation_weights[destination]
                    local_derivative = -weights[:, None] * current * residual[None, :]
                elif self.mean_fraction:
                    residual = np.where(self.valid, prediction - self.observed, 0.).reshape(-1)
                    if self.observation_weights is not None:
                        residual = residual*self.observation_weights.reshape(-1)
                    local_derivative = current * residual[:, None] / n
                else:
                    residual = np.where(self.valid[destination], prediction[destination] - self.observed[destination], 0.)
                    if self.observation_weights is not None:
                        residual = residual*self.observation_weights[destination]
                    local_derivative = residual.T if self.observation == "hazard" else current * residual.T
                d_hazard = np.zeros_like(hazard)
                d_hazard[:, core] = local_derivative * (2. / (self.count * self.scale_loss))
                options = {"with_speed": self.model.target == "fork_speed"}
                from ._hazard_transport import FastTransportHazard
                from ._hazard_stationary_source import StationarySourceHazard
                if isinstance(operator, (FastTransportHazard, StationarySourceHazard)):
                    options["with_rate"] = self.model.target == "initiation_rate"
                with measured_work_phase(middle, start + width):
                    rate_gradient, speed_gradient = operator.backward(d_hazard, **options)
                derivative = rate_gradient if self.model.target == "initiation_rate" else speed_gradient
                self.add_gradient(gradient, local, q, state, derivative, indices)
                # Do not retain another tile's forward/reverse histories.
                del operator, hazard, survival, d_hazard, derivative
        if self.model.form == "xt":
            gradient[:n] -= gradient[:n].mean()
        if self.observation == "timing":
            if not np.all(eventual_sources(final_rate, self.dx, self.radius, self.periodic)):
                if not enforce_completion:
                    raise ValueError("The kinetics or region transforms allow no continuing initiation within the influence radius of some positions; a finite mean replication time cannot be fitted.")
                invalid = True
            else:
                invalid = enforce_completion and np.any(tail_bounds > self.quadrature_tolerance)
            if invalid:
                return {"loss": np.inf, "mse": np.inf, "gradient": np.zeros(self.model.count)}
        residual = prediction - self.observed
        squared = np.where(self.valid, residual, 0.)**2
        mse = float(np.sum(squared if self.observation_weights is None else
                           squared*self.observation_weights) / self.count)
        result = dict(loss=mse / self.scale_loss, mse=mse, gradient=gradient,
                    fitted=None, survival=final_survival[None, :], prediction=prediction,
                    residual=residual, hazard=None, sensitivity=sensitivity,
                    timing_prediction=timing, final_rate=final_rate,
                    positive_rate=bool(np.any(final_rate > 0)), spatial_groups=None,
                    backend=backend_name,
                    tiled_parameters=None if need_gradient else p.copy())
        if self.observation == "timing":
            result["timing_tail_bound"] = tail_bounds
        if self.quadrature:
            result["quadrature_error"] = quadrature_errors
        return result
