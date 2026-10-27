"""Exact, compact Weibull fits for initiation that varies only in time.

With spatially uniform speed and activation, positions the same distance from
an end have identical predictions. Positions beyond the largest cone share one
prediction, too. Aggregate their observations, not their model or sampling grid.
The scalar amplitude derivative then needs no spatial convolution or adjoint.
"""

import numpy as np

from .utils import check_cancelled, report_measured_work


class TemporalWeibullModel:
    def __init__(self, target, mask, speed, time_grid, delay, resolution, periodic,
                 *, scalar_linear=False, cache_cells=1_000_000):
        self.target = np.asarray(target, dtype=float)
        self.mask = np.asarray(mask, dtype=bool)
        self.n = self.target.size
        self.t = np.asarray(time_grid, dtype=float)
        self.delay = float(delay)
        self.dx = float(resolution)
        self.periodic = bool(periodic)
        self.speed = np.asarray(speed, dtype=float).reshape(-1)
        self.scalar_linear = bool(scalar_linear)
        self.cache_cells = int(cache_cells)
        # Match WeibullCone's integration from zero for nonzero output starts.
        times, velocities = self.t, self.speed
        if times[0] > 0:
            times = np.insert(times, 0, 0.)
            velocities = np.insert(velocities, 0, velocities[0])
        self.travel_t = times
        self.travel = np.concatenate(([0.], np.cumsum(
            .5 * (velocities[:-1] + velocities[1:]) * np.diff(times))))
        self.distance = self.travel[-self.t.size:]
        radius = max(0, int(np.ceil(self.travel[-1] / self.dx)))
        self.radius = min(radius, self.n // 2 if periodic else self.n - 1)
        positions = np.arange(self.n)
        self.group_count = 1 if periodic else min((self.n + 1) // 2, self.radius + 1)
        self.groups = (np.zeros(self.n, dtype=np.intp) if periodic else
                       np.minimum(np.minimum(positions, self.n - 1 - positions), self.group_count - 1))
        self.count = np.bincount(self.groups[self.mask], minlength=self.group_count)
        sums = np.bincount(self.groups[self.mask], weights=self.target[self.mask],
                           minlength=self.group_count)
        self.mean = np.divide(sums, self.count, out=np.zeros_like(sums), where=self.count > 0)
        deviations = self.target[self.mask] - self.mean[self.groups[self.mask]]
        self.variance = float(deviations @ deviations)
        self._cache = None
        self._cache_key = None

    @classmethod
    def create(cls, target, mask, fork_speed, time_grid, delay, resolution,
               resolution_time, periodic):
        """Return None for spatially varying speed or activation."""
        from .mapping import _expand_space_time_field

        check_cancelled()
        delays = np.asarray(delay, dtype=float).reshape(-1)
        if not np.all(delays == delays[0]):
            return None
        speed = np.asarray(fork_speed, dtype=float)
        scalar = speed.size == 1
        if scalar:
            uniform = speed
        elif speed.ndim == 2 and speed.shape[0] == 1:
            uniform = speed
        elif speed.ndim == 1 and speed.size == len(target) and np.all(speed == speed[0]):
            uniform = speed[:1]
        elif speed.ndim == 2 and speed.shape[0] == len(target) and np.all(speed == speed[:1]):
            uniform = speed[:1]
        else:
            return None
        velocities = _expand_space_time_field(
            uniform, len(time_grid), 1, time_grid, resolution_time, 'fork_speed', positive=True,
        )[:, 0]
        return cls(target, mask, velocities, time_grid, delays[0], resolution, periodic,
                   scalar_linear=scalar and not periodic)

    def _basis_rows(self, shape, derivative):
        distances = np.arange(self.radius + 1, dtype=float) * self.dx
        positions = np.arange(self.group_count)
        left = np.minimum(positions, self.radius)
        right = np.minimum(self.n - 1 - positions, self.radius)
        multiplicity = np.full(self.radius + 1, 2.)
        multiplicity[0] = 1.
        if self.periodic and self.n % 2 == 0 and self.radius == self.n // 2:
            multiplicity[-1] = 1.
        for index, time in enumerate(self.t):
            check_cancelled()
            if self.scalar_linear:
                ages = np.maximum(time - self.delay - distances / self.speed[0], 0.)
            else:
                available = self.distance[index] - distances
                latest = np.interp(np.maximum(available, 0.), self.travel, self.travel_t)
                ages = np.maximum(latest - self.delay, 0.)
                ages[available < 0] = 0.
            kernel = ages ** shape
            kernels = [kernel]
            if derivative:
                log_kernel = np.zeros_like(kernel)
                positive = ages > 0
                log_kernel[positive] = kernel[positive] * np.log(ages[positive])
                kernels.append(log_kernel)
            rows = []
            for values in kernels:
                if self.periodic:
                    rows.append(np.array([self.dx * np.sum(values * multiplicity)]))
                else:
                    prefix = np.cumsum(values)
                    rows.append(self.dx * (prefix[left] + prefix[right] - values[0]))
            yield rows

    def basis_rows(self, shape, derivative=False):
        key = (float(shape), bool(derivative))
        if self._cache_key == key:
            for rows in zip(*self._cache):
                check_cancelled()
                yield rows
            return
        cache = [] if self.t.size * self.group_count * (1 + derivative) <= self.cache_cells else None
        for rows in self._basis_rows(shape, derivative):
            if cache is not None:
                cache.append(rows)
            yield rows
        self._cache = tuple(np.array(part) for part in zip(*cache)) if cache is not None else None
        self._cache_key = key if cache is not None else None

    def loss_grad(self, params, shape, weights, z0, prior=0., fit_shape=False):
        check_cancelled()
        params = np.asarray(params, dtype=float)
        alpha = float(np.exp(params[0]))
        k = float(np.exp(params[-1])) if fit_shape else float(shape)
        predicted = np.zeros(self.group_count)
        amplitude_gradient = np.zeros_like(predicted)
        shape_gradient = np.zeros_like(predicted) if fit_shape else None
        for index, (weight, rows) in enumerate(zip(weights, self.basis_rows(k, fit_shape))):
            report_measured_work(index / len(weights))
            volume = alpha * rows[0]
            survival = np.exp(-np.clip(volume, 0., 700.))
            weighted_survival = weight * survival
            predicted += weighted_survival
            amplitude_gradient -= weighted_survival * volume
            if fit_shape:
                shape_gradient -= weighted_survival * alpha * rows[1] * k
        residual = predicted - self.mean
        weighted_residual = self.count * residual
        data_loss = .5 * (float(weighted_residual @ residual) + self.variance)
        displacement = params[0] - z0[0]
        regularisation = .5 * prior * displacement ** 2
        gradient = [float(weighted_residual @ amplitude_gradient) + prior * displacement]
        if fit_shape:
            gradient.append(float(weighted_residual @ shape_gradient))
        return data_loss + regularisation, np.array(gradient), {
            '_temporal_params': params.copy(), '_timing_groups': predicted,
            'objective': data_loss + regularisation, 'data_loss': data_loss,
            'regularisation': regularisation, 'weibull_shape': k,
        }

    def result_fields(self, fitted, offset, floor, *, diagnostic_cells=1_000_000):
        from .fitting import _weibull_hazard_samples

        check_cancelled()
        fields = dict(fitted)
        params = fields.pop('_temporal_params')
        predicted = fields.pop('_timing_groups')[self.groups]
        alpha = float(np.exp(params[0]))
        shape = fields['weibull_shape']
        hazard = _weibull_hazard_samples([alpha], shape, self.t, [self.delay], floor)
        fields.update({
            'alpha': np.full(self.n, alpha), 'lambda': np.full(self.n, alpha ** (-1. / shape)),
            'initiation_rate': np.broadcast_to(hazard.T, (self.n, self.t.size)),
            'predicted_timing': predicted + offset, 'residual': predicted - self.target,
            'fork_speed': (float(self.speed[0]) if self.scalar_linear else
                           np.broadcast_to(self.speed, (self.n, self.t.size))),
            'weibull_delay': np.full(self.n, self.delay), 'time_grid': self.t,
            'target_shape': 'weibull',
            'forward_backend': ('constant_fft' if self.scalar_linear else
                                'periodic_fft' if self.periodic else 'temporal_fft'),
            'temporal_fit_groups': self.group_count,
        })
        # These optional internal space-time fields are not Mapping outputs.
        # Keep small return_info diagnostics; never allocate a chromosome-wide
        # dense grid just to return a single temporal curve.
        if self.n * self.t.size <= diagnostic_cells:
            volume = np.empty((self.n, self.t.size))
            survival = np.empty_like(volume)
            for index, rows in enumerate(self.basis_rows(shape)):
                volume[:, index] = alpha * rows[0][self.groups]
                survival[:, index] = np.exp(-np.clip(volume[:, index], 0., 700.))
            fields.update(extended_volume=volume, survival=survival)
        else:
            fields['spacetime_diagnostics_omitted'] = True
        return fields
