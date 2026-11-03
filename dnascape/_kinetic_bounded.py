"""Memory-bounded timing objectives using the existing exact kinetic kernels.

Only storage and the order of linear operations change. The native time/space
grid, temporal hats, centred source cells, objective and adjoint are unchanged.
"""

from copy import copy

import numpy as np
from scipy.fft import irfft, irfftn, next_fast_len, rfft, rfftn

from ._resources import MIB, ensure_capacity, memory_status
from .utils import check_cancelled, report_measured_work


class BoundedTimingFit:
    def __init__(self, model, observed, valid, offset, scale, dx, periodic, radius):
        from ._kinetic_fit import _trapz_weights

        self.model, self.observed, self.valid = model, observed, valid
        self.offset, self.scale = offset, scale
        self.dx, self.periodic, self.radius = dx, periodic, radius
        self.weights = _trapz_weights(model.clock)
        self.count = int(valid.sum())
        ensure_capacity(model.n * 80 + len(model.clock) * 160,
                        "Preparing the Mapping grid")

    def result(self, p, prediction, final, sensitivity, backend, state):
        residual = prediction - self.observed
        mse = float(np.sum(np.where(self.valid, residual, 0.) ** 2) / self.count)
        return dict(loss=mse / self.scale, mse=mse,
                    gradient=np.zeros(self.model.count), fitted=None, hazard=None,
                    survival=final[None, :], prediction=prediction, residual=residual,
                    sensitivity=sensitivity, positive_rate=True, spatial_groups=None,
                    backend=backend, bounded_state=(p.copy(), state))

    def fields(self, p, start, stop, *, need_state=True):
        check_cancelled()
        local = copy(self.model)
        local.clock = self.model.clock[start:stop]
        fitted, state = local.evaluate(p, need_state=need_state)
        return local, fitted, state

    def public(self, p):
        if not self.model.temporal:
            return self.model.public(self.model.evaluate(p, need_state=False)[0])
        # Materialise only the public result, never all basis/derivative fields.
        n, nt = self.model.n, len(self.model.clock)
        rows = n if self.model.form == "xt" else 1
        columns = max(self.model.delay.size, self.model.shape_count, rows)
        ensure_capacity(rows * nt * 8 + columns * 64 * 80, "Preparing the Mapping result")
        result = np.empty((rows, nt))
        for start in range(0, nt, 64):
            check_cancelled()
            _, fitted, _ = self.fields(p, start, start + 64, need_state=False)
            result[:, start:start + 64] = fitted[:, :rows].T
            del fitted
        return result

    def complete(self, result):
        p, state = result.pop("bounded_state")
        result["gradient"] += self.adjoint(p, result, state)


class StationaryConvolutionTiming(BoundedTimingFit):
    """Retain the fast spatial FFT, without retaining XT objective arrays."""

    def __init__(self, *args, speed=None, rate=None):
        super().__init__(*args)
        self.rate = rate
        self.operator = None
        self.operator_parameters = None
        if speed is not None:
            self.prepare_operator(speed)

    def prepare_operator(self, speed):
        from ._hazard_stationary_source import StationarySourceHazard

        self.operator = None
        self.operator = StationarySourceHazard(
            speed, self.model.clock, self.dx, self.periodic, self.radius,
            differentiate_speed=False,
        )
        op = self.operator
        status = memory_status()
        temporary = 32 * MIB if status is None else min(32 * MIB, status.available // 2)
        per_row = op.length * 64 + (op.width + 1) * 160
        self.block = max(1, min(64, temporary // per_row))
        ensure_capacity(self.block * per_row, "Mapping convolution blocks")
        spectrum_bytes = len(op.t) * (op.length // 2 + 1) * 16
        # Keep the entire reusable kernel when it fits. Otherwise retain no
        # partial cache that would thrash during successive forward passes.
        self.cache_kernel = status is None or spectrum_bytes + self.block * per_row < status.available
        if self.cache_kernel:
            op._kernel()
        else:
            op._prepare_primitives()

    def source(self, p):
        if self.rate is None:
            return self.model.evaluate(p, need_state=False)[0]
        if self.operator_parameters is None or not np.array_equal(self.operator_parameters, p):
            speed, _ = self.model.evaluate(p, need_state=False)
            self.prepare_operator(speed)
            self.operator_parameters = p.copy()
        return self.rate

    def kernel(self, selected):
        if self.cache_kernel:
            return self.operator._spectrum[selected]
        return self.operator._kernel_block(selected)

    def blocks(self, p):
        fitted = self.source(p)
        op = self.operator
        source = rfft(fitted[0], n=op.length)
        for start in range(0, len(op.t), self.block):
            check_cancelled()
            selected = slice(start, start + self.block)
            kernel = self.kernel(selected)
            hazard = irfft(kernel * source, n=op.length, axis=1)[:, :op.n]
            np.maximum(hazard, 0., out=hazard)
            yield selected, kernel, hazard

    def evaluate(self, p, need_gradient):
        n = self.model.n
        prediction, sensitivity = np.full(n, self.offset), np.zeros(n)
        for selected, _, hazard in self.blocks(p):
            survival = np.exp(-hazard)
            prediction += np.einsum("t,tx->x", self.weights[selected], survival)
            sensitivity += np.einsum("t,tx,tx->x", self.weights[selected], hazard, survival)
            final = survival[-1].copy()
            report_measured_work(.5 * min(selected.stop, len(self.weights)) / len(self.weights))
        result = self.result(p, prediction, final, sensitivity,
                             "blocked_stationary_source_convolution", None)
        if need_gradient:
            self.complete(result)
        return result

    def adjoint(self, p, result, state):
        if self.rate is not None:
            return self.speed_adjoint(p, result)
        op = self.operator
        residual = np.where(self.valid, result["residual"], 0.) * (2. / (self.count * self.scale))
        spectrum = np.zeros(op.length // 2 + 1, dtype=complex)
        for selected, kernel, hazard in self.blocks(p):
            gradient = np.exp(-hazard)
            gradient *= -self.weights[selected, None] * residual
            gradient *= hazard > 0.
            g = rfft(gradient, n=op.length, axis=1)
            spectrum += np.einsum("tf,tf->f", g, kernel.conj())
            report_measured_work(.5 + .5 * min(selected.stop, len(self.weights)) / len(self.weights))
        derivative = irfft(spectrum, n=op.length)[:op.n][None, :]
        return self.model.gradient(derivative, p, self.model.evaluate(p)[1])

    def speed_adjoint(self, p, result):
        self.source(p)
        op = self.operator
        residual = np.where(self.valid, result["residual"], 0.) * (2. / (self.count * self.scale))
        source = rfft(self.rate[0], n=op.length)
        distance = np.arange(op.width + 1) * op.dx
        a, b = np.maximum(0., distance - .5 * op.dx), np.minimum(op.cap, distance + .5 * op.dx)
        factor = np.ones(op.width + 1)
        factor[0] = 2.
        gd, gj, gv = (np.zeros(len(op.t)) for _ in range(3))
        for selected, _, hazard in self.blocks(p):
            g = np.exp(-hazard)
            g *= -self.weights[selected, None] * residual
            g *= hazard > 0.
            spectrum = rfft(g, n=op.length, axis=1)
            kg = irfft(spectrum * source.conj(), n=op.length, axis=1)
            ring = kg[:, :op.width + 1].copy()
            if op.width:
                ring[:, 1:] += kg[:, (-np.arange(1, op.width + 1)) % op.length]
            left, lr = op._primitive(op.travel[selected, None] - a)
            right, rr = op._primitive(op.travel[selected, None] - b)
            ring *= factor * (left > right) * (b > a)
            for sign, record in ((1., lr), (-1., rr)):
                index, s, a0, a1, c0, c1, active = record
                q = sign * ring * active
                sigma = op.t[index] + s
                gd[selected] += np.sum(q * sigma, axis=1)
                gd += np.bincount(index.ravel(), (-q * sigma).ravel(), minlength=len(op.t))
                gj += np.bincount(index.ravel(), q.ravel(), minlength=len(op.t))
                gv += np.bincount(index.ravel(), (q * ((op.t[index] - sigma) * a0 + c0)).ravel(), minlength=len(op.t))
                gv += np.bincount((index + 1).ravel(), (q * ((op.t[index] - sigma) * a1 + c1)).ravel(), minlength=len(op.t))
            report_measured_work(.5 + .5 * min(selected.stop, len(op.t)) / len(op.t))
        d, j = np.cumsum(gd[:0:-1])[::-1], np.cumsum(gj[:0:-1])[::-1]
        gv[:-1] += .5 * op.dt * d + j * (.5 * op.t[:-1] * op.dt + op.dt**2 / 6.)
        gv[1:] += .5 * op.dt * d + j * (.5 * op.t[:-1] * op.dt + op.dt**2 / 3.)
        return self.model.gradient(gv[:, None], p, self.model.evaluate(p)[1])


class UniformTiming(BoundedTimingFit):
    """Evaluate distinct boundary histories in small spatial batches."""

    def __init__(self, *args, supplied, native_clock, dt):
        super().__init__(*args)
        self.supplied, self.native_clock, self.dt = supplied, native_clock, dt

    def blocks(self, p):
        from ._hazard_uniform import UniformKineticsHazard
        from ._kinetic_fit import _supplied_on_clock

        fitted, state = self.model.evaluate(p)
        fixed = _supplied_on_clock(self.supplied, self.model.clock, self.native_clock,
                                   self.model.n, self.dt, "supplied kinetics",
                                   positive=self.model.target == "initiation_rate")
        rate, speed = (fitted, fixed) if self.model.target == "initiation_rate" else (fixed, fitted)
        geometry = UniformKineticsHazard(speed, self.model.clock, self.dx,
                                        self.periodic, self.radius, compress=True)
        status = memory_status()
        available = 32 * MIB if status is None else min(32 * MIB, status.available // 2)
        width = max(1, min(128, available // (max(1, len(self.weights)) * 400)))
        ensure_capacity(width * len(self.weights) * 400, "Mapping uniform kinetic histories")
        for start in range(0, len(geometry.left), width):
            check_cancelled()
            selected = slice(start, start + width)
            op = copy(geometry)
            distances, index = np.unique(np.r_[geometry.left[selected], geometry.right[selected]], return_inverse=True)
            size = len(geometry.left[selected])
            op.distances = geometry.distances[distances]
            op.left, op.right = index[:size], index[size:]
            yield selected, op, rate, state, geometry.groups

    def evaluate(self, p, need_gradient):
        prediction = np.empty(self.model.n)
        sensitivity, final = np.empty_like(prediction), np.empty_like(prediction)
        gradient = np.zeros(self.model.count)
        for selected, op, rate, state, groups in self.blocks(p):
            hazard = op.forward(rate, record=need_gradient)
            survival = np.exp(-hazard)
            small = np.einsum("t,tx->x", self.weights, survival) + self.offset
            receiving = (groups >= selected.start) & (groups < selected.stop)
            group = groups[receiving] - selected.start
            prediction[receiving] = small[group]
            final[receiving] = survival[-1, group]
            sensitivity[receiving] = np.einsum("t,tx,tx->x", self.weights, hazard, survival)[group]
            if need_gradient:
                residual = np.where(self.valid[receiving], prediction[receiving] - self.observed[receiving], 0.)
                q = np.bincount(group, residual, minlength=len(small))
                d_hazard = survival * (-self.weights[:, None] * q[None, :])
                d_hazard *= 2. / (self.count * self.scale)
                gi, gv = op.backward(d_hazard, with_speed=self.model.target == "fork_speed")
                gradient += self.model.gradient(gi if self.model.target == "initiation_rate" else gv, p, state)
            report_measured_work(min(selected.stop, int(groups.max()) + 1) / (int(groups.max()) + 1))
        result = self.result(p, prediction, final, sensitivity, "blocked_uniform_kinetic_primitives", None)
        result["gradient"] = gradient
        if need_gradient:
            result.pop("bounded_state")
        return result

    def adjoint(self, p, result, state):
        return self.evaluate(p, True)["gradient"]


class ConeConvolutionTiming(BoundedTimingFit):
    """Overlap-save FFT of the SAME temporal-hat cone kernel.

    The temporal difference of the cone is zero after the influence radius
    has been traversed. Convolving that finite impulse and cumulatively
    summing gives the original cone, including its half hat at time zero.
    Its transpose is a reverse cumulative sum followed by correlation.
    Neither operation needs a full XT field or full-history padded FFT.
    """

    def __init__(self, *args, speed, dt):
        super().__init__(*args)
        self.v, self.dt = float(speed), float(dt)
        if not np.isfinite(self.v) or self.v <= 0:
            raise ValueError("Fork speed must contain finite positive values.")
        n, nt = self.model.n, len(self.model.clock)
        cap = min(self.radius, .5 * n * self.dx) if self.periodic else self.radius
        self.width = min(n // 2 if self.periodic else n - 1,
                         int(np.ceil(min(cap, self.v * (self.model.clock[-1] + dt)) / self.dx + .5)))
        distance = np.arange(self.width + 1) * self.dx
        self.a = np.maximum(0., distance - .5 * self.dx)
        self.b = np.minimum(cap, distance + .5 * self.dx)
        reach = min(cap, (self.width + .5) * self.dx)
        self.support = min(nt, int(np.ceil(reach / (self.v * dt))) + 2)
        self.length = n if self.periodic else next_fast_len(n + self.width)
        # One kernel, one transform/inverse buffer, rate/Q overlap and small
        # basis batches. Unlike the old 192*N*T rule this has finite history.
        status = memory_status()
        for self.block in (128, 64, 32, 16, 8, 1):
            self.shape = next_fast_len(self.block + self.support - 1), self.length
            estimate = (np.prod(self.shape) * 40 + (self.support + self.block) * n * 16
                        + min(self.block, 32) * n * 96)
            if status is None or estimate < status.available:
                break
        ensure_capacity(estimate, "Mapping with bounded cone transforms")
        kernel = np.empty((self.support, self.length))
        for start in range(0, self.support, self.block):
            check_cancelled()
            stop = min(start + self.block, self.support)
            kernel[start:stop] = self.impulse(np.arange(start, stop))
        self.spectrum = rfftn(kernel, s=self.shape)
        self._rate_window = None

    def cone(self, indices, half=False):
        from ._hazard_fft import _hat_primitive

        age = np.maximum(indices, 0)[:, None] * self.dt
        qa, qb = age - self.a / self.v, age - self.b / self.v
        pa, _ = _hat_primitive(qa, self.dt, half)
        pb, _ = _hat_primitive(qb, self.dt, half)
        weights = self.v * (pa - pb)
        np.copyto(weights, (.5 if half else 1.) * self.dt * (self.b - self.a),
                  where=qb >= (0. if half else self.dt))
        weights[:, self.b <= self.a] = 0.
        weights[:, 0] *= 2.
        weights[indices < 0] = 0.
        return weights

    def impulse(self, indices, half=False):
        ring = self.cone(indices, half) - self.cone(indices - 1, half)
        result = np.zeros((len(indices), self.length))
        result[:, :self.width + 1] = ring
        if self.width:
            result[:, -np.arange(1, self.width + 1)] += ring[:, 1:]
        return result

    def rates(self, p, start, stop):
        result = np.zeros((stop - start, self.model.n))
        for first in range(max(0, start), min(stop, len(self.weights)), 32):
            last = min(first + 32, stop, len(self.weights))
            _, fitted, _ = self.fields(p, first, last, need_state=False)
            result[first - start:last - start] = fitted
            del fitted
        return result

    def rate_window(self, p, start):
        left, right = start - self.support + 1, start + self.block
        previous = self._rate_window
        if previous is not None and start - previous[0] == self.block:
            history = previous[1]
            history[:-self.block] = history[self.block:]
            history[-self.block:] = self.rates(p, right - self.block, right)
        elif previous is not None and previous[0] - start == self.block:
            history = previous[1]
            history[self.block:] = history[:-self.block]
            history[:self.block] = self.rates(p, left, left + self.block)
        else:
            history = self.rates(p, left, right)
        self._rate_window = start, history
        return history

    def convolve(self, values, reverse=False):
        spectrum = rfftn(values, s=self.shape)
        if reverse:
            # Conjugate the disposable input instead of copying the complete
            # cached kernel merely to conjugate it.
            np.conjugate(spectrum, out=spectrum)
            spectrum *= self.spectrum
            np.conjugate(spectrum, out=spectrum)
        else:
            spectrum *= self.spectrum
        # SciPy can overwrite the complex input for the inverse; no second
        # product array or full-history zero padding is retained.
        return irfftn(spectrum, s=self.shape, overwrite_x=True)

    def hazard(self, p, start, stop, previous):
        history = self.rate_window(p, start)
        result = self.convolve(history)[self.support - 1:self.support - 1 + stop - start, :self.model.n].copy()
        if start < self.support:
            last = min(stop, self.support)
            half = rfft(self.impulse(np.arange(start, last), half=True), axis=1)
            initial = rfft(self.rates(p, 0, 1)[0], n=self.length)
            result[:last - start] -= irfft(half * initial, n=self.length, axis=1)[:, :self.model.n]
        np.cumsum(result, axis=0, out=result)
        result += previous
        return result

    def evaluate(self, p, need_gradient):
        n, nt = self.model.n, len(self.weights)
        self._rate_window = None
        prediction, sensitivity = np.full(n, self.offset), np.zeros(n)
        previous = np.zeros(n)
        boundaries = []
        for start in range(0, nt, self.block):
            check_cancelled()
            stop = min(nt, start + self.block)
            boundaries.append(previous)
            hazard = self.hazard(p, start, stop, previous)
            previous = hazard[-1].copy()
            np.maximum(hazard, 0., out=hazard)
            survival = np.exp(-hazard)
            prediction += np.einsum("t,tx->x", self.weights[start:stop], survival)
            sensitivity += np.einsum("t,tx,tx->x", self.weights[start:stop], hazard, survival)
            final = survival[-1].copy()
            del hazard, survival
            report_measured_work(.5 * stop / nt)
        self._rate_window = None
        result = self.result(p, prediction, final, sensitivity, "blocked_exact_cone_convolution", boundaries)
        if need_gradient:
            self.complete(result)
        return result

    def adjoint(self, p, result, boundaries):
        n, nt = self.model.n, len(self.weights)
        self._rate_window = None
        residual = np.where(self.valid, result["residual"], 0.) * (2. / (self.count * self.scale))
        future = np.zeros((self.support - 1, n))
        accumulated = np.zeros(n)
        initial_gradient = np.zeros(n)
        gradient = np.zeros(self.model.count)
        for block in range(len(boundaries) - 1, -1, -1):
            check_cancelled()
            start, stop = block * self.block, min(nt, (block + 1) * self.block)
            hazard = self.hazard(p, start, stop, boundaries[block])
            g = np.exp(-np.maximum(hazard, 0.))
            g *= hazard > 0.
            g *= -self.weights[start:stop, None] * residual
            q = np.cumsum(g[::-1], axis=0)[::-1]
            q += accumulated
            accumulated = q[0].copy()
            history = np.concatenate((q, future))
            derivative = self.convolve(history, reverse=True)[:stop - start, :n].copy()
            future = history[:self.support - 1].copy()
            if start < self.support:
                last = min(stop, self.support)
                half = rfft(self.impulse(np.arange(start, last), half=True), axis=1)
                weighted = rfft(q[:last - start], n=self.length, axis=1)
                initial_gradient -= irfft(np.einsum("tf,tf->f", weighted, half.conj()), n=self.length)[:n]
            if start == 0:
                derivative[0] += initial_gradient
            del hazard, g, q, history
            for first in range(start, stop, 32):
                last = min(stop, first + 32)
                local, _, state = self.fields(p, first, last)
                gradient += local.gradient(derivative[first - start:last - start], p, state)
                del local, state
            del derivative
            report_measured_work(.5 + .5 * (nt - start) / nt)
        self._rate_window = None
        return gradient
