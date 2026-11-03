"""Adaptive timing integrals and exact discrete adjoints for static kinetics.

The spatial cone is integrated exactly within source cells. Embedded
Gauss--Kronrod panels integrate survival directly, without a PDE clock or a
space-by-time history. All panels retain the full coupled spatial adjoint.
"""

import numpy as np

from ._hazard_static import StationaryHazard
from .utils import check_cancelled


_POSITIVE = np.array([.9980022986933971, .9879925180204854, .9677390756791391,
    .9372733924007060, .8972645323440819, .8482065834104272, .7904185014424659,
    .7244177313601700, .6509967412974170, .5709721726085388, .4850818636402397,
    .3941513470775634, .2991800071531688, .2011940939974345, .1011420669187175])
_HALF_WEIGHTS = np.array([.005377479872923349, .015007947329316123,
    .02546084732671532, .035346360791375846, .044589751324764876,
    .053481524690928087, .06200956780067064, .06985412131872826,
    .07684968075772038, .08308050282313302, .08856444305621177,
    .09312659817082532, .09664272698362368, .09917359872179196,
    .1007698455238756])
_X = np.r_[_POSITIVE, 0., -_POSITIVE[::-1]]
_K = np.r_[_HALF_WEIGHTS, .10133000701479155, _HALF_WEIGHTS[::-1]]
_G = np.zeros(31)
_G[1::2] = np.polynomial.legendre.leggauss(15)[1]
_ORDER = len(_X)


class StationaryTiming(StationaryHazard):
    backend = "adaptive_static_timing"

    def __init__(self, speed, horizon, *, resolution=1., periodic=False,
                 radius=2000., receivers=None, tolerance=None, native=True):
        super().__init__(speed, np.array([0., 1.]), resolution, periodic, radius)
        self.receivers = (np.arange(self.n) if receivers is None else
                          np.asarray(receivers, dtype=int))
        if self.receivers.ndim != 1 or np.any(self.receivers < 0) or np.any(self.receivers >= self.n):
            raise ValueError("Timing receivers must lie within the selected spatial grid.")
        self.horizon = np.broadcast_to(horizon, (len(self.receivers),)).copy()
        self.tolerance = np.broadcast_to(
            np.maximum(self.horizon * 1e-7, 1e-9) if tolerance is None else tolerance,
            self.horizon.shape,
        )
        if np.any(~np.isfinite(self.horizon)) or np.any(self.horizon <= 0.):
            raise ValueError("Timing quadrature horizons must be finite and positive.")
        if np.any(~np.isfinite(self.tolerance)) or np.any(self.tolerance <= 0.):
            raise ValueError("Timing quadrature tolerances must be finite and positive.")
        self.panels = None
        self.tape = None
        from ._timing_native import timing_library

        self.native = timing_library() if native else None

    def _points(self, times, receivers, *, geometry=False):
        if self.native is not None and not geometry:
            from ._timing_native import points

            return points(self.native, self, times, receivers), None, None
        yc = self.yc[receivers]
        centers = self.centers[receivers]
        left = self._prefix(np.maximum(centers-self.cap, self._inverse(yc-times)))
        right = self._prefix(np.minimum(centers+self.cap, self._inverse(yc+times)))
        mc, qc = self.center_prefix[0][receivers], self.center_prefix[1][receivers]
        distance = yc * (2*mc-left[0]-right[0]) + left[1]+right[1]-2*qc
        hazard = np.maximum(times*(right[0]-left[0])-distance, 0.)
        return hazard, left, right

    def predict(self, rate, *, record=True):
        check_cancelled()
        self.rate = np.array(rate, dtype=float, copy=True)
        if self.rate.shape != (self.n,) or np.any(~np.isfinite(self.rate)) or np.any(self.rate < 0.):
            raise ValueError("Stationary initiation must be finite and non-negative.")
        self.mass = np.r_[0., np.cumsum(self.rate * self.dx)]
        self.moment = np.r_[0., np.cumsum(self.rate * (
            self.y[:-1]*self.dx + .5*self.slow*self.dx**2))]
        self.center_prefix = self._prefix(self.centers)
        count = len(self.receivers)
        if self.panels is None:
            pending = (np.arange(count), np.zeros(count), self.horizon.copy())
        else:
            pending = tuple(part.copy() for part in self.panels)
        accepted = [[], [], []]
        samples, sample_times, sample_weights, sample_hazards = [], [], [], []
        integral, estimate, sensitivity = np.zeros(count), np.zeros(count), np.zeros(count)
        for depth in range(24):
            check_cancelled()
            destination, lo, hi = pending
            if not len(destination):
                break
            half = .5*(hi-lo)
            times = .5*(hi+lo)[:, None] + half[:, None]*_X
            actual_receivers = np.tile(self.receivers[destination], _ORDER)
            hazard, _, _ = self._points(times.T.reshape(-1), actual_receivers)
            hazard = hazard.reshape(_ORDER, -1).T
            survival = np.exp(-hazard)
            kronrod = half*(survival @ _K)
            error = np.abs(half*(survival @ (_K-_G)))
            local_tolerance = self.tolerance[destination]*(hi-lo)/self.horizon[destination]
            keep = error <= local_tolerance
            if depth == 23 and not np.all(keep):
                raise ValueError("Timing quadrature did not reach its requested accuracy.")
            weights = half[keep, None]*_K
            selected = destination[keep]
            np.add.at(integral, selected, kronrod[keep])
            np.add.at(estimate, selected, error[keep])
            np.add.at(sensitivity, selected, np.sum(weights*survival[keep]*hazard[keep], axis=1))
            for bucket, values in zip(accepted, (selected, lo[keep], hi[keep])):
                bucket.append(values)
            if record:
                samples.append(np.tile(selected, _ORDER))
                sample_times.append(times[keep].T.reshape(-1))
                sample_weights.append(weights.T.reshape(-1))
                sample_hazards.append(hazard[keep].T.reshape(-1))
            failed = ~keep
            middle = .5*(lo[failed]+hi[failed])
            pending = (np.repeat(destination[failed], 2),
                       np.column_stack((lo[failed], middle)).reshape(-1),
                       np.column_stack((middle, hi[failed])).reshape(-1))
        self.panels = tuple(np.concatenate(bucket) for bucket in accepted)
        end_hazard, left, right = self._points(self.horizon, self.receivers, geometry=True)
        survival_end = np.exp(-end_hazard)
        slope = np.maximum(right[0]-left[0], 0.)
        # H is convex in time for stationary non-negative initiation.
        # Consequently the survival tangent gives a certified integral bound.
        upper = np.divide(survival_end, slope, out=np.full(count, np.inf), where=slope > 0.)
        centers, yc = self.centers[self.receivers], self.yc[self.receivers]
        lfull = np.maximum(centers-self.cap, 0.) if not self.periodic else centers-self.cap
        rfull = np.minimum(centers+self.cap, self.length) if not self.periodic else centers+self.cap

        def travel_coordinate(position):
            turns = np.floor(position/self.length) if self.periodic else 0.
            local = position-turns*self.length
            cell = np.clip((local/self.dx).astype(int), 0, self.n-1)
            return turns*self.y[-1] + self.y[cell] + (local-cell*self.dx)*self.slow[cell]

        closed = self.horizon >= np.maximum(yc-travel_coordinate(lfull), travel_coordinate(rfull)-yc)
        exact_tail = np.where(closed, upper, 0.)
        integral += exact_tail
        sensitivity += exact_tail*(end_hazard+1.)
        self.tail_bound = np.where(closed, 0., upper)
        self.quadrature_error = estimate
        self.final_survival = survival_end
        self.sensitivity = sensitivity
        self.tape = (np.concatenate(samples), np.concatenate(sample_times),
                     np.concatenate(sample_weights), np.concatenate(sample_hazards),
                     exact_tail, slope) if record else None
        return integral

    def backward_timing(self, weights, *, with_speed=True):
        if self.tape is None:
            raise RuntimeError("Record a timing prediction before differentiating it.")
        if self.native is not None:
            return self._native_backward(weights, with_speed)
        destination, times, quadrature, hazard, tail, slope = self.tape
        receivers = self.receivers[destination]
        g = -np.asarray(weights)[destination]*quadrature*np.exp(-hazard)
        _, left, right = self._points(times, receivers, geometry=True)
        gm, gq, gy = np.zeros(self.n+1), np.zeros(self.n+1), np.zeros(self.n+1)
        gi, gw, receiver = np.zeros(self.n), np.zeros(self.n), np.zeros(self.n)

        def accumulate(mass_gradient, moment_gradient, record):
            cell, h, turns, mass, qpart = record
            lm = mass_gradient + turns*self.y[-1]*moment_gradient
            gm[:] += np.bincount(cell, lm, minlength=self.n+1)
            gq[:] += np.bincount(cell, moment_gradient, minlength=self.n+1)
            gm[-1] += np.sum(turns*mass_gradient + .5*turns*(turns-1)*self.y[-1]*moment_gradient)
            gq[-1] += np.sum(turns*moment_gradient)
            gi[:] += np.bincount(cell, h*lm+qpart*moment_gradient, minlength=self.n)
            if with_speed:
                gy[:-1] += np.bincount(cell, self.rate[cell]*h*moment_gradient, minlength=self.n)
                gw[:] += np.bincount(cell, .5*self.rate[cell]*h*h*moment_gradient, minlength=self.n)
                gy[-1] += np.sum((.5*turns*(turns-1)*self.mass[-1]+turns*mass)*moment_gradient)

        center_mass, center_moment = np.zeros(self.n), np.zeros(self.n)

        def hazard_pullback(g, times, receivers, left, right):
            yc = self.yc[receivers]
            accumulate(g*(-times+yc), -g, left[2])
            accumulate(g*(times+yc), -g, right[2])
            center_mass[:] += np.bincount(receivers, -2*g*yc, minlength=self.n)
            center_moment[:] += np.bincount(receivers, 2*g, minlength=self.n)
            if with_speed:
                receiver[:] += np.bincount(receivers, g*(left[0]+right[0]-2*self.center_prefix[0][receivers]), minlength=self.n)

        hazard_pullback(g, times, receivers, left, right)
        _, left, right = self._points(self.horizon, self.receivers, geometry=True)
        hazard_pullback(-np.asarray(weights)*tail, self.horizon, self.receivers, left, right)
        mass_tail = np.divide(-np.asarray(weights)*tail, slope, out=np.zeros_like(tail), where=slope > 0.)
        accumulate(-mass_tail, np.zeros_like(tail), left[2])
        accumulate(mass_tail, np.zeros_like(tail), right[2])
        accumulate(center_mass, center_moment, self.center_prefix[2])
        return self._finish(gm, gq, gy, gi, gw, receiver, with_speed)

    def _native_backward(self, weights, with_speed):
        from ._timing_native import pullback

        destination, times, quadrature, hazard, tail, slope = self.tape
        receivers = self.receivers[destination]
        g = -np.asarray(weights)[destination]*quadrature*np.exp(-hazard)
        buffers = [np.zeros(self.n+1) for _ in range(3)] + [np.zeros(self.n) for _ in range(3)]
        pullback(self.native, self, times, receivers, g, np.zeros_like(g), buffers, with_speed)
        g_mass = np.divide(-np.asarray(weights)*tail, slope, out=np.zeros_like(tail), where=slope > 0.)
        pullback(self.native, self, self.horizon, self.receivers,
                 -np.asarray(weights)*tail, g_mass, buffers, with_speed)
        return self._finish(*buffers, with_speed)

    def _finish(self, gm, gq, gy, gi, gw, receiver, with_speed):
        qm = np.cumsum(gq[:0:-1])[::-1]
        gi += self.dx*np.cumsum(gm[:0:-1])[::-1]
        gi += qm*(self.y[:-1]*self.dx+.5*self.slow*self.dx**2)
        if not with_speed:
            return gi, None
        gy[:-1] += qm*self.rate*self.dx + receiver
        gw += .5*qm*self.rate*self.dx**2 + .5*self.dx*receiver
        gw += self.dx*np.cumsum(gy[:0:-1])[::-1]
        return gi, -gw/self.speed**2
