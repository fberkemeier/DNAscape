"""Conditions for a finite, unconditional mean replication time."""

import numpy as np


def remaining_timing_bound(survival, rate_floor, speed_floor, resolution=1.,
                           radius=2000., periodic=False, *, length=None, wait=0.):
    """Bound the remaining survival integral using continuing future sources.

    For any permitted radius r, new origins arrive within r/v_min. After
    that delay their integrated intensity grows at least at rate M_r.
    Thus the omitted mean time is at most S_end*(wait+r/v_min+1/M_r).
    Minimise this valid bound over nested neighbourhoods. Source densities
    are cell averages; periodic neighbourhoods count each source only once.
    ``wait`` permits a lower bound that starts after the last uploaded time.
    """
    survival, rates = np.broadcast_arrays(np.asarray(survival, float),
                                          np.asarray(rate_floor, float))
    shape = survival.shape
    rates, survival = rates.reshape(-1), survival.reshape(-1)
    dx = float(resolution)
    physical_length = rates.size*dx if length is None else float(length)
    cap = min(float(radius), physical_length/(2. if periodic else 1.))
    speed = float(np.min(speed_floor))
    if speed <= 0. or not np.isfinite(speed) or cap <= 0.:
        return np.where(survival == 0., 0., np.inf).reshape(shape)
    # A compressed uniform domain still has its original physical length.
    cell_width = physical_length/rates.size
    centers = (np.arange(rates.size)+.5)*cell_width
    edges = np.arange(rates.size+1)*cell_width
    prefix = np.r_[0., np.cumsum(rates)*cell_width]

    def primitive(x):
        if periodic:
            turns = np.floor(x/physical_length)
            return turns*prefix[-1]+np.interp(x-turns*physical_length, edges, prefix)
        return np.interp(x, edges, prefix)

    radii = np.minimum(.5*dx*2.**np.arange(
        max(1, int(np.ceil(np.log2(max(2.*cap/dx, 1.))))+1)), cap)
    factor = np.full(rates.size, np.inf)
    for r in np.unique(radii):
        mass = np.maximum(primitive(centers+r)-primitive(centers-r), 0.)
        duration = np.divide(1., mass, out=np.full_like(mass, np.inf), where=mass > 0.)
        factor = np.minimum(factor, float(wait)+r/speed+duration)
    return np.multiply(survival, factor, out=np.zeros_like(survival),
                       where=survival > 0.).reshape(shape)


def supplied_future_floor(field, end, native_clock, dt):
    """A per-position lower bound on a linearly interpolated, held-last input."""
    data = np.asarray(field, float)
    if data.ndim < 2 or data.shape[-1] == 1:
        return data.reshape(-1), 0., data.reshape(-1)
    times = (np.arange(data.shape[-1])*dt if native_clock is None else
             np.asarray(native_clock, float))
    if end >= times[-1]:
        last = data[:, -1]
        return last, 0., last
    index = max(0, int(np.searchsorted(times, end, side="right"))-1)
    fraction = np.clip((end-times[index])/(times[index+1]-times[index]), 0., 1.)
    value = data[:, index]*(1.-fraction)+data[:, index+1]*fraction
    return np.minimum(value, np.min(data[:, index+1:], axis=1)), max(0., times[-1]-end), data[:, -1]


def fitted_future_floor(model, parameters, state, final):
    """Positive Weibull trends increase after onset; decreasing trends use
    their limiting floor (plus the basal contribution for fork speed).
    Frozen final values are held after the supplied fitting clock.
    """
    if not model.temporal:
        return final
    shapes = (np.asarray([model.shape]) if model.shape_index is None else
              np.exp(parameters[model.shape_slice]))
    lower = np.full_like(np.asarray(final, float), model.bounds[0])
    if model.baseline_index is not None:
        basal = np.clip(state[1]*np.exp(parameters[model.baseline_index]), *model.bounds)
        lower = np.maximum(lower, basal)
    return np.minimum(final, np.where(shapes >= 1., final, lower))


def kinetic_timing_bound(model, parameters, state, rate, speed, survival,
                         supplied, native_clock, dt, dx, radius, periodic,
                         *, length=None):
    """A tail certificate for the model and the other, supplied kinetic field."""
    fixed, wait, last = supplied_future_floor(supplied, model.clock[-1], native_clock, dt)
    if model.target == "initiation_rate":
        source = fitted_future_floor(model, parameters, state, rate[-1])
        return remaining_timing_bound(survival, source, fixed, dx, radius, periodic, length=length)
    velocity = fitted_future_floor(model, parameters, state, speed[-1])
    bound = remaining_timing_bound(survival, fixed, velocity, dx, radius, periodic, length=length)
    if wait:
        later = remaining_timing_bound(survival, last, velocity, dx, radius, periodic,
                                       length=length, wait=wait)
        bound = np.minimum(bound, later)
    return bound


def eventual_sources(final_rate, resolution=1., radius=2000., periodic=False):
    """Mark receivers reached by initiation that continues indefinitely.

    Supplied kinetic profiles hold their final value after their native clock.
    With a positive bounded propagation speed, the survival tail decays to
    zero exactly when a positive measure of continuing source cells lies in
    every receiver's permitted neighbourhood. A finite initiation pulse alone
    leaves a non-zero probability of no origin firing, even if it is tiny.
    """
    rates = np.asarray(final_rate, dtype=float).reshape(-1)
    dx = float(resolution)
    radius = np.inf if radius is None else float(radius)
    if not rates.size or np.any(~np.isfinite(rates)) or np.any(rates < 0):
        raise ValueError("Final initiation rates must be finite and non-negative.")
    if not np.isfinite(dx) or dx <= 0 or np.isnan(radius) or radius <= 0:
        raise ValueError("Resolution and influence radius must be positive.")
    positive = rates > 0.
    length = rates.size * dx
    if not np.any(positive):
        return np.zeros(rates.size, dtype=bool)
    if radius >= length or periodic and 2. * radius >= length:
        return np.ones(rates.size, dtype=bool)
    edges = np.arange(rates.size + 1, dtype=float) * dx
    prefix = np.r_[0., np.cumsum(positive, dtype=float) * dx]
    centers = (np.arange(rates.size, dtype=float) + .5) * dx

    def integral(positions):
        if periodic:
            turns = np.floor(positions / length)
            local = positions - turns * length
            return turns * prefix[-1] + np.interp(local, edges, prefix)
        return np.interp(np.clip(positions, 0., length), edges, prefix)

    return integral(centers + radius) - integral(centers - radius) > 0.
