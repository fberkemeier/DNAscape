"""Deterministic forward fields with an automatically completed time axis."""

import numpy as np

from .mapping import (
    _field_time_length, _resolve_n_bins_from_fields, map_replicated_fraction_xt,
)
from .utils import check_cancelled


def fraction_field(firing, speed, **options):
    """Return f(x,t) and its actual clock, extending automatic clocks to completion.

    Supplied clocks retain their meaning. Beyond an uploaded temporal profile,
    the last kinetic value is held, as in the other deterministic forward maps.
    """
    options = dict(options)
    dx = float(options.get("resolution", 1.))
    dt = float(options.get("resolution_time", 5.))
    n = _resolve_n_bins_from_fields(
        firing, speed, n_bins=options.get("n_bins"), resolution_x=dx,
        genome_length=options.get("genome_length"), interval_kb=options.get("interval_kb"),
    )
    if n is None:
        raise ValueError("Provide a spatial profile or a region length.")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Time step must be finite and positive.")
    explicit = options.get("time_grid")
    maximum = options.get("max_time", options.get("max_rep_time"))
    if maximum is None:
        maximum = options.get("max_rep_time")
    automatic = explicit is None and maximum is None
    if explicit is not None:
        clock = np.asarray(explicit, dtype=float)
    elif maximum is not None:
        maximum = float(maximum)
        if not np.isfinite(maximum) or maximum <= 0:
            raise ValueError("Maximum time must be finite and positive.")
        count = maximum / dt
        if not np.isfinite(count) or count > min(99_999, 64_000_000 // int(n) - 1):
            raise ValueError("The computed time domain is too large. Increase the spatial or temporal resolution.")
        clock = np.append(np.arange(0., maximum, dt), maximum)
    else:
        rates = np.asarray(firing, dtype=float)
        speeds = np.asarray(speed, dtype=float)
        if np.any(~np.isfinite(rates)) or np.any(rates < 0):
            raise ValueError("Initiation rate must contain finite values >= 0.")
        if np.any(~np.isfinite(speeds)) or np.any(speeds <= 0):
            raise ValueError("Fork speed must contain finite positive values.")
        support = max((_field_time_length(a, n) or 1) for a in (firing, speed))
        average = float(np.mean(rates))
        velocity = float(np.median(speeds))
        characteristic = np.sqrt(np.pi / (4. * average * velocity)) if average > 0 else dt
        waiting = 1. / (average * max((n - 1) * dx, dx)) if average > 0 else dt
        maximum = max(2. * dt, (support - 1) * dt, 6. * characteristic, 12. * waiting)
        count = maximum / dt
        if not np.isfinite(count) or count > min(99_999, 64_000_000 // int(n) - 1):
            raise ValueError("The computed time domain is too large. Increase the spatial or temporal resolution.")
        clock = np.arange(int(np.ceil(count)) + 1, dtype=float) * dt
    while True:
        check_cancelled()
        # Check before allocating any space-time fields or expanding profiles.
        if clock.size > 100_000 or int(n) * clock.size > 64_000_000:
            raise ValueError("The computed time domain is too large. Increase the spatial or temporal resolution.")
        selected = {key: value for key, value in options.items()
                    if key not in {"time_grid", "max_time", "max_rep_time"}}
        field = map_replicated_fraction_xt(firing, speed, time_grid=clock, **selected)
        if not automatic or np.max(1. - field[:, -1]) <= 1e-9 or not np.any(firing):
            return clock, field
        # A finite initiation pulse can leave an unreplicated population. Once
        # its cones cover the domain, extra time cannot remove that population.
        terminal = np.asarray(firing)
        if terminal.ndim == 2 and terminal.shape[1] > 1:
            last_rate = terminal[:, -1]
            support_end = (terminal.shape[1] - 1) * dt
            crossing = max(n * dx / float(np.min(speed)), dt)
            if not np.any(last_rate) and clock[-1] >= support_end + crossing:
                return clock, field
        clock = np.arange(2 * (clock.size - 1) + 1, dtype=float) * dt


def timing_from_fraction(clock, fraction, *, require_complete=True):
    survival = 1. - np.asarray(fraction)
    if require_complete and np.max(survival[:, -1]) > 1e-7:
        raise ValueError("The firing profile leaves an unreplicated fraction; finite mean timing cannot be determined.")
    return np.trapezoid(survival, clock, axis=1)
