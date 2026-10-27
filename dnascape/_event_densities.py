"""Initiation and termination densities from the replication transport equations.

Gauthier, Norio & Bechhoefer (2012), PLoS ONE 7:e32053, Eqs. 1--6:
    j_i = I s; j_c = 2 v rho_+ rho_- / s; s = 1-f.
Both rates have units events/(kb min). Their time integrals are events/kb,
not probability densities normalized to one. Fork speed is symmetric here.
"""

import numpy as np
from scipy.integrate import solve_ivp

from ._forward_mapping import fraction_field
from .inverse import _compact_space_time_field
from .mapping import _fraction_time_axis, _resolve_n_bins_from_fields
from .utils import check_cancelled, initial_prediction, measured_work_phase, report_measured_work


def _conditional_transport(density, velocity, dx, periodic, direction):
    """Second-order upwind flux for the conditional fork density rho/(1-f)."""
    previous, following = np.roll(density, 1), np.roll(density, -1)
    if not periodic:
        previous[0] = (-density[0] if direction == 1 else
                       2*density[0] - density[min(1, len(density)-1)])
        following[-1] = (-density[-1] if direction == -1 else
                         2*density[-1] - density[max(0, len(density)-2)])
    backward, forward = density-previous, following-density
    slope = np.where(backward*forward > 0.,
                     np.sign(backward)*np.minimum(np.abs(backward), np.abs(forward)), 0.)
    face_speed = .5*(velocity + np.roll(velocity, -1))
    if direction == 1:
        flux = face_speed*np.maximum(density + .5*slope, 0.)
        incoming = np.roll(flux, 1)
        if not periodic:
            incoming[0] = 0.
            flux[-1] = velocity[-1]*max(density[-1] + .5*slope[-1], 0.)
        return -(flux-incoming)/dx
    flux = face_speed*np.maximum(np.roll(density - .5*slope, -1), 0.)
    if not periodic:
        flux[-1] = 0.
    incoming = np.roll(flux, 1)
    if not periodic:
        incoming[0] = velocity[0]*max(density[0] - .5*slope[0], 0.)
    return (flux-incoming)/dx


@initial_prediction
def kinetic_event_densities(firing, speed, *, unreplicated_floor=1e-12, cfl=.8, **options):
    """Solve the common source/sink system, including cumulative event counts.

    With g_+/- = rho_+/- / s and A = -log(s), the published equations become
    d_t g_+/- +/- d_x(v g_+/-) = I and d_t A = v(g_+ + g_-).
    This equivalent form preserves positive survival without dividing by a
    vanishing unreplicated fraction. Adaptive time integration also evolves
    the event counts, independently of the displayed temporal sampling.
    """
    floor = float(unreplicated_floor)
    if not np.isfinite(floor) or not 0 <= floor < 1:
        raise ValueError("Unreplicated floor must be finite and between zero and one.")
    cfl = float(cfl)
    if not np.isfinite(cfl) or not 0 < cfl <= 1:
        raise ValueError("CFL must be finite and between zero and one.")
    dx, dt = float(options.get("resolution", 1.)), float(options.get("resolution_time", 5.))
    if not np.isfinite(dx) or dx <= 0:
        raise ValueError("Spatial resolution must be finite and positive.")
    clock_share = 0. if options.get("time_grid") is not None else .25
    if options.get("time_grid") is not None:
        # A supplied clock needs no separate replication-fraction solve.
        n = _resolve_n_bins_from_fields(firing, speed, n_bins=options.get("n_bins"),
            resolution_x=dx, genome_length=options.get("genome_length"), interval_kb=options.get("interval_kb"))
        if n is None:
            raise ValueError("Provide a spatial profile or a region length.")
        clock = _fraction_time_axis((firing, speed), n, resolution_time=dt, time_grid=options["time_grid"])
        if clock[0] < 0:
            raise ValueError("Replication time grids cannot start before zero.")
        if len(clock) > 100_000 or n*len(clock) > 64_000_000:
            raise ValueError("The computed time domain is too large. Increase the spatial or temporal resolution.")
    else:
        with measured_work_phase(0., clock_share):
            clock, fraction = fraction_field(firing, speed, **options)
        n = fraction.shape[0]
        del fraction
    rates = _compact_space_time_field(firing, len(clock), n, clock, dt, "initiation_rate")
    velocities = _compact_space_time_field(speed, len(clock), n, clock, dt, "fork_speed", positive=True)
    periodic = bool(options.get("perQ", False))
    state = np.zeros(5 * n)
    initiation = np.zeros((n, len(clock)))
    coalescence = np.zeros_like(initiation)
    baseline = np.zeros(2 * n)

    def fields(values, rate, velocity):
        survival = np.exp(-np.maximum(values[:n], 0.))
        right = np.maximum(values[n:2*n], 0.)
        left = np.maximum(values[2*n:3*n], 0.)
        birth = rate * survival
        death = np.where(survival > floor, 2. * velocity * right * left * survival, 0.)
        return survival, right, left, birth, death

    previous = 0.
    for index, end in enumerate(clock):
        check_cancelled()
        start = previous
        rate0, rate1 = rates[max(0, index - 1)], rates[index]
        speed0, speed1 = velocities[max(0, index - 1)], velocities[index]
        if end > start:
            def rhs(time, values):
                check_cancelled()
                report_measured_work(clock_share + (1. - clock_share) * time / max(float(clock[-1]), 1e-30))
                weight = (time - start) / (end - start)
                rate = (1. - weight) * rate0 + weight * rate1
                velocity = (1. - weight) * speed0 + weight * speed1
                survival, right, left, birth, death = fields(values, rate, velocity)
                return np.concatenate((
                    velocity * (right + left),
                    rate + _conditional_transport(right, velocity, dx, periodic, 1),
                    rate + _conditional_transport(left, velocity, dx, periodic, -1),
                    birth, death,
                ))

            step = min(cfl, .5) * dx / max(float(np.max(speed0)), float(np.max(speed1)))
            solution = solve_ivp(rhs, (start, end), state, t_eval=[end],
                                 method="RK45", rtol=1e-7, atol=1e-12, max_step=step)
            if not solution.success:
                raise RuntimeError("The event-density calculation did not converge.")
            state = solution.y[:, -1]
            state = np.maximum(state, 0.)
        _, _, _, initiation[:, index], coalescence[:, index] = fields(state, rate1, speed1)
        if index == 0:
            baseline = state[3*n:].copy()
        previous = float(end)
        report_measured_work(clock_share + (1. - clock_share) * float(end) / max(float(clock[-1]), 1e-30))
    return {
        "time_grid": clock,
        "initiation_events_xt": initiation,
        "coalescence_events_xt": coalescence,
        "initiation_events": state[3*n:4*n] - baseline[:n],
        "coalescence_events": state[4*n:] - baseline[n:],
    }
