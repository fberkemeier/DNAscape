"""Published replication source/sink equations and event-density units."""

from itertools import product

import numpy as np
import pytest
from scipy.integrate import quad

from dnascape._event_densities import kinetic_event_densities
from dnascape.mapping import map_initiation_events_xt, map_coalescence_events_xt


@pytest.mark.parametrize("step", [1., 5., 20.])
def test_uniform_events_match_the_analytical_poisson_solution(step):
    rate, speed = .01, 1.4
    clock = np.arange(0., 60.01, step)
    result = kinetic_event_densities(rate, speed, n_bins=200, perQ=True, time_grid=clock)
    survival = np.exp(-rate * speed * clock**2)
    np.testing.assert_allclose(result["initiation_events_xt"], np.tile(rate * survival, (200, 1)), atol=2e-10)
    expected_coalescence = 2. * speed * rate**2 * clock**2 * survival
    np.testing.assert_allclose(result["coalescence_events_xt"], np.tile(expected_coalescence, (200, 1)), atol=2e-10)
    expected_density = .5 * np.sqrt(np.pi * rate / speed)
    np.testing.assert_allclose(result["initiation_events"], expected_density, atol=2e-9)
    np.testing.assert_allclose(result["coalescence_events"], expected_density, atol=2e-9)
    # In particular, even very sparse displayed time points retain the count.
    if step == 20.:
        assert abs(np.trapezoid(result["coalescence_events_xt"][0], clock) - expected_density) > .01


@pytest.mark.parametrize("rate_xt,speed_xt", tuple(product((False, True), repeat=2)))
def test_time_varying_rates_and_velocities_match_independent_integrals(rate_xt, speed_xt):
    clock = np.array([0., .5, 2., 5., 10., 20., 40.])
    a, b, c, d = .02, .0003, 1.3, .01
    rate = (a + b * clock)[None, :]
    speed = (c + d * clock)[None, :]
    if rate_xt:
        rate = np.repeat(rate, 128, axis=0)
    if speed_xt:
        speed = np.repeat(speed, 128, axis=0)
    def survival(t):
        return np.exp(-a*c*t*t - (b*c + 2*a*d)/3*t**3 - b*d/4*t**4)
    result = kinetic_event_densities(rate, speed, n_bins=128, perQ=True, time_grid=clock)
    expected_i = (a + b*clock) * survival(clock)
    expected_c = 2*(c+d*clock)*(a*clock + .5*b*clock**2)**2 * survival(clock)
    np.testing.assert_allclose(result["initiation_events_xt"][0], expected_i, atol=2e-9)
    np.testing.assert_allclose(result["coalescence_events_xt"][0], expected_c, atol=2e-9)
    for name, integrand in (
        ("initiation_events", lambda t: (a+b*t)*survival(t)),
        ("coalescence_events", lambda t: 2*(c+d*t)*(a*t+.5*b*t*t)**2*survival(t)),
    ):
        expected = quad(integrand, 0., clock[-1], epsabs=1e-12)[0]
        np.testing.assert_allclose(result[name], expected, atol=3e-9)


@pytest.mark.parametrize("periodic", [False, True])
def test_full_space_time_profiles_obey_event_count_balance(periodic):
    clock = np.linspace(0., 120., 13)
    errors = []
    for n in (48, 96):
        dx = 192./n
        phase = np.arange(n) * 2*np.pi/n
        rate = (.008*(1.+.5*np.sin(phase)))[:, None] * (1.+.002*clock)
        speed = (1.4*(1.+.3*np.cos(phase)))[:, None] * (1.+.001*clock)
        result = kinetic_event_densities(rate, speed, resolution=dx, time_grid=clock, perQ=periodic)
        for name in ("initiation_events", "coalescence_events", "initiation_events_xt", "coalescence_events_xt"):
            assert np.isfinite(result[name]).all() and np.all(result[name] >= 0.)
        difference = dx * np.sum(result["initiation_events"] - result["coalescence_events"])
        errors.append(abs(difference - (0. if periodic else 1.)))
    # A completed circle has equal births and mergers. An isolated linear
    # molecule has one fewer merger, because the two outer forks leave its ends.
    # Counts converge with the second-order spatial discretisation; they are
    # never rescaled artificially to impose the topological identity.
    assert errors[1] < (.001 if periodic else .01)
    assert errors[1] < .35 * errors[0]


def test_zero_firing_is_valid_and_creates_no_events():
    result = kinetic_event_densities(np.zeros((8, 5)), 1.4, time_grid=np.arange(5.))
    assert not np.any(result["initiation_events_xt"])
    assert not np.any(result["coalescence_events_xt"])
    fraction = np.full((8, 5), .5)
    np.testing.assert_array_equal(map_initiation_events_xt(np.zeros((8, 5)), fraction), 0.)


def test_firing_free_regions_can_still_contain_coalescences():
    clock = np.linspace(0., 60., 13)
    firing = np.full((48, len(clock)), .02)
    firing[18:30] = 0.
    speed = np.ones_like(firing) * (1.2 + .005*clock)
    result = kinetic_event_densities(firing, speed, time_grid=clock, perQ=True)
    assert not np.any(result["initiation_events_xt"][18:30])
    assert np.max(result["coalescence_events_xt"][18:30]) > .001


def test_supplied_observables_use_the_relative_fork_speed_and_no_probability_normalisation():
    fraction = np.array([[0., .5, 1.], [.25, .75, 1.]])
    speed = np.array([[1., 2., 3.], [3., 2., 1.]])
    right = np.array([[.1, .2, 0.], [.2, .1, 0.]])
    left = .5*right
    actual = map_coalescence_events_xt(speed, fraction, right, left)
    expected = np.zeros_like(fraction)
    np.divide(2.*speed*right*left, 1.-fraction, out=expected, where=fraction < 1.)
    np.testing.assert_allclose(actual, expected)
    np.testing.assert_allclose(map_coalescence_events_xt(speed, fraction, 2*right, left), 2*expected)
