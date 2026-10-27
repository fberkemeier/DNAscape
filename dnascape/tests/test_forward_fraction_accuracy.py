"""Independent light-cone checks for Mapping's replicated-fraction outputs."""

from itertools import product

import numpy as np
import pytest
from scipy.integrate import quad

from dnascape.mapping import map_replicated_fraction_xt


@pytest.mark.parametrize("firing_form,speed_form", tuple(product(("scalar", "x", "t", "xt"), repeat=2)))
@pytest.mark.parametrize("periodic", [False, True])
def test_uniform_fraction_is_independent_of_profile_representation(firing_form, speed_form, periodic):
    clock = np.arange(0., 1.01, .05)
    def field(form, value):
        return {"scalar": value, "x": np.full(11, value),
                "t": np.full((1, len(clock)), value), "xt": np.full((11, len(clock)), value)}[form]
    fraction = map_replicated_fraction_xt(field(firing_form, .2), field(speed_form, 1.2),
        n_bins=11, time_grid=clock, perQ=periodic)
    np.testing.assert_allclose(fraction[5], -np.expm1(-.24 * clock**2), atol=1e-13)
    if not periodic:
        np.testing.assert_allclose(fraction[0], -np.expm1(-.12 * clock**2), atol=1e-13)


@pytest.mark.parametrize("firing_xt,speed_xt", tuple(product((False, True), repeat=2)))
def test_temporal_profiles_match_the_homogeneous_continuum_law(firing_xt, speed_xt):
    clock = np.arange(0., 1.001, .025)
    firing = (.2 + .1 * clock)[None, :]
    speed = (1.2 + .5 * clock)[None, :]
    if firing_xt:
        firing = np.repeat(firing, 11, axis=0)
    if speed_xt:
        speed = np.repeat(speed, 11, axis=0)
    actual = map_replicated_fraction_xt(firing, speed, time_grid=clock, n_bins=11)[5]
    # 2 integral I(s) integral_s^t v(u) du ds for linear I and v.
    volume = .2 * 1.2 * clock**2 + (.1 * 1.2 + 2 * .2 * .5) / 3 * clock**3 + .1 * .5 / 4 * clock**4
    np.testing.assert_allclose(actual, -np.expm1(-volume), atol=2e-5)


def test_spatial_rate_and_speed_match_independent_source_interval_quadrature():
    firing = np.array([.1, .2, .05, .3, .02, .1])
    speed = np.array([1., 1.5, .8, 2., 1.2, 1.7])
    travel = np.r_[0., np.cumsum(.5 * (1. / speed[:-1] + 1. / speed[1:]))]
    clock = np.array([0., .1, .7, 2., 5.])
    expected = np.zeros((len(firing), len(clock)))
    for site, position in enumerate(travel):
        for column, time in enumerate(clock):
            volume = 0.
            for index in range(len(firing) - 1):
                rate = .5 * (firing[index] + firing[index + 1]) / (travel[index + 1] - travel[index])
                points = [p for p in (position - time, position, position + time)
                          if travel[index] < p < travel[index + 1]]
                volume += quad(lambda source: rate * max(time - abs(source - position), 0.),
                               travel[index], travel[index + 1], points=points, epsabs=1e-12)[0]
            expected[site, column] = -np.expm1(-volume)
    actual = map_replicated_fraction_xt(firing, speed, time_grid=clock)
    np.testing.assert_allclose(actual, expected, atol=1e-13)


def test_single_output_bin_retains_its_physical_length():
    from dnascape._forward_mapping import fraction_field
    clock, fraction = fraction_field(.1, 1.4, n_bins=1, resolution=10., resolution_time=1.)
    volume = .1 * np.where(2 * 1.4 * clock <= 10., 1.4 * clock**2,
                          10. * clock - 10.**2 / (4 * 1.4))
    np.testing.assert_allclose(fraction[0], -np.expm1(-volume))
    assert fraction[0, -1] > 1. - 1e-9
