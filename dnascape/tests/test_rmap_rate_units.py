"""Physical firing-density units at legacy rmap forward boundaries."""

from itertools import product

import numpy as np
import pytest

from dnascape.mapping import map_firing_timing, map_fork_densities_xt, map_replicated_fraction_xt
from dnascape.routing import rmap
from dnascape.simulation import rsim


def _source(firing, speed, explicit):
    return ({"firing_rate": firing, "fork_speed": speed}, {}) if explicit else (
        {"firing_rate": firing}, {"fork_speed": speed},
    )


@pytest.mark.parametrize(
    "dx,firing_form,speed_form,periodic,explicit",
    list(product((.5, 2., 5.), ("scalar", "x"), ("scalar", "x"), (False, True), (False, True))),
)
def test_static_routes_convert_density_once_and_preserve_legacy_direct_units(
    dx, firing_form, speed_form, periodic, explicit,
):
    n = 12
    firing = .012 if firing_form == "scalar" else np.linspace(.008, .02, n)
    speed = 1.4 if speed_form == "scalar" else np.linspace(1., 2., n)
    original = np.asarray(firing).copy()
    source, assumptions = _source(firing, speed, explicit)
    method = "firing_and_speed_to_timing_deterministic" if explicit else "firing_to_timing_deterministic"
    actual = rmap(
        source=source, target="replication_timing", method=method,
        resolution=dx, n_bins=n, perQ=periodic, **assumptions,
    )
    expected = map_firing_timing(
        np.asarray(firing) * dx, fork_speed=speed, resolution=dx,
        n_bins=n, perQ=periodic,
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
    np.testing.assert_array_equal(firing, original)
    assert not np.allclose(actual, map_firing_timing(
        firing, fork_speed=speed, resolution=dx, n_bins=n, perQ=periodic,
    ))


@pytest.mark.parametrize("dx", (.25, .5, 1.))
def test_density_timing_converges_to_independent_uniform_continuum_expectation(dx):
    firing_density = .02
    speed = 1.4
    actual = rmap(
        source={"firing_rate": firing_density, "fork_speed": speed},
        target="replication_timing", resolution=dx, genome_length=128., perQ=True,
    )
    # For a uniform infinite DNA molecule, S(t)=exp(-I*v*t**2), hence this
    # analytic mean. A finite lattice introduces a bin-crossing-time error.
    continuous_mean = np.sqrt(np.pi) / (2. * np.sqrt(firing_density * speed))
    assert abs(float(np.mean(actual)) - continuous_mean) < dx / speed


@pytest.mark.parametrize(
    "target,dx,explicit",
    list(product(
        ("replication_timing", "origin_efficiency", "inter_origin_distance",
         "s_phase_duration", "completion_time", "num_origins", "fork_directionality"),
        (.5, 3.), (False, True),
    )),
)
def test_every_simulation_forward_route_matches_seeded_legacy_bin_hazards(target, dx, explicit):
    firing = np.linspace(.025, .06, 12)
    source, assumptions = _source(firing, 1.4, explicit)
    method = f"firing_and_speed_to_{target}_simulation" if explicit else f"firing_to_{target}_simulation"
    if target == "replication_timing":
        method = "firing_and_speed_to_timing_simulation" if explicit else "firing_to_timing_simulation"
    options = dict(
        resolution=dx, sim_number=16, seed=781, verbose=False,
        retain_rep_times_per_sim=False,
        retain_inter_origin_distances=(target == "inter_origin_distance"),
    )
    expected = rsim(firing_rate=firing * dx, fork_speed=1.4, **options)[target]
    actual = rmap(source=source, target=target, method=method, **options, **assumptions)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "form,resolution_key", list(product(("scalar", "x", "t", "xt"), (
        "resolution", "resolution_x", "resolution_space",
    ))),
)
def test_stochastic_density_conversion_respects_all_forms_and_resolution_aliases(form, resolution_key):
    n = 10
    grid = np.arange(0., 41., 2.)
    firing = {
        "scalar": .02, "x": np.linspace(.015, .03, n),
        "t": np.linspace(.015, .03, len(grid)).reshape(1, -1),
        "xt": np.linspace(.015, .03, n)[:, None] * np.linspace(.8, 1.4, len(grid))[None, :],
    }[form]
    dx = 2.5
    options = dict(sim_number=16, seed=722, time_grid=grid, resolution_time=2., n_bins=n, verbose=False)
    expected = rsim(
        firing_rate=np.asarray(firing) * dx, fork_speed=1.4,
        resolution=dx, **options,
    )["replication_timing"]
    actual = rmap(
        source={"firing_rate": firing, "fork_speed": 1.4},
        target="replication_timing", method="firing_and_speed_to_timing_simulation",
        **{resolution_key: dx}, **options,
    )
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("dx,periodic", list(product((.5, 3.), (False, True))))
def test_mean_field_and_observation_routes_keep_density_units_without_rescaling(dx, periodic):
    firing = np.linspace(.01, .03, 10)
    grid = np.arange(0., 42., 2.)
    source = {"firing_rate": firing, "fork_speed": 1.4}
    options = dict(resolution=dx, time_grid=grid, perQ=periodic)
    fraction = map_replicated_fraction_xt(firing, fork_speed=1.4, **options)
    actual_fraction = rmap(source=source, target="replicated_fraction_xt", **options)
    np.testing.assert_array_equal(actual_fraction, fraction)
    bundle = map_fork_densities_xt(firing, fork_speed=1.4, **options)
    actual_bundle = rmap(source=source, target="fork_densities_xt", **options)
    for key in ("replicated_fraction_xt", "right_fork_density_xt", "left_fork_density_xt"):
        np.testing.assert_array_equal(actual_bundle[key], bundle[key])
    events = rmap(
        source={"firing_rate": firing, "replicated_fraction_xt": fraction},
        target="initiation_events_xt", resolution=dx, time_grid=grid,
    )
    np.testing.assert_allclose(events, firing[:, None] * (1. - fraction))
