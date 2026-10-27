"""Mapping's deterministic observables, clocks and kinetic dependences."""

from itertools import product

import numpy as np
import pytest

from dnascape import rmap
from dnascape import routing
from dnascape.mapping import map_replicated_fraction_xt, map_timing_directionality
from dnascape.workflows.mapping import MappingRequest, run_mapping
from dnascape.workflows.mapping_catalog import get_mapping_catalog, mapping_route


def _field(form, value, n=13, nt=13):
    x = np.linspace(.85, 1.15, n)
    t = np.linspace(.9, 1.1, nt)
    return {"scalar": value, "x": value * x,
            "t": value * t[None, :], "xt": value * x[:, None] * t[None, :]}[form]


@pytest.mark.parametrize("firing_form,speed_form", tuple(product(("scalar", "x", "t", "xt"), repeat=2)))
@pytest.mark.parametrize("periodic", [False, True])
def test_kinetic_directionality_is_deterministic_for_every_dependence(monkeypatch, firing_form, speed_form, periodic):
    def forbid(*args, **kwargs):
        pytest.fail("Directionality must not run simulations.")
    monkeypatch.setattr(routing, "rsim", forbid)
    route = mapping_route("firing_and_speed_to_directionality_deterministic")
    assert not {"sim_number", "stall_shared_directions", "stall_rate", "tau", "seed"} & {s.key for s in route.settings}
    inputs = {"initiation_rate": _field(firing_form, .12), "fork_speed": _field(speed_form, 1.4)}
    result = run_mapping(MappingRequest(route, inputs,
        {"initiation_rate": firing_form, "fork_speed": speed_form},
        settings={"perQ": periodic}, region_start_kb=0., region_end_kb=13., resolution_t=1.))
    actual = result.outputs[result.primary_output]
    assert actual.shape == (13,)
    assert np.all(np.isfinite(actual))
    assert np.all(np.abs(actual) <= 1.)
    np.testing.assert_array_equal(result.x_axis, np.arange(13.))
    if speed_form in {"scalar", "x"}:
        timing = rmap(source=inputs, target="replication_timing",
            method="firing_and_speed_to_timing_deterministic", perQ=periodic,
            n_bins=13, resolution_time=1.)
        expected = map_timing_directionality(timing, inputs["fork_speed"], perQ=periodic)
        np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_mapping_removes_requested_targets_and_routes():
    catalog = get_mapping_catalog()
    assert not {"initiation_pdf", "coalescence_pdf", "inter_origin_distance", "completion_time",
                "s_phase_duration", "num_origins"} & {t.key for t in catalog.targets}
    assert not any(t.requires_simulation for t in catalog.targets)
    assert not any(r.method.endswith("_simulation") for r in catalog.routes)
    for route in catalog.routes:
        if route.target == "fork_directionality":
            assert all("fork_density" not in item.key and "fork_densities" not in item.key for item in route.inputs)
        if route.target_key == "replicated_fraction":
            assert all(item.key != "replicated_fraction_xt" for item in route.inputs)


@pytest.mark.parametrize("speed", [1., np.ones(11)])
def test_fraction_integrates_partial_bins_and_spatial_speed_exactly(speed):
    clock = np.array([0., .1, .25, .5, 1.])
    fraction = map_replicated_fraction_xt(np.ones(11), speed, time_grid=clock)
    # At the interior the cone has volume I*v*t^2. At an open end it is half.
    np.testing.assert_allclose(fraction[5], -np.expm1(-clock**2), atol=1e-13)
    np.testing.assert_allclose(fraction[0], -np.expm1(-.5 * clock**2), atol=1e-13)


@pytest.mark.parametrize("periodic", [False, True])
def test_fraction_with_spatial_kinetics_is_monotonic_and_complete(periodic):
    rates = .05 * (1. + .5 * np.sin(np.arange(17.)))
    speed = 1.4 + .2 * np.cos(np.arange(17.))
    result = run_mapping(MappingRequest(mapping_route("firing_and_speed_to_replicated_fraction_xt_lightcone"),
        {"initiation_rate": rates, "fork_speed": speed}, {"initiation_rate": "x", "fork_speed": "x"},
        whole_genome=True, resolution_t=1., settings={"perQ": periodic}))
    fraction = result.outputs[result.primary_output]
    assert np.all(np.diff(fraction, axis=1) >= -1e-12)
    assert np.min(fraction[:, -1]) > 1. - 1e-9
    assert result.time_axis[-1] < 1200.
    assert fraction.shape[1] == len(result.time_axis)
