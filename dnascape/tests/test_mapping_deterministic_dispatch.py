"""Zero-stalling mappings keep deterministic predictions and circular boundaries."""

from unittest.mock import patch

import numpy as np
import pytest

from dnascape import fitting, rmap
from dnascape.mapping import map_fork_densities_directionality, map_fork_densities_xt
from dnascape.workflows.mapping import MappingRequest, run_mapping
from dnascape.workflows.mapping_catalog import mapping_route


def _speed(form, n=24, nt=13):
    spatial = 1.4 + .2 * np.sin(np.arange(n) * 2 * np.pi / n)
    temporal = 1. + .15 * np.sin(np.arange(nt) / 4.)
    return {"scalar": 1.4, "x": spatial, "t": 1.4 * temporal[None, :],
            "xt": spatial[:, None] * temporal[None, :]}[form]


@pytest.mark.parametrize("form", ["x", "t", "xt"])
@pytest.mark.parametrize("periodic", [False, True])
def test_spatial_and_constant_firing_fits_never_simulate_variable_speed_without_stalls(form, periodic):
    observed = 18. + np.sin(np.arange(24) / 4.)
    with patch("dnascape.fitting.rsim", side_effect=AssertionError("unexpected ensemble")):
        options = dict(fork_speed=_speed(form), resolution_time=10., perQ=periodic, maxiter=2, verbose=False)
        first = fitting.map_timing_firing(observed, refine_edge_kb=0, refine_seed=1, sim_number=2, **options)
        second = fitting.map_timing_firing(observed, refine_edge_kb=0, refine_seed=999, sim_number=500, **options)
        np.testing.assert_array_equal(first, second)
        assert np.isfinite(first).all() and (first > 0).all()
        constant = fitting.map_timing_constant_firing(observed, **options)
        assert np.isfinite(constant) and constant > 0


@pytest.mark.parametrize("form", ["scalar", "x", "t", "xt"])
def test_periodic_directionality_obeys_region_switch_and_rolls_with_the_inputs(form):
    n = 24
    rates = np.zeros(n)
    rates[-2] = .03
    speed = _speed(form)
    route = mapping_route("firing_and_speed_to_directionality_deterministic")
    settings = {"perQ": True, "max_rep_time": 120., "resolution_time": 10.}

    def run(rates, speed):
        return run_mapping(MappingRequest(route, {"initiation_rate": rates, "fork_speed": speed},
            {"initiation_rate": "x", "fork_speed": form}, settings=settings))

    with patch("dnascape.fitting.rsim", side_effect=AssertionError("unexpected ensemble")), \
         patch("dnascape.routing.rsim", side_effect=AssertionError("unexpected ensemble")):
        result = run(rates, speed)
        assert result.assumptions["perQ"] is True
        actual = result.outputs["fork_directionality"]
        assert actual[0] > -1.  # Periodic geometry must not impose a left-end anchor.
        if form in {"scalar", "x"}:
            assert actual[0] > 0  # The nearby origin reaches bin zero across the join.
        rolled_speed = np.roll(speed, 5, axis=0) if form in {"x", "xt"} else speed
        rolled = run(np.roll(rates, 5), rolled_speed)
        np.testing.assert_allclose(rolled.outputs["fork_directionality"], np.roll(actual, 5), atol=1e-12)
        if form in {"scalar", "x"}:
            from dnascape.mapping import map_timing_directionality
            timing = rmap(source={"initiation_rate": rates, "fork_speed": speed}, target="replication_timing",
                method="firing_and_speed_to_timing_deterministic", perQ=True)
            expected = map_timing_directionality(timing, speed, perQ=True)
        else:
            from dnascape.mapping import map_replicated_fraction_xt, _expand_space_time_field
            clock = np.arange(0., 121., 10.)
            fraction = map_replicated_fraction_xt(rates, speed, time_grid=clock, perQ=True)
            survival = 1. - fraction
            velocity = _expand_space_time_field(speed, len(clock), 24, clock, 10., "fork_speed").T
            expected = np.clip(np.trapezoid(.5 * (velocity + np.roll(velocity, 1, axis=0)) *
                (survival - np.roll(survival, 1, axis=0)), clock, axis=1), -1., 1.)
        np.testing.assert_allclose(actual, expected, atol=1e-12)



@pytest.mark.parametrize("form", ["scalar", "x", "t", "xt"])
def test_forward_timing_and_free_time_fits_avoid_ensembles_at_zero_stalling(form):
    speed = _speed(form)
    rates = np.full(24, .01)
    with patch("dnascape.fitting.rsim", side_effect=AssertionError("unexpected ensemble")), \
         patch("dnascape.routing.rsim", side_effect=AssertionError("unexpected ensemble")):
        result = rmap(source={"initiation_rate": rates, "fork_speed": speed}, target="replication_timing",
                      assumptions={"perQ": True, "max_rep_time": 120., "resolution_time": 10., "verbose": False})
        assert np.isfinite(result).all()
        for target in ("time", "spacetime"):
            fit = rmap(source={"replication_timing": np.full(24, 18.), "fork_speed": speed}, target="initiation_rate",
                assumptions={"target_form": target, "target_shape": "free", "perQ": True,
                             "resolution_time": 10., "maxiter": 1, "refine_edge_kb": 0, "verbose": False})
            assert np.isfinite(fit).all()


def test_nonzero_stalling_still_uses_the_stochastic_model():
    calls = []

    def ensemble(**options):
        calls.append(options)
        return {"replication_timing": np.full(24, 18.)}

    with patch("dnascape.fitting.rsim", side_effect=ensemble):
        fitting.map_timing_firing(np.full(24, 18.), fork_speed=_speed("x"), stall_rate_per_kb=.01,
                                 maxiter=2, sim_number=7, refine_edge_kb=0, verbose=False)
    assert len(calls) == 2
    assert all(call["sim_number"] == 7 and call["stall_rate"] > 0 for call in calls)


@pytest.mark.parametrize("form", ["scalar", "x"])
def test_timing_directionality_uses_the_wrapped_slope_without_an_endpoint_anchor(form):
    n = 24
    timing = 18. + .2 * np.sin(np.arange(n) * 2 * np.pi / n)
    original = timing.copy()
    speed = _speed(form)
    result = rmap({"replication_timing": timing, "fork_speed": speed}, "fork_directionality",
                  assumptions={"perQ": True})
    edge_speed = speed if form == "scalar" else .5 * (speed + np.roll(speed, 1))
    expected = edge_speed * (timing - np.roll(timing, 1))
    np.testing.assert_allclose(result, expected, atol=1e-14)
    assert abs(result[0]) < .1
    np.testing.assert_array_equal(timing, original)


@pytest.mark.parametrize("model", [{"fork_speed_std": .2}, {"ffiring_nr": 2}])
def test_explicitly_stochastic_models_are_not_silently_replaced(model):
    with patch("dnascape.fitting.rsim", return_value={"replication_timing": np.ones(24)}) as ensemble:
        fitting._mapping_prediction(initiation_rate=np.full(24, .01), fork_speed=1.4, **model)
    ensemble.assert_called_once()
