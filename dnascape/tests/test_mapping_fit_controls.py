"""Constraints affect fitted quantities, never supplied observations."""

import numpy as np
import pytest

from dnascape import fitting, rmap
from dnascape.mapping import map_firing_timing
from dnascape.workflows.mapping import MappingRequest, run_mapping
from dnascape.workflows.mapping_catalog import mapping_route


@pytest.mark.parametrize("dx", [.1, 1., 10.])
@pytest.mark.parametrize("periodic", [False, True])
def test_spatial_rate_guess_and_bounds_use_per_bin_units(dx, periodic):
    observed = map_firing_timing(.005, fork_speed=1.4, resolution=dx, n_bins=16, perQ=periodic)
    original = observed.copy()
    request = MappingRequest(
        route=mapping_route("timing_and_speed_to_initiation_rate"),
        inputs={"replication_timing": observed, "fork_speed": 1.4},
        input_forms={"replication_timing": "x", "fork_speed": "scalar"},
        settings={"perQ": periodic, "maxiter": 1, "fit_step": 1e-12, "initial_initiation_rate": .003,
                  "rate_bounds": (.002, .004)}, resolution_x=dx, initiation_rate_per_bin=True,
    )
    result = run_mapping(request)
    np.testing.assert_allclose(result.outputs["initiation_rate"], .003)
    np.testing.assert_array_equal(result.comparisons["replication_timing_fit"].observed, original)
    np.testing.assert_array_equal(observed, original)


@pytest.mark.parametrize("method", ["timing_and_speed_to_constant_initiation_rate", "timing_and_speed_to_initiation_rate"])
def test_rate_bounds_hold_during_iterations_and_accept_one_sided_limits(method):
    info = rmap({"replication_timing": np.full(20, 25.), "fork_speed": 1.4}, "initiation_rate",
                method=method, resolution=1., perQ=True, maxiter=4,
                target_form="constant" if "constant" in method else "space",
                rate_bounds=(None, .0004), initial_initiation_rate=.0003, return_info=True, verbose=False)
    assert np.all(np.asarray(info["values"]) <= .0004 * (1 + 1e-12))
    assert np.all(np.asarray(info["values"]) > 0)


def test_speed_bounds_are_independently_optional_and_keep_the_initial_guess():
    timing = map_firing_timing(.01, fork_speed=1.4, n_bins=16, perQ=True)
    result = rmap({"replication_timing": timing, "initiation_rate": .01}, "fork_speed",
                  method="timing_and_firing_to_constant_fork_speed", initial_fork_speed=1.8,
                  speed_bounds=(None, 2.), maxiter=0, perQ=True, verbose=False)
    np.testing.assert_allclose(result, 1.8)


@pytest.mark.parametrize("target_form", ["time", "spacetime"])
def test_weibull_amplitude_bounds_constrain_the_fitted_parameter(target_form):
    result = fitting._fit_weibull_survival_firing(
        np.full(8, 12.), 1.4, target_form=target_form, time_grid=np.arange(0., 41., 2.),
        maxiter=1, refine_edge_kb=0., floor=1e-12,
        rate_bounds=(1e-6, 2e-6), initial_initiation_rate=1.5e-6,
    )
    alpha = result._rmap_info["alpha"]
    assert np.all(alpha >= 1e-6 * (1 - 1e-12))
    assert np.all(alpha <= 2e-6 * (1 + 1e-12))


def test_invalid_bounds_fail_before_fitting():
    with pytest.raises(ValueError, match="bounds|Bounds"):
        rmap({"replication_timing": np.ones(10), "fork_speed": 1.4}, "initiation_rate",
             method="timing_and_speed_to_initiation_rate", rate_bounds=(.1, .01), verbose=False)


def test_one_automatic_bound_can_adapt_to_a_tighter_explicit_bound():
    from dnascape._fit_bounds import fit_bounds
    low, high = fit_bounds((None, .01), (.2, 6.))
    assert 0 < low < high == .01
    low, high = fit_bounds((10., None), (.2, 6.))
    assert low == 10. and high > low
