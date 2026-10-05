"""Regression coverage for the restored local spatial-speed iteration."""

import numpy as np
import pytest

from dnascape.fitting import map_timing_firing_spatial_speed
from dnascape.mapping import map_firing_timing
from dnascape.routing import rmap
from dnascape.utils import progress_context


@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("resolution", [1., 4.])
def test_public_spatial_speed_route_recovers_local_model_with_correct_rate_units(
    monkeypatch, periodic, resolution,
):
    def forbidden(*args, **kwargs):
        raise AssertionError("Static spatial-speed mapping must not enter the PDE inverse")

    monkeypatch.setattr("dnascape.inverse._mean_field_timing", forbidden)
    n = 96
    truth = 1.6 + .35 * np.sin(np.linspace(0, 2 * np.pi, n, endpoint=False))
    firing = .004
    timing = map_firing_timing(
        firing * resolution, truth, resolution=resolution, perQ=periodic,
    )
    result = rmap(
        source={"replication_timing": timing, "firing_rate": firing},
        target="fork_speed", target_form="space", resolution=resolution,
        perQ=periodic, maxiter=40, smoothw=None, verbose=False,
    )
    np.testing.assert_allclose(result, truth, rtol=.008)
    predicted = result._rmap_info["replication_timing_fit"]
    np.testing.assert_allclose(predicted, timing, rtol=.004)
    assert result._rmap_info["fit_mse"] == pytest.approx(
        min(result._rmap_info["fit_history"]),
    )


def test_one_update_moves_both_early_and_late_sites_and_evaluates_final_profile():
    truth = np.r_[np.full(20, .7), np.full(20, 2.8)]
    timing = map_firing_timing(.004, truth)
    initial_prediction = map_firing_timing(.004, np.full(40, 1.4))
    fit = map_timing_firing_spatial_speed(
        {"replication_timing": timing, "firing_rate": .004},
        maxiter=1, smoothw=None, verbose=False,
    )
    assert np.all(fit[:20] < 1.4)
    assert np.all(fit[20:] > 1.4)
    assert fit._rmap_info["optimizer_evaluations"] == 2
    assert fit._rmap_info["fit_mse"] < np.mean((initial_prediction - timing)**2)
    np.testing.assert_allclose(
        fit._rmap_info["replication_timing_fit"], map_firing_timing(.004, fit),
    )


def test_timing_offset_preserves_speed_and_diagnostic_coordinates():
    timing = np.linspace(20., 35., 48)
    settings = dict(maxiter=3, smoothw=None, verbose=False)
    base = map_timing_firing_spatial_speed(
        {"replication_timing": timing, "firing_rate": .001}, **settings,
    )
    shifted = map_timing_firing_spatial_speed(
        {"replication_timing": timing + 37., "firing_rate": .001}, T0=37., **settings,
    )
    np.testing.assert_allclose(shifted, base)
    np.testing.assert_allclose(
        shifted._rmap_info["replication_timing_fit"],
        base._rmap_info["replication_timing_fit"] + 37.,
    )
    np.testing.assert_allclose(shifted._rmap_info["residual"], base._rmap_info["residual"])


def test_selected_region_edges_are_fitted_instead_of_overwritten():
    # At this resolution the previous hard-coded 1000 kb exclusion discarded
    # ten positions at both ends even when the user had explicitly selected them.
    truth = np.r_[np.full(10, .7), np.full(20, 1.4), np.full(10, 2.8)]
    timing = map_firing_timing(.01, truth, resolution=100.)
    result = map_timing_firing_spatial_speed(
        {"replication_timing": timing, "firing_rate": .01},
        resolution=100., maxiter=1, smoothw=None, verbose=False,
    )
    assert result[0] < result[15] < result[-1]


def test_smoothing_preserves_a_uniform_periodic_update():
    timing = map_firing_timing(.004, np.full(48, 2.8), perQ=True)
    result = map_timing_firing_spatial_speed(
        {"replication_timing": timing, "firing_rate": .004},
        perQ=True, maxiter=1, smoothw=20, verbose=False,
    )
    assert result[0] > 1.4
    np.testing.assert_allclose(result, result[0], rtol=1e-13)


def test_cancel_is_checked_inside_the_deterministic_forward_evaluation():
    class Stopped(Exception):
        pass

    checks = 0

    def cancel():
        nonlocal checks
        checks += 1
        if checks == 3:
            raise Stopped

    with progress_context(cancel=cancel):
        with pytest.raises(Stopped):
            map_firing_timing(1e-6, np.full(256, 1.4))
    assert checks == 3


def test_supplied_bounds_and_best_iterate_are_retained():
    result = map_timing_firing_spatial_speed(
        {"replication_timing": np.linspace(10., 600., 48), "firing_rate": .001},
        maxiter=4, fit_step=8., smoothw=None, speed_bounds=(.2, 3.), verbose=False,
    )
    assert np.all((result >= .2) & (result <= 3.))
    assert result._rmap_info["fit_mse"] == min(result._rmap_info["fit_history"])


@pytest.mark.parametrize("timing,firing,kwargs,match", [
    ([1., 2.], 0., {}, "positive firing rate"),
    ([1., 2.], .001, {"T0": 3.}, "greater than T0"),
    ([1., 2.], .001, {"fit_step": 0.}, "fit_step"),
])
def test_invalid_iteration_inputs_fail_before_fitting(timing, firing, kwargs, match):
    with pytest.raises(ValueError, match=match):
        map_timing_firing_spatial_speed(
            {"replication_timing": timing, "firing_rate": firing}, **kwargs,
        )
