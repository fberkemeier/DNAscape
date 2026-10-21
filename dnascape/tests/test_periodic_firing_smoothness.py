"""Circular fitting regularizes the rate itself without changing observations."""

import numpy as np
import pytest

from dnascape._periodic_firing import PeriodicFiringSmoother
from dnascape.fitting import map_timing_firing
from dnascape.mapping import map_firing_timing


@pytest.mark.parametrize("count", [1, 2, 31, 32, 129])
def test_circular_smoothing_has_no_privileged_join_and_preserves_units(count):
    rates = np.random.default_rng(3).uniform(.001, .004, count)
    smoother = PeriodicFiringSmoother(np.full(count, 80.), 1.4, 2.)
    result = smoother(rates)
    np.testing.assert_allclose(smoother(np.roll(rates, 7)), np.roll(result, 7), rtol=2e-14)
    np.testing.assert_allclose(smoother(5 * rates), 5 * result, rtol=2e-14)
    np.testing.assert_allclose(smoother(np.full(count, .002)), .002, rtol=2e-14)
    assert np.all(result > 0)


def test_log_curvature_regularization_solves_the_circular_objective():
    smoother = PeriodicFiringSmoother(np.full(64, 80.), 1.4, 1.)
    logs = np.random.default_rng(7).normal(-8., .5, 64)
    fitted_logs = np.log(smoother(np.exp(logs)))
    curvature = np.roll(fitted_logs, 1) - 2 * fitted_logs + np.roll(fitted_logs, -1)
    stationarity = fitted_logs - logs + smoother.length_bins**4 * (
        np.roll(curvature, 1) - 2 * curvature + np.roll(curvature, -1)
    )
    np.testing.assert_allclose(stationarity, 0., atol=2e-12)


def test_space_time_smoothing_keeps_time_slices_independent_and_inactive_rows_zero():
    rates = np.asarray([[.01, .02, .03, .04], [0., 0., 0., 0.], [.03, .04, .01, .02]])
    smoother = PeriodicFiringSmoother(np.full(4, 80.), 1.4, 1.)
    result = smoother(rates, axis=1, floor=0.)
    for index in range(rates.shape[0]):
        np.testing.assert_allclose(result[index], smoother(rates[index], floor=0.))
    np.testing.assert_array_equal(result[1], 0.)
    np.testing.assert_allclose(smoother(rates.T, axis=0, floor=0.), result.T)


@pytest.mark.parametrize("bad", [1., [-1., 1., 2.], [1., np.nan, 2.], [1., 2.]])
def test_invalid_spatial_rate_arrays_are_rejected(bad):
    smoother = PeriodicFiringSmoother(np.full(3, 80.), 1.4, 1.)
    with pytest.raises(ValueError):
        smoother(bad)


def test_smoothing_length_uses_physical_distance_across_resolutions():
    first = PeriodicFiringSmoother(np.full(1000, 120.), 1.4, 1.)
    second = PeriodicFiringSmoother(np.full(2000, 120.), 1.4, .5)
    assert first.length_kb == pytest.approx(8.4)
    assert second.length_kb == pytest.approx(first.length_kb)


def test_single_bin_rates_respect_the_requested_floor():
    smoother = PeriodicFiringSmoother([80.], 1.4, 1.)
    np.testing.assert_array_equal(smoother([0.], floor=1e-12), [1e-12])
    np.testing.assert_array_equal(smoother([0.], floor=0.), [0.])


def test_mismatched_timing_does_not_create_a_rate_jump_at_the_join():
    phase = np.arange(1000.) / 1000.
    observed = 117. + 13. * np.sin(6 * np.pi * phase) + 10. * np.sin(2 * np.pi * phase) + 22. * phase
    original = observed.copy()
    observed.setflags(write=False)
    options = dict(fork_speed=1.4, perQ=True, fit_step=2., maxiter=10, verbose=False)
    legacy = map_timing_firing(observed, stabilize=False, **options)
    smooth = map_timing_firing(observed, **options)
    assert abs(legacy[-1] - legacy[0]) > 10 * np.abs(np.diff(legacy)).max()
    # The join is now as smooth as the rest of the fitted curve.
    assert abs(smooth[-1] - smooth[0]) <= 1.1 * np.abs(np.diff(smooth)).max()
    assert smooth._rmap_info["fit_mse"] < legacy._rmap_info["fit_mse"]
    prediction = map_firing_timing(smooth, fork_speed=1.4, perQ=True)
    np.testing.assert_allclose(prediction, smooth._rmap_info["replication_timing_fit"], atol=1e-10)
    assert smooth._rmap_info["fit_mse"] == pytest.approx(np.mean((prediction - observed)**2))
    np.testing.assert_array_equal(observed, original)


def test_known_smooth_periodic_rate_is_recovered_with_the_default_iteration_budget():
    phase = np.arange(1000.) / 1000.
    expected = 6e-5 * (1 + .6 * np.cos(2 * np.pi * phase) + .2 * np.sin(6 * np.pi * phase))
    observed = map_firing_timing(expected, perQ=True)
    fitted = map_timing_firing(observed, perQ=True, fit_step=2., maxiter=10, verbose=False)
    assert np.sqrt(np.mean((fitted - expected)**2)) / np.mean(expected) < .02
    assert np.sqrt(fitted._rmap_info["fit_mse"]) < .25


@pytest.mark.parametrize("speed_form", ["x", "t", "xt", "stalled"])
def test_simulation_refined_rates_are_regularized_before_scoring(monkeypatch, speed_form):
    from dnascape import fitting

    count = 64
    observed = np.linspace(50., 70., count)
    original = observed.copy()
    times = np.arange(0., 305., 5.)
    speeds = {
        "x": np.linspace(1.2, 1.6, count),
        "t": np.linspace(1.2, 1.6, times.size)[None, :],
        "xt": np.outer(np.linspace(1.2, 1.6, count), np.linspace(.9, 1.1, times.size)),
        "stalled": 1.4,
    }
    captured = []
    original_simulator = fitting.rsim

    def track(**kwargs):
        result = original_simulator(**kwargs)
        captured.append((np.asarray(kwargs["firing_rate"]).copy(), result["replication_timing"].copy()))
        assert kwargs["perQ"] is True
        return result

    monkeypatch.setattr(fitting, "rsim", track)
    result = map_timing_firing(
        observed, fork_speed=speeds[speed_form], perQ=True, maxiter=2, sim_number=8,
        refine_seed=17, verbose=False, time_grid=times, resolution_time=5.,
        stall_rate=.01 if speed_form == "stalled" else 0.,
    )
    assert len(captured) == 2
    scores = [np.mean((prediction - observed)**2) for _, prediction in captured]
    winner = int(np.argmin(scores))
    np.testing.assert_array_equal(result, captured[winner][0])
    np.testing.assert_array_equal(result._rmap_info["replication_timing_fit"], captured[winner][1])
    assert result._rmap_info["fit_mse"] == pytest.approx(scores[winner])
    assert result._rmap_info["fit_periodic_smoothing_kb"] > 0
    np.testing.assert_array_equal(observed, original)


@pytest.mark.parametrize("mode", ["factorized", "mixed"])
def test_free_space_time_fits_score_the_regularized_spatial_fields(monkeypatch, mode):
    from dnascape import fitting

    count = 32
    observed = np.linspace(30., 50., count)
    rates = np.repeat(np.linspace(.0005, .0015, count)[None, :], 25, axis=0)
    rates[0] = 0.
    original = rates.copy()
    original_simulator = fitting.rsim
    captured = []

    def track(**kwargs):
        result = original_simulator(**kwargs)
        captured.append((kwargs["firing_rate"].T.copy(), result["replication_timing"].copy()))
        np.testing.assert_array_equal(kwargs["firing_rate"][:, 0], 0.)
        return result

    monkeypatch.setattr(fitting, "rsim", track)
    result = fitting._refine_spacetime_firing(
        rates, observed, fork_speed=1.4, resolution=1., resolution_time=5.,
        perQ=True, refine_maxiter=2, spacetime_sim_number=8, refine_seed=19,
        spacetime_update=mode, refine_verbose=False,
    )
    scores = [np.mean((prediction - observed)**2) for _, prediction in captured]
    winner = int(np.argmin(scores))
    np.testing.assert_array_equal(result, captured[winner][0])
    np.testing.assert_array_equal(result._rmap_info["replication_timing_fit"], captured[winner][1])
    assert result._rmap_info["fit_mse"] == pytest.approx(scores[winner])
    assert result._rmap_info["fit_periodic_smoothing_kb"] > 0
    np.testing.assert_array_equal(rates, original)
