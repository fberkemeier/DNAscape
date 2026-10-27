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
        source={"replication_timing": timing, "initiation_rate": firing},
        target="fork_speed", target_form="space", resolution=resolution,
        perQ=periodic, maxiter=40, smoothw=None, refine_edge_kb=0., verbose=False,
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
        {"replication_timing": timing, "initiation_rate": .004},
        maxiter=1, smoothw=None, refine_edge_kb=0., verbose=False,
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
        {"replication_timing": timing, "initiation_rate": .001}, **settings,
    )
    shifted = map_timing_firing_spatial_speed(
        {"replication_timing": timing + 37., "initiation_rate": .001}, T0=37., **settings,
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
        {"replication_timing": timing, "initiation_rate": .01},
        resolution=100., maxiter=1, smoothw=None, refine_edge_kb=0., verbose=False,
    )
    assert result[0] < result[15] < result[-1]


def test_smoothing_preserves_a_uniform_periodic_update():
    timing = map_firing_timing(.004, np.full(48, 2.8), perQ=True)
    result = map_timing_firing_spatial_speed(
        {"replication_timing": timing, "initiation_rate": .004},
        perQ=True, maxiter=1, smoothw=20, verbose=False,
    )
    assert result[0] > 1.4
    np.testing.assert_allclose(result, result[0], rtol=1e-13)


@pytest.mark.parametrize("spatial_firing", [False, True])
def test_reflected_edge_context_prevents_missing_flanks_from_becoming_speed_spikes(spatial_firing):
    n = 200
    firing = np.full(n, .001)
    if spatial_firing:
        firing *= 1 + .3 * np.cos(np.linspace(0., 2 * np.pi, n))
    extended_firing = np.pad(firing, (100, 100), mode="reflect")
    timing = map_firing_timing(
        extended_firing, np.full(extended_firing.size, 1.8),
        resolution=10., neighbourhood_range=200.,
    )[100:-100]
    source = {"replication_timing": timing, "initiation_rate": firing / 10. if spatial_firing else .0001}
    common = dict(
        source=source, target="fork_speed", target_form="space",
        resolution=10., neighbourhood_range=200., maxiter=30, verbose=False,
    )
    protected = rmap(**common)
    unprotected = rmap(refine_edge_kb=0., **common)

    info = protected._rmap_info
    edge_bins = info["fit_edge_bins"]
    assert 0 < edge_bins <= n // 10
    assert info["fit_taper_bins"] == 5
    assert protected.shape == timing.shape
    assert np.all(protected > 1.4)  # Every bin remains refinable, including endpoints.
    core = np.median(protected[edge_bins:-edge_bins])
    edge_peak = max(protected[:edge_bins].max(), protected[-edge_bins:].max())
    assert edge_peak < 1.05 * core
    assert max(unprotected[0], unprotected[-1]) > 1.3 * core
    assert np.linalg.norm(protected - 1.8) < .12 * np.linalg.norm(unprotected - 1.8)
    assert np.all(np.diff(info["fit_history"]) < 0)
    assert info["fit_mse"] == pytest.approx(
        np.mean((info["replication_timing_fit"][edge_bins:-edge_bins] - timing[edge_bins:-edge_bins]) ** 2),
    )
    np.testing.assert_array_equal(source["replication_timing"], timing)
    if spatial_firing:
        np.testing.assert_array_equal(source["initiation_rate"], firing / 10.)


@pytest.mark.parametrize("resolution,expected_bins", [(10., 2), (4., 5)])
def test_speed_edge_context_is_a_physical_upper_bound_with_a_local_taper(resolution, expected_bins):
    result = map_timing_firing_spatial_speed(
        {"replication_timing": np.full(200, 30.), "initiation_rate": .001},
        resolution=resolution, refine_edge_kb=20., maxiter=1, verbose=False,
    )
    assert result._rmap_info["fit_edge_bins"] == expected_bins
    assert result._rmap_info["fit_taper_bins"] == min(expected_bins, 5)


@pytest.mark.parametrize("periodic", [False, True])
def test_disabled_context_preserves_the_finite_model_and_periodic_context_is_inert(periodic):
    source = {"replication_timing": np.linspace(20., 50., 48), "initiation_rate": .001}
    common = dict(source=source, perQ=periodic, maxiter=3, smoothw=None, verbose=False)
    unprotected = map_timing_firing_spatial_speed(refine_edge_kb=0., **common)
    disabled = map_timing_firing_spatial_speed(refine_edge_kb=None, **common)
    np.testing.assert_array_equal(disabled, unprotected)
    np.testing.assert_array_equal(disabled._rmap_info["fit_history"], unprotected._rmap_info["fit_history"])
    assert disabled._rmap_info["fit_edge_bins"] == 0
    if periodic:
        protected = map_timing_firing_spatial_speed(**common)
        np.testing.assert_array_equal(protected, unprotected)
        assert protected._rmap_info["fit_edge_bins"] == 0
    else:
        predicted = map_firing_timing(.001, unprotected)
        np.testing.assert_array_equal(unprotected._rmap_info["replication_timing_fit"], predicted)


@pytest.mark.parametrize("edge_context", [-1., np.inf, np.nan])
def test_invalid_speed_edge_context_is_rejected(edge_context):
    with pytest.raises(ValueError, match="refine_edge_kb"):
        map_timing_firing_spatial_speed(
            {"replication_timing": np.full(20, 30.), "initiation_rate": .001},
            refine_edge_kb=edge_context, maxiter=0, verbose=False,
        )


@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("resolution", [1., 4.])
@pytest.mark.parametrize("profile", ["varying", "peaks", "gaps"])
def test_spatial_firing_profile_fit_improves_with_default_smoothing(
    periodic, resolution, profile,
):
    n = 256
    phase = np.linspace(0, 2 * np.pi, n, endpoint=False)
    firing = .004 * np.exp(2 * np.sin(phase))
    if profile == "peaks":
        firing = np.full(n, .00001)
        firing[::32] = .1
    elif profile == "gaps":
        firing = np.where(np.sin(phase) > .3, .01, 0.)
    truth = 1.6 + .35 * np.cos(2 * phase)
    timing = map_firing_timing(
        firing * resolution, truth, resolution=resolution, perQ=periodic,
    )
    result = rmap(
        source={"replication_timing": timing, "initiation_rate": firing},
        target="fork_speed", target_form="space", resolution=resolution,
        perQ=periodic, maxiter=40, refine_edge_kb=0., verbose=False,
    )
    info = result._rmap_info
    history = info["fit_history"]
    assert np.all(np.diff(history) < 0)
    assert info["fit_mse"] < history[0] * .08
    assert info["fit_mse"] == pytest.approx(history[-1])
    # At the strongest firing sites timing is dominated by the local origin,
    # so speed is less identifiable than elsewhere in the same profile.
    assert np.linalg.norm(result - truth) / np.linalg.norm(truth) < .04
    np.testing.assert_allclose(
        info["replication_timing_fit"],
        map_firing_timing(firing * resolution, result, resolution=resolution, perQ=periodic),
    )


def test_large_step_is_damped_without_accepting_a_worse_fit():
    phase = np.linspace(0, 2 * np.pi, 96, endpoint=False)
    firing = .004 * np.exp(np.sin(phase))
    truth = 1.6 + .35 * np.cos(phase)
    timing = map_firing_timing(firing, truth)
    result = map_timing_firing_spatial_speed(
        {"replication_timing": timing, "initiation_rate": firing},
        fit_step=8., maxiter=10, smoothw=None, verbose=False,
    )
    info = result._rmap_info
    assert np.all(np.diff(info["fit_history"]) < 0)
    assert info["fit_mse"] < info["fit_history"][0] * .01
    assert info["optimizer_evaluations"] > len(info["fit_history"])


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
        {"replication_timing": np.linspace(10., 600., 48), "initiation_rate": .001},
        maxiter=4, fit_step=8., smoothw=None, speed_bounds=(.2, 3.), verbose=False,
    )
    assert np.all((result >= .2) & (result <= 3.))
    assert result._rmap_info["fit_mse"] == min(result._rmap_info["fit_history"])


def test_spatial_speed_progress_counts_initial_evaluation_as_iteration_zero():
    messages, metrics = [], []
    truth = 1.6 + .35 * np.sin(np.linspace(0, 2 * np.pi, 96, endpoint=False))
    timing = map_firing_timing(.004, truth)
    with progress_context(
        report=lambda value, message: messages.append(message),
        metric_report=lambda *metric: metrics.append(metric), suppress_output=True,
    ):
        result = map_timing_firing_spatial_speed(
            {"replication_timing": timing, "initiation_rate": .004},
            maxiter=10, error=None, smoothw=None, refine_edge_kb=0., verbose=False,
        )
    assert len(result._rmap_info["fit_history"]) == 11
    assert metrics[0][1:3] == (0, 10)
    assert metrics[-1][1:3] == (10, 10)
    assert all(metric[2] == 10 for metric in metrics)
    assert not any("/11" in message for message in messages)


@pytest.mark.parametrize("timing,firing,kwargs,match", [
    ([1., 2.], 0., {}, "positive initiation rate"),
    ([1., 2.], .001, {"T0": 3.}, "greater than T0"),
    ([1., 2.], .001, {"fit_step": 0.}, "fit_step"),
])
def test_invalid_iteration_inputs_fail_before_fitting(timing, firing, kwargs, match):
    with pytest.raises(ValueError, match=match):
        map_timing_firing_spatial_speed(
            {"replication_timing": timing, "initiation_rate": firing}, **kwargs,
        )
