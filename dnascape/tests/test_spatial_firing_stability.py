"""Checks for coupled descent when the historical rate update worsens MSE."""

import numpy as np
import pytest

from dnascape._spatial_firing import timing_mse_gradient
from dnascape.fitting import _map_spatial_firing_timing_with_context, map_timing_firing


@pytest.mark.parametrize("resolution,speed,edge,radius", [
    (1., 1.4, 0, 10), (25., 1.4, 3, 10), (2., .3, 4, 10), (50., 1.4, 3, 50),
])
def test_adjoint_includes_neighboring_origins_and_reflected_context(resolution, speed, edge, radius):
    rng = np.random.default_rng(42)
    rates = rng.uniform(.001, .003, 32)
    observed = rng.uniform(40., 100., rates.size)
    mask = np.ones(rates.size, dtype=bool)
    if edge:
        mask[:edge] = mask[-edge:] = False
    options = dict(fork_speed=speed, resolution=resolution,
                   neighbourhood_range=resolution * radius, fit_mask=mask, edge_bins=edge)
    mse, gradient, prediction = timing_mse_gradient(rates, observed, **options)
    historical = _map_spatial_firing_timing_with_context(
        rates, fork_speed=speed, resolution=resolution,
        neighbourhood_range=resolution * radius, perQ=False,
        edge_bins=edge, historical_linear=True,
    )
    np.testing.assert_allclose(prediction, historical, rtol=1e-11, atol=1e-10)
    assert mse == pytest.approx(np.mean((historical[mask] - observed[mask])**2))
    for position in (0, 2, 7, 16, 29, 31):
        delta = np.zeros(rates.size)
        delta[position] = rates[position] * 1e-5
        plus = timing_mse_gradient(rates + delta, observed, **options)[0]
        minus = timing_mse_gradient(rates - delta, observed, **options)[0]
        numeric = (plus - minus) / (2. * delta[position])
        assert gradient[position] == pytest.approx(numeric, rel=2e-6, abs=1e-4)


def _sharp_timing():
    timing = np.full(120, 450.)
    timing[55:59] = 160.
    return timing


def test_worsening_ratio_updates_switch_to_descent_within_ten_iteration_budget():
    observed = _sharp_timing()
    options = dict(resolution=10., maxiter=10, verbose=False)
    historical = map_timing_firing(observed, stabilize=False, **options)
    safeguarded = map_timing_firing(observed, **options)
    info = safeguarded._rmap_info
    assert np.any(np.diff(historical._rmap_info["fit_history"]) > 0)
    assert info["fit_used_descent"]
    assert len(info["fit_history"]) == 10
    assert np.all(np.diff(info["fit_history"]) <= 0)
    assert info["fit_mse"] < .9 * historical._rmap_info["fit_mse"]
    # A lower displayed score must also correspond to actual returned rates.
    prediction = _map_spatial_firing_timing_with_context(
        safeguarded, fork_speed=1.4, resolution=10., neighbourhood_range=2000.,
        perQ=False, edge_bins=info["fit_edge_bins"], historical_linear=True,
    )
    np.testing.assert_allclose(prediction, info["replication_timing_fit"], atol=1e-8)
    edge = info["fit_edge_bins"]
    score = slice(edge, -edge) if edge else slice(None)
    assert info["fit_mse"] == pytest.approx(np.mean((prediction[score] - observed[score])**2))
    assert info["fit_history"][-1] == info["fit_mse"]


def test_descent_reports_eta_and_mse_even_with_sparse_reporting(capsys):
    result = map_timing_firing(_sharp_timing(), resolution=10., maxiter=10, print_every=3)
    output = capsys.readouterr().out
    assert "[10/10]" in output and "ETA:" in output and "MSE:" in output
    assert result._rmap_info["fit_history"].size == 10


@pytest.mark.parametrize("maxiter", [0, 1])
def test_zero_or_one_iteration_still_returns_a_self_consistent_result(maxiter):
    result = map_timing_firing(_sharp_timing(), resolution=10., maxiter=maxiter, verbose=False)
    info = result._rmap_info
    assert info["fit_history"].size <= maxiter
    assert np.all(np.isfinite(result)) and np.all(result > 0)
    if maxiter:
        assert info["fit_history"][-1] == info["fit_mse"]


def test_periodic_model_keeps_its_existing_solver():
    options = dict(resolution=10., maxiter=3, perQ=True, verbose=False)
    old = map_timing_firing(_sharp_timing(), stabilize=False, **options)
    current = map_timing_firing(_sharp_timing(), **options)
    np.testing.assert_array_equal(current, old)
    assert not current._rmap_info["fit_used_descent"]
