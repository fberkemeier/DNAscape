"""Real forward/inverse checks for temporal and space-time fork speeds."""

from itertools import product

import numpy as np
import pytest

from dnascape import fitting
from dnascape.mapping import map_replicated_fraction_xt
from dnascape.routing import rmap
from dnascape.utils import progress_context


@pytest.mark.parametrize("firing_form,speed_form", tuple(product(
    ("scalar", "x", "t", "xt"), ("time", "spacetime"),
)))
def test_speed_fit_updates_time_and_keeps_the_actual_forward_timing(firing_form, speed_form):
    n = 17
    times = np.arange(0., 84., 4.)
    space = np.linspace(0., 1., n)
    firing = {
        "scalar": .003,
        "x": .003 * (1 + .5 * np.sin(2 * np.pi * space)),
        "t": (.003 * np.exp(-times / 70))[None, :],
        "xt": .003 * np.outer(1 + .5 * np.sin(2 * np.pi * space), np.exp(-times / 70)),
    }[firing_form]
    before = np.asarray(firing).copy()
    speed = 1.1 * np.exp(1.1 * times / times[-1])[None, :]
    if speed_form == "spacetime":
        speed = (1 + .25 * np.cos(2 * np.pi * space))[:, None] * speed
    offset = 7.

    def forward(field):
        fraction = map_replicated_fraction_xt(
            firing, field, resolution=4., time_grid=times, n_bins=n,
        )
        return np.trapezoid(1. - fraction, times, axis=1) + offset

    timing = forward(speed)
    initial = np.full(speed.shape, 1.4)
    initial_mse = np.mean((forward(initial) - timing) ** 2)
    metrics = []
    with progress_context(metric_report=lambda *metric: metrics.append(metric), suppress_output=True):
        result = rmap(
            source={"replication_timing": timing, "firing_rate": firing}, target="fork_speed",
            target_form=speed_form, resolution=4., time_grid=times, T0=offset,
            refine_edge_kb=0., smoothw=5, verbose=False, maxiter=10,
        )
    assert result.shape == speed.shape
    assert np.all(np.isfinite(result)) and np.all(result > 0)
    assert np.max(np.ptp(result, axis=1)) > .3
    if speed_form == "spacetime":
        assert np.max(np.ptp(result, axis=0)) > .03
    info = result._rmap_info
    fitted = np.asarray(info["replication_timing_fit"])
    assert np.all(np.isfinite(fitted))
    np.testing.assert_allclose(fitted, forward(result), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(info["residual"], fitted - timing)
    np.testing.assert_allclose(info["fit_mse"], np.mean((fitted - timing) ** 2))
    assert info["fit_mse"] < initial_mse
    if speed_form == "time":
        assert info["fit_mse"] < 1e-3
    budget = 10 if speed_form == "time" else 20
    assert metrics and all(0 <= metric[1] <= metric[2] == budget for metric in metrics)
    assert all(np.isfinite(metric[3]) for metric in metrics)
    np.testing.assert_array_equal(firing, before)


@pytest.mark.parametrize("sampled", (False, True))
def test_spacetime_timing_diagnostic_keeps_all_positions_when_edges_are_excluded(monkeypatch, sampled):
    n = 1601 if sampled else 17
    times = np.asarray([0., 5., 10.])
    firing = np.full((1, times.size), .003)
    speed = np.full((times.size, n), 1.4)
    if sampled:
        monkeypatch.setattr(fitting, "_FULL_TIMING_DIAGNOSTIC_CELL_CAP", 0)
    info = fitting._spacetime_speed_timing_fit_info(
        speed, np.full(n, 8.), firing, 1., 5., times, refine_edge_kb=4.,
    )
    assert info["replication_timing_fit_interpolated"] is sampled
    assert info["replication_timing_fit"].shape == (n,)
    assert np.all(np.isfinite(info["replication_timing_fit"]))
    assert np.all(np.isfinite(info["residual"]))


def test_crossed_speed_fit_respects_separate_phase_budgets():
    result = fitting._fit_crossed_spacetime_speed(
        np.linspace(10., 20., 9), .003, time_grid=np.arange(0., 41., 5.),
        space_maxiter=0, temporal_maxiter=2, refine_edge_kb=0., verbose=False,
    )[3]
    assert [step["phase"] for step in result["fit_history"]] == ["time", "time"]
    assert result["fork_speed_crossed_space_steps"] == 0


@pytest.mark.parametrize("firing_form,speed_form", tuple(product(
    ("scalar", "x", "t", "xt"), ("time", "spacetime"),
)))
def test_speed_fit_preserves_a_genuinely_constant_forward_solution(firing_form, speed_form):
    n = 9
    times = np.arange(0., 61., 5.)
    firing = {
        "scalar": .003,
        "x": np.linspace(.002, .004, n),
        "t": np.linspace(.004, .002, times.size)[None, :],
        "xt": np.outer(np.linspace(.002, .004, n), np.linspace(1., .5, times.size)),
    }[firing_form]
    field = np.full((1 if speed_form == "time" else n, times.size), 1.4)
    fraction = map_replicated_fraction_xt(firing, field, resolution=4., time_grid=times, n_bins=n)
    result = rmap(
        source={"replication_timing": np.trapezoid(1. - fraction, times, axis=1), "firing_rate": firing},
        target="fork_speed", target_form=speed_form, resolution=4., time_grid=times,
        refine_edge_kb=0., smoothw=5, verbose=False, maxiter=4,
    )
    np.testing.assert_allclose(result, 1.4, rtol=1e-8, atol=1e-8)
    assert result._rmap_info["fit_mse"] < 1e-14
