"""Regression coverage for the restored temporal and space-time fitters."""

import numpy as np
import pytest

from dnascape import fitting


@pytest.mark.parametrize("spacetime", [False, True])
def test_simulated_speed_retains_best_evaluated_field_and_t0(monkeypatch, spacetime):
    timing = np.array([10., 20., 30., 40.])
    axis = np.arange(0., 51., 10.)
    predictions = [np.array([20., 10., 30., 40.]),
                   np.array([11., 19., 30., 40.]), np.full(4, 60.)]
    evaluated = []

    def simulate(**kwargs):
        evaluated.append(np.asarray(kwargs["fork_speed"]).copy())
        return {"replication_timing": predictions[len(evaluated) - 1]}

    monkeypatch.setattr(fitting, "rsim", simulate)
    options = dict(observed_timing=timing + 100, firing_rate=.01,
                   resolution=2., resolution_time=10., time_grid=axis,
                   T0=100., speed_bounds=(.1, 6.), refine_edge_kb=0,
                   refine_verbose=False)
    if spacetime:
        result = fitting._refine_spacetime_speed(
            np.ones((axis.size, timing.size)), refine_maxiter=3,
            spacetime_space_fit_step=1., spacetime_time_fit_step=0., **options)
        expected = evaluated[1].T
        assert expected[0, 0] > 1
        assert expected[0, 1] < 1
    else:
        result = fitting._refine_temporal_speed(
            np.ones((1, axis.size)), temporal_maxiter=3,
            temporal_fit_step=1., **options)
        expected = evaluated[1]
        assert expected[0, 1] > 1
        assert expected[0, 2] < 1
    np.testing.assert_array_equal(result, expected)
    info = result._rmap_info
    np.testing.assert_array_equal(info["replication_timing_fit"], predictions[1] + 100)
    np.testing.assert_array_equal(info["residual"], predictions[1] - timing)
    assert info["fit_mse"] == .5
    assert info["fit_history"].min() == info["fit_mse"]


@pytest.mark.parametrize("spacetime", [False, True])
def test_simulated_speed_resamples_explicit_clock(monkeypatch, spacetime):
    axis = np.array([0., 7., 20.])
    speed = np.array([1., 2., 4.])
    firing = np.array([[.01, .03, .02]])
    calls = []
    def simulate(**kwargs):
        calls.append(kwargs)
        return {"replication_timing": np.full(4, 15.)}
    monkeypatch.setattr(fitting, "rsim", simulate)
    options = dict(observed_timing=np.full(4, 15.), firing_rate=firing,
                   resolution=2., resolution_time=5., time_grid=axis,
                   refine_edge_kb=0, refine_verbose=False)
    if spacetime:
        result = fitting._refine_spacetime_speed(
            np.broadcast_to(speed[:, None], (3, 4)), refine_maxiter=1, **options)
        assert result.shape == (3, 4)
    else:
        result = fitting._refine_temporal_speed(speed, temporal_maxiter=1, **options)
        assert result.shape == (1, 3)
    uniform = np.arange(0., 21., 5.)
    np.testing.assert_allclose(calls[0]["fork_speed"][0], np.interp(uniform, axis, speed))
    np.testing.assert_allclose(calls[0]["firing_rate"][0], np.interp(uniform, axis, firing[0]))


def test_sequential_spacetime_density_converted_only_at_legacy_boundary(monkeypatch):
    calls = {}
    axis = np.arange(0., 31., 5.)
    def spatial(source, **kwargs):
        calls["spatial"] = (source, kwargs)
        return np.full(4, 1.4)
    def temporal(timing, **kwargs):
        calls["temporal"] = (timing, kwargs)
        return fitting._RmapInfoArray(np.full((1, axis.size), 1.4), {})
    monkeypatch.setattr(fitting, "map_timing_firing_spatial_speed", spatial)
    monkeypatch.setattr(fitting, "_fit_exponential_speed_shape", temporal)
    source = {"replication_timing": np.arange(20., 24.), "firing_rate": .002}
    fitting.map_timing_firing_spacetime_speed_simulated(
        source, resolution=4, time_grid=axis, spacetime_order="space_then_time",
        T0=10, _firing_rate_is_density=True, verbose=False)
    np.testing.assert_allclose(calls["spatial"][0]["firing_rate"], .008)
    np.testing.assert_allclose(calls["temporal"][1]["firing_rate"], .002)
    assert calls["spatial"][1]["T0"] == 10
    assert calls["temporal"][1]["T0"] == 10
    assert source["firing_rate"] == .002


@pytest.mark.parametrize("general", [False, True])
def test_temporal_best_keeps_density_and_t0(monkeypatch, general):
    timing = np.linspace(8., 18., 12)
    axis = np.arange(0., 41., 5.)
    firing = .002 if not general else np.full((1, axis.size), .002)
    source = {"replication_timing": timing, "firing_rate": firing}
    entry = (fitting.map_timing_firing_temporal_speed_simulated if general
             else fitting.map_timing_firing_temporal_speed)
    options = dict(resolution=4, time_grid=axis, temporal_maxiter=4,
                   refine_edge_kb=0, verbose=False, _firing_rate_is_density=True)
    def forbidden(*args, **kwargs):
        raise AssertionError("Linear historical fit must not invoke the new inverse")
    monkeypatch.setattr(fitting, "_periodic_speed_inverse", forbidden)
    result = entry(source, **options)
    shifted = entry(dict(source, replication_timing=timing + 100), T0=100, **options)
    np.testing.assert_allclose(result, shifted, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(result._rmap_info["replication_timing_fit"] + 100,
                               shifted._rmap_info["replication_timing_fit"])
    fraction = fitting.map_replicated_fraction_xt(
        firing, fork_speed=result, resolution=4, resolution_time=5,
        time_grid=axis, n_bins=12)
    expected = np.trapezoid(1 - fraction, axis, axis=1)
    np.testing.assert_allclose(result._rmap_info["replication_timing_fit"], expected)


@pytest.mark.parametrize("general", [False, True])
def test_crossed_spacetime_best_is_offset_invariant_and_retains_improvements(monkeypatch, general):
    timing = np.linspace(8., 18., 12)
    axis = np.arange(0., 41., 5.)
    firing = .002 if general else np.full((12, axis.size), .002)
    source = {"replication_timing": timing, "firing_rate": firing}
    entry = (fitting.map_timing_firing_spacetime_speed_simulated if general
             else fitting.map_timing_firing_spacetime_speed)
    options = dict(resolution=4, time_grid=axis, space_maxiter=4, temporal_maxiter=4,
                   refine_edge_kb=0, smoothw=None, verbose=False,
                   _firing_rate_is_density=True)
    def forbidden(*args, **kwargs):
        raise AssertionError("Linear historical fit must not invoke the new inverse")
    monkeypatch.setattr(fitting, "_periodic_speed_inverse", forbidden)
    result = entry(source, **options)
    shifted = entry(dict(source, replication_timing=timing + 100), T0=100, **options)
    np.testing.assert_allclose(result, shifted, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(result._rmap_info["replication_timing_fit"] + 100,
                               shifted._rmap_info["replication_timing_fit"])
    history = result._rmap_info["fit_history"]
    errors = [step["mse"] for step in history]
    assert np.all(np.diff(errors) <= 0)
    assert result.shape == (12, axis.size)


@pytest.mark.parametrize("name,form", [
    ("map_timing_firing_temporal_speed", "time"),
    ("map_timing_firing_temporal_speed_simulated", "time"),
    ("map_timing_firing_spacetime_speed", "spacetime"),
    ("map_timing_firing_spacetime_speed_simulated", "spacetime"),
])
def test_periodic_geometry_keeps_supported_inverse(monkeypatch, name, form):
    calls = []
    def periodic(*args):
        calls.append(args)
        return np.ones((1, 3))
    monkeypatch.setattr(fitting, "_periodic_speed_inverse", periodic)
    source = {"replication_timing": np.ones(4), "firing_rate": .002}
    getattr(fitting, name)(source, perQ=True, _firing_rate_is_density=True, verbose=False)
    assert calls[0][0] is source
    assert calls[0][1] == form
