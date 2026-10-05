"""Scientific and API coverage for density-valued inverse Mapping results."""

import numpy as np
import pytest

from dnascape import fitting, rmap
from dnascape.mapping import map_firing_timing


@pytest.mark.parametrize("resolution", (0.5, 1.0, 2.0))
@pytest.mark.parametrize("form", ("constant", "space"))
def test_static_inverse_recovers_physical_density_at_multiple_resolutions(resolution, form):
    count = int(128 / resolution)
    density = 0.005
    timing = map_firing_timing(
        density * resolution, fork_speed=1.4, resolution=resolution, n_bins=count, perQ=True
    )
    result = rmap(
        {"replication_timing": timing, "fork_speed": 1.4},
        "firing_rate",
        assumptions={
            "target_form": form,
            "resolution": resolution,
            "perQ": True,
            "maxiter": 30,
            "verbose": False,
        },
        return_info=True,
    )
    np.testing.assert_allclose(result["values"], density, rtol=2e-4)
    np.testing.assert_allclose(result["replication_timing_fit"], timing, rtol=2e-4)
    assert result["firing_rate_units"] == "min^-1 kb^-1"
    # The assumed-speed public alias uses the same density boundary.
    alias = rmap(
        {"replication_timing": timing},
        "firing_rate",
        assumptions={
            "fork_speed": 1.4,
            "target_form": form,
            "resolution": resolution,
            "perQ": True,
            "maxiter": 30,
            "verbose": False,
        },
    )
    np.testing.assert_allclose(alias, result["values"])


@pytest.mark.parametrize("resolution", (0.5, 2.0))
def test_inverse_rate_bounds_are_physical_densities(resolution):
    timing = np.full(16, 12.0)
    result = rmap(
        {"replication_timing": timing, "fork_speed": 1.4},
        "firing_rate",
        assumptions={
            "target_form": "constant",
            "resolution": resolution,
            "rate_bounds": (0.003, 0.004),
            "maxiter": 0,
            "verbose": False,
        },
    )
    assert 0.003 <= float(result) <= 0.004


@pytest.mark.parametrize("form", ("constant", "space"))
def test_dynamic_speed_inverse_honors_nonuniform_source_times(form):
    timing = np.linspace(10.0, 20.0, 12)
    axis = np.asarray([0.0, 1.0, 6.0, 12.0, 30.0])
    speed = np.asarray([[0.8, 1.3, 2.0, 1.5, 1.1]])
    uniform = np.arange(0.0, 30.1, 2.0)
    resampled = np.interp(uniform, axis, speed[0])[None, :]
    function = (
        fitting.map_timing_constant_firing if form == "constant" else fitting.map_timing_firing
    )
    options = dict(
        resolution=2.0,
        resolution_time=2.0,
        refine_seed=73,
        maxiter=2,
        sim_number=20,
        verbose=False,
        perQ=True,
    )
    if form == "space":
        options["refine_edge_kb"] = 0
    actual = function(timing, fork_speed=speed, time_grid=axis, **options)
    expected = function(timing, fork_speed=resampled, **options)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(
        actual._rmap_info["replication_timing_fit"], expected._rmap_info["replication_timing_fit"]
    )
    wrong_clock = function(timing, fork_speed=resampled, **{**options, "resolution_time": 5.0})
    assert not np.allclose(
        actual._rmap_info["replication_timing_fit"],
        wrong_clock._rmap_info["replication_timing_fit"],
    )


@pytest.mark.parametrize("form", ("time", "spacetime"))
@pytest.mark.parametrize("resolution", (0.5, 2.0))
def test_free_firing_density_conversion_retains_exact_best_prediction(
    monkeypatch, form, resolution
):
    timing = np.linspace(12.0, 22.0, 12)
    options = dict(
        resolution=resolution,
        resolution_time=3.0,
        time_grid=np.arange(0.0, 30.1, 3.0),
        target_shape="free",
        space_maxiter=0,
        temporal_maxiter=1,
        temporal_sim_number=20,
        refine_edge_kb=0,
        refine_seed=17,
        verbose=False,
        perQ=True,
    )
    function = (
        fitting.map_timing_firing_time if form == "time" else fitting.map_timing_firing_spacetime
    )
    expected = function(timing, fork_speed=1.4, **options)
    calls = []
    original = fitting.rsim

    def track(**kwargs):
        calls.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(fitting, "rsim", track)
    result = rmap(
        {"replication_timing": timing, "fork_speed": 1.4},
        "firing_rate",
        assumptions={**options, "target_form": form},
        return_info=True,
    )
    np.testing.assert_allclose(result["values"], expected / resolution, atol=1e-14)
    np.testing.assert_array_equal(
        result["replication_timing_fit"], expected._rmap_info["replication_timing_fit"]
    )
    assert result["fit_mse"] == pytest.approx(
        np.mean((result["replication_timing_fit"] - timing) ** 2)
    )
    assert len(calls) == 1, (
        "The best-field prediction should be reused without an extra simulation."
    )


def test_free_nonuniform_output_reports_prediction_of_returned_density():
    timing = np.linspace(12.0, 22.0, 12)
    axis = np.asarray([0.0, 2.0, 7.0, 15.0, 30.0])
    speed = np.asarray([[1.0, 1.5, 2.0, 1.8, 1.2]])
    result = rmap(
        {"replication_timing": timing, "fork_speed": speed},
        "firing_rate",
        assumptions={
            "target_form": "time",
            "target_shape": "free",
            "resolution": 2.0,
            "resolution_time": 3.0,
            "time_grid": axis,
            "space_maxiter": 0,
            "temporal_maxiter": 1,
            "temporal_sim_number": 20,
            "refine_seed": 13,
            "refine_edge_kb": 0,
            "verbose": False,
            "perQ": True,
        },
        return_info=True,
    )
    assert result["values"].shape == (1, len(axis))
    uniform = np.arange(0.0, 30.1, 3.0)
    predicted = fitting.rsim(
        firing_rate=np.interp(uniform, axis, result["values"][0])[None, :] * 2,
        fork_speed=np.interp(uniform, axis, speed[0])[None, :],
        n_bins=12,
        resolution_space=2.0,
        resolution_time=3.0,
        seed=13,
        sim_number=20,
        perQ=True,
        verbose=False,
    )["replication_timing"]
    np.testing.assert_array_equal(result["replication_timing_fit"], predicted)
