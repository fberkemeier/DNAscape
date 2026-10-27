"""The compact temporal step keeps the native cone integral and its derivative."""

import numpy as np
import pytest

from dnascape._temporal_speed import TemporalSpeedPrediction
from dnascape.mapping import map_replicated_fraction_xt


@pytest.mark.parametrize("form", ["scalar", "space", "time", "spacetime"])
@pytest.mark.parametrize("leading", [False, True])
def test_cached_temporal_prediction_and_jacobian_match_the_native_forward(form, leading):
    n = 37
    axis = np.arange(0., 121., 3.)
    if leading:
        axis = axis[1:]
    x = np.arange(n) * .7
    rates = {"scalar": .004, "space": .004 * (1. + .2 * np.sin(x)),
             "time": .004 * (1. + .1 * np.cos(axis / 23.))[None, :],
             "spacetime": .004 * (1. + .2 * np.sin(x))[:, None] * (1. + .1 * np.cos(axis / 23.))[None, :]}[form]
    sites = np.arange(0, n, 3)
    model = TemporalSpeedPrediction(rates, n, sites, axis, .7, 3.)
    normalized = (axis - axis[0]) / 80.
    parameters = np.array([.15, .3])

    def field(params):
        return np.exp(params[0] + params[1] * normalized)

    speed = field(parameters)
    tangent = np.column_stack((speed, speed * normalized))
    predicted, jacobian, survival = model.predict(speed, tangent)
    forward = map_replicated_fraction_xt(rates, speed[None, :], resolution=.7, time_grid=axis, n_bins=n)
    np.testing.assert_allclose(predicted, np.trapezoid(1. - forward, axis, axis=1)[sites], atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(survival, 1. - forward[sites, -1], atol=1e-15, rtol=1e-10)
    for column in range(2):
        delta = np.zeros(2)
        delta[column] = 1e-6
        numerical = (model.predict(field(parameters + delta))[0] - model.predict(field(parameters - delta))[0]) / 2e-6
        np.testing.assert_allclose(jacobian[:, column], numerical, atol=2e-7, rtol=2e-6)


@pytest.mark.parametrize("form", ["scalar", "space", "time", "spacetime"])
def test_time_target_reuses_the_guarded_temporal_stage_without_fitting_space(monkeypatch, form):
    from dnascape import fitting, rmap

    n = 64
    axis = np.arange(0., 181., 5.)
    rates = {"scalar": .003, "space": np.geomspace(.0003, .003, n),
             "time": (.003 * (1. + .2 * np.sin(axis / 30.)))[None, :],
             "spacetime": np.geomspace(.0003, .003, n)[:, None] * (1. + .2 * np.sin(axis / 30.))[None, :]}[form]
    truth = (1.1 * (1. + .7 * axis / 70.))[None, :]
    forward = map_replicated_fraction_xt(rates, truth, resolution=5., time_grid=axis, n_bins=n)
    observed = np.trapezoid(1. - forward, axis, axis=1)
    original = fitting._fit_linear_increasing_temporal_factor_for_separable_speed
    spatial_components = []

    def record_temporal_step(timing, rate, spatial_speed, **kwargs):
        spatial_components.append(np.asarray(spatial_speed).copy())
        return original(timing, rate, spatial_speed, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("The temporal target must not fit a spatial component or use the old full-prediction optimizer.")

    monkeypatch.setattr(fitting, "_fit_linear_increasing_temporal_factor_for_separable_speed", record_temporal_step)
    monkeypatch.setattr(fitting, "map_timing_firing_spatial_speed", forbidden)
    monkeypatch.setattr(fitting, "_fit_exponential_speed_shape", forbidden)
    monkeypatch.setattr(fitting, "rsim", forbidden)
    result = rmap(source={"replication_timing": observed, "initiation_rate": rates}, target="fork_speed",
        assumptions=dict(target_form="time", time_grid=axis, resolution=5., speed_bounds=(.2, 6.),
                         refine_edge_kb=0., temporal_maxiter=10), return_info=True)
    assert len(spatial_components) == 1
    np.testing.assert_array_equal(spatial_components[0], np.full(n, 1.4))
    assert result["fit_mse"] < 1e-4
    assert np.ptp(result["values"]) > .5
    np.testing.assert_array_equal(result["time_grid"], axis)
    np.testing.assert_allclose(result["residual"], result["replication_timing_fit"] - observed)


def test_temporal_prediction_can_be_cancelled_during_one_forward_call():
    from concurrent.futures import CancelledError
    from dnascape.utils import progress_context

    axis = np.arange(0., 241., 3.)
    model = TemporalSpeedPrediction(.003, 96, np.arange(0, 96, 3), axis, 1., 3.)
    checks = 0

    def cancel():
        nonlocal checks
        checks += 1
        if checks == 8:
            raise CancelledError("Mapping stopped")

    with progress_context(cancel=cancel), pytest.raises(CancelledError, match="Mapping stopped"):
        model.predict(np.full(axis.size, 1.4))
    assert checks == 8


def test_zero_iterations_preserve_the_initial_temporal_profile_and_complete_its_tail():
    from dnascape._temporal_speed import fit_temporal_speed

    axis = np.arange(0., 601., 5.)
    rates = np.full(4, .002)
    forward = map_replicated_fraction_xt(rates, np.full((1, axis.size), 1.4), resolution=10., time_grid=axis, n_bins=4)
    observed = np.trapezoid(1. - forward, axis, axis=1)
    fitted = fit_temporal_speed(observed, rates, resolution=10., maxiter=0, speed_bounds=(.2, 6.), refine_edge_kb=0.)
    np.testing.assert_array_equal(fitted, np.full(fitted.shape, 1.4))
    np.testing.assert_allclose(fitted._rmap_info["replication_timing_fit"], observed, atol=1e-7)
