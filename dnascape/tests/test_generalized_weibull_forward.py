"""Numerical validation of circular forward maps and scalable Weibull fits."""

from unittest.mock import patch

import numpy as np
import pytest

from dnascape import rmap, rsim
from dnascape.fitting import (
    _fit_weibull_survival_firing,
    _make_weibull_cone,
    _weibull_cone_loss_grad,
    _weibull_survival_fields,
)
from dnascape.lightcone import ReplicationCone
from dnascape.mapping import map_firing_timing, map_replicated_fraction_xt


@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("cache_bytes", [0, 64 * 1024**2])
def test_characteristic_cone_adjoint_matches_forward_inner_product(periodic, cache_bytes):
    rng = np.random.default_rng(11)
    t = np.array([0.0, 0.4, 1.2, 2.7, 5.0])
    speed = 0.8 + rng.random((t.size, 17))
    cone = ReplicationCone(speed, t, resolution=0.7, periodic=periodic, cache_bytes=cache_bytes)
    source = rng.random((t.size - 1, 17))
    weights = rng.normal(size=speed.shape)
    np.testing.assert_allclose(np.sum(cone.apply(source) * weights),
                               np.sum(source * cone.adjoint(weights)), rtol=1e-12, atol=1e-12)


def test_periodic_homogeneous_fraction_counts_each_origin_region_only_once():
    t = np.arange(0.0, 40.25, 0.25)
    intensity, speed, length = 0.003, 1.0, 32.0
    fraction = map_replicated_fraction_xt(intensity, fork_speed=speed, n_bins=32,
                                          time_grid=t, perQ=True)
    volume = intensity * np.where(t <= length / (2 * speed), speed * t**2,
                                  length * t - length**2 / (4 * speed))
    np.testing.assert_allclose(fraction, np.broadcast_to(-np.expm1(-volume), fraction.shape), atol=1e-12)


@pytest.mark.parametrize("start", [0.0, 2.0])
def test_homogeneous_periodic_fast_path_matches_explicit_fields(start):
    t = np.arange(start, 61.0, 2.0)
    firing = (0.002 + 0.00002 * t).reshape(1, -1)
    speed = (1.2 + 0.005 * t).reshape(1, -1)
    shared = map_replicated_fraction_xt(firing, speed, n_bins=37, time_grid=t, perQ=True)
    explicit = map_replicated_fraction_xt(np.repeat(firing, 37, axis=0),
                                          np.repeat(speed, 37, axis=0), time_grid=t, perQ=True)
    np.testing.assert_allclose(shared, explicit, atol=2e-7, rtol=2e-7)


def test_periodic_variable_fields_are_rotation_equivariant():
    t = np.arange(0.0, 31.0, 2.0)
    x = np.arange(16)
    firing = (0.008 + 0.002 * np.cos(x[:, None] / 3)) * (1 + t[None, :] / 40)
    speed = (1.2 + 0.15 * np.sin(x[:, None] / 4)) * (1 + t[None, :] / 70)
    first = map_replicated_fraction_xt(firing, speed, time_grid=t, perQ=True)
    shifted = map_replicated_fraction_xt(np.roll(firing, 5, axis=0), np.roll(speed, 5, axis=0),
                                         time_grid=t, perQ=True)
    np.testing.assert_allclose(shifted, np.roll(first, 5, axis=0), atol=1e-7)
    assert np.all(np.diff(first, axis=1) >= -1e-12)


def test_periodic_initiation_routes_distinguish_mean_field_kinetics_and_supplied_survival():
    firing, speed = 0.02, 1.3
    t = np.arange(0.0, 31.0, 1.0)
    fraction = map_replicated_fraction_xt(firing, speed, n_bins=12, time_grid=t, perQ=True)
    initiation = rmap(source={"initiation_rate": firing, "fork_speed": speed}, target="initiation_events_xt",
                      assumptions={"n_bins": 12, "time_grid": t, "perQ": True})
    # The kinetic event route solves the published mean-field transport system.
    np.testing.assert_allclose(initiation, np.broadcast_to(firing*np.exp(-firing*speed*t*t), initiation.shape), atol=1e-9)
    # A supplied fraction is authoritative, including finite-circle effects
    # outside the mean-field approximation on very short periodic molecules.
    supplied = rmap(source={"initiation_rate": firing, "replicated_fraction_xt": fraction},
                    target="initiation_events_xt", time_grid=t)
    np.testing.assert_allclose(supplied, firing * (1 - fraction))


@pytest.mark.parametrize("count", [1, 2, 3, 4, 9])
def test_periodic_timing_includes_complete_waiting_time_tail(count):
    rates = np.linspace(0.002, 0.006, count)
    speed = 1.3
    actual = map_firing_timing(rates, fork_speed=speed, perQ=True, n_bins=count)
    expected = []
    for position in range(count):
        distance = np.abs(np.arange(count) - position)
        arrivals = np.minimum(distance, count - distance) / speed
        knots = np.unique(arrivals)
        value = 0.0
        for left, right in zip(knots[:-1], knots[1:]):
            active = arrivals <= left
            survival = np.exp(-np.dot(rates, np.maximum(left - arrivals, 0)))
            intensity = np.sum(rates[active])
            value += survival * -np.expm1(-intensity * (right - left)) / intensity
        survival = np.exp(-np.dot(rates, knots[-1] - arrivals))
        expected.append(value + survival / np.sum(rates))
    np.testing.assert_allclose(actual, expected, rtol=1e-12)


def test_linear_tail_collapse_preserves_finite_neighbourhood_result():
    rates = np.linspace(0.0001, 0.001, 17)
    speed = 1.4
    survival = np.ones(17)
    integrated = np.zeros(17)
    accumulated = rates.copy()
    for radius in range(2001):
        if 0 < radius < rates.size:
            accumulated[radius:] += rates[:-radius]
            accumulated[:-radius] += rates[radius:]
        following = survival * np.exp(-accumulated / speed)
        integrated += (survival - following) / accumulated
        survival = following
    np.testing.assert_allclose(map_firing_timing(rates, speed), integrated, rtol=5e-13)


@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("speed_form", ["x", "t", "xt"])
@pytest.mark.parametrize("resolution", [0.5, 2.0])
def test_generalized_weibull_gradient_matches_finite_differences(periodic, speed_form, resolution):
    t = np.array([0.0, 1.5, 4.0, 8.0, 15.0, 25.0])
    x = np.arange(7)
    speed = {"x": 1.1 + 0.1 * x,
             "t": (1.2 + 0.01 * t).reshape(1, -1),
             "xt": (1.1 + 0.04 * x[:, None]) * (1 + 0.01 * t[None, :])}[speed_form]
    delay = np.linspace(0.0, 2.0, x.size)
    operator = _make_weibull_cone(speed, t, delay, resolution, 2.0, periodic)
    z = np.log(np.full(x.size, 4e-4))
    params = np.append(z, np.log(1.7))
    target = np.full(x.size, 15.0)
    mask = np.ones(x.size, dtype=bool)
    def evaluate(p):
        return _weibull_cone_loss_grad(p, operator, target, mask, 1.7, t, 0.0,
                                       True, z, 0.01, 0.01, 1e-12, True)
    _, gradient, _ = evaluate(params)
    step = 1e-5
    numerical = []
    for index in range(params.size):
        delta = np.zeros(params.size)
        delta[index] = step
        numerical.append((evaluate(params + delta)[0] - evaluate(params - delta)[0]) / (2 * step))
    np.testing.assert_allclose(gradient, numerical, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("resolution", [0.5, 2.0])
def test_uniform_weibull_fft_forms_preserve_density_units_and_adjoint(periodic, resolution):
    n, speed, shape, delay = 11, 1.4, 1.7, 2.0
    t = np.arange(0.0, 41.0, 2.0)
    alpha = np.linspace(1e-4, 2e-4, n)
    distance = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :]) * resolution
    if periodic:
        distance = np.minimum(distance, n * resolution - distance)
    kernels = np.maximum(t[:, None, None] - delay - distance[None, :, :] / speed, 0.0)**shape
    expected = resolution * np.einsum("txy,y->tx", kernels, alpha)
    forms = (speed, np.full(n, speed), np.full((1, t.size), speed), np.full((n, t.size), speed))
    weights = np.random.default_rng(31).normal(size=(t.size, n))
    for field in forms:
        actual = _weibull_survival_fields(alpha, field, shape=shape, resolution=resolution,
                                          time_grid=t, delay=delay, perQ=periodic)
        np.testing.assert_allclose(actual["extended_volume"].T, expected, rtol=1e-12, atol=1e-12)
        operator = _make_weibull_cone(field, t, np.full(n, delay), resolution, 2.0, periodic)
        np.testing.assert_allclose(np.sum(expected * weights),
                                   np.dot(alpha, operator.alpha_adjoint(weights, shape)),
                                   rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("resolution", [0.5, 2.0])
@pytest.mark.parametrize("target_form", ["time", "spacetime"])
def test_stochastic_weibull_converts_density_to_per_bin_hazard(resolution, target_form):
    from dnascape import weibull_stochastic

    with patch.object(weibull_stochastic, "rsim", wraps=rsim) as simulation:
        mapped = _fit_weibull_survival_firing(
            np.linspace(15.0, 20.0, 8), 1.4, resolution=resolution,
            resolution_time=2.0, time_grid=np.arange(0.0, 81.0, 2.0),
            stall_rate_per_kb=0.1, tau=2.0, sim_number=2, maxiter=0,
            refine_seed=771, refine_edge_kb=0, target_form=target_form,
        )
    info = mapped._rmap_info
    assert simulation.call_count == 2
    for call in simulation.call_args_list:
        per_bin = call.kwargs["initiation_rate"]
        simulation_t = np.arange(per_bin.shape[1]) * 2.0
        age_lo = np.maximum(simulation_t[:, None] - info["weibull_delay"], 0.0)
        age_hi = np.maximum(simulation_t[:, None] + 2.0 - info["weibull_delay"], 0.0)
        density = (info["alpha"] * (age_hi**info["weibull_shape"] - age_lo**info["weibull_shape"]) / 2.0).T
        np.testing.assert_allclose(np.broadcast_to(per_bin, density.shape), density * resolution)
        np.testing.assert_allclose(np.mean(per_bin[:, :mapped.shape[1]], axis=0),
                                   np.mean(mapped, axis=0) * resolution)


@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("speed_form", ["t", "xt"])
def test_time_dependent_speed_weibull_fit_recovers_observed_timing(periodic, speed_form):
    x = np.arange(12)
    t = np.arange(0.0, 91.0, 3.0)
    speed = (1.1 + 0.003 * t).reshape(1, -1)
    if speed_form == "xt":
        speed = (1.0 + 0.05 * np.cos(x[:, None])) * speed
    truth = _weibull_survival_fields(1e-4 * (1 + 0.1 * np.sin(x)), speed,
                                    time_grid=t, perQ=periodic)
    fit = _fit_weibull_survival_firing(truth["predicted_timing"], speed, time_grid=t,
                                      perQ=periodic, maxiter=30, refine_edge_kb=0, alpha_smooth=0)
    assert fit._rmap_info["mse"] < 1e-3
    assert fit.shape == (x.size, t.size)
    assert np.isfinite(fit).all()


def test_spatial_speed_and_activation_delay_have_no_pairwise_size_limit():
    x = np.arange(1601)
    t = np.arange(0.0, 41.0, 5.0)
    fields = _weibull_survival_fields(np.full(x.size, 2e-4), 1.2 + 0.1 * np.sin(x / 30),
                                     time_grid=t, delay=2.0 + np.sin(x / 40))
    assert fields["forward_backend"] == "characteristic_cones"
    assert fields["survival"].shape == (x.size, t.size)
    assert np.isfinite(fields["predicted_timing"]).all()


@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("target_form", ["time", "spacetime"])
def test_weibull_stalling_uses_reproducible_simulation_and_monotone_refinement(periodic, target_form):
    common = dict(time_grid=np.arange(0.0, 41.0, 2.0), perQ=periodic, target_form=target_form,
                  stall_rate_per_kb=0.2, tau=3.0, sim_number=6, refine_seed=12,
                  maxiter=3, refine_edge_kb=0)
    timing = 15.0 + np.linspace(0, 0.5, 8)
    first = _fit_weibull_survival_firing(timing, 1.3, **common)
    second = _fit_weibull_survival_firing(timing, 1.3, **common)
    np.testing.assert_array_equal(first, second)
    assert first._rmap_info["forward_backend"] == "stochastic_weibull"
    assert np.all(np.diff(first._rmap_info["fit_history"]) <= 1e-12)
    assert first._rmap_info["mse"] <= first._rmap_info["fit_history"][0]
    assert np.isfinite(first).all()


@pytest.mark.parametrize("shape", [0.7, 1.0, 2.0])
def test_weibull_delayed_hazard_has_no_pre_activation_firing_or_infinite_samples(shape):
    t = np.arange(0.0, 21.0)
    fields = _weibull_survival_fields(np.full(8, 1e-3), 1.4, shape=shape,
                                     time_grid=t, delay=5.0, rate_floor=0.0, perQ=True)
    assert np.isfinite(fields["initiation_rate"]).all()
    assert np.all(fields["initiation_rate"][:, :5] == 0)
