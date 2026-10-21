"""Periodic mapping has no specially weighted or protected region endpoints."""

import numpy as np
import pytest

from dnascape.fitting import (
    _fill_nonperiodic_profile_edges,
    _fit_crop,
    _make_weibull_cone,
    _refinement_crop_mask,
    _spatial_firing_edge_controls,
    _weibull_cone_loss_grad,
    map_timing_firing,
)
from dnascape.inverse import _linear_basis, _mean_field_timing, fit_timing_fork_speed
from dnascape.weibull_stochastic import _spatial_log_roughness


@pytest.mark.parametrize("count", [12, 121])
@pytest.mark.parametrize("edge", [0., 8., 1e6])
def test_periodic_geometry_disables_all_linear_endpoint_safeguards(count, edge):
    timing = np.linspace(40., 80., count)
    mask, weights, edge_bins, taper_bins = _spatial_firing_edge_controls(
        count, 2., observed_timing=timing, fork_speed=1.4,
        neighbourhood_range=2000., refine_edge_kb=edge, perQ=True,
    )
    np.testing.assert_array_equal(mask, True)
    np.testing.assert_array_equal(weights, 1.)
    assert edge_bins == taper_bins == 0
    np.testing.assert_array_equal(_refinement_crop_mask(count, 2., True, edge), True)
    assert _fit_crop(count, 2., perQ=True) == slice(None)
    rates = np.arange(count, dtype=float)
    np.testing.assert_array_equal(
        _fill_nonperiodic_profile_edges(rates.copy(), slice(2, -2), perQ=True), rates,
    )


@pytest.mark.parametrize("count", [63, 64, 257])
def test_spatial_firing_fit_rotates_with_every_bin_and_ignores_edge_context(count):
    phase = np.arange(count) / count
    # A deliberately mismatched join must receive the same treatment as any
    # other jump after moving it into the interior of the stored array.
    observed = 65. + 8. * np.sin(4 * np.pi * phase) + 12. * phase
    original = observed.copy()
    options = dict(fork_speed=1.4, resolution=2., perQ=True, maxiter=5, fit_step=2., verbose=False)
    reference = map_timing_firing(observed, refine_edge_kb=0., **options)
    for shift, edge in ((1, 8.), (count // 3, 1e6), (count - 1, 1000.)):
        fitted = map_timing_firing(np.roll(observed, shift), refine_edge_kb=edge, **options)
        np.testing.assert_allclose(fitted, np.roll(reference, shift), rtol=2e-11, atol=1e-14)
        info = fitted._rmap_info
        assert info["fit_edge_bins"] == info["fit_taper_bins"] == 0
        np.testing.assert_allclose(info["fit_history"], reference._rmap_info["fit_history"], rtol=2e-11)
        np.testing.assert_allclose(
            info["replication_timing_fit"], np.roll(reference._rmap_info["replication_timing_fit"], shift),
            rtol=2e-11,
        )
        residual = info["replication_timing_fit"] - np.roll(observed, shift)
        assert info["fit_mse"] == pytest.approx(np.mean(residual**2), rel=2e-12)
    np.testing.assert_array_equal(observed, original)


def test_circular_interpolation_knots_have_equal_support_and_preserve_bounds():
    size, count = 24, 6
    basis = _linear_basis(size, count, periodic=True)
    for index in range(count):
        distance = np.abs((np.arange(size) - index * 4 + size / 2) % size - size / 2)
        np.testing.assert_allclose(basis[:, index], np.maximum(1 - distance / 4, 0.))
    np.testing.assert_allclose(basis.sum(axis=0), 4.)
    np.testing.assert_allclose(basis.sum(axis=1), 1.)
    assert np.min(basis) >= 0
    coefficients = np.linspace(np.log(.2), np.log(6.), count)
    values = np.exp(basis @ coefficients)
    assert np.all((values >= .2) & (values <= 6.))


@pytest.mark.parametrize("index", [0, 5, 11])
def test_speed_inverse_penalizes_a_local_perturbation_equally_at_the_join(index):
    size, strength = 12, .7
    initial = np.full(size, 1.4)
    initial[index] *= np.exp(.3)
    observed = np.full(size, 10.)
    result = fit_timing_fork_speed(
        observed, firing_rate=.008, speed_form="space", initial_fork_speed=initial,
        spatial_knots=size, regularization=strength, maxiter=0, perQ=True,
        resolution=4., time_grid=np.arange(0., 63., 3.),
    )
    # One impulse has two equally weighted neighbouring links everywhere.
    penalty = result["fit_objective"] - result["fit_mse"] / 100.
    assert penalty == pytest.approx(strength * 2 * .3**2 / size, abs=2e-14)


@pytest.mark.parametrize("form", ["space", "spacetime"])
@pytest.mark.parametrize("size,knots,shift", [(24, 6, 4), (12, 12, 5)])
def test_periodic_speed_fit_moves_with_the_ring_and_its_circular_knot_grid(form, size, knots, shift):
    t = np.arange(0., 63., 3.)
    phase = 2 * np.pi * np.arange(size) / size
    firing = .008 * (1 + .2 * np.cos(phase))
    speed = 1.5 + .3 * np.sin(phase)
    if form == "spacetime":
        speed = (1 + .1 * t[:, None] / t[-1]) * speed
    observed = _mean_field_timing(
        np.broadcast_to(firing, (t.size, size)), np.broadcast_to(speed, (t.size, size)), t, 4., True,
    )[0]
    options = dict(
        speed_form=form, perQ=True, resolution=4., time_grid=t, spatial_knots=knots,
        temporal_knots=2, regularization=.05, maxiter=18, tolerance=1e-10,
    )
    first = fit_timing_fork_speed(observed, firing_rate=firing, **options)
    shifted = fit_timing_fork_speed(np.roll(observed, shift), firing_rate=np.roll(firing, shift), **options)
    np.testing.assert_allclose(shifted["fork_speed"], np.roll(first["fork_speed"], shift, axis=0), rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(shifted["replication_timing_fit"], np.roll(first["replication_timing_fit"], shift), rtol=1e-7)
    assert shifted["fit_objective"] == pytest.approx(first["fit_objective"], rel=2e-8, abs=1e-12)


def test_automatic_periodic_speed_initialization_interpolates_across_missing_join():
    size, shift = 16, 5
    observed = 30. + 20. * np.cos(2 * np.pi * np.arange(size) / size)
    observed[[0, 1, 15]] = np.nan
    options = dict(
        firing_rate=.008, speed_form="space", initial_fork_speed=None, spatial_knots=size,
        perQ=True, resolution=4., time_grid=np.arange(0., 183., 3.), maxiter=0,
    )
    first = fit_timing_fork_speed(observed, **options)
    shifted = fit_timing_fork_speed(np.roll(observed, shift), **options)
    np.testing.assert_allclose(shifted["fork_speed"], np.roll(first["fork_speed"], shift), atol=2e-14)


def test_stochastic_weibull_smoothness_has_no_preferred_join():
    size = 16
    for position in range(size):
        impulse = np.zeros(size)
        impulse[position] = 1.
        assert _spatial_log_roughness(impulse, periodic=True) == pytest.approx(2 / size)
    values = np.random.default_rng(7).normal(size=size)
    reference = _spatial_log_roughness(values, periodic=True)
    for shift in range(size):
        assert _spatial_log_roughness(np.roll(values, shift), periodic=True) == pytest.approx(reference)
    assert _spatial_log_roughness(np.arange(5.), periodic=False) == 1.


@pytest.mark.parametrize("speed_form", ["constant", "space", "time", "spacetime"])
def test_weibull_objective_and_gradient_rotate_with_all_spatial_inputs(speed_form):
    size, shift = 12, 5
    t = np.arange(0., 33., 3.)
    phase = 2 * np.pi * np.arange(size) / size
    logs = -8 + .3 * np.sin(phase)
    observed = 12 + 2 * np.cos(phase)
    spatial_speed = 1.4 + .2 * np.sin(phase)
    temporal_speed = (1 + .2 * t / t[-1])[:, None]
    speed = {
        "constant": np.full((t.size, size), 1.4),
        "space": np.broadcast_to(spatial_speed, (t.size, size)),
        "time": np.broadcast_to(1.4 * temporal_speed, (t.size, size)),
        "spacetime": temporal_speed * spatial_speed,
    }[speed_form]
    results = []
    for amount in (0, shift):
        operator = _make_weibull_cone(np.roll(speed, amount, axis=1).T, t, np.zeros(size), 2., 3., True)
        z = np.roll(logs, amount)
        results.append(_weibull_cone_loss_grad(
            z, operator, np.roll(observed, amount), np.ones(size, dtype=bool), 2., t, 0.,
            True, z - .1, .7, .2, 1e-12, False,
        ))
    first, moved = results
    # General-speed cones cache fractional travel weights as float32.
    assert moved[0] == pytest.approx(first[0], rel=1e-7)
    np.testing.assert_allclose(moved[1], np.roll(first[1], shift), rtol=2e-7, atol=2e-7)

