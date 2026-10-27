"""The time-only fit must stay exact without chromosome-sized X-by-T work."""

from concurrent.futures import CancelledError
from unittest.mock import patch

import numpy as np
import pytest

from dnascape._temporal_weibull import TemporalWeibullModel
from dnascape.fitting import (
    _fit_weibull_survival_firing,
    _make_weibull_cone,
    _trapz_weights,
    _weibull_cone_loss_grad,
    _weibull_constant_speed_loss_grad,
    _weibull_hazard_samples,
    _weibull_survival_fields,
)
from dnascape.utils import progress_context


@pytest.mark.parametrize('periodic', [False, True])
@pytest.mark.parametrize('speed_form', ['constant', 'time'])
@pytest.mark.parametrize('n,dx,start', [(1, 1., 0.), (24, .3, 2.), (25, 2., 0.), (1003, 1., 0.)])
@pytest.mark.parametrize('fit_shape', [False, True])
def test_grouped_objective_and_gradient_match_full_discrete_model(periodic, speed_form, n, dx, start, fit_shape):
    t = np.arange(start, 81., 4.)
    speed = 1.3 if speed_form == 'constant' else (1.1 + .005 * t)[None, :]
    target = 26. + 3. * np.sin(np.arange(n) / 7.)
    mask = np.ones(n, dtype=bool)
    if n > 1:
        mask[::9] = False
    target[~mask] = np.nan
    delay = np.full(n, 3.)
    params = np.log([2.e-5, .7]) if fit_shape else np.log([2.e-5])
    z0 = np.log([1.e-5])
    model = TemporalWeibullModel.create(target, mask, speed, t, delay, dx, 4., periodic)
    actual = model.loss_grad(params, .7, _trapz_weights(t), z0, .2, fit_shape)
    if speed_form == 'constant' and not periodic:
        expected = _weibull_constant_speed_loss_grad(
            params, target, mask, speed, .7, dx, t, delay,
            variable_alpha=False, z0=z0, alpha_prior=.2, fit_shape=fit_shape,
        )
    else:
        operator = _make_weibull_cone(speed, t, delay, dx, 4., periodic)
        expected = _weibull_cone_loss_grad(
            params, operator, target, mask, .7, t, 0., False, z0, 0., .2, 1e-12, fit_shape,
        )
    np.testing.assert_allclose(actual[0], expected[0], rtol=2e-11, atol=1e-9)
    np.testing.assert_allclose(actual[1], expected[1], rtol=2e-10, atol=1e-9)
    fields = model.result_fields(actual[2], 5., 1e-12)
    np.testing.assert_allclose(fields['predicted_timing'], expected[2]['predicted_timing'] + 5., atol=1e-11)
    np.testing.assert_allclose(fields['survival'], expected[2]['survival'], atol=1e-12)
    np.testing.assert_allclose(fields['extended_volume'], expected[2]['extended_volume'], rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(fields['initiation_rate'], expected[2]['initiation_rate'], atol=1e-12)


@pytest.mark.parametrize('periodic', [False, True])
def test_whole_chromosome_time_fit_never_builds_a_dense_spacetime_grid(periodic):
    n = 250_000
    timing = 100. + 10. * np.sin(np.arange(n) / 10_000.)
    with patch('dnascape.fitting._weibull_constant_speed_loss_grad', side_effect=AssertionError('full objective')), \
         patch('dnascape.fitting._make_weibull_cone', side_effect=AssertionError('full speed grid')):
        result = _fit_weibull_survival_firing(
            timing, 1.3, target_form='time', perQ=periodic, maxiter=3, refine_verbose=False,
        )
    info = result._rmap_info
    assert result.shape == (1, len(info['time_grid']))
    assert info['predicted_timing'].shape == (n,)
    assert info['temporal_fit_groups'] < 500
    assert info['spacetime_diagnostics_omitted']
    assert 'survival' not in info and 'extended_volume' not in info
    assert info['initiation_rate'].strides[0] == 0
    assert np.isfinite(result).all()


@pytest.mark.parametrize('periodic', [False, True])
def test_time_fit_recovers_synthetic_timing_with_shape_bounds_and_offset(periodic):
    t = np.arange(0., 161., 2.)
    truth = _weibull_survival_fields(np.full(73, 8.e-5), 1.4, shape=1.6, time_grid=t, perQ=periodic)
    result = _fit_weibull_survival_firing(
        truth['predicted_timing'] + 7., 1.4, target_form='time', time_grid=t, T0=7., perQ=periodic,
        weibull_shape=1.9, fit_weibull_shape=True, maxiter=70, refine_edge_kb=0,
        rate_bounds=(1.e-6, 1.e-3), initial_initiation_rate=4.e-5,
    )
    assert result._rmap_info['mse'] < 1.e-7
    # A homogeneous circle supplies one mean timing, so amplitude and shape
    # cannot both be identified uniquely. Linear ends supply extra information.
    if not periodic:
        assert abs(result._rmap_info['weibull_shape'] - 1.6) < .01


@pytest.mark.parametrize('route', ['temporal', 'spatial', 'periodic_spatial', 'variable_speed'])
def test_stop_is_checked_inside_an_evaluation_even_when_verbose_is_off(route):
    calls = 0

    def cancel():
        nonlocal calls
        calls += 1
        if calls == 12:
            raise CancelledError('Mapping stopped.')

    speed = np.linspace(1., 1.5, 130) if route == 'variable_speed' else 1.3
    with progress_context(cancel=cancel), pytest.raises(CancelledError):
        _fit_weibull_survival_firing(
            np.full(130, 100.), speed, target_form='time' if route in {'temporal', 'variable_speed'} else 'spacetime',
            perQ=route == 'periodic_spatial', maxiter=10, refine_verbose=False, refine_edge_kb=0,
        )
    assert calls == 12


def test_hazard_sampling_checks_stop_between_bounded_chunks():
    calls = 0

    def cancel():
        nonlocal calls
        calls += 1
        if calls == 3:
            raise CancelledError

    with progress_context(cancel=cancel), pytest.raises(CancelledError):
        _weibull_hazard_samples(np.full(20_000, 1e-5), .7, np.arange(200.), 0.)
    assert calls == 3


def test_spatially_varying_inputs_keep_the_general_model():
    t = np.arange(0., 20., 2.)
    target = np.full(24, 15.)
    mask = np.ones(24, dtype=bool)
    assert TemporalWeibullModel.create(target, mask, np.linspace(1., 2., 24), t, np.zeros(24), 1., 2., False) is None
    assert TemporalWeibullModel.create(target, mask, 1.3, t, np.linspace(0., 2., 24), 1., 2., False) is None


@pytest.mark.parametrize('periodic', [False, True])
@pytest.mark.parametrize('fit_shape', [False, True])
def test_global_amplitude_gradient_matches_full_spatial_adjoint_for_variable_speed(periodic, fit_shape):
    n = 17
    t = np.arange(0., 31., 3.)
    speed = (1.2 + .1 * np.sin(np.arange(n)[:, None])) * (1. + t / 100.)
    delay = np.linspace(1., 3., n)
    target = np.linspace(10., 20., n)
    mask = np.arange(n) % 4 != 0
    operator = _make_weibull_cone(speed, t, delay, 1., 3., periodic)
    params = np.log([2e-4, 1.7]) if fit_shape else np.log([2e-4])
    spatial_params = np.full(n, params[0])
    if fit_shape:
        spatial_params = np.append(spatial_params, params[-1])
    actual = _weibull_cone_loss_grad(params, operator, target, mask, 1.7, t, 0.,
                                    False, params, 0., 0., 1e-12, fit_shape)
    full = _weibull_cone_loss_grad(spatial_params, operator, target, mask, 1.7, t, 0.,
                                  True, spatial_params, 0., 0., 1e-12, fit_shape)
    gradient = [np.sum(full[1][:-1] if fit_shape else full[1])]
    if fit_shape:
        gradient.append(full[1][-1])
    np.testing.assert_allclose(actual[0], full[0], atol=1e-12)
    np.testing.assert_allclose(actual[1], gradient, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize('periodic', [False, True])
def test_stochastic_temporal_fields_match_the_same_dense_seeded_model(periodic):
    from dnascape import weibull_stochastic as stochastic
    from dnascape.mapping import _expand_space_time_field

    n = 24
    t = np.arange(0., 51., 2.)
    settings = dict(target_form='time', time_grid=t, resolution_time=2., maxiter=2,
                    stall_rate_per_kb=.1, tau=3., sim_number=4, refine_seed=14,
                    refine_edge_kb=0., perQ=periodic)
    timing = np.linspace(15., 16., n)
    compact = _fit_weibull_survival_firing(timing, 1.3, **settings)
    interval_hazard = stochastic._interval_hazard

    def dense_hazard(*args, **kwargs):
        hazard = interval_hazard(*args, **kwargs)
        return np.broadcast_to(hazard, (n, hazard.shape[1])).copy()

    def dense_speed(speed, output_t, simulation_t, size, dt):
        field = _expand_space_time_field(speed, output_t.size, size, output_t, dt, 'fork_speed', positive=True)
        return np.array([np.interp(simulation_t, output_t, field[:, i]) for i in range(size)])

    with patch.object(stochastic, '_interval_hazard', side_effect=dense_hazard), \
         patch.object(stochastic, '_simulation_speed', side_effect=dense_speed):
        dense = _fit_weibull_survival_firing(timing, 1.3, **settings)
    np.testing.assert_allclose(compact, dense, rtol=1e-11)
    np.testing.assert_allclose(compact._rmap_info['predicted_timing'], dense._rmap_info['predicted_timing'], rtol=1e-11)
    np.testing.assert_allclose(compact._rmap_info['fit_history'], dense._rmap_info['fit_history'], rtol=1e-11)


@pytest.mark.parametrize('periodic', [False, True])
def test_stop_interrupts_a_single_stochastic_replicate(periodic):
    from dnascape.simulation import simulate_replication

    calls = 0

    def cancel():
        nonlocal calls
        calls += 1
        if calls == 5:
            raise CancelledError

    with progress_context(cancel=cancel), pytest.raises(CancelledError):
        simulate_replication(initiation_rate=.001, fork_speed=1.3, n_bins=25_000,
                             perQ=periodic, rng=np.random.default_rng(3))
    assert calls == 5


@pytest.mark.parametrize('dx', [1., 10.])
def test_whole_chromosome_time_fit_through_mapping_workflow_keeps_region_and_resolution(dx):
    from dnascape.workflows.mapping import MappingRequest, run_mapping
    from dnascape.workflows.mapping_catalog import mapping_route

    timing = 100. + 10. * np.sin(np.arange(250_000) / 1000.)
    result = run_mapping(MappingRequest(
        route=mapping_route('timing_and_speed_to_time_initiation_rate'),
        inputs={'replication_timing': timing, 'fork_speed': 1.3},
        input_forms={'replication_timing': 'x', 'fork_speed': 'scalar'},
        settings={'maxiter': 3}, whole_genome=True, resolution_x=dx,
        source_resolution_x=1., initiation_rate_per_bin=True,
    ))
    assert result.output_form == 't'
    assert result.outputs['initiation_rate'].shape == (result.time_axis.size,)
    assert result.outputs['replication_timing_fit'].shape == (int(250_000 / dx),)
    assert result.x_axis[0] == 0. and result.x_axis[-1] == 250_000 - dx
    assert result.comparisons['replication_timing_fit'].observed.size == int(250_000 / dx)
    assert result.fit_mse is not None and np.isfinite(result.fit_mse)


def test_cancelled_time_fit_releases_mapping_worker_for_the_next_run():
    from dnascape.workflows.mapping import MappingRequest, run_mapping
    from dnascape.workflows.mapping_catalog import mapping_route

    request = MappingRequest(
        route=mapping_route('timing_and_speed_to_time_initiation_rate'),
        inputs={'replication_timing': np.full(10_000, 100.), 'fork_speed': 1.3},
        input_forms={'replication_timing': 'x', 'fork_speed': 'scalar'}, settings={'maxiter': 2},
    )
    started = False
    checks = 0
    original = TemporalWeibullModel.loss_grad

    def observe(*args, **kwargs):
        nonlocal started
        started = True
        return original(*args, **kwargs)

    def cancel():
        nonlocal checks
        if started:
            checks += 1
            if checks == 12:
                raise CancelledError

    with patch.object(TemporalWeibullModel, 'loss_grad', observe), pytest.raises(CancelledError):
        run_mapping(request, cancel=cancel)
    assert started and checks == 12
    result = run_mapping(request)
    assert result.outputs['initiation_rate'].size > 0
