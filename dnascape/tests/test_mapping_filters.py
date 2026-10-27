"""Selections use inputs; frozen values constrain every scored fit proposal."""

from unittest.mock import patch

import numpy as np
import pytest

from dnascape import fitting
from dnascape.workflows.filtering import DataFilterSettings, FilterProfile
from dnascape.workflows.mapping_filter import MappingFilterSettings, MappingFitConstraint
from dnascape.workflows.mapping import MappingRequest, run_mapping, validate_mapping_request
from dnascape.workflows.mapping_catalog import mapping_route


def request(**changes):
    values = dict(route=mapping_route('timing_and_speed_to_initiation_rate'),
                  inputs={'replication_timing': np.linspace(20., 35., 16), 'fork_speed': 1.4},
                  input_forms={'replication_timing': 'x', 'fork_speed': 'scalar'},
                  settings={'maxiter': 3, 'initial_initiation_rate': .003}, initiation_rate_per_bin=True)
    values.update(changes)
    return MappingRequest(**values)


@pytest.mark.parametrize('periodic', [False, True])
def test_fixed_regions_are_present_in_every_forward_prediction(periodic):
    settings = MappingFilterSettings(DataFilterSettings(enabled=True, threshold=25., value=0.), 'replication_timing', True)
    calls = []
    forward = fitting._map_spatial_firing_timing_with_context

    def record(rates, *args, **kwargs):
        calls.append(np.array(rates))
        return forward(rates, *args, **kwargs)

    source = request(settings={'maxiter': 3, 'initial_initiation_rate': .003, 'perQ': periodic}, fit_filter=settings)
    original = source.inputs['replication_timing'].copy()
    with patch.object(fitting, '_map_spatial_firing_timing_with_context', side_effect=record):
        result = run_mapping(source)
    selected = original < 25.
    assert len(calls) >= 2
    for rates in calls:
        np.testing.assert_array_equal(rates[selected], 0.)
    np.testing.assert_array_equal(result.outputs['initiation_rate'][selected], 0.)
    np.testing.assert_array_equal(result.outputs['filter_positions'], np.flatnonzero(selected) * 1000)
    np.testing.assert_array_equal(source.inputs['replication_timing'], original)


@pytest.mark.parametrize('action,value,expected', [('multiply', 2., .006), ('add', .001, .004), ('replace', .01, .01)])
def test_transformed_initial_values_do_not_compound_across_steps(action, value, expected):
    result = run_mapping(request(fit_filter=MappingFilterSettings(
        DataFilterSettings(enabled=True, threshold=25., action=action, value=value), 'replication_timing')))
    np.testing.assert_allclose(result.outputs['initiation_rate'][:5], expected)


def test_positions_ignore_relative_input_and_respect_nonzero_region_and_complement():
    settings = MappingFilterSettings(DataFilterSettings(
        enabled=True, selection='profile', profile=FilterProfile(np.array([12_000., 14_000.]), 'positions', 'bp'),
        action='replace', value=.02, complement=True), 'invalid-input')
    result = run_mapping(request(region_start_kb=10., region_end_kb=26., fit_filter=settings,
                                 prewindowed_inputs=frozenset({'replication_timing'})))
    mask = ~np.isin(result.x_axis, [12., 14.])
    np.testing.assert_allclose(result.outputs['initiation_rate'][mask], .02)


def test_filter_promotes_time_target_to_space_time_without_modifying_inputs():
    source = request(route=mapping_route('timing_and_speed_to_time_initiation_rate'),
                     fit_filter=MappingFilterSettings(DataFilterSettings(enabled=True, threshold=25., value=0.), 'replication_timing'),
                     settings={'maxiter': 1, 'sim_number': 1, 'refine_seed': 3, 'refine_edge_kb': 0.})
    prepared = validate_mapping_request(source)
    assert prepared.route.output_form == 'xt'
    assert prepared.assumptions['target_shape'] == 'free'
    result = run_mapping(source)
    assert result.output_form == 'xt'
    np.testing.assert_array_equal(result.outputs['initiation_rate'][:5], 0.)
    assert np.isfinite(result.outputs['initiation_rate']).all()


def test_speed_fit_uses_frozen_values_instead_of_editing_its_input_rate():
    source = request(route=mapping_route('timing_and_firing_to_spatial_fork_speed'),
                     inputs={'replication_timing': np.linspace(20., 35., 16), 'initiation_rate': .01},
                     input_forms={'replication_timing': 'x', 'initiation_rate': 'scalar'},
                     settings={'maxiter': 2, 'initial_fork_speed': 1.4},
                     fit_filter=MappingFilterSettings(DataFilterSettings(enabled=True, threshold=25., value=2.), 'replication_timing'))
    result = run_mapping(source)
    np.testing.assert_allclose(result.outputs['fork_speed'][:5], 2.)
    assert source.inputs['initiation_rate'] == .01
    assert result.fit_mse is not None


def test_inactive_filter_preserves_original_result():
    original = run_mapping(request())
    inactive = run_mapping(request(fit_filter=MappingFilterSettings(DataFilterSettings(enabled=False, threshold=np.nan))))
    np.testing.assert_array_equal(original.outputs['initiation_rate'], inactive.outputs['initiation_rate'])


def test_noise_is_sampled_once_and_reference_does_not_change_with_proposals():
    constraint = MappingFitConstraint(
        MappingFilterSettings(DataFilterSettings(enabled=True, threshold=0., action='noise', standard_deviation=.01, seed=8)),
        target='initiation_rate', form='x', reference=np.array([-2., 1., -1., 2.]), reference_form='x',
        start=0., resolution=1., resolution_t=1.,
    )
    first = constraint.project(np.ones(4))
    second = constraint.project(np.full(4, 10.))
    np.testing.assert_array_equal(first[[0, 2]], second[[0, 2]])
    np.testing.assert_array_equal(second[[1, 3]], 10.)


def test_invalid_kinetic_transform_is_reported_before_fit_steps():
    with pytest.raises(ValueError, match='finite, non-negative initiation rates'):
        run_mapping(request(fit_filter=MappingFilterSettings(DataFilterSettings(enabled=True, selection='all', action='nan'))))


def test_headerless_integer_position_and_interval_exports_round_trip():
    from dnascape.workflows.mapping_exports import build_mapping_raw_export
    result = run_mapping(request(fit_filter=MappingFilterSettings(
        DataFilterSettings(enabled=True, threshold=25., value=0.), 'replication_timing', True)))
    assert build_mapping_raw_export(result, 'filter_positions', 'tsv').data == b'0\n1000\n2000\n3000\n4000\n'
    assert build_mapping_raw_export(result, 'filter_positions', 'csv', filter_layout='intervals').data == b'0,5000\n'


def test_recipe_carries_filter_and_auxiliary_arrays_without_web_application(tmp_path):
    from dnascape.workflows.mapping_filter import mapping_filter_recipe, mapping_filter_from_recipe
    settings = MappingFilterSettings(DataFilterSettings(
        enabled=True, selection='profile', profile=FilterProfile(np.array([1000., 3000.]), 'positions', 'bp'),
        action='multiply', value=2.), 'replication_timing', True)
    members = []
    recipe = mapping_filter_recipe(settings, members)
    for name, arrays in members:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays)
    restored = mapping_filter_from_recipe(recipe, tmp_path)
    first = run_mapping(request(fit_filter=settings))
    second = run_mapping(request(fit_filter=restored))
    np.testing.assert_array_equal(first.outputs['initiation_rate'], second.outputs['initiation_rate'])
    np.testing.assert_array_equal(first.outputs['filter_positions'], second.outputs['filter_positions'])


@pytest.mark.parametrize('action', ['mean', 'minimum', 'maximum', 'interpolate', 'noise', 'multiply', 'add', 'modulate'])
def test_all_numeric_transforms_match_shared_filter_and_stay_fixed(action):
    from dnascape.workflows.filtering import apply_data_filter
    from dnascape.workflows.modulation import ProfileModulationSettings
    source = np.arange(1., 9.)
    reference = np.array([1., -1., -1., 2., 3., -1., 3., 4.])
    settings = DataFilterSettings(enabled=True, threshold=0., action=action, value=2., standard_deviation=.01,
                                  seed=3, modulation=ProfileModulationSettings(operation='expression', expression='source * profile', values=np.full(8, 2.)))
    expected = apply_data_filter(source, settings, positions=np.arange(8.), selection_values=reference)
    constraint = MappingFitConstraint(MappingFilterSettings(settings), target='initiation_rate', form='x',
                                      reference=reference, reference_form='x', start=0., resolution=1., resolution_t=1.)
    actual = constraint.project(source)
    np.testing.assert_array_equal(actual, expected)
    second = constraint.project(source * 10.)
    np.testing.assert_array_equal(second[reference < 0], expected[reference < 0])
    np.testing.assert_array_equal(second[reference >= 0], source[reference >= 0] * 10.)


def test_profile_selection_without_upload_is_inert():
    original = run_mapping(request())
    filtered = run_mapping(request(fit_filter=MappingFilterSettings(DataFilterSettings(enabled=True, selection='profile'))))
    np.testing.assert_array_equal(original.outputs['initiation_rate'], filtered.outputs['initiation_rate'])


def test_empty_speed_selection_preserves_the_native_fit():
    from dataclasses import replace
    source = request(route=mapping_route('timing_and_firing_to_spatial_fork_speed'),
                     inputs={'replication_timing': np.linspace(20., 35., 16), 'initiation_rate': .01},
                     input_forms={'replication_timing': 'x', 'initiation_rate': 'scalar'},
                     settings={'maxiter': 2, 'initial_fork_speed': 1.4})
    original = run_mapping(source)
    filtered = run_mapping(replace(source, fit_filter=MappingFilterSettings(DataFilterSettings(enabled=True), 'replication_timing')))
    np.testing.assert_array_equal(original.outputs['fork_speed'], filtered.outputs['fork_speed'])


def test_uploaded_positions_cover_physical_bins_and_keep_partial_final_interval():
    source = np.arange(20.) + 20.
    result = run_mapping(request(inputs={'replication_timing': source, 'fork_speed': 1.4},
        source_resolution_x=1., resolution_x=10., region_start_kb=5., region_end_kb=16.,
        settings={'maxiter': 1, 'initial_initiation_rate': .003},
        fit_filter=MappingFilterSettings(DataFilterSettings(enabled=True, selection='profile', value=.02,
            profile=FilterProfile(np.array([15_800.]), 'positions', 'bp')), export=True)))
    np.testing.assert_array_equal(result.x_axis, [5., 15.])
    assert result.outputs['initiation_rate'][1] == .02
    np.testing.assert_array_equal(result.outputs['filter_positions'], [15_000])
    np.testing.assert_array_equal(result.diagnostics['filter_intervals'], [[15_000, 16_000]])


@pytest.mark.parametrize('form', ['space', 'time', 'spacetime'])
def test_projected_speed_field_sensitivities_match_finite_differences(form):
    from dnascape.inverse import _make_field
    n, t = 7, np.arange(0., 11., 2.)
    reference = np.array([0., 1., 0., 1., 0., 1.]) if form == 'time' else np.arange(n, dtype=float)
    constraint = MappingFitConstraint(MappingFilterSettings(DataFilterSettings(enabled=True, threshold=.5 if form == 'time' else 2., value=2.)),
        target='fork_speed', form={'space': 'x', 'time': 't', 'spacetime': 'xt'}[form],
        reference=reference, reference_form='t' if form == 'time' else 'x', start=0., resolution=1., resolution_t=2.)
    field = _make_field(form, None, 1.4, (.2, 6.), n, t, 2., 3, 3, 0, 'fork_speed')
    field.projection = constraint
    params = field.initial.copy()
    actual = field.evaluate(params)
    for index in (0, len(t) - 1):
        derivative = field.derivative(actual, index, field.count)
        numeric = np.empty_like(derivative)
        for column in range(field.count):
            delta = np.zeros_like(params)
            delta[column] = 1e-5
            numeric[:, column] = (field.evaluate(params + delta)[index] - field.evaluate(params - delta)[index]) / 2e-5
        np.testing.assert_allclose(derivative, numeric, atol=1e-9)
