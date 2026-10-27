"""Independent simulation filters operate on the sampled kinetic inputs."""

import io
from dataclasses import replace

import numpy as np
import pytest

from dnascape.workflows.filtering import DataFilterSettings, FilterProfile, apply_data_filter
from dnascape.workflows.modulation import ProfileModulationSettings
from dnascape.workflows.simulation import ConstantSimulationRequest, run_constant_simulation, validate_constant_simulation
from dnascape.workflows.simulation_filters import SimulationFilterSelection, build_simulation_filter_export


def request(**changes):
    values = dict(initiation_rate=.2, fork_speed=2., resolution_x=1., resolution_t=1.,
                  region_start_kb=0., region_end_kb=6., replicates=2, geometry='linear',
                  time_window=10., max_replication_time=100., seed=3)
    values.update(changes)
    return ConstantSimulationRequest(**values)


def test_both_filters_select_on_and_edit_their_own_inputs():
    firing = np.array([.1, .3, .1, .4, .2, .5])
    speed = np.array([1., 2., 3., 1., 2., 3.])
    original_firing, original_speed = firing.copy(), speed.copy()
    validated = validate_constant_simulation(request(initiation_rate=firing, fork_speed=speed,
        data_filters={'initiation_rate': DataFilterSettings(enabled=True, threshold=.2, value=0.),
                      'fork_speed': DataFilterSettings(enabled=True, selection='between', lower=2., upper=3., action='multiply', value=2.)},
        export_filters=frozenset({'initiation_rate', 'fork_speed'})))
    np.testing.assert_array_equal(validated.request.initiation_rate, [0., .3, 0., .4, .2, .5])
    np.testing.assert_array_equal(validated.request.fork_speed, [1., 4., 6., 1., 4., 6.])
    np.testing.assert_array_equal(firing, original_firing)
    np.testing.assert_array_equal(speed, original_speed)
    result = run_constant_simulation(validated.request)
    assert set(result.filter_state.selections) == {'initiation_rate', 'fork_speed'}
    from dnascape.workflows.preview_inputs import INPUT_PREFIX
    assert result.input_forms[INPUT_PREFIX + 'initiation_rate'] == 'x'


def test_whole_region_and_filters_use_the_resampled_genomic_grid():
    profile = np.arange(10.) / 100 + .1
    validated = validate_constant_simulation(request(initiation_rate=profile, source_resolution_x=1., resolution_x=2.,
        region_start_kb=3., region_end_kb=6., whole_region=True,
        data_filters={'initiation_rate': DataFilterSettings(enabled=True, selection='profile', value=.02,
            profile=FilterProfile(np.array([0., 4_000.]), 'positions', 'bp'))}))
    assert validated.request.region_start_kb == 0.
    assert validated.request.region_end_kb == 10.
    assert validated.n_bins == 5
    np.testing.assert_allclose(validated.request.initiation_rate, [.02, .125, .02, .165, .185])


@pytest.mark.parametrize('value', [.1, np.array([[.1, .2, .3]])])
def test_whole_region_requires_genomic_profile_coordinates(value):
    with pytest.raises(ValueError, match='genomic positions'):
        validate_constant_simulation(request(initiation_rate=value, whole_region=True))


def test_single_position_profile_can_define_the_whole_region():
    validated = validate_constant_simulation(request(initiation_rate=np.array([.2]), whole_region=True))
    assert validated.n_bins == 1 and validated.request.region_end_kb == 1.


def test_scalar_and_temporal_inputs_expand_for_position_filters():
    settings = DataFilterSettings(enabled=True, selection='profile', value=.5,
                                  profile=FilterProfile(np.array([1_000.]), 'positions', 'bp'))
    scalar = validate_constant_simulation(request(data_filters={'initiation_rate': settings}))
    np.testing.assert_array_equal(scalar.request.initiation_rate, [.2, .5, .2, .2, .2, .2])
    temporal = validate_constant_simulation(request(initiation_rate=np.array([[.1, .2, .3]]), data_filters={'initiation_rate': settings}))
    assert temporal.request.initiation_rate.shape == (6, 3)
    np.testing.assert_array_equal(temporal.request.initiation_rate[1], .5)
    np.testing.assert_array_equal(temporal.request.initiation_rate[0], [.1, .2, .3])


def test_temporal_selection_keeps_a_temporal_profile_and_exact_regions():
    validated = validate_constant_simulation(request(initiation_rate=np.array([[.1, .2, .1, .4]]),
        data_filters={'initiation_rate': DataFilterSettings(enabled=True, threshold=.15, value=0.)}))
    np.testing.assert_array_equal(validated.request.initiation_rate, [[0., .2, 0., .4]])
    selection = validated.request._filter_state.selections['initiation_rate']
    assert selection.areas('time', time_end=10.) == [[{'xAxis': 0.}, {'xAxis': 1.}], [{'xAxis': 2.}, {'xAxis': 3.}]]
    assert selection.areas('xt', time_end=10.) == [
        [{'xAxis': 0., 'yAxis': 0.}, {'xAxis': 6., 'yAxis': 1.}],
        [{'xAxis': 0., 'yAxis': 2.}, {'xAxis': 6., 'yAxis': 3.}],
    ]


@pytest.mark.parametrize('action', ['replace', 'mean', 'interpolate', 'minimum', 'maximum', 'noise', 'multiply', 'add', 'modulate'])
def test_transforms_match_data_manager(action):
    source = np.array([.1, .3, .1, .4, .2, .5])
    settings = DataFilterSettings(enabled=True, threshold=.2, action=action, value=.05,
        standard_deviation=.001, seed=7,
        modulation=ProfileModulationSettings(operation='expression', expression='source * profile', values=np.full(6, 2.)))
    expected = apply_data_filter(source, settings, positions=np.arange(6.))
    actual = validate_constant_simulation(request(initiation_rate=source, data_filters={'initiation_rate': settings}))
    np.testing.assert_allclose(actual.request.initiation_rate, expected)


def test_missing_regions_can_be_fixed_before_simulation_validation():
    validated = validate_constant_simulation(request(initiation_rate=np.array([.1, np.nan, .3, .4, .5, .6]),
        data_filters={'initiation_rate': DataFilterSettings(enabled=True, selection='nan', value=.2)}))
    np.testing.assert_array_equal(validated.request.initiation_rate, [.1, .2, .3, .4, .5, .6])


def test_noise_is_seeded_and_prepared_filters_are_not_applied_twice():
    filters = {key: DataFilterSettings(enabled=True, selection='all', action='noise', standard_deviation=.001)
               for key in ('initiation_rate', 'fork_speed')}
    original = request(data_filters=filters)
    first = validate_constant_simulation(original).request
    second = validate_constant_simulation(original).request
    repeated = validate_constant_simulation(first).request
    for key in filters:
        np.testing.assert_array_equal(getattr(first, key), getattr(second, key))
        np.testing.assert_array_equal(getattr(first, key), getattr(repeated, key))
    assert not first.data_filters
    assert not np.allclose(first.initiation_rate - .2, first.fork_speed - 2.)


def test_filter_exports_have_integer_positions_and_no_headers():
    state = validate_constant_simulation(request(data_filters={'initiation_rate': DataFilterSettings(enabled=True,
        selection='profile', value=.1, profile=FilterProfile(np.array([[1000., 3000.]]), 'intervals', 'bp'))},
        export_filters=frozenset({'initiation_rate'}))).request._filter_state
    assert build_simulation_filter_export(state, 'initiation_rate', 'tsv').data == b'1000\n2000\n'
    assert build_simulation_filter_export(state, 'initiation_rate', 'csv', layout='intervals').data == b'1000,3000\n'
    with np.load(io.BytesIO(build_simulation_filter_export(state, 'initiation_rate', 'npz').data)) as archive:
        assert archive.files == ['positions']
        np.testing.assert_array_equal(archive['positions'], [1000, 2000])


def test_region_guides_are_bounded_and_recover_detail_when_zoomed():
    mask = np.arange(10_000) % 2 == 0
    selection = SimulationFilterSelection(np.packbits(mask).tobytes(), (len(mask), 1), True, False, 0., 10_000., 1., 1.)
    assert len(selection.areas(max_areas=20)) <= 20
    assert len(selection.areas(x_limits=(9980., 9999.), max_areas=20)) == 10
    temporal = SimulationFilterSelection(np.packbits(mask).tobytes(), (1, len(mask)), False, True, 0., 6., 1., 1.)
    assert len(temporal.areas('xt', time_end=10_000., max_areas=20)) <= 20


def test_disabled_or_empty_filters_preserve_seeded_simulation_results():
    baseline = run_constant_simulation(request())
    filtered = run_constant_simulation(request(data_filters={'initiation_rate': DataFilterSettings(enabled=True)}))
    np.testing.assert_array_equal(baseline.replication_timing, filtered.replication_timing)


def test_nonpositive_fork_speed_is_rejected_after_filtering():
    with pytest.raises(ValueError, match='greater than zero'):
        validate_constant_simulation(request(data_filters={'fork_speed': DataFilterSettings(enabled=True, selection='all', value=0.)}))
