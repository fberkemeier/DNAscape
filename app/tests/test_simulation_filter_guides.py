"""Simulation filter guides remain independent and refine on both plot axes."""

from types import SimpleNamespace

import numpy as np

from app.preview_plotting import PreviewChartHandle
from app.simulation_filter_guides import SimulationFilterAreas, add_simulation_filter_guides
from dnascape.workflows.simulation_filters import SimulationFilterSelection, SimulationFilterState


def selection(mask, *, extend=False):
    mask = np.asarray(mask, dtype=bool)
    return SimulationFilterSelection(np.packbits(mask).tobytes(), mask.shape, mask.shape[0] > 1,
        mask.shape[1] > 1, 0., 6., 1., 1., extend_time=extend)


def test_two_named_filters_can_be_toggled_independently():
    state = SimulationFilterState({'initiation_rate': selection([[1], [0], [1], [0], [0], [0]]),
                                   'fork_speed': selection([[0], [0], [0], [1], [1], [0]])})
    controls = SimpleNamespace(filter_guides={key: SimpleNamespace(value=True) for key in state.selections},
        filter_guide_names={'initiation_rate': SimpleNamespace(value='Inhibited firing'), 'fork_speed': SimpleNamespace(value='Slow forks')},
        show_legend=SimpleNamespace(value=True))
    options = add_simulation_filter_guides({'series': [], 'legend': {'data': []}}, [SimpleNamespace(filter_state=state)], controls,
                                          kind='space', time_end=10.)
    assert [series['name'] for series in options['series']] == ['Inhibited firing', 'Slow forks']
    assert len(options['series'][0]['markArea']['data']) == 2
    assert len(options['series'][1]['markArea']['data']) == 1
    controls.filter_guides['initiation_rate'].value = False
    options = add_simulation_filter_guides({'series': []}, [SimpleNamespace(filter_state=state)], controls, kind='space', time_end=10.)
    assert [series['name'] for series in options['series']] == ['Slow forks']


def test_heatmap_guide_refinement_receives_both_visible_axes():
    source = selection(np.arange(36).reshape(6, 6) % 3 == 0)
    areas = SimulationFilterAreas((source,), 'xt', 6.)
    options = {'xAxis': {'min': 0., 'max': 6.}, 'yAxis': {'min': 0., 'max': 6.},
               'series': [{'type': 'line', 'data': [], 'markArea': {'data': areas}}]}
    handle = PreviewChartHandle(SimpleNamespace(options=options), None)
    handle._navigation.axes = {key: {'dimension': key, 'index': 0, 'min': 0., 'max': 6., 'log': False} for key in ('x', 'y')}
    handle._navigation.ranges = {'x': (1., 4.), 'y': (2., 5.)}
    jobs = handle._refinement_plan()
    assert jobs[0][3] == ((1., 4.), (2., 5.))
    assert handle._install_refinements(handle._compute_refinements(jobs))
    assert options['series'][0]['markArea']['data'].window == ((1., 4.), (2., 5.))


def test_speed_filter_guide_includes_the_held_last_time_sample():
    source = selection([[0, 0, 1]], extend=True)
    assert source.areas('time', time_end=10.) == [[{'xAxis': 2.}, {'xAxis': 10.}]]
    assert source.areas('xt', y_limits=(5., 9.), time_end=10.) == [
        [{'xAxis': 0., 'yAxis': 2.}, {'xAxis': 6., 'yAxis': 10.}],
    ]
