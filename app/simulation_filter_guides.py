"""Simulation-only filter guides, with independent names and zoom refinement."""

from dnascape.workflows.simulation_filters import FILTER_LABELS


class SimulationFilterAreas(list):
    def __init__(self, selections, kind, time_end, x_limits=None, y_limits=None):
        self.selections, self.kind, self.time_end = selections, kind, time_end
        self.supports_y = kind == 'xt'
        self.window = (x_limits, y_limits) if self.supports_y else x_limits
        areas = []
        for selection in selections:
            areas.extend(selection.areas(kind, x_limits=x_limits, y_limits=y_limits,
                                         time_end=time_end, max_areas=max(1, 750 // len(selections))))
        super().__init__(areas)

    def refine(self, x_limits=None, y_limits=None):
        return SimulationFilterAreas(self.selections, self.kind, self.time_end, x_limits, y_limits)


def add_simulation_filter_guides(options, results, controls, *, kind, time_end):
    if kind not in {'space', 'time', 'xt'}:
        return options
    colors = {'initiation_rate': ('#888888', 'rgba(120,120,120,0.16)'),
              'fork_speed': ('#7794ab', 'rgba(90,125,155,0.17)')}
    for key, toggle in controls.filter_guides.items():
        if not toggle.value:
            continue
        selections = tuple(result.filter_state.selections[key] for result in results
                           if result.filter_state is not None and key in result.filter_state.selections)
        if not selections:
            continue
        name = str(controls.filter_guide_names[key].value or '').strip() or FILTER_LABELS[key]
        if any(series.get('name') == name for series in options.get('series', ())):
            name += ' (filter)'
        color, fill = colors[key]
        options.setdefault('series', []).append({
            'name': name, 'type': 'line', 'data': [], 'silent': True, 'showSymbol': False,
            'tooltip': {'show': False}, 'itemStyle': {'color': color}, 'lineStyle': {'opacity': 0},
            'z': 3 if kind == 'xt' else 0,
            'markArea': {'silent': True, 'label': {'show': False}, 'emphasis': {'disabled': True},
                         'itemStyle': {'color': fill},
                         'data': SimulationFilterAreas(selections, kind, time_end)},
        })
        legend = options.setdefault('legend', {})
        legend['show'] = bool(controls.show_legend.value)
        legend.setdefault('data', []).append({'name': name, 'icon': 'roundRect'})
    return options
