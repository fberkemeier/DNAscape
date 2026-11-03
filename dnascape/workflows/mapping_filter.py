"""Fixed regional constraints selected from Mapping inputs on the fit grid."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace

import numpy as np

from .filtering import DataFilterSettings, _selected, apply_data_filter, resolve_filter


@dataclass(frozen=True, slots=True)
class MappingFilterSettings:
    regions: DataFilterSettings = field(default_factory=DataFilterSettings)
    relative_to: str | None = None
    export: bool = False


def filtered_speed_settings(route, settings):
    """Expose exactly the controls used by the constrained speed solver."""
    if route.unified_kinetics:
        return tuple(settings)
    from .mapping_catalog import MappingSetting

    allowed = {'resolution', 'resolution_time', 'time_grid', 'T0', 'perQ', 'n_bins',
               'genome_length', 'interval_kb', 'x_array', 'verbose', 'speed_bounds', 'initial_fork_speed', 'maxiter'}
    result = [item for item in settings if item.key in allowed]
    keys = {item.key for item in result}
    additions = [('maxiter', 'Fit iterations', 'integer', 20, 1, 'settings'),
                 ('resolution_time', 'Resolution', 'number', 2., 1e-12, 'region'),
                 ('spatial_knots', 'Spatial control points', 'integer', 12, 1, 'settings'),
                 ('temporal_knots', 'Time control points', 'integer', 4, 1, 'settings'),
                 ('regularization', 'Smoothness', 'number', .001, 0., 'settings'),
                 ('tolerance', 'Convergence tolerance', 'number', 1e-5, 1e-15, 'settings')]
    for key, label, kind, default, minimum, location in additions:
        if key in keys or key == 'spatial_knots' and route.output_form not in {'x', 'xt'} or key == 'temporal_knots' and route.output_form not in {'t', 'xt'}:
            continue
        result.append(MappingSetting(key, label, 'fitting', location, kind, False, True, default, (),
                                     f'{label} for the constrained fork-speed fit.',
                                     unit='min' if key == 'resolution_time' else None,
                                     min_value=minimum, step=1 if kind == 'integer' else .1))
    return tuple(result)


def prepare_filter_request(request):
    """Promote shared targets before validating their grid and solver options."""
    if not request.fit_filter.regions.enabled:
        return request
    regions = resolve_filter(request.fit_filter.regions)
    request = replace(request, fit_filter=replace(request.fit_filter, regions=regions))
    if not regions.enabled:
        return request
    from difflib import SequenceMatcher
    from .mapping_catalog import mapping_routes

    route = request.route
    reference = request.fit_filter.relative_to or next(iter(request.input_forms), None)
    dependence = compatible_filter_dependence(
        route.output_dependence, request.input_forms.get(reference, 'scalar'),
        request.fit_filter.regions.selection,
    )
    if dependence != route.output_dependence:
        keys = {item.key for item in route.inputs}
        candidates = [item for item in mapping_routes(route.target_key, dependence)
                      if {spec.key for spec in item.inputs} == keys
                      and all(request.input_forms.get(spec.key) in spec.forms for spec in item.inputs)]
        if not candidates:
            raise ValueError('No compatible target is available for this region transform and these inputs.')
        route = max(candidates, key=lambda item: SequenceMatcher(None, request.route.method, item.method).ratio())
    shape = ('free' if not route.unified_kinetics and route.target == 'initiation_rate'
             and dependence in {'time', 'spacetime'} else route.default_target_shape)
    active = route.active_settings(shape, request.input_forms)
    if route.target == 'fork_speed' and any(item.key == 'replication_timing' for item in route.inputs):
        active = filtered_speed_settings(route, active)
    accepted = {item.key for item in active}
    settings = {key: value for key, value in request.settings.items() if key in accepted}
    if 'maxiter' in request.settings and 'maxiter' not in accepted:
        for key in ('space_maxiter', 'temporal_maxiter'):
            if key in accepted:
                settings.setdefault(key, request.settings['maxiter'])
    if 'maxiter' in accepted and 'maxiter' not in settings:
        iterations = [request.settings[key] for key in ('space_maxiter', 'temporal_maxiter') if request.settings.get(key) is not None]
        if iterations:
            settings['maxiter'] = max(iterations)
    if 'fit_step' in request.settings and 'fit_step' not in accepted:
        for key in ('space_fit_step', 'temporal_fit_step'):
            if key in accepted:
                settings.setdefault(key, request.settings['fit_step'])
    return replace(request, route=route, target_shape=shape, settings=settings,
                   fit_filter=replace(request.fit_filter, relative_to=reference))


def compatible_filter_dependence(dependence, reference_form, selection):
    spatial = selection in {'profile', 'ranges'} or (selection != 'all' and reference_form in {'x', 'xt'})
    temporal = selection not in {'profile', 'ranges', 'all'} and reference_form in {'t', 'xt'}
    if dependence == 'constant':
        return 'spacetime' if spatial and temporal else 'space' if spatial else 'time' if temporal else dependence
    if dependence == 'time' and spatial or dependence == 'space' and temporal:
        return 'spacetime'
    return dependence


class MappingFitConstraint:
    def __init__(self, settings, *, target, form, reference, reference_form,
                 start, resolution, resolution_t, reference_time=None, per_bin=True, cancel=None,
                 spatial_size=None, end=None, density_model=False):
        self.settings = resolve_filter(settings.regions)
        if self.settings.action == 'noise' and self.settings.seed is None:
            # Tail planning clones this constraint before the real mask is
            # frozen. All planning and fitting evaluations share one draw.
            self.settings = replace(self.settings, seed=int(np.random.SeedSequence().generate_state(1)[0]))
        self.warnings = []
        self.settings = replace(self.settings, _warning_sink=self.warnings.append)
        self.target, self.form = target, form
        self.reference, self.reference_form = reference, reference_form
        self.start, self.dx, self.dt = start, resolution, resolution_t
        self.reference_time = reference_time
        self.spatial_size, self.end = spatial_size, end
        if density_model:
            # The shared solver always fits a density. Transform values still
            # use the caller's units, including the app's per-bin rates.
            self.scale = resolution if target == 'initiation_rate' and per_bin else 1.
        else:
            self.scale = 1. if target != 'initiation_rate' or per_bin else 1. / resolution
        self.cancel = cancel or (lambda: None)
        self.mask = self.frozen = self.positions = None

    def selects_regions(self, spatial_size):
        """An empty selection must preserve the selected route's native fit."""
        if not self.settings.enabled:
            return False
        if self.settings.selection in {'profile', 'ranges'}:
            positions = self.start + np.arange(spatial_size) * self.dx
            intervals = np.column_stack((positions, positions + self.dx))
            if self.end is not None:
                intervals[:, 1] = np.minimum(intervals[:, 1], self.end)
            return bool(_selected(np.zeros(spatial_size), self.settings, positions, intervals, None).any())
        values = np.asarray(self.reference, dtype=float)
        if self.settings.selection == 'all':
            return not self.settings.complement
        if values.ndim < 2 or self.reference_form in {'scalar', 'x', 't'}:
            return bool(_selected(values.reshape(-1), self.settings, None, None, None).any())
        return any(_selected(values[:, column], self.settings, None, None, None).any()
                   for column in range(values.shape[1]))

    def _reference_on_grid(self, n, nt):
        values = np.asarray(self.reference, dtype=float)
        if self.settings.selection in {'all', 'profile', 'ranges'}:
            return np.zeros((n, nt))
        if self.form == 't' and self.reference_form == 't':
            source = values.reshape(-1)
            source_t = np.asarray(self.reference_time) if self.reference_time is not None and len(self.reference_time) == source.size else np.arange(source.size) * self.dt
            target_t = np.asarray(self.reference_time) if self.reference_time is not None and len(self.reference_time) == n else np.arange(n) * self.dt
            return np.interp(target_t, source_t, source)[:, None]
        if self.reference_form == 'scalar':
            return np.broadcast_to(values.reshape(1, 1), (n, nt))
        if self.reference_form == 'x':
            if values.size != n:
                raise ValueError('The selection input must cover the fitted genomic region.')
            return np.broadcast_to(values.reshape(n, 1), (n, nt))
        if self.reference_form == 't':
            values = values.reshape(1, -1)
        elif self.reference_form != 'xt' or values.shape[0] != n:
            raise ValueError('Choose a numeric input with coordinates matching the fitted target.')
        if values.shape[1] != nt:
            source_t = (np.asarray(self.reference_time) if self.reference_time is not None
                        and len(self.reference_time) == values.shape[1] else np.arange(values.shape[1]) * self.dt)
            target_t = (np.asarray(self.reference_time) if self.reference_time is not None and len(self.reference_time) == nt
                        else np.arange(nt) * self.dt)
            values = np.stack([np.interp(target_t, source_t, row) for row in values])
        return np.broadcast_to(values, (n, nt))

    def project(self, values, *, transposed=False):
        self.cancel()
        raw = np.asarray(values, dtype=float)
        canonical = raw.T if transposed else raw
        if self.form == 'x':
            canonical = canonical.reshape(-1, 1)
        elif self.form == 't':
            canonical = canonical.reshape(-1, 1)
        elif self.form == 'scalar':
            canonical = canonical.reshape(1, 1)
        elif canonical.ndim == 1:
            canonical = canonical[:, None]
        if self.frozen is None:
            reference = self._reference_on_grid(*canonical.shape)
            self.positions = (np.arange(canonical.shape[0]) * self.dt if self.form == 't' else
                              self.start + np.arange(canonical.shape[0]) * self.dx)
            intervals = None if self.form == 't' else np.column_stack((self.positions, self.positions + self.dx))
            if intervals is not None and self.end is not None:
                intervals[:, 1] = np.minimum(intervals[:, 1], self.end)
            masks = []

            def capture(mask, *_args, **_kwargs):
                masks.append(mask.copy())

            transformed = apply_data_filter(
                canonical * self.scale, replace(self.settings, _position_sink=capture),
                positions=self.positions, intervals=intervals, selection_values=reference, cancel=self.cancel,
            ) / self.scale
            self.mask = np.column_stack(masks) if masks else np.zeros(canonical.shape, dtype=bool)
            if self.target in {'initiation_rate', 'fork_speed'}:
                selected = transformed[self.mask]
                invalid = ~np.isfinite(selected) | (selected < 0 if self.target == 'initiation_rate' else selected <= 0)
                if invalid.any():
                    domain = 'finite, non-negative initiation rates' if self.target == 'initiation_rate' else 'finite, positive fork speeds'
                    raise ValueError(f'The Mapping region transform must produce {domain}. Choose a compatible transform.')
            self.frozen = transformed
        if canonical.shape != self.frozen.shape:
            raise ValueError('The fitted grid changed after the region transform was frozen.')
        result = np.where(self.mask, self.frozen, canonical)
        result = result.reshape((raw.T if transposed else raw).shape)
        return result.T if transposed else result

    def prepare_tiled(self, canonical):
        """Freeze an XT constraint column by column, retaining global regions.

        Each column still sees every spatial position, so interpolation,
        regional means, extrema and convolutions keep their original anchors.
        One random generator also preserves the ordinary noise sequence.
        """
        from dnascape._kinetic_tiled import disk_array
        from .modulation import aligned_modulation

        if self.frozen is not None:
            return
        n, nt = canonical.shape
        self.positions = self.start + np.arange(n) * self.dx
        intervals = np.column_stack((self.positions, self.positions + self.dx))
        if self.end is not None:
            intervals[:, 1] = np.minimum(intervals[:, 1], self.end)
        mask = disk_array(canonical.shape, bool, order="F")
        frozen = disk_array(canonical.shape, order="F")
        rng = np.random.default_rng(None if self.settings.seed is None else
                                   np.random.SeedSequence([self.settings.seed, 0]))
        modulation = (aligned_modulation(canonical, self.settings.modulation,
                      coordinates=self.positions, warning=self.settings._warning_sink)
                      if self.settings.action == "modulate" else None)
        for column in range(nt):
            self.cancel()
            selected = []

            def capture(selection, *_args, **_kwargs):
                selected.append(selection)

            transformed = apply_data_filter(
                canonical[:, column] * self.scale,
                replace(self.settings, _position_sink=capture),
                positions=self.positions, intervals=intervals,
                selection_values=self._reference_column(n, nt, column),
                cancel=self.cancel, _rng=rng,
                _modulation=None if modulation is None else modulation[:, column:column + 1],
            ) / self.scale
            mask[:, column] = selected[0] if selected else False
            values = transformed[mask[:, column]]
            invalid = ~np.isfinite(values) | (values < 0 if self.target == 'initiation_rate' else values <= 0)
            if invalid.any():
                domain = 'finite, non-negative initiation rates' if self.target == 'initiation_rate' else 'finite, positive fork speeds'
                raise ValueError(f'The Mapping region transform must produce {domain}. Choose a compatible transform.')
            frozen[:, column] = transformed
        self.mask, self.frozen = mask, frozen

    def _reference_column(self, n, nt, column):
        """The same reference interpolation as _reference_on_grid, streamed."""
        if self.settings.selection in {'all', 'profile', 'ranges'}:
            return np.zeros(n)
        values = np.asarray(self.reference, dtype=float)
        if self.reference_form == 'scalar':
            return np.broadcast_to(values.reshape(()), (n,))
        if self.reference_form == 'x':
            if values.size != n:
                raise ValueError('The selection input must cover the fitted genomic region.')
            return values.reshape(n)
        if self.reference_form == 't':
            values = values.reshape(1, -1)
        elif self.reference_form != 'xt' or values.shape[0] != n:
            raise ValueError('Choose a numeric input with coordinates matching the fitted target.')
        if values.shape[1] == nt:
            return np.broadcast_to(values[:, column], (n,))
        source_t = (np.asarray(self.reference_time) if self.reference_time is not None
                    and len(self.reference_time) == values.shape[1] else np.arange(values.shape[1]) * self.dt)
        target_t = (self.reference_time[column] if self.reference_time is not None
                    and len(self.reference_time) == nt else column * self.dt)
        lo = int(np.clip(np.searchsorted(source_t, target_t, side='right') - 1, 0, len(source_t) - 1))
        hi = min(lo + 1, len(source_t) - 1)
        weight = np.clip((target_t - source_t[lo]) / (source_t[hi] - source_t[lo]), 0., 1.) if hi != lo else 0.
        return np.broadcast_to((1. - weight) * values[:, lo] + weight * values[:, hi], (n,))

    def export_positions(self):
        if self.mask is None:
            return np.empty(0, dtype=np.int64)
        selected = self._spatial_mask()
        positions = self.start + np.arange(selected.size) * self.dx
        return np.rint(positions[selected] * 1000.).astype(np.int64)

    def preview_selection(self):
        """Retain the frozen mask for plot guides independently of export."""
        from .simulation_filters import SimulationFilterSelection

        if self.mask is None or not self.mask.any():
            return None
        mask = self.mask.T if self.form == 't' else self.mask
        end = self.end if self.end is not None else self.start + (self.spatial_size or mask.shape[0]) * self.dx
        return SimulationFilterSelection(
            np.packbits(mask).tobytes(), mask.shape, self.form in {'x', 'xt'}, self.form in {'t', 'xt'},
            self.start, end, self.dx, self.dt,
            np.packbits(mask.any(axis=1)).tobytes(), np.packbits(mask.any(axis=0)).tobytes(),
        )

    def _spatial_mask(self):
        if self.form in {'t', 'scalar'}:
            return np.full(self.spatial_size or 1, bool(self.mask.any()))
        return np.any(self.mask, axis=1)

    def export_intervals(self):
        selected = self._spatial_mask()
        edges = np.flatnonzero(np.diff(np.r_[False, selected, False]))
        intervals = self.start + edges.reshape(-1, 2) * self.dx
        if self.end is not None:
            intervals[:, 1] = np.minimum(intervals[:, 1], self.end)
        return np.rint(intervals * 1000.).astype(np.int64)


def mapping_filter_recipe(settings, members):
    """Keep auxiliary arrays binary in runnable downloads, without caching them."""
    if settings is None or not settings.regions.enabled:
        return None
    regions = resolve_filter(settings.regions)
    ignored = {'profile', 'upload', 'modulation', '_position_sink', '_warning_sink'}
    values = {item.name: getattr(regions, item.name) for item in fields(regions) if item.name not in ignored}
    arrays = {}
    if regions.profile is not None:
        profile = regions.profile
        arrays['selection'] = np.asarray(profile.values)
        values['profile'] = {'layout': profile.layout, 'units': profile.units}
        if profile.chromosomes is not None:
            arrays['chromosomes'] = np.asarray(profile.chromosomes, dtype=str)
    if regions.action == 'modulate':
        modulation = regions.modulation
        ignored_mod = {'upload', 'values', 'profiles'}
        values['modulation'] = {item.name: getattr(modulation, item.name) for item in fields(modulation) if item.name not in ignored_mod}
        profiles = []
        for index, (chromosome, profile) in enumerate(modulation.profiles.items()):
            key = f'modulation_{index}'
            arrays[key] = np.asarray(profile.values)
            if profile.coordinates is not None:
                arrays[f'{key}_coordinates'] = np.asarray(profile.coordinates)
            profiles.append({'chromosome': chromosome, 'array': key})
        values['modulation']['profiles'] = profiles
    result = {'regions': values, 'relative_to': settings.relative_to, 'export': settings.export}
    if arrays:
        result['arrays'] = 'inputs/mapping_filter.npz'
        members.append((result['arrays'], arrays))
    return result


def mapping_filter_from_recipe(recipe, base_path):
    from pathlib import Path
    from .filtering import FilterProfile
    from .modulation import ModulationProfile, ProfileModulationSettings

    if not recipe:
        return MappingFilterSettings()
    arrays = {}
    if recipe.get('arrays'):
        with np.load(Path(base_path) / recipe['arrays'], allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
    values = dict(recipe.get('regions', {}))
    profile = values.pop('profile', None)
    if profile is not None:
        values['profile'] = FilterProfile(arrays['selection'], chromosomes=arrays.get('chromosomes'), **profile)
    modulation = values.pop('modulation', None)
    if modulation is not None:
        modulation = dict(modulation)
        profiles = modulation.pop('profiles', [])
        values['modulation'] = ProfileModulationSettings(
            **modulation, profiles={item['chromosome']: ModulationProfile(
                arrays[item['array']], arrays.get(item['array'] + '_coordinates'),
            ) for item in profiles},
        )
    return MappingFilterSettings(DataFilterSettings(**values), recipe.get('relative_to'), bool(recipe.get('export', False)))
