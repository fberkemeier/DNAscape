"""Independent edits to sampled simulation inputs and compact region guides."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import MappingProxyType

import numpy as np

from .filtering import apply_data_filter, resolve_filter
from .uploads import _resolve_missing_values

MAX_FILTER_CELLS = 16_000_000
FILTER_LABELS = {'initiation_rate': 'Initiation-rate mask', 'fork_speed': 'Fork-speed mask'}


def _runs(mask):
    return np.flatnonzero(np.diff(np.r_[False, mask, False])).reshape(-1, 2)


def _display_runs(mask, maximum):
    stride = max(1, int(np.ceil(len(mask) / max(1, 2 * maximum))))
    reduced = np.logical_or.reduceat(mask, np.arange(0, len(mask), stride))
    return np.minimum(_runs(reduced) * stride, len(mask))


@dataclass(frozen=True)
class SimulationFilterSelection:
    packed: bytes = field(repr=False)
    shape: tuple[int, int]
    spatial: bool
    temporal: bool
    start: float
    end: float
    resolution_x: float
    resolution_t: float
    spatial_union: bytes = field(default=b'', repr=False)
    temporal_union: bytes = field(default=b'', repr=False)
    extend_time: bool = False

    def mask(self):
        return np.unpackbits(np.frombuffer(self.packed, dtype=np.uint8), count=int(np.prod(self.shape))).reshape(self.shape).astype(bool)

    def positions(self):
        selected = self._union(0)
        if not self.spatial:
            count = int(np.ceil((self.end - self.start) / self.resolution_x - 1e-12))
            selected = np.full(count, bool(selected.any()))
        return np.rint((self.start + np.flatnonzero(selected) * self.resolution_x) * 1000).astype(np.int64)

    def _union(self, axis):
        packed = self.spatial_union if axis == 0 else self.temporal_union
        return (np.unpackbits(np.frombuffer(packed, dtype=np.uint8), count=self.shape[axis]).astype(bool)
                if packed else self.mask().any(axis=1 - axis))

    def intervals(self):
        positions = self.positions()
        if not positions.size:
            return np.empty((0, 2), dtype=np.int64)
        cuts = np.r_[0, np.flatnonzero(np.diff(positions) > self.resolution_x * 1000 + 1e-6) + 1, positions.size]
        return np.column_stack((positions[cuts[:-1]], np.minimum(
            np.rint(positions[cuts[1:] - 1] + self.resolution_x * 1000), round(self.end * 1000),
        ))).astype(np.int64)

    def areas(self, kind='space', *, x_limits=None, y_limits=None, time_end=0., max_areas=750):
        if kind == 'time':
            selected = self._union(1)
            first, stop = 0, len(selected)
            if x_limits is not None and self.temporal:
                first = max(0, int(np.floor(min(x_limits) / self.resolution_t)))
                if self.extend_time:
                    first = min(first, len(selected) - 1)
                stop = min(stop, int(np.ceil(max(x_limits) / self.resolution_t)) + 1)
            if first >= stop:
                return []
            ranges = ((first + _display_runs(selected[first:stop], max_areas)) * self.resolution_t if self.temporal
                      else np.array([[0., time_end]]))
            if self.extend_time and self.temporal and selected[-1] and stop == len(selected) and len(ranges):
                ranges[-1, 1] = max(ranges[-1, 1], time_end)
            return [[{'xAxis': float(a)}, {'xAxis': float(b)}] for a, b in ranges]
        if kind != 'xt' or not self.temporal:
            selected = self._union(0)
            first, stop = 0, len(selected)
            if self.spatial and x_limits is not None:
                first = max(0, int(np.floor((min(x_limits) - self.start) / self.resolution_x)))
                stop = min(stop, int(np.ceil((max(x_limits) - self.start) / self.resolution_x)) + 1)
            if first >= stop:
                return []
            ranges = (self.start + (first + _display_runs(selected[first:stop], max_areas)) * self.resolution_x if self.spatial
                      else np.array([[self.start, self.end]]))
            return [[{'xAxis': float(a)}, {'xAxis': float(min(b, self.end))}] for a, b in ranges]
        mask = self.mask()
        first, stop = 0, mask.shape[0]
        if self.spatial and x_limits is not None:
            first = max(0, int(np.floor((min(x_limits) - self.start) / self.resolution_x)))
            stop = min(stop, int(np.ceil((max(x_limits) - self.start) / self.resolution_x)) + 1)
        t_first, t_stop = 0, mask.shape[1]
        if y_limits is not None:
            t_first = max(0, int(np.floor(min(y_limits) / self.resolution_t)))
            if self.extend_time:
                t_first = min(t_first, mask.shape[1] - 1)
            t_stop = min(t_stop, int(np.ceil(max(y_limits) / self.resolution_t)) + 1)
        if first >= stop or t_first >= t_stop:
            return []
        mask = mask[first:stop, t_first:t_stop]
        nx = min(mask.shape[0], max(1, int(np.sqrt(max_areas * 2 * mask.shape[0] / mask.shape[1]))))
        nt = max(1, 2 * max_areas // nx)
        x_stride, t_stride = int(np.ceil(mask.shape[0] / nx)), max(1, int(np.ceil(mask.shape[1] / nt)))
        xs, ts = np.arange(0, mask.shape[0], x_stride), np.arange(0, mask.shape[1], t_stride)
        reduced = np.logical_or.reduceat(np.logical_or.reduceat(mask, xs, axis=0), ts, axis=1)
        areas = []
        # Adjacent identical time slices share one rectangle per spatial run.
        start = 0
        for column in range(1, len(ts) + 1):
            if column < len(ts) and np.array_equal(reduced[:, column], reduced[:, start]):
                continue
            for left, right in _runs(reduced[:, start]):
                lo = self.start + (first + xs[left]) * self.resolution_x if self.spatial else self.start
                hi = min(self.end, self.start + (first + min(mask.shape[0], right * x_stride)) * self.resolution_x) if self.spatial else self.end
                bottom = (t_first + ts[start]) * self.resolution_t
                top = min(time_end, (t_first + min(mask.shape[1], column * t_stride)) * self.resolution_t)
                if self.extend_time and column == len(ts) and t_stop == self.shape[1]:
                    top = time_end
                if top > bottom:
                    areas.append([{'xAxis': float(lo), 'yAxis': float(bottom)}, {'xAxis': float(hi), 'yAxis': float(top)}])
            start = column
        return areas


@dataclass(frozen=True)
class SimulationFilterState:
    selections: dict = field(default_factory=dict)
    exports: frozenset[str] = frozenset()
    warnings: tuple[str, ...] = ()


def apply_simulation_filters(values, settings, *, start, end, resolution_x, resolution_t,
                             n_bins, exports=frozenset(), missing_policy='error', cancel=None, seed=None):
    """Select on each original field and edit it once, after spatial sampling."""
    cancel = cancel or (lambda: None)
    result, selections, warnings = dict(values), {}, []
    for key, raw_settings in settings.items():
        if key not in FILTER_LABELS:
            raise ValueError('Choose initiation rate or fork speed for the region transform.')
        cancel()
        config = resolve_filter(raw_settings)
        if not config.enabled:
            raw = np.asarray(values[key], dtype=float)
            if np.isnan(raw).any():
                result[key] = _resolve_missing_values(raw, policy=missing_policy, expected_form='t' if raw.ndim == 2 and raw.shape[0] == 1 else 'x')
            continue
        if config.action == 'noise' and config.seed is None and seed is not None:
            config = replace(config, seed=int(np.random.SeedSequence([int(seed), int(key == 'fork_speed')]).generate_state(1)[0]))
        original = np.asarray(values[key], dtype=float)
        temporal = original.ndim == 2 and original.shape[1] > 1
        source_temporal = temporal and original.shape[0] == 1
        if original.size == 1:
            source = np.full(n_bins, float(original.reshape(-1)[0]))
        elif source_temporal and config.selection == 'profile':
            if n_bins * original.shape[1] > MAX_FILTER_CELLS:
                raise ValueError('The transformed space–time input is too large. Increase resolution or shorten the region.')
            source = np.broadcast_to(original, (n_bins, original.shape[1]))
        else:
            source = original.reshape(-1) if original.ndim == 2 and original.shape[1] == 1 else original
        if source.size > MAX_FILTER_CELLS:
            raise ValueError('The simulation mask exceeds the input size limit.')
        spatial = not (source.ndim == 2 and source.shape[0] == 1 and source.shape[1] > 1)
        positions = start + np.arange(n_bins) * resolution_x if spatial else np.arange(source.size) * resolution_t
        intervals = np.column_stack((positions, np.minimum(positions + resolution_x, end))) if spatial else None
        captured = []
        transformed = apply_data_filter(source, replace(config,
            _position_sink=lambda mask, *_a, **_k: captured.append(mask.copy()), _warning_sink=warnings.append),
            positions=positions, intervals=intervals, cancel=cancel)
        if not captured:
            continue
        mask = np.column_stack(captured) if spatial else np.asarray(captured[0])[None, :]
        if not mask.any():
            if np.isnan(original).any():
                result[key] = _resolve_missing_values(original, policy=missing_policy, expected_form='t' if source_temporal else 'x')
            continue
        transformed = _resolve_missing_values(transformed, policy=missing_policy, expected_form='t' if not spatial else 'x')
        # Keep constant inputs compact when the operation leaves them uniform.
        if original.size == 1 and np.all(transformed == transformed.flat[0]):
            transformed = float(transformed.flat[0])
        result[key] = transformed
        selections[key] = SimulationFilterSelection(np.packbits(mask).tobytes(), mask.shape, spatial,
            temporal, start, end, resolution_x, resolution_t,
            np.packbits(mask.any(axis=1)).tobytes(), np.packbits(mask.any(axis=0)).tobytes(),
            key == 'fork_speed' and temporal)
    return result, SimulationFilterState(MappingProxyType(selections), frozenset(exports), tuple(dict.fromkeys(warnings)))


def build_simulation_filter_export(state, key, file_format, *, filename=None, layout='positions', compression='none'):
    """Write the same headerless integer positions/intervals used by Data Manager."""
    import io
    from .exports import RAW_FORMATS, compressed_export_artifact, normalize_export_format, safe_export_filename

    if state is None or key not in state.selections or key not in state.exports:
        raise ValueError('This input has no exported mask.')
    if layout not in {'positions', 'intervals'}:
        raise ValueError('Choose a position list or genomic intervals.')
    file_format = normalize_export_format(file_format, allowed=RAW_FORMATS)
    selection = state.selections[key]
    values = selection.positions() if layout == 'positions' else selection.intervals()
    if file_format == 'npz':
        stream = io.BytesIO()
        np.savez_compressed(stream, **{layout: values})
        payload = stream.getvalue()
    else:
        stream = io.StringIO()
        np.savetxt(stream, values, fmt='%d', delimiter=',' if file_format == 'csv' else '\t')
        payload = stream.getvalue().encode()
    name = safe_export_filename(filename or f'rsim_{key}_filter', file_format, default_stem='filter_positions')
    return compressed_export_artifact(data=payload, filename=name, compression=compression,
                                      media_type='application/octet-stream' if file_format == 'npz' else 'text/plain; charset=utf-8')
