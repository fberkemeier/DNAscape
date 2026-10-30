"""Display time-only profiles over a spatial region without changing their data."""

from __future__ import annotations

import gzip
import io

import numpy as np


def spatial_region(region):
    start, end = map(float, region)
    if not np.isfinite([start, end]).all() or start < 0 or end <= start:
        raise ValueError('End must be greater than a finite, non-negative start.')
    return start, end


def _archive_axis(source, names):
    if source is None or source.source_format != 'npz':
        return None
    payload = gzip.decompress(source.content) if source.compressed else source.content
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        for name in names:
            if name in archive:
                return np.asarray(archive[name], dtype=float).reshape(-1)
    return None


def uploaded_spatial_region(source):
    axis = _archive_axis(source, ('position_kb', 'x_array'))
    if axis is None or axis.size < 2:
        return None
    return spatial_region((axis[0], axis[-1]))


def uploaded_time_axis(source, size):
    axis = _archive_axis(source, ('time_min', 'time_grid', 'time_array', 't_array'))
    if axis is None and source is not None:
        names = {str(name).lower().replace(' ', '_') for name in source.columns}
        if names & {'time', 'time_min', 't'}:
            from .filtering import profile_coordinates
            axis, _ = profile_coordinates(source, size)
    if axis is not None:
        if axis.size != size or not np.isfinite(axis).all() or np.any(np.diff(axis) <= 0):
            raise ValueError('Time coordinates must match the profile and increase strictly.')
    return axis


def temporal_heatmap(values, positions, times):
    """Return a bounded broadcast view in (time, position) plotting order."""
    values = np.asarray(values).reshape(-1)
    positions, times = np.asarray(positions, dtype=float), np.asarray(times, dtype=float)
    if not positions.size or times.size != values.size:
        raise ValueError('A temporal heatmap needs a spatial region and matching time coordinates.')
    # No spatial information is lost by reducing an everywhere-identical
    # field. Retain the first positive coordinate for logarithmic X axes.
    if positions.size > 720:
        indices = np.unique(np.r_[0, 1, np.linspace(0, positions.size - 1, 718, dtype=int)])
        positions = positions[indices]
    return np.broadcast_to(values[:, None], (values.size, positions.size)), positions, times


def profile_heatmap_data(profile):
    """Use the same axes in Analysis previews and exported Python workflows."""
    values = np.asarray(profile['values'])
    if profile['form'] != 't':
        return values.T, np.arange(values.shape[0], dtype=float), np.arange(values.shape[1], dtype=float)
    region = profile.get('spatial_region')
    if region is None:
        region = uploaded_spatial_region(profile.get('source'))
    if region is None:
        raise ValueError('Define a spatial region for the time-dependent profile.')
    start, end = spatial_region(region)
    times = uploaded_time_axis(profile.get('source'), values.size)
    if times is None:
        step = float(profile.get('time_step', 1.))
        if not np.isfinite(step) or step <= 0:
            raise ValueError('Time step must be finite and greater than zero.')
        times = np.arange(values.size, dtype=float) * step
    return temporal_heatmap(values, np.array([start, end]), times)
