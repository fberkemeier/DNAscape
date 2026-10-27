"""Resample spatial uploads without changing their physical interval.

Array uploads default to one sample per kb. A requested resolution describes
the destination bins, never a reinterpretation of the source coordinates.
"""

from __future__ import annotations

import math

import numpy as np


def spatial_bin_count(start: float, end: float, resolution: float) -> int:
    if not all(math.isfinite(v) for v in (start, end, resolution)):
        raise ValueError("The spatial grid must be finite.")
    if start < 0 or end <= start or resolution <= 0:
        raise ValueError("Choose a non-negative Start, a larger End and a positive resolution.")
    ratio = (end - start) / resolution
    if not math.isfinite(ratio):
        raise ValueError("The spatial grid is too large.")
    nearest = round(ratio)
    if math.isclose(ratio, nearest, rel_tol=1e-12, abs_tol=1e-12):
        ratio = float(nearest)
    return max(1, int(math.ceil(ratio)))


def resample_spatial(
    values: np.ndarray,
    *,
    source_resolution: float,
    source_start: float = 0.0,
    start: float,
    end: float,
    resolution: float,
    max_points: int = 1_000_000,
    max_values: int = 16_000_000,
) -> np.ndarray:
    """Mean into wider bins; linearly interpolate onto finer spatial grids.

    The leading dimension is space. Values (including initiation rates per bin)
    are averaged rather than summed. A final partial bin ends at ``end``.
    Missing source values remain missing for the caller's missing-value policy.
    """

    count = spatial_bin_count(start, end, resolution)
    if np.iscomplexobj(values):
        raise ValueError("Spatial data must contain real numeric values.")
    array = np.asarray(values)
    if array.ndim not in (1, 2) or not array.shape[0]:
        raise ValueError("Spatial data must be a non-empty vector or matrix.")
    columns = 1 if array.ndim == 1 else array.shape[1]
    if array.size > max_values or count > max_points or count * columns > max_values:
        raise ValueError(
            "The requested spatial grid exceeds the local limit. "
            "Increase the resolution or narrow the region."
        )
    if not math.isfinite(source_resolution) or source_resolution <= 0:
        raise ValueError("Source resolution must be positive and finite.")
    source_end = source_start + array.shape[0] * source_resolution
    tolerance = 1e-10 * max(1., abs(source_start), abs(source_end))
    if start < source_start - tolerance or end > source_end + tolerance:
        raise ValueError(
            "The selected region must lie within the source interval "
            f"({source_start:g}–{source_end:g} kb)."
        )
    first = max(0, int(math.floor((start - source_start) / source_resolution + 1e-12)))
    last = min(array.shape[0], int(math.ceil((end - source_start) / source_resolution - 1e-12)))
    if math.isclose(resolution, source_resolution, rel_tol=1e-12) and math.isclose(
        (start - source_start) / source_resolution, first, abs_tol=1e-10,
    ):
        result = np.asarray(array[first:last], dtype=float).view()
        result.setflags(write=False)
        return result
    output = np.empty((count, columns), dtype=float)
    matrix = array.reshape(array.shape[0], columns)
    factor = resolution / source_resolution
    if (
        factor >= 1 and math.isclose(factor, round(factor), rel_tol=1e-12)
        and math.isclose((start - source_start) / source_resolution, first, abs_tol=1e-10)
        and math.isclose((end - source_start) / source_resolution, last, abs_tol=1e-10)
    ):
        # Common aligned binning avoids prefix-sum cancellation and is also
        # faster for chromosome-sized profiles.
        width = round(factor)
        complete = (last - first) // width
        stop = first + complete * width
        output[:complete] = matrix[first:stop].reshape(complete, width, columns).mean(axis=1)
        if complete < count:
            output[-1] = matrix[stop:last].mean(axis=0)
        result = output.reshape((count,) + array.shape[1:])
        result.setflags(write=False)
        return result
    positions = start + np.arange(count, dtype=float) * resolution
    if resolution < source_resolution:
        # One neighbour outside the selected interval supplies the endpoint
        # slope. Inside the last source bin, extend its final sample unchanged.
        stop = min(array.shape[0], last + 1)
        native_positions = source_start + np.arange(first, stop) * source_resolution
        for column in range(columns):
            output[:, column] = np.interp(positions, native_positions, matrix[first:stop, column])
    else:
        edges = np.minimum(start + np.arange(count + 1) * resolution, end)
        coordinates = np.clip(
            (edges - (source_start + first * source_resolution)) / source_resolution,
            0., last - first,
        )
        indices = np.minimum(np.floor(coordinates).astype(int), last - first - 1)
        fractions = coordinates - indices
        for column in range(columns):
            native = np.asarray(matrix[first:last, column], dtype=float)
            finite = np.isfinite(native)
            clean = np.where(finite, native, 0.)
            integral = np.concatenate(([0.], np.cumsum(clean)))
            at_edges = integral[indices] + fractions * clean[indices]
            output[:, column] = np.diff(at_edges) / np.diff(coordinates)
            if not np.all(finite):
                missing = (~finite).astype(float)
                missing_integral = np.concatenate(([0.], np.cumsum(missing)))
                missing_edges = missing_integral[indices] + fractions * missing[indices]
                output[np.diff(missing_edges) > 1e-12, column] = np.nan
    result = output.reshape((count,) + array.shape[1:])
    result.setflags(write=False)
    return result
