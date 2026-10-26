"""Crop and rebin reusable spatial arrays on their original genomic grid."""

from __future__ import annotations

import numpy as np

from dnascape.workflows.filtering import _upload_table, profile_coordinates


def spatial_profile_rows(inspection, chromosome=None):
    """Select a chromosome from row-based tables without flattening separate regions."""
    column = next((name for name in inspection.columns if str(name).lower() in {"chr", "chrom", "chromosome"}), None)
    if column is None or inspection.source_format not in {"tsv", "csv", "txt"}:
        return None, ()
    labels = _upload_table(inspection, [column])[column].astype(str).to_numpy()
    names = tuple(dict.fromkeys(labels.tolist()))
    if chromosome is None and len(names) > 1:
        return None, names
    selected = chromosome or names[0]
    if selected not in names:
        raise ValueError("Choose a chromosome present in the source data.")
    return labels == selected, names


def spatial_profile_grid(inspection, length, *, source_bin_size=1000, source_start=0, chromosome=None):
    """Return positions and half-open intervals in kb, retaining supplied coordinates."""
    positions, intervals = profile_coordinates(inspection, length)
    rows, chromosomes = spatial_profile_rows(inspection, chromosome)
    if rows is None and len(chromosomes) > 1:
        raise ValueError("Choose one chromosome before configuring its region and binning.")
    supplied = positions is not None
    if positions is None:
        positions = (source_start + np.arange(length, dtype=float) * source_bin_size) / 1000
    positions = np.asarray(positions, dtype=float)
    if rows is not None:
        positions = positions[rows] if supplied else (source_start + np.arange(np.count_nonzero(rows), dtype=float) * source_bin_size) / 1000
        if intervals is not None:
            intervals = np.asarray(intervals)[rows]
        length = positions.size
    if not positions.size or not np.isfinite(positions).all() or np.any(np.diff(positions) <= 0):
        raise ValueError("Spatial coordinates must be finite and increasing.")
    if intervals is None:
        step = float(np.median(np.diff(positions))) if length > 1 else source_bin_size / 1000
        intervals = np.column_stack((positions, np.r_[positions[1:], positions[-1] + step]))
    intervals = np.asarray(intervals, dtype=float)
    if not np.isfinite(intervals).all() or np.any(intervals[:, 1] <= intervals[:, 0]) or np.any(intervals[1:, 0] < intervals[:-1, 1] - 1e-10):
        raise ValueError("Spatial intervals must be ordered and non-overlapping.")
    return positions, intervals, supplied


def rebin_spatial_profile(values, positions, intervals, *, start=None, end=None,
                         bin_size=1000, aggregation="mean", space_time=False,
                         max_values=10_000_000, cancel=None):
    """Aggregate interval overlaps; missing values do not contribute signal or weight."""
    source = np.asarray(values, dtype=float)
    matrix = source if space_time else source.reshape(-1, 1)
    if matrix.ndim != 2 or matrix.shape[0] != len(positions):
        raise ValueError("The spatial grid must match the source array length.")
    lower = intervals[0, 0] if start is None else start / 1000
    upper = intervals[-1, 1] if end is None else end / 1000
    step = bin_size / 1000
    if not np.isfinite([lower, upper, step]).all() or step <= 0 or upper <= lower:
        raise ValueError("Choose a positive resolution and an End greater than Start.")
    if lower < intervals[0, 0] - 1e-9 or upper > intervals[-1, 1] + 1e-9:
        raise ValueError("The selected region extends beyond the source data.")
    if aggregation not in {"mean", "sum", "min", "max"}:
        raise ValueError("Choose Mean, Sum, Minimum or Maximum for array binning.")
    count = int(np.ceil((upper - lower) / step - 1e-10))
    if count * matrix.shape[1] > max_values:
        raise ValueError("This resolution exceeds the conversion memory limit. Choose a larger bin size.")
    starts = lower + np.arange(count) * step
    ends = np.minimum(starts + step, upper)
    # Preserve samples, missing values and native positions exactly on an unchanged grid.
    first = int(np.searchsorted(intervals[:, 0], lower - 1e-10))
    last = first + count
    if last <= len(positions) and np.allclose(intervals[first:last, 0], starts, rtol=0, atol=1e-9) and np.allclose(intervals[first:last, 1], ends, rtol=0, atol=1e-9):
        output = matrix[first:last].copy()
        x = positions[first:last].copy()
    else:
        output = np.full((count, matrix.shape[1]), np.nan)
        widths = intervals[:, 1] - intervals[:, 0]
        for index, (left, right) in enumerate(zip(starts, ends)):
            if cancel is not None and index % 1024 == 0:
                cancel()
            a = np.searchsorted(intervals[:, 1], left, side="right")
            b = np.searchsorted(intervals[:, 0], right, side="left")
            overlap = np.maximum(0, np.minimum(intervals[a:b, 1], right) - np.maximum(intervals[a:b, 0], left))
            part = matrix[a:b]
            valid = np.isfinite(part) & (overlap[:, None] > 0)
            if aggregation in {"mean", "sum"}:
                weights = overlap[:, None] if aggregation == "mean" else (overlap / widths[a:b])[:, None]
                total = np.sum(np.where(valid, part, 0) * weights, axis=0)
                weight = np.sum(valid * weights, axis=0)
                output[index] = np.divide(total, weight, out=np.full(matrix.shape[1], np.nan), where=weight > 0) if aggregation == "mean" else np.where(weight > 0, total, np.nan)
            elif part.size:
                operation = np.min if aggregation == "min" else np.max
                fill = np.inf if aggregation == "min" else -np.inf
                output[index] = np.where(valid.any(axis=0), operation(np.where(valid, part, fill), axis=0), np.nan)
        x = starts
    result = output if space_time else output.reshape((-1, 1) if source.ndim == 2 and source.shape[1] == 1 else (1, -1) if source.ndim == 2 else (-1,))
    return result, x, np.column_stack((starts, ends))
