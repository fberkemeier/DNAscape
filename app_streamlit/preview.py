"""Bounded preview and memory-summary helpers for app datasets."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ArraySummary:
    """Compact numerical summary that never retains an array copy."""

    shape: tuple[int, ...]
    dtype: str
    nbytes: int
    size: int
    finite_count: int
    missing_count: int
    minimum: float | None
    maximum: float | None
    mean: float | None


def memory_label(size_bytes: int | float) -> str:
    """Format a byte estimate using compact binary units."""

    size = max(0.0, float(size_bytes))
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024.0 or unit == "TiB":
            return f"{int(size)} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TiB"


def array_summary(values: Any, *, chunk_size: int = 1_000_000) -> ArraySummary:
    """Compute exact finite-value statistics with bounded temporary memory."""

    array = np.asarray(values)
    flat = array.reshape(-1)
    step = max(1, int(chunk_size))
    finite_count = 0
    total = 0.0
    minimum = None
    maximum = None
    for start in range(0, flat.size, step):
        chunk = np.asarray(flat[start : start + step], dtype=float)
        mask = np.isfinite(chunk)
        if not np.any(mask):
            continue
        finite = chunk[mask]
        finite_count += int(finite.size)
        total += float(np.sum(finite, dtype=np.float64))
        chunk_min = float(np.min(finite))
        chunk_max = float(np.max(finite))
        minimum = chunk_min if minimum is None else min(minimum, chunk_min)
        maximum = chunk_max if maximum is None else max(maximum, chunk_max)
    return ArraySummary(
        shape=tuple(int(axis) for axis in array.shape),
        dtype=str(array.dtype),
        nbytes=int(array.nbytes),
        size=int(array.size),
        finite_count=finite_count,
        missing_count=int(array.size - finite_count),
        minimum=minimum,
        maximum=maximum,
        mean=(total / finite_count) if finite_count else None,
    )


def preview_indices(size: int, max_size: int) -> np.ndarray:
    """Return stable uniform indices capped at ``max_size``."""

    size = max(0, int(size))
    max_size = max(1, int(max_size))
    if size <= max_size:
        return np.arange(size, dtype=int)
    return np.unique(np.linspace(0, size - 1, max_size, dtype=int))


def preview_series(
    x_values: Any,
    y_values: Any,
    *,
    max_points: int = 5_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Return aligned, bounded one-dimensional preview arrays."""

    x_array = np.asarray(x_values).reshape(-1)
    y_array = np.asarray(y_values).reshape(-1)
    usable = min(x_array.size, y_array.size)
    keep = preview_indices(usable, max_points)
    return x_array[:usable][keep], y_array[:usable][keep]


def preview_matrix(
    values: Any,
    *,
    max_rows: int = 900,
    max_columns: int = 420,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a bounded matrix plus source row/column indices."""

    matrix = np.asarray(values)
    if matrix.ndim != 2:
        raise ValueError("A matrix preview requires a two-dimensional array.")
    row_indices = preview_indices(matrix.shape[0], max_rows)
    column_indices = preview_indices(matrix.shape[1], max_columns)
    return matrix[np.ix_(row_indices, column_indices)], row_indices, column_indices


def representative_preview(values: Any, *, rows: int = 8, columns: int = 8) -> np.ndarray:
    """Return a small head/tail sample suitable for a compact table."""

    array = np.asarray(values)
    matrix = array.reshape(-1, 1) if array.ndim == 1 else np.atleast_2d(array)
    row_keep = preview_indices(matrix.shape[0], rows)
    column_keep = preview_indices(matrix.shape[1], columns)
    return matrix[np.ix_(row_keep, column_keep)]


def estimated_array_bytes(shape: tuple[int, ...], dtype: Any = np.float64) -> int:
    """Return the exact dense-array byte estimate for a shape and dtype."""

    dimensions = tuple(int(axis) for axis in shape)
    if any(axis < 0 for axis in dimensions):
        raise ValueError("Array dimensions must be non-negative.")
    cells = prod(dimensions)
    return cells * int(np.dtype(dtype).itemsize)
