"""Small cross-platform BigWig reader used by the local application.

The app uses ``pybigtools`` on Python 3.10–3.13 and ``figwig`` on Python 3.14
and later. Readers load on demand; ``pyBigWig`` and the discontinued
``gwseq_io_pp`` remain compatibility fallbacks for existing environments.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager, closing, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


# Bound the per-base temporary array, even when the requested bin is very wide.
_FIGWIG_CHUNK_BASES = 1_000_000
_FIGWIG_CHUNK_BINS = 5_000


class BigWigDependencyError(ImportError):
    """Raised when no supported local BigWig reader is installed."""


@dataclass(frozen=True, slots=True)
class BigWigMetadata:
    """Bounded metadata exposed without rasterizing chromosome signals."""

    chromosomes: tuple[tuple[str, int], ...]
    backend: str
    file_info: tuple[tuple[str, str], ...] = ()


def _load_backend() -> tuple[str, Any]:
    """Select the installed reader for this interpreter without eager imports."""

    preferred = "figwig" if sys.version_info >= (3, 14) else "pybigtools"
    try:
        return preferred, importlib.import_module(preferred)
    except ImportError:
        try:
            return "pyBigWig", importlib.import_module("pyBigWig")
        except ImportError:
            try:
                return "gwseq_io_pp", importlib.import_module("gwseq_io_pp")
            except ImportError as exc:
                raise BigWigDependencyError(
                    "No compatible BigWig reader is installed. Run "
                    "`python -m pip install -r app/requirements.txt` from the repository "
                    "root. This installs pybigtools on Python 3.10–3.13 or figwig on "
                    "Python 3.14 or newer. Alternatively, use a bedGraph track."
                ) from exc


def _open_bigwig(path: Path) -> tuple[str, AbstractContextManager[Any]]:
    backend_name, backend = _load_backend()
    try:
        handle = (backend.BigWigReader(str(path)) if backend_name == "figwig"
                  else backend.open(str(path)))
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError("The uploaded BigWig file could not be opened.") from exc
    if handle is None:
        raise ValueError("The uploaded BigWig file could not be opened.")
    # Figwig opens files only during reads and has no persistent handle to close.
    context = (nullcontext(handle) if backend_name == "figwig"
               else handle if hasattr(handle, "__enter__") else closing(handle))
    return backend_name, context


def _validate_bigwig_handle(handle: Any) -> None:
    if getattr(handle, "type", None) not in {None, "bigwig"}:
        raise ValueError("The selected binary genomic track is not a BigWig file.")
    checker = getattr(handle, "is_bigwig", None)
    if checker is None:
        checker = getattr(handle, "isBigWig", None)
    if callable(checker) and not bool(checker()):
        raise ValueError("The selected binary genomic track is not a BigWig file.")


def _chromosomes(handle: Any) -> dict[str, int]:
    try:
        if hasattr(handle, "chrom_sizes"):
            raw = handle.chrom_sizes
        elif hasattr(handle, "chroms"):
            raw = handle.chroms()
        else:
            raw = {item.id: item.size for item in handle.chr_list}
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError("The BigWig chromosome index could not be read.") from exc
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError("The BigWig file contains no chromosomes.")
    chromosomes: dict[str, int] = {}
    for raw_name, raw_size in raw.items():
        name = str(raw_name)
        size = int(raw_size)
        if not name or size <= 0:
            raise ValueError("The BigWig chromosome index is invalid.")
        chromosomes[name] = size
    return chromosomes


def inspect_bigwig(path: Path) -> BigWigMetadata:
    """Read chromosome lengths and small scalar header metadata."""

    backend_name, context = _open_bigwig(path)
    try:
        with context as handle:
            _validate_bigwig_handle(handle)
            chromosomes = _chromosomes(handle)
            info_method = getattr(handle, "info", None)
            raw_info = info_method() if callable(info_method) else {}
    except (OSError, RuntimeError, ValueError) as exc:
        if isinstance(exc, ValueError):
            raise
        raise ValueError("The uploaded BigWig file is invalid.") from exc

    # Header dictionaries can differ between reader versions.  Only retain
    # scalar, display-safe values and cap the number/length of entries.
    file_info: list[tuple[str, str]] = []
    if isinstance(raw_info, Mapping):
        for raw_key, raw_value in tuple(raw_info.items())[:32]:
            if isinstance(raw_value, (str, int, float, bool)) or raw_value is None:
                file_info.append((str(raw_key)[:80], str(raw_value)[:160]))
    return BigWigMetadata(
        chromosomes=tuple((name, size) for name, size in chromosomes.items()),
        backend=backend_name,
        file_info=tuple(file_info),
    )


def _query_pybigtools(
    handle: Any,
    chromosome: str,
    start: int,
    end: int,
    bins: int | None,
    aggregation: str,
) -> np.ndarray:
    summary = "bin_covered" if aggregation == "coverage" else aggregation
    try:
        values = handle.values(
            chromosome,
            start,
            end,
            bins=bins,
            summary=summary,
            exact=True,
            uncovered=None,
            fillna=None,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError("The selected BigWig region could not be read.") from exc
    return np.asarray(values, dtype=float).reshape(-1)


def _query_pybigwig(
    handle: Any,
    chromosome: str,
    start: int,
    end: int,
    bins: int,
    aggregation: str,
) -> np.ndarray:
    stat = "coverage" if aggregation == "coverage" else aggregation
    try:
        if aggregation == "sum":
            means = handle.stats(
                chromosome,
                start,
                end,
                nBins=bins,
                type="mean",
                exact=True,
            )
            coverage = handle.stats(
                chromosome,
                start,
                end,
                nBins=bins,
                type="coverage",
                exact=True,
            )
            bin_width = (end - start) / bins
            raw = [
                None if mean is None else float(mean) * float(covered or 0.0) * bin_width
                for mean, covered in zip(means, coverage, strict=True)
            ]
        else:
            raw = handle.stats(
                chromosome,
                start,
                end,
                nBins=bins,
                type=stat,
                exact=True,
            )
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError("The selected BigWig region could not be read.") from exc
    return np.asarray([np.nan if value is None else value for value in raw], dtype=float)


def _query_bins(
    backend_name: str,
    handle: Any,
    chromosome: str,
    start: int,
    end: int,
    bins: int,
    aggregation: str,
    progress: Callable[[float], None] | None = None,
) -> np.ndarray:
    if backend_name == "pybigtools" and aggregation == "std":
        # Native summary SD subtracts large squared sums and can report NaN
        # for flat, nonempty bins. Use centred variance with either reader.
        values = _query_per_base(
            lambda first, width: _query_pybigtools(
                handle, chromosome, first, first + width, None, "mean",
            ), start, end, bins, aggregation, progress,
        )
    elif backend_name == "pybigtools":
        values = _query_pybigtools(handle, chromosome, start, end, bins, aggregation)
    elif backend_name == "pyBigWig":
        values = _query_pybigwig(handle, chromosome, start, end, bins, aggregation)
    elif backend_name == "figwig":
        values = _query_figwig(handle, chromosome, start, end, bins, aggregation, progress)
    else:
        values = _query_gwseq(
            handle,
            chromosome,
            start,
            end,
            bins,
            aggregation,
            progress=progress,
        )
    if progress is not None and backend_name != "gwseq_io_pp":
        progress(1.0)
    if values.size != bins:
        raise ValueError("The BigWig reader returned an unexpected number of bins.")
    if np.any(np.isinf(values)):
        raise ValueError("The selected BigWig region contains infinite values.")
    return values


def _figwig_window(handle: Any, chromosome: str, start: int, width: int) -> np.ndarray:
    try:
        values = np.asarray(handle.read(
            chromosome, [start], width=width, missing=np.nan, keep_nan=True, n_jobs=2,
        )).reshape(-1)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError("The selected BigWig region could not be read.") from exc
    if values.size != width:
        raise ValueError("The BigWig reader returned an unexpected number of bases.")
    if np.any(np.isinf(values)):
        raise ValueError("The selected BigWig region contains infinite values.")
    return values


def _reduce_base_values(values: np.ndarray, aggregation: str) -> np.ndarray:
    """Match exact pybigtools summaries, including empty bins and sample SD."""
    valid = ~np.isnan(values)
    counts = valid.sum(axis=1)
    if aggregation == "coverage":
        return counts / values.shape[1]
    if aggregation in {"min", "max"}:
        reduce = np.min if aggregation == "min" else np.max
        result = reduce(values, axis=1, where=valid,
                        initial=np.inf if aggregation == "min" else -np.inf).astype(float)
    else:
        totals = np.sum(values, axis=1, where=valid, dtype=np.float64)
        result = totals
        if aggregation in {"mean", "std"}:
            means = np.divide(totals, counts, out=np.zeros_like(totals), where=counts > 0)
            result = means
            if aggregation == "std":
                # Centre before squaring to avoid cancellation in nearly flat signals.
                deviations = np.subtract(values, means[:, None], dtype=np.float64)
                np.square(deviations, out=deviations)
                m2 = np.sum(deviations, axis=1, where=valid)
                result = np.sqrt(np.divide(m2, counts - 1, out=np.zeros_like(m2),
                                           where=counts > 1))
    result[counts == 0] = np.nan
    return result


def _query_figwig(
    handle: Any,
    chromosome: str,
    start: int,
    end: int,
    bins: int,
    aggregation: str,
    progress: Callable[[float], None] | None = None,
) -> np.ndarray:
    return _query_per_base(
        lambda first, width: _figwig_window(handle, chromosome, first, width),
        start, end, bins, aggregation, progress,
    )


def _query_per_base(
    read: Callable[[int, int], np.ndarray],
    start: int,
    end: int,
    bins: int,
    aggregation: str,
    progress: Callable[[float], None] | None = None,
) -> np.ndarray:
    """Reduce bounded per-base reads without allocating a whole chromosome."""
    width, remainder = divmod(end - start, bins)
    if remainder or width < 1:
        raise ValueError("The BigWig query could not be divided into fixed-width bins.")
    result = np.empty(bins, dtype=float)

    def report(bases: int) -> None:
        # Keep callbacks outside reader-error handlers so cancellation propagates.
        if progress is not None:
            progress(bases / (end - start))

    report(0)
    if width <= _FIGWIG_CHUNK_BASES:
        step = min(_FIGWIG_CHUNK_BINS, max(1, _FIGWIG_CHUNK_BASES // width))
        for first in range(0, bins, step):
            count = min(step, bins - first)
            values = read(start + first * width, count * width)
            if np.any(np.isinf(values)):
                raise ValueError("The selected BigWig region contains infinite values.")
            result[first:first + count] = _reduce_base_values(values.reshape(count, width), aggregation)
            report((first + count) * width)
        return result

    # A bin can itself exceed the temporary-array budget. Combine its chunks
    # using parallel-variance accumulation, preserving the sample-SD convention.
    for index in range(bins):
        count = 0
        total = mean = m2 = 0.0
        minimum, maximum = np.inf, -np.inf
        for offset in range(0, width, _FIGWIG_CHUNK_BASES):
            size = min(_FIGWIG_CHUNK_BASES, width - offset)
            raw = read(start + index * width + offset, size)
            if np.any(np.isinf(raw)):
                raise ValueError("The selected BigWig region contains infinite values.")
            values = raw[~np.isnan(raw)].astype(np.float64)
            n = values.size
            if n:
                chunk_mean = float(values.mean())
                delta = chunk_mean - mean
                m2 += float(np.sum((values - chunk_mean) ** 2)) + delta * delta * count * n / (count + n)
                mean += delta * n / (count + n)
                count += n
                total += float(values.sum())
                minimum = min(minimum, float(values.min()))
                maximum = max(maximum, float(values.max()))
            report(index * width + offset + size)
        result[index] = ({"mean": mean, "sum": total, "min": minimum, "max": maximum,
                          "std": np.sqrt(m2 / (count - 1)) if count > 1 else 0.0,
                          "coverage": count / width}[aggregation]
                         if count or aggregation == "coverage" else np.nan)
    return result


def _query_gwseq(
    handle: Any,
    chromosome: str,
    start: int,
    end: int,
    bins: int,
    aggregation: str,
    *,
    progress: Callable[[float], None] | None = None,
) -> np.ndarray:
    """Query the pure-Python fallback in chunks to bound temporary arrays."""

    if aggregation in {"min", "max"}:
        raise ValueError(
            "Minimum and maximum BigWig aggregation require pybigtools or pyBigWig; "
            "the portable Python fallback cannot calculate them reliably."
        )
    reduction = {
        "mean": "mean",
        "sum": "sum",
        "min": "min",
        "max": "max",
        "coverage": "count",
        "std": "sd",
    }[aggregation]
    if (end - start) % bins:
        raise ValueError("The BigWig query could not be divided into fixed-width bins.")
    width = (end - start) // bins
    chunk_size = 5_000
    chunks: list[np.ndarray] = []
    try:
        for first in range(0, bins, chunk_size):
            count = min(chunk_size, bins - first)
            starts = start + (first + np.arange(count, dtype=np.int64)) * width
            ends = starts + width
            chromosomes = np.full(count, chromosome, dtype=object)
            chunk = np.asarray(
                handle.quantify(
                    chromosomes,
                    starts=starts,
                    ends=ends,
                    reduce=reduction,
                    def_value=np.nan,
                    zoom=-1,
                ),
                dtype=float,
            ).reshape(-1)
            if aggregation == "coverage":
                chunk = chunk / width
            chunks.append(chunk)
            if progress is not None:
                progress(min(1.0, (first + count) / bins))
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError("The selected BigWig region could not be read.") from exc
    return np.concatenate(chunks) if len(chunks) > 1 else chunks[0]


def read_bigwig_bins(
    path: Path,
    *,
    chromosome: str,
    start: int,
    end: int,
    bin_size: int,
    aggregation: str,
    progress: Callable[[float], None] | None = None,
) -> np.ndarray:
    """Return exact fixed-width bins, preserving a final partial bin."""

    if bin_size <= 0 or end <= start:
        raise ValueError("BigWig binning requires a positive bin size and non-empty region.")
    supported = {"mean", "sum", "min", "max", "coverage", "std"}
    if aggregation not in supported:
        raise ValueError(f"Unsupported BigWig aggregation {aggregation!r}.")

    backend_name, context = _open_bigwig(path)
    try:
        with context as handle:
            _validate_bigwig_handle(handle)
            chromosomes = _chromosomes(handle)
            if chromosome not in chromosomes:
                raise KeyError(f"Chromosome {chromosome!r} was not found in the BigWig file.")
            if start < 0 or end > chromosomes[chromosome]:
                raise ValueError("The selected region lies outside the BigWig chromosome.")

            width = end - start
            full_bins, remainder = divmod(width, bin_size)
            total_bins = full_bins + int(bool(remainder))
            chunks: list[np.ndarray] = []
            if full_bins:
                full_end = start + full_bins * bin_size
                chunks.append(
                    _query_bins(
                        backend_name,
                        handle,
                        chromosome,
                        start,
                        full_end,
                        full_bins,
                        aggregation,
                        progress=(
                            None
                            if progress is None
                            else lambda fraction: progress(fraction * full_bins / total_bins)
                        ),
                    )
                )
            if remainder:
                chunks.append(
                    _query_bins(
                        backend_name,
                        handle,
                        chromosome,
                        start + full_bins * bin_size,
                        end,
                        1,
                        aggregation,
                        progress=(
                            None
                            if progress is None
                            else lambda fraction: progress((full_bins + fraction) / total_bins)
                        ),
                    )
                )
    except KeyError:
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        if isinstance(exc, ValueError):
            raise
        raise ValueError("The uploaded BigWig file is invalid.") from exc

    return np.concatenate(chunks) if len(chunks) > 1 else chunks[0]


__all__ = [
    "BigWigDependencyError",
    "BigWigMetadata",
    "inspect_bigwig",
    "read_bigwig_bins",
]
