"""Small cross-platform BigWig reader used by the local application.

``pybigtools`` is installed with the app on Python 3.10–3.13. ``pyBigWig`` and
the discontinued ``gwseq_io_pp`` remain compatibility fallbacks for existing
scientific environments.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager, closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


class BigWigDependencyError(ImportError):
    """Raised when no supported local BigWig reader is installed."""


@dataclass(frozen=True, slots=True)
class BigWigMetadata:
    """Bounded metadata exposed without rasterizing chromosome signals."""

    chromosomes: tuple[tuple[str, int], ...]
    backend: str
    file_info: tuple[tuple[str, str], ...] = ()


def _load_backend() -> tuple[str, Any]:
    """Prefer pybigtools and retain compatibility with existing readers."""

    try:
        return "pybigtools", importlib.import_module("pybigtools")
    except ImportError:
        try:
            return "pyBigWig", importlib.import_module("pyBigWig")
        except ImportError:
            try:
                return "gwseq_io_pp", importlib.import_module("gwseq_io_pp")
            except ImportError as exc:
                raise BigWigDependencyError(
                    "No BigWig reader is installed. For BigWig imports, use a Python "
                    "3.10–3.13 environment and run "
                    "`python -m pip install -r app/requirements.txt` from the repository "
                    "root, or install a compatible pybigtools or pyBigWig reader. "
                    "On Python 3.14 or newer, use bedGraph tracks instead."
                ) from exc


def _open_bigwig(path: Path) -> tuple[str, AbstractContextManager[Any]]:
    backend_name, backend = _load_backend()
    try:
        handle = backend.open(str(path))
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError("The uploaded BigWig file could not be opened.") from exc
    if handle is None:
        raise ValueError("The uploaded BigWig file could not be opened.")
    context = handle if hasattr(handle, "__enter__") else closing(handle)
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
        if hasattr(handle, "chroms"):
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
    bins: int,
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
    if backend_name == "pybigtools":
        values = _query_pybigtools(handle, chromosome, start, end, bins, aggregation)
    elif backend_name == "pyBigWig":
        values = _query_pybigwig(handle, chromosome, start, end, bins, aggregation)
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
