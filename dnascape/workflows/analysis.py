"""Independent smoothing and pairwise correlation reports for uploaded profiles."""

from __future__ import annotations

import csv
import io
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

CORRELATION_METHODS = {
    "spearman": "Spearman",
    "pearson": "Pearson",
    "kendall": "Kendall tau",
}


@dataclass(frozen=True)
class SmoothingSettings:
    method: str = "none"
    window: int = 5
    sigma: float = 1.0


@dataclass(frozen=True)
class AnalysisDataset:
    name: str
    values: np.ndarray
    dependence: str = "x"


@dataclass(frozen=True)
class CorrelationMatrix:
    method: str
    coefficients: np.ndarray
    sample_counts: np.ndarray


@dataclass(frozen=True)
class AnalysisReport:
    names: tuple[str, ...]
    matrices: tuple[CorrelationMatrix, ...]


def smooth_profile(values: np.ndarray, settings: SmoothingSettings, *, dependence: str = "x") -> np.ndarray:
    """Smooth along the profile axis, retaining the grid and missing samples.

    Space-time fields are smoothed over position independently at each time.
    Moving averages truncate their window at edges. Gaussian smoothing uses
    reflected edges and normalized finite-sample weights.
    """

    array = np.asarray(values, dtype=float)
    if settings.method == "none":
        return array
    if dependence not in {"x", "t", "xt"}:
        raise ValueError("Smoothing requires a space, time, or space-time profile.")
    data = array if dependence == "xt" else array.reshape(-1)
    if data.ndim not in {1, 2} or not data.size:
        raise ValueError("Choose a nonempty profile for smoothing.")
    missing = ~np.isfinite(data)
    if settings.method == "moving_average":
        import pandas as pd

        try:
            window = int(settings.window)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Smoothing window must be a whole number from 1 to 100001 points.") from exc
        if window != settings.window or window < 1 or window > 100001:
            raise ValueError("Smoothing window must be a whole number from 1 to 100001 points.")
        if data.ndim == 1:
            smoothed = pd.Series(np.where(missing, np.nan, data)).rolling(window, center=True, min_periods=1).mean().to_numpy(copy=True)
        else:
            smoothed = pd.DataFrame(np.where(missing, np.nan, data)).rolling(window, center=True, min_periods=1).mean().to_numpy(copy=True)
    elif settings.method == "gaussian":
        from scipy.ndimage import gaussian_filter1d

        try:
            sigma = float(settings.sigma)
        except (TypeError, ValueError) as exc:
            raise ValueError("Gaussian smoothing sigma must be positive and no greater than 10000 points.") from exc
        if not np.isfinite(sigma) or not 0 < sigma <= 10000:
            raise ValueError("Gaussian smoothing sigma must be positive and no greater than 10000 points.")
        numerator = gaussian_filter1d(np.where(missing, 0, data), sigma, axis=0, mode="reflect")
        denominator = gaussian_filter1d((~missing).astype(float), sigma, axis=0, mode="reflect")
        smoothed = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)
    else:
        raise ValueError("Choose no smoothing, moving average, or Gaussian smoothing.")
    smoothed[missing] = np.nan
    return smoothed.reshape(array.shape)


def _paired_values(first: AnalysisDataset, second: AnalysisDataset) -> tuple[np.ndarray, np.ndarray]:
    if first.dependence != second.dependence:
        return np.array([]), np.array([])
    a, b = np.asarray(first.values, dtype=float), np.asarray(second.values, dtype=float)
    if first.dependence == "xt":
        if a.ndim != 2 or b.ndim != 2:
            raise ValueError("Space-time correlation requires two-dimensional arrays.")
        rows, columns = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
        a, b = a[:rows, :columns].reshape(-1), b[:rows, :columns].reshape(-1)
    else:
        a, b = a.reshape(-1), b.reshape(-1)
        length = min(a.size, b.size)
        a, b = a[:length], b[:length]
    finite = np.isfinite(a) & np.isfinite(b)
    return a[finite], b[finite]


def analyse_profiles(
    datasets: Sequence[AnalysisDataset],
    *,
    methods: Sequence[str] = ("spearman", "pearson", "kendall"),
    cancel: Callable[[], None] | None = None,
) -> AnalysisReport:
    """Calculate every dataset pair using matched indices and finite samples."""

    from scipy import stats

    selected = tuple(dict.fromkeys(methods))
    if any(method not in CORRELATION_METHODS for method in selected):
        raise ValueError("Choose Spearman, Pearson, or Kendall tau correlations.")
    if not datasets:
        raise ValueError("Upload and select at least one dataset before analysing.")
    if any(item.dependence not in {"x", "t", "xt"} for item in datasets):
        raise ValueError("Correlations require space, time, or space-time datasets.")
    size = len(datasets)
    coefficients = {method: np.full((size, size), np.nan) for method in selected}
    counts = np.zeros((size, size), dtype=np.int64)
    functions = {"spearman": stats.spearmanr, "pearson": stats.pearsonr, "kendall": stats.kendalltau}
    for i, first in enumerate(datasets):
        for j in range(i, size):
            if cancel is not None:
                cancel()
            x, y = _paired_values(first, datasets[j])
            counts[i, j] = counts[j, i] = x.size
            if x.size < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
                continue
            for method in selected:
                if cancel is not None:
                    cancel()
                value = 1.0 if i == j else float(functions[method](x, y).statistic)
                coefficients[method][i, j] = coefficients[method][j, i] = value
    counts.setflags(write=False)
    matrices = []
    for method, values in coefficients.items():
        values.setflags(write=False)
        matrices.append(CorrelationMatrix(method, values, counts))
    return AnalysisReport(tuple(item.name for item in datasets), tuple(matrices))


def export_correlation_report(
    report: AnalysisReport, *, file_format: str = "csv", method: str | None = None,
) -> bytes:
    """Write all labelled coefficients and paired sample counts into one file."""

    if file_format not in {"csv", "tsv"}:
        raise ValueError("Choose CSV or TSV for the analysis report.")
    matrices = tuple(matrix for matrix in report.matrices if method is None or matrix.method == method)
    if not matrices:
        raise ValueError("Select at least one correlation to download a report.")
    output = io.StringIO(newline="")
    writer = csv.writer(output, delimiter="," if file_format == "csv" else "\t", lineterminator="\n")
    writer.writerow((*([] if method else ["Correlation"]), "Statistic", "Dataset", *report.names))
    for matrix in matrices:
        prefix = [] if method else [CORRELATION_METHODS[matrix.method]]
        for name, row in zip(report.names, matrix.coefficients, strict=True):
            writer.writerow((*prefix, "Coefficient", name,
                             *(format(float(value), ".12g") if np.isfinite(value) else "" for value in row)))
        for name, row in zip(report.names, matrix.sample_counts, strict=True):
            writer.writerow((*prefix, "Paired samples", name, *(int(value) for value in row)))
    return output.getvalue().encode("utf-8")
