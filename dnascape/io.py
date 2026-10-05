"""Data IO helpers and path resolution for DNAscape."""

import csv
import gzip
import re
import shutil
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .constants import mapt

_MODULE_ROOT = Path(__file__).resolve().parent
_PROJECT_ROOT = _MODULE_ROOT.parent

def _resolve_data_path(relative_path, for_write=False):
    rel = Path(relative_path)
    if for_write:
        target = Path('data') / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    search_bases = (
        Path('data'),
        Path('examples/data'),
        _PROJECT_ROOT / 'data',
        _PROJECT_ROOT / 'examples' / 'data',
    )
    for base in search_bases:
        cand = base / rel
        if cand.exists():
            return cand
    return Path('data') / rel

def load(d='replication_timing', cell_line='H1', chr_number=1, example='hESC'):
    if d != 'example':
        data = np.loadtxt(_resolve_data_path(f"{mapt[d]}/{mapt[d]}_{cell_line}_chr{chr_number}.txt"), dtype=float)
    else:
        data = np.loadtxt(_resolve_data_path(f"{mapt[d]}/{example}_chr{chr_number}.txt"), dtype=float)
    return data

def loadtxt(stitle='save_example', outpath="output"):
    arr = np.loadtxt(
        _resolve_data_path(f"{outpath}/{stitle}.txt"),
        dtype=float
    )
    return arr

def loadcsv(d='replication_timing', cell_line='H1', chr_number=1, example='hESC', region='region'):
    if d == 'example':
        fname = _resolve_data_path(f"{mapt[d]}/{example}.csv")
    elif d == 'region':
        fname = _resolve_data_path(f"{mapt[d]}/{region}.csv")
    else:
        fname = _resolve_data_path(f"{mapt[d]}/{mapt[d]}_{cell_line}.csv")
    df = pd.read_csv(fname)
    col = f"chr{chr_number}"
    if col not in df.columns:
        raise KeyError(f"Column {col} not found in {fname}")
    data = df[col].dropna().to_numpy(dtype=float)
    return data

def savetxt(arr, stitle='save_example', outpath="output"):
    np.savetxt(
        _resolve_data_path(f"{outpath}/{stitle}.txt", for_write=True),
        np.asarray(arr, float)
    )

def savecsv(arr, chr_number, stitle='save_example', outpath="output"):
    fname = _resolve_data_path(f"{outpath}/{stitle}.csv", for_write=True)
    col = f"chr{chr_number}"
    arr = np.asarray(arr, float)
    df = pd.read_csv(fname) if fname.exists() else pd.DataFrame()
    if len(df) < len(arr):
        df = df.reindex(range(len(arr)))
    df[col] = pd.Series(arr, index=range(len(arr)))
    cols = list(df.columns)
    cols_sorted = sorted(
        cols,
        key=lambda c: int(c[3:]) if c.startswith("chr") else float("inf")
    )
    df = df[cols_sorted]
    df.to_csv(fname, index=False)

def csv_to_tsv(input_path, output_path=None, decimal_places=2, encoding="utf-8", input_sep=None):
    """Convert a separated-value table to a tab-separated table."""
    input_path = Path(input_path)
    output_path = input_path.with_suffix(".tsv") if output_path is None else Path(output_path)

    if decimal_places is None:
        float_format = None
    else:
        decimal_places = int(decimal_places)
        if decimal_places < 0:
            raise ValueError("decimal_places must be non-negative or None.")
        float_format = f"%.{decimal_places}f"

    read_kwargs = {"encoding": encoding}
    if input_sep is None:
        read_kwargs.update({"sep": None, "engine": "python"})
    else:
        read_kwargs["sep"] = input_sep

    df = pd.read_csv(input_path, **read_kwargs)
    if decimal_places is not None:
        numeric_cols = df.select_dtypes(include=["number"]).columns
        df[numeric_cols] = df[numeric_cols].astype(float)
    df.to_csv(output_path, sep="\t", index=False, float_format=float_format, lineterminator="\n")

    return output_path

_RDATA_FORMAT_ALIASES = {
    "auto": "auto",
    "csv": "csv",
    ".csv": "csv",
    "tsv": "tsv",
    ".tsv": "tsv",
    "tab": "tsv",
    "txt": "txt",
    ".txt": "txt",
    "text": "txt",
    "npy": "npy",
    ".npy": "npy",
    "npv": "npv",
    ".npv": "npv",
    "npz": "npz",
    ".npz": "npz",
}

_RDATA_SUFFIXES = {
    "csv": ".csv",
    "tsv": ".tsv",
    "txt": ".txt",
    "npy": ".npy",
    "npv": ".npv",
    "npz": ".npz",
}

_RDATA_TRACK_FORMAT_ALIASES = {
    "bigwig": "bigwig",
    ".bigwig": "bigwig",
    "bw": "bigwig",
    ".bw": "bigwig",
    "bedgraph": "bedgraph",
    ".bedgraph": "bedgraph",
    "bdg": "bedgraph",
    ".bdg": "bedgraph",
}
_RDATA_SOURCE_FORMAT_ALIASES = {
    **_RDATA_FORMAT_ALIASES,
    **_RDATA_TRACK_FORMAT_ALIASES,
}
_RDATA_TRACK_FORMATS = {"bigwig", "bedgraph"}

_RDATA_FORM_ALIASES = {
    "auto": "auto",
    "table": "table",
    "scalar": "scalar",
    "space": "space",
    "spatial": "space",
    "x": "space",
    "array": "space",
    "time": "time",
    "temporal": "time",
    "t": "time",
    "transposed": "time",
    "spacetime": "spacetime",
    "space_time": "spacetime",
    "space-time": "spacetime",
    "xt": "spacetime",
    "matrix": "spacetime",
}


@dataclass
class _RDataPayload:
    values: np.ndarray
    table: Optional[pd.DataFrame]
    data_form: str
    labels: Optional[list]
    source_path: Optional[Path]
    source_format: Optional[str]


def _normalise_rdata_format(fmt, *, path=None):
    if fmt is None or fmt == "auto":
        if path is None:
            return "auto"
        suffixes = [suffix.lower() for suffix in Path(path).suffixes]
        if suffixes and suffixes[-1] == ".gz":
            suffixes = suffixes[:-1]
        if not suffixes:
            raise ValueError("Could not infer data format from a path without a suffix.")
        fmt = suffixes[-1]

    key = str(fmt).strip().lower()
    if key not in _RDATA_FORMAT_ALIASES:
        allowed = ", ".join(sorted(k for k in _RDATA_FORMAT_ALIASES if not k.startswith(".")))
        raise ValueError(f"Unsupported rdata format {fmt!r}. Supported formats are: {allowed}.")
    resolved = _RDATA_FORMAT_ALIASES[key]
    if resolved == "auto":
        raise ValueError("Could not infer rdata format. Pass source_format or target explicitly.")
    return resolved


def _normalise_rdata_source_format(fmt, *, path=None):
    if fmt is None or fmt == "auto":
        if path is None:
            return "auto"
        suffixes = [suffix.lower() for suffix in Path(path).suffixes]
        if suffixes and suffixes[-1] == ".gz":
            suffixes = suffixes[:-1]
        if not suffixes:
            raise ValueError("Could not infer data format from a path without a suffix.")
        fmt = suffixes[-1]

    key = str(fmt).strip().lower()
    if key not in _RDATA_SOURCE_FORMAT_ALIASES:
        allowed = ", ".join(sorted(k for k in _RDATA_SOURCE_FORMAT_ALIASES if not k.startswith(".")))
        raise ValueError(f"Unsupported rdata source format {fmt!r}. Supported formats are: {allowed}.")
    resolved = _RDATA_SOURCE_FORMAT_ALIASES[key]
    if resolved == "auto":
        raise ValueError("Could not infer rdata source format. Pass source_format explicitly.")
    return resolved


def infer_rdata_source_format(source, source_format="auto"):
    """Infer a DNAscape data or genomic-track source format."""
    if source_format not in (None, "auto"):
        return _normalise_rdata_source_format(source_format)
    if isinstance(source, (str, Path)):
        return _normalise_rdata_source_format("auto", path=source)
    if isinstance(source, pd.DataFrame):
        return "dataframe"
    return "array"


def _normalise_rdata_form(data_form):
    key = str(data_form or "auto").strip().lower().replace("-", "_")
    if key not in _RDATA_FORM_ALIASES:
        allowed = ", ".join(sorted(_RDATA_FORM_ALIASES))
        raise ValueError(f"Unsupported data_form {data_form!r}. Supported forms are: {allowed}.")
    return _RDATA_FORM_ALIASES[key]


def _default_rdata_output_path(source_path, target_format):
    if source_path is None:
        raise ValueError("output_path is required when source is not a file path.")
    source_path = Path(source_path)
    suffix = _RDATA_SUFFIXES[target_format]
    suffixes = source_path.suffixes
    if suffixes and suffixes[-1].lower() == ".gz":
        return source_path.with_suffix("").with_suffix(suffix)
    return source_path.with_suffix(suffix)


def _looks_like_path_target(target):
    if target is None:
        return False
    text = str(target)
    return bool(Path(text).suffix) or "/" in text or "\\" in text


_RDATA_PROGRESS_WEIGHTS = {
    "array": 0.25,
    "dataframe": 0.35,
    "txt": 2.0,
    "csv": 4.0,
    "tsv": 3.5,
    "npy": 1.0,
    "npv": 1.0,
    "npz": 1.5,
    "bedgraph": 4.0,
    "bigwig": 4.0,
}


def _rdata_route_source_format(source, source_format):
    if _is_track_source_collection(source):
        return _normalise_track_format(source[0], source_format=source_format)
    return infer_rdata_source_format(source, source_format=source_format)


def _rdata_progress_weight(data_format):
    return _RDATA_PROGRESS_WEIGHTS.get(data_format, 1.0)


def _format_rdata_eta(seconds):
    if seconds is None:
        return "estimating"

    seconds = max(0, int(round(float(seconds))))
    if seconds < 1:
        return "<1s"

    minutes, seconds = divmod(seconds, 60)
    if minutes < 1:
        return f"{seconds}s"

    hours, minutes = divmod(minutes, 60)
    if hours < 1:
        return f"{minutes}m {seconds:02d}s"
    return f"{hours}h {minutes:02d}m"


def _rdata_console_progress(event):
    route = event["route"]
    progress = int(round(100 * event["progress"]))
    if event["stage"] == "complete":
        elapsed = _format_rdata_eta(event["elapsed"])
        print(f"rdata {route}: complete in {elapsed}")
        return

    stage = event["stage"].replace("_", " ")
    status = "started" if event["status"] == "start" else "done"
    eta = _format_rdata_eta(event["eta"])
    print(f"rdata {route}: {stage} {status} ({progress}%, ETA {eta})")


def _normalise_rdata_progress(progress):
    if progress in (None, False):
        return None
    if progress is True:
        return _rdata_console_progress
    if callable(progress):
        return progress
    raise TypeError("progress must be False, True, or a callable.")


class _RDataProgress:
    def __init__(self, callback, source_format, target_format):
        self.callback = callback
        self.route = f"{source_format}->{target_format}"
        self.source_format = source_format
        self.target_format = target_format
        self.stages = [
            ("load", _rdata_progress_weight(source_format)),
            ("prepare", 0.5),
            ("write", _rdata_progress_weight(target_format)),
        ]
        self.total_weight = sum(weight for _, weight in self.stages)
        self.completed_weight = 0.0
        self.started_at = time.perf_counter()
        self.stage_index = 0

    def _eta(self):
        if self.completed_weight <= 0:
            return None
        remaining = max(0.0, self.total_weight - self.completed_weight)
        elapsed = time.perf_counter() - self.started_at
        return elapsed * remaining / self.completed_weight

    def _event(self, stage, status, weight, output_path=None):
        elapsed = time.perf_counter() - self.started_at
        event = {
            "route": self.route,
            "source_format": self.source_format,
            "target_format": self.target_format,
            "stage": stage,
            "status": status,
            "step": min(self.stage_index, len(self.stages)),
            "steps": len(self.stages),
            "progress": min(1.0, self.completed_weight / self.total_weight),
            "elapsed": elapsed,
            "eta": self._eta(),
            "weight": weight,
        }
        if output_path is not None:
            event["output_path"] = str(output_path)
        self.callback(event)

    def start(self, stage, output_path=None):
        self.stage_index += 1
        weight = dict(self.stages)[stage]
        self._event(stage, "start", weight, output_path=output_path)

    def finish(self, stage, output_path=None):
        weight = dict(self.stages)[stage]
        self.completed_weight += weight
        self._event(stage, "done", weight, output_path=output_path)

    def complete(self, output_path=None):
        self.completed_weight = self.total_weight
        self.stage_index = len(self.stages)
        self._event("complete", "done", 0.0, output_path=output_path)


_TRACK_AGGREGATION_ALIASES = {
    "mean": "mean",
    "average": "mean",
    "avg": "mean",
    "sum": "sum",
    "total": "sum",
    "max": "max",
    "maximum": "max",
    "min": "min",
    "minimum": "min",
    "coverage": "coverage",
    "covered": "coverage",
    "std": "std",
    "sd": "std",
}

_TRACK_MISSING_ALIASES = {
    "nan": "nan",
    "keep": "nan",
    "keep_nan": "nan",
    "none": "nan",
    "zero": "zero",
    "zeros": "zero",
    "fill": "fill",
    "value": "fill",
    "drop": "drop",
    "dropna": "drop",
}

_TRACK_INTERPOLATION_ALIASES = {
    "none": "none",
    "off": "none",
    "linear": "linear",
    "ffill": "ffill",
    "forward": "ffill",
    "forward_fill": "ffill",
    "bfill": "bfill",
    "backward": "bfill",
    "backward_fill": "bfill",
}

_TRACK_NORMALIZATION_ALIASES = {
    "none": "none",
    "off": "none",
    "minmax": "minmax",
    "min_max": "minmax",
    "0_1": "minmax",
    "zscore": "zscore",
    "z_score": "zscore",
    "standard": "zscore",
    "center": "center",
    "centred": "center",
    "centered": "center",
    "maxabs": "maxabs",
    "max_abs": "maxabs",
}


def _normalise_alias(value, aliases, name):
    key = str(value or "none").strip().lower().replace("-", "_").replace(" ", "_")
    if key not in aliases:
        allowed = ", ".join(sorted(aliases))
        raise ValueError(f"Unsupported {name} {value!r}. Supported values are: {allowed}.")
    return aliases[key]


def _normalise_track_aggregation(aggregation):
    return _normalise_alias(aggregation, _TRACK_AGGREGATION_ALIASES, "track aggregation")


def _normalise_track_missing(missing):
    return _normalise_alias(missing, _TRACK_MISSING_ALIASES, "missing-value policy")


def _normalise_track_interpolation(interpolation):
    return _normalise_alias(interpolation, _TRACK_INTERPOLATION_ALIASES, "interpolation")


def _normalise_track_normalization(normalize):
    return _normalise_alias(normalize, _TRACK_NORMALIZATION_ALIASES, "normalization")


def _normalise_track_format(source, source_format="auto"):
    if source_format in (None, "auto") and isinstance(source, pd.DataFrame):
        return "bedgraph"
    data_format = infer_rdata_source_format(source, source_format=source_format)
    if data_format not in _RDATA_TRACK_FORMATS:
        raise ValueError(
            "Genomic track conversion supports BigWig (.bw, .bigwig) and "
            "bedGraph (.bedgraph, .bdg) inputs."
        )
    return data_format


def _require_pybigwig():
    try:
        import pyBigWig  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "BigWig conversion requires pyBigWig. Install it with "
            "`pip install pyBigWig`, then retry the DNAscape track import."
        ) from exc
    return pyBigWig


def _natural_chrom_key(chromosome):
    text = str(chromosome)
    match = re.fullmatch(r"(?:chr)?(\d+)", text, flags=re.IGNORECASE)
    if match:
        return (0, int(match.group(1)), text)
    special = {"chrx": 23, "x": 23, "chry": 24, "y": 24, "chrm": 25, "chrmt": 25, "m": 25, "mt": 25}
    key = text.lower()
    return (0, special[key], text) if key in special else (1, text)


def _normalise_track_bound(value, name):
    if value in (None, ""):
        return None
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be non-negative.")
    return value


def _standardise_bedgraph_frame(frame, value_column=None):
    if frame.empty:
        raise ValueError("The bedGraph input is empty.")

    columns_by_lower = {str(column).strip().lower(): column for column in frame.columns}

    def pick(candidates, fallback_index):
        for candidate in candidates:
            if candidate in columns_by_lower:
                return columns_by_lower[candidate]
        if fallback_index < len(frame.columns):
            return frame.columns[fallback_index]
        return None

    chrom_col = pick(("chrom", "chromosome", "chr", "#chrom"), 0)
    start_col = pick(("start", "start_bp", "chromstart", "chrom_start"), 1)
    end_col = pick(("end", "end_bp", "chromend", "chrom_end"), 2)
    if value_column is None:
        signal_col = pick(("signal", "value", "score", "coverage", "rfd", "timing"), 3)
    elif isinstance(value_column, int):
        signal_col = frame.columns[value_column]
    else:
        signal_col = columns_by_lower.get(str(value_column).strip().lower(), value_column)

    # BED3 interval files deliberately omit a signal column.  Interpreting
    # each interval as value 1 turns origin/feature calls into a useful track
    # while retaining standard bedGraph behaviour whenever a fourth column is
    # present.
    presence_track = len(frame.columns) == 3 and (
        signal_col is None
        or signal_col not in frame.columns
        or str(value_column or "").strip().casefold() in {"value", "signal"}
    )
    required_columns = [
        ("chromosome", chrom_col),
        ("start", start_col),
        ("end", end_col),
    ]
    if not presence_track:
        required_columns.append(("signal", signal_col))
    missing = [
        name
        for name, column in required_columns
        if column is None or column not in frame.columns
    ]
    if missing:
        raise ValueError(f"bedGraph input is missing required columns: {', '.join(missing)}.")

    result = pd.DataFrame(
        {
            "chrom": frame[chrom_col].astype(str),
            "start": pd.to_numeric(frame[start_col], errors="coerce"),
            "end": pd.to_numeric(frame[end_col], errors="coerce"),
            "signal": (
                np.ones(len(frame), dtype=float)
                if presence_track
                else pd.to_numeric(frame[signal_col], errors="coerce")
            ),
        }
    )
    result = result[~result["chrom"].str.lower().isin({"track", "browser"})]
    result = result.dropna(subset=["start", "end", "signal"])
    result["start"] = result["start"].astype(np.int64)
    result["end"] = result["end"].astype(np.int64)
    result = result[result["end"] > result["start"]]
    if result.empty:
        raise ValueError("No valid bedGraph intervals were found.")
    return result.reset_index(drop=True)


def _bedgraph_directive_rows(path, encoding):
    """Return physical rows containing UCSC metadata rather than intervals."""

    opener = gzip.open if str(path).lower().endswith(".gz") else open
    rows = []
    with opener(path, "rt", encoding=encoding) as handle:
        for index, line in enumerate(handle):
            stripped = line.removeprefix("\ufeff").lstrip()
            if (
                not stripped.strip()
                or stripped.startswith("#")
                or re.match(r"(?:track|browser)(?:\s|$)", stripped, flags=re.IGNORECASE)
            ):
                rows.append(index)
    return rows


def _read_bedgraph_track(source, value_column=None, encoding="utf-8"):
    if isinstance(source, pd.DataFrame):
        return _standardise_bedgraph_frame(source.copy(), value_column=value_column)

    path = Path(source).expanduser()
    raw = pd.read_csv(
        path,
        sep=r"\s+|,",
        engine="python",
        comment="#",
        header=None,
        skiprows=_bedgraph_directive_rows(path, encoding),
        encoding=encoding,
    )
    return _standardise_bedgraph_frame(raw, value_column=value_column)


def _chromosome_list(chromosome, available):
    available = sorted([str(chrom) for chrom in available], key=_natural_chrom_key)
    if chromosome in (None, "", "all", "*"):
        return available
    if isinstance(chromosome, (list, tuple, set)):
        requested = [str(chrom) for chrom in chromosome]
    else:
        requested = [str(chromosome)]
    missing = [chrom for chrom in requested if chrom not in available]
    if missing:
        raise KeyError(f"Chromosome(s) not found in track: {', '.join(missing)}.")
    return requested


def _empty_track_bins(chromosome, start_bp, end_bp, bin_size, value_name):
    starts = np.arange(start_bp, end_bp, bin_size, dtype=np.int64)
    ends = np.minimum(starts + bin_size, end_bp).astype(np.int64)
    midpoint = (starts + ends) / 2.0
    return pd.DataFrame(
        {
            "chrom": str(chromosome),
            "start_bp": starts,
            "end_bp": ends,
            "position_bp": midpoint,
            "position_kb": midpoint / 1_000.0,
            value_name: np.full(starts.size, np.nan, dtype=float),
        }
    )


def _bin_bedgraph_chromosome(intervals, chromosome, start, end, bin_size, aggregation, value_name):
    selected = intervals[intervals["chrom"] == str(chromosome)]
    start_bp = _normalise_track_bound(start, "start")
    end_bp = _normalise_track_bound(end, "end")
    if selected.empty and (start_bp is None or end_bp is None):
        raise ValueError(f"No intervals for {chromosome!r}; pass start and end to create empty bins.")
    if start_bp is None:
        start_bp = int(selected["start"].min())
    if end_bp is None:
        end_bp = int(selected["end"].max())
    if end_bp <= start_bp:
        raise ValueError("Track interval end must be greater than start.")

    table = _empty_track_bins(chromosome, start_bp, end_bp, bin_size, value_name)
    if selected.empty:
        return table

    starts = table["start_bp"].to_numpy(dtype=np.int64)
    ends = table["end_bp"].to_numpy(dtype=np.int64)
    n_bins = starts.size
    values = np.full(n_bins, np.nan, dtype=float)
    covered = np.zeros(n_bins, dtype=float)

    if aggregation in {"mean", "sum", "std"}:
        weighted = np.zeros(n_bins, dtype=float)
        weighted_sq = np.zeros(n_bins, dtype=float)
    elif aggregation == "max":
        values[:] = -np.inf
    elif aggregation == "min":
        values[:] = np.inf

    for row in selected.itertuples(index=False):
        left = max(int(row.start), start_bp)
        right = min(int(row.end), end_bp)
        if right <= left:
            continue
        first_bin = max(0, (left - start_bp) // bin_size)
        last_bin = min(n_bins - 1, (right - 1 - start_bp) // bin_size)
        signal = float(row.signal)
        for bin_index in range(first_bin, last_bin + 1):
            overlap = min(ends[bin_index], right) - max(starts[bin_index], left)
            if overlap <= 0:
                continue
            if aggregation in {"mean", "sum", "std"}:
                weighted[bin_index] += overlap * signal
                weighted_sq[bin_index] += overlap * signal * signal
                covered[bin_index] += overlap
            elif aggregation == "coverage":
                covered[bin_index] += overlap
            elif aggregation == "max":
                values[bin_index] = max(values[bin_index], signal)
            elif aggregation == "min":
                values[bin_index] = min(values[bin_index], signal)

    if aggregation == "mean":
        values = np.divide(weighted, covered, out=np.full(n_bins, np.nan), where=covered > 0)
    elif aggregation == "sum":
        values = np.where(covered > 0, weighted, np.nan)
    elif aggregation == "std":
        mean = np.divide(weighted, covered, out=np.full(n_bins, np.nan), where=covered > 0)
        second = np.divide(weighted_sq, covered, out=np.full(n_bins, np.nan), where=covered > 0)
        values = np.sqrt(np.maximum(second - mean * mean, 0.0))
    elif aggregation == "coverage":
        widths = ends - starts
        values = np.divide(covered, widths, out=np.zeros(n_bins, dtype=float), where=widths > 0)
    elif aggregation == "max":
        values = np.where(np.isfinite(values), values, np.nan)
    elif aggregation == "min":
        values = np.where(np.isfinite(values), values, np.nan)
    else:
        raise ValueError(f"Unsupported bedGraph aggregation {aggregation!r}.")

    table[value_name] = values
    return table


def _bin_bigwig_chromosome(source, chromosome, start, end, bin_size, aggregation, value_name):
    pyBigWig = _require_pybigwig()
    path = Path(source).expanduser()
    with pyBigWig.open(str(path)) as handle:
        chrom_sizes = handle.chroms()
        if chromosome not in chrom_sizes:
            raise KeyError(f"Chromosome {chromosome!r} not found in {path}.")
        start_bp = _normalise_track_bound(start, "start")
        end_bp = _normalise_track_bound(end, "end")
        if start_bp is None:
            start_bp = 0
        if end_bp is None:
            end_bp = int(chrom_sizes[chromosome])
        if end_bp <= start_bp:
            raise ValueError("Track interval end must be greater than start.")
        end_bp = min(end_bp, int(chrom_sizes[chromosome]))
        if end_bp <= start_bp:
            raise ValueError("Track interval does not overlap the selected chromosome.")

        table = _empty_track_bins(chromosome, start_bp, end_bp, bin_size, value_name)
        starts = table["start_bp"].to_numpy(dtype=np.int64)
        ends = table["end_bp"].to_numpy(dtype=np.int64)

        def exact_bin_stats(stat):
            """Query fixed-width bins without stretching the final partial bin."""
            values = np.full(starts.size, np.nan, dtype=float)
            widths = ends - starts
            full_count = int(np.count_nonzero(widths == int(bin_size)))
            if full_count:
                full_end = int(starts[0] + full_count * int(bin_size))
                values[:full_count] = np.asarray(
                    handle.stats(
                        chromosome,
                        int(starts[0]),
                        full_end,
                        nBins=full_count,
                        type=stat,
                        exact=True,
                    ),
                    dtype=float,
                )
            for index in range(full_count, starts.size):
                value = handle.stats(
                    chromosome,
                    int(starts[index]),
                    int(ends[index]),
                    nBins=1,
                    type=stat,
                    exact=True,
                )[0]
                values[index] = np.nan if value is None else float(value)
            return values

        if aggregation == "sum":
            means = exact_bin_stats("mean")
            coverage = exact_bin_stats("coverage")
            values = means * np.nan_to_num(coverage, nan=0.0) * (ends - starts)
        else:
            stat = aggregation
            if stat not in {"mean", "min", "max", "coverage", "std"}:
                raise ValueError(f"Unsupported BigWig aggregation {aggregation!r}.")
            values = exact_bin_stats(stat)

    table[value_name] = values
    return table


def _apply_track_interpolation(series, interpolation):
    if interpolation == "none":
        return series
    if interpolation == "linear":
        return series.interpolate(method="linear", limit_direction="both")
    if interpolation == "ffill":
        return series.ffill()
    if interpolation == "bfill":
        return series.bfill()
    raise AssertionError(f"Unhandled interpolation {interpolation!r}.")


def _smooth_track_series(series, smooth_window):
    smooth_window = int(smooth_window or 0)
    if smooth_window <= 1:
        return series
    return series.rolling(window=smooth_window, center=True, min_periods=1).mean()


def _normalize_track_series(series, normalize):
    values = series.to_numpy(dtype=float)
    if normalize == "none":
        return series
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return pd.Series(values, index=series.index)
    if normalize == "center":
        return pd.Series(values - finite.mean(), index=series.index)
    if normalize == "zscore":
        std = finite.std()
        if not np.isfinite(std) or std == 0:
            result = np.full_like(values, np.nan, dtype=float)
            result[np.isfinite(values)] = 0.0
            return pd.Series(result, index=series.index)
        return pd.Series((values - finite.mean()) / std, index=series.index)
    if normalize == "minmax":
        minimum = finite.min()
        maximum = finite.max()
        scale = maximum - minimum
        if not np.isfinite(scale) or scale == 0:
            result = np.full_like(values, np.nan, dtype=float)
            result[np.isfinite(values)] = 0.0
            return pd.Series(result, index=series.index)
        return pd.Series((values - minimum) / scale, index=series.index)
    if normalize == "maxabs":
        scale = np.max(np.abs(finite))
        if not np.isfinite(scale) or scale == 0:
            result = np.full_like(values, np.nan, dtype=float)
            result[np.isfinite(values)] = 0.0
            return pd.Series(result, index=series.index)
        return pd.Series(values / scale, index=series.index)
    raise AssertionError(f"Unhandled normalization {normalize!r}.")


def _postprocess_track_table(
    table,
    value_columns,
    *,
    missing="nan",
    fill_value=0.0,
    interpolation="none",
    smooth_window=1,
    normalize="none",
):
    missing = _normalise_track_missing(missing)
    interpolation = _normalise_track_interpolation(interpolation)
    normalize = _normalise_track_normalization(normalize)

    result = table.copy()
    for column in value_columns:
        series = pd.to_numeric(result[column], errors="coerce").astype(float)
        series = _apply_track_interpolation(series, interpolation)
        if missing == "zero":
            series = series.fillna(0.0)
        elif missing == "fill":
            series = series.fillna(float(fill_value))
        result[column] = series

    if missing == "drop":
        result = result.dropna(subset=value_columns).reset_index(drop=True)

    for column in value_columns:
        series = pd.to_numeric(result[column], errors="coerce").astype(float)
        series = _smooth_track_series(series, smooth_window)
        series = _normalize_track_series(series, normalize)
        result[column] = series

    return result


def _values_from_track_table(table, value_columns, data_form):
    values = table[value_columns].to_numpy(dtype=float)
    if len(value_columns) == 1:
        values = values[:, 0]
    if data_form == "auto":
        data_form = "space" if len(value_columns) == 1 else "spacetime"
    values, resolved_form = _coerce_rdata_values(values, data_form=data_form)
    return values, resolved_form


def _track_source_path(source):
    if isinstance(source, (str, Path)):
        return Path(source).expanduser()
    return None


def _track_label(source, fallback_index, label=None, value_name="value"):
    if label:
        return str(label)
    if isinstance(source, (str, Path)):
        return Path(source).name.rsplit(".", 1)[0] or value_name
    return value_name if fallback_index == 0 else f"{value_name}_{fallback_index + 1}"


def _is_track_source_collection(source):
    if not isinstance(source, (list, tuple)):
        return False
    return bool(source) and all(isinstance(item, (str, Path, pd.DataFrame)) for item in source)


def _convert_single_track_to_table(
    source,
    *,
    source_format="auto",
    chromosome=None,
    start=None,
    end=None,
    bin_size=1_000,
    aggregation="mean",
    value_column=None,
    value_name="value",
    encoding="utf-8",
):
    data_format = _normalise_track_format(source, source_format=source_format)
    aggregation = _normalise_track_aggregation(aggregation)
    bin_size = int(bin_size)
    if bin_size <= 0:
        raise ValueError("bin_size must be a positive integer.")

    if data_format == "bedgraph":
        intervals = _read_bedgraph_track(source, value_column=value_column, encoding=encoding)
        chromosomes = _chromosome_list(chromosome, intervals["chrom"].unique())
        tables = [
            _bin_bedgraph_chromosome(
                intervals,
                chrom,
                start,
                end,
                bin_size,
                aggregation,
                value_name,
            )
            for chrom in chromosomes
        ]
        return pd.concat(tables, ignore_index=True), data_format, _track_source_path(source)

    if data_format == "bigwig":
        pyBigWig = _require_pybigwig()
        source_path = _track_source_path(source)
        if source_path is None:
            raise TypeError("BigWig conversion requires a file path.")
        with pyBigWig.open(str(source_path)) as handle:
            available = handle.chroms().keys()
        chromosomes = _chromosome_list(chromosome, available)
        if chromosome in (None, "", "all", "*") and len(chromosomes) != 1:
            raise ValueError("BigWig conversion requires an explicit chromosome.")
        tables = [
            _bin_bigwig_chromosome(
                source_path,
                chrom,
                start,
                end,
                bin_size,
                aggregation,
                value_name,
            )
            for chrom in chromosomes
        ]
        return pd.concat(tables, ignore_index=True), data_format, source_path

    raise AssertionError(f"Unhandled genomic track format {data_format!r}.")


def inspect_genomic_track(source, *, source_format="auto", value_column=None, encoding="utf-8"):
    """Return format and chromosome summary for a BigWig or bedGraph track."""
    data_format = _normalise_track_format(source, source_format=source_format)
    if data_format == "bedgraph":
        intervals = _read_bedgraph_track(source, value_column=value_column, encoding=encoding)
        summary = (
            intervals.groupby("chrom", sort=False)
            .agg(start=("start", "min"), end=("end", "max"), intervals=("signal", "size"))
            .reset_index()
        )
        summary = summary.sort_values("chrom", key=lambda s: s.map(_natural_chrom_key)).reset_index(drop=True)
        chrom_sizes = {str(row.chrom): int(row.end) for row in summary.itertuples(index=False)}
        return {
            "format": data_format,
            "chromosomes": chrom_sizes,
            "chromosome_names": list(chrom_sizes),
            "table": summary,
        }

    if data_format == "bigwig":
        pyBigWig = _require_pybigwig()
        source_path = _track_source_path(source)
        if source_path is None:
            raise TypeError("BigWig inspection requires a file path.")
        with pyBigWig.open(str(source_path)) as handle:
            chrom_sizes = {str(chrom): int(size) for chrom, size in handle.chroms().items()}
        return {
            "format": data_format,
            "chromosomes": chrom_sizes,
            "chromosome_names": sorted(chrom_sizes, key=_natural_chrom_key),
            "table": pd.DataFrame(
                {
                    "chrom": sorted(chrom_sizes, key=_natural_chrom_key),
                    "start": 0,
                    "end": [chrom_sizes[chrom] for chrom in sorted(chrom_sizes, key=_natural_chrom_key)],
                    "intervals": np.nan,
                }
            ),
        }

    raise AssertionError(f"Unhandled genomic track format {data_format!r}.")


def _infer_rdata_form(values):
    arr = np.asarray(values)
    if arr.ndim == 0 or arr.size == 1:
        return "scalar"
    if arr.ndim == 1:
        return "space"
    if arr.ndim == 2:
        if arr.shape[0] == 1 and arr.shape[1] > 1:
            return "time"
        if arr.shape[1] == 1 and arr.shape[0] > 1:
            return "space"
        return "spacetime"
    raise ValueError("DNAscape data must be scalar, one-dimensional, or two-dimensional.")


def _coerce_rdata_values(values, data_form="auto"):
    arr = np.asarray(values)
    form = _normalise_rdata_form(data_form)
    if form == "auto":
        form = _infer_rdata_form(arr)

    if form == "scalar":
        if arr.size != 1:
            raise ValueError("data_form='scalar' requires exactly one value.")
        return np.asarray(arr, dtype=float).reshape(()), form

    if form == "space":
        if arr.ndim == 0:
            return np.asarray([float(arr)], dtype=float), form
        if arr.ndim == 1:
            return np.asarray(arr, dtype=float), form
        if arr.ndim == 2 and 1 in arr.shape:
            return np.asarray(arr, dtype=float).reshape(-1), form
        raise ValueError("data_form='space' requires a vector or single-row/single-column array.")

    if form == "time":
        if arr.ndim == 0:
            return np.asarray([[float(arr)]], dtype=float), form
        if arr.ndim == 1:
            return np.asarray(arr, dtype=float).reshape(1, -1), form
        if arr.ndim == 2 and 1 in arr.shape:
            return np.asarray(arr, dtype=float).reshape(1, -1), form
        raise ValueError("data_form='time' requires a vector or transposed time array.")

    if form == "spacetime":
        arr = np.asarray(arr, dtype=float)
        if arr.ndim != 2:
            raise ValueError("data_form='spacetime' requires a matrix with shape (space, time).")
        if 0 in arr.shape:
            raise ValueError("data_form='spacetime' cannot be empty.")
        return arr, form

    if form == "table":
        return np.asarray(arr), form

    raise AssertionError(f"Unhandled rdata form {form!r}.")


def _sniff_rdata_separator(path, source_format, encoding="utf-8"):
    """Return an explicit delimiter while retaining legacy delimiter detection."""
    default = {"csv": ",", "tsv": "\t"}.get(source_format)
    opener = gzip.open if str(path).lower().endswith(".gz") else open
    try:
        with opener(path, "rt", encoding=encoding, errors="replace", newline="") as handle:
            sample = handle.read(16_384)
        if not sample:
            return default
        return csv.Sniffer().sniff(sample, delimiters=",\t;|").delimiter
    except (csv.Error, OSError, UnicodeError):
        return default


def _read_rdata_table(
    path,
    source_format,
    encoding="utf-8",
    input_sep=None,
    usecols=None,
):
    if input_sep is not None:
        sep = input_sep
    else:
        sep = _sniff_rdata_separator(path, source_format, encoding=encoding)
    kwargs = {"encoding": encoding, "usecols": usecols}
    if sep is None:
        kwargs.update({"sep": None, "engine": "python"})
    else:
        kwargs["sep"] = sep
    return pd.read_csv(path, **kwargs)


def _read_rdata_npz(path, array_key=None, data_form="auto"):
    opener = gzip.open if str(path).lower().endswith(".gz") else open
    with opener(path, "rb") as source, np.load(source, allow_pickle=False) as payload:
        if array_key is None:
            for preferred in ("values", "data", "arr_0"):
                if preferred in payload.files:
                    array_key = preferred
                    break
            if array_key is None:
                array_key = payload.files[0]
        values = np.asarray(payload[array_key])
        labels = None
        table = None
        if "columns" in payload.files:
            labels = [str(x) for x in np.asarray(payload["columns"]).tolist()]
            if values.ndim == 2 and len(labels) == values.shape[1]:
                table = pd.DataFrame(values, columns=labels)
        if data_form == "auto" and "data_form" in payload.files:
            data_form = str(np.asarray(payload["data_form"]).reshape(-1)[0])
    return values, table, labels, data_form


def _selected_profile_values(series):
    """Preserve genomic gaps while removing only wide-table tail padding."""

    values = series.to_numpy(dtype=float)
    present = np.flatnonzero(~np.isnan(values))
    return values[: present[-1] + 1] if present.size else values


def _rdata_payload_from_source(
    source,
    *,
    source_format="auto",
    data_form="auto",
    chromosome=None,
    value_column=None,
    labels=None,
    array_key=None,
    encoding="utf-8",
    input_sep=None,
):
    source_path = None
    table = None
    inferred_format = None

    if isinstance(source, (str, Path)):
        source_path = Path(source).expanduser()
        inferred_format = _normalise_rdata_format(source_format, path=source_path)
        selected_column = chromosome if chromosome is not None else value_column
        if inferred_format in {"csv", "tsv"} or (
            inferred_format == "txt" and selected_column is not None
        ):
            usecols = (
                (lambda column: column == selected_column)
                if selected_column is not None
                else None
            )
            table = _read_rdata_table(
                source_path,
                inferred_format,
                encoding=encoding,
                input_sep=input_sep,
                usecols=usecols,
            )
            if chromosome is not None:
                if chromosome not in table.columns:
                    raise KeyError(f"Column {chromosome!r} not found in {source_path}.")
                values = _selected_profile_values(table[chromosome])
                labels = [str(chromosome)]
                table = None
            elif value_column is not None:
                if value_column not in table.columns:
                    raise KeyError(f"Column {value_column!r} not found in {source_path}.")
                values = _selected_profile_values(table[value_column])
                labels = [str(value_column)]
                table = None
            else:
                values = table.to_numpy()
                labels = [str(col) for col in table.columns]
                data_form = "table" if data_form == "auto" else data_form
        elif inferred_format == "txt":
            values = np.loadtxt(source_path, dtype=float)
        elif inferred_format in {"npy", "npv"}:
            if str(source_path).lower().endswith(".gz"):
                with gzip.open(source_path, "rb") as source_file:
                    values = np.load(source_file, allow_pickle=False)
            else:
                values = np.load(source_path, allow_pickle=False)
        elif inferred_format == "npz":
            values, table, labels, data_form = _read_rdata_npz(
                source_path,
                array_key=array_key,
                data_form=data_form,
            )
        else:
            raise AssertionError(f"Unhandled rdata source format {inferred_format!r}.")
    elif isinstance(source, pd.DataFrame):
        table = source.copy()
        values = table.to_numpy()
        labels = [str(col) for col in table.columns]
        data_form = "table" if data_form == "auto" else data_form
    else:
        values = np.asarray(source)

    if table is None:
        values, resolved_form = _coerce_rdata_values(values, data_form=data_form)
    else:
        resolved_form = _normalise_rdata_form(data_form)
    if labels is not None:
        labels = [str(label) for label in labels]

    return _RDataPayload(
        values=np.asarray(values),
        table=table,
        data_form=resolved_form,
        labels=labels,
        source_path=source_path,
        source_format=inferred_format,
    )


def _rdata_numeric_values(payload):
    if payload.table is not None:
        numeric = payload.table.select_dtypes(include=["number"])
        if numeric.shape[1] != payload.table.shape[1]:
            raise ValueError("Plain numeric targets require all table columns to be numeric.")
        return numeric.to_numpy(dtype=float)
    return np.asarray(payload.values, dtype=float)


def _rdata_to_dataframe(payload, value_name="value"):
    if payload.table is not None:
        return payload.table.copy()

    arr = np.asarray(payload.values)
    if arr.ndim == 0:
        return pd.DataFrame({value_name: [float(arr)]})

    if payload.data_form == "space":
        label = payload.labels[0] if payload.labels else value_name
        return pd.DataFrame({label: arr.reshape(-1)})

    if payload.data_form == "time":
        values = arr.reshape(1, -1)
        columns = payload.labels if payload.labels and len(payload.labels) == values.shape[1] else [
            f"t{i}" for i in range(values.shape[1])
        ]
        return pd.DataFrame(values, columns=columns)

    if arr.ndim == 2:
        columns = payload.labels if payload.labels and len(payload.labels) == arr.shape[1] else [
            f"t{i}" for i in range(arr.shape[1])
        ]
        return pd.DataFrame(arr, columns=columns)

    return pd.DataFrame({value_name: arr.reshape(-1)})


def _write_rdata_payload(
    payload,
    output_path,
    target_format,
    *,
    decimal_places=None,
    encoding="utf-8",
    value_name="value",
):
    output_path = Path(output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() == ".gz":
        suffix = _RDATA_SUFFIXES[target_format]
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                delete=False,
                dir=output_path.parent,
                suffix=suffix,
            ) as temp_file:
                temp_path = Path(temp_file.name)
            _write_rdata_payload(
                payload,
                temp_path,
                target_format,
                decimal_places=decimal_places,
                encoding=encoding,
                value_name=value_name,
            )
            compress_to_gz(temp_path, output_path)
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
        return output_path

    if decimal_places is None:
        float_format = None
    else:
        decimal_places = int(decimal_places)
        if decimal_places < 0:
            raise ValueError("decimal_places must be non-negative or None.")
        float_format = f"%.{decimal_places}f"

    if target_format in {"csv", "tsv"}:
        sep = "," if target_format == "csv" else "\t"
        frame = _rdata_to_dataframe(payload, value_name=value_name)
        frame.to_csv(
            output_path,
            sep=sep,
            index=False,
            float_format=float_format,
            lineterminator="\n",
            encoding=encoding,
        )
    elif target_format == "txt":
        np.savetxt(output_path, _rdata_numeric_values(payload), fmt=float_format or "%.18e")
    elif target_format in {"npy", "npv"}:
        with output_path.open("wb") as handle:
            np.save(handle, _rdata_numeric_values(payload), allow_pickle=False)
    elif target_format == "npz":
        values = _rdata_numeric_values(payload)
        archive = {"values": values, "data_form": np.asarray(payload.data_form)}
        if payload.labels is not None:
            archive["columns"] = np.asarray(payload.labels)
        elif payload.table is not None:
            archive["columns"] = np.asarray([str(col) for col in payload.table.columns])
        with output_path.open("wb") as handle:
            np.savez(handle, **archive)
    else:
        raise AssertionError(f"Unhandled rdata target format {target_format!r}.")

    return output_path


def rload(
    source,
    *,
    source_format="auto",
    data_form="auto",
    chromosome=None,
    value_column=None,
    labels=None,
    array_key=None,
    encoding="utf-8",
    input_sep=None,
    as_table=False,
    return_metadata=False,
    value_name="value",
):
    """Load DNAscape-ready data from any supported plain format.

    By default, ``rload`` returns a NumPy array suitable for DNAscape and
    Python workflows. Use ``data_form`` to enforce DNAscape conventions:
    spatial data as ``(n,)``, temporal data as ``(1, T)``, and space-time
    data as ``(n, T)``. Wide labelled tables can be loaded as arrays or
    narrowed first with ``chromosome`` or ``value_column``.
    """
    payload = _rdata_payload_from_source(
        source,
        source_format=source_format,
        data_form=data_form,
        chromosome=chromosome,
        value_column=value_column,
        labels=labels,
        array_key=array_key,
        encoding=encoding,
        input_sep=input_sep,
    )

    data = (
        _rdata_to_dataframe(payload, value_name=value_name)
        if as_table
        else _rdata_numeric_values(payload)
    )

    if not return_metadata:
        return data

    return {
        "data": data,
        "values": _rdata_numeric_values(payload),
        "table": _rdata_to_dataframe(payload, value_name=value_name),
        "data_form": payload.data_form,
        "labels": payload.labels,
        "source_format": payload.source_format,
        "source_path": payload.source_path,
    }


def rsave(
    data,
    output_path,
    *,
    target_format="auto",
    data_form="auto",
    labels=None,
    decimal_places=None,
    encoding="utf-8",
    value_name="value",
):
    """Save DNAscape-ready arrays or tables to any supported plain format.

    ``target_format`` is inferred from ``output_path`` by default. Supported
    targets are ``txt``, ``csv``, ``tsv``, ``npy``, ``npv`` and ``npz``.
    ``data_form`` follows DNAscape conventions: ``space`` is ``(n,)``,
    ``time`` is stored as ``(1, T)``, and ``spacetime`` is ``(n, T)``.
    """
    output_path = Path(output_path).expanduser()
    if target_format == "auto":
        target_format = _normalise_rdata_format("auto", path=output_path)
    else:
        target_format = _normalise_rdata_format(target_format)

    payload = _rdata_payload_from_source(
        data,
        data_form=data_form,
        labels=labels,
        encoding=encoding,
    )

    return _write_rdata_payload(
        payload,
        output_path,
        target_format,
        decimal_places=decimal_places,
        encoding=encoding,
        value_name=value_name,
    )


def rtrack(
    source,
    *,
    source_format="auto",
    chromosome=None,
    start=None,
    end=None,
    bin_size=1_000,
    aggregation="mean",
    value_column=None,
    labels=None,
    missing="nan",
    fill_value=0.0,
    interpolation="none",
    smooth_window=1,
    normalize="none",
    data_form="auto",
    as_table=False,
    return_metadata=False,
    value_name="value",
    encoding="utf-8",
):
    """Convert BigWig or bedGraph genomic signals into DNAscape-ready data.

    Genomic coordinates are binned at ``bin_size`` base-pair resolution. A
    single signal is returned as a spatial vector by default; multiple track
    sources are returned as a space-time matrix with shape ``(space, tracks)``.
    Use ``as_table=True`` to preserve chromosome and coordinate columns.
    """
    sources = list(source) if _is_track_source_collection(source) else [source]
    label_list = None
    if labels is not None:
        label_list = [str(labels)] if isinstance(labels, str) else [str(label) for label in labels]
        if len(label_list) != len(sources):
            raise ValueError("labels must match the number of genomic track sources.")

    coordinate_columns = ["chrom", "start_bp", "end_bp", "position_bp", "position_kb"]
    combined = None
    value_columns = []
    source_paths = []
    source_formats = []
    shared_start = start
    shared_end = end

    for index, item in enumerate(sources):
        if len(sources) == 1:
            column_name = label_list[0] if label_list else value_name
        else:
            column_name = _track_label(
                item,
                index,
                label=label_list[index] if label_list else None,
                value_name=value_name,
            )
        if column_name in coordinate_columns or column_name in value_columns:
            column_name = f"{column_name}_{index + 1}"

        table, item_format, item_path = _convert_single_track_to_table(
            item,
            source_format=source_format,
            chromosome=chromosome,
            start=shared_start,
            end=shared_end,
            bin_size=bin_size,
            aggregation=aggregation,
            value_column=value_column,
            value_name=column_name,
            encoding=encoding,
        )
        source_formats.append(item_format)
        if item_path is not None:
            source_paths.append(item_path)

        if combined is None:
            combined = table[coordinate_columns + [column_name]].copy()
            if shared_start in (None, "") and not combined.empty:
                shared_start = int(combined["start_bp"].iloc[0])
            if shared_end in (None, "") and not combined.empty:
                shared_end = int(combined["end_bp"].iloc[-1])
        else:
            if not combined[coordinate_columns].reset_index(drop=True).equals(
                table[coordinate_columns].reset_index(drop=True)
            ):
                raise ValueError(
                    "All genomic tracks must resolve to the same bins. Pass "
                    "chromosome, start, end and bin_size explicitly for multi-track imports."
                )
            combined[column_name] = table[column_name].to_numpy(dtype=float)
        value_columns.append(column_name)

    if combined is None:
        raise ValueError("No genomic track sources were provided.")

    combined = _postprocess_track_table(
        combined,
        value_columns,
        missing=missing,
        fill_value=fill_value,
        interpolation=interpolation,
        smooth_window=smooth_window,
        normalize=normalize,
    )
    values, resolved_form = _values_from_track_table(combined, value_columns, data_form)

    if not return_metadata:
        return combined if as_table else values

    source_format_result = source_formats[0] if len(set(source_formats)) == 1 else "tracks"
    return {
        "data": combined if as_table else values,
        "values": values,
        "table": combined,
        "data_form": resolved_form,
        "labels": value_columns,
        "source_format": source_format_result,
        "source_path": source_paths[0] if source_paths else None,
        "source_paths": source_paths,
        "chromosome": chromosome,
        "start": shared_start,
        "end": shared_end,
        "bin_size": int(bin_size),
        "aggregation": _normalise_track_aggregation(aggregation),
        "value_columns": value_columns,
    }


def rdata(
    source,
    target="tsv",
    output_path=None,
    *,
    source_format="auto",
    data_form="auto",
    chromosome=None,
    value_column=None,
    labels=None,
    array_key=None,
    decimal_places=None,
    encoding="utf-8",
    input_sep=None,
    value_name="value",
    return_data=False,
    progress=False,
    bin_size=1_000,
    start=None,
    end=None,
    aggregation="mean",
    missing="nan",
    fill_value=0.0,
    interpolation="none",
    smooth_window=1,
    normalize="none",
):
    """Convert DNAscape-ready data between plain file formats.

    Parameters
    ----------
    source : path-like, array-like, or pandas.DataFrame
        Input data. Files are auto-detected from ``.txt``, ``.csv``, ``.tsv``,
        ``.npy``, ``.npv`` and ``.npz`` suffixes. Genomic tracks are also
        accepted from ``.bw``, ``.bigwig``, ``.bedgraph`` and ``.bdg`` files.
        ``.npv`` is treated as a NumPy vector/array file using the same binary
        payload as ``.npy``.
    target : str or path-like, default "tsv"
        Target file format or output path. Supported targets are ``"txt"``,
        ``"csv"``, ``"tsv"``, ``"npy"``, ``"npv"`` and ``"npz"``.
    output_path : path-like, optional
        Explicit output path. If omitted and ``source`` is a path, the suffix is
        replaced with the target suffix.
    data_form : {"auto", "space", "time", "spacetime", "table"}, default "auto"
        DNAscape shape convention. Spatial data are vectors ``(n,)``; temporal
        data are transposed row arrays ``(1, T)``; space-time data are matrices
        ``(n, T)``.
    chromosome, value_column : str, optional
        Column to extract from a table before writing. This is useful for wide
        chromosome-labelled tables such as ``chr1``, ``chr2``, etc.
    labels : sequence of str, optional
        Column labels to use when writing arrays to CSV/TSV/NPZ.
    array_key : str, optional
        Array key to read from an ``.npz`` source.
    return_data : bool, default False
        If True, return a dictionary with ``path``, ``values``, ``table``,
        ``data_form`` and ``labels``. Otherwise return the written path.
    progress : bool or callable, default False
        If True, print route-level progress and ETA messages. If callable, it
        receives event dictionaries for the ``load``, ``prepare``, ``write``
        and ``complete`` stages.
    bin_size, start, end, aggregation, missing, interpolation, smooth_window, normalize
        Genomic-track conversion options used for BigWig and bedGraph inputs.
        Bins are emitted as DNAscape spatial arrays by default; multiple track
        sources become space-time matrices.

    Returns
    -------
    pathlib.Path or dict
        Written output path, or a result dictionary when ``return_data=True``.
    """
    if _looks_like_path_target(target) and output_path is None:
        output_path = Path(target)
        target_format = _normalise_rdata_format("auto", path=output_path)
    else:
        target_format = _normalise_rdata_format(target)

    source_route_format = _rdata_route_source_format(source, source_format)
    progress_callback = _normalise_rdata_progress(progress)
    tracker = (
        _RDataProgress(progress_callback, source_route_format, target_format)
        if progress_callback is not None
        else None
    )

    if tracker is not None:
        tracker.start("load")

    track_result = None
    if source_route_format in _RDATA_TRACK_FORMATS:
        track_result = rtrack(
            source,
            source_format=source_format,
            chromosome=chromosome,
            start=start,
            end=end,
            bin_size=bin_size,
            aggregation=aggregation,
            value_column=value_column,
            labels=labels,
            missing=missing,
            fill_value=fill_value,
            interpolation=interpolation,
            smooth_window=smooth_window,
            normalize=normalize,
            data_form=data_form,
            as_table=True,
            return_metadata=True,
            value_name=value_name,
            encoding=encoding,
        )
        payload = _RDataPayload(
            values=np.asarray(track_result["values"]),
            table=track_result["table"] if target_format in {"csv", "tsv"} else None,
            data_form=track_result["data_form"],
            labels=track_result["labels"],
            source_path=track_result["source_path"],
            source_format=track_result["source_format"],
        )
    else:
        payload = _rdata_payload_from_source(
            source,
            source_format=source_format,
            data_form=data_form,
            chromosome=chromosome,
            value_column=value_column,
            labels=labels,
            array_key=array_key,
            encoding=encoding,
            input_sep=input_sep,
        )

    if tracker is not None:
        tracker.finish("load")
        tracker.start("prepare")

    if output_path is None:
        output_path = _default_rdata_output_path(payload.source_path, target_format)

    if tracker is not None:
        tracker.finish("prepare", output_path=output_path)
        tracker.start("write", output_path=output_path)

    written_path = _write_rdata_payload(
        payload,
        output_path,
        target_format,
        decimal_places=decimal_places,
        encoding=encoding,
        value_name=value_name,
    )

    if tracker is not None:
        tracker.finish("write", output_path=written_path)
        tracker.complete(output_path=written_path)

    if not return_data:
        return written_path

    return {
        "path": written_path,
        "values": payload.values,
        "table": track_result["table"] if track_result is not None else _rdata_to_dataframe(payload, value_name=value_name),
        "data_form": payload.data_form,
        "labels": payload.labels,
        "source_format": payload.source_format,
        "target_format": target_format,
    }

def compress_to_gz(input_path, output_path=None, compresslevel=9):
    """Compress a file with gzip and return the written .gz path."""
    input_path = Path(input_path)
    output_path = Path(f"{input_path}.gz") if output_path is None else Path(output_path)

    with input_path.open("rb") as source, gzip.open(output_path, "wb", compresslevel=compresslevel) as target:
        shutil.copyfileobj(source, target)

    return output_path

def _parse_gtf_attributes(attr_str: str) -> dict:
    out = {}
    for m in re.finditer(r'(\S+)\s+"([^"]*)"\s*;', attr_str):
        out[m.group(1)] = m.group(2)
    return out

def load_gene_models_from_gtf_gz(
    gtf_gz_path: str,
    chrom: str,
    start: int,
    end: int,
    keep_gene_types=None,
    max_genes=200
):
    genes = {}
    exons_by_gene = defaultdict(list)

    with gzip.open(gtf_gz_path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line or line[0] == "#":
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9:
                continue

            seqname, source, feature, s, e, score, strand, frame, attrs = fields
            if seqname != chrom:
                continue

            s = int(s) - 1
            e = int(e)
            if e <= start or s >= end:
                continue

            ad = _parse_gtf_attributes(attrs)
            gene_id = ad.get("gene_id")
            gene_name = ad.get("gene_name", gene_id)

            if keep_gene_types is not None:
                gene_type = ad.get("gene_type") or ad.get("gene_biotype")
                if gene_type not in keep_gene_types:
                    continue

            if feature == "gene":
                if gene_id is None:
                    continue
                genes[gene_id] = {
                    "gene_id": gene_id,
                    "gene_name": gene_name,
                    "chrom": seqname,
                    "start": s,
                    "end": e,
                    "strand": strand,
                }

            elif feature == "exon":
                if gene_id is None:
                    continue
                exons_by_gene[gene_id].append((s, e))

            if len(genes) > max_genes:
                break

    for gid, exs in exons_by_gene.items():
        if gid not in genes:
            min_s = min(x[0] for x in exs)
            max_e = max(x[1] for x in exs)
            genes[gid] = {
                "gene_id": gid,
                "gene_name": gid,
                "chrom": chrom,
                "start": min_s,
                "end": max_e,
                "strand": ".",
            }
        genes[gid]["exons"] = sorted(exs)

    out = list(genes.values())
    out.sort(key=lambda g: (g["start"], g["end"]))
    return out
