"""Data IO helpers and path resolution for DNAscape."""

from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional
import gzip
import re
import shutil
import tempfile
import time

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

for folder in (
    Path("figures"),
    Path("data/firing_rate"),
    Path("data/replication_timing"),
    Path("data/fork_directionality"),
    Path("data/efficiency"),
    Path("data/output"),
    Path("data/fit_error"),
    Path("data/region"),
    Path("data/other"),
):
    folder.mkdir(parents=True, exist_ok=True)

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
}


def _rdata_route_source_format(source, source_format):
    if isinstance(source, (str, Path)):
        return _normalise_rdata_format(source_format, path=Path(source).expanduser())
    if isinstance(source, pd.DataFrame):
        return "dataframe"
    return "array"


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


def _read_rdata_table(path, source_format, encoding="utf-8", input_sep=None):
    if input_sep is not None:
        sep = input_sep
    elif source_format == "csv":
        sep = None
    elif source_format == "tsv":
        sep = "\t"
    else:
        sep = None
    kwargs = {"encoding": encoding}
    if sep is None:
        kwargs.update({"sep": None, "engine": "python"})
    else:
        kwargs["sep"] = sep
    return pd.read_csv(path, **kwargs)


def _read_rdata_npz(path, array_key=None, data_form="auto"):
    with np.load(path, allow_pickle=False) as payload:
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
        if inferred_format in {"csv", "tsv"}:
            table = _read_rdata_table(source_path, inferred_format, encoding=encoding, input_sep=input_sep)
            if chromosome is not None:
                if chromosome not in table.columns:
                    raise KeyError(f"Column {chromosome!r} not found in {source_path}.")
                values = table[chromosome].dropna().to_numpy(dtype=float)
                labels = [str(chromosome)]
                table = None
            elif value_column is not None:
                if value_column not in table.columns:
                    raise KeyError(f"Column {value_column!r} not found in {source_path}.")
                values = table[value_column].dropna().to_numpy(dtype=float)
                labels = [str(value_column)]
                table = None
            else:
                values = table.to_numpy()
                labels = [str(col) for col in table.columns]
                data_form = "table" if data_form == "auto" else data_form
        elif inferred_format == "txt":
            values = np.loadtxt(source_path, dtype=float)
        elif inferred_format in {"npy", "npv"}:
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
):
    """Convert DNAscape-ready data between plain file formats.

    Parameters
    ----------
    source : path-like, array-like, or pandas.DataFrame
        Input data. Files are auto-detected from ``.txt``, ``.csv``, ``.tsv``,
        ``.npy``, ``.npv`` and ``.npz`` suffixes. ``.npv`` is treated as a
        NumPy vector/array file using the same binary payload as ``.npy``.
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
        "table": _rdata_to_dataframe(payload, value_name=value_name),
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
