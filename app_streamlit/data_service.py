"""Content-addressed loading and canonical dataset preparation for the app."""

from __future__ import annotations

import csv
import gzip
import importlib
import re
import tempfile
import time
import zipfile
from dataclasses import asdict, dataclass, field
from io import BytesIO, TextIOWrapper
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from app_streamlit.state import UploadRecord, workflow_input_signature

PLAIN_FORMATS = {"csv", "tsv", "txt", "npy", "npv", "npz"}
TRACK_FORMATS = {"bigwig", "bedgraph"}
COORDINATE_COLUMNS = (
    "position_kb",
    "position_bp",
    "start_bp",
    "end_bp",
    "time_min",
)
INSPECTION_SAMPLE_BYTES = 1 * 1024 * 1024
MAX_MATERIALIZED_SOURCE_BYTES = 512 * 1024 * 1024
MAX_PREPARED_DATASET_BINS = 10_000_000


@dataclass(frozen=True)
class SourceInspection:
    """Lightweight source metadata collected once per uploaded content digest."""

    digest: str
    name: str
    source_format: str
    size_bytes: int
    columns: tuple[str, ...] = ()
    array_keys: tuple[str, ...] = ()
    chromosomes: tuple[str, ...] = ()
    chromosome_bounds: Mapping[str, tuple[int, int]] = field(default_factory=dict)
    shape: tuple[int, ...] | None = None
    dtype: str | None = None
    delimiter: str | None = None
    headerless: bool = False
    compressed: bool = False

    @property
    def is_track(self) -> bool:
        return self.source_format in TRACK_FORMATS


@dataclass(frozen=True)
class PreparationConfig:
    """Validated loading and preprocessing options for one canonical dataset."""

    data_form: str = "auto"
    value_column: str | None = None
    array_key: str | None = None
    chromosome: str | None = None
    start_bp: int | None = None
    end_bp: int | None = None
    bin_size_bp: int = 1_000
    aggregation: str = "mean"
    missing: str = "nan"
    fill_value: float = 0.0
    interpolation: str = "none"
    smooth_window: int = 1
    normalize: str = "none"
    value_name: str = "value"

    def validate(self, inspection: SourceInspection) -> None:
        if self.data_form not in {"auto", "table", "space", "time", "spacetime"}:
            raise ValueError("Choose a supported DNAscape data form.")
        if self.value_column and inspection.columns and self.value_column not in inspection.columns:
            raise ValueError(f"Column {self.value_column!r} is not present in the source.")
        if self.array_key and inspection.array_keys and self.array_key not in inspection.array_keys:
            raise ValueError(f"Array key {self.array_key!r} is not present in the NPZ source.")
        if self.aggregation not in {"mean", "sum", "min", "max", "coverage", "std"}:
            raise ValueError("Choose a supported genomic aggregation method.")
        if self.missing not in {"nan", "zero", "fill", "drop"}:
            raise ValueError("Choose a supported missing-value policy.")
        if self.interpolation not in {"none", "linear", "ffill", "bfill"}:
            raise ValueError("Choose a supported interpolation method.")
        if self.normalize not in {"none", "center", "zscore", "minmax", "maxabs"}:
            raise ValueError("Choose a supported normalization method.")
        if not str(self.value_name).strip():
            raise ValueError("Output feature name cannot be empty.")
        if inspection.is_track:
            if not self.chromosome:
                raise ValueError("Choose one chromosome before preparing a genomic track.")
            if self.chromosome not in inspection.chromosomes:
                raise ValueError(f"Chromosome {self.chromosome!r} is not present in the track.")
            if self.start_bp is None or self.end_bp is None:
                raise ValueError("Track preparation requires a half-open start and end interval.")
            if int(self.start_bp) < 0 or int(self.end_bp) <= int(self.start_bp):
                raise ValueError("Track interval must satisfy 0 <= start < end.")
            if int(self.bin_size_bp) <= 0:
                raise ValueError("Bin size must be greater than zero.")
            if int(self.smooth_window) <= 0:
                raise ValueError("Smoothing window must be at least one bin.")
            bounds = inspection.chromosome_bounds.get(self.chromosome)
            if bounds and int(self.end_bp) > int(bounds[1]):
                raise ValueError(
                    f"The interval end exceeds {self.chromosome}'s available bound "
                    f"({int(bounds[1]):,} bp)."
                )
            bins = int(
                np.ceil(
                    (int(self.end_bp) - int(self.start_bp))
                    / int(self.bin_size_bp)
                )
            )
            if bins > MAX_PREPARED_DATASET_BINS:
                raise ValueError(
                    f"The prepared track would contain {bins:,} bins; the app limit is "
                    f"{MAX_PREPARED_DATASET_BINS:,}. Increase the bin size or narrow "
                    "the interval."
                )


@dataclass(frozen=True)
class DatasetRecord:
    """One canonical, immutable numerical dataset shared between app pages."""

    dataset_id: str
    name: str
    source_digest: str
    source_format: str
    data_form: str
    values: np.ndarray
    labels: tuple[str, ...] = ()
    axes: Mapping[str, np.ndarray] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(axis) for axis in self.values.shape)

    @property
    def dtype(self) -> str:
        return str(self.values.dtype)

    @property
    def nbytes(self) -> int:
        return int(self.values.nbytes) + sum(
            int(np.asarray(axis).nbytes) for axis in self.axes.values()
        )

    @property
    def column_options(self) -> tuple[str, ...]:
        if self.values.ndim == 1:
            return self.labels[:1] or ("value",)
        if self.values.ndim == 2 and len(self.labels) == self.values.shape[1]:
            return self.labels
        return tuple(f"value_{index + 1}" for index in range(self.values.shape[-1]))


def detect_source_format(name: str) -> str:
    """Resolve supported plain and genomic-track filename suffixes."""

    suffixes = [suffix.lower().lstrip(".") for suffix in Path(name).suffixes]
    if not suffixes:
        raise ValueError("The uploaded file needs a supported filename extension.")
    if suffixes[-1] == "gz" and len(suffixes) > 1:
        suffix = suffixes[-2]
    else:
        suffix = suffixes[-1]
    aliases = {"bw": "bigwig", "bdg": "bedgraph"}
    resolved = aliases.get(suffix, suffix)
    if resolved not in PLAIN_FORMATS | TRACK_FORMATS:
        raise ValueError(f"Unsupported source format: .{suffix}.")
    return resolved


def _materialized_source(upload: UploadRecord, source_format: str) -> tuple[bytes, str]:
    """Return package-readable bytes and a suffix, decompressing NumPy gzip once."""

    suffixes = Path(upload.name).suffixes
    compressed = bool(suffixes and suffixes[-1].lower() == ".gz")
    if compressed and source_format == "bigwig":
        raise ValueError("Gzip-compressed BigWig is not supported; upload the .bw file directly.")
    if compressed and source_format in {"npy", "npv", "npz"}:
        try:
            with gzip.GzipFile(fileobj=BytesIO(upload.payload)) as handle:
                payload = handle.read(MAX_MATERIALIZED_SOURCE_BYTES + 1)
            if len(payload) > MAX_MATERIALIZED_SOURCE_BYTES:
                raise ValueError(
                    "The expanded NumPy source exceeds the app's 512 MiB safety limit."
                )
            return payload, f".{source_format}"
        except (OSError, EOFError) as exc:
            raise ValueError("The NumPy gzip payload is invalid or truncated.") from exc
    suffix = "".join(suffixes[-2:]) if compressed else (
        suffixes[-1] if suffixes else ".dat"
    )
    return upload.payload, suffix


def _temporary_source(upload: UploadRecord, source_format: str):
    """Return a temporary-directory context plus materialized source details."""

    payload, suffix = _materialized_source(upload, source_format)
    return tempfile.TemporaryDirectory(), suffix, payload


def _bedgraph_metadata(upload: UploadRecord) -> tuple[tuple[str, ...], dict[str, tuple[int, int]]]:
    bounds: dict[str, tuple[int, int]] = {}
    binary_handle: Any
    if upload.name.lower().endswith(".gz"):
        binary_handle = gzip.GzipFile(fileobj=BytesIO(upload.payload))
    else:
        binary_handle = BytesIO(upload.payload)
    with TextIOWrapper(binary_handle, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith(("#", "track", "browser")):
                continue
            fields = stripped.split()
            if len(fields) < 4:
                continue
            try:
                chromosome, start, end = fields[0], int(fields[1]), int(fields[2])
            except ValueError:
                continue
            if not chromosome or start < 0 or end <= start:
                continue
            previous = bounds.get(chromosome)
            bounds[chromosome] = (
                start if previous is None else min(start, previous[0]),
                end if previous is None else max(end, previous[1]),
            )
    if not bounds:
        raise ValueError("No valid bedGraph intervals were found.")
    return tuple(bounds), bounds


def _bigwig_metadata(upload: UploadRecord) -> tuple[tuple[str, ...], dict[str, tuple[int, int]]]:
    try:
        pybigwig: Any = importlib.import_module("pyBigWig")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "pyBigWig is required to inspect BigWig files. Install DNAscape's app extra."
        ) from exc
    context, suffix, payload = _temporary_source(upload, "bigwig")
    with context as directory:
        path = Path(directory) / f"source{suffix}"
        path.write_bytes(payload)
        reader = pybigwig.open(str(path))
        try:
            chrom_sizes = dict(reader.chroms() or {})
        finally:
            reader.close()
    if not chrom_sizes:
        raise ValueError("The BigWig file does not contain chromosome metadata.")
    bounds = {str(chrom): (0, int(size)) for chrom, size in chrom_sizes.items()}
    return tuple(bounds), bounds


def _numpy_array_header(handle: Any) -> tuple[tuple[int, ...], str]:
    """Read an NPY header without materializing its numerical payload."""

    try:
        version = np.lib.format.read_magic(handle)
        if version == (1, 0):
            shape, _fortran_order, dtype = np.lib.format.read_array_header_1_0(
                handle
            )
        elif version in {(2, 0), (3, 0)}:
            # NumPy exposes the v2 reader publicly; v3 shares the same length
            # encoding and differs only in header text encoding.
            shape, _fortran_order, dtype = np.lib.format.read_array_header_2_0(
                handle
            )
        else:
            raise ValueError(f"Unsupported NumPy array format version: {version!r}.")
    except (EOFError, UnicodeError, ValueError) as exc:
        raise ValueError("The NumPy source has an invalid or truncated header.") from exc
    return tuple(int(axis) for axis in shape), str(dtype)


def _numpy_inspection(
    upload: UploadRecord, source_format: str
) -> tuple[tuple[int, ...] | None, str | None, tuple[str, ...]]:
    """Inspect NPY/NPZ structure without inflating contained arrays."""

    compressed = upload.name.lower().endswith(".gz")
    if source_format in {"npy", "npv"}:
        binary: Any = (
            gzip.GzipFile(fileobj=BytesIO(upload.payload))
            if compressed
            else BytesIO(upload.payload)
        )
        with binary:
            shape, dtype = _numpy_array_header(binary)
        return shape, dtype, ()

    payload = (
        _materialized_source(upload, source_format)[0]
        if compressed
        else upload.payload
    )
    try:
        with zipfile.ZipFile(BytesIO(payload)) as archive:
            members = [
                name for name in archive.namelist() if name.lower().endswith(".npy")
            ]
            keys = tuple(name[:-4] for name in members)
            if not members:
                return None, None, keys
            with archive.open(members[0]) as member:
                shape, dtype = _numpy_array_header(member)
            return shape, dtype, keys
    except (OSError, zipfile.BadZipFile) as exc:
        raise ValueError("The NPZ source is invalid or truncated.") from exc


def inspect_source(upload: UploadRecord) -> SourceInspection:
    """Inspect headers/metadata without retaining a parsed table copy."""

    source_format = detect_source_format(upload.name)
    columns: tuple[str, ...] = ()
    array_keys: tuple[str, ...] = ()
    chromosomes: tuple[str, ...] = ()
    bounds: dict[str, tuple[int, int]] = {}
    shape = None
    dtype = None
    delimiter = None
    headerless = False
    compressed = upload.name.lower().endswith(".gz")

    if source_format == "bigwig":
        chromosomes, bounds = _bigwig_metadata(upload)
    elif source_format == "bedgraph":
        chromosomes, bounds = _bedgraph_metadata(upload)
    elif source_format in {"npy", "npv"}:
        shape, dtype, _ = _numpy_inspection(upload, source_format)
    elif source_format == "npz":
        shape, dtype, array_keys = _numpy_inspection(upload, source_format)
    else:
        import pandas as pd

        if compressed:
            try:
                with gzip.GzipFile(fileobj=BytesIO(upload.payload)) as handle:
                    sample_payload = handle.read(INSPECTION_SAMPLE_BYTES)
            except (OSError, EOFError) as exc:
                raise ValueError("The gzip source is invalid or truncated.") from exc
        else:
            sample_payload = upload.payload[:INSPECTION_SAMPLE_BYTES]
        text_sample = sample_payload.decode("utf-8-sig", errors="replace")
        content_lines = [
            line for line in text_sample.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        first_line = content_lines[0] if content_lines else ""
        if source_format == "csv":
            delimiter = ","
        elif source_format == "tsv":
            delimiter = "\t"
        else:
            try:
                delimiter = csv.Sniffer().sniff(first_line, delimiters=",\t;|").delimiter
            except csv.Error:
                delimiter = None

        if source_format == "txt":
            fields = (
                first_line.split(delimiter)
                if delimiter
                else re.split(r"\s+", first_line.strip()) if first_line else []
            )
            try:
                for field_value in fields:
                    float(field_value)
                headerless = bool(fields)
            except ValueError:
                headerless = False
            if headerless:
                columns = tuple(
                    "value" if len(fields) == 1 else f"value_{index + 1}"
                    for index in range(len(fields))
                )

        if not headerless:
            table_source: Any = (
                gzip.GzipFile(fileobj=BytesIO(upload.payload))
                if compressed
                else BytesIO(upload.payload)
            )
            with table_source:
                sample = pd.read_csv(
                    table_source,
                    sep=delimiter,
                    engine="python" if delimiter is None else "c",
                    nrows=64,
                    encoding="utf-8-sig",
                )
            columns = tuple(str(column) for column in sample.columns)

    return SourceInspection(
        digest=upload.digest,
        name=upload.name,
        source_format=source_format,
        size_bytes=upload.size_bytes,
        columns=columns,
        array_keys=array_keys,
        chromosomes=chromosomes,
        chromosome_bounds=bounds,
        shape=shape,
        dtype=dtype,
        delimiter=delimiter,
        headerless=headerless,
        compressed=compressed,
    )


def _readonly_array(values: Any) -> np.ndarray:
    array = np.ascontiguousarray(np.asarray(values))
    array.setflags(write=False)
    return array


def _axes_from_table(table: Any, length: int) -> dict[str, np.ndarray]:
    axes: dict[str, np.ndarray] = {}
    if table is None or not hasattr(table, "columns"):
        return axes
    for name in COORDINATE_COLUMNS:
        if name in table.columns and len(table[name]) == length:
            axis = np.ascontiguousarray(table[name].to_numpy())
            axis.setflags(write=False)
            axes[name] = axis
    return axes


def prepare_dataset(
    upload: UploadRecord,
    inspection: SourceInspection,
    config: PreparationConfig,
) -> DatasetRecord:
    """Load/process one source through package APIs and return a canonical record."""

    if inspection.digest != upload.digest:
        raise ValueError("Inspection metadata does not match the selected upload.")
    config.validate(inspection)
    context, suffix, payload = _temporary_source(upload, inspection.source_format)
    with context as directory:
        source_path = Path(directory) / f"source{suffix}"
        source_path.write_bytes(payload)
        if inspection.is_track:
            from dnascape.io import rtrack

            start_bp = config.start_bp
            end_bp = config.end_bp
            if start_bp is None or end_bp is None:
                raise ValueError("Track preparation requires explicit interval bounds.")

            loaded = rtrack(
                source_path,
                source_format=inspection.source_format,
                chromosome=config.chromosome,
                start=int(start_bp),
                end=int(end_bp),
                bin_size=int(config.bin_size_bp),
                aggregation=config.aggregation,
                missing=config.missing,
                fill_value=float(config.fill_value),
                interpolation=config.interpolation,
                smooth_window=int(config.smooth_window),
                normalize=config.normalize,
                data_form=config.data_form,
                as_table=True,
                return_metadata=True,
                value_name=config.value_name,
            )
        else:
            from dnascape.io import rload

            if inspection.source_format == "txt" and inspection.headerless:
                raw_values = np.loadtxt(source_path, delimiter=inspection.delimiter)
                labels = inspection.columns
                if config.value_column:
                    column_index = inspection.columns.index(config.value_column)
                    if len(inspection.columns) == 1:
                        raw_values = np.asarray(raw_values).reshape(-1)
                    else:
                        raw_values = np.atleast_2d(raw_values)[:, column_index]
                    labels = (config.value_column,)
                loaded = rload(
                    raw_values,
                    data_form=config.data_form,
                    labels=labels,
                    return_metadata=True,
                    value_name=config.value_name,
                )
            else:
                loaded = rload(
                    source_path,
                    source_format=inspection.source_format,
                    data_form=config.data_form,
                    chromosome=config.value_column,
                    value_column=config.value_column,
                    array_key=config.array_key,
                    input_sep=inspection.delimiter,
                    return_metadata=True,
                    value_name=config.value_name,
                )

    if not isinstance(loaded, Mapping):
        raise TypeError("The data loader did not return the requested metadata mapping.")
    values = _readonly_array(loaded["values"])
    length = int(values.shape[0]) if values.ndim else 1
    axes = _axes_from_table(loaded.get("table"), length)
    labels = tuple(str(label) for label in (loaded.get("labels") or ()))
    preparation = asdict(config)
    dataset_id = workflow_input_signature(upload.digest, preparation)
    metadata = {
        "preparation": preparation,
        "chromosome": config.chromosome,
        "start_bp": config.start_bp,
        "end_bp": config.end_bp,
        "bin_size_bp": config.bin_size_bp if inspection.is_track else None,
        "compressed_source": inspection.compressed,
        "headerless_source": inspection.headerless,
    }
    return DatasetRecord(
        dataset_id=dataset_id,
        name=Path(upload.name).name,
        source_digest=upload.digest,
        source_format=inspection.source_format,
        data_form=str(loaded.get("data_form") or config.data_form),
        values=values,
        labels=labels,
        axes=axes,
        metadata=metadata,
    )


def prepared_dataset_id(upload: UploadRecord, config: PreparationConfig) -> str:
    """Return the canonical id used to deduplicate one prepared upload/config pair."""

    return workflow_input_signature(upload.digest, asdict(config))


def dataset_series(record: DatasetRecord, column: str | None = None) -> np.ndarray:
    """Return one view of a dataset suitable for a one-dimensional plot."""

    values = record.values
    if values.ndim == 1:
        return values
    if values.ndim != 2:
        raise ValueError("Choose a one- or two-dimensional dataset for line plotting.")
    options = record.column_options
    selected = column or options[0]
    if selected not in options:
        raise ValueError(f"Unknown dataset column: {selected!r}.")
    return values[:, options.index(selected)]


def dataset_position_axis(record: DatasetRecord, resolution_kb: float = 1.0) -> np.ndarray:
    """Return an existing genomic axis or infer a regular left-edge grid."""

    for name in ("position_kb", "position_bp", "start_bp"):
        axis = record.axes.get(name)
        if axis is None:
            continue
        values = np.asarray(axis, dtype=float)
        if name != "position_kb":
            values = values / 1_000.0
        return values
    length = record.values.shape[0] if record.values.ndim else 1
    return np.arange(length, dtype=float) * float(resolution_kb)


def dataset_display_name(record: DatasetRecord) -> str:
    """Return a concise registry label."""

    shape = " x ".join(str(axis) for axis in record.shape) or "scalar"
    return f"{record.name} - {shape} - {record.data_form}"
