"""Safe, in-memory inspection of files selected by the Mapping interface.

The browser upload surface deliberately does not write user data to disk.  This
module performs a small, bounded inspection as soon as a file arrives so the UI
can offer useful column and array-key defaults before a mapping run starts.
The same inspection object is later materialized into a numeric array.
"""

from __future__ import annotations

import csv
import gzip
import io
import math
import re
import zipfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import PurePath
from types import MappingProxyType
from typing import Any

import numpy as np

from dnascape._resources import checkpoint, ensure_capacity, hosted, hosted_operation

# These limits apply again on the server; the matching browser-side limit in
# ``data_upload`` is only a usability aid and must not be treated as security.
MAX_UPLOAD_BYTES = 512 * 1024 * 1024
MAX_DECOMPRESSED_BYTES = 1024 * 1024 * 1024
MAX_ARRAY_ELEMENTS = 64_000_000
# Delimited uploads are parsed into float64 storage, so the array-element and
# decompressed-byte limits are the authoritative memory bounds.  Keeping this
# alias makes the table-specific error clear without imposing a smaller,
# arbitrary limit than NPY/NPZ inputs.
MAX_TABLE_CELLS = min(MAX_ARRAY_ELEMENTS, MAX_DECOMPRESSED_BYTES // 8)
MAX_TABLE_COLUMNS = 10_000
MAX_ARCHIVE_MEMBERS = 128
MAX_COMPRESSION_RATIO = 250

_MISSING_VALUES = {"", "na", "n/a", "nan", "null", "none", "."}
_COORDINATE_NAMES = {
    "bin",
    "chrom",
    "chromosome",
    "end",
    "end bp",
    "index",
    "position",
    "position kb",
    "position bp",
    "pos",
    "start",
    "start bp",
    "thick start",
    "thick end",
    "t",
    "time",
    "time min",
    "x",
}
_METADATA_NAMES = {
    "id",
    "label",
    "name",
    "region",
    "sample",
    "strand",
    "igv score",
    "item rgb",
    "block count",
    "block sizes",
    "block starts",
}
_VALUE_NAME_PARTS = (
    "value",
    "signal",
    "profile",
    "timing",
    "firing",
    "speed",
    "density",
    "fraction",
    "directionality",
    "enrichment",
)
_NPZ_METADATA_KEYS = {
    "columns",
    "data_form",
    "position_kb",
    "region_names",
    "replicate",
    "semantic_form",
    "time_min",
}
_BIGWIG_MAGICS = (b"\x26\xfc\x8f\x88", b"\x88\x8f\xfc\x26")
_SOURCE_FORMAT_HINTS = frozenset({"auto", "bigwig", "bedgraph"})
_MAX_PANDAS_DIRECTIVE_ROWS = 10_000


class UploadInspectionError(ValueError):
    """Raised when an upload cannot be safely interpreted."""


class UploadTooLargeError(UploadInspectionError):
    """Raised when compressed or uncompressed input exceeds a safety limit."""


@dataclass(frozen=True)
class UploadInspection:
    """Metadata and bounded source bytes for one uploaded Mapping input."""

    name: str
    content_type: str
    content: bytes
    source_format: str
    compressed: bool
    delimiter: str | None = None
    has_header: bool | None = None
    columns: tuple[str, ...] = ()
    numeric_columns: tuple[str, ...] = ()
    default_column: str | None = None
    array_keys: tuple[str, ...] = ()
    default_array_key: str | None = None
    shape: tuple[int, ...] | None = None
    inferred_form: str = "unknown"
    expected_form: str = "auto"
    warnings: tuple[str, ...] = ()
    missing_columns: tuple[str, ...] = ()
    missing_count: int = 0
    semantic_form: str | None = None
    region_names: tuple[str, ...] = ()
    track_kind: str | None = None
    source_layout: str | None = None
    _normalized_payload: bytes | None = field(default=None, repr=False, compare=False)
    _table_data: Mapping[str, np.ndarray] = field(
        default_factory=lambda: MappingProxyType({}),
        repr=False,
        compare=False,
    )
    _array_data: Mapping[str, np.ndarray] = field(
        default_factory=lambda: MappingProxyType({}),
        repr=False,
        compare=False,
    )


def _safe_name(name: str) -> str:
    normalized = str(name or "upload").replace("\\", "/")
    return PurePath(normalized).name or "upload"


def _bounded_gunzip(content: bytes) -> bytes:
    chunks: list[bytes] = []
    total = 0
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(content)) as source:
            while True:
                checkpoint()
                ensure_capacity(total * 2 + 3 * 1024**2, "Decompressing the uploaded file")
                chunk = source.read(min(1024 * 1024, MAX_DECOMPRESSED_BYTES - total + 1))
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_DECOMPRESSED_BYTES:
                    raise UploadTooLargeError(
                        "The compressed file expands beyond the inspection limit."
                    )
                chunks.append(chunk)
    except UploadInspectionError:
        raise
    except (EOFError, OSError) as exc:
        raise UploadInspectionError("The GZ file is incomplete or invalid.") from exc
    return b"".join(chunks)


def _source_payload(
    name: str,
    content: bytes,
    source_format: str = "auto",
) -> tuple[str, bytes, bool]:
    lower_name = name.casefold()
    compressed = lower_name.endswith(".gz") or content.startswith(b"\x1f\x8b")
    payload = _bounded_gunzip(content) if compressed else content
    base_name = name[:-3] if lower_name.endswith(".gz") else name
    suffix = PurePath(base_name).suffix.casefold()

    format_hint = str(source_format or "auto").strip().casefold().lstrip(".")
    format_hint = {"bw": "bigwig", "bdg": "bedgraph"}.get(format_hint, format_hint)
    if format_hint not in _SOURCE_FORMAT_HINTS:
        raise UploadInspectionError("Manual source type must be Auto, BigWig, or bedGraph.")

    if payload.startswith(b"\x93NUMPY"):
        if format_hint != "auto":
            raise UploadInspectionError("The selected source type does not match the NPY data.")
        return "npy", payload, compressed
    if payload.startswith(b"PK\x03\x04"):
        if format_hint != "auto":
            raise UploadInspectionError("The selected source type does not match the NPZ data.")
        return "npz", payload, compressed
    is_bigwig = payload.startswith(_BIGWIG_MAGICS)
    if format_hint == "bigwig":
        if not is_bigwig:
            raise UploadInspectionError("The selected source is not a valid BigWig file.")
        return "bigwig", payload, compressed
    if format_hint == "bedgraph":
        return "bedgraph", payload, compressed
    if is_bigwig:
        return "bigwig", payload, compressed

    detected_format = {
        ".csv": "csv",
        ".tsv": "tsv",
        ".txt": "txt",
        ".npy": "npy",
        ".npv": "npv",
        ".npz": "npz",
        # Three-column BED interval sets are useful genomic profiles in the
        # Data Manager (for example, published SNS-seq origin calls).  They
        # share bedGraph's chromosome/start/end layout and are interpreted as
        # unit-valued intervals by the track reader below.
        ".bed": "bedgraph",
        ".bedgraph": "bedgraph",
        ".bdg": "bedgraph",
        ".bw": "bigwig",
        ".bigwig": "bigwig",
    }.get(suffix)
    if detected_format is None:
        # Text pasted into an extension-less or simply ``.gz`` upload remains
        # useful.  Binary data without a recognized signature does not.
        try:
            payload[:4096].decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise UploadInspectionError(
                "Unsupported file type. Use CSV, TSV, TXT, NPY, NPV, NPZ, BED, "
                "bedGraph, or BigWig."
            ) from exc
        detected_format = "txt"
    return detected_format, payload, compressed


def _decode_text(payload: bytes) -> str:
    if b"\x00" in payload[:4096]:
        raise UploadInspectionError("The selected table appears to be a binary file.")
    try:
        return payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise UploadInspectionError("Text uploads must use UTF-8 encoding.") from exc


def _geo_repliseq_payload(name: str, payload: bytes) -> bytes | None:
    """Transpose GEO's text MAT layout, retaining all genomic coordinates.

    GSE137764 stores chromosome/start/end and S1-S16 down the rows,
    with genomic bins across the columns. Ordinary table limits still apply
    to the resulting position-by-fraction table.
    """

    base_name = name.casefold().removesuffix(".gz")
    if PurePath(base_name).suffix not in {".mat", ".map"}:
        return None
    stream = io.BytesIO(payload)
    first = stream.readline().rstrip(b"\r\n").removeprefix(b"\xef\xbb\xbf")
    data_start = stream.tell()
    chromosomes = first.split(b"\t")
    if not chromosomes or not all(
        re.fullmatch(rb"chr[0-9A-Za-z_.-]+", value) for value in chromosomes
    ):
        return None
    bins = len(chromosomes)
    row_count = 1 + sum(1 for _line in stream)
    if row_count != 19:
        raise UploadInspectionError(
            "GEO Repli-seq matrices require chromosome, start, end, and 16 S-phase rows."
        )
    if row_count > MAX_TABLE_COLUMNS:
        raise UploadTooLargeError(
            "The table contains too many columns to inspect safely in this local session."
        )
    if bins * row_count > min(MAX_TABLE_CELLS, MAX_ARRAY_ELEMENTS, MAX_DECOMPRESSED_BYTES // 8):
        raise UploadTooLargeError(
            "The GEO Repli-seq matrix exceeds the local inspection limit. "
            "Split it into smaller genomic regions."
        )

    ensure_capacity(bins * 19 * 128, "Preparing the GEO Repli-seq matrix")
    numeric = np.empty((18, bins), dtype=float)
    stream.seek(data_start)
    for index, raw_line in enumerate(stream):
        line = raw_line.rstrip(b"\r\n")
        if line.count(b"\t") + 1 != bins:
            raise UploadInspectionError(
                "GEO Repli-seq rows must contain the same number of genomic bins."
            )
        try:
            values = np.fromiter(
                (
                    np.nan if cell.strip().casefold() in _MISSING_VALUES else float(cell)
                    for cell in line.decode("ascii").split("\t")
                ),
                dtype=float,
                count=bins,
            )
        except (ValueError, UnicodeDecodeError) as exc:
            raise UploadInspectionError(
                "GEO Repli-seq coordinate and fraction rows must be numeric."
            ) from exc
        if values.size != bins or np.any(np.isinf(values)):
            raise UploadInspectionError(
                "GEO Repli-seq coordinate and fraction rows must be numeric and finite or missing."
            )
        numeric[index] = values
    coordinates = numeric[:2]
    if (
        not np.isfinite(coordinates).all()
        or np.any(coordinates < 0)
        or np.any(coordinates != np.floor(coordinates))
        or np.any(coordinates[1] <= coordinates[0])
    ):
        raise UploadInspectionError(
            "GEO Repli-seq bins require non-negative integer starts and larger ends."
        )

    output = io.StringIO()
    writer = csv.writer(output, delimiter="\t", lineterminator="\n")
    writer.writerow(("chrom", "start", "end", *(f"S{i}" for i in range(1, 17))))
    for index, chromosome in enumerate(chromosomes):
        if index % 1024 == 0:
            checkpoint()
            ensure_capacity(output.tell() * 3, "Formatting the Repli-seq matrix")
        writer.writerow((
            chromosome.decode("ascii"), int(numeric[0, index]), int(numeric[1, index]),
            *numeric[2:, index],
        ))
        if output.tell() > MAX_DECOMPRESSED_BYTES:
            raise UploadTooLargeError(
                "The transposed GEO Repli-seq table exceeds the inspection limit."
            )
    return output.getvalue().encode("utf-8")


def _sniff_delimiter(text: str, source_format: str) -> str | None:
    default = {"csv": ",", "tsv": "\t"}.get(source_format)
    lines = [
        line for line in text.splitlines()[:40]
        if line.strip() and not (source_format == "bedgraph" and _is_bedgraph_directive(line))
    ]
    if source_format == "bedgraph" and lines:
        # Commas in BED RGB/block fields are contents, not table delimiters.
        if "\t" in lines[0]:
            return "\t"
        if len(lines[0].split()) >= 3:
            return None
    sample = "\n".join(lines)[:16_384]
    if not sample:
        return default
    try:
        return csv.Sniffer().sniff(sample, delimiters=",\t;|").delimiter
    except csv.Error:
        if default is not None:
            return default
        # TXT and bedGraph conventionally use arbitrary whitespace.
        return None


def _delimiter_sample(payload: bytes) -> str:
    """Decode a line-bounded prefix without materializing the whole table."""

    sample = payload[:65_536]
    if len(sample) < len(payload):
        newline = max(sample.rfind(b"\n"), sample.rfind(b"\r"))
        if newline >= 0:
            sample = sample[: newline + 1]
    return _decode_text(sample)


def _is_bedgraph_directive(line: bytes | str) -> bool:
    """Return whether a line is UCSC bedGraph metadata rather than interval data."""

    if isinstance(line, bytes):
        stripped = line.removeprefix(b"\xef\xbb\xbf").strip()
        lowered = stripped.lower()
        return (
            lowered.startswith(b"#")
            or lowered == b"track"
            or lowered.startswith(b"track ")
            or lowered == b"browser"
            or lowered.startswith(b"browser ")
        )
    stripped = line.removeprefix("\ufeff").strip()
    lowered = stripped.casefold()
    return (
        lowered.startswith("#")
        or lowered == "track"
        or lowered.startswith("track ")
        or lowered == "browser"
        or lowered.startswith("browser ")
    )


def _bedgraph_directive_rows(payload: bytes) -> list[int] | None:
    """Return bounded physical line numbers pandas should skip for bedGraph."""

    rows: list[int] = []
    for index, line in enumerate(io.BytesIO(payload)):
        if not _is_bedgraph_directive(line):
            continue
        rows.append(index)
        if len(rows) > _MAX_PANDAS_DIRECTIVE_ROWS:
            # A deliberately directive-heavy file is safer through the bounded
            # streaming parser than as an unbounded pandas skip-row list.
            return None
    return rows


def _iter_table_rows(
    payload: bytes,
    delimiter: str | None,
    *,
    source_format: str,
) -> Iterator[list[str]]:
    """Yield every numeric-table row, including entirely missing bins."""

    if b"\x00" in payload[:4096]:
        raise UploadInspectionError("The selected table appears to be a binary file.")
    stream = io.TextIOWrapper(io.BytesIO(payload), encoding="utf-8-sig", newline="")
    try:
        lines = (
            line
            for line in stream
            if not (
                source_format == "bedgraph"
                and (not line.strip() or _is_bedgraph_directive(line))
            )
        )
        if delimiter is None:
            raw_rows: Iterator[Sequence[str]] = (
                re.split(r"\s+", line.strip()) if line.strip() else [""]
                for line in lines
            )
        else:
            raw_rows = csv.reader(lines, delimiter=delimiter)
        for raw_row in raw_rows:
            row = [str(cell).strip() for cell in raw_row]
            if source_format != "bedgraph" or (row and any(row)):
                yield row or [""]
    except UnicodeDecodeError as exc:
        raise UploadInspectionError("Text uploads must use UTF-8 encoding.") from exc
    except csv.Error as exc:
        raise UploadInspectionError("The selected table is not valid delimited text.") from exc
    finally:
        stream.close()


def _preflight_table_width(
    payload: bytes,
    delimiter: str | None,
    *,
    source_format: str,
) -> None:
    """Reject pathological single rows before a parser can expand them."""

    encoded_delimiter = delimiter.encode("ascii") if delimiter is not None else None
    for line in io.BytesIO(payload):
        if source_format == "bedgraph" and _is_bedgraph_directive(line):
            continue
        if encoded_delimiter is not None:
            width = line.count(encoded_delimiter) + 1
        else:
            width = 0
            inside_value = False
            for byte in line:
                whitespace = byte in b" \t\r\n\v\f"
                if not whitespace and not inside_value:
                    width += 1
                    inside_value = True
                elif whitespace:
                    inside_value = False
        if width > MAX_TABLE_COLUMNS:
            raise UploadTooLargeError(
                "The table contains too many columns to inspect safely in this local session."
            )


def _is_number(value: str) -> bool:
    try:
        number = float(value)
    except ValueError:
        return False
    return bool(np.isfinite(number))


def _has_header(rows: Sequence[Sequence[str]], source_format: str) -> bool:
    first = tuple(rows[0])
    if source_format == "bedgraph":
        canonical = {cell.casefold() for cell in first}
        return (
            bool(canonical & {"chr", "chrom", "chromosome"})
            and {
                "start",
                "end",
            }
            <= canonical
        )
    if len(rows) == 1:
        return any(not _is_number(cell) for cell in first if cell.casefold() not in _MISSING_VALUES)

    following = rows[1 : min(len(rows), 25)]
    for index, value in enumerate(first):
        if value.casefold() in _MISSING_VALUES or _is_number(value):
            continue
        subsequent = [row[index] for row in following if index < len(row)]
        present = [item for item in subsequent if item.casefold() not in _MISSING_VALUES]
        if present and sum(_is_number(item) for item in present) / len(present) >= 0.8:
            return True
    normalized = {re.sub(r"[_-]+", " ", value.casefold()).strip() for value in first}
    return bool(
        normalized
        & {
            "chrom",
            "chromosome",
            "position",
            "start",
            "end",
            "value",
            "signal",
            "time",
        }
    )


def _deduplicate_columns(values: Sequence[str]) -> tuple[str, ...]:
    counts: dict[str, int] = {}
    columns: list[str] = []
    for index, raw in enumerate(values, start=1):
        base = str(raw).strip() or f"Column {index}"
        count = counts.get(base, 0) + 1
        counts[base] = count
        columns.append(base if count == 1 else f"{base} ({count})")
    return tuple(columns)


def _bed_interval_columns(payload: bytes, width: int, source_name: str) -> tuple[str, ...]:
    """Use declared BED fields before treating numeric display fields as signals."""

    declared: dict[int, str] = {}
    first_values: list[str] = []
    for line in io.TextIOWrapper(io.BytesIO(payload), encoding="utf-8-sig"):
        if not line.strip():
            continue
        if not _is_bedgraph_directive(line):
            first_values = re.split(r"\s+", line.strip())
            break
        match = re.match(r"\s*#\s*Column\s+(\d+)\s*:\s*(.+)", line, re.IGNORECASE)
        if match:
            description = match.group(2).strip().casefold()
            if "enrichment ratio" in description:
                label = "enrichment_ratio"
            elif "score" in description and "igv" in description:
                label = "igv_score"
            elif "thick" in description and "start" in description:
                label = "thick_start"
            elif "thick" in description and ("stop" in description or "end" in description):
                label = "thick_end"
            elif "strand" in description:
                label = "strand"
            elif "red, green, and blue" in description or "rgb" in description:
                label = "item_rgb"
            else:
                label = re.sub(r"[^a-z0-9]+", "_", description).strip("_")
            declared[int(match.group(1)) - 1] = label

    base_name = source_name.casefold().removesuffix(".gz")
    if base_name.endswith(".bed") and width >= 5 and not declared:
        try:
            quantitative_name = np.isfinite(float(first_values[3]))
        except (IndexError, ValueError):
            quantitative_name = False
        names = (
            "chromosome", "start", "end",
            "value" if quantitative_name else "name",
            "igv_score" if quantitative_name else "score", "strand",
            "thick_start", "thick_end", "item_rgb", "block_count", "block_sizes", "block_starts",
        )
    else:
        names = ("chromosome", "start", "end", "value")
    return _deduplicate_columns(
        [
            ("chromosome", "start", "end")[index]
            if index < 3
            else declared.get(index, names[index] if index < len(names) else f"Column {index + 1}")
            for index in range(width)
        ]
    )


def _table_layout(
    payload: bytes,
    source_format: str,
    *,
    source_name: str = "",
) -> tuple[str | None, bool, tuple[str, ...], int, int]:
    """Inspect table structure in constant auxiliary memory.

    The first pass retains only the small sample needed for header detection.
    It also establishes the rectangular shape before numeric storage is
    allocated, allowing ragged chromosome tracks to be padded without losing
    row alignment.
    """

    delimiter = _sniff_delimiter(_delimiter_sample(payload), source_format)
    _preflight_table_width(
        payload,
        delimiter,
        source_format=source_format,
    )
    samples: list[list[str]] = []
    row_count = 0
    width = 0
    table_limit = min(MAX_TABLE_CELLS, MAX_ARRAY_ELEMENTS, MAX_DECOMPRESSED_BYTES // 8)
    for row in _iter_table_rows(
        payload,
        delimiter,
        source_format=source_format,
    ):
        row_count += 1
        width = max(width, len(row))
        if row_count * width > table_limit + width:
            raise UploadTooLargeError(
                "The table contains more than eight million values and cannot be kept safely "
                "in this local session. Split it into smaller files or use NPY/NPZ."
            )
        # Empty bins count towards the shape without hiding the numeric
        # samples needed to distinguish chromosome headers from data.
        if len(samples) < 25 and (not samples or any(row)):
            samples.append(row)
    if not row_count:
        raise UploadInspectionError("The selected file is empty.")
    if width == 0:
        raise UploadInspectionError("The selected file contains no columns.")

    padded_samples = [row + [""] * (width - len(row)) for row in samples]
    has_header = _has_header(padded_samples, source_format)
    data_rows = row_count - int(has_header)
    if data_rows <= 0:
        raise UploadInspectionError("The selected table contains a header but no data rows.")
    cells = data_rows * width
    if cells > table_limit:
        raise UploadTooLargeError(
            "The table contains more than eight million values and cannot be kept safely "
            "in this local session. Split it into smaller files or use NPY/NPZ."
        )

    first = padded_samples[0]
    if has_header:
        columns = _deduplicate_columns(first)
    elif source_format == "bedgraph" and width >= 3:
        columns = _bed_interval_columns(payload, width, source_name)
    else:
        columns = tuple(f"Column {index}" for index in range(1, width + 1))
    return delimiter, has_header, columns, data_rows, width


def _stream_numeric_columns(
    payload: bytes,
    delimiter: str | None,
    columns: Sequence[str],
    *,
    data_rows: int,
    has_header: bool,
    source_format: str,
) -> dict[str, np.ndarray]:
    """Parse a table into one bounded float matrix without retaining strings."""

    width = len(columns)
    ensure_capacity(data_rows * width * 32, "Reading the numeric table")
    values = np.full((data_rows, width), np.nan, dtype=float)
    numeric = np.ones(width, dtype=bool)
    row_index = 0
    for source_index, row in enumerate(
        _iter_table_rows(
            payload,
            delimiter,
            source_format=source_format,
        )
    ):
        if has_header and source_index == 0:
            continue
        if row_index >= data_rows:
            raise UploadInspectionError("The table changed while it was being inspected.")
        for column_index in range(min(width, len(row))):
            raw_value = row[column_index]
            if raw_value.casefold() in _MISSING_VALUES or not numeric[column_index]:
                continue
            try:
                number = float(raw_value)
            except ValueError:
                numeric[column_index] = False
                continue
            if not np.isfinite(number):
                raise UploadInspectionError(
                    f"Column {columns[column_index]!r} contains an infinite numeric value."
                )
            values[row_index, column_index] = number
        row_index += 1
    if row_index != data_rows:
        raise UploadInspectionError("The table changed while it was being inspected.")

    result: dict[str, np.ndarray] = {}
    for index, column in enumerate(columns):
        if numeric[index]:
            column_values = values[:, index]
            column_values.setflags(write=False)
            result[column] = column_values
    return result


def _pandas_numeric_columns(
    payload: bytes,
    delimiter: str | None,
    columns: Sequence[str],
    *,
    data_rows: int,
    has_header: bool,
    expected_form: str,
    source_format: str,
) -> dict[str, np.ndarray] | None:
    """Use pandas' bounded C parser when the core dependency is available."""

    ensure_capacity(data_rows * len(columns) * 64, "Parsing the uploaded table")
    try:
        import pandas as pd
    except ImportError:  # pragma: no cover - supported minimal installations
        return None

    skipped_rows: list[int] | None = []
    if source_format == "bedgraph":
        skipped_rows = _bedgraph_directive_rows(payload)
        if skipped_rows is None:
            return None

    if not has_header:
        selected_indices = list(range(len(columns)))
    elif expected_form.casefold() == "regions":
        selected_indices = [
            index for index, column in enumerate(columns) if not _is_metadata_column(column)
        ]
    else:
        selected_indices = [
            index
            for index, column in enumerate(columns)
            if not _is_coordinate_column(column) and not _is_metadata_column(column)
        ]
    if not selected_indices:
        return {}

    separator = delimiter if delimiter is not None else r"\s+"
    try:
        frame = pd.read_csv(
            io.BytesIO(payload),
            sep=separator,
            header=0 if has_header else None,
            names=list(columns),
            usecols=selected_indices,
            encoding="utf-8-sig",
            engine="c",
            keep_default_na=True,
            low_memory=False,
            skip_blank_lines=False,
            skiprows=skipped_rows or None,
        )
    except (UnicodeDecodeError, ValueError, pd.errors.ParserError) as exc:
        raise UploadInspectionError("The selected table is not valid delimited text.") from exc
    if len(frame) != data_rows:
        # Fall back to the row-preserving parser rather than changing axes.
        return None

    result: dict[str, np.ndarray] = {}
    for index in selected_indices:
        column = columns[index]
        series: Any = frame[column]
        missing = series.isna()
        if not pd.api.types.is_numeric_dtype(series.dtype):
            normalized = series.astype("string").str.strip().str.casefold()
            missing = missing | normalized.isin(_MISSING_VALUES)
        converted: Any = pd.to_numeric(series.mask(missing), errors="coerce")
        invalid = (~missing) & converted.isna()
        if bool(invalid.any()):
            continue
        array = converted.to_numpy(dtype=float, na_value=np.nan, copy=False)
        if np.any(np.isinf(array)):
            raise UploadInspectionError(f"Column {column!r} contains an infinite numeric value.")
        array.setflags(write=False)
        result[column] = array
    return result


def _stream_table_parts(
    payload: bytes,
    source_format: str,
    expected_form: str,
    *,
    source_name: str = "",
) -> tuple[str | None, bool, tuple[str, ...], int, dict[str, np.ndarray]]:
    delimiter, has_header, columns, data_rows, _ = _table_layout(
        payload, source_format, source_name=source_name
    )
    numeric = _pandas_numeric_columns(
        payload,
        delimiter,
        columns,
        data_rows=data_rows,
        has_header=has_header,
        expected_form=expected_form,
        source_format=source_format,
    )
    if numeric is None:
        numeric = _stream_numeric_columns(
            payload,
            delimiter,
            columns,
            data_rows=data_rows,
            has_header=has_header,
            source_format=source_format,
        )
    return delimiter, has_header, columns, data_rows, numeric


def _frozen_arrays(values: Mapping[str, np.ndarray]) -> Mapping[str, np.ndarray]:
    for array in values.values():
        array.setflags(write=False)
    return MappingProxyType(dict(values))


def _normalized_column_name(column: str) -> str:
    return re.sub(r"[_-]+", " ", column.casefold()).strip()


def _is_coordinate_column(column: str) -> bool:
    name = _normalized_column_name(column)
    return name in _COORDINATE_NAMES or name.startswith("position ") or name.startswith("time ")


def _is_metadata_column(column: str) -> bool:
    return _normalized_column_name(column) in _METADATA_NAMES


def _selectable_numeric_columns(
    columns: Sequence[str],
    numeric: Mapping[str, np.ndarray],
    *,
    expected_form: str,
    source_format: str,
    has_header: bool,
) -> tuple[str, ...]:
    """Return actual signal choices, never coordinate-only fallbacks."""

    if expected_form.casefold() == "regions":
        return tuple(column for column in numeric if not _is_metadata_column(column))

    potential_signals = tuple(
        column
        for column in columns
        if not _is_coordinate_column(column) and not _is_metadata_column(column)
    )
    if has_header:
        invalid_signals = tuple(column for column in potential_signals if column not in numeric)
        if invalid_signals:
            joined = ", ".join(repr(column) for column in invalid_signals)
            raise UploadInspectionError(
                f"Signal column(s) {joined} contain nonnumeric values. Correct the data before mapping."
            )

    selectable = tuple(column for column in potential_signals if column in numeric)
    if selectable:
        return selectable

    # Headerless tables have generic names, so any numeric column is a valid
    # signal. Named coordinate-only tables are rejected rather than silently
    # mapping a position or time axis as if it were the biological profile.
    generic = all(re.fullmatch(r"Column \d+", column) for column in columns)
    if generic and numeric:
        return tuple(numeric)
    kind = "bedGraph signal" if source_format == "bedgraph" else "signal data"
    raise UploadInspectionError(f"No numeric {kind} column was detected in the selected table.")


def _default_numeric_column(columns: Sequence[str]) -> str | None:
    if not columns:
        return None

    def score(item: tuple[int, str]) -> tuple[int, int]:
        index, column = item
        normalized = _normalized_column_name(column)
        if re.fullmatch(r"chr(?:omosome)?\s*0*1", normalized):
            return (-200, index)
        value_score = -80 if any(part in normalized for part in _VALUE_NAME_PARTS) else 0
        chromosome_score = -50 if re.fullmatch(r"chr(?:omosome)?\s*\d+", normalized) else 0
        coordinate_penalty = 150 if _is_coordinate_column(column) else 0
        return (value_score + chromosome_score + coordinate_penalty, index)

    return min(enumerate(columns), key=score)[1]


def _validate_numeric_array(
    values: Any,
    *,
    label: str = "array",
    allow_missing: bool = False,
) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype.kind not in "biuf":
        raise UploadInspectionError(f"The selected {label} must contain real numeric values.")
    if array.size > MAX_ARRAY_ELEMENTS or array.nbytes > MAX_DECOMPRESSED_BYTES:
        raise UploadTooLargeError(
            f"The selected {label} is too large to keep safely in this local session."
        )
    ensure_capacity(int(array.size) * 32, "Preparing the uploaded numeric array")
    result = np.asarray(array, dtype=float)
    if result.size == 0:
        raise UploadInspectionError(f"The selected {label} is empty.")
    if np.any(np.isinf(result)):
        raise UploadInspectionError(f"The selected {label} contains infinite numeric values.")
    if not allow_missing and np.any(np.isnan(result)):
        raise UploadInspectionError(f"The selected {label} contains missing numeric values.")
    return result


def _archive_members(payload: bytes) -> None:
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            members = archive.infolist()
    except zipfile.BadZipFile as exc:
        raise UploadInspectionError("The NPZ archive is incomplete or invalid.") from exc
    if len(members) > MAX_ARCHIVE_MEMBERS:
        raise UploadTooLargeError("The NPZ archive contains too many arrays.")
    expanded = sum(member.file_size for member in members)
    ensure_capacity(expanded * 4, "Expanding the NumPy archive")
    packed = max(1, sum(member.compress_size for member in members))
    if expanded > MAX_DECOMPRESSED_BYTES or expanded / packed > MAX_COMPRESSION_RATIO:
        raise UploadTooLargeError("The NPZ archive expands beyond the safe inspection limit.")
    if hosted():
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            for member in members:
                if member.filename.endswith(".npy"):
                    with archive.open(member) as stream:
                        _preflight_npy(stream)


def _preflight_npy(payload):
    """Read declared dimensions before np.load can allocate a malicious shape."""
    if not hosted():
        return
    stream = io.BytesIO(payload) if isinstance(payload, bytes) else payload
    version = np.lib.format.read_magic(stream)
    reader = (np.lib.format.read_array_header_1_0 if version == (1, 0)
              else np.lib.format.read_array_header_2_0)
    shape, _fortran, dtype = reader(stream)
    ensure_capacity(math.prod(shape) * max(8, dtype.itemsize) * 4,
                    "Loading the NumPy array")


def _preferred_array_key(keys: Sequence[str]) -> str | None:
    for preferred in ("values", "data", "arr_0"):
        if preferred in keys:
            return preferred
    return keys[0] if keys else None


def _infer_form(shape: tuple[int, ...], expected_form: str, source_format: str) -> str:
    expected = expected_form.casefold()
    if source_format in {"bedgraph", "bigwig"}:
        return "track"
    if expected in {"scalar", "x", "t", "xt", "table", "regions"}:
        return expected
    if not shape or shape == () or int(np.prod(shape, dtype=np.int64)) == 1:
        return "scalar"
    if len(shape) == 1:
        return "x"
    if len(shape) == 2 and shape[0] == 1:
        return "t"
    if len(shape) == 2:
        return "xt"
    return "array"


def _infer_table_form(
    *,
    columns: Sequence[str],
    selectable: Sequence[str],
    data_rows: int,
    expected_form: str,
    source_format: str,
) -> str:
    """Apply rdata's shape convention after table metadata columns are removed."""

    if expected_form.casefold() != "auto" or source_format == "bedgraph":
        return _infer_form((data_rows, len(selectable)), expected_form, source_format)
    if not selectable:
        return "unknown"
    if data_rows == 1 and len(selectable) > 1:
        return "t"
    if len(selectable) > 1:
        normalized_signals = tuple(_normalized_column_name(column) for column in selectable)
        if all(
            re.fullmatch(r"chr(?:omosome)?\s*(?:\d+|x|y|m|mt)", column)
            for column in normalized_signals
        ):
            # Wide chromosome tables contain alternative spatial profiles, not
            # an X-by-T matrix. The selected header determines the active array.
            return "x"
        return "xt"
    coordinate_names = {_normalized_column_name(column) for column in columns}
    has_time_axis = bool(coordinate_names & {"t", "time", "time min"})
    has_space_axis = bool(
        coordinate_names & {"bin", "position", "position kb", "position bp", "pos", "x"}
    )
    return "t" if has_time_axis and not has_space_axis else "x"


def _inspect_table(
    *,
    name: str,
    content_type: str,
    content: bytes,
    payload: bytes,
    source_format: str,
    compressed: bool,
    expected_form: str,
) -> UploadInspection:
    delimiter, has_header, columns, data_rows, numeric = _stream_table_parts(
        payload,
        source_format,
        expected_form,
        source_name=name,
    )
    # BED3 origin/feature calls contain positions but no explicit signal.
    # Treat every interval as value 1 so the existing track-binning pipeline
    # can create coverage, sum, or mean profiles without mistaking start/end
    # coordinates for biological values.
    presence_track = source_format == "bedgraph" and len(columns) == 3
    if presence_track:
        numeric = dict(numeric)
        numeric["value"] = np.ones(data_rows, dtype=float)
        selectable = ("value",)
    else:
        selectable = _selectable_numeric_columns(
            columns,
            numeric,
            expected_form=expected_form,
            source_format=source_format,
            has_header=has_header,
        )
    if not selectable:
        raise UploadInspectionError("No numeric data column was detected in the selected table.")
    missing_columns = tuple(column for column in selectable if np.any(np.isnan(numeric[column])))
    shape = (data_rows, len(columns))
    return UploadInspection(
        name=name,
        content_type=content_type,
        content=content,
        source_format=source_format,
        compressed=compressed,
        delimiter=delimiter,
        has_header=has_header,
        columns=columns,
        numeric_columns=selectable,
        default_column=_default_numeric_column(selectable),
        shape=shape,
        inferred_form=_infer_table_form(
            columns=columns,
            selectable=selectable,
            data_rows=data_rows,
            expected_form=expected_form,
            source_format=source_format,
        ),
        expected_form=expected_form,
        missing_columns=missing_columns,
        missing_count=sum(
            int(np.count_nonzero(np.isnan(numeric[column]))) for column in selectable
        ),
        track_kind=(
            "intervals" if presence_track else "peaks"
            if source_format == "bedgraph" and (
                name.casefold().removesuffix(".gz").endswith(".bed")
                or "enrichment_ratio" in columns
            ) else "signal" if source_format == "bedgraph" else None
        ),
        _table_data=_frozen_arrays(numeric),
    )


def _inspect_npy(
    *,
    name: str,
    content_type: str,
    content: bytes,
    payload: bytes,
    source_format: str,
    compressed: bool,
    expected_form: str,
) -> UploadInspection:
    _preflight_npy(payload)
    try:
        array = _validate_numeric_array(
            np.load(io.BytesIO(payload), allow_pickle=False),
            label=source_format.upper(),
            allow_missing=True,
        )
    except UploadInspectionError:
        raise
    except (OSError, ValueError) as exc:
        raise UploadInspectionError(f"The {source_format.upper()} file is invalid.") from exc
    shape = tuple(int(value) for value in array.shape)
    return UploadInspection(
        name=name,
        content_type=content_type,
        content=content,
        source_format=source_format,
        compressed=compressed,
        shape=shape,
        inferred_form=_infer_form(shape, expected_form, source_format),
        expected_form=expected_form,
        missing_count=int(np.count_nonzero(np.isnan(array))),
        _array_data=_frozen_arrays({"values": array}),
    )


def _inspect_npz(
    *,
    name: str,
    content_type: str,
    content: bytes,
    payload: bytes,
    compressed: bool,
    expected_form: str,
) -> UploadInspection:
    _archive_members(payload)
    numeric_arrays: dict[str, np.ndarray] = {}
    labels: tuple[str, ...] = ()
    metadata_form: str | None = None
    semantic_form: str | None = None
    region_names: tuple[str, ...] = ()
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            if "columns" in archive.files:
                raw_labels = np.asarray(archive["columns"])
                if raw_labels.ndim == 1 and raw_labels.size <= 10_000:
                    labels = _deduplicate_columns([str(value) for value in raw_labels.tolist()])
            if "data_form" in archive.files:
                raw_form = np.asarray(archive["data_form"]).reshape(-1)
                if raw_form.size:
                    metadata_form = str(raw_form[0])
            if "semantic_form" in archive.files:
                raw_semantic_form = np.asarray(archive["semantic_form"]).reshape(-1)
                if raw_semantic_form.size:
                    semantic_form = str(raw_semantic_form[0])
            if "region_names" in archive.files:
                raw_names = np.asarray(archive["region_names"]).reshape(-1)
                if raw_names.size > 100_000:
                    raise UploadTooLargeError("The NPZ archive contains too many region names.")
                region_names = tuple(str(value) for value in raw_names.tolist())
            for key in archive.files:
                if key in _NPZ_METADATA_KEYS:
                    continue
                try:
                    candidate = _validate_numeric_array(
                        archive[key],
                        label=f"NPZ array {key!r}",
                        allow_missing=True,
                    )
                except UploadInspectionError:
                    raise
                numeric_arrays[key] = candidate
    except UploadInspectionError:
        raise
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise UploadInspectionError("The NPZ archive is invalid.") from exc
    if not numeric_arrays:
        raise UploadInspectionError("The NPZ archive contains no numeric arrays.")
    keys = tuple(numeric_arrays)
    default_key = _preferred_array_key(keys)
    assert default_key is not None
    default_array = numeric_arrays[default_key]
    shape = tuple(int(value) for value in default_array.shape)
    columns = (
        labels
        if (
            default_array.ndim == 2
            and len(labels) == default_array.shape[1]
        )
        or (default_array.ndim == 1 and len(labels) == 1)
        else ()
    )
    selectable_columns = tuple(
        column
        for column in columns
        if not _is_coordinate_column(column) and not _is_metadata_column(column)
    )
    if not selectable_columns:
        selectable_columns = columns
    selected_expected = expected_form
    if expected_form == "auto":
        form_aliases = {
            "scalar": "scalar",
            "space": "x",
            "spatial": "x",
            "x": "x",
            "time": "t",
            "temporal": "t",
            "t": "t",
            "spacetime": "xt",
            "space_time": "xt",
            "xt": "xt",
            "table": "table",
            "regions": "regions",
        }
        selected_expected = form_aliases.get(
            str(semantic_form or metadata_form).casefold(),
            expected_form,
        )
    return UploadInspection(
        name=name,
        content_type=content_type,
        content=content,
        source_format="npz",
        compressed=compressed,
        columns=columns,
        numeric_columns=selectable_columns,
        default_column=_default_numeric_column(selectable_columns),
        array_keys=keys,
        default_array_key=default_key,
        shape=shape,
        inferred_form=_infer_form(shape, selected_expected, "npz"),
        expected_form=expected_form,
        missing_columns=tuple(
            column
            for index, column in enumerate(columns)
            if index < default_array.shape[1] and np.any(np.isnan(default_array[:, index]))
        )
        if default_array.ndim == 2
        else (),
        missing_count=int(np.count_nonzero(np.isnan(default_array))),
        semantic_form=semantic_form,
        region_names=region_names,
        _array_data=_frozen_arrays(numeric_arrays),
    )


@hosted_operation("Reading uploaded data")
def inspect_upload(
    name: str,
    content: bytes | bytearray | memoryview,
    *,
    content_type: str = "",
    expected_form: str = "auto",
    source_format: str = "auto",
) -> UploadInspection:
    """Inspect one uploaded file without writing it to disk.

    ``expected_form`` uses the public ``rdata``/``rmap`` form names
    (``scalar``, ``x``, ``t``, ``xt``, ``table`` or ``regions``). It guides
    metadata and later materialization but never bypasses shape or numeric
    validation. ``source_format`` is a narrow manual hint for extensionless
    BigWig or bedGraph sources; binary signatures still have to agree.
    """

    ensure_capacity(len(content) * 8, "Inspecting the uploaded file")
    safe_name = _safe_name(name)
    raw = bytes(content)
    if not raw:
        raise UploadInspectionError("The selected file is empty.")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise UploadTooLargeError(f"Files larger than {MAX_UPLOAD_BYTES // 1024**2} MiB cannot be kept in this local session.")
    source_format, payload, compressed = _source_payload(
        safe_name,
        raw,
        source_format=source_format,
    )
    if len(payload) > MAX_DECOMPRESSED_BYTES:
        raise UploadTooLargeError("The selected file exceeds the safe inspection limit.")
    common = {
        "name": safe_name,
        "content_type": str(content_type or ""),
        "content": raw,
        "payload": payload,
        "compressed": compressed,
        "expected_form": expected_form,
    }
    if source_format in {"csv", "tsv", "txt", "bedgraph"}:
        normalized = _geo_repliseq_payload(safe_name, payload) if source_format == "txt" else None
        if normalized is not None:
            return replace(
                _inspect_table(source_format="tsv", **{**common, "payload": normalized}),
                source_layout="geo_repliseq",
                _normalized_payload=normalized,
                warnings=(
                    "GEO Repli-seq matrix transposed; coordinates and S1-S16 signals are preserved.",
                ),
            )
        return _inspect_table(source_format=source_format, **common)
    if source_format in {"npy", "npv"}:
        return _inspect_npy(source_format=source_format, **common)
    if source_format == "npz":
        return _inspect_npz(**common)
    if source_format == "bigwig":
        return UploadInspection(
            name=safe_name,
            content_type=str(content_type or ""),
            content=raw,
            source_format=source_format,
            compressed=compressed,
            inferred_form="track",
            expected_form=expected_form,
            warnings=("Chromosomes are read when Map runs.",),
        )
    raise AssertionError(f"Unhandled upload format {source_format!r}.")


def _payload_for_inspection(inspection: UploadInspection) -> bytes:
    if inspection._normalized_payload is not None:
        return inspection._normalized_payload
    _, payload, _ = _source_payload(inspection.name, inspection.content)
    return payload


def _coerce_expected_form(values: np.ndarray, expected_form: str) -> np.ndarray:
    """Apply only the requested shape to an already validated numeric array."""

    form = expected_form.casefold()
    # ``_resolve_missing_values`` has already checked type, size, infinities,
    # and the selected missing-value policy. Repeating those full-array scans
    # here made large uploads noticeably slower each time Map was selected.
    array = np.asarray(values)
    if form == "scalar":
        if array.size != 1:
            raise UploadInspectionError("This route requires one constant numeric value.")
        return array.reshape(())
    if form == "x":
        if array.ndim == 2 and 1 in array.shape:
            array = array.reshape(-1)
        if array.ndim != 1:
            raise UploadInspectionError("This route requires a one-dimensional space profile.")
        return array
    if form == "t":
        if array.ndim == 2 and 1 in array.shape:
            array = array.reshape(-1)
        if array.ndim != 1:
            raise UploadInspectionError("This route requires one time series.")
        return array.reshape(1, -1)
    if form == "xt":
        if array.ndim != 2:
            raise UploadInspectionError("This route requires a position-by-time matrix.")
        return array
    if form == "regions":
        if array.ndim != 2 or array.shape[1] < 2:
            raise UploadInspectionError("Region data require at least start and end columns.")
        return array
    return array


def _normalize_missing_policy(policy: str) -> str:
    key = str(policy or "error").strip().casefold().replace("-", "_").replace(" ", "_")
    aliases = {
        "error": "error",
        "raise": "error",
        "interpolate": "interpolate",
        "linear": "interpolate",
        "zero": "zero",
        "fill_zero": "zero",
        "ignore": "ignore",
        "keep": "ignore",
        "preserve": "ignore",
    }
    if key not in aliases:
        raise UploadInspectionError(
            "Missing-value policy must be Error, Ignore, Interpolate, or Fill with zero."
        )
    return aliases[key]


def _interpolate_vector(values: np.ndarray, *, label: str) -> np.ndarray:
    finite = np.isfinite(values)
    if not np.any(finite):
        raise UploadInspectionError(f"{label} contains no values to interpolate from.")
    indices = np.arange(values.size, dtype=float)
    return np.interp(indices, indices[finite], values[finite])


def _resolve_missing_values(
    values: np.ndarray,
    *,
    policy: str,
    expected_form: str,
) -> np.ndarray:
    """Apply an explicit row-preserving missing-value policy."""

    array = _validate_numeric_array(values, allow_missing=True)
    if not np.any(np.isnan(array)):
        return array
    normalized = _normalize_missing_policy(policy)
    if normalized == "error":
        raise UploadInspectionError(
            "The selected data contain missing values. Choose Ignore, Interpolate, or Fill "
            "with zero, or correct the source file."
        )
    if normalized == "ignore":
        return array
    if normalized == "zero":
        return np.nan_to_num(array, nan=0.0, copy=True)
    if expected_form.casefold() == "regions":
        raise UploadInspectionError("Region coordinates cannot be interpolated.")

    result = np.array(array, dtype=float, copy=True)
    if result.ndim <= 1:
        return _interpolate_vector(result.reshape(-1), label="The selected profile").reshape(
            result.shape
        )
    if result.ndim != 2:
        raise UploadInspectionError("Only vectors and matrices can be interpolated.")
    if expected_form.casefold() == "t" or result.shape[0] == 1:
        result[0, :] = _interpolate_vector(result[0, :], label="The selected time series")
        return result

    # First interpolate spatially within each time slice. Entirely missing
    # slices are then filled temporally within each position, preserving the
    # original (X, T) grid and never dropping a row or column.
    for column in range(result.shape[1]):
        if np.any(np.isfinite(result[:, column])):
            result[:, column] = _interpolate_vector(
                result[:, column], label=f"Matrix column {column + 1}"
            )
    for row in range(result.shape[0]):
        if np.any(np.isnan(result[row, :])) and np.any(np.isfinite(result[row, :])):
            result[row, :] = _interpolate_vector(result[row, :], label=f"Matrix row {row + 1}")
    if np.any(np.isnan(result)):
        raise UploadInspectionError("The matrix contains no values from which to interpolate.")
    return result


def _materialize_table(
    inspection: UploadInspection,
    column: str | None,
    missing_policy: str,
) -> np.ndarray:
    numeric = inspection._table_data
    if numeric:
        columns = inspection.columns
    else:
        payload = _payload_for_inspection(inspection)
        _, _, columns, _, numeric = _stream_table_parts(
            payload,
            inspection.source_format,
            inspection.expected_form,
            source_name=inspection.name,
        )
    if not numeric:
        raise UploadInspectionError("No numeric data column was detected in the selected table.")
    if column is not None:
        if column not in inspection.numeric_columns:
            raise UploadInspectionError(
                f"Column {column!r} is not a selectable numeric signal column in the upload."
            )
        values = numeric[column]
    elif (
        inspection.expected_form.casefold() == "t"
        and inspection.shape
        and inspection.shape[0] == 1
        and len(inspection.numeric_columns) > 1
    ):
        # A one-row table follows rdata's temporal row-vector convention: its
        # numeric headers are samples along T, not alternative value columns.
        values = np.asarray(
            [numeric[name][0] for name in inspection.numeric_columns],
            dtype=float,
        )
    elif inspection.expected_form.casefold() in {"xt", "regions"}:
        selected = list(numeric)
        if inspection.expected_form.casefold() == "xt":
            non_coordinates = [
                name
                for name in selected
                if not _is_coordinate_column(name) and not _is_metadata_column(name)
            ]
            selected = non_coordinates or selected
        elif inspection.expected_form.casefold() == "regions":
            named_bounds = [
                name for name in selected if _normalized_column_name(name) in {"start", "end"}
            ]
            selected = named_bounds or selected
        lengths = {len(numeric[name]) for name in selected}
        if len(lengths) != 1:
            raise UploadInspectionError(
                "The selected numeric columns contain different numbers of missing values."
            )
        values = np.column_stack([numeric[name] for name in selected])
    else:
        selected = inspection.default_column
        assert selected is not None
        values = numeric[selected]
    values_array = np.asarray(values)
    if (
        inspection.expected_form.casefold() == "x"
        and values_array.ndim == 1
        and len(inspection.numeric_columns) > 1
        and all(
            re.fullmatch(r"chr(?:omosome)?\s*(?:\d+|x|y|m|mt)", _normalized_column_name(name))
            for name in inspection.numeric_columns
        )
    ):
        present = np.flatnonzero(~np.isnan(values_array))
        if present.size:
            # Only trailing padding in a wide chromosome table can be cropped.
            # Leading and internal empty bins keep their original positions.
            values_array = values_array[: present[-1] + 1]
    resolved = _resolve_missing_values(
        values_array,
        policy=missing_policy,
        expected_form=inspection.expected_form,
    )
    return _coerce_expected_form(resolved, inspection.expected_form)


def _materialize_npz(inspection: UploadInspection, array_key: str | None) -> np.ndarray:
    key = array_key or inspection.default_array_key
    if key is None:
        raise UploadInspectionError("Choose an array from the NPZ archive.")
    if key in inspection._array_data:
        return inspection._array_data[key]
    payload = _payload_for_inspection(inspection)
    _archive_members(payload)
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            if key not in archive.files:
                raise UploadInspectionError(f"Array {key!r} is not present in the NPZ archive.")
            values = _validate_numeric_array(archive[key], label=f"NPZ array {key!r}")
    except UploadInspectionError:
        raise
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise UploadInspectionError("The NPZ archive is invalid.") from exc
    return values


@hosted_operation("Reading uploaded data")
def materialize_upload(
    inspection: UploadInspection,
    *,
    column: str | None = None,
    array_key: str | None = None,
    missing_policy: str = "error",
) -> np.ndarray:
    """Materialize a previously inspected upload under a missing-value policy."""

    if inspection.source_format in {"csv", "tsv", "txt", "bedgraph"}:
        return _materialize_table(inspection, column, missing_policy)
    if inspection.source_format in {"npy", "npv"}:
        if "values" in inspection._array_data:
            values = inspection._array_data["values"]
        else:
            payload = _payload_for_inspection(inspection)
            try:
                values = np.load(io.BytesIO(payload), allow_pickle=False)
            except (OSError, ValueError) as exc:
                raise UploadInspectionError(
                    f"The {inspection.source_format.upper()} file is invalid."
                ) from exc
    elif inspection.source_format == "npz":
        values = _materialize_npz(inspection, array_key)
    elif inspection.source_format == "bigwig":
        raise UploadInspectionError(
            "BigWig data require chromosome and interval settings before they can be materialized."
        )
    else:
        raise UploadInspectionError(f"Unsupported upload format {inspection.source_format!r}.")

    array = _validate_numeric_array(values, allow_missing=True)
    selected_column = column
    if selected_column is None and (
        inspection.expected_form.casefold() in {"scalar", "x", "t"}
        or (
            inspection.expected_form.casefold() == "auto"
            and inspection.inferred_form not in {"xt", "regions"}
        )
    ):
        selected_column = inspection.default_column
    if selected_column is not None and inspection.columns and array.ndim == 2:
        if selected_column not in inspection.numeric_columns:
            raise UploadInspectionError(
                f"Column {selected_column!r} is not a selectable numeric signal column."
            )
        array = array[:, inspection.columns.index(selected_column)]
    resolved = _resolve_missing_values(
        array,
        policy=missing_policy,
        expected_form=inspection.expected_form,
    )
    return _coerce_expected_form(resolved, inspection.expected_form)


@hosted_operation("Reading uploaded data")
def materialize_many(
    inspections: Sequence[UploadInspection],
    *,
    columns: Sequence[str | None] | None = None,
    array_keys: Sequence[str | None] | None = None,
    missing_policy: str = "error",
) -> np.ndarray:
    """Stack ordered single-track uploads into an ``(X, T)`` array."""

    if not inspections:
        raise UploadInspectionError("Upload at least one data file.")
    selected_columns = tuple(columns or (None,) * len(inspections))
    selected_keys = tuple(array_keys or (None,) * len(inspections))
    if len(selected_columns) != len(inspections) or len(selected_keys) != len(inspections):
        raise ValueError("Column and array-key selections must match the number of uploads.")
    tracks: list[np.ndarray] = []
    for inspection, column, array_key in zip(inspections, selected_columns, selected_keys):
        # A group of one-dimensional uploads *becomes* an XT matrix only after
        # stacking.  Validate each member as an X profile first.
        profile_inspection = replace(inspection, expected_form="x")
        values = materialize_upload(
            profile_inspection,
            column=column,
            array_key=array_key,
            missing_policy=missing_policy,
        )
        if values.ndim == 2 and 1 in values.shape:
            values = values.reshape(-1)
        if values.ndim != 1:
            raise UploadInspectionError(
                f"{inspection.name!r} must contain one profile when uploading multiple tracks."
            )
        tracks.append(values)
    lengths = {track.size for track in tracks}
    if len(lengths) != 1:
        raise UploadInspectionError("All uploaded tracks must contain the same number of bins.")
    return _validate_numeric_array(np.column_stack(tracks), label="space-time upload")
