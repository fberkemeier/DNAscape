"""Bounded ``rdata`` conversions for the local Data Manager."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, replace
from pathlib import Path, PurePath
from typing import Any

import numpy as np

from app.mapping_uploads import MAX_DECOMPRESSED_BYTES, UploadInspection, materialize_upload
from app.simulation_exports import ExportArtifact, safe_export_filename
from dnascape.io import inspect_genomic_track, rdata

_TARGET_FORMATS = frozenset({"npy", "npv", "npz", "txt", "tsv", "csv"})
_OUTPUT_FORMS = frozenset({"scalar", "space", "time", "spacetime", "table"})
_INPUT_FORMS = frozenset({"scalar", "x", "t", "xt", "table", "track"})
_FORM_TO_RDATA = {
    "scalar": "scalar",
    "x": "space",
    "t": "time",
    "xt": "spacetime",
    "table": "table",
    "track": "space",
}
_MEDIA_TYPES = {
    "npy": "application/octet-stream",
    "npv": "application/octet-stream",
    "npz": "application/zip",
    "txt": "text/plain; charset=utf-8",
    "tsv": "text/tab-separated-values; charset=utf-8",
    "csv": "text/csv; charset=utf-8",
}
_MAX_CONVERTED_BYTES = 128 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class TrackRegion:
    """One chromosome and its available half-open genomic bounds."""

    chromosome: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class TrackInspection:
    """Chromosome metadata read by :func:`dnascape.inspect_genomic_track`."""

    source_format: str
    regions: tuple[TrackRegion, ...]


@dataclass(frozen=True, slots=True)
class DataConversionRequest:
    """A fully resolved Data Manager conversion request."""

    inspection: UploadInspection
    input_form: str
    output_form: str
    target_format: str
    output_name: str = "dnascape_data"
    gzip_output: bool = True
    value_column: str | None = None
    array_key: str | None = None
    missing_policy: str = "error"
    decimal_places: int | None = None
    value_name: str = "value"
    chromosome: str | None = None
    start: int | None = None
    end: int | None = None
    bin_size: int = 1_000
    aggregation: str = "mean"
    track_missing: str = "nan"
    fill_value: float = 0.0
    interpolation: str = "none"
    smooth_window: int = 1
    normalize: str = "none"


@dataclass(frozen=True, slots=True)
class DataConversionResult:
    """Converted bytes and concise output metadata."""

    artifact: ExportArtifact
    source_format: str
    target_format: str
    data_form: str
    shape: tuple[int, ...]
    labels: tuple[str, ...]


def _safe_source_name(name: str) -> str:
    safe_name = PurePath(str(name or "upload").replace("\\", "/")).name
    return safe_name or "upload"


def _write_source(directory: Path, inspection: UploadInspection) -> Path:
    source_directory = directory / "source"
    source_directory.mkdir(exist_ok=True)
    source_path = source_directory / _safe_source_name(inspection.name)
    source_path.write_bytes(inspection.content)
    return source_path


def _normalized_request(request: DataConversionRequest) -> DataConversionRequest:
    input_form = str(request.input_form).strip().casefold()
    output_form = str(request.output_form).strip().casefold().replace("-", "")
    output_form = "spacetime" if output_form in {"xt", "spacetime"} else output_form
    output_form = {"x": "space", "t": "time"}.get(output_form, output_form)
    target_format = str(request.target_format).strip().casefold().lstrip(".")
    if input_form not in _INPUT_FORMS:
        raise ValueError(f"Unsupported input dependence {request.input_form!r}.")
    if output_form not in _OUTPUT_FORMS:
        raise ValueError(f"Unsupported output dependence {request.output_form!r}.")
    if target_format not in _TARGET_FORMATS:
        raise ValueError(f"Unsupported output format {request.target_format!r}.")
    if request.decimal_places is not None and int(request.decimal_places) < 0:
        raise ValueError("Decimal places must be zero or greater.")
    if not str(request.value_name or "").strip():
        raise ValueError("Value name cannot be empty.")
    return replace(
        request,
        input_form=input_form,
        output_form=output_form,
        target_format=target_format,
        decimal_places=(
            None if request.decimal_places is None else int(request.decimal_places)
        ),
        value_name=str(request.value_name).strip(),
    )


def _source_labels(request: DataConversionRequest, values: np.ndarray) -> list[str] | None:
    inspection = request.inspection
    if request.value_column:
        return [request.value_column]
    if request.input_form == "xt":
        columns = list(inspection.numeric_columns)
        if len(columns) == values.shape[1]:
            return columns
        if inspection.columns and len(inspection.columns) == values.shape[1]:
            return list(inspection.columns)
    if request.array_key and request.output_form == "space":
        return [request.array_key]
    return None


def _plain_source(request: DataConversionRequest) -> tuple[Any, list[str] | None]:
    expected_form = request.input_form
    values = np.asarray(
        materialize_upload(
            replace(request.inspection, expected_form=expected_form),
            column=request.value_column,
            array_key=request.array_key,
            missing_policy=request.missing_policy,
        )
    )
    return values, _source_labels(request, values)


def convert_data(request: DataConversionRequest) -> DataConversionResult:
    """Run one bounded conversion through the public :func:`dnascape.rdata`."""

    request = _normalized_request(request)
    inspection = request.inspection
    if len(inspection.content) > MAX_DECOMPRESSED_BYTES:
        raise ValueError("The uploaded source exceeds the local conversion limit.")

    filename = safe_export_filename(
        request.output_name,
        request.target_format,
        default_stem="dnascape_data",
        gzip_output=request.gzip_output,
    )
    with tempfile.TemporaryDirectory(prefix="dnascape-data-") as temporary:
        directory = Path(temporary)
        output_path = directory / filename
        labels: list[str] | None = None

        if request.input_form in {"track", "table"}:
            source: Any = _write_source(directory, inspection)
            source_format = inspection.source_format
        else:
            source, labels = _plain_source(request)
            source_format = "auto"

        kwargs: dict[str, Any] = {
            "source_format": source_format,
            "data_form": request.output_form,
            "labels": labels,
            "decimal_places": request.decimal_places,
            "value_name": request.value_name,
            "return_data": True,
        }
        if request.input_form == "track":
            kwargs.update(
                {
                    "chromosome": request.chromosome,
                    "value_column": request.value_column,
                    "start": request.start,
                    "end": request.end,
                    "bin_size": request.bin_size,
                    "aggregation": request.aggregation,
                    "missing": request.track_missing,
                    "fill_value": request.fill_value,
                    "interpolation": request.interpolation,
                    "smooth_window": request.smooth_window,
                    "normalize": request.normalize,
                }
            )

        raw_result = rdata(
            source,
            target=request.target_format,
            output_path=output_path,
            **kwargs,
        )
        if not isinstance(raw_result, dict):
            raise RuntimeError("rdata stopped before returning conversion metadata.")
        artifact_bytes = output_path.read_bytes()

    if not artifact_bytes:
        raise RuntimeError("rdata created an empty output file.")
    if len(artifact_bytes) > _MAX_CONVERTED_BYTES:
        raise ValueError("The converted file exceeds the 128 MB download limit.")

    raw_values = np.asarray(raw_result["values"])
    return DataConversionResult(
        artifact=ExportArtifact(
            data=artifact_bytes,
            filename=filename,
            media_type=(
                "application/gzip"
                if request.gzip_output
                else _MEDIA_TYPES[request.target_format]
            ),
        ),
        source_format=str(raw_result.get("source_format") or inspection.source_format),
        target_format=request.target_format,
        data_form=str(raw_result["data_form"]),
        shape=tuple(int(value) for value in raw_values.shape),
        labels=tuple(str(value) for value in (raw_result.get("labels") or ())),
    )


def inspect_track(inspection: UploadInspection) -> TrackInspection:
    """Read chromosome bounds from one retained BigWig or bedGraph upload."""

    if inspection.source_format not in {"bigwig", "bedgraph"}:
        raise ValueError("Chromosome inspection requires a BigWig or bedGraph source.")
    with tempfile.TemporaryDirectory(prefix="dnascape-track-") as temporary:
        source_path = _write_source(Path(temporary), inspection)
        summary = inspect_genomic_track(
            source_path,
            source_format=inspection.source_format,
            value_column=inspection.default_column if inspection.has_header else None,
        )
    table = summary["table"]
    starts = {
        str(row.chrom): int(row.start)
        for row in table[["chrom", "start"]].itertuples(index=False)
    }
    regions = tuple(
        TrackRegion(
            chromosome=str(chromosome),
            start=starts.get(str(chromosome), 0),
            end=int(summary["chromosomes"][chromosome]),
        )
        for chromosome in summary["chromosome_names"]
    )
    if not regions:
        raise ValueError("The genomic track contains no chromosomes.")
    return TrackInspection(source_format=str(summary["format"]), regions=regions)
