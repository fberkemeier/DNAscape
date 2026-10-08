"""Bounded raw serializers for live mapping results."""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from dnascape.workflows.export_settings import normalize_compression
from dnascape.workflows.mapping import MappingResult
from dnascape.workflows.result_naming import RMAP_FILE_PREFIX, output_file_token
from dnascape.workflows.exports import (
    RAW_FORMATS,
    ExportArtifact,
    compressed_export_artifact,
    normalize_export_format,
    safe_export_filename,
)

MAX_TEXT_EXPORT_VALUES = 1_000_000
MAX_BINARY_EXPORT_BYTES = 128 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class SelectedMappingOutput:
    """One validated mapping member in DNAscape's native axis order."""

    key: str
    values: np.ndarray
    data_form: str
    semantic_form: str
    position_kb: np.ndarray
    time_min: np.ndarray
    region_labels: tuple[str, ...]


def select_mapping_output(
    result: MappingResult,
    output_key: str,
) -> SelectedMappingOutput:
    """Select a numeric member and verify that its declared axes match."""

    key = str(output_key).strip()
    value = result.outputs.get(key)
    if value is None:
        raise ValueError(f"Mapping output {key!r} is not available.")
    values = np.asarray(value, dtype=float)
    if values.size == 0 or values.ndim not in {1, 2}:
        raise ValueError("Only non-empty scalar, profile, or space-time outputs can be exported.")
    form = result.output_forms.get(key)
    if form is None:
        raise ValueError(f"Mapping output {key!r} has no declared data form.")

    position = np.asarray(result.x_axis, dtype=float).reshape(-1)
    time_axis = np.asarray(result.time_axis, dtype=float).reshape(-1)
    if form == "xt":
        if values.ndim != 2:
            raise ValueError("Space-time output must be a two-dimensional matrix.")
        if values.shape != (position.size, time_axis.size):
            raise ValueError("Space-time output shape must match the position and time axes.")
        data_form = "spacetime"
    elif form == "t":
        values = values.reshape(-1)
        if values.size != time_axis.size:
            raise ValueError("Time output length must match the time axis.")
        data_form = "time"
    elif form == "scalar":
        values = values.reshape(-1)
        if values.size != 1:
            raise ValueError("Constant output must contain one value.")
        data_form = "scalar"
    elif form == "x":
        values = values.reshape(-1)
        if values.size != position.size:
            raise ValueError("Spatial output length must match the position axis.")
        data_form = "space"
    elif form == "regions":
        values = values.reshape(-1)
        if result.region_labels and len(result.region_labels) != values.size:
            raise ValueError("Regional output length must match its region labels.")
        # ``rdata`` deliberately has no regional/ensemble storage form.  Keep
        # these one-dimensional outputs loadable as ordinary spatial vectors
        # and retain their richer application meaning as separate metadata.
        data_form = "space"
    else:
        values = values.reshape(-1)
        data_form = "space"

    values = np.array(values, dtype=float, copy=True)
    values.setflags(write=False)
    return SelectedMappingOutput(
        key=key,
        values=values,
        data_form=data_form,
        semantic_form=str(form),
        position_kb=position,
        time_min=time_axis,
        region_labels=result.region_labels,
    )


def build_mapping_raw_export(
    result: MappingResult,
    output_key: str,
    file_format: str,
    *,
    filename: str | None = None,
    compression: str = "none",
    gzip_output: bool | None = None,
) -> ExportArtifact:
    """Serialize one mapped member as CSV/TSV/TXT/NPY/NPV/NPZ.

    Shapes follow :func:`dnascape.rdata`: spatial data are ``(X,)``, temporal
    data are ``(1, T)``, and space-time data are ``(X, T)``.  Bundle targets
    are exported one selected numeric member at a time.
    """

    normalized_format = normalize_export_format(file_format, allowed=RAW_FORMATS)
    selected = select_mapping_output(result, output_key)
    if normalized_format in {"csv", "tsv", "txt"}:
        if selected.values.size > MAX_TEXT_EXPORT_VALUES:
            raise ValueError(
                f"Text export contains {selected.values.size:,} values; the interactive "
                f"limit is {MAX_TEXT_EXPORT_VALUES:,}. Use NPY or NPZ instead."
            )
        data = _text_bytes(selected, normalized_format)
    else:
        estimated = selected.values.nbytes + selected.position_kb.nbytes + selected.time_min.nbytes
        if estimated > MAX_BINARY_EXPORT_BYTES:
            raise ValueError(
                f"Binary export would contain at least {estimated / 1024**2:.1f} MiB; "
                f"the local limit is {MAX_BINARY_EXPORT_BYTES / 1024**2:.0f} MiB."
            )
        data = _binary_bytes(selected, normalized_format)

    default_name = f"{RMAP_FILE_PREFIX}_{output_file_token(selected.key)}"
    base_name = filename or default_name
    inner_filename = safe_export_filename(
        base_name,
        normalized_format,
        default_stem=default_name,
    )
    selected_compression = (
        "gzip"
        if gzip_output
        else "none"
        if gzip_output is not None
        else normalize_compression(compression)
    )
    return compressed_export_artifact(
        data=data,
        filename=inner_filename,
        media_type={
            "csv": "text/csv; charset=utf-8",
            "tsv": "text/tab-separated-values; charset=utf-8",
            "txt": "text/plain; charset=utf-8",
            "npy": "application/octet-stream",
            "npv": "application/octet-stream",
            "npz": "application/octet-stream",
        }[normalized_format],
        compression=selected_compression,
    )


def build_mapping_comparison_raw_export(
    result: MappingResult,
    fitted_key: str,
    file_format: str,
    *,
    filename: str | None = None,
    compression: str = "none",
    gzip_output: bool | None = None,
) -> ExportArtifact:
    """Serialize one observed-versus-fitted profile with a shared axis."""

    comparison = result.comparisons.get(str(fitted_key))
    if comparison is None:
        raise ValueError(f"Fit comparison {fitted_key!r} is not available.")
    observed = np.asarray(comparison.observed, dtype=float).reshape(-1)
    fitted = np.asarray(comparison.fitted, dtype=float).reshape(-1)
    if observed.size != fitted.size or observed.size == 0:
        raise ValueError("Observed and fitted profiles must have the same non-zero length.")
    if comparison.form == "x" and result.x_axis.size == observed.size:
        axis_name = "position_kb"
        axis = np.asarray(result.x_axis, dtype=float)
    elif comparison.form == "t" and result.time_axis.size == observed.size:
        axis_name = "time_min"
        axis = np.asarray(result.time_axis, dtype=float)
    else:
        axis_name = "index"
        axis = np.arange(observed.size, dtype=float)

    normalized_format = normalize_export_format(file_format, allowed=RAW_FORMATS)
    matrix = np.column_stack((axis, observed, fitted))
    if normalized_format in {"csv", "tsv", "txt"}:
        if matrix.size > MAX_TEXT_EXPORT_VALUES:
            raise ValueError(
                f"Comparison export contains {matrix.size:,} values; the interactive limit "
                f"is {MAX_TEXT_EXPORT_VALUES:,}. Use NPY or NPZ instead."
            )
        stream: io.StringIO | io.BytesIO = io.StringIO(newline="")
        if normalized_format == "txt":
            np.savetxt(stream, matrix, fmt="%.18e")
        else:
            writer = csv.writer(
                stream,
                delimiter="," if normalized_format == "csv" else "\t",
                lineterminator="\n",
            )
            writer.writerow((axis_name, "observed", "fitted"))
            writer.writerows(
                (
                    _number_text(axis_value),
                    _number_text(observed_value),
                    _number_text(fitted_value),
                )
                for axis_value, observed_value, fitted_value in matrix
            )
        data = cast(io.StringIO, stream).getvalue().encode("utf-8")
    else:
        estimated = matrix.nbytes
        if estimated > MAX_BINARY_EXPORT_BYTES:
            raise ValueError(
                f"Binary comparison export would contain {estimated / 1024**2:.1f} MiB; "
                f"the local limit is {MAX_BINARY_EXPORT_BYTES / 1024**2:.0f} MiB."
            )
        binary = io.BytesIO()
        if normalized_format in {"npy", "npv"}:
            np.save(binary, matrix, allow_pickle=False)
        else:
            np.savez(
                binary,
                axis=axis,
                axis_name=np.asarray(axis_name),
                observed=observed,
                fitted=fitted,
                source_key=np.asarray(comparison.source_key),
                fitted_key=np.asarray(comparison.fitted_key),
            )
        data = binary.getvalue()

    default_name = f"{RMAP_FILE_PREFIX}_{output_file_token(comparison.source_key)}-cmp"
    base_name = filename or default_name
    inner_filename = safe_export_filename(
        base_name,
        normalized_format,
        default_stem=default_name,
    )
    selected_compression = (
        "gzip"
        if gzip_output
        else "none"
        if gzip_output is not None
        else normalize_compression(compression)
    )
    return compressed_export_artifact(
        data=data,
        filename=inner_filename,
        media_type={
            "csv": "text/csv; charset=utf-8",
            "tsv": "text/tab-separated-values; charset=utf-8",
            "txt": "text/plain; charset=utf-8",
            "npy": "application/octet-stream",
            "npv": "application/octet-stream",
            "npz": "application/octet-stream",
        }[normalized_format],
        compression=selected_compression,
    )


def _rdata_values(selected: SelectedMappingOutput) -> np.ndarray:
    if selected.data_form == "scalar":
        return selected.values.reshape(())
    if selected.data_form == "time":
        return selected.values.reshape(1, -1)
    return selected.values


def _text_bytes(selected: SelectedMappingOutput, file_format: str) -> bytes:
    values = _rdata_values(selected)
    if file_format == "txt":
        stream = io.StringIO(newline="")
        np.savetxt(stream, np.atleast_1d(values), fmt="%.18e")
        return stream.getvalue().encode("utf-8")

    delimiter = "," if file_format == "csv" else "\t"
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, delimiter=delimiter, lineterminator="\n")
    if selected.semantic_form == "regions" and selected.region_labels:
        writer.writerow(("region", _column_name(selected.key)))
        writer.writerows(
            (label, _number_text(value))
            for label, value in zip(selected.region_labels, values.reshape(-1), strict=True)
        )
    elif values.ndim == 0:
        writer.writerow((_column_name(selected.key),))
        writer.writerow((_number_text(float(values)),))
    elif values.ndim == 1:
        writer.writerow((_column_name(selected.key),))
        writer.writerows((_number_text(value),) for value in values)
    else:
        writer.writerow(tuple(f"t{index}" for index in range(values.shape[1])))
        writer.writerows(tuple(_number_text(value) for value in row) for row in values)
    return stream.getvalue().encode("utf-8")


def _binary_bytes(selected: SelectedMappingOutput, file_format: str) -> bytes:
    values = _rdata_values(selected)
    stream = io.BytesIO()
    if file_format in {"npy", "npv"}:
        np.save(stream, values, allow_pickle=False)
        return stream.getvalue()

    payload: dict[str, np.ndarray] = {
        "values": values,
        "data_form": np.asarray(selected.data_form),
        "semantic_form": np.asarray(selected.semantic_form),
    }
    if selected.key == "fit_history":
        payload["iteration"] = np.arange(1, values.size + 1, dtype=int)
    elif selected.data_form in {"space", "spacetime"} and selected.position_kb.size:
        payload["position_kb"] = selected.position_kb
    if selected.data_form in {"time", "spacetime"}:
        payload["time_min"] = selected.time_min
        payload["columns"] = np.asarray(
            tuple(f"t{index}" for index in range(selected.time_min.size))
        )
    if selected.semantic_form == "regions" and selected.region_labels:
        payload["region_names"] = np.asarray(selected.region_labels)
    if selected.semantic_form == "ensemble":
        payload["replicate"] = np.arange(1, values.size + 1, dtype=int)
    np.savez(stream, **cast(Any, payload))
    return stream.getvalue()


def _number_text(value: float) -> str:
    text = repr(float(value))
    return text[:-2] if text.endswith(".0") else text


def _column_name(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_") or "value"
    return "mse" if normalized == "fit_history" else normalized


__all__ = [
    "MAX_BINARY_EXPORT_BYTES",
    "MAX_TEXT_EXPORT_VALUES",
    "SelectedMappingOutput",
    "build_mapping_raw_export",
    "select_mapping_output",
]
