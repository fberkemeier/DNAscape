"""Bounded raw serializers for live mapping results."""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from dnascape._resources import ensure_capacity, hosted_operation
from dnascape.workflows.export_settings import normalize_compression
from dnascape.workflows.exports import (
    RAW_FORMATS,
    ExportArtifact,
    compressed_export_artifact,
    normalize_export_format,
    safe_export_filename,
)
from dnascape.workflows.mapping import MappingResult
from dnascape.workflows.result_naming import (
    RMAP_FILE_PREFIX,
    dependence_file_suffix,
    output_file_token,
)

MAX_TEXT_EXPORT_VALUES = 64_000_000
MAX_BINARY_EXPORT_BYTES = 512 * 1024 * 1024
MAX_TEXT_EXPORT_BYTES = 1024 * 1024 * 1024
_TEXT_BLOCK_VALUES = 262_144


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


def mapping_output_dependence_suffix(result: MappingResult, output_key: str) -> str:
    """Use the selected output's axes, including comparisons and input previews."""
    if output_key.startswith("comparison:"):
        key = output_key.removeprefix("comparison:")
        comparison = result.comparisons.get(key)
        return dependence_file_suffix(comparison.form if comparison is not None else result.output_forms.get(key, ""))
    from .preview_inputs import input_preview_result
    result = input_preview_result(result, output_key)
    return dependence_file_suffix(result.output_forms.get(output_key, ""))


def select_mapping_output(
    result: MappingResult,
    output_key: str,
) -> SelectedMappingOutput:
    """Select a numeric member and verify that its declared axes match."""

    key = str(output_key).strip()
    from .preview_inputs import input_preview_result
    result = input_preview_result(result, key)
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

    # Mapping results are immutable, including chromosome-scale disk-backed
    # fields. Keep those arrays mapped instead of copying the entire grid just
    # to download it. Mutable caller-owned arrays still need a frozen snapshot.
    if values.flags.writeable:
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


@hosted_operation("Exporting Mapping results")
def build_mapping_raw_export(
    result: MappingResult,
    output_key: str,
    file_format: str,
    *,
    filename: str | None = None,
    compression: str = "none",
    gzip_output: bool | None = None,
    filter_layout: str = 'positions',
) -> ExportArtifact:
    """Serialize one mapped member as CSV/TSV/TXT/NPZ.

    Shapes follow :func:`dnascape.rdata`: spatial data are ``(X,)``, temporal
    data are ``(1, T)``, and space-time data are ``(X, T)``.  Bundle targets
    are exported one selected numeric member at a time.
    """

    normalized_format = normalize_export_format(file_format, allowed=RAW_FORMATS)
    selected = select_mapping_output(result, output_key)
    ensure_capacity(selected.values.size * (100 if normalized_format != "npz" else 32),
                    "Preparing this Mapping download")
    if output_key == 'filter_positions':
        if filter_layout not in {'positions', 'intervals'}:
            raise ValueError('Choose a position list or genomic intervals for the mask.')
        values = (np.asarray(result.diagnostics.get('filter_intervals')) if filter_layout == 'intervals'
                  else selected.values)
        if values.size > MAX_TEXT_EXPORT_VALUES:
            raise ValueError('The mask export is too large for this format.')
        integers = np.rint(values).astype(np.int64)
        if normalized_format == 'npz':
            stream = io.BytesIO()
            np.savez_compressed(stream, **{filter_layout: integers})
            data = stream.getvalue()
        else:
            stream = io.StringIO()
            np.savetxt(stream, integers, fmt='%d', delimiter=',' if normalized_format == 'csv' else '\t')
            data = stream.getvalue().encode()
    elif normalized_format in {"csv", "tsv", "txt"}:
        if selected.values.size > MAX_TEXT_EXPORT_VALUES:
            raise ValueError(
                f"Text export contains {selected.values.size:,} values; the interactive "
                f"limit is {MAX_TEXT_EXPORT_VALUES:,}. Use NPZ instead."
            )
        data = _text_bytes(selected, normalized_format)
    else:
        estimated = selected.values.nbytes + selected.position_kb.nbytes + selected.time_min.nbytes
        if estimated > MAX_BINARY_EXPORT_BYTES:
            raise ValueError(
                f"Binary export would contain at least {estimated / 1024**2:.1f} MiB; "
                f"the local limit is {MAX_BINARY_EXPORT_BYTES / 1024**2:.0f} MiB."
            )
        data = _binary_bytes(selected)

    suffix = dependence_file_suffix(selected.semantic_form)
    token = output_file_token(selected.key)
    if suffix:
        token = token.removesuffix(f"-{suffix}")
    default_name = f"{RMAP_FILE_PREFIX}_{token}"
    if suffix:
        default_name += f"_{suffix}"
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
            "npz": "application/octet-stream",
        }[normalized_format],
        compression=selected_compression,
        # Prefer download latency over maximum compression for
        # large fraction fields. All values and the inner file remain identical.
        compression_level=(
            1 if selected.key == "replicated_fraction_xt" and selected.values.size >= 1_000_000 else 6
        ),
    )


@hosted_operation("Exporting Mapping comparisons")
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
    ensure_capacity(observed.size * 256, "Preparing the fit comparison download")
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
                f"is {MAX_TEXT_EXPORT_VALUES:,}. Use NPZ instead."
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
    suffix = dependence_file_suffix(comparison.form)
    if suffix:
        default_name += f"_{suffix}"
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
    class BoundedBuffer(io.BytesIO):
        def write(self, data):
            ensure_capacity(len(data) + self.tell(), "Formatting this Mapping download")
            if self.tell() + len(data) > MAX_TEXT_EXPORT_BYTES:
                raise ValueError("Text export exceeds 1 GiB. Use NPZ for this dataset.")
            return super().write(data)
    binary = BoundedBuffer()
    stream = io.TextIOWrapper(binary, encoding="utf-8", newline="")
    if file_format == "txt":
        np.savetxt(stream, np.atleast_1d(values), fmt="%.18e")
        stream.flush()
        data = binary.getvalue()
        stream.detach()
        return data

    delimiter = "," if file_format == "csv" else "\t"
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
        _write_numeric_rows(stream, values, delimiter)
    stream.flush()
    data = binary.getvalue()
    stream.detach()
    return data


def _write_numeric_rows(stream: io.TextIOWrapper, values: np.ndarray, delimiter: str) -> None:
    """Write bounded blocks using Python's exact, shortest float representation.

    Boxing a NumPy block in C and formatting a whole row avoids a Python call
    for every cell. Numeric fields require no CSV quoting. Keep the existing
    spelling of integral values (including negative zero) and scientific notation.
    """
    row_format = delimiter.join(["%s"] * values.shape[1]) + "\n"
    rows_per_block = max(1, _TEXT_BLOCK_VALUES // values.shape[1])
    for start in range(0, len(values), rows_per_block):
        block = values[start:start + rows_per_block]
        objects = block.astype(object)
        # repr(float) uses fixed notation below 1e16. Only those integral
        # values previously lost their trailing '.0' in _number_text.
        integral = np.isfinite(block) & (np.abs(block) < 1e16) & (block == np.trunc(block))
        objects[integral] = block[integral].astype(np.int64)
        objects[(block == 0) & np.signbit(block)] = "-0"
        stream.writelines(row_format % tuple(row) for row in objects.tolist())


def _binary_bytes(selected: SelectedMappingOutput) -> bytes:
    values = _rdata_values(selected)
    stream = io.BytesIO()
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
    "mapping_output_dependence_suffix",
    "select_mapping_output",
]
