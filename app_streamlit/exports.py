"""Lazy single-format serialization for figures and scientific arrays."""

from __future__ import annotations

import gzip
import json
import re
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Mapping, Sequence

import numpy as np

FIGURE_MIME = {
    "png": "image/png",
    "pdf": "application/pdf",
    "svg": "image/svg+xml",
}
ARRAY_MIME = {
    "csv": "text/csv",
    "tsv": "text/tab-separated-values",
    "txt": "text/plain",
    "npy": "application/octet-stream",
    "npv": "application/octet-stream",
    "npz": "application/octet-stream",
}


@dataclass(frozen=True)
class ExportPayload:
    """One fully prepared download payload."""

    payload: bytes
    extension: str
    mime: str


def _format(value: str, allowed: Mapping[str, str]) -> str:
    resolved = str(value).strip().lower().lstrip(".")
    if resolved not in allowed:
        raise ValueError(f"Unsupported export format: {value!r}.")
    return resolved


def serialize_figure(
    figure: Any,
    image_format: str = "png",
    *,
    dpi: int = 180,
    transparent: bool = True,
) -> ExportPayload:
    """Serialize one requested figure format; no other formats are built."""

    image_format = _format(image_format, FIGURE_MIME)
    buffer = BytesIO()
    figure.savefig(
        buffer,
        format=image_format,
        bbox_inches="tight",
        dpi=int(dpi),
        transparent=bool(transparent),
        facecolor="none" if transparent else None,
        edgecolor="none" if transparent else None,
    )
    return ExportPayload(buffer.getvalue(), image_format, FIGURE_MIME[image_format])


def _safe_axis_key(name: object) -> str:
    key = re.sub(r"[^A-Za-z0-9_]+", "_", str(name)).strip("_")
    return f"axis_{key or 'values'}"


def _tabular_frame(
    values: np.ndarray,
    axes: Mapping[str, Any],
    labels: Sequence[str],
):
    import pandas as pd

    if values.ndim == 1:
        frame = pd.DataFrame({labels[0] if labels else "value": values})
    elif values.ndim == 2:
        names = list(labels) if len(labels) == values.shape[1] else [
            f"value_{index + 1}" for index in range(values.shape[1])
        ]
        frame = pd.DataFrame(values, columns=names)
    else:
        raise ValueError("CSV, TSV, and TXT exports support one- or two-dimensional arrays.")
    for name, axis in reversed(list(axes.items())):
        axis_values = np.asarray(axis).reshape(-1)
        if axis_values.size == len(frame):
            frame.insert(0, str(name), axis_values)
    return frame


def serialize_array(
    values: Any,
    output_format: str,
    *,
    axes: Mapping[str, Any] | None = None,
    labels: Sequence[str] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ExportPayload:
    """Serialize one array format, preserving axes/shape in NPZ output."""

    output_format = _format(output_format, ARRAY_MIME)
    array = np.asarray(values)
    axes = dict(axes or {})
    labels = tuple(str(label) for label in (labels or ()))
    metadata = dict(metadata or {})
    buffer = BytesIO()

    if output_format in {"npy", "npv"}:
        np.save(buffer, array, allow_pickle=False)
    elif output_format == "npz":
        fields: dict[str, Any] = {
            "values": array,
            "labels": np.asarray(labels, dtype=str),
            "metadata_json": np.asarray(
                json.dumps(metadata, sort_keys=True, default=str), dtype=str
            ),
        }
        for name, axis in axes.items():
            key = _safe_axis_key(name)
            if key in fields:
                raise ValueError(
                    f"Axis names collide after export-key sanitization: {name!r}."
                )
            fields[key] = np.asarray(axis)
        np.savez_compressed(buffer, **fields)
    elif output_format in {"csv", "tsv"}:
        separator = "," if output_format == "csv" else "\t"
        frame = _tabular_frame(array, axes, labels)
        buffer.write(frame.to_csv(index=False, sep=separator).encode("utf-8"))
    else:
        frame = _tabular_frame(array, axes, labels)
        np.savetxt(buffer, frame.to_numpy(), fmt="%.12g", delimiter="\t")

    return ExportPayload(buffer.getvalue(), output_format, ARRAY_MIME[output_format])


def gzip_payload(payload: bytes, extension: str) -> ExportPayload:
    """Compress a payload only when the caller explicitly requests it."""

    return ExportPayload(gzip.compress(payload), f"{extension}.gz", "application/gzip")


def array_export_target(
    output_format: str,
    *,
    compressed: bool = False,
) -> tuple[str, str]:
    """Return the extension and MIME type without serializing array data."""

    resolved = _format(output_format, ARRAY_MIME)
    if compressed:
        return f"{resolved}.gz", "application/gzip"
    return resolved, ARRAY_MIME[resolved]


def figure_export_target(image_format: str) -> tuple[str, str]:
    """Return the extension and MIME type without constructing a figure payload."""

    resolved = _format(image_format, FIGURE_MIME)
    return resolved, FIGURE_MIME[resolved]
