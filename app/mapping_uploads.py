"""Compatibility imports for :mod:`dnascape.workflows.uploads`.

The app and headless scripts share the same module instance, including
process-local locks, export preferences, and private helper overrides.
"""

from __future__ import annotations

import sys
from typing import Any

from nicegui import run

from dnascape.workflows import uploads as _workflow
from dnascape.workflows.uploads import (
    MAX_ARCHIVE_MEMBERS as MAX_ARCHIVE_MEMBERS,
    MAX_ARRAY_ELEMENTS as MAX_ARRAY_ELEMENTS,
    MAX_COMPRESSION_RATIO as MAX_COMPRESSION_RATIO,
    MAX_DECOMPRESSED_BYTES as MAX_DECOMPRESSED_BYTES,
    MAX_TABLE_CELLS as MAX_TABLE_CELLS,
    MAX_TABLE_COLUMNS as MAX_TABLE_COLUMNS,
    MAX_UPLOAD_BYTES as MAX_UPLOAD_BYTES,
    UploadInspection as UploadInspection,
    UploadInspectionError as UploadInspectionError,
    UploadTooLargeError as UploadTooLargeError,
    inspect_upload as inspect_upload,
    materialize_many as materialize_many,
    materialize_upload as materialize_upload,
)


async def inspect_upload_event(
    event: Any,
    *,
    expected_form: str = "auto",
    source_format: str = "auto",
) -> UploadInspection:
    """Read and inspect a NiceGUI upload event with a server-side size check."""

    upload = event.file
    size = int(upload.size())
    if size <= 0:
        raise _workflow.UploadInspectionError("The selected file is empty.")
    if size > _workflow.MAX_UPLOAD_BYTES:
        raise _workflow.UploadTooLargeError(f"Files larger than {_workflow.MAX_UPLOAD_BYTES // 1024**2} MiB cannot be kept in this local session.")
    content = await upload.read()
    if len(content) != size and len(content) > _workflow.MAX_UPLOAD_BYTES:
        raise _workflow.UploadTooLargeError(f"Files larger than {_workflow.MAX_UPLOAD_BYTES // 1024**2} MiB cannot be kept in this local session.")
    inspection = await _workflow.__dict__["run"].io_bound(
        _workflow.inspect_upload,
        upload.name,
        content,
        content_type=upload.content_type,
        expected_form=expected_form,
        source_format=source_format,
    )
    if inspection is None:  # NiceGUI returns None only while its executor is shutting down.
        raise RuntimeError("Upload inspection was interrupted while the app was stopping.")
    return inspection

_workflow.__dict__["inspect_upload_event"] = inspect_upload_event
_workflow.__dict__["run"] = run

sys.modules[__name__] = _workflow
