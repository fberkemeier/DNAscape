"""Shared, deterministic archives for groups of generated app artifacts."""

from __future__ import annotations

import io
import zipfile
from collections.abc import Iterable
from pathlib import PurePath

from app.simulation_exports import ExportArtifact

DEFAULT_BATCH_ARCHIVE_SOURCE_BYTES = 256 * 1024 * 1024


def build_artifact_archive(
    artifacts: Iterable[ExportArtifact],
    *,
    filename: str = "results.zip",
    max_source_bytes: int | None = None,
    compress_all: bool = False,
) -> ExportArtifact:
    """Bundle generated artifacts into one safe, deterministic ZIP download.

    Only basenames are retained, duplicates receive stable numeric suffixes,
    and archive timestamps are fixed so the same inputs produce the same ZIP.
    Already-compressed payloads are stored directly unless ``compress_all``
    is requested (for example, for uncompressed NPZ members).
    """

    used_names: set[str] = set()
    artifact_count = 0
    source_bytes = 0
    output = io.BytesIO()
    with zipfile.ZipFile(output, mode="w") as archive:
        for artifact in artifacts:
            artifact_count += 1
            source_bytes += len(artifact.data)
            if max_source_bytes is not None and source_bytes > max_source_bytes:
                maximum_mib = max_source_bytes / 1024**2
                raise ValueError(
                    "The bulk download would contain more than "
                    f"{maximum_mib:g} MiB of raw data. Download the files "
                    "individually or reduce the batch."
                )
            basename = (
                PurePath(str(artifact.filename).replace("\\", "/")).name.replace("\x00", "").strip()
                or "result"
            )
            suffixes = "".join(PurePath(basename).suffixes)
            stem = basename[: -len(suffixes)] if suffixes else basename
            candidate = basename
            duplicate = 2
            while candidate.casefold() in used_names:
                candidate = f"{stem}_{duplicate}{suffixes}"
                duplicate += 1
            used_names.add(candidate.casefold())

            compression = (
                zipfile.ZIP_STORED
                if not compress_all and candidate.casefold().endswith((".gz", ".npz", ".zip"))
                else zipfile.ZIP_DEFLATED
            )
            info = zipfile.ZipInfo(candidate, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = compression
            info.external_attr = 0o600 << 16
            archive.writestr(info, artifact.data, compress_type=compression)

    if artifact_count == 0:
        raise ValueError("There are no generated files to download.")

    safe_archive_name = (
        PurePath(str(filename).replace("\\", "/")).name.replace("\x00", "").strip()
        or "results.zip"
    )
    if not safe_archive_name.casefold().endswith(".zip"):
        safe_archive_name += ".zip"
    return ExportArtifact(
        data=output.getvalue(),
        filename=safe_archive_name,
        media_type="application/zip",
    )
