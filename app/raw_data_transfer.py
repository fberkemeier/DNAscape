"""Move generated data into a page's normal upload controls, within one client."""

from __future__ import annotations

import io
import zipfile
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import PurePath

from nicegui import ui
from app import background as run

from app.mapping_uploads import MAX_ARCHIVE_MEMBERS, MAX_DECOMPRESSED_BYTES, MAX_UPLOAD_BYTES
from app.notifications import notify
from app.session_cache import current_session_cache
from app.simulation_exports import ExportArtifact

RAW_DATA_PAGES = (
    ("data-management", "Data Manager"),
    ("mapping", "Mapping"),
    ("simulation", "Simulation"),
    ("plotting", "Analysis"),
)


@dataclass(frozen=True)
class RawInputTarget:
    key: str
    label: str
    receive: Callable[[tuple[ExportArtifact, ...]], Awaitable[None]]
    kind: str = "data"


def unpack_raw_artifact(artifact: ExportArtifact) -> tuple[ExportArtifact, ...]:
    """Expand a generated ZIP's numeric files; an NPZ remains one input file."""
    if not artifact.filename.casefold().endswith(".zip"):
        return (artifact,)
    with zipfile.ZipFile(io.BytesIO(artifact.data)) as archive:
        members = [item for item in archive.infolist() if not item.is_dir()]
        if len(members) > MAX_ARCHIVE_MEMBERS or sum(item.file_size for item in members) > MAX_DECOMPRESSED_BYTES:
            raise ValueError("The data archive is too large to open in this local session.")
        files = []
        for item in members:
            name = PurePath(item.filename.replace("\\", "/")).name
            if not name.casefold().endswith((".tsv", ".txt", ".csv", ".npz", ".tsv.gz", ".txt.gz", ".csv.gz", ".npz.gz")):
                continue
            if item.file_size > MAX_UPLOAD_BYTES:
                raise ValueError("A profile is too large to open in this local session.")
            files.append(ExportArtifact(data=archive.read(item), filename=name, media_type="application/octet-stream"))
    if not files:
        raise ValueError("The archive contains no supported profile files.")
    return tuple(files)


async def open_raw_artifact(
    artifact: ExportArtifact, destination: str, navigate: Callable[[str], None], *, preferred_input: str | None = None,
) -> None:
    """Navigate, choose an input if ambiguous, and append through upload inspection."""
    try:
        if destination not in dict(RAW_DATA_PAGES):
            raise ValueError("Choose a data page to open this file.")
        files = await run.io_bound(unpack_raw_artifact, artifact)
        # Navigation may delete the source page and its menu. Keep new UI and
        # callbacks rooted in the client rather than that disposable slot.
        with ui.context.client:
            navigate(destination)
            kind = "mask" if preferred_input == "filter_positions" else "data"
            targets = tuple(item for item in current_session_cache().input_targets(destination) if item.kind == kind)
            if not targets:
                raise ValueError("The selected page has no available file input.")
            preferred_input = preferred_input.removesuffix("_xt") if preferred_input else None
            target = next((item for item in targets if item.key == preferred_input), None)
            if target is None and len(targets) == 1:
                target = targets[0]
            if target is None:
                with ui.dialog() as dialog, ui.card().classes("dn-settings-dialog dn-raw-open-dialog"):
                    with ui.row().classes("dn-settings-dialog-head"):
                        ui.label("Choose region transform" if kind == "mask" else "Choose input").classes("dn-panel-title")
                        ui.space()
                        ui.button(icon="close", on_click=lambda: dialog.submit(None)).props('flat round dense aria-label="Cancel opening data"')
                    with ui.column().classes("dn-settings-section"):
                        ui.label(f"Add {'this file' if len(files) == 1 else 'these files'} to:").classes("dn-field-heading")
                        for item in targets:
                            ui.button(item.label, on_click=lambda _event, key=item.key: dialog.submit(key)).props("flat no-caps").classes("dn-raw-open-input").mark(f"raw-open-input-{item.key}")
                try:
                    selected = await dialog
                finally:
                    dialog.delete()
                target = next((item for item in targets if item.key == selected), None)
            if target is not None:
                await target.receive(files)
    except (OSError, RuntimeError, TypeError, ValueError, zipfile.BadZipFile) as exc:
        notify(str(exc), type="negative")
