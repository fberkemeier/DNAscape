"""Bounded, browser-session-local inventory for DNAscape files and results."""

from __future__ import annotations

import dataclasses
import hashlib
import time
import uuid
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import PurePath
from typing import Any, Literal

import numpy as np
from nicegui import context

from app.simulation_exports import ExportArtifact

CacheCategory = Literal["generated", "uploads", "datasets", "workflows"]
CacheKind = Literal["plot", "raw", "upload", "dataset", "workflow"]
CacheSort = Literal["newest", "oldest", "largest", "smallest", "name"]

CACHE_CATEGORIES: tuple[CacheCategory, ...] = (
    "generated",
    "uploads",
    "datasets",
    "workflows",
)
DEFAULT_MAX_ITEMS = 30
DEFAULT_MAX_TOTAL_ITEMS = 80
DEFAULT_MAX_TOTAL_BYTES = 256 * 1024 * 1024
MAX_ITEMS = 200
MAX_TOTAL_ITEMS = 500
MAX_TOTAL_BYTES = 2 * 1024 * 1024 * 1024
MAX_SESSION_CACHES = 32


def _safe_filename(value: str) -> str:
    name = PurePath(str(value or "file").replace("\\", "/")).name
    return name or "file"


def _artifact_digest(artifact: ExportArtifact) -> str:
    digest = hashlib.sha256()
    digest.update(artifact.filename.encode("utf-8", errors="replace"))
    digest.update(b"\0")
    digest.update(artifact.data)
    return digest.hexdigest()


def _shape_text(shape: Any) -> str:
    try:
        values = tuple(int(value) for value in shape)
    except (TypeError, ValueError):
        return ""
    return " x ".join(f"{value:,}" for value in values)


def estimate_retained_bytes(value: Any, _seen: set[int] | None = None) -> int:
    """Estimate bytes retained by numeric results without copying their arrays."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return 0
    seen = set() if _seen is None else _seen
    identity = id(value)
    if identity in seen:
        return 0
    seen.add(identity)
    if isinstance(value, memoryview):
        return int(value.nbytes)
    if isinstance(value, (bytes, bytearray)):
        return len(value)
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if isinstance(value, Mapping):
        return sum(estimate_retained_bytes(item, seen) for item in value.values())
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return sum(
            estimate_retained_bytes(getattr(value, item.name), seen)
            for item in dataclasses.fields(value)
        )
    if isinstance(value, Sequence):
        return sum(estimate_retained_bytes(item, seen) for item in value)
    return 0


@dataclass(frozen=True, slots=True)
class CacheEntry:
    """One downloadable artifact or retained workflow reference."""

    id: str
    category: CacheCategory
    name: str
    source: str
    kind: CacheKind
    created_at: float
    size_bytes: int
    detail: str = ""
    artifact: ExportArtifact | None = None
    payload: Any = field(default=None, repr=False, compare=False)
    navigate_to: str | None = None
    digest: str | None = None

    @property
    def format_label(self) -> str:
        """Return a concise extension label, retaining compression information."""

        suffixes = [suffix.removeprefix(".").upper() for suffix in PurePath(self.name).suffixes]
        if not suffixes:
            return "RESULT" if self.kind == "workflow" else "FILE"
        if len(suffixes) >= 2 and suffixes[-1] in {"GZ", "XZ", "ZIP", "7Z"}:
            return f"{suffixes[-2]} + {suffixes[-1]}"
        return suffixes[-1]


@dataclass(slots=True)
class SessionCache:
    """Mutable, bounded cache state owned by one NiceGUI browser tab."""

    max_items: int = DEFAULT_MAX_ITEMS
    max_total_items: int = DEFAULT_MAX_TOTAL_ITEMS
    max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES
    _entries: OrderedDict[str, CacheEntry] = field(default_factory=OrderedDict)
    _active_entries: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.max_items = max(1, min(MAX_ITEMS, int(self.max_items)))
        self.max_total_items = max(1, min(MAX_TOTAL_ITEMS, int(self.max_total_items)))
        self.max_total_bytes = max(1, min(MAX_TOTAL_BYTES, int(self.max_total_bytes)))

    def add_artifact(
        self,
        artifact: ExportArtifact,
        *,
        category: CacheCategory = "generated",
        source: str,
        kind: CacheKind,
        detail: str = "",
        payload: Any = None,
        navigate_to: str | None = None,
    ) -> CacheEntry:
        """Insert an artifact, refreshing rather than duplicating identical bytes."""

        if category not in CACHE_CATEGORIES:
            raise ValueError(f"Unknown cache category {category!r}.")
        if kind == "workflow":
            raise ValueError("Workflow references must be added with add_reference().")
        normalized = ExportArtifact(
            data=artifact.data,
            filename=_safe_filename(artifact.filename),
            media_type=str(artifact.media_type or "application/octet-stream"),
        )
        digest = _artifact_digest(normalized)
        existing = next(
            (
                entry
                for entry in self._entries.values()
                if entry.category == category and entry.digest == digest
            ),
            None,
        )
        now = time.time()
        retained_size = estimate_retained_bytes((normalized, payload))
        if existing is not None:
            entry = replace(
                existing,
                source=source,
                kind=kind,
                created_at=now,
                detail=detail,
                artifact=normalized,
                payload=payload,
                navigate_to=navigate_to,
                size_bytes=retained_size,
            )
            self._entries[entry.id] = entry
            self._entries.move_to_end(entry.id)
            self._enforce_limits(category)
            return entry

        entry = CacheEntry(
            id=uuid.uuid4().hex,
            category=category,
            name=normalized.filename,
            source=str(source).strip() or "DNAscape",
            kind=kind,
            created_at=now,
            size_bytes=retained_size,
            detail=str(detail).strip(),
            artifact=normalized,
            payload=payload,
            navigate_to=navigate_to,
            digest=digest,
        )
        self._entries[entry.id] = entry
        self._enforce_limits(category)
        return entry

    def add_reference(
        self,
        *,
        name: str,
        source: str,
        detail: str,
        payload: Any,
        navigate_to: str,
        size_bytes: int | None = None,
    ) -> CacheEntry:
        """Retain a completed workflow result for reopening in its source page."""

        entry = CacheEntry(
            id=uuid.uuid4().hex,
            category="workflows",
            name=str(name).strip() or "Workflow result",
            source=str(source).strip() or "DNAscape",
            kind="workflow",
            created_at=time.time(),
            size_bytes=(
                max(0, int(size_bytes))
                if size_bytes is not None
                else estimate_retained_bytes(payload)
            ),
            detail=str(detail).strip(),
            payload=payload,
            navigate_to=navigate_to,
        )
        self._entries[entry.id] = entry
        self._enforce_limits("workflows")
        return entry

    def get(self, entry_id: str) -> CacheEntry | None:
        return self._entries.get(entry_id)

    def remove(self, entry_id: str) -> bool:
        """Remove one entry and any pending open action tied to it."""

        removed = self._entries.pop(entry_id, None)
        if removed is None:
            return False
        self._active_entries = {
            slug: active_id
            for slug, active_id in self._active_entries.items()
            if active_id != entry_id
        }
        return True

    def clear(self, category: CacheCategory) -> int:
        """Clear one inventory group and report how many entries were removed."""

        identifiers = [
            entry.id for entry in self._entries.values() if entry.category == category
        ]
        for identifier in identifiers:
            self.remove(identifier)
        return len(identifiers)

    def set_max_items(self, value: int) -> None:
        normalized = max(1, min(MAX_ITEMS, int(value)))
        self.max_items = normalized
        for category in CACHE_CATEGORIES:
            self._trim(category)
        self._trim_global()

    def list_entries(
        self,
        category: CacheCategory | None,
        *,
        query: str = "",
        kind: str = "all",
        sort: CacheSort = "newest",
    ) -> list[CacheEntry]:
        """Return a filtered and predictably ordered snapshot of one group."""

        normalized_query = str(query).strip().casefold()
        normalized_kind = str(kind).strip().casefold()
        entries = [
            entry
            for entry in self._entries.values()
            if category is None or entry.category == category
        ]
        if normalized_query:
            entries = [
                entry
                for entry in entries
                if normalized_query
                in " ".join((entry.name, entry.source, entry.kind, entry.detail)).casefold()
            ]
        if category in {None, "generated"} and normalized_kind in {"plot", "raw"}:
            entries = [entry for entry in entries if entry.kind == normalized_kind]

        if sort == "oldest":
            entries.sort(key=lambda entry: (entry.created_at, entry.name.casefold()))
        elif sort == "largest":
            entries.sort(key=lambda entry: (-entry.size_bytes, -entry.created_at))
        elif sort == "smallest":
            entries.sort(key=lambda entry: (entry.size_bytes, -entry.created_at))
        elif sort == "name":
            entries.sort(key=lambda entry: (entry.name.casefold(), -entry.created_at))
        else:
            entries.sort(key=lambda entry: (-entry.created_at, entry.name.casefold()))
        return entries

    def count(self, category: CacheCategory | None = None) -> int:
        return sum(
            category is None or entry.category == category for entry in self._entries.values()
        )

    def size(self, category: CacheCategory | None = None) -> int:
        return sum(
            entry.size_bytes
            for entry in self._entries.values()
            if category is None or entry.category == category
        )

    def activate(self, entry_id: str) -> CacheEntry:
        """Mark a retained result to be restored when its source page opens."""

        entry = self._entries.get(entry_id)
        if entry is None or entry.navigate_to is None or entry.payload is None:
            raise ValueError("This cached item cannot be opened in a workflow page.")
        self._entries.move_to_end(entry.id)
        self._active_entries[entry.navigate_to] = entry.id
        return entry

    def consume_active(self, slug: str) -> CacheEntry | None:
        identifier = self._active_entries.pop(slug, None)
        return self._entries.get(identifier) if identifier is not None else None

    def _trim(self, category: CacheCategory) -> None:
        identifiers = [
            entry.id for entry in self._entries.values() if entry.category == category
        ]
        for identifier in identifiers[: max(0, len(identifiers) - self.max_items)]:
            self.remove(identifier)

    def _trim_global(self) -> None:
        while self.count() > self.max_total_items or self.size() > self.max_total_bytes:
            identifier = next(iter(self._entries), None)
            if identifier is None:
                return
            self.remove(identifier)

    def _enforce_limits(self, category: CacheCategory) -> None:
        self._trim(category)
        self._trim_global()


_SESSION_CACHES: OrderedDict[str, SessionCache] = OrderedDict()


def current_session_cache() -> SessionCache:
    """Return the cache associated with the active NiceGUI browser tab."""

    client = context.client
    session_id = str(client.tab_id or client.id)
    cache = _SESSION_CACHES.get(session_id)
    if cache is None:
        cache = SessionCache()
        _SESSION_CACHES[session_id] = cache
    _SESSION_CACHES.move_to_end(session_id)
    while len(_SESSION_CACHES) > MAX_SESSION_CACHES:
        _SESSION_CACHES.popitem(last=False)
    return cache


def cache_artifact(
    artifact: ExportArtifact,
    *,
    source: str,
    kind: Literal["plot", "raw"],
    detail: str = "",
    navigate_to: str | None = None,
) -> CacheEntry:
    return current_session_cache().add_artifact(
        artifact,
        source=source,
        kind=kind,
        detail=detail,
        navigate_to=navigate_to,
    )


def cache_upload(
    inspection: Any,
    *,
    source: str,
    navigate_to: str,
) -> CacheEntry:
    shape = _shape_text(getattr(inspection, "shape", ()))
    dependence = str(getattr(inspection, "inferred_form", "unknown")).replace("_", " ").title()
    detail_parts = [str(getattr(inspection, "source_format", "file")).upper(), dependence]
    if shape:
        detail_parts.append(shape)
    artifact = ExportArtifact(
        data=bytes(inspection.content),
        filename=_safe_filename(str(inspection.name)),
        media_type=str(getattr(inspection, "content_type", "") or "application/octet-stream"),
    )
    return current_session_cache().add_artifact(
        artifact,
        category="uploads",
        source=source,
        kind="upload",
        detail=" | ".join(detail_parts),
        payload=inspection,
        navigate_to=navigate_to,
    )


def cache_dataset(result: Any, *, source: str = "Data Manager") -> CacheEntry:
    shape = _shape_text(getattr(result, "shape", ()))
    form = str(getattr(result, "data_form", "data")).replace("_", " ").title()
    detail = " | ".join(part for part in (form, shape) if part)
    return current_session_cache().add_artifact(
        result.artifact,
        category="datasets",
        source=source,
        kind="dataset",
        detail=detail,
        payload=result,
        navigate_to="data-management",
    )


def cache_workflow(
    result: Any,
    *,
    name: str,
    source: str,
    detail: str,
    navigate_to: Literal["mapping", "simulation"],
) -> CacheEntry:
    return current_session_cache().add_reference(
        name=name,
        source=source,
        detail=detail,
        payload=result,
        navigate_to=navigate_to,
    )


def consume_cached_workflow(slug: Literal["mapping", "simulation"]) -> CacheEntry | None:
    return current_session_cache().consume_active(slug)
