"""Bounded, browser-session-local inventory for DNAscape files and results."""

from __future__ import annotations

import dataclasses
import hashlib
import mmap
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from pathlib import PurePath
from typing import Any, Literal

import numpy as np
from nicegui import context

from app.simulation_exports import ExportArtifact
from dnascape import _resources
from dnascape._resources import MIB, hosted

CacheCategory = Literal["generated", "uploads", "datasets", "workflows"]
CacheKind = Literal["plot", "raw", "upload", "dataset", "workflow"]
CacheSort = Literal["newest", "oldest", "largest", "smallest", "name"]

CACHE_CATEGORIES: tuple[CacheCategory, ...] = (
    "generated",
    "uploads",
    "datasets",
    "workflows",
)
DEFAULT_TRACKED_CATEGORIES: frozenset[CacheCategory] = frozenset(
    {"generated", "datasets", "workflows"}
)
DEFAULT_MAX_ITEMS = 30
DEFAULT_MAX_TOTAL_ITEMS = 80
DEFAULT_MAX_TOTAL_BYTES = 256 * 1024 * 1024
MAX_ITEMS = 200
MAX_TOTAL_ITEMS = 500
MAX_TOTAL_BYTES = 2 * 1024 * 1024 * 1024
MAX_SESSION_CACHES = 32
HOSTED_CACHE_BYTES = 4 * MIB
HOSTED_SHARED_CACHE_BYTES = 8 * MIB
HOSTED_CACHE_ITEMS = 5
HOSTED_CACHE_TOTAL_ITEMS = 10
_INTERNAL_UPLOAD = ContextVar("dnascape_internal_upload", default=False)


@contextmanager
def internal_upload():
    """Do not duplicate transferred data or individual simulations in cache."""
    token = _INTERNAL_UPLOAD.set(True)
    try:
        yield
    finally:
        _INTERNAL_UPLOAD.reset(token)


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
    if isinstance(value, mmap.mmap):
        return len(value)
    if isinstance(value, np.ndarray):
        # A small slice keeps its entire parent allocation alive.
        if value.base is not None:
            return estimate_retained_bytes(value.base, seen)
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
    keep: bool = True
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
    _output_clearers: dict[str, Callable[[], None]] = field(default_factory=dict)
    _page_refreshers: dict[str, Callable[[], None]] = field(default_factory=dict)
    _input_target_providers: dict[str, Callable[[], Sequence[Any]]] = field(default_factory=dict)
    _client_ids: set[str] = field(default_factory=set)
    _tracked_categories: set[CacheCategory] = field(
        default_factory=lambda: set(DEFAULT_TRACKED_CATEGORIES)
    )

    def __post_init__(self) -> None:
        self.max_items = max(1, min(MAX_ITEMS, int(self.max_items)))
        self.max_total_items = max(1, min(MAX_TOTAL_ITEMS, int(self.max_total_items)))
        self.max_total_bytes = max(1, min(MAX_TOTAL_BYTES, int(self.max_total_bytes)))
        self._tracked_categories = {
            category for category in self._tracked_categories if category in CACHE_CATEGORIES
        }
        if hosted():
            self.max_items = min(self.max_items, HOSTED_CACHE_ITEMS)
            self.max_total_items = min(self.max_total_items, HOSTED_CACHE_TOTAL_ITEMS)
            self.max_total_bytes = min(self.max_total_bytes, HOSTED_CACHE_BYTES)
            self._tracked_categories.discard("workflows")

    @property
    def byte_limit(self) -> int:
        """The effective tab allowance, including current hosted headroom."""
        status = _resources.memory_status()
        if status is None:
            return self.max_total_bytes
        return min(self.max_total_bytes, HOSTED_CACHE_BYTES, status.limit // 128,
                   status.available // 4)

    def set_byte_limit(self, value: int) -> None:
        maximum = HOSTED_CACHE_BYTES if hosted() else MAX_TOTAL_BYTES
        self.max_total_bytes = max(0, min(maximum, int(value)))
        self._trim_global()

    def is_tracking(self, category: CacheCategory) -> bool:
        """Return whether future items in ``category`` should be retained automatically."""

        if category not in CACHE_CATEGORIES:
            raise ValueError(f"Unknown cache category {category!r}.")
        return category in self._tracked_categories

    def set_tracking(self, category: CacheCategory, enabled: bool) -> None:
        """Enable or disable future automatic retention for one category.

        Existing entries intentionally remain untouched; the preference only
        controls items produced after it changes.
        """

        if category not in CACHE_CATEGORIES:
            raise ValueError(f"Unknown cache category {category!r}.")
        if enabled:
            self._tracked_categories.add(category)
        else:
            self._tracked_categories.discard(category)

    def tracked_categories(self) -> tuple[CacheCategory, ...]:
        """Return enabled categories in the same stable order as the inventory tabs."""

        return tuple(category for category in CACHE_CATEGORIES if self.is_tracking(category))

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
        keep: bool | None = None,
    ) -> CacheEntry | None:
        """Insert an artifact, refreshing rather than duplicating identical bytes."""

        if category not in CACHE_CATEGORIES:
            raise ValueError(f"Unknown cache category {category!r}.")
        if kind == "workflow":
            raise ValueError("Workflow references must be added with add_reference().")
        retained_size = estimate_retained_bytes((artifact, payload))
        if hosted():
            self._enforce_limits(category)
            if self.byte_limit == 0 or retained_size > self.byte_limit:
                return None
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
        if existing is not None:
            retained_artifact = ExportArtifact(
                data=normalized.data,
                filename=existing.name,
                media_type=normalized.media_type,
            )
            entry = replace(
                existing,
                source=source,
                kind=kind,
                created_at=now,
                detail=detail,
                keep=existing.keep if keep is None else bool(keep),
                artifact=retained_artifact,
                payload=payload,
                navigate_to=navigate_to,
                size_bytes=retained_size,
            )
            self._entries[entry.id] = entry
            self._entries.move_to_end(entry.id)
            self._enforce_limits(category)
            return entry

        unique_name = self._unique_name(normalized.filename)
        normalized = ExportArtifact(
            data=normalized.data,
            filename=unique_name,
            media_type=normalized.media_type,
        )
        entry = CacheEntry(
            id=uuid.uuid4().hex,
            category=category,
            name=unique_name,
            source=str(source).strip() or "DNAscape",
            kind=kind,
            created_at=now,
            size_bytes=retained_size,
            detail=str(detail).strip(),
            keep=category != "uploads" if keep is None else bool(keep),
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
        keep: bool = True,
    ) -> CacheEntry | None:
        """Retain a completed workflow result for reopening in its source page."""

        retained_size = max(0, int(size_bytes)) if size_bytes is not None else estimate_retained_bytes(payload)
        if hosted():
            self._enforce_limits("workflows")
            if self.byte_limit == 0 or retained_size > self.byte_limit:
                return None
        entry = CacheEntry(
            id=uuid.uuid4().hex,
            category="workflows",
            name=self._unique_name(str(name).strip() or "Workflow result"),
            source=str(source).strip() or "DNAscape",
            kind="workflow",
            created_at=time.time(),
            size_bytes=retained_size,
            detail=str(detail).strip(),
            keep=bool(keep),
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

    def set_keep(self, entry_id: str, value: bool) -> CacheEntry:
        """Set whether automatic eviction should protect one retained item."""

        entry = self._entries.get(entry_id)
        if entry is None:
            raise ValueError("This cached item is no longer available.")
        updated = replace(entry, keep=bool(value))
        self._entries[entry_id] = updated
        return updated

    def clear(self, category: CacheCategory) -> int:
        """Clear one inventory group and report how many entries were removed."""

        identifiers = [entry.id for entry in self._entries.values() if entry.category == category]
        for identifier in identifiers:
            self.remove(identifier)
        return len(identifiers)

    def clear_all(self) -> int:
        """Clear every cached entry while leaving mounted page controls untouched."""

        identifiers = tuple(self._entries)
        for identifier in identifiers:
            self.remove(identifier)
        return len(identifiers)

    def reset(self) -> int:
        """Clear retained state and restore cache preferences to their defaults."""

        removed = self.clear_all()
        self._active_entries.clear()
        self.max_items = DEFAULT_MAX_ITEMS
        self.max_total_items = DEFAULT_MAX_TOTAL_ITEMS
        self.max_total_bytes = DEFAULT_MAX_TOTAL_BYTES
        self._tracked_categories = set(DEFAULT_TRACKED_CATEGORIES)
        self.__post_init__()
        return removed

    def register_output_clearer(self, slug: str, callback: Callable[[], None]) -> None:
        """Register one mounted page's result-only reset callback."""

        self._output_clearers[str(slug)] = callback

    def unregister_output_clearer(self, slug: str) -> None:
        self._output_clearers.pop(str(slug), None)

    def clear_page_outputs(self) -> None:
        """Release mounted workflow outputs without resetting page controls."""

        for callback in tuple(self._output_clearers.values()):
            callback()

    def register_page_refresher(self, slug: str, callback: Callable[[], None]) -> None:
        """Register work that should run whenever a mounted page is revealed."""

        self._page_refreshers[str(slug)] = callback

    def unregister_page_refresher(self, slug: str) -> None:
        self._page_refreshers.pop(str(slug), None)

    def register_input_targets(self, slug: str, provider: Callable[[], Sequence[Any]]) -> None:
        self._input_target_providers[str(slug)] = provider

    def unregister_input_targets(self, slug: str) -> None:
        self._input_target_providers.pop(str(slug), None)

    def input_targets(self, slug: str) -> Sequence[Any]:
        provider = self._input_target_providers.get(str(slug))
        return provider() if provider is not None else ()

    def refresh_page(self, slug: str) -> None:
        callback = self._page_refreshers.get(str(slug))
        if callback is not None:
            callback()

    def set_max_items(self, value: int) -> None:
        normalized = max(1, min(MAX_ITEMS, int(value)))
        if hosted():
            normalized = min(normalized, HOSTED_CACHE_ITEMS)
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

    def has_active(self, slug: str) -> bool:
        """Return whether a workflow page has a pending cached-result restore."""

        return str(slug) in self._active_entries

    def _trim(self, category: CacheCategory) -> None:
        identifiers = [entry.id for entry in self._entries.values() if entry.category == category]
        overflow = max(0, len(identifiers) - self.max_items)
        eviction_order = [
            identifier for identifier in identifiers if not self._entries[identifier].keep
        ] + [identifier for identifier in identifiers if self._entries[identifier].keep]
        for identifier in eviction_order[:overflow]:
            self.remove(identifier)

    def _trim_global(self) -> None:
        byte_limit = self.byte_limit
        while self.count() > self.max_total_items or self.size() > byte_limit:
            identifier = next(
                (entry.id for entry in self._entries.values() if not entry.keep),
                next(iter(self._entries), None),
            )
            if identifier is None:
                return
            self.remove(identifier)

    def _enforce_limits(self, category: CacheCategory) -> None:
        self._trim(category)
        self._trim_global()
        status = _resources.memory_status()
        if status is not None:
            # One small shared allowance across visitors, reduced under pressure.
            caches = list(_SESSION_CACHES.values())
            if all(cache is not self for cache in caches):
                caches.append(self)
            limit = min(HOSTED_SHARED_CACHE_BYTES, status.limit // 64, status.available // 2)
            while sum(cache.size() for cache in caches) > limit:
                candidates = [(entry.created_at, cache, entry.id)
                              for cache in caches for entry in cache._entries.values()]
                if not candidates:
                    break
                _, cache, identifier = min(candidates, key=lambda item: item[0])
                cache.remove(identifier)

    def _unique_name(self, requested: str) -> str:
        """Return a case-insensitively unique cache name with compound suffixes intact."""

        name = _safe_filename(requested)
        existing = {entry.name.casefold() for entry in self._entries.values()}
        if name.casefold() not in existing:
            return name
        suffix = "".join(PurePath(name).suffixes)
        stem = name[: -len(suffix)] if suffix else name
        stem = stem or "file"
        index = 2
        while True:
            candidate = f"{stem} ({index}){suffix}"
            if candidate.casefold() not in existing:
                return candidate
            index += 1


_SESSION_CACHES: OrderedDict[str, SessionCache] = OrderedDict()


def current_session_cache() -> SessionCache:
    """Return the cache associated with the active NiceGUI browser tab."""

    client = context.client
    session_id = str(client.tab_id or client.id)
    cache = _SESSION_CACHES.get(session_id)
    if cache is None:
        cache = SessionCache()
        _SESSION_CACHES[session_id] = cache
    if hosted() and client.id not in cache._client_ids:
        cache._client_ids.add(client.id)

        def release_client() -> None:
            cache._client_ids.discard(client.id)
            if cache._client_ids:
                return
            cache.clear_all()
            cache._output_clearers.clear()
            cache._page_refreshers.clear()
            cache._input_target_providers.clear()
            if _SESSION_CACHES.get(session_id) is cache:
                _SESSION_CACHES.pop(session_id)

        # Delete follows NiceGUI's reconnect grace period, unlike disconnect.
        client.on_delete(release_client)
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
) -> CacheEntry | None:
    cache = current_session_cache()
    if not cache.is_tracking("generated"):
        return None
    return cache.add_artifact(
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
) -> CacheEntry | None:
    if _INTERNAL_UPLOAD.get():
        return None
    cache = current_session_cache()
    if not cache.is_tracking("uploads"):
        return None
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
    return cache.add_artifact(
        artifact,
        category="uploads",
        source=source,
        kind="upload",
        detail=" | ".join(detail_parts),
        payload=inspection,
        navigate_to=navigate_to,
    )


def cache_dataset(result: Any, *, source: str = "Data Manager") -> CacheEntry | None:
    cache = current_session_cache()
    if not cache.is_tracking("datasets"):
        return None
    shape = _shape_text(getattr(result, "shape", ()))
    form = str(getattr(result, "data_form", "data")).replace("_", " ").title()
    detail = " | ".join(part for part in (form, shape) if part)
    return cache.add_artifact(
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
) -> CacheEntry | None:
    cache = current_session_cache()
    if not cache.is_tracking("workflows"):
        return None
    return cache.add_reference(
        name=name,
        source=source,
        detail=detail,
        payload=result,
        navigate_to=navigate_to,
    )


def consume_cached_workflow(slug: Literal["mapping", "simulation"]) -> CacheEntry | None:
    return current_session_cache().consume_active(slug)
