"""Functional inventory for files and results retained in the current session."""

from __future__ import annotations

import math
from collections.abc import Callable
from datetime import datetime
from typing import Any, Literal, cast

from nicegui import ui

from app.pages.common import page_intro, panel, panel_title, stat_strip
from app.session_cache import CacheCategory, CacheEntry, CacheSort, current_session_cache

CacheView = Literal["all", "generated", "uploads", "datasets", "workflows"]

_CATEGORY_LABELS: dict[CacheView, str] = {
    "all": "All items",
    "generated": "Generated files",
    "uploads": "Uploads",
    "datasets": "Prepared datasets",
    "workflows": "Workflow results",
}
_CATEGORY_ICONS: dict[CacheView, str] = {
    "all": "inventory_2",
    "generated": "download_done",
    "uploads": "upload_file",
    "datasets": "dataset",
    "workflows": "science",
}
_KIND_ICONS = {
    "plot": "image",
    "raw": "data_object",
    "upload": "upload_file",
    "dataset": "dataset",
    "workflow": "science",
}
_SORT_OPTIONS: dict[CacheSort, str] = {
    "newest": "Newest first",
    "oldest": "Oldest first",
    "largest": "Largest first",
    "smallest": "Smallest first",
    "name": "Name",
}


def _size_text(size: int) -> str:
    value = float(max(0, size))
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"


def _created_text(timestamp: float) -> str:
    created = datetime.fromtimestamp(timestamp).astimezone()
    now = datetime.now().astimezone()
    if created.date() == now.date():
        return f"Today {created:%H:%M}"
    if created.year == now.year:
        return created.strftime("%d %b, %H:%M")
    return created.strftime("%d %b %Y")


def _entry_row(
    entry: CacheEntry,
    *,
    download: Callable[[str], None],
    open_entry: Callable[[str], None],
    remove: Callable[[str], None],
) -> None:
    with ui.element("div").classes("dn-cache-item").mark(f"cache-entry-{entry.id}"):
        with ui.element("div").classes("dn-cache-kind-icon"):
            ui.icon(_KIND_ICONS[entry.kind])
        with ui.element("div").classes("dn-cache-identity"):
            ui.label(entry.name).classes("dn-cache-name")
            detail = " | ".join(part for part in (entry.source, entry.detail) if part)
            ui.label(detail).classes("dn-cache-detail")
        with ui.element("div").classes("dn-cache-meta dn-cache-created"):
            ui.label("Created").classes("dn-cache-meta-label")
            ui.label(_created_text(entry.created_at)).classes("dn-cache-meta-value")
        with ui.element("div").classes("dn-cache-meta dn-cache-size"):
            ui.label("Memory").classes("dn-cache-meta-label")
            ui.label(_size_text(entry.size_bytes)).classes("dn-cache-meta-value")
        with ui.element("div").classes("dn-cache-meta dn-cache-format"):
            ui.label("Format").classes("dn-cache-meta-label")
            ui.label(entry.format_label).classes("dn-cache-meta-value")
        with ui.row().classes("dn-cache-actions"):
            if entry.artifact is not None:
                with (
                    ui.button(
                        icon="download",
                        on_click=lambda _event, entry_id=entry.id: download(entry_id),
                    )
                    .props('flat round dense aria-label="Download cached file"')
                    .classes("dn-cache-icon-action")
                    .mark(f"cache-download-{entry.id}")
                ):
                    ui.tooltip(f"Download {entry.name}")
            elif entry.navigate_to is not None:
                with (
                    ui.button(
                        icon="open_in_new",
                        on_click=lambda _event, entry_id=entry.id: open_entry(entry_id),
                    )
                    .props('flat round dense aria-label="Open cached workflow"')
                    .classes("dn-cache-icon-action")
                    .mark(f"cache-open-{entry.id}")
                ):
                    ui.tooltip(f"Open in {entry.source}")
            with (
                ui.button(
                    icon="close",
                    on_click=lambda _event, entry_id=entry.id: remove(entry_id),
                )
                .props('flat round dense aria-label="Remove cached item"')
                .classes("dn-cache-icon-action dn-cache-remove-action")
                .mark(f"cache-remove-{entry.id}")
            ):
                ui.tooltip(f"Remove {entry.name}")


def render(navigate: Callable[[str], None]) -> None:
    """Render the bounded cache associated with the active browser session."""

    cache = current_session_cache()
    filters: dict[str, Any] = {
        "query": "",
        "kind": "all",
        "sort": "newest",
        "rows": 10,
    }
    pages: dict[CacheView, int] = {
        "all": 1,
        "generated": 1,
        "uploads": 1,
        "datasets": 1,
        "workflows": 1,
    }

    page_intro(
        "Cached Files",
        "Session files and results",
        "Find, download, or reopen items retained in this browser session.",
    )

    with panel(extra_classes="dn-cache-panel"):
        with ui.row().classes("dn-cache-panel-head"):
            panel_title("Inventory")
            ui.space()
            ui.icon("memory").classes("dn-cache-memory-note").tooltip(
                f"Automatically retains up to {cache.max_total_items} recent items and "
                f"{_size_text(cache.max_total_bytes)} in this browser tab."
            )

        @ui.refreshable
        def render_summary() -> None:
            stat_strip(
                (
                    ("Items", f"{cache.count():,} / {cache.max_total_items:,}"),
                    ("Memory", f"{_size_text(cache.size())} / {_size_text(cache.max_total_bytes)}"),
                    (
                        "Files",
                        f"{cache.count('generated') + cache.count('uploads') + cache.count('datasets'):,}",
                    ),
                    ("Results", f"{cache.count('workflows'):,}"),
                )
            )

        render_summary()

        with ui.element("div").classes("dn-cache-toolbar"):
            search = (
                ui.input(label="Search", placeholder="Name, source, or type")
                .props("dense clearable debounce=250")
                .classes("dn-field dn-cache-search")
                .mark("cache-search")
            )
            generated_kind = (
                ui.select(
                    {"all": "All generated", "raw": "Raw data", "plot": "Plots"},
                    value="all",
                    label="Generated type",
                )
                .props("dense options-dense")
                .classes("dn-field")
                .mark("cache-kind-filter")
            )
            sort_control = (
                ui.select(_SORT_OPTIONS, value="newest", label="Sort")
                .props("dense options-dense")
                .classes("dn-field")
                .mark("cache-sort")
            )
            clear_button = (
                ui.button("Clear generated", icon="delete_sweep")
                .props("flat no-caps")
                .classes("dn-action-secondary dn-cache-clear")
                .mark("clear-generated-cache", "action-clear-generated-cache")
            )

        with ui.tabs().classes("dn-tabs dn-cache-tabs") as tabs:
            tab_objects: dict[CacheView, Any] = {
                category: ui.tab(label, icon=_CATEGORY_ICONS[category])
                for category, label in _CATEGORY_LABELS.items()
            }

        def refresh_all() -> None:
            render_summary.refresh()
            render_inventory.refresh()

        def download_entry(entry_id: str) -> None:
            entry = cache.get(entry_id)
            if entry is None or entry.artifact is None:
                ui.notify("This cached file is no longer available.", type="warning")
                refresh_all()
                return
            artifact = entry.artifact
            ui.download(artifact.data, artifact.filename, artifact.media_type)

        def open_entry(entry_id: str) -> None:
            try:
                entry = cache.activate(entry_id)
            except ValueError as exc:
                ui.notify(str(exc), type="warning")
                refresh_all()
                return
            assert entry.navigate_to is not None
            navigate(entry.navigate_to)

        def remove_entry(entry_id: str) -> None:
            entry = cache.get(entry_id)
            if entry is None:
                return
            cache.remove(entry_id)
            ui.notify(f"Removed {entry.name}.", type="positive")
            refresh_all()

        def set_page(category: CacheView, value: Any, maximum: int) -> None:
            try:
                numeric = int(value)
            except (TypeError, ValueError):
                numeric = 1
            pages[category] = max(1, min(maximum, numeric))
            render_inventory.refresh()

        def render_empty(category: CacheView, *, filtered: bool) -> None:
            with ui.column().classes("dn-cache-empty"):
                ui.icon("search_off" if filtered else _CATEGORY_ICONS[category])
                ui.label(
                    "No matching items"
                    if filtered
                    else "No cached items yet"
                    if category == "all"
                    else f"No {_CATEGORY_LABELS[category].lower()} yet"
                )

        def render_group(category: CacheView) -> None:
            entries = cache.list_entries(
                None if category == "all" else cast(CacheCategory, category),
                query=str(filters["query"]),
                kind=str(filters["kind"]),
                sort=cast(CacheSort, filters["sort"]),
            )
            row_count = max(1, int(filters["rows"]))
            maximum_page = max(1, math.ceil(len(entries) / row_count))
            pages[category] = min(pages[category], maximum_page)
            first = (pages[category] - 1) * row_count
            visible_entries = entries[first : first + row_count]

            with ui.row().classes("dn-cache-group-summary"):
                ui.label(f"{len(entries):,} items").classes("dn-cache-group-count")
                ui.label(_size_text(sum(entry.size_bytes for entry in entries))).classes(
                    "dn-cache-group-size"
                )
            if not visible_entries:
                render_empty(
                    category,
                    filtered=bool(str(filters["query"]).strip())
                    or (category in {"all", "generated"} and filters["kind"] != "all"),
                )
                return
            with ui.element("div").classes("dn-cache-list"):
                for entry in visible_entries:
                    _entry_row(
                        entry,
                        download=download_entry,
                        open_entry=open_entry,
                        remove=remove_entry,
                    )
            with ui.element("div").classes("dn-cache-pagination"):
                rows_control = (
                    ui.select([10, 25, 50], value=row_count, label="Rows")
                    .props("dense options-dense")
                    .classes("dn-field dn-cache-rows")
                    .mark(f"cache-rows-{category}")
                )
                page_control = (
                    ui.number(
                        label=f"Page of {maximum_page}",
                        value=pages[category],
                        min=1,
                        max=maximum_page,
                        step=1,
                    )
                    .props("dense")
                    .classes("dn-field dn-cache-page")
                    .mark(f"cache-page-{category}")
                )
                previous = (
                    ui.button(
                        icon="chevron_left",
                        on_click=lambda _event, category=category: set_page(
                            category,
                            pages[category] - 1,
                            maximum_page,
                        ),
                    )
                    .props('flat round dense aria-label="Previous cache page"')
                    .classes("dn-cache-icon-action")
                )
                next_button = (
                    ui.button(
                        icon="chevron_right",
                        on_click=lambda _event, category=category: set_page(
                            category,
                            pages[category] + 1,
                            maximum_page,
                        ),
                    )
                    .props('flat round dense aria-label="Next cache page"')
                    .classes("dn-cache-icon-action")
                )
                if pages[category] <= 1:
                    previous.disable()
                if pages[category] >= maximum_page:
                    next_button.disable()

                def rows_changed() -> None:
                    filters["rows"] = int(rows_control.value or 10)
                    for key in pages:
                        pages[key] = 1
                    render_inventory.refresh()

                rows_control.on_value_change(rows_changed)
                page_control.on_value_change(
                    lambda _event, category=category: set_page(
                        category,
                        page_control.value,
                        maximum_page,
                    )
                )

        with ui.tab_panels(tabs, value=tab_objects["all"], animated=False).classes(
            "dn-tab-panels dn-cache-tab-panels"
        ):

            @ui.refreshable
            def render_inventory() -> None:
                for category, tab in tab_objects.items():
                    with ui.tab_panel(tab).classes("dn-cache-tab-panel"):
                        render_group(category)

            render_inventory()

        with ui.dialog() as clear_dialog, ui.card().classes(
            "dn-settings-dialog dn-cache-dialog"
        ):
            with ui.row().classes("dn-settings-dialog-head"):
                ui.label("Clear generated files?").classes("dn-panel-title")
                ui.space()
                ui.button(icon="close", on_click=clear_dialog.close).props(
                    'flat round dense aria-label="Close clear cache dialog"'
                )
            ui.label(
                "Plots and raw-data exports will be released from session memory. "
                "Uploads and retained results will stay available."
            ).classes("dn-about-copy")
            with ui.row().classes("dn-cache-dialog-actions"):
                ui.button("Cancel", on_click=clear_dialog.close).props("flat no-caps")

                def confirm_clear() -> None:
                    removed = cache.clear("generated")
                    clear_dialog.close()
                    ui.notify(f"Cleared {removed:,} generated files.", type="positive")
                    refresh_all()

                (
                    ui.button("Clear", icon="delete_sweep", on_click=confirm_clear)
                    .props("no-caps")
                    .classes("dn-action-primary")
                    .mark("confirm-clear-generated-cache")
                )

        def filters_changed() -> None:
            filters.update(
                query=str(search.value or ""),
                kind=str(generated_kind.value or "all"),
                sort=str(sort_control.value or "newest"),
            )
            for category in pages:
                pages[category] = 1
            render_inventory.refresh()

        search.on_value_change(filters_changed)
        generated_kind.on_value_change(filters_changed)
        sort_control.on_value_change(filters_changed)
        clear_button.on_click(clear_dialog.open)
