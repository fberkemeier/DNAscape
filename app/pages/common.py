"""Reusable components for the local NiceGUI interface."""

from __future__ import annotations

import asyncio
import weakref
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from typing import Any

from nicegui import background_tasks, events, helpers, ui

from app.export_settings import DATA_FILE_FORMAT_OPTIONS, get_export_settings
from app.icons import DATA_ICON, DATA_ICON_SVG, PLOT_ICON_SVG
from app.mapping_uploads import MAX_DECOMPRESSED_BYTES, MAX_UPLOAD_BYTES

# Keep every data-export selector in the same user-facing order.
RAW_DATA_FORMATS = tuple(DATA_FILE_FORMAT_OPTIONS.values())


@dataclass(frozen=True)
class SourceInputControls:
    """Handles for reading a scalar/profile source control at submit time."""

    mode: Any
    constant: Any


@dataclass(frozen=True)
class RawOutputControls:
    """Live controls for a workflow's raw-data download."""

    file_name: Any
    file_type: Any
    compress: Any
    download_button: Any
    row: Any
    remove_button: Any | None = None
    extra_slot: Any | None = None


@dataclass(kw_only=True)
class UploadClearEventArguments(events.UiEventArguments):
    """Removal event; an empty ``name`` denotes bulk clearing of ``names``."""

    name: str
    names: tuple[str, ...] = ()


@dataclass
class _DataUploadState:
    """Typed server-side state associated with one NiceGUI upload element."""

    locked: bool
    remove_controls: list[Any]
    inspecting_names: list[str]


_DATA_UPLOAD_STATES: weakref.WeakKeyDictionary[Any, _DataUploadState] = weakref.WeakKeyDictionary()


def _make_upload_row_drop_target(
    row: Any,
    uploader: Any,
    *,
    multiple: bool,
    filename: str,
) -> None:
    """Forward files dropped on a completed row to its hidden uploader.

    Single-file workflow controls hide Quasar's empty upload surface after a
    successful inspection.  Keeping the completed filename row as a native
    drop target lets users replace that file directly without first clearing
    it.  Quasar still owns acceptance, size checks, and auto-upload; this
    client-side bridge forwards the browser ``FileList`` through Quasar's
    native validation and auto-upload path.
    """

    row.classes("dn-upload-file-drop-target").props(
        'data-upload-drop-target="true"'
    )
    # The native browser tooltip is intentionally scoped to the completed
    # filename row. It exposes truncated names without adding layout or a
    # tooltip over the surrounding upload surface.
    row._props["title"] = filename
    row.on(
        "dragenter",
        js_handler=(
            "event => { "
            "if (!Array.from(event.dataTransfer?.types || []).includes('Files')) return; "
            "event.preventDefault(); "
            "}"
        ),
    )
    row.on(
        "dragover",
        js_handler=(
            "event => { "
            "if (!Array.from(event.dataTransfer?.types || []).includes('Files')) return; "
            "event.preventDefault(); "
            "event.dataTransfer.dropEffect = 'copy'; "
            "}"
        ),
    )
    row.on(
        "drop",
        js_handler=(
            "event => { "
            "event.preventDefault(); event.stopPropagation(); "
            "const files = Array.from(event.dataTransfer?.files || []); "
            f"const q = getElement({uploader.id})?.$refs?.qRef; "
            "if (!q || q.disable || files.length === 0) return; "
            f"{'q.removeUploadedFiles(); ' if multiple else 'q.reset(); '}"
            f"{'q.addFiles(files); ' if multiple else 'q.addFiles(files.slice(0, 1)); '}"
            "}"
        ),
    )


def page_intro(kicker: str, title: str, body: str) -> None:
    """Render the heading shared by every workflow page."""

    with ui.element("section").classes("dn-hero"):
        ui.label(kicker).classes("dn-eyebrow")
        ui.label(title).classes("dn-page-title")
        ui.label(body).classes("dn-lead")


def panel_title(title: str) -> None:
    ui.label(title).classes("dn-panel-title")


def muted(text: str) -> None:
    ui.label(text).classes("dn-muted")


@contextmanager
def panel(*, extra_classes: str = "") -> Iterator[None]:
    """Render a visual card using the DNAscape panel style."""

    classes = f"dn-card {extra_classes}".strip()
    with ui.card().classes(classes):
        yield


@contextmanager
def expansion(
    title: str,
    *,
    expanded: bool = False,
    icon: str | None = None,
    extra_classes: str = "",
) -> Iterator[None]:
    """Render a compact workflow section."""

    classes = f"dn-expansion {extra_classes}".strip()
    custom_data_icon = icon == DATA_ICON
    section = (
        ui.expansion(title, value=expanded, icon=None if custom_data_icon else icon)
        .classes(classes)
        .props("dense")
    )
    if custom_data_icon:
        with section.add_slot("header"):
            with ui.item_section().props("avatar"):
                ui.html(DATA_ICON_SVG).classes("dn-expansion-data-icon dn-outline-icon").mark(
                    "data-outline-icon"
                )
            with ui.item_section():
                ui.item_label(title)
    with section:
        with ui.column().classes("dn-expansion-content"):
            yield


def inert_button(
    label: str,
    *,
    icon: str | None = None,
    primary: bool = False,
    compact: bool = False,
) -> Any:
    """Create a deliberately inert prototype action button."""

    button = ui.button(label, icon=icon).props("no-caps")
    marker = "-".join(label.casefold().replace("/", " ").split())
    button.mark(f"action-{marker}")
    button.classes("dn-action-primary" if primary else "dn-action-secondary")
    if compact:
        button.classes("dn-action-compact")
    return button


def plot_action_button() -> Any:
    """Create the Plot action with the shared axis-and-line symbol."""

    with (
        ui.button().props("no-caps").classes("dn-action-primary dn-plot-action").mark("action-plot")
    ) as button:
        ui.html(PLOT_ICON_SVG).classes("dn-action-plot-icon dn-plot-icon-wrap")
        ui.label("Plot").classes("dn-action-label")
    return button


def plot_toolbar(
    *,
    on_zoom: Callable[[], Any] | None = None,
    on_download: Callable[[str], Any] | None = None,
) -> None:
    """Render the shared hover toolbar used by every scientific preview."""

    with ui.row().classes("dn-plot-toolbar").mark("plot-toolbar"):
        zoom = (
            ui.button(icon="fit_screen", on_click=on_zoom)
            .props('flat round dense aria-label="Fit plot to window"')
            .classes("dn-plot-tool")
            .mark("action-fit-plot-to-window")
        )
        zoom.tooltip("Fit plot to window")
        with (
            ui.button(icon="download")
            .props('flat round dense aria-label="Download plot"')
            .classes("dn-plot-tool dn-plot-download-trigger")
            .mark("action-download-plot")
        ):
            ui.tooltip("Download plot")
            with (
                ui.menu()
                .props('anchor="bottom right" self="top right"')
                .classes("dn-plot-download-menu")
                .mark("plot-download-menu")
            ):
                for file_format in ("PNG", "SVG", "PDF"):
                    handler = None if on_download is None else partial(on_download, file_format)
                    (
                        ui.menu_item(file_format, on_click=handler)
                        .classes("dn-plot-download-item")
                        .mark(f"download-plot-{file_format.casefold()}")
                    )


@contextmanager
def action_dock() -> Iterator[None]:
    """Keep a workflow's primary action in a predictable card footer."""

    with ui.element("div").classes("dn-action-dock").mark("scientific-action-dock"):
        yield
        (
            ui.button("Code", color="#2596be")
            .props('no-caps icon-right="arrow_drop_down"')
            .classes("dn-action-primary dn-action-code")
            .mark("action-code")
            .bind_visibility_from(get_export_settings(), "dev_mode")
        )


def set_data_upload_locked(uploader: Any, locked: bool) -> None:
    """Atomically freeze or release an upload surface and its remove actions."""

    state = _DATA_UPLOAD_STATES.get(uploader)
    if state is None:
        raise ValueError("The supplied control is not a DNAscape data upload surface.")
    state.locked = bool(locked)
    uploader.disable() if locked else uploader.enable()
    removals_locked = bool(locked or state.inspecting_names)
    for control in tuple(state.remove_controls):
        if getattr(control, "is_deleted", False):
            continue
        control.disable() if removals_locked else control.enable()


def is_data_upload(control: Any) -> bool:
    """Return whether ``control`` was created by :func:`data_upload`."""

    try:
        return control in _DATA_UPLOAD_STATES
    except TypeError:
        return False


def data_upload(
    *,
    accept: str,
    accepted_formats: str,
    marker: str | None = None,
    show_accepted_formats: bool = True,
    on_upload: Callable[[events.UploadEventArguments], Any] | Callable[[], Any] | None = None,
    on_clear: Callable[[UploadClearEventArguments], Any] | Callable[[], Any] | None = None,
    on_rejected: Callable[[events.UiEventArguments], Any] | Callable[[], Any] | None = None,
    multiple: bool = False,
    max_file_size: int = MAX_UPLOAD_BYTES,
    max_total_size: int = MAX_DECOMPRESSED_BYTES,
    max_files: int | None = None,
    hide_when_loaded: bool = False,
    initial_names: Sequence[str] = (),
) -> Any:
    """Render the shared drag-and-drop surface with immediate upload.

    Quasar's completed queue exposes byte counts and a passive status glyph.
    DNAscape replaces it with a compact filename row and an explicit remove
    button. ``on_clear`` is the authoritative hook for an explicit removal;
    accepted replacements are committed by ``on_upload`` without briefly
    clearing the previous page state.
    """

    uploaded_names: list[str] = list(dict.fromkeys(str(name) for name in initial_names))
    inspecting_names: list[str] = []
    upload_lock = asyncio.Lock()

    async def invoke_callback(callback: Callable[..., Any] | None, argument: Any) -> Any:
        if callback is None:
            return None
        result = callback(argument) if helpers.expects_arguments(callback) else callback()
        if helpers.should_await(result):
            return await result
        return result

    async def handle_rejected(event: events.UiEventArguments) -> None:
        ui.notify(
            "This file type or size is not accepted here.",
            type="warning",
        )
        await invoke_callback(on_rejected, event)

    with ui.element("div").classes("dn-upload-field"):
        accept_tokens = [token.strip() for token in accept.split(",") if token.strip()]
        if ".csv" in accept_tokens and "text/csv" not in accept_tokens:
            accept_tokens.append("text/csv")
        normalized_accept = ",".join(accept_tokens)
        upload_prompt = (
            "Choose files or drag them here"
            if multiple
            else "Choose a file or drag it here"
        )
        uploader = (
            ui.upload(
                label=upload_prompt,
                auto_upload=True,
                multiple=multiple,
                max_file_size=max_file_size,
                max_total_size=max_total_size,
                max_files=max_files,
                on_rejected=handle_rejected,
            )
            .props(
                'flat bordered tabindex="0" role="button" '
                f'aria-label="{upload_prompt}"'
            )
            .classes(
                "dn-upload-compact dn-upload-multiple"
                if multiple
                else "dn-upload-compact"
            )
        )
        uploader._props["accept"] = normalized_accept
        if marker:
            uploader.mark(marker)
        upload_state = _DataUploadState(
            locked=False,
            remove_controls=[],
            inspecting_names=inspecting_names,
        )
        _DATA_UPLOAD_STATES[uploader] = upload_state
        # Quasar does not finish its XHR entry until after the Python upload
        # callback and HTTP response complete. Resetting the finished queue
        # here (rather than inside that callback) also clears failed entries
        # and keeps max_files=1 uploaders ready for the next click or drop.
        uploader.on(
            "finish",
            js_handler=(f"() => runMethod({uploader.id}, 'reset', [])"),
        )
        uploader.on(
            "click",
            js_handler=(
                "event => { if (event.target.closest('input, button')) return; "
                "event.currentTarget.querySelector('input[type=file]')?.click(); }"
            ),
        )
        uploader.on(
            "keydown",
            js_handler=(
                "event => { if (event.key !== 'Enter' && event.key !== ' ') return; "
                "event.preventDefault(); "
                "event.currentTarget.querySelector('input[type=file]')?.click(); }"
            ),
        )
        with ui.element("div").classes("dn-upload-files"):

            @ui.refreshable
            def uploaded_file_rows() -> None:
                uploader.visible = not (hide_when_loaded and (uploaded_names or inspecting_names))
                upload_state.remove_controls.clear()
                for name in inspecting_names:
                    row_marker = f"{marker}-inspecting" if marker else None
                    with ui.element("div").classes(
                        "dn-upload-file dn-upload-file-inspecting"
                    ) as row:
                        if row_marker:
                            row.mark(row_marker)
                        ui.spinner(size="xs").classes("dn-upload-file-spinner")
                        ui.label(name).classes("dn-upload-file-name")
                        ui.label("Inspecting…").classes("dn-upload-file-state")
                for name in uploaded_names:
                    row_marker = f"{marker}-file" if marker else None
                    remove_marker = f"{marker}-remove" if marker else None

                    def remove_current_file(
                        _event: Any = None,
                        upload_name: str = name,
                    ) -> None:
                        background_tasks.create(remove_uploaded_file(upload_name))

                    with ui.element("div").classes("dn-upload-file") as row:
                        if row_marker:
                            row.mark(row_marker)
                        _make_upload_row_drop_target(
                            row,
                            uploader,
                            multiple=multiple,
                            filename=name,
                        )
                        ui.icon("description").classes("dn-upload-file-icon")
                        ui.label(name).classes("dn-upload-file-name")
                        remove = (
                            ui.button(icon="close", on_click=remove_current_file)
                            .props(
                                'flat round dense aria-label="Remove uploaded file" '
                                'data-upload-status="complete"'
                            )
                            .classes("dn-upload-remove")
                        )
                        if remove_marker:
                            remove.mark(remove_marker)
                        remove._props["data-upload-name"] = name
                        upload_state.remove_controls.append(remove)
                        remove.bind_enabled_from(uploader, "enabled")
                        if upload_state.locked or inspecting_names:
                            remove.disable()
                        remove.tooltip(f"Remove {name}")
                if multiple and len(uploaded_names) > 1:

                    def remove_all_current_files(_event: Any = None) -> None:
                        background_tasks.create(remove_all_uploaded_files())

                    with ui.element("div").classes("dn-upload-delete-all-row"):
                        remove_all = (
                            ui.button(
                                "Delete all",
                                icon="delete_sweep",
                                on_click=remove_all_current_files,
                            )
                            .props(
                                'flat no-caps dense aria-label="Remove all uploaded files"'
                            )
                            .classes("dn-upload-delete-all")
                        )
                        if marker:
                            remove_all.mark(f"{marker}-delete-all")
                        upload_state.remove_controls.append(remove_all)
                        remove_all.bind_enabled_from(uploader, "enabled")
                        if upload_state.locked or inspecting_names:
                            remove_all.disable()
                        remove_all.tooltip(f"Remove all {len(uploaded_names)} files")

            async def remove_uploaded_file(name: str) -> None:
                # The uploader is disabled while a scientific worker owns its
                # retained bytes.  Mirror that state server-side as well as on
                # the visible remove button so a queued click cannot mutate an
                # in-flight Mapping request.
                if (
                    upload_state.locked
                    or inspecting_names
                    or not uploader.enabled
                    or name not in uploaded_names
                ):
                    return
                uploaded_names.remove(name)
                uploader.run_method("removeUploadedFiles")
                await invoke_callback(
                    on_clear,
                    UploadClearEventArguments(
                        sender=uploader,
                        client=uploader.client,
                        name=name,
                        names=(name,),
                    ),
                )
                if not uploader.is_deleted:
                    uploaded_file_rows.refresh()

            async def remove_all_uploaded_files() -> None:
                # Clear one stable snapshot of completed rows. Removal actions
                # remain disabled during inspection so a same-name replacement
                # cannot be mistaken for the completed file it supersedes.
                if (
                    upload_state.locked
                    or inspecting_names
                    or not uploader.enabled
                    or len(uploaded_names) < 2
                ):
                    return
                removed_names = tuple(uploaded_names)
                uploaded_names.clear()
                uploader.run_method("removeUploadedFiles")
                try:
                    await invoke_callback(
                        on_clear,
                        UploadClearEventArguments(
                            sender=uploader,
                            client=uploader.client,
                            name="",
                            names=removed_names,
                        ),
                    )
                finally:
                    if not uploader.is_deleted:
                        uploaded_file_rows.refresh()

            uploaded_file_rows()

        async def handle_upload(event: events.UploadEventArguments) -> None:
            # NiceGUI dispatches files in a multiple-upload batch as individual
            # event tasks. Serialize them to preserve the user's file order
            # through inspection, state storage, and scientific pairing.
            async with upload_lock:
                allow_multiple = bool(uploader._props.get("multiple", multiple))
                # Keep the existing completed row and page state in place while
                # the candidate is inspected. A rejected replacement therefore
                # leaves the last valid source untouched.
                inspecting_names.append(event.file.name)
                uploaded_file_rows.refresh()
                try:
                    accepted = await invoke_callback(on_upload, event)
                except Exception:
                    inspecting_names.remove(event.file.name)
                    if not uploader.is_deleted:
                        uploaded_file_rows.refresh()
                    raise
                inspecting_names.remove(event.file.name)
                if uploader.is_deleted:
                    return
                # A callback can explicitly reject a file after inspecting its
                # contents.  Exceptions likewise stop here, so an invalid file can
                # never acquire the completed/check presentation.
                if accepted is False:
                    uploaded_file_rows.refresh()
                    return
                if allow_multiple:
                    # Replacing a same-name item must retain its position: the
                    # scientific batch pairer uses this stable upload order.
                    if event.file.name not in uploaded_names:
                        uploaded_names.append(event.file.name)
                else:
                    uploaded_names.clear()
                    uploaded_names.append(event.file.name)
                uploaded_file_rows.refresh()

        uploader.on_upload(handle_upload)
        if show_accepted_formats:
            ui.label(f"Accepted: {accepted_formats}").classes("dn-upload-formats")
    return uploader


def source_input(
    label: str,
    *,
    default_mode: str = "Profile",
    constant: float = 1.0,
    units: str = "",
    show_heading: bool = True,
) -> SourceInputControls:
    """Render a compact scalar/profile input and return its live controls."""

    marker = "-".join(label.casefold().replace("/", " ").split())
    with ui.element("div").classes("dn-source-compact").mark(f"source-{marker}"):
        with ui.element("div").classes("dn-source-head"):
            if show_heading:
                ui.label(label).classes("dn-field-heading")
            mode = (
                ui.select(
                    ["Profile", "Constant"],
                    value=default_mode,
                    label="Input type",
                )
                .props("dense options-dense")
                .classes("dn-source-mode")
                .mark(f"source-{marker}-mode")
            )
        with (
            ui.element("div")
            .classes("dn-source-fields dn-source-profile-fields")
            .mark(f"source-{marker}-profile")
        ) as profile_fields:
            ui.select(
                ["Upload a profile", "Prepared data or result"],
                value="Upload a profile",
                label="Profile source",
            ).props("dense options-dense").classes("dn-field")
            data_upload(
                accept=".csv,.tsv,.txt,.npy,.npz,.gz",
                accepted_formats="CSV, TSV, TXT, NPY, NPZ, GZ",
                marker=f"source-{marker}-upload",
            )
        with ui.element("div").classes("dn-source-fields") as constant_fields:
            constant_input = (
                ui.number(
                    label=f"Constant value{f' ({units})' if units else ''}",
                    value=constant,
                    min=0,
                    step=0.001 if constant < 0.1 else 0.1,
                )
                .props("dense")
                .classes("dn-field")
                .mark(f"source-{marker}-constant")
            )
        profile_fields.bind_visibility_from(
            mode, "value", backward=lambda value: value == "Profile"
        )
        constant_fields.bind_visibility_from(
            mode, "value", backward=lambda value: value == "Constant"
        )
    return SourceInputControls(mode=mode, constant=constant_input)


def preview_plot(
    title: str,
    *,
    kind: str = "line",
    caption: str | None = None,
) -> None:
    """Render a compact placeholder until a workflow has produced real data."""

    del kind  # Retained for backwards-compatible page calls.
    with ui.element("div").classes("dn-preview-guidance").mark("preview-guidance"):
        ui.label(title).classes("dn-preview-guidance-title")
        if caption:
            ui.label(caption).classes("dn-preview-guidance-copy")


def stat_strip(items: Sequence[tuple[str, str]]) -> dict[str, Any]:
    """Render compact metadata and return value labels for optional live updates."""

    values: dict[str, Any] = {}
    with ui.element("div").classes("dn-stat-grid"):
        for label, value in items:
            with ui.element("div").classes("dn-stat"):
                ui.label(label).classes("dn-stat-label")
                values[label] = ui.label(value).classes("dn-stat-value")
    return values


def raw_output_file_row(
    *,
    file_name: str,
    default_type: str = "NPY",
    default_compress: bool = True,
    on_download: Callable[[RawOutputControls], Any] | None = None,
    on_remove: Callable[[RawOutputControls], Any] | None = None,
    markers: Mapping[str, str] | None = None,
    extra_classes: str = "",
) -> RawOutputControls:
    """Render one editable raw-data artifact row without a surrounding section."""

    resolved_markers = {
        "row": "raw-output-row",
        "compression": "raw-output-compression",
        "download": "action-download-raw-data",
        "remove": "action-remove-raw-data",
        **dict(markers or {}),
    }
    removable_class = "dn-raw-output-row-removable" if on_remove is not None else ""
    row_classes = (
        f"dn-download-file-row dn-raw-output-row {removable_class} {extra_classes}"
    ).strip()
    with ui.element("div").classes(row_classes) as row:
        if marker := resolved_markers.get("row"):
            row.mark(marker)
        ui.icon("description").classes("dn-download-file-icon")
        with ui.element("div").classes("dn-raw-output-fields"):
            file_name_control = (
                ui.input(label="File name", value=file_name)
                .classes("dn-field dn-raw-output-name")
                .props("dense")
            )
            if marker := resolved_markers.get("file_name"):
                file_name_control.mark(marker)
            file_type_control = (
                ui.select(
                    list(RAW_DATA_FORMATS),
                    value=default_type,
                    label="File type",
                )
                .classes("dn-field dn-raw-output-type")
                .props("dense options-dense")
            )
            if marker := resolved_markers.get("file_type"):
                file_type_control.mark(marker)
        compress_control = ui.checkbox("Compress", value=default_compress).classes(
            "dn-raw-output-compress"
        )
        if marker := resolved_markers.get("compression"):
            compress_control.mark(marker)
        download_button = (
            ui.button(icon="download")
            .props('flat round dense aria-label="Download raw data"')
            .classes("dn-download-file-action dn-raw-output-download")
        )
        if marker := resolved_markers.get("download"):
            download_button.mark(marker)
        download_button.tooltip("Download")
        remove_button = None
        if on_remove is not None:
            remove_button = (
                ui.button(icon="close")
                .props('flat round dense aria-label="Remove raw data"')
                .classes("dn-download-file-action dn-raw-output-remove")
            )
            if marker := resolved_markers.get("remove"):
                remove_button.mark(marker)
            remove_button.tooltip("Remove")
    controls = RawOutputControls(
        file_name=file_name_control,
        file_type=file_type_control,
        compress=compress_control,
        download_button=download_button,
        row=row,
        remove_button=remove_button,
    )
    if on_download is not None:
        download_button.on_click(lambda: on_download(controls))
    if remove_button is not None:
        remove_button.on_click(lambda: on_remove(controls))
    return controls


def raw_output_controls(
    *,
    file_name: str,
    default_type: str = "NPY",
    default_compress: bool = True,
    on_download: Callable[[RawOutputControls], Any] | None = None,
    on_remove: Callable[[RawOutputControls], Any] | None = None,
) -> RawOutputControls:
    """Render the shared raw-data export section and an optional extension slot."""

    with expansion(
        "Raw data",
        icon=DATA_ICON,
        extra_classes="dn-output-section",
    ):
        controls = raw_output_file_row(
            file_name=file_name,
            default_type=default_type,
            default_compress=default_compress,
            on_download=on_download,
            on_remove=on_remove,
        )
        extra_slot = ui.element("div").classes("dn-raw-output-extra")
    return RawOutputControls(
        file_name=controls.file_name,
        file_type=controls.file_type,
        compress=controls.compress,
        download_button=controls.download_button,
        row=controls.row,
        remove_button=controls.remove_button,
        extra_slot=extra_slot,
    )


def empty_state(title: str, body: str, *, icon: str = "science") -> None:
    with ui.element("div").classes("dn-empty-state"):
        ui.icon(icon).classes("dn-empty-icon")
        ui.label(title).classes("dn-empty-title")
        muted(body)


def mini_table(
    columns: Sequence[tuple[str, str]],
    rows: Sequence[dict[str, Any]],
) -> None:
    """Render a compact, synthetic preview table."""

    formatted_columns = [
        {"name": name, "label": label, "field": name, "align": "left"} for name, label in columns
    ]
    ui.table(columns=formatted_columns, rows=list(rows), row_key=columns[0][0]).props(
        "flat bordered dense hide-bottom"
    ).classes("dn-table")


def two_column_fields() -> Any:
    """Return a responsive two-column field grid."""

    return ui.element("div").classes("dn-field-grid")


def three_column_fields() -> Any:
    """Return a responsive three-column field grid."""

    return ui.element("div").classes("dn-field-grid dn-field-grid-three")
