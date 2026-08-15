"""Reusable components for the local NiceGUI interface."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from typing import Any

from nicegui import ui

DATA_ICON = "storage"
RAW_DATA_FORMATS = ("CSV", "TSV", "TXT", "NPY", "NPZ")
PLOT_ICON_SVG = """
<svg class="dn-plot-icon" viewBox="0 0 24 24" aria-hidden="true" focusable="false">
  <path class="dn-plot-icon-axis" d="M4 3.5V20h16.5" />
  <path class="dn-plot-icon-line" d="m7 16 4-5 3.2 3.1L19 7.5" />
  <circle class="dn-plot-icon-point" cx="7" cy="16" r="1" />
  <circle class="dn-plot-icon-point" cx="11" cy="11" r="1" />
  <circle class="dn-plot-icon-point" cx="14.2" cy="14.1" r="1" />
  <circle class="dn-plot-icon-point" cx="19" cy="7.5" r="1" />
</svg>
"""


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
    gzip_output: Any
    download_button: Any


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
    with (
        ui.expansion(title, value=expanded, icon=icon)
        .classes(classes)
        .props("dense expand-separator")
    ):
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
                    handler = (
                        None if on_download is None else partial(on_download, file_format)
                    )
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


def data_upload(
    *,
    accept: str,
    accepted_formats: str,
    marker: str | None = None,
) -> Any:
    """Render the shared, drag-aware data upload surface."""

    with ui.element("div").classes("dn-upload-field"):
        uploader = (
            ui.upload(label="Upload data", auto_upload=False)
            .props(f"flat bordered accept={accept}")
            .classes("dn-upload-compact")
        )
        if marker:
            uploader.mark(marker)
        uploader.on(
            "dragenter",
            js_handler="event => event.currentTarget.classList.add('dn-upload-dragging')",
        )
        uploader.on(
            "dragleave",
            js_handler=(
                "event => { if (!event.currentTarget.contains(event.relatedTarget)) "
                "event.currentTarget.classList.remove('dn-upload-dragging'); }"
            ),
        )
        uploader.on(
            "drop",
            js_handler="event => event.currentTarget.classList.remove('dn-upload-dragging')",
        )
        ui.label(f"Accepted: {accepted_formats}").classes("dn-upload-formats")
    return uploader


def source_input(
    label: str,
    *,
    default_mode: str = "Profile",
    constant: float = 1.0,
    units: str = "",
) -> SourceInputControls:
    """Render a compact scalar/profile input and return its live controls."""

    marker = "-".join(label.casefold().replace("/", " ").split())
    with ui.element("div").classes("dn-source-compact").mark(f"source-{marker}"):
        with ui.element("div").classes("dn-source-head"):
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
    """Draw a lightweight, clearly labelled prototype chart."""

    paths = {
        "line": '<polyline points="14,132 55,112 94,119 135,73 178,89 220,46 266,62 308,30"/>',
        "mapping": '<polyline points="14,126 52,118 92,97 130,105 168,62 206,75 248,36 308,48"/>',
        "simulation": '<polyline points="14,139 45,128 78,132 109,105 142,112 176,76 210,83 244,46 278,54 308,24"/>',
        "histogram": (
            '<rect x="24" y="108" width="24" height="38"/><rect x="58" y="82" width="24" height="64"/>'
            '<rect x="92" y="48" width="24" height="98"/><rect x="126" y="24" width="24" height="122"/>'
            '<rect x="160" y="41" width="24" height="105"/><rect x="194" y="72" width="24" height="74"/>'
            '<rect x="228" y="102" width="24" height="44"/><rect x="262" y="124" width="24" height="22"/>'
        ),
    }
    geometry = paths.get(kind, paths["line"])
    primitive_class = "dn-bars" if kind == "histogram" else "dn-line"
    markup = f"""
        <div class="dn-chart" role="img" aria-label="{title} placeholder">
          <div class="dn-chart-label">{title}</div>
          <svg viewBox="0 0 324 168" preserveAspectRatio="none" aria-hidden="true">
            <g class="{primitive_class}">{geometry}</g>
          </svg>
        </div>
    """
    with ui.dialog() as zoom_dialog, ui.card().classes("dn-plot-dialog"):
        with ui.row().classes("dn-plot-dialog-header"):
            ui.label(title).classes("dn-panel-title")
            ui.space()
            ui.button(icon="close", on_click=zoom_dialog.close).props(
                'flat round dense aria-label="Close enlarged plot"'
            )
        ui.html(markup).classes("dn-chart-wrap dn-chart-wrap-large")
    with ui.element("div").classes("dn-plot-shell"):
        ui.html(markup).classes("dn-chart-wrap")
        plot_toolbar(on_zoom=zoom_dialog.open)
    if caption:
        muted(caption)


def stat_strip(items: Sequence[tuple[str, str]]) -> dict[str, Any]:
    """Render compact metadata and return value labels for optional live updates."""

    values: dict[str, Any] = {}
    with ui.element("div").classes("dn-stat-grid"):
        for label, value in items:
            with ui.element("div").classes("dn-stat"):
                ui.label(label).classes("dn-stat-label")
                values[label] = ui.label(value).classes("dn-stat-value")
    return values


def raw_output_controls(
    *,
    file_name: str,
    default_type: str = "CSV",
    on_download: Callable[[RawOutputControls], Any] | None = None,
) -> RawOutputControls:
    """Render the shared raw-data export surface."""

    with expansion(
        "Raw data",
        icon=DATA_ICON,
        extra_classes="dn-output-section",
    ):
        with ui.element("div").classes("dn-field-grid dn-field-grid-three"):
            file_name_control = ui.input(label="File name", value=file_name).classes(
                "dn-field"
            )
            file_type_control = ui.select(
                list(RAW_DATA_FORMATS),
                value=default_type,
                label="File type",
            ).classes("dn-field")
            gzip_control = ui.checkbox("GZ", value=False)
        download_button = (
            ui.button("Download raw data", icon="download")
            .props("no-caps")
            .classes("dn-action-secondary dn-action-compact")
            .mark("action-download-raw-data")
        )
        controls = RawOutputControls(
            file_name=file_name_control,
            file_type=file_type_control,
            gzip_output=gzip_control,
            download_button=download_button,
        )
        if on_download is not None:
            download_button.on_click(lambda: on_download(controls))
        return controls


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
