"""Reusable presentation-only components for the NiceGUI prototype."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

from nicegui import ui

DATA_ICON = "storage"
RAW_DATA_FORMATS = ("CSV", "TSV", "TXT", "NPY", "NPZ")


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
    """Render a visual card using the Streamlit application's panel style."""

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
    with ui.expansion(title, value=expanded, icon=icon).classes(classes).props(
        "dense expand-separator"
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


def inert_icon_button(label: str, *, icon: str) -> Any:
    """Create an icon-only preview control without attaching an action."""

    marker = "-".join(label.casefold().replace("/", " ").split())
    button = (
        ui.button(icon=icon)
        .props(f'flat round dense aria-label="{label}"')
        .classes("dn-plot-tool")
        .mark(f"action-{marker}")
    )
    button.tooltip(label)
    return button


def source_input(
    label: str,
    *,
    default_mode: str = "Profile",
    constant: float = 1.0,
    units: str = "",
) -> None:
    """Render a representative scalar/profile input without retaining data."""

    marker = "-".join(label.casefold().replace("/", " ").split())
    with ui.element("div").classes("dn-source-block").mark(f"source-{marker}"):
        ui.label(label).classes("dn-field-heading")
        mode = ui.radio(["Profile", "Constant"], value=default_mode).props("inline dense")
        with ui.column().classes("w-full gap-2").mark(f"source-{marker}-profile") as profile_fields:
            ui.select(
                ["Upload a profile", "Prepared data or result"],
                value="Upload a profile",
                label="Profile source",
            ).classes("dn-field")
            ui.upload(label=f"Choose {label.lower()} file", auto_upload=False).props(
                "flat bordered accept=.csv,.tsv,.txt,.npy,.npz,.gz"
            ).classes("dn-upload-compact")
        constant_field = ui.number(
            label=f"Constant value{f' ({units})' if units else ''}",
            value=constant,
            min=0,
            step=0.001 if constant < 0.1 else 0.1,
        ).classes("dn-field").mark(f"source-{marker}-constant")
        profile_fields.bind_visibility_from(mode, "value", backward=lambda value: value == "Profile")
        constant_field.bind_visibility_from(mode, "value", backward=lambda value: value == "Constant")


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
            <g class="dn-grid-lines">
              <path d="M12 24H314M12 64H314M12 104H314M12 144H314" />
              <path d="M12 16V148M88 16V148M164 16V148M240 16V148M314 16V148" />
            </g>
            <g class="{primitive_class}">{geometry}</g>
          </svg>
        </div>
    """
    with ui.element("div").classes("dn-plot-shell"):
        ui.html(markup).classes("dn-chart-wrap")
        with ui.row().classes("dn-plot-toolbar"):
            inert_icon_button("Fit plot to window", icon="fit_screen")
            inert_icon_button("Download plot", icon="download")
            inert_icon_button("Clear plot", icon="delete_outline")
    if caption:
        muted(caption)


def stat_strip(items: Sequence[tuple[str, str]]) -> None:
    """Render the compact preview metadata strip shared by scientific pages."""

    with ui.element("div").classes("dn-stat-grid"):
        for label, value in items:
            with ui.element("div").classes("dn-stat"):
                ui.label(label).classes("dn-stat-label")
                ui.label(value).classes("dn-stat-value")


def raw_output_controls(
    *,
    file_name: str,
    default_type: str = "CSV",
) -> None:
    """Render the shared raw-data export surface."""

    with expansion(
        "Raw data",
        icon=DATA_ICON,
        extra_classes="dn-output-section",
    ):
        with ui.element("div").classes("dn-field-grid dn-field-grid-three"):
            ui.input(label="File name", value=file_name).classes("dn-field")
            ui.select(
                list(RAW_DATA_FORMATS),
                value=default_type,
                label="File type",
            ).classes("dn-field")
            ui.checkbox("GZ", value=False)
        inert_button("Download raw data", icon="download", compact=True)


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
        {"name": name, "label": label, "field": name, "align": "left"}
        for name, label in columns
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
