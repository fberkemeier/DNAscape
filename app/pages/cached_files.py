"""Cached-files inventory interface prototype."""

from __future__ import annotations

from collections.abc import Callable

from nicegui import ui

from app.pages.common import inert_button, muted, page_intro, panel, panel_title


def _example_row(
    name: str,
    detail: str,
    size: str,
    *,
    primary_action: str,
    secondary_action: str,
) -> None:
    with ui.element("div").classes("dn-cache-row"):
        with ui.column().classes("gap-0"):
            ui.label(name).classes("dn-cache-name")
            muted(detail)
        with ui.column().classes("gap-0"):
            muted("Size")
            ui.label(size)
        ui.select(["CSV", "NPY", "NPZ"], value="CSV", label="Format").classes("dn-field")
        with ui.element("div").classes("dn-inline-actions"):
            inert_button(primary_action, icon="download", compact=True)
            inert_button(secondary_action, icon="delete", compact=True)


def render(_navigate: Callable[[str], None]) -> None:
    """Render generated and retained storage controls using example-only rows."""

    page_intro(
        "Cached Files",
        "Manage generated files",
        "Review and download data and figures saved during this DNAscape session.",
    )
    with panel():
        panel_title("Cache settings")
        with ui.element("div").classes("dn-cache-controls"):
            ui.number(label="Max. files", value=30, min=1, max=200, step=1).classes("dn-field")
            with ui.column().classes("gap-0"):
                ui.checkbox("Raw output", value=False)
                ui.checkbox("Plots", value=False)
            with ui.element("div").classes("dn-inline-actions"):
                inert_button("Date ↓", icon="schedule", compact=True)
                inert_button("Size", icon="sort", compact=True)
            inert_button("Clear generated cache", icon="delete_sweep")
        muted("1 file · 24.6 KB · date descending")
        with ui.element("div").classes("dn-cache-row"):
            with ui.column().classes("gap-0"):
                ui.label("dnascape_plot.png").classes("dn-cache-name")
                muted("Plotting · Plot")
            with ui.column().classes("gap-0"):
                muted("Created")
                ui.label("This session")
            ui.select(["PNG", "SVG", "PDF"], value="PNG", label="Format").classes("dn-field")
            ui.checkbox("GZ", value=False)
            with ui.element("div").classes("dn-inline-actions"):
                inert_button("Download", icon="download", compact=True)
                inert_button("Remove", icon="delete", compact=True)

    ui.label("Retained session data").classes("dn-subheading")
    muted(
        "Uploads, prepared datasets, and completed workflow results are retained independently "
        "from generated download files."
    )
    with ui.tabs().classes("dn-tabs") as tabs:
        uploads = ui.tab("Uploads", icon="upload_file")
        datasets = ui.tab("Prepared datasets", icon="dataset")
        workflows = ui.tab("Workflow results", icon="science")
    with ui.tab_panels(tabs, value=uploads, animated=True).classes("dn-tab-panels"):
        with ui.tab_panel(uploads):
            _example_row(
                "profile.csv",
                "SHA-256 available after upload",
                "18.2 KB",
                primary_action="Download",
                secondary_action="Remove",
            )
        with ui.tab_panel(datasets):
            _example_row(
                "profile · space",
                "Canonical dataset and source identifiers will appear here",
                "31.4 KB",
                primary_action="Open plot",
                secondary_action="Remove",
            )
        with ui.tab_panel(workflows):
            _example_row(
                "Simulation result",
                "Input signature and completion time will appear here",
                "2.1 MB",
                primary_action="Open",
                secondary_action="Remove result",
            )
    with ui.element("div").classes("dn-cache-controls"):
        ui.select([10, 25, 50], value=10, label="Rows per page").classes("dn-field")
        ui.number(label="Page", value=1, min=1, step=1).classes("dn-field")
