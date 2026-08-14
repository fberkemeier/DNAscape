"""Overview page for the presentation-only NiceGUI prototype."""

from __future__ import annotations

from collections.abc import Callable

from nicegui import ui

from app.pages.common import DATA_ICON

WORKFLOWS = (
    (
        "data-management",
        "Data Manager",
        "Prepare sequencing tables, genomic tracks, and profile formats for downstream analysis.",
        DATA_ICON,
    ),
    (
        "mapping",
        "Mapping",
        "Infer spatial, temporal, or feature-dependent replication landscapes from kinetic inputs.",
        "map",
    ),
    (
        "simulation",
        "Simulation",
        "Generate replication dynamics under selected kinetic assumptions and compare outputs.",
        "monitor",
    ),
    (
        "plotting",
        "Plotting",
        "Create polished previews of input profiles, inferred features, simulations, and exports.",
        "show_chart",
    ),
    (
        "cached-files",
        "Cached Files",
        "Manage data and figure exports retained during this local application session.",
        "inventory_2",
    ),
    (
        "resources",
        "Resources",
        "Find documentation, examples, source code, and citation guidance for the project.",
        "menu_book",
    ),
)


def _navigate_handler(navigate: Callable[[str], None], slug: str):
    def handle() -> None:
        navigate(slug)

    return handle


def render(navigate: Callable[[str], None]) -> None:
    """Render the overview and its six navigable workflow cards."""

    with ui.column().classes("dn-overview"):
        ui.element("img").props(
            'src="/dnascape-assets/dnascape_logo_small.svg" alt="DNAscape" draggable=false'
        ).classes("dn-main-logo")
        with ui.column().classes("dn-overview-copy"):
            ui.label("Overview").classes("dn-eyebrow")
            ui.label(
                "Build quantitative models of DNA replication from genomic sequencing data, "
                "then move cleanly from prepared inputs to kinetic mapping and stochastic simulation."
            ).classes("dn-lead")

        with ui.element("div").classes("dn-overview-grid"):
            for slug, title, body, icon in WORKFLOWS:
                with ui.element("button").props("type=button").classes("dn-workflow-card").on(
                    "click", _navigate_handler(navigate, slug)
                ):
                    with ui.element("div"):
                        ui.label(title).classes("dn-workflow-title")
                        ui.label(body).classes("dn-workflow-copy")
                    ui.icon(icon).classes("dn-workflow-icon")
