"""Overview page for the presentation-only NiceGUI prototype."""

from __future__ import annotations

from collections.abc import Callable

from nicegui import ui

from app.icons import OUTLINE_ICON_SVGS, PLOT_ICON_SVG

WORKFLOWS = (
    (
        "data-management",
        "Data Manager",
        "Prepare sequencing tables, genomic tracks, and profile formats for downstream analysis.",
        OUTLINE_ICON_SVGS["data"],
    ),
    (
        "mapping",
        "Mapping",
        "Infer spatial, temporal, or feature-dependent replication landscapes from kinetic inputs.",
        OUTLINE_ICON_SVGS["mapping"],
    ),
    (
        "simulation",
        "Simulation",
        "Generate replication dynamics under selected kinetic assumptions and compare outputs.",
        OUTLINE_ICON_SVGS["simulation"],
    ),
    (
        "plotting",
        "Plotting",
        "Create polished previews of input profiles, inferred features, simulations, and exports.",
        None,
    ),
    (
        "cached-files",
        "Cached Files",
        "Reopen recent results and download files retained automatically in this browser tab.",
        OUTLINE_ICON_SVGS["cached"],
    ),
    (
        "resources",
        "Resources",
        "Find documentation, examples, source code, and citation guidance for the project.",
        OUTLINE_ICON_SVGS["resources"],
    ),
)


def _navigate_handler(navigate: Callable[[str], None], slug: str):
    def handle() -> None:
        navigate(slug)

    return handle


def render(navigate: Callable[[str], None]) -> None:
    """Render the overview and its six navigable workflow cards."""

    with ui.column().classes("dn-overview"):
        with ui.element("div").classes("dn-main-logo-frame"):
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
                with (
                    ui.element("button")
                    .props("type=button")
                    .classes("dn-workflow-card")
                    .on("click", _navigate_handler(navigate, slug))
                ):
                    with ui.element("div"):
                        ui.label(title).classes("dn-workflow-title")
                        ui.label(body).classes("dn-workflow-copy")
                    if slug == "plotting":
                        ui.html(PLOT_ICON_SVG).classes("dn-workflow-icon dn-plot-icon-wrap")
                    else:
                        assert icon is not None
                        ui.html(icon).classes("dn-workflow-icon dn-workflow-symbol")
