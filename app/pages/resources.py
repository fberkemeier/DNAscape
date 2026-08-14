"""Project links and citation interface prototype."""

from __future__ import annotations

from collections.abc import Callable

from nicegui import ui

from app.pages.common import inert_button, muted, page_intro, panel, panel_title

GITHUB_URL = "https://github.com/fberkemeier/DNAscape"
DOCS_URL = "https://dnascape.readthedocs.io/en/latest/"


def render(_navigate: Callable[[str], None]) -> None:
    """Render repository, documentation, and citation cards."""

    page_intro(
        "Resources",
        "Project links and citation guidance",
        "Documentation, source code, examples, and citation guidance.",
    )
    with ui.element("div").classes("dn-resource-grid"):
        with panel():
            panel_title("Repository")
            ui.icon("code").classes("dn-empty-icon")
            ui.link("DNAscape on GitHub", GITHUB_URL, new_tab=True).classes("dn-resource-link")
            muted("Source code, issues, releases, and development history.")
        with panel():
            panel_title("Documentation")
            ui.icon("menu_book").classes("dn-empty-icon")
            ui.link("DNAscape Read the Docs", DOCS_URL, new_tab=True).classes("dn-resource-link")
            muted("Installation notes, worked examples, and Python API guidance.")
    with panel():
        panel_title("Citation")
        ui.label("Using DNAscape in research").classes("dn-subheading")
        muted(
            "Cite the relevant DNAscape publication or software release and include the version "
            "or commit used. Formal citation metadata can be added here when finalized."
        )
        with ui.element("div").classes("dn-inline-actions"):
            inert_button("Copy citation", icon="content_copy", compact=True)
            inert_button("Download BibTeX", icon="download", compact=True)
