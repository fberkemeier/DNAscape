"""Project links and publication citation."""

from __future__ import annotations

from collections.abc import Callable

from nicegui import ui

from app.icons import GITHUB_ICON_SVG
from app.pages.common import muted, page_intro, panel, panel_title

GITHUB_URL = "https://github.com/fberkemeier/DNAscape"
DOCS_URL = "https://dnascape.readthedocs.io/en/latest/"

CITATION_TITLE = "DNAscape: Software for mapping and simulating DNA replication kinetics"
CITATION_AUTHORS = "Francisco Berkemeier and Michael A. Boemo"
CITATION_TEXT = f"{CITATION_AUTHORS} (2026). {CITATION_TITLE}. bioRxiv."
CITATION_BIBTEX = """@article{berkemeier2026dnascape,
  author = {Berkemeier, Francisco and Boemo, Michael A.},
  title = {{DNAscape}: Software for mapping and simulating DNA replication kinetics},
  journal = {bioRxiv},
  year = {2026}
}
"""


def render(_navigate: Callable[[str], None]) -> None:
    """Render repository, documentation, and citation cards."""

    page_intro(
        "Resources",
        "Project links and citation guidance",
        "Documentation, source code, examples, and citation guidance.",
    )
    with ui.element("div").classes("dn-resource-grid"):
        with panel():
            panel_title("Documentation")
            with (
                ui.link(target=DOCS_URL, new_tab=True)
                .classes("dn-resource-link")
                .mark("resource-documentation")
            ):
                ui.icon("menu_book").classes("dn-resource-link-icon")
                ui.label("DNAscape Read the Docs").classes("dn-resource-link-label")
            muted("Installation notes, tutorials, and Python API guidance.")
        with panel():
            panel_title("Repository")
            with (
                ui.link(target=GITHUB_URL, new_tab=True)
                .classes("dn-resource-link")
                .mark("resource-repository")
            ):
                ui.html(GITHUB_ICON_SVG).classes("dn-resource-link-icon")
                ui.label("DNAscape on GitHub").classes("dn-resource-link-label")
            muted("Source code, issues, releases, and development history.")
        with panel(extra_classes="dn-resource-citation-panel"):
            panel_title("Citation")
            muted(
                "If you use DNAscape in your research, please cite the publication below "
                "and include the software version used."
            )
            with ui.element("blockquote").classes("dn-resource-citation"):
                ui.label(CITATION_TEXT).mark("resource-citation-text")
            with ui.element("div").classes("dn-inline-actions"):
                (
                    ui.button(
                        "Copy citation", icon="content_copy",
                        on_click=lambda: ui.clipboard.write(CITATION_TEXT),
                    )
                    .props("no-caps")
                    .classes("dn-action-secondary dn-action-compact")
                    .mark("action-copy-citation")
                )
                (
                    ui.button(
                        "Download BibTeX", icon="download",
                        on_click=lambda: ui.download(
                            CITATION_BIBTEX.encode("utf-8"), "dnascape.bib", "application/x-bibtex",
                        ),
                    )
                    .props("no-caps")
                    .classes("dn-action-secondary dn-action-compact")
                    .mark("action-download-bibtex")
                )
