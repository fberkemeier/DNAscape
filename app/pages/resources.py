"""Project links and publication citation."""

from __future__ import annotations

from collections.abc import Callable

from nicegui import ui

from app import __version__
from app.icons import GITHUB_ICON_SVG
from app.pages.common import muted, page_intro, panel, panel_title

GITHUB_URL = "https://github.com/fberkemeier/DNAscape"
DOCS_URL = "https://dnascape.readthedocs.io/en/latest/"
REPLICANVAS_URL = "https://fberkemeier.github.io/RepliCanvas/"
SOFTWARE_VERSION = __version__.removesuffix("-prototype")

CITATION_TITLE = "DNAscape: Software for mapping and simulating DNA replication kinetics"
CITATION_AUTHORS = "Francisco Berkemeier and Michael A. Boemo"
REPOSITORY_CITATION_TEXT = (
    f"{CITATION_AUTHORS}. {CITATION_TITLE} (Version {SOFTWARE_VERSION}) "
    f"[Computer software]. GitHub. {GITHUB_URL}"
)
REPOSITORY_CITATION_BIBTEX = f"""@software{{berkemeier_dnascape,
  author = {{Berkemeier, Francisco and Boemo, Michael A.}},
  title = {{{{DNAscape}}: Software for mapping and simulating DNA replication kinetics}},
  version = {{{SOFTWARE_VERSION}}},
  url = {{{GITHUB_URL}}}
}}
"""
CITATION_TEXT = f"{CITATION_AUTHORS} (2026). {CITATION_TITLE}. bioRxiv."
CITATION_BIBTEX = """@article{berkemeier2026dnascape,
  author = {Berkemeier, Francisco and Boemo, Michael A.},
  title = {{DNAscape}: Software for mapping and simulating DNA replication kinetics},
  journal = {bioRxiv},
  year = {2026}
}
"""


def render(_navigate: Callable[[str], None]) -> None:
    """Render project links, RepliCanvas, and software/preprint citations."""

    page_intro(
        "Resources",
        "Project links and citation guidance",
        "Source code, documentation, illustration tools, and citation guidance.",
    )
    with ui.element("div").classes("dn-resource-grid"):
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
            panel_title("Art")
            with (
                ui.link(target=REPLICANVAS_URL, new_tab=True)
                .props('aria-label="Open RepliCanvas"')
                .classes("dn-resource-link")
                .mark("resource-replicanvas")
            ):
                (
                    ui.element("span")
                    .props('role="img" aria-label="RepliCanvas"')
                    .classes("dn-resource-link-icon dn-resource-replicanvas-logo")
                    .mark("resource-replicanvas-logo")
                )
                ui.label("RepliCanvas").classes("dn-resource-link-label")
            muted("Make publication-ready illustrations of DNA replication.")
        with panel(extra_classes="dn-resource-citation-panel"):
            panel_title("Citation")
            muted(
                "If you use DNAscape in your research, please cite the software "
                "and include the software version and model settings used."
            )
            ui.html(
                "On GitHub, select Cite this repository for APA or BibTeX, or view "
                f'<a href="{GITHUB_URL}/blob/main/CITATION.cff" '
                'target="_blank" rel="noopener noreferrer">CITATION.cff</a>.',
                tag="p",
            ).classes("dn-muted dn-resource-citation-guidance")
            with ui.element("blockquote").classes("dn-resource-citation"):
                ui.label(REPOSITORY_CITATION_TEXT).mark("resource-repository-citation-text")
            with ui.element("div").classes("dn-inline-actions"):
                (
                    ui.button(
                        "Copy repository citation", icon="content_copy",
                        on_click=lambda: ui.clipboard.write(REPOSITORY_CITATION_TEXT),
                    )
                    .props("no-caps")
                    .classes("dn-action-secondary dn-action-compact")
                    .mark("action-copy-repository-citation")
                )
                (
                    ui.button(
                        "Download repository BibTeX", icon="download",
                        on_click=lambda: ui.download(
                            REPOSITORY_CITATION_BIBTEX.encode("utf-8"),
                            "dnascape-software.bib", "application/x-bibtex",
                        ),
                    )
                    .props("no-caps")
                    .classes("dn-action-secondary dn-action-compact")
                    .mark("action-download-repository-bibtex")
                )
            muted("Once available, please also cite the accompanying preprint:")
            with ui.element("blockquote").classes("dn-resource-citation"):
                ui.label(CITATION_TEXT).mark("resource-citation-text")
            with ui.element("div").classes("dn-inline-actions"):
                (
                    ui.button(
                        "Copy preprint citation", icon="content_copy",
                        on_click=lambda: ui.clipboard.write(CITATION_TEXT),
                    )
                    .props("no-caps")
                    .classes("dn-action-secondary dn-action-compact")
                    .mark("action-copy-citation")
                )
                (
                    ui.button(
                        "Download preprint BibTeX", icon="download",
                        on_click=lambda: ui.download(
                            CITATION_BIBTEX.encode("utf-8"), "dnascape.bib", "application/x-bibtex",
                        ),
                    )
                    .props("no-caps")
                    .classes("dn-action-secondary dn-action-compact")
                    .mark("action-download-bibtex")
                )
