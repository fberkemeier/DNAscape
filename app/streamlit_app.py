"""DNAscape Streamlit app shell and lightweight page router."""

import importlib.util
import sys
from pathlib import Path

import streamlit as st


APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import app_shell as app_shell_module

from app_shell import (
    DOCS_URL,
    GITHUB_URL,
    LOGO_PATH,
    active_page_slug,
    configure_page,
    render_main_logo,
    render_sidebar,
    set_active_page,
)


PAGE_MODULES = {
    "data-management": APP_DIR / "pages" / "1_Data_Management.py",
    "mapping": APP_DIR / "pages" / "2_Mapping.py",
    "simulation": APP_DIR / "pages" / "3_Simulation.py",
    "plotting": APP_DIR / "pages" / "4_Plotting.py",
    "cached-files": APP_DIR / "pages" / "5_Cached_Files.py",
    "resources": APP_DIR / "pages" / "6_Resources.py",
}


def render_workflow_card(title: str, body: str, page: str) -> None:
    """Render a compact workflow card as a single click target."""
    with st.container(key=f"workflow-card-{page}"):
        st.markdown(
            f"""
        <div class="workflow-card">
          <div class="workflow-card-title">{title}</div>
          <div class="workflow-card-copy">{body}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
        st.button(
            title,
            key=f"workflow-card-button-{page}",
            on_click=set_active_page,
            args=(page,),
            width="stretch",
        )


def render_overview() -> None:
    """Render the DNAscape overview page."""
    st.markdown(
        '<span class="overview-page-marker" aria-hidden="true"></span>',
        unsafe_allow_html=True,
    )

    left, center, right = st.columns([1, 2, 1])
    with center:
        render_main_logo(LOGO_PATH, "DNAscape")

    st.markdown(
        """
        <section class="hero">
          <p class="eyebrow">Overview</p>
          <p class="lead">
            Build quantitative models of DNA replication from genomic sequencing
            data, then move cleanly from prepared inputs to kinetic mapping and
            stochastic simulation.
          </p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    workflow_cards = [
        (
            "Data Management",
            "Convert sequencing data and genomic tracks into DNAscape-ready inputs.",
            "data-management",
        ),
        (
            "Mapping",
            "Explore rmap workflows for kinetic inference and feature translation.",
            "mapping",
        ),
        (
            "Simulation",
            "Set up rsim workflows for stochastic replication programme simulation.",
            "simulation",
        ),
        (
            "Plotting",
            "Build compact visual summaries for inputs, mapped features, and simulations.",
            "plotting",
        ),
        (
            "Cached Files",
            "Review reusable uploads, generated intermediates, figures, and exports.",
            "cached-files",
        ),
        (
            "Resources",
            "Open project documentation, source code, examples, and citation notes.",
            "resources",
        ),
    ]

    for row_start in range(0, len(workflow_cards), 3):
        if row_start:
            st.markdown("<div class='workflow-grid-row-spacer'></div>", unsafe_allow_html=True)

        cols = st.columns(3)
        for col, (title, body, page) in zip(cols, workflow_cards[row_start : row_start + 3]):
            with col:
                render_workflow_card(title, body, page)

    st.markdown(
        """
        <div class="workflow-line">
          <span>Sequencing tracks</span>
          <span>DNAscape inputs</span>
          <span>rmap</span>
          <span>rsim</span>
          <span>comparison</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="footer-note">
          Source code: <a href="{GITHUB_URL}">GitHub repository</a>.
          Documentation: <a href="{DOCS_URL}">Read the Docs</a>.
          If you use DNAscape in research, cite the corresponding publication or
          software release.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_page_script(slug: str) -> None:
    """Execute a legacy page script inside the persistent app shell."""
    page_path = PAGE_MODULES[slug]
    module_name = f"_dnascape_router_{slug.replace('-', '_')}"
    spec = importlib.util.spec_from_file_location(module_name, page_path)
    if spec is None or spec.loader is None:
        st.error(f"Could not load page: {page_path.name}")
        return

    def noop(*_args: object, **_kwargs: object) -> None:
        return None

    original_configure_page = app_shell_module.configure_page
    original_render_sidebar = app_shell_module.render_sidebar
    app_shell_module.configure_page = noop
    app_shell_module.render_sidebar = noop
    try:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        app_shell_module.configure_page = original_configure_page
        app_shell_module.render_sidebar = original_render_sidebar


def render_page(slug: str) -> None:
    """Dispatch the active page while leaving the left app shell stable."""
    if slug == "overview":
        render_overview()
        return

    if slug not in PAGE_MODULES:
        st.session_state["dnascape_active_page"] = "overview"
        render_overview()
        return

    render_page_script(slug)


configure_page("DNAscape")
current_page = active_page_slug()
render_sidebar(current_page, use_router=True)
render_page(current_page)
