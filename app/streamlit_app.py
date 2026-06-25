"""DNAscape Streamlit app shell and lightweight page router."""

import importlib.util
import sys
from pathlib import Path

import streamlit as st


APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
for import_path in (ROOT_DIR, APP_DIR):
    import_path_text = str(import_path)
    if import_path_text not in sys.path:
        sys.path.insert(0, import_path_text)

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

WORKFLOW_CARD_ICONS = {
    "data-management": """
      <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
        <path d="M4 6c0-1.7 3.6-3 8-3s8 1.3 8 3-3.6 3-8 3-8-1.3-8-3Z" />
        <path d="M4 6v6c0 1.7 3.6 3 8 3s8-1.3 8-3V6" />
        <path d="M4 12v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6" />
      </svg>
    """,
    "mapping": """
      <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
        <path d="M9 18 3 21V6l6-3 6 3 6-3v15l-6 3-6-3Z" />
        <path d="M9 3v15" />
        <path d="M15 6v15" />
      </svg>
    """,
    "simulation": """
      <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
        <rect x="3" y="4" width="18" height="12" rx="2" />
        <path d="M8 20h8" />
        <path d="M12 16v4" />
      </svg>
    """,
    "plotting": """
      <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
        <path d="M4 19V5" />
        <path d="M4 19h16" />
        <path d="m7 15 3.3-4 3.2 2.6L18 8" />
      </svg>
    """,
    "cached-files": """
      <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
        <path d="M4 7h16" />
        <path d="M6 7V5c0-1.1.9-2 2-2h8c1.1 0 2 .9 2 2v2" />
        <rect x="5" y="7" width="14" height="14" rx="2" />
        <path d="M9 12h6" />
      </svg>
    """,
    "resources": """
      <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
        <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20" />
        <path d="M6.5 3H20v18H6.5A2.5 2.5 0 0 1 4 18.5v-13A2.5 2.5 0 0 1 6.5 3Z" />
      </svg>
    """,
}


def render_workflow_card(title: str, body: str, page: str) -> None:
    """Render a compact workflow card as a single click target."""
    icon = WORKFLOW_CARD_ICONS.get(page, "")
    with st.container(key=f"workflow-card-{page}"):
        st.markdown(
            f"""
        <div class="workflow-card">
          <div class="workflow-card-content">
            <div class="workflow-card-title">{title}</div>
            <div class="workflow-card-copy">{body}</div>
          </div>
          <div class="workflow-card-icon">{icon}</div>
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
            "Prepare sequencing tables, genomic tracks, and other profile formats for downstream analysis.",
            "data-management",
        ),
        (
            "Mapping",
            "Infer spatial, temporal, or feature-dependent replication landscapes from compatible kinetic inputs.",
            "mapping",
        ),
        (
            "Simulation",
            "Generate replication dynamics under selected kinetic assumptions and compare simulated outputs.",
            "simulation",
        ),
        (
            "Plotting",
            "Create polished visual previews of input profiles, inferred features, simulations, and exports.",
            "plotting",
        ),
        (
            "Cached Files",
            "Manage reusable uploads, generated intermediates, figures, and analysis exports in one place.",
            "cached-files",
        ),
        (
            "Resources",
            "Find documentation, worked examples, source code, and citation guidance for the project.",
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
