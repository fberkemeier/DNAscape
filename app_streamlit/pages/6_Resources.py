"""Resources page for the DNAscape Streamlit application."""

import streamlit as st

from app_streamlit.app_shell import (
    DOCS_URL,
    GITHUB_URL,
    render_page_intro,
    render_standard_content_marker,
)


def render_page() -> None:
    """Render project links and citation guidance."""

    render_page_intro(
        "Resources",
        "Project links and citation guidance",
        "Documentation, source code, examples, and citation guidance.",
    )
    render_standard_content_marker("resources-page-marker")
    with st.container(key="dnascape-standard-content"):
        col1, col2 = st.columns(2)
        with col1:
            with st.container(border=True):
                st.markdown("### Repository")
                st.markdown(f"[DNAscape on GitHub]({GITHUB_URL})")
                st.markdown(
                    "<p class='small-muted'>Source code, issues, and development history.</p>",
                    unsafe_allow_html=True,
                )
        with col2:
            with st.container(border=True):
                st.markdown("### Documentation")
                st.markdown(f"[DNAscape Read the Docs]({DOCS_URL})")
                st.markdown(
                    "<p class='small-muted'>Installation notes, examples, and API guidance.</p>",
                    unsafe_allow_html=True,
                )
        with st.container(border=True):
            st.markdown("### Citation")
            st.markdown(
                """
                <p class="small-muted">
                If you use DNAscape in research, cite the relevant publication or
                software release and include the version or commit used.
                </p>
                """,
                unsafe_allow_html=True,
            )
