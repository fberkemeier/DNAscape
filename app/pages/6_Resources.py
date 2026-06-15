"""Resources page for the DNAscape Streamlit prototype."""

import importlib
import sys
from pathlib import Path

import streamlit as st


APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import app_shell

importlib.reload(app_shell)

from app_shell import DOCS_URL, GITHUB_URL, configure_page, render_page_intro, render_sidebar


configure_page("DNAscape | Resources")
render_sidebar("Resources")

render_page_intro(
    "Resources",
    "Project links and citation notes.",
    "A compact reference page for the repository, documentation, examples, and "
    "citation guidance.",
)

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
        If you use DNAscape in research, cite the corresponding publication or
        software release. A more specific citation block can be added here once
        the final reference is fixed.
        </p>
        """,
        unsafe_allow_html=True,
    )
