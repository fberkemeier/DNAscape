"""Cached Files page for the DNAscape Streamlit prototype."""

import sys
from pathlib import Path

import streamlit as st


APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from app_shell import (
    configure_page,
    render_page_intro,
    render_sidebar,
    render_standard_content_marker,
)


configure_page("DNAscape | Cached Files")
render_sidebar("Cached Files")

render_page_intro(
    "Cached Files",
    "Manage reusable app artifacts",
    "Review generated inputs, intermediate tables, plots, simulation outputs, "
    "and other files cached during a DNAscape session.",
)

render_standard_content_marker("cached-files-page-marker")

with st.container(key="dnascape-standard-content"):
    with st.container(border=True):
        st.markdown("### Prototype scope")
        st.markdown(
            """
            <p class="small-muted">
            This page will expose cached uploads, converted DNAscape inputs,
            intermediate outputs, download actions, and cache-clearing controls.
            </p>
            """,
            unsafe_allow_html=True,
        )
