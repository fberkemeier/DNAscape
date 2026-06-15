"""Mapping page for the DNAscape Streamlit prototype."""

import importlib
import sys
from pathlib import Path

import streamlit as st


APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import app_shell

importlib.reload(app_shell)

from app_shell import configure_page, render_page_intro, render_sidebar


configure_page("DNAscape | Mapping")
render_sidebar("Mapping")

render_page_intro(
    "Mapping",
    "Explore rmap workflows.",
    "Use DNAscape mapping to infer kinetic quantities, translate between "
    "replication features, and inspect the assumptions behind each route.",
)

with st.container(border=True):
    st.markdown("### Prototype scope")
    st.markdown(
        """
        <p class="small-muted">
        This page will expose rmap source features, target features, assumptions,
        route summaries, preview plots, and reproducible code export.
        </p>
        """,
        unsafe_allow_html=True,
    )
