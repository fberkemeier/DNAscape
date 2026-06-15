"""Data Management page for the DNAscape Streamlit prototype."""

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


configure_page("DNAscape | Data Management")
render_sidebar("Data Management")

render_page_intro(
    "Data Management",
    "Prepare sequencing data for DNAscape.",
    "Convert sequencing outputs and genomic tracks into tidy, validated inputs "
    "that can be passed into mapping and simulation workflows.",
)

with st.container(border=True):
    st.markdown("### Prototype scope")
    st.markdown(
        """
        <p class="small-muted">
        This page will focus on upload, format checks, binning, normalization,
        metadata capture, and export of DNAscape-ready tables.
        </p>
        """,
        unsafe_allow_html=True,
    )
