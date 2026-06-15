"""Plotting page for the DNAscape Streamlit prototype."""

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


configure_page("DNAscape | Plotting")
render_sidebar("Plotting")

render_page_intro(
    "Plotting",
    "Visualize DNAscape outputs",
    "Create compact views for prepared inputs, mapped kinetic features, "
    "simulation results, and comparisons across genomic regions.",
)

with st.container(border=True):
    st.markdown("### Prototype scope")
    st.markdown(
        """
        <p class="small-muted">
        This page will collect plotting controls, figure presets, genomic
        region selectors, export options, and reusable visualization templates.
        </p>
        """,
        unsafe_allow_html=True,
    )
