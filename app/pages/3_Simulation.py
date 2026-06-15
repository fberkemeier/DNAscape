"""Simulation page for the DNAscape Streamlit prototype."""

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


configure_page("DNAscape | Simulation")
render_sidebar("Simulation")

render_page_intro(
    "Simulation",
    "Configure rsim experiments.",
    "Use stochastic replication simulations to test origin firing, fork-speed "
    "assumptions, perturbations, and expected genomic readouts.",
)

with st.container(border=True):
    st.markdown("### Prototype scope")
    st.markdown(
        """
        <p class="small-muted">
        This page will collect rsim inputs, simulation controls, output feature
        choices, comparison plots, and downloadable simulation results.
        </p>
        """,
        unsafe_allow_html=True,
    )
