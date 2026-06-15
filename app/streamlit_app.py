"""DNAscape Streamlit home page."""

import importlib
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import app_shell

importlib.reload(app_shell)

from app_shell import (
    DOCS_URL,
    GITHUB_URL,
    LOGO_PATH,
    configure_page,
    render_main_logo,
    render_sidebar,
)

import streamlit as st


configure_page("DNAscape")
render_sidebar("Overview")


def render_workflow_card(title: str, body: str, url: str) -> None:
    """Render a compact workflow card as a single click target."""
    st.markdown(
        f"""
        <a class="workflow-card" href="{url}" target="_self">
          <div class="workflow-card-title">{title}</div>
          <div class="workflow-card-copy">{body}</div>
        </a>
        """,
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

cols = st.columns(3)
with cols[0]:
    render_workflow_card(
        "Data Management",
        "Convert sequencing data and genomic tracks into DNAscape-ready inputs.",
        "/Data_Management",
    )

with cols[1]:
    render_workflow_card(
        "Mapping",
        "Explore rmap workflows for kinetic inference and feature translation.",
        "/Mapping",
    )

with cols[2]:
    render_workflow_card(
        "Simulation",
        "Set up rsim workflows for stochastic replication programme simulation.",
        "/Simulation",
    )

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
