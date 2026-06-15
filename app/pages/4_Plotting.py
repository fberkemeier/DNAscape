"""Plotting page for the DNAscape Streamlit prototype."""

import importlib
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
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
    "Upload kinetic profiles, choose a genomic window, and preview DNAscape outputs.",
)

controls, preview = st.columns([0.9, 1.35], gap="large")

PREVIEW_DPI = 300
EXPORT_FORMATS = (
    ("PNG", "png", "image/png", ":material/download:"),
    ("PDF", "pdf", "application/pdf", ":material/picture_as_pdf:"),
    ("SVG", "svg", "image/svg+xml", ":material/code:"),
)
PLOT_STATE_KEY = "plotting_last_plot"


def figure_bytes(fig, image_format: str) -> bytes:
    image = BytesIO()
    fig.savefig(
        image,
        format=image_format,
        bbox_inches="tight",
        dpi=PREVIEW_DPI,
        transparent=True,
        facecolor="none",
        edgecolor="none",
    )
    return image.getvalue()


def safe_download_stem(*parts: object) -> str:
    raw_name = "_".join(str(part) for part in parts if part is not None)
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in raw_name)


def figure_export_bytes(fig) -> dict[str, bytes]:
    return {
        image_format: figure_bytes(fig, image_format)
        for _, image_format, _, _ in EXPORT_FORMATS
    }


def render_figure_preview(png_bytes: bytes) -> None:
    st.image(png_bytes, width="stretch", output_format="PNG")


def render_download_buttons(export_bytes: dict[str, bytes], file_stem: str) -> None:
    export_cols = st.columns(
        [0.2, 0.30, 0.30, 0.30, 1.32],
        gap="small",
        vertical_alignment="center",
    )
    with export_cols[0]:
        st.markdown("**Export**")

    for column, (label, image_format, mime, icon) in zip(export_cols[1:4], EXPORT_FORMATS):
        with column:
            st.download_button(
                label,
                data=export_bytes[image_format],
                file_name=f"{file_stem}.{image_format}",
                mime=mime,
                icon=icon,
                on_click="ignore",
                width="stretch",
            )

with controls:
    with st.container(border=True):
        st.markdown("### Timing data")
        uploaded_file = st.file_uploader(
            "Upload file",
            type=["tsv", "csv", "txt", "gz"],
            accept_multiple_files=False,
        )

        table = None
        numeric_columns = []
        if uploaded_file is not None:
            try:
                compression = (
                    "gzip" if uploaded_file.name.lower().endswith(".gz") else None
                )
                table = pd.read_csv(
                    uploaded_file,
                    sep="\t",
                    compression=compression,
                )
            except Exception as exc:
                st.error(f"Could not read TSV file: {exc}")

            if table is not None:
                numeric_columns = [
                    column
                    for column in table.columns
                    if pd.to_numeric(table[column], errors="coerce").notna().any()
                ]

                if not numeric_columns:
                    st.warning("No numeric chromosome columns were found.")

        st.markdown("### Region")
        chromosome = st.selectbox(
            "Chromosome",
            numeric_columns,
            disabled=not numeric_columns,
            placeholder="Select a column",
        )

        resolution = st.number_input(
            "Spatial resolution (kb/bin)",
            min_value=0.000001,
            value=1.0,
            step=1.0,
            format="%.1f",
        )

        selected_values = None
        positions = None
        if table is not None and chromosome:
            raw_values = pd.to_numeric(table[chromosome], errors="coerce").to_numpy(
                dtype=float
            )
            raw_positions = np.arange(raw_values.size, dtype=float) * float(resolution)
            finite_mask = np.isfinite(raw_values)
            selected_values = raw_values[finite_mask]
            positions = raw_positions[finite_mask]

        if positions is not None and positions.size:
            max_position = max(0, int(np.ceil(float(np.nanmax(positions)))))
            min_position = 0
        else:
            max_position = max(1, int(np.ceil(float(resolution))))
            min_position = 0

        region_cols = st.columns(2)
        with region_cols[0]:
            region_start = st.number_input(
                "Start (kb)",
                min_value=min_position,
                max_value=max_position,
                value=min_position,
                step=1,
                format="%d",
                disabled=positions is None or not positions.size,
            )
        with region_cols[1]:
            region_end = st.number_input(
                "End (kb)",
                min_value=min_position,
                max_value=max_position,
                value=max_position,
                step=1,
                format="%d",
                disabled=positions is None or not positions.size,
            )

        plot_requested = st.button(
            "Plot",
            type="primary",
            width="stretch",
            disabled=selected_values is None or selected_values.size == 0,
        )

with preview:
    with st.container(border=True):
        st.markdown("### Preview")

        last_plot = st.session_state.get(PLOT_STATE_KEY)
        plot_feedback_shown = False

        if plot_requested:
            plot_feedback_shown = True
            if region_end < region_start:
                st.error("Region end must be greater than or equal to region start.")
            else:
                region_mask = (positions >= float(region_start)) & (
                    positions <= float(region_end)
                )
                plot_values = selected_values[region_mask]
                plot_positions = positions[region_mask]

                if plot_values.size == 0:
                    st.warning("The selected region does not contain any bins.")
                else:
                    try:
                        import matplotlib.pyplot as plt
                        from dnascape.plotting import plotf
                    except ModuleNotFoundError as exc:
                        missing = exc.name or "a plotting dependency"
                        st.error(
                            f"{missing} is required to render DNAscape plots. "
                            "Install the project requirements, then restart Streamlit."
                        )
                    else:
                        fig, _ = plotf(
                            plot_values,
                            x_array=plot_positions,
                            labels=[chromosome],
                            title=f"{chromosome}: {region_start:g}-{region_end:g} kb",
                            xtitle="Genomic position (kb)",
                            ytitle="Replication timing",
                            figsize=(9.5, 3.8),
                            showQ=False,
                            return_handles=True,
                        )
                        file_stem = safe_download_stem(
                            "dnascape",
                            chromosome,
                            f"{region_start:g}-{region_end:g}kb",
                        )
                        last_plot = {
                            "exports": figure_export_bytes(fig),
                            "file_stem": file_stem,
                        }
                        st.session_state[PLOT_STATE_KEY] = last_plot
                        plt.close(fig)

        if last_plot is not None:
            render_figure_preview(last_plot["exports"]["png"])
            render_download_buttons(last_plot["exports"], last_plot["file_stem"])
        elif plot_feedback_shown:
            pass
        elif table is None:
            st.markdown(
                """
                <p class="small-muted">
                Upload a timing table to preview a chromosome profile.
                </p>
                """,
                unsafe_allow_html=True,
            )
        elif selected_values is None or selected_values.size == 0:
            st.info("Choose a numeric chromosome column.")
        else:
            st.dataframe(
                table.loc[:, numeric_columns].head(8),
                width="stretch",
                hide_index=True,
            )
