"""Plotting page for the DNAscape Streamlit prototype."""

import base64
import html
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

st.markdown(
    """
    <span class="plotting-page-marker" aria-hidden="true"></span>
    <style>
      body:has(.plotting-page-marker) .plotting-control-label {
        color: var(--dnascape-muted);
        font-size: 0.88rem;
        font-weight: 520;
        line-height: 1.25;
        padding-top: 0.1rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

controls, preview = st.columns([0.35, 0.65], gap="medium")

PREVIEW_DPI = 300
EXPORT_FORMATS = (
    ("PNG", "png", "image/png", ":material/download:"),
    ("PDF", "pdf", "application/pdf", ":material/picture_as_pdf:"),
    ("SVG", "svg", "image/svg+xml", ":material/code:"),
)
PLOT_STATE_KEY = "plotting_last_plot"
DATA_TYPE_DEFAULTS = {
    "Replication timing": {
        "ytitle": "Replication timing",
        "color": "#1f77b4",
    },
    "Origin firing": {
        "ytitle": "Origin firing",
        "color": "#b03412",
    },
    "Fork speed": {
        "ytitle": "Fork speed",
        "color": "#2ca02c",
    },
    "Fork directionality": {
        "ytitle": "Fork directionality",
        "color": "#9467bd",
    },
    "Initiation density": {
        "ytitle": "Initiation density",
        "color": "#ff7f0e",
    },
    "Custom profile": {
        "ytitle": "Value",
        "color": "#1f77b4",
    },
}
LEGEND_LOCATIONS = {
    "Lower right": "lower right",
    "Lower left": "lower left",
    "Upper right": "upper right",
    "Upper left": "upper left",
    "Best": "best",
}


def control_row(label: str, weights: tuple[float, float] = (0.38, 0.62)):
    label_column, field_column = st.columns(
        weights,
        gap="small",
        vertical_alignment="center",
    )
    with label_column:
        st.markdown(
            f'<div class="plotting-control-label">{html.escape(label)}</div>',
            unsafe_allow_html=True,
        )
    return field_column


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


def render_figure_preview(export_bytes: dict[str, bytes], file_stem: str) -> None:
    preview_src = base64.b64encode(export_bytes["png"]).decode("ascii")
    expand_toggle_id = f"plot-preview-expand-{safe_download_stem(file_stem) or 'figure'}"
    download_formats = (EXPORT_FORMATS[0], EXPORT_FORMATS[2], EXPORT_FORMATS[1])
    download_links = []
    for label, image_format, mime, _ in download_formats:
        payload = base64.b64encode(export_bytes[image_format]).decode("ascii")
        download_links.append(
            f'<a href="data:{mime};base64,{payload}" download="{file_stem}.{image_format}">{label.lower()}</a>'
        )

    st.markdown(
        f"""
        <div class="plot-preview-frame">
          <input class="plot-preview-expand-toggle" id="{expand_toggle_id}" type="checkbox" aria-hidden="true" />
          <div class="plot-preview-actions" aria-label="Plot preview actions">
            <div class="plot-download-menu">
              <button class="plot-preview-action" type="button" aria-label="Download plot" aria-haspopup="menu" title="Download plot">
                <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
                  <path d="M12 3v10.2l3.6-3.6 1.4 1.4-6 6-6-6 1.4-1.4 3.6 3.6V3h2zm-7 14h14v2H5v-2z" fill="currentColor" />
                </svg>
              </button>
              <div class="plot-download-menu-panel" role="menu">
                {''.join(download_links)}
              </div>
            </div>
            <label class="plot-preview-action plot-preview-expand-button" for="{expand_toggle_id}" aria-label="Expand plot" title="Expand plot">
              <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
                <path d="M5 5h6v2H8.4l3.3 3.3-1.4 1.4L7 8.4V11H5V5zm8 0h6v6h-2V8.4l-3.3 3.3-1.4-1.4L15.6 7H13V5zM7 15.6l3.3-3.3 1.4 1.4L8.4 17H11v2H5v-6h2v2.6zm9.6 1.4-3.3-3.3 1.4-1.4 3.3 3.3V13h2v6h-6v-2h2.6z" fill="currentColor" />
              </svg>
            </label>
          </div>
          <img class="plot-preview-image" src="data:image/png;base64,{preview_src}" alt="Plot preview" />
        </div>
        """,
        unsafe_allow_html=True,
    )


with controls:
    with st.container(border=True):
        st.markdown(
            '<div class="plotting-panel-title">Data</div>',
            unsafe_allow_html=True,
        )
        with control_row("Type"):
            data_type = st.selectbox(
                "Type",
                list(DATA_TYPE_DEFAULTS),
                index=0,
                label_visibility="collapsed",
            )
        data_defaults = DATA_TYPE_DEFAULTS[data_type]

        with control_row("Upload"):
            uploaded_file = st.file_uploader(
                "Upload a file",
                type=["tsv", "csv", "txt", "gz"],
                accept_multiple_files=False,
                label_visibility="collapsed",
            )

        table = None
        numeric_columns = []
        if uploaded_file is not None:
            try:
                suffixes = [suffix.lower() for suffix in Path(uploaded_file.name).suffixes]
                compression = (
                    "gzip" if uploaded_file.name.lower().endswith(".gz") else None
                )
                separator = "," if ".csv" in suffixes and ".tsv" not in suffixes else "\t"
                table = pd.read_csv(
                    uploaded_file,
                    sep=separator,
                    compression=compression,
                )
            except Exception as exc:
                st.error(f"Could not read data file: {exc}")

            if table is not None:
                numeric_columns = [
                    column
                    for column in table.columns
                    if pd.to_numeric(table[column], errors="coerce").notna().any()
                ]

                if not numeric_columns:
                    st.warning("No numeric chromosome columns were found.")

        st.markdown(
            '<div class="plotting-panel-title plotting-panel-title--spaced">Region</div>',
            unsafe_allow_html=True,
        )
        with control_row("Chromosome"):
            chromosome = st.selectbox(
                "Chromosome",
                numeric_columns,
                disabled=not numeric_columns,
                placeholder="Select column",
                label_visibility="collapsed",
            )
        with control_row("Resolution"):
            resolution = st.number_input(
                "Resolution",
                min_value=0.000001,
                value=1.0,
                step=1.0,
                format="%.1f",
                help="Spatial resolution in kb/bin.",
                label_visibility="collapsed",
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

        with control_row("Start (kb)"):
            region_start = st.number_input(
                "Start (kb)",
                min_value=min_position,
                max_value=max_position,
                value=min_position,
                step=1,
                format="%d",
                disabled=positions is None or not positions.size,
                label_visibility="collapsed",
            )
        with control_row("End (kb)"):
            region_end = st.number_input(
                "End (kb)",
                min_value=min_position,
                max_value=max_position,
                value=max_position,
                step=1,
                format="%d",
                disabled=positions is None or not positions.size,
                label_visibility="collapsed",
            )

        with st.expander("Plot styling", expanded=False):
            with control_row("Title"):
                title_override = st.text_input(
                    "Title",
                    value="",
                    placeholder="Auto",
                    label_visibility="collapsed",
                )
            with control_row("x-axis"):
                x_axis_title = st.text_input(
                    "x-axis",
                    value="Genomic position (kb)",
                    label_visibility="collapsed",
                )
            with control_row("y-axis"):
                y_axis_title = st.text_input(
                    "y-axis",
                    value=data_defaults["ytitle"],
                    label_visibility="collapsed",
                )

            with control_row("Color"):
                line_color = st.color_picker(
                    "Color",
                    value=data_defaults["color"],
                    label_visibility="collapsed",
                )
            with control_row("Label"):
                legend_label = st.text_input(
                    "Label",
                    value="",
                    placeholder=chromosome or "Profile",
                    label_visibility="collapsed",
                )

            with control_row("Legend"):
                show_legend = st.checkbox(
                    "Legend",
                    value=True,
                    label_visibility="collapsed",
                )
            with control_row("Position"):
                legend_location_label = st.selectbox(
                    "Position",
                    list(LEGEND_LOCATIONS),
                    index=0,
                    disabled=not show_legend,
                    label_visibility="collapsed",
                )

            with control_row("Invert y-axis"):
                invert_y_axis = st.checkbox(
                    "Invert y-axis",
                    value=False,
                    label_visibility="collapsed",
                )
            with control_row("Log y-axis"):
                log_y_axis = st.checkbox(
                    "Log y-axis",
                    value=False,
                    label_visibility="collapsed",
                )

            with control_row("Width"):
                figure_width = st.number_input(
                    "Width",
                    min_value=3.0,
                    max_value=16.0,
                    value=9.5,
                    step=0.5,
                    format="%.1f",
                    label_visibility="collapsed",
                )
            with control_row("Height"):
                figure_height = st.number_input(
                    "Height",
                    min_value=2.0,
                    max_value=10.0,
                    value=3.8,
                    step=0.2,
                    format="%.1f",
                    label_visibility="collapsed",
                )

            finite_values = (
                selected_values[np.isfinite(selected_values)]
                if selected_values is not None
                else np.array([])
            )
            y_limit_default_min = (
                float(np.nanmin(finite_values)) if finite_values.size else 0.0
            )
            y_limit_default_max = (
                float(np.nanmax(finite_values)) if finite_values.size else 1.0
            )
            with control_row("Y limits"):
                use_y_limits = st.checkbox(
                    "Set y-axis limits",
                    value=False,
                    disabled=not finite_values.size,
                    label_visibility="collapsed",
                )
            with control_row("Min"):
                y_limit_min = st.number_input(
                    "Min",
                    value=y_limit_default_min,
                    step=1.0,
                    disabled=not use_y_limits,
                    label_visibility="collapsed",
                )
            with control_row("Max"):
                y_limit_max = st.number_input(
                    "Max",
                    value=y_limit_default_max,
                    step=1.0,
                    disabled=not use_y_limits,
                    label_visibility="collapsed",
                )

        plot_requested = st.button(
            "Plot",
            type="primary",
            width="stretch",
            disabled=selected_values is None or selected_values.size == 0,
        )

with preview:
    with st.container(border=True):
        st.markdown(
            '<div class="plotting-panel-title">Preview</div>',
            unsafe_allow_html=True,
        )

        last_plot = st.session_state.get(PLOT_STATE_KEY)
        plot_feedback_shown = False

        if plot_requested:
            plot_feedback_shown = True
            if region_end < region_start:
                st.error("Region end must be greater than or equal to region start.")
            elif use_y_limits and y_limit_max <= y_limit_min:
                st.error("The y-axis maximum must be greater than the minimum.")
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
                        plot_title = title_override.strip() or (
                            f"{chromosome}: {region_start:g}-{region_end:g} kb"
                        )
                        plot_label = legend_label.strip() or chromosome
                        fig, _ = plotf(
                            plot_values,
                            x_array=plot_positions,
                            labels=[plot_label] if show_legend else None,
                            title=plot_title,
                            xtitle=x_axis_title,
                            ytitle=y_axis_title,
                            figsize=(figure_width, figure_height),
                            limsy=(
                                (y_limit_min, y_limit_max)
                                if use_y_limits
                                else (None, None)
                            ),
                            invyQ=invert_y_axis,
                            logyQ=log_y_axis,
                            label_loc=LEGEND_LOCATIONS[legend_location_label],
                            colors=[line_color],
                            showQ=False,
                            return_handles=True,
                        )
                        file_stem = safe_download_stem(
                            "dnascape",
                            data_type,
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
            render_figure_preview(last_plot["exports"], last_plot["file_stem"])
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
