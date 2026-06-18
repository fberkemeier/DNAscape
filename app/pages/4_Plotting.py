"""Plotting page for the DNAscape Streamlit prototype."""

import base64
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
      body:has(.plotting-page-marker) [data-testid="stWidgetLabel"] {
        margin-bottom: 0.16rem;
      }

      body:has(.plotting-page-marker) [data-testid="stWidgetLabel"] p {
        font-size: 0.86rem;
        line-height: 1.25;
      }

      body:has(.plotting-page-marker) div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stVerticalBlock"] {
        gap: 0.62rem !important;
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
CLEAR_PREVIEW_PARAM = "dnascape_clear_preview"
CLEAR_PREVIEW_TOKEN = "plotting"
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
DISPLAY_UNITS = ("kb", "Mb")


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


def consume_preview_clear_request(state_key: str, token: str) -> None:
    query_params = getattr(st, "query_params", None)
    if query_params is not None:
        current_value = query_params.get(CLEAR_PREVIEW_PARAM)
        if isinstance(current_value, list):
            current_value = current_value[-1] if current_value else None
        if current_value == token:
            st.session_state.pop(state_key, None)
            try:
                del query_params[CLEAR_PREVIEW_PARAM]
            except Exception:
                try:
                    query_params.clear()
                except Exception:
                    pass
            st.rerun()
        return

    current_value = st.experimental_get_query_params().get(
        CLEAR_PREVIEW_PARAM,
        [None],
    )[-1]
    if current_value == token:
        st.session_state.pop(state_key, None)
        params = st.experimental_get_query_params()
        params.pop(CLEAR_PREVIEW_PARAM, None)
        st.experimental_set_query_params(**params)
        st.rerun()


def render_figure_preview(export_bytes: dict[str, bytes], file_stem: str) -> None:
    preview_src = base64.b64encode(export_bytes["png"]).decode("ascii")
    expand_toggle_id = f"plot-preview-expand-{safe_download_stem(file_stem) or 'figure'}"
    clear_href = f"?{CLEAR_PREVIEW_PARAM}={CLEAR_PREVIEW_TOKEN}"
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
            <a class="plot-preview-action plot-preview-clear-button" href="{clear_href}" aria-label="Clear plot preview" title="Clear preview">
              <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
                <path d="M9 3h6l1 1h4v2H4V4h4l1-1Zm-3 4h12v12a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2V7Zm3 3v8h2v-8H9Zm4 0v8h2v-8h-2Z" fill="currentColor" />
              </svg>
            </a>
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


consume_preview_clear_request(PLOT_STATE_KEY, CLEAR_PREVIEW_TOKEN)


with controls:
    with st.container(border=True):
        with st.expander("Data", expanded=True):
            data_cols = st.columns(2, gap="small")
            with data_cols[0]:
                data_type = st.selectbox(
                    "Type",
                    list(DATA_TYPE_DEFAULTS),
                    index=0,
                )
            data_defaults = DATA_TYPE_DEFAULTS[data_type]

            with data_cols[1]:
                uploaded_file = st.file_uploader(
                    "Upload",
                    type=["tsv", "csv", "txt", "gz"],
                    accept_multiple_files=False,
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

        with st.expander("Region", expanded=False):
            region_cols = st.columns(3, gap="small")
            with region_cols[0]:
                chromosome_options = numeric_columns if numeric_columns else [""]
                chromosome = st.selectbox(
                    "Chromosome",
                    chromosome_options,
                    format_func=lambda value: value if value else "Select chromosome",
                )
                if chromosome == "":
                    chromosome = None

            region_detail_cols = st.columns(3, gap="small")
            with region_detail_cols[0]:
                resolution = st.number_input(
                    "Resolution",
                    min_value=0.000001,
                    value=1.0,
                    step=1.0,
                    format="%.1f",
                    help="Spatial resolution in kb/bin.",
                )
            with region_detail_cols[1]:
                display_unit = st.selectbox(
                    "Display",
                    DISPLAY_UNITS,
                    index=0,
                    help="Unit used for the plot axis and default title.",
                )
            with region_detail_cols[2]:
                smooth_bins = st.number_input(
                    "Smooth",
                    min_value=1,
                    max_value=501,
                    value=1,
                    step=1,
                    help="Rolling mean window in bins. Use 1 to keep raw values.",
                )

            selected_values = None
            positions = None
            if table is not None and chromosome:
                raw_values = pd.to_numeric(table[chromosome], errors="coerce").to_numpy(
                    dtype=float
                )
                raw_positions = np.arange(raw_values.size, dtype=float) * float(
                    resolution
                )
                finite_mask = np.isfinite(raw_values)
                selected_values = raw_values[finite_mask]
                positions = raw_positions[finite_mask]

            if positions is not None and positions.size:
                max_position = max(0, int(np.ceil(float(np.nanmax(positions)))))
                min_position = 0
            else:
                max_position = max(1, int(np.ceil(float(resolution))))
                min_position = 0

            with region_cols[1]:
                region_start = st.number_input(
                    "Start (kb)",
                    min_value=min_position,
                    max_value=max_position,
                    value=min_position,
                    step=1,
                    format="%d",
                )
            with region_cols[2]:
                region_end = st.number_input(
                    "End (kb)",
                    min_value=min_position,
                    max_value=max_position,
                    value=max_position,
                    step=1,
                    format="%d",
                )

        with st.expander("Settings", expanded=False):
            finite_values = (
                selected_values[np.isfinite(selected_values)]
                if selected_values is not None
                else np.array([])
            )
            x_limit_default_min = float(region_start) / (
                1000.0 if display_unit == "Mb" else 1.0
            )
            x_limit_default_max = float(region_end) / (
                1000.0 if display_unit == "Mb" else 1.0
            )
            y_limit_default_min = (
                float(np.nanmin(finite_values)) if finite_values.size else 0.0
            )
            y_limit_default_max = (
                float(np.nanmax(finite_values)) if finite_values.size else 1.0
            )
            x_limit_step = max(
                abs(x_limit_default_max - x_limit_default_min) / 100.0,
                0.01 if display_unit == "Mb" else 1.0,
            )
            y_limit_step = max(
                abs(y_limit_default_max - y_limit_default_min) / 100.0,
                0.001,
            )
            text_cols = st.columns(2, gap="small")
            with text_cols[0]:
                include_title = st.checkbox("Title", value=True)
                title_override = st.text_input(
                    "Title text",
                    value="",
                    placeholder="Auto",
                    label_visibility="collapsed",
                )
            with text_cols[1]:
                include_label = st.checkbox("Label", value=True)
                legend_label = st.text_input(
                    "Label text",
                    value="",
                    placeholder=chromosome or "Profile",
                    label_visibility="collapsed",
                )

            axis_cols = st.columns(2, gap="small")
            with axis_cols[0]:
                include_x_axis = st.checkbox("x-axis", value=True)
                x_axis_title = st.text_input(
                    "x-axis title",
                    value=f"Genomic position ({display_unit})",
                    label_visibility="collapsed",
                )
            with axis_cols[1]:
                include_y_axis = st.checkbox("y-axis", value=True)
                y_axis_title = st.text_input(
                    "y-axis title",
                    value=data_defaults["ytitle"],
                    label_visibility="collapsed",
                )

            limit_toggle_cols = st.columns(2, gap="small")
            with limit_toggle_cols[0]:
                use_x_limits = st.checkbox("x limits", value=False)
            with limit_toggle_cols[1]:
                use_y_limits = st.checkbox(
                    "y limits",
                    value=False,
                )

            x_limit_min = x_limit_default_min
            x_limit_max = x_limit_default_max
            y_limit_min = y_limit_default_min
            y_limit_max = y_limit_default_max
            if use_x_limits or use_y_limits:
                limit_value_cols = st.columns(4, gap="small")
                with limit_value_cols[0]:
                    x_limit_min = st.number_input(
                        "x min",
                        value=x_limit_default_min,
                        step=x_limit_step,
                        format="%.3f" if display_unit == "Mb" else "%.1f",
                    )
                with limit_value_cols[1]:
                    x_limit_max = st.number_input(
                        "x max",
                        value=x_limit_default_max,
                        step=x_limit_step,
                        format="%.3f" if display_unit == "Mb" else "%.1f",
                    )
                with limit_value_cols[2]:
                    y_limit_min = st.number_input(
                        "y min",
                        value=y_limit_default_min,
                        step=y_limit_step,
                        format="%.6g",
                    )
                with limit_value_cols[3]:
                    y_limit_max = st.number_input(
                        "y max",
                        value=y_limit_default_max,
                        step=y_limit_step,
                        format="%.6g",
                    )

            style_cols = st.columns([0.78, 1.02, 0.86], gap="small")
            with style_cols[0]:
                show_legend = st.checkbox(
                    "Legend",
                    value=True,
                )
            with style_cols[1]:
                legend_location_label = st.selectbox(
                    "Position",
                    list(LEGEND_LOCATIONS),
                    index=0,
                )
            with style_cols[2]:
                line_color = st.color_picker(
                    "Color",
                    value=data_defaults["color"],
                )

            transform_cols = st.columns([0.92, 0.92, 0.8, 0.8], gap="small")
            with transform_cols[0]:
                log_y_axis = st.checkbox(
                    "Log y-axis",
                    value=False,
                )
            with transform_cols[1]:
                invert_y_axis = st.checkbox(
                    "Invert y-axis",
                    value=False,
                )
            with transform_cols[2]:
                figure_width = st.number_input(
                    "Width",
                    min_value=3.0,
                    max_value=16.0,
                    value=9.5,
                    step=0.5,
                    format="%.1f",
                )
            with transform_cols[3]:
                figure_height = st.number_input(
                    "Height",
                    min_value=2.0,
                    max_value=10.0,
                    value=3.8,
                    step=0.2,
                    format="%.1f",
                )

        plot_requested = st.button(
            "Plot",
            type="primary",
            width="stretch",
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
            if uploaded_file is None:
                st.error("Upload a data file before plotting.")
            elif table is None:
                st.error("Fix the uploaded data file before plotting.")
            elif not numeric_columns:
                st.error("The uploaded file does not contain numeric chromosome columns.")
            elif chromosome is None:
                st.error("Choose a chromosome column before plotting.")
            elif selected_values is None or selected_values.size == 0:
                st.error("The selected chromosome does not contain numeric data to plot.")
            elif positions is None or not positions.size:
                st.error("Choose a valid region before plotting.")
            elif region_end < region_start:
                st.error("Region end must be greater than or equal to region start.")
            elif use_x_limits and x_limit_max <= x_limit_min:
                st.error("The x-axis maximum must be greater than the minimum.")
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
                        if smooth_bins > 1:
                            plot_values = (
                                pd.Series(plot_values)
                                .rolling(
                                    window=int(smooth_bins),
                                    center=True,
                                    min_periods=1,
                                )
                                .mean()
                                .to_numpy(dtype=float)
                            )

                        position_scale = 1000.0 if display_unit == "Mb" else 1.0
                        plot_positions_display = plot_positions / position_scale
                        title_start = float(region_start) / position_scale
                        title_end = float(region_end) / position_scale
                        plot_title = title_override.strip() or (
                            f"{chromosome}: {title_start:g}-{title_end:g} {display_unit}"
                        )
                        plot_label = legend_label.strip() or chromosome
                        fig, _ = plotf(
                            plot_values,
                            x_array=plot_positions_display,
                            labels=[plot_label] if show_legend and include_label else None,
                            title=plot_title if include_title else "",
                            xtitle=x_axis_title if include_x_axis else "",
                            ytitle=y_axis_title if include_y_axis else "",
                            figsize=(figure_width, figure_height),
                            limsx=(
                                (x_limit_min, x_limit_max)
                                if use_x_limits
                                else (None, None)
                            ),
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
                            f"{title_start:g}-{title_end:g}{display_unit}",
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
