"""Mapping page for the DNAscape Streamlit prototype."""

import base64
import contextlib
import importlib
import queue
import re
import sys
import threading
import time
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


configure_page("DNAscape | Mapping")
render_sidebar("Mapping")

render_page_intro(
    "Mapping",
    "Infer kinetic landscapes",
    "Load replication timing, choose a genomic window, and preview an rmap-derived initiation profile.",
)

st.markdown(
    '<span class="plotting-page-marker mapping-page-marker" aria-hidden="true"></span>',
    unsafe_allow_html=True,
)

controls, preview = st.columns([0.35, 0.65], gap="medium")

PREVIEW_DPI = 300
EXPORT_FORMATS = (
    ("PNG", "png", "image/png", ":material/download:"),
    ("PDF", "pdf", "application/pdf", ":material/picture_as_pdf:"),
    ("SVG", "svg", "image/svg+xml", ":material/code:"),
)
MAP_STATE_KEY = "mapping_last_map"
MAP_JOB_KEY = "mapping_active_job"
RMAP_PROGRESS_PATTERN = re.compile(
    r"\[(?P<iteration>\d+)/(?P<total>\d+)\].*?"
    r"Elapsed:\s*(?P<elapsed>\d{2}:\d{2}:\d{2}).*?"
    r"ETA:\s*(?P<eta>\d{2}:\d{2}:\d{2})"
    r"(?:.*?MSE:\s*(?P<mse>[^\s\r\n]+))?"
)
LEGEND_LOCATIONS = {
    "Lower right": "lower right",
    "Lower left": "lower left",
    "Upper right": "upper right",
    "Upper left": "upper left",
    "Best": "best",
}


class MappingCancelled(RuntimeError):
    """Raised inside a running mapping job when the user requests cancellation."""


class RMapProgressWriter:
    """Capture rmap ETA output and pass progress updates back to the UI thread."""

    def __init__(self, updates, cancel_event):
        self.updates = updates
        self.cancel_event = cancel_event
        self.buffer = ""

    def write(self, text: str) -> int:
        if self.cancel_event.is_set():
            raise MappingCancelled()
        if not text:
            return 0

        self.buffer = (self.buffer + text)[-1600:]
        match = None
        for candidate in RMAP_PROGRESS_PATTERN.finditer(self.buffer):
            match = candidate

        if match is not None:
            iteration = int(match.group("iteration"))
            total = max(1, int(match.group("total")))
            eta = match.group("eta")
            mse = match.group("mse")
            fraction = min(0.95, max(0.08, iteration / total))
            label = f"Mapping... {iteration}/{total} - ETA {eta}"
            if mse is not None:
                label = f"{label} - MSE {mse}"
            self.updates.put(
                {
                    "status": "progress",
                    "value": fraction,
                    "text": label,
                }
            )

        return len(text)

    def flush(self) -> None:
        if self.cancel_event.is_set():
            raise MappingCancelled()
        return None


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


def raw_result_frame(raw_result: dict[str, np.ndarray]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "position_kb": raw_result["position_kb"],
            "initiation_rate": raw_result["initiation_rate"],
        }
    )


def raw_download_payload(
    raw_result: dict[str, np.ndarray],
    format_label: str,
) -> tuple[bytes, str, str]:
    frame = raw_result_frame(raw_result)
    if format_label == "CSV":
        return frame.to_csv(index=False).encode("utf-8"), "csv", "text/csv"
    if format_label == "TXT":
        return (
            frame.to_csv(index=False, sep="\t").encode("utf-8"),
            "txt",
            "text/plain",
        )

    array_buffer = BytesIO()
    np.save(
        array_buffer,
        np.column_stack(
            [
                raw_result["position_kb"],
                raw_result["initiation_rate"],
            ]
        ),
    )
    return array_buffer.getvalue(), "npy", "application/octet-stream"


def build_mapping_preview(
    initiation_rate: np.ndarray,
    map_positions: np.ndarray,
    settings: dict[str, object],
) -> dict[str, object]:
    import matplotlib.pyplot as plt
    from dnascape.plotting import plotf

    plot_title = settings["title"] or f"{settings['chromosome']}: inferred initiation rate"
    plot_label = settings["label"] or "Initiation rate"
    fig, _ = plotf(
        initiation_rate,
        x_array=map_positions,
        labels=[plot_label] if settings["show_legend"] else None,
        title=plot_title,
        xtitle=settings["x_axis_title"],
        ytitle=settings["y_axis_title"],
        figsize=(settings["figure_width"], settings["figure_height"]),
        limsy=(
            (settings["y_limit_min"], settings["y_limit_max"])
            if settings["use_y_limits"]
            else (None, None)
        ),
        logyQ=settings["log_y_axis"],
        label_loc=LEGEND_LOCATIONS[settings["legend_location_label"]],
        colors=[settings["line_color"]],
        showQ=False,
        return_handles=True,
    )
    file_stem = safe_download_stem(
        "dnascape",
        "mapping",
        "initiation_rate",
        settings["chromosome"],
        f"{settings['region_start']:g}-{settings['region_end']:g}kb",
    )
    mapped_result = {
        "exports": figure_export_bytes(fig),
        "file_stem": file_stem,
        "raw": {
            "position_kb": np.asarray(map_positions, dtype=float),
            "initiation_rate": np.asarray(initiation_rate, dtype=float),
        },
    }
    plt.close(fig)
    return mapped_result


def run_mapping_job(
    timing_values: np.ndarray,
    map_positions: np.ndarray,
    assumptions: dict[str, object],
    updates,
    cancel_event,
) -> None:
    try:
        from dnascape import rmap

        with contextlib.redirect_stdout(RMapProgressWriter(updates, cancel_event)):
            initiation_rate = rmap(
                source="replication_timing",
                data=timing_values,
                target="firing_rate",
                assumptions=assumptions,
            )
        if cancel_event.is_set():
            raise MappingCancelled()
    except MappingCancelled:
        updates.put({"status": "cancelled"})
    except Exception as exc:
        updates.put({"status": "error", "message": str(exc)})
    else:
        updates.put(
            {
                "status": "complete",
                "initiation_rate": np.asarray(initiation_rate, dtype=float),
                "position_kb": np.asarray(map_positions, dtype=float),
            }
        )


def start_mapping_job(
    timing_values: np.ndarray,
    map_positions: np.ndarray,
    assumptions: dict[str, object],
    plot_settings: dict[str, object],
) -> dict[str, object]:
    updates = queue.Queue()
    cancel_event = threading.Event()
    job = {
        "updates": updates,
        "cancel_event": cancel_event,
        "plot_settings": plot_settings,
        "progress": {"value": 0.05, "text": "Mapping..."},
    }
    thread = threading.Thread(
        target=run_mapping_job,
        args=(
            np.asarray(timing_values, dtype=float),
            np.asarray(map_positions, dtype=float),
            dict(assumptions),
            updates,
            cancel_event,
        ),
        daemon=True,
    )
    job["thread"] = thread
    thread.start()
    return job


def drain_mapping_job(job: dict[str, object]) -> dict[str, object] | None:
    terminal_update = None
    updates = job["updates"]
    while True:
        try:
            update = updates.get_nowait()
        except queue.Empty:
            break

        if update["status"] == "progress":
            job["progress"] = {
                "value": update["value"],
                "text": update["text"],
            }
        else:
            terminal_update = update

    return terminal_update


def render_figure_preview(export_bytes: dict[str, bytes], file_stem: str) -> None:
    preview_src = base64.b64encode(export_bytes["png"]).decode("ascii")
    expand_toggle_id = f"map-preview-expand-{safe_download_stem(file_stem) or 'figure'}"
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
          <div class="plot-preview-actions" aria-label="Mapping preview actions">
            <div class="plot-download-menu">
              <button class="plot-preview-action" type="button" aria-label="Download map preview" aria-haspopup="menu" title="Download">
                <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
                  <path d="M12 3v10.2l3.6-3.6 1.4 1.4-6 6-6-6 1.4-1.4 3.6 3.6V3h2zm-7 14h14v2H5v-2z" fill="currentColor" />
                </svg>
              </button>
              <div class="plot-download-menu-panel" role="menu">
                {''.join(download_links)}
              </div>
            </div>
            <label class="plot-preview-action plot-preview-expand-button" for="{expand_toggle_id}" aria-label="Expand map preview" title="Expand">
              <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
                <path d="M5 5h6v2H8.4l3.3 3.3-1.4 1.4L7 8.4V11H5V5zm8 0h6v6h-2V8.4l-3.3 3.3-1.4-1.4L15.6 7H13V5zM7 15.6l3.3-3.3 1.4 1.4L8.4 17H11v2H5v-6h2v2.6zm9.6 1.4-3.3-3.3 1.4-1.4 3.3 3.3V13h2v6h-6v-2h2.6z" fill="currentColor" />
              </svg>
            </label>
          </div>
          <img class="plot-preview-image" src="data:image/png;base64,{preview_src}" alt="Mapping preview" />
        </div>
        """,
        unsafe_allow_html=True,
    )


active_mapping_job = st.session_state.get(MAP_JOB_KEY)

with controls:
    with st.container(border=True):
        st.markdown(
            '<div class="plotting-panel-title">Data</div>',
            unsafe_allow_html=True,
        )
        input_type = st.selectbox(
            "Input",
            ["Replication timing"],
            index=0,
        )
        target_type = st.selectbox(
            "Target",
            ["Space-dependent initiation rate"],
            index=0,
        )

        uploaded_file = st.file_uploader(
            "Upload a file",
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

        st.markdown(
            '<div class="plotting-panel-title plotting-panel-title--spaced">Region</div>',
            unsafe_allow_html=True,
        )
        region_top_cols = st.columns([1.25, 0.75], gap="small")
        with region_top_cols[0]:
            chromosome = st.selectbox(
                "Chromosome",
                numeric_columns,
                disabled=not numeric_columns,
                placeholder="Select column",
            )
        with region_top_cols[1]:
            resolution = st.number_input(
                "Resolution",
                min_value=0.000001,
                value=1.0,
                step=1.0,
                format="%.1f",
                help="Spatial resolution in kb/bin.",
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

        region_cols = st.columns(2, gap="small")
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

        st.markdown(
            '<div class="plotting-panel-title plotting-panel-title--spaced">Mapping</div>',
            unsafe_allow_html=True,
        )
        map_cols = st.columns(2, gap="small")
        with map_cols[0]:
            fork_speed = st.number_input(
                "Fork speed",
                min_value=0.000001,
                value=1.4,
                step=0.1,
                format="%.2f",
                help="Constant fork speed in kb/min.",
            )
        with map_cols[1]:
            max_iterations = st.number_input(
                "Iterations",
                min_value=0,
                max_value=200,
                value=10,
                step=1,
                format="%d",
            )

        with st.expander("rmap settings", expanded=False):
            fit_cols = st.columns(2, gap="small")
            with fit_cols[0]:
                fit_step = st.number_input(
                    "Fit step",
                    min_value=0.01,
                    max_value=2.0,
                    value=0.6,
                    step=0.05,
                    format="%.2f",
                )
            with fit_cols[1]:
                neighbourhood_range = st.number_input(
                    "Neighbourhood",
                    min_value=0,
                    value=2000,
                    step=100,
                    format="%d",
                    help="Neighbourhood range in kb.",
                )

            threshold_cols = st.columns(2, gap="small")
            with threshold_cols[0]:
                err_threshold = st.number_input(
                    "Error floor",
                    min_value=1.0,
                    max_value=30.0,
                    value=15.0,
                    step=1.0,
                    format="%.1f",
                    help="Lower firing-rate floor is 10^(-error floor).",
                )
            with threshold_cols[1]:
                periodic_genome = st.checkbox("Periodic", value=False)

            use_deviation_threshold = st.checkbox(
                "Freeze close bins",
                value=False,
            )
            deviation_threshold = st.number_input(
                "Tolerance",
                min_value=0.0,
                value=0.0,
                step=0.1,
                disabled=not use_deviation_threshold,
                help="Bins with timing deviation below this value are left unchanged.",
            )

        with st.expander("Plot styling", expanded=False):
            title_override = st.text_input(
                "Title",
                value="",
                placeholder="Auto",
            )
            axis_cols = st.columns(2, gap="small")
            with axis_cols[0]:
                x_axis_title = st.text_input(
                    "x-axis",
                    value="Genomic position (kb)",
                )
            with axis_cols[1]:
                y_axis_title = st.text_input(
                    "y-axis",
                    value="Initiation rate",
                )

            style_cols = st.columns(2, gap="small")
            with style_cols[0]:
                line_color = st.color_picker(
                    "Color",
                    value="#b03412",
                )
            with style_cols[1]:
                legend_label = st.text_input(
                    "Label",
                    value="",
                    placeholder="Initiation rate",
                )

            legend_cols = st.columns([0.85, 1.15], gap="small")
            with legend_cols[0]:
                show_legend = st.checkbox("Legend", value=True)
            with legend_cols[1]:
                legend_location_label = st.selectbox(
                    "Position",
                    list(LEGEND_LOCATIONS),
                    index=0,
                    disabled=not show_legend,
                )

            scale_cols = st.columns(2, gap="small")
            with scale_cols[0]:
                log_y_axis = st.checkbox("Log y-axis", value=False)
            with scale_cols[1]:
                use_y_limits = st.checkbox("Set y-axis limits", value=False)

            y_limit_cols = st.columns(2, gap="small")
            with y_limit_cols[0]:
                y_limit_min = st.number_input(
                    "Min",
                    value=0.0,
                    step=0.00001,
                    format="%.6g",
                    disabled=not use_y_limits,
                )
            with y_limit_cols[1]:
                y_limit_max = st.number_input(
                    "Max",
                    value=0.0001,
                    step=0.00001,
                    format="%.6g",
                    disabled=not use_y_limits,
                )

            size_cols = st.columns(2, gap="small")
            with size_cols[0]:
                figure_width = st.number_input(
                    "Width",
                    min_value=3.0,
                    max_value=16.0,
                    value=9.5,
                    step=0.5,
                    format="%.1f",
                )
            with size_cols[1]:
                figure_height = st.number_input(
                    "Height",
                    min_value=2.0,
                    max_value=10.0,
                    value=3.8,
                    step=0.2,
                    format="%.1f",
                )

        map_requested = st.button(
            "Map",
            type="primary",
            width="stretch",
            disabled=(
                selected_values is None
                or selected_values.size == 0
                or active_mapping_job is not None
            ),
        )

with preview:
    with st.container(border=True):
        st.markdown(
            '<div class="plotting-panel-title">Preview</div>',
            unsafe_allow_html=True,
        )

        last_map = st.session_state.get(MAP_STATE_KEY)
        map_feedback_shown = False
        active_mapping_job = st.session_state.get(MAP_JOB_KEY)

        if map_requested:
            map_feedback_shown = True
            if region_end < region_start:
                st.error("Region end must be greater than or equal to region start.")
            elif use_y_limits and y_limit_max <= y_limit_min:
                st.error("The y-axis maximum must be greater than the minimum.")
            else:
                region_mask = (positions >= float(region_start)) & (
                    positions <= float(region_end)
                )
                timing_values = selected_values[region_mask]
                map_positions = positions[region_mask]

                if timing_values.size == 0:
                    st.warning("The selected region does not contain any bins.")
                else:
                    active_mapping_job = start_mapping_job(
                        timing_values,
                        map_positions,
                        {
                            "fork_speed": fork_speed,
                            "resolution": resolution,
                            "target_form": "space",
                            "fit_step": fit_step,
                            "maxiter": int(max_iterations),
                            "neighbourhood_range": int(neighbourhood_range),
                            "err_threshold": err_threshold,
                            "deviation_threshold": (
                                deviation_threshold
                                if use_deviation_threshold
                                else None
                            ),
                            "perQ": periodic_genome,
                            "verbose": True,
                            "print_every": 1,
                        },
                        {
                            "chromosome": chromosome,
                            "region_start": region_start,
                            "region_end": region_end,
                            "title": title_override.strip(),
                            "label": legend_label.strip(),
                            "x_axis_title": x_axis_title,
                            "y_axis_title": y_axis_title,
                            "figure_width": figure_width,
                            "figure_height": figure_height,
                            "use_y_limits": use_y_limits,
                            "y_limit_min": y_limit_min,
                            "y_limit_max": y_limit_max,
                            "log_y_axis": log_y_axis,
                            "legend_location_label": legend_location_label,
                            "show_legend": show_legend,
                            "line_color": line_color,
                        },
                    )
                    st.session_state[MAP_JOB_KEY] = active_mapping_job

        if active_mapping_job is not None:
            terminal_update = drain_mapping_job(active_mapping_job)

            if terminal_update is None:
                progress = active_mapping_job["progress"]
                st.progress(progress["value"], text=progress["text"])
                if st.button(
                    "Stop mapping",
                    type="secondary",
                    width="stretch",
                    key="mapping_stop_button",
                ):
                    active_mapping_job["cancel_event"].set()
                    active_mapping_job["progress"] = {
                        "value": progress["value"],
                        "text": "Stopping mapping...",
                    }
                    st.rerun()

                if active_mapping_job["thread"].is_alive():
                    time.sleep(0.25)
                    st.rerun()
                else:
                    time.sleep(0.05)
                    st.rerun()
            else:
                completed_job = active_mapping_job
                st.session_state.pop(MAP_JOB_KEY, None)
                active_mapping_job = None

                if terminal_update["status"] == "complete":
                    render_slot = st.empty()
                    render_slot.progress(0.98, text="Rendering preview...")
                    try:
                        last_map = build_mapping_preview(
                            terminal_update["initiation_rate"],
                            terminal_update["position_kb"],
                            completed_job["plot_settings"],
                        )
                    except Exception as exc:
                        render_slot.empty()
                        st.error(f"Could not render the mapping preview: {exc}")
                    else:
                        st.session_state[MAP_STATE_KEY] = last_map
                        render_slot.empty()
                elif terminal_update["status"] == "cancelled":
                    st.info("Mapping stopped.")
                elif terminal_update["status"] == "error":
                    st.error(
                        f"rmap could not complete this mapping: {terminal_update['message']}"
                    )

        if active_mapping_job is not None:
            pass
        elif last_map is not None:
            render_figure_preview(last_map["exports"], last_map["file_stem"])
        elif map_feedback_shown:
            pass
        elif table is None:
            st.markdown(
                """
                <p class="small-muted">
                Upload a replication-timing table to infer an initiation-rate profile.
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

    with st.container(border=True):
        st.markdown(
            '<div class="plotting-panel-title">Raw output</div>',
            unsafe_allow_html=True,
        )

        last_map_for_download = st.session_state.get(MAP_STATE_KEY)
        if last_map_for_download is not None and "raw" in last_map_for_download:
            st.markdown(
                """
                <p class="small-muted">
                Download the mapped initiation-rate profile with genomic positions.
                </p>
                """,
                unsafe_allow_html=True,
            )
            export_cols = st.columns([0.42, 0.58], gap="small")
            with export_cols[0]:
                raw_format = st.selectbox(
                    "Format",
                    ["CSV", "TXT", "Array"],
                    index=0,
                    key="mapping_raw_format",
                )
            payload, extension, mime = raw_download_payload(
                last_map_for_download["raw"],
                raw_format,
            )
            with export_cols[1]:
                st.download_button(
                    "Download",
                    data=payload,
                    file_name=f"{last_map_for_download['file_stem']}_raw.{extension}",
                    mime=mime,
                    on_click="ignore",
                    width="stretch",
                )
        else:
            st.markdown(
                """
                <p class="small-muted">
                Run a mapping to enable raw output export.
                </p>
                """,
                unsafe_allow_html=True,
            )
