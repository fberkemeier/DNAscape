"""Simulation page for the DNAscape Streamlit prototype."""

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


configure_page("DNAscape | Simulation")
render_sidebar("Simulation")

render_page_intro(
    "Simulation",
    "Configure rsim experiments",
    "Simulate stochastic replication dynamics, vary kinetic assumptions, and preview DNAscape-ready outputs.",
)

st.markdown(
    """
    <span class="plotting-page-marker simulation-page-marker" aria-hidden="true"></span>
    <style>
      body:has(.simulation-page-marker) [data-testid="stWidgetLabel"] {
        margin-bottom: 0.16rem;
      }

      body:has(.simulation-page-marker) [data-testid="stWidgetLabel"] p {
        font-size: 0.86rem;
        line-height: 1.25;
      }

      body:has(.simulation-page-marker) div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stVerticalBlock"] {
        gap: 0.62rem !important;
      }

      body:has(.simulation-page-marker) [data-testid="stSelectbox"] [data-baseweb="select"],
      body:has(.simulation-page-marker) [data-testid="stSelectbox"] [data-baseweb="select"] *,
      body:has(.simulation-page-marker) [data-testid="stSelectbox"] [data-baseweb="select"] input {
        color: var(--dnascape-text) !important;
        opacity: 1 !important;
        -webkit-text-fill-color: var(--dnascape-text) !important;
      }

      body:has(.simulation-page-marker) [data-testid="stSelectbox"] [data-baseweb="select"] svg {
        color: var(--dnascape-muted) !important;
        fill: currentColor !important;
        -webkit-text-fill-color: currentColor !important;
      }

      body:has(.simulation-page-marker) [data-baseweb="popover"] [role="option"],
      body:has(.simulation-page-marker) [data-baseweb="popover"] [role="option"] *,
      body:has(.simulation-page-marker) [data-baseweb="menu"] [role="option"],
      body:has(.simulation-page-marker) [data-baseweb="menu"] [role="option"] * {
        color: var(--dnascape-text) !important;
        opacity: 1 !important;
        -webkit-text-fill-color: var(--dnascape-text) !important;
      }

      body:has(.simulation-page-marker) [data-testid="stButton"] button[kind="primary"]:disabled,
      body:has(.simulation-page-marker) [data-testid="stButton"] button[kind="primary"][aria-disabled="true"] {
        background: var(--dnascape-accent) !important;
        border-color: var(--dnascape-accent) !important;
        color: white !important;
        opacity: 0.88 !important;
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
SIM_STATE_KEY = "simulation_last_preview"
SIM_JOB_KEY = "simulation_active_job"
RSIM_PROGRESS_PATTERN = re.compile(
    r"\[(?P<label>[^\]]+)\]\s*\[(?P<iteration>\d+)/(?P<total>\d+)\]\s*"
    r"Elapsed:\s*(?P<elapsed>\d{2}:\d{2}:\d{2})\s*ETA:\s*(?P<eta>\d{2}:\d{2}:\d{2})"
)
LINE_INPUT_MODES = ("Profile", "Constant")
GEOMETRY_OPTIONS = ("Linear", "Periodic")
HEATMAP_PALETTES = ("viridis", "plasma", "magma", "cividis", "inferno")
LEGEND_LOCATIONS = {
    "Lower right": "lower right",
    "Lower left": "lower left",
    "Upper right": "upper right",
    "Upper left": "upper left",
    "Best": "best",
}
SIMULATION_OUTPUTS = {
    "Replication timing": {
        "key": "replication_timing",
        "kind": "space",
        "ytitle": "Replication timing",
        "color": "#1f77b4",
    },
    "Origin efficiency": {
        "key": "origin_efficiency",
        "kind": "space",
        "ytitle": "Origin efficiency",
        "color": "#b03412",
    },
    "Fork directionality": {
        "key": "fork_directionality",
        "kind": "space",
        "ytitle": "Fork directionality",
        "color": "#9467bd",
    },
    "Replicated fraction": {
        "key": "replicated_fraction",
        "kind": "time",
        "ytitle": "Replicated fraction",
        "color": "#2ca02c",
    },
    "Total forks": {
        "key": "total_forks",
        "kind": "time",
        "ytitle": "Total forks",
        "color": "#ff7f0e",
    },
    "Active forks": {
        "key": "active_forks",
        "kind": "time",
        "ytitle": "Active forks",
        "color": "#1f77b4",
    },
    "Stalled forks": {
        "key": "stalled_forks",
        "kind": "time",
        "ytitle": "Stalled forks",
        "color": "#8c564b",
    },
    "Firing factors": {
        "key": "firing_factors",
        "kind": "time",
        "ytitle": "Firing factors",
        "color": "#7f7f7f",
    },
    "Replicated fraction (space-time)": {
        "key": "replicated_fraction_xt",
        "kind": "xt",
        "label": "Replicated fraction",
        "cmap": "viridis",
    },
    "Initiation events (space-time)": {
        "key": "initiation_events_xt",
        "kind": "xt",
        "label": "Initiation events",
        "cmap": "magma",
    },
    "Coalescence events (space-time)": {
        "key": "coalescence_events_xt",
        "kind": "xt",
        "label": "Coalescence events",
        "cmap": "plasma",
    },
    "Right fork density (space-time)": {
        "key": "right_moving_fork_density_xt",
        "kind": "xt",
        "label": "Right-moving fork density",
        "cmap": "cividis",
    },
    "Left fork density (space-time)": {
        "key": "left_moving_fork_density_xt",
        "kind": "xt",
        "label": "Left-moving fork density",
        "cmap": "cividis",
    },
}
SIMULATION_OUTPUTS_BY_KEY = {
    str(spec["key"]): {
        "title": title,
        "label": str(spec.get("label", title)),
        **spec,
    }
    for title, spec in SIMULATION_OUTPUTS.items()
}
SIMULATION_VIEW_OPTIONS = {
    "space": ("Single view", "Multiple view", "Histogram"),
    "time": ("Single view", "Multiple view", "Histogram"),
    "xt": ("Time array", "Multiple view", "Histogram"),
}


class SimulationCancelled(RuntimeError):
    """Raised inside a running rsim job when the user requests cancellation."""


class RSimProgressWriter:
    """Capture rsim ETA output and stream progress updates back to the UI thread."""

    def __init__(self, updates, cancel_event):
        self.updates = updates
        self.cancel_event = cancel_event
        self.buffer = ""

    def write(self, text: str) -> int:
        if self.cancel_event.is_set():
            raise SimulationCancelled()
        if not text:
            return 0

        self.buffer = (self.buffer + text)[-2000:]
        match = None
        for candidate in RSIM_PROGRESS_PATTERN.finditer(self.buffer):
            match = candidate

        if match is not None:
            iteration = int(match.group("iteration"))
            total = max(1, int(match.group("total")))
            eta = match.group("eta")
            label = match.group("label").strip().lower()
            prefix = "Simulating" if label == "rsim" else label.replace("_", " ").title()
            self.updates.put(
                {
                    "status": "progress",
                    "value": min(0.95, max(0.08, iteration / total)),
                    "text": f"{prefix}... {iteration}/{total} - ETA {eta}",
                }
            )

        return len(text)

    def flush(self) -> None:
        if self.cancel_event.is_set():
            raise SimulationCancelled()
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


def clear_preview_state(state_key: str) -> None:
    st.session_state.pop(state_key, None)


def read_numeric_table(uploaded_file) -> tuple[pd.DataFrame | None, list[str], str | None]:
    if uploaded_file is None:
        return None, [], None
    try:
        suffixes = [suffix.lower() for suffix in Path(uploaded_file.name).suffixes]
        compression = "gzip" if uploaded_file.name.lower().endswith(".gz") else None
        separator = "," if ".csv" in suffixes and ".tsv" not in suffixes else "\t"
        table = pd.read_csv(
            uploaded_file,
            sep=separator,
            compression=compression,
        )
    except Exception as exc:
        return None, [], str(exc)

    numeric_columns = [
        column
        for column in table.columns
        if pd.to_numeric(table[column], errors="coerce").notna().any()
    ]
    return table, numeric_columns, None


def render_figure_preview(export_bytes: dict[str, bytes], file_stem: str) -> None:
    preview_src = base64.b64encode(export_bytes["png"]).decode("ascii")
    expand_toggle_id = f"simulation-preview-expand-{safe_download_stem(file_stem) or 'figure'}"
    download_formats = (EXPORT_FORMATS[0], EXPORT_FORMATS[2], EXPORT_FORMATS[1])
    download_links = []
    for label, image_format, mime, _ in download_formats:
        payload = base64.b64encode(export_bytes[image_format]).decode("ascii")
        download_links.append(
            f'<a href="data:{mime};base64,{payload}" download="{file_stem}.{image_format}">{label.lower()}</a>'
        )

    with st.container(key="simulation-preview-shell"):
        st.button(
            "Clear preview",
            key="simulation-preview-clear",
            icon=":material/delete:",
            on_click=clear_preview_state,
            args=(SIM_STATE_KEY,),
        )
        st.markdown(
            f"""
            <div class="plot-preview-frame">
              <input class="plot-preview-expand-toggle" id="{expand_toggle_id}" type="checkbox" aria-hidden="true" />
              <div class="plot-preview-actions" aria-label="Simulation preview actions">
                <div class="plot-download-menu">
                  <button class="plot-preview-action" type="button" aria-label="Download simulation preview" aria-haspopup="menu" title="Download">
                    <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
                      <path d="M12 3v10.2l3.6-3.6 1.4 1.4-6 6-6-6 1.4-1.4 3.6 3.6V3h2zm-7 14h14v2H5v-2z" fill="currentColor" />
                    </svg>
                  </button>
                  <div class="plot-download-menu-panel" role="menu">
                    {''.join(download_links)}
                  </div>
                </div>
                <span class="plot-preview-action plot-preview-clear-placeholder" aria-hidden="true">
                  <svg viewBox="0 0 24 24" focusable="false">
                    <path d="M9 3h6l1 1h4v2H4V4h4l1-1Zm-3 4h12v12a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2V7Zm3 3v8h2v-8H9Zm4 0v8h2v-8h-2Z" fill="currentColor" />
                  </svg>
                </span>
                <label class="plot-preview-action plot-preview-expand-button" for="{expand_toggle_id}" aria-label="Expand simulation preview" title="Expand">
                  <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
                    <path d="M5 5h6v2H8.4l3.3 3.3-1.4 1.4L7 8.4V11H5V5zm8 0h6v6h-2V8.4l-3.3 3.3-1.4-1.4L15.6 7H13V5zM7 15.6l3.3-3.3 1.4 1.4L8.4 17H11v2H5v-6h2v2.6zm9.6 1.4-3.3-3.3 1.4-1.4 3.3 3.3V13h2v6h-6v-2h2.6z" fill="currentColor" />
                  </svg>
                </label>
              </div>
              <img class="plot-preview-image" src="data:image/png;base64,{preview_src}" alt="Simulation preview" />
            </div>
            """,
            unsafe_allow_html=True,
        )


def simulation_time_grid(resolution_time: float, time_max: float) -> np.ndarray:
    return np.arange(0.0, float(time_max) + float(resolution_time), float(resolution_time))


def interval_positions(region_start: float, region_end: float, resolution: float) -> np.ndarray:
    n_bins = max(1, int(np.ceil((float(region_end) - float(region_start)) / float(resolution))))
    return float(region_start) + np.arange(n_bins, dtype=float) * float(resolution)


def simulation_output_meta(output_key: str) -> dict[str, object]:
    meta = SIMULATION_OUTPUTS_BY_KEY.get(str(output_key))
    if meta is not None:
        return meta

    title = str(output_key).replace("_", " ").title()
    return {
        "title": title,
        "label": title,
        "key": output_key,
        "kind": "space",
        "ytitle": title,
        "color": "#1f77b4",
    }


def normalize_preview_series(values: object, axis: np.ndarray | None = None) -> np.ndarray:
    axis_array = (
        np.asarray(axis, dtype=float).reshape(-1)
        if axis is not None
        else np.array([], dtype=float)
    )
    series = np.asarray(values, dtype=float)
    if series.ndim == 0:
        target_size = axis_array.size or 1
        return np.full(target_size, float(series), dtype=float)

    series = series.reshape(-1)
    if axis_array.size and series.size != axis_array.size:
        if series.size == 1:
            return np.full(axis_array.size, float(series[0]), dtype=float)
        source_axis = np.linspace(0.0, 1.0, series.size)
        target_axis = np.linspace(0.0, 1.0, axis_array.size)
        return np.interp(target_axis, source_axis, series)
    return series


def normalize_preview_matrix(
    values: object,
    positions: np.ndarray | None = None,
    time_grid: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    position_array = (
        np.asarray(positions, dtype=float).reshape(-1)
        if positions is not None
        else np.array([], dtype=float)
    )
    time_array = (
        np.asarray(time_grid, dtype=float).reshape(-1)
        if time_grid is not None
        else np.array([], dtype=float)
    )

    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2:
        matrix = np.atleast_2d(matrix)

    if position_array.size and matrix.shape[0] != position_array.size and matrix.shape[1] == position_array.size:
        matrix = matrix.T

    if position_array.size != matrix.shape[0]:
        position_array = np.arange(matrix.shape[0], dtype=float)

    if time_array.size != matrix.shape[1]:
        time_array = np.arange(matrix.shape[1], dtype=float)

    return matrix, position_array, time_array


def simulation_raw_frame(raw_result: dict[str, object]) -> pd.DataFrame:
    kind = raw_result["kind"]
    values = np.asarray(raw_result["values"], dtype=float)
    value_column = str(raw_result["value_column"])

    if kind == "space":
        positions = np.asarray(raw_result.get("position_kb", []), dtype=float)
        if positions.size != values.size:
            positions = np.arange(values.size, dtype=float)
        return pd.DataFrame(
            {
                "position_kb": positions,
                value_column: values.reshape(-1),
            }
        )

    if kind == "time":
        time_grid = np.asarray(raw_result.get("time_min", []), dtype=float)
        if time_grid.size != values.size:
            time_grid = np.arange(values.size, dtype=float)
        return pd.DataFrame(
            {
                "time_min": time_grid,
                value_column: values.reshape(-1),
            }
        )

    positions = np.asarray(raw_result.get("position_kb", []), dtype=float)
    time_grid = np.asarray(raw_result.get("time_min", []), dtype=float)
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2:
        matrix = np.atleast_2d(matrix)
    if matrix.shape[0] != positions.size and matrix.shape[1] == positions.size:
        matrix = matrix.T
    if positions.size != matrix.shape[0]:
        positions = np.arange(matrix.shape[0], dtype=float)
    if time_grid.size != matrix.shape[1]:
        time_grid = np.arange(matrix.shape[1], dtype=float)

    rows = []
    for index, time_value in enumerate(time_grid):
        rows.append(
            pd.DataFrame(
                {
                    "position_kb": positions,
                    "time_min": time_value,
                    value_column: matrix[:, index],
                }
            )
        )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def simulation_raw_payload(
    raw_result: dict[str, object],
    format_label: str,
) -> tuple[bytes, str, str]:
    frame = simulation_raw_frame(raw_result)
    if format_label == "CSV":
        return frame.to_csv(index=False).encode("utf-8"), "csv", "text/csv"
    if format_label == "TXT":
        return (
            frame.to_csv(index=False, sep="\t").encode("utf-8"),
            "txt",
            "text/plain",
        )

    array_buffer = BytesIO()
    np.save(array_buffer, frame.to_numpy())
    return array_buffer.getvalue(), "npy", "application/octet-stream"


def build_simulation_preview(
    simulation_result: dict[str, object],
    settings: dict[str, object],
) -> dict[str, object]:
    import matplotlib.pyplot as plt
    from dnascape.plotting import mplotf, ploth, plotf

    output_label = str(settings["preview_output_label"])
    output_spec = SIMULATION_OUTPUTS[output_label]
    output_key = str(output_spec["key"])
    values = simulation_result.get(output_key)
    if values is None:
        raise ValueError(f"{output_label} is not available for the current rsim settings.")

    kind = str(output_spec["kind"])
    preview_mode = str(
        settings.get(
            "preview_mode",
            "Time array" if kind == "xt" else "Single view",
        )
    )
    title = settings["title"] or f"{settings['chromosome']}: {output_label.lower()}"
    positions = np.asarray(settings.get("position_kb", []), dtype=float)
    time_grid = np.asarray(simulation_result.get("time_grid", []), dtype=float)
    label = settings["label"] or str(output_spec.get("label", output_label))
    y_limits = (
        (settings["y_limit_min"], settings["y_limit_max"])
        if settings["use_y_limits"]
        else (None, None)
    )

    if preview_mode == "Histogram":
        if kind == "xt":
            matrix, positions, time_grid = normalize_preview_matrix(
                values,
                positions,
                time_grid,
            )
            histogram_values = matrix.reshape(-1)
        elif kind == "time":
            histogram_values = normalize_preview_series(values, time_grid)
            if time_grid.size != histogram_values.size:
                time_grid = np.arange(histogram_values.size, dtype=float)
        else:
            histogram_values = normalize_preview_series(values, positions)
            if positions.size != histogram_values.size:
                positions = np.arange(histogram_values.size, dtype=float)

        histogram_values = np.asarray(histogram_values, dtype=float)
        histogram_values = histogram_values[np.isfinite(histogram_values)]
        if histogram_values.size == 0:
            raise ValueError("No finite values are available for the selected histogram preview.")

        if float(np.nanmin(histogram_values)) >= 0.0:
            ploth(
                histogram_values,
                labels=[label] if settings["show_legend"] else None,
                bin_size=settings["hist_bin_size"],
                density=settings["hist_density"],
                statsQ=False,
                figsize=(settings["figure_width"], settings["figure_height"]),
                xtitle=settings["x_axis_title"],
                ytitle=settings["y_axis_title"],
            )
            fig = plt.gcf()
            fig.axes[0].set_title(title)
        else:
            fig, ax = plt.subplots(
                figsize=(settings["figure_width"], settings["figure_height"])
            )
            if settings["hist_bin_size"] > 0:
                data_min = float(np.nanmin(histogram_values))
                data_max = float(np.nanmax(histogram_values))
                bins = np.arange(
                    data_min,
                    data_max + settings["hist_bin_size"],
                    settings["hist_bin_size"],
                )
                if bins.size < 2:
                    bins = 40
            else:
                bins = 40

            ax.hist(
                histogram_values,
                bins=bins,
                density=settings["hist_density"],
                alpha=0.8,
                color=settings["line_color"],
                label=label if settings["show_legend"] else None,
            )
            ax.set_title(title)
            ax.set_xlabel(settings["x_axis_title"])
            ax.set_ylabel(settings["y_axis_title"])
            if settings["show_legend"]:
                ax.legend(loc=LEGEND_LOCATIONS[settings["legend_location_label"]])
            fig.tight_layout()
            fig.patch.set_facecolor("none")
            fig.patch.set_alpha(0)
            ax.set_facecolor("none")
            ax.patch.set_alpha(0)
    elif preview_mode == "Multiple view":
        if kind == "time":
            companion_keys = [output_key]
            for candidate in (
                "replicated_fraction",
                "active_forks",
                "total_forks",
                "firing_factors",
                "stalled_forks",
            ):
                if candidate == output_key or simulation_result.get(candidate) is None:
                    continue
                companion_keys.append(candidate)
                if len(companion_keys) == 3:
                    break

            shared_time = time_grid
            panels = []
            panel_titles = []
            panel_labels = []
            panel_colors = []
            panel_invert = []
            panel_log = []
            panel_limits = []
            panel_label_locations = []
            for candidate_key in companion_keys:
                meta = simulation_output_meta(candidate_key)
                series = normalize_preview_series(
                    simulation_result.get(candidate_key),
                    shared_time,
                )
                if shared_time.size != series.size:
                    shared_time = np.arange(series.size, dtype=float)
                panels.append(series)
                panel_titles.append(
                    settings["y_axis_title"]
                    if candidate_key == output_key
                    else str(meta.get("ytitle", meta["label"]))
                )
                panel_labels.append(
                    [label]
                    if candidate_key == output_key and settings["show_legend"]
                    else None
                )
                panel_colors.append(
                    [settings["line_color"]]
                    if candidate_key == output_key
                    else [str(meta.get("color", "#1f77b4"))]
                )
                panel_invert.append(
                    settings["invert_y_axis"] if candidate_key == output_key else False
                )
                panel_log.append(
                    settings["log_y_axis"] if candidate_key == output_key else False
                )
                panel_limits.append(y_limits if candidate_key == output_key else (None, None))
                panel_label_locations.append(
                    LEGEND_LOCATIONS[settings["legend_location_label"]]
                )

            panel_count = len(panels)
            panel_height = max(
                1.55,
                float(settings["figure_height"]) / max(panel_count, 1),
            )
            mplotf(
                panels,
                mlabels=panel_labels,
                mx_array=[shared_time] * panel_count,
                mytitle=panel_titles,
                mxtitle=settings["x_axis_title"],
                mcolors=panel_colors,
                minvyQ=panel_invert,
                mlogyQ=panel_log,
                mlimsy=panel_limits,
                mlabel_loc=panel_label_locations,
                figsize=(settings["figure_width"], panel_height),
                hspace=0.2,
                mheight_ratios=[0.95, 0.95, 1.1][:panel_count],
            )
            fig = plt.gcf()
            fig.suptitle(title)
            fig.subplots_adjust(top=0.93)
            time_grid = shared_time
        else:
            if kind == "xt":
                plot_values, positions, time_grid = normalize_preview_matrix(
                    values,
                    positions,
                    time_grid,
                )
                output_panel = plot_values
                panel_titles = [
                    "Firing rate",
                    "Fork speed",
                    settings["y_axis_title"],
                ]
                panel_time_axes = [None, None, time_grid]
                panel_cmaps = [None, None, settings["cmap"]]
                panel_colorbars = [False, False, settings["show_colorbar"]]
                panel_colors = [["#b03412"], ["#2ca02c"], None]
                panel_height_ratios = [0.9, 0.9, 1.2]
            else:
                plot_values = normalize_preview_series(values, positions)
                if positions.size != plot_values.size:
                    positions = np.arange(plot_values.size, dtype=float)
                output_panel = plot_values
                panel_titles = [
                    "Firing rate",
                    "Fork speed",
                    settings["y_axis_title"],
                ]
                panel_time_axes = [None, None, None]
                panel_cmaps = [None, None, None]
                panel_colorbars = [False, False, False]
                panel_colors = [["#b03412"], ["#2ca02c"], [settings["line_color"]]]
                panel_height_ratios = [0.95, 0.95, 1.1]

            firing_profile = normalize_preview_series(
                settings.get("input_firing_rate", []),
                positions,
            )
            fork_profile = normalize_preview_series(
                settings.get("input_fork_speed", []),
                positions,
            )
            panels = [firing_profile, fork_profile, output_panel]
            panel_count = len(panels)
            panel_height = max(
                1.55,
                float(settings["figure_height"]) / max(panel_count, 1),
            )
            mplotf(
                panels,
                mlabels=[
                    None,
                    None,
                    [label]
                    if (settings["show_colorbar"] if kind == "xt" else settings["show_legend"])
                    else None,
                ],
                mx_array=[positions] * panel_count,
                mtime_array=panel_time_axes,
                mytitle=panel_titles,
                mxtitle=settings["x_axis_title"],
                mcolors=panel_colors,
                minvyQ=[False, False, settings["invert_y_axis"]],
                mlogyQ=[False, False, settings["log_y_axis"]],
                mlimsy=[(None, None), (None, None), y_limits],
                mcmap=panel_cmaps,
                mcolorbarQ=panel_colorbars,
                mlabel_loc=[
                    LEGEND_LOCATIONS[settings["legend_location_label"]],
                    LEGEND_LOCATIONS[settings["legend_location_label"]],
                    LEGEND_LOCATIONS[settings["legend_location_label"]],
                ],
                figsize=(settings["figure_width"], panel_height),
                hspace=0.2,
                mheight_ratios=panel_height_ratios,
            )
            fig = plt.gcf()
            fig.suptitle(title)
            fig.subplots_adjust(top=0.93)
    elif kind == "space":
        plot_values = normalize_preview_series(values, positions)
        if positions.size != plot_values.size:
            positions = np.arange(plot_values.size, dtype=float)
        fig, _ = plotf(
            plot_values,
            x_array=positions,
            labels=[label] if settings["show_legend"] else None,
            title=title,
            xtitle=settings["x_axis_title"],
            ytitle=settings["y_axis_title"],
            figsize=(settings["figure_width"], settings["figure_height"]),
            invyQ=settings["invert_y_axis"],
            limsy=y_limits,
            logyQ=settings["log_y_axis"],
            label_loc=LEGEND_LOCATIONS[settings["legend_location_label"]],
            colors=[settings["line_color"]],
            showQ=False,
            return_handles=True,
        )
    elif kind == "time":
        plot_values = normalize_preview_series(values, time_grid)
        if time_grid.size != plot_values.size:
            time_grid = np.arange(plot_values.size, dtype=float)
        fig, _ = plotf(
            plot_values,
            x_array=time_grid,
            labels=[label] if settings["show_legend"] else None,
            title=title,
            xtitle=settings["x_axis_title"],
            ytitle=settings["y_axis_title"],
            figsize=(settings["figure_width"], settings["figure_height"]),
            invyQ=settings["invert_y_axis"],
            limsy=y_limits,
            logyQ=settings["log_y_axis"],
            label_loc=LEGEND_LOCATIONS[settings["legend_location_label"]],
            colors=[settings["line_color"]],
            showQ=False,
            return_handles=True,
        )
    else:
        matrix, positions, time_grid = normalize_preview_matrix(
            values,
            positions,
            time_grid,
        )
        fig, _ = plotf(
            matrix,
            x_array=positions,
            time_array=time_grid,
            labels=[label] if settings["show_colorbar"] else None,
            title=title,
            xtitle=settings["x_axis_title"],
            ytitle=settings["y_axis_title"],
            figsize=(settings["figure_width"], settings["figure_height"]),
            invyQ=settings["invert_y_axis"],
            limsy=y_limits,
            logyQ=settings["log_y_axis"],
            cmap=settings["cmap"],
            colorbarQ=settings["show_colorbar"],
            showQ=False,
            return_handles=True,
        )

    file_stem = safe_download_stem(
        "dnascape",
        "simulation",
        output_key,
        safe_download_stem(preview_mode.lower().replace(" ", "_")),
        settings["chromosome"],
        f"{settings['region_start']:g}-{settings['region_end']:g}kb",
    )
    preview_result = {
        "exports": figure_export_bytes(fig),
        "file_stem": file_stem,
        "raw": {
            "kind": kind,
            "value_column": safe_download_stem(output_key).lower() or "value",
            "values": np.asarray(values, dtype=float),
            "position_kb": positions,
            "time_min": time_grid,
        },
    }
    plt.close(fig)
    return preview_result


def run_simulation_job(
    source_payload: dict[str, object],
    assumptions: dict[str, object],
    updates,
    cancel_event,
) -> None:
    try:
        from dnascape import rsim

        with contextlib.redirect_stdout(RSimProgressWriter(updates, cancel_event)):
            result = rsim(
                source=source_payload,
                assumptions=assumptions,
            )
        if cancel_event.is_set():
            raise SimulationCancelled()
    except SimulationCancelled:
        updates.put({"status": "cancelled"})
    except Exception as exc:
        updates.put({"status": "error", "message": str(exc)})
    else:
        updates.put({"status": "complete", "result": result})


def start_simulation_job(
    source_payload: dict[str, object],
    assumptions: dict[str, object],
    preview_settings: dict[str, object],
) -> dict[str, object]:
    updates = queue.Queue()
    cancel_event = threading.Event()
    job = {
        "updates": updates,
        "cancel_event": cancel_event,
        "preview_settings": preview_settings,
        "progress": {"value": 0.05, "text": "Simulating..."},
    }
    thread = threading.Thread(
        target=run_simulation_job,
        args=(
            dict(source_payload),
            dict(assumptions),
            updates,
            cancel_event,
        ),
        daemon=True,
    )
    job["thread"] = thread
    thread.start()
    return job


def drain_simulation_job(job: dict[str, object]) -> dict[str, object] | None:
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


active_simulation_job = st.session_state.get(SIM_JOB_KEY)
source_inputs = {}
primary_table = None
primary_columns = []
primary_values = None
primary_feature = None
chromosome = "simulated_region"

with controls:
    with st.container(border=True):
        with st.expander("Data", expanded=True):
            source_cols = st.columns(2, gap="small")

            with source_cols[0]:
                firing_mode = st.selectbox(
                    "Firing rate",
                    LINE_INPUT_MODES,
                    index=0,
                )
                if firing_mode == "Profile":
                    firing_upload = st.file_uploader(
                        "Upload profile",
                        type=["tsv", "csv", "txt", "gz"],
                        accept_multiple_files=False,
                        key="simulation_firing_upload",
                    )
                    firing_table, firing_columns, firing_error = read_numeric_table(
                        firing_upload
                    )
                    if firing_error:
                        st.error(f"Could not read firing rate profile: {firing_error}")
                    elif firing_table is not None and not firing_columns:
                        st.warning("No numeric chromosomes were found in the firing rate profile.")

                    firing_chromosome = st.selectbox(
                        "Chromosome",
                        firing_columns,
                        index=0 if firing_columns else None,
                        disabled=not firing_columns,
                        placeholder="Select chromosome",
                        key="simulation_firing_chromosome",
                    )
                    firing_values = None
                    if firing_table is not None and firing_chromosome:
                        firing_values = pd.to_numeric(
                            firing_table[firing_chromosome],
                            errors="coerce",
                        ).to_numpy(dtype=float)

                    source_inputs["firing_rate"] = {
                        "mode": "profile",
                        "table": firing_table,
                        "columns": firing_columns,
                        "chromosome": firing_chromosome,
                        "values": firing_values,
                    }
                    if primary_values is None and firing_values is not None:
                        primary_table = firing_table
                        primary_columns = firing_columns
                        primary_values = firing_values
                        primary_feature = "firing_rate"
                        chromosome = firing_chromosome or "firing_rate"
                else:
                    firing_value = st.number_input(
                        "Constant rate",
                        min_value=0.0,
                        value=0.01,
                        step=0.001,
                        format="%.6g",
                    )
                    source_inputs["firing_rate"] = {
                        "mode": "constant",
                        "value": float(firing_value),
                    }

            with source_cols[1]:
                fork_mode = st.selectbox(
                    "Fork speed",
                    LINE_INPUT_MODES[::-1],
                    index=0,
                )
                if fork_mode == "Profile":
                    fork_upload = st.file_uploader(
                        "Upload profile",
                        type=["tsv", "csv", "txt", "gz"],
                        accept_multiple_files=False,
                        key="simulation_fork_upload",
                    )
                    fork_table, fork_columns, fork_error = read_numeric_table(
                        fork_upload
                    )
                    if fork_error:
                        st.error(f"Could not read fork-speed profile: {fork_error}")
                    elif fork_table is not None and not fork_columns:
                        st.warning("No numeric chromosomes were found in the fork-speed profile.")

                    fork_chromosome = st.selectbox(
                        "Chromosome",
                        fork_columns,
                        index=0 if fork_columns else None,
                        disabled=not fork_columns,
                        placeholder="Select chromosome",
                        key="simulation_fork_chromosome",
                    )
                    fork_values = None
                    if fork_table is not None and fork_chromosome:
                        fork_values = pd.to_numeric(
                            fork_table[fork_chromosome],
                            errors="coerce",
                        ).to_numpy(dtype=float)

                    source_inputs["fork_speed"] = {
                        "mode": "profile",
                        "table": fork_table,
                        "columns": fork_columns,
                        "chromosome": fork_chromosome,
                        "values": fork_values,
                    }
                    if primary_values is None and fork_values is not None:
                        primary_table = fork_table
                        primary_columns = fork_columns
                        primary_values = fork_values
                        primary_feature = "fork_speed"
                        chromosome = fork_chromosome or "fork_speed"
                else:
                    fork_value = st.number_input(
                        "Constant speed",
                        min_value=0.000001,
                        value=1.4,
                        step=0.1,
                        format="%.3g",
                        help="Fork speed in kb/min.",
                    )
                    source_inputs["fork_speed"] = {
                        "mode": "constant",
                        "value": float(fork_value),
                    }

        with st.expander("Region", expanded=False):
            region_top_cols = st.columns(2, gap="small")
            with region_top_cols[0]:
                resolution = st.number_input(
                    "Resolution",
                    min_value=0.000001,
                    value=1.0,
                    step=1.0,
                    format="%.1f",
                    help="Spatial resolution in kb/bin.",
                )
            with region_top_cols[1]:
                resolution_time = st.number_input(
                    "Time step",
                    min_value=0.000001,
                    value=5.0,
                    step=1.0,
                    format="%.1f",
                    help="Temporal resolution in min/bin for time-dependent outputs.",
                )

            if primary_values is not None:
                finite_primary = np.isfinite(primary_values)
                primary_positions_full = (
                    np.arange(primary_values.size, dtype=float) * float(resolution)
                )
                primary_positions = primary_positions_full[finite_primary]
                max_position = (
                    max(1, int(np.ceil(float(np.nanmax(primary_positions)))))
                    if primary_positions.size
                    else max(1, int(np.ceil(float(resolution))))
                )
                region_cols = st.columns(2, gap="small")
                with region_cols[0]:
                    region_start = st.number_input(
                        "Start (kb)",
                        min_value=0,
                        max_value=max_position,
                        value=0,
                        step=1,
                        format="%d",
                    )
                with region_cols[1]:
                    region_end = st.number_input(
                        "End (kb)",
                        min_value=0,
                        max_value=max_position,
                        value=max_position,
                        step=1,
                        format="%d",
                    )
            else:
                region_cols = st.columns(2, gap="small")
                with region_cols[0]:
                    region_start = st.number_input(
                        "Start (kb)",
                        min_value=0,
                        value=0,
                        step=1,
                        format="%d",
                    )
                with region_cols[1]:
                    region_end = st.number_input(
                        "End (kb)",
                        min_value=1,
                        value=250,
                        step=1,
                        format="%d",
                    )

        with st.expander("Simulation settings", expanded=False):
            setup_cols = st.columns(2, gap="small")
            with setup_cols[0]:
                sim_number = st.number_input(
                    "Replicates",
                    min_value=1,
                    max_value=5000,
                    value=20,
                    step=1,
                    format="%d",
                )
            with setup_cols[1]:
                geometry_label = st.selectbox(
                    "Geometry",
                    GEOMETRY_OPTIONS,
                    index=0,
                )

            timing_cols = st.columns(2, gap="small")
            with timing_cols[0]:
                time_max = st.number_input(
                    "Time window",
                    min_value=1.0,
                    value=300.0,
                    step=10.0,
                    format="%.1f",
                    help="Upper bound of the preview time grid in minutes.",
                )
            with timing_cols[1]:
                max_rep_time = st.number_input(
                    "Max replication time",
                    min_value=1.0,
                    value=1200.0,
                    step=50.0,
                    format="%.1f",
                )

            stall_cols = st.columns(2, gap="small")
            with stall_cols[0]:
                stall_rate = st.number_input(
                    "Stall rate",
                    min_value=0.0,
                    value=0.0,
                    step=0.01,
                    format="%.3f",
                )
            with stall_cols[1]:
                recover_stalls = st.checkbox(
                    "Recover stalled forks",
                    value=True,
                    disabled=stall_rate <= 0.0,
                )

            tau = st.number_input(
                "Stall duration",
                min_value=0.000001,
                value=15.0,
                step=1.0,
                format="%.1f",
                disabled=stall_rate <= 0.0 or not recover_stalls,
                help="Mean recovery time in minutes for transient stalls.",
            )

            use_firing_factors = st.checkbox(
                "Finite firing-factor pool",
                value=False,
            )
            if use_firing_factors:
                factor_cols = st.columns(2, gap="small")
                with factor_cols[0]:
                    ffiring_nr = st.number_input(
                        "Pool size",
                        min_value=2.0,
                        value=60.0,
                        step=1.0,
                        format="%.1f",
                    )
                with factor_cols[1]:
                    ffiring_recycle = st.number_input(
                        "Recycle rate",
                        min_value=0.0,
                        value=0.0,
                        step=0.05,
                        format="%.2f",
                    )
                ffiring_forkQ = st.checkbox(
                    "Couple to active forks",
                    value=False,
                )
            else:
                ffiring_nr = None
                ffiring_recycle = 0.0
                ffiring_forkQ = False

            seed_cols = st.columns(2, gap="small")
            with seed_cols[0]:
                use_seed = st.checkbox("Fixed seed", value=False)
            with seed_cols[1]:
                seed_value = st.number_input(
                    "Seed",
                    min_value=0,
                    value=1,
                    step=1,
                    format="%d",
                    disabled=not use_seed,
                )

        with st.expander("Plotting settings", expanded=False):
            selector_cols = st.columns([1.2, 0.8], gap="small")
            with selector_cols[0]:
                preview_output_label = st.selectbox(
                    "Output",
                    list(SIMULATION_OUTPUTS),
                    index=0,
                )

            output_spec = SIMULATION_OUTPUTS[preview_output_label]
            output_kind = str(output_spec["kind"])
            is_heatmap = output_kind == "xt"
            view_options = SIMULATION_VIEW_OPTIONS[output_kind]
            with selector_cols[1]:
                preview_mode = st.selectbox(
                    "View",
                    view_options,
                    index=0,
                )

            is_histogram = preview_mode == "Histogram"
            if is_histogram:
                default_x_axis_title = str(output_spec.get("label", preview_output_label))
                default_y_axis_title = "Frequency"
            elif output_kind == "space":
                default_x_axis_title = "Genomic position (kb)"
                default_y_axis_title = str(output_spec["ytitle"])
            elif output_kind == "time":
                default_x_axis_title = "S-phase time (min)"
                default_y_axis_title = str(output_spec["ytitle"])
            else:
                default_x_axis_title = "Genomic position (kb)"
                default_y_axis_title = "S-phase time (min)"

            text_cols = st.columns([1.15, 0.85], gap="small")
            with text_cols[0]:
                title_override = st.text_input(
                    "Title",
                    value="",
                    placeholder="Auto",
                )
            with text_cols[1]:
                label_override = st.text_input(
                    "Label",
                    value="",
                    placeholder=str(output_spec.get("label", preview_output_label)),
                )

            axis_cols = st.columns(2, gap="small")
            with axis_cols[0]:
                x_axis_title = st.text_input(
                    "x-axis",
                    value=default_x_axis_title,
                )
            with axis_cols[1]:
                y_axis_title = st.text_input(
                    "y-axis",
                    value=default_y_axis_title,
                )

            line_color = str(output_spec.get("color", "#1f77b4"))
            show_legend = False
            legend_location_label = "Best"
            show_colorbar = False
            cmap = str(output_spec.get("cmap", HEATMAP_PALETTES[0]))
            invert_y_axis = False
            log_y_axis = False
            use_y_limits = False
            hist_density = False
            hist_bin_size = 5.0 if output_kind != "time" else 2.0

            if is_histogram:
                histogram_cols = st.columns([0.78, 0.74, 0.98], gap="small")
                with histogram_cols[0]:
                    show_legend = st.checkbox("Legend", value=False)
                with histogram_cols[1]:
                    hist_density = st.checkbox("Density", value=False)
                with histogram_cols[2]:
                    hist_bin_size = st.number_input(
                        "Bin size",
                        min_value=0.1,
                        value=hist_bin_size,
                        step=0.5 if output_kind != "time" else 0.2,
                        format="%.2f",
                    )
            elif is_heatmap:
                style_cols = st.columns([0.78, 1.22], gap="small")
                with style_cols[0]:
                    show_colorbar = st.checkbox("Colorbar", value=True)
                with style_cols[1]:
                    cmap = st.selectbox(
                        "Palette",
                        HEATMAP_PALETTES,
                        index=HEATMAP_PALETTES.index(str(output_spec["cmap"])),
                    )

                scale_cols = st.columns(3, gap="small")
                with scale_cols[0]:
                    invert_y_axis = st.checkbox("Invert y-axis", value=False)
                with scale_cols[1]:
                    log_y_axis = st.checkbox("Log y-axis", value=False)
                with scale_cols[2]:
                    use_y_limits = st.checkbox("Y limits", value=False)
            else:
                style_cols = st.columns([0.72, 1.0, 0.92], gap="small")
                with style_cols[0]:
                    show_legend = st.checkbox("Legend", value=True)
                with style_cols[1]:
                    legend_location_label = st.selectbox(
                        "Position",
                        list(LEGEND_LOCATIONS),
                        index=0,
                        disabled=not show_legend,
                    )
                with style_cols[2]:
                    line_color = st.color_picker(
                        "Color",
                        value=str(output_spec["color"]),
                    )

                scale_cols = st.columns(3, gap="small")
                with scale_cols[0]:
                    invert_y_axis = st.checkbox("Invert y-axis", value=False)
                with scale_cols[1]:
                    log_y_axis = st.checkbox("Log y-axis", value=False)
                with scale_cols[2]:
                    use_y_limits = st.checkbox("Y limits", value=False)

            default_figure_height = 3.6 if is_histogram else (4.6 if preview_mode == "Multiple view" else (4.1 if is_heatmap else 3.8))

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
                    max_value=12.0,
                    value=default_figure_height,
                    step=0.2,
                    format="%.1f",
                )

            y_limit_min = 0.0
            y_limit_max = 1.0
            if use_y_limits:
                y_limit_cols = st.columns(2, gap="small")
                with y_limit_cols[0]:
                    y_limit_min = st.number_input(
                        "Min",
                        value=0.0,
                        step=1.0 if output_spec["kind"] != "time" else 0.1,
                    )
                with y_limit_cols[1]:
                    y_limit_max = st.number_input(
                        "Max",
                        value=1.0,
                        step=1.0 if output_spec["kind"] != "time" else 0.1,
                    )

        missing_simulation_inputs = []
        if (
            source_inputs.get("firing_rate", {}).get("mode") == "profile"
            and source_inputs.get("firing_rate", {}).get("values") is None
        ):
            missing_simulation_inputs.append("Firing rate")
        if (
            source_inputs.get("fork_speed", {}).get("mode") == "profile"
            and source_inputs.get("fork_speed", {}).get("values") is None
        ):
            missing_simulation_inputs.append("Fork speed")

        simulate_requested = st.button(
            "Simulate",
            type="primary",
            width="stretch",
        )

with preview:
    with st.container(border=True):
        st.markdown(
            '<div class="plotting-panel-title">Preview</div>',
            unsafe_allow_html=True,
        )

        last_simulation = st.session_state.get(SIM_STATE_KEY)
        simulation_feedback_shown = False
        active_simulation_job = st.session_state.get(SIM_JOB_KEY)

        if simulate_requested:
            simulation_feedback_shown = True
            if missing_simulation_inputs:
                st.info(
                    "Add the required input data for "
                    + ", ".join(missing_simulation_inputs)
                    + " before simulating."
                )
            elif region_end <= region_start:
                st.error("Region end must be greater than the region start.")
            elif time_max < resolution_time:
                st.error("The time window must be at least as large as the selected time step.")
            elif use_y_limits and y_limit_max <= y_limit_min:
                st.error("The y-axis maximum must be greater than the minimum.")
            else:
                source_payload = {}
                preview_positions = np.array([], dtype=float)
                reference_mask = None

                for feature, input_state in source_inputs.items():
                    if input_state["mode"] == "constant":
                        source_payload[feature] = float(input_state["value"])
                        continue

                    raw_values = np.asarray(input_state["values"], dtype=float)
                    raw_positions = np.arange(raw_values.size, dtype=float) * float(
                        resolution
                    )
                    region_mask = (
                        np.isfinite(raw_values)
                        & (raw_positions >= float(region_start))
                        & (raw_positions <= float(region_end))
                    )

                    if reference_mask is None:
                        reference_mask = region_mask
                        preview_positions = raw_positions[region_mask]
                    elif raw_values.shape[0] == reference_mask.shape[0]:
                        region_mask = reference_mask & np.isfinite(raw_values)

                    source_payload[feature] = raw_values[region_mask]

                if reference_mask is None:
                    preview_positions = interval_positions(
                        region_start,
                        region_end,
                        resolution,
                    )

                uploaded_lengths = [
                    int(np.asarray(value).size)
                    for value in source_payload.values()
                    if np.asarray(value).ndim > 0
                ]
                if uploaded_lengths and any(length == 0 for length in uploaded_lengths):
                    st.warning("The selected region does not contain any bins.")
                elif len(set(uploaded_lengths)) > 1:
                    st.error(
                        "Uploaded spatial profiles do not share the same number of bins in the selected region."
                    )
                else:
                    if active_simulation_job is not None:
                        active_simulation_job["cancel_event"].set()

                    simulation_assumptions = {
                        "resolution": float(resolution),
                        "resolution_time": float(resolution_time),
                        "sim_number": int(sim_number),
                        "geometry": geometry_label.lower(),
                        "stall_rate": float(stall_rate),
                        "tau": float(tau) if stall_rate > 0.0 and recover_stalls else np.inf,
                        "ffiring_nr": ffiring_nr,
                        "ffiring_recycle": float(ffiring_recycle),
                        "ffiring_forkQ": bool(ffiring_forkQ),
                        "max_rep_time": float(max_rep_time),
                        "seed": int(seed_value) if use_seed else None,
                        "verbose": True,
                        "print_every": 1,
                    }

                    output_spec = SIMULATION_OUTPUTS[preview_output_label]
                    if output_spec["kind"] in {"time", "xt"}:
                        time_grid = simulation_time_grid(resolution_time, time_max)
                        simulation_assumptions["time_grid"] = time_grid
                        if output_spec["kind"] == "time":
                            simulation_assumptions["time_statsQ"] = True
                        else:
                            simulation_assumptions["time_stats_xtQ"] = True

                    if not uploaded_lengths:
                        simulation_assumptions["interval_kb"] = (
                            float(region_start),
                            float(region_end),
                        )

                    active_simulation_job = start_simulation_job(
                        source_payload,
                        simulation_assumptions,
                        {
                            "chromosome": chromosome,
                            "region_start": region_start,
                            "region_end": region_end,
                            "preview_output_label": preview_output_label,
                            "preview_mode": preview_mode,
                            "position_kb": np.asarray(preview_positions, dtype=float),
                            "input_firing_rate": source_payload.get("firing_rate"),
                            "input_fork_speed": source_payload.get("fork_speed"),
                            "title": title_override.strip(),
                            "label": label_override.strip(),
                            "x_axis_title": x_axis_title,
                            "y_axis_title": y_axis_title,
                            "figure_width": figure_width,
                            "figure_height": figure_height,
                            "use_y_limits": use_y_limits,
                            "y_limit_min": y_limit_min,
                            "y_limit_max": y_limit_max,
                            "invert_y_axis": invert_y_axis,
                            "log_y_axis": log_y_axis,
                            "show_legend": show_legend,
                            "legend_location_label": legend_location_label,
                            "line_color": line_color,
                            "show_colorbar": show_colorbar,
                            "cmap": cmap,
                            "hist_bin_size": hist_bin_size,
                            "hist_density": hist_density,
                        },
                    )
                    st.session_state[SIM_JOB_KEY] = active_simulation_job

        if active_simulation_job is not None:
            terminal_update = drain_simulation_job(active_simulation_job)

            if terminal_update is None:
                progress = active_simulation_job["progress"]
                st.progress(progress["value"], text=progress["text"])
                if st.button(
                    "Stop simulation",
                    type="secondary",
                    width="stretch",
                    key="simulation_stop_button",
                ):
                    active_simulation_job["cancel_event"].set()
                    active_simulation_job["progress"] = {
                        "value": progress["value"],
                        "text": "Stopping simulation...",
                    }
                    st.rerun()

                if active_simulation_job["thread"].is_alive():
                    time.sleep(0.25)
                    st.rerun()
                else:
                    time.sleep(0.05)
                    st.rerun()
            else:
                completed_job = active_simulation_job
                st.session_state.pop(SIM_JOB_KEY, None)
                active_simulation_job = None

                if terminal_update["status"] == "complete":
                    render_slot = st.empty()
                    render_slot.progress(0.98, text="Rendering preview...")
                    try:
                        last_simulation = build_simulation_preview(
                            terminal_update["result"],
                            completed_job["preview_settings"],
                        )
                    except Exception as exc:
                        render_slot.empty()
                        st.error(f"Could not render the simulation preview: {exc}")
                    else:
                        st.session_state[SIM_STATE_KEY] = last_simulation
                        render_slot.empty()
                elif terminal_update["status"] == "cancelled":
                    st.info("Simulation stopped.")
                elif terminal_update["status"] == "error":
                    st.error(
                        f"rsim could not complete this simulation: {terminal_update['message']}"
                    )

        if active_simulation_job is not None:
            pass
        elif last_simulation is not None:
            render_figure_preview(last_simulation["exports"], last_simulation["file_stem"])
        elif simulation_feedback_shown:
            pass
        elif primary_table is None:
            st.markdown(
                """
                <p class="small-muted">
                Choose kinetic inputs, define a genomic interval, and run rsim to preview a simulated output.
                </p>
                """,
                unsafe_allow_html=True,
            )
        elif primary_values is None or primary_values.size == 0:
            st.info("Choose a numeric input chromosome.")
        else:
            st.dataframe(
                primary_table.loc[:, primary_columns].head(8),
                width="stretch",
                hide_index=True,
            )

        with st.expander("Raw output", expanded=False):
            last_simulation_for_download = st.session_state.get(SIM_STATE_KEY)
            if (
                last_simulation_for_download is not None
                and "raw" in last_simulation_for_download
            ):
                st.markdown(
                    """
                    <p class="small-muted">
                    Download the currently previewed rsim output together with its coordinate axes.
                    </p>
                    """,
                    unsafe_allow_html=True,
                )
                st.caption("Export")
                export_cols = st.columns([0.34, 0.26, 0.40], gap="small")
                with export_cols[0]:
                    raw_format = st.selectbox(
                        "Format",
                        ["CSV", "TXT", "Array"],
                        index=0,
                        key="simulation_raw_format",
                        label_visibility="collapsed",
                    )
                payload, extension, mime = simulation_raw_payload(
                    last_simulation_for_download["raw"],
                    raw_format,
                )
                with export_cols[1]:
                    st.download_button(
                        "Download",
                        data=payload,
                        file_name=f"{last_simulation_for_download['file_stem']}_raw.{extension}",
                        mime=mime,
                        on_click="ignore",
                        width="stretch",
                    )
            else:
                st.markdown(
                    """
                    <p class="small-muted">
                    Run a simulation to enable raw output export.
                    </p>
                    """,
                    unsafe_allow_html=True,
                )
