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
    "Choose a target, set a precise rmap route, and preview mapped DNAscape outputs.",
)

st.markdown(
    """
    <span class="plotting-page-marker mapping-page-marker" aria-hidden="true"></span>
    <style>
      body:has(.mapping-page-marker) [data-testid="stSelectbox"] [data-baseweb="select"],
      body:has(.mapping-page-marker) [data-testid="stSelectbox"] [data-baseweb="select"] *,
      body:has(.mapping-page-marker) [data-testid="stSelectbox"] [data-baseweb="select"] input {
        color: var(--dnascape-text) !important;
        opacity: 1 !important;
        -webkit-text-fill-color: var(--dnascape-text) !important;
      }

      body:has(.mapping-page-marker) [data-testid="stSelectbox"] [data-baseweb="select"] svg {
        color: var(--dnascape-muted) !important;
        fill: currentColor !important;
        -webkit-text-fill-color: currentColor !important;
      }

      body:has(.mapping-page-marker) [data-baseweb="popover"] [role="option"],
      body:has(.mapping-page-marker) [data-baseweb="popover"] [role="option"] *,
      body:has(.mapping-page-marker) [data-baseweb="menu"] [role="option"],
      body:has(.mapping-page-marker) [data-baseweb="menu"] [role="option"] * {
        color: var(--dnascape-text) !important;
        opacity: 1 !important;
        -webkit-text-fill-color: var(--dnascape-text) !important;
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
MAP_STATE_KEY = "mapping_last_map"
MAP_JOB_KEY = "mapping_active_job"
CLEAR_PREVIEW_PARAM = "dnascape_clear_preview"
CLEAR_PREVIEW_TOKEN = "mapping"
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
FEATURE_COLORS = {
    "replication_timing": "#1f77b4",
    "firing_rate": "#b03412",
    "fork_speed": "#2ca02c",
    "fork_directionality": "#9467bd",
}
FEATURE_LABELS = {
    "replication_timing": "Replication timing",
    "firing_rate": "Initiation rate",
    "fork_speed": "Fork speed",
    "fork_directionality": "Fork directionality",
}
PREVIEW_CONTENT_OPTIONS = ("Target", "Input", "Both")
TARGET_OPTIONS = {
    "Initiation rate": "firing_rate",
    "Fork speed": "fork_speed",
    "Replication timing": "replication_timing",
    "Fork directionality": "fork_directionality",
}
DEPENDENCE_TYPES = {
    "Constant": "constant",
    "Space": "space",
    "Time": "time",
    "Space-time": "spacetime",
}
TARGET_DEPENDENCE_OPTIONS = {
    "firing_rate": ["Space", "Time", "Space-time", "Constant"],
    "fork_speed": ["Space", "Time", "Space-time", "Constant"],
    "replication_timing": ["Space"],
    "fork_directionality": ["Space"],
}
TARGET_SHAPES = {
    "Weibull": "weibull",
    "Free": "free",
}
ROUTE_CATALOG = (
    {
        "id": "timing_to_firing",
        "target": "firing_rate",
        "forms": ("constant", "space", "time", "spacetime"),
        "label": "Replication timing + fork speed",
        "inputs": (
            {"feature": "replication_timing", "mode": "upload"},
            {
                "feature": "fork_speed",
                "mode": "optional_constant",
                "default_constant": True,
                "value": 1.4,
                "step": 0.1,
                "format": "%.2f",
                "help": "Constant fork speed in kb/min.",
            },
        ),
    },
    {
        "id": "timing_to_speed",
        "target": "fork_speed",
        "forms": ("constant", "space", "time", "spacetime"),
        "label": "Replication timing + initiation rate",
        "inputs": (
            {"feature": "replication_timing", "mode": "upload"},
            {
                "feature": "firing_rate",
                "mode": "optional_constant",
                "default_constant": False,
                "value": 0.0001,
                "step": 0.00001,
                "format": "%.6g",
            },
        ),
    },
    {
        "id": "firing_to_timing",
        "target": "replication_timing",
        "forms": ("space",),
        "label": "Initiation rate + fork speed",
        "inputs": (
            {"feature": "firing_rate", "mode": "upload"},
            {
                "feature": "fork_speed",
                "mode": "optional_constant",
                "default_constant": True,
                "value": 1.4,
                "step": 0.1,
                "format": "%.2f",
                "help": "Constant fork speed in kb/min.",
            },
        ),
    },
    {
        "id": "timing_to_directionality",
        "target": "fork_directionality",
        "forms": ("space",),
        "label": "Replication timing + fork speed",
        "inputs": (
            {"feature": "replication_timing", "mode": "upload"},
            {
                "feature": "fork_speed",
                "mode": "optional_constant",
                "default_constant": True,
                "value": 1.4,
                "step": 0.1,
                "format": "%.2f",
                "help": "Constant fork speed in kb/min.",
            },
        ),
    },
)


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


def feature_label(feature: str) -> str:
    return FEATURE_LABELS.get(feature, feature.replace("_", " ").title())


def feature_column_name(feature: str) -> str:
    return safe_download_stem(feature).lower() or "value"


def feature_color(feature: str) -> str:
    return FEATURE_COLORS.get(feature, "#7d8590")


def routes_for(target: str, target_form: str) -> list[dict[str, object]]:
    return [
        route
        for route in ROUTE_CATALOG
        if route["target"] == target and target_form in route["forms"]
    ]


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


def raw_result_frame(raw_result: dict[str, np.ndarray]) -> pd.DataFrame:
    values = np.asarray(raw_result["values"], dtype=float)
    target_form = raw_result.get("target_form", "space")
    value_column = raw_result.get("value_column", "value")

    if values.size == 1:
        return pd.DataFrame({value_column: values.reshape(-1)})

    if target_form == "time" or (values.ndim == 2 and values.shape[0] == 1):
        time_axis = np.asarray(raw_result.get("time_min", []), dtype=float)
        temporal = values.reshape(-1)
        if time_axis.size != temporal.size:
            time_axis = np.arange(temporal.size, dtype=float)
        return pd.DataFrame(
            {
                "time_min": time_axis,
                value_column: temporal,
            }
        )

    positions = np.asarray(raw_result["position_kb"], dtype=float)
    if target_form == "spacetime" or values.ndim == 2:
        time_axis = np.asarray(raw_result.get("time_min", []), dtype=float)
        matrix = values
        if matrix.ndim != 2:
            matrix = np.atleast_2d(matrix)
        if matrix.shape[0] != positions.size and matrix.shape[1] == positions.size:
            matrix = matrix.T
        if not time_axis.size or time_axis.size != matrix.shape[1]:
            time_axis = np.arange(matrix.shape[1], dtype=float)

        rows = []
        spatial_axis = positions[: matrix.shape[0]]
        if spatial_axis.size != matrix.shape[0]:
            spatial_axis = np.arange(matrix.shape[0], dtype=float)
        for time_index, time_value in enumerate(time_axis):
            rows.append(
                pd.DataFrame(
                    {
                        "position_kb": spatial_axis,
                        "time_min": time_value,
                        value_column: matrix[:, time_index],
                    }
                )
            )
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    spatial = values.reshape(-1)
    spatial_axis = positions[: spatial.size]
    if spatial_axis.size != spatial.size:
        spatial_axis = np.arange(spatial.size, dtype=float)
    return pd.DataFrame(
        {
            "position_kb": spatial_axis,
            value_column: spatial,
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
    np.save(array_buffer, frame.to_numpy())
    return array_buffer.getvalue(), "npy", "application/octet-stream"


def build_time_grid(timing_values: np.ndarray, resolution_time: float) -> np.ndarray:
    resolution_time = float(resolution_time)
    finite = np.asarray(timing_values, dtype=float)
    finite = finite[np.isfinite(finite) & (finite > 0)]
    max_timing = float(np.nanmax(finite)) if finite.size else resolution_time
    max_time = max(max_timing * 3.0, max_timing + 5.0 * resolution_time)
    return np.arange(0.0, max_time + resolution_time, resolution_time)


def input_preview_series(settings: dict[str, object]) -> tuple[np.ndarray, np.ndarray, str, str, str] | None:
    input_feature = settings.get("input_feature")
    input_values = np.asarray(settings.get("input_values", []), dtype=float)
    input_positions = np.asarray(settings.get("input_positions", []), dtype=float)
    chromosome = str(settings.get("chromosome", "profile"))

    if input_feature is None or input_values.size == 0:
        return None

    if input_positions.size != input_values.size:
        input_positions = np.arange(input_values.size, dtype=float)

    input_label = feature_label(str(input_feature))
    return (
        input_positions,
        input_values.reshape(-1),
        "Genomic position (kb)",
        f"{chromosome}: {input_label.lower()}",
        input_label,
    )


def mapping_preview_series(
    mapped_values: np.ndarray,
    map_positions: np.ndarray,
    settings: dict[str, object],
) -> tuple[np.ndarray, np.ndarray, str, str, str]:
    values = np.asarray(mapped_values, dtype=float)
    target_form = settings["target_form"]
    chromosome = settings["chromosome"]
    target_label = feature_label(settings["target_feature"])

    if values.size == 1:
        return (
            np.array([0.0], dtype=float),
            values.reshape(-1),
            "",
            f"{chromosome}: {target_label.lower()}",
            target_label,
        )

    if target_form == "time":
        y_values = values.reshape(-1)
        x_values = np.asarray(settings.get("time_grid", []), dtype=float)
        if x_values.size != y_values.size:
            x_values = np.arange(y_values.size, dtype=float)
        return (
            x_values,
            y_values,
            "S-phase time (min)",
            f"{chromosome}: temporal {target_label.lower()}",
            target_label,
        )

    if target_form == "spacetime":
        matrix = values
        positions = np.asarray(map_positions, dtype=float)
        if matrix.ndim != 2:
            matrix = np.atleast_2d(matrix)
        if matrix.shape[0] != positions.size and matrix.shape[1] == positions.size:
            matrix = matrix.T
        y_values = np.nanmean(matrix, axis=1)
        x_values = positions[: y_values.size]
        if x_values.size != y_values.size:
            x_values = np.arange(y_values.size, dtype=float)
        return (
            x_values,
            y_values,
            "Genomic position (kb)",
            f"{chromosome}: mean {target_label.lower()}",
            f"Mean {target_label.lower()}",
        )

    y_values = values.reshape(-1)
    positions = np.asarray(map_positions, dtype=float)
    x_values = positions[: y_values.size]
    if x_values.size != y_values.size:
        x_values = np.arange(y_values.size, dtype=float)
    return (
        x_values,
        y_values,
        "Genomic position (kb)",
        f"{chromosome}: inferred {target_label.lower()}",
        target_label,
    )


def build_mapping_preview(
    mapped_values: np.ndarray,
    map_positions: np.ndarray,
    settings: dict[str, object],
) -> dict[str, object]:
    import matplotlib.pyplot as plt
    from dnascape.plotting import plotf

    preview_mode = str(settings.get("preview_mode", "Target"))
    target_x, target_y, target_x_title, target_default_title, target_default_label = (
        mapping_preview_series(mapped_values, map_positions, settings)
    )
    input_preview = input_preview_series(settings)
    target_label = settings["label"] or target_default_label

    if preview_mode == "Input" and input_preview is not None:
        input_x, input_y, input_x_title, input_default_title, input_default_label = (
            input_preview
        )
        fig, _ = plotf(
            input_y,
            x_array=input_x,
            labels=[input_default_label] if settings["show_legend"] else None,
            title=settings["title"] or input_default_title,
            xtitle=input_x_title,
            ytitle=input_default_label,
            figsize=(settings["figure_width"], settings["figure_height"]),
            invyQ=settings["invert_y_axis"],
            limsy=(
                (settings["y_limit_min"], settings["y_limit_max"])
                if settings["use_y_limits"]
                else (None, None)
            ),
            logyQ=settings["log_y_axis"],
            label_loc=LEGEND_LOCATIONS[settings["legend_location_label"]],
            colors=[feature_color(str(settings.get("input_feature")))],
            showQ=False,
            return_handles=True,
        )
    elif preview_mode == "Both" and input_preview is not None:
        input_x, input_y, input_x_title, _, input_default_label = input_preview
        plot_title = settings["title"] or (
            f"{settings['chromosome']}: {input_default_label.lower()} and {target_default_label.lower()}"
        )
        fig, axes = plt.subplots(
            2,
            1,
            figsize=(settings["figure_width"], settings["figure_height"] * 1.7),
            sharex=False,
        )
        plotf(
            input_y,
            x_array=input_x,
            labels=[input_default_label] if settings["show_legend"] else None,
            title="Input",
            xtitle=input_x_title,
            ytitle=input_default_label,
            invyQ=settings["invert_y_axis"],
            limsy=(
                (settings["y_limit_min"], settings["y_limit_max"])
                if settings["use_y_limits"]
                else (None, None)
            ),
            logyQ=settings["log_y_axis"],
            label_loc=LEGEND_LOCATIONS[settings["legend_location_label"]],
            colors=[feature_color(str(settings.get("input_feature")))],
            showQ=False,
            return_handles=True,
            fig=fig,
            ax=axes[0],
        )
        plotf(
            target_y,
            x_array=target_x,
            labels=[target_label] if settings["show_legend"] else None,
            title="Target",
            xtitle=settings["x_axis_title"] or target_x_title,
            ytitle=settings["y_axis_title"],
            invyQ=settings["invert_y_axis"],
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
            fig=fig,
            ax=axes[1],
        )
        fig.suptitle(plot_title)
        fig.subplots_adjust(hspace=0.34, top=0.9)
    else:
        plot_title = settings["title"] or target_default_title
        fig, _ = plotf(
            target_y,
            x_array=target_x,
            labels=[target_label] if settings["show_legend"] else None,
            title=plot_title,
            xtitle=settings["x_axis_title"] or target_x_title,
            ytitle=settings["y_axis_title"],
            figsize=(settings["figure_width"], settings["figure_height"]),
            invyQ=settings["invert_y_axis"],
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
        settings["target_form"],
        settings["target_feature"],
        settings["chromosome"],
        f"{settings['region_start']:g}-{settings['region_end']:g}kb",
    )
    mapped_result = {
        "exports": figure_export_bytes(fig),
        "file_stem": file_stem,
        "raw": {
            "position_kb": np.asarray(map_positions, dtype=float),
            "time_min": np.asarray(settings.get("time_grid", []), dtype=float),
            "values": np.asarray(mapped_values, dtype=float),
            "target_feature": settings["target_feature"],
            "target_form": settings["target_form"],
            "value_column": feature_column_name(settings["target_feature"]),
        },
    }
    plt.close(fig)
    return mapped_result


def run_mapping_job(
    source_payload: dict[str, object],
    target_feature: str,
    map_positions: np.ndarray,
    assumptions: dict[str, object],
    updates,
    cancel_event,
) -> None:
    try:
        from dnascape import rmap

        with contextlib.redirect_stdout(RMapProgressWriter(updates, cancel_event)):
            mapped_values = rmap(
                source=source_payload,
                target=target_feature,
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
                "mapped_values": np.asarray(mapped_values, dtype=float),
                "position_kb": np.asarray(map_positions, dtype=float),
            }
        )


def start_mapping_job(
    source_payload: dict[str, object],
    target_feature: str,
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
            dict(source_payload),
            target_feature,
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
            <a class="plot-preview-action plot-preview-clear-button" href="{clear_href}" aria-label="Clear mapping preview" title="Clear preview">
              <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
                <path d="M9 3h6l1 1h4v2H4V4h4l1-1Zm-3 4h12v12a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2V7Zm3 3v8h2v-8H9Zm4 0v8h2v-8h-2Z" fill="currentColor" />
              </svg>
            </a>
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


consume_preview_clear_request(MAP_STATE_KEY, CLEAR_PREVIEW_TOKEN)


active_mapping_job = st.session_state.get(MAP_JOB_KEY)
source_inputs = {}
primary_table = None
primary_columns = []
primary_values = None
primary_positions = None
primary_feature = None
chromosome = "profile"

with controls:
    with st.container(border=True):
        with st.expander("Data", expanded=True):
            target_cols = st.columns(2, gap="small")
            with target_cols[0]:
                target_label = st.selectbox(
                    "Target",
                    list(TARGET_OPTIONS),
                    index=0,
                )
            target_feature = TARGET_OPTIONS[target_label]
            dependence_options = TARGET_DEPENDENCE_OPTIONS[target_feature]
            with target_cols[1]:
                target_dependence_label = st.selectbox(
                    "Dependence",
                    dependence_options,
                    index=0,
                )
            target_form = DEPENDENCE_TYPES[target_dependence_label]

            route_options = routes_for(target_feature, target_form)
            route_labels = [str(route["label"]) for route in route_options]
            route_label = st.selectbox(
                "Route",
                route_labels,
                index=0 if route_labels else None,
                disabled=not route_labels,
            )
            selected_route = (
                route_options[route_labels.index(route_label)] if route_labels else None
            )

            if selected_route is not None:
                for input_spec in selected_route["inputs"]:
                    input_feature = str(input_spec["feature"])
                    input_label = feature_label(input_feature)
                    input_mode = input_spec["mode"]

                    if input_mode in {"constant", "optional_constant"}:
                        effective_mode = input_mode
                        if input_mode == "optional_constant":
                            use_constant = st.toggle(
                                f"Use constant {input_label.lower()}",
                                value=bool(input_spec.get("default_constant", False)),
                                key=f"mapping_use_constant_{selected_route['id']}_{input_feature}",
                            )
                            effective_mode = "constant" if use_constant else "upload"

                        if effective_mode == "upload":
                            upload_cols = st.columns([1.15, 0.85], gap="small")
                            with upload_cols[0]:
                                uploaded_file = st.file_uploader(
                                    f"{input_label} file",
                                    type=["tsv", "csv", "txt", "gz"],
                                    accept_multiple_files=False,
                                    key=f"mapping_upload_{selected_route['id']}_{input_feature}",
                                )

                            table, numeric_columns, read_error = read_numeric_table(
                                uploaded_file
                            )
                            if read_error:
                                st.error(
                                    f"Could not read {input_label.lower()}: {read_error}"
                                )
                            elif table is not None and not numeric_columns:
                                st.warning(
                                    f"No numeric chromosomes were found in {input_label.lower()}."
                                )

                            with upload_cols[1]:
                                selected_chromosome = st.selectbox(
                                    "Chromosome",
                                    numeric_columns,
                                    index=0 if numeric_columns else None,
                                    disabled=not numeric_columns,
                                    placeholder="Select chromosome",
                                    key=f"mapping_chromosome_{selected_route['id']}_{input_feature}",
                                )

                            values = None
                            if table is not None and selected_chromosome:
                                values = pd.to_numeric(
                                    table[selected_chromosome],
                                    errors="coerce",
                                ).to_numpy(dtype=float)

                            source_inputs[input_feature] = {
                                "mode": "upload",
                                "table": table,
                                "columns": numeric_columns,
                                "chromosome": selected_chromosome,
                                "values": values,
                            }

                            if primary_values is None and values is not None:
                                primary_table = table
                                primary_columns = numeric_columns
                                primary_values = values
                                primary_feature = input_feature
                                chromosome = selected_chromosome or input_label
                            continue

                        value = st.number_input(
                            input_label,
                            min_value=0.0,
                            value=float(input_spec.get("value", 1.0)),
                            step=float(input_spec.get("step", 0.1)),
                            format=str(input_spec.get("format", "%.6g")),
                            help=input_spec.get("help"),
                            key=f"mapping_constant_{selected_route['id']}_{input_feature}",
                        )
                        source_inputs[input_feature] = {
                            "mode": "constant",
                            "value": float(value),
                        }
                        continue

                    upload_cols = st.columns([1.15, 0.85], gap="small")
                    with upload_cols[0]:
                        uploaded_file = st.file_uploader(
                            f"{input_label} file",
                            type=["tsv", "csv", "txt", "gz"],
                            accept_multiple_files=False,
                            key=f"mapping_upload_{selected_route['id']}_{input_feature}",
                        )

                    table, numeric_columns, read_error = read_numeric_table(uploaded_file)
                    if read_error:
                        st.error(f"Could not read {input_label.lower()}: {read_error}")
                    elif table is not None and not numeric_columns:
                        st.warning(f"No numeric chromosomes were found in {input_label.lower()}.")

                    with upload_cols[1]:
                        selected_chromosome = st.selectbox(
                            "Chromosome",
                            numeric_columns,
                            index=0 if numeric_columns else None,
                            disabled=not numeric_columns,
                            placeholder="Select chromosome",
                            key=f"mapping_chromosome_{selected_route['id']}_{input_feature}",
                        )

                    values = None
                    if table is not None and selected_chromosome:
                        values = pd.to_numeric(
                            table[selected_chromosome],
                            errors="coerce",
                        ).to_numpy(dtype=float)

                    source_inputs[input_feature] = {
                        "mode": "upload",
                        "table": table,
                        "columns": numeric_columns,
                        "chromosome": selected_chromosome,
                        "values": values,
                    }

                    if primary_values is None and values is not None:
                        primary_table = table
                        primary_columns = numeric_columns
                        primary_values = values
                        primary_feature = input_feature
                        chromosome = selected_chromosome or input_label

        with st.expander("Region", expanded=False):
            region_top_cols = st.columns([1, 1], gap="small")
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
                    disabled=target_form not in {"time", "spacetime"},
                    help="Temporal resolution in min/bin for time-dependent targets.",
                )

            if primary_values is not None:
                finite_primary = np.isfinite(primary_values)
                primary_positions_full = (
                    np.arange(primary_values.size, dtype=float) * float(resolution)
                )
                primary_positions = primary_positions_full[finite_primary]
                max_position = max(0, int(np.ceil(float(np.nanmax(primary_positions)))))
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
                    disabled=primary_positions is None or not primary_positions.size,
                )
            with region_cols[1]:
                region_end = st.number_input(
                    "End (kb)",
                    min_value=min_position,
                    max_value=max_position,
                    value=max_position,
                    step=1,
                    format="%d",
                    disabled=primary_positions is None or not primary_positions.size,
                )

        with st.expander("Mapping settings", expanded=False):
            max_iterations = st.number_input(
                "Iterations",
                min_value=0,
                max_value=200,
                value=10,
                step=1,
                format="%d",
            )

            target_shape_label = st.selectbox(
                "Time model",
                list(TARGET_SHAPES),
                index=0,
                disabled=not (
                    target_feature == "firing_rate"
                    and target_form in {"time", "spacetime"}
                ),
            )

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

        with st.expander("Plotting settings", expanded=False):
            default_x_axis_title = (
                "S-phase time (min)"
                if target_form == "time"
                else "Genomic position (kb)"
            )
            default_y_axis_title = feature_label(target_feature)
            preview_mode = st.selectbox(
                "Show",
                PREVIEW_CONTENT_OPTIONS,
                index=0,
                help="Choose whether the preview shows the mapped target, the input profile, or both.",
            )
            text_cols = st.columns([1.25, 0.95], gap="small")
            with text_cols[0]:
                title_override = st.text_input(
                    "Title",
                    value="",
                    placeholder="Auto",
                )
            with text_cols[1]:
                legend_label = st.text_input(
                    "Label",
                    value="",
                    placeholder=default_y_axis_title,
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

            legend_cols = st.columns([0.84, 1.16], gap="small")
            with legend_cols[0]:
                show_legend = st.checkbox("Legend", value=True)
            with legend_cols[1]:
                legend_location_label = st.selectbox(
                    "Position",
                    list(LEGEND_LOCATIONS),
                    index=0,
                    disabled=not show_legend,
                )

            appearance_cols = st.columns(2, gap="small")
            with appearance_cols[0]:
                line_color = st.color_picker(
                    "Color",
                    value=feature_color(target_feature),
                )
            with appearance_cols[1]:
                invert_y_axis = st.checkbox("Invert y-axis", value=False)

            scale_cols = st.columns(2, gap="small")
            with scale_cols[0]:
                log_y_axis = st.checkbox("Log y-axis", value=False)
            with scale_cols[1]:
                use_y_limits = st.checkbox("Y limits", value=False)

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

            y_limit_min = 0.0
            y_limit_max = 0.0001
            if use_y_limits:
                y_limit_cols = st.columns(2, gap="small")
                with y_limit_cols[0]:
                    y_limit_min = st.number_input(
                        "Min",
                        value=0.0,
                        step=0.00001,
                        format="%.6g",
                    )
                with y_limit_cols[1]:
                    y_limit_max = st.number_input(
                        "Max",
                        value=0.0001,
                        step=0.00001,
                        format="%.6g",
                    )

        missing_mapping_inputs = []
        if selected_route is not None:
            for input_spec in selected_route["inputs"]:
                input_feature = str(input_spec["feature"])
                input_state = source_inputs.get(input_feature)
                if input_state is None:
                    missing_mapping_inputs.append(feature_label(input_feature))
                    continue
                if (
                    input_state.get("mode") == "upload"
                    and input_state.get("values") is None
                ):
                    missing_mapping_inputs.append(feature_label(input_feature))

        map_requested = st.button(
            "Map",
            type="primary",
            width="stretch",
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
            if selected_route is None:
                st.info("Choose a mapping route before starting.")
            elif missing_mapping_inputs:
                st.info(
                    "Add the required input data for "
                    + ", ".join(missing_mapping_inputs)
                    + " before mapping."
                )
            elif primary_values is None or primary_values.size == 0:
                st.info(
                    "Upload a numeric input profile and choose its chromosome before mapping."
                )
            elif region_end < region_start:
                st.error("Region end must be greater than or equal to region start.")
            elif use_y_limits and y_limit_max <= y_limit_min:
                st.error("The y-axis maximum must be greater than the minimum.")
            else:
                source_payload = {}
                map_positions = np.array([], dtype=float)
                reference_mask = None
                reference_timing = None

                for input_spec in selected_route["inputs"]:
                    input_feature = str(input_spec["feature"])
                    input_state = source_inputs[input_feature]

                    if input_state["mode"] == "constant":
                        source_payload[input_feature] = float(input_state["value"])
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

                    if input_feature == primary_feature:
                        reference_mask = region_mask
                        map_positions = raw_positions[region_mask]
                    elif (
                        reference_mask is not None
                        and raw_values.shape[0] == reference_mask.shape[0]
                    ):
                        region_mask = reference_mask & np.isfinite(raw_values)

                    selected_input = raw_values[region_mask]
                    source_payload[input_feature] = selected_input
                    if input_feature == "replication_timing":
                        reference_timing = selected_input

                if reference_mask is None and primary_values is not None:
                    raw_positions = np.arange(primary_values.size, dtype=float) * float(
                        resolution
                    )
                    reference_mask = (
                        np.isfinite(primary_values)
                        & (raw_positions >= float(region_start))
                        & (raw_positions <= float(region_end))
                    )
                    map_positions = raw_positions[reference_mask]

                empty_uploads = [
                    feature_label(feature)
                    for feature, values in source_payload.items()
                    if np.asarray(values).ndim > 0 and np.asarray(values).size == 0
                ]
                if empty_uploads:
                    st.warning("The selected region does not contain any bins.")
                else:
                    if active_mapping_job is not None:
                        active_mapping_job["cancel_event"].set()

                    mapping_assumptions = {
                        "resolution": resolution,
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
                    }
                    time_grid = np.array([], dtype=float)
                    if target_feature in {"firing_rate", "fork_speed"}:
                        mapping_assumptions["target_form"] = target_form
                    if target_feature == "fork_speed":
                        mapping_assumptions["target_shape"] = "best"
                    if target_form in {"time", "spacetime"}:
                        time_grid_source = (
                            reference_timing
                            if reference_timing is not None
                            else np.asarray(next(iter(source_payload.values())))
                        )
                        time_grid = build_time_grid(time_grid_source, resolution_time)
                        mapping_assumptions.update(
                            {
                                "resolution_time": resolution_time,
                                "time_grid": time_grid,
                            }
                        )
                    if (
                        target_feature == "firing_rate"
                        and target_form in {"time", "spacetime"}
                    ):
                        mapping_assumptions["target_shape"] = TARGET_SHAPES[
                            target_shape_label
                        ]

                    preview_input_values = np.array([], dtype=float)
                    if primary_feature is not None:
                        primary_payload = source_payload.get(primary_feature)
                        if primary_payload is not None and np.asarray(primary_payload).ndim > 0:
                            preview_input_values = np.asarray(
                                primary_payload,
                                dtype=float,
                            ).reshape(-1)

                    active_mapping_job = start_mapping_job(
                        source_payload,
                        target_feature,
                        map_positions,
                        mapping_assumptions,
                        {
                            "chromosome": chromosome,
                            "region_start": region_start,
                            "region_end": region_end,
                            "route": selected_route["label"],
                            "target_feature": target_feature,
                            "target_form": target_form,
                            "time_grid": time_grid,
                            "preview_mode": preview_mode,
                            "input_feature": primary_feature,
                            "input_values": preview_input_values,
                            "input_positions": np.asarray(map_positions, dtype=float),
                            "title": title_override.strip(),
                            "label": legend_label.strip(),
                            "x_axis_title": x_axis_title,
                            "y_axis_title": y_axis_title,
                            "figure_width": figure_width,
                            "figure_height": figure_height,
                            "use_y_limits": use_y_limits,
                            "y_limit_min": y_limit_min,
                            "y_limit_max": y_limit_max,
                            "invert_y_axis": invert_y_axis,
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
                            terminal_update["mapped_values"],
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
        elif primary_table is None:
            st.markdown(
                """
                <p class="small-muted">
                Select a route and upload the required input profile to preview a map.
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
            last_map_for_download = st.session_state.get(MAP_STATE_KEY)
            if last_map_for_download is not None and "raw" in last_map_for_download:
                st.markdown(
                    """
                    <p class="small-muted">
                    Download the mapped output with genomic positions.
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
                        key="mapping_raw_format",
                        label_visibility="collapsed",
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
