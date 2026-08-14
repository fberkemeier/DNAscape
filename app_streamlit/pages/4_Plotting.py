"""Explicit, bounded plotting workflow for canonical DNAscape data."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import streamlit as st

from app_streamlit.app_shell import render_control_panel_marker, render_page_intro, safe_name_stem
from app_streamlit.exports import figure_export_target, serialize_figure
from app_streamlit.plot_service import (
    heatmap_figure,
    histogram_figure,
    line_figure,
    release_figure,
)
from app_streamlit.preview import preview_indices, preview_matrix, preview_series
from app_streamlit.preview_ui import render_image_preview
from app_streamlit.state import persistent_widget_key, workflow_input_signature
from app_streamlit.workflow_ui import (
    DatasetView,
    available_data_views,
    render_profile_view,
    view_axis,
    view_series,
)

PAGE_NAMESPACE = "plotting"
RENDERED_PLOT_KEY = "plotting_rendered_preview"
_Option = TypeVar("_Option")


def _clear_plot_output() -> None:
    """Remove the retained preview."""

    st.session_state.pop(RENDERED_PLOT_KEY, None)


def _stable_selectbox(
    label: str,
    options: Sequence[_Option],
    name: str,
    *,
    default: _Option | None = None,
    format_func: Callable[[_Option], str] | None = None,
    help_text: str | None = None,
) -> _Option:
    options = list(options)
    if not options:
        raise ValueError(f"{label} requires at least one option.")
    key = persistent_widget_key(PAGE_NAMESPACE, name)
    if key in st.session_state and st.session_state[key] not in options:
        st.session_state.pop(key, None)
    if key not in st.session_state and options:
        st.session_state[key] = default if default in options else options[0]
    selected = st.selectbox(
        label,
        options,
        key=key,
        format_func=format_func or str,
        help=help_text,
    )
    if selected is None:  # Defensive guard for future optional-index callers.
        raise ValueError(f"{label} requires a selection.")
    return selected


def _stable_number(
    label: str,
    name: str,
    default: int | float,
    *,
    min_value: int | float,
    max_value: int | float,
    step: int | float,
    help_text: str | None = None,
):
    key = persistent_widget_key(PAGE_NAMESPACE, name)
    current = st.session_state.get(key, default)
    integer_input = isinstance(default, int) and not isinstance(default, bool)
    try:
        numeric = float(current)
    except (TypeError, ValueError, OverflowError):
        numeric = float(default)
    if not np.isfinite(numeric):
        numeric = float(default)
    numeric = min(max(numeric, float(min_value)), float(max_value))
    if integer_input:
        numeric = int(numeric) if numeric.is_integer() else int(default)
    st.session_state[key] = int(numeric) if integer_input else float(numeric)
    return st.number_input(
        label,
        min_value=min_value,
        max_value=max_value,
        step=step,
        key=key,
        help=help_text,
    )


def _stable_checkbox(
    label: str,
    name: str,
    default: bool,
    *,
    help_text: str | None = None,
) -> bool:
    key = persistent_widget_key(PAGE_NAMESPACE, name)
    if not isinstance(st.session_state.get(key), bool):
        st.session_state[key] = bool(default)
    return bool(st.checkbox(label, key=key, help=help_text))


def _stable_color(label: str, name: str, default: str) -> str:
    key = persistent_widget_key(PAGE_NAMESPACE, name)
    current = st.session_state.get(key)
    valid = (
        isinstance(current, str)
        and len(current) == 7
        and current.startswith("#")
        and all(character in "0123456789abcdefABCDEF" for character in current[1:])
    )
    if not valid:
        st.session_state[key] = default
    return str(st.color_picker(label, key=key))


def _series_labels(view: DatasetView) -> tuple[str, ...]:
    if view.values.ndim != 2 or view.data_form == "time":
        return ()
    if len(view.labels) == view.values.shape[1]:
        return view.labels
    return tuple(f"value_{index + 1}" for index in range(view.values.shape[1]))


def _optional_limits(text: Any, label: str) -> tuple[float, float] | None:
    text = str(text or "").strip()
    if not text:
        return None
    try:
        values = [float(value.strip()) for value in text.split(",")]
    except ValueError as exc:
        raise ValueError(f"{label} limits must contain two comma-separated numbers.") from exc
    if len(values) != 2 or not np.all(np.isfinite(values)) or values[1] <= values[0]:
        raise ValueError(f"{label} limits must be finite and satisfy minimum < maximum.")
    return values[0], values[1]


def _dimension_axis(
    view: DatasetView,
    dimension: int,
    *,
    spatial_step_kb: float,
    time_step_min: float,
) -> np.ndarray:
    size = view.values.shape[dimension]
    names = (
        ("position_kb", 1.0),
        ("position_bp", 0.001),
        ("start_bp", 0.001),
    ) if dimension == 0 else (
        ("time_min", 1.0),
        ("time_grid", 1.0),
    )
    for name, scale in names:
        axis = view.axes.get(name)
        if axis is not None:
            axis_array = np.asarray(axis)
            if axis_array.ndim == 1 and axis_array.size == size:
                try:
                    return np.asarray(axis_array, dtype=float) * scale
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Coordinate axis {name!r} must be numeric.") from exc
    step = spatial_step_kb if dimension == 0 else time_step_min
    return np.arange(size, dtype=float) * float(step)


def _presentation_transform(values: np.ndarray, mode: str) -> np.ndarray:
    """Normalize only a selected plot slice, never its canonical source."""

    if mode not in {"None", "Z-score", "Min-max [0,1]"}:
        raise ValueError(f"Unsupported presentation normalization: {mode!r}.")
    if np.asarray(values).dtype.kind not in "biuf":
        raise ValueError("Plot data must contain real numeric values.")
    if mode == "None":
        return values
    array = np.asarray(values, dtype=float)
    finite_mask = np.isfinite(array)
    if not np.any(finite_mask):
        raise ValueError("Presentation normalization requires finite values.")
    finite = array[finite_mask]
    if mode == "Z-score":
        center = float(np.mean(finite))
        scale = float(np.std(finite))
    else:
        center = float(np.min(finite))
        scale = float(np.max(finite) - center)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Presentation normalization is undefined for constant data.")
    return (array - center) / scale


def _view_by_id(view_id: str) -> DatasetView | None:
    if not view_id:
        return None
    return next(
        (view for view in available_data_views() if view.view_id == view_id),
        None,
    )


def _build_figure(view: DatasetView, config: dict[str, Any], *, full_resolution: bool):
    plot_type = str(config.get("plot_type", ""))
    if plot_type not in {"Line", "Heatmap", "Histogram"}:
        raise ValueError(f"Unsupported plot type: {plot_type or 'missing'}.")
    try:
        row_start, row_end = int(config["row_start"]), int(config["row_end"])
        col_start = int(config.get("column_start", 0))
        col_end = int(config.get("column_end", 1))
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Plot region indices must be integers.") from exc
    if min(row_start, col_start) < 0:
        raise ValueError("Plot region indices cannot be negative.")
    if row_end <= row_start or col_end <= col_start:
        raise ValueError("Plot regions use half-open ranges and must contain at least one value.")

    if plot_type == "Line":
        series = view_series(view, config.get("series"))
        if series.size == 0:
            raise ValueError("Line plots require a non-empty source series.")
        axis = view_axis(view, series.size, resolution_kb=config["spatial_step_kb"])
        start = min(row_start, series.size)
        end = min(max(start + 1, row_end), series.size)
        if start >= end:
            raise ValueError("The selected line region falls outside the source series.")
        x_plot, y_plot = axis[start:end], series[start:end]
        if not full_resolution:
            x_plot, y_plot = preview_series(x_plot, y_plot, max_points=5_000)
        y_plot = _presentation_transform(y_plot, config["normalization"])
        comparison = _view_by_id(config.get("comparison_view_id", ""))
        comparison_values: tuple[np.ndarray, np.ndarray] | None = None
        if comparison is not None:
            requested_series = config.get("series")
            comparison_series = (
                requested_series
                if requested_series in _series_labels(comparison)
                else None
            )
            other = view_series(comparison, comparison_series)
            other_axis = view_axis(
                comparison,
                other.size,
                resolution_kb=config["spatial_step_kb"],
            )
            if other.size != series.size or not np.allclose(
                other_axis,
                axis,
                equal_nan=False,
            ):
                raise ValueError(
                    "Comparison lines must share the same length and physical coordinate axis. "
                    "Align them on Data Management before plotting."
                )
            other_x, other_y = other_axis[start:end], other[start:end]
            if not full_resolution:
                other_x, other_y = preview_series(other_x, other_y, max_points=5_000)
            comparison_values = (
                other_x,
                _presentation_transform(other_y, config["normalization"]),
            )
        figure, axes = line_figure(
            x_plot,
            y_plot,
            title=config["title"],
            x_label=config["x_label"],
            y_label=config["y_label"],
            color=config["color"],
            line_width=config["line_width"],
            show_grid=config["show_grid"],
            x_limits=config["x_limits"],
            y_limits=config["y_limits"],
            log_y=config["log_y"],
            invert_y=config["invert_y"],
            label=view.name if comparison is not None else None,
            full_resolution=True,
        )
        if comparison is not None and comparison_values is not None:
            try:
                axes.plot(
                    *comparison_values,
                    color=config["comparison_color"],
                    linewidth=config["line_width"],
                    alpha=0.9,
                    label=comparison.name,
                )
                axes.legend(loc="best")
            except Exception:
                release_figure(figure)
                raise
        return figure, axes

    if plot_type == "Heatmap":
        matrix = view.values
        if matrix.ndim != 2:
            raise ValueError("Heatmaps require a two-dimensional source.")
        row_end = min(row_end, matrix.shape[0])
        col_end = min(col_end, matrix.shape[1])
        selected = matrix[row_start:row_end, col_start:col_end]
        y_axis = _dimension_axis(
            view,
            0,
            spatial_step_kb=config["spatial_step_kb"],
            time_step_min=config["time_step_min"],
        )[row_start:row_end]
        x_axis = _dimension_axis(
            view,
            1,
            spatial_step_kb=config["spatial_step_kb"],
            time_step_min=config["time_step_min"],
        )[col_start:col_end]
        if not full_resolution:
            selected, row_indices, column_indices = preview_matrix(
                selected,
                max_rows=900,
                max_columns=420,
            )
            y_axis = y_axis[row_indices]
            x_axis = x_axis[column_indices]
        selected = _presentation_transform(selected, config["normalization"])
        return heatmap_figure(
            selected,
            x_values=x_axis,
            y_values=y_axis,
            title=config["title"],
            x_label=config["x_label"],
            y_label=config["y_label"],
            colorbar_label=config["colorbar_label"],
            cmap=config["cmap"],
            vmin=config["vmin"],
            vmax=config["vmax"],
            show_colorbar=config["show_colorbar"],
            invert_y=config["invert_y"],
            full_resolution=True,
        )

    values = view.values
    if values.ndim == 2:
        values = values[
            row_start : min(row_end, values.shape[0]),
            col_start : min(col_end, values.shape[1]),
        ]
    else:
        values = values.reshape(-1)[row_start:min(row_end, values.size)]
    values = np.asarray(values).reshape(-1)
    if not full_resolution:
        values = values[preview_indices(values.size, 250_000)]
    values = _presentation_transform(values, config["normalization"])
    return histogram_figure(
        values,
        bins=config["bins"],
        title=config["title"],
        x_label=config["x_label"],
        y_label=config["y_label"],
        color=config["color"],
        full_resolution=True,
    )


def _render_plot_controls(view: DatasetView) -> dict[str, Any] | None:
    values = view.values
    matrix = values.ndim == 2 and min(values.shape) > 1
    plot_types = ["Line", "Histogram"]
    if matrix:
        plot_types.insert(1, "Heatmap")
    default_type = "Heatmap" if view.data_form == "spacetime" and matrix else "Line"

    plot_type_key = persistent_widget_key(PAGE_NAMESPACE, "plot_type")
    if st.session_state.get(plot_type_key) not in plot_types:
        st.session_state[plot_type_key] = default_type
    plot_type = st.session_state[plot_type_key]

    row_count = values.shape[0] if values.ndim else 1
    column_count = values.shape[1] if values.ndim == 2 else 1
    if view.data_form == "time":
        row_count = values.size
        column_count = 1

    with st.expander("Region", expanded=False):
        region_cols = st.columns(2, gap="small")
        with region_cols[0]:
            row_start = int(
                _stable_number(
                    "Start index (inclusive)",
                    "row_start",
                    0,
                    min_value=0,
                    max_value=max(0, row_count - 1),
                    step=1,
                )
            )
        with region_cols[1]:
            row_end = int(
                _stable_number(
                    "End index (exclusive)",
                    "row_end",
                    row_count,
                    min_value=1,
                    max_value=max(1, row_count),
                    step=1,
                )
            )
        column_start, column_end = 0, column_count
        if matrix and plot_type in {"Heatmap", "Histogram"}:
            column_cols = st.columns(2, gap="small")
            with column_cols[0]:
                column_start = int(
                    _stable_number(
                        "First time/feature index",
                        "column_start",
                        0,
                        min_value=0,
                        max_value=max(0, column_count - 1),
                        step=1,
                    )
                )
            with column_cols[1]:
                column_end = int(
                    _stable_number(
                        "Last time/feature index (exclusive)",
                        "column_end",
                        column_count,
                        min_value=1,
                        max_value=max(1, column_count),
                        step=1,
                    )
                )
        resolution_cols = st.columns(2, gap="small")
        with resolution_cols[0]:
            spatial_step_kb = float(
                _stable_number(
                    "Resolution (kb/bin)",
                    "spatial_step",
                    1.0,
                    min_value=1e-9,
                    max_value=1e9,
                    step=0.1,
                )
            )
        with resolution_cols[1]:
            time_step_min = float(
                _stable_number(
                    "Time step (min)",
                    "time_step",
                    1.0,
                    min_value=1e-9,
                    max_value=1e9,
                    step=0.1,
                )
            )

    with st.expander("Plotting settings", expanded=False):
        plot_type = _stable_selectbox(
            "Plot type",
            plot_types,
            "plot_type",
            default=default_type,
        )
        series = None
        labels = _series_labels(view)
        if plot_type == "Line" and labels:
            series = _stable_selectbox("Feature / series", labels, "series", default=labels[0])

        comparison_view_id = ""
        comparison_color = "#d97757"
        if plot_type == "Line":
            comparison_views = [
                candidate
                for candidate in available_data_views()
                if candidate.view_id != view.view_id
            ]
            comparison_ids = [""] + [candidate.view_id for candidate in comparison_views]
            comparison_by_id = {candidate.view_id: candidate for candidate in comparison_views}
            comparison_view_id = _stable_selectbox(
                "Optional comparison line",
                comparison_ids,
                "comparison_view",
                default="",
                format_func=lambda value: (
                    "None" if not value else comparison_by_id[value].name
                ),
                help_text="Comparison is accepted only when physical coordinates align exactly.",
            )
            if comparison_view_id:
                comparison_color = _stable_color(
                    "Comparison color",
                    "comparison_color",
                    "#d97757",
                )

        title_key = persistent_widget_key(PAGE_NAMESPACE, "title")
        if title_key not in st.session_state:
            st.session_state[title_key] = Path(view.name).stem
        title = st.text_input("Title", key=title_key)
        label_cols = st.columns(2)
        x_key = persistent_widget_key(PAGE_NAMESPACE, "x_label")
        y_key = persistent_widget_key(PAGE_NAMESPACE, "y_label")
        if x_key not in st.session_state:
            st.session_state[x_key] = "Time (min)" if view.data_form == "time" else "Position (kb)"
        if y_key not in st.session_state:
            st.session_state[y_key] = "Value"
        with label_cols[0]:
            x_label = st.text_input("X-axis label", key=x_key)
        with label_cols[1]:
            y_label = st.text_input("Y-axis label", key=y_key)

        normalization = _stable_selectbox(
            "Presentation normalization",
            ("None", "Z-score", "Min-max [0,1]"),
            "normalization",
            default="None",
            help_text=(
                "Applies to this plot only; the source data remain unchanged."
            ),
        )
        color = _stable_color("Line / bar color", "color", "#2c7f77")
        cmap = _stable_selectbox(
            "Heatmap palette",
            ("viridis", "magma", "plasma", "cividis", "coolwarm"),
            "cmap",
            default="viridis",
        )
        style_cols = st.columns(2, gap="small")
        with style_cols[0]:
            line_width = float(
                _stable_number(
                    "Line width",
                    "line_width",
                    1.8,
                    min_value=0.2,
                    max_value=8.0,
                    step=0.1,
                )
            )
        with style_cols[1]:
            bins = int(
                _stable_number(
                    "Histogram bins",
                    "histogram_bins",
                    60,
                    min_value=1,
                    max_value=2_000,
                    step=1,
                )
            )
        option_cols = st.columns(4)
        with option_cols[0]:
            show_grid = _stable_checkbox("Grid", "show_grid", True)
        with option_cols[1]:
            show_colorbar = _stable_checkbox("Colorbar", "colorbar", True)
        with option_cols[2]:
            log_y = _stable_checkbox("Log Y", "log_y", False)
        with option_cols[3]:
            invert_y = _stable_checkbox("Invert Y", "invert_y", False)
        limit_cols = st.columns(2, gap="small")
        with limit_cols[0]:
            x_limits_text = st.text_input(
                "X limits (min,max; optional)",
                key=persistent_widget_key(PAGE_NAMESPACE, "x_limits"),
            )
        with limit_cols[1]:
            y_limits_text = st.text_input(
                "Y limits (min,max; optional)",
                key=persistent_widget_key(PAGE_NAMESPACE, "y_limits"),
            )
        colorbar_key = persistent_widget_key(PAGE_NAMESPACE, "colorbar_label")
        if colorbar_key not in st.session_state:
            st.session_state[colorbar_key] = "Value"
        colorbar_label = st.text_input("Colorbar label", key=colorbar_key)
        value_limit_cols = st.columns(2, gap="small")
        with value_limit_cols[0]:
            vmin_text = st.text_input(
                "Heatmap minimum (optional)",
                key=persistent_widget_key(PAGE_NAMESPACE, "vmin"),
            )
        with value_limit_cols[1]:
            vmax_text = st.text_input(
                "Heatmap maximum (optional)",
                key=persistent_widget_key(PAGE_NAMESPACE, "vmax"),
            )

    try:
        x_limits = _optional_limits(x_limits_text, "X") if plot_type == "Line" else None
        y_limits = _optional_limits(y_limits_text, "Y") if plot_type == "Line" else None
        vmin = (
            float(vmin_text)
            if plot_type == "Heatmap" and str(vmin_text).strip()
            else None
        )
        vmax = (
            float(vmax_text)
            if plot_type == "Heatmap" and str(vmax_text).strip()
            else None
        )
        if vmin is not None and not np.isfinite(vmin):
            raise ValueError("Heatmap minimum must be finite.")
        if vmax is not None and not np.isfinite(vmax):
            raise ValueError("Heatmap maximum must be finite.")
        if vmin is not None and vmax is not None and vmax <= vmin:
            raise ValueError("Heatmap maximum must be greater than its minimum.")
        if row_end <= row_start or column_end <= column_start:
            raise ValueError(
                "Plot regions use half-open ranges and must contain at least one value."
            )
        if log_y and y_limits is not None and y_limits[0] <= 0:
            raise ValueError("Y limits must be positive when using a log scale.")
    except ValueError as exc:
        st.error(str(exc))
        return None

    return {
        "plot_type": plot_type,
        "series": series,
        "comparison_view_id": comparison_view_id,
        "comparison_color": comparison_color,
        "normalization": normalization,
        "row_start": row_start,
        "row_end": row_end,
        "column_start": column_start,
        "column_end": column_end,
        "title": title,
        "x_label": x_label,
        "y_label": y_label,
        "colorbar_label": colorbar_label,
        "spatial_step_kb": spatial_step_kb,
        "time_step_min": time_step_min,
        "color": color,
        "cmap": cmap,
        "line_width": line_width,
        "bins": bins,
        "show_grid": show_grid,
        "show_colorbar": show_colorbar,
        "log_y": log_y,
        "invert_y": invert_y,
        "x_limits": x_limits,
        "y_limits": y_limits,
        "vmin": vmin,
        "vmax": vmax,
    }


def _plot_figure_download_bytes(
    view: DatasetView,
    config: dict[str, Any],
    export_format: str,
    export_dpi: int,
    full_resolution: bool,
) -> bytes:
    """Build the requested plot export only when Download is clicked."""

    figure = None
    try:
        figure, _ = _build_figure(
            view,
            dict(config),
            full_resolution=full_resolution,
        )
        return serialize_figure(
            figure,
            export_format,
            dpi=export_dpi,
        ).payload
    finally:
        if figure is not None:
            release_figure(figure)


def _render_preview_and_export(
    view: DatasetView,
    config: dict[str, Any],
    render_clicked: bool,
) -> None:
    signature = workflow_input_signature(view.view_id, config)
    if render_clicked:
        figure = None
        try:
            with st.spinner("Building plot preview..."):
                figure, _ = _build_figure(view, config, full_resolution=False)
                preview_payload = serialize_figure(figure, "png", dpi=110).payload
            st.session_state[RENDERED_PLOT_KEY] = {
                "signature": signature,
                "view_id": view.view_id,
                "config": dict(config),
                "payload": preview_payload,
                "file_name": (
                    f"{safe_name_stem('dnascape', Path(view.name).stem, config['plot_type'])}.png"
                ),
            }
        except (
            IndexError,
            KeyError,
            MemoryError,
            OSError,
            OverflowError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            st.error(f"Could not build the plot preview: {exc}")
        finally:
            if figure is not None:
                release_figure(figure)

    rendered = st.session_state.get(RENDERED_PLOT_KEY)
    if not isinstance(rendered, Mapping) or rendered.get("view_id") != view.view_id:
        st.caption("Choose the settings, then click Plot.")
        return
    payload = rendered.get("payload")
    stored_config = rendered.get("config")
    if not isinstance(payload, (bytes, bytearray, memoryview)) or not isinstance(
        stored_config,
        Mapping,
    ):
        st.session_state.pop(RENDERED_PLOT_KEY, None)
        st.warning("The retained preview was invalid and has been cleared. Select Plot again.")
        return

    if rendered.get("signature") != signature:
        st.caption("Settings changed - select Update plot.")

    render_image_preview(
        bytes(payload),
        file_name=rendered.get("file_name", "dnascape_plot.png"),
        shell_key="plot-preview-shell",
        clear_key="plot-preview-clear",
        on_clear=_clear_plot_output,
        alt="Plot preview",
        downloads={},
    )

    try:
        selected_rows = max(
            0,
            min(int(stored_config["row_end"]), int(view.values.shape[0]))
            - int(stored_config["row_start"]),
        )
        stored_plot_type = str(stored_config["plot_type"])
    except (KeyError, TypeError, ValueError, OverflowError):
        st.session_state.pop(RENDERED_PLOT_KEY, None)
        st.warning("The retained plot settings were invalid and have been cleared.")
        return
    selected_values = selected_rows
    if view.values.ndim == 2 and stored_plot_type != "Line":
        try:
            selected_columns = max(
                0,
                min(int(stored_config["column_end"]), int(view.values.shape[1]))
                - int(stored_config["column_start"]),
            )
        except (KeyError, TypeError, ValueError, OverflowError):
            selected_columns = 0
        selected_values *= selected_columns
    st.caption(
        f"{stored_plot_type} | {selected_values:,} selected values | {view.name}"
    )

    with st.expander("Export plot", expanded=False):
        export_columns = st.columns([0.2, 0.2, 0.26, 0.34], gap="small")
        with export_columns[0]:
            export_format = _stable_selectbox(
                "Format",
                ("png", "svg", "pdf"),
                "export_format",
                default="png",
                format_func=str.upper,
            )
        with export_columns[1]:
            export_dpi = int(
                _stable_number(
                    "DPI",
                    "export_dpi",
                    180,
                    min_value=72,
                    max_value=600,
                    step=10,
                )
            )
        with export_columns[2]:
            full_resolution = _stable_checkbox(
                "Full resolution",
                "export_full_resolution",
                False,
                help_text="Off keeps the export bounded like the interactive preview.",
            )
        extension, mime = figure_export_target(export_format)
        file_stem = safe_name_stem(
            "dnascape",
            Path(view.name).stem,
            stored_plot_type,
        )
        export_columns[3].download_button(
            "Download",
            data=partial(
                _plot_figure_download_bytes,
                view,
                dict(stored_config),
                export_format,
                export_dpi,
                bool(full_resolution),
            ),
            file_name=f"{file_stem}.{extension}",
            mime=mime,
            on_click="ignore",
            width="stretch",
            help="The selected format is generated only when Download is clicked.",
        )


@st.fragment
def render_plotting_workspace() -> None:
    controls, preview = st.columns((0.35, 0.65), gap="medium")
    view: DatasetView | None = None
    config: dict[str, Any] | None = None

    with controls:
        control_stack = st.container(key="dnascape-control-stack")

    with control_stack:
        with st.container(
            border=True,
            height="stretch",
            key="dnascape-control-card",
        ):
            with st.expander("Data", expanded=True):
                view = render_profile_view(PAGE_NAMESPACE, allow_upload=True)
                if view is not None:
                    st.caption(
                        f"{view.name} - {' x '.join(map(str, view.shape))} - "
                        f"{view.data_form}."
                    )
            if view is None:
                st.caption("Select or upload data to configure a plot.")
            else:
                config = _render_plot_controls(view)

    with control_stack:
        with st.container(key="dnascape-control-action"):
            retained = st.session_state.get(RENDERED_PLOT_KEY)
            has_preview = (
                isinstance(retained, dict)
                and view is not None
                and retained.get("view_id") == view.view_id
            )
            render_clicked = st.button(
                "Update plot" if has_preview else "Plot",
                type="primary",
                key="plotting_run_button",
                disabled=view is None or config is None,
                width="stretch",
            )

    with preview:
        with st.container(border=True, key="dnascape-preview-card"):
            st.markdown(
                '<div class="plotting-panel-title">Preview</div>',
                unsafe_allow_html=True,
            )
            if view is None:
                st.markdown(
                    '<p class="small-muted">Select or upload data to begin.</p>',
                    unsafe_allow_html=True,
                )
            elif config is None:
                st.caption("Check the plot settings.")
            else:
                _render_preview_and_export(view, config, render_clicked)


def render_page() -> None:
    """Render Plotting inside the persistent DNAscape shell."""

    render_page_intro(
        "Plotting",
        "Visualize DNAscape outputs",
        "Upload kinetic profiles, choose a genomic window, and preview DNAscape outputs.",
    )
    render_control_panel_marker()
    render_plotting_workspace()
