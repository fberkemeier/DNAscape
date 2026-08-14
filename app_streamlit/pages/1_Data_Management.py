"""Staged, content-addressed data preparation for DNAscape workflows."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Callable, Sequence, TypeVar

import numpy as np
import streamlit as st

from app_streamlit.app_shell import (
    render_control_panel_marker,
    render_page_intro,
    safe_name_stem,
)
from app_streamlit.data_service import (
    DatasetRecord,
    PreparationConfig,
    SourceInspection,
    dataset_display_name,
    prepared_dataset_id,
)
from app_streamlit.exports import (
    array_export_target,
    gzip_payload,
    serialize_array,
    serialize_figure,
)
from app_streamlit.plot_service import heatmap_figure, line_figure, release_figure
from app_streamlit.preview import memory_label, representative_preview
from app_streamlit.state import (
    get_dataset,
    get_source_inspection,
    initialized_widget_key,
    persistent_widget_key,
)
from app_streamlit.workflow_ui import (
    dataset_record_view,
    get_or_prepare_upload_dataset,
    inspect_retained_upload,
    render_upload_source_selector,
    summarize_view,
    view_axis,
    view_series,
)

PAGE_NAMESPACE = "data_management"
ACTIVE_DATASET_KEY = "data_management_active_dataset_id"
PREVIEW_IMAGE_KEY = "data_management_preview_image"
EXPORT_KEY = "data_management_prepared_export"
UPLOAD_TYPES = (
    "csv",
    "tsv",
    "txt",
    "npy",
    "npv",
    "npz",
    "gz",
    "bw",
    "bigwig",
    "bedgraph",
    "bdg",
)
DATA_FORMS = {
    "Auto-detect": "auto",
    "Table": "table",
    "Space profile": "space",
    "Time profile": "time",
    "Space-time matrix": "spacetime",
}
_Option = TypeVar("_Option")


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
    try:
        current_is_valid = key not in st.session_state or st.session_state[key] in options
    except (TypeError, ValueError):
        current_is_valid = False
    if not current_is_valid:
        st.session_state.pop(key, None)
    if key not in st.session_state:
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
    min_value: int | float | None = None,
    max_value: int | float | None = None,
    step: int | float = 1,
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
    if min_value is not None:
        numeric = max(numeric, float(min_value))
    if max_value is not None:
        numeric = min(numeric, float(max_value))
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


def _render_source_panel():
    st.markdown('<div class="dnascape-panel-title">Input</div>', unsafe_allow_html=True)
    upload = render_upload_source_selector(
        PAGE_NAMESPACE,
        label="Upload",
        allowed_types=UPLOAD_TYPES,
        help_text=(
            "CSV, TSV, TXT, NumPy, BigWig, bedGraph, and supported gzip variants."
        ),
    )
    if upload is None:
        st.caption("Upload a file to continue.")
        return None, None

    st.caption(f"{upload.name} - {memory_label(upload.size_bytes)}")
    inspection = get_source_inspection(upload.digest)
    if not isinstance(inspection, SourceInspection):
        try:
            with st.spinner("Reading source metadata..."):
                inspection = inspect_retained_upload(upload)
        except (
            ValueError,
            OSError,
            EOFError,
            UnicodeError,
            ModuleNotFoundError,
            MemoryError,
        ) as exc:
            st.error(str(exc))
        if not isinstance(inspection, SourceInspection):
            return upload, None

    details = [inspection.source_format.upper()]
    if inspection.compressed:
        details.append("gzip")
    if inspection.shape:
        details.append(" x ".join(str(axis) for axis in inspection.shape))
    if inspection.dtype:
        details.append(inspection.dtype)
    if inspection.chromosomes:
        details.append(f"{len(inspection.chromosomes)} chromosome(s)")
    if inspection.headerless:
        details.append("headerless numeric text")
    st.success(" - ".join(details))
    if inspection.columns:
        shown = ", ".join(inspection.columns[:8])
        suffix = " ..." if len(inspection.columns) > 8 else ""
        st.caption(f"Columns: {shown}{suffix}")
    if inspection.array_keys:
        st.caption(f"NPZ arrays: {', '.join(inspection.array_keys)}")
    return upload, inspection


def _render_preparation_controls(inspection: SourceInspection) -> PreparationConfig:
    # Preparation remains explicitly executed by the action button in the
    # original left-hand control stack.  These widgets only build a lightweight
    # configuration and never load or transform the source on their own.
    with st.container():
        if inspection.is_track:
            chromosome = _stable_selectbox(
                "Chromosome",
                inspection.chromosomes,
                "track_chromosome",
                help_text="Only this chromosome and interval are read and binned.",
            )
            lower, upper = inspection.chromosome_bounds[chromosome]
            interval_namespace = f"{inspection.digest[:10]}_{chromosome}"
            bounds_cols = st.columns(2)
            with bounds_cols[0]:
                start_bp = int(
                    _stable_number(
                        "Start (bp, inclusive)",
                        f"{interval_namespace}_start",
                        int(lower),
                        min_value=0,
                        max_value=max(0, int(upper) - 1),
                        step=1,
                    )
                )
            with bounds_cols[1]:
                end_bp = int(
                    _stable_number(
                        "End (bp, exclusive)",
                        f"{interval_namespace}_end",
                        int(upper),
                        min_value=1,
                        max_value=int(upper),
                        step=1,
                    )
                )
            bin_size = int(
                _stable_number(
                    "Bin size (bp)",
                    "track_bin_size",
                    1_000,
                    min_value=1,
                    step=100,
                    help_text="Genomic resolution; bins follow the selected half-open interval.",
                )
            )
            aggregation = _stable_selectbox(
                "Aggregation",
                ("mean", "sum", "min", "max", "coverage", "std"),
                "track_aggregation",
                default="mean",
            )
            with st.expander("Advanced preprocessing"):
                missing = _stable_selectbox(
                    "Missing values",
                    ("nan", "zero", "fill", "drop"),
                    "track_missing",
                    default="nan",
                )
                fill_value = float(
                    _stable_number(
                        "Custom fill value",
                        "track_fill_value",
                        0.0,
                        step=0.1,
                        help_text="Used only when the missing-value policy is 'fill'.",
                    )
                )
                interpolation = _stable_selectbox(
                    "Interpolation",
                    ("none", "linear", "ffill", "bfill"),
                    "track_interpolation",
                    default="none",
                )
                smooth_window = int(
                    _stable_number(
                        "Centered smoothing window (bins)",
                        "track_smooth_window",
                        1,
                        min_value=1,
                        step=1,
                    )
                )
                normalize = _stable_selectbox(
                    "Normalization",
                    ("none", "center", "zscore", "minmax", "maxabs"),
                    "track_normalization",
                    default="none",
                )
            estimated_bins = max(0, int(np.ceil(max(0, end_bp - start_bp) / bin_size)))
            st.caption(
                f"Estimated output: {estimated_bins:,} bins - "
                f"approximately {memory_label(estimated_bins * 8)} for one float64 profile."
            )
            config = PreparationConfig(
                data_form="space",
                chromosome=chromosome,
                start_bp=start_bp,
                end_bp=end_bp,
                bin_size_bp=bin_size,
                aggregation=aggregation,
                missing=missing,
                fill_value=fill_value,
                interpolation=interpolation,
                smooth_window=smooth_window,
                normalize=normalize,
            )
        else:
            data_form_label = _stable_selectbox(
                "DNAscape data form",
                tuple(DATA_FORMS),
                "plain_data_form",
                default="Auto-detect",
                help_text="Space is (n,), time is (1, T), and space-time is (n, T).",
            )
            value_column = None
            if inspection.columns:
                column_options = [""] + list(inspection.columns)
                value_column = _stable_selectbox(
                    "Value column",
                    column_options,
                    "plain_value_column",
                    default="",
                    format_func=lambda value: "All numeric columns" if not value else value,
                    help_text="Choose one feature/profile, or retain all numeric columns.",
                ) or None
            array_key = None
            if inspection.array_keys:
                array_key = _stable_selectbox(
                    "NPZ array",
                    inspection.array_keys,
                    "plain_array_key",
                    default=inspection.array_keys[0],
                )
            value_name_key = persistent_widget_key(PAGE_NAMESPACE, "value_name")
            if not isinstance(st.session_state.get(value_name_key), str):
                st.session_state[value_name_key] = "value"
            value_name = st.text_input(
                "Output feature name",
                key=value_name_key,
                help="Used when the source does not already provide a column label.",
            ).strip() or "value"
            config = PreparationConfig(
                data_form=DATA_FORMS[data_form_label],
                value_column=value_column,
                array_key=array_key,
                value_name=value_name,
            )

    return config


def _preview_image(record: DatasetRecord) -> bytes:
    cached = st.session_state.get(PREVIEW_IMAGE_KEY)
    if isinstance(cached, dict) and cached.get("dataset_id") == record.dataset_id:
        payload = cached.get("payload")
        if isinstance(payload, bytes):
            return payload

    view = dataset_record_view(record)
    if record.values.ndim == 1 or record.data_form == "time" or (
        record.values.ndim == 2 and 1 in record.values.shape
    ):
        series = view_series(view)
        axis = view_axis(view, series.size)
        x_label = "Time" if record.data_form == "time" else "Position (kb)"
        figure, _ = line_figure(
            axis,
            series,
            title=record.name,
            x_label=x_label,
            y_label=record.labels[0] if record.labels else "Value",
        )
    else:
        y_axis = view_axis(view, record.values.shape[0])
        figure, _ = heatmap_figure(
            record.values,
            y_values=y_axis,
            title=record.name,
            x_label="Time / feature index",
            y_label="Position (kb)",
        )
    try:
        payload = serialize_figure(figure, "png", dpi=135).payload
    finally:
        release_figure(figure)
    st.session_state[PREVIEW_IMAGE_KEY] = {
        "dataset_id": record.dataset_id,
        "payload": payload,
    }
    return payload


def _dataset_download_bytes(
    record: DatasetRecord,
    output_format: str,
    compress: bool,
) -> bytes:
    """Serialize the converted dataset only when Download is clicked."""

    exported = serialize_array(
        record.values,
        output_format,
        axes=record.axes,
        labels=record.labels,
        metadata=record.metadata,
    )
    if compress:
        exported = gzip_payload(exported.payload, exported.extension)
    return exported.payload


def _clear_data_output() -> None:
    """Hide the current conversion output without deleting its shared dataset."""

    for key in (ACTIVE_DATASET_KEY, PREVIEW_IMAGE_KEY, EXPORT_KEY):
        st.session_state.pop(key, None)


def _render_dataset_result(record: DatasetRecord) -> None:
    view = dataset_record_view(record)
    with st.spinner("Summarizing prepared data..."):
        summary = summarize_view(view)

    value_range = "n/a" if summary.minimum is None else f"{summary.minimum:.4g} to {summary.maximum:.4g}"
    st.caption(
        f"{dataset_display_name(record)} | shape {' x '.join(map(str, summary.shape)) or 'scalar'} | "
        f"{summary.dtype} | {memory_label(record.nbytes)} | range {value_range} | "
        f"{summary.missing_count:,} missing/non-finite value(s)"
    )

    with st.expander("Preview", expanded=False):
        sample = representative_preview(record.values, rows=12, columns=10)
        st.dataframe(sample, width="stretch", hide_index=True, height=min(420, 54 + sample.shape[0] * 35))
        st.caption("Representative sample; calculations and exports use the full data.")
        preview_key = initialized_widget_key(
            PAGE_NAMESPACE, f"show_preview_{str(record.dataset_id)[:16]}", False
        )
        if not isinstance(st.session_state.get(preview_key), bool):
            st.session_state[preview_key] = False
        show_plot = st.toggle(
            "Render plot preview",
            key=preview_key,
            help="Plot construction is deferred until enabled.",
        )
        if show_plot:
            try:
                st.image(_preview_image(record), width="stretch")
                st.caption("Representative preview of the converted data.")
            except (ValueError, TypeError, OSError, MemoryError) as exc:
                st.warning(f"A representative plot could not be built: {exc}")

    st.caption("Export")
    formats = ["NPY", "NPV", "NPZ"]
    if record.values.ndim <= 2:
        formats = ["CSV", "TSV", "TXT"] + formats
    export_cols = st.columns([0.38, 0.18, 0.12, 0.32], gap="small")
    output_name_key = "data_management_output_file_name"
    if not isinstance(st.session_state.get(output_name_key, ""), str):
        st.session_state[output_name_key] = ""
    output_name = export_cols[0].text_input(
        "File name",
        key=output_name_key,
        placeholder="Name (auto)",
        label_visibility="collapsed",
    )
    output_format_key = "data_management_export_format"
    if st.session_state.get(output_format_key) not in formats:
        st.session_state[output_format_key] = formats[0]
    output_label = export_cols[1].selectbox(
        "File type",
        formats,
        key=output_format_key,
        label_visibility="collapsed",
    )
    gzip_key = "data_management_export_gzip"
    if not isinstance(st.session_state.get(gzip_key), bool):
        st.session_state[gzip_key] = False
    compress = export_cols[2].checkbox(
        "GZ",
        key=gzip_key,
        help="Compression starts when Download is clicked.",
    )
    extension, mime = array_export_target(
        output_label,
        compressed=bool(compress),
    )
    default_stem = safe_name_stem("dnascape", Path(record.name).stem, record.data_form)
    stem = safe_name_stem(output_name) if output_name.strip() else default_stem
    export_cols[3].download_button(
        "Download",
        data=partial(
            _dataset_download_bytes,
            record,
            output_label.lower(),
            bool(compress),
        ),
        file_name=f"{stem}.{extension}",
        mime=mime,
        on_click="ignore",
        width="stretch",
    )


def render_data_management_workspace() -> None:
    left, right = st.columns([0.35, 0.65], gap="medium")
    with left:
        with st.container(key="dnascape-control-stack"):
            with st.container(border=True, height="stretch", key="dnascape-control-card"):
                upload, inspection = _render_source_panel()
            with st.container(key="dnascape-control-action"):
                convert_button_slot = st.empty()

    with right:
        with st.container(border=True, key="dnascape-preview-card"):
            st.markdown(
                '<div class="dnascape-panel-title">Conversion</div>',
                unsafe_allow_html=True,
            )
            config = None
            if inspection is not None:
                try:
                    config = _render_preparation_controls(inspection)
                except (ValueError, KeyError, TypeError, OverflowError) as exc:
                    st.error(f"The retained source settings are invalid: {exc}")

            with convert_button_slot:
                submitted = st.button(
                    "Convert",
                    type="primary",
                    key="data_management_convert_button",
                    width="stretch",
                    help="Loading and preprocessing run only when this button is pressed.",
                    disabled=upload is None or inspection is None or config is None,
                )

            if submitted and upload is not None and inspection is not None and config is not None:
                try:
                    config.validate(inspection)
                    with st.spinner("Loading and preparing the selected data..."):
                        record = get_or_prepare_upload_dataset(upload, inspection, config)
                    st.session_state[ACTIVE_DATASET_KEY] = record.dataset_id
                    st.session_state.pop(EXPORT_KEY, None)
                except (
                    ValueError,
                    KeyError,
                    OSError,
                    EOFError,
                    TypeError,
                    ImportError,
                    MemoryError,
                ) as exc:
                    st.error(str(exc))

            active = get_dataset(st.session_state.get(ACTIVE_DATASET_KEY))
            if not isinstance(active, DatasetRecord):
                st.caption("Upload a source, choose the conversion settings, then click Convert.")
            elif upload is not None and (
                active.source_digest != upload.digest
                or (config is not None and active.dataset_id != prepared_dataset_id(upload, config))
            ):
                st.warning(
                    "The source or conversion settings changed. Click Convert to update "
                    "the previous result shown below."
                )
                try:
                    _render_dataset_result(active)
                except (ValueError, TypeError, KeyError, IndexError, OverflowError, MemoryError) as exc:
                    st.error(f"The retained dataset cannot be displayed: {exc}")
            else:
                try:
                    _render_dataset_result(active)
                except (ValueError, TypeError, KeyError, IndexError, OverflowError, MemoryError) as exc:
                    st.error(f"The retained dataset cannot be displayed: {exc}")


def render_page() -> None:
    """Render Data Management inside the persistent DNAscape shell."""

    render_page_intro(
        "Data Management",
        "Convert genomic data files",
        "Upload a supported file and convert it between DNAscape-compatible formats.",
    )
    render_control_panel_marker("data-management-page-marker")
    render_data_management_workspace()
