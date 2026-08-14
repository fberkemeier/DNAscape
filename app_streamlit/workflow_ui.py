"""Shared source-selection and zero-copy dataset views for app workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import streamlit as st

from app_streamlit.data_service import (
    DatasetRecord,
    PreparationConfig,
    SourceInspection,
    dataset_display_name,
    inspect_source,
    prepare_dataset,
    prepared_dataset_id,
)
from app_streamlit.preview import ArraySummary, array_summary
from app_streamlit.state import (
    UploadRecord,
    datasets,
    get_dataset,
    get_dataset_summary,
    get_source_inspection,
    get_upload,
    persistent_widget_key,
    register_dataset,
    resolve_uploaded_file,
    set_dataset_summary,
    set_source_inspection,
    uploads,
    workflow_results,
)


@dataclass(frozen=True)
class DatasetView:
    """A non-owning numerical view of a dataset or workflow result."""

    view_id: str
    name: str
    values: np.ndarray
    data_form: str
    labels: tuple[str, ...] = ()
    axes: Mapping[str, np.ndarray] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    dataset: DatasetRecord | None = None

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(axis) for axis in self.values.shape)

    @property
    def nbytes(self) -> int:
        return int(self.values.nbytes)


@dataclass(frozen=True)
class SourceSelection:
    """One selected prepared dataset or retained upload."""

    kind: str
    dataset: DatasetRecord | None = None
    upload: UploadRecord | None = None
    inspection: SourceInspection | None = None


def _safe_selectbox_state(key: str, options: Sequence[Any]) -> None:
    """Drop a stale preserved select value before Streamlit validates options."""

    try:
        current_is_valid = key not in st.session_state or st.session_state[key] in options
    except (TypeError, ValueError):
        current_is_valid = False
    if not current_is_valid:
        st.session_state.pop(key, None)


def inspect_retained_upload(
    upload: UploadRecord,
    *,
    refresh: bool = False,
) -> SourceInspection:
    """Inspect an upload once per SHA-256 digest for the current session."""

    if not refresh:
        cached = get_source_inspection(upload.digest)
        if isinstance(cached, SourceInspection):
            return cached
    inspected = inspect_source(upload)
    return set_source_inspection(upload.digest, inspected)


def get_or_prepare_upload_dataset(
    upload: UploadRecord,
    inspection: SourceInspection,
    config: PreparationConfig,
) -> DatasetRecord:
    """Reuse or explicitly prepare the canonical dataset for an upload/config pair."""

    expected_id = prepared_dataset_id(upload, config)
    existing = get_dataset(expected_id)
    if isinstance(existing, DatasetRecord):
        return existing
    return register_dataset(prepare_dataset(upload, inspection, config))


def render_upload_source_selector(
    namespace: str,
    *,
    label: str = "Upload source data",
    allowed_types: Sequence[str] | None = None,
    help_text: str | None = None,
) -> UploadRecord | None:
    """Render one uploader plus a retained-upload chooser without reparsing data."""

    upload_key = f"{namespace}_file_uploader"
    uploaded_file = st.file_uploader(
        label,
        type=list(allowed_types) if allowed_types else None,
        accept_multiple_files=False,
        help=help_text,
        key=upload_key,
    )
    try:
        resolved = resolve_uploaded_file(uploaded_file, upload_key)
    except (TypeError, ValueError, MemoryError) as exc:
        st.error(str(exc))
        resolved = None
    retained = sorted(uploads(), key=lambda item: item.created_at, reverse=True)
    if not retained:
        return resolved

    by_digest = {item.digest: item for item in retained}
    choice_key = persistent_widget_key(namespace, "retained_upload")
    options = list(by_digest)
    if resolved is not None and uploaded_file is not None:
        st.session_state[choice_key] = resolved.digest
    _safe_selectbox_state(choice_key, options)
    selected_digest = st.selectbox(
        "Retained source",
        options,
        format_func=lambda digest: (
            f"{by_digest[digest].name} ({by_digest[digest].size_bytes / 1024:.1f} KiB, "
            f"{digest[:10]})"
        ),
        key=choice_key,
        help="Uploads are retained by content digest while you move between pages.",
    )
    return get_upload(selected_digest)


def select_prepared_dataset(
    namespace: str,
    *,
    label: str = "Prepared dataset",
    allow_none: bool = True,
) -> DatasetRecord | None:
    """Select a canonical session dataset without copying its numerical arrays."""

    records = [item for item in datasets() if isinstance(item, DatasetRecord)]
    by_id = {record.dataset_id: record for record in records}
    options = ([""] if allow_none else []) + list(by_id)
    if not options:
        st.info("Prepare a dataset on Data Management before continuing.")
        return None
    key = persistent_widget_key(namespace, "prepared_dataset")
    _safe_selectbox_state(key, options)
    selected = st.selectbox(
        label,
        options,
        format_func=lambda value: "Choose a dataset" if not value else dataset_display_name(by_id[value]),
        key=key,
    )
    return get_dataset(selected) if selected else None


def render_scientific_source_selector(
    namespace: str,
    *,
    allowed_types: Sequence[str] | None = None,
) -> SourceSelection:
    """Select either a canonical dataset or an inspected retained upload."""

    modes = ["Prepared dataset", "Retained upload"]
    key = persistent_widget_key(namespace, "source_kind")
    _safe_selectbox_state(key, modes)
    mode = st.radio("Source", modes, horizontal=True, key=key)
    if mode == "Prepared dataset":
        return SourceSelection(kind="dataset", dataset=select_prepared_dataset(namespace))
    upload = render_upload_source_selector(namespace, allowed_types=allowed_types)
    inspection = get_source_inspection(upload.digest) if upload else None
    return SourceSelection(
        kind="upload",
        upload=upload,
        inspection=inspection if isinstance(inspection, SourceInspection) else None,
    )


def render_profile_view(
    namespace: str,
    *,
    allow_upload: bool = True,
    allowed_types: Sequence[str] | None = None,
) -> DatasetView | None:
    """Select an existing view or explicitly prepare a retained plain-data upload."""

    existing_views = available_data_views(science_compatible=True)
    modes = (["Prepared data or result"] if existing_views else []) + (
        ["Upload data"] if allow_upload else []
    )
    if not modes:
        st.info("Prepare a dataset on Data Management before continuing.")
        return None
    mode_key = persistent_widget_key(namespace, "profile_source_mode")
    _safe_selectbox_state(mode_key, modes)
    mode = st.radio("Profile source", modes, horizontal=True, key=mode_key)
    if mode == "Prepared data or result":
        return select_data_view(namespace, science_compatible=True)

    upload = render_upload_source_selector(
        f"{namespace}_profile",
        allowed_types=allowed_types or ("csv", "tsv", "txt", "npy", "npv", "npz", "gz"),
    )
    if upload is None:
        return None
    inspection = get_source_inspection(upload.digest)
    if not isinstance(inspection, SourceInspection):
        if st.button("Inspect source", key=f"{namespace}_inspect_profile"):
            try:
                with st.spinner("Inspecting source metadata..."):
                    inspection = inspect_retained_upload(upload)
            except (ValueError, OSError, UnicodeError, ModuleNotFoundError) as exc:
                st.error(str(exc))
        if not isinstance(inspection, SourceInspection):
            return None
    if inspection.is_track:
        st.info("Prepare genomic tracks on Data Management before selecting them here.")
        return None

    form_key = persistent_widget_key(namespace, "profile_data_form")
    form_options = ["auto", "space", "time", "spacetime", "table"]
    _safe_selectbox_state(form_key, form_options)
    with st.form(f"{namespace}_prepare_profile_form"):
        data_form = st.selectbox("Data form", form_options, key=form_key)
        value_column = None
        if inspection.columns:
            column_options = [""] + list(inspection.columns)
            column_key = persistent_widget_key(namespace, "profile_value_column")
            _safe_selectbox_state(column_key, column_options)
            value_column = st.selectbox(
                "Value column",
                column_options,
                format_func=lambda value: "All numeric columns" if not value else value,
                key=column_key,
            ) or None
        array_key = None
        if inspection.array_keys:
            array_key_key = persistent_widget_key(namespace, "profile_array_key")
            _safe_selectbox_state(array_key_key, inspection.array_keys)
            array_key = st.selectbox("NPZ array", inspection.array_keys, key=array_key_key)
        prepare_clicked = st.form_submit_button("Prepare profile", type="primary")

    active_key = f"{namespace}_prepared_profile_dataset_id"
    if prepare_clicked:
        try:
            config = PreparationConfig(
                data_form=data_form,
                value_column=value_column,
                array_key=array_key,
            )
            with st.spinner("Preparing profile..."):
                record = get_or_prepare_upload_dataset(upload, inspection, config)
            st.session_state[active_key] = record.dataset_id
        except (ValueError, KeyError, OSError, TypeError, MemoryError) as exc:
            st.error(str(exc))
            return None

    record = get_dataset(st.session_state.get(active_key))
    if isinstance(record, DatasetRecord) and record.source_digest == upload.digest:
        return dataset_record_view(record)
    return None


def dataset_record_view(record: DatasetRecord) -> DatasetView:
    """Return a non-owning view over one canonical dataset record."""

    return DatasetView(
        view_id=f"dataset:{record.dataset_id}",
        name=record.name,
        values=record.values,
        data_form=record.data_form,
        labels=record.labels,
        axes=record.axes,
        metadata=record.metadata,
        dataset=record,
    )


def _resolved_data_form(values: np.ndarray) -> str:
    if values.ndim == 1:
        return "space"
    if values.ndim == 2:
        return "time" if values.shape[0] == 1 else "spacetime"
    return "array"


def _payload_array_views(
    workflow: str,
    signature: str,
    payload: Any,
) -> list[DatasetView]:
    """Expose top-level numerical workflow arrays without retaining copies."""

    if isinstance(payload, DatasetRecord):
        return [dataset_record_view(payload)]
    output_forms = (
        payload.get("output_forms", {})
        if isinstance(payload, Mapping) and isinstance(payload.get("output_forms"), Mapping)
        else {}
    )
    if isinstance(payload, np.ndarray):
        candidates = [("values", payload, {})]
    elif isinstance(payload, Mapping):
        candidates: list[tuple[str, np.ndarray, Mapping[str, Any]]] = []
        containers = [("", payload)]
        for container_name in ("raw", "outputs", "result"):
            nested = payload.get(container_name)
            if isinstance(nested, Mapping):
                containers.append((container_name, nested))
        axis_names = {"position_kb", "position_bp", "start_bp", "time_grid", "time_min", "x", "t"}
        for prefix, container in containers:
            for name, value in container.items():
                if str(name) in axis_names or not isinstance(value, np.ndarray):
                    continue
                if value.ndim not in {1, 2} or value.dtype.kind not in "biufc":
                    continue
                candidates.append((f"{prefix}.{name}".strip("."), value, container))
    else:
        return []

    views: list[DatasetView] = []
    seen: set[int] = set()
    for name, values, container in candidates:
        if id(values) in seen:
            continue
        seen.add(id(values))
        field_name = name.rsplit(".", 1)[-1]
        data_form = str(output_forms.get(field_name) or _resolved_data_form(values))
        axes: dict[str, np.ndarray] = {}
        if data_form == "space":
            allowed_axes = ("position_kb", "position_bp", "start_bp")
        elif data_form == "time":
            allowed_axes = ("time_grid", "time_min")
        elif data_form == "spacetime":
            allowed_axes = (
                "position_kb", "position_bp", "start_bp", "time_grid", "time_min"
            )
        elif data_form == "ensemble_space":
            allowed_axes = ("position_kb", "position_bp", "start_bp", "realization")
        else:
            allowed_axes = ()
        for axis_name in allowed_axes:
            axis = container.get(axis_name) if isinstance(container, Mapping) else None
            if not isinstance(axis, np.ndarray):
                continue
            if axis.ndim == 1 and axis.size in values.shape:
                axes[axis_name] = axis
        views.append(
            DatasetView(
                view_id=f"workflow:{workflow}:{signature[:16]}:{name}",
                name=f"{workflow.title()} - {name.replace('_', ' ')}",
                values=values,
                data_form=data_form,
                axes=axes,
                metadata={"workflow": workflow, "input_signature": signature, "field": name},
            )
        )
    return views


def workflow_result_views(*, include_legacy: bool = True) -> tuple[DatasetView, ...]:
    """Return zero-copy views of current mapping/simulation results."""

    views: list[DatasetView] = []
    for workflow, result in workflow_results().items():
        views.extend(_payload_array_views(workflow, result.input_signature, result.payload))

    if include_legacy:
        for workflow, key in (
            ("mapping", "mapping_last_map"),
            ("simulation", "simulation_last_preview"),
        ):
            payload = st.session_state.get(key)
            if payload is not None:
                views.extend(_payload_array_views(workflow, f"legacy-{id(payload)}", payload))

    unique: dict[str, DatasetView] = {}
    for view in views:
        unique.setdefault(view.view_id, view)
    return tuple(unique.values())


def available_data_views(
    *,
    include_workflows: bool = True,
    science_compatible: bool = False,
) -> tuple[DatasetView, ...]:
    """Return canonical dataset views followed by current workflow-result views."""

    views = [dataset_record_view(record) for record in datasets() if isinstance(record, DatasetRecord)]
    if include_workflows:
        views.extend(workflow_result_views())
    if science_compatible:
        views = [
            view for view in views
            if view.data_form in {"space", "time", "spacetime", "table"}
        ]
    return tuple(views)


def select_data_view(
    namespace: str,
    *,
    label: str = "Data source",
    include_workflows: bool = True,
    science_compatible: bool = False,
) -> DatasetView | None:
    """Render a stable selector for prepared datasets and workflow outputs."""

    views = available_data_views(
        include_workflows=include_workflows,
        science_compatible=science_compatible,
    )
    by_id = {view.view_id: view for view in views}
    options = [""] + list(by_id)
    key = persistent_widget_key(namespace, "data_view")
    _safe_selectbox_state(key, options)
    selected = st.selectbox(
        label,
        options,
        format_func=lambda value: (
            "Choose prepared data or a workflow result"
            if not value
            else f"{by_id[value].name} - {' x '.join(map(str, by_id[value].shape))}"
        ),
        key=key,
    )
    return by_id.get(selected)


def summarize_view(view: DatasetView) -> ArraySummary:
    """Compute exact statistics once for a canonical dataset/result view."""

    cache_key = view.dataset.dataset_id if view.dataset is not None else view.view_id
    cached = get_dataset_summary(cache_key)
    if isinstance(cached, ArraySummary):
        return cached
    return set_dataset_summary(cache_key, array_summary(view.values))


def view_series(view: DatasetView, label: str | None = None) -> np.ndarray:
    """Return a one-dimensional, non-owning series from a selected view."""

    values = view.values
    if values.ndim == 1 or view.data_form == "time":
        return values.reshape(-1)
    if values.ndim != 2:
        raise ValueError("Choose a one- or two-dimensional source for line plotting.")
    labels = view.labels if len(view.labels) == values.shape[1] else tuple(
        f"value_{index + 1}" for index in range(values.shape[1])
    )
    selected = label or labels[0]
    if selected not in labels:
        raise ValueError(f"Unknown data series: {selected!r}.")
    return values[:, labels.index(selected)]


def view_axis(view: DatasetView, length: int, *, resolution_kb: float = 1.0) -> np.ndarray:
    """Return a matching existing axis or a lightweight regular coordinate array."""

    preferred = (
        ("time_min", 1.0),
        ("time_grid", 1.0),
    ) if view.data_form == "time" else (
        ("position_kb", 1.0),
        ("position_bp", 1 / 1_000.0),
        ("start_bp", 1 / 1_000.0),
    )
    for name, scale in preferred:
        axis = view.axes.get(name)
        if axis is not None and np.asarray(axis).size == int(length):
            return np.asarray(axis, dtype=float) * scale
    return np.arange(int(length), dtype=float) * float(resolution_kb)
