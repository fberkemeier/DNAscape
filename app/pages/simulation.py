"""Interactive NiceGUI simulation workflow."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import CancelledError
from dataclasses import dataclass, replace
from itertools import chain
from threading import Event, Lock
from typing import Any

import numpy as np
from nicegui import run, ui
from dnascape._resources import ensure_single_source, hosted, LOCAL_SETUP_URL

from app.artifact_archives import (
    DEFAULT_BATCH_ARCHIVE_SOURCE_BYTES,
    build_artifact_archive,
)
from app.batch_workflows import (
    BatchSourceSummary,
    unique_dataset_names,
)
from app.export_settings import get_export_settings
from app.field_hints import field_hint
from app.notifications import notify
from app.data_filter_controls import DataFilterControls
from app.simulation_filter_guides import add_simulation_filter_guides
from dnascape.workflows.simulation_filters import FILTER_LABELS, build_simulation_filter_export
from app.statistics_exports import HistogramStatisticsExport
from app.raw_data_transfer import RawInputTarget, open_raw_artifact
from app.mapping_uploads import (
    UploadInspection,
    inspect_upload_event,
    materialize_upload,
)
from app.page_activity import current_page_activity
from dnascape.workflows.preview_inputs import (
    INPUT_PREFIX, add_simulation_input_previews, input_preview_result,
)
from app.pages.common import (
    progress_message,
    DATA_ICON,
    RawOutputControls,
    action_dock,
    workflow_action_button,
    workflow_action_icon,
    data_upload,
    append_raw_uploads,
    expansion,
    page_intro,
    panel,
    panel_title,
    preview_plot,
    raw_output_controls,
    raw_output_file_row,
    set_data_upload_locked,
    stat_strip,
    two_column_fields,
)
from app.preview_plotting import (
    CURVE_TYPES,
    axis_limits_from_controls,
    FIXED_QUANTITY_Y_LIMITS,
    HistogramBinSynchronizer,
    PreviewBand,
    PreviewLine,
    automatic_y_limits,
    bind_preview_settings_updates,
    build_preview_plot_options,
    compact_color_input,
    histogram_x_limits_from_controls,
    has_finite_values,
    plotting_settings_ready,
    preview_settings_controls,
    preview_settings_markers,
    render_preview_chart,
    sync_preview_settings_visibility,
)
from app.result_naming import (
    RSIM_FILE_PREFIX,
    ResultName,
    compact_export_stem,
    normalize_name_style,
    output_file_token,
)
from app.session_cache import (
    cache_artifact,
    cache_upload,
    cache_workflow,
    consume_cached_workflow,
    current_session_cache,
)
from app.simulation_exports import (
    build_chart_capture_export,
    build_raw_export,
    select_result_output,
)
from app.simulation_service import (
    MAX_REPLICATES,
    ConstantSimulationRequest,
    ConstantSimulationResult,
    run_constant_simulation,
    validate_constant_simulation,
)
from app.simulation_profile_exports import (
    build_individual_profile_archive,
    build_standard_deviation_export,
    individual_profile_count as retained_individual_profile_count,
    limit_replication_profiles,
    profile_archive_stem,
)


@dataclass(frozen=True)
class _OutputSpec:
    key: str
    x_label: str
    y_label: str
    kind: str


_BASE_OUTPUTS = {
    "Replication timing": _OutputSpec(
        "replication_timing", "Position (kb)", "Replication timing", "space"
    ),
    "Origin efficiency": _OutputSpec(
        "origin_efficiency", "Position (kb)", "Origin efficiency", "space"
    ),
    "Fork directionality": _OutputSpec(
        "fork_directionality", "Position (kb)", "Fork directionality", "space"
    ),
}
_DISTRIBUTION_OUTPUTS = {
    "S-phase duration": _OutputSpec(
        "s_phase_duration", "Replicate", "Duration (min)", "distribution"
    ),
    "Origins per replicate": _OutputSpec("num_origins", "Replicate", "Origins", "distribution"),
    "Inter-origin distance": _OutputSpec(
        "inter_origin_distances",
        "Sample",
        "Distance (kb)",
        "distribution",
    ),
}
_TIME_OUTPUTS = {
    "Replicated fraction (time)": _OutputSpec(
        "replicated_fraction", "Time (min)", "Replicated fraction", "time"
    ),
    "Total forks (time)": _OutputSpec("total_forks", "Time (min)", "Total forks", "time"),
    "Active forks (time)": _OutputSpec("active_forks", "Time (min)", "Active forks", "time"),
    "Stalled forks (time)": _OutputSpec("stalled_forks", "Time (min)", "Stalled forks", "time"),
    "Origin firing (time)": _OutputSpec("origin_firing", "Time (min)", "Origins per interval", "time"),
    "Firing factors (time)": _OutputSpec("firing_factors", "Time (min)", "Available factors", "time"),
}
_TIME_DISTRIBUTION_OUTPUTS = {
    "Replicon length": _OutputSpec("replicon_lengths", "Sample", "Length (kb)", "distribution"),
}
_TIME_TRACKED_OUTPUTS = {**_TIME_OUTPUTS, **_TIME_DISTRIBUTION_OUTPUTS}
_SPACETIME_OUTPUTS = {
    "Replicated fraction (space-time)": _OutputSpec(
        "replicated_fraction_xt", "Position (kb)", "Time (min)", "xt"
    ),
    "Initiation events (space-time)": _OutputSpec(
        "initiation_events_xt", "Position (kb)", "Time (min)", "xt"
    ),
    "Coalescence events (space-time)": _OutputSpec(
        "coalescence_events_xt", "Position (kb)", "Time (min)", "xt"
    ),
    "Right fork density (space-time)": _OutputSpec(
        "right_moving_fork_density_xt", "Position (kb)", "Time (min)", "xt"
    ),
    "Left fork density (space-time)": _OutputSpec(
        "left_moving_fork_density_xt", "Position (kb)", "Time (min)", "xt"
    ),
    "Total fork density (space-time)": _OutputSpec(
        "fork_density_xt", "Position (kb)", "Time (min)", "xt"
    ),
}
_STANDARD_OUTPUTS = {**_BASE_OUTPUTS, **_DISTRIBUTION_OUTPUTS}
_INPUT_OUTPUTS = {
    "Initiation rate (input)": _OutputSpec("input:initiation_rate", "Position (kb)", "Initiation rate", "input"),
    "Fork speed (input)": _OutputSpec("input:fork_speed", "Position (kb)", "Fork speed", "input"),
}
_OUTPUT_SPECS = {
    **_STANDARD_OUTPUTS,
    **_TIME_TRACKED_OUTPUTS,
    **_SPACETIME_OUTPUTS,
    **_INPUT_OUTPUTS,
}


def _output_spec(label: str, result: ConstantSimulationResult | None = None) -> _OutputSpec:
    spec = _OUTPUT_SPECS[label]
    if spec.kind != "input":
        return spec
    form = getattr(result, "input_forms", {}).get(spec.key, "x")
    if form == "xt":
        return replace(spec, kind="xt", y_label="Time (min)")
    if form == "t":
        return replace(spec, kind="xt", y_label="Time (min)")
    return replace(spec, kind="space")

# Some simulation quantities have a meaningful physical domain which should not
# shrink to the particular values observed in one run. Other outputs retain an
# automatic, data-dependent range chosen by ECharts.
_NONNEGATIVE_LINE_OUTPUTS = {
    "Origin efficiency",
    "Total forks (time)",
    "Active forks (time)",
    "Stalled forks (time)",
    "Origin firing (time)",
    "Firing factors (time)",
}

_SOURCE_ACCEPT = ".csv,.tsv,.txt,.npy,.npv,.npz,.gz"
_SOURCE_DEPENDENCE = {
    "x": "Space",
    "t": "Time",
    "xt": "Space-time",
}

from dnascape.workflows.simulation_recipe import (
    _PreparedSimulationSource, _format_simulation_component,
    _plan_simulation_cases, _simulation_case_label,
)

_MAX_PREVIEW_PROFILES = 50
_BATCH_RAW_PAGE_SIZE = 50
_BATCH_NAV_PAGE_SIZE = 50
_MAX_COMBINED_PREVIEW_DATASETS = 50


@dataclass(frozen=True)
class _SimulationSourceControls:
    """Live controls and retained upload state for one rsim kinetic field."""

    mode: Any
    form: Any
    constant: Any
    constants: list[Any]
    constant_actions: list[Any]
    refresh_constants: Callable[[], Any]
    uploader: Any
    inspections: list[UploadInspection]
    selection: dict[str, str | None]
    selections: dict[str, dict[str, str | None]]
    selection_controls: list[Any]
    materialized_cache: dict[tuple[Any, ...], np.ndarray]


@dataclass
class _SimulationConstantState:
    """Retained multiple constants keyed by manual row or counterpart file."""

    values_by_key: dict[str, float]
    manual_keys: list[str]
    next_manual_index: int

    @classmethod
    def create(cls) -> _SimulationConstantState:
        return cls(values_by_key={}, manual_keys=["manual-1", "manual-2"], next_manual_index=3)










@dataclass(frozen=True)
class _SimulationBatchItem:
    """One named result retained by the Simulation preview."""

    key: str
    identity: ResultName
    result: ConstantSimulationResult
    full_export_stem: str = ""
    short_export_stem: str = ""

    @property
    def name(self) -> str:
        """Retain the fully descriptive result name used by diagnostics."""

        return self.identity.full

    @property
    def export_stem(self) -> str:
        """Retain the fully descriptive legacy export-stem interface."""

        return self.full_export_stem


@dataclass(frozen=True)
class _SimulationBatchPayload:
    """Cache payload used only when a workflow contains several datasets."""

    items: tuple[_SimulationBatchItem, ...]


@dataclass
class _SimulationRawPreference:
    """Editable raw-export values retained across batch-row refreshes."""

    file_name: str
    file_type: str = "TSV"
    compress: bool = False
    automatic_name: str = ""
    customized: bool = False




def _simulation_case_stem(display_name: str) -> str:
    """Create a bounded case-only stem through the shared filename policy."""

    return compact_export_stem(
        _simulation_case_label(display_name),
        fallback="dataset",
        maximum=110,
    )


def _simulation_raw_default_filename(
    item: _SimulationBatchItem,
    output_key: str,
) -> str:
    """Return a stable output-first filename for one simulation dataset."""

    base = f"{RSIM_FILE_PREFIX}_{output_file_token(output_key)}"
    case_stem = _simulation_item_stem(item)
    return base if case_stem == "dataset" else f"{base}_{case_stem}"


def _simulation_plot_default_filename(
    item: _SimulationBatchItem,
    output_key: str,
) -> str:
    """Use the same workflow/output/case order for plot captures."""

    return f"{_simulation_raw_default_filename(item, output_key)}_plot"


def _simulation_item_stem(item: _SimulationBatchItem) -> str:
    """Return the collision-safe stem matching the visible naming style."""

    if _simulation_name_style() == "full":
        return item.full_export_stem or _simulation_case_stem(item.identity.full)
    return (
        item.short_export_stem
        or item.full_export_stem
        or _simulation_case_stem(item.identity.short)
    )


def _simulation_display_name(item: _SimulationBatchItem) -> str:
    """Resolve one visible label through the global naming preference."""

    return item.identity.display(_simulation_name_style())


def _simulation_name_style() -> str:
    """Read the shared preference, tolerating narrow test doubles."""

    return normalize_name_style(
        getattr(get_export_settings(), "result_name_style", "short")
    )


def _simulation_raw_page_bounds(total: int, requested_page: int) -> tuple[int, int, int, int]:
    """Return ``page, page_count, start, end`` for bounded batch-row rendering."""

    normalized_total = max(0, int(total))
    page_count = max(1, math.ceil(normalized_total / _BATCH_RAW_PAGE_SIZE))
    page = min(max(1, int(requested_page)), page_count)
    start = (page - 1) * _BATCH_RAW_PAGE_SIZE
    end = min(normalized_total, start + _BATCH_RAW_PAGE_SIZE)
    return page, page_count, start, end


_BATCH_COLORS = (
    "#b03412",
    "#2596be",
    "#3f6da8",
    "#8b6f37",
    "#6f5b8f",
    "#4f7d3a",
    "#a64f70",
    "#4f737f",
)


def _remove_simulation_uploads(
    inspections: list[UploadInspection],
    selections: dict[str, dict[str, str | None]],
    event: Any,
) -> None:
    """Remove exactly the files represented by one shared upload event."""

    removed_name = str(getattr(event, "name", "") or "")
    removed_names = tuple(
        str(name) for name in (getattr(event, "names", ()) or ()) if str(name)
    )
    if removed_name:
        names_to_remove = {removed_name}
    elif removed_names:
        # A bulk event is a snapshot. Files completing inspection after that
        # snapshot must remain in the scientific source state.
        names_to_remove = set(removed_names)
    else:
        # Backwards compatibility for pre-bulk events with an empty name.
        names_to_remove = {item.name for item in inspections}
    inspections[:] = [
        item for item in inspections if item.name not in names_to_remove
    ]
    for name in names_to_remove:
        selections.pop(name, None)


def _render_simulation_multiple_constants(
    *,
    label: str,
    marker: str,
    default: float,
    units: str,
    retained: _SimulationConstantState,
    peer_rows: Callable[[], tuple[tuple[str, str], ...]],
    combination_mode: dict[str, str],
    constant_controls: list[Any],
    action_controls: list[Any],
    on_change: Callable[[], Any],
    on_structure_change: Callable[[], Any],
) -> None:
    """Render Mapping-equivalent retained number rows for one rsim factor."""

    aligned_rows = peer_rows()
    manual = not aligned_rows
    if manual:
        aligned_rows = tuple(
            (key, f"Value {index}")
            for index, key in enumerate(retained.manual_keys, start=1)
        )
    for index, (row_key, _row_label) in enumerate(aligned_rows):
        if row_key in retained.values_by_key:
            continue
        manual_key = retained.manual_keys[index] if index < len(retained.manual_keys) else ""
        retained.values_by_key[row_key] = retained.values_by_key.get(
            manual_key,
            default,
        )

    with (
        ui.element("div")
        .classes("dn-mapping-constant-batch")
        .mark(f"source-{marker}-constants")
    ):
        matching = (
            ui.select(
                {"pairwise": "Pairwise", "all": "All combinations"},
                value=combination_mode.get("value", "pairwise"),
                label="Matching",
            )
            .props("dense options-dense")
            .classes("dn-field dn-mapping-constant-matching")
            .mark(f"source-{marker}-combination-mode")
        )
        action_controls.append(matching)

        def matching_changed() -> None:
            selected = str(matching.value or "pairwise")
            combination_mode["value"] = (
                selected if selected in {"pairwise", "all"} else "pairwise"
            )
            on_change()
            on_structure_change()

        matching.on_value_change(matching_changed)
        with (
            ui.element("div")
            .classes("dn-mapping-constant-list")
            .mark(f"source-{marker}-constant-list")
        ):
            for index, (row_key, row_label) in enumerate(aligned_rows, start=1):
                with ui.element("div").classes("dn-mapping-constant-row"):
                    row_name = ui.label(row_label).classes("dn-mapping-constant-label")
                    row_name.tooltip(row_label)
                    control = (
                        ui.number(
                            value=retained.values_by_key[row_key],
                            min=0,
                            step=0.001 if label.casefold() == "initiation rate" else 0.1,
                        )
                        .props("dense")
                        .classes("dn-field dn-mapping-constant-value")
                        .mark(f"source-{marker}-constant-{index}")
                    )
                    control._props["aria-label"] = f"{label} for {row_label}"
                    field_hint(control, f"Constant {label.lower()} for this dataset.")

                    def value_changed(
                        value_control: Any = control,
                        identity: str = row_key,
                    ) -> None:
                        try:
                            value = float(value_control.value)
                        except (TypeError, ValueError):
                            on_change()
                            return
                        if math.isfinite(value):
                            retained.values_by_key[identity] = value
                        on_change()

                    control.on_value_change(value_changed)
                    constant_controls.append(control)
                    if manual:
                        with ui.element("div").classes("dn-mapping-constant-actions"):
                            remove = (
                                ui.button(icon="remove")
                                .props(
                                    'flat round dense aria-label="Remove constant value"'
                                )
                                .mark(f"source-{marker}-constant-remove-{index}")
                            )
                            remove.set_enabled(len(retained.manual_keys) > 1)
                            action_controls.append(remove)

                            def remove_value(identity: str = row_key) -> None:
                                if len(retained.manual_keys) <= 1:
                                    return
                                retained.manual_keys.remove(identity)
                                retained.values_by_key.pop(identity, None)
                                on_change()
                                on_structure_change()

                            remove.on_click(remove_value)
        if manual:
            with ui.element("div").classes("dn-mapping-constant-add-row"):
                add = (
                    ui.button("Add value", icon="add")
                    .props("flat no-caps dense")
                    .classes("dn-action-secondary")
                    .mark(f"source-{marker}-constant-add")
                )
                action_controls.append(add)

                def add_value() -> None:
                    identity = f"manual-{retained.next_manual_index}"
                    retained.next_manual_index += 1
                    retained.manual_keys.append(identity)
                    retained.values_by_key[identity] = default
                    on_change()
                    on_structure_change()

                add.on_click(add_value)


def _render_simulation_source(
    label: str,
    *,
    constant: float,
    units: str,
    retained_constants: _SimulationConstantState | None = None,
    peer_constant_rows: Callable[[], tuple[tuple[str, str], ...]] | None = None,
    combination_mode: dict[str, str] | None = None,
    on_profile_change: Callable[[], Any] | None = None,
    on_structure_change: Callable[[], Any] | None = None,
) -> _SimulationSourceControls:
    """Render one scalar-or-upload rsim source using Mapping's upload surface."""

    marker = "-".join(label.casefold().replace("/", " ").split())
    inspections: list[UploadInspection] = []
    selection: dict[str, str | None] = {"column": None, "array_key": None}
    selections: dict[str, dict[str, str | None]] = {}
    selection_controls: list[Any] = []
    materialized_cache: dict[tuple[Any, ...], np.ndarray] = {}
    profile_changed = on_profile_change or (lambda: None)
    structure_changed = on_structure_change or (lambda: None)
    constant_state = retained_constants or _SimulationConstantState.create()
    shared_combination_mode = combination_mode or {"value": "pairwise"}
    constant_controls: list[Any] = []
    constant_actions: list[Any] = []
    local_constant_refresh: dict[str, Callable[[], Any]] = {"call": lambda: None}

    def constant_structure_changed() -> None:
        if on_structure_change is None:
            local_constant_refresh["call"]()
        else:
            structure_changed()

    with ui.element("div").classes("dn-source-compact").mark(f"source-{marker}"):
        with ui.element("div").classes("dn-source-head"):
            mode = (
                ui.select(
                    {
                        "scalar": "Constant",
                        **({} if hosted() else {"constants": "Multiple constants"}),
                        "profile": "Profile",
                    },
                    value="scalar",
                    label="Dependence",
                )
                .props("dense options-dense")
                .classes("dn-field dn-source-mode")
                .mark(f"source-{marker}-mode")
            )
            form = (
                ui.select(
                    {"x": "Space", "t": "Time", "xt": "Space-time"},
                    value="x",
                )
                .props('dense options-dense aria-label="Detected profile dependence"')
                .classes("dn-simulation-exact-form")
                .mark(f"source-{marker}-form")
            )
            form.set_visibility(False)
        with ui.element("div").classes("dn-source-fields") as constant_fields:
            constant_input = (
                ui.number(
                    label=f"Constant value ({units})",
                    value=constant,
                    min=0,
                    step=0.001 if constant < 0.1 else 0.1,
                )
                .props("dense")
                .classes("dn-field")
                .mark(f"source-{marker}-constant")
            )
        with ui.element("div").classes("dn-source-fields") as constants_fields:

            @ui.refreshable
            def render_constant_fields() -> None:
                constant_controls.clear()
                constant_actions.clear()
                _render_simulation_multiple_constants(
                    label=label,
                    marker=marker,
                    default=constant,
                    units=units,
                    retained=constant_state,
                    peer_rows=peer_constant_rows or (lambda: ()),
                    combination_mode=shared_combination_mode,
                    constant_controls=constant_controls,
                    action_controls=constant_actions,
                    on_change=profile_changed,
                    on_structure_change=constant_structure_changed,
                )

            render_constant_fields()
            local_constant_refresh["call"] = render_constant_fields.refresh

        with (
            ui.element("div")
            .classes("dn-source-fields dn-source-profile-fields")
            .mark(f"source-{marker}-profile")
        ) as upload_fields:

            def sync_selection() -> None:
                if not inspections:
                    selection.update(column=None, array_key=None)
                    selections.clear()
                    return
                inspection = inspections[0]
                row_vector = bool(
                    str(form.value) == "t"
                    and inspection.shape
                    and inspection.shape[0] == 1
                    and len(inspection.numeric_columns) > 1
                )
                if str(form.value) in {"x", "t"} and inspection.numeric_columns and not row_vector:
                    if selection["column"] not in inspection.numeric_columns:
                        selection["column"] = inspection.default_column
                else:
                    selection["column"] = None
                if inspection.array_keys:
                    if selection["array_key"] not in inspection.array_keys:
                        selection["array_key"] = inspection.default_array_key
                else:
                    selection["array_key"] = None

                retained_names = {item.name for item in inspections}
                for stale_name in tuple(selections):
                    if stale_name not in retained_names:
                        selections.pop(stale_name, None)
                for item in inspections:
                    item_selection = selections.setdefault(
                        item.name,
                        {"column": None, "array_key": None},
                    )
                    item_row_vector = bool(
                        str(form.value) == "t"
                        and item.shape
                        and item.shape[0] == 1
                        and len(item.numeric_columns) > 1
                    )
                    if (
                        str(form.value) in {"x", "t"}
                        and item.numeric_columns
                        and not item_row_vector
                    ):
                        shared_column = selection["column"]
                        if shared_column in item.numeric_columns:
                            item_selection["column"] = shared_column
                        elif item_selection["column"] not in item.numeric_columns:
                            item_selection["column"] = item.default_column
                    else:
                        item_selection["column"] = None
                    if item.array_keys:
                        shared_array = selection["array_key"]
                        if shared_array in item.array_keys:
                            item_selection["array_key"] = shared_array
                        elif item_selection["array_key"] not in item.array_keys:
                            item_selection["array_key"] = item.default_array_key
                    else:
                        item_selection["array_key"] = None

            @ui.refreshable
            def render_selection() -> None:
                selection_controls.clear()
                sync_selection()
                if not inspections:
                    return
                inspection = inspections[0]
                matrix_columns = (
                    inspection.inferred_form == "xt" and len(inspection.numeric_columns) > 1
                )
                row_vector = bool(
                    str(form.value) == "t"
                    and inspection.shape
                    and inspection.shape[0] == 1
                    and len(inspection.numeric_columns) > 1
                )
                if matrix_columns:
                    column_options: Any = {
                        "__all_columns__": "All columns (space-time)",
                        **{column: column for column in inspection.numeric_columns},
                    }
                    column_value = selection["column"] or "__all_columns__"
                    column_label = "Profile"
                elif (
                    str(form.value) in {"x", "t"}
                    and len(inspection.numeric_columns) > 1
                    and not row_vector
                ):
                    column_options = list(inspection.numeric_columns)
                    column_value = selection["column"]
                    column_label = "Value column"
                else:
                    column_options = None
                    column_value = None
                    column_label = "Value column"
                if column_options is not None:
                    column_control = (
                        ui.select(
                            column_options,
                            value=column_value,
                            label=column_label,
                        )
                        .props("dense options-dense")
                        .classes("dn-field dn-simulation-source-select")
                        .mark(f"source-{marker}-value-column")
                    )
                    def store_column() -> None:
                        selected = (
                            None if column_control.value is None else str(column_control.value)
                        )
                        if matrix_columns and selected == "__all_columns__":
                            selection["column"] = None
                            form.set_value("xt")
                        else:
                            selection["column"] = selected
                            if matrix_columns:
                                form.set_value("x")
                        materialized_cache.clear()
                        sync_selection()
                        render_summary.refresh()
                        profile_changed()

                    column_control.on_value_change(store_column)
                    selection_controls.append(column_control)
                if inspection.array_keys:
                    array_control = (
                        ui.select(
                            list(inspection.array_keys),
                            value=selection["array_key"],
                            label="Array",
                        )
                        .props("dense options-dense")
                        .classes("dn-field dn-simulation-source-select")
                        .mark(f"source-{marker}-array-key")
                    )
                    if len(inspection.array_keys) == 1:
                        array_control.disable()

                    def store_array() -> None:
                        selection["array_key"] = (
                            None if array_control.value is None else str(array_control.value)
                        )
                        materialized_cache.clear()
                        sync_selection()
                        render_summary.refresh()
                        profile_changed()

                    array_control.on_value_change(store_array)
                    selection_controls.append(array_control)

            async def uploaded(event: Any) -> bool:
                try:
                    inspection = await inspect_upload_event(
                        event,
                        expected_form="auto",
                    )
                except (OSError, TypeError, ValueError) as exc:
                    notify(str(exc), type="negative")
                    return False
                detected_form = str(inspection.inferred_form)
                if detected_form not in _SOURCE_DEPENDENCE:
                    notify(
                        f"{label} must be a space, time, or space-time profile.",
                        type="negative",
                    )
                    return False
                # Match the shared uploader's replacement semantics: an upload
                # with the same basename replaces its previous retained bytes,
                # while a new basename extends the ordered batch.
                replaced_index = next(
                    (
                        index
                        for index, item in enumerate(inspections)
                        if item.name == inspection.name
                    ),
                    None,
                )
                selections.pop(inspection.name, None)
                materialized_cache.clear()
                if hosted():
                    inspections[:] = [inspection]
                elif replaced_index is None:
                    inspections.append(inspection)
                else:
                    # Keep N/N initiation-rate and fork-speed pairing stable when
                    # one same-name dataset is replaced.
                    inspections[replaced_index] = inspection
                if len(inspections) == 1:
                    form.set_value(detected_form)
                sync_selection()
                render_selection.refresh()
                render_summary.refresh()
                cache_upload(inspection, source="Simulation", navigate_to="simulation")
                profile_changed()
                constant_structure_changed()
                return True

            def cleared(event: Any) -> None:
                _remove_simulation_uploads(inspections, selections, event)
                materialized_cache.clear()
                if not inspections:
                    selection.update(column=None, array_key=None)
                else:
                    form.set_value(str(inspections[0].inferred_form))
                    sync_selection()
                render_selection.refresh()
                render_summary.refresh()
                profile_changed()
                constant_structure_changed()

            uploader = data_upload(
                accept=_SOURCE_ACCEPT,
                accepted_formats="CSV, TSV, TXT, NPY, NPV, NPZ, GZ",
                marker=f"source-{marker}-upload",
                show_accepted_formats=False,
                on_upload=uploaded,
                on_clear=cleared,
                multiple=not hosted(),
                max_files=1 if hosted() else None,
                hide_when_loaded=False,
            )

            @ui.refreshable
            def render_summary() -> None:
                if not inspections:
                    return
                inspection = inspections[0]
                shape = inspection.shape
                try:
                    selected = _materialize_simulation_source(
                        _SimulationSourceControls(
                            mode=mode,
                            form=form,
                            constant=constant_input,
                            constants=constant_controls,
                            constant_actions=constant_actions,
                            refresh_constants=render_constant_fields.refresh,
                            uploader=uploader,
                            inspections=inspections,
                            selection=selection,
                            selections=selections,
                            selection_controls=selection_controls,
                            materialized_cache=materialized_cache,
                        ),
                        label=label,
                        missing_policy="zero",
                    )
                    shape = tuple(np.asarray(selected).shape)
                except (OSError, TypeError, ValueError):
                    pass
                size = " x ".join(str(value) for value in shape) if shape else "1"
                dependence = _SOURCE_DEPENDENCE.get(str(form.value), "Unknown")
                with (
                    ui.element("div")
                    .classes("dn-mapping-detected-data dn-profile-metadata")
                    .mark(f"source-{marker}-detected-data")
                ):
                    ui.label(f"Dependence: {dependence}").classes(
                        "dn-mapping-detected-summary dn-mapping-detected-dependence"
                    )
                    ui.label(f"Array size: {size}").classes("dn-mapping-detected-summary")
                    if len(inspections) > 1:
                        ui.label(f"Files: {len(inspections):,}").classes(
                            "dn-mapping-detected-summary"
                        )

            render_summary()
            render_selection()

        constant_fields.bind_visibility_from(
            mode,
            "value",
            backward=lambda value: value == "scalar",
        )
        constants_fields.bind_visibility_from(
            mode,
            "value",
            backward=lambda value: value == "constants",
        )
        upload_fields.bind_visibility_from(
            mode,
            "value",
            backward=lambda value: value == "profile",
        )

        def mode_changed() -> None:
            selection["column"] = None
            materialized_cache.clear()
            render_selection.refresh()
            render_summary.refresh()
            profile_changed()
            constant_structure_changed()

        mode.on_value_change(mode_changed)

    return _SimulationSourceControls(
        mode=mode,
        form=form,
        constant=constant_input,
        constants=constant_controls,
        constant_actions=constant_actions,
        refresh_constants=render_constant_fields.refresh,
        uploader=uploader,
        inspections=inspections,
        selection=selection,
        selections=selections,
        selection_controls=selection_controls,
        materialized_cache=materialized_cache,
    )


def _materialize_simulation_source(
    controls: _SimulationSourceControls,
    *,
    label: str,
    missing_policy: str,
    inspection: UploadInspection | None = None,
) -> float | np.ndarray:
    """Build one rsim field from its scalar control or retained upload."""

    if str(controls.mode.value) == "scalar":
        return _number(controls.constant, label)
    form = str(controls.form.value)
    if form not in {"x", "t", "xt"}:
        raise ValueError(f"Choose a valid input type for {label.casefold()}.")
    if not controls.inspections and inspection is None:
        raise ValueError(f"Upload {label.casefold()} data.")
    retained_inspection = inspection or controls.inspections[0]
    selected_fields = controls.selections.get(
        retained_inspection.name,
        controls.selection,
    )
    cache_key = (
        id(retained_inspection),
        form,
        selected_fields["column"],
        selected_fields["array_key"],
        missing_policy,
    )
    cached = controls.materialized_cache.get(cache_key)
    if cached is not None:
        return cached
    if hosted():
        controls.materialized_cache.clear()
    selected_inspection = replace(retained_inspection, expected_form=form)
    materialized = np.asarray(
        materialize_upload(
            selected_inspection,
            column=selected_fields["column"] if form in {"x", "t"} else None,
            array_key=selected_fields["array_key"],
            missing_policy=missing_policy,
        )
    )
    controls.materialized_cache[cache_key] = materialized
    return materialized


def _materialize_simulation_sources(
    controls: _SimulationSourceControls,
    *,
    label: str,
    missing_policy: str,
) -> tuple[float | np.ndarray, ...]:
    """Materialize every retained file, or one scalar for broadcast."""

    mode = str(controls.mode.value)
    if mode == "scalar":
        return (_number(controls.constant, label),)
    if mode == "constants":
        ensure_single_source(len(controls.constants), label)
        return _read_simulation_constant_controls(
            controls.constants,
            label=label,
            positive=label.casefold() == "fork speed",
        )
    if not controls.inspections:
        raise ValueError(f"Upload {label.casefold()} data.")
    ensure_single_source(len(controls.inspections), label)
    return tuple(
        _materialize_simulation_source(
            controls,
            label=label,
            missing_policy=missing_policy,
            inspection=inspection,
        )
        for inspection in controls.inspections
    )


def _validate_simulation_batch_selection(
    controls: _SimulationSourceControls,
    *,
    label: str,
) -> None:
    """Reject ambiguous per-file signal fallbacks before batch materialization.

    A differently named sole numeric column/array is unambiguous and remains
    convenient. When a later file exposes several choices, however, silently
    substituting its default could simulate a different biological signal.
    """

    if str(controls.mode.value) != "profile":
        return
    form = str(controls.form.value)
    detected_forms = tuple(
        _simulation_batch_detected_form(inspection, form)
        for inspection in controls.inspections
    )
    if form == "xt":
        incompatible = tuple(
            inspection.name
            for inspection, detected in zip(
                controls.inspections,
                detected_forms,
                strict=True,
            )
            if detected not in {None, "xt"}
        )
        if incompatible:
            raise ValueError(
                f"{label} expects one complete space-time matrix per dataset; "
                "these files are not space-time matrices: "
                + ", ".join(incompatible)
                + "."
            )
    known_forms = {detected for detected in detected_forms if detected is not None}
    if len(known_forms) > 1:
        readable = ", ".join(sorted(known_forms))
        raise ValueError(
            f"{label} files do not have one consistent data type ({readable})."
        )
    if len(controls.inspections) < 2:
        return
    shared_column = controls.selection.get("column")
    shared_array = controls.selection.get("array_key")
    for inspection in controls.inspections[1:]:
        if inspection.array_keys:
            if (
                shared_array is not None
                and shared_array not in inspection.array_keys
                and len(inspection.array_keys) > 1
            ):
                raise ValueError(
                    f"The {label.lower()} input does not contain the selected "
                    f"array {shared_array!r}; choose files with one consistent signal."
                )
            continue
        row_vector = bool(
            form == "t"
            and inspection.shape
            and inspection.shape[0] == 1
            and len(inspection.numeric_columns) > 1
        )
        if (
            form in {"x", "t"}
            and not row_vector
            and shared_column is not None
            and shared_column not in inspection.numeric_columns
            and len(inspection.numeric_columns) > 1
        ):
            raise ValueError(
                f"The {label.lower()} input does not contain the selected "
                f"column {shared_column!r}; choose files with one consistent signal."
            )


def _simulation_batch_detected_form(
    inspection: UploadInspection,
    selected_form: str,
) -> str | None:
    """Resolve X/T vector ambiguity while preserving true XT mismatches."""

    detected = str(inspection.inferred_form)
    if detected not in {"x", "t", "xt"}:
        return None
    if selected_form in {"x", "t"} and detected in {"x", "t"}:
        return selected_form
    return detected


def _simulation_value_form(value: float | np.ndarray) -> str:
    """Return rsim's semantic form for one already-materialized field."""

    array = np.asarray(value)
    if array.size == 1:
        return "scalar"
    if array.ndim == 1 or (array.ndim == 2 and array.shape[1] == 1):
        return "x"
    if array.ndim == 2 and array.shape[0] == 1:
        return "t"
    if array.ndim == 2:
        return "xt"
    raise ValueError("Simulation profiles must be scalar, space, time, or space-time data.")


def _simulation_value_shape(value: float | np.ndarray) -> tuple[int, ...]:
    array = np.asarray(value)
    return () if array.size == 1 else tuple(int(size) for size in array.shape)


def _simulation_spatial_size(value: float | np.ndarray) -> int | None:
    """Return the rsim spatial dimension, if the field has one."""

    array = np.asarray(value)
    form = _simulation_value_form(value)
    if form == "x":
        return int(array.size)
    if form == "xt":
        return int(array.shape[0])
    return None


def _source_batch_summary(
    controls: _SimulationSourceControls,
    values: Sequence[float | np.ndarray],
    *,
    label: str,
) -> BatchSourceSummary | None:
    mode = str(controls.mode.value)
    if mode == "scalar":
        return None
    if mode == "constants":
        if len(values) <= 1:
            return None
        return BatchSourceSummary(
            label=label,
            filenames=tuple(f"Dataset {index}" for index in range(1, len(values) + 1)),
            forms=tuple("scalar" for _value in values),
            shapes=tuple(() for _value in values),
        )
    return BatchSourceSummary(
        label=label,
        filenames=tuple(inspection.name for inspection in controls.inspections),
        forms=tuple(
            _simulation_batch_detected_form(
                inspection,
                str(controls.form.value),
            )
            for inspection in controls.inspections
        ),
        shapes=tuple(_simulation_value_shape(value) for value in values),
    )


def _prepare_simulation_source(
    controls: _SimulationSourceControls,
    values: Sequence[float | np.ndarray],
    *,
    label: str,
    units: str,
) -> _PreparedSimulationSource:
    """Attach validation and descriptive naming metadata to one rsim input."""

    retained_values = tuple(values)
    mode = str(controls.mode.value)
    summary = _source_batch_summary(controls, retained_values, label=label)
    if mode == "profile":
        components = unique_dataset_names(
            tuple(inspection.name for inspection in controls.inspections)
        )
    else:
        components = tuple(
            _format_simulation_component(float(value), units=units)
            for value in retained_values
        )
    return _PreparedSimulationSource(
        label=label,
        values=retained_values,
        components=components,
        summary=summary,
    )


class _ProgressState:
    """Thread-safe hand-off from DNAscape's worker callback to the UI loop."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._value = 0.0
        self._message = "Waiting to run"

    def report(self, value: float, message: str) -> None:
        with self._lock:
            self._value = max(0.0, min(1.0, float(value)))
            self._message = progress_message(message)

    def snapshot(self) -> tuple[float, str]:
        with self._lock:
            return self._value, self._message


def _number(control: Any, label: str) -> float:
    """Read a NiceGUI numeric control with a useful validation message."""

    if control.value is None:
        raise ValueError(f"{label} is required.")
    try:
        return float(control.value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a number.") from exc


def _read_simulation_constant_controls(
    controls: Sequence[Any],
    *,
    label: str,
    positive: bool,
) -> tuple[float, ...]:
    """Read Mapping-style constant rows only when Simulation starts."""

    if not controls:
        raise ValueError(f"Add at least one {label.casefold()} value.")
    values = tuple(
        _number(control, f"{label} value {index}")
        for index, control in enumerate(controls, start=1)
    )
    if any(not math.isfinite(value) for value in values):
        raise ValueError(f"{label} values must be finite.")
    if positive and any(value <= 0 for value in values):
        raise ValueError(f"{label} values must be greater than zero.")
    if not positive and any(value < 0 for value in values):
        raise ValueError(f"{label} values must be non-negative.")
    return values


def _integer(control: Any, label: str) -> int:
    value = _number(control, label)
    if not np.isfinite(value) or not value.is_integer():
        raise ValueError(f"{label} must be a whole number.")
    return int(value)


def _bounded_profile_count(value: Any, available: int) -> int | None:
    """Return a complete profile-count edit without emitting transient errors."""

    if value in (None, "") or available < 1:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric) or not numeric.is_integer() or numeric < 1:
        return None
    return min(int(numeric), min(_MAX_PREVIEW_PROFILES, int(available)))


def _replication_profile_arrays(
    result: ConstantSimulationResult,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return validated optional ensemble arrays from a detailed result."""

    profiles = getattr(result, "replication_time_profiles", None)
    standard_deviation = getattr(result, "replication_timing_std", None)
    profile_array = None if profiles is None else np.asarray(profiles, dtype=float)
    if profile_array is not None and (
        profile_array.ndim != 2 or profile_array.shape[1] != result.position_kb.size
    ):
        profile_array = None
    deviation_array = (
        None
        if standard_deviation is None
        else np.asarray(standard_deviation, dtype=float).reshape(-1)
    )
    if deviation_array is not None and deviation_array.size != result.position_kb.size:
        deviation_array = None
    return profile_array, deviation_array


def _automatic_y_limits(
    result: ConstantSimulationResult,
    *,
    output_label: str,
    view_mode: str,
    histogram_bins: int,
    density: bool,
    kde_bandwidth: float = 1.0,
    log_y: bool,
    log_x: bool = False,
    histogram_bin_size: float | None = None,
    histogram_x_limits: tuple[float, float] | None = None,
    histogram_boundary: str = "left",
    show_individual_profiles: bool = False,
    individual_profile_count: int = 10,
    show_standard_deviation: bool = False,
) -> tuple[float, float]:
    """Return stable, editable plot bounds derived from the selected result."""

    spec = _output_spec(output_label, result)
    result = input_preview_result(result, spec.key)
    selected = select_result_output(result, spec.key)
    plot_type = (
        "Heatmap"
        if spec.kind == "xt"
        else "Boxplot"
        if view_mode == "Boxplot"
        else "Histogram"
        if spec.kind == "distribution" or view_mode == "Histogram"
        else "Line"
    )
    plotted_values: list[Any] = [selected.values]
    if output_label == "Replication timing" and plot_type == "Line":
        profiles, standard_deviation = _replication_profile_arrays(result)
        if profiles is not None and show_individual_profiles:
            count = min(
                max(1, int(individual_profile_count)),
                profiles.shape[0],
                _MAX_PREVIEW_PROFILES,
            )
            plotted_values.extend(profiles[:count])
        if standard_deviation is not None and show_standard_deviation:
            mean = np.asarray(selected.values, dtype=float).reshape(-1)
            plotted_values.extend((mean - standard_deviation, mean + standard_deviation))
    limits = automatic_y_limits(
        plotted_values,
        plot_type=plot_type,
        histogram_bins=histogram_bins,
        density=density,
        kde_bandwidth=kde_bandwidth,
        log_y=log_y,
        log_x=log_x,
        histogram_bin_size=histogram_bin_size,
        histogram_x_limits=histogram_x_limits,
        histogram_boundary=histogram_boundary,
        y_axis=result.time_grid if spec.kind == "xt" else None,
        fixed_limits=_default_y_limits(
            output_label,
            view_mode=view_mode,
            log_y=log_y,
        ),
        floor_at_zero=output_label in _NONNEGATIVE_LINE_OUTPUTS,
    )
    if spec.kind != "xt" or spec.key.startswith(INPUT_PREFIX):
        return limits

    # Detailed fields are tracked over the requested simulation window, which
    # can extend far beyond the biologically informative part of S phase. Use
    # the latest finite ensemble-mean replication time as the visible cutoff
    # for every space-time output. Never extend beyond the retained time grid:
    # doing so would introduce an empty band in a truncated result.
    mean_replication_times = np.asarray(result.replication_timing, dtype=float).reshape(-1)
    mean_replication_times = mean_replication_times[np.isfinite(mean_replication_times)]
    tracked_times = np.asarray(result.time_grid, dtype=float).reshape(-1)
    tracked_times = tracked_times[np.isfinite(tracked_times)]
    if log_y:
        mean_replication_times = mean_replication_times[mean_replication_times > 0]
        tracked_times = tracked_times[tracked_times > 0]
    if mean_replication_times.size == 0 or tracked_times.size == 0:
        return limits
    y_maximum = min(
        float(np.max(mean_replication_times)),
        float(np.max(tracked_times)),
    )
    if y_maximum <= limits[0]:
        return limits
    return limits[0], y_maximum


def _default_y_limits(
    output_label: str,
    *,
    view_mode: str = "Default",
    log_y: bool = False,
) -> tuple[float, float] | None:
    """Return a fixed physical Y domain when the active view has one.

    Histogram Y values are frequencies rather than the selected quantity, and
    logarithmic axes cannot use the zero/negative lower bounds in these domains.
    Those cases therefore continue to use a data-derived automatic range.
    """

    spec = _output_spec(output_label)
    if spec.kind in {"distribution", "xt"} or view_mode in {"Histogram", "Boxplot"} or log_y:
        return None
    if spec.key == "origin_efficiency":
        return None
    return FIXED_QUANTITY_Y_LIMITS.get(spec.key)


def _default_axis_labels(
    *,
    output_label: str,
    view_mode: str,
    density: bool,
    result: ConstantSimulationResult | None = None,
) -> tuple[str, str]:
    """Return labels for the quantity and frequency axes in the active view."""

    spec = _output_spec(output_label, result)
    if view_mode == "Boxplot":
        return "", spec.y_label
    if spec.kind == "distribution" or view_mode == "Histogram":
        return spec.y_label, "Density" if density else "Count"
    return spec.x_label, spec.y_label


def _result_plot_options(
    result: ConstantSimulationResult,
    *,
    output_label: str,
    title: str,
    series_label: str,
    x_axis_label: str,
    y_axis_label: str,
    color: str,
    palette: str,
    view_mode: str,
    histogram_bins: int,
    density: bool,
    show_legend: bool,
    show_colorbar: bool,
    log_y: bool,
    invert_y: bool,
    y_limits: tuple[float, float] | None,
    legend_position: str = "Top right",
    line_width: float = 1.8,
    line_type: str = "solid",
    show_markers: bool = False,
    show_grid: bool = False,
    log_z: bool = False,
    log_x: bool = False,
    invert_x: bool = False,
    invert_z: bool = False,
    show_histogram_statistics: bool = False,
    histogram_statistics_position: str = "Top right",
    histogram_bin_size: float | None = None,
    histogram_x_limits: tuple[float, float] | None = None,
    histogram_touching: bool = True,
    kde_bandwidth: float = 1.0,
    histogram_grouped: bool = False,
    histogram_boundary: str = "left",
    show_individual_profiles: bool = False,
    individual_profile_count: int = 10,
    show_standard_deviation: bool = False,
    x_limits: tuple[float, float] | None = None,
    z_limits: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Build bounded ECharts options for any supported simulation output."""

    spec = _output_spec(output_label, result)
    result = input_preview_result(result, spec.key)
    values = result.outputs.get(spec.key)
    if values is None:
        if spec.key == "firing_factors":
            raise ValueError("Enable a finite firing-factor pool to plot firing factors.")
        if output_label in _TIME_TRACKED_OUTPUTS:
            raise ValueError(
                "Run the simulation again with time courses enabled to plot this result."
            )
        if output_label in _SPACETIME_OUTPUTS:
            raise ValueError(
                "Run the simulation again with space-time outputs enabled to plot this result."
            )
        raise ValueError(f"DNAscape did not return {output_label.lower()}.")
    values = np.asarray(values)
    if not has_finite_values(values):
        if spec.key == "inter_origin_distances":
            raise ValueError(
                "No inter-origin distances were recorded. At least two origins must "
                "fire in a realization."
            )
        if spec.key == "replicon_lengths":
            raise ValueError(
                "No completed replicons were recorded. Increase the maximum replication time or initiation rate."
            )
        raise ValueError(
            "The selected output has no finite values. Try a positive initiation rate or a "
            "longer replication-time cap."
        )

    plot_type = (
        "Heatmap"
        if spec.kind == "xt"
        else "Boxplot"
        if view_mode == "Boxplot"
        else "Histogram"
        if spec.kind == "distribution" or view_mode == "Histogram"
        else "Line"
    )
    axis = result.position_kb if spec.kind == "space" else result.time_grid
    if (spec.key == "fork_directionality" and plot_type == "Line"
            and result.geometry == "periodic" and axis.size > 1):
        # Positions are bin starts; the circle ends one full bin beyond the
        # last sample. Close only the preview, retaining the raw n-bin output.
        axis = np.append(axis, axis[-1] + (axis[-1] - axis[-2]))
        values = np.append(values, values[0])
    lines: tuple[PreviewLine, ...] = ()
    bands: tuple[PreviewBand, ...] = ()
    ensemble_active = False
    if output_label == "Replication timing" and plot_type == "Line":
        profiles, standard_deviation = _replication_profile_arrays(result)
        individual_lines: tuple[PreviewLine, ...] = ()
        if profiles is not None and show_individual_profiles:
            count = min(
                max(1, int(individual_profile_count)),
                profiles.shape[0],
                _MAX_PREVIEW_PROFILES,
            )
            individual_lines = tuple(
                PreviewLine(
                    name="Individual simulations",
                    x=result.position_kb,
                    y=profile,
                    color=color,
                    opacity=0.22,
                    width=max(0.55, float(line_width) * 0.55),
                    show_in_legend=index == 0,
                    z=2,
                )
                for index, profile in enumerate(profiles[:count])
            )
            ensemble_active = bool(individual_lines)
        mean = np.asarray(values, dtype=float).reshape(-1)
        if standard_deviation is not None and show_standard_deviation:
            bands = (
                PreviewBand(
                    name="Mean ± SD",
                    x=result.position_kb,
                    lower=mean - standard_deviation,
                    upper=mean + standard_deviation,
                    color=color,
                    opacity=0.16,
                ),
            )
            ensemble_active = True
        if ensemble_active:
            lines = (
                *individual_lines,
                PreviewLine("Mean", result.position_kb, mean, color, z=4),
            )
    heatmap_values = values.T if spec.kind == 'xt' else values
    heatmap_positions = result.position_kb
    if spec.kind == 'xt' and result.input_forms.get(spec.key) == 't':
        from dnascape.workflows.temporal_preview import temporal_heatmap
        heatmap_values, heatmap_positions, _ = temporal_heatmap(values, result.position_kb, result.time_grid)
    options = build_preview_plot_options(
        plot_type=plot_type,
        x_limits=x_limits,
        z_limits=z_limits,
        title=title.strip() or output_label,
        x_label=x_axis_label.strip(),
        y_label=y_axis_label.strip(),
        color=color,
        palette=palette,
        histogram_bins=histogram_bins,
        density=density,
        show_legend=show_legend,
        show_colorbar=show_colorbar,
        log_y=log_y,
        log_z=log_z,
        invert_y=invert_y,
        y_limits=y_limits,
        series_label=(
            output_label
            if plot_type == "Heatmap"
            else series_label.strip() or ("Mean" if plot_type == "Line" else output_label)
        ),
        values=heatmap_values,
        x_axis=heatmap_positions if spec.kind == "xt" else axis,
        y_axis=result.time_grid if spec.kind == "xt" else None,
        lines=lines,
        bands=bands,
        legend_position=legend_position,
        line_width=line_width,
        line_type=line_type,
        show_markers=show_markers,
        show_grid=show_grid,
        interactive=not ensemble_active or view_mode == "Circular",
        log_x=log_x,
        invert_x=invert_x,
        invert_z=invert_z,
        show_histogram_statistics=show_histogram_statistics,
        histogram_statistics_position=histogram_statistics_position,
        histogram_bin_size=histogram_bin_size,
        histogram_x_limits=histogram_x_limits,
        histogram_touching=histogram_touching,
        kde_bandwidth=kde_bandwidth,
        histogram_grouped=histogram_grouped,
        histogram_boundary=histogram_boundary,
    )
    if ensemble_active:
        for series in options["series"]:
            series["dnascapeCircularGroup"] = "replication_timing"
            series["dnascapeCircularEnsembleMember"] = series.get("name") == "Individual simulations"
    if plot_type == "Heatmap":
        options["dnascapeDefaultAxes"]["y"] = list(_automatic_y_limits(
            result, output_label=output_label, view_mode=view_mode, histogram_bins=histogram_bins,
            density=density, log_y=log_y, log_x=log_x, kde_bandwidth=kde_bandwidth,
        ))
    return options

def _combined_result_plot_options(
    items: Sequence[_SimulationBatchItem],
    *,
    output_label: str,
    title: str,
    x_axis_label: str,
    y_axis_label: str,
    colors: Mapping[str, str],
    y_limits: tuple[float, float] | None,
    show_legend: bool,
    legend_position: str,
    line_width: float,
    show_grid: bool,
    log_x: bool,
    invert_x: bool,
    log_y: bool,
    invert_y: bool,
    x_limits: tuple[float, float] | None = None,
    line_type: str = "solid",
    line_types: Mapping[str, str] | None = None,
    curve_names: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build one overlay for axis-compatible one-dimensional batch outputs."""

    if len(items) < 2:
        raise ValueError("A combined preview requires at least two datasets.")
    if len(items) > _MAX_COMBINED_PREVIEW_DATASETS:
        raise ValueError(
            "Combined previews support at most "
            f"{_MAX_COMBINED_PREVIEW_DATASETS} datasets; choose Individual."
        )
    spec = _output_spec(output_label, items[0].result)
    if spec.kind not in {"space", "time"}:
        raise ValueError("This output is shown one dataset at a time.")
    selected = tuple(select_result_output(item.result, spec.key) for item in items)
    for item, output_value in zip(items, selected, strict=True):
        axis = np.asarray(output_value.x, dtype=float).reshape(-1)
        values = np.asarray(output_value.values, dtype=float).reshape(-1)
        if axis.size != values.size:
            raise ValueError(
                f"{_simulation_display_name(item)} has incompatible plot coordinates."
            )
    lines = tuple(
        PreviewLine(
            name=(curve_names or {}).get(item.key, _simulation_display_name(item)),
            x=output_value.x,
            y=np.asarray(output_value.values, dtype=float).reshape(-1),
            color=colors.get(item.key, _BATCH_COLORS[index % len(_BATCH_COLORS)]),
            width=line_width,
            line_type=(line_types or {}).get(item.key),
        )
        for index, (item, output_value) in enumerate(zip(items, selected, strict=True))
    )
    return build_preview_plot_options(
        plot_type="Line",
        x_limits=x_limits,
        title=title.strip() or output_label,
        x_label=x_axis_label.strip(),
        y_label=y_axis_label.strip(),
        color=lines[0].color,
        palette="viridis",
        histogram_bins=60,
        density=False,
        show_legend=show_legend,
        show_colorbar=False,
        log_y=log_y,
        invert_y=invert_y,
        y_limits=y_limits,
        series_label=lines[0].name,
        values=np.asarray(selected[0].values, dtype=float),
        x_axis=np.asarray(selected[0].x, dtype=float).reshape(-1),
        lines=lines,
        line_width=line_width,
        line_type=line_type,
        show_markers=False,
        legend_position=legend_position,
        show_grid=show_grid,
        interactive=True,
        log_x=log_x,
        invert_x=invert_x,
    )


def _render_result_plot(
    result: ConstantSimulationResult,
    *,
    output_label: str,
    title: str,
    series_label: str,
    x_axis_label: str,
    y_axis_label: str,
    color: str,
    palette: str,
    view_mode: str,
    histogram_bins: int,
    density: bool,
    show_legend: bool,
    legend_position: str,
    show_colorbar: bool,
    log_y: bool,
    log_z: bool,
    invert_y: bool,
    y_limits: tuple[float, float] | None,
    on_download: Callable[[str], Any],
    line_width: float = 1.8,
    line_type: str = "solid",
    show_markers: bool = False,
    show_grid: bool = False,
    log_x: bool = False,
    invert_x: bool = False,
    invert_z: bool = False,
    show_histogram_statistics: bool = False,
    histogram_statistics_position: str = "Top right",
    histogram_bin_size: float | None = None,
    histogram_x_limits: tuple[float, float] | None = None,
    histogram_touching: bool = True,
    kde_bandwidth: float = 1.0,
    histogram_grouped: bool = False,
    histogram_boundary: str = "left",
    show_individual_profiles: bool = False,
    individual_profile_count: int = 10,
    show_standard_deviation: bool = False,
    x_limits: tuple[float, float] | None = None,
    z_limits: tuple[float, float] | None = None,
) -> Any:
    """Render a bounded ECharts preview with shared zoom/download tools."""

    options = _result_plot_options(
        result,
        x_limits=x_limits,
        z_limits=z_limits,
        output_label=output_label,
        title=title,
        series_label=series_label,
        x_axis_label=x_axis_label,
        y_axis_label=y_axis_label,
        color=color,
        palette=palette,
        view_mode=view_mode,
        histogram_bins=histogram_bins,
        density=density,
        show_legend=show_legend,
        legend_position=legend_position,
        show_colorbar=show_colorbar,
        log_y=log_y,
        log_z=log_z,
        invert_y=invert_y,
        y_limits=y_limits,
        line_width=line_width,
        line_type=line_type,
        show_markers=show_markers,
        show_grid=show_grid,
        log_x=log_x,
        invert_x=invert_x,
        invert_z=invert_z,
        show_histogram_statistics=show_histogram_statistics,
        histogram_statistics_position=histogram_statistics_position,
        histogram_bin_size=histogram_bin_size,
        histogram_x_limits=histogram_x_limits,
        histogram_touching=histogram_touching,
        kde_bandwidth=kde_bandwidth,
        histogram_grouped=histogram_grouped,
        histogram_boundary=histogram_boundary,
        show_individual_profiles=show_individual_profiles,
        individual_profile_count=individual_profile_count,
        show_standard_deviation=show_standard_deviation,
    )
    return render_preview_chart(
        options,
        title=title.strip() or output_label,
        marker="simulation-result-plot",
        on_download=on_download,
    )


def render(_navigate: Callable[[str], None]) -> None:
    """Render the local simulation controls and live preview."""

    running = False
    page_activity = current_page_activity()
    active_cancel: Event | None = None
    restored_entry = consume_cached_workflow("simulation")
    restored_result = restored_entry.payload if restored_entry is not None else None
    if isinstance(restored_result, ConstantSimulationResult):
        latest_items: tuple[_SimulationBatchItem, ...] = (
            _SimulationBatchItem(
                key="simulation-case-1",
                identity=ResultName(short="Simulation", full="Simulation"),
                result=restored_result,
                full_export_stem="dataset",
                short_export_stem="dataset",
            ),
        )
    elif isinstance(restored_result, _SimulationBatchPayload):
        latest_items = restored_result.items
    else:
        latest_items = ()
    active_dataset_index = 0
    latest_result: ConstantSimulationResult | None = (
        latest_items[0].result if latest_items else None
    )
    result_chart: Any | None = None
    y_limits_automatic = True
    updating_y_limits = False
    updating_presentation = False
    batch_colors: dict[str, str] = {}
    batch_line_types: dict[str, str] = {}
    batch_names: dict[str, str] = {}
    batch_navigation_page = {"value": 1}
    batch_raw_preferences: dict[tuple[str, str], _SimulationRawPreference] = {}
    ensemble_raw_preferences: dict[tuple[str, str], _SimulationRawPreference] = {}
    batch_raw_page = {"value": 1}
    filter_raw_refresh = {'call': lambda: None}
    rendered_name_style = {"value": _simulation_name_style()}
    source_region_refresh: dict[str, Callable[[], Any]] = {"call": lambda: None}
    simulation_constant_states = {
        "initiation_rate": _SimulationConstantState.create(),
        "fork_speed": _SimulationConstantState.create(),
    }
    simulation_combination_mode = {"value": "pairwise"}
    simulation_sources: dict[str, _SimulationSourceControls] = {}
    filter_controls: dict[str, DataFilterControls] = {}
    refreshing_constant_surfaces = False

    page_intro(
        "Simulation",
        "Simulate DNA replication",
        "Run stochastic replication ensembles and inspect their outputs.",
    )
    if hosted():
        ui.html(
            "Render accepts one constant or profile per input and one simulation at a time. "
            "Uploading a new profile replaces the previous one. "
            f'<a href="{LOCAL_SETUP_URL}" target="_blank" rel="noopener noreferrer">'
            "Run locally for multiple constants, profiles and batch comparisons</a>."
        ).classes("dn-about-copy dn-hosted-resource-note").mark("simulation-hosted-input-note")
    with ui.element("div").classes("dn-workspace"):
        with ui.column().classes("dn-control-stack"):
            with panel(extra_classes="dn-control-card"):
                panel_title("Controls")
                with expansion(
                    "Data",
                    icon=DATA_ICON,
                    expanded=True,
                    extra_classes="dn-data-section",
                ):
                    def counterpart_rows(
                        current_key: str,
                    ) -> tuple[tuple[str, str], ...]:
                        """Align multiple constants with the other uploaded factor."""

                        peer_key = (
                            "fork_speed"
                            if current_key == "initiation_rate"
                            else "initiation_rate"
                        )
                        peer = simulation_sources.get(peer_key)
                        if (
                            peer is None
                            or str(peer.mode.value) != "profile"
                            or not peer.inspections
                        ):
                            return ()
                        return tuple(
                            (f"{peer_key}:{inspection.name}", inspection.name)
                            for inspection in peer.inspections
                        )

                    def refresh_constant_surfaces() -> None:
                        """Rebuild constant rows after peer cardinality changes."""

                        nonlocal refreshing_constant_surfaces
                        if refreshing_constant_surfaces:
                            return
                        refreshing_constant_surfaces = True
                        try:
                            for source in tuple(simulation_sources.values()):
                                if str(source.mode.value) == "constants":
                                    source.refresh_constants()
                        finally:
                            refreshing_constant_surfaces = False

                    def scientific_source_structure_changed() -> None:
                        source_region_refresh["call"]()
                        refresh_constant_surfaces()

                    with ui.element("div").classes("dn-simulation-data-grid"):
                        with ui.column().classes("dn-setting-group"):
                            ui.label("Initiation rate").classes("dn-setting-group-title")
                            firing_source = _render_simulation_source(
                                "Initiation rate",
                                constant=0.01,
                                units="min⁻¹ per bin",
                                retained_constants=simulation_constant_states[
                                    "initiation_rate"
                                ],
                                peer_constant_rows=lambda: counterpart_rows(
                                    "initiation_rate"
                                ),
                                combination_mode=simulation_combination_mode,
                                on_profile_change=lambda: source_region_refresh["call"](),
                                on_structure_change=scientific_source_structure_changed,
                            )
                            simulation_sources["initiation_rate"] = firing_source
                        with ui.column().classes("dn-setting-group"):
                            ui.label("Fork speed").classes("dn-setting-group-title")
                            speed_source = _render_simulation_source(
                                "Fork speed",
                                constant=1.4,
                                units="kb/min",
                                retained_constants=simulation_constant_states[
                                    "fork_speed"
                                ],
                                peer_constant_rows=lambda: counterpart_rows(
                                    "fork_speed"
                                ),
                                combination_mode=simulation_combination_mode,
                                on_profile_change=lambda: source_region_refresh["call"](),
                                on_structure_change=scientific_source_structure_changed,
                            )
                            simulation_sources["fork_speed"] = speed_source
                    # The first source is rendered before its peer exists.
                    refresh_constant_surfaces()
                with expansion("Region", icon="straighten"):
                    with ui.row().classes('dn-simulation-region-scope'):
                        whole_region = ui.switch('Whole region', value=False).mark('simulation-whole-region')
                        periodic = ui.switch('Periodic', value=False).mark('simulation-periodic')
                        show_plot = ui.switch('Show plot', value=True).mark('simulation-show-plot')
                    whole_region.disable()
                    field_hint(whole_region, 'Use the full genomic extent of an uploaded profile.')
                    field_hint(periodic, 'Join the two region boundaries into a circle.')
                    field_hint(show_plot, 'Show the simulation preview.')
                    with two_column_fields():
                        resolution_x = (
                            ui.number(
                                label="Resolution (kb/bin)", value=1.0, min=0.000001, step=0.1
                            )
                            .props("dense")
                            .classes("dn-field")
                            .mark("simulation-resolution-x")
                        )
                        resolution_t = (
                            ui.number(
                                label="Time step (min/bin)", value=5.0, min=0.000001, step=1.0
                            )
                            .props("dense")
                            .classes("dn-field")
                            .mark("simulation-resolution-t")
                        )
                        region_start = (
                            ui.number(label="Start (kb)", value=0.0, min=0.0, step=1.0)
                            .props("dense")
                            .classes("dn-field")
                            .mark("simulation-region-start")
                        )
                        region_end = (
                            ui.number(label="End (kb)", value=1_000.0, min=0.000001, step=1.0)
                            .props("dense")
                            .classes("dn-field")
                            .mark("simulation-region-end")
                        )

                    automatic_region_end: float | None = None

                    def sync_region_to_profiles() -> None:
                        nonlocal automatic_region_end
                        sizes: set[int] = set()
                        for source, source_label in (
                            (firing_source, "Initiation rate"),
                            (speed_source, "Fork speed"),
                        ):
                            if str(source.mode.value) != "profile" or not source.inspections:
                                continue
                            try:
                                # The first file defines the shared region. The
                                # complete batch is materialized and checked in
                                # a worker only after Simulate is clicked.
                                value = _materialize_simulation_source(
                                    source,
                                    label=source_label,
                                    missing_policy="zero",
                                )
                            except (OSError, TypeError, ValueError):
                                continue
                            spatial_size = _simulation_spatial_size(value)
                            if spatial_size is None and str(source.form.value) == 'x':
                                array = np.asarray(value)
                                spatial_size = array.shape[0] if array.ndim else None
                            if spatial_size is not None:
                                sizes.add(spatial_size)
                        if len(sizes) > 1:
                            # Keep uploads permissive. The complete, named
                            # consistency error is reported transactionally when
                            # Simulate is selected.
                            whole_region.disable()
                            return
                        if not sizes:
                            whole_region.set_value(False)
                            whole_region.disable()
                            if not running:
                                region_start.enable()
                                region_end.enable()
                            return
                        if not running:
                            whole_region.enable()
                        was_automatic = automatic_region_end is None or (
                            region_start.value == 0.0 and region_end.value == automatic_region_end
                        )
                        # Array uploads are natively sampled at 1 kb. Changing
                        # the destination bin width only changes the sample count.
                        automatic_region_end = float(sizes.pop())
                        if was_automatic or whole_region.value:
                            region_start.set_value(0.0)
                            region_end.set_value(automatic_region_end)
                        region_start.set_enabled(not running and not whole_region.value)
                        region_end.set_enabled(not running and not whole_region.value)

                    source_region_refresh["call"] = sync_region_to_profiles
                    resolution_x.on_value_change(sync_region_to_profiles)
                    whole_region.on_value_change(sync_region_to_profiles)
                    sync_region_to_profiles()

                with expansion("Simulation settings", icon="tune"):
                    with ui.element("div").classes("dn-simulation-settings-grid"):
                        with ui.column().classes("dn-setting-group"):
                            ui.label("Ensemble").classes("dn-setting-group-title")
                            replicates = (
                                ui.number(
                                    label="Simulations (replicates)",
                                    value=20,
                                    min=1,
                                    max=MAX_REPLICATES,
                                    step=1,
                                )
                                .props("dense")
                                .classes("dn-field")
                                .mark("simulation-replicates")
                            )
                            missing_values = (
                                ui.select(
                                    {"error": "Error", "interpolate": "Interpolate", "zero": "Zero"},
                                    value="error",
                                    label="Missing values",
                                )
                                .props("dense options-dense")
                                .classes("dn-field")
                                .mark("simulation-missing-values")
                            )
                            with ui.element("div").classes("dn-setting-inline"):
                                fixed_seed = ui.checkbox("Fix seed", value=False).mark("simulation-fix-seed")
                                seed = (
                                    ui.number(label="Seed", value=1, min=0, step=1)
                                    .props("dense")
                                    .classes("dn-field dn-field-xs")
                                    .mark("simulation-seed")
                                )
                            seed.bind_visibility_from(fixed_seed, "value")

                        with ui.column().classes("dn-setting-group"):
                            ui.label("Timing & outputs").classes("dn-setting-group-title")
                            time_window = (
                                ui.number(
                                    label="Time window (min)",
                                    value=300.0,
                                    min=0.000001,
                                    step=10.0,
                                )
                                .props("dense")
                                .classes("dn-field")
                            )
                            max_replication_time = (
                                ui.number(
                                    label="Maximum time (min)",
                                    value=1200.0,
                                    min=0.000001,
                                    step=50.0,
                                )
                                .props("dense")
                                .classes("dn-field")
                            )
                            track_time_outputs = ui.checkbox(
                                "Track time courses",
                                value=bool(
                                    latest_result is not None
                                    and getattr(latest_result, "time_outputs", False)
                                ),
                            ).mark("simulation-track-time-outputs")
                            track_detailed_outputs = ui.checkbox(
                                "Track space-time outputs",
                                value=bool(
                                    latest_result is not None and latest_result.detailed_outputs
                                ),
                            ).mark("simulation-track-detailed-outputs")

                        with ui.column().classes("dn-setting-group dn-fork-behaviour-group"):
                            ui.label("Origin & fork behaviour").classes("dn-setting-group-title")
                            with ui.element("div").classes("dn-fork-behaviour-grid"):
                                with (
                                    ui.element("div")
                                    .classes("dn-setting-subgroup")
                                    .mark("simulation-factor-pool-settings")
                                ):
                                    ui.label("Firing factor").classes(
                                        "dn-setting-subgroup-title"
                                    )
                                    with ui.element("div").classes("dn-firing-factor-grid"):
                                        finite_pool = ui.checkbox("Finite pool", value=False).mark(
                                            "simulation-finite-pool"
                                        )
                                        factor_coupling = ui.checkbox("Fork coupling", value=False).mark(
                                            "simulation-fork-coupling"
                                        )
                                        factor_coupling.bind_visibility_from(finite_pool, "value")
                                    with (
                                        ui.element("div")
                                        .classes("dn-firing-factor-grid")
                                        .mark("simulation-pool-fields") as pool_fields
                                    ):
                                        pool_size = (
                                            ui.number(label="Pool size", value=100, min=2, step=1)
                                            .props("dense").classes("dn-field")
                                            .mark("simulation-pool-size")
                                        )
                                        recycle_rate = (
                                            ui.number(
                                                label="Recycle rate",
                                                value=0.0,
                                                min=0.0,
                                                step=0.001,
                                            )
                                            .props("dense")
                                            .classes("dn-field")
                                            .mark("simulation-recycle-rate")
                                        )
                                    pool_fields.bind_visibility_from(finite_pool, "value")

                                with (
                                    ui.element("div")
                                    .classes("dn-setting-subgroup")
                                    .mark("simulation-stalling-settings")
                                ):
                                    ui.label("Fork movement").classes("dn-setting-subgroup-title")
                                    with ui.element("div").classes("dn-fork-movement-grid"):
                                        with ui.column().classes("dn-fork-movement-options"):
                                            fork_stalling = ui.checkbox("Fork stalling", value=False).mark("simulation-fork-stalling")
                                            with ui.column().classes("dn-fork-movement-options").bind_visibility_from(fork_stalling, "value"):
                                                stall_rate = (
                                                    ui.number(label="Stall rate (stalls/kb)", value=0.0, min=0.0, step=0.001)
                                                    .props("dense").classes("dn-field")
                                                    .mark("simulation-stall-rate-per-kb")
                                                )
                                                shared_stalls = ui.checkbox(
                                                    "Shared between directions", value=False,
                                                ).mark("simulation-stall-shared-directions")
                                                stall_recovery = ui.checkbox("Stall recovery", value=False).mark("simulation-stall-recovery")
                                                stall_duration = (
                                                    ui.number(label="Recovery time (min)", value=10.0, min=0.000001, step=1.0)
                                                    .props("dense").classes("dn-field")
                                                    .mark("simulation-stall-duration")
                                                )
                                                stall_duration.bind_visibility_from(stall_recovery, "value")
                                            fork_stalling.on_value_change(lambda: stall_rate.set_value(0.0) if not fork_stalling.value else None)
                                        with ui.column().classes("dn-fork-movement-options"):
                                            stochastic_movement = ui.checkbox("Stochastic movement", value=False).mark("simulation-stochastic-movement")
                                            fork_speed_std = (
                                                ui.number(label="Standard deviation (kb/min)", value=0.1, min=0.0, step=0.1)
                                                .props("dense").classes("dn-field")
                                                .mark("simulation-fork-speed-std")
                                            )
                                            fork_speed_std.bind_visibility_from(stochastic_movement, "value")

                filter_refresh = {'call': lambda: None}
                with expansion('Region transforms', icon='filter_alt'):
                    filter_relative = ui.select(
                        {'initiation_rate': 'Initiation rate', 'fork_speed': 'Fork speed', 'both': 'Both'},
                        value='initiation_rate', label='Relative to',
                    ).props('dense options-dense').classes('dn-field').mark('simulation-filter-relative-to')
                    field_hint(filter_relative, 'Choose which simulation inputs to transform independently.')
                    filter_panels = {}

                    def sync_simulation_filters() -> None:
                        for key, section in filter_panels.items():
                            section.set_visibility(filter_relative.value in {key, 'both'})
                        filter_refresh['call']()

                    for key, label in (('initiation_rate', 'Initiation rate'), ('fork_speed', 'Fork speed')):
                        with ui.column().classes('dn-simulation-filter-input') as section:
                            filter_panels[key] = section
                            filter_controls[key] = DataFilterControls(
                                sync_simulation_filters, title=label, embedded=True,
                                marker_prefix=f'simulation-filter-{key.replace("_", "-")}',
                                allow_export=True, allow_complement=True, profile_units='bp',
                                align_modulation_from_start=True, on_change=sync_simulation_filters,
                            )
                    filter_relative.on_value_change(sync_simulation_filters)
                    sync_simulation_filters()

                def combined_preview_compatible() -> bool:
                    if not 2 <= len(latest_items) <= _MAX_COMBINED_PREVIEW_DATASETS:
                        return False
                    selected_label = str(output.value or "Replication timing")
                    spec = _output_spec(selected_label, latest_result)
                    if spec.kind not in {"space", "time"}:
                        return False
                    if str(view.value or "Default") not in {"Default", "Circular"}:
                        return False
                    try:
                        selected = tuple(
                            select_result_output(item.result, spec.key)
                            for item in latest_items
                        )
                    except ValueError:
                        return False
                    return all(
                        np.asarray(item.x).size == np.asarray(item.values).size
                        and np.asarray(item.x).size > 0
                        for item in selected
                    )

                def combined_preview_active() -> bool:
                    return bool(
                        len(latest_items) > 1
                        and str(presentation.value or "Combined") == "Combined"
                        and combined_preview_compatible()
                    )

                def current_y_limits() -> tuple[float, float] | None:
                    if y_minimum.value is None or y_maximum.value is None:
                        return None
                    limits = (
                        _number(y_minimum, "Y minimum"),
                        _number(y_maximum, "Y maximum"),
                    )
                    if limits[1] <= limits[0]:
                        raise ValueError("Y maximum must be greater than Y minimum.")
                    return limits

                def active_plot_type() -> str:
                    selected = str(output.value or "Replication timing")
                    spec = _output_spec(selected, latest_result)
                    requested_view = str(view.value or "Default")
                    if spec.kind in {"xt", "distribution"} and requested_view == "Circular":
                        view.set_value("Default")
                        requested_view = "Default"
                    if spec.kind == "xt" and requested_view == "Default":
                        return "Heatmap"
                    if requested_view == "Boxplot":
                        return "Boxplot"
                    if spec.kind == "distribution" or requested_view == "Histogram":
                        return "Histogram"
                    return "Line"

                def populate_automatic_y_limits(*, redraw: bool = True) -> None:
                    """Restore the selected output's data-derived Y range."""

                    nonlocal updating_y_limits, y_limits_automatic
                    y_limits_automatic = True
                    if latest_result is None:
                        return
                    if not plotting_settings_ready(
                        plot_settings,
                        plot_type=active_plot_type(),
                        allow_empty_y_limits=y_limits_automatic,
                    ):
                        return
                    profile_count = selected_profile_count()
                    if (
                        bool(show_individual_profiles.value)
                        and available_profile_count() > 0
                        and profile_count is None
                        and not combined_preview_active()
                    ):
                        return
                    try:
                        selected_label = str(output.value or "Replication timing")
                        if combined_preview_active():
                            spec = _output_spec(selected_label, latest_result)
                            values = [
                                select_result_output(item.result, spec.key).values
                                for item in latest_items
                            ]
                            limits = automatic_y_limits(
                                values,
                                plot_type="Line",
                                histogram_bins=_integer(
                                    histogram_bins,
                                    "Histogram bins",
                                ),
                                density=False,
                                log_y=bool(log_y.value),
                                log_x=bool(log_x.value),
                                fixed_limits=_default_y_limits(
                                    selected_label,
                                    view_mode="Default",
                                    log_y=bool(log_y.value),
                                ),
                                floor_at_zero=selected_label in _NONNEGATIVE_LINE_OUTPUTS,
                            )
                        else:
                            limits = _automatic_y_limits(
                                latest_result,
                                output_label=selected_label,
                                view_mode=str(view.value or "Default"),
                                histogram_bins=_integer(histogram_bins, "Histogram bins"),
                                density=bool(density.value),
                                kde_bandwidth=float(plot_settings.kde_bandwidth.value or 1.0),
                                log_y=bool(log_y.value),
                                log_x=bool(log_x.value),
                                histogram_bin_size=(
                                    float(plot_settings.histogram_bin_size.value)
                                    if plot_settings.histogram_bin_size.value is not None
                                    else None
                                ),
                                histogram_x_limits=histogram_x_limits_from_controls(
                                    plot_settings,
                                    log_x=bool(log_x.value),
                                ),
                                histogram_boundary=str(
                                    plot_settings.histogram_boundary.value or "left"
                                ),
                                show_individual_profiles=bool(
                                    show_individual_profiles.value
                                ),
                                individual_profile_count=profile_count or 1,
                                show_standard_deviation=bool(
                                    show_standard_deviation.value
                                ),
                            )
                    except ValueError:
                        limits = None
                    updating_y_limits = True
                    try:
                        y_minimum.value = None if limits is None else float(f"{limits[0]:.10g}")
                        y_maximum.value = None if limits is None else float(f"{limits[1]:.10g}")
                        y_minimum.update()
                        y_maximum.update()
                    finally:
                        updating_y_limits = False
                    if redraw:
                        render_latest_result()

                def use_custom_y_limits() -> None:
                    nonlocal y_limits_automatic
                    if updating_y_limits:
                        return
                    y_limits_automatic = False
                    render_latest_result()

                def refresh_axis_range() -> None:
                    if y_limits_automatic:
                        populate_automatic_y_limits()
                    else:
                        render_latest_result()

                async def download_plot(file_format: str) -> None:
                    if latest_result is None or result_chart is None:
                        notify("Run a simulation before downloading a plot.", type="warning")
                        return
                    try:
                        selected_label = str(output.value or "Replication timing")
                        spec = _OUTPUT_SPECS[selected_label]
                        normalized_dpi = int(get_export_settings().plot_dpi)
                        pixel_ratio = max(1.0, min(4.0, normalized_dpi / 96.0))
                        capture_format = (
                            "png" if str(file_format).casefold() == "png" else "svg"
                        )
                        capture_options: dict[str, Any] = {"type": capture_format}
                        if capture_format == "png":
                            capture_options["pixelRatio"] = pixel_ratio
                        data_uri = await result_chart.run_chart_method(
                            "getDataURL",
                            capture_options,
                            timeout=10,
                        )
                        artifact = await run.io_bound(
                            build_chart_capture_export,
                            data_uri,
                            file_format,
                            filename=(
                                f"{RSIM_FILE_PREFIX}_{output_file_token(spec.key)}_batch_plot"
                                if combined_preview_active()
                                else _simulation_plot_default_filename(
                                    latest_items[active_dataset_index],
                                    spec.key,
                                )
                            ),
                            dpi=normalized_dpi,
                        )
                        if artifact is None or not artifact.data:
                            raise RuntimeError("Plot export stopped before returning data.")
                    except (RuntimeError, TimeoutError, TypeError, ValueError) as exc:
                        notify(str(exc), type="negative")
                        return
                    cache_artifact(
                        artifact,
                        source="Simulation",
                        kind="plot",
                        detail=selected_label,
                        navigate_to="simulation",
                    )
                    ui.download(artifact.data, artifact.filename, artifact.media_type)

                async def download_raw(controls: RawOutputControls, *, open_in: str | None = None) -> None:
                    if latest_result is None:
                        notify("Run a simulation before downloading raw data.", type="warning")
                        return
                    try:
                        selected_label = str(output.value or "Replication timing")
                        artifact = await run.io_bound(
                            build_raw_export,
                            latest_result,
                            _OUTPUT_SPECS[selected_label].key,
                            str(controls.file_type.value or "TSV"),
                            filename=str(
                                controls.file_name.value
                                or (
                                    f"{RSIM_FILE_PREFIX}_"
                                    f"{output_file_token(_OUTPUT_SPECS[selected_label].key)}"
                                )
                            ),
                            compression=(
                                get_export_settings().raw_compression
                                if controls.compress.value
                                else "none"
                            ),
                        )
                        if artifact is None or not artifact.data:
                            raise RuntimeError("Raw export stopped before returning data.")
                    except Exception as exc:
                        notify(str(exc), type="negative")
                        return
                    if open_in is not None:
                        await open_raw_artifact(artifact, open_in, _navigate, preferred_input=_OUTPUT_SPECS[selected_label].key)
                        return
                    cache_artifact(
                        artifact,
                        source="Simulation",
                        kind="raw",
                        detail=selected_label,
                        navigate_to="simulation",
                    )
                    ui.download(artifact.data, artifact.filename, artifact.media_type)

                async def download_ensemble_raw(
                    item_key: str, kind: str, controls: RawOutputControls,
                    *, open_in: str | None = None,
                ) -> None:
                    item = next((item for item in latest_items if item.key == item_key), None)
                    if item is None:
                        notify("Run a simulation before downloading raw data.", type="warning")
                        return
                    result_snapshot = item.result
                    file_type = str(controls.file_type.value or "TSV")
                    filename = str(controls.file_name.value or "")
                    count = _bounded_profile_count(individual_profile_count.value, _MAX_PREVIEW_PROFILES)
                    compression = get_export_settings().raw_compression if controls.compress.value else "none"
                    controls.download_button.disable()
                    try:
                        if kind == "profiles":
                            if count is None:
                                raise ValueError("Enter a whole number for Profiles shown.")
                            artifact = await run.io_bound(
                                build_individual_profile_archive, result_snapshot, count,
                                file_type, filename=filename,
                            )
                        else:
                            artifact = await run.io_bound(
                                build_standard_deviation_export, result_snapshot,
                                file_type, filename=filename, compression=compression,
                            )
                        if artifact is None or not artifact.data:
                            raise RuntimeError("The export stopped before returning data.")
                        if open_in is not None:
                            await open_raw_artifact(artifact, open_in, _navigate)
                            return
                        # Individual profiles and their ZIP never enter the cache.
                        if kind == "standard_deviation":
                            cache_artifact(
                                artifact, source="Simulation", kind="raw",
                                detail="Replication timing standard deviation", navigate_to="simulation",
                            )
                        ui.download(artifact.data, artifact.filename, artifact.media_type)
                    except Exception as exc:
                        notify(str(exc), type="negative")
                    finally:
                        if not controls.download_button.is_deleted:
                            controls.download_button.enable()

                def batch_raw_artifact(
                    item: _SimulationBatchItem,
                    *,
                    output_key: str,
                    file_name: str,
                    file_type: str,
                    compression: str,
                ) -> Any:
                    return build_raw_export(
                        item.result,
                        output_key,
                        file_type,
                        filename=file_name,
                        compression=compression,
                    )

                def retained_raw_preference(
                    item: _SimulationBatchItem,
                    output_key: str,
                ) -> _SimulationRawPreference:
                    """Update an automatic name without overwriting a manual edit."""

                    automatic_name = _simulation_raw_default_filename(item, output_key)
                    preference = batch_raw_preferences.setdefault(
                        (item.key, output_key),
                        _SimulationRawPreference(
                            file_name=automatic_name,
                            automatic_name=automatic_name,
                        ),
                    )
                    if not preference.customized:
                        preference.file_name = automatic_name
                    preference.automatic_name = automatic_name
                    return preference

                async def download_batch_raw(
                    item: _SimulationBatchItem,
                    controls: RawOutputControls,
                    *, open_in: str | None = None,
                ) -> None:
                    if not any(item is current for current in latest_items):
                        notify("Run a simulation before downloading raw data.", type="warning")
                        return
                    # Snapshot mutable UI values before yielding to a worker;
                    # later output or dataset changes cannot alter this export.
                    selected_label = str(output.value or "Replication timing")
                    output_key = _OUTPUT_SPECS[selected_label].key
                    item_snapshot = item
                    file_name = str(
                        controls.file_name.value
                        or _simulation_raw_default_filename(item_snapshot, output_key)
                    )
                    file_type = str(controls.file_type.value or "TSV")
                    compression = (
                        get_export_settings().raw_compression
                        if bool(controls.compress.value)
                        else "none"
                    )
                    try:
                        artifact = await run.io_bound(
                            batch_raw_artifact,
                            item_snapshot,
                            output_key=output_key,
                            file_name=file_name,
                            file_type=file_type,
                            compression=compression,
                        )
                    except Exception as exc:
                        notify(str(exc), type="negative")
                        return
                    if open_in is not None:
                        await open_raw_artifact(artifact, open_in, _navigate, preferred_input=output_key)
                        return
                    cache_artifact(
                        artifact,
                        source="Simulation",
                        kind="raw",
                        detail=(
                            f"{_simulation_display_name(item_snapshot)} · "
                            f"{selected_label}"
                        ),
                        navigate_to="simulation",
                    )
                    ui.download(artifact.data, artifact.filename, artifact.media_type)

                async def download_all_batch_raw() -> None:
                    items_snapshot = tuple(latest_items)
                    if len(items_snapshot) < 2:
                        return
                    selected_label = str(output.value or "Replication timing")
                    output_key = _OUTPUT_SPECS[selected_label].key
                    enabled_compression = get_export_settings().raw_compression
                    # Rows on unvisited pages use defaults; every edited row
                    # contributes its independently retained values.
                    settings_snapshot: tuple[
                        tuple[_SimulationBatchItem, str, str, str], ...
                    ] = tuple(
                        (
                            item,
                            (
                                preference.file_name
                                or _simulation_raw_default_filename(item, output_key)
                            ),
                            preference.file_type or "TSV",
                            enabled_compression if preference.compress else "none",
                        )
                        for item in items_snapshot
                        for preference in (
                            retained_raw_preference(item, output_key),
                        )
                    )
                    try:
                        statistics_snapshot = statistics_export.snapshot() if statistics_export.surface.visible else None
                        def prepare_archive() -> Any:
                            artifacts = (
                                batch_raw_artifact(
                                    item,
                                    output_key=output_key,
                                    file_name=file_name,
                                    file_type=file_type,
                                    compression=compression,
                                )
                                for item, file_name, file_type, compression in settings_snapshot
                            )
                            if statistics_snapshot is not None:
                                artifacts = chain(artifacts, (statistics_export.artifact(statistics_snapshot),))
                            return build_artifact_archive(
                                artifacts,
                                filename=(
                                    f"{RSIM_FILE_PREFIX}_"
                                    f"{output_file_token(_OUTPUT_SPECS[selected_label].key)}"
                                    "_batch.zip"
                                ),
                                max_source_bytes=DEFAULT_BATCH_ARCHIVE_SOURCE_BYTES,
                            )

                        archive = await run.io_bound(prepare_archive)
                    except Exception as exc:
                        notify(str(exc), type="negative")
                        return
                    cache_artifact(
                        archive,
                        source="Simulation",
                        kind="raw",
                        detail=f"{len(items_snapshot):,} datasets · {selected_label}",
                        navigate_to="simulation",
                    )
                    ui.download(archive.data, archive.filename, archive.media_type)

                def render_latest_result() -> None:
                    nonlocal result_chart
                    plot_settings.surface.visible = bool(show_plot.value)
                    if plot_settings._statistics_export is not None:
                        plot_settings._statistics_export.refresh()
                    if latest_result is None:
                        return
                    if not show_plot.value:
                        result_chart = None
                        result_slot.clear()
                        with result_slot:
                            preview_plot('Plot hidden', kind='simulation')
                        return
                    if not plotting_settings_ready(
                        plot_settings,
                        plot_type=active_plot_type(),
                        allow_empty_y_limits=y_limits_automatic,
                    ):
                        return
                    profile_count = selected_profile_count()
                    if (
                        bool(show_individual_profiles.value)
                        and available_profile_count() > 0
                        and profile_count is None
                        and not combined_preview_active()
                    ):
                        return
                    result_chart = None
                    result_slot.clear()
                    with result_slot:
                        try:
                            if combined_preview_active():
                                options = _combined_result_plot_options(
                                    latest_items,
                                    output_label=str(
                                        output.value or "Replication timing"
                                    ),
                                    title=str(title.value or ""),
                                    x_axis_label=str(x_axis_label.value or ""),
                                    y_axis_label=str(y_axis_label.value or ""),
                                    colors=batch_colors,
                                    line_types=batch_line_types,
                                    curve_names=batch_names,
                                    line_type=str(plot_settings.line_type.value or "solid"),
                                    y_limits=current_y_limits(),
                                    x_limits=axis_limits_from_controls(plot_settings, "x"),
                                    show_legend=bool(show_legend.value),
                                    legend_position=str(
                                        legend_position.value or "Top right"
                                    ),
                                    line_width=float(
                                        _number(line_width, "Thickness")
                                    ),
                                    show_grid=bool(show_grid.value),
                                    log_x=bool(log_x.value),
                                    invert_x=bool(invert_x.value),
                                    log_y=bool(log_y.value),
                                    invert_y=bool(invert_y.value),
                                )
                                result_chart = render_preview_chart(
                                    options,
                                    title=str(title.value or output.value or "Simulation"),
                                    marker="simulation-result-plot",
                                    on_download=download_plot,
                                )
                            else:
                                result_chart = _render_result_plot(
                                    latest_result,
                                    x_limits=axis_limits_from_controls(plot_settings, "x"),
                                    z_limits=axis_limits_from_controls(plot_settings, "z"),
                                    output_label=str(output.value or "Replication timing"),
                                    title=str(title.value or ""),
                                    series_label=str(output.value or "Replication timing"),
                                    x_axis_label=str(x_axis_label.value or ""),
                                    y_axis_label=str(y_axis_label.value or ""),
                                    color=str(line_or_bar_color.value or "#b03412"),
                                    palette=str(heatmap_palette.value or "viridis"),
                                    view_mode=str(view.value or "Default"),
                                    histogram_bins=_integer(histogram_bins, "Histogram bins"),
                                    density=bool(density.value),
                                    show_legend=bool(show_legend.value),
                                    legend_position=str(
                                        legend_position.value or "Top right"
                                    ),
                                    show_histogram_statistics=bool(
                                        show_histogram_statistics.value
                                    ),
                                    histogram_statistics_position=str(
                                        histogram_statistics_position.value or "Bottom right"
                                    ),
                                    histogram_bin_size=(
                                        float(plot_settings.histogram_bin_size.value)
                                        if plot_settings.histogram_bin_size.value is not None
                                        else None
                                    ),
                                    histogram_x_limits=histogram_x_limits_from_controls(
                                        plot_settings,
                                        log_x=bool(log_x.value),
                                    ),
                                    histogram_touching=bool(
                                        plot_settings.histogram_touching.value
                                    ),
                                    kde_bandwidth=float(plot_settings.kde_bandwidth.value or 1.0),
                                    histogram_grouped=bool(plot_settings.histogram_grouped.value),
                                    histogram_boundary=str(
                                        plot_settings.histogram_boundary.value or "left"
                                    ),
                                    show_individual_profiles=bool(
                                        show_individual_profiles.value
                                    ),
                                    individual_profile_count=profile_count or 1,
                                    show_standard_deviation=bool(
                                        show_standard_deviation.value
                                    ),
                                    line_width=float(
                                        _number(line_width, "Thickness")
                                    ),
                                    line_type=str(plot_settings.line_type.value or "solid"),
                                    show_markers=False,
                                    show_grid=bool(show_grid.value),
                                    show_colorbar=bool(colorbar.value),
                                    log_y=bool(log_y.value),
                                    log_z=bool(log_z.value),
                                    invert_y=bool(invert_y.value),
                                    log_x=bool(log_x.value),
                                    invert_x=bool(invert_x.value),
                                    invert_z=bool(invert_z.value),
                                    # Automatic fields contain outward-rounded bounds
                                    # (or a fixed physical domain); applying the same
                                    # values keeps the editor and preview in agreement.
                                    y_limits=current_y_limits(),
                                    on_download=download_plot,
                                )
                            selected = str(output.value or 'Replication timing')
                            spec = _output_spec(selected, latest_result)
                            if spec.kind in {'space', 'time', 'xt'} and str(view.value or 'Default') in {'Default', 'Circular'}:
                                shown = tuple(item.result for item in latest_items) if combined_preview_active() else (latest_result,)
                                selected_outputs = [select_result_output(result, spec.key) for result in shown]
                                temporal_axes = ([input_preview_result(result, spec.key).time_grid for result in shown] if spec.kind == 'xt' else
                                                 [selected.x for selected in selected_outputs] if spec.kind == 'time' else
                                                 [result.time_grid for result in shown])
                                guide_time_end = max((float(axis[-1]) for axis in temporal_axes if axis is not None and len(axis)),
                                                     default=float(time_window.value or 0.))
                                add_simulation_filter_guides(
                                    result_chart.options, shown, plot_settings, kind=spec.kind,
                                    time_end=guide_time_end,
                                )
                                result_chart.update()
                            result_chart.attach_settings(plot_settings)
                        except (RuntimeError, TypeError, ValueError) as exc:
                            result_chart = None
                            notify(str(exc), type="negative")
                            preview_plot(
                                (
                                    "No inter-origin distances were recorded; at least two "
                                    "origins must fire in a realization."
                                    if _OUTPUT_SPECS[
                                        str(output.value or "Replication timing")
                                    ].key
                                    == "inter_origin_distances"
                                    else "Output unavailable"
                                ),
                                kind="simulation",
                            )

                def simulation_request_settings(selected_filters, selected_filter_exports):
                    return dict(
                        resolution_x=_number(resolution_x, "Spatial resolution"),
                        source_resolution_x=1.0,
                        resolution_t=_number(resolution_t, "Time step"),
                        region_start_kb=_number(region_start, "Region start"),
                        replicates=_integer(replicates, "Simulations (replicates)"),
                        geometry='periodic' if periodic.value else 'linear',
                        whole_region=bool(whole_region.value),
                        data_filters=selected_filters,
                        export_filters=selected_filter_exports,
                        time_window=_number(time_window, "Time window"),
                        max_replication_time=_number(
                            max_replication_time,
                            "Maximum replication time",
                        ),
                        missing_values=str(missing_values.value),
                        seed=_integer(seed, "Seed") if fixed_seed.value else None,
                        stall_rate_per_kb=_number(stall_rate, "Stall rate (stalls/kb)") if fork_stalling.value else 0.0,
                        stall_shared_directions=bool(shared_stalls.value) if fork_stalling.value else False,
                        fork_speed_std=_number(fork_speed_std, "Standard deviation") if stochastic_movement.value else 0.0,
                        stall_duration=(
                            math.inf
                            if not fork_stalling.value or not stall_recovery.value
                            else _number(stall_duration, "Stall duration")
                        ),
                        firing_factors=(
                            float(_integer(pool_size, "Firing factor"))
                            if finite_pool.value
                            else None
                        ),
                        firing_factor_recycle=(
                            _number(recycle_rate, "Firing-factor recycle rate")
                            if finite_pool.value
                            else 0.0
                        ),
                        firing_factor_fork_coupling=(
                            bool(factor_coupling.value) if finite_pool.value else False
                        ),
                        track_time_outputs=bool(track_time_outputs.value),
                        track_detailed_outputs=bool(track_detailed_outputs.value),
                        retained_profile_count=_integer(individual_profile_count, "Profiles shown"),
                    )

                async def download_simulation_code() -> None:
                    from app.code_export import CodeInputs, CodeSnapshot, build_python, filter_code_settings, placeholder_upload, statistics_code_settings

                    if running or any(control.loading for control in filter_controls.values()):
                        notify("Wait for the data operation to finish before exporting code.", type="warning")
                        return
                    try:
                        inputs = CodeInputs()
                        selected_filters = {key: filter_code_settings(control) for key, control in filter_controls.items()
                                            if filter_relative.value in {key, 'both'} and control.controls['enabled'].value}
                        exports = frozenset(key for key in selected_filters if filter_controls[key].controls['export'].value)
                        source_options = {}
                        for key, source, label in (("initiation_rate", firing_source, "Initiation rate"),
                                                   ("fork_speed", speed_source, "Fork speed")):
                            _validate_simulation_batch_selection(source, label=label)
                            mode = str(source.mode.value)
                            options = {"mode": mode, "form": str(source.form.value)}
                            if mode == "scalar":
                                options["values"] = [_number(source.constant, label)]
                            elif mode == "constants":
                                options["values"] = list(_read_simulation_constant_controls(
                                    source.constants, label=label, positive=key == "fork_speed"))
                            else:
                                inspections = source.inspections or (placeholder_upload(key, form=options["form"]),)
                                options["files"] = [dict(source=inputs.encode(inspection),
                                    **source.selections.get(inspection.name, source.selection)) for inspection in inspections]
                            source_options[key] = options
                        preferences = {}
                        for (dataset, key), preference in {**batch_raw_preferences, **ensemble_raw_preferences}.items():
                            preferences.setdefault(dataset, {})[key] = dict(
                                file_name=preference.file_name, file_type=preference.file_type,
                                compress=preference.compress,
                            )
                            if key.endswith("_filter"):
                                preferences[dataset][key]["layout"] = filter_layouts.get((dataset, key.removesuffix("_filter")), "positions")
                        statistics = statistics_code_settings(plot_settings)
                        if statistics is not None and not combined_preview_active():
                            statistics["dataset_index"] = active_dataset_index
                        snapshot = CodeSnapshot("simulation", dict(
                            inputs=source_options,
                            settings=inputs.encode(simulation_request_settings(selected_filters, exports)),
                            region_end_kb=_number(region_end, "Region end"), automatic_region_end=automatic_region_end,
                            combination_mode=simulation_combination_mode.get("value", "pairwise")
                                if any(source.mode.value == "constants" for source in (firing_source, speed_source)) else "pairwise",
                            export=dict(output=_OUTPUT_SPECS[str(output.value or "Replication timing")].key,
                                filename=str(raw_controls.file_name.value or "") if len(latest_items) <= 1 else "",
                                file_type=str(raw_controls.file_type.value), compress=bool(raw_controls.compress.value),
                                compression=get_export_settings().raw_compression,
                                statistics=statistics,
                                name_style=_simulation_name_style(), per_dataset=preferences,
                                standard_deviation=bool(show_standard_deviation.value),
                                individual_profiles=bool(show_individual_profiles.value),
                                profile_count=_integer(individual_profile_count, "Profiles shown")),
                        ), tuple(inputs.files.items()))
                        artifact = await run.io_bound(build_python, snapshot)
                        ui.download(artifact.data, artifact.filename, artifact.media_type)
                    except (OSError, RuntimeError, TypeError, ValueError) as exc:
                        notify(str(exc), type="negative")

                async def simulate() -> None:
                    nonlocal active_cancel, active_dataset_index, latest_items
                    nonlocal latest_result, result_chart, running
                    if running:
                        if active_cancel is not None and not active_cancel.is_set():
                            page_activity.stop("simulation")
                            active_cancel.set()
                            status.text = "Stopping simulation..."
                            simulate_button.disable()
                        return
                    if any(control.loading for control in filter_controls.values()):
                        notify('Wait for the mask and modulation uploads to finish.', type='warning')
                        return
                    selected_filters = {key: control.snapshot() for key, control in filter_controls.items()
                                        if filter_relative.value in {key, 'both'} and control.controls['enabled'].value}
                    selected_filter_exports = frozenset(key for key in selected_filters if filter_controls[key].controls['export'].value)

                    cancel_event = Event()

                    def check_cancelled() -> None:
                        if cancel_event.is_set():
                            raise CancelledError("Simulation stopped.")

                    frozen_controls = [
                        control
                        for control in (
                            firing_source.mode,
                            firing_source.constant,
                            *firing_source.constants,
                            *firing_source.constant_actions,
                            *firing_source.selection_controls,
                            speed_source.mode,
                            speed_source.constant,
                            *speed_source.constants,
                            *speed_source.constant_actions,
                            *speed_source.selection_controls,
                            resolution_x,
                            resolution_t,
                            region_start,
                            region_end,
                            replicates,
                            individual_profile_count,
                            whole_region,
                            periodic,
                            filter_relative,
                            missing_values,
                            fixed_seed,
                            seed,
                            time_window,
                            max_replication_time,
                            track_time_outputs,
                            track_detailed_outputs,
                            stall_rate,
                            fork_stalling,
                            stochastic_movement,
                            fork_speed_std,
                            stall_recovery,
                            stall_duration,
                            shared_stalls,
                            finite_pool,
                            pool_size,
                            recycle_rate,
                            factor_coupling,
                        )
                        if bool(getattr(control, "enabled", False))
                    ]

                    def cleanup_simulation() -> None:
                        nonlocal active_cancel, running
                        running = False
                        active_cancel = None
                        set_data_upload_locked(firing_source.uploader, False)
                        set_data_upload_locked(speed_source.uploader, False)
                        for control in frozen_controls:
                            control.enable()
                        for control in filter_controls.values():
                            control.set_locked(False)
                        sync_region_to_profiles()
                        simulate_button.set_text("Simulate")
                        simulate_button.set_icon(workflow_action_icon("simulation"))
                        simulate_button.enable()
                        # Input validation can return before the worker's finally block.
                        page_activity.finish(
                            "simulation", page_run, success=completed_successfully
                        )

                    running = True
                    active_cancel = cancel_event
                    page_run = page_activity.begin("simulation", cancel=cancel_event.set)
                    completed_successfully = False
                    set_data_upload_locked(firing_source.uploader, True)
                    set_data_upload_locked(speed_source.uploader, True)
                    for control in frozen_controls:
                        control.disable()
                    for control in filter_controls.values():
                        control.set_locked(True)
                    simulate_button.set_text("Stop")
                    simulate_button.set_icon("stop")
                    progress.visible = True
                    progress.value = 0.01
                    status.text = "Reading inputs..."

                    try:
                        missing_policy = str(missing_values.value)

                        def prepare_simulation_sources() -> tuple[
                            tuple[float | np.ndarray, ...],
                            tuple[float | np.ndarray, ...],
                        ]:
                            _validate_simulation_batch_selection(
                                firing_source,
                                label="Initiation rate",
                            )
                            _validate_simulation_batch_selection(
                                speed_source,
                                label="Fork speed",
                            )
                            return (
                                _materialize_simulation_sources(
                                    firing_source,
                                    label="Initiation rate",
                                    missing_policy='ignore' if 'initiation_rate' in selected_filters else missing_policy,
                                ),
                                _materialize_simulation_sources(
                                    speed_source,
                                    label="Fork speed",
                                    missing_policy='ignore' if 'fork_speed' in selected_filters else missing_policy,
                                ),
                            )

                        preparation = asyncio.create_task(
                            run.io_bound(prepare_simulation_sources)
                        )
                        while not preparation.done():
                            if cancel_event.is_set():
                                preparation.cancel()
                                try:
                                    await preparation
                                except asyncio.CancelledError:
                                    pass
                                check_cancelled()
                            await asyncio.sleep(0.04)
                        source_values = await preparation
                        check_cancelled()
                        if source_values is None:
                            raise RuntimeError(
                                "Input preparation stopped before returning simulation data."
                            )
                        firing_values, speed_values = source_values
                        prepared_sources = (
                            _prepare_simulation_source(
                                firing_source,
                                firing_values,
                                label="Initiation rate",
                                units="min⁻¹ per bin",
                            ),
                            _prepare_simulation_source(
                                speed_source,
                                speed_values,
                                label="Fork speed",
                                units="kb/min",
                            ),
                        )
                        has_multiple_constants = any(
                            str(source.mode.value) == "constants"
                            for source in (firing_source, speed_source)
                        )
                        cases = _plan_simulation_cases(
                            prepared_sources,
                            combination_mode=(
                                simulation_combination_mode.get(
                                    "value",
                                    "pairwise",
                                )
                                if has_multiple_constants
                                else "pairwise"
                            ),
                        )
                        resolution_x_value = _number(resolution_x, "Spatial resolution")
                        region_start_value = _number(region_start, "Region start")
                        configured_region_end = _number(region_end, "Region end")
                        use_selected_end = (
                            configured_region_end != automatic_region_end
                            or region_start_value != 0.0
                        )
                        paired_values: list[
                            tuple[float | np.ndarray, float | np.ndarray, float]
                        ] = []
                        for case in cases:
                            dataset_name = case.identity.display(_simulation_name_style())
                            initiation_rate = prepared_sources[0].values[
                                case.item_indices[0]
                            ]
                            fork_speed = prepared_sources[1].values[
                                case.item_indices[1]
                            ]
                            paired_sizes = {
                                size
                                for size in (
                                    _simulation_spatial_size(initiation_rate),
                                    _simulation_spatial_size(fork_speed),
                                )
                                if size is not None
                            }
                            if len(paired_sizes) > 1:
                                sizes = ", ".join(
                                    f"{size:,}" for size in sorted(paired_sizes)
                                )
                                raise ValueError(
                                    f"{dataset_name}: initiation-rate and fork-speed spatial "
                                    f"axes disagree ({sizes} bins)."
                                )
                            region_end_value = (
                                region_start_value
                                + next(iter(paired_sizes))
                                if paired_sizes and not use_selected_end
                                else configured_region_end
                            )
                            paired_values.append(
                                (initiation_rate, fork_speed, region_end_value)
                            )
                        shared_request = simulation_request_settings(selected_filters, selected_filter_exports)
                        requests = tuple(
                            ConstantSimulationRequest(
                                initiation_rate=initiation_rate,
                                fork_speed=fork_speed,
                                region_end_kb=request_region_end,
                                **shared_request,
                            )
                            for initiation_rate, fork_speed, request_region_end in paired_values
                        )
                        # Validate the entire batch before the first scientific
                        # worker starts, so a mismatch cannot leave a partially
                        # published set of results.
                        prepared_requests = []
                        for case, request in zip(
                            cases,
                            requests,
                            strict=True,
                        ):
                            try:
                                validated = await run.io_bound(validate_constant_simulation, request, check_cancelled)
                                if validated is None:
                                    raise RuntimeError('Simulation validation stopped before returning.')
                                prepared_requests.append(validated.request if request.data_filters else request)
                            except (TypeError, ValueError) as exc:
                                raise ValueError(
                                    f"{case.identity.display(_simulation_name_style())}: "
                                    f"{exc}"
                                ) from exc
                        requests = tuple(prepared_requests)
                        check_cancelled()
                    except CancelledError:
                        progress.visible = False
                        status.text = "Simulation stopped"
                        stats_container.visible = bool(latest_items)
                        if not latest_items:
                            result_slot.clear()
                            with result_slot:
                                preview_plot("Simulation stopped", kind="simulation")
                        cleanup_simulation()
                        return
                    except Exception as exc:
                        notify(str(exc), type="negative")
                        status.text = "Check the settings"
                        progress.visible = False
                        cleanup_simulation()
                        return

                    progress.value = 0.02
                    status.text = "Preparing simulation..."
                    try:
                        staged_items: list[_SimulationBatchItem] = []
                        batch_size = len(requests)
                        for index, (case, request) in enumerate(
                            zip(cases, requests, strict=True)
                        ):
                            dataset_suffix = (
                                f" · Dataset {index + 1}/{batch_size}" if batch_size > 1 else ""
                            )
                            check_cancelled()
                            progress_state = _ProgressState()

                            def report_batch_progress(
                                value: float,
                                message: str,
                                *,
                                dataset_index: int = index,
                                suffix: str = dataset_suffix,
                                state: _ProgressState = progress_state,
                            ) -> None:
                                state.report(
                                    (dataset_index + max(0.0, min(1.0, float(value))))
                                    / batch_size,
                                    f"{message}{suffix}",
                                )

                            status.text = (
                                f"Preparing simulation...{dataset_suffix}"
                            )
                            worker = asyncio.create_task(
                                run.io_bound(
                                    run_constant_simulation,
                                    request,
                                    report_batch_progress,
                                    check_cancelled,
                                )
                            )
                            while not worker.done():
                                if cancel_event.is_set():
                                    status.text = "Stopping simulation..."
                                else:
                                    value, message = progress_state.snapshot()
                                    progress.value = value
                                    status.text = message
                                await asyncio.sleep(0.08)
                            result = await worker
                            if result is None:
                                check_cancelled()
                                raise RuntimeError(
                                    "The simulation worker stopped before returning data."
                                )
                            result = add_simulation_input_previews(result, request)
                            result = limit_replication_profiles(result, request.retained_profile_count)
                            staged_items.append(
                                _SimulationBatchItem(
                                    key=case.key,
                                    identity=case.identity,
                                    result=result,
                                    full_export_stem=case.full_export_stem,
                                    short_export_stem=case.short_export_stem,
                                )
                            )

                        # Commit only after every worker succeeds. Until here the
                        # last complete preview remains live and downloadable.
                        latest_items = tuple(staged_items)
                        retained_keys = {item.key for item in latest_items}
                        for preference_key in tuple(batch_raw_preferences):
                            if preference_key[0] not in retained_keys:
                                batch_raw_preferences.pop(preference_key, None)
                        for preference_key in tuple(ensemble_raw_preferences):
                            if preference_key[0] not in retained_keys:
                                ensemble_raw_preferences.pop(preference_key, None)
                        for color_key in tuple(batch_colors):
                            if color_key not in retained_keys:
                                batch_colors.pop(color_key, None)
                        for type_key in tuple(batch_line_types):
                            if type_key not in retained_keys:
                                batch_line_types.pop(type_key, None)
                        for name_key in tuple(batch_names):
                            if name_key not in retained_keys:
                                batch_names.pop(name_key, None)
                        batch_raw_page["value"] = 1
                        batch_navigation_page["value"] = 1
                        active_dataset_index = 0
                        latest_result = latest_items[0].result
                        retained_count = available_profile_count()
                        if retained_count:
                            displayed_count = _bounded_profile_count(
                                individual_profile_count.value, retained_count,
                            )
                            if displayed_count is not None:
                                individual_profile_count.set_value(displayed_count)
                        result_chart = None
                        for index, item in enumerate(
                            latest_items[:_MAX_COMBINED_PREVIEW_DATASETS]
                        ):
                            batch_colors.setdefault(
                                item.key,
                                _BATCH_COLORS[index % len(_BATCH_COLORS)],
                            )
                        progress.value = 1.0
                        elapsed = sum(item.result.elapsed_seconds for item in latest_items)
                        completion_status = (
                            f"Complete in {elapsed:.2f} s"
                            if len(latest_items) == 1
                            else f"{len(latest_items):,} datasets complete in {elapsed:.2f} s"
                        )
                        # Keep the public readiness signal behind every result
                        # surface update. Otherwise a fast click (and simulated
                        # users) can observe "datasets complete" while the
                        # refreshable raw rows are still being rebuilt.
                        status.text = "Preparing results..."
                        replicate_stats.text = f"{result.replicates:,}"
                        resolution_stats.text = (
                            f"{request.resolution_x:g} kb/bin · {request.resolution_t:g} min/bin"
                        )
                        region_stats.text = (
                            f"{request.region_start_kb:g}–{request.region_end_kb:g} kb"
                        )
                        geometry_stats.text = result.geometry.title()
                        latest_result = latest_items[0].result
                        sync_output_options()
                        sync_single_raw_filename()
                        truncated = [
                            item
                            for item in latest_items
                            if item.result.inter_origin_distances_truncated
                        ]
                        if truncated:
                            notify(
                                "Inter-origin distance retention was truncated for "
                                "some datasets because it exceeded the local data limit.",
                                type="warning",
                            )
                        if any(item.result.replicon_lengths_truncated for item in latest_items):
                            notify(
                                "Replicon length retention was truncated because it exceeded the local data limit.",
                                type="warning",
                            )
                        for warning in dict.fromkeys(message for item in latest_items if item.result.filter_state is not None
                                                     for message in item.result.filter_state.warnings):
                            notify(warning, type='warning')
                        if len(latest_items) > 1:
                            presentation.value = "Combined"
                            presentation.update()
                        sync_batch_presentation(redraw=False)
                        sync_batch_surfaces()
                        update_active_stats()
                        histogram_bin_sync.sync_to_data(redraw=False)
                        sync_view_controls()
                        populate_automatic_y_limits(redraw=False)
                        stats_container.visible = True
                        render_latest_result()
                        # Cached results retain aggregate statistics only. The
                        # displayed individual profiles belong to this page.
                        cached_items = tuple(
                            replace(item, result=limit_replication_profiles(item.result, 0))
                            for item in latest_items
                        )
                        cached_payload: Any = (
                            cached_items[0].result
                            if len(cached_items) == 1
                            else _SimulationBatchPayload(cached_items)
                        )
                        cache_workflow(
                            cached_payload,
                            name=(
                                "Simulation result"
                                if len(latest_items) == 1
                                else f"Simulation batch ({len(latest_items):,})"
                            ),
                            source="Simulation",
                            detail=(
                                f"{latest_result.replicates:,} replicates | "
                                f"{requests[0].region_start_kb:g}-"
                                f"{requests[0].region_end_kb:g} kb | "
                                f"{latest_result.geometry.title()} | "
                                f"{len(latest_items):,} dataset(s)"
                            ),
                            navigate_to="simulation",
                        )
                        status.text = completion_status
                        completed_successfully = True
                    except CancelledError:
                        progress.visible = False
                        status.text = "Simulation stopped"
                        stats_container.visible = bool(latest_items)
                        if not latest_items:
                            result_slot.clear()
                            with result_slot:
                                preview_plot(
                                    "Simulation stopped",
                                    kind="simulation",
                                )
                    except Exception as exc:  # keep malformed local jobs inside the preview
                        progress.visible = False
                        status.text = "Simulation could not be completed"
                        stats_container.visible = bool(latest_items)
                        notify(str(exc) or type(exc).__name__, type="negative")
                        if not latest_items:
                            result_slot.clear()
                            with result_slot:
                                preview_plot(
                                    "No result",
                                    kind="simulation",
                                )
                    finally:
                        cleanup_simulation()

                with action_dock(on_code=download_simulation_code):
                    simulate_button = (
                        workflow_action_button("Simulate", symbol="simulation", on_click=simulate)
                        .props("no-caps")
                        .classes("dn-action-primary")
                        .mark("action-simulate")
                    )

        with ui.column().classes("dn-preview-stack"):
            with panel(extra_classes="dn-preview-card"):
                with (
                    ui.element("div")
                    .classes("dn-preview-heading-row")
                    .mark("simulation-preview-header-row")
                ):
                    panel_title("Preview")
                with (
                    ui.element("div")
                    .classes("dn-stat-grid dn-simulation-summary-grid")
                    .mark("simulation-preview-output-bar")
                ):
                    with (
                        ui.element("div")
                        .classes("dn-stat dn-simulation-output-stat")
                        .mark("simulation-output-stat")
                    ):
                        ui.label("Output").classes("dn-stat-label")
                        output = (
                            ui.select(
                                [
                                    *list(_STANDARD_OUTPUTS),
                                    *(
                                        [
                                            label
                                            for label, spec in _TIME_TRACKED_OUTPUTS.items()
                                            if spec.key in latest_result.outputs
                                        ]
                                        if latest_result is not None
                                        and getattr(latest_result, "time_outputs", False)
                                        else []
                                    ),
                                    *(
                                        [
                                            label
                                            for label, spec in _SPACETIME_OUTPUTS.items()
                                            if spec.key in latest_result.outputs
                                        ]
                                        if latest_result is not None
                                        and latest_result.detailed_outputs
                                        else []
                                    ),
                                ],
                                value="Replication timing",
                            )
                            .props('dense borderless options-dense aria-label="Preview output"')
                            .classes("dn-field dn-simulation-output-select")
                            .mark("simulation-output")
                        )
                    with (
                        ui.element("div")
                        .classes("dn-simulation-run-stats")
                        .mark("simulation-stats") as stats_container
                    ):
                        stats = stat_strip(
                            (
                                ("Simulations", ""),
                                ("Resolution", ""),
                                ("Region", ""),
                                ("Geometry", ""),
                            )
                        )
                replicate_stats = stats["Simulations"]
                resolution_stats = stats["Resolution"]
                region_stats = stats["Region"]
                geometry_stats = stats["Geometry"]
                stats_container.visible = False
                with (
                    ui.row()
                    .classes("w-full items-center dn-setting-inline")
                    .mark("simulation-batch-navigation") as batch_navigation
                ):
                    presentation = (
                        ui.select(
                            ["Combined", "Individual"],
                            value="Combined",
                            label="Layout",
                        )
                        .props("dense options-dense")
                        .classes("dn-field dn-field-compact")
                        .mark("simulation-presentation")
                    )
                    with (
                        ui.element("div")
                        .classes("grow min-w-0")
                        .mark("simulation-dataset-selector") as dataset_selector_surface
                    ):

                        @ui.refreshable
                        def render_dataset_selector() -> None:
                            if (
                                len(latest_items) < 2
                                or str(presentation.value or "Combined") != "Individual"
                            ):
                                return
                            selected_index = min(
                                max(0, active_dataset_index),
                                len(latest_items) - 1,
                            )
                            selected_page = (
                                selected_index // _BATCH_NAV_PAGE_SIZE
                            ) + 1
                            page_count = max(
                                1,
                                math.ceil(
                                    len(latest_items) / _BATCH_NAV_PAGE_SIZE
                                ),
                            )
                            page = min(
                                max(
                                    1,
                                    int(
                                        batch_navigation_page.get(
                                            "value",
                                            selected_page,
                                        )
                                    ),
                                ),
                                page_count,
                            )
                            if not (
                                (page - 1) * _BATCH_NAV_PAGE_SIZE
                                <= selected_index
                                < page * _BATCH_NAV_PAGE_SIZE
                            ):
                                page = selected_page
                            batch_navigation_page["value"] = page
                            start = (page - 1) * _BATCH_NAV_PAGE_SIZE
                            end = min(
                                len(latest_items),
                                start + _BATCH_NAV_PAGE_SIZE,
                            )
                            with (
                                ui.row()
                                .classes("w-full items-end dn-dataset-selector-row")
                                .mark("simulation-dataset-selector-row")
                            ):
                                if page_count > 1:
                                    page_control = (
                                        ui.number(
                                            label="Page",
                                            value=page,
                                            min=1,
                                            max=page_count,
                                            step=1,
                                        )
                                        .props("dense")
                                        .classes("dn-field dn-dataset-page-select")
                                        .mark("simulation-dataset-page")
                                    )

                                    def page_changed() -> None:
                                        try:
                                            requested = int(page_control.value)
                                        except (TypeError, ValueError):
                                            return
                                        requested = min(max(1, requested), page_count)
                                        batch_navigation_page["value"] = requested
                                        select_dataset(
                                            latest_items[
                                                (requested - 1) * _BATCH_NAV_PAGE_SIZE
                                            ].key
                                        )
                                        render_dataset_selector.refresh()

                                    page_control.on_value_change(page_changed)
                                options = {
                                    item.key: _simulation_display_name(item)
                                    for item in latest_items[start:end]
                                }
                                dataset_select = (
                                    ui.select(
                                        options,
                                        value=latest_items[selected_index].key,
                                        label="Dataset",
                                    )
                                    .props(
                                        "dense options-dense use-input "
                                        "input-debounce=0"
                                    )
                                    .classes("dn-field dn-dataset-select")
                                    .mark("simulation-dataset-select")
                                )
                                dataset_select.on_value_change(
                                    lambda event: select_dataset(str(event.value))
                                )

                        render_dataset_selector()
                batch_navigation.visible = len(latest_items) > 1
                dataset_selector_surface.visible = (
                    len(latest_items) > 1 and str(presentation.value) == "Individual"
                )
                progress = (
                    ui.linear_progress(value=0.0, show_value=False)
                    .props("rounded instant-feedback")
                    .classes("dn-simulation-progress")
                    .mark("simulation-progress")
                )
                progress.visible = False
                with ui.element("div").classes("dn-run-status-line").mark("simulation-status-line"):
                    status = ui.label("Ready").classes("dn-muted").mark("simulation-status")
                with (
                    ui.element("div")
                    .classes("w-full")
                    .mark("simulation-result-container") as result_slot
                ):
                    preview_plot(
                        "No result yet",
                        kind="simulation",
                    )

                plot_settings = preview_settings_controls(
                    filter_labels=FILTER_LABELS,
                    initial_title="Replication timing",
                    initial_x_label="Position (kb)",
                    initial_y_label="Replication timing",
                    on_reset_y_limits=lambda: populate_automatic_y_limits(),
                    markers=preview_settings_markers(
                        "simulation",
                        overrides={
                            "primary_color": "simulation-line-or-bar-color",
                            "secondary_color": "simulation-second-curve-color",
                        },
                    ),
                )
                view = plot_settings.view
                title = plot_settings.title
                x_axis_label = plot_settings.x_axis_label
                y_axis_label = plot_settings.y_axis_label
                line_or_bar_color = plot_settings.primary_color
                heatmap_palette = plot_settings.heatmap_palette
                log_x = plot_settings.log_x
                invert_x = plot_settings.invert_x
                log_z = plot_settings.log_z
                invert_z = plot_settings.invert_z
                line_width = plot_settings.line_width
                histogram_bins = plot_settings.histogram_bins
                show_individual_profiles = plot_settings.show_individual_profiles
                individual_profile_count = plot_settings.individual_profile_count
                show_standard_deviation = plot_settings.show_standard_deviation
                show_histogram_statistics = plot_settings.show_histogram_statistics
                histogram_statistics_position = plot_settings.histogram_statistics_position
                show_legend = plot_settings.show_legend
                legend_position = plot_settings.legend_position
                show_grid = plot_settings.show_grid
                density = plot_settings.density
                colorbar = plot_settings.colorbar
                log_y = plot_settings.log_y
                invert_y = plot_settings.invert_y
                y_minimum = plot_settings.y_minimum
                y_maximum = plot_settings.y_maximum
                with (
                    ui.element("div")
                    .classes("w-full")
                    .mark("simulation-dataset-color-settings") as batch_color_container
                ):
                    with expansion(
                        "Curves",
                        icon="palette",
                        extra_classes="dn-preview-settings-section",
                    ):

                        @ui.refreshable
                        def render_batch_colors() -> None:
                            if not combined_preview_active():
                                return
                            with ui.element("div").classes("dn-preview-settings-grid"):
                                for index, item in enumerate(latest_items):
                                    default_color = _BATCH_COLORS[
                                        index % len(_BATCH_COLORS)
                                    ]
                                    batch_colors.setdefault(item.key, default_color)
                                    name_control = (
                                        ui.input("Name", value=batch_names.get(item.key, _simulation_display_name(item)))
                                        .props("dense").classes("dn-field").style("grid-column: 1 / -1")
                                        .mark(f"simulation-dataset-name-{index + 1}")
                                    )

                                    def name_changed(
                                        _event: Any = None, *,
                                        item_key: str = item.key, control: Any = name_control,
                                    ) -> None:
                                        value = str(control.value or "").strip()
                                        if value:
                                            batch_names[item_key] = value
                                        else:
                                            batch_names.pop(item_key, None)
                                        render_latest_result()

                                    name_control.on_value_change(name_changed)
                                    color_surface, color_control = compact_color_input(
                                        "Color",
                                        batch_colors[item.key],
                                        marker=(
                                            f"simulation-dataset-color-{index + 1}"
                                        ),
                                    )
                                    curve_type = ui.select(
                                        CURVE_TYPES,
                                        value=batch_line_types.get(
                                            item.key, str(plot_settings.line_type.value or "solid")
                                        ),
                                        label="Type",
                                    ).props("dense options-dense").classes("dn-field").mark(
                                        f"simulation-dataset-curve-type-{index + 1}"
                                    )

                                    def type_changed(
                                        _event: Any = None, *,
                                        item_key: str = item.key, control: Any = curve_type,
                                    ) -> None:
                                        batch_line_types[item_key] = str(control.value or "solid")
                                        render_latest_result()

                                    curve_type.on_value_change(type_changed)

                                    def color_changed(
                                        _event: Any = None,
                                        *,
                                        item_key: str = item.key,
                                        control: Any = color_control,
                                    ) -> None:
                                        batch_colors[item_key] = str(
                                            control.value or "#b03412"
                                        )
                                        render_latest_result()

                                    color_control.on_value_change(color_changed)

                        render_batch_colors()
                batch_color_container.visible = bool(
                    len(latest_items) > 1
                    and str(presentation.value or "Combined") == "Combined"
                )
                initial_raw_name = (
                    _simulation_raw_default_filename(
                        latest_items[0],
                        "replication_timing",
                    )
                    if len(latest_items) == 1
                    else f"{RSIM_FILE_PREFIX}_{output_file_token('replication_timing')}"
                )
                raw_controls = raw_output_controls(
                    file_name=initial_raw_name,
                    default_type="TSV",
                    default_compress=False,
                    on_download=download_raw,
                    on_open=lambda controls, destination: download_raw(controls, open_in=destination),
                    on_remove=lambda _controls: remove_simulation_result(
                        latest_items[0].key if latest_items else ""
                    ),
                    extra_classes="dn-simulation-raw-data",
                )

                def sync_single_raw_filename() -> None:
                    """Keep an automatic single-result name aligned with its case."""

                    if len(latest_items) != 1:
                        return
                    selected_label = str(output.value or "Replication timing")
                    output_key = _OUTPUT_SPECS[selected_label].key
                    preference = retained_raw_preference(
                        latest_items[0],
                        output_key,
                    )
                    raw_controls.file_name.value = preference.file_name
                    raw_controls.file_name.update()

                def store_single_raw_filename() -> None:
                    """Remember manual single-result edits across style changes."""

                    if len(latest_items) != 1:
                        return
                    selected_label = str(output.value or "Replication timing")
                    output_key = _OUTPUT_SPECS[selected_label].key
                    preference = retained_raw_preference(
                        latest_items[0],
                        output_key,
                    )
                    preference.file_name = str(raw_controls.file_name.value or "")
                    preference.customized = bool(
                        preference.file_name
                        and preference.file_name != preference.automatic_name
                    )

                raw_controls.file_name.on_value_change(store_single_raw_filename)
                assert raw_controls.extra_slot is not None
                with raw_controls.extra_slot:
                    statistics_export = HistogramStatisticsExport(
                        plot_settings, get_chart=lambda: result_chart if latest_result is not None else None,
                        filename="simulation_statistics", marker="simulation-statistics", navigate=_navigate,
                    )
                    with (
                        ui.element("div")
                        .classes("w-full")
                        .mark("simulation-batch-raw-surface") as batch_raw_container
                    ):

                        @ui.refreshable
                        def render_batch_raw_files() -> None:
                            filter_raw_refresh['call']()
                            if len(latest_items) < 2:
                                return
                            selected_label = str(output.value or "Replication timing")
                            output_key = _OUTPUT_SPECS[selected_label].key
                            page, page_count, start, end = _simulation_raw_page_bounds(
                                len(latest_items),
                                int(batch_raw_page["value"]),
                            )
                            batch_raw_page["value"] = page

                            if page_count > 1:
                                options = {
                                    page_number: (
                                        f"{(page_number - 1) * _BATCH_RAW_PAGE_SIZE + 1:,}–"
                                        f"{min(len(latest_items), page_number * _BATCH_RAW_PAGE_SIZE):,}"
                                    )
                                    for page_number in range(1, page_count + 1)
                                }
                                with ui.row().classes("w-full items-end justify-between"):
                                    ui.label(
                                        f"Files {start + 1:,}–{end:,} of "
                                        f"{len(latest_items):,}"
                                    ).classes("dn-data-manager-file-detail")
                                    page_control = (
                                        ui.select(
                                            options,
                                            value=page,
                                            label="File range",
                                        )
                                        .props("dense options-dense")
                                        .classes("dn-field")
                                        .mark("simulation-batch-raw-page")
                                    )

                                    def change_page(_event: Any = None) -> None:
                                        try:
                                            batch_raw_page["value"] = int(
                                                page_control.value or 1
                                            )
                                        except (TypeError, ValueError):
                                            return
                                        render_batch_raw_files.refresh()
                                        render_ensemble_raw_files.refresh()

                                    page_control.on_value_change(change_page)

                            with ui.element("div").classes("dn-download-file-list"):
                                for index in range(start, end):
                                    item = latest_items[index]
                                    preference = retained_raw_preference(
                                        item,
                                        output_key,
                                    )
                                    with (
                                        ui.element("div")
                                        .classes("dn-mapping-batch-raw-item")
                                        .mark(f"simulation-batch-raw-{index + 1}")
                                    ):
                                        title = ui.label(
                                            _simulation_display_name(item)
                                        ).classes(
                                            "dn-download-file-name dn-mapping-batch-raw-title"
                                        )
                                        title.tooltip(item.identity.full)
                                        row_controls = raw_output_file_row(
                                            file_name=preference.file_name,
                                            default_type=preference.file_type,
                                            default_compress=preference.compress,
                                            on_open=lambda controls, destination, batch_item=item: download_batch_raw(batch_item, controls, open_in=destination),
                                            on_download=(
                                                lambda controls, batch_item=item: (
                                                    download_batch_raw(
                                                        batch_item,
                                                        controls,
                                                    )
                                                )
                                            ),
                                            on_remove=(
                                                lambda _controls, item_key=item.key: (
                                                    remove_simulation_result(item_key)
                                                )
                                            ),
                                            markers={
                                                "row": (
                                                    "simulation-batch-raw-row-"
                                                    f"{index + 1}"
                                                ),
                                                "file_name": (
                                                    "simulation-batch-raw-name-"
                                                    f"{index + 1}"
                                                ),
                                                "file_type": (
                                                    "simulation-batch-raw-type-"
                                                    f"{index + 1}"
                                                ),
                                                "compression": (
                                                    "simulation-batch-raw-compression-"
                                                    f"{index + 1}"
                                                ),
                                                "download": (
                                                    "action-download-simulation-raw-"
                                                    f"{index + 1}"
                                                ),
                                                "remove": (
                                                    "action-remove-simulation-raw-"
                                                    f"{index + 1}"
                                                ),
                                            },
                                            extra_classes="dn-mapping-batch-raw-row",
                                        )

                                        def store_file_name(
                                            _event: Any = None,
                                            *,
                                            control: Any = row_controls.file_name,
                                            retained: _SimulationRawPreference = preference,
                                        ) -> None:
                                            retained.file_name = str(control.value or "")
                                            retained.customized = bool(
                                                retained.file_name
                                                and retained.file_name
                                                != retained.automatic_name
                                            )

                                        def store_file_type(
                                            _event: Any = None,
                                            *,
                                            control: Any = row_controls.file_type,
                                            retained: _SimulationRawPreference = preference,
                                        ) -> None:
                                            retained.file_type = str(control.value or "TSV")

                                        def store_compression(
                                            _event: Any = None,
                                            *,
                                            control: Any = row_controls.compress,
                                            retained: _SimulationRawPreference = preference,
                                        ) -> None:
                                            retained.compress = bool(control.value)

                                        row_controls.file_name.on_value_change(
                                            store_file_name
                                        )
                                        row_controls.file_type.on_value_change(
                                            store_file_type
                                        )
                                        row_controls.compress.on_value_change(
                                            store_compression
                                        )
                            with ui.row().classes("w-full justify-end"):
                                (
                                    ui.button(
                                        "Download all",
                                        icon="archive",
                                        on_click=download_all_batch_raw,
                                    )
                                    .props("flat no-caps")
                                    .classes("dn-action-secondary")
                                    .mark("action-download-all-simulation-raw")
                                )

                        render_batch_raw_files()
                    with (
                        ui.element("div").classes("dn-download-file-list w-full")
                        .mark("simulation-ensemble-raw-files") as ensemble_raw_container
                    ):

                        @ui.refreshable
                        def render_ensemble_raw_files() -> None:
                            requested_count = _bounded_profile_count(
                                individual_profile_count.value, _MAX_PREVIEW_PROFILES,
                            )
                            _, _, start, end = _simulation_raw_page_bounds(
                                len(latest_items), int(batch_raw_page["value"]),
                            )
                            shown = False
                            for index in range(start, end):
                                item = latest_items[index]
                                profiles, deviation = _replication_profile_arrays(item.result)
                                count = 0 if requested_count is None else retained_individual_profile_count(
                                    item.result, requested_count,
                                )
                                stem = _simulation_raw_default_filename(item, "replication_timing")
                                for kind, label, available, automatic_name in (
                                    ("standard_deviation", "Replication timing standard deviation",
                                     bool(show_standard_deviation.value) and deviation is not None,
                                     f"{stem}_standard_deviation"),
                                    ("profiles", "Individual replication timing profiles",
                                     bool(show_individual_profiles.value) and profiles is not None and count > 0,
                                     profile_archive_stem(f"{stem}_individual_profiles", count)),
                                ):
                                    if not available:
                                        continue
                                    shown = True
                                    preference = ensemble_raw_preferences.setdefault(
                                        (item.key, kind), _SimulationRawPreference(
                                            file_name=automatic_name, automatic_name=automatic_name,
                                            compress=kind == "profiles",
                                        ),
                                    )
                                    if not preference.customized:
                                        preference.file_name = automatic_name
                                    preference.automatic_name = automatic_name
                                    prefix = f"simulation-{kind.replace('_', '-')}-{index + 1}"
                                    ui.label(
                                        f"{_simulation_display_name(item)} · {label}"
                                        if len(latest_items) > 1 else label
                                    ).classes("dn-download-file-name")
                                    row_controls = raw_output_file_row(
                                        file_name=preference.file_name,
                                        default_type=preference.file_type,
                                        default_compress=preference.compress,
                                        reserve_remove_space=True,
                                        on_open=lambda controls, destination, key=item.key, output_kind=kind: download_ensemble_raw(key, output_kind, controls, open_in=destination),
                                        on_download=lambda controls, key=item.key, output_kind=kind: (
                                            download_ensemble_raw(key, output_kind, controls)
                                        ),
                                        markers={
                                            key: f"{prefix}-{suffix}" for key, suffix in (
                                                ("row", "row"), ("file_name", "file-name"),
                                                ("file_type", "file-type"), ("compression", "compression"),
                                                ("download", "download"),
                                            )
                                        },
                                    )
                                    if kind == "profiles":
                                        row_controls.compress.set_value(True)
                                        row_controls.compress.disable()
                                        field_hint(row_controls.compress, "Individual profiles always download as a ZIP.")

                                    def remember_export(
                                        _event: Any = None, *, controls=row_controls, retained=preference,
                                    ) -> None:
                                        retained.file_name = str(controls.file_name.value or "")
                                        retained.customized = retained.file_name != retained.automatic_name
                                        retained.file_type = str(controls.file_type.value or "TSV")
                                        retained.compress = bool(controls.compress.value)

                                    for control in (row_controls.file_name, row_controls.file_type, row_controls.compress):
                                        control.on_value_change(remember_export)
                            ensemble_raw_container.visible = shown

                        render_ensemble_raw_files()
                    filter_layouts = {}

                    @ui.refreshable
                    def render_filter_raw_files() -> None:
                        _, _, first, stop = _simulation_raw_page_bounds(len(latest_items), int(batch_raw_page['value']))
                        for item in latest_items[first:stop]:
                            state = item.result.filter_state
                            if state is None:
                                continue
                            for key, label in FILTER_LABELS.items():
                                if key not in state.exports or key not in state.selections:
                                    continue
                                kind = f'{key}_filter'
                                name = _simulation_raw_default_filename(item, kind)
                                preference = ensemble_raw_preferences.setdefault((item.key, kind), _SimulationRawPreference(file_name=name, automatic_name=name))
                                ui.label((f'{_simulation_display_name(item)} · ' if len(latest_items) > 1 else '') + label + ' positions').classes('dn-download-file-name')
                                handlers = {}
                                prefix = f'simulation-{key.replace("_", "-")}-filter-{item.key}'
                                controls = raw_output_file_row(
                                    file_name=preference.file_name, default_type=preference.file_type,
                                    default_compress=preference.compress, reserve_remove_space=True,
                                    on_open=lambda _controls, destination, handlers=handlers: handlers['download'](open_in=destination),
                                    markers={'row': prefix + '-raw', 'download': prefix + '-download'},
                                    extra_classes='dn-simulation-filter-raw-row',
                                )
                                with controls.file_type.parent_slot.parent:
                                    layout = ui.select({'positions': 'Position list', 'intervals': 'Genomic intervals'},
                                        value=filter_layouts.get((item.key, key), 'positions'), label='Layout').props('dense options-dense').classes('dn-field')
                                layout.move(controls.file_type.parent_slot.parent, target_index=1)
                                def remember(*, controls=controls, preference=preference, layout=layout, token=(item.key, key)):
                                    preference.file_name = str(controls.file_name.value or '')
                                    preference.customized = preference.file_name != preference.automatic_name
                                    preference.file_type = str(controls.file_type.value or 'TSV')
                                    preference.compress = bool(controls.compress.value)
                                    filter_layouts[token] = layout.value
                                for control in (controls.file_name, controls.file_type, controls.compress, layout):
                                    control.on_value_change(remember)
                                async def download(*, open_in=None, state=state, key=key, controls=controls, layout=layout):
                                    try:
                                        artifact = await run.io_bound(build_simulation_filter_export, state, key, controls.file_type.value,
                                            filename=controls.file_name.value, layout=layout.value,
                                            compression=get_export_settings().raw_compression if controls.compress.value else 'none')
                                        if open_in is not None:
                                            await open_raw_artifact(artifact, open_in, _navigate, preferred_input='filter_positions')
                                        else:
                                            ui.download(artifact.data, artifact.filename, artifact.media_type)
                                    except (RuntimeError, ValueError, TypeError) as exc:
                                        notify(str(exc), type='negative')
                                handlers['download'] = download
                                controls.download_button.on_click(download)

                    render_filter_raw_files()
                    filter_raw_refresh['call'] = render_filter_raw_files.refresh
                raw_controls.row.set_visibility(len(latest_items) <= 1)
                if raw_controls.remove_button is not None:
                    raw_controls.remove_button.set_visibility(
                        len(latest_items) == 1
                    )
                batch_raw_container.visible = len(latest_items) > 1

                def remove_simulation_result(item_key: str) -> None:
                    """Remove one completed dataset while preserving the remaining batch."""

                    nonlocal active_dataset_index, latest_items, latest_result
                    if running:
                        notify(
                            "Stop the current simulation before removing results.",
                            type="warning",
                        )
                        return
                    removed_index = next(
                        (
                            index
                            for index, item in enumerate(latest_items)
                            if item.key == item_key
                        ),
                        None,
                    )
                    if removed_index is None:
                        return
                    latest_items = tuple(
                        item for item in latest_items if item.key != item_key
                    )
                    for preference_key in tuple(batch_raw_preferences):
                        if preference_key[0] == item_key:
                            batch_raw_preferences.pop(preference_key, None)
                    for preference_key in tuple(ensemble_raw_preferences):
                        if preference_key[0] == item_key:
                            ensemble_raw_preferences.pop(preference_key, None)
                    batch_colors.pop(item_key, None)
                    batch_line_types.pop(item_key, None)
                    batch_names.pop(item_key, None)
                    if not latest_items:
                        clear_simulation_output()
                        return

                    active_dataset_index = min(removed_index, len(latest_items) - 1)
                    latest_result = latest_items[active_dataset_index].result
                    batch_raw_page["value"] = min(
                        int(batch_raw_page["value"]),
                        _simulation_raw_page_bounds(
                            len(latest_items),
                            int(batch_raw_page["value"]),
                        )[1],
                    )
                    sync_batch_presentation(redraw=False)
                    sync_single_raw_filename()
                    select_dataset(latest_items[active_dataset_index].key)

                def available_profile_count() -> int:
                    if latest_result is None:
                        return 0
                    profiles, _standard_deviation = _replication_profile_arrays(latest_result)
                    return 0 if profiles is None else int(profiles.shape[0])

                def selected_profile_count() -> int | None:
                    return _bounded_profile_count(
                        individual_profile_count.value,
                        available_profile_count(),
                    )

                def update_active_stats() -> None:
                    if latest_result is None:
                        stats_container.visible = False
                        return
                    position_steps = np.diff(latest_result.position_kb)
                    time_steps = np.diff(latest_result.time_grid)
                    position_resolution = (
                        float(np.median(position_steps)) if position_steps.size else 0.0
                    )
                    time_resolution = (
                        float(np.median(time_steps)) if time_steps.size else 0.0
                    )
                    replicate_stats.text = f"{latest_result.replicates:,}"
                    resolution_stats.text = (
                        f"{position_resolution:g} kb/bin | {time_resolution:g} min/bin"
                    )
                    region_stats.text = (
                        f"{latest_result.position_kb[0]:g}-"
                        f"{latest_result.position_kb[-1]:g} kb"
                    )
                    geometry_stats.text = latest_result.geometry.title()
                    for value in stats.values():
                        value.update()
                    stats_container.visible = True

                def select_dataset(key: str) -> None:
                    nonlocal active_dataset_index, latest_result
                    if not latest_items:
                        return
                    selected_index = next(
                        (
                            index
                            for index, item in enumerate(latest_items)
                            if item.key == key
                        ),
                        None,
                    )
                    if selected_index is None:
                        return
                    active_dataset_index = selected_index
                    selected_page = (
                        active_dataset_index // _BATCH_NAV_PAGE_SIZE
                    ) + 1
                    batch_navigation_page["value"] = selected_page
                    latest_result = latest_items[active_dataset_index].result
                    update_active_stats()
                    sync_output_options()
                    histogram_bin_sync.sync_to_data(redraw=False)
                    sync_view_controls()
                    populate_automatic_y_limits(redraw=False)
                    render_latest_result()
                    render_dataset_selector.refresh()

                def sync_batch_surfaces() -> None:
                    multiple = len(latest_items) > 1
                    individual = bool(
                        multiple
                        and str(presentation.value or "Combined") == "Individual"
                    )
                    combined = bool(
                        multiple
                        and str(presentation.value or "Combined") == "Combined"
                        and combined_preview_compatible()
                    )
                    batch_navigation.visible = multiple
                    dataset_selector_surface.visible = individual
                    batch_color_container.visible = combined
                    raw_controls.row.set_visibility(not multiple)
                    if raw_controls.remove_button is not None:
                        raw_controls.remove_button.set_visibility(
                            bool(latest_items) and not multiple
                        )
                    batch_raw_container.visible = multiple
                    if individual:
                        render_dataset_selector.refresh()
                    if combined:
                        render_batch_colors.refresh()
                    render_batch_raw_files.refresh()

                def sync_batch_presentation(*, redraw: bool = True) -> None:
                    nonlocal updating_presentation
                    if updating_presentation:
                        return
                    updating_presentation = True
                    try:
                        compatible = combined_preview_compatible()
                        options = (
                            ["Combined", "Individual"]
                            if len(latest_items) > 1 and compatible
                            else ["Individual"]
                        )
                        presentation.options = options
                        if presentation.value not in options:
                            presentation.value = options[0]
                        presentation.visible = len(latest_items) > 1
                        presentation.update()
                        sync_batch_surfaces()
                        sync_view_controls()
                    finally:
                        updating_presentation = False
                    if redraw and latest_result is not None:
                        populate_automatic_y_limits(redraw=False)
                        render_latest_result()

                def sync_view_controls() -> None:
                    """Show controls for the representation rendered by Default."""

                    selected = str(output.value or "Replication timing")
                    spec = _output_spec(selected, latest_result)
                    requested_view = str(view.value or "Default")
                    allow_individual_profiles = (
                        selected == "Replication timing"
                        and requested_view in {"Default", "Circular"}
                        and not combined_preview_active()
                    )
                    allow_standard_deviation = (
                        selected == "Replication timing"
                        and requested_view in {"Default", "Circular"}
                        and not combined_preview_active()
                    )
                    if allow_individual_profiles:
                        # A larger next run may retain more profiles than the
                        # current result, so keep the configuration editable.
                        maximum = _MAX_PREVIEW_PROFILES
                        individual_profile_count._props["max"] = maximum
                        current_count = _bounded_profile_count(
                            individual_profile_count.value,
                            maximum,
                        )
                        if current_count is not None and (
                            individual_profile_count.value != current_count
                        ):
                            individual_profile_count.value = current_count
                        individual_profile_count.update()
                    if requested_view == "Default":
                        time_array = spec.kind == "xt"
                        histogram = spec.kind == "distribution"
                        boxplot = False
                    else:
                        time_array = False
                        histogram = spec.kind == "distribution" or requested_view == "Histogram"
                        boxplot = requested_view == "Boxplot"
                    sync_preview_settings_visibility(
                        plot_settings,
                        plot_type=(
                            "Heatmap"
                            if time_array
                            else "Boxplot"
                            if boxplot
                            else "Histogram"
                            if histogram
                            else "Line"
                        ),
                        allow_histogram=spec.kind != "xt",
                        allow_boxplot=spec.kind == "distribution",
                        allow_circular=spec.kind not in {"xt", "distribution"},
                        allow_individual_profiles=allow_individual_profiles,
                        allow_standard_deviation=allow_standard_deviation,
                        available_filters=tuple(
                            {key for item in latest_items if item.result.filter_state is not None
                             for key in item.result.filter_state.selections}
                            | {key for key, controls in filter_controls.items()
                               if filter_relative.value in {key, 'both'} and controls.controls['enabled'].value}
                        ),
                    )
                    # A combined batch uses one colour control per dataset; the
                    # single-series colour field would otherwise be ambiguous.
                    if combined_preview_active():
                        plot_settings.primary_color_surface.visible = False
                    render_ensemble_raw_files.refresh()
                    render_filter_raw_files.refresh()

                def sync_output_options() -> None:
                    """Expose only outputs retained by the latest simulation."""

                    if latest_items:
                        available_keys = set(latest_items[0].result.outputs)
                        for item in latest_items[1:]:
                            available_keys.intersection_update(item.result.outputs)
                        labels = [
                            label
                            for label, spec in _OUTPUT_SPECS.items()
                            if spec.key in available_keys
                        ]
                    elif latest_result is None:
                        labels = list(_STANDARD_OUTPUTS)
                        if track_time_outputs.value:
                            labels.extend(_TIME_TRACKED_OUTPUTS)
                        if track_detailed_outputs.value:
                            labels.extend(_SPACETIME_OUTPUTS)
                    else:
                        labels = [
                            label
                            for label, spec in _STANDARD_OUTPUTS.items()
                            if spec.key in latest_result.outputs
                        ]
                        labels.extend(
                            label
                            for label, spec in _TIME_TRACKED_OUTPUTS.items()
                            if spec.key in latest_result.outputs
                        )
                        labels.extend(
                            label
                            for label, spec in _SPACETIME_OUTPUTS.items()
                            if spec.key in latest_result.outputs
                        )
                        labels.extend(
                            label for label, spec in _INPUT_OUTPUTS.items()
                            if spec.key in latest_result.outputs
                        )
                    if not labels:
                        labels = ["Replication timing"]
                    output.options = labels
                    selection_changed = output.value not in labels
                    if output.value not in labels:
                        output.value = "Replication timing"
                    output.update()
                    if selection_changed:
                        sync_plot_labels()
                    else:
                        sync_view_controls()

                def sync_plot_labels() -> None:
                    """Switch the existing result locally without starting a new run."""

                    selected = str(output.value or "Replication timing")
                    spec = _output_spec(selected, latest_result)
                    title.value = selected
                    if len(latest_items) == 1:
                        sync_single_raw_filename()
                    else:
                        raw_controls.file_name.value = (
                            f"{RSIM_FILE_PREFIX}_{output_file_token(spec.key)}"
                        )
                    # "Default" resolves to the natural representation for the
                    # selected quantity: line, histogram, or space-time heatmap.
                    view.value = "Default"
                    sync_view_controls()
                    default_x_label, default_y_label = _default_axis_labels(
                        output_label=selected,
                        view_mode=str(view.value or "Default"),
                        density=bool(density.value),
                        result=latest_result,
                    )
                    x_axis_label.value = default_x_label
                    y_axis_label.value = default_y_label
                    for control in (
                        title,
                        x_axis_label,
                        y_axis_label,
                        view,
                        raw_controls.file_name,
                    ):
                        control.update()
                    render_batch_raw_files.refresh()
                    sync_batch_presentation(redraw=False)
                    if latest_result is not None:
                        histogram_bin_sync.sync_to_data(redraw=False)
                        populate_automatic_y_limits(redraw=False)
                        render_latest_result()

                def sync_view_axis_labels() -> None:
                    """Keep histogram quantity/count labels consistent with its view."""

                    sync_view_controls()
                    sync_batch_presentation(redraw=False)
                    selected = str(output.value or "Replication timing")
                    default_x_label, default_y_label = _default_axis_labels(
                        output_label=selected,
                        view_mode=str(view.value or "Default"),
                        density=bool(density.value),
                        result=latest_result,
                    )
                    x_axis_label.value = default_x_label
                    y_axis_label.value = default_y_label
                    x_axis_label.update()
                    y_axis_label.update()
                    histogram_bin_sync.sync_to_data(redraw=False)
                    refresh_axis_range()

                def histogram_values() -> Any | None:
                    if latest_result is None:
                        return None
                    selected = str(output.value or "Replication timing")
                    spec = _output_spec(selected, latest_result)
                    return latest_result.outputs.get(spec.key)

                histogram_bin_sync = HistogramBinSynchronizer(
                    plot_settings,
                    values=histogram_values,
                    on_valid_change=refresh_axis_range,
                    log_x=lambda: bool(log_x.value),
                )

                def log_x_changed() -> None:
                    histogram_bin_sync.sync_to_data(redraw=False)
                    render_latest_result()

                def ensemble_display_changed() -> None:
                    sync_view_controls()
                    refresh_axis_range()

                def individual_profile_count_changed() -> None:
                    nonlocal latest_items, latest_result
                    count = _bounded_profile_count(individual_profile_count.value, _MAX_PREVIEW_PROFILES)
                    if count is not None:
                        latest_items = tuple(
                            replace(item, result=limit_replication_profiles(item.result, count))
                            for item in latest_items
                        )
                        if latest_items:
                            latest_result = latest_items[active_dataset_index].result
                        render_batch_raw_files.refresh()
                        render_ensemble_raw_files.refresh()
                        refresh_axis_range()

                track_time_outputs.on_value_change(sync_output_options)
                track_detailed_outputs.on_value_change(sync_output_options)
                output.on_value_change(sync_plot_labels)
                presentation.on_value_change(sync_batch_presentation)

                def line_type_changed() -> None:
                    render_batch_colors.refresh()
                    render_latest_result()

                bind_preview_settings_updates(
                    plot_settings,
                    on_redraw=render_latest_result,
                    on_view_change=sync_view_axis_labels,
                    on_density_change=sync_view_axis_labels,
                    on_histogram_range_change=refresh_axis_range,
                    on_log_x_change=log_x_changed,
                    on_y_limits_change=use_custom_y_limits,
                    on_visibility_change=sync_view_controls,
                    on_line_type_change=line_type_changed,
                )
                show_individual_profiles.on_value_change(ensemble_display_changed)
                show_standard_deviation.on_value_change(ensemble_display_changed)
                individual_profile_count.on_value_change(individual_profile_count_changed)
                show_plot.on_value_change(render_latest_result)
                filter_refresh['call'] = sync_view_controls
                sync_view_controls()

                if latest_result is not None:
                    histogram_bin_sync.sync_to_data(redraw=False)
                    update_active_stats()
                    sync_output_options()
                    sync_batch_presentation(redraw=False)
                    sync_batch_surfaces()
                    status.text = "Restored from Cached Files"
                    populate_automatic_y_limits(redraw=False)
                    render_latest_result()

                def clear_simulation_output() -> None:
                    nonlocal active_dataset_index, latest_items
                    nonlocal latest_result, result_chart
                    latest_items = ()
                    batch_colors.clear()
                    batch_line_types.clear()
                    batch_names.clear()
                    batch_raw_preferences.clear()
                    ensemble_raw_preferences.clear()
                    batch_raw_page["value"] = 1
                    batch_navigation_page["value"] = 1
                    active_dataset_index = 0
                    latest_result = None
                    result_chart = None
                    progress.visible = False
                    stats_container.visible = False
                    status.text = "Ready"
                    selected_label = str(output.value or "Replication timing")
                    raw_controls.file_name.value = (
                        f"{RSIM_FILE_PREFIX}_"
                        f"{output_file_token(_OUTPUT_SPECS[selected_label].key)}"
                    )
                    raw_controls.file_name.update()
                    sync_output_options()
                    sync_batch_presentation(redraw=False)
                    sync_batch_surfaces()
                    sync_view_controls()
                    result_slot.clear()
                    with result_slot:
                        preview_plot("No result yet", kind="simulation")

                current_session_cache().register_output_clearer(
                    "simulation",
                    clear_simulation_output,
                )

                def refresh_simulation_names() -> None:
                    """Apply a changed global name style without rerunning rsim."""

                    current_style = _simulation_name_style()
                    if current_style == rendered_name_style["value"]:
                        return
                    rendered_name_style["value"] = current_style
                    sync_single_raw_filename()
                    render_dataset_selector.refresh()
                    render_batch_colors.refresh()
                    render_batch_raw_files.refresh()
                    render_ensemble_raw_files.refresh()
                    if combined_preview_active():
                        render_latest_result()

                current_session_cache().register_page_refresher(
                    "simulation",
                    refresh_simulation_names,
                )

                def raw_input_targets():
                    async def receive(key, files):
                        if running:
                            raise ValueError("Stop the simulation before adding input data.")
                        simulation_sources[key].mode.set_value("profile")
                        for _ in range(4):
                            await asyncio.sleep(0)
                        await append_raw_uploads(lambda: simulation_sources[key].uploader, files)
                    async def receive_mask(key, files):
                        if running:
                            raise ValueError("Stop the simulation before adding a mask.")
                        if filter_relative.value not in {key, "both"}:
                            filter_relative.set_value(key)
                        await filter_controls[key].receive_mask(files)
                    return tuple(
                        RawInputTarget(key, label, lambda files, key=key: receive(key, files))
                        for key, label in (("initiation_rate", "Initiation rate"), ("fork_speed", "Fork speed"))
                    ) + tuple(
                        RawInputTarget(key + "_mask", label + " mask", lambda files, key=key: receive_mask(key, files), kind="mask")
                        for key, label in (("initiation_rate", "Initiation rate"), ("fork_speed", "Fork speed"))
                    )

                current_session_cache().register_input_targets("simulation", raw_input_targets)
