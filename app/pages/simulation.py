"""Interactive NiceGUI simulation workflow."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import CancelledError
from dataclasses import dataclass, replace
from threading import Event, Lock
from typing import Any

import numpy as np
from nicegui import run, ui

from app.artifact_archives import (
    DEFAULT_BATCH_ARCHIVE_SOURCE_BYTES,
    build_artifact_archive,
)
from app.batch_workflows import (
    BatchSourceSummary,
    CombinationSource,
    combination_names,
    plan_batch_sources,
    plan_combination_indices,
    unique_dataset_names,
)
from app.export_settings import get_export_settings
from app.notifications import notify
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
    DATA_ICON,
    RawOutputControls,
    action_dock,
    workflow_action_button,
    workflow_action_icon,
    data_upload,
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
    unique_compact_export_stems,
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
    "Replicated fraction": _OutputSpec(
        "replicated_fraction", "Time (min)", "Replicated fraction", "time"
    ),
    "Total forks": _OutputSpec("total_forks", "Time (min)", "Total forks", "time"),
    "Active forks": _OutputSpec("active_forks", "Time (min)", "Active forks", "time"),
    "Stalled forks": _OutputSpec("stalled_forks", "Time (min)", "Stalled forks", "time"),
    "Firing factors": _OutputSpec("firing_factors", "Time (min)", "Available factors", "time"),
}
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
}
_STANDARD_OUTPUTS = {**_BASE_OUTPUTS, **_DISTRIBUTION_OUTPUTS}
_INPUT_OUTPUTS = {
    "Firing rate (input)": _OutputSpec("input:firing_rate", "Position (kb)", "Firing rate", "input"),
    "Fork speed (input)": _OutputSpec("input:fork_speed", "Position (kb)", "Fork speed", "input"),
}
_OUTPUT_SPECS = {
    **_STANDARD_OUTPUTS,
    **_TIME_OUTPUTS,
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
        return replace(spec, kind="time", x_label="Time (min)")
    return replace(spec, kind="space")

# Some simulation quantities have a meaningful physical domain which should not
# shrink to the particular values observed in one run. Other outputs retain an
# automatic, data-dependent range chosen by ECharts.
_NONNEGATIVE_LINE_OUTPUTS = {
    "Origin efficiency",
    "Total forks",
    "Active forks",
    "Stalled forks",
    "Firing factors",
}

_SOURCE_ACCEPT = ".csv,.tsv,.txt,.npy,.npv,.npz,.gz"
_SOURCE_DEPENDENCE = {
    "x": "Space",
    "t": "Time",
    "xt": "Space-time",
}

_MAX_PREVIEW_PROFILES = 50
_BATCH_RAW_PAGE_SIZE = 50
_BATCH_NAV_PAGE_SIZE = 50
_MAX_COMBINED_PREVIEW_DATASETS = 50
_MAX_SIMULATION_CASES = 50_000


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


@dataclass(frozen=True, slots=True)
class _PreparedSimulationSource:
    """One fully materialized Simulation factor and its naming metadata."""

    label: str
    values: tuple[float | np.ndarray, ...]
    components: tuple[str, ...]
    summary: BatchSourceSummary | None = None


@dataclass(frozen=True, slots=True)
class _SimulationBatchCase:
    """One explicit pairwise or Cartesian Simulation request."""

    key: str
    identity: ResultName
    full_export_stem: str
    short_export_stem: str
    item_indices: tuple[int, ...]

    @property
    def name(self) -> str:
        """Retain the fully descriptive legacy planner interface."""

        return self.identity.full

    @property
    def export_stem(self) -> str:
        """Retain the fully descriptive legacy export-stem interface."""

        return self.full_export_stem


def _format_simulation_component(value: float, *, units: str = "") -> str:
    """Format a constant compactly for descriptive batch names."""

    numeric = f"{float(value):.10g}"
    return f"{numeric} {units}" if units else numeric


def _plan_simulation_cases(
    inputs: Sequence[_PreparedSimulationSource],
    *,
    combination_mode: str = "pairwise",
) -> tuple[_SimulationBatchCase, ...]:
    """Resolve explicit pairwise or Cartesian cases before starting rsim."""

    for item in inputs:
        if not item.values:
            raise ValueError(f"{item.label} must contain at least one value or file.")
        if item.summary is not None:
            plan_batch_sources((item.summary,), require_matching_shapes=False)
        if len(item.components) != len(item.values):
            raise ValueError(f"{item.label} naming metadata are incomplete.")

    mode = "all" if combination_mode == "all" else "pairwise"
    source_sizes = tuple(len(item.values) for item in inputs)
    projected = (
        math.prod(source_sizes)
        if mode == "all"
        else max(source_sizes, default=1)
    )
    if projected > _MAX_SIMULATION_CASES:
        raise ValueError(
            f"These inputs would create {projected:,} simulation runs. "
            f"Reduce the inputs or use at most {_MAX_SIMULATION_CASES:,} combinations."
        )
    combinations = plan_combination_indices(source_sizes, mode=mode)
    identities = combination_names(
        "Simulation",
        tuple(
            CombinationSource(
                item.label,
                item.components,
                include_singleton_in_short=item.summary is not None,
            )
            for item in inputs
        ),
        combinations,
        include_singletons=True,
    )
    full_stems = unique_compact_export_stems(
        tuple(_simulation_case_label(identity.full) for identity in identities),
        fallback="dataset",
        maximum=110,
    )
    short_stems = unique_compact_export_stems(
        tuple(_simulation_case_label(identity.short) for identity in identities),
        fallback="dataset",
        maximum=110,
    )
    # A single run driven entirely by fixed scalar assumptions has no useful
    # case identity in an exported filename. Keep its full label descriptive
    # in the UI, but retain the concise workflow/output filename convention.
    if len(combinations) == 1 and not any(item.summary is not None for item in inputs):
        full_stems = ("dataset",)
        short_stems = ("dataset",)
    cases: list[_SimulationBatchCase] = []
    for identity, full_stem, short_stem, indices in zip(
        identities,
        full_stems,
        short_stems,
        combinations,
        strict=True,
    ):
        cases.append(
            _SimulationBatchCase(
                key=f"simulation-case-{full_stem}",
                identity=identity,
                full_export_stem=full_stem,
                short_export_stem=short_stem,
                item_indices=tuple(indices),
            )
        )
    return tuple(cases)


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


def _simulation_case_label(display_name: str) -> str:
    """Remove the workflow prefix before constructing a case-only stem."""

    candidate = str(display_name).strip()
    prefix = "Simulation · "
    if candidate.startswith(prefix):
        return candidate.removeprefix(prefix)
    return "dataset" if candidate == "Simulation" else candidate


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
                            step=0.001 if label.casefold() == "firing rate" else 0.1,
                        )
                        .props("dense")
                        .classes("dn-field dn-mapping-constant-value")
                        .mark(f"source-{marker}-constant-{index}")
                    )
                    control._props["aria-label"] = f"{label} for {row_label}"
                    if units:
                        control.tooltip(units)

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
                        "constants": "Multiple constants",
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
                if replaced_index is None:
                    inspections.append(inspection)
                else:
                    # Keep N/N firing-rate and fork-speed pairing stable when
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
                multiple=True,
                max_files=None,
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
        return _read_simulation_constant_controls(
            controls.constants,
            label=label,
            positive=label.casefold() == "fork speed",
        )
    if not controls.inspections:
        raise ValueError(f"Upload {label.casefold()} data.")
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
                    f"{label} file {inspection.name!r} does not contain the selected "
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
                f"{label} file {inspection.name!r} does not contain the selected "
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


def _trim_simulation_profile(
    value: float | np.ndarray,
    *,
    start: float,
    end: float,
    resolution: float,
) -> float | np.ndarray:
    """Keep the spatial bins requested by an explicitly edited region end."""

    if _simulation_spatial_size(value) is None:
        return value
    if not math.isfinite(resolution) or resolution <= 0:
        raise ValueError("Spatial resolution must be finite and greater than zero.")
    if not math.isfinite(start) or not math.isfinite(end):
        raise ValueError("Region start and end must be finite.")
    if end <= start:
        raise ValueError("Region end must be greater than region start.")
    ratio = (end - start) / resolution
    if not math.isfinite(ratio):
        raise ValueError("The requested spatial grid is not finite.")
    nearest = round(ratio)
    if math.isclose(ratio, nearest, rel_tol=1e-12, abs_tol=1e-12):
        ratio = float(nearest)
    return np.asarray(value)[:math.ceil(ratio)]


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


def _uploaded_spatial_size(
    controls: _SimulationSourceControls,
    value: float | np.ndarray,
) -> int | None:
    """Return the spatial axis length encoded by an uploaded rsim field."""

    if str(controls.mode.value) != "profile":
        return None
    form = str(controls.form.value)
    array = np.asarray(value)
    if form == "x":
        return int(array.size)
    if form == "xt":
        return int(array.shape[0])
    return None


class _ProgressState:
    """Thread-safe hand-off from DNAscape's worker callback to the UI loop."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._value = 0.0
        self._message = "Waiting to run"

    def report(self, value: float, message: str) -> None:
        with self._lock:
            self._value = max(0.0, min(1.0, float(value)))
            self._message = str(message)

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
    if spec.kind != "xt":
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
    show_grid: bool = True,
    log_z: bool = False,
    log_x: bool = False,
    invert_x: bool = False,
    invert_z: bool = False,
    show_histogram_statistics: bool = False,
    histogram_statistics_position: str = "Top right",
    histogram_bin_size: float | None = None,
    histogram_x_limits: tuple[float, float] | None = None,
    histogram_touching: bool = True,
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
        if output_label in _TIME_OUTPUTS:
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
        raise ValueError(
            "The selected output has no finite values. Try a positive firing rate or a "
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
    return build_preview_plot_options(
        plot_type=plot_type,
        x_limits=x_limits,
        z_limits=z_limits,
        title=title.strip() or output_label,
        x_label=x_axis_label.strip()
        or (
            spec.y_label
            if plot_type == "Histogram"
            else ""
            if plot_type == "Boxplot"
            else spec.x_label
        ),
        y_label=y_axis_label.strip() or ("Density" if density else "Count")
        if plot_type == "Histogram"
        else y_axis_label.strip() or spec.y_label,
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
        values=values.T if spec.kind == "xt" else values,
        x_axis=result.position_kb if spec.kind == "xt" else axis,
        y_axis=result.time_grid if spec.kind == "xt" else None,
        lines=lines,
        bands=bands,
        legend_position=legend_position,
        line_width=line_width,
        line_type=line_type,
        show_markers=show_markers,
        show_grid=show_grid,
        interactive=not ensemble_active,
        log_x=log_x,
        invert_x=invert_x,
        invert_z=invert_z,
        show_histogram_statistics=show_histogram_statistics,
        histogram_statistics_position=histogram_statistics_position,
        histogram_bin_size=histogram_bin_size,
        histogram_x_limits=histogram_x_limits,
        histogram_touching=histogram_touching,
        histogram_boundary=histogram_boundary,
    )


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
        x_label=x_axis_label.strip() or spec.x_label,
        y_label=y_axis_label.strip() or spec.y_label,
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
    show_grid: bool = True,
    log_x: bool = False,
    invert_x: bool = False,
    invert_z: bool = False,
    show_histogram_statistics: bool = False,
    histogram_statistics_position: str = "Top right",
    histogram_bin_size: float | None = None,
    histogram_x_limits: tuple[float, float] | None = None,
    histogram_touching: bool = True,
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
    batch_raw_page = {"value": 1}
    rendered_name_style = {"value": _simulation_name_style()}
    source_region_refresh: dict[str, Callable[[], Any]] = {"call": lambda: None}
    simulation_constant_states = {
        "firing_rate": _SimulationConstantState.create(),
        "fork_speed": _SimulationConstantState.create(),
    }
    simulation_combination_mode = {"value": "pairwise"}
    simulation_sources: dict[str, _SimulationSourceControls] = {}
    refreshing_constant_surfaces = False

    page_intro(
        "Simulation",
        "Simulate DNA replication",
        "Run stochastic replication ensembles and inspect their outputs.",
    )
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
                            if current_key == "firing_rate"
                            else "firing_rate"
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
                            ui.label("Firing rate").classes("dn-setting-group-title")
                            firing_source = _render_simulation_source(
                                "Firing rate",
                                constant=0.01,
                                units="min⁻¹ per bin",
                                retained_constants=simulation_constant_states[
                                    "firing_rate"
                                ],
                                peer_constant_rows=lambda: counterpart_rows(
                                    "firing_rate"
                                ),
                                combination_mode=simulation_combination_mode,
                                on_profile_change=lambda: source_region_refresh["call"](),
                                on_structure_change=scientific_source_structure_changed,
                            )
                            simulation_sources["firing_rate"] = firing_source
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
                            ui.number(
                                label="End (kb, exclusive)", value=1000.0, min=0.000001, step=1.0
                            )
                            .props("dense")
                            .classes("dn-field")
                            .mark("simulation-region-end")
                        )

                    automatic_region_end: float | None = None

                    def sync_region_to_profiles() -> None:
                        nonlocal automatic_region_end
                        sizes: set[int] = set()
                        for source, source_label in (
                            (firing_source, "Firing rate"),
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
                            if spatial_size is not None:
                                sizes.add(spatial_size)
                        if len(sizes) > 1:
                            # Keep uploads permissive. The complete, named
                            # consistency error is reported transactionally when
                            # Simulate is selected.
                            return
                        if not sizes:
                            return
                        raw_resolution = resolution_x.value
                        if raw_resolution is None:
                            return
                        try:
                            spatial_resolution = float(raw_resolution)
                        except (TypeError, ValueError):
                            return
                        if not math.isfinite(spatial_resolution) or spatial_resolution <= 0:
                            return
                        automatic_region_end = sizes.pop() * spatial_resolution
                        region_start.set_value(0.0)
                        region_end.set_value(automatic_region_end)

                    source_region_refresh["call"] = sync_region_to_profiles
                    resolution_x.on_value_change(sync_region_to_profiles)
                    sync_region_to_profiles()

                with expansion("Simulation settings", icon="tune"):
                    with ui.element("div").classes("dn-simulation-settings-grid"):
                        with ui.column().classes("dn-setting-group"):
                            ui.label("Ensemble").classes("dn-setting-group-title")
                            replicates = (
                                ui.number(
                                    label="Replicates",
                                    value=20,
                                    min=1,
                                    max=MAX_REPLICATES,
                                    step=1,
                                )
                                .props("dense")
                                .classes("dn-field dn-field-compact")
                                .mark("simulation-replicates")
                            )
                            geometry = (
                                ui.select(
                                    ["linear", "periodic"],
                                    value="linear",
                                    label="Geometry",
                                )
                                .props("dense options-dense")
                                .classes("dn-field")
                            )
                            missing_values = (
                                ui.select(
                                    ["error", "interpolate", "zero"],
                                    value="error",
                                    label="Missing values",
                                )
                                .props("dense options-dense")
                                .classes("dn-field")
                            )
                            with ui.element("div").classes("dn-setting-inline"):
                                fixed_seed = ui.checkbox("Fixed seed", value=False)
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
                                .classes("dn-field dn-field-compact")
                            )
                            max_replication_time = (
                                ui.number(
                                    label="Maximum time (min)",
                                    value=1200.0,
                                    min=0.000001,
                                    step=50.0,
                                )
                                .props("dense")
                                .classes("dn-field dn-field-compact")
                            )
                            track_time_outputs = ui.checkbox(
                                "Track time courses",
                                value=bool(
                                    latest_result is not None
                                    and getattr(latest_result, "time_outputs", False)
                                ),
                            ).mark("simulation-track-time-outputs")
                            track_time_outputs.tooltip(
                                "Retain replicated fraction, fork counts, and firing-factor "
                                "availability across time."
                            )
                            track_detailed_outputs = ui.checkbox(
                                "Track space-time outputs",
                                value=bool(
                                    latest_result is not None and latest_result.detailed_outputs
                                ),
                            ).mark("simulation-track-detailed-outputs")
                            track_detailed_outputs.tooltip(
                                "Retain position-by-time heatmaps; these require more memory."
                            )

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
                                    with ui.element("div").classes("dn-setting-inline"):
                                        finite_pool = ui.checkbox("Finite pool", value=False).mark(
                                            "simulation-finite-pool"
                                        )
                                        pool_size = (
                                            ui.number(label="Pool size", value=100, min=2, step=1)
                                            .props("dense")
                                            .classes("dn-field dn-field-xs")
                                        )
                                    pool_size.bind_visibility_from(finite_pool, "value")
                                    with (
                                        ui.element("div")
                                        .classes("dn-setting-inline")
                                        .mark("simulation-pool-fields") as pool_fields
                                    ):
                                        recycle_rate = (
                                            ui.number(
                                                label="Recycle rate",
                                                value=0.0,
                                                min=0.0,
                                                step=0.001,
                                            )
                                            .props("dense")
                                            .classes("dn-field dn-field-compact")
                                        )
                                        factor_coupling = ui.checkbox("Fork coupling", value=False)
                                    pool_fields.bind_visibility_from(finite_pool, "value")

                                with (
                                    ui.element("div")
                                    .classes("dn-setting-subgroup")
                                    .mark("simulation-stalling-settings")
                                ):
                                    ui.label("Stalling").classes("dn-setting-subgroup-title")
                                    with ui.element("div").classes("dn-setting-inline"):
                                        stall_rate = (
                                            ui.number(
                                                label="Stall rate (stalls/kb)",
                                                value=0.0,
                                                min=0.0,
                                                step=0.001,
                                            )
                                            .props("dense")
                                            .classes("dn-field dn-field-compact")
                                            .mark("simulation-stall-rate-per-kb")
                                        )
                                        non_recovering = ui.checkbox("Non-recovering", value=True)
                                    stall_duration = (
                                        ui.number(
                                            label="Recovery time (min)",
                                            value=10.0,
                                            min=0.000001,
                                            step=1.0,
                                        )
                                        .props("dense")
                                        .classes("dn-field dn-field-compact")
                                        .mark("simulation-stall-duration")
                                    )
                                    stall_duration.bind_visibility_from(
                                        non_recovering,
                                        "value",
                                        backward=lambda value: not value,
                                    )
                                    stall_probability = (
                                        ui.label()
                                        .classes("dn-settings-help")
                                        .mark("simulation-stall-probability")
                                    )

                                    def sync_stall_probability() -> None:
                                        try:
                                            rate_value = stall_rate.value
                                            bin_width_value = resolution_x.value
                                            if rate_value is None or bin_width_value is None:
                                                raise ValueError
                                            rate = float(rate_value)
                                            bin_width = float(bin_width_value)
                                            if (
                                                not math.isfinite(rate)
                                                or rate < 0.0
                                                or not math.isfinite(bin_width)
                                                or bin_width <= 0.0
                                            ):
                                                raise ValueError
                                        except (TypeError, ValueError):
                                            stall_probability.text = (
                                                "Converted automatically using the spatial resolution."
                                            )
                                            return
                                        probability = -math.expm1(-(rate * bin_width))
                                        stall_probability.text = (
                                            f"At {bin_width:g} kb/bin: {probability:.1%} "
                                            "probability per directed bin."
                                        )

                                    stall_rate.on_value_change(sync_stall_probability)
                                    resolution_x.on_value_change(sync_stall_probability)
                                    sync_stall_probability()

                def combined_preview_compatible() -> bool:
                    if not 2 <= len(latest_items) <= _MAX_COMBINED_PREVIEW_DATASETS:
                        return False
                    selected_label = str(output.value or "Replication timing")
                    spec = _output_spec(selected_label, latest_result)
                    if spec.kind not in {"space", "time"}:
                        return False
                    if str(view.value or "Default") != "Default":
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

                async def download_raw(controls: RawOutputControls) -> None:
                    if latest_result is None:
                        notify("Run a simulation before downloading raw data.", type="warning")
                        return
                    try:
                        selected_label = str(output.value or "Replication timing")
                        artifact = await run.io_bound(
                            build_raw_export,
                            latest_result,
                            _OUTPUT_SPECS[selected_label].key,
                            str(controls.file_type.value or "NPY"),
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
                    cache_artifact(
                        artifact,
                        source="Simulation",
                        kind="raw",
                        detail=selected_label,
                        navigate_to="simulation",
                    )
                    ui.download(artifact.data, artifact.filename, artifact.media_type)

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
                    file_type = str(controls.file_type.value or "NPY")
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
                            preference.file_type or "NPY",
                            enabled_compression if preference.compress else "none",
                        )
                        for item in items_snapshot
                        for preference in (
                            retained_raw_preference(item, output_key),
                        )
                    )
                    try:
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
                                        _number(line_width, "Curve thickness")
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
                                        histogram_statistics_position.value or "Top right"
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
                                        _number(line_width, "Curve thickness")
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
                            geometry,
                            missing_values,
                            fixed_seed,
                            seed,
                            time_window,
                            max_replication_time,
                            track_time_outputs,
                            track_detailed_outputs,
                            stall_rate,
                            non_recovering,
                            stall_duration,
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
                                label="Firing rate",
                            )
                            _validate_simulation_batch_selection(
                                speed_source,
                                label="Fork speed",
                            )
                            return (
                                _materialize_simulation_sources(
                                    firing_source,
                                    label="Firing rate",
                                    missing_policy=missing_policy,
                                ),
                                _materialize_simulation_sources(
                                    speed_source,
                                    label="Fork speed",
                                    missing_policy=missing_policy,
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
                                label="Firing rate",
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
                            firing_rate = prepared_sources[0].values[
                                case.item_indices[0]
                            ]
                            fork_speed = prepared_sources[1].values[
                                case.item_indices[1]
                            ]
                            paired_sizes = {
                                size
                                for size in (
                                    _simulation_spatial_size(firing_rate),
                                    _simulation_spatial_size(fork_speed),
                                )
                                if size is not None
                            }
                            if len(paired_sizes) > 1:
                                sizes = ", ".join(
                                    f"{size:,}" for size in sorted(paired_sizes)
                                )
                                raise ValueError(
                                    f"{dataset_name}: firing-rate and fork-speed spatial "
                                    f"axes disagree ({sizes} bins)."
                                )
                            region_end_value = (
                                region_start_value
                                + next(iter(paired_sizes)) * resolution_x_value
                                if paired_sizes and not use_selected_end
                                else configured_region_end
                            )
                            if use_selected_end:
                                firing_rate = _trim_simulation_profile(
                                    firing_rate, start=region_start_value,
                                    end=configured_region_end, resolution=resolution_x_value,
                                )
                                fork_speed = _trim_simulation_profile(
                                    fork_speed, start=region_start_value,
                                    end=configured_region_end, resolution=resolution_x_value,
                                )
                            paired_values.append(
                                (firing_rate, fork_speed, region_end_value)
                            )
                        shared_request = dict(
                            resolution_x=resolution_x_value,
                            resolution_t=_number(resolution_t, "Time step"),
                            region_start_kb=region_start_value,
                            replicates=_integer(replicates, "Replicates"),
                            geometry=str(geometry.value),
                            time_window=_number(time_window, "Time window"),
                            max_replication_time=_number(
                                max_replication_time,
                                "Maximum replication time",
                            ),
                            missing_values=str(missing_values.value),
                            seed=_integer(seed, "Seed") if fixed_seed.value else None,
                            stall_rate_per_kb=_number(
                                stall_rate,
                                "Stall rate (stalls/kb)",
                            ),
                            stall_duration=(
                                math.inf
                                if non_recovering.value
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
                        )
                        requests = tuple(
                            ConstantSimulationRequest(
                                firing_rate=firing_rate,
                                fork_speed=fork_speed,
                                region_end_kb=request_region_end,
                                **shared_request,
                            )
                            for firing_rate, fork_speed, request_region_end in paired_values
                        )
                        # Validate the entire batch before the first scientific
                        # worker starts, so a mismatch cannot leave a partially
                        # published set of results.
                        for case, request in zip(
                            cases,
                            requests,
                            strict=True,
                        ):
                            try:
                                validate_constant_simulation(request)
                            except (TypeError, ValueError) as exc:
                                raise ValueError(
                                    f"{case.identity.display(_simulation_name_style())}: "
                                    f"{exc}"
                                ) from exc
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
                            dataset_name = case.identity.display(_simulation_name_style())
                            check_cancelled()
                            progress_state = _ProgressState()

                            def report_batch_progress(
                                value: float,
                                message: str,
                                *,
                                dataset_index: int = index,
                                name: str = dataset_name,
                                state: _ProgressState = progress_state,
                            ) -> None:
                                state.report(
                                    (dataset_index + max(0.0, min(1.0, float(value))))
                                    / batch_size,
                                    f"Dataset {dataset_index + 1}/{batch_size} · "
                                    f"{name} · {message}",
                                )

                            status.text = (
                                f"Dataset {index + 1}/{batch_size} · "
                                f"{dataset_name} · Preparing simulation..."
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
                                + ", ".join(
                                    _simulation_display_name(item)
                                    for item in truncated
                                )
                                + ".",
                                type="warning",
                            )
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
                        cached_payload: Any = (
                            latest_result
                            if len(latest_items) == 1
                            else _SimulationBatchPayload(latest_items)
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

                with action_dock():
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
                                            for label, spec in _TIME_OUTPUTS.items()
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
                                ("Replicates", ""),
                                ("Resolution", ""),
                                ("Region", ""),
                                ("Geometry", ""),
                            )
                        )
                replicate_stats = stats["Replicates"]
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
                                dataset_select.tooltip(
                                    latest_items[selected_index].identity.full
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
                                    color_surface.tooltip(item.identity.full)
                                    curve_type = ui.select(
                                        CURVE_TYPES,
                                        value=batch_line_types.get(
                                            item.key, str(plot_settings.line_type.value or "solid")
                                        ),
                                        label="Curve type",
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
                    on_remove=lambda _controls: remove_simulation_result(
                        latest_items[0].key if latest_items else ""
                    ),
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
                    with (
                        ui.element("div")
                        .classes("w-full")
                        .mark("simulation-batch-raw-surface") as batch_raw_container
                    ):

                        @ui.refreshable
                        def render_batch_raw_files() -> None:
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
                                            retained.file_type = str(control.value or "NPY")

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

                def standard_deviation_available() -> bool:
                    if latest_result is None:
                        return False
                    _profiles, standard_deviation = _replication_profile_arrays(latest_result)
                    return standard_deviation is not None

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
                    available_profiles = available_profile_count()
                    allow_individual_profiles = (
                        selected == "Replication timing"
                        and requested_view == "Default"
                        and available_profiles > 0
                        and not combined_preview_active()
                    )
                    allow_standard_deviation = (
                        selected == "Replication timing"
                        and requested_view == "Default"
                        and standard_deviation_available()
                        and not combined_preview_active()
                    )
                    if allow_individual_profiles:
                        maximum = min(_MAX_PREVIEW_PROFILES, available_profiles)
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
                        histogram = requested_view == "Histogram"
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
                        allow_individual_profiles=allow_individual_profiles,
                        allow_standard_deviation=allow_standard_deviation,
                    )
                    # A combined batch uses one colour control per dataset; the
                    # single-series colour field would otherwise be ambiguous.
                    if combined_preview_active():
                        plot_settings.primary_color_surface.visible = False

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
                            labels.extend(_TIME_OUTPUTS)
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
                            for label, spec in _TIME_OUTPUTS.items()
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
                    if selected_profile_count() is not None:
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
                    if combined_preview_active():
                        render_latest_result()

                current_session_cache().register_page_refresher(
                    "simulation",
                    refresh_simulation_names,
                )
