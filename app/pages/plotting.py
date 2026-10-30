"""Interactive profile plotting, smoothing, and correlation analysis."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import CancelledError
from dataclasses import dataclass, field, replace
from pathlib import PurePath
from threading import Event
from typing import Any

import numpy as np
from nicegui import run, ui

from app.export_settings import get_export_settings
from app.data_filter_controls import DataFilterControls
from app.data_transform_controls import DataTransformControls
from app.icons import PLOT_ICON_SVG
from app.mapping_uploads import UploadInspection, inspect_upload_event, materialize_upload
from app.notifications import notify
from app.statistics_exports import HistogramStatisticsExport, StatisticsExportPreference
from app.page_activity import current_page_activity
from app.pages.common import (
    DATA_ICON,
    action_dock,
    data_upload,
    append_raw_uploads,
    expansion,
    page_intro,
    panel,
    panel_title,
    plot_action_button,
    preview_plot,
    raw_data_open_menu,
    raw_output_file_row,
)
from app.preview_plotting import (
    CURVE_TYPES,
    axis_limits_from_controls,
    HistogramBinSynchronizer,
    PreviewLine,
    automatic_y_limits,
    bind_preview_settings_updates,
    build_preview_plot_options,
    compact_color_input,
    histogram_x_limits_from_controls,
    plotting_settings_ready,
    preview_settings_controls,
    preview_settings_markers,
    render_preview_chart,
    sync_preview_settings_visibility,
)
from app.result_naming import (
    PLOTTING_FILE_PREFIX,
    compact_export_stem,
    unique_compact_export_stems,
)
from app.session_cache import cache_artifact, cache_upload, current_session_cache
from app.raw_data_transfer import RawInputTarget, open_raw_artifact
from app.simulation_exports import ExportArtifact, build_chart_capture_export
from dnascape.workflows.analysis import (
    CORRELATION_METHODS, AnalysisDataset, AnalysisReport, SmoothingSettings,
    analyse_profiles, export_correlation_report, export_profile_data, smooth_profile,
)
from dnascape.workflows.filtering import DataFilterSettings
from dnascape.workflows.exports import safe_export_filename
from dnascape.workflows.analysis_transforms import analysis_values, transform_profiles, analysis_mask_artifact, profile_key

_PROFILE_ACCEPT = ".csv,.tsv,.txt,.npy,.npv,.npz,.gz"
_DEPENDENCE_LABELS = {
    "x": "Space",
    "t": "Time",
    "xt": "Space-time",
}
_NATURAL_SELECTION = "__natural_profile__"
_MATRIX_SELECTION = "__all_columns__"
_LINE_COLOURS = (
    "#b03412",
    "#2596be",
    "#3f6da8",
    "#8b6f37",
    "#6f5b8f",
    "#4f7d3a",
    "#a64f70",
    "#4f737f",
)


@dataclass
class _ProfileSourceState:
    source_id: int
    inspection: UploadInspection | None = None
    selection: tuple[str, ...] = ()
    array_cache: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass(frozen=True)
class _PlotSeries:
    series_id: tuple[int, str]
    name: str
    form: str
    values: np.ndarray
    full_identity: str = ""
    transformed: bool = False
    mask: np.ndarray | None = field(default=None, repr=False, compare=False)
    analysis_on_mask_only: bool = False
    source: UploadInspection | None = field(default=None, repr=False, compare=False)
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class _TransformationSettings:
    data_filter: DataFilterSettings
    target: tuple[int, str] | None = None
    keep_originals: bool = False
    analysis_on_mask_only: bool = False


@dataclass
class _CurveStyle:
    color: str
    width: float = 1.8
    line_type: str | None = None


@dataclass
class _SeparatePlotContext:
    """Controls and chart owned by one independently presented profile."""

    group: tuple[_PlotSeries, ...]
    controls: Any
    chart: Any
    y_limits_automatic: bool = True
    updating_y_limits: bool = False
    histogram_bin_sync: HistogramBinSynchronizer | None = None


@dataclass
class _SeparatePlotSettingsSnapshot:
    """Persisted values for one source profile's independent plot controls."""

    values: dict[str, Any]
    y_limits_automatic: bool = True
    histogram_basis: str = "bins"
    histogram_x_limits_automatic: bool = True
    automatic_axes: frozenset[str] = frozenset({"x", "z"})


_SEPARATE_SETTING_VALUE_FIELDS = (
    "view",
    "title",
    "x_axis_label",
    "y_axis_label",
    "primary_color",
    "secondary_color",
    "heatmap_palette",
    "log_x",
    "invert_x",
    "log_z",
    "invert_z",
    "line_width",
    "line_type",
    "show_fill", "fill_side",
    "aspect_ratio", "circular_combine", "circular_track_height", "circular_inner_radius",
    "histogram_bins",
    "histogram_bin_size",
    "histogram_touching",
    "kde_bandwidth", "histogram_grouped", "show_hover",
    "histogram_boundary",
    "show_individual_profiles",
    "individual_profile_count",
    "show_standard_deviation",
    "histogram_x_minimum",
    "histogram_x_maximum",
    "show_histogram_statistics",
    "export_histogram_statistics",
    "histogram_statistics_position",
    "show_legend",
    "legend_position",
    "show_grid",
    "density",
    "colorbar",
    "log_y",
    "invert_y",
    "y_minimum",
    "y_maximum",
    "x_minimum", "x_maximum", "z_minimum", "z_maximum",
)


def _plotting_setting_markers(*, separate_index: int | None = None) -> dict[str, str]:
    """Return stable shared or per-plot markers for the common control factory."""

    return preview_settings_markers(
        "plotting",
        qualifier=(
            None if separate_index is None else f"separate-{separate_index}"
        ),
        overrides={
            "container": "plotting-settings",
            "primary_color": "plotting-primary-color",
            "secondary_color": "plotting-secondary-color",
        },
    )


def _infer_profile_form(values: Any) -> str:
    array = np.asarray(values)
    if array.size <= 1:
        return "scalar"
    if array.ndim == 1:
        return "x"
    if array.ndim == 2 and array.shape[0] == 1:
        return "t"
    if array.ndim == 2:
        return "xt"
    return "array"


def _file_identity(name: str) -> str:
    path = PurePath(name)
    if path.suffix.casefold() == ".gz":
        path = path.with_suffix("")
    return path.stem.replace("_", " ").strip() or "Profile"


def _full_series_identity(inspection_name: str, token: str) -> str:
    """Retain source context for Full filenames without changing plot labels."""

    source = _file_identity(inspection_name)
    if token in {_NATURAL_SELECTION, _MATRIX_SELECTION}:
        return source
    selected = str(token).strip() or "Profile"
    if selected.casefold() == source.casefold():
        return source
    return f"{source} · {selected}"


def _plot_download_filenames(
    groups: Sequence[Sequence[_PlotSeries]],
) -> tuple[str, ...]:
    """Return consistent, collision-safe defaults for rendered profile plots."""

    use_full_names = get_export_settings().result_name_style == "full"
    identities = tuple(
        " + ".join(
            (
                item.full_identity
                if use_full_names and item.full_identity
                else item.name
            ).strip()
            or "Profile"
            for item in group
        )
        or "Profile"
        for group in groups
    )
    concise = tuple(
        compact_export_stem(
            identity,
            fallback=f"profile-{index}",
            maximum=96,
        )
        for index, identity in enumerate(identities, start=1)
    )
    unique = unique_compact_export_stems(
        concise,
        fallback="profile",
        maximum=96,
    )
    return tuple(
        f"{PLOTTING_FILE_PREFIX}_{identity}"
        for identity in unique
    )


def _npz_values(state: _ProfileSourceState, key: str) -> np.ndarray:
    if key in state.array_cache:
        return state.array_cache[key]
    inspection = state.inspection
    if inspection is None:
        raise ValueError("Choose a profile file first.")
    values = materialize_upload(
        replace(inspection, expected_form="array"),
        array_key=key,
        missing_policy="zero",
    )
    state.array_cache[key] = values
    return values


def _profile_array_keys(state: _ProfileSourceState) -> tuple[str, ...]:
    inspection = state.inspection
    if inspection is None:
        return ()
    return tuple(
        key
        for key in inspection.array_keys
        if _infer_profile_form(_npz_values(state, key)) in _DEPENDENCE_LABELS
    )


def _source_selection_options(state: _ProfileSourceState) -> Mapping[str, str] | None:
    inspection = state.inspection
    if inspection is None:
        return None
    if inspection.array_keys:
        return {key: key for key in _profile_array_keys(state)}

    columns = inspection.numeric_columns
    shape = tuple(inspection.shape or ())
    row_vector = bool(
        inspection.inferred_form == "t" and shape and shape[0] == 1 and len(columns) > 1
    )
    if row_vector or not columns:
        return None
    if inspection.inferred_form == "xt" and len(columns) > 1:
        return {
            _MATRIX_SELECTION: "All columns (space-time)",
            **{column: column for column in columns},
        }
    return {column: column for column in columns}


def _selection_form(state: _ProfileSourceState, token: str) -> str:
    inspection = state.inspection
    if inspection is None:
        return "unknown"
    if token == _NATURAL_SELECTION:
        return str(inspection.inferred_form)
    if token == _MATRIX_SELECTION:
        return "xt"
    if inspection.array_keys:
        return _infer_profile_form(_npz_values(state, token))
    return "x"


def _set_default_selection(state: _ProfileSourceState) -> None:
    inspection = state.inspection
    if inspection is None:
        state.selection = ()
        return
    options = _source_selection_options(state)
    if options is None:
        form = str(inspection.inferred_form)
        if form not in _DEPENDENCE_LABELS:
            raise ValueError(
                "Analysis requires a space, time, or space-time profile; "
                f"the selected data are {form}."
            )
        state.selection = (_NATURAL_SELECTION,)
        return
    if not options:
        raise ValueError("The selected archive contains no plottable profiles.")
    if _MATRIX_SELECTION in options:
        state.selection = (_MATRIX_SELECTION,)
        return
    preferred = inspection.default_array_key if inspection.array_keys else inspection.default_column
    selected = preferred if preferred is not None and preferred in options else next(iter(options))
    state.selection = (selected,)


def _normalize_multi_selection(
    state: _ProfileSourceState,
    selected: Sequence[str],
) -> tuple[str, ...]:
    options = _source_selection_options(state) or {}
    normalized = tuple(dict.fromkeys(token for token in selected if token in options))
    matrix_tokens = tuple(token for token in normalized if _selection_form(state, token) == "xt")
    if not matrix_tokens or len(normalized) == 1:
        return normalized

    previous = set(state.selection)
    newly_selected_matrix = next(
        (token for token in matrix_tokens if token not in previous),
        None,
    )
    if newly_selected_matrix is not None:
        return (newly_selected_matrix,)
    return tuple(token for token in normalized if token not in matrix_tokens)


def _materialize_source(
    state: _ProfileSourceState,
    *,
    missing_policy: str = "error",
) -> list[_PlotSeries]:
    inspection = state.inspection
    if inspection is None:
        return []
    series: list[_PlotSeries] = []
    for token in state.selection:
        form = _selection_form(state, token)
        if form not in _DEPENDENCE_LABELS:
            raise ValueError("The selected data does not contain a plottable profile.")
        column = None
        array_key = None
        if inspection.array_keys:
            array_key = token
        elif token not in {_NATURAL_SELECTION, _MATRIX_SELECTION}:
            column = token
        values = np.asarray(
            materialize_upload(
                replace(inspection, expected_form=form),
                column=column,
                array_key=array_key,
                missing_policy=missing_policy,
            ),
            dtype=float,
        )
        if token in {_NATURAL_SELECTION, _MATRIX_SELECTION}:
            name = _file_identity(inspection.name)
        else:
            name = token
        series.append(
            _PlotSeries(
                series_id=(state.source_id, token),
                name=name,
                form=form,
                values=values,
                full_identity=_full_series_identity(inspection.name, token),
            )
        )
    return series


def _unique_series_names(series: Sequence[_PlotSeries]) -> tuple[_PlotSeries, ...]:
    counts = Counter(item.name for item in series)
    occurrences: Counter[str] = Counter()
    normalized: list[_PlotSeries] = []
    for item in series:
        occurrences[item.name] += 1
        name = f"{item.name} {occurrences[item.name]}" if counts[item.name] > 1 else item.name
        normalized.append(replace(item, name=name))
    return tuple(normalized)


def _collect_plot_series(
    sources: Sequence[_ProfileSourceState], *, missing_policy: str = "error",
) -> tuple[_PlotSeries, ...]:
    collected = [item for source in sources for item in _materialize_source(source, missing_policy=missing_policy)]
    if not collected:
        raise ValueError("Upload and select at least one profile.")
    normalized = _unique_series_names(collected)
    incompatibility = _series_compatibility_error(normalized)
    if incompatibility is not None:
        raise ValueError(incompatibility)
    return normalized


def _prepare_analysis_profiles(
    sources: Sequence[_ProfileSourceState],
    smoothing: Mapping[tuple[int, str], SmoothingSettings],
    methods: Sequence[str] | None,
    cancel: Callable[[], None],
    transformations: _TransformationSettings | None = None,
    data_transformations: dict | None = None,
) -> tuple[tuple[_PlotSeries, ...], AnalysisReport | None]:
    cancel()
    transform_enabled = (transformations is not None and transformations.data_filter.enabled) or bool(data_transformations and data_transformations.get("enabled"))
    selected = _collect_plot_series(sources, missing_policy="ignore" if methods is not None or transform_enabled else "error")
    records = []
    for item in selected:
        cancel()
        source = next(source for source in sources if source.source_id == item.series_id[0])
        records.append(dict(id=item.series_id, name=item.name, form=item.form, full_identity=item.full_identity,
                            values=smooth_profile(item.values, smoothing.get(item.series_id, SmoothingSettings()), dependence=item.form),
                            source=source.inspection))
    region = (dict(settings=transformations.data_filter, target=transformations.target,
                   keep_originals=transformations.keep_originals, analysis_on_mask_only=transformations.analysis_on_mask_only)
              if transformations is not None else None)
    records = transform_profiles(records, data_transforms=data_transformations, region_transforms=region, cancel=cancel)
    processed = [_PlotSeries(series_id=item["id"], name=item["name"], form=item["form"], values=item["values"],
                             full_identity=item.get("full_identity", ""), transformed=item.get("transformed", False),
                             mask=item.get("mask"), analysis_on_mask_only=item.get("analysis_on_mask_only", False), source=item["source"], warnings=item.get("warnings", ()))
                 for item in records]
    processed = _unique_series_names(processed)
    report = None
    if methods is not None:
        report = analyse_profiles(
            [AnalysisDataset(item.name, analysis_values(vars(item)), item.form) for item in processed],
            methods=methods, cancel=cancel,
        )
    cancel()
    return tuple(processed), report


def _series_compatibility_error(series: Sequence[_PlotSeries]) -> str | None:
    """Return the one intrinsically unsupported dependence mixture, if present."""

    forms = {item.form for item in series}
    if "t" in forms and "xt" in forms:
        return (
            "Time profiles cannot be mixed with space-time heatmaps. "
            "Remove one of these data types before plotting."
        )
    return None


def _combined_series_compatible(series: Sequence[_PlotSeries]) -> bool:
    """Whether all series can share one Cartesian line plot."""

    if len(series) <= 1:
        return True
    forms = {item.form for item in series}
    return len(forms) == 1 and forms <= {"x", "t"}


def _plot_groups(
    series: Sequence[_PlotSeries],
    presentation: str,
) -> tuple[tuple[_PlotSeries, ...], ...]:
    """Group selected profiles into one overlay or independent plot panels."""

    if not series:
        return ()
    incompatibility = _series_compatibility_error(series)
    if incompatibility is not None:
        raise ValueError(incompatibility)
    if presentation == "Separate":
        return tuple((item,) for item in series)
    if presentation != "Combined":
        raise ValueError(f"Unknown plot presentation {presentation!r}.")
    if not _combined_series_compatible(series):
        raise ValueError(
            "These profiles need separate plots because they do not share "
            "one 1D dependence."
        )
    return (tuple(series),)


def _axis_labels(form: str, view: str, density: bool) -> tuple[str, str]:
    if form == "xt":
        return "Position (kb)", "Time (min)"
    if view == "Boxplot":
        return "", "Value"
    if view == "Histogram":
        return "Value", "Density" if density else "Count"
    return "Position (kb)", "Value"


def _numeric_value(control: Any, label: str) -> float:
    if control.value is None:
        raise ValueError(f"{label} is required.")
    try:
        value = float(control.value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a number.") from exc
    if not np.isfinite(value):
        raise ValueError(f"{label} must be finite.")
    return value


def _integer_value(control: Any, label: str) -> int:
    value = _numeric_value(control, label)
    if not value.is_integer():
        raise ValueError(f"{label} must be a whole number.")
    return int(value)


def render(_navigate: Callable[[str], None]) -> None:
    """Render staged profile controls and an explicitly committed preview."""

    sources: list[_ProfileSourceState] = []
    page_activity = current_page_activity()
    next_source_id = 1
    latest_series: tuple[_PlotSeries, ...] = ()
    result_charts: list[Any] = []
    separate_plot_contexts: list[_SeparatePlotContext] = []
    curve_styles: dict[tuple[int, str], _CurveStyle] = {}
    curve_names: dict[tuple[int, str], str] = {}
    box_positions: dict[tuple[int, str], float] = {}
    box_labels: dict[tuple[int, str], str] = {}
    histogram_opacities: dict[tuple[int, str], float] = {}
    dataset_settings_expanded: dict[str, bool] = {}
    secondary_y_limits_automatic = True
    separate_plot_settings: dict[tuple[int, str], _SeparatePlotSettingsSnapshot] = {}
    y_limits_automatic = True
    updating_y_limits = False
    running = False
    active_cancel: Event | None = None
    latest_analysis: AnalysisReport | None = None
    smoothing_controls: list[Any] = []
    value_transform_settings: dict[str, dict[str, Any]] = {}
    data_transformation_context: dict[str, Any] = {}
    data_transformation_controls = None
    report_file_format = {"value": "csv"}
    report_name = {"value": "analysis-report"}
    transformation_context: dict[str, Any] = {}
    transformation_targets: dict[str, tuple[int, str]] = {}
    transformations_locked = False
    transformed_export_preferences: dict[tuple[int, str], dict[str, Any]] = {}
    statistics_export_preferences: dict[Any, StatisticsExportPreference] = {}
    mask_export_preferences: dict[str, dict] = {}

    page_intro(
        "Analysis",
        "Explore DNAscape data",
        "Transform and plot profiles, smooth individual datasets, and compare their correlations.",
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

                    async def uploaded(event: Any) -> bool:
                        """Inspect and atomically append or replace one plotting source."""

                        nonlocal next_source_id
                        try:
                            inspection = await inspect_upload_event(
                                event,
                                expected_form="auto",
                            )
                            replaced_index = next(
                                (
                                    index
                                    for index, source in enumerate(sources)
                                    if source.inspection is not None
                                    and source.inspection.name == inspection.name
                                ),
                                None,
                            )
                            previous = (
                                sources[replaced_index]
                                if replaced_index is not None
                                else None
                            )
                            candidate = _ProfileSourceState(
                                source_id=(
                                    previous.source_id
                                    if previous is not None
                                    else next_source_id
                                ),
                                inspection=inspection,
                            )
                            _set_default_selection(candidate)
                            if previous is not None:
                                options = _source_selection_options(candidate)
                                allowed = (
                                    {_NATURAL_SELECTION}
                                    if options is None
                                    else set(options)
                                )
                                retained_selection = tuple(
                                    token
                                    for token in previous.selection
                                    if token in allowed
                                )
                                if retained_selection:
                                    candidate.selection = retained_selection
                            # Validate every selected profile before changing the
                            # retained collection. A bad same-name replacement
                            # therefore leaves the previous file fully usable.
                            _materialize_source(candidate, missing_policy="ignore")
                        except (OSError, RuntimeError, TypeError, ValueError) as exc:
                            notify(str(exc), type="negative")
                            return False

                        if replaced_index is None:
                            sources.append(candidate)
                            next_source_id += 1
                        else:
                            sources[replaced_index] = candidate
                        cache_upload(
                            inspection,
                            source="Analysis",
                            navigate_to="plotting",
                        )
                        staged_sources_changed()
                        return True

                    def cleared(event: Any) -> None:
                        """Mirror the shared uploader's single or bulk removal."""

                        name = str(getattr(event, "name", ""))
                        if name:
                            sources[:] = [
                                source
                                for source in sources
                                if source.inspection is None
                                or source.inspection.name != name
                            ]
                        elif removed_names := {
                            str(value)
                            for value in getattr(event, "names", ())
                            if str(value)
                        }:
                            sources[:] = [
                                source
                                for source in sources
                                if source.inspection is None
                                or source.inspection.name not in removed_names
                            ]
                        else:
                            sources.clear()
                        staged_sources_changed()

                    def render_profile_source(name: str) -> None:
                        source = next(
                            (
                                state for state in sources
                                if state.inspection is not None
                                and state.inspection.name == name
                            ),
                            None,
                        )
                        if source is None:
                            return
                        with (
                            ui.element("div")
                            .classes("dn-plot-profile-source")
                            .mark(f"plotting-profile-source-{source.source_id}")
                        ):

                            @ui.refreshable
                            def details_panel(state: _ProfileSourceState = source) -> None:
                                options = _source_selection_options(state)
                                # Omit selectors which cannot change anything:
                                # plain arrays and one-signal-column tables.
                                if options and len(options) > 1:
                                    selector = (
                                        ui.select(
                                            dict(options),
                                            value=list(state.selection),
                                            label="Profiles",
                                            multiple=True,
                                            clearable=True,
                                        )
                                        .props("dense options-dense use-chips")
                                        .classes(
                                            "dn-field dn-mapping-column-select "
                                            "dn-plot-profile-selector"
                                        )
                                        .mark(f"plotting-profile-selection-{state.source_id}")
                                    )
                                    selector._props["data-upload-name"] = name

                                    def selection_changed() -> None:
                                        raw = tuple(
                                            str(token) for token in (selector.value or [])
                                        )
                                        normalized = _normalize_multi_selection(state, raw)
                                        state.selection = normalized
                                        expected_value = list(normalized)
                                        if selector.value != expected_value:
                                            selector.value = expected_value
                                            selector.update()
                                        details_panel.refresh()
                                        staged_sources_changed()

                                    selector.on_value_change(selection_changed)

                                try:
                                    selected = _materialize_source(
                                        state,
                                        missing_policy="zero",
                                    )
                                except (OSError, TypeError, ValueError):
                                    selected = []
                                if not selected:
                                    dependence = "Not selected"
                                    size = "0"
                                else:
                                    forms = {item.form for item in selected}
                                    dependence = (
                                        _DEPENDENCE_LABELS.get(next(iter(forms)), "Unknown")
                                        if len(forms) == 1
                                        else "Mixed"
                                    )
                                    shapes = [
                                        " x ".join(str(value) for value in item.values.shape)
                                        or "1"
                                        for item in selected
                                    ]
                                    size = ", ".join(shapes)
                                with (
                                    ui.element("div")
                                    .classes("dn-mapping-detected-data dn-profile-metadata")
                                    .mark(f"plotting-detected-data-{state.source_id}")
                                ):
                                    ui.label(f"Dependence: {dependence}").classes(
                                        "dn-mapping-detected-summary "
                                        "dn-mapping-detected-dependence"
                                    )
                                    ui.label(f"Array size: {size}").classes(
                                        "dn-mapping-detected-summary"
                                    )

                            details_panel()

                    profile_uploader = data_upload(
                        accept=_PROFILE_ACCEPT,
                        accepted_formats="CSV, TSV, TXT, NPY, NPV, NPZ, GZ",
                        marker="plotting-data-upload",
                        show_accepted_formats=False,
                        on_upload=uploaded,
                        on_clear=cleared,
                        multiple=True,
                        max_files=None,
                        hide_when_loaded=False,
                        initial_names=tuple(
                            source.inspection.name
                            for source in sources
                            if source.inspection is not None
                        ),
                        file_details=render_profile_source,
                    )
                    compatibility_note = (
                        ui.label("")
                        .classes("dn-settings-help dn-plot-compatibility-note")
                        .mark("plotting-compatibility-note")
                    )
                    compatibility_note.visible = False

                with expansion("Analysis", icon_svg=PLOT_ICON_SVG, expanded=True):
                    analysis_enabled = ui.switch("Enable analysis", value=False).mark("analysis-enabled")
                    with ui.column().classes("dn-analysis-options w-full").bind_visibility_from(analysis_enabled, "value"):
                        ui.label("Correlations").classes("dn-setting-group-title")
                        with ui.row().classes("dn-inline-actions dn-analysis-methods"):
                            correlation_controls = {
                                method: ui.checkbox(label, value=True).mark(f"analysis-{method}")
                                for method, label in CORRELATION_METHODS.items()
                            }
                        for control in correlation_controls.values():
                            control.disable()
                        show_p_values = ui.checkbox("Show p-values", value=False).mark("analysis-show-p-values")

                def data_transform_sources():
                    profiles = _unique_series_names([
                        item for source in sources for item in _materialize_source(source, missing_policy="ignore")
                    ])
                    target = data_transformation_context["target"].value if "target" in data_transformation_context else "all"
                    return [item for item in profiles if target == "all" or profile_key(item.series_id) == target]

                value_controls = ui.column().classes("w-full")
                with value_controls:
                    @ui.refreshable
                    def render_smoothing_controls() -> None:
                        smoothing_controls.clear()
                        profiles = data_transform_sources()
                        combining = bool(data_transformation_controls and data_transformation_controls.combine.value and len(profiles) > 1)
                        if profiles and "pending" in value_transform_settings:
                            value_transform_settings.setdefault(profile_key(profiles[0].series_id), value_transform_settings.pop("pending"))
                        rows = [("combined", "Combined")] if combining else [(profile_key(item.series_id), item.name) for item in profiles]
                        for index, (key, name) in enumerate(rows or [("pending", "Dataset")], 1):
                            settings = value_transform_settings.setdefault(key, {})
                            smoothing = settings.get("smoothing", SmoothingSettings())
                            with ui.column().classes("dn-analysis-smoothing-row w-full"):
                                if len(rows) > 1:
                                    ui.label(f"Dataset {index}").classes("dn-upload-formats")
                                with ui.element("div").classes("dn-data-manager-polishing-grid dn-data-manager-transform-grid"):
                                    with ui.column().classes("dn-data-manager-polishing-column"):
                                        method = ui.select({"none": "Off", "moving_average": "Moving average", "gaussian": "Gaussian"},
                                            value=smoothing.method, label="Smoothing").props("dense options-dense").classes("dn-field").mark(f"analysis-smoothing-method-{index}")
                                        window = ui.number("Window (bins)", value=smoothing.window, min=1, max=100001, step=1).props("dense").classes("dn-field").mark(f"analysis-smoothing-window-{index}")
                                        sigma = ui.number("Sigma (bins)", value=smoothing.sigma, min=.01, max=10000, step=.1).props("dense").classes("dn-field").mark(f"analysis-smoothing-sigma-{index}")
                                    with ui.column().classes("dn-data-manager-polishing-column"):
                                        scaling = ui.select({"none": "None", "interval": "Target interval", "multiplier": "Multiplier", "center": "Centre at zero", "zscore": "Z-score", "maxabs": "Maximum absolute value"},
                                            value=settings.get("scaling", "none"), label="Value scaling").props("dense options-dense").classes("dn-field").mark(f"analysis-value-scaling-{index}")
                                        multiplier = ui.number("Multiplier", value=settings.get("multiplier", 1.)).props("dense").classes("dn-field").mark(f"analysis-scale-multiplier-{index}")
                                        with ui.element("div").classes("dn-data-manager-range-grid") as interval:
                                            minimum = ui.number("Minimum", value=settings.get("minimum", 0.)).props("dense").classes("dn-field").mark(f"analysis-target-minimum-{index}")
                                            maximum = ui.number("Maximum", value=settings.get("maximum", 1.)).props("dense").classes("dn-field").mark(f"analysis-target-maximum-{index}")
                                    invert = ui.checkbox("Invert", value=settings.get("invert", False)).classes("dn-data-manager-inline-toggle").mark(f"analysis-invert-{index}")
                                window.set_visibility(smoothing.method == "moving_average")
                                sigma.set_visibility(smoothing.method == "gaussian")
                                multiplier.set_visibility(scaling.value == "multiplier")
                                interval.set_visibility(scaling.value == "interval")
                                controls = (method, window, sigma, scaling, multiplier, minimum, maximum, invert)
                                smoothing_controls.extend(controls)
                                for control in controls:
                                    control.set_enabled(not running)

                                def changed(_event=None, *, key=key, method=method, window=window, sigma=sigma,
                                            scaling=scaling, multiplier=multiplier, minimum=minimum, maximum=maximum, invert=invert, interval=interval):
                                    value_transform_settings[key] = dict(smoothing=SmoothingSettings(method.value, window.value, sigma.value),
                                        scaling=scaling.value, multiplier=multiplier.value, minimum=minimum.value, maximum=maximum.value, invert=bool(invert.value))
                                    window.set_visibility(method.value == "moving_average")
                                    sigma.set_visibility(method.value == "gaussian")
                                    multiplier.set_visibility(scaling.value == "multiplier")
                                    interval.set_visibility(scaling.value == "interval")
                                    invalidate_analysis_report()
                                for control in controls:
                                    control.on_value_change(changed)
                    render_smoothing_controls()

                def data_transform_header():
                    data_transformation_context["target"] = ui.select({"all": "All datasets"}, value="all", label="Dataset").props("dense options-dense").classes("dn-field").mark("analysis-data-transform-target")

                def data_transform_changed():
                    invalidate_analysis_report()
                    render_smoothing_controls.refresh()
                    sync_transformation_targets(refresh_data=False)
                    sync_presentation_options()

                data_transformation_controls = DataTransformControls(
                    lambda: [(item.series_id, item.name) for item in data_transform_sources()],
                    value_controls=value_controls, marker_prefix="analysis-data-transform", source_label="Dataset",
                    header=data_transform_header, on_change=data_transform_changed,
                )

                def data_target_changed():
                    data_transformation_controls.refresh()
                    data_transform_changed()
                data_transformation_context["target"].on_value_change(data_target_changed)

                def transformation_header() -> None:
                    transformation_context["target"] = ui.select(
                        {"all": "All datasets"}, value="all", label="Dataset",
                    ).props("dense options-dense").classes("dn-field").mark("analysis-transform-target")
                    transformation_context["keep_originals"] = ui.checkbox(
                        "Keep original profiles", value=False,
                    ).mark("analysis-transform-keep-originals")

                def transformation_switches():
                    transformation_context["analysis_on_mask_only"] = ui.switch("Analysis on mask only", value=False).mark("analysis-transform-mask-only")

                transformation_controls = DataFilterControls(
                    lambda: (invalidate_analysis_report(), sync_presentation_options()),
                    marker_prefix="analysis-transform", header=transformation_header,
                    switches_footer=transformation_switches, allow_export=True, allow_complement=True,
                    profile_units="bp", align_modulation_from_start=True,
                )

                with action_dock(on_code=lambda: download_plotting_code()):
                    plot_button = plot_action_button()
                    plot_button.mark("action-plot", "action-analyse")
                    action_label = next(child for child in plot_button.default_slot.children if "dn-action-label" in child._classes)
                    action_symbol = next(child for child in plot_button.default_slot.children if "dn-action-plot-icon" in child._classes)
                    action_label.mark("analysis-action-label")

        with ui.column().classes("dn-preview-stack"):
            with panel(extra_classes="dn-preview-card"):
                with ui.row().classes("dn-preview-output-bar"):
                    panel_title("Preview")
                    presentation = (
                        ui.select(
                            ["Combined", "Separate"],
                            value="Combined",
                            label="Layout",
                        )
                        .props("dense options-dense")
                        .classes("dn-field dn-preview-output-control")
                        .mark("plotting-presentation")
                    )
                with (
                    ui.element("div")
                    .classes("w-full dn-plot-result-list")
                    .mark("plotting-result-container") as result_slot
                ):
                    preview_plot("No plot yet", kind="plotting")

                shared_settings_surface = (
                    ui.element("div")
                    .classes("w-full")
                    .mark("plotting-shared-settings")
                )
                with shared_settings_surface:
                    plot_settings = preview_settings_controls(
                        initial_title="Profile",
                        initial_x_label="Position (kb)",
                        initial_y_label="Value",
                        on_reset_y_limits=lambda: reset_y_limits(),
                        allow_dual_y_axes=True,
                        on_reset_secondary_y_limits=lambda: reset_secondary_y_limits(),
                        markers=_plotting_setting_markers(),
                    )

                @ui.refreshable
                def render_curve_settings() -> None:
                    if str(presentation.value or "Combined") != "Combined":
                        return
                    view = str(plot_settings.view.value or "Default")
                    prefix = {"Default": "curve", "Boxplot": "box", "Histogram": "histogram"}.get(view)
                    if prefix is None:
                        return
                    curve_series = [
                        item
                        for item in latest_series
                        if item.form in {"x", "t"}
                    ]
                    if len(curve_series) < 2:
                        return
                    with ui.expansion(
                        {"Default": "Curves", "Boxplot": "Boxes", "Histogram": "Histograms"}[view],
                        icon="palette", value=dataset_settings_expanded.get(prefix, False),
                    ).props("dense").classes("dn-expansion dn-preview-settings-section dn-plot-curve-settings").mark(f"plotting-{prefix}-settings") as dataset_section:
                        dataset_section.on_value_change(lambda event, key=prefix: dataset_settings_expanded.update({key: bool(event.value)}))
                        with ui.column().classes("dn-expansion-content dn-plot-curve-list"):
                            for index, item in enumerate(curve_series):
                                style = curve_styles.setdefault(
                                    item.series_id,
                                    _CurveStyle(_LINE_COLOURS[index % len(_LINE_COLOURS)]),
                                )
                                with ui.element("div").classes("dn-plot-curve-row" + (" dn-plot-distribution-row" if view != "Default" else "")):
                                    name = (
                                        ui.input("Name", value=curve_names.get(item.series_id, item.name))
                                        .props("dense")
                                        .classes("dn-field dn-plot-curve-name")
                                        .mark(f"plotting-{prefix}-name-{index + 1}")
                                    )

                                    def name_changed(
                                        _event: Any = None, *,
                                        series_id: tuple[int, str] = item.series_id,
                                        control: Any = name,
                                    ) -> None:
                                        value = str(control.value or "").strip()
                                        if value:
                                            curve_names[series_id] = value
                                        else:
                                            curve_names.pop(series_id, None)
                                        update_combined_plot()
                                        render_smoothing_controls.refresh()
                                        render_analysis_report.refresh()

                                    name.on_value_change(name_changed)
                                    color_surface, color = compact_color_input(
                                        "Color",
                                        style.color,
                                        marker=f"plotting-{prefix}-color-{index + 1}",
                                    )
                                    color_surface.classes("dn-plot-curve-color")

                                    def color_changed(_event=None, *, series_id=item.series_id, control=color) -> None:
                                        curve_styles[series_id] = replace(curve_styles[series_id], color=str(control.value or _LINE_COLOURS[0]))
                                        update_combined_plot()

                                    color.on_value_change(color_changed)
                                    if view == "Histogram":
                                        opacity = ui.number("Opacity", value=histogram_opacities.get(item.series_id, .6), min=0, max=1, step=.05).props("dense").classes("dn-field dn-field-compact").mark(f"plotting-histogram-opacity-{index + 1}")

                                        def opacity_changed(_event=None, *, series_id=item.series_id, control=opacity):
                                            try:
                                                value = _numeric_value(control, "Opacity")
                                            except ValueError:
                                                return
                                            if 0 <= value <= 1:
                                                histogram_opacities[series_id] = value
                                                update_combined_plot()

                                        opacity.on_value_change(opacity_changed)
                                        if not plot_settings.density.value:
                                            continue
                                    width = (
                                        ui.number(
                                            label="Thickness",
                                            value=style.width,
                                            min=0.2,
                                            max=8.0,
                                            step=0.1,
                                        )
                                        .props("dense")
                                        .classes("dn-field dn-field-compact")
                                        .mark(f"plotting-{prefix}-width-{index + 1}")
                                    )
                                    curve_type = None if view == "Boxplot" else ui.select(
                                        CURVE_TYPES,
                                        value=style.line_type or str(plot_settings.line_type.value or "solid"),
                                        label="Type",
                                    ).props("dense options-dense").classes("dn-field dn-field-compact").mark(
                                        f"plotting-{prefix}-type-{index + 1}"
                                    )

                                    def style_changed(
                                        _event: Any = None,
                                        *,
                                        series_id: tuple[int, str] = item.series_id,
                                        color_control: Any = color,
                                        width_control: Any = width,
                                        type_control: Any = curve_type,
                                    ) -> None:
                                        numeric_width = width_control.value
                                        if numeric_width is None:
                                            return
                                        try:
                                            parsed_width = float(numeric_width)
                                        except (TypeError, ValueError):
                                            return
                                        if not np.isfinite(parsed_width) or parsed_width <= 0:
                                            return
                                        curve_styles[series_id] = _CurveStyle(
                                            str(color_control.value or _LINE_COLOURS[0]),
                                            parsed_width,
                                            str(type_control.value or "solid") if type_control is not None else curve_styles[series_id].line_type,
                                        )
                                        update_combined_plot()

                                    width.on_value_change(style_changed)
                                    if curve_type is not None:
                                        curve_type.on_value_change(style_changed)
                                    if view == "Boxplot":
                                        if item.series_id not in box_positions:
                                            available = float(index + 1)
                                            while available in box_positions.values():
                                                available += 1
                                            box_positions[item.series_id] = available
                                        position = ui.number("X position", value=box_positions[item.series_id]).props("dense").classes("dn-field").mark(f"plotting-box-position-{index + 1}")
                                        label = ui.input("X label", value=box_labels.get(item.series_id, ""), placeholder="Automatic").props("dense clearable").classes("dn-field dn-plot-box-label").mark(f"plotting-box-label-{index + 1}")

                                        def box_changed(_event=None, *, series_id=item.series_id, position_control=position, label_control=label):
                                            try:
                                                value = _numeric_value(position_control, "X position")
                                            except ValueError:
                                                return
                                            if any(key != series_id and other == value for key, other in box_positions.items() if key in {entry.series_id for entry in latest_series}):
                                                notify("Choose a different X position for each box.", type="warning")
                                                return
                                            box_positions[series_id] = value
                                            box_labels[series_id] = str(label_control.value or "")
                                            update_combined_plot()

                                        position.on_value_change(box_changed)
                                        label.on_value_change(box_changed)

                with shared_settings_surface:
                    render_curve_settings()

                @ui.refreshable
                def render_transformed_data() -> None:
                    transformed = [item for item in latest_series if item.transformed]
                    with expansion("Raw data", icon=DATA_ICON, expanded=True, extra_classes="dn-output-section") as raw_section:
                        for index, item in enumerate(transformed, start=1):
                            default_name = compact_export_stem(item.name, fallback="profile") + "_transformed"
                            preferences = transformed_export_preferences.setdefault(item.series_id, {
                                "name": default_name, "type": "TSV", "compress": False,
                            })

                            async def download_transformed(controls, retained=item, *, open_in=None):
                                try:
                                    artifact = await run.io_bound(
                                        export_profile_data, retained.values, dependence=retained.form,
                                        file_format=str(controls.file_type.value), filename=str(controls.file_name.value or "transformed_profile"),
                                        compress=bool(controls.compress.value),
                                    )
                                    if open_in is not None:
                                        await open_raw_artifact(artifact, open_in, _navigate)
                                        return
                                    cache_artifact(artifact, source="Analysis", kind="raw", detail="Transformed profile", navigate_to="plotting")
                                    ui.download(artifact.data, artifact.filename, artifact.media_type)
                                except (OSError, RuntimeError, TypeError, ValueError) as exc:
                                    notify(str(exc), type="negative")

                            row = raw_output_file_row(
                                file_name=preferences["name"], default_type=preferences["type"], default_compress=preferences["compress"],
                                on_download=download_transformed,
                                on_open=lambda controls, destination, handler=download_transformed: handler(controls, open_in=destination),
                                reserve_remove_space=True,
                                markers={"row": f"analysis-transform-result-{index}", "file_name": f"analysis-transform-name-{index}",
                                         "file_type": f"analysis-transform-format-{index}", "compression": f"analysis-transform-compress-{index}",
                                         "download": f"analysis-transform-download-{index}"},
                            )
                            for control, key in ((row.file_name, "name"), (row.file_type, "type"), (row.compress, "compress")):
                                control.on_value_change(lambda event, key=key, prefs=preferences: prefs.update({key: event.value}))
                        masks = [item for item in transformed if item.mask is not None] if transformation_controls.controls["enabled"].value and transformation_controls.controls["export"].value else []
                        for index, item in enumerate(masks, 1):
                            key = profile_key(item.series_id)
                            prefs = mask_export_preferences.setdefault(key, dict(filename=compact_export_stem(item.name, fallback="profile") + "_mask_positions",
                                file_format="TSV", compress=False, layout="positions"))
                            with ui.column().classes("w-full dn-filter-export-row").mark(f"analysis-mask-result-{index}"):
                                ui.label("Mask positions").classes("dn-setting-group-title")
                                async def download_mask(controls, retained=item, preference=prefs, *, open_in=None):
                                    try:
                                        artifact = await run.io_bound(analysis_mask_artifact, vars(retained), **preference)
                                        if open_in is not None:
                                            await open_raw_artifact(artifact, open_in, _navigate, preferred_input="filter_positions")
                                        else:
                                            ui.download(artifact.data, artifact.filename, artifact.media_type)
                                    except (OSError, RuntimeError, TypeError, ValueError) as exc:
                                        notify(str(exc), type="negative")
                                row = raw_output_file_row(file_name=prefs["filename"], default_type=prefs["file_format"], default_compress=prefs["compress"],
                                    reserve_remove_space=True, on_download=download_mask,
                                    on_open=lambda controls, destination, handler=download_mask: handler(controls, open_in=destination),
                                    markers={"row": f"analysis-mask-row-{index}", "file_name": f"analysis-mask-name-{index}", "file_type": f"analysis-mask-type-{index}",
                                             "compression": f"analysis-mask-compress-{index}", "download": f"analysis-mask-download-{index}"})
                                fields = row.file_type.parent_slot.parent
                                fields.classes(add="dn-filter-export-fields")
                                with fields:
                                    layout = ui.select({"positions": "Positions", "intervals": "Intervals"}, value=prefs["layout"], label="Export as").props("dense options-dense").classes("dn-field").mark(f"analysis-mask-layout-{index}")
                                layout.move(fields, target_index=1)
                                for control, name in ((row.file_name, "filename"), (row.file_type, "file_format"), (row.compress, "compress"), (layout, "layout")):
                                    control.on_value_change(lambda event, name=name, prefs=prefs: prefs.update({name: event.value}))
                        HistogramStatisticsExport(
                            plot_settings,
                            get_chart=lambda: result_charts[0] if latest_series and result_charts and presentation.value == "Combined" else None,
                            filename="analysis_statistics", marker="analysis-statistics", navigate=_navigate,
                            preference=statistics_export_preferences.setdefault(
                                "combined", StatisticsExportPreference("analysis_statistics"),
                            ),
                            section=raw_section, has_other_rows=lambda: bool(transformed),
                        )

                render_transformed_data()

                @ui.refreshable
                def render_analysis_report() -> None:
                    if not analysis_enabled.value:
                        return
                    with expansion("Analysis report", icon="table_chart", expanded=True,
                                   extra_classes="dn-preview-settings-section dn-analysis-report"):
                        if latest_analysis is None:
                            ui.label("Click Analyse to generate the report.").classes("dn-settings-help")
                            return
                        report = current_analysis_report()
                        assert report is not None
                        if not report.matrices:
                            ui.label("Select a correlation in Analysis, then click Analyse.").classes("dn-settings-help")
                            return
                        columns = [{"name": "dataset", "label": "Dataset", "field": "dataset", "align": "left"}]
                        for matrix in report.matrices:
                            columns.extend({
                                "name": f"{matrix.method}_{index}", "label": name,
                                "field": f"{matrix.method}_{index}", "align": "right",
                                "group": CORRELATION_METHODS[matrix.method], "groupStart": index == 0,
                                "showPValues": bool(show_p_values.value),
                            } for index, name in enumerate(report.names))
                        rows = []
                        for index, name in enumerate(report.names):
                            row = {"key": index, "dataset": name, "counts": {}, "p_values": {}}
                            for matrix in report.matrices:
                                for column, value in enumerate(matrix.coefficients[index]):
                                    key = f"{matrix.method}_{column}"
                                    row[key] = f"{value:.3f}" if np.isfinite(value) else "—"
                                    row["counts"][key] = int(matrix.sample_counts[index, column])
                                    p = matrix.p_values[index, column] if matrix.p_values is not None else np.nan
                                    row["p_values"][key] = (
                                        "—" if not np.isfinite(p) else "0" if p == 0 else f"{p:.2e}" if p < 0.001 else f"{p:.3f}"
                                    )
                            rows.append(row)
                        table = ui.table(columns=columns, rows=rows, row_key="key", pagination={"rowsPerPage": 0}).props(
                            "dense flat hide-bottom wrap-cells"
                        ).classes("dn-analysis-matrix w-full").mark("analysis-report-table")
                        table.add_slot("header", '''
                            <q-tr :props="props" class="dn-analysis-group-heading">
                                <q-th rowspan="2" class="dn-analysis-dataset text-left">Dataset</q-th>
                                <q-th v-for="col in props.cols.filter(c => c.groupStart)" :key="col.group"
                                      :colspan="props.cols.filter(c => c.group === col.group).length"
                                      class="dn-analysis-group-start text-center">{{ col.group }}</q-th>
                            </q-tr>
                            <q-tr :props="props" class="dn-analysis-column-heading">
                                <q-th v-for="col in props.cols.filter(c => c.name !== 'dataset')" :key="col.name"
                                      :class="{'dn-analysis-group-start': col.groupStart}" class="text-right">
                                    {{ col.label }}
                                </q-th>
                            </q-tr>
                        ''')
                        table.add_slot("body-cell", '''
                            <q-td :props="props" :class="{'dn-analysis-dataset': props.col.name === 'dataset',
                                                         'dn-analysis-group-start': props.col.groupStart}">
                                <span>{{ props.value }}</span>
                                <template v-if="props.col.name !== 'dataset'">
                                    <div v-if="props.col.showPValues" class="dn-analysis-p-value">
                                        p = {{ props.row.p_values[props.col.name] }}
                                    </div>
                                    <q-tooltip>Paired values: {{ props.row.counts[props.col.name] }}</q-tooltip>
                                </template>
                            </q-td>
                        ''')
                        with ui.element("div").classes("dn-download-file-row dn-analysis-report-download"):
                            ui.icon("description").classes("dn-download-file-icon")
                            with ui.element("div").classes("dn-raw-output-fields"):
                                file_name = (
                                    ui.input(label="Report name", value=report_name["value"])
                                    .props("dense").classes("dn-field").mark("analysis-report-name")
                                )
                                file_name.on_value_change(lambda: report_name.update(value=str(file_name.value or "")))
                                file_format = (
                                    ui.select({"csv": "CSV", "tsv": "TSV"}, value=report_file_format["value"], label="File type")
                                    .props("dense options-dense").classes("dn-field").mark("analysis-report-format")
                                )
                                file_format.on_value_change(lambda: report_file_format.update(value=str(file_format.value)))
                            ui.button(icon="download", on_click=download_analysis_report).props(
                                'flat round dense aria-label="Download analysis report"'
                            ).classes("dn-download-file-action dn-raw-output-download").mark("analysis-download-report").tooltip("Download")
                            raw_data_open_menu(lambda destination: download_analysis_report(open_in=destination), marker="analysis-open-report")
                            ui.button(icon="close", on_click=lambda: clear_plot_output()).props(
                                'flat round dense aria-label="Clear preview"'
                            ).classes("dn-download-file-action dn-raw-output-remove").mark("analysis-clear-preview").tooltip("Clear preview")

                render_analysis_report()

    def current_analysis_report() -> AnalysisReport | None:
        if latest_analysis is None:
            return None
        return replace(latest_analysis, names=tuple(curve_names.get(item.series_id, item.name) for item in latest_series))

    async def download_analysis_report(*, open_in: str | None = None) -> None:
        report = current_analysis_report()
        if report is None:
            return
        file_format = report_file_format["value"]
        data = export_correlation_report(report, file_format=file_format, include_p_values=bool(show_p_values.value), combined=True)
        filename = safe_export_filename(report_name["value"], file_format, default_stem="analysis-report")
        artifact = ExportArtifact(data, filename, "text/csv; charset=utf-8" if file_format == "csv" else "text/tab-separated-values; charset=utf-8")
        if open_in is not None:
            await open_raw_artifact(artifact, open_in, _navigate)
            return
        cache_artifact(artifact, source="Analysis", kind="raw", detail="Correlation report", navigate_to="plotting")
        ui.download(data, filename, artifact.media_type)

    def invalidate_analysis_report() -> None:
        nonlocal latest_analysis
        latest_analysis = None
        render_analysis_report.refresh()

    def analysis_enabled_changed() -> None:
        for control in correlation_controls.values():
            control.set_enabled(bool(analysis_enabled.value) and not running)
        sync_presentation_options()
        render_analysis_report.refresh()

    def sync_analysis_action() -> None:
        nonlocal transformations_locked
        stopping = running and active_cancel is not None and active_cancel.is_set()
        label = "Stopping…" if stopping else "Stop" if running else "Analyse" if analysis_enabled.value else "Plot"
        action_label.set_text(label)
        plot_button.props(f'aria-label="{label}"')
        action_symbol.set_content(
            '<svg class="dn-plot-icon" viewBox="0 0 24 24" aria-hidden="true"><path fill="currentColor" d="M6 6h12v12H6z"/></svg>'
            if running else PLOT_ICON_SVG
        )
        if running:
            plot_button.set_enabled(not stopping)
        analysis_enabled.set_enabled(not running)
        show_p_values.set_enabled(not running)
        for control in correlation_controls.values():
            control.set_enabled(bool(analysis_enabled.value) and not running)
        for control in smoothing_controls:
            control.set_enabled(not running)
        if transformations_locked != running:
            transformation_controls.set_locked(running)
            data_transformation_controls.set_locked(running)
            transformations_locked = running
        for control in (*transformation_context.values(), *data_transformation_context.values()):
            control.set_enabled(not running)
        if not running and transformation_controls.loading:
            plot_button.disable()

    def staged_forms() -> tuple[str, ...]:
        return tuple(
            _selection_form(source, token)
            for source in sources
            for token in source.selection
            if _selection_form(source, token) in _DEPENDENCE_LABELS
        )

    def preview_forms() -> tuple[str, ...]:
        """Describe the committed preview, falling back to staged data initially."""

        if latest_series:
            return tuple(item.form for item in latest_series)
        return staged_forms()

    def requested_plot_type(*, form: str, multiple: bool, controls: Any = None) -> str:
        if form == "xt":
            return "Heatmap"
        active_controls = plot_settings if controls is None else controls
        view = str(active_controls.view.value or "Default")
        if view == "Histogram":
            return "Histogram"
        if view == "Boxplot":
            return "Boxplot"
        return "Line"

    def staged_compatibility_error() -> str | None:
        forms = set(staged_forms())
        if "t" in forms and "xt" in forms:
            return (
                "Time profiles cannot be mixed with space-time heatmaps. "
                "Remove one of these data types."
            )
        return None

    def sync_presentation_options(*, sync_action: bool = True) -> None:
        forms = preview_forms()
        multiple = len(forms) > 1
        combined_compatible = (
            not multiple
            or (len(set(forms)) == 1 and set(forms) <= {"x", "t"})
        )
        options = ["Combined", "Separate"] if combined_compatible else ["Separate"]
        if list(presentation.options) != options:
            presentation.options = options
        if presentation.value not in options:
            presentation.value = options[0]
        presentation.visible = multiple
        presentation.update()

        incompatibility = staged_compatibility_error()
        compatibility_note.visible = incompatibility is not None
        compatibility_note.text = incompatibility or ""
        compatibility_note.update()
        if sync_action:
            if incompatibility is None and staged_forms():
                plot_button.enable()
            else:
                plot_button.disable()
            sync_analysis_action()

    def latest_groups() -> tuple[tuple[_PlotSeries, ...], ...]:
        return _plot_groups(
            latest_series,
            str(presentation.value or "Combined"),
        )

    def group_plot_type(
        group: Sequence[_PlotSeries],
        controls: Any = None,
    ) -> str:
        first = next(iter(group), None)
        if first is None:
            raise ValueError("A plot group cannot be empty.")
        return requested_plot_type(
            form=first.form,
            multiple=len(group) > 1,
            controls=controls,
        )

    def sync_staged_controls(*, reset_axes: bool = False) -> None:
        # Mapping's settings always describe its displayed result. Keep the
        # same contract here after the first Plot commit; newly staged files
        # must not change controls for a chart which is still on screen.
        forms = preview_forms()
        multiple = len(forms) > 1
        unique_forms = set(forms)
        form = forms[0] if forms and len(unique_forms) == 1 else "xt" if "xt" in forms else "x"
        separate = str(presentation.value or "Combined") == "Separate"
        plot_type = requested_plot_type(
            form=form,
            multiple=multiple and not separate,
        )
        sync_preview_settings_visibility(
            plot_settings,
            plot_type=plot_type,
            multiple_curves=multiple and not separate,
            allow_histogram=form != "xt",
            allow_boxplot=form != "xt",
        )
        dual_available = len(forms) == 2 and not separate and len(unique_forms) == 1 and form in {"x", "t"}
        plot_settings.dual_y_axes.visible = dual_available
        dual_active = dual_available and bool(plot_settings.dual_y_axes.value)
        plot_settings.secondary_y_limit_editor.visible = dual_active
        plot_settings.secondary_y_axis_label.visible = dual_active
        for control in (
            plot_settings.secondary_x_axis_label, plot_settings.secondary_x_limit_editor,
            plot_settings.secondary_x_axis_options, plot_settings.secondary_y_axis_options,
        ):
            control.visible = dual_active
        plot_settings.x_axis_caption.set_text("X1" if dual_active else "X")
        plot_settings.y_axis_caption.set_text("Y1" if dual_active else "Y")
        if plot_type == "Boxplot" and multiple and not separate:
            plot_settings.x_axis_options.visible = True
        for control, label in (
            (plot_settings.x_minimum, "X1 min" if dual_active else "X min"),
            (plot_settings.x_maximum, "X1 max" if dual_active else "X max"),
            (plot_settings.histogram_x_minimum, "X1 min" if dual_active else "X min"),
            (plot_settings.histogram_x_maximum, "X1 max" if dual_active else "X max"),
            (plot_settings.x_axis_label, "X1 label" if dual_active else "X-axis label"),
            (plot_settings.y_minimum, "Y1 min" if dual_active else "Y min"),
            (plot_settings.y_maximum, "Y1 max" if dual_active else "Y max"),
            (plot_settings.y_axis_label, "Y1 label" if dual_active else "Y label"),
        ):
            control.props(f'label="{label}"')
        # Multiple curves have individual styles in Curves; a single profile
        # uses the shared Style controls, as it does in Mapping.
        if plot_type != "Heatmap" and len(latest_series) > 1:
            plot_settings.primary_color_surface.visible = False
            plot_settings.secondary_color_surface.visible = False
            plot_settings.line_width.visible = False
            plot_settings.line_type.visible = False
        if reset_axes:
            if len(unique_forms) > 1:
                x_label, y_label = "Index (bin)", "Value"
            else:
                x_label, y_label = _axis_labels(
                    form,
                    str(plot_settings.view.value or "Default"),
                    bool(plot_settings.density.value),
                )
            plot_settings.x_axis_label.value = x_label
            plot_settings.secondary_x_axis_label.set_value(x_label)
            plot_settings.y_axis_label.value = y_label
            plot_settings.x_axis_label.update()
            plot_settings.y_axis_label.update()

    def clear_automatic_y_limits() -> None:
        nonlocal updating_y_limits, y_limits_automatic, secondary_y_limits_automatic
        y_limits_automatic = True
        secondary_y_limits_automatic = True
        updating_y_limits = True
        try:
            plot_settings.y_minimum.value = None
            plot_settings.y_maximum.value = None
            plot_settings.y_minimum.update()
            plot_settings.y_maximum.update()
            for control in (plot_settings.secondary_y_minimum, plot_settings.secondary_y_maximum):
                control.set_value(None)
        finally:
            updating_y_limits = False

    def staged_sources_changed() -> None:
        clear_automatic_y_limits()
        sync_presentation_options()
        sync_staged_controls(reset_axes=True)
        render_smoothing_controls.refresh()
        sync_transformation_targets()

    def sync_transformation_targets(*, refresh_data=True) -> None:
        profiles = _unique_series_names([
            item for source in sources for item in _materialize_source(source, missing_policy="ignore")
        ])
        data_options = {"all": "All datasets", **{profile_key(item.series_id): item.name for item in profiles}}
        data_target = data_transformation_context["target"]
        data_target.set_options(data_options, value=data_target.value if data_target.value in data_options else "all")
        if refresh_data:
            data_transformation_controls.refresh()
            render_smoothing_controls.refresh()
        if data_transformation_controls.combining and profiles:
            profiles = [replace(profiles[0], series_id=(profiles[0].series_id[0], "__combined__"), name="Combined")]
        transformation_targets.clear()
        options = {"all": "All datasets"}
        for item in profiles:
            key = profile_key(item.series_id)
            transformation_targets[key] = item.series_id
            options[key] = item.name
        target = transformation_context["target"]
        target.set_options(options, value=target.value if target.value in options else "all")

    def data_transform_snapshot(profiles=None):
        target = data_transformation_context["target"].value
        profiles = data_transform_sources() if profiles is None else profiles
        if target != "all" and not profiles:
            raise ValueError("Choose an available dataset to transform.")
        return dict(enabled=bool(data_transformation_controls.enabled.value),
                    target=profiles[0].series_id if target != "all" and profiles else None,
                    combine=bool(data_transformation_controls.combining),
                    formulas={profile_key(item.series_id): data_transformation_controls.formulas.get(item.series_id,
                        data_transformation_controls.formulas.get(None, "source")) for item in profiles},
                    combined_formula=data_transformation_controls.combined_formula,
                    values={profile_key(profiles[0].series_id) if key == "pending" and profiles else key: dict(value)
                            for key, value in value_transform_settings.items()})

    def dual_y_axes_active(group: Sequence[_PlotSeries], controls: Any = None) -> bool:
        active_controls = plot_settings if controls is None else controls
        return bool(
            active_controls is plot_settings and plot_settings.dual_y_axes.value
            and str(presentation.value or "Combined") == "Combined"
            and len(group) == 2 and _combined_series_compatible(group)
        )

    def current_secondary_y_limits() -> tuple[float, float] | None:
        lower, upper = plot_settings.secondary_y_minimum.value, plot_settings.secondary_y_maximum.value
        if lower is None or upper is None:
            return None
        limits = float(lower), float(upper)
        if not np.isfinite(limits).all() or limits[1] <= limits[0]:
            raise ValueError("Y2 maximum must be greater than its finite minimum.")
        return limits

    def populate_secondary_y_limits() -> None:
        nonlocal updating_y_limits, secondary_y_limits_automatic
        secondary_y_limits_automatic = True
        if not dual_y_axes_active(latest_series):
            return
        limits = automatic_y_limits(
            [latest_series[1].values], plot_type=group_plot_type(latest_series),
            histogram_bins=_integer_value(plot_settings.histogram_bins, "Histogram bins"),
            density=bool(plot_settings.density.value),
            kde_bandwidth=float(plot_settings.kde_bandwidth.value or 1.0),
            log_y=bool(plot_settings.secondary_log_y.value),
            log_x=bool(plot_settings.secondary_log_x.value),
            histogram_bin_size=None,
            histogram_x_limits=axis_limits_from_controls(plot_settings, "secondary_x"),
            histogram_boundary=str(plot_settings.histogram_boundary.value or "left"),
        )
        updating_y_limits = True
        try:
            for control, value in zip(
                (plot_settings.secondary_y_minimum, plot_settings.secondary_y_maximum), limits, strict=True,
            ):
                control.set_value(float(f"{value:.10g}"))
        finally:
            updating_y_limits = False

    def reset_secondary_y_limits() -> None:
        try:
            populate_secondary_y_limits()
            update_combined_plot()
        except ValueError as exc:
            notify(str(exc), type="negative")

    def current_y_limits(controls: Any = None) -> tuple[float, float] | None:
        active_controls = plot_settings if controls is None else controls
        if (
            active_controls.y_minimum.value is None
            or active_controls.y_maximum.value is None
        ):
            return None
        limits = (
            _numeric_value(active_controls.y_minimum, "Y minimum"),
            _numeric_value(active_controls.y_maximum, "Y maximum"),
        )
        if limits[1] <= limits[0]:
            raise ValueError("Y maximum must be greater than Y minimum.")
        return limits

    def latest_plot_types() -> tuple[str, ...]:
        return tuple(group_plot_type(group) for group in latest_groups())

    def populate_automatic_y_limits() -> None:
        nonlocal updating_y_limits, y_limits_automatic
        y_limits_automatic = True
        if not latest_series:
            clear_automatic_y_limits()
            return
        plot_types = latest_plot_types()
        if not plot_types:
            clear_automatic_y_limits()
            return
        if len(set(plot_types)) != 1:
            # A line and heatmap do not share a meaningful vertical scale;
            # allow each renderer to derive its own complete range.
            clear_automatic_y_limits()
            return
        plot_type = plot_types[0]
        if not plotting_settings_ready(
            plot_settings,
            plot_type=plot_type,
            allow_empty_y_limits=y_limits_automatic,
        ):
            return
        limits = automatic_y_limits(
            [item.values for item in (latest_series[:1] if dual_y_axes_active(latest_series) else latest_series)],
            plot_type=plot_type,
            histogram_bins=_integer_value(plot_settings.histogram_bins, "Histogram bins"),
            density=bool(plot_settings.density.value),
            kde_bandwidth=float(plot_settings.kde_bandwidth.value or 1.0),
            log_y=bool(plot_settings.log_y.value),
            log_x=bool(plot_settings.log_x.value),
            histogram_bin_size=(
                float(plot_settings.histogram_bin_size.value)
                if plot_settings.histogram_bin_size.value is not None
                else None
            ),
            histogram_x_limits=histogram_x_limits_from_controls(
                plot_settings,
                log_x=bool(plot_settings.log_x.value),
            ),
            histogram_boundary=str(plot_settings.histogram_boundary.value or "left"),
            y_axis=(
                np.arange(
                    max(item.values.shape[1] for item in latest_series),
                    dtype=float,
                )
                if plot_type == "Heatmap"
                else None
            ),
        )
        updating_y_limits = True
        try:
            plot_settings.y_minimum.value = float(f"{limits[0]:.10g}")
            plot_settings.y_maximum.value = float(f"{limits[1]:.10g}")
            plot_settings.y_minimum.update()
            plot_settings.y_maximum.update()
        finally:
            updating_y_limits = False
        if secondary_y_limits_automatic:
            populate_secondary_y_limits()

    def reset_y_limits() -> None:
        try:
            populate_automatic_y_limits()
            update_combined_plot()
        except ValueError as exc:
            notify(str(exc), type="negative")

    def plot_options(
        group: Sequence[_PlotSeries],
        *,
        controls: Any = None,
        use_curve_styles: bool = True,
        capture_arguments: bool = False,
    ) -> dict[str, Any]:
        active_controls = plot_settings if controls is None else controls
        first = next(iter(group), None)
        if first is None:
            raise ValueError("Upload and select at least one profile.")
        plot_type = group_plot_type(group, active_controls)
        if plot_type in {"Histogram", "Boxplot"}:
            group = tuple(replace(item, values=analysis_values(vars(item))) for item in group)
            first = group[0]
        profile = np.asarray(first.values)
        values: Any = profile
        x_axis: Any = None
        y_axis: Any = None
        lines: tuple[PreviewLine, ...] = ()
        if plot_type == "Heatmap":
            values = profile.T
            x_axis = np.arange(profile.shape[0], dtype=float)
            y_axis = np.arange(profile.shape[1], dtype=float)
        elif plot_type in {"Line", "Histogram", "Boxplot"}:
            individual_styles = use_curve_styles and len(group) > 1
            lines = tuple(
                PreviewLine(
                    name=curve_names.get(item.series_id, item.name),
                    x=np.arange(item.values.size, dtype=float),
                    y=np.asarray(item.values, dtype=float).reshape(-1),
                    color=(
                        curve_styles.get(item.series_id, _CurveStyle(_LINE_COLOURS[index % len(_LINE_COLOURS)])).color
                        if individual_styles
                        else str(active_controls.primary_color.value or _LINE_COLOURS[0])
                    ),
                    width=(
                        curve_styles.get(item.series_id, _CurveStyle(_LINE_COLOURS[index % len(_LINE_COLOURS)])).width
                        if individual_styles
                        else _numeric_value(active_controls.line_width, "Thickness")
                    ),
                    line_type=curve_styles.get(item.series_id, _CurveStyle(_LINE_COLOURS[index % len(_LINE_COLOURS)])).line_type if individual_styles else None,
                    opacity=histogram_opacities.get(item.series_id, .6) if plot_type == "Histogram" else 1.0,
                    box_position=box_positions.get(item.series_id),
                    box_label=box_labels.get(item.series_id) or None,
                    z=3 + len(group) - index,
                )
                for index, item in enumerate(group)
            )
        else:
            values = profile.reshape(-1)
            x_axis = np.arange(values.size, dtype=float)

        x_label = str(active_controls.x_axis_label.value or "")
        y_label = str(active_controls.y_axis_label.value or "")

        arguments = dict(
            plot_type=plot_type,
            title=str(active_controls.title.value or ""),
            x_label=x_label,
            y_label=y_label,
            color=str(active_controls.primary_color.value or _LINE_COLOURS[0]),
            palette=str(active_controls.heatmap_palette.value or "viridis"),
            histogram_bins=_integer_value(
                active_controls.histogram_bins,
                "Histogram bins",
            ),
            density=bool(active_controls.density.value),
            show_legend=bool(active_controls.show_legend.value),
            show_colorbar=bool(active_controls.colorbar.value),
            log_y=bool(active_controls.log_y.value),
            log_z=bool(active_controls.log_z.value),
            invert_y=bool(active_controls.invert_y.value),
            log_x=bool(active_controls.log_x.value),
            invert_x=bool(active_controls.invert_x.value),
            invert_z=bool(active_controls.invert_z.value),
            show_histogram_statistics=bool(
                active_controls.show_histogram_statistics.value
            ),
            histogram_statistics_position=str(
                active_controls.histogram_statistics_position.value or "Bottom right"
            ),
            histogram_bin_size=(
                float(active_controls.histogram_bin_size.value)
                if active_controls.histogram_bin_size.value is not None
                else None
            ),
            histogram_x_limits=histogram_x_limits_from_controls(
                active_controls,
                log_x=bool(active_controls.log_x.value),
            ),
            histogram_touching=bool(active_controls.histogram_touching.value),
            kde_bandwidth=float(active_controls.kde_bandwidth.value or 1.0),
            histogram_grouped=bool(active_controls.histogram_grouped.value),
            histogram_boundary=str(active_controls.histogram_boundary.value or "left"),
            y_limits=current_y_limits(active_controls),
            x_limits=axis_limits_from_controls(active_controls, "x"),
            z_limits=axis_limits_from_controls(active_controls, "z"),
            aspect_ratio=(
                float(active_controls.aspect_ratio.value)
                if active_controls.aspect_ratio.value is not None else None
            ),
            series_label=first.name,
            values=values,
            x_axis=x_axis,
            y_axis=y_axis,
            lines=lines,
            line_width=_numeric_value(active_controls.line_width, "Thickness"),
            line_type=str(active_controls.line_type.value or "solid"),
            secondary_y_label=(
                str(plot_settings.secondary_y_axis_label.value or "")
                if dual_y_axes_active(group, active_controls) else None
            ),
            secondary_y_limits=(current_secondary_y_limits() if dual_y_axes_active(group, active_controls) else None),
            secondary_x_label=(str(plot_settings.secondary_x_axis_label.value or "") if dual_y_axes_active(group, active_controls) else None),
            secondary_x_limits=(axis_limits_from_controls(plot_settings, "secondary_x") if dual_y_axes_active(group, active_controls) else None),
            secondary_log_x=(bool(plot_settings.secondary_log_x.value) if dual_y_axes_active(group, active_controls) else None),
            secondary_invert_x=(bool(plot_settings.secondary_invert_x.value) if dual_y_axes_active(group, active_controls) else None),
            secondary_log_y=(bool(plot_settings.secondary_log_y.value) if dual_y_axes_active(group, active_controls) else None),
            secondary_invert_y=(bool(plot_settings.secondary_invert_y.value) if dual_y_axes_active(group, active_controls) else None),
            show_markers=False,
            legend_position=str(active_controls.legend_position.value or "Top right"),
            show_grid=bool(active_controls.show_grid.value),
            interactive=True,
        )
        return arguments if capture_arguments else build_preview_plot_options(**arguments)

    def sync_separate_controls(
        context: _SeparatePlotContext,
        *,
        reset_axes: bool = False,
    ) -> None:
        """Apply representation-specific visibility to one separate plot."""

        controls = context.controls
        first = context.group[0]
        plot_type = group_plot_type(context.group, controls)
        sync_preview_settings_visibility(
            controls,
            plot_type=plot_type,
            multiple_curves=False,
            allow_histogram=first.form != "xt",
            allow_boxplot=first.form != "xt",
        )
        if reset_axes:
            x_label, y_label = _axis_labels(
                first.form,
                str(controls.view.value or "Default"),
                bool(controls.density.value),
            )
            controls.x_axis_label.value = x_label
            controls.y_axis_label.value = y_label
            controls.x_axis_label.update()
            controls.y_axis_label.update()

    def clear_separate_y_limits(context: _SeparatePlotContext) -> None:
        context.y_limits_automatic = True
        context.updating_y_limits = True
        try:
            context.controls.y_minimum.value = None
            context.controls.y_maximum.value = None
            context.controls.y_minimum.update()
            context.controls.y_maximum.update()
        finally:
            context.updating_y_limits = False

    def populate_separate_y_limits(context: _SeparatePlotContext) -> None:
        """Populate one plot's editable limits from only that plot's data."""

        controls = context.controls
        plot_type = group_plot_type(context.group, controls)
        context.y_limits_automatic = True
        if not plotting_settings_ready(
            controls,
            plot_type=plot_type,
            allow_empty_y_limits=True,
        ):
            return
        first = context.group[0]
        limits = automatic_y_limits(
            [item.values for item in context.group],
            plot_type=plot_type,
            histogram_bins=_integer_value(controls.histogram_bins, "Histogram bins"),
            density=bool(controls.density.value),
            kde_bandwidth=float(controls.kde_bandwidth.value or 1.0),
            log_y=bool(controls.log_y.value),
            log_x=bool(controls.log_x.value),
            histogram_bin_size=(
                float(controls.histogram_bin_size.value)
                if controls.histogram_bin_size.value is not None
                else None
            ),
            histogram_x_limits=histogram_x_limits_from_controls(
                controls,
                log_x=bool(controls.log_x.value),
            ),
            histogram_boundary=str(controls.histogram_boundary.value or "left"),
            y_axis=(
                np.arange(first.values.shape[1], dtype=float)
                if plot_type == "Heatmap"
                else None
            ),
        )
        context.updating_y_limits = True
        try:
            controls.y_minimum.value = float(f"{limits[0]:.10g}")
            controls.y_maximum.value = float(f"{limits[1]:.10g}")
            controls.y_minimum.update()
            controls.y_maximum.update()
        finally:
            context.updating_y_limits = False

    def update_separate_chart(context: _SeparatePlotContext) -> None:
        """Update one committed chart without rebuilding its settings controls."""

        controls = context.controls
        context.chart.attach_settings(controls)
        plot_type = group_plot_type(context.group, controls)
        if not plotting_settings_ready(
            controls,
            plot_type=plot_type,
            allow_empty_y_limits=context.y_limits_automatic,
        ):
            return
        try:
            options = plot_options(
                context.group,
                controls=controls,
                use_curve_styles=False,
            )
            context.chart.options.clear()
            context.chart.options.update(options)
            context.chart.update()
        except (RuntimeError, TypeError, ValueError) as exc:
            notify(str(exc), type="negative")

    def reset_separate_y_limits(context: _SeparatePlotContext) -> None:
        try:
            populate_separate_y_limits(context)
            update_separate_chart(context)
        except (RuntimeError, TypeError, ValueError) as exc:
            notify(str(exc), type="negative")

    def remember_separate_plot_settings(context: _SeparatePlotContext) -> None:
        """Snapshot one plot before its transient controls are rebuilt."""

        first = context.group[0]
        synchronizer = context.histogram_bin_sync
        separate_plot_settings[first.series_id] = _SeparatePlotSettingsSnapshot(
            automatic_axes=frozenset(context.controls._automatic_axes),
            values={
                field_name: getattr(context.controls, field_name).value
                for field_name in _SEPARATE_SETTING_VALUE_FIELDS
            },
            y_limits_automatic=context.y_limits_automatic,
            histogram_basis=str(getattr(synchronizer, "_basis", "bins")),
            histogram_x_limits_automatic=bool(
                getattr(synchronizer, "_x_limits_automatic", True)
            ),
        )

    def remember_all_separate_plot_settings() -> None:
        for context in tuple(separate_plot_contexts):
            remember_separate_plot_settings(context)

    def restore_separate_plot_settings(
        controls: Any,
        snapshot: _SeparatePlotSettingsSnapshot,
    ) -> None:
        """Restore saved primitive values before wiring live callbacks."""
        controls._automatic_axes.clear()
        controls._automatic_axes.update(snapshot.automatic_axes)

        for field_name, value in snapshot.values.items():
            control = getattr(controls, field_name)
            control.value = value
            control.update()

    def render_separate_settings(
        *,
        index: int,
        group: tuple[_PlotSeries, ...],
        chart: Any,
    ) -> _SeparatePlotContext:
        """Render and wire one independent settings block below its chart."""

        first = group[0]
        saved_settings = separate_plot_settings.get(first.series_id)
        x_label, y_label = _axis_labels(first.form, "Default", False)
        holder: dict[str, _SeparatePlotContext] = {}
        with (
            ui.element("div")
            .classes("w-full")
            .mark(f"plotting-separate-settings-surface-{index}")
        ):
            controls = preview_settings_controls(
                initial_title=first.name,
                initial_x_label=x_label,
                initial_y_label=y_label,
                on_reset_y_limits=lambda: reset_separate_y_limits(holder["context"]),
                markers=_plotting_setting_markers(separate_index=index),
            )
        context = _SeparatePlotContext(
            group=group,
            controls=controls,
            chart=chart,
            y_limits_automatic=(
                saved_settings.y_limits_automatic if saved_settings is not None else True
            ),
        )
        holder["context"] = context
        controls.line_type.set_value(curve_styles.get(first.series_id, _CurveStyle(_LINE_COLOURS[0])).line_type or str(plot_settings.line_type.value or "solid"))
        if saved_settings is not None:
            restore_separate_plot_settings(controls, saved_settings)

        def range_changed() -> None:
            clear_separate_y_limits(context)
            if context.y_limits_automatic:
                populate_separate_y_limits(context)
            update_separate_chart(context)

        def view_changed_separately() -> None:
            clear_separate_y_limits(context)
            sync_separate_controls(context, reset_axes=True)
            assert context.histogram_bin_sync is not None
            context.histogram_bin_sync.sync_to_data(redraw=False)
            if context.y_limits_automatic:
                populate_separate_y_limits(context)
            update_separate_chart(context)

        def density_changed_separately() -> None:
            clear_separate_y_limits(context)
            sync_separate_controls(context, reset_axes=True)
            if context.y_limits_automatic:
                populate_separate_y_limits(context)
            update_separate_chart(context)

        def x_scale_changed_separately() -> None:
            assert context.histogram_bin_sync is not None
            context.histogram_bin_sync.sync_to_data(redraw=False)
            range_changed()

        def custom_y_limits_changed_separately() -> None:
            if context.updating_y_limits:
                return
            context.y_limits_automatic = False
            update_separate_chart(context)

        def sync_guides_separately() -> None:
            sync_separate_controls(context)

        context.histogram_bin_sync = HistogramBinSynchronizer(
            controls,
            values=lambda: first.values,
            on_valid_change=range_changed,
            log_x=lambda: bool(controls.log_x.value),
        )
        if saved_settings is not None:
            context.histogram_bin_sync._basis = saved_settings.histogram_basis
            context.histogram_bin_sync._x_limits_automatic = (
                saved_settings.histogram_x_limits_automatic
            )
        bind_preview_settings_updates(
            controls,
            on_redraw=lambda: update_separate_chart(context),
            on_view_change=view_changed_separately,
            on_density_change=density_changed_separately,
            on_histogram_range_change=range_changed,
            on_log_x_change=x_scale_changed_separately,
            on_y_limits_change=custom_y_limits_changed_separately,
            on_visibility_change=sync_guides_separately,
        )

        sync_separate_controls(context, reset_axes=saved_settings is None)
        if saved_settings is None:
            context.histogram_bin_sync.sync_to_data(redraw=False)
        if context.y_limits_automatic:
            populate_separate_y_limits(context)
        update_separate_chart(context)
        with expansion("Raw data", icon=DATA_ICON, expanded=True, extra_classes="dn-output-section") as raw_section:
            HistogramStatisticsExport(
                controls, get_chart=lambda: chart,
                filename=compact_export_stem(first.name, fallback="profile") + "_statistics",
                marker=f"analysis-statistics-{index}", navigate=_navigate,
                preference=statistics_export_preferences.setdefault(
                    first.series_id, StatisticsExportPreference(
                        compact_export_stem(first.name, fallback="profile") + "_statistics",
                    ),
                ),
                section=raw_section,
            )
        return context

    async def download_plot(
        file_format: str,
        *,
        chart: Any,
        filename: str,
        detail: str,
    ) -> None:
        if not latest_series or chart is None:
            notify("Plot profiles before downloading the figure.", type="warning")
            return
        try:
            dpi = int(get_export_settings().plot_dpi)
            capture_format = "png" if str(file_format).casefold() == "png" else "svg"
            capture_options: dict[str, Any] = {"type": capture_format}
            if capture_format == "png":
                capture_options["pixelRatio"] = max(1.0, min(4.0, dpi / 96.0))
            data_uri = await chart.run_chart_method(
                "getDataURL",
                capture_options,
                timeout=10,
            )
            artifact = await run.io_bound(
                build_chart_capture_export,
                data_uri,
                file_format,
                filename=filename,
                dpi=dpi,
            )
            if artifact is None or not artifact.data:
                raise RuntimeError("Plot export stopped before returning data.")
        except (RuntimeError, TimeoutError, TypeError, ValueError) as exc:
            notify(str(exc), type="negative")
            return
        cache_artifact(
            artifact,
            source="Analysis",
            kind="plot",
            detail=detail,
            navigate_to="plotting",
        )
        ui.download(artifact.data, artifact.filename, artifact.media_type)

    def render_latest_plot() -> None:
        if plot_settings._statistics_export is not None:
            plot_settings._statistics_export.refresh()
        if not latest_series:
            return
        try:
            groups = latest_groups()
        except ValueError as exc:
            notify(str(exc), type="negative")
            return
        separate = str(presentation.value or "Combined") == "Separate"
        shared_settings_surface.visible = not separate
        shared_settings_surface.update()
        if not separate:
            if any(
                not plotting_settings_ready(
                    plot_settings,
                    plot_type=group_plot_type(group),
                    allow_empty_y_limits=y_limits_automatic,
                )
                for group in groups
            ):
                return
            try:
                prepared = tuple((group, plot_options(group)) for group in groups)
            except (RuntimeError, TypeError, ValueError) as exc:
                notify(str(exc), type="negative")
                return
        else:
            # Each separate plot receives a fresh natural representation and
            # its own live controls immediately below the rendered chart.
            prepared = tuple((group, plot_options(group)) for group in groups)
        remember_all_separate_plot_settings()
        result_charts.clear()
        separate_plot_contexts.clear()
        result_slot.clear()
        with result_slot:
            try:
                for index, (group, options) in enumerate(prepared):
                    first = group[0]
                    chart_title = (
                        first.name
                        if separate
                        else str(plot_settings.title.value or "Profiles")
                    )
                    with ui.element("div").classes("dn-plot-result-panel"):
                        if separate:
                            ui.label(first.name).classes("dn-setting-group-title")
                        async def download_current(
                            file_format: str,
                            *,
                            plot_index: int = index,
                            plot_group: tuple[_PlotSeries, ...] = tuple(group),
                            plot_detail: str = chart_title,
                        ) -> None:
                            try:
                                current_names = _plot_download_filenames(latest_groups())
                                plot_filename = current_names[plot_index]
                            except (IndexError, RuntimeError, TypeError, ValueError):
                                # The rendered chart remains downloadable while
                                # controls are between two valid layout states.
                                plot_filename = _plot_download_filenames((plot_group,))[0]
                            await download_plot(
                                file_format,
                                chart=result_charts[plot_index],
                                filename=plot_filename,
                                detail=plot_detail,
                            )

                        chart = render_preview_chart(
                            options,
                            title=chart_title,
                            marker=(
                                "plotting-result-plot"
                                if index == 0
                                else f"plotting-result-plot-{index + 1}"
                            ),
                            on_download=download_current,
                        )
                        result_charts.append(chart)
                        if not separate:
                            chart.attach_settings(plot_settings)
                        if separate:
                            separate_plot_contexts.append(
                                render_separate_settings(
                                    index=index + 1,
                                    group=tuple(group),
                                    chart=chart,
                                )
                            )
            except (RuntimeError, TypeError, ValueError) as exc:
                result_charts.clear()
                separate_plot_contexts.clear()
                notify(str(exc), type="negative")
                preview_plot("Plot unavailable", kind="plotting")

    def update_combined_plot() -> None:
        """Update shared-layout charts in place so the controls remain stable."""

        if not latest_series or str(presentation.value or "Combined") != "Combined":
            return
        try:
            groups = latest_groups()
            if len(result_charts) != len(groups):
                render_latest_plot()
                return
            for group, chart in zip(groups, result_charts):
                if not plotting_settings_ready(
                    plot_settings, plot_type=group_plot_type(group),
                    allow_empty_y_limits=y_limits_automatic,
                ):
                    return
                options = plot_options(group)
                if len(latest_series) == 1 and latest_series[0].form in {"x", "t"}:
                    # Retain edits from Style if this profile later joins a
                    # combined plot with per-curve controls.
                    curve_styles[latest_series[0].series_id] = _CurveStyle(
                        str(plot_settings.primary_color.value or _LINE_COLOURS[0]),
                        _numeric_value(plot_settings.line_width, "Thickness"),
                    )
                chart.options.clear()
                chart.options.update(options)
                chart.update()
        except ValueError:
            # Empty or partially edited numeric fields are transient states;
            # leave the last valid chart in place until the edit is complete.
            return
        except (RuntimeError, TypeError) as exc:
            notify(str(exc), type="negative")

    async def download_plotting_code() -> None:
        from dataclasses import fields
        from app.code_export import CodeInputs, CodeSnapshot, build_python, filter_code_settings, placeholder_upload

        if running or transformation_controls.loading:
            notify("Wait for the data operation to finish before exporting code.", type="warning")
            return
        try:
            has_selection = any(source.inspection is not None and source.selection for source in sources)
            selected = (_collect_plot_series(sources, missing_policy="ignore") if has_selection else
                        (_PlotSeries((1, _NATURAL_SELECTION), "Profile", "x", np.empty(0)),))
            inputs = CodeInputs()
            profiles = []
            for item in selected:
                inspection = next((source.inspection for source in sources
                    if source.source_id == item.series_id[0] and source.selection), None) or placeholder_upload("profile", form=item.form)
                token = item.series_id[1]
                profiles.append(dict(
                    id=list(item.series_id), name=item.name, form=item.form,
                    source=inputs.encode(inspection),
                    column=token if not inspection.array_keys and token not in {_NATURAL_SELECTION, _MATRIX_SELECTION} else None,
                    array_key=token if inspection.array_keys else None,
                    smoothing=inputs.encode(SmoothingSettings()),
                ))
            target = transformation_context["target"].value
            if transformation_controls.controls["enabled"].value and target != "all" and target not in transformation_targets:
                raise ValueError("Choose an available dataset to transform.")
            transform = dict(settings=inputs.encode(filter_code_settings(transformation_controls)),
                             target=transformation_targets.get(target),
                             keep_originals=bool(transformation_context["keep_originals"].value),
                             analysis_on_mask_only=bool(transformation_context["analysis_on_mask_only"].value))
            data_transform = data_transform_snapshot(selected if not has_selection else None)
            # Describe output identities without running formulas or analysis.
            preview_series = list(selected)
            if data_transform["enabled"]:
                if data_transform["combine"] and len(preview_series) > 1:
                    preview_series = [replace(preview_series[0], series_id=(preview_series[0].series_id[0], "__combined__"), name="Combined", transformed=True)]
                else:
                    preview_series = [replace(item, transformed=True) if data_transform["target"] is None or item.series_id == data_transform["target"] else item for item in preview_series]
            region_series = []
            for item in preview_series:
                transforming = transformation_controls.controls["enabled"].value and (target == "all" or item.series_id == transformation_targets.get(target))
                if not transforming or transform["keep_originals"]:
                    region_series.append(item)
                if transforming:
                    region_series.append(replace(item, series_id=(-item.series_id[0], item.series_id[1]) if transform["keep_originals"] else item.series_id,
                        name=item.name + " (transformed)" if transform["keep_originals"] else item.name, transformed=True))
            preview_series = region_series
            groups = _plot_groups(_unique_series_names(preview_series), str(presentation.value or "Combined"))
            plots = []
            for index, group in enumerate(groups, 1):
                separate = presentation.value == "Separate"
                context = next((context for context in separate_plot_contexts
                                if context.group[0].series_id == group[0].series_id), None) if separate else None
                active = context.controls if context is not None else plot_settings
                arguments = plot_options(group, controls=active, use_curve_styles=not separate, capture_arguments=True)
                curves = [{field.name: getattr(line, field.name) for field in fields(line) if field.name not in {"x", "y"}}
                          for line in arguments.get("lines", ())]
                for key in ("lines", "values", "x_axis", "y_axis"):
                    arguments.pop(key, None)
                style = {key: getattr(active, key).value for key in (
                    "view", "show_fill", "fill_side", "reverse_order", "circular_combine", "circular_inner_radius",
                    "circular_track_height", "z_axis_label", "show_hover")}
                style["histogram_dataset"] = active._histogram_dataset
                style["theme_color"] = get_export_settings().theme_color
                specification = dict(profiles=[list(item.series_id) for item in group],
                    arguments=arguments, curves=curves, presentation=style, filename=f"plot_{index}")
                if active.export_histogram_statistics.value and arguments["plot_type"] == "Histogram":
                    exporter = active._statistics_export
                    preference = exporter.preference if exporter is not None else StatisticsExportPreference(f"plot_{index}_statistics")
                    specification["statistics"] = dict(filename=preference.file_name,
                        file_format=preference.file_type, compression=get_export_settings().raw_compression if preference.compress else "none")
                plots.append(specification)
            snapshot = CodeSnapshot("plotting", dict(
                profiles=profiles, transformations=transform, data_transforms=inputs.encode(data_transform),
                mask_exports={profile_key(item.series_id): mask_export_preferences.get(profile_key(item.series_id),
                    dict(filename=compact_export_stem(item.name, fallback="profile") + "_mask_positions", file_format="TSV", compress=False, layout="positions"))
                    for item in preview_series if item.transformed and (target == "all" or (abs(item.series_id[0]), item.series_id[1]) == transformation_targets.get(target))}
                    if transformation_controls.controls["enabled"].value and transformation_controls.controls["export"].value else {},
                methods=[method for method, control in correlation_controls.items() if control.value] if analysis_enabled.value else None,
                report=dict(filename=report_name["value"], file_format=report_file_format["value"], include_p_values=bool(show_p_values.value)),
                report_names={f"{key[0]}:{key[1]}": name for key, name in curve_names.items()},
                data_exports={f"{key[0]}:{key[1]}": value for key, value in transformed_export_preferences.items()},
                plots=plots, plot_dpi=get_export_settings().plot_dpi,
                preview_points=get_export_settings().preview_points,
            ), tuple(inputs.files.items()))
            artifact = await run.io_bound(build_python, snapshot)
            ui.download(artifact.data, artifact.filename, artifact.media_type)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            notify(str(exc), type="negative")

    async def plot_profiles() -> None:
        nonlocal latest_series, latest_analysis, running, active_cancel
        if running:
            if active_cancel is not None:
                active_cancel.set()
                page_activity.stop("plotting")
                sync_analysis_action()
            return
        running = True
        active_cancel = Event()
        cancel_event = active_cancel
        sync_analysis_action()
        page_run = page_activity.begin("plotting", cancel=cancel_event.set)
        completed_successfully = False

        def check_cancelled() -> None:
            if cancel_event.is_set():
                raise CancelledError

        try:
            preparation_arguments = (
                tuple(replace(source, selection=tuple(source.selection), array_cache=dict(source.array_cache)) for source in sources),
                {},
                tuple(method for method, control in correlation_controls.items() if control.value) if analysis_enabled.value else None,
                check_cancelled,
            )
            if transformation_controls.controls["enabled"].value:
                target = transformation_context["target"].value
                if target != "all" and target not in transformation_targets:
                    raise ValueError("Choose an available dataset to transform.")
                preparation_arguments += (_TransformationSettings(
                    transformation_controls.snapshot(), transformation_targets.get(target),
                    bool(transformation_context["keep_originals"].value),
                    bool(transformation_context["analysis_on_mask_only"].value),
                ),)
            if data_transformation_controls.enabled.value:
                if len(preparation_arguments) == 4:
                    preparation_arguments += (None,)
                preparation_arguments += (data_transform_snapshot(),)
            result = await run.io_bound(_prepare_analysis_profiles, *preparation_arguments)
            check_cancelled()
            if result is None:
                raise RuntimeError("Profile preparation stopped before returning data.")
            selected, report = result
            first = next(iter(selected), None)
            if first is None:
                raise RuntimeError("Plot preparation returned no profiles.")
            latest_series = selected
            latest_analysis = report
            for warning in dict.fromkeys(warning for item in selected for warning in item.warnings):
                notify(warning, type="warning")
            multiple = len(selected) > 1
            previous_styles = dict(curve_styles)
            if not multiple and first.form in {"x", "t"}:
                if first.series_id not in previous_styles:
                    previous_styles[first.series_id] = _CurveStyle(
                        str(plot_settings.primary_color.value or _LINE_COLOURS[0]),
                        _numeric_value(plot_settings.line_width, "Thickness"),
                    )
            curve_styles.clear()
            curve_styles.update(
                {
                    item.series_id: previous_styles.get(
                        item.series_id,
                        _CurveStyle(_LINE_COLOURS[index % len(_LINE_COLOURS)]),
                    )
                    for index, item in enumerate(selected)
                    if item.form in {"x", "t"}
                }
            )
            if not multiple and first.form in {"x", "t"}:
                style = curve_styles[first.series_id]
                plot_settings.primary_color.set_value(style.color)
                plot_settings.line_width.set_value(style.width)
                if style.line_type is not None:
                    plot_settings.line_type.set_value(style.line_type)
            sync_presentation_options(sync_action=False)
            histogram_bin_sync.sync_to_data(redraw=False)
            sync_staged_controls(reset_axes=False)
            render_curve_settings.refresh()
            if str(plot_settings.title.value or "").strip() in {"", "Profile", "Profiles"}:
                plot_settings.title.value = "Profiles" if multiple else first.name
                plot_settings.title.update()
            if y_limits_automatic:
                populate_automatic_y_limits()
            render_latest_plot()
            render_transformed_data.refresh()
            render_analysis_report.refresh()
            completed_successfully = True
        except CancelledError:
            pass
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            notify(str(exc), type="negative")
        finally:
            running = False
            active_cancel = None
            sync_presentation_options()
            page_activity.finish("plotting", page_run, success=completed_successfully)

    def view_changed() -> None:
        clear_automatic_y_limits()
        sync_staged_controls(reset_axes=True)
        histogram_bin_sync.sync_to_data(redraw=False)
        if y_limits_automatic:
            populate_automatic_y_limits()
        render_curve_settings.refresh()
        update_combined_plot()

    def density_changed() -> None:
        clear_automatic_y_limits()
        sync_staged_controls(reset_axes=True)
        if y_limits_automatic:
            populate_automatic_y_limits()
        render_curve_settings.refresh()
        update_combined_plot()

    def range_basis_changed() -> None:
        clear_automatic_y_limits()
        if y_limits_automatic:
            populate_automatic_y_limits()
        update_combined_plot()

    def custom_y_limits_changed() -> None:
        nonlocal y_limits_automatic
        if updating_y_limits:
            return
        y_limits_automatic = False
        update_combined_plot()

    def custom_secondary_y_limits_changed() -> None:
        nonlocal secondary_y_limits_automatic
        if updating_y_limits or plot_settings._resetting_limits:
            return
        secondary_y_limits_automatic = False
        update_combined_plot()

    def dual_y_axes_changed() -> None:
        clear_automatic_y_limits()
        sync_staged_controls()
        histogram_bin_sync.sync_to_data(redraw=False)
        if dual_y_axes_active(latest_series) and plot_settings.secondary_y_axis_label.value == "Value":
            plot_settings.secondary_y_axis_label.set_value(latest_series[1].name)
        try:
            populate_automatic_y_limits()
            # ECharts merges axes when the series count stays unchanged.
            # Rebuild only when switching between one and two Y axes so no
            # previous right axis survives turning this option off.
            render_latest_plot()
        except ValueError as exc:
            notify(str(exc), type="negative")

    def sync_guide_visibility() -> None:
        sync_staged_controls(reset_axes=False)

    def live_setting_changed() -> None:
        if secondary_y_limits_automatic and not updating_y_limits:
            try:
                populate_secondary_y_limits()
            except ValueError:
                return
        update_combined_plot()

    def line_type_changed() -> None:
        render_curve_settings.refresh()
        update_combined_plot()

    def x_scale_changed() -> None:
        histogram_bin_sync.sync_to_data(redraw=False)
        range_basis_changed()

    def presentation_changed() -> None:
        sync_staged_controls(reset_axes=True)
        if y_limits_automatic:
            populate_automatic_y_limits()
        render_curve_settings.refresh()
        render_latest_plot()

    def histogram_values() -> Any | None:
        if not latest_series:
            return None
        if dual_y_axes_active(latest_series):
            return latest_series[0].values
        return np.concatenate([item.values.reshape(-1) for item in latest_series])

    histogram_bin_sync = HistogramBinSynchronizer(
        plot_settings,
        values=histogram_values,
        on_valid_change=range_basis_changed,
        log_x=lambda: bool(plot_settings.log_x.value),
    )

    bind_preview_settings_updates(
        plot_settings,
        on_redraw=live_setting_changed,
        on_view_change=view_changed,
        on_density_change=density_changed,
        on_histogram_range_change=range_basis_changed,
        on_log_x_change=x_scale_changed,
        on_y_limits_change=custom_y_limits_changed,
        on_visibility_change=sync_guide_visibility,
        on_line_type_change=line_type_changed,
    )
    plot_settings.dual_y_axes.on_value_change(dual_y_axes_changed)
    plot_settings.secondary_y_minimum.on_value_change(custom_secondary_y_limits_changed)
    plot_settings.secondary_y_maximum.on_value_change(custom_secondary_y_limits_changed)
    plot_settings.secondary_y_axis_label.on_value_change(live_setting_changed)
    plot_settings.secondary_x_axis_label.on_value_change(live_setting_changed)
    plot_settings.secondary_invert_x.on_value_change(live_setting_changed)
    plot_settings.secondary_invert_y.on_value_change(live_setting_changed)

    def secondary_scale_changed(*, reset_x: bool = False) -> None:
        if reset_x:
            plot_settings._automatic_axes.add("secondary_x")
        try:
            populate_secondary_y_limits()
            update_combined_plot()
        except ValueError as exc:
            notify(str(exc), type="negative")

    plot_settings.secondary_log_x.on_value_change(lambda: secondary_scale_changed(reset_x=True))
    plot_settings.secondary_log_y.on_value_change(secondary_scale_changed)
    analysis_enabled.on_value_change(analysis_enabled_changed)
    show_p_values.on_value_change(lambda: render_analysis_report.refresh())
    for control in correlation_controls.values():
        control.on_value_change(invalidate_analysis_report)
    for control in (*transformation_controls.controls.values(), *transformation_context.values()):
        if hasattr(control, "on_value_change"):
            control.on_value_change(invalidate_analysis_report)
    transformation_controls.controls["export"].on_value_change(lambda: render_transformed_data.refresh())
    transformation_controls.controls["enabled"].on_value_change(lambda: render_transformed_data.refresh())
    presentation.on_value_change(presentation_changed)
    plot_button.on_click(plot_profiles)
    sync_presentation_options()
    sync_staged_controls()

    def clear_plot_output() -> None:
        nonlocal latest_series, latest_analysis
        if active_cancel is not None:
            active_cancel.set()
            page_activity.stop("plotting")
            sync_analysis_action()
        latest_series = ()
        latest_analysis = None
        result_charts.clear()
        separate_plot_contexts.clear()
        curve_styles.clear()
        curve_names.clear()
        box_positions.clear()
        box_labels.clear()
        histogram_opacities.clear()
        separate_plot_settings.clear()
        shared_settings_surface.visible = True
        shared_settings_surface.update()
        render_curve_settings.refresh()
        render_analysis_report.refresh()
        render_transformed_data.refresh()
        result_slot.clear()
        with result_slot:
            preview_plot("No plot yet", kind="plotting")

    current_session_cache().register_output_clearer("plotting", clear_plot_output)

    async def receive_raw_files(files) -> None:
        if active_cancel is not None:
            raise ValueError("Wait for the current analysis to finish before adding data.")
        await append_raw_uploads(lambda: profile_uploader, files)

    current_session_cache().register_input_targets(
        "plotting", lambda: (
            RawInputTarget("profiles", "Data", receive_raw_files),
            RawInputTarget("filter_positions", "Region transforms", transformation_controls.receive_mask, kind="mask"),
        ),
    )
