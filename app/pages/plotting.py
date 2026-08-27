"""Interactive single- and multi-profile plotting workspace."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import PurePath
from typing import Any

import numpy as np
from nicegui import run, ui

from app.export_settings import get_export_settings
from app.mapping_uploads import UploadInspection, inspect_upload_event, materialize_upload
from app.pages.common import (
    DATA_ICON,
    action_dock,
    data_upload,
    expansion,
    page_intro,
    panel,
    panel_title,
    plot_action_button,
    preview_plot,
)
from app.preview_plotting import (
    HistogramBinSynchronizer,
    PreviewLine,
    automatic_y_limits,
    build_preview_plot_options,
    compact_color_input,
    histogram_x_limits_from_controls,
    plotting_settings_ready,
    preview_settings_controls,
    render_preview_chart,
    sync_preview_settings_visibility,
)
from app.session_cache import cache_artifact, cache_upload, current_session_cache
from app.simulation_exports import build_chart_capture_export

_PROFILE_ACCEPT = ".csv,.tsv,.txt,.npy,.npv,.npz,.gz"
_DEPENDENCE_LABELS = {
    "x": "Space",
    "t": "Time",
    "xt": "Space-time",
}
_NATURAL_SELECTION = "__natural_profile__"
_MATRIX_SELECTION = "__all_columns__"
_MAX_PROFILE_FILES = 8
_LINE_COLOURS = (
    "#2c7f77",
    "#b03412",
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
    name: str
    form: str
    values: np.ndarray


@dataclass
class _CurveStyle:
    color: str
    width: float = 1.8


@dataclass
class _SeparatePlotContext:
    """Controls and chart owned by one independently presented profile."""

    group: tuple[_PlotSeries, ...]
    controls: Any
    chart: Any
    y_limits_automatic: bool = True
    updating_y_limits: bool = False
    histogram_bin_sync: HistogramBinSynchronizer | None = None


_PLOTTING_SETTING_MARKERS = {
    "container": "plotting-settings",
    "view": "plotting-view",
    "title": "plotting-title",
    "x_axis_label": "plotting-x-axis-label",
    "y_axis_label": "plotting-y-axis-label",
    "primary_color": "plotting-primary-color",
    "secondary_color": "plotting-secondary-color",
    "heatmap_palette": "plotting-heatmap-palette",
    "log_x": "plotting-log-x",
    "invert_x": "plotting-invert-x",
    "log_z": "plotting-log-z",
    "invert_z": "plotting-invert-z",
    "line_width": "plotting-line-width",
    "histogram_bins": "plotting-histogram-bins",
    "histogram_bin_size": "plotting-histogram-bin-size",
    "histogram_touching": "plotting-histogram-touching",
    "histogram_boundary": "plotting-histogram-boundary",
    "histogram_x_minimum": "plotting-histogram-x-minimum",
    "histogram_x_maximum": "plotting-histogram-x-maximum",
    "reset_histogram_x_limits": "plotting-reset-histogram-x-limits",
    "show_histogram_statistics": "plotting-show-histogram-statistics",
    "histogram_statistics_position": "plotting-histogram-statistics-position",
    "show_legend": "plotting-show-legend",
    "legend_position": "plotting-legend-position",
    "show_grid": "plotting-show-grid",
    "density": "plotting-density",
    "colorbar": "plotting-colorbar",
    "log_y": "plotting-log-y",
    "invert_y": "plotting-invert-y",
    "y_minimum": "plotting-y-minimum",
    "y_maximum": "plotting-y-maximum",
    "reset_y_limits": "plotting-reset-y-limits",
}


def _plotting_setting_markers(*, separate_index: int | None = None) -> dict[str, str]:
    """Return stable shared or per-plot markers for the common control factory."""

    if separate_index is None:
        return dict(_PLOTTING_SETTING_MARKERS)
    return {
        key: f"{marker}-separate-{separate_index}"
        for key, marker in _PLOTTING_SETTING_MARKERS.items()
    }


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
                "Plotting requires a space, time, or space-time profile; "
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
            raise ValueError(f"{inspection.name} does not contain a plottable profile.")
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
        series.append(_PlotSeries(name=name, form=form, values=values))
    return series


def _unique_series_names(series: Sequence[_PlotSeries]) -> tuple[_PlotSeries, ...]:
    counts = Counter(item.name for item in series)
    occurrences: Counter[str] = Counter()
    normalized: list[_PlotSeries] = []
    for item in series:
        occurrences[item.name] += 1
        name = f"{item.name} {occurrences[item.name]}" if counts[item.name] > 1 else item.name
        normalized.append(_PlotSeries(name=name, form=item.form, values=item.values))
    return tuple(normalized)


def _collect_plot_series(sources: Sequence[_ProfileSourceState]) -> tuple[_PlotSeries, ...]:
    collected = [item for source in sources for item in _materialize_source(source)]
    if not collected:
        raise ValueError("Upload and select at least one profile.")
    normalized = _unique_series_names(collected)
    incompatibility = _series_compatibility_error(normalized)
    if incompatibility is not None:
        raise ValueError(incompatibility)
    return normalized


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
        return "Position (bin)", "Time (bin)"
    if view == "Boxplot":
        return "", "Value"
    if view == "Histogram":
        return "Value", "Density" if density else "Count"
    return ("Time (bin)" if form == "t" else "Position (bin)"), "Value"


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

    sources = [_ProfileSourceState(source_id=1)]
    next_source_id = 2
    latest_series: tuple[_PlotSeries, ...] = ()
    result_charts: list[Any] = []
    separate_plot_contexts: list[_SeparatePlotContext] = []
    curve_styles: dict[str, _CurveStyle] = {}
    y_limits_automatic = True
    updating_y_limits = False

    page_intro(
        "Plotting",
        "Visualize DNAscape outputs",
        "Upload profiles and compare them together or in separate panels.",
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

                    def add_source() -> None:
                        nonlocal next_source_id
                        if len(sources) >= _MAX_PROFILE_FILES:
                            ui.notify(
                                f"Up to {_MAX_PROFILE_FILES} profile files can be combined.",
                                type="warning",
                            )
                            return
                        sources.append(_ProfileSourceState(source_id=next_source_id))
                        next_source_id += 1
                        render_profile_sources.refresh()
                        staged_sources_changed()

                    def remove_source(source_id: int) -> None:
                        if len(sources) == 1:
                            return
                        sources[:] = [source for source in sources if source.source_id != source_id]
                        render_profile_sources.refresh()
                        staged_sources_changed()

                    @ui.refreshable
                    def render_profile_sources() -> None:
                        for position, state in enumerate(sources, start=1):
                            marker = (
                                "plotting-data-upload"
                                if position == 1
                                else f"plotting-data-upload-{state.source_id}"
                            )
                            with (
                                ui.element("div")
                                .classes("dn-plot-profile-source")
                                .mark(f"plotting-profile-source-{state.source_id}")
                            ):
                                with ui.row().classes("dn-plot-profile-source-head"):
                                    ui.label(f"Profile file {position}").classes(
                                        "dn-setting-group-title"
                                    )
                                    ui.space()
                                    if len(sources) > 1:

                                        def remove_current(
                                            _event: Any = None,
                                            source_id: int = state.source_id,
                                        ) -> None:
                                            remove_source(source_id)

                                        remove = (
                                            ui.button(icon="close", on_click=remove_current)
                                            .props(
                                                'flat round dense aria-label="Remove profile file"'
                                            )
                                            .classes("dn-upload-remove")
                                            .mark(f"plotting-remove-profile-{state.source_id}")
                                        )
                                        remove.tooltip("Remove profile file")

                                async def uploaded(
                                    event: Any,
                                    source: _ProfileSourceState = state,
                                ) -> bool:
                                    try:
                                        inspection = await inspect_upload_event(
                                            event,
                                            expected_form="auto",
                                        )
                                        candidate = _ProfileSourceState(
                                            source_id=source.source_id,
                                            inspection=inspection,
                                        )
                                        _set_default_selection(candidate)
                                        _materialize_source(candidate)
                                    except (OSError, RuntimeError, TypeError, ValueError) as exc:
                                        ui.notify(str(exc), type="negative")
                                        return False
                                    source.inspection = candidate.inspection
                                    source.selection = candidate.selection
                                    source.array_cache = candidate.array_cache
                                    cache_upload(
                                        inspection,
                                        source="Plotting",
                                        navigate_to="plotting",
                                    )
                                    render_profile_sources.refresh()
                                    staged_sources_changed()
                                    return True

                                def cleared(
                                    _event: Any,
                                    source: _ProfileSourceState = state,
                                ) -> None:
                                    source.inspection = None
                                    source.selection = ()
                                    source.array_cache.clear()
                                    render_profile_sources.refresh()
                                    staged_sources_changed()

                                data_upload(
                                    accept=_PROFILE_ACCEPT,
                                    accepted_formats="CSV, TSV, TXT, NPY, NPV, NPZ, GZ",
                                    marker=marker,
                                    show_accepted_formats=False,
                                    on_upload=uploaded,
                                    on_clear=cleared,
                                    max_files=1,
                                    hide_when_loaded=True,
                                    initial_names=(
                                        (state.inspection.name,)
                                        if state.inspection is not None
                                        else ()
                                    ),
                                )

                                @ui.refreshable
                                def selection_panel(
                                    source: _ProfileSourceState = state,
                                ) -> None:
                                    options = _source_selection_options(source)
                                    if not options:
                                        return
                                    multiple = len(options) > 1
                                    value: Any = (
                                        list(source.selection)
                                        if multiple
                                        else source.selection[0]
                                        if source.selection
                                        else None
                                    )
                                    selector = (
                                        ui.select(
                                            dict(options),
                                            value=value,
                                            label="Profiles" if multiple else "Profile",
                                            multiple=multiple,
                                            clearable=multiple,
                                        )
                                        .props(
                                            "dense options-dense use-chips"
                                            if multiple
                                            else "dense options-dense"
                                        )
                                        .classes("dn-field dn-plot-profile-selector")
                                        .mark(f"plotting-profile-selection-{source.source_id}")
                                    )
                                    if not multiple:
                                        selector.disable()

                                    def selection_changed() -> None:
                                        raw = (
                                            tuple(str(token) for token in (selector.value or []))
                                            if multiple
                                            else (str(selector.value),)
                                            if selector.value is not None
                                            else ()
                                        )
                                        normalized = _normalize_multi_selection(source, raw)
                                        source.selection = normalized
                                        expected_value: Any = (
                                            list(normalized)
                                            if multiple
                                            else normalized[0]
                                            if normalized
                                            else None
                                        )
                                        if selector.value != expected_value:
                                            selector.value = expected_value
                                            selector.update()
                                        render_profile_sources.refresh()
                                        staged_sources_changed()

                                    selector.on_value_change(selection_changed)

                                selection_panel()

                                @ui.refreshable
                                def summary_panel(
                                    source: _ProfileSourceState = state,
                                ) -> None:
                                    if source.inspection is None:
                                        return
                                    try:
                                        selected = _materialize_source(
                                            source,
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
                                        .mark(f"plotting-detected-data-{source.source_id}")
                                    ):
                                        ui.label(f"Dependence: {dependence}").classes(
                                            "dn-mapping-detected-summary "
                                            "dn-mapping-detected-dependence"
                                        )
                                        ui.label(f"Array size: {size}").classes(
                                            "dn-mapping-detected-summary"
                                        )

                                summary_panel()

                        with ui.row().classes("w-full items-center"):
                            add_file = (
                                ui.button("Add file", icon="add", on_click=add_source)
                                .props("flat dense no-caps")
                                .classes("dn-add-profile-source")
                                .mark("action-add-plot-profile")
                            )
                            add_file.tooltip("Add another profile file")
                            ui.space()
                            delete_all = (
                                ui.button(
                                    "Delete all",
                                    icon="delete_outline",
                                    on_click=lambda: delete_all_profiles(),
                                )
                                .props("flat dense no-caps")
                                .classes("dn-add-profile-source")
                                .mark("action-delete-all-plot-profiles")
                            )
                            delete_all.tooltip("Remove every loaded plotting profile")

                    render_profile_sources()
                    compatibility_note = (
                        ui.label("")
                        .classes("dn-settings-help dn-plot-compatibility-note")
                        .mark("plotting-compatibility-note")
                    )
                    compatibility_note.visible = False

                with action_dock():
                    plot_button = plot_action_button()

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
                        initial_x_label="Position (bin)",
                        initial_y_label="Value",
                        on_reset_y_limits=lambda: reset_y_limits(),
                        markers=_plotting_setting_markers(),
                    )

                @ui.refreshable
                def render_curve_settings() -> None:
                    curve_series = [
                        item
                        for item in latest_series
                        if item.form in {"x", "t"}
                        and str(plot_settings.view.value or "Default") == "Default"
                    ]
                    if not curve_series:
                        return
                    with expansion(
                        "Curves",
                        icon="show_chart",
                        extra_classes="dn-plot-curve-settings",
                    ):
                        with ui.column().classes("dn-plot-curve-list"):
                            for index, item in enumerate(curve_series):
                                style = curve_styles.setdefault(
                                    item.name,
                                    _CurveStyle(_LINE_COLOURS[index % len(_LINE_COLOURS)]),
                                )
                                with ui.element("div").classes("dn-plot-curve-row"):
                                    ui.label(item.name).classes("dn-plot-curve-name")
                                    color_surface, color = compact_color_input(
                                        "Color",
                                        style.color,
                                        marker=f"plotting-curve-color-{index + 1}",
                                    )
                                    color_surface.classes("dn-plot-curve-color")
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
                                        .mark(f"plotting-curve-width-{index + 1}")
                                    )

                                    def style_changed(
                                        _event: Any = None,
                                        *,
                                        series_name: str = item.name,
                                        color_control: Any = color,
                                        width_control: Any = width,
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
                                        curve_styles[series_name] = _CurveStyle(
                                            str(color_control.value or _LINE_COLOURS[0]),
                                            parsed_width,
                                        )
                                        update_combined_plot()

                                    color.on_value_change(style_changed)
                                    width.on_value_change(style_changed)

                with shared_settings_surface:
                    render_curve_settings()

    def staged_forms() -> tuple[str, ...]:
        return tuple(
            _selection_form(source, token)
            for source in sources
            for token in source.selection
            if _selection_form(source, token) in _DEPENDENCE_LABELS
        )

    def requested_plot_type(*, form: str, multiple: bool, controls: Any = None) -> str:
        if form == "xt":
            return "Heatmap"
        if multiple:
            return "Line"
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

    def sync_presentation_options() -> None:
        forms = staged_forms()
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
        if incompatibility is None:
            plot_button.enable()
        else:
            plot_button.disable()

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
        forms = staged_forms()
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
            multiple_curves=False,
            allow_histogram=not multiple and form != "xt",
            allow_boxplot=not multiple and form != "xt",
        )
        # Plotting owns per-curve color and thickness controls in the adjacent
        # Curves section, avoiding ambiguous "first" and "second" fields.
        if plot_type == "Line" and latest_series:
            plot_settings.primary_color_surface.visible = False
            plot_settings.secondary_color_surface.visible = False
            plot_settings.line_width.visible = False
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
            plot_settings.y_axis_label.value = y_label
            plot_settings.x_axis_label.update()
            plot_settings.y_axis_label.update()

    def clear_automatic_y_limits() -> None:
        nonlocal updating_y_limits, y_limits_automatic
        y_limits_automatic = True
        updating_y_limits = True
        try:
            plot_settings.y_minimum.value = None
            plot_settings.y_maximum.value = None
            plot_settings.y_minimum.update()
            plot_settings.y_maximum.update()
        finally:
            updating_y_limits = False

    def staged_sources_changed() -> None:
        clear_automatic_y_limits()
        sync_presentation_options()
        sync_staged_controls(reset_axes=True)

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
            [item.values for item in latest_series],
            plot_type=plot_type,
            histogram_bins=_integer_value(plot_settings.histogram_bins, "Histogram bins"),
            density=bool(plot_settings.density.value),
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

    def reset_y_limits() -> None:
        try:
            populate_automatic_y_limits()
            update_combined_plot()
        except ValueError as exc:
            ui.notify(str(exc), type="negative")

    def plot_options(
        group: Sequence[_PlotSeries],
        *,
        controls: Any = None,
        use_curve_styles: bool = True,
    ) -> dict[str, Any]:
        active_controls = plot_settings if controls is None else controls
        first = next(iter(group), None)
        if first is None:
            raise ValueError("Upload and select at least one profile.")
        plot_type = group_plot_type(group, active_controls)
        profile = np.asarray(first.values, dtype=float)
        values: Any = profile
        x_axis: Any = None
        y_axis: Any = None
        lines: tuple[PreviewLine, ...] = ()
        if plot_type == "Heatmap":
            values = profile.T
            x_axis = np.arange(profile.shape[0], dtype=float)
            y_axis = np.arange(profile.shape[1], dtype=float)
        elif plot_type == "Line":
            lines = tuple(
                PreviewLine(
                    name=item.name,
                    x=np.arange(item.values.size, dtype=float),
                    y=np.asarray(item.values, dtype=float).reshape(-1),
                    color=(
                        curve_styles[item.name].color
                        if use_curve_styles
                        else str(active_controls.primary_color.value or _LINE_COLOURS[0])
                    ),
                    width=(
                        curve_styles[item.name].width
                        if use_curve_styles
                        else _numeric_value(active_controls.line_width, "Curve thickness")
                    ),
                )
                for item in group
            )
        else:
            values = profile.reshape(-1)
            x_axis = np.arange(values.size, dtype=float)

        x_label = str(active_controls.x_axis_label.value or "")
        y_label = str(active_controls.y_axis_label.value or "")

        return build_preview_plot_options(
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
                active_controls.histogram_statistics_position.value or "Top right"
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
            histogram_boundary=str(active_controls.histogram_boundary.value or "left"),
            y_limits=current_y_limits(active_controls),
            series_label=first.name,
            values=values,
            x_axis=x_axis,
            y_axis=y_axis,
            lines=lines,
            line_width=_numeric_value(active_controls.line_width, "Curve thickness"),
            show_markers=False,
            legend_position=str(active_controls.legend_position.value or "Top right"),
            show_grid=bool(active_controls.show_grid.value),
            interactive=first.form not in {"t", "xt"},
        )

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
            ui.notify(str(exc), type="negative")

    def reset_separate_y_limits(context: _SeparatePlotContext) -> None:
        try:
            populate_separate_y_limits(context)
            update_separate_chart(context)
        except (RuntimeError, TypeError, ValueError) as exc:
            ui.notify(str(exc), type="negative")

    def render_separate_settings(
        *,
        index: int,
        group: tuple[_PlotSeries, ...],
        chart: Any,
    ) -> _SeparatePlotContext:
        """Render and wire one independent settings block below its chart."""

        first = group[0]
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
        )
        holder["context"] = context

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

        def guide_changed_separately() -> None:
            sync_separate_controls(context)
            update_separate_chart(context)

        context.histogram_bin_sync = HistogramBinSynchronizer(
            controls,
            values=lambda: first.values,
            on_valid_change=range_changed,
            log_x=lambda: bool(controls.log_x.value),
        )
        controls.view.on_value_change(view_changed_separately)
        controls.density.on_value_change(density_changed_separately)
        controls.histogram_boundary.on_value_change(range_changed)
        controls.log_y.on_value_change(range_changed)
        controls.log_x.on_value_change(x_scale_changed_separately)
        controls.y_minimum.on_value_change(custom_y_limits_changed_separately)
        controls.y_maximum.on_value_change(custom_y_limits_changed_separately)
        controls.show_histogram_statistics.on_value_change(guide_changed_separately)
        controls.show_legend.on_value_change(guide_changed_separately)
        for live_control in (
            controls.title,
            controls.x_axis_label,
            controls.y_axis_label,
            controls.primary_color,
            controls.secondary_color,
            controls.heatmap_palette,
            controls.invert_x,
            controls.log_z,
            controls.invert_z,
            controls.line_width,
            controls.histogram_touching,
            controls.histogram_statistics_position,
            controls.legend_position,
            controls.show_grid,
            controls.colorbar,
            controls.invert_y,
        ):
            live_control.on_value_change(lambda: update_separate_chart(context))

        sync_separate_controls(context, reset_axes=True)
        context.histogram_bin_sync.sync_to_data(redraw=False)
        populate_separate_y_limits(context)
        update_separate_chart(context)
        return context

    async def download_plot(
        file_format: str,
        *,
        chart: Any,
        filename: str,
        detail: str,
    ) -> None:
        if not latest_series or chart is None:
            ui.notify("Plot profiles before downloading the figure.", type="warning")
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
            ui.notify(str(exc), type="negative")
            return
        cache_artifact(
            artifact,
            source="Plotting",
            kind="plot",
            detail=detail,
            navigate_to="plotting",
        )
        ui.download(artifact.data, artifact.filename, artifact.media_type)

    def render_latest_plot() -> None:
        if not latest_series:
            return
        try:
            groups = latest_groups()
        except ValueError as exc:
            ui.notify(str(exc), type="negative")
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
                ui.notify(str(exc), type="negative")
                return
        else:
            # Each separate plot receives a fresh natural representation and
            # its own live controls immediately below the rendered chart.
            prepared = tuple((group, plot_options(group)) for group in groups)
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
                    filename = "dnascape_profile_plot"
                    if separate:
                        safe_name = "_".join(first.name.casefold().split()) or f"plot_{index + 1}"
                        filename = f"dnascape_{safe_name}"
                    with ui.element("div").classes("dn-plot-result-panel"):
                        if separate:
                            ui.label(first.name).classes("dn-setting-group-title")
                        async def download_current(
                            file_format: str,
                            *,
                            plot_index: int = index,
                            plot_filename: str = filename,
                            plot_detail: str = chart_title,
                        ) -> None:
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
                ui.notify(str(exc), type="negative")
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
                options = plot_options(group)
                chart.options.clear()
                chart.options.update(options)
                chart.update()
        except ValueError:
            # Empty or partially edited numeric fields are transient states;
            # leave the last valid chart in place until the edit is complete.
            return
        except (RuntimeError, TypeError) as exc:
            ui.notify(str(exc), type="negative")

    async def plot_profiles() -> None:
        nonlocal latest_series
        plot_button.disable()
        try:
            selected = await run.io_bound(_collect_plot_series, tuple(sources))
            if selected is None:
                raise RuntimeError("Plot preparation stopped before returning data.")
            first = next(iter(selected), None)
            if first is None:
                raise RuntimeError("Plot preparation returned no profiles.")
            latest_series = selected
            multiple = len(selected) > 1
            previous_styles = dict(curve_styles)
            curve_styles.clear()
            curve_styles.update(
                {
                    item.name: previous_styles.get(
                        item.name,
                        _CurveStyle(_LINE_COLOURS[index % len(_LINE_COLOURS)]),
                    )
                    for index, item in enumerate(selected)
                    if item.form in {"x", "t"}
                }
            )
            histogram_bin_sync.sync_to_data(redraw=False)
            sync_staged_controls(reset_axes=False)
            render_curve_settings.refresh()
            if str(plot_settings.title.value or "").strip() in {"", "Profile", "Profiles"}:
                plot_settings.title.value = "Profiles" if multiple else first.name
                plot_settings.title.update()
            if y_limits_automatic:
                populate_automatic_y_limits()
            render_latest_plot()
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            ui.notify(str(exc), type="negative")
        finally:
            sync_presentation_options()

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

    def legend_changed() -> None:
        sync_staged_controls(reset_axes=False)
        update_combined_plot()

    def live_setting_changed() -> None:
        update_combined_plot()

    def x_scale_changed() -> None:
        histogram_bin_sync.sync_to_data(redraw=False)
        range_basis_changed()

    def presentation_changed() -> None:
        sync_staged_controls(reset_axes=True)
        if y_limits_automatic:
            populate_automatic_y_limits()
        render_latest_plot()

    def histogram_values() -> Any | None:
        return latest_series[0].values if latest_series else None

    histogram_bin_sync = HistogramBinSynchronizer(
        plot_settings,
        values=histogram_values,
        on_valid_change=range_basis_changed,
        log_x=lambda: bool(plot_settings.log_x.value),
    )

    plot_settings.view.on_value_change(view_changed)
    plot_settings.density.on_value_change(density_changed)
    plot_settings.histogram_boundary.on_value_change(range_basis_changed)
    plot_settings.log_y.on_value_change(range_basis_changed)
    plot_settings.log_x.on_value_change(x_scale_changed)
    plot_settings.y_minimum.on_value_change(custom_y_limits_changed)
    plot_settings.y_maximum.on_value_change(custom_y_limits_changed)
    plot_settings.show_histogram_statistics.on_value_change(legend_changed)
    plot_settings.show_legend.on_value_change(legend_changed)
    for live_control in (
        plot_settings.title,
        plot_settings.x_axis_label,
        plot_settings.y_axis_label,
        plot_settings.primary_color,
        plot_settings.secondary_color,
        plot_settings.heatmap_palette,
        plot_settings.invert_x,
        plot_settings.log_z,
        plot_settings.invert_z,
        plot_settings.line_width,
        plot_settings.histogram_touching,
        plot_settings.histogram_statistics_position,
        plot_settings.legend_position,
        plot_settings.show_grid,
        plot_settings.colorbar,
        plot_settings.invert_y,
    ):
        live_control.on_value_change(live_setting_changed)
    presentation.on_value_change(presentation_changed)
    plot_button.on_click(plot_profiles)
    sync_presentation_options()
    sync_staged_controls()

    def clear_plot_output() -> None:
        nonlocal latest_series
        latest_series = ()
        result_charts.clear()
        separate_plot_contexts.clear()
        curve_styles.clear()
        shared_settings_surface.visible = True
        shared_settings_surface.update()
        render_curve_settings.refresh()
        result_slot.clear()
        with result_slot:
            preview_plot("No plot yet", kind="plotting")

    def delete_all_profiles() -> None:
        """Remove every staged upload and committed plotting result."""

        nonlocal next_source_id
        sources[:] = [_ProfileSourceState(source_id=1)]
        next_source_id = 2
        clear_plot_output()
        presentation.options = ["Combined", "Separate"]
        presentation.value = "Combined"
        presentation.visible = False
        presentation.update()
        plot_settings.view.value = "Default"
        plot_settings.title.value = "Profile"
        plot_settings.x_axis_label.value = "Position (bin)"
        plot_settings.y_axis_label.value = "Value"
        for control in (
            plot_settings.view,
            plot_settings.title,
            plot_settings.x_axis_label,
            plot_settings.y_axis_label,
        ):
            control.update()
        clear_automatic_y_limits()
        render_profile_sources.refresh()
        staged_sources_changed()

    current_session_cache().register_output_clearer("plotting", clear_plot_output)
