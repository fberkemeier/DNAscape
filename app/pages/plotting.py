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
    PreviewLine,
    automatic_y_limits,
    build_preview_plot_options,
    preview_settings_controls,
    render_preview_chart,
    sync_preview_settings_visibility,
)
from app.session_cache import cache_artifact, cache_upload
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
        inspection.inferred_form == "t"
        and shape
        and shape[0] == 1
        and len(columns) > 1
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
    matrix_tokens = tuple(
        token for token in normalized if _selection_form(state, token) == "xt"
    )
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
        name = (
            f"{item.name} {occurrences[item.name]}"
            if counts[item.name] > 1
            else item.name
        )
        normalized.append(_PlotSeries(name=name, form=item.form, values=item.values))
    return tuple(normalized)


def _collect_plot_series(sources: Sequence[_ProfileSourceState]) -> tuple[_PlotSeries, ...]:
    collected = [item for source in sources for item in _materialize_source(source)]
    if not collected:
        raise ValueError("Upload and select at least one profile.")
    if len(collected) > 1:
        forms = {item.form for item in collected}
        if "xt" in forms:
            raise ValueError(
                "A space-time heatmap must be plotted on its own. "
                "Select individual columns to compare curves."
            )
        if len(forms) != 1:
            raise ValueError(
                "Multiple curves must share one dependence. "
                "Plot space and time profiles separately."
            )
    return _unique_series_names(collected)


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
    latest_form = "unknown"
    result_chart: Any | None = None
    y_limits_automatic = True
    updating_y_limits = False

    page_intro(
        "Plotting",
        "Visualize DNAscape outputs",
        "Upload one or more profiles, arrange their presentation, and plot them together.",
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
                        sources[:] = [
                            source for source in sources if source.source_id != source_id
                        ]
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
                            with ui.element("div").classes("dn-plot-profile-source").mark(
                                f"plotting-profile-source-{state.source_id}"
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
                                            .mark(
                                                f"plotting-remove-profile-{state.source_id}"
                                            )
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
                                        source.inspection = inspection
                                        source.array_cache.clear()
                                        _set_default_selection(source)
                                        _materialize_source(source)
                                    except (OSError, RuntimeError, TypeError, ValueError) as exc:
                                        source.inspection = None
                                        source.selection = ()
                                        source.array_cache.clear()
                                        render_profile_sources.refresh()
                                        ui.notify(str(exc), type="negative")
                                        return False
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
                                        else source.selection[0] if source.selection else None
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
                                        .mark(
                                            f"plotting-profile-selection-{source.source_id}"
                                        )
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
                                            else normalized[0] if normalized else None
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
                                    with ui.element("div").mark(
                                        f"plotting-detected-data-{source.source_id}"
                                    ):
                                        ui.label(f"Dependence: {dependence}").classes(
                                            "dn-mapping-detected-summary "
                                            "dn-mapping-detected-dependence"
                                        )
                                        ui.label(f"Array size: {size}").classes(
                                            "dn-mapping-detected-summary"
                                        )

                                summary_panel()

                        add_file = (
                            ui.button("Add file", icon="add", on_click=add_source)
                            .props("flat dense no-caps")
                            .classes("dn-add-profile-source")
                            .mark("action-add-plot-profile")
                        )
                        add_file.tooltip("Add another profile file")

                    render_profile_sources()

                with ui.element("div").classes("dn-control-plot-settings"):
                    plot_settings = preview_settings_controls(
                        initial_title="Profile",
                        initial_series_label="Profile",
                        initial_x_label="Position (bin)",
                        initial_y_label="Value",
                        on_reset_y_limits=lambda: reset_y_limits(),
                        markers={
                            "container": "plotting-settings",
                            "view": "plotting-view",
                            "title": "plotting-title",
                            "series_label": "plotting-series-label",
                            "x_axis_label": "plotting-x-axis-label",
                            "y_axis_label": "plotting-y-axis-label",
                            "primary_color": "plotting-primary-color",
                            "secondary_color": "plotting-secondary-color",
                            "heatmap_palette": "plotting-heatmap-palette",
                            "log_z": "plotting-log-z",
                            "line_width": "plotting-line-width",
                            "histogram_bins": "plotting-histogram-bins",
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
                        },
                    )

                with action_dock():
                    plot_button = plot_action_button()

        with ui.column().classes("dn-preview-stack"):
            with panel(extra_classes="dn-preview-card"):
                panel_title("Preview")
                with ui.element("div").classes("w-full").mark(
                    "plotting-result-container"
                ) as result_slot:
                    preview_plot("No plot yet", kind="plotting")

    def staged_forms() -> tuple[str, ...]:
        return tuple(
            _selection_form(source, token)
            for source in sources
            for token in source.selection
            if _selection_form(source, token) in _DEPENDENCE_LABELS
        )

    def requested_plot_type(*, form: str, multiple: bool) -> str:
        if form == "xt":
            return "Heatmap"
        if multiple:
            return "Line"
        view = str(plot_settings.view.value or "Default")
        if view == "Histogram":
            return "Histogram"
        if view == "Boxplot":
            return "Boxplot"
        return "Line"

    def sync_staged_controls(*, reset_axes: bool = False) -> None:
        forms = staged_forms()
        multiple = len(forms) > 1
        form = forms[0] if forms and len(set(forms)) == 1 else "x"
        plot_type = requested_plot_type(form=form, multiple=multiple)
        sync_preview_settings_visibility(
            plot_settings,
            plot_type=plot_type,
            multiple_curves=multiple,
            allow_histogram=not multiple and form != "xt",
            allow_boxplot=not multiple and form != "xt",
        )
        plot_settings.series_label.visible = not multiple
        if reset_axes:
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
        sync_staged_controls(reset_axes=True)

    def current_y_limits() -> tuple[float, float] | None:
        if plot_settings.y_minimum.value is None or plot_settings.y_maximum.value is None:
            return None
        limits = (
            _numeric_value(plot_settings.y_minimum, "Y minimum"),
            _numeric_value(plot_settings.y_maximum, "Y maximum"),
        )
        if limits[1] <= limits[0]:
            raise ValueError("Y maximum must be greater than Y minimum.")
        return limits

    def latest_plot_type() -> str:
        return requested_plot_type(
            form=latest_form,
            multiple=len(latest_series) > 1,
        )

    def populate_automatic_y_limits() -> None:
        nonlocal updating_y_limits, y_limits_automatic
        y_limits_automatic = True
        if not latest_series:
            clear_automatic_y_limits()
            return
        plot_type = latest_plot_type()
        limits = automatic_y_limits(
            [item.values for item in latest_series],
            plot_type=plot_type,
            histogram_bins=_integer_value(plot_settings.histogram_bins, "Histogram bins"),
            density=bool(plot_settings.density.value),
            log_y=bool(plot_settings.log_y.value),
            y_axis=(
                np.arange(latest_series[0].values.shape[1], dtype=float)
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
        except ValueError as exc:
            ui.notify(str(exc), type="negative")

    def plot_options() -> dict[str, Any]:
        if not latest_series:
            raise ValueError("Upload and select at least one profile.")
        plot_type = latest_plot_type()
        first = latest_series[0]
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
            primary = str(plot_settings.primary_color.value or _LINE_COLOURS[0])
            secondary = str(plot_settings.secondary_color.value or _LINE_COLOURS[1])
            colours = (primary, secondary, *_LINE_COLOURS[2:])
            lines = tuple(
                PreviewLine(
                    name=(
                        str(plot_settings.series_label.value or item.name)
                        if len(latest_series) == 1
                        else item.name
                    ),
                    x=np.arange(item.values.size, dtype=float),
                    y=np.asarray(item.values, dtype=float).reshape(-1),
                    color=colours[index % len(colours)],
                )
                for index, item in enumerate(latest_series)
            )
        else:
            values = profile.reshape(-1)
            x_axis = np.arange(values.size, dtype=float)

        return build_preview_plot_options(
            plot_type=plot_type,
            title=str(plot_settings.title.value or ""),
            x_label=str(plot_settings.x_axis_label.value or ""),
            y_label=str(plot_settings.y_axis_label.value or ""),
            color=str(plot_settings.primary_color.value or _LINE_COLOURS[0]),
            palette=str(plot_settings.heatmap_palette.value or "viridis"),
            histogram_bins=_integer_value(plot_settings.histogram_bins, "Histogram bins"),
            density=bool(plot_settings.density.value),
            show_legend=bool(plot_settings.show_legend.value),
            show_colorbar=bool(plot_settings.colorbar.value),
            log_y=bool(plot_settings.log_y.value),
            log_z=bool(plot_settings.log_z.value),
            invert_y=bool(plot_settings.invert_y.value),
            y_limits=current_y_limits(),
            series_label=(
                first.name
                if plot_type == "Heatmap"
                else str(plot_settings.series_label.value or first.name)
            ),
            values=values,
            x_axis=x_axis,
            y_axis=y_axis,
            lines=lines,
            line_width=_numeric_value(plot_settings.line_width, "Curve thickness"),
            show_markers=False,
            legend_position=str(plot_settings.legend_position.value or "Top right"),
            show_grid=bool(plot_settings.show_grid.value),
            interactive=latest_form not in {"t", "xt"},
        )

    async def download_plot(file_format: str) -> None:
        if not latest_series or result_chart is None:
            ui.notify("Plot profiles before downloading the figure.", type="warning")
            return
        try:
            dpi = int(get_export_settings().plot_dpi)
            data_uri = await result_chart.run_chart_method(
                "getDataURL",
                {"type": "png", "pixelRatio": max(1.0, min(4.0, dpi / 96.0))},
                timeout=10,
            )
            artifact = await run.io_bound(
                build_chart_capture_export,
                data_uri,
                file_format,
                filename="dnascape_profile_plot",
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
            detail=str(plot_settings.title.value or "Profiles"),
            navigate_to="plotting",
        )
        ui.download(artifact.data, artifact.filename, artifact.media_type)

    def render_latest_plot() -> None:
        nonlocal result_chart
        result_chart = None
        result_slot.clear()
        with result_slot:
            try:
                options = plot_options()
                result_chart = render_preview_chart(
                    options,
                    title=str(plot_settings.title.value or "Profiles"),
                    marker="plotting-result-plot",
                    on_download=download_plot,
                )
            except (RuntimeError, TypeError, ValueError) as exc:
                ui.notify(str(exc), type="negative")
                preview_plot("Plot unavailable", kind="plotting")

    async def plot_profiles() -> None:
        nonlocal latest_form, latest_series
        plot_button.disable()
        try:
            selected = await run.io_bound(_collect_plot_series, tuple(sources))
            if selected is None:
                raise RuntimeError("Plot preparation stopped before returning data.")
            first = next(iter(selected), None)
            if first is None:
                raise RuntimeError("Plot preparation returned no profiles.")
            latest_series = selected
            latest_form = first.form
            multiple = len(selected) > 1
            sync_staged_controls(reset_axes=False)
            if str(plot_settings.title.value or "").strip() in {"", "Profile", "Profiles"}:
                plot_settings.title.value = "Profiles" if multiple else first.name
                plot_settings.title.update()
            if not multiple and str(plot_settings.series_label.value or "").strip() in {
                "",
                "Profile",
            }:
                plot_settings.series_label.value = first.name
                plot_settings.series_label.update()
            if y_limits_automatic:
                populate_automatic_y_limits()
            render_latest_plot()
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            ui.notify(str(exc), type="negative")
        finally:
            plot_button.enable()

    def view_changed() -> None:
        clear_automatic_y_limits()
        sync_staged_controls(reset_axes=True)

    def density_changed() -> None:
        clear_automatic_y_limits()
        sync_staged_controls(reset_axes=True)

    def range_basis_changed() -> None:
        clear_automatic_y_limits()

    def custom_y_limits_changed() -> None:
        nonlocal y_limits_automatic
        if not updating_y_limits:
            y_limits_automatic = False

    def legend_changed() -> None:
        sync_staged_controls(reset_axes=False)

    plot_settings.view.on_value_change(view_changed)
    plot_settings.density.on_value_change(density_changed)
    plot_settings.histogram_bins.on_value_change(range_basis_changed)
    plot_settings.log_y.on_value_change(range_basis_changed)
    plot_settings.y_minimum.on_value_change(custom_y_limits_changed)
    plot_settings.y_maximum.on_value_change(custom_y_limits_changed)
    plot_settings.show_legend.on_value_change(legend_changed)
    plot_button.on_click(plot_profiles)
    sync_staged_controls()
