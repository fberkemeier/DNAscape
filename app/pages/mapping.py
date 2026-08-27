"""Mapping workflow interface."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Callable
from concurrent.futures import CancelledError
from dataclasses import dataclass, field, replace
from threading import Event, Lock
from typing import Any, cast

import numpy as np
from nicegui import run, ui

from app.export_settings import get_export_settings
from app.mapping_catalog import (
    DataForm,
    MappingInput,
    MappingRoute,
    MappingRouteFamily,
    SettingLocation,
    get_mapping_catalog,
    mapping_route_families,
    mapping_target,
    mapping_targets,
)
from app.mapping_exports import (
    build_mapping_comparison_raw_export,
    build_mapping_raw_export,
)
from app.mapping_service import (
    MappingRequest,
    MappingResult,
    materialize_many_tracks,
    materialize_track_upload,
    run_mapping,
)
from app.mapping_uploads import (
    UploadInspection,
    inspect_upload_event,
    materialize_many,
    materialize_upload,
)
from app.pages.common import (
    DATA_ICON,
    RawOutputControls,
    action_dock,
    data_upload,
    expansion,
    is_data_upload,
    page_intro,
    panel,
    panel_title,
    preview_plot,
    raw_output_controls,
    set_data_upload_locked,
    stat_strip,
)
from app.preview_plotting import (
    FIXED_QUANTITY_Y_LIMITS,
    PreviewLine,
    automatic_y_limits,
    build_preview_plot_options,
    finite_values,
    preview_settings_controls,
    render_preview_chart,
    sync_preview_settings_visibility,
)
from app.session_cache import (
    cache_artifact,
    cache_upload,
    cache_workflow,
    consume_cached_workflow,
)
from app.simulation_exports import build_chart_capture_export

_DEPENDENCE_LABELS = {
    "constant": "Constant",
    "space": "Space",
    "time": "Time",
    "spacetime": "Space–time",
    "regions": "Regions",
    "ensemble": "Ensemble",
}

_FORM_LABELS = {
    "scalar": "Constant",
    "x": "Space",
    "t": "Time",
    "xt": "Space–time",
    "supplied": "Prepared data",
    "bundle": "Prepared result",
    "regions": "Regions",
}

_FORMAT_LABELS = {
    "csv": "CSV",
    "tsv": "TSV",
    "txt": "TXT",
    "npy": "NPY",
    "npv": "NPV",
    "npz": "NPZ",
    "bedgraph": "bedGraph",
    "bigwig": "BigWig",
}

_SCALAR_DEFAULTS = {
    "firing_rate": 0.001,
    "fork_speed": 1.4,
}

_SCALAR_UNITS = {
    "firing_rate": "min⁻¹ kb⁻¹",
    "fork_speed": "kb/min",
}

_SETTING_CATEGORY_LABELS = {
    "domain": "Grid & domain",
    "fitting": "Fitting",
    "simulation": "Simulation",
    "numerics": "Numerics",
    "integration": "Integration",
    "advanced": "Advanced",
}

# Keep the full controls available, but start expensive fitting routes with a
# responsive local-preview budget. Users can raise these values for final runs.
_INTERACTIVE_SETTING_CAPS = {
    "maxiter": 5,
    "space_maxiter": 5,
    "temporal_maxiter": 5,
    "space_init_maxiter": 5,
    "joint_maxiter": 5,
    "sim_number": 20,
    "temporal_sim_number": 20,
    "spacetime_sim_number": 20,
    "space_init_sim_number": 20,
    "joint_sim_number": 20,
}

_OUTPUT_LABELS = {
    "firing_rate": "Firing rate",
    "fork_speed": "Fork speed",
    "replication_timing": "Replication timing",
    "replication_timing_fit": "Replication timing fit",
    "fit_history": "Fit history",
    "residual": "Residual",
    "replicated_fraction": "Replicated fraction",
    "replicated_fraction_xt": "Replicated fraction (space-time)",
    "fork_directionality": "Fork directionality",
    "right_fork_density_xt": "Right-moving fork density",
    "left_fork_density_xt": "Left-moving fork density",
    "fork_density_xt": "Total fork density",
}

_BUNDLE_OUTPUTS = {
    "kinetic_fields": (
        "firing_rate",
        "fork_speed",
        "replication_timing_fit",
        "fit_history",
    ),
    "fork_densities": (
        "replicated_fraction_xt",
        "right_fork_density_xt",
        "left_fork_density_xt",
        "fork_density_xt",
    ),
}

_COMPARISON_PREFIX = "comparison:"
_FIT_DIAGNOSTIC_KEY = "fit_diagnostic"
_PLOT_TYPES = ["Line", "Heatmap", "Histogram", "Boxplot"]

_WEIBULL_ONLY_SETTINGS = {"weibull_shape", "weibull_delay", "fit_weibull_shape"}


class _ProgressState:
    """Thread-safe progress hand-off from the mapping worker to NiceGUI."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._value = 0.0
        self._message = "Waiting to map"

    def report(self, value: float, message: str) -> None:
        with self._lock:
            self._value = max(0.0, min(1.0, float(value)))
            self._message = str(message)

    def snapshot(self) -> tuple[float, str]:
        with self._lock:
            return self._value, self._message


@dataclass
class _SettingControl:
    """A route setting and the live widgets needed to read it at submit time."""

    setting: Any
    control: Any
    override: Any | None = None


@dataclass
class _MappingInputControl:
    """Live controls and parsed payload state for one biological input."""

    spec: MappingInput
    form: Any
    refresh_surface: Callable[[], Any]
    override: Any | None
    surface: dict[str, Any]


@dataclass
class _MappingUploadState:
    """Parsed upload data retained independently of disposable route widgets."""

    inspections: list[UploadInspection] = field(default_factory=list)
    selection_values: dict[str, dict[str, str | None]] = field(default_factory=dict)
    materialized_cache: dict[str, Any] = field(default_factory=lambda: {"key": None, "value": None})


# Building the scientific route catalogue is the only non-trivial part of
# Mapping-page setup. Resolve it while the local app starts so opening Mapping
# never pays that cold cost or leaves the navigation looking unresponsive.
_MAPPING_TARGETS = mapping_targets()


def _slug(value: str) -> str:
    """Return the stable marker fragment used by route-driven inputs."""

    return "-".join(value.casefold().replace("_", " ").split())


def _select_options(values: tuple[str, ...], labels: dict[str, str]) -> dict[str, str]:
    return {value: labels.get(value, value.replace("_", " ").title()) for value in values}


def _route_options(families: tuple[MappingRouteFamily, ...]) -> dict[str, str]:
    """Return biological factor combinations without implementation details."""

    return {family.key: family.label for family in families}


def _route_family(key: str) -> MappingRouteFamily:
    """Look up one route family from the already cached catalogue."""

    return get_mapping_catalog().route_family(key)


def _default_form_selections(family: MappingRouteFamily) -> dict[str, DataForm]:
    """Choose a complete, implemented starting combination for a family."""

    selections: dict[str, DataForm] = {}
    for key in family.required_factor_keys:
        forms = family.allowed_forms(key, selections)
        if not forms:
            raise ValueError(f"Route family {family.key!r} has no valid form for {key!r}.")
        selections[key] = forms[0]
    return selections


def _retained_form_selections(
    family: MappingRouteFamily,
    upload_states: dict[str, _MappingUploadState],
) -> dict[str, DataForm]:
    """Choose a route variant which reuses the most compatible uploaded inputs."""

    detected = {
        key: form
        for key in family.input_keys
        if (
            form := _detected_profile_form(
                upload_states.get(_friendly_input_key(key), _MappingUploadState()).inspections
            )
        )
        is not None
    }
    if not detected:
        return _default_form_selections(family)

    def retained_matches(route: MappingRoute) -> int:
        matches = 0
        for key, form in detected.items():
            try:
                matches += form in family.variant_input(route, key).forms
            except KeyError:
                continue
        return matches

    template = max(family.variants, key=retained_matches)
    selections: dict[str, DataForm] = {}
    for key in family.required_factor_keys:
        exact_input = family.variant_input(template, key)
        retained_form = detected.get(key)
        selections[key] = (
            retained_form if retained_form in exact_input.forms else exact_input.forms[0]
        )
    return selections


def _format_copy(input_spec: MappingInput, form: str) -> str:
    formats = input_spec.formats_for(form)  # type: ignore[arg-type]
    labels = [_FORMAT_LABELS.get(item, item.upper()) for item in formats]
    if ".gz" in input_spec.accept_for(form):  # type: ignore[arg-type]
        labels.append("GZ")
    return ", ".join(labels)


def _scalar_default(input_spec: MappingInput) -> float:
    value = input_spec.default
    if isinstance(value, (int, float)):
        return float(value)
    return _SCALAR_DEFAULTS.get(input_spec.key, 1.0)


def _display_default(value: Any) -> str:
    """Format a registry default without exposing Python implementation noise."""

    if value is None:
        return "automatic"
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, (tuple, list)):
        return ", ".join(str(item) for item in value)
    return str(value)


def _friendly_input_key(key: str) -> str:
    return key.removesuffix("_xt")


def _detected_profile_form(inspections: list[UploadInspection]) -> DataForm | None:
    """Resolve rdata's shape convention for one retained profile input."""

    if not inspections:
        return None
    if len(inspections) > 1:
        return cast(DataForm, "xt")
    if all(item.source_format in {"bedgraph", "bigwig"} for item in inspections):
        return cast(DataForm, "x")
    detected = str(inspections[0].inferred_form)
    return cast(DataForm, detected) if detected in {"x", "t", "xt"} else None


def _detected_dependence_label(inspections: list[UploadInspection]) -> str:
    form = _detected_profile_form(inspections)
    return _FORM_LABELS.get(str(form), "Unknown")


def _preview_outputs(route: MappingRoute) -> dict[str, str]:
    """Return the actual plottable members produced by the selected route."""

    output_keys = _BUNDLE_OUTPUTS.get(route.target_key, (route.target,))
    outputs = {
        key: (
            f"{_OUTPUT_LABELS.get(key, key.removesuffix('_xt').replace('_', ' ').title())} (target)"
            if key == route.target
            else _OUTPUT_LABELS.get(
                key,
                key.removesuffix("_xt").replace("_", " ").title(),
            )
        )
        for key in output_keys
        if key not in {"fit_history", "residual"}
    }
    input_keys = {_friendly_input_key(item.key) for item in route.inputs}
    if "replication_timing" in input_keys and route.target_key in {
        "firing_rate",
        "kinetic_fields",
    }:
        outputs.setdefault("replication_timing_fit", "Replication timing fit")
        outputs[f"{_COMPARISON_PREFIX}replication_timing_fit"] = (
            "Replication timing: observed vs fitted"
        )
    outputs[_FIT_DIAGNOSTIC_KEY] = "Residual / misfit error"
    return outputs


def _result_preview_outputs(result: MappingResult) -> dict[str, str]:
    """Return every output available in one completed Mapping result."""

    outputs = {
        key: (
            f"{_OUTPUT_LABELS.get(key, key.removesuffix('_xt').replace('_', ' ').title())} "
            "(target)"
            if key == result.target
            else _OUTPUT_LABELS.get(
                key,
                key.removesuffix("_xt").replace("_", " ").title(),
            )
        )
        for key in result.outputs
        if key not in {"fit_history", "residual"}
    }
    if "residual" in result.outputs or "fit_history" in result.outputs:
        outputs[_FIT_DIAGNOSTIC_KEY] = "Residual / misfit error"
    for fitted_key, comparison in result.comparisons.items():
        source_label = _OUTPUT_LABELS.get(
            comparison.source_key,
            comparison.source_key.replace("_", " ").title(),
        )
        outputs[f"{_COMPARISON_PREFIX}{fitted_key}"] = f"{source_label}: observed vs fitted"
    return outputs


def _diagnostic_output_key(result: Any) -> str:
    """Resolve the route-specific residual or optimizer-error output."""

    outputs = getattr(result, "outputs", {})
    if "residual" in outputs:
        return "residual"
    if "fit_history" in outputs:
        return "fit_history"
    raise ValueError("This mapping route did not return residuals or a fit-error history.")


def _resolved_output_key(result: Any, output_key: str) -> str:
    """Resolve a virtual diagnostic key without failing UI synchronization."""

    if output_key != _FIT_DIAGNOSTIC_KEY:
        return output_key
    outputs = getattr(result, "outputs", {})
    return next(
        (candidate for candidate in ("residual", "fit_history") if candidate in outputs),
        output_key,
    )


def _result_axis(result: Any, *, kind: str, length: int) -> np.ndarray:
    """Return a matching result axis while tolerating scalar and legacy results."""

    names = (
        ("time_min", "time_axis", "time_grid")
        if kind == "time"
        else ("position_kb", "x_axis", "position")
    )
    for name in names:
        candidate = getattr(result, name, None)
        if candidate is None:
            continue
        axis = np.asarray(candidate, dtype=float).reshape(-1)
        if axis.size == length:
            return axis
    return np.arange(length, dtype=float)


def _periodic_mapping(result: Any) -> bool:
    """Return whether rmap evaluated the selected region as a closed ring."""

    assumptions = getattr(result, "assumptions", {})
    geometry = str(assumptions.get("geometry", "")).strip().casefold()
    return bool(assumptions.get("perQ", False)) or geometry == "periodic"


def _closed_periodic_profile(
    axis: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Append the first ring bin at the next coordinate for an honest closed preview."""

    coordinates = np.asarray(axis, dtype=float).reshape(-1)
    profile = np.asarray(values, dtype=float).reshape(-1)
    if coordinates.size < 2 or coordinates.size != profile.size:
        return coordinates, profile
    steps = np.diff(coordinates)
    finite_steps = steps[np.isfinite(steps) & (steps > 0)]
    step = float(np.median(finite_steps)) if finite_steps.size else 1.0
    return (
        np.append(coordinates, coordinates[-1] + step),
        np.append(profile, profile[0]),
    )


def _mapping_plot_options(
    result: Any,
    *,
    output_key: str,
    plot_type: str,
    normalization: str,
    title: str,
    x_label: str,
    y_label: str,
    color: str,
    palette: str,
    histogram_bins: int,
    line_width: float,
    show_markers: bool,
    show_colorbar: bool,
    log_y: bool,
    invert_y: bool,
    log_z: bool = False,
    series_label: str = "",
    density: bool = False,
    show_legend: bool = True,
    legend_position: str = "Top right",
    show_grid: bool = True,
    y_limits: tuple[float, float] | None = None,
    secondary_color: str = "#b03412",
) -> dict[str, Any]:
    """Build a bounded ECharts view for scalar, X, T, XT, and bundle outputs."""

    if output_key == _FIT_DIAGNOSTIC_KEY:
        output_key = _diagnostic_output_key(result)

    if output_key.startswith(_COMPARISON_PREFIX):
        fitted_key = output_key.removeprefix(_COMPARISON_PREFIX)
        comparison = getattr(result, "comparisons", {}).get(fitted_key)
        if comparison is None:
            raise ValueError("The selected fit comparison was not returned by DNAscape.")
        if plot_type != "Line":
            raise ValueError("Observed-versus-fitted comparisons use a line plot.")
        observed = np.asarray(comparison.observed, dtype=float).reshape(-1)
        fitted = np.asarray(comparison.fitted, dtype=float).reshape(-1)
        finite = np.concatenate((finite_values(observed), finite_values(fitted)))
        if finite.size == 0:
            raise ValueError("The fit comparison contains no finite values.")
        if normalization == "Z-score":
            center = float(np.mean(finite))
            scale = float(np.std(finite))
            observed = observed - center
            fitted = fitted - center
            if scale > 0:
                observed = observed / scale
                fitted = fitted / scale
        elif normalization == "Min-max [0,1]":
            lower = float(np.min(finite))
            span = float(np.max(finite) - lower)
            observed = observed - lower
            fitted = fitted - lower
            if span > 0:
                observed = observed / span
                fitted = fitted / span
        axis_kind = "time" if comparison.form == "t" else "space"
        axis = (
            np.arange(observed.size, dtype=float)
            if comparison.form in {"scalar", "regions", "supplied"}
            else _result_axis(result, kind=axis_kind, length=observed.size)
        )
        if comparison.form == "x" and _periodic_mapping(result):
            observed_axis, observed = _closed_periodic_profile(axis, observed)
            fitted_axis, fitted = _closed_periodic_profile(axis, fitted)
        else:
            observed_axis = fitted_axis = axis
        return build_preview_plot_options(
            plot_type="Line",
            title=title.strip() or "Observed vs fitted",
            x_label=x_label,
            y_label=y_label,
            color=color,
            palette=palette,
            histogram_bins=histogram_bins,
            density=density,
            show_legend=show_legend,
            show_colorbar=show_colorbar,
            log_y=log_y,
            invert_y=invert_y,
            y_limits=y_limits,
            series_label=series_label,
            lines=(
                PreviewLine("Observed", observed_axis, observed, color),
                PreviewLine("Fitted", fitted_axis, fitted, secondary_color),
            ),
            line_width=line_width,
            show_markers=show_markers,
            legend_position=legend_position,
            show_grid=show_grid,
            interactive=comparison.form != "t",
        )

    outputs = getattr(result, "outputs", {})
    if output_key not in outputs:
        raise ValueError("The selected mapping output was not returned by DNAscape.")
    values = np.asarray(outputs[output_key], dtype=float)
    if values.size == 0 or not np.isfinite(values).any():
        raise ValueError("The selected mapping output contains no finite values.")
    if normalization == "Z-score":
        finite = finite_values(values)
        scale = float(np.std(finite))
        values = values - float(np.mean(finite))
        if scale > 0:
            values = values / scale
    elif normalization == "Min-max [0,1]":
        finite = finite_values(values)
        lower = float(np.min(finite))
        span = float(np.max(finite) - lower)
        values = values - lower
        if span > 0:
            values = values / span

    resolved_title = title.strip() or _OUTPUT_LABELS.get(
        output_key,
        output_key.replace("_", " ").title(),
    )
    if plot_type == "Heatmap":
        output_form = str(getattr(result, "output_forms", {}).get(output_key, ""))
        if output_form == "t":
            time_axis = _result_axis(result, kind="time", length=values.size)
            space_axis = np.asarray(getattr(result, "x_axis", []), dtype=float).reshape(-1)
            if space_axis.size == 0:
                space_axis = np.asarray([0.0])
            if _periodic_mapping(result) and space_axis.size > 1:
                space_axis, _ = _closed_periodic_profile(
                    space_axis,
                    np.zeros(space_axis.size, dtype=float),
                )
            display_matrix = np.repeat(values.reshape(-1, 1), space_axis.size, axis=1)
        elif output_form == "xt":
            matrix = np.asarray(values, dtype=float)
            if matrix.ndim != 2:
                raise ValueError("Heatmap view requires a temporal or space-time mapping output.")
            space_axis = _result_axis(result, kind="space", length=matrix.shape[0])
            time_axis = _result_axis(result, kind="time", length=matrix.shape[1])
            if _periodic_mapping(result) and matrix.shape[0] > 1:
                space_axis, _ = _closed_periodic_profile(space_axis, matrix[:, 0])
                matrix = np.concatenate((matrix, matrix[:1, :]), axis=0)
            # rmap returns canonical (position, time) arrays. DNAscape's
            # plotting API transposes them for a position-X, time-Y image.
            display_matrix = matrix.T
        else:
            raise ValueError("Heatmap view requires a temporal or space-time mapping output.")
        return build_preview_plot_options(
            plot_type="Heatmap",
            title=resolved_title,
            x_label=x_label,
            y_label=y_label,
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
            series_label=resolved_title,
            values=display_matrix,
            x_axis=space_axis,
            y_axis=time_axis,
            show_grid=show_grid,
            interactive=False,
        )

    flattened = values.reshape(-1)
    if plot_type == "Histogram":
        return build_preview_plot_options(
            plot_type="Histogram",
            title=resolved_title,
            x_label=x_label,
            y_label=y_label or ("Density" if density else "Count"),
            color=color,
            palette=palette,
            histogram_bins=histogram_bins,
            density=density,
            show_legend=show_legend,
            show_colorbar=show_colorbar,
            log_y=log_y,
            invert_y=invert_y,
            y_limits=y_limits,
            series_label=series_label.strip() or resolved_title,
            values=flattened,
            legend_position=legend_position,
            show_grid=show_grid,
        )

    if plot_type == "Boxplot":
        return build_preview_plot_options(
            plot_type="Boxplot",
            title=resolved_title,
            x_label=x_label,
            y_label=y_label,
            color=color,
            palette=palette,
            histogram_bins=histogram_bins,
            density=density,
            show_legend=False,
            show_colorbar=show_colorbar,
            log_y=log_y,
            invert_y=invert_y,
            y_limits=y_limits,
            series_label=series_label.strip() or resolved_title,
            values=flattened,
            legend_position=legend_position,
            show_grid=show_grid,
        )

    if values.ndim > 1 and 1 not in values.shape:
        raise ValueError("Choose Heatmap to view this two-dimensional mapping output.")
    dependence = str(getattr(result, "output_dependence", "space"))
    output_form = str(getattr(result, "output_forms", {}).get(output_key, ""))
    axis_kind = "time" if output_form == "t" or dependence == "time" else "space"
    if output_key == "fit_history" or dependence in {"ensemble", "regions", "constant"}:
        axis = np.arange(flattened.size, dtype=float)
    else:
        axis = _result_axis(result, kind=axis_kind, length=flattened.size)
    if axis_kind == "space" and output_form == "x" and _periodic_mapping(result):
        axis, flattened = _closed_periodic_profile(axis, flattened)
    return build_preview_plot_options(
        plot_type="Line",
        title=resolved_title,
        x_label=x_label,
        y_label=y_label,
        color=color,
        palette=palette,
        histogram_bins=histogram_bins,
        density=density,
        show_legend=show_legend,
        show_colorbar=show_colorbar,
        log_y=log_y,
        invert_y=invert_y,
        y_limits=y_limits,
        series_label=series_label.strip() or resolved_title,
        values=flattened,
        x_axis=axis,
        line_width=line_width,
        show_markers=show_markers,
        legend_position=legend_position,
        show_grid=show_grid,
        interactive=output_form != "t" and dependence != "time",
    )


def _mapping_automatic_y_limits(
    result: Any,
    *,
    output_key: str,
    plot_type: str,
    histogram_bins: int,
    density: bool,
    log_y: bool,
) -> tuple[float, float]:
    """Return editable, data-derived bounds for the active mapping plot."""

    if output_key == _FIT_DIAGNOSTIC_KEY:
        output_key = _diagnostic_output_key(result)

    if output_key.startswith(_COMPARISON_PREFIX):
        comparison = getattr(result, "comparisons", {}).get(
            output_key.removeprefix(_COMPARISON_PREFIX)
        )
        if comparison is None:
            raise ValueError("The selected fit comparison was not returned by DNAscape.")
        values = [comparison.observed, comparison.fitted]
        y_axis = None
        fixed_limits = None
        quantity_key = comparison.source_key.removesuffix("_xt")
    else:
        output = np.asarray(getattr(result, "outputs", {}).get(output_key), dtype=float)
        output_form = str(getattr(result, "output_forms", {}).get(output_key, ""))
        if output_form == "t":
            time_length = output.size
        elif output_form == "xt" and output.ndim == 2:
            time_length = output.shape[1]
        else:
            time_length = 0
        if plot_type == "Heatmap" and time_length == 0:
            raise ValueError("Heatmap view requires a temporal or space-time output.")
        values = [output]
        y_axis = (
            _result_axis(result, kind="time", length=time_length)
            if plot_type == "Heatmap"
            else None
        )
        quantity_key = output_key.removesuffix("_xt")
        fixed_limits = FIXED_QUANTITY_Y_LIMITS.get(quantity_key)
    nonnegative = quantity_key not in {"fork_directionality", "residual"}
    return automatic_y_limits(
        values,
        plot_type=plot_type,
        histogram_bins=histogram_bins,
        density=density,
        log_y=log_y,
        y_axis=y_axis,
        fixed_limits=fixed_limits,
        floor_at_zero=nonnegative,
    )


def _render_mapping_result(
    result: Any,
    *,
    output_key: str,
    plot_type: str,
    normalization: str,
    title: str,
    series_label: str,
    x_label: str,
    y_label: str,
    color: str,
    secondary_color: str,
    palette: str,
    histogram_bins: int,
    line_width: float,
    show_markers: bool,
    show_colorbar: bool,
    log_y: bool,
    log_z: bool,
    invert_y: bool,
    density: bool,
    show_legend: bool,
    legend_position: str,
    y_limits: tuple[float, float] | None,
    on_download: Callable[[str], Any],
    show_grid: bool = True,
) -> Any:
    options = _mapping_plot_options(
        result,
        output_key=output_key,
        plot_type=plot_type,
        normalization=normalization,
        title=title,
        x_label=x_label,
        y_label=y_label,
        color=color,
        secondary_color=secondary_color,
        palette=palette,
        histogram_bins=histogram_bins,
        line_width=line_width,
        show_markers=show_markers,
        show_colorbar=show_colorbar,
        log_y=log_y,
        log_z=log_z,
        invert_y=invert_y,
        series_label=series_label,
        density=density,
        show_legend=show_legend,
        legend_position=legend_position,
        show_grid=show_grid,
        y_limits=y_limits,
    )
    return render_preview_chart(
        options,
        title=title,
        marker="mapping-result-plot",
        on_download=on_download,
    )


def _setting_value(setting: Any) -> Any:
    """Choose a useful initial control value from exact route metadata."""

    if (
        setting.key in _INTERACTIVE_SETTING_CAPS
        and setting.has_default
        and isinstance(setting.default, (int, float))
        and not isinstance(setting.default, bool)
    ):
        return min(setting.default, _INTERACTIVE_SETTING_CAPS[setting.key])
    if setting.kind in {"number_pair", "number_list"}:
        if setting.has_default and setting.default is not None:
            value = setting.default
            if isinstance(value, (tuple, list)):
                return ", ".join(str(item) for item in value)
            return str(value)
        return ""
    if setting.kind == "text":
        return "" if setting.default is None else str(setting.default)
    if setting.has_default and setting.default is not None:
        return setting.default
    if setting.kind == "boolean":
        return False
    if setting.kind == "integer":
        return int(setting.min_value or 0)
    if setting.kind == "number":
        return float(setting.min_value or 0.0)
    if setting.allowed_values:
        return setting.allowed_values[0]
    return ""


def _setting_label(setting: Any) -> str:
    if setting.key == "tau":
        label = "Stall lifetime"
    else:
        label = setting.label
    return f"{label} ({setting.unit})" if setting.unit else label


def _render_setting_field(setting: Any) -> Any:
    """Render one typed, route-valid setting and return its NiceGUI control."""

    marker = f"mapping-setting-{_slug(setting.key)}"
    value = _setting_value(setting)
    label = _setting_label(setting)
    if setting.kind == "boolean":
        return (
            ui.checkbox(label, value=bool(value))
            .classes("dn-mapping-setting-checkbox")
            .mark(marker)
        )
    if setting.kind == "choice":
        options = {item: str(item).replace("_", " ").title() for item in setting.allowed_values}
        return (
            ui.select(options, value=value, label=label)
            .props("dense options-dense")
            .classes("dn-field")
            .mark(marker)
        )
    if setting.kind in {"integer", "number"}:
        if isinstance(value, float) and not math.isfinite(value):
            return (
                ui.input(label=label, value="infinite")
                .props("dense")
                .classes("dn-field")
                .mark(marker)
            )
        kwargs: dict[str, Any] = {"label": label, "value": value}
        if setting.min_value is not None:
            kwargs["min"] = setting.min_value
        if setting.max_value is not None:
            kwargs["max"] = setting.max_value
        if setting.step is not None:
            kwargs["step"] = setting.step
        return ui.number(**kwargs).props("dense").classes("dn-field").mark(marker)
    placeholder = "e.g. 0.5, 2.0" if setting.kind == "number_pair" else None
    control = ui.input(label=label, value=value, placeholder=placeholder)
    return control.props("dense").classes("dn-field").mark(marker)


def _render_setting(
    setting: Any,
    on_change: Callable[[], Any] | None = None,
) -> _SettingControl:
    """Render a setting, progressively disclosing nullable opt-in values."""

    with ui.element("div").classes("dn-mapping-setting"):
        needs_override = (not setting.required) and (
            not setting.has_default or (setting.nullable and setting.default is None)
        )
        if needs_override:
            override_label = (
                "Allow stall recovery" if setting.key == "tau" else f"Set {_setting_label(setting)}"
            )
            override = (
                ui.checkbox(override_label, value=False)
                .classes("dn-mapping-setting-override")
                .mark(f"mapping-setting-{_slug(setting.key)}-override")
            )
            with ui.element("div").classes("dn-mapping-setting-optional") as field_holder:
                control = _render_setting_field(setting)
            field_holder.bind_visibility_from(override, "value")
        else:
            override = None
            control = _render_setting_field(setting)
        control.tooltip(setting.description)
        if on_change is not None:
            control.on_value_change(on_change)
            if override is not None:
                override.on_value_change(on_change)
    return _SettingControl(setting=setting, control=control, override=override)


def _render_route_setting_groups(
    route: MappingRoute,
    *,
    location: SettingLocation,
    target_shape: str | None = None,
    on_change: Callable[[], Any] | None = None,
) -> dict[str, _SettingControl]:
    """Render categorized controls for one route and one UI location."""

    settings = tuple(
        setting
        for setting in route.settings_for(location=location)
        if target_shape == "weibull" or setting.key not in _WEIBULL_ONLY_SETTINGS
    )
    if not settings:
        ui.label("No additional controls are needed for this route.").classes(
            "dn-mapping-input-hint"
        )
        return {}
    controls: dict[str, _SettingControl] = {}
    present_categories = set(setting.category for setting in settings)
    categories = tuple(
        category for category in _SETTING_CATEGORY_LABELS if category in present_categories
    ) + tuple(
        dict.fromkeys(
            setting.category
            for setting in settings
            if setting.category not in _SETTING_CATEGORY_LABELS
        )
    )
    with ui.element("div").classes("dn-mapping-settings-themes"):
        for category in categories:
            category_settings = tuple(
                setting for setting in settings if setting.category == category
            )
            with (
                ui.column()
                .classes("dn-setting-group dn-mapping-settings-theme")
                .mark(f"mapping-settings-{category}")
            ):
                ui.label(_SETTING_CATEGORY_LABELS.get(category, category.title())).classes(
                    "dn-setting-group-title"
                )
                for setting in category_settings:
                    controls[setting.key] = _render_setting(setting, on_change)
    return controls


def _read_number(control: Any, label: str) -> float:
    if control.value is None or str(control.value).strip() == "":
        raise ValueError(f"{label} is required.")
    try:
        value = float(control.value)
    except (TypeError, ValueError) as exc:
        if str(control.value).strip().casefold() in {"inf", "infinite", "infinity"}:
            return math.inf
        raise ValueError(f"{label} must be a number.") from exc
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite.")
    return value


def _read_setting(live: _SettingControl) -> Any:
    setting = live.setting
    if live.override is not None and not bool(live.override.value):
        raise KeyError(setting.key)
    value = live.control.value
    if setting.kind == "boolean":
        return bool(value)
    if setting.kind == "integer":
        numeric = _read_number(live.control, setting.label)
        if not numeric.is_integer():
            raise ValueError(f"{setting.label} must be a whole number.")
        return int(numeric)
    if setting.kind == "number":
        return _read_number(live.control, setting.label)
    if setting.kind in {"number_pair", "number_list"}:
        parts = [part for part in str(value or "").replace(",", " ").split() if part]
        if setting.kind == "number_pair" and len(parts) != 2:
            raise ValueError(f"{setting.label} must contain exactly two numbers.")
        if not parts:
            if setting.nullable:
                return None
            raise ValueError(f"{setting.label} is required.")
        try:
            return tuple(float(part) for part in parts)
        except ValueError as exc:
            raise ValueError(f"{setting.label} must contain only numbers.") from exc
    if setting.kind == "text" and str(value or "").strip() == "" and setting.nullable:
        return None
    return value


def _render_prepared_input(input_spec: MappingInput, marker: str) -> None:
    with (
        ui.element("div")
        .classes("dn-mapping-prepared-result")
        .mark(f"mapping-input-{marker}-prepared")
    ):
        ui.icon("inventory_2").classes("dn-mapping-prepared-icon")
        with ui.column().classes("dn-mapping-prepared-copy"):
            ui.label("No compatible prepared result in this session").classes(
                "dn-mapping-prepared-title"
            )
        prepared = (
            ui.select(
                ["No compatible result"],
                value="No compatible result",
                label="Result",
            )
            .props("dense options-dense")
            .classes("dn-field dn-mapping-prepared-select")
            .mark(f"mapping-input-{marker}-prepared-select")
        )
        prepared.disable()


def _render_regions_input(
    input_spec: MappingInput,
    marker: str,
    surface: dict[str, Any],
    on_change: Callable[[], Any],
    on_error: Callable[[str], Any],
) -> None:
    source = (
        ui.select(
            {"manual": "Enter regions", "upload": "Upload a region table"},
            value="manual",
            label="Region source",
        )
        .props("dense options-dense")
        .classes("dn-field dn-mapping-source-choice")
        .mark("mapping-regions-source")
    )
    with ui.element("div").classes("dn-mapping-regions-manual") as manual_fields:
        editor = (
            ui.textarea(
                label="Regions",
                placeholder="origin_1, 100, 150\norigin_2, 320, 380",
            )
            .props("dense outlined autogrow rows=2")
            .classes("dn-field dn-mapping-regions-editor")
            .mark("mapping-regions-editor")
        )
        surface.update(kind="regions-manual", editor=editor, source=source)
        editor.on_value_change(on_change)
    with ui.element("div").classes("dn-mapping-regions-upload") as upload_fields:
        inspections: list[UploadInspection] = []

        async def uploaded(event: Any) -> bool:
            try:
                inspection = await inspect_upload_event(event, expected_form="regions")
            except (OSError, TypeError, ValueError) as exc:
                on_error(str(exc))
                return False
            inspections[:] = [inspection]
            surface.update(kind="upload", inspections=inspections, source=source)
            cache_upload(inspection, source="Mapping", navigate_to="mapping")
            on_change()
            return True

        def cleared(_event: Any) -> None:
            inspections.clear()
            on_change()

        uploader = data_upload(
            accept=input_spec.accept_for("regions"),
            accepted_formats=_format_copy(input_spec, "regions"),
            marker="mapping-regions-upload",
            show_accepted_formats=False,
            on_upload=uploaded,
            on_clear=cleared,
            hide_when_loaded=True,
        )
        surface["uploader"] = uploader
    manual_fields.bind_visibility_from(source, "value", backward=lambda value: value == "manual")
    upload_fields.bind_visibility_from(source, "value", backward=lambda value: value == "upload")
    source.on_value_change(on_change)


def _render_upload_input(
    input_spec: MappingInput,
    profile_forms: tuple[DataForm, ...],
    marker: str,
    surface: dict[str, Any],
    retained: _MappingUploadState,
    on_change: Callable[[], Any],
    on_error: Callable[[str], Any],
    on_detected_form: Callable[[DataForm], Any],
    single_file: bool = False,
) -> None:
    # A per-client prepared-result workspace is not available yet.  Ordinary
    # factors therefore expose only the working upload path instead of a dead
    # selectable option.
    source = None
    allow_multiple = not single_file and any(
        input_spec.multiple_for(candidate)  # type: ignore[arg-type]
        for candidate in profile_forms
    )
    with ui.element("div").classes("dn-mapping-upload-source"):
        inspections = retained.inspections
        selector_controls: dict[str, tuple[Any | None, Any | None]] = {}
        selection_values = retained.selection_values
        materialized_cache = retained.materialized_cache
        track_options: Any | None = None

        def sync_selection_values() -> None:
            """Preserve valid table-column and NPZ-array choices across refreshes."""

            active_names = {inspection.name for inspection in inspections}
            for stale_name in tuple(selection_values):
                if stale_name not in active_names:
                    selection_values.pop(stale_name, None)
            for inspection in inspections:
                selected = selection_values.setdefault(inspection.name, {})
                if inspection.numeric_columns:
                    valid_columns = set(inspection.numeric_columns)
                    default_column = inspection.default_column
                    if _detected_profile_form(inspections) == "xt" and len(inspections) == 1:
                        valid_columns.add("__all__")
                        default_column = "__all__"
                    if selected.get("column") not in valid_columns:
                        selected["column"] = default_column
                else:
                    selected["column"] = None
                if inspection.array_keys:
                    if selected.get("array_key") not in inspection.array_keys:
                        selected["array_key"] = inspection.default_array_key
                else:
                    selected["array_key"] = None

        def sync_track_options_visibility() -> None:
            if track_options is None:
                return
            track_options.set_visibility(
                bool(inspections)
                and all(
                    inspection.source_format in {"bedgraph", "bigwig"} for inspection in inspections
                )
            )

        @ui.refreshable
        def render_inspection() -> None:
            sync_selection_values()
            if not inspections:
                return
            with (
                ui.element("div")
                .classes("dn-mapping-detected-data")
                .mark(f"mapping-input-{marker}-detected-data")
            ):
                ui.label(f"Dependence: {_detected_dependence_label(inspections)}").classes(
                    "dn-mapping-detected-summary dn-mapping-detected-dependence"
                )
                for inspection in inspections:
                    with ui.element("div").classes("dn-mapping-detected-file"):
                        shape = (
                            " x ".join(str(size) for size in inspection.shape)
                            if inspection.shape
                            else "determined on Map"
                        )
                        ui.label(f"Array size: {shape}").classes("dn-mapping-detected-summary")

        async def uploaded(event: Any) -> bool:
            try:
                inspection = await inspect_upload_event(event, expected_form="auto")
            except (OSError, TypeError, ValueError) as exc:
                on_error(str(exc))
                return False
            materialized_cache.update(key=None, value=None)
            if not allow_multiple:
                inspections.clear()
            inspections[:] = [item for item in inspections if item.name != inspection.name]
            inspections.append(inspection)
            detected_form = _detected_profile_form(inspections)
            if detected_form not in profile_forms:
                inspections[:] = [item for item in inspections if item.name != inspection.name]
                expected = ", ".join(_FORM_LABELS[item] for item in profile_forms)
                message = (
                    f"{input_spec.label} was detected as "
                    f"{_FORM_LABELS.get(str(detected_form), 'an unsupported profile')}; "
                    f"this route accepts {expected}."
                )
                on_error(message)
                return False
            sync_selection_values()
            surface.update(
                inspections=inspections,
                selector_controls=selector_controls,
                selection_values=selection_values,
                source=source,
            )
            render_inspection.refresh()
            sync_track_options_visibility()
            assert detected_form is not None
            on_detected_form(detected_form)
            cache_upload(inspection, source="Mapping", navigate_to="mapping")
            on_change()
            return True

        def cleared(event: Any) -> None:
            materialized_cache.update(key=None, value=None)
            name = str(getattr(event, "name", ""))
            if name:
                inspections[:] = [item for item in inspections if item.name != name]
            else:
                inspections.clear()
            sync_selection_values()
            render_inspection.refresh()
            sync_track_options_visibility()
            on_change()

        available_formats = {
            item
            for candidate in profile_forms
            for item in input_spec.formats_for(candidate)  # type: ignore[arg-type]
        }
        accept_tokens = {
            extension.strip()
            for candidate in profile_forms
            for extension in input_spec.accept_for(candidate).split(",")  # type: ignore[arg-type]
            if extension.strip()
        }
        upload_accept = ",".join(sorted(accept_tokens))
        if single_file and profile_forms == ("xt",):
            available_formats.difference_update({"bigwig", "bedgraph"})
            track_extensions = {".bedgraph", ".bdg", ".bw", ".bigwig"}
            upload_accept = ",".join(
                extension
                for extension in upload_accept.split(",")
                if extension not in track_extensions
            )
        uploader = data_upload(
            accept=upload_accept,
            accepted_formats=", ".join(
                _FORMAT_LABELS.get(item, item.upper()) for item in sorted(available_formats)
            ),
            marker=f"source-{marker}-upload",
            show_accepted_formats=False,
            on_upload=uploaded,
            on_clear=cleared,
            multiple=allow_multiple,
            max_files=None if allow_multiple else 1,
            hide_when_loaded=True,
            initial_names=tuple(item.name for item in inspections),
        )
        render_inspection()
        surface.update(
            inspections=inspections,
            selector_controls=selector_controls,
            selection_values=selection_values,
            source=source,
            uploader=uploader,
            materialized_cache=materialized_cache,
        )
        track_capable = bool({"bigwig", "bedgraph"} & available_formats)
        if track_capable:
            with (
                ui.element("div")
                .classes("dn-mapping-track-options")
                .mark(f"mapping-input-{marker}-track-options")
            ) as track_options:
                ui.label("Genomic interval & binning").classes("dn-field-heading")
                with ui.element("div").classes("dn-mapping-track-grid"):
                    track_chromosome = (
                        ui.input(label="Chromosome", placeholder="e.g. chr1")
                        .props("dense")
                        .classes("dn-field")
                        .mark(f"mapping-input-{marker}-track-chromosome")
                    )
                    track_bin_size = (
                        ui.number(label="Bin size (bp)", value=1000, min=1, step=1)
                        .props("dense")
                        .classes("dn-field")
                        .mark(f"mapping-input-{marker}-track-bin-size")
                    )
                    track_start = (
                        ui.number(label="Start (bp, inclusive; optional)", min=0, step=1)
                        .props("dense")
                        .classes("dn-field")
                        .mark(f"mapping-input-{marker}-track-start")
                    )
                    track_end = (
                        ui.number(label="End (bp, exclusive; optional)", min=1, step=1)
                        .props("dense")
                        .classes("dn-field")
                        .mark(f"mapping-input-{marker}-track-end")
                    )
                    track_aggregation = (
                        ui.select(
                            {
                                "mean": "Mean",
                                "sum": "Sum",
                                "max": "Maximum",
                                "min": "Minimum",
                                "coverage": "Coverage",
                                "std": "Standard deviation",
                            },
                            value="mean",
                            label="Bin aggregation",
                        )
                        .props("dense options-dense")
                        .classes("dn-field")
                        .mark(f"mapping-input-{marker}-track-aggregation")
                    )
                    track_value_column = (
                        ui.input(label="Value column (bedGraph)")
                        .props("dense")
                        .classes("dn-field")
                        .mark(f"mapping-input-{marker}-track-value-column")
                    )
                with (
                    ui.element("details")
                    .classes("dn-mapping-track-preprocessing")
                    .mark(f"mapping-input-{marker}-track-preprocessing")
                ):
                    with ui.element("summary").classes("dn-mapping-import-summary"):
                        ui.label("Preprocessing")
                    with ui.element("div").classes("dn-mapping-track-grid"):
                        missing = (
                            ui.select(
                                {
                                    "nan": "Keep missing",
                                    "zero": "Replace with zero",
                                    "fill": "Replace with value",
                                    "drop": "Drop missing bins",
                                },
                                value="nan",
                                label="Missing values",
                            )
                            .props("dense options-dense")
                            .classes("dn-field")
                            .mark(f"mapping-input-{marker}-track-missing")
                        )
                        fill_value = (
                            ui.number(label="Fill value", value=0.0, step=0.1)
                            .props("dense")
                            .classes("dn-field")
                            .mark(f"mapping-input-{marker}-track-fill-value")
                        )
                        interpolation = (
                            ui.select(
                                {
                                    "none": "None",
                                    "linear": "Linear",
                                    "ffill": "Forward fill",
                                    "bfill": "Backward fill",
                                },
                                value="none",
                                label="Interpolation",
                            )
                            .props("dense options-dense")
                            .classes("dn-field")
                            .mark(f"mapping-input-{marker}-track-interpolation")
                        )
                        smooth_window = (
                            ui.number(label="Smoothing window (bins)", value=1, min=1, step=1)
                            .props("dense")
                            .classes("dn-field")
                            .mark(f"mapping-input-{marker}-track-smooth-window")
                        )
                        normalization = (
                            ui.select(
                                {
                                    "none": "None",
                                    "minmax": "Min-max [0,1]",
                                    "zscore": "Z-score",
                                    "center": "Center",
                                    "maxabs": "Maximum absolute value",
                                },
                                value="none",
                                label="Normalization",
                            )
                            .props("dense options-dense")
                            .classes("dn-field")
                            .mark(f"mapping-input-{marker}-track-normalize")
                        )
                    fill_value.bind_visibility_from(
                        missing, "value", backward=lambda value: value == "fill"
                    )
                surface["track"] = {
                    "chromosome": track_chromosome,
                    "bin_size": track_bin_size,
                    "start": track_start,
                    "end": track_end,
                    "aggregation": track_aggregation,
                    "value_column": track_value_column,
                    "missing": missing,
                    "fill_value": fill_value,
                    "interpolation": interpolation,
                    "smooth_window": smooth_window,
                    "normalize": normalization,
                }
                for control in surface["track"].values():
                    control.on_value_change(on_change)
            sync_track_options_visibility()


def _render_mapping_input(
    input_spec: MappingInput,
    *,
    forms: tuple[DataForm, ...] | None = None,
    selected_form: DataForm | None = None,
    retained: _MappingUploadState | None = None,
    on_change: Callable[[], Any] | None = None,
    on_error: Callable[[str], Any] | None = None,
    single_file: bool = False,
) -> _MappingInputControl:
    """Render one route-family input and expose its live form control.

    Route families merge exact ``rmap`` variants which use the same biological
    factors.  The caller owns compatibility between their form selectors; this
    renderer owns only the selected input's surface.
    """

    marker = _slug(input_spec.key)
    available_forms = forms or input_spec.forms
    if not available_forms:
        raise ValueError(f"Mapping input {input_spec.key!r} has no supported forms.")
    upload_state = retained or _MappingUploadState()
    profile_forms: tuple[DataForm, ...] = tuple(
        cast(DataForm, form) for form in available_forms if form in {"x", "t", "xt"}
    )
    retained_form = _detected_profile_form(upload_state.inspections)
    if retained_form in profile_forms:
        default_form = retained_form
    else:
        default_form = selected_form if selected_form in available_forms else available_forms[0]
    defaulted_override = input_spec.role == "feature_assumption" and input_spec.has_default
    surface: dict[str, Any] = {}
    scientific_change = on_change or (lambda: None)
    scientific_error = on_error or (lambda _message: None)
    with ui.element("section").classes("dn-mapping-input-card").mark(f"mapping-input-{marker}"):
        with ui.element("div").classes("dn-mapping-input-head"):
            heading = ui.label(input_spec.label).classes("dn-setting-group-title")
            heading.tooltip(input_spec.description)
            if defaulted_override:
                ui.label("Uses package default").classes("dn-mapping-input-status")
        override = None
        if defaulted_override:
            with ui.element("div").classes("dn-mapping-default-row"):
                override = (
                    ui.checkbox("Override", value=False)
                    .classes("dn-mapping-default-toggle")
                    .mark(f"mapping-input-{marker}-override")
                )
                ui.label(f"Package default: {_display_default(input_spec.default)}").classes(
                    "dn-mapping-default-copy"
                )

        with ui.element("div").classes("dn-mapping-override-fields") as input_fields:
            if profile_forms:
                mode_options: dict[str, str] = {"profile": "Profile"}
                if "scalar" in available_forms:
                    mode_options = {"scalar": "Constant", **mode_options}
                default_mode = "scalar" if default_form == "scalar" else "profile"
            else:
                mode_options = {
                    str(form): _FORM_LABELS.get(str(form), str(form).replace("_", " ").title())
                    for form in available_forms
                }
                default_mode = str(default_form)
            mode_control = (
                ui.select(mode_options, value=default_mode, label="Input type")
                .props("dense options-dense")
                .classes("dn-field dn-mapping-input-mode")
                .mark(f"mapping-input-{marker}-type")
            )
            if len(mode_options) == 1:
                mode_control.disable()

            # The exact rmap form is inferred from profile shape. Keep it as a
            # live route-resolution value without making users choose an
            # implementation detail before they upload their data.
            form_control = (
                ui.select(
                    _select_options(available_forms, _FORM_LABELS),
                    value=default_form,
                )
                .props('dense options-dense aria-label="Detected profile dependence"')
                .classes("dn-mapping-exact-form")
                .mark(f"mapping-input-{marker}-form")
            )
            form_control.set_visibility(False)

            def detected_form_changed(detected: DataForm) -> None:
                if detected not in profile_forms:
                    return
                form_control.set_value(detected)
                scientific_change()

            @ui.refreshable
            def render_input_surface() -> None:
                mode = str(mode_control.value or default_mode)
                surface.clear()
                surface["mode"] = mode_control
                with (
                    ui.element("div")
                    .classes("dn-mapping-input-surface")
                    .mark(f"mapping-input-{marker}-surface")
                ):
                    if input_spec.prepared_result_only or mode == "bundle":
                        _render_prepared_input(input_spec, marker)
                        surface["prepared_only"] = True
                    elif mode == "regions":
                        _render_regions_input(
                            input_spec,
                            marker,
                            surface,
                            scientific_change,
                            scientific_error,
                        )
                    elif mode == "scalar":
                        units = _SCALAR_UNITS.get(input_spec.key, "")
                        label = f"Constant value{f' ({units})' if units else ''}"
                        scalar = (
                            ui.number(
                                label=label,
                                value=_scalar_default(input_spec),
                                min=0,
                                step=0.001 if input_spec.key == "firing_rate" else 0.1,
                            )
                            .props("dense")
                            .classes("dn-field")
                            .mark(f"mapping-input-{marker}-constant")
                        )
                        surface["scalar"] = scalar
                        scalar.on_value_change(scientific_change)
                    else:
                        _render_upload_input(
                            input_spec,
                            profile_forms,
                            marker,
                            surface,
                            upload_state,
                            scientific_change,
                            scientific_error,
                            detected_form_changed,
                            single_file=single_file,
                        )

            render_input_surface()

            def mode_changed() -> None:
                mode = str(mode_control.value or default_mode)
                if mode == "profile":
                    detected = _detected_profile_form(upload_state.inspections)
                    desired = (
                        detected if detected in profile_forms else cast(DataForm, profile_forms[0])
                    )
                else:
                    desired = cast(DataForm, mode)
                form_control.set_value(desired)
                render_input_surface.refresh()
                scientific_change()

            mode_control.on_value_change(mode_changed)

            def refresh_input_surface() -> None:
                form = str(form_control.value or default_form)
                desired_mode = "profile" if form in profile_forms else form
                if desired_mode in mode_options and mode_control.value != desired_mode:
                    mode_control.value = desired_mode
                    mode_control.update()
                render_input_surface.refresh()

        if override is not None:
            input_fields.bind_visibility_from(override, "value")
    return _MappingInputControl(
        spec=input_spec,
        form=form_control,
        refresh_surface=refresh_input_surface,
        override=override,
        surface=surface,
    )


def _selected_upload_value(live: _MappingInputControl) -> Any:
    """Materialize the current upload only when Map is selected."""

    source = live.surface.get("source")
    if source is not None and str(source.value) == "prepared":
        raise ValueError(f"{live.spec.label}: choose an uploaded file.")
    inspections = cast(list[UploadInspection], live.surface.get("inspections", []))
    if not inspections:
        raise ValueError(f"Upload {live.spec.label.lower()} data before selecting Map.")
    selection_values = cast(
        dict[str, dict[str, str | None]],
        live.surface.get("selection_values", {}),
    )

    def choices(inspection: UploadInspection) -> tuple[str | None, str | None]:
        selected = selection_values.get(inspection.name, {})
        column = selected.get("column")
        array_key = selected.get("array_key")
        return (
            None if column is None or str(column) == "__all__" else str(column),
            None if array_key is None else str(array_key),
        )

    expected_form = str(live.form.value or "auto")
    prepared_inspections = [
        replace(inspection, expected_form=expected_form) for inspection in inspections
    ]

    if len(prepared_inspections) == 1:
        column, array_key = choices(prepared_inspections[0])
        return materialize_upload(
            prepared_inspections[0],
            column=column,
            array_key=array_key,
        )
    columns: list[str | None] = []
    array_keys: list[str | None] = []
    for inspection in prepared_inspections:
        column, array_key = choices(inspection)
        columns.append(column)
        array_keys.append(array_key)
    return materialize_many(
        prepared_inspections,
        columns=columns,
        array_keys=array_keys,
    )


def _read_mapping_input(live: _MappingInputControl) -> Any:
    form = str(live.form.value)
    if live.override is not None and not bool(live.override.value):
        raise KeyError(live.spec.key)
    if form == "scalar":
        control = live.surface.get("scalar")
        if control is None:
            raise ValueError(f"{live.spec.label} has no constant value control.")
        return _read_number(control, live.spec.label)
    if form == "regions":
        source = live.surface.get("source")
        if source is not None and str(source.value) == "upload":
            return _selected_upload_value(live)
        editor = live.surface.get("editor")
        lines = [line.strip() for line in str(editor.value if editor else "").splitlines()]
        parsed: list[list[float]] = []
        for line in lines:
            if not line:
                continue
            fields = [field.strip() for field in line.replace("\t", ",").split(",")]
            numeric = fields[-2:]
            try:
                parsed.append([float(numeric[0]), float(numeric[1])])
            except (IndexError, ValueError) as exc:
                raise ValueError(f"{live.spec.label}: enter one start/end pair per line.") from exc
        if not parsed:
            raise ValueError(f"Enter at least one region for {live.spec.label.lower()}.")
        return np.asarray(parsed, dtype=float)
    if live.surface.get("prepared_only"):
        raise ValueError(
            f"{live.spec.label} requires a prepared DNAscape result. No compatible result "
            "is available in this session; choose another route or first create that result."
        )
    return _selected_upload_value(live)


async def _read_mapping_input_async(
    live: _MappingInputControl,
    *,
    missing_policy: str = "error",
    region_start_kb: float | None = None,
    region_end_kb: float | None = None,
    resolution_x: float | None = None,
    whole_genome: bool = False,
) -> tuple[Any, bool]:
    """Read a live input without parsing uploaded arrays on NiceGUI's event loop."""

    form = str(live.form.value)
    if live.surface.get("prepared_only"):
        return _read_mapping_input(live), False
    source = live.surface.get("source")
    uploaded = form not in {"scalar", "regions", "bundle"} or (
        form == "regions" and source is not None and str(source.value) == "upload"
    )
    if not uploaded:
        return _read_mapping_input(live), False
    if live.override is not None and not bool(live.override.value):
        raise KeyError(live.spec.key)
    if source is not None and str(source.value) == "prepared":
        raise ValueError(f"{live.spec.label}: choose an uploaded file.")
    inspections = tuple(cast(list[UploadInspection], live.surface.get("inspections", [])))
    if not inspections:
        raise ValueError(f"Upload {live.spec.label.lower()} data before selecting Map.")
    materialized_cache = cast(
        dict[str, Any],
        live.surface.setdefault("materialized_cache", {"key": None, "value": None}),
    )
    inspection_key = tuple((id(item), item.name) for item in inspections)
    selection_values = cast(
        dict[str, dict[str, str | None]],
        live.surface.get("selection_values", {}),
    )
    track_controls = cast(dict[str, Any] | None, live.surface.get("track"))
    use_track = track_controls is not None and all(
        item.source_format in {"bedgraph", "bigwig"} for item in inspections
    )
    if use_track:
        assert track_controls is not None
        chromosome = str(track_controls["chromosome"].value or "").strip() or None
        if any(item.source_format == "bigwig" for item in inspections) and chromosome is None:
            raise ValueError("Choose a chromosome before mapping BigWig data.")
        explicit_value_column = str(track_controls["value_column"].value or "").strip() or None
        detected_value_columns = {
            str(selected["column"])
            for selected in selection_values.values()
            if selected.get("column") is not None and str(selected["column"]) != "__all__"
        }
        if explicit_value_column is None and len(detected_value_columns) > 1:
            raise ValueError("Choose the same bedGraph value column for every uploaded time point.")
        selected_value_column = explicit_value_column or next(
            iter(detected_value_columns),
            None,
        )
        use_region_window = (
            not whole_genome and region_start_kb is not None and region_end_kb is not None
        )
        if whole_genome and resolution_x is not None:
            start_bp = None
            end_bp = None
            bin_size_bp = int(round(resolution_x * 1_000))
            if bin_size_bp <= 0:
                raise ValueError("Spatial resolution must span at least one base pair.")
        elif use_region_window:
            assert resolution_x is not None
            assert region_start_kb is not None
            assert region_end_kb is not None
            start_bp = int(round(region_start_kb * 1_000))
            end_bp = int(round(region_end_kb * 1_000))
            bin_size_bp = int(round(resolution_x * 1_000))
            if bin_size_bp <= 0:
                raise ValueError("Spatial resolution must span at least one base pair.")
        else:
            start_bp = (
                None
                if track_controls["start"].value is None
                else int(track_controls["start"].value)
            )
            end_bp = (
                None if track_controls["end"].value is None else int(track_controls["end"].value)
            )
            bin_size_bp = int(_read_number(track_controls["bin_size"], "Track bin size"))
        track_kwargs = {
            "chromosome": chromosome,
            "start": start_bp,
            "end": end_bp,
            "bin_size": bin_size_bp,
            "aggregation": str(track_controls["aggregation"].value or "mean"),
            "value_column": selected_value_column,
            "missing": str(track_controls["missing"].value or "nan"),
            "fill_value": _read_number(track_controls["fill_value"], "Track fill value"),
            "interpolation": str(track_controls["interpolation"].value or "none"),
            "smooth_window": int(_read_number(track_controls["smooth_window"], "Smoothing window")),
            "normalize": str(track_controls["normalize"].value or "none"),
        }
        cache_key = ("track", inspection_key, tuple(track_kwargs.items()))
        if materialized_cache.get("key") == cache_key:
            return materialized_cache["value"], use_region_window
        if len(inspections) == 1:
            values = await run.io_bound(
                materialize_track_upload,
                inspections[0],
                **track_kwargs,
            )
        else:
            values = await run.io_bound(
                materialize_many_tracks,
                inspections,
                **track_kwargs,
            )
        materialized_cache.update(key=cache_key, value=values)
        return values, use_region_window
    columns: list[str | None] = []
    array_keys: list[str | None] = []
    for inspection in inspections:
        selected = selection_values.get(inspection.name, {})
        column = selected.get("column")
        array_key = selected.get("array_key")
        columns.append(None if column is None or str(column) == "__all__" else str(column))
        array_keys.append(None if array_key is None else str(array_key))
    prepared_inspections = tuple(
        replace(inspection, expected_form=form) for inspection in inspections
    )
    cache_key = (
        "numeric",
        inspection_key,
        form,
        tuple(columns),
        tuple(array_keys),
        missing_policy,
    )
    if materialized_cache.get("key") == cache_key:
        return materialized_cache["value"], False
    if len(inspections) == 1:
        values = await run.io_bound(
            materialize_upload,
            prepared_inspections[0],
            column=columns[0],
            array_key=array_keys[0],
            missing_policy=missing_policy,
        )
    else:
        values = await run.io_bound(
            materialize_many,
            prepared_inspections,
            columns=columns,
            array_keys=array_keys,
            missing_policy=missing_policy,
        )
    materialized_cache.update(key=cache_key, value=values)
    return values, False


def render(_navigate: Callable[[str], None]) -> None:
    """Render mapping routes, inputs, configuration, and output placeholders."""

    running = False
    active_cancel: Event | None = None
    restored_entry = consume_cached_workflow("mapping")
    restored_result = restored_entry.payload if restored_entry is not None else None
    latest_result: MappingResult | None = (
        restored_result if isinstance(restored_result, MappingResult) else None
    )
    result_chart: Any | None = None
    updating_y_limits = False
    y_limits_automatic = True
    mapping_input_controls: dict[str, _MappingInputControl] = {}
    mapping_upload_states: dict[str, _MappingUploadState] = {}
    region_setting_controls: dict[str, _SettingControl] = {}
    mapping_setting_controls: dict[str, _SettingControl] = {}
    region_selector_refresh: dict[str, Callable[[], Any]] = {"call": lambda: None}

    page_intro(
        "Mapping",
        "Map replication kinetic landscapes",
        "Infer replication features from measured profiles.",
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
                    targets = _MAPPING_TARGETS
                    selected_target = targets[0]
                    selected_dependence = selected_target.default_dependence
                    selected_families = mapping_route_families(
                        selected_target.key,
                        selected_dependence,
                    )
                    selected_family = selected_families[0]
                    initial_form_selections = _default_form_selections(selected_family)
                    selected_route = selected_family.resolve(initial_form_selections)
                    selected_route_state = {"route": selected_route}

                    with ui.element("div").classes("dn-mapping-request-grid"):
                        with ui.column().classes("dn-setting-group dn-mapping-request-group"):
                            ui.label("Target").classes("dn-setting-group-title").mark(
                                "mapping-heading-target"
                            )
                            target_control = (
                                ui.select(
                                    {target.key: target.label for target in targets},
                                    value=selected_target.key,
                                    label="Mapped output",
                                )
                                .props("dense options-dense")
                                .classes("dn-field")
                                .mark("mapping-target")
                            )

                        with ui.column().classes("dn-setting-group dn-mapping-request-group"):
                            ui.label("Dependence").classes("dn-setting-group-title").mark(
                                "mapping-heading-dependence"
                            )
                            dependence_control = (
                                ui.select(
                                    _select_options(
                                        selected_target.dependences,
                                        _DEPENDENCE_LABELS,
                                    ),
                                    value=selected_dependence,
                                    label="Output form",
                                )
                                .props("dense options-dense")
                                .classes("dn-field")
                                .mark("mapping-target-form")
                            )
                            if len(selected_target.dependences) == 1:
                                dependence_control.disable()

                            initial_shapes = selected_route.target_shapes
                            initial_shape = selected_route.default_target_shape or (
                                initial_shapes[0] if initial_shapes else None
                            )
                            with (
                                ui.element("div")
                                .classes("dn-mapping-target-shape")
                                .mark("mapping-target-shape-field")
                            ) as target_shape_container:
                                target_shape_control = (
                                    ui.select(
                                        {
                                            shape: shape.replace("_", " ").title()
                                            for shape in initial_shapes
                                        },
                                        value=initial_shape,
                                        label="Target shape",
                                    )
                                    .props("dense options-dense")
                                    .classes("dn-field")
                                    .mark("mapping-target-shape")
                                )
                            target_shape_container.set_visibility(len(initial_shapes) > 1)

                        with ui.column().classes("dn-setting-group dn-mapping-request-group"):
                            ui.label("Route").classes("dn-setting-group-title").mark(
                                "mapping-heading-route"
                            )
                            route_control = (
                                ui.select(
                                    _route_options(selected_families),
                                    value=selected_family.key,
                                    label="Input factors",
                                )
                                .props("dense options-dense")
                                .classes("dn-field dn-mapping-route-select")
                                .mark("mapping-route")
                            )

                    with ui.element("div").classes("dn-mapping-required-heading"):
                        ui.label("Inputs").classes("dn-setting-group-title").mark(
                            "mapping-heading-inputs"
                        )

                    def sync_target_shape(route: MappingRoute) -> None:
                        shapes = route.target_shapes
                        target_shape_control.options = {
                            shape: shape.replace("_", " ").title() for shape in shapes
                        }
                        target_shape_control.value = route.default_target_shape or (
                            shapes[0] if shapes else None
                        )
                        target_shape_control.update()
                        target_shape_container.set_visibility(len(shapes) > 1)

                    def sync_preview_selection(route: MappingRoute) -> None:
                        invalidate_mapping_result()
                        if latest_result is not None:
                            return
                        mapping_stats["Form"].text = _DEPENDENCE_LABELS.get(
                            str(dependence_control.value),
                            str(dependence_control.value).replace("_", " ").title(),
                        )
                        mapping_stats["Form"].update()
                        outputs = _preview_outputs(route)
                        previous_output = str(preview_output.value or "")
                        preview_output.options = outputs
                        preview_output.value = (
                            previous_output if previous_output in outputs else next(iter(outputs))
                        )
                        if len(outputs) == 1:
                            preview_output.disable()
                        else:
                            preview_output.enable()
                        preview_output.update()
                        apply_preview_output()

                    def sync_exact_route(route: MappingRoute) -> None:
                        """Apply the exact method resolved from the visible form choices."""

                        selected_route_state["route"] = route
                        sync_target_shape(route)
                        render_region_settings.refresh()
                        render_mapping_settings.refresh()
                        region_selector_refresh["call"]()
                        sync_preview_selection(route)

                    def scientific_input_changed() -> None:
                        invalidate_mapping_result()
                        region_selector_refresh["call"]()

                    @ui.refreshable
                    def render_required_inputs() -> None:
                        mapping_input_controls.clear()
                        family = _route_family(str(route_control.value))
                        selections = _retained_form_selections(
                            family,
                            mapping_upload_states,
                        )
                        initial_route = family.resolve(selections)
                        displayed_forms: dict[str, DataForm] = {}
                        controls: dict[str, Any] = {}
                        refreshers: dict[str, Callable[[], Any]] = {}
                        overrides: dict[str, Any] = {}
                        single_file = len(family.required_factor_keys) == 1
                        synchronizing = False

                        with (
                            ui.element("div")
                            .classes("dn-mapping-input-grid")
                            .mark("mapping-required-inputs")
                        ):
                            for input_spec in family.inputs:
                                key = input_spec.key
                                allowed = family.allowed_forms(key)
                                initial_form = family.variant_input(initial_route, key).forms[0]
                                if initial_form not in allowed:
                                    initial_form = allowed[0]
                                displayed_forms[key] = selections.get(key, initial_form)
                                retained = mapping_upload_states.setdefault(
                                    _friendly_input_key(key),
                                    _MappingUploadState(),
                                )
                                live = _render_mapping_input(
                                    input_spec,
                                    forms=allowed,
                                    selected_form=displayed_forms[key],
                                    retained=retained,
                                    on_change=scientific_input_changed,
                                    on_error=lambda message: report_mapping_input_error(message),
                                    single_file=single_file,
                                )
                                control = live.form
                                controls[key] = control
                                refreshers[key] = live.refresh_surface
                                mapping_input_controls[key] = live
                                if live.override is not None:
                                    overrides[key] = live.override

                        def sync_input_forms(changed_key: str | None = None) -> None:
                            """Keep every input-form selector on an implemented combination."""

                            nonlocal synchronizing
                            if synchronizing:
                                return
                            if changed_key is not None and controls[changed_key].value is None:
                                return
                            synchronizing = True
                            try:
                                if changed_key is not None:
                                    selected_value = cast(
                                        DataForm,
                                        str(controls[changed_key].value),
                                    )
                                    displayed_forms[changed_key] = selected_value
                                    selections[changed_key] = selected_value
                                candidates = family.compatible_routes(selections)
                                if not candidates:
                                    if changed_key is None:
                                        raise ValueError(
                                            "No route accepts the selected input forms."
                                        )
                                    assert changed_key is not None
                                    candidates = family.compatible_routes(
                                        {changed_key: selections[changed_key]}
                                    )
                                if not candidates:
                                    selection_copy = ", ".join(
                                        f"{key}={value}" for key, value in selections.items()
                                    )
                                    raise ValueError(f"No route accepts {selection_copy}.")

                                def mismatch_count(route: MappingRoute) -> int:
                                    return sum(
                                        selections[key]
                                        not in family.variant_input(route, key).forms
                                        for key in selections
                                        if key != changed_key
                                    )

                                template = min(candidates, key=mismatch_count)
                                for key in tuple(selections):
                                    if key == changed_key:
                                        continue
                                    template_input = family.variant_input(template, key)
                                    if selections[key] not in template_input.forms:
                                        selections[key] = template_input.forms[0]
                                        displayed_forms[key] = selections[key]

                                for key in family.input_keys:
                                    allowed_forms = family.allowed_forms(key)
                                    control = controls[key]
                                    previous = str(control.value)
                                    current = selections.get(key, displayed_forms[key])
                                    if current not in allowed_forms:
                                        current = allowed_forms[0]
                                    displayed_forms[key] = current
                                    if key in selections:
                                        selections[key] = current
                                    control.options = _select_options(
                                        allowed_forms,
                                        _FORM_LABELS,
                                    )
                                    control.value = current
                                    if len(allowed_forms) == 1:
                                        control.disable()
                                    else:
                                        control.enable()
                                    control.update()
                                    if previous != current or key == changed_key:
                                        refreshers[key]()

                                sync_exact_route(family.resolve(selections))
                            finally:
                                synchronizing = False

                        def sync_optional_input(key: str) -> None:
                            if bool(overrides[key].value):
                                selections[key] = displayed_forms[key]
                                sync_input_forms(key)
                            else:
                                selections.pop(key, None)
                                sync_input_forms()

                        for key, control in controls.items():
                            control.on_value_change(
                                lambda input_key=key: sync_input_forms(input_key)
                            )
                        for key, override in overrides.items():
                            override.on_value_change(
                                lambda input_key=key: sync_optional_input(input_key)
                            )

                    def sync_selected_family() -> None:
                        family_key = route_control.value
                        if family_key is None or family_key not in route_control.options:
                            return
                        family = _route_family(str(family_key))
                        selections = _retained_form_selections(
                            family,
                            mapping_upload_states,
                        )
                        selected = family.resolve(selections)
                        render_required_inputs.refresh()
                        sync_exact_route(selected)

                    def sync_route_families() -> None:
                        if target_control.value is None or dependence_control.value is None:
                            return
                        families = mapping_route_families(
                            str(target_control.value),
                            str(dependence_control.value),
                        )
                        route_control.options = _route_options(families)
                        route_control.value = families[0].key
                        route_control.update()
                        sync_selected_family()

                    def sync_target() -> None:
                        if target_control.value is None:
                            return
                        target = mapping_target(str(target_control.value))
                        dependence_control.options = _select_options(
                            target.dependences,
                            _DEPENDENCE_LABELS,
                        )
                        dependence_control.value = target.default_dependence
                        dependence_control.update()
                        if len(target.dependences) == 1:
                            dependence_control.disable()
                        else:
                            dependence_control.enable()
                        sync_route_families()

                    render_required_inputs()
                    target_control.on_value_change(sync_target)
                    dependence_control.on_value_change(sync_route_families)
                    route_control.on_value_change(sync_selected_family)

                    def target_shape_changed() -> None:
                        render_mapping_settings.refresh()
                        invalidate_mapping_result()

                    target_shape_control.on_value_change(target_shape_changed)

                with expansion("Region", icon="straighten"):

                    @ui.refreshable
                    def render_region_upload_selectors() -> None:
                        """Place uploaded table/array choices beside the region scope."""

                        for live in mapping_input_controls.values():
                            inspections = cast(
                                list[UploadInspection],
                                live.surface.get("inspections", []),
                            )
                            selection_values = cast(
                                dict[str, dict[str, str | None]],
                                live.surface.get("selection_values", {}),
                            )
                            selector_controls = cast(
                                dict[str, tuple[Any | None, Any | None]],
                                live.surface.setdefault("selector_controls", {}),
                            )
                            selector_controls.clear()
                            for inspection in inspections:
                                selected = selection_values.setdefault(inspection.name, {})
                                column_control = None
                                if inspection.numeric_columns:
                                    column_options: Any = list(inspection.numeric_columns)
                                    if str(live.form.value) == "xt" and len(inspections) == 1:
                                        column_options = {
                                            "__all__": "All signal columns",
                                            **{
                                                column: column
                                                for column in inspection.numeric_columns
                                            },
                                        }
                                    column_control = (
                                        ui.select(
                                            column_options,
                                            value=selected.get("column"),
                                            label="Value column",
                                        )
                                        .props("dense options-dense")
                                        .classes("dn-field dn-mapping-column-select")
                                        .mark(f"mapping-input-{_slug(live.spec.key)}-value-column")
                                    )
                                    column_control._props["data-upload-name"] = inspection.name
                                    if (
                                        len(inspection.numeric_columns) == 1
                                        and str(live.form.value) != "xt"
                                    ):
                                        column_control.disable()

                                    def store_column(
                                        control: Any = column_control,
                                        values: dict[str, str | None] = selected,
                                        surface: dict[str, Any] = live.surface,
                                    ) -> None:
                                        values["column"] = (
                                            None if control.value is None else str(control.value)
                                        )
                                        surface["materialized_cache"].update(
                                            key=None,
                                            value=None,
                                        )
                                        invalidate_mapping_result()

                                    column_control.on_value_change(store_column)
                                array_control = None
                                if inspection.array_keys:
                                    array_control = (
                                        ui.select(
                                            list(inspection.array_keys),
                                            value=selected.get("array_key"),
                                            label="Array",
                                        )
                                        .props("dense options-dense")
                                        .classes("dn-field dn-mapping-column-select")
                                        .mark(f"mapping-input-{_slug(live.spec.key)}-array-key")
                                    )
                                    array_control._props["data-upload-name"] = inspection.name
                                    if len(inspection.array_keys) == 1:
                                        array_control.disable()

                                    def store_array(
                                        control: Any = array_control,
                                        values: dict[str, str | None] = selected,
                                        surface: dict[str, Any] = live.surface,
                                    ) -> None:
                                        values["array_key"] = (
                                            None if control.value is None else str(control.value)
                                        )
                                        surface["materialized_cache"].update(
                                            key=None,
                                            value=None,
                                        )
                                        invalidate_mapping_result()

                                    array_control.on_value_change(store_array)
                                selector_controls[inspection.name] = (
                                    column_control,
                                    array_control,
                                )

                    with ui.element("div").classes("dn-mapping-region-choice-grid"):
                        with (
                            ui.element("div")
                            .classes("dn-mapping-region-selectors")
                            .mark("mapping-region-value-selectors")
                        ):
                            render_region_upload_selectors()
                        whole_region = (
                            ui.checkbox("Whole region", value=False)
                            .classes("dn-mapping-whole-region")
                            .mark("mapping-whole-region")
                        )
                    region_selector_refresh["call"] = render_region_upload_selectors.refresh
                    with (
                        ui.element("div")
                        .classes("dn-mapping-window-grid")
                        .mark("mapping-spatial-window")
                    ):
                        region_start = (
                            ui.number(label="Start (kb)", value=0.0, min=0.0, step=1.0)
                            .props("dense")
                            .classes("dn-field")
                            .mark("mapping-region-start")
                        )
                        region_end = (
                            ui.number(
                                label="End (kb, exclusive)",
                                value=1000.0,
                                min=0.000001,
                                step=1.0,
                            )
                            .props("dense")
                            .classes("dn-field")
                            .mark("mapping-region-end")
                        )

                    @ui.refreshable
                    def render_region_settings() -> None:
                        route = selected_route_state["route"]
                        region_setting_controls.clear()
                        with (
                            ui.element("div")
                            .classes("dn-mapping-settings")
                            .mark("mapping-region-settings")
                        ):
                            region_setting_controls.update(
                                _render_route_setting_groups(
                                    route,
                                    location="region",
                                    on_change=lambda: invalidate_mapping_result(),
                                )
                            )

                    render_region_settings()

                with expansion("Mapping settings", icon="tune"):
                    with ui.element("div").classes("dn-mapping-settings-grid"):
                        _region_missing = (
                            ui.select(
                                {
                                    "error": "Report error",
                                    "interpolate": "Interpolate",
                                    "zero": "Replace with zero",
                                },
                                value="error",
                                label="Missing values",
                            )
                            .props("dense options-dense")
                            .classes("dn-field")
                            .mark("mapping-region-missing")
                        )

                    @ui.refreshable
                    def render_mapping_settings() -> None:
                        route = selected_route_state["route"]
                        mapping_setting_controls.clear()
                        with (
                            ui.element("div")
                            .classes("dn-mapping-settings")
                            .mark("mapping-settings")
                        ):
                            mapping_setting_controls.update(
                                _render_route_setting_groups(
                                    route,
                                    location="settings",
                                    target_shape=(
                                        str(target_shape_control.value)
                                        if target_shape_control.value is not None
                                        else None
                                    ),
                                    on_change=lambda: invalidate_mapping_result(),
                                )
                            )

                    render_mapping_settings()

                with action_dock():
                    map_button = (
                        ui.button("Map", icon="map")
                        .props("no-caps")
                        .classes("dn-action-primary")
                        .mark("action-map")
                    )

        with ui.column().classes("dn-preview-stack"):
            with panel(extra_classes="dn-preview-card"):
                with (
                    ui.element("div")
                    .classes("dn-preview-heading-row")
                    .mark("mapping-preview-header-row")
                ):
                    panel_title("Preview")
                initial_outputs = (
                    _result_preview_outputs(latest_result)
                    if latest_result is not None
                    else _preview_outputs(selected_route)
                )
                initial_output_key = (
                    latest_result.primary_output
                    if latest_result is not None
                    and latest_result.primary_output in initial_outputs
                    else next(iter(initial_outputs))
                )
                initial_output_label = _OUTPUT_LABELS.get(
                    initial_output_key,
                    initial_output_key.replace("_", " ").title(),
                )
                with (
                    ui.element("div")
                    .classes("dn-stat-grid dn-simulation-summary-grid dn-mapping-summary-grid")
                    .mark("mapping-preview-output-bar")
                ):
                    with (
                        ui.element("div")
                        .classes("dn-stat dn-simulation-output-stat")
                        .mark("mapping-output-stat")
                    ):
                        ui.label("Output").classes("dn-stat-label")
                        preview_output = (
                            ui.select(initial_outputs, value=initial_output_key)
                            .props('dense borderless options-dense aria-label="Preview output"')
                            .classes("dn-field dn-simulation-output-select")
                            .mark("mapping-preview-output")
                        )
                        if len(initial_outputs) == 1:
                            preview_output.disable()
                    with (
                        ui.element("div")
                        .classes("dn-mapping-run-stats")
                        .mark("mapping-stats") as stats_container
                    ):
                        mapping_stats = stat_strip(
                            (
                                ("Form", ""),
                                ("Points", ""),
                                ("Method", ""),
                            )
                        )
                stats_container.visible = False
                progress = (
                    ui.linear_progress(value=0.0, show_value=False)
                    .props("rounded instant-feedback")
                    .classes("dn-simulation-progress dn-mapping-progress")
                    .mark("mapping-progress")
                )
                progress.visible = False
                status = ui.label("Ready").classes("dn-muted").mark("mapping-status")
                with (
                    ui.element("div")
                    .classes("w-full")
                    .mark("mapping-result-container") as result_slot
                ):
                    preview_plot(
                        "No result yet",
                        kind="mapping",
                    )
                plot_settings = preview_settings_controls(
                    initial_title=initial_output_label,
                    initial_series_label="Mapped",
                    initial_x_label="Position (kb)",
                    initial_y_label=initial_output_label,
                    on_reset_y_limits=lambda: populate_automatic_y_limits(),
                    markers={
                        "container": "mapping-preview-settings",
                        "view": "mapping-view",
                        "title": "mapping-preview-title",
                        "series_label": "mapping-series-label",
                        "x_axis_label": "mapping-preview-x-label",
                        "y_axis_label": "mapping-preview-y-label",
                        "primary_color": "mapping-line-or-bar-color",
                        "secondary_color": "mapping-second-curve-color",
                        "line_width": "mapping-line-width",
                        "legend_position": "mapping-legend-position",
                        "show_grid": "mapping-show-grid",
                        "heatmap_palette": "mapping-heatmap-palette",
                        "log_z": "mapping-log-z",
                        "histogram_bins": "mapping-histogram-bins",
                        "density": "mapping-density",
                        "colorbar": "mapping-colorbar",
                        "y_minimum": "mapping-y-minimum",
                        "y_maximum": "mapping-y-maximum",
                        "reset_y_limits": "mapping-reset-y-limits",
                    },
                )
                view = plot_settings.view
                preview_title = plot_settings.title
                series_label = plot_settings.series_label
                preview_x_label = plot_settings.x_axis_label
                preview_y_label = plot_settings.y_axis_label
                line_or_bar_color = plot_settings.primary_color
                second_curve_color = plot_settings.secondary_color
                heatmap_palette = plot_settings.heatmap_palette
                log_z = plot_settings.log_z
                line_width = plot_settings.line_width
                histogram_bins = plot_settings.histogram_bins
                show_legend = plot_settings.show_legend
                legend_position = plot_settings.legend_position
                show_grid = plot_settings.show_grid
                density = plot_settings.density
                colorbar = plot_settings.colorbar
                log_y = plot_settings.log_y
                invert_y = plot_settings.invert_y
                y_minimum = plot_settings.y_minimum
                y_maximum = plot_settings.y_maximum
                raw_controls = raw_output_controls(
                    file_name=f"dnascape_mapping_{initial_output_key}",
                    default_type="NPY",
                )
                raw_controls.file_name.mark("mapping-raw-filename")

                def report_mapping_input_error(message: str) -> None:
                    """Report upload failures through the shared transient message surface."""

                    ui.notify(message, type="negative")
                    status.text = "Check the uploaded data"

                def invalidate_mapping_result() -> None:
                    """A changed request must not disturb the last completed preview."""

                    if running:
                        return
                    if latest_result is None:
                        status.text = "Ready"

                def resolved_plot_type() -> str:
                    output_key = str(
                        preview_output.value
                        or (latest_result.primary_output if latest_result else initial_output_key)
                    )
                    if output_key.startswith(_COMPARISON_PREFIX):
                        return "Line"
                    actual_output_key = (
                        _resolved_output_key(latest_result, output_key)
                        if latest_result is not None
                        else output_key
                    )
                    output_form = (
                        latest_result.output_forms.get(actual_output_key)
                        if latest_result is not None
                        else selected_route_state["route"].output_form
                    )
                    if output_form in {"t", "xt"}:
                        return "Heatmap"
                    requested_view = str(view.value or "Default")
                    if requested_view == "Histogram":
                        return "Histogram"
                    if requested_view == "Boxplot":
                        return "Boxplot"
                    dependence = (
                        latest_result.output_dependence
                        if latest_result is not None
                        else str(dependence_control.value)
                    )
                    return "Histogram" if dependence == "ensemble" else "Line"

                def sync_plot_controls() -> None:
                    plot_type = resolved_plot_type()
                    output_key = str(
                        preview_output.value
                        or (latest_result.primary_output if latest_result else initial_output_key)
                    )
                    multiple_curves = output_key.startswith(_COMPARISON_PREFIX)
                    actual_output_key = (
                        _resolved_output_key(latest_result, output_key)
                        if latest_result is not None
                        else output_key
                    )
                    output_form = (
                        latest_result.output_forms.get(actual_output_key)
                        if latest_result is not None
                        else selected_route_state["route"].output_form
                    )
                    dependence = (
                        latest_result.output_dependence
                        if latest_result is not None
                        else str(dependence_control.value)
                    )
                    sync_preview_settings_visibility(
                        plot_settings,
                        plot_type=plot_type,
                        multiple_curves=multiple_curves,
                        allow_histogram=output_form not in {"t", "xt"} and not multiple_curves,
                        allow_boxplot=dependence == "ensemble" and not multiple_curves,
                    )

                def current_y_limits() -> tuple[float, float] | None:
                    if y_minimum.value is None or y_maximum.value is None:
                        return None
                    limits = (
                        _read_number(y_minimum, "Y minimum"),
                        _read_number(y_maximum, "Y maximum"),
                    )
                    if limits[1] < limits[0] or (
                        limits[1] == limits[0] and resolved_plot_type() != "Heatmap"
                    ):
                        raise ValueError("Y maximum must be greater than Y minimum.")
                    return limits

                def populate_automatic_y_limits(*, redraw: bool = True) -> None:
                    nonlocal updating_y_limits, y_limits_automatic
                    y_limits_automatic = True
                    if latest_result is None:
                        return
                    try:
                        limits = _mapping_automatic_y_limits(
                            latest_result,
                            output_key=str(preview_output.value or latest_result.primary_output),
                            plot_type=resolved_plot_type(),
                            histogram_bins=int(_read_number(histogram_bins, "Histogram bins")),
                            density=bool(density.value),
                            log_y=bool(log_y.value),
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
                    sync_plot_controls()
                    if y_limits_automatic:
                        populate_automatic_y_limits()
                    else:
                        render_latest_result()

                def setting_values() -> dict[str, Any]:
                    values: dict[str, Any] = {}
                    for key, live in {
                        **region_setting_controls,
                        **mapping_setting_controls,
                    }.items():
                        try:
                            values[key] = _read_setting(live)
                        except KeyError:
                            continue
                    return values

                def freeze_scientific_controls() -> list[Any]:
                    """Freeze the request surface so a running result cannot become stale."""

                    frozen: list[Any] = []
                    visited: set[int] = set()

                    def freeze(value: Any) -> None:
                        identity = id(value)
                        if identity in visited:
                            return
                        visited.add(identity)
                        if isinstance(value, dict):
                            for item in value.values():
                                freeze(item)
                            return
                        if isinstance(value, (tuple, list)):
                            for item in value:
                                freeze(item)
                            return
                        if is_data_upload(value):
                            if bool(getattr(value, "enabled", False)):
                                set_data_upload_locked(value, True)
                                frozen.append(value)
                            return
                        if hasattr(value, "disable") and bool(getattr(value, "enabled", False)):
                            value.disable()
                            frozen.append(value)

                    for control in (
                        target_control,
                        dependence_control,
                        route_control,
                        target_shape_control,
                        whole_region,
                        region_start,
                        region_end,
                        _region_missing,
                    ):
                        freeze(control)
                    for live in mapping_input_controls.values():
                        freeze(live.form)
                        freeze(live.override)
                        freeze(live.surface)
                    for live in {
                        **region_setting_controls,
                        **mapping_setting_controls,
                    }.values():
                        freeze(live.control)
                        freeze(live.override)
                    return frozen

                def restore_scientific_controls(frozen: list[Any]) -> None:
                    for control in frozen:
                        if is_data_upload(control):
                            set_data_upload_locked(control, False)
                        else:
                            control.enable()

                async def download_plot(file_format: str) -> None:
                    if latest_result is None or result_chart is None:
                        ui.notify("Run Mapping before downloading a plot.", type="warning")
                        return
                    try:
                        normalized_dpi = int(get_export_settings().plot_dpi)
                        pixel_ratio = max(1.0, min(4.0, normalized_dpi / 96.0))
                        data_uri = await result_chart.run_chart_method(
                            "getDataURL",
                            {
                                "type": "png",
                                "pixelRatio": pixel_ratio,
                                "backgroundColor": "#ffffff",
                            },
                            timeout=10,
                        )
                        output_key = str(preview_output.value or latest_result.primary_output)
                        output_stem = output_key.replace(":", "_")
                        artifact = await run.io_bound(
                            build_chart_capture_export,
                            data_uri,
                            file_format,
                            filename=f"dnascape_mapping_{output_stem}_plot",
                            dpi=normalized_dpi,
                        )
                        if artifact is None or not artifact.data:
                            raise RuntimeError("Plot export stopped before returning data.")
                    except (RuntimeError, TimeoutError, TypeError, ValueError) as exc:
                        ui.notify(str(exc), type="negative")
                        return
                    cache_artifact(
                        artifact,
                        source="Mapping",
                        kind="plot",
                        detail=output_key.replace(":", " ").replace("_", " ").title(),
                        navigate_to="mapping",
                    )
                    ui.download(artifact.data, artifact.filename, artifact.media_type)

                async def download_raw(controls: RawOutputControls) -> None:
                    if latest_result is None:
                        ui.notify("Run Mapping before downloading data.", type="warning")
                        return
                    output_key = str(preview_output.value or latest_result.primary_output)
                    try:
                        if output_key.startswith(_COMPARISON_PREFIX):
                            fitted_key = output_key.removeprefix(_COMPARISON_PREFIX)
                            artifact = await run.io_bound(
                                build_mapping_comparison_raw_export,
                                latest_result,
                                fitted_key,
                                str(controls.file_type.value or "NPY"),
                                filename=str(
                                    controls.file_name.value
                                    or f"dnascape_mapping_{fitted_key}_comparison"
                                ),
                                compression=(
                                    get_export_settings().raw_compression
                                    if controls.compress.value
                                    else "none"
                                ),
                            )
                        else:
                            exported_output_key = (
                                _diagnostic_output_key(latest_result)
                                if output_key == _FIT_DIAGNOSTIC_KEY
                                else output_key
                            )
                            artifact = await run.io_bound(
                                build_mapping_raw_export,
                                latest_result,
                                exported_output_key,
                                str(controls.file_type.value or "NPY"),
                                filename=str(
                                    controls.file_name.value
                                    or f"dnascape_mapping_{exported_output_key}"
                                ),
                                compression=(
                                    get_export_settings().raw_compression
                                    if controls.compress.value
                                    else "none"
                                ),
                            )
                        if artifact is None or not artifact.data:
                            raise RuntimeError("Raw export stopped before returning data.")
                    except (RuntimeError, TypeError, ValueError) as exc:
                        ui.notify(str(exc), type="negative")
                        return
                    cache_artifact(
                        artifact,
                        source="Mapping",
                        kind="raw",
                        detail=output_key.replace(":", " ").replace("_", " ").title(),
                        navigate_to="mapping",
                    )
                    ui.download(artifact.data, artifact.filename, artifact.media_type)

                def render_latest_result() -> None:
                    nonlocal result_chart
                    if latest_result is None:
                        return
                    result_chart = None
                    result_slot.clear()
                    with result_slot:
                        try:
                            result_chart = _render_mapping_result(
                                latest_result,
                                output_key=str(
                                    preview_output.value or latest_result.primary_output
                                ),
                                plot_type=resolved_plot_type(),
                                normalization="None",
                                title=str(preview_title.value or ""),
                                series_label=str(series_label.value or ""),
                                x_label=str(preview_x_label.value or ""),
                                y_label=str(preview_y_label.value or ""),
                                color=str(line_or_bar_color.value or "#2c7f77"),
                                secondary_color=str(second_curve_color.value or "#b03412"),
                                palette=str(heatmap_palette.value or "viridis"),
                                histogram_bins=int(_read_number(histogram_bins, "Histogram bins")),
                                line_width=float(_read_number(line_width, "Curve thickness")),
                                show_markers=False,
                                show_colorbar=bool(colorbar.value),
                                log_y=bool(log_y.value),
                                log_z=bool(log_z.value),
                                invert_y=bool(invert_y.value),
                                density=bool(density.value),
                                show_legend=bool(show_legend.value),
                                legend_position=str(legend_position.value or "Top right"),
                                show_grid=bool(show_grid.value),
                                y_limits=current_y_limits(),
                                on_download=download_plot,
                            )
                        except (RuntimeError, TypeError, ValueError) as exc:
                            ui.notify(str(exc), type="negative")
                            preview_plot("Output unavailable", kind="mapping")

                async def map_data() -> None:
                    nonlocal active_cancel, latest_result, result_chart, running
                    if running:
                        if active_cancel is not None and not active_cancel.is_set():
                            active_cancel.set()
                            status.text = "Stopping mapping..."
                            map_button.disable()
                        return

                    cancel_event = Event()

                    def check_cancelled() -> None:
                        if cancel_event.is_set():
                            raise CancelledError("Mapping stopped.")

                    running = True
                    active_cancel = cancel_event
                    map_button.set_text("Stop")
                    map_button.set_icon("stop")
                    frozen_controls = freeze_scientific_controls()
                    progress.visible = True
                    progress.value = 0.01
                    status.text = "Reading inputs..."
                    stats_container.visible = False
                    latest_result = None
                    result_chart = None
                    result_slot.clear()
                    with result_slot:
                        ui.label("Mapping is running in the background...").classes("dn-muted")
                    try:
                        route = selected_route_state["route"]
                        inputs: dict[str, Any] = {}
                        forms: dict[str, DataForm] = {}
                        prewindowed_inputs: set[str] = set()
                        missing_policy = str(_region_missing.value or "error")
                        settings = setting_values()
                        selected_start = _read_number(region_start, "Region start")
                        selected_end = _read_number(region_end, "Region end")
                        resolution_value = settings.get("resolution")
                        if resolution_value is None:
                            resolution_value = route.setting_defaults_dict.get("resolution", 1.0)
                        selected_resolution = float(
                            1.0 if resolution_value is None else resolution_value
                        )
                        use_whole_region = bool(whole_region.value)

                        async def read_input(
                            key: str,
                            live: _MappingInputControl,
                        ) -> tuple[str, Any, DataForm | None, bool]:
                            try:
                                value, prewindowed = await _read_mapping_input_async(
                                    live,
                                    missing_policy=missing_policy,
                                    region_start_kb=selected_start,
                                    region_end_kb=selected_end,
                                    resolution_x=selected_resolution,
                                    whole_genome=use_whole_region,
                                )
                            except KeyError:
                                return key, None, None, False
                            return key, value, cast(DataForm, str(live.form.value)), prewindowed

                        preparation = asyncio.gather(
                            *(read_input(key, live) for key, live in mapping_input_controls.items())
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
                        prepared_inputs = await preparation
                        check_cancelled()
                        for key, value, form, prewindowed in prepared_inputs:
                            if form is None:
                                continue
                            inputs[key] = value
                            forms[key] = form
                            if prewindowed:
                                prewindowed_inputs.add(key)
                        request = MappingRequest(
                            route=route,
                            inputs=inputs,
                            input_forms=forms,
                            settings=settings,
                            target_shape=(
                                str(target_shape_control.value)
                                if target_shape_control.value is not None
                                else None
                            ),
                            region_start_kb=selected_start,
                            region_end_kb=selected_end,
                            resolution_x=selected_resolution,
                            resolution_t=(
                                float(settings["resolution_time"])
                                if settings.get("resolution_time") is not None
                                else None
                            ),
                            whole_genome=use_whole_region,
                            prewindowed_inputs=frozenset(prewindowed_inputs),
                        )

                        progress_state = _ProgressState()
                        worker = asyncio.create_task(
                            run.io_bound(
                                run_mapping,
                                request,
                                progress_state.report,
                                check_cancelled,
                            )
                        )
                        while not worker.done():
                            if cancel_event.is_set():
                                status.text = "Stopping mapping..."
                            else:
                                value, message = progress_state.snapshot()
                                progress.value = max(0.02, value)
                                status.text = message
                            await asyncio.sleep(0.04)
                        result = await worker
                        if result is None:
                            check_cancelled()
                            raise RuntimeError("The mapping worker stopped before returning data.")

                        requested_output = str(preview_output.value or "")
                        latest_result = result
                        output_options = _result_preview_outputs(result)
                        preview_output.options = output_options
                        preview_output.value = (
                            requested_output
                            if requested_output in output_options
                            else result.primary_output
                            if result.primary_output in output_options
                            else next(iter(output_options))
                        )
                        if len(output_options) == 1:
                            preview_output.disable()
                        else:
                            preview_output.enable()
                        preview_output.update()
                        apply_preview_output()

                        primary = np.asarray(result.outputs[result.primary_output])
                        mapping_stats["Form"].text = _DEPENDENCE_LABELS.get(
                            result.output_dependence,
                            result.output_dependence.replace("_", " ").title(),
                        )
                        mapping_stats["Points"].text = f"{primary.size:,}"
                        mapping_stats["Method"].text = route.method.replace("_", " ").title()
                        for value in mapping_stats.values():
                            value.update()
                        stats_container.visible = True
                        progress.value = 1.0
                        status.text = f"Complete in {result.elapsed_seconds:.2f} s"
                        populate_automatic_y_limits(redraw=False)
                        render_latest_result()
                        cache_workflow(
                            result,
                            name=f"Mapping: {result.target_label}",
                            source="Mapping",
                            detail=(
                                f"{mapping_stats['Form'].text} | {primary.size:,} points | "
                                f"{mapping_stats['Method'].text}"
                            ),
                            navigate_to="mapping",
                        )
                    except CancelledError:
                        progress.visible = False
                        status.text = "Mapping stopped"
                        stats_container.visible = False
                        latest_result = None
                        result_slot.clear()
                        with result_slot:
                            preview_plot(
                                "Mapping stopped",
                                kind="mapping",
                            )
                    except Exception as exc:
                        progress.visible = False
                        status.text = "Mapping could not be completed"
                        ui.notify(str(exc) or type(exc).__name__, type="negative")
                        stats_container.visible = False
                        latest_result = None
                        result_slot.clear()
                        with result_slot:
                            preview_plot(
                                "No result",
                                kind="mapping",
                            )
                    finally:
                        running = False
                        active_cancel = None
                        restore_scientific_controls(frozen_controls)
                        map_button.set_text("Map")
                        map_button.set_icon("map")
                        map_button.enable()

                def apply_preview_output() -> None:
                    """Keep preview presentation and raw export tied to the actual output."""

                    output_key = str(preview_output.value)
                    output_label = dict(preview_output.options).get(
                        output_key,
                        _OUTPUT_LABELS.get(
                            output_key,
                            output_key.replace("_", " ").title(),
                        ),
                    )
                    output_label = output_label.removesuffix(" (target)")
                    dependence = (
                        latest_result.output_dependence
                        if latest_result is not None
                        else str(dependence_control.value)
                    )
                    comparison = (
                        latest_result.comparisons.get(output_key.removeprefix(_COMPARISON_PREFIX))
                        if latest_result is not None and output_key.startswith(_COMPARISON_PREFIX)
                        else None
                    )
                    comparison_source_key = (
                        comparison.source_key
                        if comparison is not None
                        else output_key.removeprefix(_COMPARISON_PREFIX).removesuffix("_fit")
                        if output_key.startswith(_COMPARISON_PREFIX)
                        else None
                    )
                    comparison_form = (
                        comparison.form
                        if comparison is not None
                        else next(
                            (
                                item.forms[0]
                                for item in selected_route_state["route"].inputs
                                if _friendly_input_key(item.key) == comparison_source_key
                            ),
                            "x",
                        )
                        if comparison_source_key is not None
                        else None
                    )
                    actual_output_key = (
                        _resolved_output_key(latest_result, output_key)
                        if latest_result is not None
                        else output_key
                    )
                    output_form = (
                        comparison_form
                        if comparison_source_key is not None
                        else (
                            latest_result.output_forms.get(actual_output_key)
                            if latest_result is not None
                            else selected_route_state["route"].output_form
                        )
                    )
                    preview_title.value = output_label
                    view.value = "Default"
                    if comparison_source_key is not None:
                        view.disable()
                        preview_x_label.value = (
                            "Time (min)" if comparison_form == "t" else "Position (kb)"
                        )
                        preview_y_label.value = _OUTPUT_LABELS.get(
                            comparison_source_key,
                            comparison_source_key.replace("_", " ").title(),
                        )
                    else:
                        view.enable()
                    if actual_output_key == "fit_history":
                        preview_x_label.value = "Iteration"
                        preview_y_label.value = "Fit error"
                    elif comparison_source_key is None and output_form in {"t", "xt"}:
                        preview_x_label.value = "Chromosome position (kb)"
                        preview_y_label.value = "Time (min)"
                    elif comparison_source_key is None and (
                        output_form == "t" or dependence == "time"
                    ):
                        preview_x_label.value = "Time (min)"
                        preview_y_label.value = output_label
                    elif comparison_source_key is None and dependence == "ensemble":
                        preview_x_label.value = "Replicate"
                        preview_y_label.value = output_label
                    elif comparison_source_key is None and dependence == "regions":
                        preview_x_label.value = "Region"
                        preview_y_label.value = output_label
                    elif comparison_source_key is None:
                        preview_x_label.value = "Position (kb)"
                        preview_y_label.value = output_label
                    raw_controls.file_name.value = (
                        f"dnascape_mapping_{comparison_source_key}_comparison"
                        if comparison_source_key is not None
                        else f"dnascape_mapping_{actual_output_key}"
                    )
                    for control in (
                        preview_title,
                        preview_x_label,
                        preview_y_label,
                        view,
                        raw_controls.file_name,
                    ):
                        control.update()
                    sync_plot_controls()

                def sync_view_axis_labels() -> None:
                    sync_plot_controls()
                    output_key = str(preview_output.value or initial_output_key)
                    output_label = _OUTPUT_LABELS.get(
                        output_key,
                        output_key.replace("_", " ").title(),
                    )
                    plot_type = resolved_plot_type()
                    if plot_type == "Histogram":
                        preview_x_label.value = output_label
                        preview_y_label.value = "Density" if density.value else "Count"
                    elif plot_type == "Boxplot":
                        preview_x_label.value = ""
                        preview_y_label.value = output_label
                    elif plot_type == "Heatmap":
                        preview_x_label.value = "Chromosome position (kb)"
                        preview_y_label.value = "Time (min)"
                    else:
                        dependence = (
                            latest_result.output_dependence
                            if latest_result is not None
                            else str(dependence_control.value)
                        )
                        preview_x_label.value = (
                            "Time (min)" if dependence == "time" else "Position (kb)"
                        )
                        preview_y_label.value = output_label
                    preview_x_label.update()
                    preview_y_label.update()
                    refresh_axis_range()

                def refresh_preview_output() -> None:
                    apply_preview_output()
                    populate_automatic_y_limits(redraw=False)
                    render_latest_result()

                preview_output.on_value_change(refresh_preview_output)
                for plot_control in (
                    preview_title,
                    series_label,
                    preview_x_label,
                    preview_y_label,
                    line_or_bar_color,
                    second_curve_color,
                    line_width,
                    heatmap_palette,
                    log_z,
                    show_legend,
                    legend_position,
                    show_grid,
                    colorbar,
                    invert_y,
                ):
                    plot_control.on_value_change(render_latest_result)
                show_legend.on_value_change(sync_plot_controls)
                view.on_value_change(sync_view_axis_labels)
                density.on_value_change(sync_view_axis_labels)
                histogram_bins.on_value_change(refresh_axis_range)
                log_y.on_value_change(refresh_axis_range)
                y_minimum.on_value_change(use_custom_y_limits)
                y_maximum.on_value_change(use_custom_y_limits)
                sync_plot_controls()
                raw_controls.download_button.on_click(lambda: download_raw(raw_controls))
                map_button.on_click(map_data)
                region_start.on_value_change(invalidate_mapping_result)
                region_end.on_value_change(invalidate_mapping_result)
                _region_missing.on_value_change(invalidate_mapping_result)

                def sync_whole_region() -> None:
                    if bool(whole_region.value):
                        region_start.disable()
                        region_end.disable()
                    else:
                        region_start.enable()
                        region_end.enable()
                    invalidate_mapping_result()

                whole_region.on_value_change(sync_whole_region)

                if latest_result is not None:
                    restored_primary = np.asarray(
                        latest_result.outputs[latest_result.primary_output]
                    )
                    mapping_stats["Form"].text = _DEPENDENCE_LABELS.get(
                        latest_result.output_dependence,
                        latest_result.output_dependence.replace("_", " ").title(),
                    )
                    mapping_stats["Points"].text = f"{restored_primary.size:,}"
                    mapping_stats["Method"].text = latest_result.method.replace(
                        "_", " "
                    ).title()
                    for value in mapping_stats.values():
                        value.update()
                    stats_container.visible = True
                    status.text = "Restored from Cached Files"
                    apply_preview_output()
                    populate_automatic_y_limits(redraw=False)
                    render_latest_result()
