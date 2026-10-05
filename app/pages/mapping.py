"""Mapping workflow interface."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import CancelledError
from dataclasses import dataclass, field, replace
from threading import Event, Lock
from typing import Any, cast

import numpy as np
from nicegui import run, ui

from app.artifact_archives import (
    DEFAULT_BATCH_ARCHIVE_SOURCE_BYTES,
    build_artifact_archive,
)
from app.batch_workflows import (
    BatchPlan,
    BatchSourceSummary,
    CombinationSource,
    combination_names,
    plan_batch_sources,
    plan_combination_indices,
    unique_dataset_names,
)
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
from app.mapping_eta import MappingEta
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
    validate_mapping_request,
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
    raw_output_file_row,
    set_data_upload_locked,
    stat_strip,
)
from app.preview_plotting import (
    FIXED_QUANTITY_Y_LIMITS,
    HistogramBinSynchronizer,
    PreviewLine,
    automatic_y_limits,
    bind_preview_settings_updates,
    build_preview_plot_options,
    compact_color_input,
    finite_values,
    histogram_x_limits_from_controls,
    limit_curve_hover,
    plotting_settings_ready,
    preview_settings_controls,
    preview_settings_markers,
    render_preview_chart,
    sync_preview_settings_visibility,
)
from app.result_naming import (
    RMAP_FILE_PREFIX,
    compact_export_stem,
    output_file_token,
    safe_filename_stem,
    unique_compact_export_stems,
)
from app.session_cache import (
    cache_artifact,
    cache_upload,
    cache_workflow,
    consume_cached_workflow,
    current_session_cache,
)
from app.simulation_exports import ExportArtifact, build_chart_capture_export

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

_OUTPUT_LABELS = {
    "firing_rate": "Firing rate",
    "fork_speed": "Fork speed",
    "replication_timing": "Replication timing",
    "replication_timing_fit": "Replication timing fit",
    "fit_history": "MSE",
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
_MSE_DIAGNOSTIC_KEY = "mse_diagnostic"
_PLOT_TYPES = ["Line", "Heatmap", "Histogram", "Boxplot"]
_BATCH_COLOURS = (
    "#2c7f77",
    "#b03412",
    "#6f5a9c",
    "#c47a1b",
    "#3f6fa3",
    "#7d8f3a",
    "#9c4d70",
    "#4e7d8d",
)

_MAX_MAPPING_CASES = 50_000
_MAX_COMBINED_MAPPING_DATASETS = 50
_BATCH_RAW_PAGE_SIZE = 50
_BATCH_NAV_PAGE_SIZE = 50


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
    holder: Any | None = None
    active: bool = True


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


@dataclass
class _MappingConstantState:
    """Multiple scalar values retained across disposable Mapping widgets."""

    values_by_key: dict[str, float] = field(default_factory=dict)
    manual_keys: list[str] = field(default_factory=lambda: ["manual-1", "manual-2"])
    next_manual_index: int = 3


@dataclass(frozen=True, slots=True)
class _PreparedBatchInput:
    """One route input materialized as one or several independent datasets."""

    key: str
    label: str
    form: DataForm
    values: tuple[Any, ...]
    prewindowed: tuple[bool, ...]
    filenames: tuple[str, ...] = ()
    detected_forms: tuple[str | None, ...] = ()
    components: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _MappingBatchCase:
    """One fully resolved request in a pairwise or Cartesian Mapping run."""

    key: str
    name: str
    export_stem: str
    item_indices: tuple[int, ...]
    short_name: str = ""
    short_export_stem: str = ""


@dataclass(frozen=True, slots=True)
class _NamedMappingResult:
    """A completed mapping result paired with its stable dataset label."""

    key: str
    name: str
    export_stem: str
    result: MappingResult
    short_name: str = ""
    short_export_stem: str = ""


@dataclass
class _MappingRawPreference:
    """Editable raw-export values retained across refreshable batch rows."""

    file_name: str
    file_type: str = "TSV"
    compress: bool = False
    automatic_name: str = ""
    customized: bool = False


@dataclass(frozen=True, slots=True)
class _MappingRawExportSelection:
    """One immutable raw-export choice captured before asynchronous work."""

    output_key: str
    file_name: str
    file_type: str
    compression: str


# Building the scientific route catalogue is the only non-trivial part of
# Mapping-page setup. Resolve it while the local app starts so opening Mapping
# never pays that cold cost or leaves the navigation looking unresponsive.
_MAPPING_TARGETS = mapping_targets()


def _mapping_route_families_for_ui(
    target: str,
    dependence: str,
) -> tuple[MappingRouteFamily, ...]:
    """Expose every catalog family, with defaults folded into its factors."""

    return mapping_route_families(target, dependence)


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


def _calculation_routes(
    family: MappingRouteFamily,
    selections: dict[str, DataForm],
) -> dict[str, MappingRoute]:
    """Offer distinct calculations without duplicating defaulted-factor aliases."""

    calculations: dict[str, MappingRoute] = {}
    for route in family.compatible_routes(selections):
        # Replication timing has two forward solvers with overlapping input
        # forms. Other variants select a backend through their input forms.
        key = "automatic"
        if family.target_key == "replication_timing":
            key = "simulation" if "simulation" in route.method else "deterministic"
        calculations.setdefault(key, route)
    return calculations


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
            form := _retained_profile_form(
                upload_states.get(_friendly_input_key(key), _MappingUploadState()).inspections,
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


def _retained_profile_form(
    inspections: list[UploadInspection],
) -> DataForm | None:
    detected = {
        value
        for inspection in inspections
        if (value := _inspection_profile_form(inspection)) is not None
    }
    if len(detected) != 1:
        return None
    return cast(DataForm, next(iter(detected)))


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


def _inspection_profile_form(inspection: UploadInspection) -> str | None:
    """Return one file's independently detected profile form."""

    if inspection.source_format in {"bedgraph", "bigwig"}:
        return "x"
    detected = str(inspection.inferred_form)
    return detected if detected in {"x", "t", "xt"} else None


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
        "fork_speed",
        "kinetic_fields",
    }:
        outputs.setdefault("replication_timing_fit", "Replication timing fit")
        outputs[f"{_COMPARISON_PREFIX}replication_timing_fit"] = (
            "Replication timing: observed vs fitted"
        )
    outputs[_FIT_DIAGNOSTIC_KEY] = "Residual"
    outputs[_MSE_DIAGNOSTIC_KEY] = "MSE"
    return outputs


def _result_preview_outputs(result: MappingResult) -> dict[str, str]:
    """Return every output available in one completed Mapping result."""

    outputs = {
        key: (
            f"{_OUTPUT_LABELS.get(key, key.removesuffix('_xt').replace('_', ' ').title())} (target)"
            if key == result.target
            else _OUTPUT_LABELS.get(
                key,
                key.removesuffix("_xt").replace("_", " ").title(),
            )
        )
        for key in result.outputs
        if key not in {"fit_history", "residual"}
    }
    for fitted_key, comparison in result.comparisons.items():
        source_label = _OUTPUT_LABELS.get(
            comparison.source_key,
            comparison.source_key.replace("_", " ").title(),
        )
        outputs[f"{_COMPARISON_PREFIX}{fitted_key}"] = f"{source_label}: observed vs fitted"
    # Keep fit diagnostics together at the end of the selector: the residual
    # profile first and the optimizer's MSE trace after it.
    if "residual" in result.outputs:
        outputs["residual"] = "Residual"
    if "fit_history" in result.outputs:
        outputs["fit_history"] = "MSE"
    return outputs


def _common_result_preview_outputs(
    results: Sequence[MappingResult],
) -> dict[str, str]:
    """Return the ordered output intersection shared by a completed batch."""

    if not results:
        return {}
    output_sets = [_result_preview_outputs(result) for result in results]
    return {
        key: label
        for key, label in output_sets[0].items()
        if all(key in options for options in output_sets[1:])
    }


def _resolved_output_key(result: Any, output_key: str) -> str:
    """Resolve a virtual diagnostic key without failing UI synchronization."""

    if output_key == _MSE_DIAGNOSTIC_KEY:
        outputs = getattr(result, "outputs", {})
        return "fit_history" if "fit_history" in outputs else output_key
    if output_key != _FIT_DIAGNOSTIC_KEY:
        return output_key
    outputs = getattr(result, "outputs", {})
    return "residual" if "residual" in outputs else output_key


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
    log_x: bool = False,
    invert_x: bool = False,
    invert_z: bool = False,
    show_histogram_statistics: bool = False,
    histogram_statistics_position: str = "Top right",
    histogram_bin_size: float | None = None,
    histogram_x_limits: tuple[float, float] | None = None,
    histogram_touching: bool = True,
    histogram_boundary: str = "left",
) -> dict[str, Any]:
    """Build a bounded ECharts view for scalar, X, T, XT, and bundle outputs."""

    output_key = _resolved_output_key(result, output_key)

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
            log_x=log_x,
            invert_x=invert_x,
        )

    outputs = getattr(result, "outputs", {})
    if output_key not in outputs:
        raise ValueError("The selected mapping output was not returned by DNAscape.")
    values = np.asarray(outputs[output_key], dtype=float)
    if values.size == 0 and output_key == "inter_origin_distance":
        return {
            "title": {"text": title.strip() or "Inter-origin distance"},
            "xAxis": {"type": "value", "name": x_label},
            "yAxis": {"type": "value", "name": y_label},
            "series": [],
            "graphic": [
                {
                    "type": "text", "left": "center", "top": "middle",
                    "style": {"text": "No pairs of fired origins were observed.", "fill": "#777"},
                }
            ],
        }
    if values.size and not np.isfinite(values).any() and getattr(result, "diagnostics", {}).get(
        "unidentifiable_count"
    ) == values.size:
        return {
            "title": {"text": title.strip() or _OUTPUT_LABELS.get(output_key, output_key)},
            "xAxis": {"type": "value", "name": x_label},
            "yAxis": {"type": "value", "name": y_label},
            "series": [],
            "graphic": [{
                "type": "text", "left": "center", "top": "middle",
                "style": {"text": "No identifiable values in this interval.", "fill": "#777"},
            }],
        }
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
            log_x=log_x,
            invert_x=invert_x,
            invert_z=invert_z,
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
            log_x=log_x,
            invert_x=invert_x,
            show_histogram_statistics=show_histogram_statistics,
            histogram_statistics_position=histogram_statistics_position,
            histogram_bin_size=histogram_bin_size,
            histogram_x_limits=histogram_x_limits,
            histogram_touching=histogram_touching,
            histogram_boundary=histogram_boundary,
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
            invert_x=invert_x,
        )

    if values.ndim > 1 and 1 not in values.shape:
        raise ValueError("Choose Heatmap to view this two-dimensional mapping output.")
    dependence = str(getattr(result, "output_dependence", "space"))
    output_form = str(getattr(result, "output_forms", {}).get(output_key, ""))
    axis_kind = "time" if output_form == "t" or dependence == "time" else "space"
    if output_key == "fit_history":
        axis = np.arange(1, flattened.size + 1, dtype=float)
    elif dependence in {"ensemble", "regions", "constant"}:
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
        show_markers=show_markers or (output_key == "fit_history" and flattened.size == 1),
        legend_position=legend_position,
        show_grid=show_grid,
        interactive=output_form != "t" and dependence != "time",
        log_x=log_x,
        invert_x=invert_x,
    )


def _mapping_automatic_y_limits(
    result: Any,
    *,
    output_key: str,
    plot_type: str,
    histogram_bins: int,
    density: bool,
    log_y: bool,
    log_x: bool = False,
    histogram_bin_size: float | None = None,
    histogram_x_limits: tuple[float, float] | None = None,
    histogram_boundary: str = "left",
) -> tuple[float, float]:
    """Return editable, data-derived bounds for the active mapping plot."""

    output_key = _resolved_output_key(result, output_key)

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
    return automatic_y_limits(
        values,
        plot_type=plot_type,
        histogram_bins=histogram_bins,
        density=density,
        log_y=log_y,
        log_x=log_x,
        histogram_bin_size=histogram_bin_size,
        histogram_x_limits=histogram_x_limits,
        histogram_boundary=histogram_boundary,
        y_axis=y_axis,
        fixed_limits=fixed_limits,
        # A nonnegative physical quantity does not imply that zero belongs in
        # every preview.  Keep ordinary mapping profiles centred on their
        # observed extrema; genuinely bounded quantities above use their
        # explicit scientific domains, and histograms anchor their own count
        # baseline inside ``automatic_y_limits``.
        floor_at_zero=False,
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
    log_x: bool = False,
    invert_x: bool = False,
    invert_z: bool = False,
    show_histogram_statistics: bool = False,
    histogram_statistics_position: str = "Top right",
    histogram_bin_size: float | None = None,
    histogram_x_limits: tuple[float, float] | None = None,
    histogram_touching: bool = True,
    histogram_boundary: str = "left",
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
    return render_preview_chart(
        options,
        title=title,
        marker="mapping-result-plot",
        on_download=on_download,
    )


def _combined_mapping_plot_options(
    results: Sequence[_NamedMappingResult],
    *,
    colors: Mapping[tuple[str, str, int], str] | None = None,
    **plot_kwargs: Any,
) -> dict[str, Any]:
    """Merge compatible one-dimensional Mapping previews into one EChart."""

    if len(results) < 2:
        raise ValueError("A combined Mapping preview requires several results.")
    plot_type = str(plot_kwargs.get("plot_type") or "Line")
    histogram = plot_type == "Histogram"
    combined_series: list[dict[str, Any]] = []
    combined_names: list[str] = []
    x_minima: list[float] = []
    x_maxima: list[float] = []
    base: dict[str, Any] | None = None
    output_key = str(plot_kwargs.get("output_key") or "output")
    curve_index = 0
    for named in results:
        color = _BATCH_COLOURS[curve_index % len(_BATCH_COLOURS)]
        kwargs = dict(plot_kwargs)
        kwargs["color"] = color
        kwargs["secondary_color"] = _BATCH_COLOURS[
            (curve_index + 1) % len(_BATCH_COLOURS)
        ]
        options = _mapping_plot_options(named.result, **kwargs)
        series = list(options.get("series", ()))
        accepted_series = {"custom"} if histogram else {"line", "scatter"}
        if not series or any(item.get("type") not in accepted_series for item in series):
            raise ValueError(
                "This output cannot be combined. Choose Individual to view each dataset."
            )
        if base is None:
            base = options
        x_axis = options.get("xAxis", {})
        try:
            x_minima.append(float(x_axis["min"]))
            x_maxima.append(float(x_axis["max"]))
        except (KeyError, TypeError, ValueError):
            pass
        for series_index, item in enumerate(series):
            curve_color = (colors or {}).get(
                (named.key, output_key, series_index),
                _BATCH_COLOURS[curve_index % len(_BATCH_COLOURS)],
            )
            line_style = dict(item.get("lineStyle", {}))
            line_style["color"] = curve_color
            item["lineStyle"] = line_style
            item_style = dict(item.get("itemStyle", {}))
            item_style["color"] = curve_color
            if histogram:
                # Semi-transparent bars preserve every distribution when
                # their shared bins occupy the same coordinates.
                item_style["opacity"] = 0.48
            item["itemStyle"] = item_style
            original_name = str(item.get("name") or "Output")
            result_name = _mapping_display_name(named)
            display_name = (
                result_name
                if len(series) == 1
                else f"{result_name} · {original_name}"
            )
            item["name"] = display_name
            combined_names.append(display_name)
            combined_series.append(item)
            curve_index += 1
    assert base is not None
    base["series"] = combined_series
    if x_minima and x_maxima:
        base_x_axis = dict(base.get("xAxis", {}))
        base_x_axis["min"] = min(x_minima)
        base_x_axis["max"] = max(x_maxima)
        base["xAxis"] = base_x_axis
    if histogram:
        # The one-result histogram renderer intentionally has no legend. A
        # combined distribution needs one, and summary statistics should
        # describe the pooled values rather than whichever dataset happened
        # to become the base chart.
        show_legend = bool(plot_kwargs.get("show_legend", True))
        position = str(plot_kwargs.get("legend_position") or "Top right").casefold()
        bottom_legend = show_legend and position.startswith("bottom")
        base["legend"] = {
            "show": show_legend,
            "type": "scroll",
            "selectedMode": True,
            "orient": "horizontal" if bottom_legend else "vertical",
            **({"top": 24} if position.startswith("top") else {"bottom": 8}),
            **({"left": 84} if position.endswith("left") else {"right": 28}),
            "itemWidth": 16,
            "itemHeight": 8,
            "backgroundColor": "rgba(255,255,255,0.84)",
            "borderColor": "#ddd7d1",
            "borderWidth": 1,
            "borderRadius": 4,
            "padding": [4, 6],
            "textStyle": {"color": "#625c57", "fontSize": 11},
            "data": combined_names,
        }
        if bottom_legend:
            grid = dict(base.get("grid", {}))
            grid["bottom"] = max(106, int(grid.get("bottom", 0)))
            base["grid"] = grid
        if bool(plot_kwargs.get("show_histogram_statistics", False)):
            actual_keys = tuple(
                _resolved_output_key(named.result, output_key) for named in results
            )
            pooled = np.concatenate(
                tuple(
                    np.asarray(named.result.outputs[actual], dtype=float).reshape(-1)
                    for named, actual in zip(results, actual_keys, strict=True)
                )
            )
            first = results[0].result
            first_key = actual_keys[0]
            aggregate = replace(
                first,
                outputs={**first.outputs, first_key: pooled},
            )
            aggregate_kwargs = dict(plot_kwargs)
            aggregate_kwargs["color"] = _BATCH_COLOURS[0]
            aggregate_kwargs["secondary_color"] = _BATCH_COLOURS[1]
            aggregate_options = _mapping_plot_options(aggregate, **aggregate_kwargs)
            base["graphic"] = aggregate_options.get("graphic", [])
    else:
        legend = dict(base.get("legend", {}))
        legend["data"] = combined_names
        base["legend"] = legend
    return limit_curve_hover(base)


def _setting_value(setting: Any) -> Any:
    """Choose a useful initial control value from exact route metadata."""

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

    with ui.element("div").classes("dn-mapping-setting") as holder:
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
    return _SettingControl(setting=setting, control=control, override=override, holder=holder)


def _render_route_setting_groups(
    route: MappingRoute,
    *,
    location: SettingLocation,
    target_shape: str | None = None,
    input_forms: Mapping[str, DataForm] | None = None,
    on_change: Callable[[], Any] | None = None,
) -> dict[str, _SettingControl]:
    """Render categorized controls for one route and one UI location."""

    settings = tuple(
        setting
        for setting in route.active_settings(target_shape, input_forms)
        if setting.location == location
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
                .classes(
                    "dn-setting-group dn-mapping-settings-theme"
                    + (" dn-mapping-settings-theme-untitled" if category == "domain" else "")
                )
                .mark(f"mapping-settings-{category}")
            ):
                if category != "domain":
                    ui.label(_SETTING_CATEGORY_LABELS.get(category, category.title())).classes(
                        "dn-setting-group-title"
                    )
                for setting in category_settings:
                    controls[setting.key] = _render_setting(setting, on_change)
    if route.method in {
        "timing_and_speed_to_firing_rate", "timing_to_firing_constant_fork_speed",
    } and "fit_step" in controls:
        automatic_step = 2.0
        custom_step = False

        def remember_custom_step() -> None:
            nonlocal custom_step
            if controls["fit_step"].control.value != automatic_step:
                custom_step = True

        def sync_spatial_firing_step() -> None:
            nonlocal automatic_step
            if custom_step:
                return
            # The page exposes physical density rather than per-bin probability.
            key = "stall_rate_per_kb" if "stall_rate_per_kb" in controls else "stall_rate"
            stall = controls.get(key)
            enabled = stall is not None and (stall.override is None or stall.override.value)
            automatic_step = .6 if enabled and (stall.control.value or 0.) > 0. else 2.
            controls["fit_step"].control.set_value(automatic_step)

        controls["fit_step"].control.on_value_change(remember_custom_step)
        for key in ("stall_rate", "stall_rate_per_kb"):
            if key in controls:
                controls[key].control.on_value_change(sync_spatial_firing_step)
                if controls[key].override is not None:
                    controls[key].override.on_value_change(sync_spatial_firing_step)
        sync_spatial_firing_step()
    if route.target == "kinetic_fields" and {"firing_form", "speed_form"} <= controls.keys():
        def sync_kinetic_controls() -> None:
            forms = {controls[key].control.value for key in ("firing_form", "speed_form")}
            applicable = {
                "spatial_knots": bool(forms & {"space", "spacetime"}),
                "temporal_knots": bool(forms & {"time", "spacetime"}),
                "regularization": forms != {"constant"},
            }
            for key, active in applicable.items():
                if key in controls:
                    controls[key].active = active
                    holder = controls[key].holder
                    if holder is not None:
                        holder.set_visibility(active)

        for key in ("firing_form", "speed_form"):
            controls[key].control.on_value_change(sync_kinetic_controls)
        sync_kinetic_controls()
    if route.target == "firing_rate" and "spacetime_update" in controls:
        def sync_firing_update_controls() -> None:
            mixed = controls["spacetime_update"].control.value == "mixed"
            for key, active in {
                "spacetime_fit_step": mixed, "spacetime_sim_number": mixed,
                "spacetime_kernel_t": mixed, "spacetime_order": not mixed,
                "temporal_sim_number": not mixed,
            }.items():
                if key in controls:
                    controls[key].active = active
                    holder = controls[key].holder
                    if holder is not None:
                        holder.set_visibility(active)

        controls["spacetime_update"].control.on_value_change(sync_firing_update_controls)
        sync_firing_update_controls()
    if route.target == "fork_speed" and "spacetime_order" in controls:
        def sync_speed_order_controls() -> None:
            sequential = controls["spacetime_order"].control.value != "crossed"
            for key in ("neighbourhood_range", "refine_verbose"):
                if key in controls:
                    controls[key].active = sequential
                    holder = controls[key].holder
                    if holder is not None:
                        holder.set_visibility(sequential)

        controls["spacetime_order"].control.on_value_change(sync_speed_order_controls)
        sync_speed_order_controls()
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


def _read_multiple_constants(control: Any, label: str) -> tuple[float, ...]:
    """Parse a compact comma/newline list without validating while users type."""

    parts = [part for part in str(control.value or "").replace(",", " ").split() if part]
    if not parts:
        raise ValueError(f"Enter at least one value for {label.lower()}.")
    values: list[float] = []
    for part in parts:
        try:
            value = float(part)
        except ValueError as exc:
            raise ValueError(f"{label} values must be numbers.") from exc
        if not math.isfinite(value):
            raise ValueError(f"{label} values must be finite.")
        if value < 0:
            raise ValueError(f"{label} values cannot be negative.")
        values.append(value)
    return tuple(values)


def _read_multiple_constant_controls(
    controls: Sequence[Any],
    label: str,
) -> tuple[float, ...]:
    """Read the aligned scalar fields only when the user starts Mapping."""

    if not controls:
        raise ValueError(f"Add at least one value for {label.lower()}.")
    values: list[float] = []
    for index, control in enumerate(controls, start=1):
        value = _read_number(control, f"{label} value {index}")
        if value < 0:
            raise ValueError(f"{label} values cannot be negative.")
        values.append(value)
    return tuple(values)


def _format_mapping_component(value: float, *, units: str = "") -> str:
    """Format a scalar compactly for target-first result labels."""

    numeric = f"{float(value):.10g}"
    return f"{numeric} {units}" if units else numeric


def _mapping_export_stem(display_name: str, index: int) -> str:
    """Create a bounded target-first stem without treating decimals as suffixes."""

    return compact_export_stem(
        display_name,
        fallback=f"mapping-result-{index}",
        maximum=150,
    )


def _mapping_display_name(named: _NamedMappingResult) -> str:
    """Resolve a completed result through the application naming preference."""

    if get_export_settings().result_name_style == "full":
        return named.name
    return named.short_name or named.name


def _mapping_result_stem(named: _NamedMappingResult) -> str:
    """Resolve the matching collision-safe file stem."""

    if get_export_settings().result_name_style == "full":
        return named.export_stem
    return named.short_export_stem or named.export_stem


def _mapping_output_file_token(result: MappingResult, output_key: str) -> str:
    """Resolve comparison aliases and diagnostics to one concise file token."""

    if output_key.startswith(_COMPARISON_PREFIX):
        source_key = output_key.removeprefix(_COMPARISON_PREFIX).removesuffix("_fit")
        return f"{output_file_token(source_key)}-cmp"
    return output_file_token(_resolved_output_key(result, output_key))


def _mapping_raw_default_filename(
    named: _NamedMappingResult,
    output_key: str,
) -> str:
    """Return a target-first filename for one mapped result and selected output."""

    identity_stem = _mapping_result_stem(named)[:120].rstrip("-") or "mapping-result"
    actual_output = _resolved_output_key(named.result, output_key)
    comparison = output_key.startswith(_COMPARISON_PREFIX)
    output_stem = _mapping_output_file_token(named.result, output_key)
    base = f"{RMAP_FILE_PREFIX}_{identity_stem}"
    if not comparison and actual_output == named.result.primary_output:
        return base
    return f"{base}_{_mapping_export_stem(output_stem, 1)}"


def _mapping_raw_preference(
    preferences: dict[tuple[str, str], _MappingRawPreference],
    named: _NamedMappingResult,
    output_key: str,
) -> _MappingRawPreference:
    """Return one stable preference, updating only untouched automatic names."""

    key = (named.key, output_key)
    automatic_name = _mapping_raw_default_filename(named, output_key)
    preference = preferences.setdefault(
        key,
        _MappingRawPreference(
            file_name=automatic_name,
            automatic_name=automatic_name,
        ),
    )
    if not preference.automatic_name:
        preference.automatic_name = preference.file_name
    if not preference.customized:
        preference.file_name = automatic_name
        preference.automatic_name = automatic_name
    return preference


def _mapping_raw_page_bounds(
    total: int,
    requested_page: int,
) -> tuple[int, int, int, int]:
    """Return ``page, page_count, start, end`` for bounded raw-row rendering."""

    normalized_total = max(0, int(total))
    page_count = max(1, math.ceil(normalized_total / _BATCH_RAW_PAGE_SIZE))
    page = min(max(1, int(requested_page)), page_count)
    start = (page - 1) * _BATCH_RAW_PAGE_SIZE
    end = min(normalized_total, start + _BATCH_RAW_PAGE_SIZE)
    return page, page_count, start, end


def _read_setting(live: _SettingControl) -> Any:
    setting = live.setting
    if not live.active:
        raise KeyError(setting.key)
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


def _render_prepared_input(
    input_spec: MappingInput,
    marker: str,
    surface: dict[str, Any],
    prepared_results: Callable[[], Sequence[_NamedMappingResult]] | None = None,
    on_change: Callable[[], Any] | None = None,
) -> None:
    """Select a compatible numeric result already computed in this client."""

    required_members = (
        "replicated_fraction_xt", "right_fork_density_xt", "left_fork_density_xt"
    )
    compatible: dict[str, _NamedMappingResult] = {}
    for named in prepared_results() if prepared_results is not None else ():
        outputs = named.result.outputs
        if input_spec.key not in {"fork_densities", "fork_densities_xt"} or any(
            member not in outputs for member in required_members
        ):
            continue
        shapes = {np.asarray(outputs[member]).shape for member in required_members}
        if len(shapes) != 1:
            continue
        shape = next(iter(shapes))
        if len(shape) != 2 or shape != (
            named.result.x_axis.size, named.result.time_axis.size
        ):
            continue
        compatible[named.key] = named
    surface["prepared_results"] = compatible
    with (
        ui.element("div")
        .classes("dn-mapping-prepared-result")
        .mark(f"mapping-input-{marker}-prepared")
    ):
        ui.icon("inventory_2").classes("dn-mapping-prepared-icon")
        with ui.column().classes("dn-mapping-prepared-copy"):
            ui.label(
                "Use a computed fork-density result"
                if compatible else "No compatible prepared result in this session"
            ).classes(
                "dn-mapping-prepared-title"
            )
        prepared = (
            ui.select(
                {key: named.name for key, named in compatible.items()},
                value=next(iter(compatible), None),
                label="Result",
            )
            .props("dense options-dense")
            .classes("dn-field dn-mapping-prepared-select")
            .mark(f"mapping-input-{marker}-prepared-select")
        )
        surface["prepared_selection"] = prepared
        if not compatible:
            prepared.disable()
        if on_change is not None:
            prepared.on_value_change(on_change)


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
    on_structure_change: Callable[[], Any],
) -> None:
    # A per-client prepared-result workspace is not available yet.  Ordinary
    # factors therefore expose only the working upload path instead of a dead
    # selectable option.
    source = None
    with ui.element("div").classes("dn-mapping-upload-source"):
        inspections = retained.inspections
        selector_controls: dict[str, tuple[Any | None, Any | None]] = {}
        selection_values = retained.selection_values
        materialized_cache = retained.materialized_cache
        track_options: Any | None = None

        def uses_matrix_columns(inspection: UploadInspection) -> bool:
            return _inspection_profile_form(inspection) == "xt"

        def supports_all_profile_columns(inspection: UploadInspection) -> bool:
            """Whether one table can be expanded into independent profiles."""

            return (
                len(inspection.numeric_columns) > 1
                and inspection.source_format not in {"bedgraph", "bigwig"}
            )

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
                    if supports_all_profile_columns(inspection):
                        valid_columns.add("__all__")
                    if uses_matrix_columns(inspection):
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
                summary = f"Datasets: {len(inspections)}"
                ui.label(summary).classes(
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

        @ui.refreshable
        def render_selection() -> None:
            """Render file-specific array choices directly under the uploaded profile."""

            selector_controls.clear()
            if not inspections:
                return
            with ui.element("div").classes("dn-mapping-upload-selectors"):
                for inspection in inspections:
                    selected = selection_values.setdefault(inspection.name, {})
                    column_control = None
                    # A selector adds no information when there is only one
                    # usable signal column: rdata already chooses it safely.
                    if len(inspection.numeric_columns) > 1:
                        column_options: Any = list(inspection.numeric_columns)
                        if supports_all_profile_columns(inspection):
                            column_options = {
                                "__all__": "All",
                                **{column: column for column in inspection.numeric_columns},
                            }
                        column_control = (
                            ui.select(
                                column_options,
                                value=selected.get("column"),
                                label="Value column",
                            )
                            .props("dense options-dense")
                            .classes("dn-field dn-mapping-column-select")
                            .mark(f"mapping-input-{marker}-value-column")
                        )
                        column_control._props["data-upload-name"] = inspection.name
                        def store_column(
                            control: Any = column_control,
                            values: dict[str, str | None] = selected,
                        ) -> None:
                            values["column"] = None if control.value is None else str(control.value)
                            materialized_cache.update(key=None, value=None)
                            on_change()

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
                            .mark(f"mapping-input-{marker}-array-key")
                        )
                        array_control._props["data-upload-name"] = inspection.name
                        if len(inspection.array_keys) == 1:
                            array_control.disable()

                        def store_array(
                            control: Any = array_control,
                            values: dict[str, str | None] = selected,
                        ) -> None:
                            values["array_key"] = (
                                None if control.value is None else str(control.value)
                            )
                            materialized_cache.update(key=None, value=None)
                            on_change()

                        array_control.on_value_change(store_array)
                    selector_controls[inspection.name] = (
                        column_control,
                        array_control,
                    )

        async def uploaded(event: Any) -> bool:
            try:
                inspection = await inspect_upload_event(event, expected_form="auto")
            except (OSError, TypeError, ValueError) as exc:
                on_error(str(exc))
                return False
            candidates = list(inspections)
            replaced_index = next(
                (
                    index
                    for index, item in enumerate(candidates)
                    if item.name == inspection.name
                ),
                None,
            )
            if replaced_index is None:
                candidates.append(inspection)
            else:
                # Preserve the pairing position when retained bytes for one
                # named dataset are replaced.
                candidates[replaced_index] = inspection
            materialized_cache.update(key=None, value=None)
            inspections[:] = candidates
            sync_selection_values()
            surface.update(
                inspections=inspections,
                selector_controls=selector_controls,
                selection_values=selection_values,
                source=source,
            )
            render_inspection.refresh()
            render_selection.refresh()
            sync_track_options_visibility()
            individual_form = _inspection_profile_form(inspection)
            if len(candidates) == 1 and individual_form in profile_forms:
                on_detected_form(cast(DataForm, individual_form))
            cache_upload(inspection, source="Mapping", navigate_to="mapping")
            on_change()
            on_structure_change()
            return True

        def cleared(event: Any) -> None:
            materialized_cache.update(key=None, value=None)
            name = str(getattr(event, "name", ""))
            if name:
                inspections[:] = [item for item in inspections if item.name != name]
            elif removed_names := {
                str(value)
                for value in getattr(event, "names", ())
                if str(value)
            }:
                # Delete all snapshots the completed rows it owns. Keep an
                # upload whose inspection finishes concurrently instead of
                # clearing scientific state that remains visible afterwards.
                inspections[:] = [
                    item for item in inspections if item.name not in removed_names
                ]
            else:
                inspections.clear()
            sync_selection_values()
            render_inspection.refresh()
            render_selection.refresh()
            sync_track_options_visibility()
            on_change()
            on_structure_change()

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
        if profile_forms == ("xt",):
            # Genomic tracks are one-dimensional. A batch XT input consists
            # of one complete matrix per file, never one track per time point.
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
            multiple=True,
            max_files=None,
            hide_when_loaded=False,
            initial_names=tuple(item.name for item in inspections),
        )
        render_inspection()
        render_selection()
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


def _render_multiple_constant_input(
    input_spec: MappingInput,
    marker: str,
    surface: dict[str, Any],
    retained: _MappingConstantState,
    peer_rows: Callable[[], tuple[tuple[str, str], ...]],
    combination_mode: dict[str, str],
    on_change: Callable[[], Any],
    on_structure_change: Callable[[], Any],
) -> None:
    """Render a compact value row beside every uploaded counterpart dataset."""

    units = _SCALAR_UNITS.get(_friendly_input_key(input_spec.key), "")
    aligned_rows = peer_rows()
    manual = not aligned_rows
    if manual:
        aligned_rows = tuple(
            (key, f"Value {index}")
            for index, key in enumerate(retained.manual_keys, start=1)
        )

    # Populate newly encountered peer identities from the matching manual row
    # before falling back to the package's scalar default. Values already tied
    # to a filename survive removals and same-name replacements.
    for index, (row_key, _label) in enumerate(aligned_rows):
        if row_key in retained.values_by_key:
            continue
        manual_key = (
            retained.manual_keys[index] if index < len(retained.manual_keys) else ""
        )
        retained.values_by_key[row_key] = retained.values_by_key.get(
            manual_key,
            _scalar_default(input_spec),
        )

    with (
        ui.element("div")
        .classes("dn-mapping-constant-batch")
        .mark(f"mapping-input-{marker}-constants")
    ):
        matching = (
            ui.select(
                {
                    "pairwise": "Pairwise",
                    "all": "All combinations",
                },
                value=combination_mode.get("value", "pairwise"),
                label="Matching",
            )
            .props("dense options-dense")
            .classes("dn-field dn-mapping-constant-matching")
            .mark(f"mapping-input-{marker}-combination-mode")
        )

        def matching_changed() -> None:
            value = str(matching.value or "pairwise")
            combination_mode["value"] = value if value in {"pairwise", "all"} else "pairwise"
            on_change()
            on_structure_change()

        matching.on_value_change(matching_changed)
        constant_controls: list[Any] = []
        with (
            ui.element("div")
            .classes("dn-mapping-constant-list")
            .mark(f"mapping-input-{marker}-constant-list")
        ):
            for index, (row_key, row_label) in enumerate(aligned_rows, start=1):
                with ui.element("div").classes("dn-mapping-constant-row"):
                    label = ui.label(row_label).classes("dn-mapping-constant-label")
                    label.tooltip(row_label)
                    control = (
                        ui.number(
                            value=retained.values_by_key[row_key],
                            min=0,
                            step=0.001 if input_spec.key == "firing_rate" else 0.1,
                        )
                        .props("dense")
                        .classes("dn-field dn-mapping-constant-value")
                        .mark(f"mapping-input-{marker}-constant-{index}")
                    )
                    control._props["aria-label"] = f"{input_spec.label} for {row_label}"
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
                                .mark(f"mapping-input-{marker}-constant-remove-{index}")
                            )
                            remove.set_enabled(len(retained.manual_keys) > 1)

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
                    .mark(f"mapping-input-{marker}-constant-add")
                )

                def add_value() -> None:
                    identity = f"manual-{retained.next_manual_index}"
                    retained.next_manual_index += 1
                    retained.manual_keys.append(identity)
                    retained.values_by_key[identity] = _scalar_default(input_spec)
                    on_change()
                    on_structure_change()

                add.on_click(add_value)
        surface.update(
            constants=constant_controls,
            constant_controls=constant_controls,
            constant_row_keys=tuple(key for key, _label in aligned_rows),
            combination_mode=matching,
        )


def _render_mapping_input(
    input_spec: MappingInput,
    *,
    forms: tuple[DataForm, ...] | None = None,
    selected_form: DataForm | None = None,
    retained: _MappingUploadState | None = None,
    retained_constants: _MappingConstantState | None = None,
    peer_constant_rows: Callable[[], tuple[tuple[str, str], ...]] | None = None,
    combination_mode: dict[str, str] | None = None,
    prepared_results: Callable[[], Sequence[_NamedMappingResult]] | None = None,
    on_change: Callable[[], Any] | None = None,
    on_structure_change: Callable[[], Any] | None = None,
    on_error: Callable[[str], Any] | None = None,
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
    constant_state = retained_constants or _MappingConstantState()
    profile_forms: tuple[DataForm, ...] = tuple(
        cast(DataForm, form) for form in available_forms if form in {"x", "t", "xt"}
    )
    retained_form = _retained_profile_form(upload_state.inspections)
    if retained_form in profile_forms:
        default_form = retained_form
    else:
        default_form = selected_form if selected_form in available_forms else available_forms[0]
    defaulted_override = input_spec.role == "feature_assumption" and input_spec.has_default
    surface: dict[str, Any] = {}
    scientific_change = on_change or (lambda: None)
    scientific_structure_change = on_structure_change or (lambda: None)
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
                    if _friendly_input_key(input_spec.key) in {
                        "firing_rate",
                        "fork_speed",
                    }:
                        mode_options = {
                            "scalar": "Constant",
                            "constants": "Multiple constants",
                            "profile": "Profile",
                        }
                default_mode = "scalar" if default_form == "scalar" else "profile"
            else:
                mode_options = {
                    str(form): _FORM_LABELS.get(str(form), str(form).replace("_", " ").title())
                    for form in available_forms
                }
                if available_forms == ("scalar",) and _friendly_input_key(input_spec.key) in {
                    "firing_rate",
                    "fork_speed",
                }:
                    mode_options = {
                        "scalar": "Constant",
                        "constants": "Multiple constants",
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
                        _render_prepared_input(
                            input_spec, marker, surface, prepared_results, scientific_change
                        )
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
                    elif mode == "constants":
                        _render_multiple_constant_input(
                            input_spec,
                            marker,
                            surface,
                            constant_state,
                            peer_constant_rows or (lambda: ()),
                            combination_mode or {"value": "pairwise"},
                            scientific_change,
                            scientific_structure_change,
                        )
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
                            scientific_structure_change,
                        )

            render_input_surface()

            def mode_changed() -> None:
                mode = str(mode_control.value or default_mode)
                if mode == "profile":
                    detected = _retained_profile_form(upload_state.inspections)
                    desired = (
                        detected if detected in profile_forms else cast(DataForm, profile_forms[0])
                    )
                elif mode == "constants":
                    desired = cast(DataForm, "scalar")
                else:
                    desired = cast(DataForm, mode)
                form_control.set_value(desired)
                render_input_surface.refresh()
                scientific_change()
                scientific_structure_change()

            mode_control.on_value_change(mode_changed)

            def refresh_input_surface() -> None:
                form = str(form_control.value or default_form)
                if form == "scalar" and mode_control.value == "constants":
                    desired_mode = "constants"
                else:
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
        constants = live.surface.get("constants")
        if constants is not None:
            if isinstance(constants, (tuple, list)):
                return _read_multiple_constant_controls(constants, live.spec.label)
            return _read_multiple_constants(constants, live.spec.label)
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
        selected = live.surface.get("prepared_selection")
        candidates = live.surface.get("prepared_results", {})
        named = candidates.get(getattr(selected, "value", None))
        if named is None:
            raise ValueError(
                f"{live.spec.label} requires a prepared DNAscape result. No compatible result "
                "is available in this session; first map the directional fork densities."
            )
        result = named.result
        members = (
            "replicated_fraction_xt", "right_fork_density_xt", "left_fork_density_xt",
            "fork_density_xt",
        )
        return {
            **{key: result.outputs[key] for key in members if key in result.outputs},
            "x_array": result.x_axis,
            "time_grid": result.time_axis,
        }
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


async def _read_mapping_input_batch_async(
    key: str,
    live: _MappingInputControl,
    *,
    missing_policy: str = "error",
    region_start_kb: float | None = None,
    region_end_kb: float | None = None,
    resolution_x: float | None = None,
    whole_genome: bool = False,
) -> _PreparedBatchInput:
    """Materialize one live route input as broadcast or per-dataset values."""

    if live.override is not None and not bool(live.override.value):
        raise KeyError(live.spec.key)
    form = cast(DataForm, str(live.form.value))
    inspections = tuple(cast(list[UploadInspection], live.surface.get("inspections", [])))
    source = live.surface.get("source")
    uploaded = form not in {"scalar", "regions", "bundle"} or (
        form == "regions" and source is not None and str(source.value) == "upload"
    )
    if str(getattr(live.surface.get("mode"), "value", "")) == "constants":
        constant_controls = cast(
            Sequence[Any],
            live.surface.get("constant_controls", live.surface.get("constants", ())),
        )
        if not constant_controls:
            raise ValueError(f"{live.spec.label} has no multiple-constant control.")
        values = _read_multiple_constant_controls(constant_controls, live.spec.label)
        units = _SCALAR_UNITS.get(_friendly_input_key(live.spec.key), "")
        return _PreparedBatchInput(
            key=key,
            label=live.spec.label,
            form=form,
            values=values,
            prewindowed=(False,) * len(values),
            components=tuple(
                _format_mapping_component(value, units=units) for value in values
            ),
        )
    selection_values = cast(
        dict[str, dict[str, str | None]],
        live.surface.get("selection_values", {}),
    )
    expand_table_columns = bool(
        uploaded
        and form in {"x", "t"}
        and any(
            selection_values.get(inspection.name, {}).get("column") == "__all__"
            for inspection in inspections
        )
    )
    if expand_table_columns:
        materialized_cache = cast(
            dict[str, Any],
            live.surface.setdefault("materialized_cache", {"key": None, "value": None}),
        )
        selections = tuple(
            (
                inspection.name,
                selection_values.get(inspection.name, {}).get("column"),
                selection_values.get(inspection.name, {}).get("array_key"),
            )
            for inspection in inspections
        )
        cache_key = (
            "numeric-profile-columns",
            tuple((id(item), item.name) for item in inspections),
            form,
            selections,
            missing_policy,
        )
        cached_values = (
            materialized_cache.get("value")
            if materialized_cache.get("key") == cache_key
            else None
        )
        values: list[Any] = [] if cached_values is None else list(cached_values)
        filenames: list[str] = []
        detected_forms: list[str | None] = []
        components: list[str] = []
        value_index = 0
        for inspection in inspections:
            selected = selection_values.get(inspection.name, {})
            selected_column = selected.get("column")
            columns: tuple[str | None, ...] = (
                tuple(inspection.numeric_columns)
                if selected_column == "__all__"
                else (
                    None if selected_column is None else str(selected_column),
                )
            )
            dataset_label = unique_dataset_names((inspection.name,))[0]
            for column in columns:
                if cached_values is None:
                    values.append(
                        await run.io_bound(
                            materialize_upload,
                            replace(inspection, expected_form=form),
                            column=column,
                            array_key=selected.get("array_key"),
                            missing_policy=missing_policy,
                        )
                    )
                column_label = str(column or inspection.default_column or "values")
                expanded = selected_column == "__all__"
                component = (
                    f"{dataset_label} · {column_label}" if expanded else dataset_label
                )
                synthetic_stem = safe_filename_stem(
                    f"{dataset_label}-{column_label}" if expanded else dataset_label,
                    fallback=f"dataset-{value_index + 1}",
                )
                filenames.append(f"{synthetic_stem}.profile")
                components.append(component)
                detected_forms.append(_batch_detected_form(inspection, form))
                value_index += 1
        if cached_values is None:
            materialized_cache.update(key=cache_key, value=tuple(values))
        return _PreparedBatchInput(
            key=key,
            label=live.spec.label,
            form=form,
            values=tuple(values),
            prewindowed=(False,) * len(values),
            filenames=tuple(filenames),
            detected_forms=tuple(detected_forms),
            components=tuple(components),
        )
    if not uploaded or len(inspections) <= 1:
        value, prewindowed = await _read_mapping_input_async(
            live,
            missing_policy=missing_policy,
            region_start_kb=region_start_kb,
            region_end_kb=region_end_kb,
            resolution_x=resolution_x,
            whole_genome=whole_genome,
        )
        filenames = tuple(item.name for item in inspections) if uploaded else ()
        detected = (
            tuple(_batch_detected_form(item, form) for item in inspections) if uploaded else ()
        )
        return _PreparedBatchInput(
            key=key,
            label=live.spec.label,
            form=form,
            values=(value,),
            prewindowed=(prewindowed,),
            filenames=filenames,
            detected_forms=detected,
            components=(
                unique_dataset_names(filenames)
                if filenames
                else (
                    _format_mapping_component(
                        float(value),
                        units=_SCALAR_UNITS.get(_friendly_input_key(live.spec.key), ""),
                    ),
                )
                if form == "scalar"
                else ("Manual",)
            ),
        )

    values: list[Any] = []
    prewindowed_values: list[bool] = []
    for inspection in inspections:
        # Reuse the proven single-upload materializer while retaining the
        # original file-specific column/array selection and shared track setup.
        surface = dict(live.surface)
        surface["inspections"] = [inspection]
        surface["materialized_cache"] = {"key": None, "value": None}
        single_live = _MappingInputControl(
            spec=live.spec,
            form=live.form,
            refresh_surface=live.refresh_surface,
            override=live.override,
            surface=surface,
        )
        value, prewindowed = await _read_mapping_input_async(
            single_live,
            missing_policy=missing_policy,
            region_start_kb=region_start_kb,
            region_end_kb=region_end_kb,
            resolution_x=resolution_x,
            whole_genome=whole_genome,
        )
        values.append(value)
        prewindowed_values.append(prewindowed)
    return _PreparedBatchInput(
        key=key,
        label=live.spec.label,
        form=form,
        values=tuple(values),
        prewindowed=tuple(prewindowed_values),
        filenames=tuple(item.name for item in inspections),
        detected_forms=tuple(_batch_detected_form(item, form) for item in inspections),
        components=unique_dataset_names(tuple(item.name for item in inspections)),
    )


def _batch_detected_form(inspection: UploadInspection, selected_form: DataForm) -> str | None:
    """Resolve the X/T vector ambiguity while retaining true XT mismatches."""

    detected = _inspection_profile_form(inspection)
    if selected_form in {"x", "t"} and detected in {"x", "t"}:
        return str(selected_form)
    return detected


def _plan_mapping_batch(inputs: Sequence[_PreparedBatchInput]) -> BatchPlan:
    """Validate submit-time Mapping cardinality and file-form consistency."""

    for item in inputs:
        if item.form != "xt":
            continue
        incompatible = tuple(
            filename
            for filename, detected in zip(
                item.filenames,
                item.detected_forms,
                strict=True,
            )
            if detected not in {None, "xt"}
        )
        if incompatible:
            names = ", ".join(incompatible)
            raise ValueError(
                f"{item.label} expects one complete space-time matrix per dataset; "
                f"these files are not space-time matrices: {names}."
            )

    file_plan = plan_batch_sources(
        tuple(
            BatchSourceSummary(
                label=item.label,
                filenames=item.filenames,
                forms=item.detected_forms,
                shapes=tuple(
                    tuple(int(size) for size in np.asarray(value).shape) for value in item.values
                ),
            )
            for item in inputs
            if item.filenames
        ),
        fallback_name="mapping",
        require_matching_shapes=False,
    )
    size = max((len(item.values) for item in inputs), default=1)
    size = max(size, file_plan.size)
    for item in inputs:
        count = len(item.values)
        if count not in {1, size}:
            raise ValueError(
                f"{item.label} has {count} datasets, but this batch needs either "
                f"one shared value/file or {size} paired values/files."
            )

    if size == file_plan.size:
        names = file_plan.names
    elif size == 1:
        names = file_plan.names
    else:
        names = tuple(f"Dataset {index}" for index in range(1, size + 1))
    return BatchPlan(size=size, names=names)


def _prepared_components(item: _PreparedBatchInput) -> tuple[str, ...]:
    """Return one deterministic naming component for every prepared value."""

    if len(item.components) == len(item.values):
        return item.components
    if len(item.filenames) == len(item.values):
        return unique_dataset_names(item.filenames)
    if item.form == "scalar":
        units = _SCALAR_UNITS.get(_friendly_input_key(item.key), "")
        return tuple(
            _format_mapping_component(float(value), units=units) for value in item.values
        )
    return tuple(f"Dataset {index}" for index in range(1, len(item.values) + 1))


def _validate_mapping_source_metadata(inputs: Sequence[_PreparedBatchInput]) -> None:
    """Validate every source independently before planning cross-source cases."""

    for item in inputs:
        if not item.values:
            raise ValueError(f"{item.label} must contain at least one value or file.")
        if len(item.prewindowed) != len(item.values):
            raise ValueError(f"{item.label} preprocessing metadata are incomplete.")
        if item.filenames and len(item.filenames) != len(item.values):
            raise ValueError(f"{item.label} upload metadata are incomplete.")
        if item.filenames and len(item.detected_forms) != len(item.filenames):
            raise ValueError(f"{item.label} inspection metadata are incomplete.")
        if item.form == "xt":
            incompatible = tuple(
                filename
                for filename, detected in zip(
                    item.filenames,
                    item.detected_forms,
                    strict=True,
                )
                if detected not in {None, "xt"}
            )
            if incompatible:
                names = ", ".join(incompatible)
                raise ValueError(
                    f"{item.label} expects one complete space-time matrix per dataset; "
                    f"these files are not space-time matrices: {names}."
                )
        if item.filenames:
            plan_batch_sources(
                (
                    BatchSourceSummary(
                        label=item.label,
                        filenames=item.filenames,
                        forms=item.detected_forms,
                        shapes=tuple(
                            tuple(int(size) for size in np.asarray(value).shape)
                            for value in item.values
                        ),
                    ),
                ),
                require_matching_shapes=False,
            )


def _plan_mapping_cases(
    inputs: Sequence[_PreparedBatchInput],
    *,
    target_label: str,
    combination_mode: str = "pairwise",
) -> tuple[_MappingBatchCase, ...]:
    """Resolve pairwise or full Cartesian Mapping cases with target-first names."""

    _validate_mapping_source_metadata(inputs)
    mode = "all" if combination_mode == "all" else "pairwise"
    source_sizes = tuple(len(item.values) for item in inputs)
    if mode == "all":
        projected = math.prod(source_sizes) if source_sizes else 1
    else:
        projected = max(source_sizes, default=1)
    if projected > _MAX_MAPPING_CASES:
        raise ValueError(
            f"These inputs would create {projected:,} mapping runs. "
            f"Reduce the inputs or use at most {_MAX_MAPPING_CASES:,} combinations."
        )
    combinations = plan_combination_indices(source_sizes, mode=mode)
    sources = tuple(
        CombinationSource(
            item.label,
            _prepared_components(item),
            include_singleton_in_short=item.form != "scalar",
        )
        for item in inputs
    )
    identities = combination_names(
        target_label,
        sources,
        combinations,
        include_singletons=True,
    )
    full_stems = unique_compact_export_stems(
        tuple(identity.full for identity in identities),
        fallback="mapping-result",
        maximum=150,
    )
    short_stems = unique_compact_export_stems(
        tuple(identity.short for identity in identities),
        fallback="mapping-result",
        maximum=150,
    )
    cases: list[_MappingBatchCase] = []
    for index, (identity, export_stem, short_export_stem, indices) in enumerate(
        zip(
            identities,
            full_stems,
            short_stems,
            combinations,
            strict=True,
        ),
        start=1,
    ):
        cases.append(
            _MappingBatchCase(
                key=f"mapping-case-{index}",
                name=identity.full,
                export_stem=export_stem,
                item_indices=tuple(indices),
                short_name=identity.short,
                short_export_stem=short_export_stem,
            )
        )
    return tuple(cases)


def _validate_mapping_case_axes(
    inputs: Sequence[_PreparedBatchInput],
    item_indices: Sequence[int],
    dataset_name: str,
) -> None:
    """Validate time axes; spatial inputs are aligned by the Mapping service."""

    if len(inputs) != len(item_indices):
        raise ValueError("A Mapping case must choose one item from every input.")
    temporal: dict[str, int] = {}
    for prepared, item_index in zip(inputs, item_indices, strict=True):
        value = prepared.values[item_index]
        array = np.asarray(value)
        if prepared.form == "t":
            temporal[prepared.label] = int(array.size)
        elif prepared.form == "xt" and array.ndim == 2:
            temporal[prepared.label] = int(array.shape[1])
    # Uploaded chromosome profiles and shorter local outputs can have different
    # spatial lengths. The service applies Start/End and caps every input to
    # their shortest usable length before validating the resulting grid.
    if len(set(temporal.values())) > 1:
        detail = ", ".join(f"{label}: {size:,}" for label, size in temporal.items())
        raise ValueError(f"Dataset {dataset_name!r} has inconsistent time points ({detail}).")


def _validate_paired_batch_axes(
    inputs: Sequence[_PreparedBatchInput],
    index: int,
    dataset_name: str,
) -> None:
    """Validate paired time axes before the service aligns their spatial inputs."""
    _validate_mapping_case_axes(
        inputs,
        tuple(0 if len(item.values) == 1 else index for item in inputs),
        dataset_name,
    )


def render(_navigate: Callable[[str], None]) -> None:
    """Render mapping routes, inputs, configuration, and output placeholders."""

    running = False
    active_cancel: Event | None = None
    restored_entry = consume_cached_workflow("mapping")
    restored_result = restored_entry.payload if restored_entry is not None else None
    latest_result: MappingResult | None = (
        restored_result if isinstance(restored_result, MappingResult) else None
    )
    restored_label = latest_result.target_label if latest_result is not None else ""
    latest_results: list[_NamedMappingResult] = (
        [
            _NamedMappingResult(
                "mapping-case-1",
                restored_label,
                _mapping_export_stem(restored_label, 1),
                latest_result,
            )
        ]
        if latest_result is not None
        else []
    )
    prepared_bundle_results: list[_NamedMappingResult] = []

    def prepared_result_candidates() -> tuple[_NamedMappingResult, ...]:
        # Retain the last fork-density batch when a downstream map replaces
        # the visible results. This state belongs only to the current client.
        compatible = [
            named for named in latest_results
            if named.result.target == "fork_densities_xt"
        ]
        if compatible:
            prepared_bundle_results[:] = compatible
        return tuple(prepared_bundle_results)

    active_dataset_key = latest_results[0].key if latest_results else ""
    result_chart: Any | None = None
    updating_y_limits = False
    y_limits_automatic = True
    mapping_input_controls: dict[str, _MappingInputControl] = {}
    mapping_upload_states: dict[str, _MappingUploadState] = {}
    mapping_constant_states: dict[str, _MappingConstantState] = {}
    mapping_combination_mode = {"value": "pairwise"}
    mapping_raw_preferences: dict[tuple[str, str], _MappingRawPreference] = {}
    mapping_batch_colors: dict[tuple[str, str, int], str] = {}
    batch_raw_page = {"value": 1}
    dataset_page = {"value": 1}
    region_setting_controls: dict[str, _SettingControl] = {}
    retained_spatial_resolution: dict[str, Any] = {}
    mapping_setting_controls: dict[str, _SettingControl] = {}

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
                    selected_families = _mapping_route_families_for_ui(
                        selected_target.key,
                        selected_dependence,
                    )
                    selected_family = selected_families[0]
                    initial_form_selections = _default_form_selections(selected_family)
                    selected_route = selected_family.resolve(initial_form_selections)
                    selected_route_state = {"route": selected_route}
                    selected_forms = dict(initial_form_selections)
                    calculation_state = _calculation_routes(selected_family, selected_forms)
                    synchronizing_calculation = False

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
                            calculation_control = (
                                ui.select(
                                    {key: key.capitalize() for key in calculation_state},
                                    value=next(iter(calculation_state)),
                                    label="Calculation",
                                )
                                .props("dense options-dense")
                                .classes("dn-field")
                                .mark("mapping-calculation")
                            )
                            calculation_control.set_visibility(len(calculation_state) > 1)

                    with ui.element("div").classes("dn-mapping-required-heading"):
                        ui.label("Inputs").classes("dn-setting-group-title").mark(
                            "mapping-heading-inputs"
                        )

                    def sync_target_shape(route: MappingRoute, *, retain: bool = False) -> None:
                        shapes = route.shapes_for(selected_forms)
                        previous = target_shape_control.value if retain else None
                        target_shape_control.options = {
                            shape: shape.replace("_", " ").title() for shape in shapes
                        }
                        if previous in shapes:
                            target_shape_control.value = previous
                        elif route.default_target_shape in shapes:
                            target_shape_control.value = route.default_target_shape
                        else:
                            target_shape_control.value = shapes[0] if shapes else None
                        target_shape_control.update()
                        target_shape_container.set_visibility(len(shapes) > 1)

                    def sync_preview_selection(route: MappingRoute) -> None:
                        invalidate_mapping_result()
                        if latest_result is not None:
                            return
                        mapping_stats["Form"].text = "Per field" if route.target == "kinetic_fields" else _DEPENDENCE_LABELS.get(
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

                        retain_shape = selected_route_state["route"].method == route.method
                        selected_route_state["route"] = route
                        sync_target_shape(route, retain=retain_shape)
                        render_region_settings.refresh()
                        render_mapping_settings.refresh()
                        sync_preview_selection(route)

                    def sync_calculation(
                        family: MappingRouteFamily,
                        selections: dict[str, DataForm],
                        *,
                        retain: bool = True,
                    ) -> None:
                        nonlocal synchronizing_calculation
                        selected_forms.clear()
                        selected_forms.update(selections)
                        calculation_state.clear()
                        calculation_state.update(_calculation_routes(family, selections))
                        previous = str(calculation_control.value) if retain else ""
                        choice = (
                            previous if previous in calculation_state else next(iter(calculation_state))
                        )
                        synchronizing_calculation = True
                        try:
                            calculation_control.set_options(
                                {key: key.capitalize() for key in calculation_state},
                                value=choice,
                            )
                            calculation_control.set_visibility(len(calculation_state) > 1)
                        finally:
                            synchronizing_calculation = False
                        sync_exact_route(calculation_state[choice])

                    def calculation_changed() -> None:
                        if (
                            not synchronizing_calculation
                            and calculation_control.value in calculation_state
                        ):
                            sync_exact_route(calculation_state[str(calculation_control.value)])

                    calculation_control.on_value_change(calculation_changed)

                    def scientific_input_changed() -> None:
                        invalidate_mapping_result()

                    refreshing_constant_surfaces = False

                    def counterpart_rows(current_key: str) -> tuple[tuple[str, str], ...]:
                        """Describe the uploaded datasets a constant list must align with."""

                        sources: list[tuple[str, str, tuple[UploadInspection, ...]]] = []
                        for input_key, live in mapping_input_controls.items():
                            if input_key == current_key:
                                continue
                            mode = str(getattr(live.surface.get("mode"), "value", ""))
                            inspections = tuple(
                                cast(
                                    list[UploadInspection],
                                    live.surface.get("inspections", []),
                                )
                            )
                            if mode == "profile" and inspections:
                                sources.append((input_key, live.spec.label, inspections))
                        if not sources:
                            return ()
                        row_count = max(len(inspections) for _key, _label, inspections in sources)
                        rows: list[tuple[str, str]] = []
                        for row_index in range(row_count):
                            identities: list[str] = []
                            labels: list[str] = []
                            for source_key, source_label, inspections in sources:
                                item_index = 0 if len(inspections) == 1 else row_index
                                if item_index >= len(inspections):
                                    identities.append(f"{source_key}:unpaired-{row_index + 1}")
                                    labels.append(f"{source_label}: unpaired")
                                    continue
                                inspection = inspections[item_index]
                                identities.append(f"{source_key}:{inspection.name}")
                                labels.append(
                                    inspection.name
                                    if len(sources) == 1
                                    else f"{source_label}: {inspection.name}"
                                )
                            rows.append(("|".join(identities), " · ".join(labels)))
                        return tuple(rows)

                    def refresh_constant_surfaces() -> None:
                        nonlocal refreshing_constant_surfaces
                        if refreshing_constant_surfaces:
                            return
                        refreshing_constant_surfaces = True
                        try:
                            for live in tuple(mapping_input_controls.values()):
                                mode = str(getattr(live.surface.get("mode"), "value", ""))
                                if mode == "constants":
                                    live.refresh_surface()
                        finally:
                            refreshing_constant_surfaces = False

                    def scientific_input_structure_changed() -> None:
                        """Refresh only cardinality-dependent constant rows."""

                        invalidate_mapping_result()
                        refresh_constant_surfaces()

                    @ui.refreshable
                    def render_required_inputs() -> None:
                        mapping_input_controls.clear()
                        family = _route_family(str(route_control.value))
                        selections = _retained_form_selections(family, mapping_upload_states)
                        initial_route = family.resolve(selections)
                        displayed_forms: dict[str, DataForm] = {}
                        controls: dict[str, Any] = {}
                        refreshers: dict[str, Callable[[], Any]] = {}
                        overrides: dict[str, Any] = {}
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
                                retained_constants = mapping_constant_states.setdefault(
                                    _friendly_input_key(key),
                                    _MappingConstantState(),
                                )
                                live = _render_mapping_input(
                                    input_spec,
                                    forms=allowed,
                                    selected_form=displayed_forms[key],
                                    retained=retained,
                                    retained_constants=retained_constants,
                                    peer_constant_rows=(
                                        lambda input_key=key: counterpart_rows(input_key)
                                    ),
                                    combination_mode=mapping_combination_mode,
                                    prepared_results=prepared_result_candidates,
                                    on_change=scientific_input_changed,
                                    on_structure_change=scientific_input_structure_changed,
                                    on_error=lambda message: report_mapping_input_error(message),
                                )
                                control = live.form
                                controls[key] = control
                                refreshers[key] = live.refresh_surface
                                mapping_input_controls[key] = live
                                if live.override is not None:
                                    overrides[key] = live.override

                        # A retained peer upload may be rendered after its
                        # constant counterpart. Rebuild only those constant
                        # rows once the complete route input grid exists.
                        refresh_constant_surfaces()

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

                                sync_calculation(family, selections)
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
                        selections = _retained_form_selections(family, mapping_upload_states)
                        render_required_inputs.refresh()
                        sync_calculation(family, selections, retain=False)

                    def sync_route_families() -> None:
                        if target_control.value is None or dependence_control.value is None:
                            return
                        families = _mapping_route_families_for_ui(
                            str(target_control.value),
                            str(dependence_control.value),
                        )
                        route_control.set_options(
                            _route_options(families),
                            value=families[0].key,
                        )
                        sync_selected_family()

                    def sync_target() -> None:
                        if target_control.value is None:
                            return
                        target = mapping_target(str(target_control.value))
                        dependence_control.options = _select_options(
                            target.dependences,
                            {**_DEPENDENCE_LABELS, "spacetime": "Per field"}
                            if target.key == "kinetic_fields" else _DEPENDENCE_LABELS,
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
                        render_region_settings.refresh()
                        render_mapping_settings.refresh()
                        invalidate_mapping_result()

                    target_shape_control.on_value_change(target_shape_changed)

                with expansion("Region", icon="straighten"):
                    with ui.element("div").classes("dn-mapping-region-scope"):
                        whole_region = (
                            ui.checkbox("Whole region", value=False)
                            .classes("dn-mapping-whole-region")
                            .mark("mapping-whole-region")
                        )
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
                                label="End boundary (kb, exclusive)",
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
                        if "resolution" in region_setting_controls:
                            retained_spatial_resolution["value"] = (
                                region_setting_controls["resolution"].control.value
                            )
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
                                    target_shape=str(target_shape_control.value or ""),
                                    input_forms=selected_forms,
                                    on_change=lambda: invalidate_mapping_result(),
                                )
                            )
                        # Bin width describes the uploaded data, so input-form
                        # changes must not reset it and shift Start/End indices.
                        if (
                            "resolution" in region_setting_controls
                            and "value" in retained_spatial_resolution
                        ):
                            region_setting_controls["resolution"].control.set_value(
                                retained_spatial_resolution["value"]
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
                                    input_forms=selected_forms,
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
                    if latest_result is not None and latest_result.primary_output in initial_outputs
                    else next(iter(initial_outputs))
                )
                initial_output_label = _OUTPUT_LABELS.get(
                    initial_output_key,
                    initial_output_key.replace("_", " ").title(),
                )
                with (
                    ui.element("div")
                    .classes("dn-stat-grid dn-simulation-summary-grid dn-mapping-summary-grid")
                    .mark("mapping-preview-output-bar") as mapping_summary_bar
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
                        .classes("dn-stat dn-simulation-output-stat")
                        .mark("mapping-layout-stat") as layout_stat
                    ):
                        ui.label("Layout").classes("dn-stat-label")
                        preview_layout = (
                            ui.select(
                                ["Combined", "Individual"],
                                value="Combined",
                            )
                            .props('dense borderless options-dense aria-label="Preview layout"')
                            .classes("dn-field dn-simulation-output-select")
                            .mark("mapping-preview-layout")
                        )
                    layout_stat.visible = False
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

                @ui.refreshable
                def render_dataset_selector() -> None:
                    if len(latest_results) <= 1 or str(preview_layout.value) != "Individual":
                        return
                    active_index = next(
                        (
                            index
                            for index, item in enumerate(latest_results)
                            if item.key == active_dataset_key
                        ),
                        0,
                    )
                    active_page = active_index // _BATCH_NAV_PAGE_SIZE + 1
                    page_count = max(
                        1,
                        math.ceil(len(latest_results) / _BATCH_NAV_PAGE_SIZE),
                    )
                    requested_page = min(
                        max(1, int(dataset_page.get("value", active_page))),
                        page_count,
                    )
                    page_start = (requested_page - 1) * _BATCH_NAV_PAGE_SIZE
                    page_end = min(
                        len(latest_results),
                        page_start + _BATCH_NAV_PAGE_SIZE,
                    )
                    if not page_start <= active_index < page_end:
                        requested_page = active_page
                        page_start = (requested_page - 1) * _BATCH_NAV_PAGE_SIZE
                        page_end = min(
                            len(latest_results),
                            page_start + _BATCH_NAV_PAGE_SIZE,
                        )
                    dataset_page["value"] = requested_page
                    with (
                        ui.row()
                        .classes("w-full items-end dn-dataset-selector-row")
                        .mark("mapping-dataset-selector")
                    ):
                        if page_count > 1:
                            page_control = (
                                ui.number(
                                    label="Page",
                                    value=requested_page,
                                    min=1,
                                    max=page_count,
                                    step=1,
                                )
                                .props("dense")
                                .classes("dn-field dn-dataset-page-select")
                                .mark("mapping-dataset-page")
                            )

                            def page_changed() -> None:
                                try:
                                    page = int(page_control.value)
                                except (TypeError, ValueError):
                                    return
                                page = min(max(1, page), page_count)
                                dataset_page["value"] = page
                                select_dataset(
                                    latest_results[
                                        (page - 1) * _BATCH_NAV_PAGE_SIZE
                                    ].key
                                )
                                render_dataset_selector.refresh()

                            page_control.on_value_change(page_changed)
                        options = {
                            item.key: _mapping_display_name(item)
                            for item in latest_results[page_start:page_end]
                        }
                        dataset_select = (
                            ui.select(
                                options,
                                value=active_dataset_key,
                                label="Dataset",
                            )
                            .props("dense options-dense use-input input-debounce=0")
                            .classes("dn-field dn-dataset-select")
                            .mark("mapping-dataset-select")
                        )
                        active_item = latest_results[active_index]
                        dataset_select.tooltip(active_item.name)
                        dataset_select.on_value_change(
                            lambda event: select_dataset(str(event.value))
                        )

                render_dataset_selector()
                stats_container.visible = False
                progress = (
                    ui.linear_progress(value=0.0, show_value=False)
                    .props("rounded instant-feedback")
                    .classes("dn-simulation-progress dn-mapping-progress")
                    .mark("mapping-progress")
                )
                progress.visible = False
                status = ui.label("Ready").classes("dn-muted").mark("mapping-status")
                eta_label = ui.label("").classes("dn-muted").mark("mapping-eta")
                eta_label.visible = False
                eta_timer: ui.timer | None = None

                def stop_disconnected_eta() -> None:
                    if eta_timer is not None:
                        eta_timer.cancel()

                ui.context.client.on_disconnect(stop_disconnected_eta)
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
                    initial_x_label="Position (kb)",
                    initial_y_label=initial_output_label,
                    on_reset_y_limits=lambda: populate_automatic_y_limits(),
                    markers=preview_settings_markers(
                        "mapping",
                        overrides={
                            "x_axis_label": "mapping-preview-x-label",
                            "y_axis_label": "mapping-preview-y-label",
                            "primary_color": "mapping-line-or-bar-color",
                            "secondary_color": "mapping-second-curve-color",
                        },
                    ),
                )
                view = plot_settings.view
                preview_title = plot_settings.title
                preview_x_label = plot_settings.x_axis_label
                preview_y_label = plot_settings.y_axis_label
                line_or_bar_color = plot_settings.primary_color
                second_curve_color = plot_settings.secondary_color
                heatmap_palette = plot_settings.heatmap_palette
                log_x = plot_settings.log_x
                invert_x = plot_settings.invert_x
                log_z = plot_settings.log_z
                invert_z = plot_settings.invert_z
                line_width = plot_settings.line_width
                histogram_bins = plot_settings.histogram_bins
                histogram_touching = plot_settings.histogram_touching
                histogram_boundary = plot_settings.histogram_boundary
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
                    .mark("mapping-dataset-color-settings") as batch_color_container
                ):
                    with expansion(
                        "Curves",
                        icon="palette",
                        extra_classes="dn-preview-settings-section",
                    ):

                        @ui.refreshable
                        def render_batch_colors() -> None:
                            if (
                                len(latest_results) < 2
                                or str(preview_layout.value or "Combined") != "Combined"
                            ):
                                return
                            output_key = str(
                                preview_output.value
                                or latest_results[0].result.primary_output
                            )
                            roles = (
                                ("Observed", "Fitted")
                                if output_key.startswith(_COMPARISON_PREFIX)
                                else ("",)
                            )
                            with ui.element("div").classes("dn-preview-settings-grid"):
                                curve_index = 0
                                for item in latest_results:
                                    for series_index, role in enumerate(roles):
                                        color_key = (
                                            item.key,
                                            output_key,
                                            series_index,
                                        )
                                        default_color = _BATCH_COLOURS[
                                            curve_index % len(_BATCH_COLOURS)
                                        ]
                                        mapping_batch_colors.setdefault(
                                            color_key,
                                            default_color,
                                        )
                                        short_label = _mapping_display_name(item)
                                        control_label = (
                                            short_label
                                            if not role
                                            else f"{short_label} · {role}"
                                        )
                                        color_surface, color_control = compact_color_input(
                                            control_label,
                                            mapping_batch_colors[color_key],
                                            marker=(
                                                "mapping-dataset-color-"
                                                f"{curve_index + 1}"
                                            ),
                                        )
                                        color_surface.tooltip(
                                            item.name if not role else f"{item.name} · {role}"
                                        )

                                        def color_changed(
                                            _event: Any = None,
                                            *,
                                            retained_key: tuple[str, str, int] = color_key,
                                            control: Any = color_control,
                                        ) -> None:
                                            mapping_batch_colors[retained_key] = str(
                                                control.value or "#2c7f77"
                                            )
                                            render_latest_result()

                                        color_control.on_value_change(color_changed)
                                        curve_index += 1

                        render_batch_colors()
                batch_color_container.visible = bool(
                    len(latest_results) > 1
                    and str(preview_layout.value or "Combined") == "Combined"
                )
                raw_controls = raw_output_controls(
                    file_name=f"{RMAP_FILE_PREFIX}_{output_file_token(initial_output_key)}",
                    default_type="TSV",
                    default_compress=False,
                    on_remove=lambda _controls: remove_mapping_result(
                        active_dataset_key
                    ),
                )
                single_raw_automatic_name = {
                    "value": f"{RMAP_FILE_PREFIX}_{output_file_token(initial_output_key)}"
                }

                def set_single_raw_automatic_name(next_name: str) -> None:
                    """Update an untouched filename while preserving manual edits."""

                    current_name = str(raw_controls.file_name.value or "")
                    if current_name in {
                        "",
                        str(single_raw_automatic_name["value"]),
                    }:
                        raw_controls.file_name.value = next_name
                        raw_controls.file_name.update()
                    single_raw_automatic_name["value"] = next_name
                raw_controls.file_name.mark("mapping-raw-filename")
                raw_controls.file_type.mark("mapping-raw-file-type")
                assert raw_controls.extra_slot is not None

                with raw_controls.extra_slot:

                    @ui.refreshable
                    def render_batch_raw_files() -> None:
                        multiple = len(latest_results) > 1
                        single = len(latest_results) == 1
                        raw_controls.row.set_visibility(single)
                        if raw_controls.remove_button is not None:
                            raw_controls.remove_button.set_visibility(single)
                        if not multiple:
                            batch_raw_page["value"] = 1
                            return
                        output_key = str(
                            preview_output.value
                            or latest_results[0].result.primary_output
                        )
                        rows: list[
                            tuple[_NamedMappingResult, _MappingRawPreference]
                        ] = []
                        for item in latest_results:
                            preference = _mapping_raw_preference(
                                mapping_raw_preferences,
                                item,
                                output_key,
                            )
                            rows.append((item, preference))

                        page, page_count, start, end = _mapping_raw_page_bounds(
                            len(rows),
                            int(batch_raw_page["value"]),
                        )
                        batch_raw_page["value"] = page
                        with (
                            ui.element("div")
                            .classes("dn-download-file-list dn-mapping-batch-raw-list")
                            .mark("mapping-batch-raw-files")
                        ):
                            if page_count > 1:

                                def raw_page_changed(event: Any) -> None:
                                    try:
                                        requested = int(event.value)
                                    except (TypeError, ValueError):
                                        return
                                    requested = min(max(requested, 1), page_count)
                                    if requested == batch_raw_page["value"]:
                                        return
                                    batch_raw_page["value"] = requested
                                    render_batch_raw_files.refresh()

                                with ui.row().classes("dn-mapping-batch-raw-pager"):
                                    (
                                        ui.number(
                                            "Page",
                                            value=page,
                                            min=1,
                                            max=page_count,
                                            step=1,
                                            on_change=raw_page_changed,
                                        )
                                        .props("dense")
                                        .classes("dn-field")
                                        .mark("mapping-batch-raw-page")
                                    )
                                    ui.label(
                                        f"of {page_count} · {len(rows):,} files"
                                    ).classes("dn-mapping-batch-raw-page-count")
                            for item, preference in rows[start:end]:
                                with ui.element("div").classes(
                                    "dn-mapping-batch-raw-item"
                                ):
                                    title = ui.label(_mapping_display_name(item)).classes(
                                        "dn-download-file-name dn-mapping-batch-raw-title"
                                    )
                                    title.tooltip(item.name)
                                    row_controls = raw_output_file_row(
                                        file_name=preference.file_name,
                                        default_type=preference.file_type,
                                        default_compress=preference.compress,
                                        on_download=(
                                            lambda controls, named=item: download_named_raw(
                                                named,
                                                controls,
                                            )
                                        ),
                                        on_remove=(
                                            lambda _controls, result_key=item.key: (
                                                remove_mapping_result(result_key)
                                            )
                                        ),
                                        markers={
                                            "row": f"mapping-batch-raw-row-{item.key}",
                                            "file_name": f"mapping-batch-raw-name-{item.key}",
                                            "file_type": f"mapping-batch-raw-type-{item.key}",
                                            "compression": (
                                                "mapping-batch-raw-compression-"
                                                f"{item.key}"
                                            ),
                                            "download": f"mapping-download-raw-{item.key}",
                                            "remove": f"mapping-remove-raw-{item.key}",
                                        },
                                        extra_classes="dn-mapping-batch-raw-row",
                                    )
                                    def store_file_name(
                                        control: Any = row_controls.file_name,
                                        retained: _MappingRawPreference = preference,
                                    ) -> None:
                                        retained.file_name = str(control.value or "")
                                        retained.customized = bool(
                                            retained.file_name
                                            and retained.file_name != retained.automatic_name
                                        )

                                    def store_file_type(
                                        control: Any = row_controls.file_type,
                                        retained: _MappingRawPreference = preference,
                                    ) -> None:
                                        retained.file_type = str(control.value or "TSV")

                                    def store_compression(
                                        control: Any = row_controls.compress,
                                        retained: _MappingRawPreference = preference,
                                    ) -> None:
                                        retained.compress = bool(control.value)

                                    row_controls.file_name.on_value_change(store_file_name)
                                    row_controls.file_type.on_value_change(store_file_type)
                                    row_controls.compress.on_value_change(store_compression)
                            with ui.row().classes("dn-data-manager-delete-all-row"):
                                (
                                    ui.button(
                                        "Download all",
                                        icon="download",
                                        on_click=download_all_raw,
                                    )
                                    .props("flat no-caps")
                                    .classes("dn-action-secondary")
                                    .mark("mapping-download-all-raw")
                                )
                                (
                                    ui.button(
                                        "Delete all",
                                        icon="delete_sweep",
                                        on_click=lambda: remove_all_mapping_results(),
                                    )
                                    .props("flat no-caps")
                                    .classes("dn-action-secondary")
                                    .mark("mapping-delete-all-raw")
                                )

                    render_batch_raw_files()

                def remove_mapping_result(result_key: str) -> None:
                    """Remove one completed result and reconcile the live batch preview."""

                    nonlocal active_dataset_key, latest_result
                    if running:
                        ui.notify(
                            "Stop the current mapping run before removing results.",
                            type="warning",
                        )
                        return
                    removed_index = next(
                        (
                            index
                            for index, item in enumerate(latest_results)
                            if item.key == result_key
                        ),
                        None,
                    )
                    if removed_index is None:
                        return
                    latest_results.pop(removed_index)
                    for preference_key in tuple(mapping_raw_preferences):
                        if preference_key[0] == result_key:
                            mapping_raw_preferences.pop(preference_key, None)
                    for color_key in tuple(mapping_batch_colors):
                        if color_key[0] == result_key:
                            mapping_batch_colors.pop(color_key, None)
                    if not latest_results:
                        clear_mapping_output()
                        return

                    replacement = next(
                        (
                            item
                            for item in latest_results
                            if item.key == active_dataset_key
                        ),
                        latest_results[min(removed_index, len(latest_results) - 1)],
                    )
                    active_dataset_key = replacement.key
                    latest_result = replacement.result
                    sync_batch_presentation()
                    render_dataset_selector.refresh()
                    render_batch_raw_files.refresh()
                    select_dataset(replacement.key)

                def remove_all_mapping_results() -> None:
                    """Remove every completed Mapping result from the live preview."""

                    if running:
                        ui.notify(
                            "Stop the current mapping run before removing results.",
                            type="warning",
                        )
                        return
                    clear_mapping_output()

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

                def select_dataset(key: str) -> None:
                    """Activate one completed dataset without re-running Mapping."""

                    nonlocal active_dataset_key, latest_result
                    selected = next(
                        (item for item in latest_results if item.key == key),
                        None,
                    )
                    if selected is None:
                        return
                    active_dataset_key = selected.key
                    latest_result = selected.result
                    requested_output = str(preview_output.value or "")
                    options = (
                        _common_result_preview_outputs(
                            tuple(item.result for item in latest_results)
                        )
                        if len(latest_results) > 1
                        else _result_preview_outputs(latest_result)
                    )
                    preview_output.options = options
                    preview_output.value = (
                        requested_output
                        if requested_output in options
                        else latest_result.primary_output
                        if latest_result.primary_output in options
                        else next(iter(options))
                    )
                    if len(options) == 1:
                        preview_output.disable()
                    else:
                        preview_output.enable()
                    preview_output.update()
                    primary = np.asarray(latest_result.outputs[latest_result.primary_output])
                    mapping_stats["Form"].text = "Per field" if latest_result.target == "kinetic_fields" else _DEPENDENCE_LABELS.get(
                        latest_result.output_dependence,
                        latest_result.output_dependence.replace("_", " ").title(),
                    )
                    mapping_stats["Points"].text = f"{primary.size:,}"
                    mapping_stats["Method"].text = latest_result.method.replace("_", " ").title()
                    for value in mapping_stats.values():
                        value.update()
                    apply_preview_output()
                    populate_automatic_y_limits(redraw=False)
                    render_latest_result()

                def combined_output_supported() -> bool:
                    plot_type = resolved_plot_type()
                    if (
                        not 2 <= len(latest_results) <= _MAX_COMBINED_MAPPING_DATASETS
                        or plot_type not in {"Line", "Histogram"}
                    ):
                        return False
                    output_key = str(
                        preview_output.value
                        or (latest_result.primary_output if latest_result else "")
                    )
                    if output_key.startswith(_COMPARISON_PREFIX):
                        return plot_type == "Line" and all(
                            output_key.removeprefix(_COMPARISON_PREFIX) in item.result.comparisons
                            for item in latest_results
                        )
                    return all(
                        (actual := _resolved_output_key(item.result, output_key))
                        in item.result.outputs
                        and np.asarray(item.result.outputs[actual]).ndim <= 1
                        for item in latest_results
                    )

                def sync_batch_presentation() -> None:
                    multiple = len(latest_results) > 1
                    layout_stat.visible = multiple
                    mapping_summary_bar.classes(
                        add="dn-mapping-summary-grid-batch" if multiple else None,
                        remove=None if multiple else "dn-mapping-summary-grid-batch",
                    )
                    options = (
                        ["Combined", "Individual"]
                        if multiple and combined_output_supported()
                        else ["Individual"]
                        if multiple
                        else ["Combined"]
                    )
                    preview_layout.options = options
                    if preview_layout.value not in options:
                        preview_layout.value = options[0]
                    preview_layout.update()
                    combined = bool(
                        multiple
                        and str(preview_layout.value or "Combined") == "Combined"
                        and combined_output_supported()
                    )
                    batch_color_container.visible = combined
                    if combined:
                        render_batch_colors.refresh()
                    render_dataset_selector.refresh()

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
                    comparison_curves = output_key.startswith(_COMPARISON_PREFIX)
                    combined_batch = bool(
                        len(latest_results) > 1
                        and str(preview_layout.value or "Combined") == "Combined"
                    )
                    multiple_curves = comparison_curves or combined_batch
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
                        allow_histogram=(
                            output_form not in {"t", "xt"}
                            and actual_output_key != "fit_history"
                            and not comparison_curves
                        ),
                        allow_boxplot=dependence == "ensemble" and not multiple_curves,
                    )
                    if (
                        len(latest_results) > 1
                        and str(preview_layout.value or "Combined") == "Combined"
                    ):
                        plot_settings.primary_color_surface.visible = False
                        plot_settings.secondary_color_surface.visible = False

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
                    if not plotting_settings_ready(
                        plot_settings,
                        plot_type=resolved_plot_type(),
                        allow_empty_y_limits=y_limits_automatic,
                    ):
                        return
                    try:
                        limit_results = (
                            [item.result for item in latest_results]
                            if len(latest_results) > 1 and str(preview_layout.value) == "Combined"
                            else [latest_result]
                        )
                        candidates = [
                            _mapping_automatic_y_limits(
                                item,
                                output_key=str(preview_output.value or item.primary_output),
                                plot_type=resolved_plot_type(),
                                histogram_bins=int(_read_number(histogram_bins, "Histogram bins")),
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
                                histogram_boundary=str(histogram_boundary.value or "left"),
                            )
                            for item in limit_results
                        ]
                        finite_limits = [item for item in candidates if item is not None]
                        limits = (
                            (
                                min(item[0] for item in finite_limits),
                                max(item[1] for item in finite_limits),
                            )
                            if finite_limits
                            else None
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
                        calculation_control,
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
                        capture_format = "png" if str(file_format).casefold() == "png" else "svg"
                        capture_options: dict[str, Any] = {
                            "type": capture_format,
                            "backgroundColor": "#ffffff",
                        }
                        if capture_format == "png":
                            capture_options["pixelRatio"] = pixel_ratio
                        data_uri = await result_chart.run_chart_method(
                            "getDataURL",
                            capture_options,
                            timeout=10,
                        )
                        output_key = str(preview_output.value or latest_result.primary_output)
                        active_named = next(
                            (
                                item
                                for item in latest_results
                                if item.key == active_dataset_key
                            ),
                            None,
                        )
                        individual_result = (
                            active_named is not None
                            and (
                                len(latest_results) == 1
                                or str(preview_layout.value) == "Individual"
                            )
                        )
                        plot_name = (
                            f"{_mapping_raw_default_filename(active_named, output_key)}_plot"
                            if individual_result and active_named is not None
                            else f"{RMAP_FILE_PREFIX}_"
                            f"{_mapping_output_file_token(latest_result, output_key)}"
                            "_combined_plot"
                        )
                        artifact = await run.io_bound(
                            build_chart_capture_export,
                            data_uri,
                            file_format,
                            filename=plot_name,
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

                def snapshot_named_raw_export(
                    named: _NamedMappingResult,
                    *,
                    output_key: str | None = None,
                    controls: RawOutputControls | None = None,
                    preference: _MappingRawPreference | None = None,
                ) -> _MappingRawExportSelection:
                    """Freeze one row before any export work yields to the UI."""

                    selected_output = str(
                        output_key
                        or preview_output.value
                        or named.result.primary_output
                    )
                    if controls is not None:
                        file_name = str(controls.file_name.value or "").strip()
                        file_type = str(controls.file_type.value or "TSV")
                        compress = bool(controls.compress.value)
                    elif preference is not None:
                        file_name = str(preference.file_name or "").strip()
                        file_type = str(preference.file_type or "TSV")
                        compress = bool(preference.compress)
                    else:
                        raise RuntimeError("Raw export settings are unavailable.")
                    return _MappingRawExportSelection(
                        output_key=selected_output,
                        file_name=(
                            file_name
                            or _mapping_raw_default_filename(named, selected_output)
                        ),
                        file_type=file_type,
                        compression=(
                            get_export_settings().raw_compression
                            if compress
                            else "none"
                        ),
                    )

                async def build_named_raw_artifact(
                    named: _NamedMappingResult,
                    selection: _MappingRawExportSelection,
                ) -> tuple[ExportArtifact, str]:
                    if selection.output_key.startswith(_COMPARISON_PREFIX):
                        fitted_key = selection.output_key.removeprefix(
                            _COMPARISON_PREFIX
                        )
                        artifact = await run.io_bound(
                            build_mapping_comparison_raw_export,
                            named.result,
                            fitted_key,
                            selection.file_type,
                            filename=selection.file_name,
                            compression=selection.compression,
                        )
                    else:
                        exported_output_key = _resolved_output_key(
                            named.result,
                            selection.output_key,
                        )
                        artifact = await run.io_bound(
                            build_mapping_raw_export,
                            named.result,
                            exported_output_key,
                            selection.file_type,
                            filename=selection.file_name,
                            compression=selection.compression,
                        )
                    if artifact is None or not artifact.data:
                        raise RuntimeError("Raw export stopped before returning data.")
                    return artifact, selection.output_key

                async def download_named_raw(
                    named: _NamedMappingResult,
                    controls: RawOutputControls,
                ) -> None:
                    try:
                        selection = snapshot_named_raw_export(
                            named,
                            controls=controls,
                        )
                        artifact, output_key = await build_named_raw_artifact(
                            named,
                            selection,
                        )
                    except (RuntimeError, TypeError, ValueError) as exc:
                        ui.notify(str(exc), type="negative")
                        return
                    cache_artifact(
                        artifact,
                        source="Mapping",
                        kind="raw",
                        detail=(
                            f"{_mapping_display_name(named)} · "
                            + output_key.replace(":", " ").replace("_", " ").title()
                        ),
                        navigate_to="mapping",
                    )
                    ui.download(artifact.data, artifact.filename, artifact.media_type)

                async def download_all_raw() -> None:
                    if not latest_results:
                        ui.notify("Run Mapping before downloading data.", type="warning")
                        return
                    try:
                        named_results = tuple(latest_results)
                        output_key = str(
                            preview_output.value
                            or named_results[0].result.primary_output
                        )
                        selections = tuple(
                            snapshot_named_raw_export(
                                item,
                                output_key=output_key,
                                preference=_mapping_raw_preference(
                                    mapping_raw_preferences,
                                    item,
                                    output_key,
                                ),
                            )
                            for item in named_results
                        )
                        # Build sequentially to avoid launching one large raw
                        # conversion per dataset at the same time.
                        prepared: list[tuple[ExportArtifact, str]] = []
                        prepared_bytes = 0
                        for item, selection in zip(
                            named_results,
                            selections,
                            strict=True,
                        ):
                            built = await build_named_raw_artifact(item, selection)
                            prepared_bytes += len(built[0].data)
                            if prepared_bytes > DEFAULT_BATCH_ARCHIVE_SOURCE_BYTES:
                                raise ValueError(
                                    "The bulk download would contain more than 256 MiB "
                                    "of raw data. Download the files individually or "
                                    "reduce the batch."
                                )
                            prepared.append(built)
                        artifacts = tuple(item[0] for item in prepared)
                        archive = await run.io_bound(
                            build_artifact_archive,
                            artifacts,
                            filename=(
                                f"{RMAP_FILE_PREFIX}_"
                                f"{_mapping_output_file_token(named_results[0].result, output_key)}"
                                "_batch.zip"
                            ),
                            max_source_bytes=DEFAULT_BATCH_ARCHIVE_SOURCE_BYTES,
                        )
                    except (OSError, RuntimeError, TypeError, ValueError) as exc:
                        ui.notify(str(exc), type="negative")
                        return
                    for named, (artifact, output_key) in zip(
                        named_results,
                        prepared,
                        strict=True,
                    ):
                        cache_artifact(
                            artifact,
                            source="Mapping",
                            kind="raw",
                            detail=(
                                f"{_mapping_display_name(named)} · "
                                + output_key.replace(":", " ").replace("_", " ").title()
                            ),
                            navigate_to="mapping",
                        )
                    ui.download(archive.data, archive.filename, archive.media_type)

                async def download_raw(controls: RawOutputControls) -> None:
                    if latest_result is None:
                        ui.notify("Run Mapping before downloading data.", type="warning")
                        return
                    named = next(
                        (item for item in latest_results if item.result is latest_result),
                        _NamedMappingResult(
                            "mapping-case-1",
                            latest_result.target_label,
                            _mapping_export_stem(latest_result.target_label, 1),
                            latest_result,
                        ),
                    )
                    await download_named_raw(named, controls)

                def render_latest_result() -> None:
                    nonlocal result_chart
                    if latest_result is None:
                        return
                    if not plotting_settings_ready(
                        plot_settings,
                        plot_type=resolved_plot_type(),
                        allow_empty_y_limits=y_limits_automatic,
                    ):
                        return
                    result_chart = None
                    result_slot.clear()
                    with result_slot:
                        try:
                            plot_kwargs = dict(
                                output_key=str(
                                    preview_output.value or latest_result.primary_output
                                ),
                                plot_type=resolved_plot_type(),
                                normalization="None",
                                title=str(preview_title.value or ""),
                                series_label=dict(preview_output.options)
                                .get(
                                    str(preview_output.value),
                                    str(preview_output.value or "Mapped output")
                                    .replace("_", " ")
                                    .title(),
                                )
                                .removesuffix(" (target)"),
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
                                log_x=bool(log_x.value),
                                invert_x=bool(invert_x.value),
                                invert_z=bool(invert_z.value),
                                density=bool(density.value),
                                show_legend=bool(show_legend.value),
                                legend_position=str(legend_position.value or "Top right"),
                                show_histogram_statistics=bool(show_histogram_statistics.value),
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
                                histogram_touching=bool(histogram_touching.value),
                                histogram_boundary=str(histogram_boundary.value or "left"),
                                show_grid=bool(show_grid.value),
                                y_limits=current_y_limits(),
                            )
                            if len(latest_results) > 1 and str(preview_layout.value) == "Combined":
                                options = _combined_mapping_plot_options(
                                    latest_results,
                                    colors=mapping_batch_colors,
                                    **plot_kwargs,
                                )
                                result_chart = render_preview_chart(
                                    options,
                                    title=str(preview_title.value or ""),
                                    marker="mapping-result-plot",
                                    on_download=download_plot,
                                )
                            else:
                                result_chart = _render_mapping_result(
                                    latest_result,
                                    **plot_kwargs,
                                    on_download=download_plot,
                                )
                        except (RuntimeError, TypeError, ValueError) as exc:
                            ui.notify(str(exc), type="negative")
                            preview_plot("Output unavailable", kind="mapping")

                async def map_data() -> None:
                    nonlocal \
                        active_cancel, \
                        active_dataset_key, \
                        latest_result, \
                        result_chart, \
                        running, \
                        eta_timer
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
                    eta = MappingEta()
                    eta_label.text = eta.text()
                    eta_label.visible = True
                    eta_timer = ui.timer(.2, lambda: eta_label.set_text(eta.text()))
                    eta_timer.mark("mapping-eta-heartbeat")
                    stats_container.visible = False
                    if not latest_results:
                        result_chart = None
                        result_slot.clear()
                        with result_slot:
                            ui.label("Mapping is running in the background...").classes("dn-muted")
                    try:
                        route = selected_route_state["route"]
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
                        # Bin any accompanying genomic tracks on the retained
                        # result's intrinsic grid, before service validation.
                        for live in mapping_input_controls.values():
                            if live.surface.get("prepared_only"):
                                bundle = _read_mapping_input(live)
                                source_axis = np.asarray(bundle["x_array"], dtype=float)
                                if source_axis.size > 1:
                                    selected_resolution = float(source_axis[1] - source_axis[0])
                                break
                        use_whole_region = bool(whole_region.value)

                        async def read_input(
                            key: str,
                            live: _MappingInputControl,
                        ) -> _PreparedBatchInput | None:
                            try:
                                return await _read_mapping_input_batch_async(
                                    key,
                                    live,
                                    missing_policy=missing_policy,
                                    region_start_kb=selected_start,
                                    region_end_kb=selected_end,
                                    resolution_x=selected_resolution,
                                    whole_genome=use_whole_region,
                                )
                            except KeyError:
                                return None

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
                        prepared_inputs = tuple(
                            item for item in await preparation if item is not None
                        )
                        check_cancelled()
                        has_multiple_constants = any(
                            str(getattr(live.surface.get("mode"), "value", ""))
                            == "constants"
                            for live in mapping_input_controls.values()
                        )
                        cases = _plan_mapping_cases(
                            prepared_inputs,
                            target_label=mapping_target(route.target_key).label,
                            combination_mode=(
                                mapping_combination_mode.get("value", "pairwise")
                                if has_multiple_constants
                                else "pairwise"
                            ),
                        )

                        requests: list[tuple[_MappingBatchCase, MappingRequest]] = []
                        for case in cases:
                            _validate_mapping_case_axes(
                                prepared_inputs,
                                case.item_indices,
                                case.name,
                            )
                            inputs = {
                                item.key: item.values[item_index]
                                for item, item_index in zip(
                                    prepared_inputs,
                                    case.item_indices,
                                    strict=True,
                                )
                            }
                            forms = {item.key: item.form for item in prepared_inputs}
                            prewindowed_inputs = {
                                item.key
                                for item, item_index in zip(
                                    prepared_inputs,
                                    case.item_indices,
                                    strict=True,
                                )
                                if item.prewindowed[item_index]
                            }
                            requests.append(
                                (
                                    case,
                                    MappingRequest(
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
                                    ),
                                )
                            )

                        # Validate every paired request before the first rmap
                        # worker starts, so a bad batch cannot publish partial data.
                        status.text = "Checking batch consistency..."
                        for _case, request in requests:
                            checked = await run.io_bound(
                                validate_mapping_request,
                                request,
                                check_cancelled,
                            )
                            if checked is None:
                                raise RuntimeError("Mapping validation stopped before returning.")
                        check_cancelled()

                        staged_results: list[_NamedMappingResult] = []
                        total_requests = len(requests)
                        for index, (case, request) in enumerate(requests):
                            progress_state = _ProgressState()
                            eta.start_case(total_requests)

                            def report_mapping(
                                value: float,
                                message: str,
                                state: _ProgressState = progress_state,
                            ) -> None:
                                state.report(value, message)
                                eta.report(value, message)

                            worker = asyncio.create_task(
                                run.io_bound(
                                    run_mapping,
                                    request,
                                    report_mapping,
                                    check_cancelled,
                                )
                            )
                            while not worker.done():
                                if cancel_event.is_set():
                                    status.text = "Stopping mapping..."
                                else:
                                    value, message = progress_state.snapshot()
                                    progress.value = max(
                                        0.02,
                                        (index + value) / total_requests,
                                    )
                                    status.text = (
                                        f"Dataset {index + 1}/{total_requests} · {message}"
                                        if total_requests > 1
                                        else message
                                    )
                                await asyncio.sleep(0.04)
                            result = await worker
                            if result is None:
                                check_cancelled()
                                raise RuntimeError(
                                    "The mapping worker stopped before returning data."
                                )
                            eta.complete_case()
                            staged_results.append(
                                _NamedMappingResult(
                                    case.key,
                                    case.name,
                                    case.export_stem,
                                    result,
                                    case.short_name,
                                    case.short_export_stem,
                                )
                            )
                        check_cancelled()

                        requested_output = str(preview_output.value or "")
                        output_options = _common_result_preview_outputs(
                            tuple(item.result for item in staged_results)
                        )
                        if not output_options:
                            raise RuntimeError("Mapped datasets returned no common preview output.")
                        retained_keys = {item.key for item in staged_results}
                        for preference_key in tuple(mapping_raw_preferences):
                            if preference_key[0] not in retained_keys:
                                mapping_raw_preferences.pop(preference_key, None)
                        for color_key in tuple(mapping_batch_colors):
                            if color_key[0] not in retained_keys:
                                mapping_batch_colors.pop(color_key, None)
                        latest_results[:] = staged_results
                        prepared_result_candidates()
                        active_dataset_key = staged_results[0].key
                        dataset_page["value"] = 1
                        latest_result = staged_results[0].result
                        result = latest_result
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
                        sync_batch_presentation()
                        render_dataset_selector.refresh()
                        render_batch_raw_files.refresh()
                        apply_preview_output()

                        primary = np.asarray(result.outputs[result.primary_output])
                        mapping_stats["Form"].text = "Per field" if result.target == "kinetic_fields" else _DEPENDENCE_LABELS.get(
                            result.output_dependence,
                            result.output_dependence.replace("_", " ").title(),
                        )
                        mapping_stats["Points"].text = f"{primary.size:,}"
                        mapping_stats["Method"].text = route.method.replace("_", " ").title()
                        for value in mapping_stats.values():
                            value.update()
                        stats_container.visible = True
                        progress.value = 1.0
                        total_elapsed = sum(item.result.elapsed_seconds for item in latest_results)
                        status.text = (
                            f"Complete · {len(latest_results)} datasets in {total_elapsed:.2f} s"
                            if len(latest_results) > 1
                            else f"Complete in {result.elapsed_seconds:.2f} s"
                        )
                        if len(latest_results) == 1 and result.fit_mse is not None:
                            status.text += f" · MSE {result.fit_mse:.3e}"
                        masked = result.diagnostics.get("unidentifiable_count", 0)
                        if masked:
                            action = "filled with zero" if result.diagnostics.get("invalid_policy") == "zero" else "masked"
                            status.text += f" · {masked:,} unidentifiable values {action}"
                        if result.diagnostics.get("converged") is False:
                            status.text += " · Fit budget reached; inspect residuals"
                        if result.diagnostics.get("unreplicated_at_end", 0) > .01:
                            status.text += " · Time window ends before replication completes"
                        if result.inter_origin_distances_truncated:
                            total = result.inter_origin_distance_count
                            retained = int(primary.size)
                            count_text = f" of {total:,}" if total is not None else ""
                            status.text += (
                                f" · showing the first {retained:,}{count_text} distances"
                            )
                            ui.notify(
                                "The inter-origin-distance distribution exceeded the "
                                f"local retained-data limit; showing the first {retained:,} "
                                "values. Narrow the region or reduce the ensemble for a "
                                "complete distribution.",
                                type="warning",
                            )
                        histogram_bin_sync.sync_to_data(redraw=False)
                        populate_automatic_y_limits(redraw=False)
                        render_latest_result()
                        for named in latest_results:
                            named_primary = np.asarray(
                                named.result.outputs[named.result.primary_output]
                            )
                            cache_workflow(
                                named.result,
                                name=_mapping_display_name(named),
                                source="Mapping",
                                detail=(
                                    f"{mapping_stats['Form'].text} | "
                                    f"{named_primary.size:,} points | "
                                    f"{mapping_stats['Method'].text}"
                                ),
                                navigate_to="mapping",
                            )
                        eta.finish("complete")
                    except (CancelledError, asyncio.CancelledError):
                        cancel_event.set()
                        eta.finish("stopped")
                        progress.visible = False
                        status.text = "Mapping stopped"
                        if latest_results:
                            stats_container.visible = True
                            render_latest_result()
                        else:
                            stats_container.visible = False
                            latest_result = None
                            result_slot.clear()
                            with result_slot:
                                preview_plot(
                                    "Mapping stopped",
                                    kind="mapping",
                                )
                    except Exception as exc:
                        eta.finish("unavailable")
                        progress.visible = False
                        status.text = "Mapping could not be completed"
                        ui.notify(str(exc) or type(exc).__name__, type="negative")
                        if latest_results:
                            stats_container.visible = True
                            render_latest_result()
                        else:
                            stats_container.visible = False
                            latest_result = None
                            result_slot.clear()
                            with result_slot:
                                preview_plot(
                                    "No result",
                                    kind="mapping",
                                )
                    finally:
                        eta_label.text = eta.text()
                        if eta_timer is not None:
                            eta_timer.cancel()
                            eta_timer = None
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
                        preview_y_label.value = "MSE"
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
                    selected_named = next(
                        (
                            item
                            for item in latest_results
                            if item.key == active_dataset_key
                        ),
                        None,
                    )
                    next_automatic_name = (
                        _mapping_raw_default_filename(selected_named, output_key)
                        if selected_named is not None
                        else (
                            f"{RMAP_FILE_PREFIX}_"
                            f"{output_file_token(comparison_source_key)}-cmp"
                            if comparison_source_key is not None
                            else f"{RMAP_FILE_PREFIX}_"
                            f"{output_file_token(actual_output_key)}"
                        )
                    )
                    set_single_raw_automatic_name(next_automatic_name)
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
                    # A histogram or boxplot cannot represent several datasets
                    # as one overlaid line chart. Reconcile the layout before
                    # the redraw below so the plotter never sees an unsupported
                    # Combined/view pairing.
                    sync_batch_presentation()
                    histogram_bin_sync.sync_to_data(redraw=False)
                    refresh_axis_range()

                def refresh_preview_output() -> None:
                    apply_preview_output()
                    sync_batch_presentation()
                    render_batch_raw_files.refresh()
                    histogram_bin_sync.sync_to_data(redraw=False)
                    populate_automatic_y_limits(redraw=False)
                    render_latest_result()

                def refresh_preview_layout() -> None:
                    sync_batch_presentation()
                    sync_plot_controls()
                    render_dataset_selector.refresh()
                    histogram_bin_sync.sync_to_data(redraw=False)
                    populate_automatic_y_limits(redraw=False)
                    render_latest_result()

                def histogram_values() -> Any | None:
                    if latest_result is None:
                        return None
                    output_key = str(preview_output.value or latest_result.primary_output)
                    if output_key.startswith(_COMPARISON_PREFIX):
                        return None
                    if (
                        len(latest_results) > 1
                        and str(preview_layout.value or "Combined") == "Combined"
                        and resolved_plot_type() == "Histogram"
                    ):
                        values = tuple(
                            np.asarray(
                                item.result.outputs[
                                    _resolved_output_key(item.result, output_key)
                                ],
                                dtype=float,
                            ).reshape(-1)
                            for item in latest_results
                        )
                        return np.concatenate(values) if values else None
                    actual_output_key = _resolved_output_key(latest_result, output_key)
                    return latest_result.outputs.get(actual_output_key)

                histogram_bin_sync = HistogramBinSynchronizer(
                    plot_settings,
                    values=histogram_values,
                    on_valid_change=refresh_axis_range,
                    log_x=lambda: bool(log_x.value),
                )

                def log_x_changed() -> None:
                    histogram_bin_sync.sync_to_data(redraw=False)
                    render_latest_result()

                preview_output.on_value_change(refresh_preview_output)
                preview_layout.on_value_change(refresh_preview_layout)
                bind_preview_settings_updates(
                    plot_settings,
                    on_redraw=render_latest_result,
                    on_view_change=sync_view_axis_labels,
                    on_density_change=sync_view_axis_labels,
                    on_histogram_range_change=refresh_axis_range,
                    on_log_x_change=log_x_changed,
                    on_y_limits_change=use_custom_y_limits,
                    on_visibility_change=sync_plot_controls,
                )
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
                    histogram_bin_sync.sync_to_data(redraw=False)
                    restored_primary = np.asarray(
                        latest_result.outputs[latest_result.primary_output]
                    )
                    mapping_stats["Form"].text = "Per field" if latest_result.target == "kinetic_fields" else _DEPENDENCE_LABELS.get(
                        latest_result.output_dependence,
                        latest_result.output_dependence.replace("_", " ").title(),
                    )
                    mapping_stats["Points"].text = f"{restored_primary.size:,}"
                    mapping_stats["Method"].text = latest_result.method.replace("_", " ").title()
                    for value in mapping_stats.values():
                        value.update()
                    stats_container.visible = True
                    status.text = "Restored from Cached Files"
                    sync_batch_presentation()
                    apply_preview_output()
                    populate_automatic_y_limits(redraw=False)
                    render_latest_result()

                def clear_mapping_output() -> None:
                    nonlocal active_dataset_key, latest_result, result_chart
                    latest_results.clear()
                    prepared_bundle_results.clear()
                    for live in mapping_input_controls.values():
                        if live.surface.get("prepared_only"):
                            live.refresh_surface()
                    mapping_raw_preferences.clear()
                    mapping_batch_colors.clear()
                    active_dataset_key = ""
                    dataset_page["value"] = 1
                    latest_result = None
                    result_chart = None
                    progress.visible = False
                    stats_container.visible = False
                    status.text = "Ready"
                    eta_label.visible = False
                    result_slot.clear()
                    layout_stat.visible = False
                    batch_color_container.visible = False
                    render_dataset_selector.refresh()
                    render_batch_colors.refresh()
                    render_batch_raw_files.refresh()
                    with result_slot:
                        preview_plot("No result yet", kind="mapping")

                current_session_cache().register_output_clearer(
                    "mapping",
                    clear_mapping_output,
                )

                def refresh_mapping_page() -> None:
                    """Refresh style-dependent labels without rerunning Mapping."""

                    render_dataset_selector.refresh()
                    if batch_color_container.visible:
                        render_batch_colors.refresh()
                    render_batch_raw_files.refresh()
                    if latest_result is None:
                        return
                    active_named = next(
                        (
                            item
                            for item in latest_results
                            if item.key == active_dataset_key
                        ),
                        None,
                    )
                    if active_named is not None:
                        set_single_raw_automatic_name(
                            _mapping_raw_default_filename(
                                active_named,
                                str(
                                    preview_output.value
                                    or active_named.result.primary_output
                                ),
                            )
                        )
                    render_latest_result()

                current_session_cache().register_page_refresher(
                    "mapping",
                    refresh_mapping_page,
                )
