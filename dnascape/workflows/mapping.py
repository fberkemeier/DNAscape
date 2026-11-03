"""Bounded, framework-independent execution bridge for :data:`dnascape.rmap`.

The NiceGUI page materialises uploaded data before constructing a
:class:`MappingRequest`.  This module then validates the exact catalogue route,
normalises its scientific inputs, runs the mapper in a worker thread, and
returns immutable arrays with explicit plotting axes.  No NiceGUI objects are
used here, which keeps the expensive work straightforward to test and safe to
run through ``run.io_bound``.
"""

from __future__ import annotations

import gzip
import io
import math
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from types import MappingProxyType
from typing import Any, Protocol, cast

import numpy as np

from dnascape.workflows.mapping_catalog import (
    DATA_FORMS,
    DataForm,
    MappingInput,
    MappingRoute,
    MappingSetting,
    mapping_route,
    mapping_target,
)
from dnascape.workflows.spatial_grid import resample_spatial, spatial_bin_count
from dnascape.workflows.simulation_filters import SimulationFilterSelection
from dnascape.workflows.mapping_filter import (
    MappingFilterSettings, MappingFitConstraint, prepare_filter_request,
    filtered_speed_settings,
)

ProgressReporter = Callable[[float, str], None]
CancelChecker = Callable[[], None]

# Interactive limits are intentionally independent of the scientific package.
# They prevent a malformed upload or a mistyped grid/iteration count from
# exhausting the browser process while remaining generous for local analyses.
MAX_INPUT_VALUES = 64_000_000
MAX_INPUT_BYTES = 512 * 1024 * 1024
MAX_OUTPUT_VALUES = 128_000_000
MAX_OUTPUT_BYTES = 1024 * 1024 * 1024
MAX_SPATIAL_POINTS = 1_000_000
MAX_TIME_POINTS = 100_000
MAX_GRID_CELLS = 64_000_000
MAX_MAPPING_SIMULATIONS = 50_000
MAX_MAPPING_IOD_SAMPLES = 2_000_000
MAX_TRACK_UPLOAD_BYTES = 32 * 1024 * 1024
MAX_TRACK_PAYLOAD_BYTES = 64 * 1024 * 1024
MAX_TRACK_FILES = 128

_RUN_LOCK = Lock()
_AXIS_OUTPUT_KEYS = {"time_grid", "time_axis", "x_array", "position_kb"}
_FIT_OUTPUT_KEYS = {
    "fit_history",
    "replication_timing_fit",
    "residual",
}
_DIAGNOSTIC_KEYS = {
    'filter_intervals',
    'filter_warnings',
    "valid_mask", "unidentifiable_count", "invalid_policy", "converged",
    "convergence_message", "optimizer_evaluations", "unreplicated_at_end",
    "identifiability", "forward_model", "firing_form", "speed_form",
    "regularization", "speed_anchor", "fit_objective", "fit_mse", "fit_used_descent",
    "fit_edge_bins", "fit_taper_bins", "fit_periodic_smoothing_kb",
    "fork_speed_time_origin", "fork_speed_time_scale",
    "terminal_survival", "extrapolated_tail", "tail_policy", "initial_policy",
    "timing_is_restricted_mean", "clipped_derivative_count", "derivative_method",
    "monotonic_policy", "temporal_smoothw", "density_floor", "unreplicated_floor",
    "forward_backend", "fitting_backend", "phase_history", "space_iterations", "time_iterations",
    "iterations", "iteration_budget", "objective_evaluations", "observation",
    "weibull_reference", "weibull_baseline", "warnings", "finite_influence_approximation",
    "weibull_shape", "weibull_delay", "alpha", "initiation_rate_units",
    "restricted_mean_timing",
    "weibull_onset_cutoff_min",
    "coarse_fit_history", "coarse_resolution", "native_resolution", "adaptive_fitting",
}


class TrackUpload(Protocol):
    """Structural subset of an inspected genomic-track upload."""

    @property
    def name(self) -> str: ...

    @property
    def content(self) -> bytes: ...

    @property
    def source_format(self) -> str: ...

    @property
    def columns(self) -> tuple[str, ...]: ...


@dataclass(frozen=True, slots=True)
class MappingRequest:
    """One resolved and fully materialised mapping operation.

    ``inputs`` may use either the exact route input name or its family name
    (for example ``initiation_events`` for an exact
    ``initiation_events_xt`` input).  Optional defaulted feature inputs are
    omitted unless their override is enabled in the UI.

    Set ``source_resolution_x`` to resample native uploads onto the requested
    spatial grid (ordinary app uploads use 1 kb). ``initiation_rate_per_bin`` opts
    into the app/Simulation rate convention; the default retains public rmap's
    rate density per kb for existing Python callers.
    """

    route: MappingRoute
    inputs: Mapping[str, Any]
    input_forms: Mapping[str, DataForm]
    settings: Mapping[str, Any] = field(default_factory=dict)
    target_shape: str | None = None
    region_start_kb: float | None = None
    region_end_kb: float | None = None
    resolution_x: float | None = None
    resolution_t: float | None = None
    whole_genome: bool = False
    prewindowed_inputs: frozenset[str] = frozenset()
    # None preserves the public workflow's historical already-binned contract.
    # The app supplies the native upload spacing, independently of resolution_x.
    source_resolution_x: float | None = None
    initiation_rate_per_bin: bool = False
    fit_filter: MappingFilterSettings = field(default_factory=MappingFilterSettings)


@dataclass(frozen=True, slots=True)
class MappingComparison:
    """Observed input and fitted profile returned by one rmap fit."""

    source_key: str
    fitted_key: str
    form: DataForm
    observed: np.ndarray
    fitted: np.ndarray


@dataclass(frozen=True, slots=True)
class MappingResult:
    """Immutable mapped outputs and their browser-facing scientific axes."""

    target: str
    target_key: str
    target_label: str
    method: str
    output_form: DataForm
    output_dependence: str
    primary_output: str
    outputs: Mapping[str, np.ndarray]
    output_forms: Mapping[str, DataForm]
    x_axis: np.ndarray
    time_axis: np.ndarray
    region_labels: tuple[str, ...]
    elapsed_seconds: float
    assumptions: Mapping[str, Any]
    comparisons: Mapping[str, MappingComparison] = field(default_factory=dict)
    inter_origin_distance_count: int | None = None
    inter_origin_distances_truncated: bool = False
    fit_mse: float | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    input_axes: Mapping[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    filter_selection: SimulationFilterSelection | None = None


@dataclass(frozen=True, slots=True)
class _PreparedCall:
    route: MappingRoute
    source: dict[str, Any]
    assumptions: dict[str, Any]
    start_kb: float
    resolution_x: float
    resolution_t: float
    spatial_size: int
    input_forms: Mapping[str, DataForm]
    initiation_rate_per_bin: bool = False
    end_kb: float | None = None
    fit_filter: MappingFilterSettings = field(default_factory=MappingFilterSettings)


@dataclass(frozen=True, slots=True)
class _SpatialSelection:
    """One zero-origin genomic window applied to uploaded spatial arrays."""

    start_kb: float
    end_kb: float
    start_index: int
    stop_index: int
    source_bins: int


def _friendly_input_key(key: str) -> str:
    return key[:-3] if key.endswith("_xt") else key


def _finite_number(value: Any, label: str, *, positive: bool = False) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric.") from exc
    if not np.isfinite(number) or (positive and number <= 0):
        qualifier = "finite and greater than zero" if positive else "finite"
        raise ValueError(f"{label} must be {qualifier}.")
    return number


def _readonly(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


def _numeric_array(value: Any, label: str) -> np.ndarray:
    if isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{label} must contain parsed numeric data, not text or bytes.")
    if np.iscomplexobj(value):
        raise ValueError(f"{label} must contain real numeric values.")
    try:
        raw_array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain numeric values.") from exc
    estimated_bytes = max(int(raw_array.nbytes), int(raw_array.size) * np.dtype(float).itemsize)
    if raw_array.size > MAX_INPUT_VALUES or estimated_bytes > MAX_INPUT_BYTES:
        raise ValueError(
            f"{label} contains {raw_array.size:,} values / "
            f"{estimated_bytes / 1024**2:.1f} MiB, which exceeds the local input limit."
        )
    try:
        # Upload inspection already owns immutable float64 arrays. Reuse those
        # directly; copying them here and again while applying X/T shapes made
        # input preparation scale with the same data two or three times.
        if (
            raw_array.dtype == np.dtype(float)
            and raw_array.flags.c_contiguous
            and not raw_array.flags.writeable
        ):
            array = raw_array
        else:
            array = np.array(raw_array, dtype=float, copy=True, order="C")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain numeric values.") from exc
    if array.size == 0:
        raise ValueError(f"{label} must not be empty.")
    if array.ndim > 2:
        raise ValueError(f"{label} must be a scalar, vector, or two-dimensional matrix.")
    if np.any(~np.isfinite(array)):
        raise ValueError(
            f"{label} contains missing or infinite values. Resolve them in the import controls."
        )
    return _readonly(array)


def _preflight_input_payload(
    values: Mapping[str, Any],
    cancel: CancelChecker | None = None,
) -> None:
    total_values = 0
    total_bytes = 0
    check_cancelled = cancel or (lambda: None)

    def visit(value: Any) -> None:
        nonlocal total_values, total_bytes
        check_cancelled()
        if isinstance(value, Mapping):
            if len(value) > 64:
                raise ValueError("An uploaded result bundle contains too many members.")
            for member in value.values():
                visit(member)
            return
        if isinstance(value, (str, bytes, bytearray)):
            return
        try:
            array = np.asarray(value)
        except (TypeError, ValueError):
            return
        total_values += int(array.size)
        total_bytes += max(
            int(array.nbytes),
            int(array.size) * np.dtype(float).itemsize,
        )

    for value in values.values():
        visit(value)
    if total_values > MAX_INPUT_VALUES or total_bytes > MAX_INPUT_BYTES:
        raise ValueError(
            f"Mapped inputs contain {total_values:,} values / {total_bytes / 1024**2:.1f} MiB; "
            f"the interactive limits are {MAX_INPUT_VALUES:,} values and "
            f"{MAX_INPUT_BYTES / 1024**2:.0f} MiB."
        )


def _normalise_regions(value: Any, label: str) -> Any:
    if isinstance(value, Mapping):
        if not value:
            raise ValueError(f"{label} must contain at least one region.")
        if len(value) > MAX_SPATIAL_POINTS:
            raise ValueError(f"{label} contains too many regions for an interactive run.")
        regions: dict[str, tuple[float, float]] = {}
        for raw_name, raw_bounds in value.items():
            bounds = _numeric_array(raw_bounds, f"{label} region {raw_name!r}").reshape(-1)
            if bounds.size != 2 or bounds[1] <= bounds[0]:
                raise ValueError(f"{label} region {raw_name!r} must be a (start, end) pair.")
            regions[str(raw_name)] = (float(bounds[0]), float(bounds[1]))
        return regions
    array = _numeric_array(value, label)
    if array.ndim == 1:
        if array.size != 2:
            raise ValueError(f"{label} must contain (start, end) coordinate pairs.")
        array = array.reshape(1, 2)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"{label} must contain (start, end) coordinate pairs.")
    if np.any(array[:, 1] <= array[:, 0]):
        raise ValueError(f"Every {label.lower()} region must have positive width.")
    return _readonly(array)


def _normalise_bundle(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{label} must be a prepared DNAscape result bundle.")
    if len(value) > 64:
        raise ValueError(f"{label} contains too many result members.")
    bundle: dict[str, Any] = {}
    for raw_key, member in value.items():
        key = str(raw_key)
        if isinstance(member, Mapping):
            bundle[key] = _normalise_bundle(member, f"{label}.{key}")
        elif member is not None:
            bundle[key] = _numeric_array(member, f"{label}.{key}")
    if not bundle:
        raise ValueError(f"{label} contains no numeric result members.")
    return bundle


def _normalise_input(value: Any, input_spec: MappingInput, form: DataForm) -> Any:
    label = input_spec.label
    if form == "regions":
        return _normalise_regions(value, label)
    if form == "bundle":
        return _normalise_bundle(value, label)
    if form == "supplied" and isinstance(value, Mapping):
        return _normalise_bundle(value, label)

    array = _numeric_array(value, label)
    if form == "scalar":
        if array.size != 1:
            raise ValueError(f"{label} must contain one constant value.")
        return float(array.reshape(-1)[0])
    if form == "x":
        squeezed = np.squeeze(array)
        if squeezed.ndim == 0 and array.size == 1:
            squeezed = squeezed.reshape(1)
        if squeezed.ndim != 1:
            raise ValueError(f"{label} must be a one-dimensional spatial profile.")
        return _readonly(np.ascontiguousarray(squeezed, dtype=float))
    if form == "t":
        squeezed = np.squeeze(array)
        if squeezed.ndim != 1:
            raise ValueError(f"{label} must be a one-dimensional temporal profile.")
        return _readonly(np.ascontiguousarray(squeezed, dtype=float).reshape(1, -1))
    if form == "xt":
        if array.ndim != 2 or 1 in array.shape:
            raise ValueError(f"{label} must be a position-by-time matrix.")
        return array
    return array


def _input_size(value: Any) -> tuple[int, int]:
    if isinstance(value, Mapping):
        sizes = tuple(_input_size(item) for item in value.values())
        return sum(item[0] for item in sizes), sum(item[1] for item in sizes)
    array = np.asarray(value)
    return int(array.size), int(array.nbytes)


def _validate_input_shapes(
    route: MappingRoute,
    inputs: Mapping[str, Any],
    forms: Mapping[str, DataForm],
) -> tuple[int, int]:
    spatial_lengths: set[int] = set()
    time_lengths: set[int] = set()

    def add_bundle_shapes(value: Any, key: str = "") -> None:
        if isinstance(value, Mapping):
            for member_key, member in value.items():
                add_bundle_shapes(member, str(member_key))
            return
        array = np.asarray(value)
        if key in {"time_grid", "time_axis"}:
            time_lengths.add(int(array.size))
        elif key == "x_array":
            spatial_lengths.add(int(array.size))
        elif array.ndim == 2 and (key.endswith("_xt") or 1 not in array.shape):
            spatial_lengths.add(int(array.shape[0]))
            time_lengths.add(int(array.shape[1]))

    for input_spec in route.inputs:
        if input_spec.key not in inputs:
            continue
        form = forms[input_spec.key]
        if form == "x":
            spatial_lengths.add(int(np.asarray(inputs[input_spec.key]).size))
        elif form == "t":
            time_lengths.add(int(np.asarray(inputs[input_spec.key]).size))
        elif form == "xt":
            shape = np.asarray(inputs[input_spec.key]).shape
            spatial_lengths.add(int(shape[0]))
            time_lengths.add(int(shape[1]))
        elif form == "bundle":
            add_bundle_shapes(inputs[input_spec.key])
    if len(spatial_lengths) > 1:
        lengths = ", ".join(f"{value:,}" for value in sorted(spatial_lengths))
        raise ValueError(f"Spatial inputs must use the same number of bins; received {lengths}.")
    if len(time_lengths) > 1:
        lengths = ", ".join(f"{value:,}" for value in sorted(time_lengths))
        raise ValueError(f"Temporal inputs must use the same number of points; received {lengths}.")
    spatial = next(iter(spatial_lengths), 0)
    temporal = next(iter(time_lengths), 0)
    if spatial > MAX_SPATIAL_POINTS or temporal > MAX_TIME_POINTS:
        raise ValueError("An input axis exceeds the local interactive grid limit.")
    if not route.unified_kinetics and spatial and temporal and spatial * temporal > MAX_GRID_CELLS:
        raise ValueError(
            "The uploaded space-time grid is too large for an interactive mapping run."
        )
    return spatial, temporal


def _find_supplied_key(
    route: MappingRoute,
    supplied_key: str,
) -> str:
    exact = {item.key for item in route.inputs}
    if supplied_key in exact:
        return supplied_key
    matches = tuple(
        item.key for item in route.inputs if _friendly_input_key(item.key) == supplied_key
    )
    if len(matches) == 1:
        return matches[0]
    raise ValueError(f"Input {supplied_key!r} is not accepted by {route.label!r}.")


def _request_resolution_x(request: MappingRequest, route: MappingRoute) -> float:
    # Prepared results carry intrinsic coordinates. The previous page's
    # resolution control must not rescale a retained scientific result.
    value = None
    for supplied in (() if request.source_resolution_x is not None else request.inputs.values()):
        if isinstance(supplied, Mapping) and "x_array" in supplied:
            axis = np.asarray(supplied["x_array"], dtype=float).reshape(-1)
            if axis.size > 1:
                value = float(axis[1] - axis[0])
                break
    if value is None:
        value = request.resolution_x
    if value is None:
        value = request.settings.get("resolution")
    if value is None:
        value = route.setting_defaults_dict.get("resolution", 1.0)
    return _finite_number(value, "Spatial resolution", positive=True)


def _bundle_spatial_bins(value: Any) -> int:
    """Infer the leading spatial axis of a prepared numeric bundle."""

    matrix_lengths: set[int] = set()
    vector_lengths: set[int] = set()

    def visit(member: Any, key: str = "") -> None:
        if isinstance(member, Mapping):
            for raw_key, nested in member.items():
                visit(nested, str(raw_key))
            return
        if isinstance(member, (str, bytes, bytearray)):
            return
        try:
            array = np.asarray(member)
        except (TypeError, ValueError):
            return
        if array.ndim >= 2 and (key.endswith("_xt") or array.shape[0] > 1):
            matrix_lengths.add(int(array.shape[0]))
        elif (
            array.ndim == 1
            and array.size > 1
            and key not in {"time_grid", "time_axis", "fit_history"}
        ):
            vector_lengths.add(int(array.size))

    visit(value)
    candidates = matrix_lengths or vector_lengths
    return max(candidates, default=0)


def _raw_spatial_bins(value: Any, form: DataForm) -> int:
    if form == "x":
        try:
            array = np.squeeze(np.asarray(value))
        except (TypeError, ValueError):
            return 0
        return int(array.size) if array.ndim <= 1 else 0
    if form == "xt":
        try:
            array = np.asarray(value)
        except (TypeError, ValueError):
            return 0
        return int(array.shape[0]) if array.ndim == 2 else 0
    if form in {"bundle", "supplied"} and isinstance(value, Mapping):
        return _bundle_spatial_bins(value)
    return 0


def _spatial_selection(
    request: MappingRequest,
    *,
    source_bins: int,
    resolution_x: float,
    source_label: str = "the longest uploaded spatial input",
    source_start_kb: float = 0.0,
) -> _SpatialSelection:
    source_end = source_start_kb + source_bins * resolution_x
    if request.whole_genome:
        return _SpatialSelection(source_start_kb, source_end, 0, source_bins, source_bins)

    start = (
        source_start_kb
        if request.region_start_kb is None
        else _finite_number(request.region_start_kb, "Region start")
    )
    end = (
        source_end
        if request.region_end_kb is None
        else _finite_number(request.region_end_kb, "Region end")
    )
    if start < 0:
        raise ValueError(f"Region start must be at least 0 kb; received {start:g} kb.")
    if start < source_start_kb:
        raise ValueError(
            f"Region start must be at least {source_start_kb:g} kb for the prepared result."
        )
    if end <= start:
        raise ValueError(
            f"Region end must be greater than region start; received Start {start:g} kb "
            f"and End {end:g} kb."
        )

    start_index = max(0, int(math.floor((start - source_start_kb) / resolution_x + 1e-12)))
    stop_index = min(source_bins, int(math.ceil((end - source_start_kb) / resolution_x - 1e-12)))
    if start_index >= source_bins or stop_index <= start_index:
        raise ValueError(
            f"Region start {start:g} kb is outside {source_label} "
            f"(0-{source_end:g} kb at {resolution_x:g} kb/bin). Choose a Start below "
            f"{source_end:g} kb."
        )
    return _SpatialSelection(
        start_kb=source_start_kb + start_index * resolution_x,
        end_kb=source_start_kb + stop_index * resolution_x,
        start_index=start_index,
        stop_index=stop_index,
        source_bins=source_bins,
    )


def _slice_bundle_spatial(
    value: Any,
    selection: _SpatialSelection,
    key: str = "",
) -> Any:
    if key in {"time_grid", "time_axis", "fit_history"}:
        return value
    if isinstance(value, Mapping):
        return {
            str(member_key): _slice_bundle_spatial(member, selection, str(member_key))
            for member_key, member in value.items()
        }
    if isinstance(value, (str, bytes, bytearray)):
        return value
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return value
    if array.ndim == 1 and array.size == selection.source_bins:
        return array[selection.start_index : selection.stop_index]
    if array.ndim >= 2 and array.shape[0] == selection.source_bins:
        return array[selection.start_index : selection.stop_index, ...]
    return value


def _slice_spatial_value(
    value: Any,
    form: DataForm,
    selection: _SpatialSelection,
) -> Any:
    if form == "x":
        array = np.squeeze(np.asarray(value))
        if array.ndim == 1:
            return array[selection.start_index : selection.stop_index]
    elif form == "xt":
        array = np.asarray(value)
        if array.ndim == 2:
            return array[selection.start_index : selection.stop_index, :]
    elif form in {"bundle", "supplied"} and isinstance(value, Mapping):
        return _slice_bundle_spatial(value, selection)
    return value


def _resample_spatial_inputs(
    request: MappingRequest,
    route: MappingRoute,
    values: Mapping[str, Any],
    forms: Mapping[str, DataForm],
    *,
    cancel: CancelChecker,
    prewindowed_inputs: frozenset[str],
) -> tuple[dict[str, Any], _SpatialSelection | None]:
    """Keep native upload coordinates separate from the requested fit grid."""

    resolution = _request_resolution_x(request, route)
    native_resolution = _finite_number(
        request.source_resolution_x, "Source resolution", positive=True,
    )
    spatial: list[tuple[str, Any, DataForm, int, float, float, bool]] = []
    for key, value in values.items():
        cancel()
        form = forms[key]
        bins = _raw_spatial_bins(value, form)
        if not bins:
            continue
        prewindowed = key in prewindowed_inputs
        origin = 0.0
        step = resolution if prewindowed else native_resolution
        if isinstance(value, Mapping) and "x_array" in value:
            axis = np.asarray(value["x_array"], dtype=float)
            if axis.ndim != 1 or axis.size != bins or not np.all(np.isfinite(axis)):
                raise ValueError(
                    "The prepared result's position axis does not match its spatial fields."
                )
            origin = float(axis[0])
            if axis.size > 1:
                step = float(axis[1] - axis[0])
                if step <= 0 or not np.allclose(np.diff(axis), step):
                    raise ValueError(
                        "The prepared result must have a regular, increasing position axis."
                    )
        spatial.append((key, value, form, bins, step, origin, prewindowed))
    if not spatial:
        return dict(values), None
    reference = max(
        (item for item in spatial if not item[6]),
        key=lambda item: item[3] * item[4], default=spatial[0],
    )
    if request.whole_genome:
        start = max(item[5] for item in spatial)
        end = min(item[5] + item[3] * item[4] for item in spatial)
    else:
        start = (
            reference[5] if request.region_start_kb is None
            else _finite_number(request.region_start_kb, "Region start")
        )
        end = (
            reference[5] + reference[3] * reference[4] if request.region_end_kb is None
            else _finite_number(request.region_end_kb, "Region end")
        )
        ordinary = tuple(item for item in spatial if not item[6])
        if ordinary and len({item[3] * item[4] for item in ordinary}) == 1:
            # Preserve the existing default-End behaviour for short uploads:
            # data cannot extend beyond its native extent. Mixed local/genomic
            # profiles must still cover the complete explicitly selected width.
            end = min(end, reference[5] + reference[3] * reference[4])
    count = spatial_bin_count(start, end, resolution)
    if count > MAX_SPATIAL_POINTS:
        raise ValueError(
            "The requested spatial grid exceeds the local limit. "
            "Increase the resolution or narrow the region."
        )
    result = dict(values)

    def rebin(value: Any, *, bins: int, step: float, origin: float, key: str = "") -> Any:
        cancel()
        if key in {"time_grid", "time_axis", "fit_history"}:
            return value
        if isinstance(value, Mapping):
            return {
                str(k): rebin(v, bins=bins, step=step, origin=origin, key=str(k))
                for k, v in value.items()
            }
        if isinstance(value, (str, bytes, bytearray)):
            return value
        array = np.asarray(value)
        if key in {"x_array", "position_kb"} and array.ndim == 1 and array.size == bins:
            return _readonly(start + np.arange(count, dtype=float) * resolution)
        if array.ndim in (1, 2) and array.shape[0] == bins:
            return resample_spatial(
                array, source_resolution=step, source_start=origin,
                start=start, end=end, resolution=resolution,
                max_points=MAX_SPATIAL_POINTS, max_values=MAX_INPUT_VALUES,
            )
        return value

    for key, value, form, bins, step, origin, prewindowed in spatial:
        # Shorter ordinary uploads retain the established local-profile
        # convention: their first value is at the selected Start.
        is_local = bins * step < reference[3] * reference[4] and not isinstance(value, Mapping)
        if not request.whole_genome and (prewindowed or is_local):
            origin = start
        if form == "x":
            result[key] = rebin(np.asarray(value).reshape(-1), bins=bins, step=step, origin=origin)
        else:
            result[key] = rebin(value, bins=bins, step=step, origin=origin)
    return result, _SpatialSelection(start, end, 0, count, reference[3])


def _crop_spatial_inputs(
    request: MappingRequest,
    route: MappingRoute,
    values: Mapping[str, Any],
    forms: Mapping[str, DataForm],
    *,
    cancel: CancelChecker,
    prewindowed_inputs: frozenset[str] = frozenset(),
) -> tuple[dict[str, Any], _SpatialSelection | None]:
    """Align spatial inputs to one interval before validation and rmap.

    The longest ordinary upload defines genomic coordinates. Shorter profiles
    begin at the selected Start and must cover the complete selected width.
    Whole region uses the first bins of every input, from zero to the shortest
    profile's length. Mixed-length inputs never silently shorten a window.
    """

    if request.source_resolution_x is not None:
        return _resample_spatial_inputs(
            request, route, values, forms, cancel=cancel,
            prewindowed_inputs=prewindowed_inputs,
        )
    resolution_x = _request_resolution_x(request, route)
    labels = {item.key: item.label for item in route.inputs}
    spatial_inputs: list[tuple[str, Any, DataForm, int, bool]] = []
    source_origins: dict[str, float] = {}
    for key, value in values.items():
        cancel()
        form = forms.get(key)
        if form is None:
            continue
        source_bins = _raw_spatial_bins(value, form)
        if source_bins:
            if isinstance(value, Mapping) and "x_array" in value:
                axis = np.asarray(value["x_array"], dtype=float)
                if axis.ndim != 1 or axis.size != source_bins or not np.all(np.isfinite(axis)):
                    raise ValueError(
                        "The prepared result's position axis does not match its spatial fields."
                    )
                if axis.size > 1 and not np.allclose(np.diff(axis), resolution_x):
                    raise ValueError(
                        "Spatial resolution must match the prepared result's position axis."
                    )
                source_origins[key] = float(axis[0])
            spatial_inputs.append((key, value, form, source_bins, key in prewindowed_inputs))

    if not spatial_inputs:
        cancel()
        return dict(values), None

    coordinate_inputs = tuple(item for item in spatial_inputs if not item[4])
    shortest_bins = min(item[3] for item in spatial_inputs)
    mixed_lengths = len({item[3] for item in spatial_inputs}) > 1
    reference_bins = max(item[3] for item in coordinate_inputs or spatial_inputs)
    if request.whole_genome:
        selected_bins = shortest_bins
        selection = _SpatialSelection(
            start_kb=0., end_kb=shortest_bins * resolution_x,
            start_index=0, stop_index=shortest_bins, source_bins=reference_bins,
        )
    elif coordinate_inputs:
        reference_bins = max(item[3] for item in coordinate_inputs)
        reference_key = next(item[0] for item in coordinate_inputs if item[3] == reference_bins)
        reference_label = labels.get(reference_key, _friendly_input_key(reference_key))
        requested = _spatial_selection(
            request,
            source_bins=reference_bins,
            resolution_x=resolution_x,
            source_label=f"{reference_label}, the longest uploaded spatial input",
            source_start_kb=source_origins.get(reference_key, 0.0),
        )
        requested_bins = requested.stop_index - requested.start_index
        if mixed_lengths and request.region_end_kb is not None:
            # Validate the complete entered width before considering the
            # longest profile's end, rather than accepting a clipped window.
            end = _finite_number(request.region_end_kb, "Region end")
            reference_origin = source_origins.get(reference_key, 0.0)
            stop_index = int(math.ceil((end - reference_origin) / resolution_x - 1e-12))
            requested_bins = stop_index - requested.start_index
        selected_bins = requested_bins
        selection = _SpatialSelection(
            start_kb=requested.start_kb,
            end_kb=requested.start_kb + selected_bins * resolution_x,
            start_index=requested.start_index,
            stop_index=requested.start_index + selected_bins,
            source_bins=reference_bins,
        )
    else:
        reference_bins = max(item[3] for item in spatial_inputs)
        start = (
            0.0
            if request.whole_genome or request.region_start_kb is None
            else _finite_number(request.region_start_kb, "Region start")
        )
        if start < 0:
            raise ValueError(f"Region start must be at least 0 kb; received {start:g} kb.")
        if request.region_end_kb is None:
            requested_bins = shortest_bins
        else:
            end = _finite_number(request.region_end_kb, "Region end")
            if end <= start:
                raise ValueError(
                    f"Region end must be greater than region start; received Start "
                    f"{start:g} kb and End {end:g} kb."
                )
            requested_bins = int(math.ceil((end - start) / resolution_x))
        selected_bins = requested_bins
        selection = _SpatialSelection(
            start_kb=start,
            end_kb=start + selected_bins * resolution_x,
            start_index=0,
            stop_index=selected_bins,
            source_bins=reference_bins,
        )

    if selected_bins > shortest_bins:
        shortest_key = next(item[0] for item in spatial_inputs if item[3] == shortest_bins)
        shortest_label = labels.get(shortest_key, _friendly_input_key(shortest_key))
        raise ValueError(
            f"The selected region requires {selected_bins:,} bins "
            f"({selected_bins * resolution_x:g} kb), but {shortest_label}, the shortest "
            f"spatial profile, contains only {shortest_bins:,} bins "
            f"({shortest_bins * resolution_x:g} kb). Start and End are chromosome "
            f"coordinates; choose a width of at most {shortest_bins * resolution_x:g} kb "
            f"or select Whole region."
        )
    if mixed_lengths and not request.whole_genome and coordinate_inputs:
        source_start = source_origins.get(reference_key, 0.0)
        source_end = source_start + reference_bins * resolution_x
        if selection.end_kb > source_end:
            raise ValueError(
                f"Region end {selection.end_kb:g} kb is outside {reference_label}, "
                f"the longest uploaded spatial input "
                f"({source_start:g}-{source_end:g} kb at {resolution_x:g} kb/bin). "
                f"Choose an End at or below {source_end:g} kb."
            )

    # Shorter prepared profiles are local too. Rebase their position axes to
    # the selected chromosome start; retain their data and intrinsic bin width.
    relocated_inputs = {
        key for key, _value, _form, source_bins, prewindowed in spatial_inputs
        if request.whole_genome or (not prewindowed and source_bins < reference_bins)
    }
    coordinated_offsets: dict[str, int] = {}
    for key, _value, _form, source_bins, _prewindowed in spatial_inputs:
        if key not in source_origins or key in relocated_inputs:
            continue
        offset = (selection.start_kb - source_origins[key]) / resolution_x
        start_index = int(round(offset))
        if not np.isclose(offset, start_index) or not 0 <= start_index < source_bins:
            raise ValueError(
                "The selected region must start on the prepared result's position axis."
            )
        coordinated_offsets[key] = start_index
        if start_index + selected_bins > source_bins:
            raise ValueError("The selected region must end within the prepared result's position axis.")

    cropped = dict(values)
    for key, value, form, source_bins, prewindowed in spatial_inputs:
        cancel()
        if key in coordinated_offsets:
            start_index = coordinated_offsets[key]
            input_selection = _SpatialSelection(
                start_kb=selection.start_kb,
                end_kb=selection.end_kb,
                start_index=start_index,
                stop_index=start_index + selected_bins,
                source_bins=source_bins,
            )
        elif request.whole_genome or prewindowed or source_bins < reference_bins:
            input_selection = _SpatialSelection(
                start_kb=selection.start_kb,
                end_kb=selection.end_kb,
                start_index=0,
                stop_index=selected_bins,
                source_bins=source_bins,
            )
        else:
            input_selection = selection
        cropped[key] = _slice_spatial_value(value, form, input_selection)
        if key in relocated_inputs and key in source_origins:
            cropped[key]["x_array"] = (
                selection.start_kb + np.arange(selected_bins, dtype=float) * resolution_x
            )
    cancel()
    return cropped, selection


def _parse_number_sequence(value: Any, label: str) -> tuple[float, ...]:
    if isinstance(value, str):
        parts = tuple(part.strip() for part in value.replace(";", ",").split(","))
        value = tuple(part for part in parts if part)
    if isinstance(value, np.ndarray):
        value = value.reshape(-1).tolist()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{label} must be a comma-separated numeric sequence.")
    numbers = tuple(_finite_number(item, label) for item in value)
    if not numbers:
        raise ValueError(f"{label} must not be empty.")
    return numbers


def _normalise_setting(
    route: MappingRoute, key: str, value: Any, setting: MappingSetting | None = None
) -> Any:
    setting = route.setting(key) if setting is None else setting
    if value is None:
        if setting.nullable:
            return None
        raise ValueError(f"{setting.label} cannot be automatic or empty.")
    if setting.kind == "boolean":
        if not isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{setting.label} must be true or false.")
        return bool(value)
    if setting.kind == "choice":
        if value not in setting.allowed_values:
            choices = ", ".join(str(item) for item in setting.allowed_values)
            raise ValueError(f"{setting.label} must be one of: {choices}.")
        return value
    if setting.kind == "integer":
        number = _finite_number(value, setting.label)
        if not number.is_integer():
            raise ValueError(f"{setting.label} must be a whole number.")
        normalized: Any = int(number)
    elif setting.kind == "number":
        normalized = _finite_number(value, setting.label)
    elif setting.kind == "number_pair":
        if key in {"rate_bounds", "speed_bounds", "firing_bounds"} and isinstance(value, (tuple, list)):
            normalized = tuple(None if item is None else _finite_number(item, setting.label, positive=True) for item in value)
        else:
            normalized = _parse_number_sequence(value, setting.label)
        if len(normalized) != 2 or (all(item is not None for item in normalized) and normalized[1] <= normalized[0]):
            raise ValueError(f"{setting.label} must contain increasing lower and upper values.")
    elif setting.kind == "number_list":
        normalized = np.asarray(_parse_number_sequence(value, setting.label), dtype=float)
        if key in {"time_grid", "x_array"} and (
            normalized.size > 1 and np.any(np.diff(normalized) <= 0)
        ):
            raise ValueError(f"{setting.label} must be strictly increasing.")
        normalized = _readonly(normalized)
    elif key == "region_names":
        if isinstance(value, str):
            normalized = tuple(part.strip() for part in value.split(",") if part.strip())
        else:
            normalized = tuple(str(part).strip() for part in value if str(part).strip())
        if not normalized:
            raise ValueError("Region names must not be empty.")
    else:
        normalized = str(value).strip()
        if not normalized:
            raise ValueError(f"{setting.label} must not be empty.")

    if isinstance(normalized, (int, float)):
        if setting.min_value is not None and normalized < setting.min_value:
            raise ValueError(f"{setting.label} must be at least {setting.min_value}.")
        if setting.max_value is not None and normalized > setting.max_value:
            raise ValueError(f"{setting.label} must be at most {setting.max_value}.")
        if "sim_number" in key and normalized > MAX_MAPPING_SIMULATIONS:
            raise ValueError(
                f"{setting.label} exceeds the interactive limit of "
                f"{MAX_MAPPING_SIMULATIONS:,} simulations."
            )
    return normalized


def _accepted_assumptions(route: MappingRoute) -> set[str]:
    return (
        set(route.required_assumptions)
        | set(route.optional_assumptions)
        | {key for key, _value in route.defaults}
    )


def _apply_domain(
    request: MappingRequest,
    route: MappingRoute,
    assumptions: dict[str, Any],
    *,
    input_spatial_points: int,
    spatial_selection: _SpatialSelection | None = None,
) -> tuple[float, float, float]:
    accepted = _accepted_assumptions(route)
    explicit_interval = assumptions.get("interval_kb")
    interval_start = float(explicit_interval[0]) if explicit_interval is not None else None
    start = (
        spatial_selection.start_kb
        if spatial_selection is not None
        else (
            (0.0 if interval_start is None else interval_start)
            if request.region_start_kb is None
            else _finite_number(request.region_start_kb, "Region start")
        )
    )
    if start < 0:
        raise ValueError(f"Region start must be at least 0 kb; received {start:g} kb.")
    end: float | None = None
    if not request.whole_genome and request.region_end_kb is not None:
        end = _finite_number(request.region_end_kb, "Region end")
        if end <= start:
            raise ValueError(
                f"Region end must be greater than region start; received Start {start:g} kb "
                f"and End {end:g} kb."
            )

    resolution_x = _request_resolution_x(request, route)
    if "resolution" in accepted:
        assumptions["resolution"] = resolution_x

    resolution_t_value = request.resolution_t
    if resolution_t_value is None:
        resolution_t_value = assumptions.get("resolution_time")
    if resolution_t_value is None:
        resolution_t_value = route.setting_defaults_dict.get("resolution_time", 5.0)
    resolution_t = _finite_number(resolution_t_value, "Time resolution", positive=True)
    if "resolution_time" in accepted:
        assumptions.setdefault("resolution_time", resolution_t)

    if input_spatial_points:
        n_bins = input_spatial_points
        implied_end = start + n_bins * resolution_x
        if assumptions.get("n_bins") is not None and int(assumptions["n_bins"]) != n_bins:
            raise ValueError(
                f"The uploaded spatial inputs contain {n_bins:,} bins, but the explicit "
                f"n_bins setting requests {int(assumptions['n_bins']):,}."
            )
        if explicit_interval is not None:
            explicit_start, explicit_end = map(float, explicit_interval)
            interval_bins = spatial_bin_count(explicit_start, explicit_end, resolution_x)
            if explicit_start != start or interval_bins != n_bins:
                raise ValueError(
                    "The explicit genomic interval does not match the uploaded spatial "
                    "profile length and resolution."
                )
        if assumptions.get("genome_length") is not None and not np.isclose(
            float(assumptions["genome_length"]), n_bins * resolution_x
        ):
            raise ValueError(
                "The explicit genome length does not match the uploaded spatial profile."
            )
        if (
            assumptions.get("x_array") is not None
            and np.asarray(assumptions["x_array"]).size != n_bins
        ):
            raise ValueError(
                "The explicit position axis does not match the uploaded spatial profile."
            )
        # Uploaded X/XT data and their declared resolution are authoritative.
        # The page's default Region end must not silently turn a 64-bin upload
        # into a 1,000-bin request.
        end = (
            spatial_selection.end_kb
            if request.source_resolution_x is not None and spatial_selection is not None
            else implied_end
        )
    elif end is not None:
        n_bins = int(math.ceil((end - start) / resolution_x))
    else:
        n_bins = 0

    if n_bins:
        length = n_bins * resolution_x if input_spatial_points else cast(float, end) - start
        if n_bins > MAX_SPATIAL_POINTS:
            raise ValueError(
                f"The requested region contains {n_bins:,} bins; the local limit is "
                f"{MAX_SPATIAL_POINTS:,}. Increase the resolution or narrow the region."
            )
        if "interval_kb" in accepted:
            assumptions.setdefault("interval_kb", (start, cast(float, end)))
        if "genome_length" in accepted:
            assumptions.setdefault("genome_length", length)
        if "n_bins" in accepted:
            assumptions.setdefault("n_bins", n_bins)
        if "x_array" in accepted:
            assumptions.setdefault(
                "x_array",
                _readonly(start + np.arange(n_bins, dtype=float) * resolution_x),
            )

    n_bins_value = assumptions.get("n_bins")
    if n_bins_value is not None and int(n_bins_value) > MAX_SPATIAL_POINTS:
        raise ValueError(f"Spatial bins exceed the local limit of {MAX_SPATIAL_POINTS:,}.")
    return start, resolution_x, resolution_t


def _apply_time_domain(
    assumptions: dict[str, Any],
    settings: Mapping[str, MappingSetting],
    *,
    resolution_t: float,
    input_time_points: int,
) -> None:
    """Give forward mappings a bounded clock while retaining supplied axes."""
    if not {"time_grid", "max_time"} <= settings.keys():
        return
    maximum = assumptions.get("max_time", assumptions.get("max_rep_time"))
    if maximum is None:
        maximum = settings["max_time"].default
    if maximum is None:
        return
    maximum = float(maximum)
    if not math.isfinite(maximum) or maximum <= 0:
        raise ValueError("Maximum time must be finite and positive.")
    assumptions["max_time"] = maximum
    if assumptions.get("time_grid") is not None:
        return
    if input_time_points:
        # Temporal uploads define their own length. A default forward limit
        # must not resample them or contradict their intrinsic time axis.
        grid = np.arange(input_time_points, dtype=float) * resolution_t
    else:
        intervals = maximum / resolution_t
        if not math.isfinite(intervals) or intervals > MAX_TIME_POINTS - 1:
            raise ValueError(
                "The requested time grid exceeds the local limit of "
                f"{MAX_TIME_POINTS:,} points. Increase the time step or shorten the time domain."
            )
        steps = int(math.floor(intervals))
        grid = np.arange(steps + 1, dtype=float) * resolution_t
        # A partial final interval ends exactly at the requested maximum.
        grid = np.append(grid[grid < maximum], maximum)
    assumptions["time_grid"] = _readonly(grid)


def _validate_grid(
    assumptions: Mapping[str, Any],
    *,
    input_spatial_points: int = 0,
    input_time_points: int = 0,
    tiled_kinetics: bool = False,
) -> None:
    time_grid = assumptions.get("time_grid")
    if time_grid is not None:
        time_points = int(np.asarray(time_grid).size)
    else:
        max_time = assumptions.get("max_time", assumptions.get("max_rep_time"))
        resolution_t = assumptions.get("resolution_time")
        time_points = (
            int(math.floor(float(max_time) / float(resolution_t))) + 1
            if max_time is not None and resolution_t is not None
            else 0
        )
    if time_points > MAX_TIME_POINTS:
        raise ValueError(
            f"The requested time grid contains {time_points:,} points; the local limit is "
            f"{MAX_TIME_POINTS:,}. Increase the time step or shorten the time domain."
        )
    if input_time_points and time_points and input_time_points != time_points:
        raise ValueError(
            f"Uploaded temporal inputs contain {input_time_points:,} points, but the "
            f"explicit time grid contains {time_points:,}."
        )
    time_points = time_points or input_time_points
    n_bins = int(assumptions.get("n_bins", 0) or 0) or input_spatial_points
    if not tiled_kinetics and n_bins and time_points and n_bins * time_points > MAX_GRID_CELLS:
        raise ValueError(
            "The requested space-time grid is too large for an interactive mapping run. "
            "Coarsen the spatial or temporal resolution."
        )


def validate_mapping_request(
    request: MappingRequest,
    cancel: CancelChecker | None = None,
) -> _PreparedCall:
    """Validate and normalise a request before calling or allocating in rmap."""

    check_cancelled = cancel or (lambda: None)
    check_cancelled()
    request = prepare_filter_request(request)
    if not isinstance(request.route, MappingRoute):
        raise ValueError("Mapping request must contain a resolved mapping route.")
    if not isinstance(request.whole_genome, (bool, np.bool_)):
        raise ValueError("Whole genome must be true or false.")
    try:
        route = mapping_route(request.route.method)
    except KeyError as exc:
        raise ValueError(f"Unknown mapping method {request.route.method!r}.") from exc
    if route.target != request.route.target:
        raise ValueError(
            f"Method {route.method!r} maps to {route.target!r}, not {request.route.target!r}."
        )

    raw_inputs: dict[str, Any] = {}
    for raw_key, value in request.inputs.items():
        check_cancelled()
        exact_key = _find_supplied_key(route, str(raw_key))
        if exact_key in raw_inputs:
            raise ValueError(f"Input {exact_key!r} was supplied more than once.")
        raw_inputs[exact_key] = value

    raw_forms: dict[str, DataForm] = {}
    for raw_key, form in request.input_forms.items():
        check_cancelled()
        exact_key = _find_supplied_key(route, str(raw_key))
        if form not in DATA_FORMS:
            raise ValueError(f"Unknown data form {form!r} for input {raw_key!r}.")
        raw_forms[exact_key] = form

    for input_spec in route.inputs:
        check_cancelled()
        if input_spec.key not in raw_inputs:
            if input_spec.required:
                raise ValueError(f"{input_spec.label} is required for the selected route.")
            continue
        form = raw_forms.get(input_spec.key)
        if form is None:
            raise ValueError(f"Choose the data form for {input_spec.label}.")
        if form not in input_spec.forms:
            allowed = ", ".join(input_spec.forms)
            raise ValueError(
                f"{input_spec.label} does not accept {form!r} for this route; choose {allowed}."
            )

    raw_inputs, spatial_selection = _crop_spatial_inputs(
        request,
        route,
        raw_inputs,
        raw_forms,
        cancel=check_cancelled,
        prewindowed_inputs=frozenset(
            _find_supplied_key(route, str(key)) for key in request.prewindowed_inputs
        ),
    )
    if request.whole_genome and spatial_selection is None:
        raise ValueError(
            "Whole region mapping needs an uploaded spatial profile to determine its length."
        )
    _preflight_input_payload(raw_inputs, check_cancelled)

    normalized_inputs: dict[str, Any] = {}
    for input_spec in route.inputs:
        check_cancelled()
        if input_spec.key not in raw_inputs:
            continue
        form = raw_forms[input_spec.key]
        normalized_inputs[input_spec.key] = _normalise_input(
            raw_inputs[input_spec.key], input_spec, form
        )
        check_cancelled()

    total_values, total_bytes = _input_size(normalized_inputs)
    if total_values > MAX_INPUT_VALUES or total_bytes > MAX_INPUT_BYTES:
        raise ValueError(
            f"Mapped inputs contain {total_values:,} values / {total_bytes / 1024**2:.1f} MiB; "
            f"the interactive limits are {MAX_INPUT_VALUES:,} values and "
            f"{MAX_INPUT_BYTES / 1024**2:.0f} MiB."
        )
    input_spatial_points, input_time_points = _validate_input_shapes(
        route, normalized_inputs, raw_forms
    )

    shape = request.target_shape if request.target_shape is not None else route.default_target_shape
    available_shapes = route.shapes_for(raw_forms)
    if (not route.unified_kinetics and (request.fit_filter.regions.enabled or request.fit_filter.relative_to is not None)
            and route.target == 'initiation_rate' and route.output_dependence in {'time', 'spacetime'}):
        available_shapes = (*available_shapes, 'free')
    if shape is not None and shape not in available_shapes:
        choices = ", ".join(available_shapes) or "no configurable shapes"
        raise ValueError(f"Target shape must be one of: {choices} for the selected input forms.")
    if not route.supports_periodic(shape) and request.settings.get("perQ"):
        raise ValueError("The selected target and route support linear geometry only.")
    active_settings = {item.key: item for item in route.active_settings(shape, raw_forms)}
    if request.fit_filter.regions.enabled and route.target == 'fork_speed' and any(item.key == 'replication_timing' for item in route.inputs):
        active_settings = {item.key: item for item in filtered_speed_settings(route, active_settings.values())}
    supplied_settings = dict(request.settings)
    for alias, canonical in route.single_phase_fit_aliases.items():
        if alias in supplied_settings:
            value = supplied_settings.pop(alias)
            if value is not None:
                # Match the legacy fitter's explicit alias precedence while
                # exposing only one control and one saved effective setting.
                supplied_settings[canonical] = value
    setting_keys = set(active_settings)
    # Accept an explicit linear boundary from older callers, while exposing
    # no editable periodic control for a branch which cannot execute it.
    if request.settings.get("perQ") is False and any(item.key == "perQ" for item in route.settings):
        setting_keys.add("perQ")
    unknown_settings = tuple(str(key) for key in supplied_settings if key not in setting_keys)
    if unknown_settings:
        raise ValueError(
            f"The selected route does not accept setting(s): {', '.join(unknown_settings)}."
        )
    assumptions: dict[str, Any] = {}
    for key, value in supplied_settings.items():
        check_cancelled()
        if value is not None:
            assumptions[str(key)] = _normalise_setting(
                route, str(key), value, active_settings.get(str(key))
            )
    if "time_grid" in setting_keys:
        for input_spec in route.inputs:
            bundle = normalized_inputs.get(input_spec.key)
            if not isinstance(bundle, Mapping):
                continue
            for axis_key in ("time_grid", "time_axis"):
                if axis_key not in bundle:
                    continue
                time_grid = _normalise_setting(route, "time_grid", bundle[axis_key])
                existing = assumptions.get("time_grid")
                if existing is not None and not np.array_equal(existing, time_grid):
                    raise ValueError(
                        "The supplied time grid does not match the prepared result's time axis."
                    )
                assumptions["time_grid"] = time_grid
    if shape is not None:
        assumptions["target_shape"] = shape

    accepted_assumptions = _accepted_assumptions(route)
    geometry = str(assumptions.get("geometry", "")).strip().casefold()
    if "perQ" in request.settings and "geometry" not in request.settings and "geometry" in accepted_assumptions:
        geometry = "periodic" if bool(assumptions.get("perQ")) else "linear"
        assumptions["geometry"] = geometry
    if geometry in {"linear", "periodic"} and "perQ" in accepted_assumptions:
        # Keep the legacy rmap boundary flag explicit in the dispatched request
        # and in MappingResult.assumptions. This mirrors rmap's own geometry
        # normalization and makes the boundary condition auditable by the UI.
        assumptions["perQ"] = geometry == "periodic"
    if "target_form" in accepted_assumptions:
        target_form = route.output_dependence
        allowed_target_forms = dict(route.assumption_values).get("target_form", ())
        if allowed_target_forms and target_form not in allowed_target_forms:
            raise ValueError(
                f"Resolved method {route.method!r} does not accept its declared "
                f"{target_form!r} output form."
            )
        assumptions["target_form"] = target_form

    # IODs are a one-dimensional pooled ensemble distribution, not a detailed
    # space-time output.  Retain them for the IOD target, but bound the local
    # app payload before it reaches browser/export memory.  Direct Python rmap
    # calls remain unrestricted unless callers provide this optional setting.
    if (
        route.target == "inter_origin_distance"
        and "retain_inter_origin_distances_limit" in accepted_assumptions
    ):
        assumptions.setdefault(
            "retain_inter_origin_distances_limit",
            MAX_MAPPING_IOD_SAMPLES,
        )

    if spatial_selection is not None and input_spatial_points:
        selected_end = (
            spatial_selection.end_kb if request.source_resolution_x is not None
            else spatial_selection.start_kb
            + input_spatial_points * _request_resolution_x(request, route)
        )
        if "interval_kb" in accepted_assumptions:
            assumptions["interval_kb"] = (spatial_selection.start_kb, selected_end)
        if "genome_length" in accepted_assumptions:
            assumptions["genome_length"] = input_spatial_points * _request_resolution_x(
                request, route
            )
        if "n_bins" in accepted_assumptions:
            assumptions["n_bins"] = input_spatial_points
        if assumptions.get("x_array") is not None:
            explicit_x = np.asarray(assumptions["x_array"])
            if request.source_resolution_x is not None:
                assumptions["x_array"] = _readonly(
                    spatial_selection.start_kb
                    + np.arange(input_spatial_points) * _request_resolution_x(request, route)
                )
            elif explicit_x.size == spatial_selection.source_bins:
                assumptions["x_array"] = _readonly(
                    np.asarray(
                        explicit_x[spatial_selection.start_index : spatial_selection.stop_index],
                        dtype=float,
                    ).reshape(-1)
                )

    source: dict[str, Any] = {}
    for input_spec in route.inputs:
        check_cancelled()
        if input_spec.key not in normalized_inputs:
            continue
        if input_spec.role == "source":
            source[input_spec.key] = normalized_inputs[input_spec.key]
        else:
            assumptions[input_spec.key] = normalized_inputs[input_spec.key]

    if not source:
        raise ValueError("The selected mapping route has no materialised source data.")
    start, resolution_x, resolution_t = _apply_domain(
        request,
        route,
        assumptions,
        input_spatial_points=input_spatial_points,
        spatial_selection=spatial_selection,
    )
    _apply_time_domain(
        assumptions, active_settings,
        resolution_t=resolution_t, input_time_points=input_time_points,
    )
    _validate_grid(
        assumptions,
        input_spatial_points=input_spatial_points,
        input_time_points=input_time_points,
        tiled_kinetics=route.unified_kinetics and not any(
            float(assumptions.get(key) or 0.) > 0.
            for key in ("stall_rate", "stall_rate_per_kb")
        ),
    )
    return _PreparedCall(
        route=route,
        source=source,
        assumptions=assumptions,
        start_kb=start,
        resolution_x=resolution_x,
        resolution_t=resolution_t,
        spatial_size=input_spatial_points,
        input_forms=MappingProxyType(dict(raw_forms)),
        initiation_rate_per_bin=request.initiation_rate_per_bin,
        end_kb=spatial_selection.end_kb if spatial_selection is not None else request.region_end_kb,
        fit_filter=request.fit_filter,
    )


def _result_array(value: Any, label: str, *, allow_empty: bool = False) -> np.ndarray | None:
    if value is None or isinstance(value, (str, bytes, bytearray, Mapping)):
        return None
    if np.iscomplexobj(value):
        raise RuntimeError(f"DNAscape returned complex values for {label!r}.")
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return None
    if array.dtype.kind not in "biuf":
        return None
    if (array.size == 0 and not allow_empty) or array.ndim > 2:
        return None
    if np.any(np.isinf(array)):
        raise RuntimeError(f"DNAscape returned infinite values for {label!r}.")
    if array.ndim == 0:
        array = array.reshape(1)
    return _readonly(array)


def _fit_history_array(value: Any) -> np.ndarray | None:
    """Normalize numeric and structured optimizer histories to one MSE array."""

    if (
        isinstance(value, (list, tuple))
        and value
        and all(isinstance(item, Mapping) for item in value)
    ):
        history: list[float] = []
        for item in value:
            raw_mse = item.get("mse")
            if raw_mse is None:
                history.append(math.nan)
                continue
            try:
                numeric = float(raw_mse)
            except (TypeError, ValueError):
                history.append(math.nan)
                continue
            history.append(numeric if math.isfinite(numeric) and numeric >= 0 else math.nan)
        if history and any(math.isfinite(item) for item in history):
            return _readonly(np.asarray(history, dtype=float))
        return None
    return _result_array(value, "fit_history")


def _region_labels(values: Any) -> tuple[str, ...]:
    if not isinstance(values, Mapping) or "names" not in values:
        return ()
    names = np.asarray(values["names"]).reshape(-1)
    return tuple(str(name) for name in names)


def _collect_result_outputs(
    route: MappingRoute,
    info: Mapping[str, Any],
    input_forms: Mapping[str, DataForm],
) -> tuple[dict[str, np.ndarray], dict[str, DataForm], str, tuple[str, ...], np.ndarray | None]:
    values = info.get("values")
    outputs: dict[str, np.ndarray] = {}
    time_grid: np.ndarray | None = None
    labels = _region_labels(values)

    if isinstance(values, Mapping):
        for raw_key, raw_value in values.items():
            key = str(raw_key)
            if key in {"names", "regions"} or key in _DIAGNOSTIC_KEYS:
                continue
            if key in _AXIS_OUTPUT_KEYS:
                axis = _result_array(raw_value, key)
                if key in {"time_grid", "time_axis"} and axis is not None:
                    time_grid = axis.reshape(-1)
                continue
            array = (
                _fit_history_array(raw_value)
                if key == "fit_history"
                else _result_array(raw_value, key)
            )
            if array is not None:
                outputs[key] = array
        if route.target == "origin_efficiency" and "efficiency" in outputs:
            outputs.setdefault("origin_efficiency", outputs["efficiency"])
    else:
        array = _result_array(
            values, route.target, allow_empty=route.target == "inter_origin_distance"
        )
        if array is None:
            raise RuntimeError(f"DNAscape returned no numeric {route.target!r} values.")
        outputs[route.target] = array

    supplemental_keys = {
        str(key) for key in info if str(key) in _FIT_OUTPUT_KEYS or str(key).endswith("_fit")
    }
    for key in supplemental_keys:
        if key in outputs:
            continue
        array = (
            _fit_history_array(info[key]) if key == "fit_history" else _result_array(info[key], key)
        )
        if array is not None:
            outputs[key] = array
    if "time_grid" in info and time_grid is None:
        axis = _result_array(info["time_grid"], "time_grid")
        if axis is not None:
            time_grid = axis.reshape(-1)

    if not outputs:
        raise RuntimeError("DNAscape returned no plottable numeric mapping output.")
    total_values = sum(array.size for array in outputs.values())
    total_bytes = sum(array.nbytes for array in outputs.values())
    disk_backed_fit = route.unified_kinetics and info.get("tiled_fitting") and info.get("disk_backed_output")
    if not disk_backed_fit and (total_values > MAX_OUTPUT_VALUES or total_bytes > MAX_OUTPUT_BYTES):
        raise RuntimeError(
            f"Mapped output contains {total_values:,} values / {total_bytes / 1024**2:.1f} MiB; "
            "reduce the requested grid before previewing it in the local app."
        )

    if route.target in outputs:
        primary = route.target
    elif route.target == "fork_densities_xt" and "replicated_fraction_xt" in outputs:
        primary = "replicated_fraction_xt"
    else:
        primary = next(iter(outputs))

    forms: dict[str, DataForm] = {}
    for key, array in outputs.items():
        if key.endswith("_xt") or (
            route.output_form in {"xt", "bundle"}
            and key in {"initiation_rate", "fork_speed"}
            and array.ndim == 2
        ):
            forms[key] = "xt"
        elif key.endswith("_fit"):
            source_key = key.removesuffix("_fit")
            matching_input = next(
                (
                    input_key
                    for input_key in input_forms
                    if input_key == source_key or _friendly_input_key(input_key) == source_key
                ),
                None,
            )
            forms[key] = (
                input_forms[matching_input]
                if matching_input is not None
                else ("xt" if array.ndim == 2 and 1 not in array.shape else "x")
            )
        elif key == "residual":
            fraction_input = next((name for name in ("replicated_fraction_xt", "replicated_fraction")
                                   if name in input_forms), None)
            if fraction_input is not None and route.unified_kinetics:
                forms[key] = input_forms[fraction_input]
            elif array.ndim == 2 and 1 not in array.shape:
                forms[key] = "xt"
            elif array.ndim == 2 and array.shape[0] == 1:
                forms[key] = "t"
            else:
                forms[key] = "x"
        elif key == "fit_history":
            forms[key] = "supplied"
        elif route.output_form == "regions":
            forms[key] = "regions"
        elif key == route.target:
            forms[key] = route.output_form
        elif array.ndim == 2:
            forms[key] = "xt"
        else:
            forms[key] = "supplied"
    for key, form in forms.items():
        if form in {"scalar", "x", "t", "regions", "supplied"}:
            outputs[key] = _readonly(np.asarray(outputs[key], dtype=float).reshape(-1))
    return outputs, forms, primary, labels, time_grid


def _axis_lengths(
    outputs: Mapping[str, np.ndarray], forms: Mapping[str, DataForm]
) -> tuple[int, int]:
    spatial = 0
    temporal = 0
    for key, array in outputs.items():
        form = forms[key]
        if form == "x":
            spatial = max(spatial, int(array.reshape(-1).size))
        elif form == "t":
            temporal = max(temporal, int(array.reshape(-1).size))
        elif form == "xt" and array.ndim == 2:
            spatial = max(spatial, int(array.shape[0]))
            temporal = max(temporal, int(array.shape[1]))
    return spatial, temporal


def _fit_comparisons(
    prepared: _PreparedCall,
    outputs: Mapping[str, np.ndarray],
    forms: Mapping[str, DataForm],
) -> dict[str, MappingComparison]:
    """Pair any one-dimensional ``*_fit`` output with its observed input."""

    available_inputs: dict[str, Any] = {
        **prepared.assumptions,
        **prepared.source,
    }
    comparisons: dict[str, MappingComparison] = {}
    for fitted_key, fitted_value in outputs.items():
        if not fitted_key.endswith("_fit"):
            continue
        source_name = fitted_key.removesuffix("_fit")
        source_key = next(
            (
                key
                for key in prepared.input_forms
                if key == source_name or _friendly_input_key(key) == source_name
            ),
            None,
        )
        if source_key is None or source_key not in available_inputs:
            continue
        form = prepared.input_forms[source_key]
        if form not in {"scalar", "x", "t", "regions", "supplied"}:
            continue
        raw_observed = available_inputs[source_key]
        if isinstance(raw_observed, Mapping):
            continue
        try:
            observed = np.squeeze(np.asarray(raw_observed, dtype=float))
            fitted = np.squeeze(np.asarray(fitted_value, dtype=float))
        except (TypeError, ValueError):
            continue
        if observed.ndim > 1 or fitted.ndim > 1 or observed.size != fitted.size:
            continue
        observed = _readonly(np.ascontiguousarray(observed, dtype=float).reshape(-1))
        fitted = _readonly(np.ascontiguousarray(fitted, dtype=float).reshape(-1))
        comparisons[fitted_key] = MappingComparison(
            source_key=source_key,
            fitted_key=fitted_key,
            form=form,
            observed=observed,
            fitted=fitted,
        )
    return comparisons


def _fit_mse(
    info: Mapping[str, Any],
    outputs: Mapping[str, np.ndarray],
    comparisons: Mapping[str, MappingComparison],
) -> float | None:
    """Return the final finite fit MSE without rerunning scientific work.

    Most fitting routes expose the final deterministic value as ``fit_mse``.
    Joint optimizers instead expose an MSE history, while older routes may
    only return an observed/fitted comparison.  These fallbacks preserve the
    same meaning and make the UI diagnostic available consistently.
    """

    raw_mse = info.get("fit_mse")
    if raw_mse is None and isinstance(info.get("values"), Mapping):
        raw_mse = info["values"].get("fit_mse")
    try:
        numeric_mse = float(raw_mse) if raw_mse is not None else math.nan
    except (TypeError, ValueError):
        numeric_mse = math.nan
    if math.isfinite(numeric_mse) and numeric_mse >= 0:
        return numeric_mse

    history = outputs.get("fit_history")
    if history is not None:
        finite_history = np.asarray(history, dtype=float).reshape(-1)
        finite_history = finite_history[np.isfinite(finite_history)]
        if finite_history.size and np.any(finite_history >= 0):
            # Optimizers return the best retained profile, which need not be
            # the final attempted step when a later proposal was rejected.
            return float(np.min(finite_history[finite_history >= 0]))

    if comparisons:
        comparison = next(iter(comparisons.values()))
        residual = np.asarray(comparison.fitted, dtype=float) - np.asarray(
            comparison.observed,
            dtype=float,
        )
    else:
        residual = np.asarray(outputs.get("residual", []), dtype=float)
    finite_residual = residual.reshape(-1)
    finite_residual = finite_residual[np.isfinite(finite_residual)]
    if finite_residual.size:
        return float(np.mean(np.square(finite_residual)))
    return None


def _build_axes(
    prepared: _PreparedCall,
    outputs: Mapping[str, np.ndarray],
    forms: Mapping[str, DataForm],
    returned_time_grid: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    spatial_size, time_size = _axis_lengths(outputs, forms)
    if spatial_size == 0:
        explicit_x = prepared.assumptions.get("x_array")
        spatial_size = (
            int(np.asarray(explicit_x).size)
            if explicit_x is not None
            else int(prepared.assumptions.get("n_bins", 0) or prepared.spatial_size)
        )
    explicit_x = prepared.assumptions.get("x_array")
    if explicit_x is not None and np.asarray(explicit_x).size == spatial_size:
        x_axis = np.array(explicit_x, dtype=float, copy=True).reshape(-1)
    else:
        x_axis = prepared.start_kb + np.arange(spatial_size, dtype=float) * prepared.resolution_x
    explicit_time = returned_time_grid
    if explicit_time is None and prepared.assumptions.get("time_grid") is not None:
        explicit_time = np.asarray(prepared.assumptions["time_grid"], dtype=float).reshape(-1)
    if explicit_time is not None and time_size and explicit_time.size != time_size:
        raise RuntimeError(
            f"DNAscape returned {time_size:,} time columns but a "
            f"{explicit_time.size:,}-point time axis."
        )
    time_axis = (
        np.array(explicit_time, dtype=float, copy=True)
        if explicit_time is not None
        else np.arange(time_size, dtype=float) * prepared.resolution_t
    )
    return _readonly(x_axis), _readonly(time_axis)


def _bounded_track_payload(upload: TrackUpload) -> bytes:
    raw = bytes(upload.content)
    if not raw:
        raise ValueError("The selected genomic track is empty.")
    if len(raw) > MAX_TRACK_UPLOAD_BYTES:
        raise ValueError(
            f"Genomic-track uploads are limited to {MAX_TRACK_UPLOAD_BYTES / 1024**2:.0f} MiB."
        )
    compressed = upload.name.casefold().endswith(".gz") or raw.startswith(b"\x1f\x8b")
    if not compressed:
        return raw

    chunks: list[bytes] = []
    total = 0
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(raw)) as source:
            while True:
                chunk = source.read(min(1024 * 1024, MAX_TRACK_PAYLOAD_BYTES - total + 1))
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_TRACK_PAYLOAD_BYTES:
                    raise ValueError(
                        "The compressed genomic track expands beyond the 64 MiB local limit."
                    )
                chunks.append(chunk)
    except ValueError:
        raise
    except (EOFError, OSError) as exc:
        raise ValueError("The compressed genomic track is incomplete or invalid.") from exc
    return b"".join(chunks)


def materialize_track_upload(
    upload: TrackUpload,
    *,
    chromosome: str | None,
    start: int | None = None,
    end: int | None = None,
    bin_size: int = 1_000,
    aggregation: str = "mean",
    value_column: str | int | None = None,
    missing: str = "nan",
    fill_value: float = 0.0,
    interpolation: str = "none",
    smooth_window: int = 1,
    normalize: str = "none",
) -> np.ndarray:
    """Materialise one inspected bedGraph/BigWig through public ``rtrack``.

    BigWig's native reader requires a path.  The retained, bounded upload is
    therefore written to one narrowly scoped temporary file and removed in a
    ``finally`` block.  DNAscape is imported lazily in this worker-side module,
    leaving upload inspection lightweight.
    """

    source_format = str(upload.source_format).casefold()
    if source_format not in {"bedgraph", "bigwig"}:
        raise ValueError("Track materialization requires bedGraph or BigWig data.")
    try:
        normalized_bin_size = int(bin_size)
        normalized_smooth_window = int(smooth_window)
    except (TypeError, ValueError) as exc:
        raise ValueError("Track bin size and smoothing window must be whole numbers.") from exc
    if normalized_bin_size <= 0:
        raise ValueError("Track bin size must be a positive integer.")
    if normalized_smooth_window <= 0:
        raise ValueError("Track smoothing window must be a positive integer.")
    if start is not None and end is not None and int(end) <= int(start):
        raise ValueError("Track end must be greater than track start.")

    payload = _bounded_track_payload(upload)
    suffix = ".bedgraph" if source_format == "bedgraph" else ".bw"
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temporary:
            temporary.write(payload)
            temporary_path = Path(temporary.name)

        from dnascape.io import rtrack

        selected_column: str | int | None = value_column
        if source_format == "bedgraph" and selected_column is None:
            selected_column = getattr(upload, "default_column", None)
        if (
            source_format == "bedgraph"
            and isinstance(selected_column, str)
            and selected_column in upload.columns
        ):
            selected_column = upload.columns.index(selected_column)
        values = rtrack(
            temporary_path,
            source_format=source_format,
            chromosome=chromosome,
            start=start,
            end=end,
            bin_size=normalized_bin_size,
            aggregation=aggregation,
            value_column=selected_column,
            missing=missing,
            fill_value=_finite_number(fill_value, "Track fill value"),
            interpolation=interpolation,
            smooth_window=normalized_smooth_window,
            normalize=normalize,
            data_form="space",
        )
        profile = _numeric_array(values, "Binned genomic track")
        # A valid one-bin track squeezes to a scalar; preserve it as a
        # one-dimensional profile rather than rejecting the smallest region.
        profile = np.atleast_1d(np.squeeze(profile))
        if profile.ndim != 1:
            raise ValueError("A genomic track must resolve to one spatial profile.")
        return _readonly(profile)
    except ImportError:
        raise
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise ValueError(str(exc)) from exc
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def materialize_many_tracks(
    uploads: Sequence[TrackUpload],
    **track_options: Any,
) -> np.ndarray:
    """Bin ordered genomic tracks and stack them as an ``(X, T)`` field."""

    if not uploads:
        raise ValueError("Upload at least one genomic track.")
    if len(uploads) > MAX_TRACK_FILES:
        raise ValueError(f"At most {MAX_TRACK_FILES:,} genomic tracks may be mapped at once.")
    total_bytes = sum(len(bytes(upload.content)) for upload in uploads)
    if total_bytes > MAX_INPUT_BYTES:
        raise ValueError(
            f"Genomic-track uploads total {total_bytes / 1024**2:.1f} MiB; "
            f"the local limit is {MAX_INPUT_BYTES / 1024**2:.0f} MiB."
        )
    profiles = [materialize_track_upload(upload, **track_options) for upload in uploads]
    lengths = {profile.size for profile in profiles}
    if len(lengths) != 1:
        raise ValueError("All genomic tracks must resolve to the same number of bins.")
    values = np.column_stack(profiles)
    if values.size > MAX_INPUT_VALUES or values.nbytes > MAX_INPUT_BYTES:
        raise ValueError("The stacked genomic tracks exceed the local input limit.")
    return _readonly(values)


def run_mapping(
    request: MappingRequest,
    report: ProgressReporter | None = None,
    cancel: CancelChecker | None = None,
) -> MappingResult:
    """Run one exact ``rmap`` route with progress and cancellation plumbing."""

    from dnascape import rmap
    from dnascape.utils import mapping_progress, progress_context

    reporting_enabled = report is not None
    report = report or (lambda _value, _message: None)
    cancel = cancel or (lambda: None)
    if not _RUN_LOCK.acquire(blocking=False):
        raise RuntimeError("Another mapping operation is already running in this local app.")
    try:
        started = time.perf_counter()
        cancel()
        report(0.02, "Preparing mapping...")
        prepared = validate_mapping_request(request, cancel=cancel)
        cancel()
        last_progress = 0.02
        reported_fit_history: list[float] = []

        def monotonic_report(value: float, message: str) -> None:
            nonlocal last_progress
            numeric = min(0.97, max(last_progress, float(value)))
            last_progress = numeric
            text = str(message)
            if text.casefold().startswith("rmap"):
                text = f"Mapping{text[4:]}"
            elif text.startswith("Fitting ") and "..." in text:
                text = "Mapping..." + text.split("...", 1)[1]
            report(numeric, text)

        def retain_fit_mse(
            _label: str,
            _iteration: int,
            _total: int,
            value: float,
        ) -> None:
            """Retain the exact optimizer metric behind the formatted progress text."""

            numeric = float(value)
            if math.isfinite(numeric) and numeric >= 0:
                reported_fit_history.append(numeric)

        mapper = cast(Callable[..., Mapping[str, Any]], rmap)
        execution_assumptions = dict(prepared.assumptions)
        execution_source = dict(prepared.source)
        constraint = None
        if prepared.fit_filter.regions.enabled:
            reference_key = prepared.fit_filter.relative_to
            if prepared.fit_filter.regions.selection not in {'profile', 'ranges', 'all'} and reference_key not in prepared.input_forms:
                raise ValueError('Choose the input used to define the mask.')
            reference = prepared.source.get(reference_key, prepared.assumptions.get(reference_key, 0.))
            density_fit = prepared.route.unified_kinetics
            if (density_fit and prepared.route.target == 'initiation_rate'
                    and prepared.route.output_form in {'scalar', 'x'}
                    and any(float(execution_assumptions.get(key) or 0.) > 0.
                            for key in ('stall_rate', 'stall_rate_per_kb'))):
                # Stalled spatial fits retain the legacy per-bin solver. The
                # temporal Weibull fallback already uses density-valued fields.
                density_fit = False
            constraint = MappingFitConstraint(
                prepared.fit_filter, target=prepared.route.target, form=prepared.route.output_form,
                reference=reference, reference_form=prepared.input_forms.get(reference_key, 'scalar'),
                start=prepared.start_kb, resolution=prepared.resolution_x, resolution_t=prepared.resolution_t,
                reference_time=prepared.assumptions.get('time_grid'),
                per_bin=prepared.initiation_rate_per_bin, cancel=cancel,
                spatial_size=prepared.spatial_size, end=prepared.end_kb,
                density_model=density_fit,
            )
            if not constraint.selects_regions(prepared.spatial_size):
                constraint = None
            if prepared.route.output_form == 'xt':
                grid = prepared.assumptions.get('time_grid')
                timing = prepared.source.get('replication_timing')
                if grid is not None:
                    count = len(grid)
                elif timing is not None:
                    horizon = float(np.nanmax(timing)) * (4 if prepared.route.target == 'fork_speed' else 1)
                    count = int(np.ceil(horizon / prepared.resolution_t)) + 2
                else:
                    count = 1
                if prepared.spatial_size * count > MAX_GRID_CELLS:
                    raise ValueError('The transformed space–time target is too large. Increase the spatial or time resolution.')
        if prepared.initiation_rate_per_bin:
            # Public rmap accepts rates per kb; the page and rsim accept rates
            # per spatial bin. Convert at this boundary only. Fork speed stays
            # in kb/min: the engine already divides it by the spatial step.
            def density_inputs(values: Mapping[str, Any]) -> dict[str, Any]:
                converted = dict(values)
                for key, value in values.items():
                    if key in {"initiation_rate", "initiation_rate_xt"}:
                        converted[key] = np.asarray(value, dtype=float) / prepared.resolution_x
                    elif isinstance(value, Mapping):
                        converted[key] = density_inputs(value)
                return converted

            execution_source = density_inputs(execution_source)
            execution_assumptions = density_inputs(execution_assumptions)
            for key in ("rate_bounds", "floor", "initial_initiation_rate"):
                if execution_assumptions.get(key) is not None:
                    if key == "rate_bounds":
                        from dnascape._fit_bounds import scale_bounds
                        execution_assumptions[key] = scale_bounds(execution_assumptions[key], 1 / prepared.resolution_x)
                    else:
                        execution_assumptions[key] = np.asarray(execution_assumptions[key]) / prepared.resolution_x
            # rmap converts this logarithmic cutoff to its per-bin solver.
            if "err_threshold" in _accepted_assumptions(prepared.route):
                execution_assumptions["err_threshold"] = (
                    float(execution_assumptions.get("err_threshold", 15.))
                    + math.log10(prepared.resolution_x)
                )
        if reporting_enabled and "verbose" in _accepted_assumptions(prepared.route):
            # Optimizers gate their ETA and fit-error callbacks on verbose.
            # The worker context sends those callbacks to the page and keeps
            # console output suppressed; this is execution plumbing, not a
            # scientific setting to add to the saved mapping assumptions.
            execution_assumptions["verbose"] = True
        monotonic_report(
            0.04,
            f"Mapping {mapping_target(prepared.route.target_key).label.lower()}...",
        )
        with progress_context(
            report=monotonic_report,
            cancel=cancel,
            suppress_output=True,
            metric_report=retain_fit_mse,
        ), mapping_progress(execution_assumptions):
            from dnascape._fit_constraints import fit_constraint_context
            with fit_constraint_context(constraint):
                if (constraint is not None and not prepared.route.unified_kinetics
                        and prepared.route.target == 'fork_speed' and 'replication_timing' in execution_source):
                    from dnascape.inverse import fit_timing_fork_speed
                    accepted = {'initial_fork_speed', 'resolution_time', 'time_grid', 'T0', 'perQ',
                                'maxiter', 'spatial_knots', 'temporal_knots', 'regularization', 'speed_bounds', 'tolerance', 'verbose'}
                    options = {key: value for key, value in execution_assumptions.items() if key in accepted and value is not None}
                    options.setdefault('maxiter', max(int(execution_assumptions.get(key) or 0) for key in ('space_maxiter', 'temporal_maxiter', 'maxiter')) or 10)
                    fitted = fit_timing_fork_speed(
                        execution_source['replication_timing'],
                        initiation_rate=execution_source.get('initiation_rate', execution_assumptions.get('initiation_rate')),
                        speed_form=prepared.route.output_dependence, resolution=prepared.resolution_x,
                        fit_projection=constraint, **options,
                    )
                    info = {**fitted, 'values': fitted['fork_speed']}
                else:
                    if constraint is None and prepared.fit_filter.regions.enabled and prepared.route.target == 'fork_speed':
                        accepted = _accepted_assumptions(prepared.route) | {'target_shape', 'target_form'}
                        execution_assumptions = {key: value for key, value in execution_assumptions.items() if key in accepted}
                    info = mapper(
                        source=execution_source,
                        target=prepared.route.target,
                        assumptions=execution_assumptions,
                        method=prepared.route.method,
                        return_info=True,
                    )
                    if constraint is not None and constraint.frozen is None:
                        if prepared.route.target in {'initiation_rate', 'fork_speed'}:
                            raise ValueError('This route does not expose a fitted numeric profile for regional constraints.')
                        info = dict(info)
                        info['values'] = constraint.project(info['values'])
        info = dict(info)
        if constraint is not None and constraint.mask is not None and prepared.fit_filter.export:
            info['filter_intervals'] = constraint.export_intervals()
        if constraint is not None and constraint.warnings:
            info['filter_warnings'] = tuple(dict.fromkeys(constraint.warnings))
        for source_name, canonical_name in {
            "predicted_timing": "replication_timing_fit",
            "mse": "fit_mse",
            "optimisation_success": "converged",
            "optimisation_message": "convergence_message",
        }.items():
            if source_name in info:
                info.setdefault(canonical_name, info[source_name])
        cancel()
        report(0.98, "Preparing mapped preview...")
        outputs, forms, primary, labels, returned_time = _collect_result_outputs(
            prepared.route,
            info,
            prepared.input_forms,
        )
        if constraint is not None and prepared.fit_filter.export and constraint.mask is not None and constraint.mask.any():
            outputs['filter_positions'] = _readonly(constraint.export_positions())
            forms['filter_positions'] = 'supplied'
        if prepared.initiation_rate_per_bin:
            for key in outputs.keys() & {"initiation_rate", "initiation_rate_xt", "initiation_rate_fit", "initiation_rate_xt_fit"}:
                outputs[key] = _readonly(outputs[key] * prepared.resolution_x)
        if "fit_history" not in outputs and reported_fit_history:
            outputs["fit_history"] = _readonly(np.asarray(reported_fit_history, dtype=float))
            forms["fit_history"] = "supplied"
        inter_origin_distance_count: int | None = None
        inter_origin_distances_truncated = False
        if prepared.route.target == "inter_origin_distance":
            retained_count = int(np.asarray(outputs[primary]).size)
            raw_count = info.get("inter_origin_distance_count", retained_count)
            try:
                numeric_count = float(raw_count)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    "DNAscape returned an invalid inter-origin distance count."
                ) from exc
            if (
                not np.isfinite(numeric_count)
                or not numeric_count.is_integer()
                or numeric_count < retained_count
            ):
                raise RuntimeError("DNAscape returned an invalid inter-origin distance count.")
            inter_origin_distance_count = int(numeric_count)
            inter_origin_distances_truncated = bool(
                info.get(
                    "inter_origin_distances_truncated",
                    inter_origin_distance_count > retained_count,
                )
            )
        cancel()
        comparisons = _fit_comparisons(prepared, outputs, forms)
        if comparisons:
            comparison = next(iter(comparisons.values()))
            if "residual" not in outputs:
                outputs["residual"] = _readonly(
                    np.ascontiguousarray(
                        comparison.fitted - comparison.observed,
                        dtype=float,
                    )
                )
            if np.asarray(outputs["residual"]).size == comparison.observed.size:
                forms["residual"] = comparison.form
        fit_mse = _fit_mse(info, outputs, comparisons)
        if "fit_history" not in outputs and fit_mse is not None:
            # Some deterministic routes return only their final MSE and do not
            # report optimizer iterations. Preserve it as a one-value array so
            # the Mapping preview and every raw export format remain available.
            outputs["fit_history"] = _readonly(np.asarray([fit_mse], dtype=float))
            forms["fit_history"] = "supplied"
        x_axis, time_axis = _build_axes(
            prepared,
            outputs,
            forms,
            returned_time,
        )
        report(1.0, "Mapping complete")
        result = MappingResult(
            target=prepared.route.target,
            target_key=prepared.route.target_key,
            target_label=mapping_target(prepared.route.target_key).label,
            method=prepared.route.method,
            output_form=prepared.route.output_form,
            output_dependence=prepared.route.output_dependence,
            primary_output=primary,
            outputs=MappingProxyType(outputs),
            output_forms=MappingProxyType(forms),
            x_axis=x_axis,
            time_axis=time_axis,
            region_labels=labels,
            elapsed_seconds=float(time.perf_counter() - started),
            assumptions=MappingProxyType(dict(prepared.assumptions)),
            comparisons=MappingProxyType(comparisons),
            inter_origin_distance_count=inter_origin_distance_count,
            inter_origin_distances_truncated=inter_origin_distances_truncated,
            fit_mse=fit_mse,
            filter_selection=constraint.preview_selection() if constraint is not None else None,
            diagnostics=MappingProxyType({
                key: _readonly(value) if isinstance(value, np.ndarray) else value
                for metadata in (
                    info,
                    cast(Mapping[str, Any], info["values"])
                    if isinstance(info.get("values"), Mapping) else {},
                )
                for key, value in metadata.items()
                if key in _DIAGNOSTIC_KEYS
            }),
        )
        from .preview_inputs import add_mapping_input_previews

        return add_mapping_input_previews(result, prepared)
    finally:
        _RUN_LOCK.release()


__all__ = [
    "MAX_GRID_CELLS",
    "MAX_INPUT_BYTES",
    "MAX_INPUT_VALUES",
    "MAX_MAPPING_IOD_SAMPLES",
    "MAX_MAPPING_SIMULATIONS",
    "MAX_OUTPUT_BYTES",
    "MAX_OUTPUT_VALUES",
    "MAX_SPATIAL_POINTS",
    "MAX_TIME_POINTS",
    "MAX_TRACK_FILES",
    "MAX_TRACK_PAYLOAD_BYTES",
    "MAX_TRACK_UPLOAD_BYTES",
    "MappingRequest",
    "MappingComparison",
    "MappingResult",
    "TrackUpload",
    "materialize_many_tracks",
    "materialize_track_upload",
    "run_mapping",
    "validate_mapping_request",
]
