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
import json
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

from app.mapping_catalog import (
    DATA_FORMS,
    DataForm,
    MappingInput,
    MappingRoute,
    MappingSetting,
    mapping_route,
    mapping_target,
)

ProgressReporter = Callable[[float, str], None]
CancelChecker = Callable[[], None]

# Interactive limits are intentionally independent of the scientific package.
# They prevent a malformed upload or a mistyped grid/iteration count from
# exhausting the browser process while remaining generous for local analyses.
MAX_INPUT_VALUES = 16_000_000
MAX_INPUT_BYTES = 128 * 1024 * 1024
MAX_OUTPUT_VALUES = 24_000_000
MAX_OUTPUT_BYTES = 192 * 1024 * 1024
MAX_SPATIAL_POINTS = 1_000_000
MAX_TIME_POINTS = 100_000
MAX_GRID_CELLS = 12_000_000
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
    "valid_mask", "unidentifiable_count", "invalid_policy", "converged",
    "convergence_message", "optimizer_evaluations", "unreplicated_at_end",
    "identifiability", "forward_model", "firing_form", "speed_form",
    "regularization", "speed_anchor", "fit_objective", "fit_mse", "fit_used_descent",
    "terminal_survival", "extrapolated_tail", "tail_policy", "initial_policy",
    "timing_is_restricted_mean", "clipped_derivative_count", "derivative_method",
    "monotonic_policy", "temporal_smoothw", "density_floor", "unreplicated_floor",
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
    if spatial and temporal and spatial * temporal > MAX_GRID_CELLS:
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
    for supplied in request.inputs.values():
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
        return int(array.size) if array.ndim == 1 else 0
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

    The longest ordinary upload defines genomic coordinates. Shorter uploads
    are treated as local profiles beginning at the selected genomic start, and
    the effective interval is capped to the shortest available spatial input.
    """

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
    if coordinate_inputs:
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
        selected_bins = min(requested_bins, shortest_bins)
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
        if request.whole_genome or request.region_end_kb is None:
            requested_bins = shortest_bins
        else:
            end = _finite_number(request.region_end_kb, "Region end")
            if end <= start:
                raise ValueError(
                    f"Region end must be greater than region start; received Start "
                    f"{start:g} kb and End {end:g} kb."
                )
            requested_bins = int(math.ceil((end - start) / resolution_x))
        selected_bins = min(requested_bins, shortest_bins)
        selection = _SpatialSelection(
            start_kb=start,
            end_kb=start + selected_bins * resolution_x,
            start_index=0,
            stop_index=selected_bins,
            source_bins=reference_bins,
        )

    # A prepared bundle has explicit coordinates even when another input is
    # longer. Its first row is not necessarily the selected region's start.
    coordinated_offsets: dict[str, int] = {}
    for key, _value, _form, source_bins, _prewindowed in spatial_inputs:
        if key not in source_origins:
            continue
        offset = (selection.start_kb - source_origins[key]) / resolution_x
        start_index = int(round(offset))
        if not np.isclose(offset, start_index) or not 0 <= start_index < source_bins:
            raise ValueError(
                "The selected region must start on the prepared result's position axis."
            )
        coordinated_offsets[key] = start_index
        selected_bins = min(selected_bins, source_bins - start_index)
    if selection.stop_index - selection.start_index != selected_bins:
        selection = _SpatialSelection(
            start_kb=selection.start_kb,
            end_kb=selection.start_kb + selected_bins * resolution_x,
            start_index=selection.start_index,
            stop_index=selection.start_index + selected_bins,
            source_bins=selection.source_bins,
        )

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
        elif prewindowed or source_bins < reference_bins:
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
        normalized = _parse_number_sequence(value, setting.label)
        if len(normalized) != 2 or normalized[1] <= normalized[0]:
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
    elif key == "initial_kinetics":
        if isinstance(value, str):
            if len(value) > MAX_INPUT_BYTES:
                raise ValueError("Initial kinetics exceeds the local input limit.")
            try:
                value = json.loads(value)
            except (json.JSONDecodeError, RecursionError) as exc:
                raise ValueError(
                    "Initial kinetics must be a JSON object, for example "
                    '{"firing_rate": 0.001, "fork_speed": 1.4}.'
                ) from exc
        if (
            not isinstance(value, Mapping)
            or not value
            or set(value) - {"firing_rate", "fork_speed"}
        ):
            raise ValueError("Initial kinetics must contain firing_rate and/or fork_speed.")
        normalized = _normalise_bundle(value, setting.label)
        for field_key, field_value in normalized.items():
            if isinstance(field_value, Mapping):
                raise ValueError(f"Initial {field_key} must be a numeric field.")
            field_array = np.asarray(field_value)
            if field_array.ndim > 2:
                raise ValueError(f"Initial {field_key} must be a scalar, vector, or matrix.")
            if (
                np.any(~np.isfinite(field_array))
                or np.any(field_array < 0)
                or (field_key == "fork_speed" and np.any(field_array == 0))
            ):
                raise ValueError(f"Initial {field_key} contains invalid kinetic values.")
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
            interval_bins = int(math.ceil((explicit_end - explicit_start) / resolution_x))
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
        end = implied_end
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


def _validate_grid(
    assumptions: Mapping[str, Any],
    *,
    input_spatial_points: int = 0,
    input_time_points: int = 0,
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
    if n_bins and time_points and n_bins * time_points > MAX_GRID_CELLS:
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
    if shape is not None and shape not in available_shapes:
        choices = ", ".join(available_shapes) or "no configurable shapes"
        raise ValueError(f"Target shape must be one of: {choices} for the selected input forms.")
    if not route.supports_periodic(shape) and request.settings.get("perQ"):
        raise ValueError("The selected target and route support linear geometry only.")
    active_settings = {item.key: item for item in route.active_settings(shape, raw_forms)}
    setting_keys = set(active_settings)
    # Accept an explicit linear boundary from older callers, while exposing
    # no editable periodic control for a branch which cannot execute it.
    if request.settings.get("perQ") is False and any(item.key == "perQ" for item in route.settings):
        setting_keys.add("perQ")
    unknown_settings = tuple(str(key) for key in request.settings if key not in setting_keys)
    if unknown_settings:
        raise ValueError(
            f"The selected route does not accept setting(s): {', '.join(unknown_settings)}."
        )
    assumptions: dict[str, Any] = {}
    for key, value in request.settings.items():
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
        selected_end = spatial_selection.start_kb + input_spatial_points * _request_resolution_x(
            request, route
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
            if explicit_x.size == spatial_selection.source_bins:
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
    _validate_grid(
        assumptions,
        input_spatial_points=input_spatial_points,
        input_time_points=input_time_points,
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
    if total_values > MAX_OUTPUT_VALUES or total_bytes > MAX_OUTPUT_BYTES:
        raise RuntimeError(
            f"Mapped output contains {total_values:,} values / {total_bytes / 1024**2:.1f} MiB; "
            "reduce the requested grid before previewing it in the local app."
        )

    if route.target in outputs:
        primary = route.target
    elif route.target == "kinetic_fields" and "firing_rate" in outputs:
        primary = "firing_rate"
    elif route.target == "fork_densities_xt" and "replicated_fraction_xt" in outputs:
        primary = "replicated_fraction_xt"
    else:
        primary = next(iter(outputs))

    forms: dict[str, DataForm] = {}
    for key, array in outputs.items():
        kinetic_form = None
        if route.target == "kinetic_fields" and key in {"firing_rate", "fork_speed"}:
            metadata_key = "firing_form" if key == "firing_rate" else "speed_form"
            metadata = values if isinstance(values, Mapping) else info
            kinetic_form = {
                "constant": "scalar", "space": "x", "time": "t", "spacetime": "xt",
            }.get(str(metadata.get(metadata_key, "")))
        if kinetic_form is not None:
            forms[key] = cast(DataForm, kinetic_form)
        elif key.endswith("_xt") or (
            route.output_form in {"xt", "bundle"}
            and key in {"firing_rate", "fork_speed"}
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
            if array.ndim == 2 and 1 not in array.shape:
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
    from dnascape.utils import progress_context

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
        ):
            info = mapper(
                source=prepared.source,
                target=prepared.route.target,
                assumptions=execution_assumptions,
                method=prepared.route.method,
                return_info=True,
            )
        info = dict(info)
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
        return MappingResult(
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
