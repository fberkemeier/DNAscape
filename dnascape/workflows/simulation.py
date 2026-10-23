"""Small, framework-independent bridge to DNAscape's simulation engine.

The NiceGUI page passes plain configuration and already-materialized kinetic
fields into this module. Keeping the scientific call outside the UI makes it
straightforward to run in a worker thread and test without a browser session.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from threading import Lock
from typing import Any, cast

import numpy as np

ProgressReporter = Callable[[float, str], None]
CancelChecker = Callable[[], None]

# These limits protect genuine in-memory allocations. Large ensembles remain
# cancellable and are bounded separately by ``MAX_REPLICATES``.
MAX_SPATIAL_BINS = 1_000_000
MAX_TIME_POINTS = 100_000
MAX_REPLICATES = 50_000

# ``rsim`` allocates six float32 (time, position) accumulators when detailed
# space-time statistics are requested. Keep their combined preallocation below
# 64 MiB; ensemble size itself is independently capped and cancellable.
DETAILED_XT_ARRAY_COUNT = 6
DETAILED_XT_ITEM_BYTES = np.dtype(np.float32).itemsize
MAX_DETAILED_XT_BYTES = 64 * 1024 * 1024
MAX_DETAILED_XT_CELLS = MAX_DETAILED_XT_BYTES // (DETAILED_XT_ARRAY_COUNT * DETAILED_XT_ITEM_BYTES)
MAX_RETAINED_REPLICATION_PROFILES = 50
MAX_RETAINED_REPLICATION_PROFILE_BYTES = 64 * 1024 * 1024
# ``rsim`` retains an exact prefix of neighbouring-origin distances and reports
# both the total count and whether that prefix was truncated.
MAX_INTER_ORIGIN_SAMPLES = 2_000_000
MAX_REPLICON_SAMPLES = 2_000_000

CORE_OUTPUT_KEYS = (
    "replication_timing",
    "origin_efficiency",
    "fork_directionality",
)
REPLICATE_OUTPUT_KEYS = (
    "s_phase_duration",
    "completion_time",
    "num_origins",
)
POOLED_DISTRIBUTION_OUTPUT_KEYS = ("inter_origin_distances",)
TIME_DISTRIBUTION_OUTPUT_KEYS = ("replicon_lengths",)
DISTRIBUTION_OUTPUT_KEYS = REPLICATE_OUTPUT_KEYS + POOLED_DISTRIBUTION_OUTPUT_KEYS + TIME_DISTRIBUTION_OUTPUT_KEYS
TIME_OUTPUT_KEYS = (
    "replicated_fraction",
    "total_forks",
    "active_forks",
    "stalled_forks",
    "origin_firing",
    "firing_factors",
)
XT_OUTPUT_KEYS = (
    "replicated_fraction_xt",
    "initiation_events_xt",
    "coalescence_events_xt",
    "right_moving_fork_density_xt",
    "left_moving_fork_density_xt",
    "fork_density_xt",
)
_RUN_LOCK = Lock()


@dataclass(frozen=True)
class ConstantSimulationRequest:
    """Inputs for one local simulation ensemble.

    Kinetic fields use rsim's native forms: scalar, ``(n,)``, ``(1, T)``, or
    ``(n, T)``. The historical class name remains part of the app API even
    though uploaded arrays are now supported.
    """

    firing_rate: float | np.ndarray
    fork_speed: float | np.ndarray
    resolution_x: float
    resolution_t: float
    region_start_kb: float
    region_end_kb: float
    replicates: int
    geometry: str
    time_window: float
    max_replication_time: float
    missing_values: str = "error"
    seed: int | None = 1
    stall_rate: float = 0.0
    stall_duration: float = math.inf
    firing_factors: float | None = None
    firing_factor_recycle: float = 0.0
    firing_factor_fork_coupling: bool = False
    track_time_outputs: bool = False
    track_detailed_outputs: bool = False
    stall_rate_per_kb: float | None = None
    fork_speed_std: float = 0.0
    retained_profile_count: int = 10
    stall_shared_directions: bool = False


@dataclass(frozen=True)
class ConstantSimulationResult:
    """Compact outputs used by the live simulation preview."""

    position_kb: np.ndarray
    replication_timing: np.ndarray
    elapsed_seconds: float
    replicates: int
    geometry: str
    time_grid: np.ndarray
    outputs: Mapping[str, np.ndarray]
    time_outputs: bool = False
    detailed_outputs: bool = False
    replication_time_profiles: np.ndarray | None = None
    replication_timing_std: np.ndarray | None = None
    inter_origin_distance_count: int = 0
    inter_origin_distances_truncated: bool = False
    input_forms: Mapping[str, str] = field(default_factory=dict)
    input_axes: Mapping[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    replicon_length_count: int = 0
    replicon_lengths_truncated: bool = False


@dataclass(frozen=True)
class _ValidatedRequest:
    request: ConstantSimulationRequest
    n_bins: int
    time_points: int
    time_grid: np.ndarray
    retained_profile_count: int


def _finite(value: float, label: str, *, positive: bool = False) -> float:
    number = float(value)
    if not np.isfinite(number) or (positive and number <= 0):
        qualifier = "finite and greater than zero" if positive else "finite"
        raise ValueError(f"{label} must be {qualifier}.")
    return number


def _positive_integer(value: int, label: str, *, allow_zero: bool = False) -> int:
    number = float(value)
    minimum = 0 if allow_zero else 1
    if not np.isfinite(number) or not number.is_integer() or number < minimum:
        qualifier = "a non-negative integer" if allow_zero else "a positive integer"
        raise ValueError(f"{label} must be {qualifier}.")
    return int(number)


def _kinetic_field(
    value: float | np.ndarray,
    label: str,
    *,
    n_bins: int,
    positive: bool,
) -> float | np.ndarray:
    """Validate and normalize one field using rsim's public shape contract."""

    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain numeric values.") from exc
    if array.size == 0:
        raise ValueError(f"{label} cannot be empty.")
    if array.size == 1:
        scalar = _finite(float(array.reshape(-1)[0]), label, positive=positive)
        if not positive and scalar < 0:
            raise ValueError(f"{label} must be non-negative.")
        return scalar
    if array.ndim == 1:
        if array.size != n_bins:
            raise ValueError(
                f"{label} spatial data must contain {n_bins:,} values; got {array.size:,}."
            )
    elif array.ndim == 2:
        is_time = array.shape[0] == 1 and array.shape[1] > 1
        is_space = array.shape[1] == 1 and array.shape[0] > 1
        is_space_time = array.shape[0] > 1 and array.shape[1] > 1
        if (is_space or is_space_time) and array.shape[0] != n_bins:
            raise ValueError(
                f"{label} spatial data must contain {n_bins:,} rows; got {array.shape[0]:,}."
            )
        if not (is_time or is_space or is_space_time):
            raise ValueError(f"{label} must be scalar, (n,), (1, T), (n, 1), or (n, T).")
    else:
        raise ValueError(f"{label} must be scalar, (n,), (1, T), (n, 1), or (n, T).")

    if np.any(~np.isfinite(array)):
        raise ValueError(f"{label} values must be finite.")
    if positive and np.any(array <= 0):
        raise ValueError(f"{label} values must be greater than zero.")
    if not positive and np.any(array < 0):
        raise ValueError(f"{label} values must be non-negative.")
    normalized = array.astype(float, copy=False).view()
    normalized.setflags(write=False)
    return normalized


def validate_constant_simulation(request: ConstantSimulationRequest) -> _ValidatedRequest:
    """Validate user-controlled values before allocating arrays or starting work."""

    if not isinstance(request.track_time_outputs, bool):
        raise ValueError("Time-output tracking must be true or false.")
    if not isinstance(request.track_detailed_outputs, bool):
        raise ValueError("Space-time-output tracking must be true or false.")

    resolution_x = _finite(request.resolution_x, "Spatial resolution", positive=True)
    resolution_t = _finite(request.resolution_t, "Time step", positive=True)
    start = _finite(request.region_start_kb, "Region start")
    end = _finite(request.region_end_kb, "Region end")
    if start < 0:
        raise ValueError("Region start must be non-negative.")
    if end <= start:
        raise ValueError("Region end must be greater than region start.")

    ratio = (end - start) / resolution_x
    if not np.isfinite(ratio):
        raise ValueError("The requested spatial grid is not finite.")
    nearest_bin_count = round(ratio)
    if math.isclose(
        ratio,
        nearest_bin_count,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        # Region ends derived from ``start + n * resolution`` can land a few
        # floating-point ulps above the exact integer ratio (for example,
        # 0.3 / 0.1). Do not manufacture an extra spatial bin in that case.
        ratio = float(nearest_bin_count)
    n_bins = int(math.ceil(ratio))
    if n_bins > MAX_SPATIAL_BINS:
        raise ValueError(
            f"The region contains {n_bins:,} bins; the local app limit is "
            f"{MAX_SPATIAL_BINS:,}. Increase the resolution or narrow the region."
        )

    firing_rate = _kinetic_field(
        request.firing_rate,
        "Firing rate",
        n_bins=n_bins,
        positive=False,
    )
    fork_speed = _kinetic_field(
        request.fork_speed,
        "Fork speed",
        n_bins=n_bins,
        positive=True,
    )
    request = replace(
        request,
        firing_rate=firing_rate,
        fork_speed=fork_speed,
    )

    replicates = _positive_integer(request.replicates, "Simulations (replicates)")
    if replicates > MAX_REPLICATES:
        raise ValueError(
            f"Simulations (replicates) exceed the local app limit of {MAX_REPLICATES:,}. "
            "Use a smaller interactive ensemble."
        )
    retained_profile_count = min(
        _positive_integer(request.retained_profile_count, "Profiles shown", allow_zero=True),
        replicates,
        MAX_RETAINED_REPLICATION_PROFILES,
        MAX_RETAINED_REPLICATION_PROFILE_BYTES
        // (n_bins * np.dtype(float).itemsize),
    )
    geometry = str(request.geometry).strip().lower()
    if geometry not in {"linear", "periodic"}:
        raise ValueError("Geometry must be 'linear' or 'periodic'.")
    missing_values = str(request.missing_values).strip().lower()
    if missing_values not in {"error", "interpolate", "zero"}:
        raise ValueError("Missing values must be 'error', 'interpolate', or 'zero'.")

    time_window = _finite(request.time_window, "Time window", positive=True)
    _finite(
        request.max_replication_time,
        "Maximum replication time",
        positive=True,
    )
    tracks_time = request.track_time_outputs or request.track_detailed_outputs
    if tracks_time:
        if time_window < resolution_t:
            raise ValueError("Time window must be at least one time step.")
        time_ratio = time_window / resolution_t
        if not np.isfinite(time_ratio):
            raise ValueError("The requested time grid is not finite.")
        steps = int(math.floor(time_ratio))
        time_points = steps + 1
        last = steps * resolution_t
        if last < time_window and not np.isclose(last, time_window):
            time_points += 1
        if time_points > MAX_TIME_POINTS:
            raise ValueError(
                f"The time grid contains {time_points:,} points; increase the time step "
                "or shorten the time window."
            )
        if request.track_detailed_outputs:
            xt_cells = n_bins * time_points
            if xt_cells > MAX_DETAILED_XT_CELLS:
                estimated_mib = (
                    xt_cells
                    * DETAILED_XT_ARRAY_COUNT
                    * DETAILED_XT_ITEM_BYTES
                    / 1024**2
                )
                raise ValueError(
                    "Detailed space-time outputs would preallocate "
                    f"approximately {estimated_mib:.1f} MiB across six matrices; "
                    f"the local limit is {MAX_DETAILED_XT_BYTES / 1024**2:.0f} MiB. "
                    "Increase the spatial or time resolution, or narrow the requested windows."
                )
        time_grid = np.arange(steps + 1, dtype=float) * resolution_t
        if time_points > time_grid.size:
            time_grid = np.append(time_grid, time_window)
    else:
        time_points = 0
        time_grid = np.empty(0, dtype=float)

    stall_rate = _finite(request.stall_rate, "Stall rate")
    if not isinstance(request.stall_shared_directions, bool):
        raise ValueError("Shared directional stalling must be a boolean.")
    speed_std = _finite(request.fork_speed_std, "Fork-speed standard deviation")
    if speed_std < 0:
        raise ValueError("Fork-speed standard deviation must be non-negative.")
    if not 0 <= stall_rate <= 1:
        raise ValueError("Legacy stall probability must lie between zero and one.")
    if request.stall_rate_per_kb is not None:
        stall_rate_per_kb = _finite(
            request.stall_rate_per_kb,
            "Stall rate (stalls/kb)",
        )
        if stall_rate_per_kb < 0:
            raise ValueError("Stall rate (stalls/kb) must be non-negative.")
        if stall_rate != 0.0:
            raise ValueError(
                "A legacy per-bin stall probability cannot be combined with "
                "a stall rate in stalls/kb."
            )
    stall_duration = float(request.stall_duration)
    if not (stall_duration == math.inf or np.isfinite(stall_duration) and stall_duration > 0):
        raise ValueError("Stall duration must be positive or infinite.")
    if request.seed is not None:
        _positive_integer(request.seed, "Seed", allow_zero=True)
    if request.firing_factors is not None:
        factors = _finite(request.firing_factors, "Firing-factor pool")
        if factors < 2 or not factors.is_integer():
            raise ValueError("Firing-factor pool must be an integer of at least two.")
    recycle = _finite(request.firing_factor_recycle, "Firing-factor recycle rate")
    if recycle < 0:
        raise ValueError("Firing-factor recycle rate must be non-negative.")

    return _ValidatedRequest(
        request=request,
        n_bins=n_bins,
        time_points=time_points,
        time_grid=time_grid,
        retained_profile_count=retained_profile_count,
    )


def _collect_outputs(
    raw: Mapping[str, Any],
    validated: _ValidatedRequest,
) -> dict[str, np.ndarray]:
    """Collect canonical result arrays and verify their dimensions."""

    request = validated.request
    expected_shapes: dict[str, tuple[int, ...]] = {
        "replication_timing": (validated.n_bins,),
        "origin_efficiency": (validated.n_bins,),
        "fork_directionality": (validated.n_bins,),
    }
    required_keys: list[str] = list(CORE_OUTPUT_KEYS)
    if request.track_time_outputs:
        expected_shapes.update(
            {
                name: (validated.time_points,)
                for name in TIME_OUTPUT_KEYS
                if name != "firing_factors"
            }
        )
        required_keys.extend(name for name in TIME_OUTPUT_KEYS if name != "firing_factors")
    if request.track_detailed_outputs:
        expected_shapes.update(
            {name: (validated.n_bins, validated.time_points) for name in XT_OUTPUT_KEYS}
        )
        required_keys.extend(XT_OUTPUT_KEYS)

    outputs: dict[str, np.ndarray] = {}
    for name in required_keys:
        value = raw.get(name)
        if value is None:
            raise RuntimeError(f"DNAscape did not return the requested {name!r} output.")
        # Preserve rsim's float32 memmap-backed XT views. Converting these to
        # float64 would silently materialize additional matrices in RAM.
        array = np.asarray(value) if name in XT_OUTPUT_KEYS else np.asarray(value, dtype=float)
        expected = expected_shapes[name]
        if array.shape != expected:
            raise RuntimeError(
                f"DNAscape returned {name!r} with shape {array.shape}; expected {expected}."
            )
        array.setflags(write=False)
        outputs[name] = array

    if request.track_time_outputs and raw.get("firing_factors") is not None:
        firing_factors = np.asarray(raw["firing_factors"], dtype=float)
        expected = (validated.time_points,)
        if firing_factors.shape != expected:
            raise RuntimeError(
                "DNAscape returned 'firing_factors' with shape "
                f"{firing_factors.shape}; expected {expected}."
            )
        firing_factors.setflags(write=False)
        outputs["firing_factors"] = firing_factors
    for name in REPLICATE_OUTPUT_KEYS:
        value = raw.get(name)
        if value is None:
            raise RuntimeError(f"DNAscape did not return the requested {name!r} output.")
        distribution = np.asarray(value, dtype=float).reshape(-1)
        if distribution.size != int(request.replicates):
            raise RuntimeError(
                f"DNAscape returned {name!r} with {distribution.size} values; "
                f"expected {int(request.replicates)}."
            )
        distribution.setflags(write=False)
        outputs[name] = distribution
    distribution_keys = POOLED_DISTRIBUTION_OUTPUT_KEYS + (TIME_DISTRIBUTION_OUTPUT_KEYS if request.track_time_outputs else ())
    for name in distribution_keys:
        value = raw.get(name)
        if value is None and name == "inter_origin_distances":
            # ``rsim`` exposes both spellings. Keep the plural name as the app
            # contract while accepting results from an older engine.
            value = raw.get("inter_origin_distance")
        if value is None:
            raise RuntimeError(
                f"DNAscape did not return the requested {name!r} output."
            )
        distribution = np.asarray(value, dtype=float).reshape(-1)
        if np.any(~np.isfinite(distribution)) or np.any(distribution <= 0):
            raise RuntimeError(
                f"DNAscape returned invalid {name.replace('_', ' ')}; expected "
                "finite values greater than zero."
            )
        distribution.setflags(write=False)
        outputs[name] = distribution
    return outputs


def _collect_replication_profile_statistics(
    raw: Mapping[str, Any],
    validated: _ValidatedRequest,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Validate immutable ensemble-profile data kept outside selectable outputs."""

    std_value = raw.get("replication_timing_std")
    if std_value is None:
        raise RuntimeError(
            "DNAscape did not return the requested 'replication_timing_std' output."
        )
    else:
        replication_timing_std = np.asarray(std_value, dtype=float)
        expected_std_shape = (validated.n_bins,)
        if replication_timing_std.shape != expected_std_shape:
            raise RuntimeError(
                "DNAscape returned 'replication_timing_std' with shape "
                f"{replication_timing_std.shape}; expected {expected_std_shape}."
            )
        invalid_std = (~np.isnan(replication_timing_std)) & (
            (~np.isfinite(replication_timing_std)) | (replication_timing_std < 0)
        )
        if np.any(invalid_std):
            raise RuntimeError(
                "DNAscape returned invalid replication-time standard deviations; "
                "expected finite non-negative values or NaN where no replicate finished."
            )
        replication_timing_std.setflags(write=False)

    profiles_value = raw.get("rep_times_per_sim")
    if validated.retained_profile_count == 0:
        if profiles_value is not None and np.asarray(profiles_value).size:
            raise RuntimeError(
                "DNAscape retained per-simulation profiles despite a zero retention limit."
            )
        return None, replication_timing_std
    if profiles_value is None:
        raise RuntimeError(
            "DNAscape did not return the requested per-simulation replication-time profiles."
        )
    replication_time_profiles = np.asarray(profiles_value, dtype=float)
    expected_profiles_shape = (
        validated.retained_profile_count,
        validated.n_bins,
    )
    if replication_time_profiles.shape != expected_profiles_shape:
        raise RuntimeError(
            "DNAscape returned per-simulation replication-time profiles with shape "
            f"{replication_time_profiles.shape}; expected {expected_profiles_shape}."
        )
    replication_time_profiles.setflags(write=False)
    return replication_time_profiles, replication_timing_std


def run_constant_simulation(
    request: ConstantSimulationRequest,
    report: ProgressReporter | None = None,
    cancel: CancelChecker | None = None,
) -> ConstantSimulationResult:
    """Run ``dnascape.simulation.rsim`` with scalar or array kinetic inputs."""

    from dnascape import rsim
    from dnascape.utils import progress_context

    simulate = cast(Callable[..., dict[str, Any]], rsim)
    report = report or (lambda _value, _message: None)
    cancel = cancel or (lambda: None)
    if not _RUN_LOCK.acquire(blocking=False):
        raise RuntimeError("Another simulation is already running in this local app.")
    try:
        started = time.perf_counter()
        cancel()
        report(0.02, "Preparing simulation...")
        validated = validate_constant_simulation(request)
        request = validated.request
        cancel()
        time_outputs = bool(request.track_time_outputs)
        detailed = bool(request.track_detailed_outputs)
        progress_interval = max(1, math.ceil(int(request.replicates) / 500))
        assumptions = {
            "sim_number": int(request.replicates),
            "resolution_x": float(request.resolution_x),
            "resolution_t": float(request.resolution_t),
            "geometry": str(request.geometry).strip().lower(),
            "tau": float(request.stall_duration),
            "ffiring_nr": request.firing_factors,
            "ffiring_recycle": float(request.firing_factor_recycle),
            "ffiring_forkQ": bool(request.firing_factor_fork_coupling),
            "fork_speed_std": float(request.fork_speed_std),
            "stall_shared_directions": bool(request.stall_shared_directions),
            "time_statsQ": time_outputs,
            "time_stats_xtQ": detailed,
            "max_rep_time": float(request.max_replication_time),
            "seed": None if request.seed is None else int(request.seed),
            "verbose": True,
            # Keep progress smooth without forwarding tens of thousands of
            # cross-thread updates for very large ensembles.
            "print_every": progress_interval,
            "interval_kb": (float(request.region_start_kb), float(request.region_end_kb)),
            # These ensemble summaries do not depend on time or space-time
            # tracking. Retain a memory-bounded number of profiles for the
            # preview and a bounded prefix of the pooled IOD distribution.
            "retain_rep_times_per_sim": validated.retained_profile_count > 0,
            "retain_rep_times_limit": validated.retained_profile_count,
            "retain_inter_origin_distances": True,
            "retain_inter_origin_distances_limit": MAX_INTER_ORIGIN_SAMPLES,
            "retain_replicon_lengths_limit": MAX_REPLICON_SAMPLES,
        }
        if request.stall_rate_per_kb is None:
            assumptions["stall_rate"] = float(request.stall_rate)
        else:
            assumptions["stall_rate_per_kb"] = float(request.stall_rate_per_kb)
        if time_outputs or detailed:
            assumptions["time_grid"] = validated.time_grid
        with progress_context(report=report, cancel=cancel, suppress_output=True):
            raw = simulate(
                source={
                    "firing_rate": request.firing_rate,
                    "fork_speed": request.fork_speed,
                },
                assumptions=assumptions,
            )
        cancel()
        report(0.98, "Preparing preview...")

        outputs = _collect_outputs(raw, validated)
        replication_time_profiles, replication_timing_std = (
            _collect_replication_profile_statistics(raw, validated)
        )
        retained_iods = outputs["inter_origin_distances"]
        raw_iod_count = raw.get("inter_origin_distance_count", retained_iods.size)
        try:
            inter_origin_distance_count = int(raw_iod_count)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "DNAscape returned an invalid inter-origin distance count."
            ) from exc
        if inter_origin_distance_count < retained_iods.size:
            raise RuntimeError(
                "DNAscape returned fewer total inter-origin distances than retained values."
            )
        inter_origin_distances_truncated = (
            bool(raw.get("inter_origin_distances_truncated", False))
            or inter_origin_distance_count > retained_iods.size
        )
        retained_replicons = outputs.get("replicon_lengths", np.empty(0))
        try:
            replicon_length_count = int(raw.get("replicon_length_count", retained_replicons.size))
        except (TypeError, ValueError) as exc:
            raise RuntimeError("DNAscape returned an invalid replicon length count.") from exc
        if replicon_length_count < retained_replicons.size:
            raise RuntimeError("DNAscape returned fewer total replicon lengths than retained values.")
        replicon_lengths_truncated = (
            bool(raw.get("replicon_lengths_truncated", False)) or replicon_length_count > retained_replicons.size
        )
        cancel()
        replication_timing = outputs["replication_timing"]
        position_kb = float(request.region_start_kb) + np.arange(
            validated.n_bins, dtype=float
        ) * float(request.resolution_x)
        for array in (position_kb, validated.time_grid):
            array.setflags(write=False)
        report(1.0, "Simulation complete")
        result = ConstantSimulationResult(
            position_kb=position_kb,
            replication_timing=replication_timing,
            elapsed_seconds=float(time.perf_counter() - started),
            replicates=int(request.replicates),
            geometry=str(request.geometry).strip().lower(),
            time_grid=validated.time_grid,
            outputs=outputs,
            time_outputs=time_outputs,
            detailed_outputs=detailed,
            replication_time_profiles=replication_time_profiles,
            replication_timing_std=replication_timing_std,
            inter_origin_distance_count=inter_origin_distance_count,
            inter_origin_distances_truncated=inter_origin_distances_truncated,
            replicon_length_count=replicon_length_count,
            replicon_lengths_truncated=replicon_lengths_truncated,
        )
        from .preview_inputs import add_simulation_input_previews

        return add_simulation_input_previews(result, request)
    finally:
        _RUN_LOCK.release()
