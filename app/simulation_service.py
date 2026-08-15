"""Small, framework-independent bridge to DNAscape's simulation engine.

The NiceGUI page deliberately passes only plain scalar configuration into this
module.  Keeping the scientific call outside the UI makes it straightforward
to run in a worker thread and to test without constructing a browser session.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from threading import Lock
from typing import Any, cast

import numpy as np

ProgressReporter = Callable[[float, str], None]

# These limits protect a local browser session from accidental multi-million
# replicate jobs while still leaving ample room for useful interactive runs.
MAX_SPATIAL_BINS = 1_000_000
MAX_TIME_POINTS = 100_000
MAX_BIN_REPLICATES = 20_000_000
MAX_REPLICATES = 5_000

# ``rsim`` allocates five float32 (time, position) accumulators when detailed
# space-time statistics are requested.  Keep their combined preallocation
# below 64 MiB and independently cap the replicate-scaled work.
DETAILED_XT_ARRAY_COUNT = 5
DETAILED_XT_ITEM_BYTES = np.dtype(np.float32).itemsize
MAX_DETAILED_XT_BYTES = 64 * 1024 * 1024
MAX_DETAILED_XT_CELLS = MAX_DETAILED_XT_BYTES // (DETAILED_XT_ARRAY_COUNT * DETAILED_XT_ITEM_BYTES)
MAX_DETAILED_WORK_CELLS = 50_000_000

CORE_OUTPUT_KEYS = (
    "replication_timing",
    "origin_efficiency",
    "fork_directionality",
)
DISTRIBUTION_OUTPUT_KEYS = (
    "completion_time",
    "num_origins",
)
TIME_OUTPUT_KEYS = (
    "replicated_fraction",
    "total_forks",
    "active_forks",
    "stalled_forks",
    "firing_factors",
)
XT_OUTPUT_KEYS = (
    "replicated_fraction_xt",
    "initiation_events_xt",
    "coalescence_events_xt",
    "right_moving_fork_density_xt",
    "left_moving_fork_density_xt",
)
_RUN_LOCK = Lock()


@dataclass(frozen=True)
class ConstantSimulationRequest:
    """Validated scalar inputs for one local simulation ensemble."""

    firing_rate: float
    fork_speed: float
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
    track_detailed_outputs: bool = False


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
    detailed_outputs: bool = False


@dataclass(frozen=True)
class _ValidatedRequest:
    request: ConstantSimulationRequest
    n_bins: int
    time_points: int
    time_grid: np.ndarray


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


def validate_constant_simulation(request: ConstantSimulationRequest) -> _ValidatedRequest:
    """Validate user-controlled values before allocating arrays or starting work."""

    if not isinstance(request.track_detailed_outputs, bool):
        raise ValueError("Detailed-output tracking must be true or false.")

    firing_rate = _finite(request.firing_rate, "Firing rate")
    if firing_rate < 0:
        raise ValueError("Firing rate must be non-negative.")
    _finite(request.fork_speed, "Fork speed", positive=True)
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
    n_bins = int(math.ceil(ratio))
    if n_bins > MAX_SPATIAL_BINS:
        raise ValueError(
            f"The region contains {n_bins:,} bins; the local app limit is "
            f"{MAX_SPATIAL_BINS:,}. Increase the resolution or narrow the region."
        )

    replicates = _positive_integer(request.replicates, "Replicates")
    if replicates > MAX_REPLICATES:
        raise ValueError(
            f"Replicates exceed the local app limit of {MAX_REPLICATES:,}. "
            "Use a smaller interactive ensemble."
        )
    if n_bins * replicates > MAX_BIN_REPLICATES:
        raise ValueError(
            "This ensemble is too large for an interactive run "
            f"({n_bins:,} bins x {replicates:,} replicates). Reduce the region, "
            "coarsen the resolution, or use fewer replicates."
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
    if time_window < resolution_t:
        raise ValueError("Time window must be at least one time step.")
    steps = int(math.floor(time_window / resolution_t))
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
            estimated_mib = xt_cells * DETAILED_XT_ARRAY_COUNT * DETAILED_XT_ITEM_BYTES / 1024**2
            raise ValueError(
                "Detailed space-time outputs would preallocate "
                f"approximately {estimated_mib:.1f} MiB across five matrices; "
                f"the local limit is {MAX_DETAILED_XT_BYTES / 1024**2:.0f} MiB. "
                "Increase the spatial or time resolution, or narrow the requested windows."
            )
        if xt_cells * replicates > MAX_DETAILED_WORK_CELLS:
            raise ValueError(
                "The requested detailed space-time workload is too large for an "
                "interactive run. Reduce the region, time window, or replicates."
            )
        time_grid = np.arange(steps + 1, dtype=float) * resolution_t
        if time_points > time_grid.size:
            time_grid = np.append(time_grid, time_window)
    else:
        time_grid = np.empty(0, dtype=float)

    stall_rate = _finite(request.stall_rate, "Stall rate")
    if not 0 <= stall_rate <= 1:
        raise ValueError("Stall rate must lie between zero and one.")
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
    if request.track_detailed_outputs:
        expected_shapes.update(
            {
                name: (validated.time_points,)
                for name in TIME_OUTPUT_KEYS
                if name != "firing_factors"
            }
        )
        expected_shapes.update(
            {name: (validated.n_bins, validated.time_points) for name in XT_OUTPUT_KEYS}
        )
        required_keys.extend(name for name in TIME_OUTPUT_KEYS if name != "firing_factors")
        required_keys.extend(XT_OUTPUT_KEYS)

    outputs: dict[str, np.ndarray] = {}
    for name in required_keys:
        value = raw.get(name)
        if value is None:
            raise RuntimeError(f"DNAscape did not return the requested {name!r} output.")
        # Preserve rsim's float32 memmap-backed XT views. Converting these to
        # float64 would silently materialize five additional matrices in RAM.
        array = np.asarray(value) if name in XT_OUTPUT_KEYS else np.asarray(value, dtype=float)
        expected = expected_shapes[name]
        if array.shape != expected:
            raise RuntimeError(
                f"DNAscape returned {name!r} with shape {array.shape}; expected {expected}."
            )
        array.setflags(write=False)
        outputs[name] = array

    if request.track_detailed_outputs and raw.get("firing_factors") is not None:
        firing_factors = np.asarray(raw["firing_factors"], dtype=float)
        expected = (validated.time_points,)
        if firing_factors.shape != expected:
            raise RuntimeError(
                "DNAscape returned 'firing_factors' with shape "
                f"{firing_factors.shape}; expected {expected}."
            )
        firing_factors.setflags(write=False)
        outputs["firing_factors"] = firing_factors
    if request.track_detailed_outputs:
        for name in DISTRIBUTION_OUTPUT_KEYS:
            value = raw.get(name)
            if value is None:
                continue
            distribution = np.asarray(value, dtype=float).reshape(-1)
            if distribution.size != int(request.replicates):
                raise RuntimeError(
                    f"DNAscape returned {name!r} with {distribution.size} values; "
                    f"expected {int(request.replicates)}."
                )
            distribution.setflags(write=False)
            outputs[name] = distribution
    return outputs


def run_constant_simulation(
    request: ConstantSimulationRequest,
    report: ProgressReporter | None = None,
) -> ConstantSimulationResult:
    """Run ``dnascape.simulation.rsim`` for two scalar kinetic inputs."""

    from dnascape import rsim
    from dnascape.utils import progress_context

    simulate = cast(Callable[..., dict[str, Any]], rsim)
    validated = validate_constant_simulation(request)
    request = validated.request
    report = report or (lambda _value, _message: None)
    if not _RUN_LOCK.acquire(blocking=False):
        raise RuntimeError("Another simulation is already running in this local app.")
    try:
        report(0.02, "Preparing simulation...")
        detailed = bool(request.track_detailed_outputs)
        assumptions = {
            "sim_number": int(request.replicates),
            "resolution_x": float(request.resolution_x),
            "resolution_t": float(request.resolution_t),
            "geometry": str(request.geometry).strip().lower(),
            "stall_rate": float(request.stall_rate),
            "tau": float(request.stall_duration),
            "ffiring_nr": request.firing_factors,
            "ffiring_recycle": float(request.firing_factor_recycle),
            "ffiring_forkQ": bool(request.firing_factor_fork_coupling),
            "time_statsQ": detailed,
            "time_stats_xtQ": detailed,
            "max_rep_time": float(request.max_replication_time),
            "seed": None if request.seed is None else int(request.seed),
            "verbose": True,
            "print_every": 1,
            "interval_kb": (float(request.region_start_kb), float(request.region_end_kb)),
            "retain_rep_times_per_sim": False,
            "retain_inter_origin_distances": False,
        }
        if detailed:
            assumptions["time_grid"] = validated.time_grid
        started = time.perf_counter()
        with progress_context(report=report, suppress_output=True):
            raw = simulate(
                source={
                    "firing_rate": float(request.firing_rate),
                    "fork_speed": float(request.fork_speed),
                },
                assumptions=assumptions,
            )
        report(0.98, "Preparing preview...")

        outputs = _collect_outputs(raw, validated)
        replication_timing = outputs["replication_timing"]
        position_kb = float(request.region_start_kb) + np.arange(
            validated.n_bins, dtype=float
        ) * float(request.resolution_x)
        for array in (position_kb, validated.time_grid):
            array.setflags(write=False)
        report(1.0, "Simulation complete")
        return ConstantSimulationResult(
            position_kb=position_kb,
            replication_timing=replication_timing,
            elapsed_seconds=float(time.perf_counter() - started),
            replicates=int(request.replicates),
            geometry=str(request.geometry).strip().lower(),
            time_grid=validated.time_grid,
            outputs=outputs,
            detailed_outputs=detailed,
        )
    finally:
        _RUN_LOCK.release()
