"""Pure configuration, validation, and cost helpers for app workflows.

The Streamlit pages deliberately keep scientific configuration separate from
plot configuration.  This module has no Streamlit dependency, so its rules can
be exercised in ordinary unit tests and reused by other front ends.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Any, Mapping, Sequence

import numpy as np

MIB = 1024**2
MAX_SPATIAL_BINS = 10_000_000
MAX_TIME_GRID_POINTS = 10_000_000


def _bounded_integer(value: Any, name: str, *, minimum: int = 0) -> int:
    """Return an exact finite integer with a concise validation error."""

    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be an integer.") from exc
    if not np.isfinite(numeric) or not numeric.is_integer() or numeric < minimum:
        qualifier = "positive" if minimum == 1 else f"at least {minimum}"
        raise ValueError(f"{name} must be a {qualifier} integer.")
    return int(numeric)


def _normalise_for_hash(value: Any) -> Any:
    """Return a deterministic, JSON-compatible representation of ``value``."""
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return {
            # NumPy arrays implement Python's buffer protocol, although the
            # current NumPy type stubs do not expose that fact to Pyright.
            "__ndarray__": sha256(
                memoryview(array).cast("B")  # pyright: ignore[reportArgumentType]
            ).hexdigest(),
            "dtype": array.dtype.str,
            "shape": list(array.shape),
        }
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "__dataclass_fields__"):
        return _normalise_for_hash(asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): _normalise_for_hash(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_normalise_for_hash(item) for item in value]
    if isinstance(value, set):
        return sorted(_normalise_for_hash(item) for item in value)
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
    return value


def scientific_signature(*parts: Any) -> str:
    """Hash scientific inputs without including presentation-only settings."""
    payload = json.dumps(
        _normalise_for_hash(parts),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def half_open_positions(
    n_bins: int,
    resolution_kb: float,
    start_kb: float = 0.0,
) -> np.ndarray:
    """Return left-edge coordinates for a zero-based half-open genomic grid."""
    n_bins = _bounded_integer(n_bins, "n_bins")
    resolution_kb = float(resolution_kb)
    start_kb = float(start_kb)
    if n_bins > MAX_SPATIAL_BINS:
        raise ValueError(
            f"The spatial grid contains {n_bins:,} bins; the app limit is "
            f"{MAX_SPATIAL_BINS:,}. Coarsen the resolution or narrow the interval."
        )
    if not np.isfinite(resolution_kb) or resolution_kb <= 0:
        raise ValueError("Spatial resolution must be finite and greater than zero.")
    if not np.isfinite(start_kb) or start_kb < 0:
        raise ValueError("Region start must be finite and non-negative.")
    return start_kb + np.arange(n_bins, dtype=float) * resolution_kb


def interval_bin_count(start_kb: float, end_kb: float, resolution_kb: float) -> int:
    """Return bins required for the half-open interval ``[start_kb, end_kb)``."""
    start_kb = float(start_kb)
    end_kb = float(end_kb)
    resolution_kb = float(resolution_kb)
    if not all(np.isfinite(value) for value in (start_kb, end_kb, resolution_kb)):
        raise ValueError("Region bounds and resolution must be finite.")
    if start_kb < 0:
        raise ValueError("Region start must be non-negative.")
    if end_kb <= start_kb:
        raise ValueError("Region end must be greater than region start.")
    if resolution_kb <= 0:
        raise ValueError("Spatial resolution must be greater than zero.")
    ratio = (end_kb - start_kb) / resolution_kb
    if not np.isfinite(ratio) or ratio > MAX_SPATIAL_BINS:
        raise ValueError(
            f"The spatial grid exceeds the app limit of {MAX_SPATIAL_BINS:,} bins. "
            "Coarsen the resolution or narrow the interval."
        )
    return int(math.ceil(ratio))


def time_grid_point_count(end_min: float, resolution_min: float) -> int:
    """Return the exact time-grid size without allocating the grid."""

    end_min = float(end_min)
    resolution_min = float(resolution_min)
    if not np.isfinite(end_min) or end_min <= 0:
        raise ValueError("Time window must be finite and greater than zero.")
    if not np.isfinite(resolution_min) or resolution_min <= 0:
        raise ValueError("Time step must be finite and greater than zero.")
    if end_min < resolution_min:
        raise ValueError("Time window must be at least one time step.")
    ratio = end_min / resolution_min
    if not np.isfinite(ratio) or ratio >= MAX_TIME_GRID_POINTS:
        raise ValueError(
            f"The time grid exceeds the app limit of {MAX_TIME_GRID_POINTS:,} points. "
            "Increase the time step or shorten the time window."
        )
    steps = int(math.floor(ratio))
    last = steps * resolution_min
    points = steps + 1 + int(last < end_min and not np.isclose(last, end_min))
    if points > MAX_TIME_GRID_POINTS:
        raise ValueError(
            f"The time grid exceeds the app limit of {MAX_TIME_GRID_POINTS:,} points. "
            "Increase the time step or shorten the time window."
        )
    return points


def _fill_missing(values: np.ndarray, policy: str, name: str) -> np.ndarray:
    finite = np.isfinite(values)
    if finite.all():
        return values
    policy = str(policy).strip().lower()
    missing = int(values.size - np.count_nonzero(finite))
    if policy == "error":
        raise ValueError(
            f"{name} contains {missing:,} non-finite bin(s). Choose interpolation "
            "or fill them before running the scientific operation."
        )
    if policy == "zero":
        return np.nan_to_num(values, copy=True, nan=0.0, posinf=0.0, neginf=0.0)
    if policy != "interpolate":
        raise ValueError("Missing-value policy must be 'error', 'interpolate', or 'zero'.")
    if not finite.any():
        raise ValueError(f"{name} contains no finite bins to interpolate.")
    indices = np.arange(values.size, dtype=float)
    result = values.copy()
    result[~finite] = np.interp(indices[~finite], indices[finite], values[finite])
    return result


def select_uniform_region(
    profiles: Mapping[str, Sequence[float] | np.ndarray],
    *,
    resolution_kb: float,
    start_kb: float,
    end_kb: float,
    grid_start_kb: float = 0.0,
    missing: str = "error",
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Select aligned profiles without collapsing missing genomic bins.

    Input profiles are interpreted on one regular grid whose first left edge is
    ``grid_start_kb``.  The selected interval follows the package convention
    ``[start_kb, end_kb)``.  Missing values are handled in place on that grid;
    bins are never removed because doing so changes genomic distances.
    """
    if not profiles:
        return {}, half_open_positions(
            interval_bin_count(start_kb, end_kb, resolution_kb),
            resolution_kb,
            start_kb,
        )

    arrays = {
        str(name): np.asarray(values, dtype=float).reshape(-1)
        for name, values in profiles.items()
    }
    lengths = {array.size for array in arrays.values()}
    if len(lengths) != 1:
        detail = ", ".join(f"{name}: {array.size:,}" for name, array in arrays.items())
        raise ValueError(f"Input profiles must have equal bin counts ({detail}).")

    n_bins = lengths.pop()
    full_positions = half_open_positions(n_bins, resolution_kb, grid_start_kb)
    if end_kb <= start_kb:
        raise ValueError("Region end must be greater than region start.")
    mask = (full_positions >= float(start_kb)) & (full_positions < float(end_kb))
    if not np.any(mask):
        raise ValueError("The selected half-open region contains no input bins.")

    selected = {
        name: _fill_missing(array[mask], missing, name)
        for name, array in arrays.items()
    }
    return selected, full_positions[mask]


def build_time_grid(end_min: float, resolution_min: float) -> np.ndarray:
    """Build a finite time grid including zero and the configured upper bound."""
    end_min = float(end_min)
    resolution_min = float(resolution_min)
    point_count = time_grid_point_count(end_min, resolution_min)
    steps = int(math.floor(end_min / resolution_min))
    grid = np.arange(steps + 1, dtype=float) * resolution_min
    if point_count > grid.size:
        grid = np.append(grid, end_min)
    return grid


@dataclass(frozen=True)
class WorkEstimate:
    """Compact cost estimate shown before a mapping or simulation starts."""

    spatial_bins: int
    time_points: int = 0
    realizations: int = 1
    estimated_bytes: int = 0

    @property
    def cells(self) -> int:
        return self.spatial_bins * max(1, self.time_points)

    @property
    def memory_mib(self) -> float:
        return self.estimated_bytes / MIB

    @property
    def expensive(self) -> bool:
        return self.estimated_bytes >= 512 * MIB or self.cells >= 100_000_000

    @property
    def impractical(self) -> bool:
        return self.estimated_bytes >= 4 * 1024 * MIB or self.cells >= 750_000_000


@dataclass(frozen=True)
class MappingConfig:
    """Scientific inputs for one explicit ``rmap`` execution."""

    target: str
    target_form: str
    route_id: str
    resolution_x: float
    resolution_t: float
    region_start_kb: float
    region_end_kb: float
    periodic: bool = False
    maxiter: int = 10
    target_shape: str | None = None
    fit_step: float | None = None
    neighbourhood_range: int | None = None
    error_floor: float | None = None
    error: float | None = None
    deviation_threshold: float | None = None
    smooth_window: int | None = None
    time_max: float | None = None
    automatic_time_horizon_min: float | None = None
    weibull_shape: float = 2.0
    weibull_delay: float = 0.0
    fit_weibull_shape: bool = False
    sim_number: int = 1
    seed: int | None = 1
    alpha: float | None = None
    cfl: float | None = None

    def validate(self) -> None:
        interval_bin_count(self.region_start_kb, self.region_end_kb, self.resolution_x)
        if not np.isfinite(self.resolution_t) or self.resolution_t <= 0:
            raise ValueError("Temporal resolution must be finite and greater than zero.")
        _bounded_integer(self.maxiter, "Iterations")
        _bounded_integer(self.sim_number, "Simulation replicates", minimum=1)
        if self.seed is not None:
            _bounded_integer(self.seed, "Seed")
        if self.fit_step is not None and (
            not np.isfinite(self.fit_step) or self.fit_step <= 0
        ):
            raise ValueError("Fit step must be finite and greater than zero.")
        if not np.isfinite(self.weibull_shape) or self.weibull_shape <= 0:
            raise ValueError("Weibull shape must be finite and greater than zero.")
        if not np.isfinite(self.weibull_delay) or self.weibull_delay < 0:
            raise ValueError("Weibull delay must be finite and non-negative.")
        if self.alpha is not None and (
            not np.isfinite(self.alpha) or self.alpha < 0
        ):
            raise ValueError("Regularization strength must be finite and non-negative.")
        if self.cfl is not None and (
            not np.isfinite(self.cfl) or self.cfl <= 0
        ):
            raise ValueError("CFL factor must be finite and greater than zero.")
        if self.automatic_time_horizon_min is not None and (
            not np.isfinite(self.automatic_time_horizon_min)
            or self.automatic_time_horizon_min <= 0
        ):
            raise ValueError("Automatic time-horizon estimate must be finite and positive.")
        if self.time_max is not None:
            time_grid_point_count(self.time_max, self.resolution_t)
        if self.automatic_time_horizon_min is not None:
            time_grid_point_count(
                self.automatic_time_horizon_min, self.resolution_t
            )
        if self.error_floor is not None and (
            not np.isfinite(self.error_floor) or self.error_floor < 0
        ):
            raise ValueError("Error-floor exponent must be finite and non-negative.")
        if self.error is not None and (
            not np.isfinite(self.error) or self.error < 0
        ):
            raise ValueError("Error threshold must be finite and non-negative.")
        if self.deviation_threshold is not None and (
            not np.isfinite(self.deviation_threshold)
            or self.deviation_threshold < 0
        ):
            raise ValueError("Deviation threshold must be finite and non-negative.")
        if self.smooth_window is not None:
            _bounded_integer(self.smooth_window, "Smoothing window", minimum=1)

    def assumptions(self, *, include_interval: bool = False) -> dict[str, Any]:
        """Return route-aware ``rmap`` assumptions without overriding hidden defaults."""
        self.validate()
        output: dict[str, Any] = {
            "resolution_x": float(self.resolution_x),
            "resolution_t": float(self.resolution_t),
            "target_form": str(self.target_form),
            "perQ": bool(self.periodic),
            "maxiter": int(self.maxiter),
            "verbose": False,
        }
        if include_interval:
            output["interval_kb"] = (
                float(self.region_start_kb),
                float(self.region_end_kb),
            )
        optional = {
            "target_shape": self.target_shape,
            "fit_step": self.fit_step,
            "neighbourhood_range": self.neighbourhood_range,
            "error": self.error,
            "deviation_threshold": self.deviation_threshold,
            "smoothw": self.smooth_window,
        }
        for name, value in optional.items():
            if value is not None:
                output[name] = value
        if self.error_floor is not None:
            output["err_threshold"] = float(self.error_floor)
        if self.target_shape == "weibull":
            output.update(
                weibull_shape=float(self.weibull_shape),
                weibull_delay=float(self.weibull_delay),
                fit_weibull_shape=bool(self.fit_weibull_shape),
            )
        if self.time_max is not None:
            output["time_grid"] = build_time_grid(self.time_max, self.resolution_t)
        if self.alpha is not None:
            output["alpha"] = float(self.alpha)
        if self.cfl is not None:
            output["cfl"] = float(self.cfl)
        return output

    def estimate(self, n_bins: int | None = None) -> WorkEstimate:
        bins = (
            interval_bin_count(
                self.region_start_kb, self.region_end_kb, self.resolution_x
            )
            if n_bins is None
            else _bounded_integer(n_bins, "Spatial bins", minimum=1)
        )
        if bins > MAX_SPATIAL_BINS:
            raise ValueError(
                f"Spatial bins exceed the app limit of {MAX_SPATIAL_BINS:,}."
            )
        effective_horizon = (
            self.time_max
            if self.time_max is not None
            else self.automatic_time_horizon_min
        )
        points = (
            time_grid_point_count(effective_horizon, self.resolution_t)
            if effective_horizon is not None
            else 0
        )
        matrices = 4 if self.target_form == "spacetime" else 1
        return WorkEstimate(
            spatial_bins=bins,
            time_points=points,
            estimated_bytes=int(bins * max(1, points) * matrices * 8),
        )


@dataclass(frozen=True)
class SimulationConfig:
    """Scientific inputs for one explicit ``rsim`` execution."""

    resolution_x: float
    resolution_t: float
    region_start_kb: float
    region_end_kb: float
    realizations: int = 20
    geometry: str = "linear"
    time_max: float = 300.0
    max_rep_time: float = 1200.0
    stall_rate: float = 0.0
    stall_duration: float = math.inf
    firing_factors: float | None = None
    firing_factor_recycle: float = 0.0
    firing_factor_fork_coupling: bool = False
    seed: int | None = 1
    time_stats: bool = False
    spacetime_stats: bool = False
    retain_rep_times_per_sim: bool = False
    retain_inter_origin_distances: bool = False
    dens_block_n: int = 4096

    def validate(self) -> None:
        interval_bin_count(self.region_start_kb, self.region_end_kb, self.resolution_x)
        time_grid_point_count(self.time_max, self.resolution_t)
        _bounded_integer(self.realizations, "Replicates", minimum=1)
        if str(self.geometry).lower() not in {"linear", "periodic"}:
            raise ValueError("Geometry must be 'linear' or 'periodic'.")
        if not np.isfinite(self.max_rep_time) or self.max_rep_time <= 0:
            raise ValueError("Maximum replication time must be finite and positive.")
        if not np.isfinite(self.stall_rate) or not 0 <= self.stall_rate <= 1:
            raise ValueError("Stall probability must lie between 0 and 1 per bin encounter.")
        if not (
            self.stall_duration == math.inf
            or (np.isfinite(self.stall_duration) and self.stall_duration > 0)
        ):
            raise ValueError("Stall duration must be positive or infinite.")
        if self.seed is not None:
            _bounded_integer(self.seed, "Seed")
        if self.firing_factors is not None and (
            not np.isfinite(self.firing_factors) or self.firing_factors < 2
        ):
            raise ValueError("A finite firing-factor pool must contain at least two factors.")
        if (
            not np.isfinite(self.firing_factor_recycle)
            or self.firing_factor_recycle < 0
        ):
            raise ValueError("Firing-factor recycling must be finite and non-negative.")
        _bounded_integer(
            self.dens_block_n, "Space-time block size", minimum=1
        )

    def assumptions(self, *, include_interval: bool = False) -> dict[str, Any]:
        self.validate()
        output: dict[str, Any] = {
            "resolution_x": float(self.resolution_x),
            "resolution_t": float(self.resolution_t),
            "sim_number": int(self.realizations),
            "geometry": str(self.geometry).lower(),
            "stall_rate": float(self.stall_rate),
            "tau": float(self.stall_duration),
            "ffiring_nr": self.firing_factors,
            "ffiring_recycle": float(self.firing_factor_recycle),
            "ffiring_forkQ": bool(self.firing_factor_fork_coupling),
            "max_rep_time": float(self.max_rep_time),
            "seed": None if self.seed is None else int(self.seed),
            "verbose": False,
            "time_statsQ": bool(self.time_stats),
            "time_stats_xtQ": bool(self.spacetime_stats),
            "retain_rep_times_per_sim": bool(self.retain_rep_times_per_sim),
            "retain_inter_origin_distances": bool(
                self.retain_inter_origin_distances
            ),
            "dens_block_n": int(self.dens_block_n),
        }
        if self.time_stats or self.spacetime_stats:
            output["time_grid"] = build_time_grid(self.time_max, self.resolution_t)
        if include_interval:
            output["interval_kb"] = (
                float(self.region_start_kb),
                float(self.region_end_kb),
            )
        return output

    def estimate(self, n_bins: int | None = None) -> WorkEstimate:
        bins = (
            interval_bin_count(
                self.region_start_kb, self.region_end_kb, self.resolution_x
            )
            if n_bins is None
            else _bounded_integer(n_bins, "Spatial bins", minimum=1)
        )
        if bins > MAX_SPATIAL_BINS:
            raise ValueError(
                f"Spatial bins exceed the app limit of {MAX_SPATIAL_BINS:,}."
            )
        points = time_grid_point_count(self.time_max, self.resolution_t)
        retained = (
            bins * int(self.realizations) * 8
            if self.retain_rep_times_per_sim
            else 0
        )
        # rsim currently maintains five float32 space-time accumulators.
        spacetime = bins * points * 5 * 4 if self.spacetime_stats else 0
        summaries = (
            bins * 6 * 8 + points * 8 * 6
            if self.time_stats or self.spacetime_stats
            else bins * 6 * 8
        )
        return WorkEstimate(
            spatial_bins=bins,
            time_points=points if self.time_stats or self.spacetime_stats else 0,
            realizations=int(self.realizations),
            estimated_bytes=int(retained + spacetime + summaries),
        )


def validate_positive_fields(source: Mapping[str, Any]) -> None:
    """Validate non-negative firing and strictly positive fork-speed fields."""
    if "firing_rate" in source:
        firing = np.asarray(source["firing_rate"], dtype=float)
        if not np.all(np.isfinite(firing)) or np.any(firing < 0):
            raise ValueError("Firing rate must contain finite, non-negative values.")
    if "fork_speed" in source:
        speed = np.asarray(source["fork_speed"], dtype=float)
        if not np.all(np.isfinite(speed)) or np.any(speed <= 0):
            raise ValueError("Fork speed must contain finite values greater than zero.")
