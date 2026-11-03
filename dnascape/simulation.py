"""Stochastic replication simulation engines."""

import heapq
import inspect
import os
import tempfile
import time
import weakref

import numpy as np

from .utils import check_cancelled, etaf, report_measured_work

_SOURCE_ALIASES = {
    "firing_rate": "initiation_rate",  # Read older structured inputs.
    "fr": "initiation_rate",
    "firing": "initiation_rate",
    "origin_rate": "initiation_rate",
    "ori_rate": "initiation_rate",
    "initiation": "initiation_rate",
    "initiation_rate": "initiation_rate",
    "ir": "initiation_rate",
    "fork_speeds": "fork_speed",
    "speed": "fork_speed",
    "v": "fork_speed",
    "fs": "fork_speed",
}

_ASSUMPTION_ALIASES = {
    "bin_size": "resolution",
    "resolution_kb": "resolution",
    "resolution_x": "resolution_space",
    "resolution_x_kb": "resolution_space",
    "resolution_space_kb": "resolution_space",
    "resolution_t": "resolution_time",
    "resolution_t_min": "resolution_time",
    "dt": "resolution_time",
    "periodic": "perQ",
    "perq": "perQ",
    "time_stats": "time_statsQ",
    "time_stats_xt": "time_stats_xtQ",
    "time_stats_dens": "time_stats_densQ",
    "stall_density": "stall_rate_per_kb",
    "stall_density_per_kb": "stall_rate_per_kb",
    "stalls_per_kb": "stall_rate_per_kb",
    "stall_probability": "stall_rate",
    "stall_probability_per_bin": "stall_rate",
}

# Keep the legacy default grid immutable. ``rsim`` converts it to an ndarray
# before use, so callers cannot accidentally mutate the default shared by
# later simulations.
_DEFAULT_TIME_GRID = tuple(float(value) for value in range(1501))


def _normalise_structured_key(name, aliases):
    text = str(name).strip().replace("-", "_")
    key = text.lower()
    return aliases.get(key, text)


def _n_bins_from_interval(resolution_space, genome_length=None, interval_kb=None):
    if interval_kb is not None:
        if len(interval_kb) != 2:
            raise ValueError("interval_kb must be a (start_kb, end_kb) pair.")
        start_kb, end_kb = map(float, interval_kb)
        genome_length = end_kb - start_kb
    if genome_length is None:
        return None
    genome_length = float(genome_length)
    if not np.isfinite(genome_length) or genome_length <= 0:
        raise ValueError("genome_length must be a finite positive value in kb.")
    n = int(np.ceil(genome_length / resolution_space))
    if n <= 0:
        raise ValueError("genome_length/resolution_x must define at least one bin.")
    return n


def _resolve_n_bins(
    initiation_rate,
    fork_speed,
    n_bins=None,
    resolution_space=1.0,
    genome_length=None,
    interval_kb=None,
):
    """Infer the spatial genome length from spatial or space-time inputs."""
    candidates = []

    def add_candidate(field, name):
        if field is None:
            return
        arr = np.asarray(field)
        if arr.size == 1:
            return
        if arr.ndim == 1:
            candidates.append((name, int(arr.size)))
            return
        if arr.ndim == 2:
            if arr.shape[0] == 1 and arr.shape[1] > 1:
                return
            if arr.shape[1] == 1 and arr.shape[0] > 1:
                candidates.append((name, int(arr.shape[0])))
                return
            candidates.append((name, int(arr.shape[0])))
            return
        raise ValueError(f"{name} must be scalar, (n,), (1,T), (n,1), or (n,T).")

    add_candidate(initiation_rate, "initiation_rate")
    add_candidate(fork_speed, "fork_speed")
    interval_n = _n_bins_from_interval(
        resolution_space,
        genome_length=genome_length,
        interval_kb=interval_kb,
    )
    if interval_n is not None:
        candidates.append(("interval", interval_n))

    if n_bins is not None:
        n = int(n_bins)
        if n <= 0:
            raise ValueError("n_bins must be a positive integer.")
        mismatches = [f"{name} has n={candidate}" for name, candidate in candidates if candidate != n]
        if mismatches:
            raise ValueError("n_bins is inconsistent with supplied fields: " + "; ".join(mismatches))
        return n

    unique = {candidate for _, candidate in candidates}
    if len(unique) == 1:
        return unique.pop()
    if len(unique) > 1:
        detail = ", ".join(f"{name}: n={candidate}" for name, candidate in candidates)
        raise ValueError(f"Spatial field lengths disagree ({detail}).")
    raise ValueError(
        "Cannot infer genome length from scalar or time-only inputs. "
        "Pass n_bins, genome_length, interval_kb, or provide at least one spatial "
        "profile or space-time field."
    )


def _as_rate_field(field, name, n_bins):
    arr = np.asarray(field, dtype=float)
    if arr.size == 1:
        value = float(arr.reshape(-1)[0])
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} scalar must be finite and >= 0.")
        return "scalar", value, None
    if arr.ndim == 1:
        if arr.size != n_bins:
            raise ValueError(f"{name} (n,) must have length {n_bins}, got {arr.size}.")
        if np.any(~np.isfinite(arr)) or np.any(arr < 0):
            raise ValueError(f"{name} (n,) must be finite and >= 0.")
        return "x", arr.astype(float, copy=False), None
    if arr.ndim == 2:
        if arr.shape[0] == 1 and arr.shape[1] > 1:
            values_t = arr.reshape(-1)
            if np.any(~np.isfinite(values_t)) or np.any(values_t < 0):
                raise ValueError(f"{name} (1,T) must be finite and >= 0.")
            return "t", values_t.astype(float, copy=False), int(values_t.size)
        if arr.shape[1] == 1 and arr.shape[0] > 1:
            values_x = arr.reshape(-1)
            if values_x.size != n_bins:
                raise ValueError(f"{name} (n,1) must have length {n_bins}, got {values_x.size}.")
            if np.any(~np.isfinite(values_x)) or np.any(values_x < 0):
                raise ValueError(f"{name} (n,1) must be finite and >= 0.")
            return "x", values_x.astype(float, copy=False), None
        if arr.shape[0] != n_bins:
            raise ValueError(f"{name} (n,T) must have n={n_bins} rows, got {arr.shape[0]}.")
        if np.any(~np.isfinite(arr)) or np.any(arr < 0):
            raise ValueError(f"{name} (n,T) must be finite and >= 0.")
        return "xt", arr.T.astype(float, copy=False), int(arr.shape[1])
    raise ValueError(f"{name} must be scalar, (n,), (1,T), (n,1), or (n,T).")


def _as_speed_field(field, name, n_bins):
    arr = np.asarray(field, dtype=float)
    if arr.size == 1:
        value = float(arr.reshape(-1)[0])
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} scalar must be finite and > 0.")
        return "scalar", value, None
    if arr.ndim == 1:
        if arr.size != n_bins:
            raise ValueError(f"{name} (n,) must have length {n_bins}, got {arr.size}.")
        if np.any(~np.isfinite(arr)) or np.any(arr <= 0):
            raise ValueError(f"{name} (n,) must be finite and > 0.")
        return "x", arr.astype(float, copy=False), None
    if arr.ndim == 2:
        if arr.shape[0] == 1 and arr.shape[1] > 1:
            values_t = arr.reshape(-1)
            if np.any(~np.isfinite(values_t)) or np.any(values_t <= 0):
                raise ValueError(f"{name} (1,T) must be finite and > 0.")
            return "t", values_t.astype(float, copy=False), int(values_t.size)
        if arr.shape[1] == 1 and arr.shape[0] > 1:
            values_x = arr.reshape(-1)
            if values_x.size != n_bins:
                raise ValueError(f"{name} (n,1) must have length {n_bins}, got {values_x.size}.")
            if np.any(~np.isfinite(values_x)) or np.any(values_x <= 0):
                raise ValueError(f"{name} (n,1) must be finite and > 0.")
            return "x", values_x.astype(float, copy=False), None
        if arr.shape[0] != n_bins:
            raise ValueError(f"{name} (n,T) must have n={n_bins} rows, got {arr.shape[0]}.")
        if np.any(~np.isfinite(arr)) or np.any(arr <= 0):
            raise ValueError(f"{name} (n,T) must be finite and > 0.")
        return "xt", arr.T.astype(float, copy=False), int(arr.shape[1])
    raise ValueError(f"{name} must be scalar, (n,), (1,T), (n,1), or (n,T).")


def _as_inverse_speed_field(field, name, n_bins):
    """Validate a speed field and prepare its invariant reciprocal."""
    kind, values, time_points = _as_speed_field(field, name, n_bins)
    return kind, 1.0 / values, time_points


def _validated_fork_speed_std(value):
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("fork_speed_std must be finite and non-negative (kb/min).") from exc
    if not np.isfinite(value) or value < 0:
        raise ValueError("fork_speed_std must be finite and non-negative (kb/min).")
    return value


def _positive_speed_samples(rng, mean, standard_deviation, *, size):
    """Positive lognormal speeds with the requested arithmetic mean and SD.

    A normal distribution conditioned on v > 0 has nonzero density at zero,
    giving 1/v an infinite mean and arbitrarily large bin-crossing artifacts.
    Matching lognormal moments avoids that singularity without a speed floor.
    """
    means = np.broadcast_to(np.asarray(mean, dtype=float), (size,))
    if standard_deviation == 0:
        return means.copy()
    log_means = np.log(means)
    log_variance = np.logaddexp(0., 2 * (np.log(standard_deviation) - log_means))
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        speeds = rng.lognormal(log_means - .5 * log_variance, np.sqrt(log_variance), size=size)
    if np.any(~np.isfinite(speeds)) or np.any(speeds <= 0):
        raise ValueError(
            "Speed variation exceeds the numeric range. Reduce the standard deviation or increase the input speed."
        )
    return speeds


def _validated_stall_shared_directions(value):
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError("stall_shared_directions must be a boolean.")
    return bool(value)


def _legacy_initiation_input(options):
    value = options.pop("firing_rate", None)
    if options:
        raise TypeError(f"Unexpected simulation argument: {next(iter(options))!r}.")
    return value


def _normalise_initiation_rate(initiation_rate=None, ori_rate=None, legacy_rate=None):
    supplied = [
        name
        for name, value in (
            ("initiation_rate", initiation_rate),
            ("ori_rate", ori_rate),
            ("legacy_rate", legacy_rate),
        )
        if value is not None
    ]
    if len(supplied) > 1:
        raise ValueError(
            "Pass initiation rate only once, using initiation_rate."
        )
    if initiation_rate is not None:
        return initiation_rate
    if ori_rate is not None:
        return ori_rate
    if legacy_rate is not None:
        return legacy_rate
    raise ValueError("An initiation_rate input is required.")


def _validated_time_grid(time_grid, *, required=False):
    if time_grid is None:
        if required:
            raise ValueError("time statistics require a time_grid array.")
        return None
    values = np.asarray(time_grid, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("time_grid must be a non-empty one-dimensional array.")
    if np.any(~np.isfinite(values)):
        raise ValueError("time_grid must contain only finite values.")
    if values.size > 1 and np.any(np.diff(values) <= 0):
        raise ValueError("time_grid must be strictly increasing.")
    return values


def _validated_positive_integer(value, name):
    """Return a positive integer without silently truncating fractions."""
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer.") from exc
    if not np.isfinite(numeric) or numeric <= 0 or not numeric.is_integer():
        raise ValueError(f"{name} must be a positive integer.")
    return int(numeric)


def _validated_nonnegative_integer(value, name):
    """Return a non-negative integer without silently truncating fractions."""
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a non-negative integer.") from exc
    if not np.isfinite(numeric) or numeric < 0 or not numeric.is_integer():
        raise ValueError(f"{name} must be a non-negative integer.")
    return int(numeric)


def stall_probability_per_bin(stall_rate_per_kb, resolution_space):
    """Convert a physical stall density into a per-bin encounter probability.

    Stalls are treated as a Poisson process along genomic distance. The chance
    of at least one encounter while crossing a bin of width
    ``resolution_space`` is ``1 - exp(-rate * width)``. The stochastic engine
    represents that encounter with one Bernoulli draw per bin and direction.
    """

    try:
        rate = float(stall_rate_per_kb)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "stall_rate_per_kb must be a finite non-negative rate."
        ) from exc
    if not np.isfinite(rate) or rate < 0.0:
        raise ValueError("stall_rate_per_kb must be a finite non-negative rate.")
    try:
        resolution = float(resolution_space)
    except (TypeError, ValueError) as exc:
        raise ValueError("resolution_x must be a finite positive number.") from exc
    if not np.isfinite(resolution) or resolution <= 0.0:
        raise ValueError("resolution_x must be a finite positive number.")
    with np.errstate(over="ignore"):
        exposure = rate * resolution
    return float(-np.expm1(-exposure))


def _resolve_stall_probability(stall_rate, stall_rate_per_kb, resolution_space):
    """Resolve legacy per-bin and physical per-kb stalling inputs."""

    try:
        probability = 0.0 if stall_rate is None else float(stall_rate)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "stall_rate must be a finite probability between 0 and 1."
        ) from exc
    if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("stall_rate must be a finite probability between 0 and 1.")
    if stall_rate_per_kb is None:
        return probability
    if probability != 0.0:
        raise ValueError(
            "A non-zero stall_rate (probability per bin) cannot be combined "
            "with stall_rate_per_kb (stalls/kb)."
        )
    return stall_probability_per_bin(stall_rate_per_kb, resolution_space)


def _close_and_unlink_memmap(mapping, path):
    """Close a memmap handle before unlinking its temporary backing file."""
    try:
        mapping.flush()
    finally:
        mapping.close()
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


def _phase_timing_summary(fired_times_min, replication_time_min):
    """Return absolute completion time and elapsed S-phase duration.

    The simulation clock starts at zero, before the first stochastic origin
    firing. ``completion_time`` therefore remains an absolute timestamp,
    while the biological S-phase duration starts at the first *realised*
    initiation and ends when replication completes. A realization without a
    fired origin never starts or completes S phase, so both values are NaN.
    """

    fired = np.asarray(fired_times_min, dtype=float).reshape(-1)
    realised = fired[np.isfinite(fired)]
    replication = np.asarray(replication_time_min, dtype=float).reshape(-1)
    if realised.size == 0 or replication.size == 0 or np.isnan(replication).any():
        return np.nan, np.nan

    completion = float(np.max(replication))
    first_initiation = float(np.min(realised))
    duration = completion - first_initiation
    if np.isnan(duration) or duration < 0.0:
        # A negative interval can only arise from a censored/inconsistent
        # realization (for example, a cap preceding the first initiation).
        return np.nan, np.nan
    return completion, duration


def _replicon_lengths(src, arrival_times, fired_idx, resolution_space, max_rep_time):
    """Completed DNA domains owned by individual fired origins, in kb.

    Ownership joins domains across the periodic seam. Exclude a domain if any
    of its bins remain unreplicated at the simulation's stopping time.
    """
    if not len(fired_idx):
        return np.empty(0, dtype=float)
    lengths = np.bincount(src, minlength=len(src))[fired_idx]
    last_arrival = np.full(len(src), -np.inf)
    np.maximum.at(last_arrival, src, arrival_times)
    completed = np.isfinite(last_arrival[fired_idx])
    if max_rep_time is not None:
        completed &= last_arrival[fired_idx] <= max_rep_time
    return lengths[completed].astype(float) * resolution_space


def simulate_replication(
    initiation_rate=None,
    fork_speed=1.4,
    resolution_space=1.0,
    resolution_time=5.0,
    resolution_x=None,
    resolution_t=None,
    rng=None,
    perQ=False,
    stall_rate=0.0,
    tau=np.inf,
    ffiring_nr=None,
    ffiring_recycle=0.0,
    ffiring_forkQ=False,
    time_statsQ=False,
    time_stats_xtQ=False,  # space-time stats are handled in rsim; kept for API symmetry
    time_stats_densQ=None,  # deprecated alias; kept for backwards compatibility
    time_grid=None,
    max_rep_time: "float | None" = 1200.0,
    n_bins=None,
    genome_length=None,
    interval_kb=None,
    ori_rate=None,
    _firing_cumulative_hazard=None,
    _prepared_firing_field=None,
    _prepared_inverse_speed_field=None,
    stall_rate_per_kb=None,
    _compute_inter_origin_distances=True,
    fork_speed_std=0.0,
    stall_shared_directions=False,
    **legacy_kwargs,
):
    """Run one stochastic replication simulation.

    ``initiation_rate`` and ``fork_speed`` may be scalar, spatial ``(n,)``,
    time-only ``(1, T)``, or space-time ``(n, T)`` fields. If neither input
    contains a spatial axis, pass ``n_bins``, ``genome_length``, or
    ``interval_kb``. ``stall_rate_per_kb`` is a physical stall density and is
    converted to a per-bin probability after resolving the spatial resolution.
    ``stall_rate`` remains the legacy probability-per-bin input.
    ``fork_speed_std`` enables positive lognormal speeds matching the local
    input mean and the requested SD in kb/min, independently by bin, direction
    and replicate (also by time slice for temporal fields). Temporal stochastic
    speeds are integrated across each crossing. ``stall_shared_directions``
    makes stall positions and recovery delays identical in both directions.
    """
    rng = np.random.default_rng() if rng is None else rng
    fork_speed_std = _validated_fork_speed_std(fork_speed_std)
    stall_shared_directions = _validated_stall_shared_directions(stall_shared_directions)
    if resolution_x is not None:
        resolution_space = resolution_x
    if resolution_t is not None:
        resolution_time = resolution_t

    resolution_space = float(resolution_space)
    if not np.isfinite(resolution_space) or resolution_space <= 0:
        raise ValueError("resolution_x must be a finite positive number.")

    resolution_time = float(resolution_time)
    if not np.isfinite(resolution_time) or resolution_time <= 0:
        raise ValueError("resolution_t must be a finite positive number.")

    stall_rate = _resolve_stall_probability(
        stall_rate,
        stall_rate_per_kb,
        resolution_space,
    )
    tau = float(tau)
    if np.isnan(tau) or tau <= 0:
        raise ValueError("tau must be positive or np.inf.")
    if max_rep_time is not None:
        max_rep_time = float(max_rep_time)
        if not np.isfinite(max_rep_time) or max_rep_time <= 0:
            raise ValueError("max_rep_time must be finite and positive, or None.")

    initiation_rate = _normalise_initiation_rate(
        initiation_rate=initiation_rate,
        ori_rate=ori_rate,
        legacy_rate=_legacy_initiation_input(legacy_kwargs),
    )
    n = _resolve_n_bins(
        initiation_rate,
        fork_speed,
        n_bins=n_bins,
        resolution_space=resolution_space,
        genome_length=genome_length,
        interval_kb=interval_kb,
    )

    if _prepared_firing_field is None:
        I_kind, I_val, _TI = _as_rate_field(initiation_rate, "initiation_rate", n_bins=n)
    else:
        I_kind, I_val, _TI = _prepared_firing_field

    if _prepared_inverse_speed_field is None:
        v_kind, inv_v, Tv = _as_inverse_speed_field(
            fork_speed,
            "fork_speed",
            n_bins=n,
        )
    else:
        v_kind, inv_v, Tv = _prepared_inverse_speed_field

    inv_v_values = np.asarray(inv_v, dtype=float)
    inv_v_scalar = None
    inv_v_x = None
    inv_v_t = None
    inv_v_xt = None
    if v_kind == "scalar":
        if inv_v_values.size != 1:
            raise RuntimeError("A scalar prepared fork-speed field must contain one value.")
        inv_v_scalar = float(inv_v_values.reshape(-1)[0])
    elif v_kind == "x":
        if inv_v_values.shape != (n,):
            raise RuntimeError(f"A spatial prepared fork-speed field must have shape ({n},).")
        inv_v_x = inv_v_values
    elif v_kind == "t":
        if inv_v_values.ndim != 1 or inv_v_values.size == 0:
            raise RuntimeError("A temporal prepared fork-speed field must be one-dimensional.")
        inv_v_t = inv_v_values
        Tv = int(inv_v_t.size)
    elif v_kind == "xt":
        if inv_v_values.ndim != 2 or inv_v_values.shape[1] != n:
            raise RuntimeError(
                f"A space-time prepared fork-speed field must have shape (time, {n})."
            )
        inv_v_xt = inv_v_values
        Tv = int(inv_v_xt.shape[0])
    else:
        raise RuntimeError(f"Unknown prepared fork-speed field kind: {v_kind!r}.")

    p = stall_rate
    shared_stalls = stall_shared_directions and p > 0
    stall_left_mask = rng.random(n) < p
    stall_right_mask = stall_left_mask if shared_stalls else rng.random(n) < p
    stall_delay_left = np.zeros(n, float)
    stall_delay_right = np.zeros(n, float)
    if p > 0.0:
        if np.isfinite(tau):
            if stall_left_mask.any():
                stall_delay_left[stall_left_mask] = rng.exponential(scale=tau, size=int(stall_left_mask.sum()))
            if shared_stalls:
                stall_delay_right[:] = stall_delay_left
            elif stall_right_mask.any():
                stall_delay_right[stall_right_mask] = rng.exponential(scale=tau, size=int(stall_right_mask.sum()))
        else:
            stall_delay_left[stall_left_mask] = np.inf
            stall_delay_right[stall_right_mask] = np.inf

    def _row_for_time(t):
        if not np.isfinite(t) or t < 0.0:
            return 0
        r = int(t // resolution_time)
        if inv_v_t is not None or inv_v_xt is not None:
            if Tv is None:
                raise RuntimeError("A time-dependent speed field has no time axis.")
            if r >= Tv:
                r = Tv - 1
        return r

    def _edge_forward(t_here, k_to):
        if inv_v_xt is not None:
            r = _row_for_time(t_here)
            return resolution_space * inv_v_xt[r, k_to] + stall_delay_left[k_to]
        if inv_v_t is not None:
            r = _row_for_time(t_here)
            return resolution_space * inv_v_t[r] + stall_delay_left[k_to]
        if inv_v_x is not None:
            return resolution_space * inv_v_x[k_to] + stall_delay_left[k_to]
        if inv_v_scalar is None:
            raise RuntimeError("The prepared fork-speed field has an invalid kind.")
        return resolution_space * inv_v_scalar + stall_delay_left[k_to]

    def _edge_backward(t_here, k_to):
        if inv_v_xt is not None:
            r = _row_for_time(t_here)
            return resolution_space * inv_v_xt[r, k_to] + stall_delay_right[k_to]
        if inv_v_t is not None:
            r = _row_for_time(t_here)
            return resolution_space * inv_v_t[r] + stall_delay_right[k_to]
        if inv_v_x is not None:
            return resolution_space * inv_v_x[k_to] + stall_delay_right[k_to]
        if inv_v_scalar is None:
            raise RuntimeError("The prepared fork-speed field has an invalid kind.")
        return resolution_space * inv_v_scalar + stall_delay_right[k_to]

    if fork_speed_std > 0:
        if v_kind in {"scalar", "x"}:
            means = 1.0 / inv_v_values
            inverse_forward = 1.0 / _positive_speed_samples(rng, means, fork_speed_std, size=n)
            inverse_backward = 1.0 / _positive_speed_samples(rng, means, fork_speed_std, size=n)

            def _edge_forward(t_here, k_to):
                return resolution_space * inverse_forward[k_to] + stall_delay_left[k_to]

            def _edge_backward(t_here, k_to):
                return resolution_space * inverse_backward[k_to] + stall_delay_right[k_to]
        else:
            # Sample only encountered space-time cells. Reuse each draw when
            # propagation is revisited (for example with fork-coupled factors),
            # without allocating two additional full space-time matrices.
            forward_samples = {}
            backward_samples = {}

            def stochastic_speed(row, k_to, samples):
                key = row * n + k_to
                if key not in samples:
                    inverse = inv_v_t[row] if inv_v_t is not None else inv_v_xt[row, k_to]
                    samples[key] = _positive_speed_samples(rng, 1.0 / inverse, fork_speed_std, size=1)[0]
                return samples[key]

            def stochastic_crossing(t_here, k_to, samples):
                # Integrate distance over clock slices. Freezing entry speed
                # over a long crossing can let a later fork arrive first,
                # violating the FIFO condition used by propagation.
                if not np.isfinite(t_here):
                    return np.inf
                current = t_here
                remaining = resolution_space
                row = _row_for_time(current)
                while True:
                    if row % 256 == 0:
                        check_cancelled()
                    speed = stochastic_speed(row, k_to, samples)
                    if row == Tv - 1:
                        return current - t_here + remaining / speed
                    boundary = (row + 1) * resolution_time
                    available_time = max(0., boundary - current)
                    distance = available_time * speed
                    if remaining <= distance:
                        return current - t_here + remaining / speed
                    remaining -= distance
                    current = max(current, boundary)
                    row += 1

            def _edge_forward(t_here, k_to):
                return stochastic_crossing(t_here, k_to, forward_samples) + stall_delay_left[k_to]

            def _edge_backward(t_here, k_to):
                return stochastic_crossing(t_here, k_to, backward_samples) + stall_delay_right[k_to]

    # firing proposals t0
    t0 = np.full(n, np.inf, float)
    if I_kind == "x":
        rate = np.asarray(I_val, dtype=float)
        pos = rate > 0
        if np.any(pos):
            t0[pos] = rng.exponential(scale=1.0 / rate[pos])
    elif I_kind == "scalar":
        scalar_rate = float(np.asarray(I_val, dtype=float).reshape(-1)[0])
        if scalar_rate > 0:
            t0[:] = rng.exponential(scale=1.0 / scalar_rate, size=n)
    elif I_kind == "t":
        rates_t = np.asarray(I_val, dtype=float)
        u = rng.random(n)
        target = -np.log(u)
        H = (
            np.asarray(_firing_cumulative_hazard, dtype=float)
            if _firing_cumulative_hazard is not None
            else np.cumsum(rates_t * resolution_time)
        )
        if H.shape != rates_t.shape:
            raise ValueError("Prepared firing cumulative hazard does not match initiation_rate.")
        hit = target <= H[-1] if H.size else np.zeros(n, dtype=bool)
        if np.any(hit):
            kh = np.flatnonzero(hit).astype(np.int64, copy=False)
            jh = np.searchsorted(H, target[hit], side="left").astype(np.int64, copy=False)
            prevH = np.zeros(kh.size, float)
            m0 = jh > 0
            if np.any(m0):
                prevH[m0] = H[jh[m0] - 1]
            lam = rates_t[jh]
            frac = np.ones_like(lam)
            poslam = lam > 0
            frac[poslam] = (target[kh[poslam]] - prevH[poslam]) / (lam[poslam] * resolution_time)
            frac = np.clip(frac, 0.0, 1.0)
            t0[kh] = (jh.astype(float) + frac) * resolution_time
    elif I_kind == "xt":
        rates_xt = np.asarray(I_val, dtype=float)
        u = rng.random(n)
        target = -np.log(u)
        H = (
            np.asarray(_firing_cumulative_hazard, dtype=float)
            if _firing_cumulative_hazard is not None
            else np.cumsum(rates_xt * resolution_time, axis=0)
        )
        if H.shape != rates_xt.shape:
            raise ValueError("Prepared firing cumulative hazard does not match initiation_rate.")
        ge = H >= target[None, :]
        hit = ge.any(axis=0)
        j = np.argmax(ge, axis=0).astype(np.int64, copy=False)
        if np.any(hit):
            jh = j[hit]
            kh = np.flatnonzero(hit).astype(np.int64, copy=False)
            prevH = np.zeros(kh.size, float)
            m0 = jh > 0
            if np.any(m0):
                prevH[m0] = H[jh[m0] - 1, kh[m0]]
            lam = rates_xt[jh, kh]
            frac = np.ones_like(lam)
            poslam = lam > 0
            frac[poslam] = (target[kh[poslam]] - prevH[poslam]) / (lam[poslam] * resolution_time)
            frac = np.clip(frac, 0.0, 1.0)
            t0[kh] = (jh.astype(float) + frac) * resolution_time
    else:
        raise RuntimeError(f"Unknown prepared initiation-rate field kind: {I_kind!r}.")

    # firing-factor gating (unchanged)
    use_factors = ffiring_nr is not None
    if use_factors:
        Amax = float(ffiring_nr)
        if not np.isfinite(Amax) or Amax <= 0:
            raise ValueError("ffiring_nr must be a finite positive number.")
        cost = 2.0
        if Amax < cost:
            raise ValueError("ffiring_nr must be >= 2 if modelling two forks per origin firing.")

        recycle_rate = float(ffiring_recycle)
        if not np.isfinite(recycle_rate) or recycle_rate < 0:
            raise ValueError("ffiring_recycle must be a finite non-negative number.")
        r = 0.0 if ffiring_forkQ else recycle_rate

        def recover(A_now, dt):
            if r == 0.0 or dt <= 0.0:
                return A_now
            return Amax - (Amax - A_now) * np.exp(-r * dt)

        def wait_time_to_cost(A_now):
            if A_now >= cost:
                return 0.0
            if r == 0.0:
                return np.inf
            denom = Amax - A_now
            num = Amax - cost
            if denom <= 0.0 or num <= 0.0:
                return np.inf
            return (1.0 / r) * np.log(denom / num)

        def propagate_from_times(t):
            if not perQ:
                g = np.empty(n, float)
                src = np.empty(n, np.int32)
                g[0], src[0] = t[0], 0
                for k in range(1, n):
                    cand = g[k - 1] + _edge_forward(g[k - 1], k)
                    if t[k] <= cand:
                        g[k], src[k] = t[k], k
                    else:
                        g[k], src[k] = cand, src[k - 1]
                for k in range(n - 2, -1, -1):
                    cand = g[k + 1] + _edge_backward(g[k + 1], k)
                    if cand < g[k]:
                        g[k], src[k] = cand, src[k + 1]
                return g, src
            g = t.copy()
            src = np.arange(n, dtype=np.int32)
            visited = np.zeros(n, bool)
            pq = [(g[i], i) for i in range(n) if np.isfinite(g[i])]
            heapq.heapify(pq)
            while pq:
                time_u, u = heapq.heappop(pq)
                if visited[u] or time_u > g[u]:
                    continue
                visited[u] = True
                v = (u + 1) % n
                w = _edge_forward(time_u, v)
                if np.isfinite(w):
                    cand = time_u + w
                    if cand < g[v]:
                        g[v] = cand
                        src[v] = src[u]
                        heapq.heappush(pq, (cand, v))
                v = (u - 1) % n
                w = _edge_backward(time_u, v)
                if np.isfinite(w):
                    cand = time_u + w
                    if cand < g[v]:
                        g[v] = cand
                        src[v] = src[u]
                        heapq.heappush(pq, (cand, v))
            return g, src

        def schedule_with_releases(t_base, release_guess):
            t_adj = np.full(n, np.inf, float)
            A = Amax
            current_time = 0.0
            cand_idx = np.flatnonzero(np.isfinite(t_base)).astype(np.int64, copy=False)
            cand_heap = [(t_base[i], int(i)) for i in cand_idx]
            heapq.heapify(cand_heap)
            rel_heap = []
            firedQ = np.zeros(n, bool)
            while cand_heap or rel_heap:
                next_cand_t = cand_heap[0][0] if cand_heap else np.inf
                next_rel_t = rel_heap[0][0] if rel_heap else np.inf
                if next_rel_t <= next_cand_t:
                    te, add = heapq.heappop(rel_heap)
                    if te > current_time:
                        A = recover(A, te - current_time)
                        current_time = te
                    A = min(Amax, A + add)
                    continue
                ti, i = heapq.heappop(cand_heap)
                if firedQ[i]:
                    continue
                if ti > current_time:
                    A = recover(A, ti - current_time)
                    current_time = ti
                if A >= cost:
                    A -= cost
                    firedQ[i] = True
                    t_adj[i] = current_time
                    rt = release_guess[i]
                    if np.isfinite(rt) and rt >= current_time:
                        heapq.heappush(rel_heap, (rt, cost))
                else:
                    dtw = wait_time_to_cost(A)
                    if not np.isfinite(dtw):
                        firedQ[i] = True
                        t_adj[i] = np.inf
                        continue
                    heapq.heappush(cand_heap, (current_time + dtw, i))
            return t_adj

        if not ffiring_forkQ:
            t = t0.copy()
            heap_idx = np.flatnonzero(np.isfinite(t)).astype(np.int64, copy=False)
            heap = [(t[i], int(i)) for i in heap_idx]
            heapq.heapify(heap)
            done = np.zeros(n, bool)
            A = Amax
            current_time = 0.0
            t_adj = t.copy()
            while heap:
                ti, i = heapq.heappop(heap)
                if done[i]:
                    continue
                if ti > current_time:
                    A = recover(A, ti - current_time)
                    current_time = ti
                if A >= cost:
                    A -= cost
                    done[i] = True
                    t_adj[i] = current_time
                else:
                    dtw = wait_time_to_cost(A)
                    if not np.isfinite(dtw):
                        t_adj[i] = np.inf
                        done[i] = True
                        continue
                    current_time = current_time + dtw
                    A = recover(A, dtw)
                    heapq.heappush(heap, (current_time, i))
            t = t_adj
        else:
            t = t0.copy()
            release_guess = np.full(n, np.inf, float)
            t = schedule_with_releases(t0, release_guess)
            for _ in range(3):
                rep_time_tmp, src_tmp = propagate_from_times(t)
                if not perQ:
                    changes = np.flatnonzero(np.r_[True, src_tmp[1:] != src_tmp[:-1], True])
                    starts = changes[:-1]
                    ends = changes[1:] - 1
                    labels = src_tmp[starts].astype(np.int64, copy=False)
                    release_guess.fill(np.inf)
                    release_guess[labels] = np.maximum(rep_time_tmp[starts], rep_time_tmp[ends])
                else:
                    release_guess.fill(np.inf)
                    fired = np.flatnonzero((src_tmp == np.arange(n, dtype=np.int32)) & np.isfinite(t)).astype(np.int64, copy=False)
                    for ori in fired:
                        seg = np.flatnonzero(src_tmp == ori)
                        if seg.size:
                            release_guess[ori] = np.nanmax(rep_time_tmp[seg])
                t_new = schedule_with_releases(t0, release_guess)
                if np.allclose(np.nan_to_num(t_new, nan=np.inf, posinf=np.inf),
                               np.nan_to_num(t, nan=np.inf, posinf=np.inf),
                               rtol=0, atol=1e-12):
                    t = t_new
                    break
                t = t_new
    else:
        t = t0

    # propagation with prev
    check_cancelled()
    fork_directionality = np.zeros(n, float)
    if not perQ:
        g = np.empty(n, float)
        src = np.empty(n, np.int32)
        prev = np.full(n, -1, np.int32)
        g[0], src[0], prev[0] = t[0], 0, 0
        for k in range(1, n):
            if k % 1024 == 0:
                check_cancelled()
            cand = g[k - 1] + _edge_forward(g[k - 1], k)
            if t[k] <= cand:
                g[k], src[k], prev[k] = t[k], k, k
            else:
                g[k], src[k], prev[k] = cand, src[k - 1], k - 1
        for k in range(n - 2, -1, -1):
            if k % 1024 == 0:
                check_cancelled()
            cand = g[k + 1] + _edge_backward(g[k + 1], k)
            if cand < g[k]:
                g[k], src[k], prev[k] = cand, src[k + 1], k + 1
    else:
        g = t.copy()
        src = np.arange(n, dtype=np.int32)
        prev = np.arange(n, dtype=np.int32)
        visited = np.zeros(n, bool)
        pq = [(g[i], i) for i in range(n) if np.isfinite(g[i])]
        heapq.heapify(pq)
        propagation_steps = 0
        while pq:
            if propagation_steps % 1024 == 0:
                check_cancelled()
            propagation_steps += 1
            time_u, u = heapq.heappop(pq)
            if visited[u] or time_u > g[u]:
                continue
            visited[u] = True
            v = (u + 1) % n
            w = _edge_forward(time_u, v)
            if np.isfinite(w):
                cand = time_u + w
                if cand < g[v]:
                    g[v] = cand
                    src[v] = src[u]
                    prev[v] = u
                    fork_directionality[v] = 1.0
                    heapq.heappush(pq, (cand, v))
                elif cand == g[v] and prev[v] != v and fork_directionality[v] < 0:
                    # Opposing fronts meet exactly at this bin. Their net
                    # direction is zero, independent of heap/index ordering.
                    fork_directionality[v] = 0.0
            v = (u - 1) % n
            w = _edge_backward(time_u, v)
            if np.isfinite(w):
                cand = time_u + w
                if cand < g[v]:
                    g[v] = cand
                    src[v] = src[u]
                    prev[v] = u
                    fork_directionality[v] = -1.0
                    heapq.heappush(pq, (cand, v))
                elif cand == g[v] and prev[v] != v and fork_directionality[v] > 0:
                    fork_directionality[v] = 0.0

    idx = np.arange(n, dtype=np.int32)
    fired_mask = (src == idx) & np.isfinite(t)
    fired_idx = np.flatnonzero(fired_mask).astype(np.int64, copy=False)

    if not perQ:
        fork_directionality[prev == (idx - 1)] = 1.0
        fork_directionality[prev == (idx + 1)] = -1.0

    fired_times_min = t[fired_idx]
    fired_pos_kb = fired_idx.astype(float) * resolution_space

    if _compute_inter_origin_distances and fired_idx.size >= 2:
        # ``flatnonzero`` already returns monotonically increasing indices, so
        # there is no need to sort every realization before taking neighbour
        # differences.
        diffs = np.diff(fired_idx)
        if perQ:
            wrap = n - fired_idx[-1] + fired_idx[0]
            interorigin_dist_kb = np.concatenate([diffs, [wrap]]).astype(float) * resolution_space
            if interorigin_dist_kb.size != fired_idx.size:
                raise RuntimeError("perQ=True: expected len(IODs) == number of fired origins.")
        else:
            interorigin_dist_kb = diffs.astype(float) * resolution_space
    else:
        interorigin_dist_kb = np.array([], float)

    replication_time_min = g
    if max_rep_time is not None:
        replication_time_min = np.minimum(replication_time_min, float(max_rep_time))

    # --- time statistics (1D only, stays here) ---
    time_stats = None
    if time_statsQ:
        time_grid = _validated_time_grid(time_grid, required=True)
        assert time_grid is not None
        t_max_grid = float(time_grid[-1])

        def count_from_intervals(starts, ends):
            starts = np.asarray(starts, float)
            ends = np.asarray(ends, float)
            if starts.size == 0:
                return np.zeros_like(time_grid, float)
            starts.sort()
            ends.sort()
            started = np.searchsorted(starts, time_grid, side="right")
            ended = np.searchsorted(ends, time_grid, side="left")
            return (started - ended).astype(float)

        # build fork/stall intervals via traversal edges (cheap vs m×n)
        pk = prev.astype(np.int64, copy=False)
        good = (pk >= 0) & np.isfinite(replication_time_min) & np.isfinite(replication_time_min[pk])
        cols = np.flatnonzero(good).astype(np.int64, copy=False)
        edge_start = replication_time_min[pk[cols]]
        edge_end = replication_time_min[cols]

        fork_ok = np.isfinite(edge_start) & np.isfinite(edge_end) & (edge_start < edge_end) & (edge_start <= t_max_grid)
        fork_starts = edge_start[fork_ok].tolist()
        fork_ends = np.minimum(edge_end[fork_ok], t_max_grid).tolist()

        d = fork_directionality[cols]
        delay = np.where(d > 0, stall_delay_left[cols], stall_delay_right[cols])
        stall_ok = np.isfinite(delay) & (delay > 0.0) & np.isfinite(edge_end) & (edge_end <= t_max_grid)
        stall_starts = (edge_end[stall_ok] - delay[stall_ok]).tolist()
        stall_ends = np.minimum(edge_end[stall_ok], t_max_grid).tolist()

        total_forks = count_from_intervals(fork_starts, fork_ends)
        stalled_forks = count_from_intervals(stall_starts, stall_ends)
        active_forks = total_forks - stalled_forks

        # firing_factors 1D (as before)
        firing_factors = None
        if use_factors:
            Amax = float(ffiring_nr)
            cost = 2.0
            firing_factors = np.empty_like(time_grid, float)

            if ffiring_forkQ:
                release = np.full(n, np.inf, float)
                if not perQ:
                    changes = np.flatnonzero(np.r_[True, src[1:] != src[:-1], True])
                    starts = changes[:-1]
                    ends = changes[1:] - 1
                    labels = src[starts].astype(np.int64, copy=False)
                    release[labels] = np.maximum(replication_time_min[starts], replication_time_min[ends])
                else:
                    for ori in fired_idx:
                        seg = np.flatnonzero(src == ori)
                        if seg.size:
                            release[ori] = np.nanmax(replication_time_min[seg])

                fire_events = np.sort(fired_times_min[np.isfinite(fired_times_min)])
                rel_events = np.sort(release[fired_idx][np.isfinite(release[fired_idx])])
                A = Amax
                fi = 0
                ri = 0
                for j, tj in enumerate(time_grid):
                    while fi < fire_events.size and fire_events[fi] <= tj:
                        A = max(0.0, A - cost)
                        fi += 1
                    while ri < rel_events.size and rel_events[ri] <= tj:
                        A = min(Amax, A + cost)
                        ri += 1
                    firing_factors[j] = A
            else:
                r = float(ffiring_recycle)
                A = Amax
                last_t = 0.0
                events = np.sort(fired_times_min[np.isfinite(fired_times_min)])

                def _recover(A_now, dt):
                    if r == 0.0 or dt <= 0.0:
                        return A_now
                    return Amax - (Amax - A_now) * np.exp(-r * dt)

                ei = 0
                for j, tj in enumerate(time_grid):
                    while ei < events.size and events[ei] <= tj:
                        te = events[ei]
                        A = _recover(A, te - last_t)
                        A = max(0.0, A - cost)
                        last_t = te
                        ei += 1
                    firing_factors[j] = _recover(A, tj - last_t)

        event_cutoff = min(t_max_grid, max_rep_time) if max_rep_time is not None else t_max_grid
        origin_times = fired_times_min[np.isfinite(fired_times_min) & (fired_times_min <= event_cutoff)]
        origin_bins = np.searchsorted(time_grid, origin_times, side="left")
        time_stats = dict(
            time_grid=time_grid,
            total_forks=total_forks,
            active_forks=active_forks,
            stalled_forks=stalled_forks,
            firing_factors=firing_factors,
            replicon_lengths=_replicon_lengths(src, g, fired_idx, resolution_space, max_rep_time),
            origin_firing=np.bincount(origin_bins, minlength=len(time_grid)).astype(float),
        )

    return (
        fired_idx.astype(np.int64, copy=False),
        fired_pos_kb,
        fired_times_min,
        replication_time_min,
        interorigin_dist_kb,
        time_stats,
        fork_directionality,
        # NEW minimal extra return for post dens compilation (no m×n arrays):
        t, src, prev,
    )

def rsim(
    initiation_rate=None,
    fork_speed=1.4,
    sim_number=1,
    resolution_space=1.0,
    resolution_time=5.0,
    resolution_x=None,
    resolution_t=None,
    perQ=False,
    stall_rate=0.0,
    tau=np.inf,
    ffiring_nr=None,
    ffiring_recycle=0.0,
    ffiring_forkQ=False,
    time_statsQ=False,
    time_stats_xtQ=False,
    time_stats_densQ=None,  # deprecated alias; kept for backwards compatibility
    time_grid=_DEFAULT_TIME_GRID,
    max_rep_time=1200.0,
    seed=None,
    verbose=True,
    print_every=1,
    dens_block_n=4096,
    dens_memmap_dir=None,
    n_bins=None,
    genome_length=None,
    interval_kb=None,
    ori_rate=None,
    geometry=None,
    resolution=None,
    source=None,
    assumptions=None,
    retain_rep_times_per_sim=True,
    retain_inter_origin_distances=True,
    retain_rep_times_limit=None,
    stall_rate_per_kb=None,
    retain_inter_origin_distances_limit=None,
    fork_speed_std=0.0,
    retain_replicon_lengths_limit=None,
    stall_shared_directions=False,
    **legacy_kwargs,
):
    """Run replicate stochastic simulations and return averaged features.

    ``initiation_rate`` describes the rate of origin firing.
    ``ori_rate`` is accepted as a compatibility alias.
    Initiation rate and fork speed support four field forms: scalar, ``(n,)``,
    ``(1, T)``, and ``(n, T)``. If both kinetic inputs lack a spatial axis,
    pass ``n_bins``, ``genome_length``, or ``interval_kb``.

    For consistency with ``rmap``, kinetic inputs can also be supplied as
    ``source={"initiation_rate": ..., "fork_speed": ...}``, while simulation
    settings can be supplied through ``assumptions={...}``. Prefer
    ``stall_rate_per_kb`` for resolution-independent physical inputs;
    ``stall_rate`` is retained as the legacy probability per spatial bin.
    Per-simulation profiles, pooled inter-origin distances, one-dimensional
    time courses, and space-time fields have independent retention/statistics
    controls; inter-origin distances do not require either time-statistics flag.
    ``time_statsQ`` also records actual origin firing per time-grid interval and
    completed replicon lengths. Replicons are the DNA domains copied from each
    fired origin; unfinished domains are excluded. Their retained sample count
    can be bounded with ``retain_replicon_lengths_limit``.
    ``fork_speed_std`` enables positive lognormal speed draws matching the local
    input mean and requested SD, for every supported dependence. Stochastic
    temporal speeds are integrated through clock boundaries during crossings.
    ``stall_shared_directions=True`` shares stall positions and recovery delays
    between both fork directions; False preserves independent directional stalls.
    """
    params = {
        "initiation_rate": initiation_rate,
        "fork_speed": fork_speed,
        "fork_speed_std": fork_speed_std,
        "sim_number": sim_number,
        "resolution_space": resolution_space,
        "resolution_time": resolution_time,
        "resolution_x": resolution_x,
        "resolution_t": resolution_t,
        "perQ": perQ,
        "stall_rate": stall_rate,
        "stall_rate_per_kb": stall_rate_per_kb,
        "stall_shared_directions": stall_shared_directions,
        "tau": tau,
        "ffiring_nr": ffiring_nr,
        "ffiring_recycle": ffiring_recycle,
        "ffiring_forkQ": ffiring_forkQ,
        "time_statsQ": time_statsQ,
        "time_stats_xtQ": time_stats_xtQ,
        "time_stats_densQ": time_stats_densQ,
        "time_grid": time_grid,
        "max_rep_time": max_rep_time,
        "seed": seed,
        "verbose": verbose,
        "print_every": print_every,
        "dens_block_n": dens_block_n,
        "dens_memmap_dir": dens_memmap_dir,
        "n_bins": n_bins,
        "genome_length": genome_length,
        "interval_kb": interval_kb,
        "ori_rate": ori_rate,
        "_legacy_rate": _legacy_initiation_input(legacy_kwargs),
        "geometry": geometry,
        "resolution": resolution,
        "retain_rep_times_per_sim": retain_rep_times_per_sim,
        "retain_inter_origin_distances": retain_inter_origin_distances,
        "retain_rep_times_limit": retain_rep_times_limit,
        "retain_inter_origin_distances_limit": retain_inter_origin_distances_limit,
        "retain_replicon_lengths_limit": retain_replicon_lengths_limit,
    }

    if source is not None:
        if not isinstance(source, dict):
            raise ValueError("rsim source must be a dictionary with initiation_rate and/or fork_speed.")
        allowed_source = {"initiation_rate", "fork_speed"}
        for key, value in source.items():
            name = _normalise_structured_key(key, _SOURCE_ALIASES)
            if name not in allowed_source:
                raise ValueError(
                    "rsim source only accepts kinetic inputs: initiation_rate and fork_speed."
                )
            if name == "initiation_rate" and (
                params["initiation_rate"] is not None
                or params["ori_rate"] is not None
                or params["_legacy_rate"] is not None
            ):
                raise ValueError("Pass initiation_rate either directly or in source, not both.")
            params[name] = value

    if assumptions is not None:
        if not isinstance(assumptions, dict):
            raise ValueError("rsim assumptions must be a dictionary.")
        source_only = {
            "initiation_rate",
            "fork_speed",
            "ori_rate",
            "_legacy_rate",
        }
        allowed_assumptions = set(params) - source_only
        for key, value in assumptions.items():
            name = _normalise_structured_key(key, _ASSUMPTION_ALIASES)
            if name in source_only:
                raise ValueError(
                    "Put kinetic inputs such as initiation_rate and fork_speed in source, not assumptions."
                )
            if name not in allowed_assumptions:
                raise ValueError(f"Unknown rsim assumption: {key!r}.")
            params[name] = value

    initiation_rate = params["initiation_rate"]
    fork_speed = params["fork_speed"]
    fork_speed_std = _validated_fork_speed_std(params["fork_speed_std"])
    sim_number = params["sim_number"]
    resolution_space = params["resolution_space"]
    resolution_time = params["resolution_time"]
    resolution_x = params["resolution_x"]
    resolution_t = params["resolution_t"]
    perQ = params["perQ"]
    stall_rate = params["stall_rate"]
    stall_rate_per_kb = params["stall_rate_per_kb"]
    stall_shared_directions = _validated_stall_shared_directions(params["stall_shared_directions"])
    tau = params["tau"]
    ffiring_nr = params["ffiring_nr"]
    ffiring_recycle = params["ffiring_recycle"]
    ffiring_forkQ = params["ffiring_forkQ"]
    time_statsQ = params["time_statsQ"]
    time_stats_xtQ = params["time_stats_xtQ"]
    time_stats_densQ = params["time_stats_densQ"]
    time_grid = params["time_grid"]
    max_rep_time = params["max_rep_time"]
    seed = params["seed"]
    verbose = params["verbose"]
    print_every = params["print_every"]
    dens_block_n = params["dens_block_n"]
    dens_memmap_dir = params["dens_memmap_dir"]
    n_bins = params["n_bins"]
    genome_length = params["genome_length"]
    interval_kb = params["interval_kb"]
    ori_rate = params["ori_rate"]
    geometry = params["geometry"]
    resolution = params["resolution"]
    retain_rep_times_per_sim = bool(params["retain_rep_times_per_sim"])
    retain_inter_origin_distances = bool(params["retain_inter_origin_distances"])
    retain_rep_times_limit = params["retain_rep_times_limit"]
    retain_inter_origin_distances_limit = params[
        "retain_inter_origin_distances_limit"
    ]
    retain_replicon_lengths_limit = params["retain_replicon_lengths_limit"]

    if time_stats_densQ is not None:
        time_stats_xtQ = bool(time_stats_densQ)

    initiation_rate = _normalise_initiation_rate(
        initiation_rate=initiation_rate,
        ori_rate=ori_rate,
        legacy_rate=params["_legacy_rate"],
    )

    if resolution is not None:
        resolution_space = resolution
    if resolution_x is not None:
        resolution_space = resolution_x
    if resolution_t is not None:
        resolution_time = resolution_t

    resolution_space = float(resolution_space)
    if not np.isfinite(resolution_space) or resolution_space <= 0:
        raise ValueError("resolution_x must be a finite positive number.")
    resolution_time = float(resolution_time)
    if not np.isfinite(resolution_time) or resolution_time <= 0:
        raise ValueError("resolution_t must be a finite positive number.")
    stall_rate = _resolve_stall_probability(
        stall_rate,
        stall_rate_per_kb,
        resolution_space,
    )
    tau = float(tau)
    if np.isnan(tau) or tau <= 0:
        raise ValueError("tau must be positive or np.inf.")
    if max_rep_time is not None:
        max_rep_time = float(max_rep_time)
        if not np.isfinite(max_rep_time) or max_rep_time <= 0:
            raise ValueError("max_rep_time must be finite and positive, or None.")
    if ffiring_nr is not None:
        ffiring_nr = float(ffiring_nr)
        if not np.isfinite(ffiring_nr) or ffiring_nr < 2:
            raise ValueError("ffiring_nr must be finite and >= 2 when enabled.")
        ffiring_recycle = float(ffiring_recycle)
        if not np.isfinite(ffiring_recycle) or ffiring_recycle < 0:
            raise ValueError("ffiring_recycle must be a finite non-negative number.")

    if geometry is not None:
        if not isinstance(geometry, str):
            raise ValueError("geometry must be 'linear' or 'periodic'.")
        g = geometry.strip().lower()
        if g in {"linear", "chromosome", "open"}:
            perQ = False
        elif g in {"periodic", "circular", "circle"}:
            perQ = True
        else:
            raise ValueError("geometry must be 'linear' or 'periodic'.")

    n = _resolve_n_bins(
        initiation_rate,
        fork_speed,
        n_bins=n_bins,
        resolution_space=resolution_space,
        genome_length=genome_length,
        interval_kb=interval_kb,
    )

    sim_number = _validated_positive_integer(sim_number, "sim_number")
    print_every = _validated_positive_integer(print_every, "print_every")
    if retain_rep_times_per_sim:
        retained_rep_times = (
            sim_number
            if retain_rep_times_limit is None
            else min(
                sim_number,
                _validated_nonnegative_integer(
                    retain_rep_times_limit,
                    "retain_rep_times_limit",
                ),
            )
        )
    else:
        # A limit has no effect unless profile retention itself is enabled.
        retained_rep_times = 0

    if retain_inter_origin_distances and retain_inter_origin_distances_limit is not None:
        retained_iod_limit = _validated_nonnegative_integer(
            retain_inter_origin_distances_limit,
            "retain_inter_origin_distances_limit",
        )
    else:
        retained_iod_limit = None

    retained_replicon_limit = (
        _validated_nonnegative_integer(retain_replicon_lengths_limit, "retain_replicon_lengths_limit")
        if time_statsQ and retain_replicon_lengths_limit is not None else None
    )

    ss = np.random.SeedSequence(seed) if seed is not None else np.random.SeedSequence()

    sum_rep_time = np.zeros(n, dtype=float)
    count_rep_time = np.zeros(n, dtype=np.int64)
    running_mean_rep_time = np.zeros(n, dtype=float)
    m2_rep_time = np.zeros(n, dtype=float)

    rep_times_per_sim = (
        np.empty((retained_rep_times, n), dtype=float)
        if retain_rep_times_per_sim
        else None
    )
    s_phase_durations_min = np.empty(sim_number, dtype=float)
    completion_times_min = np.empty(sim_number, dtype=float)
    nr_oris = np.empty(sim_number, dtype=np.int64)

    all_iods = [] if retain_inter_origin_distances else None
    retained_iod_count = 0
    total_iod_count = 0
    all_replicons = [] if time_statsQ else None
    retained_replicon_count = 0
    total_replicon_count = 0
    fire_counts = np.zeros(n, dtype=np.int64)

    sum_fdir = np.zeros(n, dtype=float)
    count_fdir = np.zeros(n, dtype=np.int64)

    need_time_grid = time_statsQ or time_stats_xtQ
    if need_time_grid:
        time_grid = _validated_time_grid(time_grid, required=True)
        assert time_grid is not None
        m = int(time_grid.size)
        dt_grid_out = float(time_grid[1] - time_grid[0]) if m >= 2 else 1.0
        sum_replicated_fraction = np.zeros(m, float)
    else:
        time_grid = None
        dt_grid_out = None
        m = None
        sum_replicated_fraction = None

    if time_statsQ:
        assert m is not None
        sum_total_forks = np.zeros(m, float)
        sum_active_forks = np.zeros(m, float)
        sum_stalled_forks = np.zeros(m, float)
        sum_origin_firing = np.zeros(m, float)
        sum_firing_factors = np.zeros(m, float)
        count_firing_factors = 0
    else:
        sum_total_forks = None
        sum_active_forks = None
        sum_stalled_forks = None
        sum_origin_firing = None
        sum_firing_factors = None
        count_firing_factors = 0

    # --- allocate space-time accumulators (disk-backed) ---
    if time_stats_xtQ:
        assert m is not None
        assert time_grid is not None
        if dens_memmap_dir is None:
            dens_memmap_dir = tempfile.gettempdir()
        os.makedirs(dens_memmap_dir, exist_ok=True)

        dens_block_n = _validated_positive_integer(dens_block_n, "dens_block_n")

        def _mm(name, shape, dtype):
            path = os.path.join(dens_memmap_dir, f"{name}_{int(time.time()*1e6)}.dat")
            array = np.memmap(path, mode="w+", dtype=dtype, shape=shape)
            # Returned transposed views retain this root as their base. The
            # finalizer therefore runs only after the last output view is gone.
            mapping = getattr(array, "_mmap", None)
            if mapping is None:
                raise RuntimeError("Could not attach cleanup to the density memmap.")
            weakref.finalize(
                array,
                _close_and_unlink_memmap,
                mapping,
                path,
            )
            return array

        # store sums as float32 (you can change to uint16 if sim_number <= 65535)
        sum_replicated_fraction_xt = _mm("sum_replicated_fraction_xt", (m, n), np.float32)
        sum_initiation_events_xt = _mm("sum_initiation_events_xt", (m, n), np.float32)
        sum_coalescence_events_xt = _mm("sum_coalescence_events_xt", (m, n), np.float32)
        sum_right_moving_fork_density_xt = _mm("sum_right_moving_fork_density_xt", (m, n), np.float32)
        sum_left_moving_fork_density_xt = _mm("sum_left_moving_fork_density_xt", (m, n), np.float32)
        sum_fork_density_xt = _mm("sum_fork_density_xt", (m, n), np.float32)

        sum_replicated_fraction_xt[:] = 0.0
        sum_initiation_events_xt[:] = 0.0
        sum_coalescence_events_xt[:] = 0.0
        sum_right_moving_fork_density_xt[:] = 0.0
        sum_left_moving_fork_density_xt[:] = 0.0
        sum_fork_density_xt[:] = 0.0

    else:
        sum_replicated_fraction_xt = None
        sum_initiation_events_xt = None
        sum_coalescence_events_xt = None
        sum_right_moving_fork_density_xt = None
        sum_left_moving_fork_density_xt = None
        sum_fork_density_xt = None

    start_time = time.time()

    prepared_firing_field = _as_rate_field(
        initiation_rate,
        "initiation_rate",
        n_bins=n,
    )
    firing_kind, firing_values, _ = prepared_firing_field
    prepared_inverse_speed_field = _as_inverse_speed_field(
        fork_speed,
        "fork_speed",
        n_bins=n,
    )
    if firing_kind == "t":
        firing_cumulative_hazard = np.cumsum(
            firing_values * resolution_time,
        )
    elif firing_kind == "xt":
        firing_cumulative_hazard = np.cumsum(
            firing_values * resolution_time,
            axis=0,
        )
    else:
        firing_cumulative_hazard = None

    for i in range(1, sim_number + 1):
        # Cancellation is independent of console/progress verbosity. Mapping
        # fits intentionally silence nested ensembles, but Stop must still be
        # observed between individual simulations.
        check_cancelled()
        # Spawn lazily so large ensembles do not retain tens of thousands of
        # Generator and SeedSequence objects at once. Repeated spawn(1) calls
        # produce the same child sequence as one eager spawn(sim_number).
        g = np.random.default_rng(ss.spawn(1)[0])
        retain_this_iod = all_iods is not None and (
            retained_iod_limit is None or retained_iod_count < retained_iod_limit
        )
        (
            fired_idx,
            _,
            fired_times,
            rep_time,
            iod,
            time_stats,
            fdir,
            t_fire,
            src,
            prev,
        ) = simulate_replication(
            initiation_rate=initiation_rate,
            fork_speed=fork_speed,
            resolution_space=resolution_space,
            resolution_time=resolution_time,
            resolution_x=None,
            resolution_t=None,
            rng=g,
            perQ=perQ,
            stall_rate=stall_rate,
            tau=tau,
            ffiring_nr=ffiring_nr,
            ffiring_recycle=ffiring_recycle,
            ffiring_forkQ=ffiring_forkQ,
            time_statsQ=time_statsQ,
            time_stats_xtQ=time_stats_xtQ,
            time_stats_densQ=time_stats_densQ,
            time_grid=time_grid,
            max_rep_time=max_rep_time,
            n_bins=n,
            genome_length=None,
            interval_kb=None,
            _firing_cumulative_hazard=firing_cumulative_hazard,
            _prepared_firing_field=prepared_firing_field,
            _prepared_inverse_speed_field=prepared_inverse_speed_field,
            _compute_inter_origin_distances=retain_this_iod,
            fork_speed_std=fork_speed_std,
            stall_shared_directions=stall_shared_directions,
        )

        if rep_times_per_sim is not None and i <= retained_rep_times:
            rep_times_per_sim[i - 1, :] = rep_time
        completion_time, s_phase_duration = _phase_timing_summary(
            fired_times,
            rep_time,
        )
        completion_times_min[i - 1] = completion_time
        s_phase_durations_min[i - 1] = s_phase_duration
        nr_oris[i - 1] = int(fired_idx.size)

        finite_mask = np.isfinite(rep_time)
        next_counts = count_rep_time[finite_mask] + 1
        delta = rep_time[finite_mask] - running_mean_rep_time[finite_mask]
        next_means = running_mean_rep_time[finite_mask] + delta / next_counts
        m2_rep_time[finite_mask] += delta * (
            rep_time[finite_mask] - next_means
        )
        running_mean_rep_time[finite_mask] = next_means
        sum_rep_time[finite_mask] += rep_time[finite_mask]
        count_rep_time[finite_mask] = next_counts

        fired_count = int(fired_idx.size)
        realization_iod_count = (
            fired_count
            if perQ and fired_count >= 2
            else max(0, fired_count - 1)
        )
        total_iod_count += realization_iod_count
        if retain_this_iod:
            if int(iod.size) != realization_iod_count:
                raise RuntimeError(
                    "simulate_replication returned an unexpected number of "
                    "inter-origin distances."
                )
            if iod.size:
                remaining = (
                    int(iod.size)
                    if retained_iod_limit is None
                    else max(0, retained_iod_limit - retained_iod_count)
                )
                if remaining:
                    retained = np.asarray(iod[:remaining], dtype=float).copy()
                    all_iods.append(retained)
                    retained_iod_count += int(retained.size)
        if fired_idx.size:
            fire_counts[fired_idx] += 1

        sum_fdir[finite_mask] += fdir[finite_mask]
        count_fdir[finite_mask] += 1

        # --- 1D replicated-fraction time series (RAM) ---
        if time_grid is not None:
            assert sum_replicated_fraction is not None
            rep_sorted = np.sort(rep_time[finite_mask])
            if rep_sorted.size:
                sum_replicated_fraction += (
                    np.searchsorted(rep_sorted, time_grid, side="right") / float(n)
                )

        # --- 1D time series accumulate (RAM) ---
        if time_statsQ and time_stats is not None:
            assert sum_total_forks is not None
            assert sum_active_forks is not None
            assert sum_stalled_forks is not None
            assert sum_firing_factors is not None
            sum_total_forks += time_stats["total_forks"]
            sum_active_forks += time_stats["active_forks"]
            sum_stalled_forks += time_stats["stalled_forks"]
            assert sum_origin_firing is not None
            assert all_replicons is not None
            sum_origin_firing += time_stats["origin_firing"]
            lengths = time_stats["replicon_lengths"]
            total_replicon_count += len(lengths)
            remaining = (
                len(lengths) if retained_replicon_limit is None
                else max(0, retained_replicon_limit - retained_replicon_count)
            )
            if remaining:
                retained = lengths[:remaining].copy()
                all_replicons.append(retained)
                retained_replicon_count += len(retained)
            ff = time_stats.get("firing_factors", None)
            if ff is not None:
                sum_firing_factors += ff
                count_firing_factors += 1

        # --- 2D space-time stats accumulate (disk memmap, block-by-block, no giant temporaries) ---
        if time_stats_xtQ:
            assert time_grid is not None
            assert sum_replicated_fraction_xt is not None
            assert sum_initiation_events_xt is not None
            assert sum_coalescence_events_xt is not None
            assert sum_right_moving_fork_density_xt is not None
            assert sum_left_moving_fork_density_xt is not None
            t_max_grid = float(time_grid[-1])

            # (a) replicated fraction in (t,x)
            for x0 in range(0, n, dens_block_n):
                x1 = min(n, x0 + dens_block_n)
                blk = (rep_time[x0:x1][None, :] <= time_grid[:, None])
                sum_replicated_fraction_xt[:, x0:x1] += blk.astype(np.float32)

            # (b) initiation events: sparse binning
            tf = fired_times
            ok = np.isfinite(tf) & (tf <= t_max_grid) & (tf <= (max_rep_time if max_rep_time is not None else np.inf))
            if np.any(ok):
                cols = fired_idx[ok]
                bins = np.searchsorted(time_grid, tf[ok], side="left").astype(np.int64, copy=False)
                np.add.at(sum_initiation_events_xt, (bins, cols), 1.0)

            # (c) coalescence events: sparse binning on boundaries
            if not perQ:
                boundary_k = (np.flatnonzero(src[1:] != src[:-1]) + 1).astype(np.int64, copy=False)
                if boundary_k.size:
                    km1 = boundary_k - 1
                    t_meet = np.maximum(rep_time[km1], rep_time[boundary_k])
                    okm = np.isfinite(t_meet) & (t_meet <= t_max_grid)
                    if np.any(okm):
                        bins = np.searchsorted(time_grid, t_meet[okm], side="left").astype(np.int64, copy=False)
                        cols = boundary_k[okm].astype(np.int64, copy=False)
                        np.add.at(sum_coalescence_events_xt, (bins, cols), 1.0)
            else:
                boundary_k = np.flatnonzero(src[np.r_[1:n, 0]] != src).astype(np.int64, copy=False)
                if boundary_k.size:
                    km1 = (boundary_k - 1) % n
                    t_meet = np.maximum(rep_time[km1], rep_time[boundary_k])
                    okm = np.isfinite(t_meet) & (t_meet <= t_max_grid)
                    if np.any(okm):
                        bins = np.searchsorted(time_grid, t_meet[okm], side="left").astype(np.int64, copy=False)
                        cols = boundary_k[okm].astype(np.int64, copy=False)
                        np.add.at(sum_coalescence_events_xt, (bins, cols), 1.0)

            # (d) fork densities in (t,x)
            pk = prev.astype(np.int64, copy=False)
            good = (pk >= 0) & np.isfinite(rep_time) & np.isfinite(rep_time[pk])
            cols0 = np.flatnonzero(good).astype(np.int64, copy=False)
            if cols0.size:
                t_end = rep_time[cols0]
                t_start = rep_time[pk[cols0]]
                ok2 = (t_start < t_end) & (t_start <= t_max_grid)
                cols0 = cols0[ok2]
                if cols0.size:
                    t_start = t_start[ok2]
                    t_end = np.minimum(t_end[ok2], t_max_grid)
                    j0 = np.searchsorted(time_grid, t_start, side="left").astype(np.int64, copy=False)
                    j1 = np.searchsorted(time_grid, t_end, side="left").astype(np.int64, copy=False)
                    ok3 = j1 > j0
                    cols0 = cols0[ok3]
                    if cols0.size:
                        j0 = j0[ok3]
                        j1 = j1[ok3]
                        d = fdir[cols0]
                        Rm = d > 0
                        Lm = d < 0

                        def _range_add(target_mm, cols, a, b, weight=1.0):
                            for kk in range(cols.size):
                                c = int(cols[kk])
                                target_mm[a[kk]:b[kk], c] += weight

                        if np.any(Rm):
                            _range_add(sum_right_moving_fork_density_xt, cols0[Rm], j0[Rm], j1[Rm])
                        if np.any(Lm):
                            _range_add(sum_left_moving_fork_density_xt, cols0[Lm], j0[Lm], j1[Lm])
                        tied = d == 0
                        if np.any(tied):
                            _range_add(sum_right_moving_fork_density_xt, cols0[tied], j0[tied], j1[tied], .5)
                            _range_add(sum_left_moving_fork_density_xt, cols0[tied], j0[tied], j1[tied], .5)

        report_measured_work(i / sim_number)
        if verbose and (i % print_every == 0 or i == sim_number):
            etaf(it=i, maxiter=int(sim_number), start_time=start_time, label="rsim")

    if all_iods is None:
        all_iods_kb = None
    else:
        all_iods_kb = (
            np.concatenate(all_iods)
            if all_iods
            else np.array([], dtype=float)
        )

    avg_rep_time_min = np.full(n, np.inf, dtype=float)
    nonzero = count_rep_time > 0
    avg_rep_time_min[nonzero] = sum_rep_time[nonzero] / count_rep_time[nonzero].astype(float)
    replication_timing_std = np.full(n, np.nan, dtype=float)
    replication_timing_std[nonzero] = np.sqrt(
        np.maximum(
            m2_rep_time[nonzero] / count_rep_time[nonzero].astype(float),
            0.0,
        )
    )

    fork_directionality = np.full(n, np.nan, float)
    fd_ok = count_fdir > 0
    fork_directionality[fd_ok] = sum_fdir[fd_ok] / count_fdir[fd_ok].astype(float)

    efficiencies = fire_counts.astype(float) / float(sim_number)

    avg_replicated_fraction = (
        sum_replicated_fraction / float(sim_number)
        if sum_replicated_fraction is not None
        else None
    )

    # --- finalise outputs ---
    if time_statsQ:
        assert sum_total_forks is not None
        assert sum_active_forks is not None
        assert sum_stalled_forks is not None
        assert sum_origin_firing is not None
        assert sum_firing_factors is not None
        invS = 1.0 / float(sim_number)
        avg_total_forks = sum_total_forks * invS
        avg_active_forks = sum_active_forks * invS
        avg_stalled_forks = sum_stalled_forks * invS
        avg_origin_firing = sum_origin_firing * invS
        avg_firing_factors = (sum_firing_factors / float(count_firing_factors)) if count_firing_factors > 0 else None
    else:
        avg_total_forks = None
        avg_active_forks = None
        avg_stalled_forks = None
        avg_origin_firing = None
        avg_firing_factors = None

    if time_stats_xtQ:
        assert time_grid is not None
        assert sum_replicated_fraction_xt is not None
        assert sum_initiation_events_xt is not None
        assert sum_coalescence_events_xt is not None
        assert sum_right_moving_fork_density_xt is not None
        assert sum_left_moving_fork_density_xt is not None
        assert sum_fork_density_xt is not None
        post_start = time.time()
        post_blocks = (n + dens_block_n - 1) // dens_block_n
        for j, x0 in enumerate(range(0, n, dens_block_n), start=1):
            x1 = min(n, x0 + dens_block_n)
            sum_replicated_fraction_xt[:, x0:x1] /= float(sim_number)
            sum_initiation_events_xt[:, x0:x1] /= float(sim_number)
            sum_coalescence_events_xt[:, x0:x1] /= float(sim_number)
            sum_right_moving_fork_density_xt[:, x0:x1] /= float(sim_number)
            sum_left_moving_fork_density_xt[:, x0:x1] /= float(sim_number)
            np.add(sum_right_moving_fork_density_xt[:, x0:x1], sum_left_moving_fork_density_xt[:, x0:x1],
                   out=sum_fork_density_xt[:, x0:x1])
            if verbose:
                etaf(
                    it=j,
                    maxiter=post_blocks,
                    start_time=post_start,
                    label="rsim_finalize",
                )

        avg_replicated_fraction_xt = sum_replicated_fraction_xt.T
        avg_initiation_events_xt = sum_initiation_events_xt.T
        avg_coalescence_events_xt = sum_coalescence_events_xt.T
        avg_right_moving_fork_density_xt = sum_right_moving_fork_density_xt.T
        avg_left_moving_fork_density_xt = sum_left_moving_fork_density_xt.T
        avg_fork_density_xt = sum_fork_density_xt.T
        dt_grid_out = float(time_grid[1] - time_grid[0]) if time_grid is not None and time_grid.size >= 2 else 1.0
    else:
        avg_replicated_fraction_xt = None
        avg_initiation_events_xt = None
        avg_coalescence_events_xt = None
        avg_right_moving_fork_density_xt = None
        avg_left_moving_fork_density_xt = None
        avg_fork_density_xt = None

    nr_oris_list = nr_oris.tolist()
    return dict(
        replication_timing=avg_rep_time_min,
        replication_timing_std=replication_timing_std,
        rep_times_per_sim=rep_times_per_sim,
        s_phase_duration=s_phase_durations_min,
        completion_time=completion_times_min,
        num_oris=nr_oris_list,
        num_origins=nr_oris_list,
        inter_origin_distances=all_iods_kb,
        inter_origin_distance=all_iods_kb,
        inter_origin_distance_count=total_iod_count,
        inter_origin_distances_truncated=(
            all_iods_kb is not None and retained_iod_count < total_iod_count
        ),
        replicon_lengths=(
            (np.concatenate(all_replicons) if all_replicons else np.empty(0, dtype=float))
            if time_statsQ else None
        ),
        replicon_length_count=total_replicon_count,
        replicon_lengths_truncated=(time_statsQ and retained_replicon_count < total_replicon_count),
        efficiency=efficiencies,
        origin_efficiency=efficiencies,
        fork_directionality=fork_directionality,
        time_grid=time_grid,
        dt_grid=dt_grid_out,
        replicated_fraction=avg_replicated_fraction,
        replicated_fraction_xt=avg_replicated_fraction_xt,
        initiation_events_xt=avg_initiation_events_xt,
        coalescence_events_xt=avg_coalescence_events_xt,
        right_moving_fork_density_xt=avg_right_moving_fork_density_xt,
        right_fork_density_xt=avg_right_moving_fork_density_xt,
        left_moving_fork_density_xt=avg_left_moving_fork_density_xt,
        left_fork_density_xt=avg_left_moving_fork_density_xt,
        fork_density_xt=avg_fork_density_xt,
        total_forks=avg_total_forks,
        active_forks=avg_active_forks,
        stalled_forks=avg_stalled_forks,
        origin_firing=avg_origin_firing,
        firing_factors=avg_firing_factors,
    )


_RSIM_INPUT_DESCRIPTIONS = {
    "initiation_rate": (
        "Origin firing or initiation propensity. Preferred direct kinetic "
        "input; accepts scalar, spatial, time-only, or space-time forms."
    ),
    "fork_speed": (
        "Replication fork speed in kb/min. Accepts scalar, spatial, "
        "time-only, or space-time forms."
    ),
    "fork_speed_std": (
        "Arithmetic SD in kb/min of positive lognormal speeds whose mean is the local "
        "input speed. Temporal speeds are integrated during crossings. Zero keeps existing speed behavior."
    ),
    "sim_number": "Number of independent stochastic S phases to simulate and summarize.",
    "resolution_space": "Spatial bin size in kb.",
    "resolution_time": "Time step in minutes for time-dependent inputs and default time grids.",
    "resolution_x": "Alias for resolution_space, matching the rmap notation.",
    "resolution_t": "Alias for resolution_time.",
    "perQ": "Use periodic/circular geometry instead of linear geometry.",
    "stall_rate": (
        "Legacy probability that a moving fork encounters a stall in one spatial bin."
    ),
    "stall_rate_per_kb": (
        "Physical stall density in stalls/kb. Converted to the per-bin probability "
        "1 - exp(-stall_rate_per_kb * resolution_space)."
    ),
    "stall_shared_directions": (
        "Share stall positions and recovery delays between both fork directions. "
        "False retains independent directional stalls; this does not couple sister forks."
    ),
    "tau": "Mean stall duration. np.inf represents non-recovering stalls.",
    "ffiring_nr": "Finite firing-factor pool size. None disables finite-factor limitation.",
    "ffiring_recycle": "Rate at which used firing factors recycle into the available pool.",
    "ffiring_forkQ": "Whether active forks also affect the finite firing-factor pool.",
    "time_statsQ": (
        "Return one-dimensional time courses, including actual origin firing, "
        "and the completed replicon-length distribution."
    ),
    "time_stats_xtQ": (
        "Return space-time fields. This is more memory intensive than "
        "time_statsQ."
    ),
    "time_stats_densQ": "Deprecated compatibility alias for time_stats_xtQ.",
    "time_grid": "Time points at which time-course and space-time summaries are evaluated.",
    "max_rep_time": (
        "Maximum simulated replication time before unresolved loci are "
        "treated as unfinished."
    ),
    "seed": "Random seed for reproducible stochastic ensembles.",
    "verbose": "Print progress while simulations are running.",
    "print_every": "Progress-reporting interval in number of simulated S phases.",
    "dens_block_n": (
        "Number of genomic bins processed per block for space-time density "
        "accumulation."
    ),
    "dens_memmap_dir": "Directory for temporary memmap files used by space-time fields.",
    "retain_rep_times_per_sim": (
        "Retain per-simulation replication-time profiles. By default this is "
        "the full (sim_number, n) matrix; retain_rep_times_limit can bound it."
    ),
    "retain_rep_times_limit": (
        "Maximum number of per-simulation replication-time profiles to retain. "
        "None retains all profiles; the setting is ignored when "
        "retain_rep_times_per_sim=False."
    ),
    "retain_inter_origin_distances": (
        "Retain pooled per-realization inter-origin distances. Disable to "
        "avoid storing the pooled distribution."
    ),
    "retain_inter_origin_distances_limit": (
        "Maximum number of pooled inter-origin distances to retain. None "
        "retains the complete distribution. The full event count is still "
        "reported when retention is bounded or disabled."
    ),
    "retain_replicon_lengths_limit": (
        "Maximum number of pooled completed replicon lengths to retain with "
        "time_statsQ=True. None retains all; zero keeps only the total count."
    ),
    "n_bins": "Number of genomic bins when no spatial kinetic input is supplied.",
    "genome_length": "Genome length in kb, used with the spatial resolution to infer n_bins.",
    "interval_kb": (
        "Genomic interval (start_kb, end_kb), also used to infer the "
        "simulated length."
    ),
    "ori_rate": "Compatibility alias for initiation_rate.",
    "geometry": "String geometry selector, usually linear or periodic.",
    "resolution": "Alias for resolution_space.",
    "source": (
        "Dictionary input style, usually containing initiation_rate and fork_speed; "
        "mirrors rmap."
    ),
    "assumptions": (
        "Dictionary of simulation settings; mirrors rmap and keeps kinetic "
        "inputs separate from options."
    ),
}


_RSIM_OUTPUT_RECORDS = [
    {
        "output": "replication_timing",
        "shape/type": "(n,) array",
        "returned when": "always",
        "description": "Mean replication time of each genomic bin across the ensemble.",
    },
    {
        "output": "replication_timing_std",
        "shape/type": "(n,) array",
        "returned when": "always",
        "description": (
            "Population standard deviation of finite replication times at each "
            "genomic bin across the full ensemble."
        ),
    },
    {
        "output": "rep_times_per_sim",
        "shape/type": "(retained simulations, n) array or None",
        "returned when": "retain_rep_times_per_sim=True",
        "description": (
            "Replication-time profiles for the first simulations retained, bounded "
            "by retain_rep_times_limit when provided."
        ),
    },
    {
        "output": "s_phase_duration",
        "shape/type": "(sim_number,) array",
        "returned when": "always",
        "description": (
            "Elapsed time from the first realised origin initiation to "
            "replication completion in each simulation."
        ),
    },
    {
        "output": "completion_time",
        "shape/type": "(sim_number,) array",
        "returned when": "always",
        "description": (
            "Absolute simulation-clock time at which replication completed "
            "in each simulation."
        ),
    },
    {
        "output": "num_oris",
        "shape/type": "list",
        "returned when": "always",
        "description": "Number of fired origins in each simulation; alias of num_origins.",
    },
    {
        "output": "num_origins",
        "shape/type": "list",
        "returned when": "always",
        "description": "Number of fired origins in each simulation.",
    },
    {
        "output": "inter_origin_distances",
        "shape/type": "1D array or None",
        "returned when": "retain_inter_origin_distances=True",
        "description": (
            "Pooled distances between neighboring fired origins, optionally "
            "bounded by retain_inter_origin_distances_limit; alias of "
            "inter_origin_distance."
        ),
    },
    {
        "output": "inter_origin_distance",
        "shape/type": "1D array or None",
        "returned when": "retain_inter_origin_distances=True",
        "description": (
            "Pooled distances between neighboring fired origins, optionally "
            "bounded; alias of inter_origin_distances."
        ),
    },
    {
        "output": "inter_origin_distance_count",
        "shape/type": "integer",
        "returned when": "always",
        "description": (
            "Total number of inter-origin distances produced before any "
            "retention limit; computed without retaining the distribution."
        ),
    },
    {
        "output": "inter_origin_distances_truncated",
        "shape/type": "boolean",
        "returned when": "always",
        "description": (
            "True when distance retention was enabled but its limit retained "
            "fewer values than were produced."
        ),
    },
    {
        "output": "replicon_lengths",
        "shape/type": "1D array or None",
        "returned when": "time_statsQ=True",
        "description": "Pooled completed DNA-domain lengths per fired origin, in kb; unfinished domains are excluded.",
    },
    {
        "output": "replicon_length_count",
        "shape/type": "integer",
        "returned when": "always",
        "description": "Total completed replicons before the retention limit (zero without time_statsQ).",
    },
    {
        "output": "replicon_lengths_truncated",
        "shape/type": "boolean",
        "returned when": "always",
        "description": "Whether the retained replicon lengths are a bounded prefix of the full sample pool.",
    },
    {
        "output": "efficiency",
        "shape/type": "(n,) array",
        "returned when": "always",
        "description": (
            "Fraction of simulations in which each bin fired; alias of "
            "origin_efficiency."
        ),
    },
    {
        "output": "origin_efficiency",
        "shape/type": "(n,) array",
        "returned when": "always",
        "description": (
            "Fraction of simulations in which each bin fired before passive "
            "replication."
        ),
    },
    {
        "output": "fork_directionality",
        "shape/type": "(n,) array",
        "returned when": "always",
        "description": "Average direction from which each bin was replicated.",
    },
    {
        "output": "time_grid",
        "shape/type": "(m,) array or None",
        "returned when": "time statistics",
        "description": "Time points used for time-course and space-time summaries.",
    },
    {
        "output": "dt_grid",
        "shape/type": "float or None",
        "returned when": "time statistics",
        "description": "Spacing of time_grid, when defined.",
    },
    {
        "output": "replicated_fraction",
        "shape/type": "(m,) array or None",
        "returned when": "time_statsQ=True or time_stats_xtQ=True",
        "description": "Fraction of the simulated region replicated at each time point.",
    },
    {
        "output": "replicated_fraction_xt",
        "shape/type": "(n, m) array or None",
        "returned when": "time_stats_xtQ=True",
        "description": (
            "Local replicated fraction as a function of genomic position and time."
        ),
    },
    {
        "output": "initiation_events_xt",
        "shape/type": "(n, m) array or None",
        "returned when": "time_stats_xtQ=True",
        "description": "Realized initiation-event counts binned by position and time.",
    },
    {
        "output": "coalescence_events_xt",
        "shape/type": "(n, m) array or None",
        "returned when": "time_stats_xtQ=True",
        "description": "Fork-coalescence events binned by position and time.",
    },
    {
        "output": "right_moving_fork_density_xt",
        "shape/type": "(n, m) array or None",
        "returned when": "time_stats_xtQ=True",
        "description": (
            "Right-moving fork-density field; alias of right_fork_density_xt."
        ),
    },
    {
        "output": "right_fork_density_xt",
        "shape/type": "(n, m) array or None",
        "returned when": "time_stats_xtQ=True",
        "description": "Right-moving fork-density field.",
    },
    {
        "output": "left_moving_fork_density_xt",
        "shape/type": "(n, m) array or None",
        "returned when": "time_stats_xtQ=True",
        "description": (
            "Left-moving fork-density field; alias of left_fork_density_xt."
        ),
    },
    {
        "output": "left_fork_density_xt",
        "shape/type": "(n, m) array or None",
        "returned when": "time_stats_xtQ=True",
        "description": "Left-moving fork-density field.",
    },
    {
        "output": "fork_density_xt",
        "shape/type": "(n, m) array or None",
        "returned when": "time_stats_xtQ=True",
        "description": "Total fork-density field, equal to the right- plus left-moving fields.",
    },
    {
        "output": "origin_firing",
        "shape/type": "(m,) array or None",
        "returned when": "time_statsQ=True",
        "description": (
            "Mean actual initiations per time-grid interval ending at each grid point; "
            "the first bin includes events up to the first point. Excludes passive replication."
        ),
    },
    {
        "output": "total_forks",
        "shape/type": "(m,) array or None",
        "returned when": "time_statsQ=True",
        "description": "Total number of forks present at each time point.",
    },
    {
        "output": "active_forks",
        "shape/type": "(m,) array or None",
        "returned when": "time_statsQ=True",
        "description": "Number of non-stalled forks present at each time point.",
    },
    {
        "output": "stalled_forks",
        "shape/type": "(m,) array or None",
        "returned when": "time_statsQ=True",
        "description": "Number of stalled forks present at each time point.",
    },
    {
        "output": "firing_factors",
        "shape/type": "(m,) array or None",
        "returned when": "finite-factor model",
        "description": (
            "Available firing-factor pool through time, when finite factors "
            "are enabled."
        ),
    },
]


def _rsim_table(records, columns):
    try:
        import pandas as pd

        return pd.DataFrame(records, columns=columns)
    except ImportError:
        return records


def _rsim_default_text(value):
    if value is inspect._empty:
        return "required"
    if value is None:
        return "None"
    if isinstance(value, np.ndarray):
        if value.ndim == 1 and value.size > 6:
            step = value[1] - value[0]
            return f"np.arange({value[0]:g}, {value[-1]:g} + {step:g}, {step:g})"
        return repr(value)
    if isinstance(value, float) and np.isposinf(value):
        return "np.inf"
    return repr(value)


def _rsim_input():
    """Return a table describing the inputs accepted by rsim."""
    records = []
    for name, parameter in inspect.signature(rsim).parameters.items():
        if parameter.kind in {inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL}:
            continue
        records.append(
            {
                "argument": name,
                "default": _rsim_default_text(parameter.default),
                "description": _RSIM_INPUT_DESCRIPTIONS[name],
            }
        )
    return _rsim_table(records, ("argument", "default", "description"))


def _rsim_output():
    """Return a table describing the dictionary outputs produced by rsim."""
    return _rsim_table(
        _RSIM_OUTPUT_RECORDS,
        ("output", "shape/type", "returned when", "description"),
    )


rsim.input = _rsim_input  # type: ignore[reportFunctionMemberAccess]
rsim.output = _rsim_output  # type: ignore[reportFunctionMemberAccess]
