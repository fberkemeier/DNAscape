"""Deterministic mapping functions between replication observables."""

import numpy as np
from scipy.integrate import cumulative_trapezoid

from .utils import fast_shift_add, interp_nans, logistic, smoothf


def _normalise_resolution_x(resolution=1.0, resolution_x=None):
    if resolution_x is not None:
        resolution = resolution_x
    resolution = float(resolution)
    if not np.isfinite(resolution) or resolution <= 0:
        raise ValueError("resolution_x must be a finite positive value.")
    return resolution


def _n_bins_from_interval(resolution_x, genome_length=None, interval_kb=None):
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
    n = int(np.ceil(genome_length / resolution_x))
    if n <= 0:
        raise ValueError("genome_length/resolution_x must define at least one bin.")
    return n


def _spatial_length(field):
    arr = np.asarray(field)
    if arr.size == 1:
        return None
    if arr.ndim == 1:
        return int(arr.size)
    if arr.ndim == 2:
        if arr.shape[0] == 1 and arr.shape[1] > 1:
            return None
        if arr.shape[1] == 1 and arr.shape[0] > 1:
            return int(arr.shape[0])
        return int(arr.shape[1])
    return None


def _resolve_n_bins_from_fields(*fields, n_bins=None, resolution_x=1.0, genome_length=None, interval_kb=None):
    candidates = [length for length in (_spatial_length(field) for field in fields) if length is not None]
    interval_n = _n_bins_from_interval(
        resolution_x,
        genome_length=genome_length,
        interval_kb=interval_kb,
    )
    if interval_n is not None:
        candidates.append(interval_n)

    if n_bins is not None:
        n = int(n_bins)
        if n <= 0:
            raise ValueError("n_bins must be a positive integer.")
        if any(candidate != n for candidate in candidates):
            raise ValueError("n_bins is inconsistent with supplied spatial fields or interval.")
        return n

    unique = set(candidates)
    if len(unique) == 1:
        return unique.pop()
    if len(unique) > 1:
        raise ValueError("Spatial fields or interval imply inconsistent numbers of bins.")
    return None


def _edge_fork_speed(fork_speed, n, perQ=False):
    """Return scalar or edge-local speeds for timing-gradient maps."""
    v = np.asarray(fork_speed, dtype=float)
    if v.size == 1:
        scalar = float(v.reshape(-1)[0])
        if not np.isfinite(scalar) or scalar <= 0:
            raise ValueError("fork_speed must be finite and positive.")
        return scalar

    v = np.squeeze(v)
    if v.ndim != 1 or v.size != n:
        raise ValueError("fork_speed must be scalar or an x-dependent array matching timing.")
    if np.any(~np.isfinite(v)) or np.any(v <= 0):
        raise ValueError("fork_speed entries must be finite and positive.")

    if perQ:
        return 0.5 * (v + np.roll(v, 1))

    edge_speed = np.empty_like(v, dtype=float)
    edge_speed[0] = v[0]
    edge_speed[1:] = 0.5 * (v[1:] + v[:-1])
    return edge_speed


def _cumtrapz_rows(y, x):
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    dx = np.diff(x)
    out = np.zeros_like(y, dtype=float)
    out[:, 1:] = np.cumsum(0.5 * (y[:, :-1] + y[:, 1:]) * dx[None, :], axis=1)
    return out


def _cumtrapz_1d(y, x):
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(y, dtype=float)
    out[1:] = np.cumsum(0.5 * (y[:-1] + y[1:]) * np.diff(x))
    return out


def _field_time_length(field, n_positions):
    arr = np.asarray(field)
    if arr.size == 1 or arr.ndim <= 1:
        return None
    if arr.ndim == 2:
        if arr.shape[0] == 1 and arr.shape[1] > 1:
            return int(arr.shape[1])
        if arr.shape[1] == 1:
            return None
        if arr.shape[1] == n_positions:
            return int(arr.shape[0])
    return None


def _fraction_time_axis(fields, n_positions, resolution_time=1.0, time_grid=None, max_time=None):
    resolution_time = float(resolution_time)
    if not np.isfinite(resolution_time) or resolution_time <= 0:
        raise ValueError("resolution_t must be a finite positive value.")

    if time_grid is not None:
        t = np.asarray(time_grid, dtype=float)
        if t.ndim != 1 or t.size == 0:
            raise ValueError("time_grid must be a non-empty one-dimensional array.")
        if np.any(~np.isfinite(t)) or (t.size > 1 and np.any(np.diff(t) <= 0)):
            raise ValueError("time_grid must contain finite, strictly increasing values.")
        return t

    lengths = [
        length for length in (_field_time_length(field, n_positions) for field in fields)
        if length is not None
    ]
    if lengths:
        return np.arange(max(lengths), dtype=float) * resolution_time

    if max_time is None:
        raise ValueError("replicated_fraction_xt requires time_grid or max_time for time-independent inputs.")
    max_time = float(max_time)
    if not np.isfinite(max_time) or max_time <= 0:
        raise ValueError("max_time must be finite and positive.")
    return np.arange(0.0, max_time + resolution_time, resolution_time)


def _interp_time_rows(arr, time_grid, resolution_time):
    arr = np.asarray(arr, dtype=float)
    if arr.shape[0] == time_grid.size:
        return arr.copy()

    source_time = time_grid[0] + np.arange(arr.shape[0], dtype=float) * float(resolution_time)
    idx = np.searchsorted(source_time, time_grid, side="right") - 1
    idx = np.clip(idx, 0, source_time.size - 1)
    nxt = np.clip(idx + 1, 0, source_time.size - 1)
    denom = np.maximum(source_time[nxt] - source_time[idx], 1e-12)
    w = np.clip((time_grid - source_time[idx]) / denom, 0.0, 1.0)
    return (1.0 - w)[:, None] * arr[idx] + w[:, None] * arr[nxt]


def _expand_space_time_field(field, n_time, n_positions, time_grid, resolution_time, name, positive=False):
    arr = np.asarray(field, dtype=float)
    if arr.size == 1:
        out = np.full((n_time, n_positions), float(arr.reshape(-1)[0]), dtype=float)
    elif arr.ndim == 1:
        if arr.size != n_positions:
            raise ValueError(f"{name} must be scalar, x-dependent, time-dependent, or space-time dependent.")
        out = np.broadcast_to(arr[None, :], (n_time, n_positions)).copy()
    elif arr.ndim == 2 and arr.shape[0] == 1 and arr.shape[1] > 1:
        temporal = _interp_time_rows(arr.reshape(-1, 1), time_grid, resolution_time).reshape(-1)
        out = np.broadcast_to(temporal[:, None], (n_time, n_positions)).copy()
    elif arr.ndim == 2 and arr.shape[1] == 1:
        if arr.shape[0] != n_positions:
            raise ValueError(f"{name} column profiles must match the number of positions.")
        out = np.broadcast_to(arr.reshape(1, -1), (n_time, n_positions)).copy()
    elif arr.ndim == 2 and arr.shape[1] == n_positions:
        out = _interp_time_rows(arr, time_grid, resolution_time)
    else:
        raise ValueError(f"{name} must be scalar, x-dependent, time-dependent, or space-time dependent.")

    if np.any(~np.isfinite(out)) or np.any(out < 0):
        raise ValueError(f"{name} must contain finite values >= 0.")
    if positive and np.any(out <= 0):
        raise ValueError(f"{name} must contain finite positive values.")
    return out


def _integrate_cdf_between(cdf_row, x, left, right):
    a = np.clip(np.minimum(left, right), x[0], x[-1])
    b = np.clip(np.maximum(left, right), x[0], x[-1])
    return np.maximum(np.interp(b, x, cdf_row) - np.interp(a, x, cdf_row), 0.0)


def _replicated_fraction_constant_speed_static_I(x, time_grid, firing_rate_x, fork_speed):
    F = _cumtrapz_1d(firing_rate_x, x)
    G = _cumtrapz_1d(firing_rate_x * x, x)
    A = np.zeros((time_grid.size, x.size), dtype=float)

    for i, t in enumerate(time_grid):
        radius = fork_speed * t
        left = np.clip(x - radius, x[0], x[-1])
        right = np.clip(x + radius, x[0], x[-1])

        F_left = np.interp(left, x, F)
        G_left = np.interp(left, x, G)
        F_right = np.interp(right, x, F)
        G_right = np.interp(right, x, G)

        left_mass = F - F_left
        right_mass = F_right - F
        left_moment = G - G_left
        right_moment = G_right - G

        A[i] = (
            (t - x / fork_speed) * left_mass
            + left_moment / fork_speed
            + (t + x / fork_speed) * right_mass
            - right_moment / fork_speed
        )

    return np.maximum(A, 0.0)


def _replicated_fraction_constant_speed(x, time_grid, firing_rate_xt, fork_speed):
    I_mid = 0.5 * (firing_rate_xt[:-1] + firing_rate_xt[1:])
    cdf = _cumtrapz_rows(I_mid, x)
    A = np.zeros_like(firing_rate_xt, dtype=float)

    for n in range(1, time_grid.size):
        total = np.zeros(x.size, dtype=float)
        for k in range(n):
            dt = time_grid[k + 1] - time_grid[k]
            tau_mid = 0.5 * (time_grid[k + 1] + time_grid[k])
            radius = fork_speed * (time_grid[n] - tau_mid)
            total += dt * _integrate_cdf_between(cdf[k], x, x - radius, x + radius)
        A[n] = total
    return A


def _replicated_fraction_general_speed(x, time_grid, firing_rate_xt, fork_speed_xt):
    I_mid = 0.5 * (firing_rate_xt[:-1] + firing_rate_xt[1:])
    cdf = _cumtrapz_rows(I_mid, x)
    A = np.zeros_like(firing_rate_xt, dtype=float)

    def velocity_at(pos, tau):
        if tau <= time_grid[0]:
            row = fork_speed_xt[0]
        elif tau >= time_grid[-1]:
            row = fork_speed_xt[-1]
        else:
            k = np.searchsorted(time_grid, tau) - 1
            w = (tau - time_grid[k]) / (time_grid[k + 1] - time_grid[k])
            row = (1.0 - w) * fork_speed_xt[k] + w * fork_speed_xt[k + 1]
        return np.interp(pos, x, row, left=0.0, right=0.0)

    def step(pos, tau, h, sign):
        k1 = sign * velocity_at(pos, tau)
        mid = np.clip(pos + 0.5 * h * k1, x[0], x[-1])
        k2 = sign * velocity_at(mid, tau + 0.5 * h)
        return np.clip(pos + h * k2, x[0], x[-1])

    for n in range(1, time_grid.size):
        left = x.copy()
        right = x.copy()
        total = np.zeros(x.size, dtype=float)
        for k in range(n - 1, -1, -1):
            dt = time_grid[k + 1] - time_grid[k]
            tau_hi = time_grid[k + 1]
            left_mid = step(left, tau_hi, -0.5 * dt, sign=1.0)
            right_mid = step(right, tau_hi, -0.5 * dt, sign=-1.0)
            total += dt * _integrate_cdf_between(cdf[k], x, left_mid, right_mid)
            left = step(left, tau_hi, -dt, sign=1.0)
            right = step(right, tau_hi, -dt, sign=-1.0)
        A[n] = total
    return A


def map_replicated_fraction_xt(
    firing_rate,
    fork_speed=1.4,
    resolution=1.0,
    resolution_x=None,
    resolution_time=1.0,
    resolution_t=None,
    time_grid=None,
    max_time=None,
    max_rep_time=None,
    n_bins=None,
    genome_length=None,
    interval_kb=None,
    perQ=False,
):
    resolution = _normalise_resolution_x(resolution, resolution_x=resolution_x)
    if resolution_t is not None:
        resolution_time = resolution_t
    if max_rep_time is not None and max_time is None:
        max_time = max_rep_time
    if perQ:
        raise ValueError("Analytical replicated_fraction_xt currently supports linear geometry only.")

    n_bins = _resolve_n_bins_from_fields(
        firing_rate,
        fork_speed,
        n_bins=n_bins,
        resolution_x=resolution,
        genome_length=genome_length,
        interval_kb=interval_kb,
    )
    if n_bins is None:
        raise ValueError("Cannot infer genomic length. Provide an x-dependent field, n_bins, genome_length, or interval_kb.")

    start = float(interval_kb[0]) if interval_kb is not None else 0.0
    x = start + np.arange(int(n_bins), dtype=float) * resolution
    t = _fraction_time_axis(
        (firing_rate, fork_speed),
        int(n_bins),
        resolution_time=resolution_time,
        time_grid=time_grid,
        max_time=max_time,
    )

    I = _expand_space_time_field(firing_rate, t.size, int(n_bins), t, resolution_time, "firing_rate")
    v = _expand_space_time_field(fork_speed, t.size, int(n_bins), t, resolution_time, "fork_speed", positive=True)

    v_raw = np.asarray(fork_speed, dtype=float)
    if v_raw.size == 1:
        speed = float(v_raw.reshape(-1)[0])
        if _field_time_length(firing_rate, int(n_bins)) is None:
            A = _replicated_fraction_constant_speed_static_I(x, t, I[0], speed)
        else:
            A = _replicated_fraction_constant_speed(x, t, I, speed)
    else:
        A = _replicated_fraction_general_speed(x, t, I, v)

    return -np.expm1(-np.maximum(A, 0.0))


def map_replicated_fraction(*args, **kwargs):
    return np.nanmean(map_replicated_fraction_xt(*args, **kwargs), axis=1)


def bwmap(chrom, cell_line='Rat', bin_size=10_000,
          bins=["S1","S2","S3","S4"],
          min_total=1e-9, fill_nans=True):

    from scipy.optimize import curve_fit
    try:
        import pybigtools
    except ImportError as exc:
        raise ImportError("bwmap requires optional dependency 'pybigtools'.") from exc

    x = np.arange(1, len(bins) + 1, dtype=float)
    bws = [pybigtools.open(f"data/repli-seq/RepliSeq_{cell_line}_{b}.bw", "r") for b in bins]

    chroms = bws[0].chroms()
    c = str(chrom)
    if c not in chroms:
        c = f"chr{c}" if (not c.startswith("chr") and f"chr{c}" in chroms) else (c[3:] if c.startswith("chr") and c[3:] in chroms else None)
    if c is None:
        raise KeyError(f"Chromosome {chrom!r} not in bigWig.")

    L = int(chroms[c])
    n = (L + bin_size - 1) // bin_size

    frac = np.vstack([
        np.asarray(bw.values(c, 0, L, bins=n, summary="mean", missing=np.nan), float)
        for bw in bws
    ])
    for bw in bws: bw.close()

    rt = np.full(n, np.nan, float)
    for j in range(n):
        s = frac[:, j]
        if np.any(~np.isfinite(s)): 
            continue
        tot = s.sum()
        if tot < min_total:
            continue
        cum = np.cumsum(s / tot)
        try:
            (k, x0), _ = curve_fit(logistic, x, cum,
                                   p0=(1.0, (len(bins)+1)/2),
                                   maxfev=500)
            rt[j] = x0
        except Exception:
            pass

    return interp_nans(rt) if fill_nans else rt

def map_firing_timing(
    firing_rate,
    fork_speed=1.4,
    resolution=1,
    resolution_x=None,
    neighbourhood_range=2000,
    perQ=False,
    n_bins=None,
    genome_length=None,
    interval_kb=None,
):
    """Map firing rate to replication timing.

    ``firing_rate`` may be a spatial profile ``I(x)`` or a scalar ``I0``.
    Scalar inputs are expanded using the length inferred from any spatial input,
    or from ``n_bins``, ``genome_length`` or ``interval_kb``. ``fork_speed`` may
    be a scalar ``v0`` or a spatial effective speed profile ``v(x)`` with the
    same length as the firing profile.
    """
    resolution = _normalise_resolution_x(resolution, resolution_x=resolution_x)
    n_bins = _resolve_n_bins_from_fields(
        firing_rate,
        fork_speed,
        n_bins=n_bins,
        resolution_x=resolution,
        genome_length=genome_length,
        interval_kb=interval_kb,
    )

    x = np.asarray(firing_rate, dtype=float)
    if x.size == 1:
        if n_bins is None:
            raise ValueError(
                "Cannot infer the number of genomic bins from scalar inputs. "
                "Provide a spatial firing_rate/fork_speed profile, or pass "
                "n_bins, genome_length, or interval_kb."
            )
        value = float(x.reshape(-1)[0])
        x = np.full(n_bins, value, dtype=float)
    elif x.ndim == 1:
        x = x.astype(float, copy=False)
        if n_bins is not None and int(n_bins) != x.size:
            raise ValueError("n_bins must match the length of firing_rate.")
    elif x.ndim == 2 and x.shape[1] == 1:
        x = x.reshape(-1).astype(float, copy=False)
        if n_bins is not None and int(n_bins) != x.size:
            raise ValueError("n_bins must match the length of firing_rate.")
    else:
        raise ValueError("firing_rate must be scalar, (n,), or (n,1) for this deterministic map.")

    if np.any(~np.isfinite(x)) or np.any(x < 0):
        raise ValueError("firing_rate must contain finite values >= 0.")

    n = x.size
    fork_speed_arr = np.asarray(fork_speed, dtype=float)
    if fork_speed_arr.size == 1:
        fork_speed = float(fork_speed_arr.reshape(-1)[0])
        if not np.isfinite(fork_speed) or fork_speed <= 0:
            raise ValueError("fork_speed must be a finite positive scalar.")
        fork_speed_per_bin = fork_speed / resolution
    elif fork_speed_arr.ndim == 1:
        if fork_speed_arr.size != n:
            raise ValueError("fork_speed must match the length of firing_rate.")
        if np.any(~np.isfinite(fork_speed_arr)) or np.any(fork_speed_arr <= 0):
            raise ValueError("fork_speed must contain finite positive values.")
        fork_speed_per_bin = fork_speed_arr.astype(float, copy=False) / resolution
    elif fork_speed_arr.ndim == 2 and fork_speed_arr.shape[1] == 1:
        fork_speed_arr = fork_speed_arr.reshape(-1).astype(float, copy=False)
        if fork_speed_arr.size != n:
            raise ValueError("fork_speed must match the length of firing_rate.")
        if np.any(~np.isfinite(fork_speed_arr)) or np.any(fork_speed_arr <= 0):
            raise ValueError("fork_speed must contain finite positive values.")
        fork_speed_per_bin = fork_speed_arr / resolution
    else:
        raise ValueError("fork_speed must be scalar, (n,), or (n,1) for this deterministic map.")

    # Requested neighbourhood radius in bins
    L_req = max(1, int(neighbourhood_range / resolution))

    # Finite periodic cap to avoid repeated counting on the circle.
    # Your proposed R = ceil((n-3)/2) equals (n-2)//2 for integers.
    if perQ:
        R = max(0, (n - 2) // 2)
        L = min(L_req, R)
    else:
        L = L_req

    y = np.zeros_like(x)
    last_raw = np.zeros_like(x)
    last_exp = np.ones_like(x)
    unitary = x.copy()

    for k in range(L + 1):
        if k:
            fast_shift_add(unitary, x,  k, perQ=perQ)
            fast_shift_add(unitary, x, -k, perQ=perQ)

        exp2_raw = last_raw + unitary / fork_speed_per_bin
        exp2 = np.exp(-exp2_raw)
        y += (last_exp - exp2) / np.clip(unitary, 1e-80, None)
        last_raw, last_exp = exp2_raw, exp2

    return y

def map_timing_directionality(
    timing,
    fork_speed=1.4,
    resolution=1.0,
    resolution_x=None,
    T0=None,
    smoothw=None,
    perQ=False,
):
    resolution = _normalise_resolution_x(resolution, resolution_x=resolution_x)
    T = np.asarray(timing, dtype=float)
    if T0 is None:
        T0 = T[0] if T.size else 0.0
    edge_speed = _edge_fork_speed(fork_speed, T.size, perQ=perQ)

    dT_per_bin = np.empty_like(T)

    if perQ:
        # Periodic finite difference: wrap last to first.
        dT_per_bin[:] = T - np.roll(T, 1)
    else:
        # Non-periodic behaviour, anchored by T0 at the first bin.
        dT_per_bin[0] = T[0] - T0
        dT_per_bin[1:] = np.diff(T)

    # RFD = v_edge * dT/dx with dx = resolution.
    rfd = (edge_speed / resolution) * dT_per_bin

    # optional smoothing of RFD
    rfd = smoothf(rfd, smoothw)

    # enforce biological bounds
    rfd = np.clip(rfd, -1.0, 1.0)

    return rfd

def map_directionality_timing(
    rfd,
    fork_speed=1.4,
    resolution=1.0,
    resolution_x=None,
    T0=0.0,
    smoothw=None,
    perQ=False,
):
    resolution = _normalise_resolution_x(resolution, resolution_x=resolution_x)
    rfd = np.asarray(rfd, dtype=float)
    edge_speed = _edge_fork_speed(fork_speed, rfd.size, perQ=perQ)

    # optional smoothing of RFD before integration
    rfd = smoothf(rfd, smoothw)

    # Each bin contributes dT = (resolution / v_edge) * RFD.
    dT_per_bin = (resolution / edge_speed) * rfd

    if perQ:
        # periodic integration over circle
        T = np.cumsum(dT_per_bin)
        # remove global offset by enforcing circular consistency
        shift = (T[-1] + dT_per_bin[0]) / len(T)
        T = T - shift * np.arange(len(T))
    else:
        # standard linear integration
        T = T0 + np.cumsum(dT_per_bin)

    return T


def __getattr__(name):
    if name in {"rmap", "DataTypeMapper", "MappingSpec"}:
        from . import routing
        return getattr(routing, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
