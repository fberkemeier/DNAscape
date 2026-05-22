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

    dT_per_bin = np.empty_like(T)

    if perQ:
        # periodic finite difference: wrap last â†’ first
        dT_per_bin[:] = T - np.roll(T, 1)
    else:
        # original non-periodic behaviour
        dT_per_bin[0] = T[0] - T0
        dT_per_bin[1:] = np.diff(T)

    # RFD = v * dT/dx with dx = resolution
    rfd = (fork_speed / resolution) * dT_per_bin

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

    # optional smoothing of RFD before integration
    rfd = smoothf(rfd, smoothw)

    # each bin contributes Î”T = (resolution / v) * RFD
    dT_per_bin = (resolution / fork_speed) * rfd

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
