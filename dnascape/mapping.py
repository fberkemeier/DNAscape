"""Deterministic mapping functions between replication observables."""

import numpy as np
from scipy.integrate import cumulative_trapezoid

from .utils import fast_shift_add, interp_nans, logistic, smoothf

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
    neighbourhood_range=2000,
    perQ=False
):
    x = np.asarray(firing_rate, dtype=float)
    n = x.size
    fork_speed_per_bin = fork_speed / resolution

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
    T0=None,
    smoothw=None,
    perQ=False,
):
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
    T0=0.0,
    smoothw=None,
    perQ=False,
):
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
