"""Shared numerical and utility helpers for DNAscape."""

import time
import numpy as np
from numpy.fft import rfft, irfft

def logistic(x, k, x0):
    return 1.0 / (1.0 + np.exp(-k * (x - x0)))

def interp_nans(y):
    y = np.asarray(y, float)
    x = np.arange(y.size)
    m = np.isfinite(y)
    return y if m.sum() < 2 else np.where(m, y, np.interp(x, x[m], y[m]))

def smoothf(data, window=50):
    data = np.asarray(data, float)
    if window is None:
        return data
    n = len(data)
    if n == 0:
        return data
    window = int(window)
    if window <= 1:
        return data
    win = min(window, n)
    kernel = np.ones(win, float) / win
    size = n + win - 1
    fft_A = rfft(data, size)
    fft_B = rfft(kernel, size)
    conv_full = irfft(fft_A * fft_B, size)
    conv = conv_full[win//2 : win//2 + n]
    return conv

def rescale(arr, interval):
    new_min, new_max = interval
    arr = np.asarray(arr, dtype=float)
    old_min, old_max = np.nanmin(arr), np.nanmax(arr)
    if old_max == old_min:
        return np.full_like(arr, (new_min + new_max) / 2.0)
    return (arr - old_min) / (old_max - old_min) * (new_max - new_min) + new_min

def fast_shift_add(dst, src, shift, perQ=False):
    
    if shift == 0:
        dst += src
        return
    
    if perQ:
        # periodic wrap
        dst += np.roll(src, shift)
    else:
        # standard linear genome
        if shift > 0:
            dst[shift:] += src[:-shift]
        else:
            dst[:shift] += src[-shift:]

def match_yaxis_scales(ax1, ax2, y1, y2):
    y1 = np.asarray(y1, float)
    y2 = np.asarray(y2, float)

    # Normalise both curves
    y1n = (y1 - y1.min()) / (y1.max() - y1.min() + 1e-12)
    y2n = (y2 - y2.min()) / (y2.max() - y2.min() + 1e-12)

    # Best affine match: y2n ~ alpha*y1n + beta
    A = np.vstack([y1n, np.ones_like(y1n)]).T
    alpha, beta = np.linalg.lstsq(A, y2n, rcond=None)[0]

    y1_matched = alpha * y1n + beta

    # Combined normalised range
    lo = min(y1_matched.min(), y2n.min())
    hi = max(y1_matched.max(), y2n.max())

    # Map back to original scale
    y1_lo = lo * (y1.max() - y1.min()) + y1.min()
    y1_hi = hi * (y1.max() - y1.min()) + y1.min()

    y2_lo = lo * (y2.max() - y2.min()) + y2.min()
    y2_hi = hi * (y2.max() - y2.min()) + y2.min()

    # ------------------------------------------------------------------
    # SAFETY FOR LOG-SCALE AXES â€” enforce positive bounds > 0
    # ------------------------------------------------------------------
    eps = 1e-9
    if ax1.get_yscale() == 'log':
        y1_lo = max(y1_lo, eps)
        y1_hi = max(y1_hi, y1_lo * 1.1)

    if ax2.get_yscale() == 'log':
        y2_lo = max(y2_lo, eps)
        y2_hi = max(y2_hi, y2_lo * 1.1)

    # Apply final limits
    ax1.set_ylim(y1_lo, y1_hi)
    ax2.set_ylim(y2_lo, y2_hi)

def mean_squared_error(a, b):
    return float(np.mean((a - b) ** 2))

def hms(t):
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def arrint(a, b):
    return np.intersect1d(a, b)

def arrcomp(a, b):
    return np.setdiff1d(a, b)

def etaf(it, maxiter, start_time):
    elapsed = time.time() - start_time
    sims_left = maxiter - it
    avg_time_per_sim = elapsed / it
    eta_sec = sims_left * avg_time_per_sim
    print(
        f"[{it}/{maxiter}]  Elapsed: {hms(elapsed)}  ETA: {hms(eta_sec)}",
        end="\r",
        flush=True,
    )

def etaferr(it, maxiter, start_time, err):
    elapsed = time.time() - start_time
    sims_left = maxiter - it
    avg_time_per_sim = elapsed / it
    eta_sec = sims_left * avg_time_per_sim
    print(
        f"[{it}/{maxiter}]  Elapsed: {hms(elapsed)}  ETA: {hms(eta_sec)}  MSE: {err:.4e}",
        end="\r",
        flush=True,
    )
