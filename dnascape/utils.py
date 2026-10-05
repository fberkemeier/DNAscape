"""Shared numerical and utility helpers for DNAscape."""

import time
from contextlib import contextmanager
from threading import local

import numpy as np
from numpy.fft import irfft, rfft

_PROGRESS_STATE = local()


@contextmanager
def progress_context(report=None, cancel=None, *, suppress_output=False, metric_report=None):
    """Route ETA updates to one worker-local callback.

    This avoids redirecting process-global stdout when a local app runs
    scientific work in a background thread.  Outside this context ``etaf`` and
    ``etaferr`` retain their existing console behaviour.
    """

    previous = (
        getattr(_PROGRESS_STATE, "report", None),
        getattr(_PROGRESS_STATE, "cancel", None),
        getattr(_PROGRESS_STATE, "suppress_output", False),
        getattr(_PROGRESS_STATE, "metric_report", None),
    )
    _PROGRESS_STATE.report = report
    _PROGRESS_STATE.cancel = cancel
    _PROGRESS_STATE.suppress_output = bool(suppress_output)
    _PROGRESS_STATE.metric_report = metric_report
    try:
        yield
    finally:
        (
            _PROGRESS_STATE.report,
            _PROGRESS_STATE.cancel,
            _PROGRESS_STATE.suppress_output,
            _PROGRESS_STATE.metric_report,
        ) = previous


def check_cancelled():
    """Raise through the worker's cancellation callback when a run was stopped."""

    cancel = getattr(_PROGRESS_STATE, "cancel", None)
    if callable(cancel):
        cancel()


def _report_progress(value, message):
    """Report a solver stage without printing or changing its scientific options."""
    check_cancelled()
    report = getattr(_PROGRESS_STATE, "report", None)
    if callable(report):
        report(float(value), str(message))


def _progress_update(it, maxiter, eta_sec, label=None, err=None):
    check_cancelled()

    total = max(1, int(maxiter))
    iteration = min(total, max(0, int(it)))
    fraction = iteration / total
    normalized_label = str(label or "Working").strip().lower()
    metric_report = getattr(_PROGRESS_STATE, "metric_report", None)
    if err is not None and callable(metric_report):
        metric_report(normalized_label, iteration, total, float(err))

    report = getattr(_PROGRESS_STATE, "report", None)
    if not callable(report):
        return
    if normalized_label in {"rsim_finalize", "rsim_finalise", "rsim_post"}:
        value = 0.95 + 0.04 * fraction
        prefix = "Finalizing"
    elif normalized_label == "rsim":
        value = 0.04 + 0.90 * fraction
        prefix = "Simulating"
    else:
        value = 0.04 + 0.90 * fraction
        prefix = str(label or "Mapping").replace("_", " ").title()
    text = f"{prefix}... {iteration}/{total} - ETA {hms(eta_sec)}"
    if err is not None:
        text += f" - MSE {float(err):.3e}"
    report(min(0.99, value), text)


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
    conv = conv_full[win // 2 : win // 2 + n]
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
    # SAFETY FOR LOG-SCALE AXES — enforce positive bounds > 0
    # ------------------------------------------------------------------
    eps = 1e-9
    if ax1.get_yscale() == "log":
        y1_lo = max(y1_lo, eps)
        y1_hi = max(y1_hi, y1_lo * 1.1)

    if ax2.get_yscale() == "log":
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


def _eta_label(label):
    return "" if label is None else f"[{label}]  "


def etaf(it, maxiter, start_time, done=None, label=None):
    elapsed = time.time() - start_time
    sims_left = maxiter - it
    avg_time_per_sim = elapsed / it
    eta_sec = sims_left * avg_time_per_sim
    if done is None:
        done = it >= maxiter
    if done:
        eta_sec = 0.0
    _progress_update(it, maxiter, eta_sec, label=label)
    if not getattr(_PROGRESS_STATE, "suppress_output", False):
        print(
            f"{_eta_label(label)}[{it}/{maxiter}]  Elapsed: {hms(elapsed)}  ETA: {hms(eta_sec)}",
            end="\n" if done else "\r",
            flush=True,
        )


def etaferr(it, maxiter, start_time, err, done=None, label="rmap"):
    elapsed = time.time() - start_time
    sims_left = maxiter - it
    avg_time_per_sim = elapsed / it
    eta_sec = sims_left * avg_time_per_sim
    if done is None:
        done = it >= maxiter
    if done:
        eta_sec = 0.0
    _progress_update(it, maxiter, eta_sec, label=label, err=err)
    if not getattr(_PROGRESS_STATE, "suppress_output", False):
        print(
            f"{_eta_label(label)}[{it}/{maxiter}]  Elapsed: {hms(elapsed)}  ETA: {hms(eta_sec)}  MSE: {err:.4e}",
            end="\n" if done else "\r",
            flush=True,
        )
