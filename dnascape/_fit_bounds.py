"""Optional, independently specified limits for positive fitted fields."""

import numpy as np


def fit_bounds(bounds, defaults, *, label="Bounds", allow_zero=False):
    if bounds is None:
        return defaults
    if len(bounds) != 2:
        raise ValueError(f"{label} require a lower and upper value.")
    if any(value is not None and not np.isfinite(float(value)) for value in bounds):
        raise ValueError(f"{label} must contain finite positive increasing values.")
    low, high = (fallback if value is None else float(value)
                 for value, fallback in zip(bounds, defaults))
    if low >= high:
        if bounds[0] is None and high > 0:
            low = high * .1
        elif bounds[1] is None and low > 0:
            high = low * 10.
    if not np.isfinite(low) or np.isnan(high) or low < 0 or (not allow_zero and low == 0) or high <= low:
        raise ValueError(f"{label} must contain positive increasing values.")
    return low, high


def initial_rate(value, default, low, high):
    if value is None:
        return np.clip(default, low, high)
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("Initial initiation rate must be finite and positive.")
    return np.clip(np.full_like(np.asarray(default, dtype=float), value), low, high)


def scale_bounds(bounds, scale):
    return None if bounds is None else tuple(None if item is None else float(item) * scale for item in bounds)
