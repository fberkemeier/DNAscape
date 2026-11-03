"""Conditions for a finite, unconditional mean replication time."""

import numpy as np


def eventual_sources(final_rate, resolution=1., radius=2000., periodic=False):
    """Mark receivers reached by initiation that continues indefinitely.

    Supplied kinetic profiles hold their final value after their native clock.
    With a positive bounded propagation speed, the survival tail decays to
    zero exactly when a positive measure of continuing source cells lies in
    every receiver's permitted neighbourhood. A finite initiation pulse alone
    leaves a non-zero probability of no origin firing, even if it is tiny.
    """
    rates = np.asarray(final_rate, dtype=float).reshape(-1)
    dx = float(resolution)
    radius = np.inf if radius is None else float(radius)
    if not rates.size or np.any(~np.isfinite(rates)) or np.any(rates < 0):
        raise ValueError("Final initiation rates must be finite and non-negative.")
    if not np.isfinite(dx) or dx <= 0 or np.isnan(radius) or radius <= 0:
        raise ValueError("Resolution and influence radius must be positive.")
    positive = rates > 0.
    length = rates.size * dx
    if not np.any(positive):
        return np.zeros(rates.size, dtype=bool)
    if radius >= length or periodic and 2. * radius >= length:
        return np.ones(rates.size, dtype=bool)
    edges = np.arange(rates.size + 1, dtype=float) * dx
    prefix = np.r_[0., np.cumsum(positive, dtype=float) * dx]
    centers = (np.arange(rates.size, dtype=float) + .5) * dx

    def integral(positions):
        if periodic:
            turns = np.floor(positions / length)
            local = positions - turns * length
            return turns * prefix[-1] + np.interp(local, edges, prefix)
        return np.interp(np.clip(positions, 0., length), edges, prefix)

    return integral(centers + radius) - integral(centers - radius) > 0.
