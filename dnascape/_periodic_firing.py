"""Positive circular curvature regularization for fitted firing-rate fields."""

import numpy as np


class PeriodicFiringSmoother:
    """Regularize log rates with the same curvature penalty at every ring bin.

    The Fourier multiplier solves ``min ||z - log(I)||^2 + s^4 ||D2 z||^2``,
    using circular second differences. It therefore couples the two sides of
    the join without privileging that join or modifying observed timing.
    The scale is 5% of a typical fork-travel distance, at least one bin and
    at most 1/32 of the region (or one bin for very small regions).
    """

    def __init__(self, observed_timing, fork_speed, resolution):
        timing = np.asarray(observed_timing, dtype=float).reshape(-1)
        self.size = timing.size
        positive = timing[np.isfinite(timing) & (timing > 0)]
        if not self.size or not positive.size:
            raise ValueError("Periodic firing fitting requires positive timing values.")
        speed = float(np.mean(np.asarray(fork_speed, dtype=float)))
        resolution = float(resolution)
        if not np.isfinite(speed) or speed <= 0 or not np.isfinite(resolution) or resolution <= 0:
            raise ValueError("Fork speed and spatial resolution must be finite and positive.")
        self.length_bins = min(
            max(1., .05 * speed * float(np.median(positive)) / resolution),
            max(1., self.size / 32.),
        )
        self.length_kb = self.length_bins * resolution
        frequencies = np.fft.rfftfreq(self.size)
        self._transfer = 1. / (1. + 16. * self.length_bins**4 * np.sin(np.pi * frequencies)**4)

    def __call__(self, values, *, axis=-1, floor=1e-15):
        rates = np.asarray(values, dtype=float)
        if rates.ndim < 1 or not -rates.ndim <= axis < rates.ndim:
            raise ValueError("Periodic firing rates require a spatial array axis.")
        axis %= rates.ndim
        if rates.shape[axis] != self.size:
            raise ValueError("Periodic firing rates must match the observed spatial grid.")
        if np.any(~np.isfinite(rates)) or np.any(rates < 0):
            raise ValueError("Periodic firing rates must contain finite nonnegative values.")
        if not np.isfinite(floor) or floor < 0:
            raise ValueError("The firing-rate floor must be finite and nonnegative.")
        if self.size == 1:
            return np.maximum(rates, float(floor))
        minimum = max(float(floor), 1e-80)
        logs = np.log(np.maximum(rates, minimum))
        shape = [1] * rates.ndim
        shape[axis] = self._transfer.size
        smoothed = np.fft.irfft(
            np.fft.rfft(logs, axis=axis) * self._transfer.reshape(shape),
            n=self.size, axis=axis,
        )
        result = np.maximum(np.exp(smoothed), float(floor))
        # Time slices with no firing anywhere remain inactive when permitted.
        if floor <= 0:
            result = np.where(np.all(rates == 0, axis=axis, keepdims=True), 0., result)
        return result
