"""Scientific timing estimates from sixteen equal S-phase fractions."""

from concurrent.futures import CancelledError

import numpy as np
import pytest
from scipy.special import expit

from dnascape.workflows.repliseq import replication_timing_from_fractions


def test_sigmoid_recovers_known_cumulative_midpoints_and_scales_to_minutes() -> None:
    boundaries = np.arange(17)
    midpoints = np.array([4., 8., 12.])
    cumulative = expit((boundaries[None, :] - midpoints[:, None]) / 0.4)
    cumulative[:, 0] = 0
    cumulative[:, -1] = 1
    signals = np.diff(cumulative, axis=1) * 100
    fitted = replication_timing_from_fractions(signals)
    np.testing.assert_allclose(fitted, midpoints / 16 * 600, atol=0.01)
    np.testing.assert_allclose(replication_timing_from_fractions(signals * 7), fitted, atol=1e-7)
    np.testing.assert_allclose(replication_timing_from_fractions(signals, s_phase_duration_min=300), fitted / 2)


@pytest.mark.parametrize("method", ["sigmoid", "mean"])
def test_single_fraction_early_middle_and_late_timing_uses_equal_intervals(method: str) -> None:
    signals = np.zeros((3, 16))
    signals[np.arange(3), [0, 7, 15]] = 100
    timing = replication_timing_from_fractions(signals, method=method)
    np.testing.assert_allclose(timing, [18.75, 281.25, 581.25], atol=0.01)


def test_mean_uses_signal_mass_and_equal_interval_midpoints() -> None:
    signal = np.zeros((1, 16))
    signal[0, 0] = 30
    signal[0, 15] = 70
    result = replication_timing_from_fractions(signal, method="mean")
    np.testing.assert_allclose(result, [(0.3 * 0.5 + 0.7 * 15.5) / 16 * 600])


@pytest.mark.parametrize("method", ["sigmoid", "mean"])
def test_incomplete_and_zero_signal_bins_stay_missing_without_changing_positions(method: str) -> None:
    values = np.ones((5, 16))
    values[1] = np.nan
    values[2, 8] = np.nan
    values[3] = 0
    original = values.copy()
    result = replication_timing_from_fractions(values, method=method)
    assert result.shape == (5,)
    assert np.isfinite(result[[0, 4]]).all()
    assert np.isnan(result[1:4]).all()
    np.testing.assert_allclose(values, original, equal_nan=True)


@pytest.mark.parametrize("duration", [0, -1, np.inf, np.nan])
def test_invalid_s_phase_duration_is_rejected(duration: float) -> None:
    with pytest.raises(ValueError, match="duration"):
        replication_timing_from_fractions(np.ones((2, 16)), s_phase_duration_min=duration)


def test_invalid_fractions_are_rejected_and_fitting_is_cancellable() -> None:
    with pytest.raises(ValueError, match="sixteen"):
        replication_timing_from_fractions(np.ones((2, 15)))
    with pytest.raises(ValueError, match="non-negative"):
        replication_timing_from_fractions(-np.ones((2, 16)))
    with pytest.raises(ValueError, match="non-negative"):
        replication_timing_from_fractions(np.full((2, 16), np.inf))
    reports = []

    def cancel():
        if reports:
            raise CancelledError

    with pytest.raises(CancelledError):
        replication_timing_from_fractions(np.ones((100, 16)), progress=reports.append, cancel=cancel)
    assert reports and reports[0] == 0
