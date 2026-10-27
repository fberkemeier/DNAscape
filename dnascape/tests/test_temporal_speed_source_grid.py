"""Sampling timing observations must retain the complete spatial firing integral."""

from concurrent.futures import CancelledError

import numpy as np
import pytest

from dnascape.mapping import _cumtrapz_1d, _static_firing_timing_time_speed, map_replicated_fraction_xt
from dnascape.utils import progress_context


@pytest.mark.parametrize("n,dx", [(17, .3), (64, 5.)])
def test_sampled_observations_match_the_full_closed_forward_model(n, dx):
    rng = np.random.default_rng(23)
    firing = rng.uniform(0., .02, n)
    firing[::3] = 0.
    source_x = np.arange(n) * dx
    axis = np.array([0., 2., 7., 15., 30., 55.])
    speed = 1.3 + .01 * axis
    sites = np.arange(0, n, 3)
    actual, survival = _static_firing_timing_time_speed(
        source_x, _cumtrapz_1d(firing, source_x), source_x[sites], axis, speed,
    )
    full = map_replicated_fraction_xt(firing, speed[None, :], resolution=dx, time_grid=axis, n_bins=n)
    np.testing.assert_allclose(actual, np.trapezoid(1. - full, axis, axis=1)[sites], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(survival, 1. - full[sites, -1], rtol=1e-12, atol=1e-15)


def test_temporal_speed_prediction_can_be_stopped_within_an_evaluation():
    calls = 0
    def cancel():
        nonlocal calls
        calls += 1
        if calls == 8:
            raise CancelledError("Mapping stopped.")
    source_x = np.arange(64.)
    axis = np.arange(0., 100., 2.)
    with progress_context(cancel=cancel), pytest.raises(CancelledError):
        _static_firing_timing_time_speed(source_x, _cumtrapz_1d(np.full(64, .01), source_x),
                                        source_x, axis, np.full(axis.size, 1.4))
    assert calls == 8
