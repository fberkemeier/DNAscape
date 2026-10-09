"""Homogeneous timing preserves the full discrete model without repeated bins."""
import numpy as np
import pytest

from dnascape import fitting
from dnascape.mapping import map_firing_timing


@pytest.mark.parametrize("count", (2, 9, 32, 201))
@pytest.mark.parametrize("rate", (0.0, 0.003, 2.0))
@pytest.mark.parametrize("periodic", (False, True))
@pytest.mark.parametrize("radius", (1.0, 7.0, 400.0))
def test_constant_kernel_matches_explicit_profile(count, rate, periodic, radius):
    options = dict(fork_speed=1.4, resolution=0.5, neighbourhood_range=radius, perQ=periodic)
    expected = map_firing_timing(np.full(count, rate), **options)
    actual = map_firing_timing(rate, n_bins=count, **options)
    np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=1e-11)


def test_constant_fit_preserves_rate_and_best_timing(monkeypatch):
    observed = np.linspace(12.0, 30.0, 257)
    options = dict(maxiter=4, verbose=False, neighbourhood_range=80.0)
    actual = fitting.map_timing_constant_firing(observed, **options)
    original = fitting.map_firing_timing

    def explicit(rate, **kwargs):
        count = kwargs.pop("n_bins")
        return original(np.full(count, rate), **kwargs)

    monkeypatch.setattr(fitting, "map_firing_timing", explicit)
    expected = fitting.map_timing_constant_firing(observed, **options)
    assert float(actual) == pytest.approx(float(expected), rel=1e-14)
    np.testing.assert_allclose(
        actual._rmap_info["replication_timing_fit"],
        expected._rmap_info["replication_timing_fit"], rtol=2e-14,
    )
