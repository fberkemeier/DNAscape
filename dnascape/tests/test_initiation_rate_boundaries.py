"""Regression tests for endpoint-stable spatial initiation-rate inference."""

import math

import numpy as np
import pytest

from dnascape import rmap
from dnascape.fitting import _pad_spatial_firing_inputs, map_timing_firing
from dnascape.mapping import map_firing_timing


def _uniform_timing(n=100, resolution=10.0, neighbourhood_range=200.0):
    truth = np.full(n, 1e-3, dtype=float)
    timing = map_firing_timing(
        truth,
        fork_speed=1.4,
        resolution=resolution,
        neighbourhood_range=neighbourhood_range,
        perQ=True,
    )
    return truth, timing


def test_linear_subregion_fit_does_not_convert_missing_flanks_to_rate_spikes():
    truth, timing = _uniform_timing()

    inferred = map_timing_firing(
        timing,
        fork_speed=1.4,
        resolution=10.0,
        neighbourhood_range=200.0,
        maxiter=8,
        perQ=False,
        verbose=False,
    )

    edge_bins = inferred._rmap_info["fit_edge_bins"]
    taper_bins = inferred._rmap_info["fit_taper_bins"]
    assert edge_bins == 10
    assert taper_bins == 3
    assert np.isfinite(inferred._rmap_info["fit_mse"])

    # The old safeguard froze 25 bins at each end at this smooth analytical
    # seed.  The context-aware fit now updates every bin and uses only a narrow
    # three-bin taper.
    initial = (math.pi / (4.0 * (1.4 / 10.0))) * timing**-2
    assert np.count_nonzero(~np.isclose(inferred, initial, rtol=1e-8, atol=1e-15)) == inferred.size

    core = np.median(inferred[edge_bins:-edge_bins])
    terminal_max = max(np.max(inferred[:edge_bins]), np.max(inferred[-edge_bins:]))
    assert terminal_max < 1.1 * core
    np.testing.assert_allclose(inferred, truth, rtol=0.04)


def test_refine_edge_width_is_an_upper_bound_not_a_frozen_band():
    _, timing = _uniform_timing()

    inferred = map_timing_firing(
        timing,
        fork_speed=1.4,
        resolution=10.0,
        neighbourhood_range=200.0,
        maxiter=4,
        perQ=False,
        refine_edge_kb=20.0,
        verbose=False,
    )

    assert inferred._rmap_info["fit_edge_bins"] == 2
    assert inferred._rmap_info["fit_taper_bins"] == 2


@pytest.mark.parametrize(
    ("speed", "expected_shape"),
    (
        (1.4, None),
        (np.arange(4, dtype=float) + 1.0, (8,)),
        ((np.arange(4, dtype=float) + 1.0).reshape(4, 1), (8, 1)),
        (np.arange(5, dtype=float).reshape(1, 5) + 1.0, (1, 5)),
        (np.arange(20, dtype=float).reshape(4, 5) + 1.0, (8, 5)),
    ),
)
def test_endpoint_context_preserves_rsim_fork_speed_forms(speed, expected_shape):
    rates = np.arange(4, dtype=float) + 1.0

    padded_rates, padded_speed, output_slice = _pad_spatial_firing_inputs(
        rates,
        speed,
        edge_bins=2,
    )

    assert padded_rates.shape == (8,)
    np.testing.assert_array_equal(padded_rates[output_slice], rates)
    if expected_shape is None:
        assert padded_speed == pytest.approx(1.4)
    else:
        assert np.asarray(padded_speed).shape == expected_shape
        original = np.asarray(speed)
        if original.shape[0] == rates.size:
            np.testing.assert_array_equal(np.asarray(padded_speed)[2:6], original)
        else:
            np.testing.assert_array_equal(padded_speed, original)


def test_endpoint_context_rejects_mismatched_spatial_speed_fields():
    with pytest.raises(ValueError, match="spatial length matching initiation_rate"):
        _pad_spatial_firing_inputs(
            np.ones(4),
            np.ones((3, 2)),
            edge_bins=2,
        )


def test_true_linear_chromosome_ends_can_still_be_fitted_explicitly():
    _, timing = _uniform_timing()

    inferred = map_timing_firing(
        timing,
        fork_speed=1.4,
        resolution=10.0,
        neighbourhood_range=200.0,
        maxiter=8,
        perQ=False,
        refine_edge_kb=0.0,
        verbose=False,
    )

    assert inferred._rmap_info["fit_edge_bins"] == 0
    core = np.median(inferred[25:75])
    # A flat timing profile at a real chromosome end legitimately requires a
    # higher terminal initiation rate because there are no flanking origins.
    assert inferred[0] > 1.5 * core
    assert inferred[-1] > 1.5 * core


def test_periodic_fit_has_full_wraparound_support_regardless_of_edge_setting():
    _, timing = _uniform_timing(n=60, neighbourhood_range=100.0)
    common = {
        "fork_speed": 1.4,
        "resolution": 10.0,
        "neighbourhood_range": 100.0,
        "maxiter": 4,
        "perQ": True,
        "verbose": False,
    }

    protected = map_timing_firing(timing, refine_edge_kb=1000.0, **common)
    unprotected = map_timing_firing(timing, refine_edge_kb=0.0, **common)

    np.testing.assert_allclose(protected, unprotected)
    assert protected._rmap_info["fit_edge_bins"] == 0


def test_rmap_exposes_edge_refinement_for_spatial_firing_routes():
    route = next(
        spec for spec in rmap.routes() if spec.method == "timing_and_speed_to_initiation_rate"
    )
    assert "refine_edge_kb" in route.optional_assumptions


@pytest.mark.parametrize("value", [-1.0, math.inf, math.nan])
def test_invalid_edge_width_is_rejected(value):
    _, timing = _uniform_timing(n=20, neighbourhood_range=50.0)
    with pytest.raises(ValueError, match="refine_edge_kb"):
        map_timing_firing(
            timing,
            resolution=10.0,
            neighbourhood_range=50.0,
            maxiter=0,
            refine_edge_kb=value,
            verbose=False,
        )
