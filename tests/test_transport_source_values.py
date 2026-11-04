"""Prediction-only source integration keeps centered-cell boundary behavior."""

import numpy as np
import pytest

from dnascape._hazard_transport import FastTransportHazard


@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("n", [1, 2, 17])
@pytest.mark.parametrize("dx", [.3, 1., 2.7])
def test_source_value_without_record_matches_cell_integral(periodic, n, dx):
    rng = np.random.default_rng(731 + n)
    operator = FastTransportHazard(np.ones((2, n)), [0., 1.], dx, periodic)
    prefix = np.r_[0., np.cumsum(rng.uniform(.1, 3., n)) * dx]
    positions = np.r_[
        (np.arange(-3 * n, 3 * n + 1) - .5) * dx,
        rng.uniform(-3 * n * dx, 3 * n * dx, 200),
    ]
    expected, _ = operator._source_sample(prefix, positions, with_position=False)
    actual, record = operator._source_sample(
        prefix, positions, with_position=False, cache_geometry=False,
    )
    assert record is None
    np.testing.assert_allclose(actual, expected, rtol=3e-14, atol=3e-14)


@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("n", [1, 2, 3, 17])
def test_potential_prediction_without_adjoint_record(periodic, n):
    rng = np.random.default_rng(121 + n)
    operator = FastTransportHazard(np.ones((2, n)), [0., 1.], .3, periodic)
    values = np.cumsum(rng.uniform(.01, 4., n))
    for direction in (-1., 1.):
        positions = operator.x + rng.uniform(-2., 2., n)
        expected, _ = operator._potential_sample(
            values, positions, values[-1] + 1., direction, with_position=False,
        )
        actual, record = operator._potential_sample(
            values, positions, values[-1] + 1., direction,
            with_position=False, record=False,
        )
        assert record is None
        np.testing.assert_array_equal(actual, expected)
