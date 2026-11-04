"""Independent cell-integral and adjoint checks for the optimized kernels."""

from types import SimpleNamespace

import numpy as np
import pytest

from dnascape._hazard_static import StationaryHazard
from dnascape._hazard_transport import FastTransportHazard


@pytest.mark.parametrize("periodic", [False, True])
def test_inverse_travel_coordinate_at_cells_boundaries_and_outside_the_domain(periodic):
    op = StationaryHazard(np.array([1., 2., .5]), np.array([0., 1.]), 2., periodic)
    query = np.array([-1., 0., 1., 2., 2.5, 3., 5., 7., 8.])
    expected = np.array([-.5 if periodic else 0., 0., 1., 2., 3., 4., 5., 6., 7. if periodic else 6.])
    np.testing.assert_allclose(op._inverse(query), expected, atol=1e-14)
    if periodic:
        np.testing.assert_allclose(op._inverse(query + 3 * 7.), expected + 3 * 6., atol=1e-13)


def _hat_integral(z, time):
    distance = min(abs(z), time)
    return np.sign(z) * (time * distance - .5 * distance**2)


@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("radius", [.6, 3.7, 2000.])
def test_stationary_hazard_matches_independently_integrated_source_cells(periodic, radius):
    speed = np.array([.5, 2.1, .7, 1.8, 1.2])
    rate = np.array([.001, .2, .003, .015, .002])
    dx, clock = 1.3, np.array([0., .17, 1.1, 4., 9.])
    op = StationaryHazard(speed, clock, dx, periodic, radius)
    actual = op.forward(rate, record=True)
    length, travel = len(rate) * dx, np.r_[0., np.cumsum(dx / speed)]
    cap = min(radius, length / 2) if periodic else radius
    for receiver in range(len(rate)):
        x = (receiver + .5) * dx
        y = travel[receiver] + .5 * dx / speed[receiver]
        for k, time in enumerate(clock):
            expected = 0.
            for source in range(len(rate)):
                for turn in ((-1, 0, 1) if periodic else (0,)):
                    left = max(source * dx + turn * length, x - cap)
                    right = min((source + 1) * dx + turn * length, x + cap)
                    if right <= left:
                        continue
                    lo = travel[source] + turn * travel[-1] + (left - source * dx - turn * length) / speed[source] - y
                    hi = travel[source] + turn * travel[-1] + (right - source * dx - turn * length) / speed[source] - y
                    expected += rate[source] * speed[source] * (_hat_integral(hi, time) - _hat_integral(lo, time))
            assert actual[k, receiver] == pytest.approx(expected, abs=2e-13)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 19])
@pytest.mark.parametrize("periodic", [False, True])
def test_interpolation_pullback_matches_value_total_and_position_perturbations(n, periodic):
    rng = np.random.default_rng(n)
    values = np.cumsum(rng.uniform(.1, 1., n))
    positions = rng.uniform(-1., n + .5, n)
    total, dx = float(values[-1] + 1.), .8
    operator = SimpleNamespace(n=n, dx=dx, periodic=periodic)
    predict = lambda v, p, t: FastTransportHazard._potential_sample(operator, v, p, t, 1.)
    output, record = predict(values, positions, total)
    g, dv, dp = (rng.normal(size=n) for _ in range(3))
    dt, step = .37, 1e-7
    gv, gt, gp = FastTransportHazard._potential_pullback(operator, g, record)
    high = predict(values + step * dv, positions + step * dp, total + step * dt)[0]
    low = predict(values - step * dv, positions - step * dp, total - step * dt)[0]
    finite = np.dot(g, high - low) / (2 * step)
    exact = np.dot(gv, dv) + gt * dt + np.dot(gp, dp)
    assert np.isfinite(output).all()
    assert finite == pytest.approx(exact, abs=3e-7, rel=3e-7)


@pytest.mark.parametrize("periodic", [False, True])
def test_general_speed_and_initiation_adjoint_with_partial_cell_radius(periodic):
    rng = np.random.default_rng(54)
    n, clock = 29, np.array([0., .4, 1.5, 4., 7.])
    speed = (1.1 + .15 * np.cos(np.arange(n) / 3))[None, :] * (1 + .03 * clock[:, None])
    rate = rng.uniform(.002, .015, speed.shape)
    seed, di, dv = (rng.normal(size=speed.shape) for _ in range(3))
    op = FastTransportHazard(speed, clock, periodic=periodic, radius=3.7)
    op.forward(rate, True)
    gi, gv = op.backward(seed)
    step = 1e-7
    high = FastTransportHazard(speed + step * dv, clock, periodic=periodic, radius=3.7).forward(rate + step * di)
    low = FastTransportHazard(speed - step * dv, clock, periodic=periodic, radius=3.7).forward(rate - step * di)
    finite = np.sum(seed * (high - low)) / (2 * step)
    assert finite == pytest.approx(np.sum(gi * di) + np.sum(gv * dv), rel=3e-6, abs=3e-6)
