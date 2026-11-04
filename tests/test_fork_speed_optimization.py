"""Independent derivatives and cache/boundary invariants for speed fitting."""
import numpy as np
import pytest

from dnascape._hazard_stationary_source import StationarySourceHazard
from dnascape._hazard_transport import FastTransportHazard
from dnascape._kinetic_fit import _KineticModel


@pytest.mark.parametrize('form', ['t', 'xt'])
@pytest.mark.parametrize('shape', [.25, 1., 2., 8.])
@pytest.mark.parametrize('local_delay', [False, True])
def test_prediction_only_weibull_field_matches_differentiable_field_with_delays_and_bounds(form, shape, local_delay):
    clock = np.array([0., .3, 1., 4., 10., 30., 80.])
    n = 9
    delay = np.linspace(0., 12., n) if local_delay else np.array([3.])
    model = _KineticModel('fork_speed', form, n, clock, 20., delay, shape, True, (.2, 3.))
    parameters = model.initial(1.4, np.full(n, 20.), .001)
    if form == 'xt':
        parameters[:n] = np.linspace(-2., 2., n)
        parameters[model.shape_slice] = np.log(np.linspace(.25, 8., n))
    reference = model.evaluate(parameters)[0]
    prediction, state = model.evaluate(parameters, need_state=False)
    assert state is None
    np.testing.assert_allclose(prediction, reference, atol=1e-14, rtol=1e-14)


@pytest.mark.parametrize('periodic', [False, True])
@pytest.mark.parametrize('n', [1, 2, 3, 29])
@pytest.mark.parametrize('radius', [.37, 4.3, 2000.])
def test_compact_departure_cache_matches_uncached_joint_and_speed_only_adjoint(periodic, n, radius):
    rng = np.random.default_rng(13 + n)
    clock = np.array([0., .2, .9, 2., 4.5])
    speed = rng.uniform(.3, 2.1, (len(clock), n))
    rate = rng.uniform(.002, .02, speed.shape)
    seed = rng.normal(size=speed.shape)
    cached = FastTransportHazard(speed, clock, .8, periodic, radius)
    uncached = FastTransportHazard(speed, clock, .8, periodic, radius)
    uncached._CACHE_BYTES = 0
    np.testing.assert_array_equal(cached.forward(rate, True), uncached.forward(rate, True))
    gi, gv = cached.backward(seed)
    ui, uv = uncached.backward(seed)
    np.testing.assert_allclose(gi, ui, atol=1e-11, rtol=1e-11)
    np.testing.assert_allclose(gv, uv, atol=1e-11, rtol=1e-10)
    _, speed_only = cached.backward(seed, with_rate=False)
    np.testing.assert_allclose(gv, speed_only, atol=1e-11, rtol=1e-10)


@pytest.mark.parametrize('periodic', [False, True])
def test_cached_speed_only_adjoint_matches_directional_finite_difference(periodic):
    rng = np.random.default_rng(431)
    n, clock = 19, np.array([0., .17, .7, 1.9, 4., 6.])
    speed = rng.uniform(.4, 2., (len(clock), n))
    rate = rng.uniform(.001, .01, speed.shape)
    seed, direction = (rng.normal(size=speed.shape) for _ in range(2))
    op = FastTransportHazard(speed, clock, .7, periodic, 3.73)
    op.forward(rate, True)
    _, derivative = op.backward(seed, with_rate=False)
    step = 1e-7
    high = FastTransportHazard(speed + step * direction, clock, .7, periodic, 3.73).forward(rate)
    low = FastTransportHazard(speed - step * direction, clock, .7, periodic, 3.73).forward(rate)
    numeric = np.sum(seed * (high - low)) / (2 * step)
    assert numeric == pytest.approx(np.sum(derivative * direction), abs=3e-7, rel=3e-6)


@pytest.mark.parametrize('periodic', [False, True])
@pytest.mark.parametrize('n', [1, 2, 3, 28, 29])
@pytest.mark.parametrize('radius', [.17, 3.7, 2000.])
def test_shared_source_edges_match_direct_cell_primitives_and_speed_derivative(periodic, n, radius):
    rng = np.random.default_rng(123 + n)
    clock = np.array([0., .13, .8, 1.9, 5., 12.])
    velocity = rng.uniform(.7, 1.7, len(clock))
    speed = np.broadcast_to(velocity[:, None], (len(clock), n))
    rate = np.broadcast_to(rng.uniform(.003, .03, n), speed.shape)
    op = StationarySourceHazard(speed, clock, .8, periodic, radius)
    actual = op.forward(rate, True)
    # Independently assemble the two primitives for each (possibly partial)
    # centered source cell; this also checks even/odd circular diameters.
    op._prepare_primitives()
    direct = np.zeros_like(actual)
    for receiver in range(n):
        for source in range(n):
            for turn in ((-1, 0, 1) if periodic else (0,)):
                d = (source - receiver + turn * n) * op.dx
                a, b = max(0., abs(d) - .5 * op.dx), min(op.cap, abs(d) + .5 * op.dx)
                if b <= a:
                    continue
                left = op._primitive(op.travel - a)[0]
                right = op._primitive(op.travel - b)[0]
                direct[:, receiver] += rate[0, source] * (left - right) * (2. if d == 0. else 1.)
    np.testing.assert_allclose(actual, direct, atol=1e-13, rtol=1e-12)
    seed = rng.normal(size=actual.shape)
    gi, gv = op.backward(seed)
    _, compact = op.backward(seed, with_rate=False, reduce_speed=True)
    assert compact.shape == (len(clock), 1)
    np.testing.assert_array_equal(compact[:, 0], gv.sum(axis=1))
    dv = rng.normal(size=len(clock))
    dr = rng.normal(size=n)
    step = 1e-7
    high = StationarySourceHazard(speed + step * dv[:, None], clock, .8, periodic, radius).forward(rate + step * dr)
    low = StationarySourceHazard(speed - step * dv[:, None], clock, .8, periodic, radius).forward(rate - step * dr)
    numeric = np.sum(seed * (high - low)) / (2 * step)
    assert numeric == pytest.approx(np.sum(gv * dv[:, None]) + np.sum(gi * dr), abs=2e-7, rel=3e-6)
