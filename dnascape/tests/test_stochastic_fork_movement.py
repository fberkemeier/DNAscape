"""Stochastic fork movement respects local speeds, seeds, and all field dependences."""

import numpy as np
import pytest

from dnascape import rsim
from dnascape import simulation as engine


def speed_field(form, n=12):
    if form == "scalar":
        return 1.4
    if form == "x":
        return np.linspace(1, 2, n)
    if form == "t":
        return np.array([[1., 1.5, 2.]])
    return np.linspace(1, 2, n)[:, None] * np.array([[1., 1.5, 2.]])


def common(form, **changes):
    return dict(
        firing_rate=np.linspace(0.01, 0.08, 12), fork_speed=speed_field(form),
        sim_number=4, resolution_t=0.5, n_bins=12, seed=719, verbose=False, **changes,
    )


@pytest.mark.parametrize("form", ["scalar", "x", "t", "xt"])
def test_zero_deviation_preserves_seeded_results_exactly(form):
    deterministic = rsim(**common(form))
    zero = rsim(**common(form), fork_speed_std=0)
    for key in ("replication_timing", "replication_timing_std", "rep_times_per_sim", "efficiency",
                "fork_directionality", "s_phase_duration", "completion_time", "inter_origin_distances"):
        np.testing.assert_array_equal(deterministic[key], zero[key])


@pytest.mark.parametrize("form", ["scalar", "x", "t", "xt"])
@pytest.mark.parametrize("geometry", ["linear", "periodic"])
def test_stochastic_motion_is_reproducible_and_changes_each_speed_dependence(form, geometry):
    kwargs = common(form, geometry=geometry, time_statsQ=True, time_stats_xtQ=True,
                    time_grid=np.arange(0, 61, 1.))
    first = rsim(**kwargs, fork_speed_std=0.3)
    second = rsim(**kwargs, fork_speed_std=0.3)
    deterministic = rsim(**kwargs)
    np.testing.assert_array_equal(first["rep_times_per_sim"], second["rep_times_per_sim"])
    assert not np.array_equal(first["rep_times_per_sim"], deterministic["rep_times_per_sim"])
    assert np.isfinite(first["replication_timing"]).all()
    assert np.all(first["replication_timing_std"] >= 0)
    assert np.all(np.diff(first["replicated_fraction"]) >= -1e-12)
    assert first["replicated_fraction_xt"].shape == (12, 61)


@pytest.mark.parametrize("form", ["scalar", "x", "t", "xt"])
def test_local_mean_uses_the_current_position_and_time(monkeypatch, form):
    calls = []

    def doubled_speed(_rng, mean, standard_deviation, *, size):
        assert standard_deviation == 0.2
        calls.append(np.array(mean, copy=True))
        return np.broadcast_to(np.asarray(mean) * 2, (size,)).copy()

    monkeypatch.setattr(engine, "_positive_speed_samples", doubled_speed)
    field = speed_field(form, n=5)
    result = engine.simulate_replication(
        firing_rate=np.array([1000., 0, 0, 0, 0]), fork_speed=field, n_bins=5,
        resolution_t=0.25, rng=np.random.default_rng(921), fork_speed_std=0.2,
    )
    times = result[3]
    expected = np.empty(5)
    expected[0] = result[2][0]
    for position in range(1, 5):
        if form in {"scalar", "x"}:
            mean = field if form == "scalar" else field[position]
            expected[position] = expected[position - 1] + 1 / (2 * mean)
        else:
            # Independently invert the cumulative distance curve, including
            # the held final sample, rather than freezing speed at entry.
            speeds = 2 * (field[0] if form == "t" else field[position])
            clock = np.arange(4) * .25
            distance = np.r_[0., np.cumsum(speeds * .25)]
            start = expected[position - 1]
            travelled = (np.interp(start, clock, distance) if start <= clock[-1]
                         else distance[-1] + (start - clock[-1]) * speeds[-1])
            target = travelled + 1
            expected[position] = (np.interp(target, distance, clock) if target <= distance[-1]
                                  else clock[-1] + (target - distance[-1]) / speeds[-1])
    np.testing.assert_allclose(times, expected)
    assert calls


def test_positive_speed_draws_preserve_requested_moments_and_finite_mean_crossing_time():
    speeds = engine._positive_speed_samples(np.random.default_rng(71), 2., 0.2, size=100_000)
    assert speeds.mean() == pytest.approx(2, abs=0.003)
    assert speeds.std() == pytest.approx(0.2, abs=0.002)
    broad = engine._positive_speed_samples(np.random.default_rng(73), 1.4, 1., size=1_000_000)
    assert np.isfinite(broad).all() and np.all(broad > 0)
    assert broad.mean() == pytest.approx(1.4, abs=.005)
    assert broad.std() == pytest.approx(1., abs=.01)
    assert np.mean(1 / broad) == pytest.approx((1 + (1 / 1.4) ** 2) / 1.4, rel=.01)


def test_zero_deviation_does_not_draw_and_extreme_variation_has_a_clear_error():
    np.testing.assert_array_equal(engine._positive_speed_samples(None, [1., 2.], 0., size=2), [1., 2.])
    with pytest.raises(ValueError, match="numeric range"):
        engine._positive_speed_samples(np.random.default_rng(1), 1e-300, 1e300, size=2)


@pytest.mark.parametrize("form", ["t", "xt"])
def test_temporal_stochastic_crossings_use_later_speeds_and_preserve_arrival_order(monkeypatch, form):
    monkeypatch.setattr(engine, "_positive_speed_samples", lambda rng, mean, sd, *, size: np.full(size, mean))

    class OriginAt:
        def __init__(self, time):
            self.time = time

        def random(self, size):
            return np.ones(size)

        def exponential(self, scale, size=None):
            return np.full(np.asarray(scale).shape, self.time)

    speeds = np.array([[.01, 10.]])
    if form == "xt":
        speeds = np.repeat(speeds, 2, axis=0)
    arguments = dict(firing_rate=np.array([1., 0.]), fork_speed=speeds,
                     resolution_t=1., fork_speed_std=.3, max_rep_time=None)
    early = engine.simulate_replication(**arguments, rng=OriginAt(0.))[3][1]
    late = engine.simulate_replication(**arguments, rng=OriginAt(1.))[3][1]
    assert early == pytest.approx(1.099)
    assert late == pytest.approx(1.1)
    assert early <= late  # The entry-speed approximation produced 100 versus 1.1.


@pytest.mark.parametrize("form", ["scalar", "x", "t", "xt"])
def test_sd_one_does_not_create_the_seeded_neighbouring_bin_spike(form):
    n = 100
    speed = {"scalar": 1.4, "x": np.linspace(1., 1.8, n),
             "t": np.array([[1., 1.4, 1.8]]),
             "xt": np.linspace(1., 1.8, n)[:, None] * np.array([[1., 1.2, .8]])}[form]
    result = rsim(firing_rate=.01, fork_speed=speed, fork_speed_std=1., sim_number=10,
                  n_bins=n, resolution_t=5., seed=6, verbose=False)
    assert np.max(np.abs(np.diff(result["replication_timing"]))) < 5


@pytest.mark.parametrize("invalid", [-0.1, np.nan, np.inf, "invalid"])
def test_invalid_speed_deviation_is_rejected(invalid):
    with pytest.raises(ValueError, match="fork_speed_std"):
        rsim(**common("scalar"), fork_speed_std=invalid)


def test_stochastic_movement_supports_stalls_and_fork_coupled_finite_factors():
    kwargs = common("xt", fork_speed_std=0.2, ffiring_nr=8, ffiring_forkQ=True,
                    stall_rate_per_kb=0.1, tau=2.)
    result = rsim(**kwargs)
    assert np.isfinite(result["replication_timing"]).all()
    assert np.all(result["replication_timing"] >= 0)
    structured = rsim(source={"firing_rate": kwargs.pop("firing_rate"), "fork_speed": kwargs.pop("fork_speed")},
                      assumptions=kwargs)
    np.testing.assert_array_equal(result["rep_times_per_sim"], structured["rep_times_per_sim"])
