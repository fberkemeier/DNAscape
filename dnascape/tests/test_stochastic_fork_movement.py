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

    monkeypatch.setattr(engine, "_positive_normal_speeds", doubled_speed)
    field = speed_field(form, n=5)
    result = engine.simulate_replication(
        firing_rate=np.array([1000., 0, 0, 0, 0]), fork_speed=field, n_bins=5,
        resolution_t=0.25, rng=np.random.default_rng(921), fork_speed_std=0.2,
    )
    times = result[3]
    expected = np.empty(5)
    expected[0] = result[2][0]
    for position in range(1, 5):
        row = min(int(expected[position - 1] // 0.25), 2)
        mean = field if form == "scalar" else field[position] if form == "x" else field[0, row] if form == "t" else field[position, row]
        expected[position] = expected[position - 1] + 1 / (2 * mean)
    np.testing.assert_allclose(times, expected)
    assert calls


def test_normal_draws_have_requested_centre_and_spread_and_resample_negative_tail():
    speeds = engine._positive_normal_speeds(np.random.default_rng(71), 2., 0.2, size=100_000)
    assert speeds.mean() == pytest.approx(2, abs=0.003)
    assert speeds.std() == pytest.approx(0.2, abs=0.002)
    broad = engine._positive_normal_speeds(np.random.default_rng(71), 0.1, 2., size=10_000)
    assert np.isfinite(broad).all() and np.all(broad > 0)


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
