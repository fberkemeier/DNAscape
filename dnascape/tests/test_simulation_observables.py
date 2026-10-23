"""Actual initiations, completed replicons, and total directional fork density."""

import gc

import numpy as np
import pytest

import dnascape.simulation as simulation


def ensemble(**overrides):
    arguments = dict(
        firing_rate=.08, fork_speed=2., n_bins=32, sim_number=5,
        resolution_x=1., resolution_t=1., seed=12, verbose=False,
        time_statsQ=True, time_stats_xtQ=True, max_rep_time=128.,
        time_grid=np.array([0., .25, .7, 1.2, 2., 4., 8., 16., 32., 64., 128.]),
    )
    arguments.update(overrides)
    return simulation.rsim(**arguments)


@pytest.mark.parametrize("periodic", [False, True])
def test_origin_firing_counts_actual_events_and_replicon_lengths_cover_the_genome(monkeypatch, tmp_path, periodic):
    realizations = []
    original = simulation.simulate_replication

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        realizations.append((result[0].copy(), result[2].copy(), result[8].copy()))
        return result

    monkeypatch.setattr(simulation, "simulate_replication", capture)
    result = ensemble(perQ=periodic, dens_memmap_dir=tmp_path)
    expected_firing = np.zeros(len(result["time_grid"]))
    expected_lengths = []
    for indices, times, owners in realizations:
        expected_firing += np.bincount(np.searchsorted(result["time_grid"], times), minlength=len(expected_firing))
        expected_lengths.extend(np.count_nonzero(owners == origin) for origin in indices)
    expected_firing /= len(realizations)
    np.testing.assert_allclose(result["origin_firing"], expected_firing)
    np.testing.assert_allclose(result["initiation_events_xt"].sum(axis=0), expected_firing, atol=1e-6)
    assert result["origin_firing"].sum() == pytest.approx(np.mean(result["num_origins"]))
    assert result["origin_firing"].sum() < 32  # Passive proposals are not initiation events.
    np.testing.assert_equal(result["replicon_lengths"], expected_lengths)
    assert result["replicon_length_count"] == sum(result["num_origins"])
    assert result["replicon_lengths"].sum() == 32 * 5
    np.testing.assert_equal(result["fork_density_xt"],
                            result["right_fork_density_xt"] + result["left_fork_density_xt"])
    assert result["fork_density_xt"].dtype == np.float32
    assert len(list(tmp_path.glob("*.dat"))) == 6
    del result
    gc.collect()
    assert not list(tmp_path.glob("*.dat"))


def test_replicon_lengths_join_periodic_domains_and_exclude_censored_or_unreachable_domains():
    owners = np.array([0, 0, 0, 3, 3, 0])
    arrivals = np.array([0., 1., 2., 0., 1., 1.])
    origins = np.array([0, 3])
    np.testing.assert_equal(simulation._replicon_lengths(owners, arrivals, origins, 2., None), [8, 4])
    np.testing.assert_equal(simulation._replicon_lengths(owners, arrivals, origins, 2., 1.5), [4])
    arrivals[2] = np.inf
    np.testing.assert_equal(simulation._replicon_lengths(owners, arrivals, origins, 2., None), [4])
    assert not simulation._replicon_lengths(owners, arrivals, [], 2., None).size
    np.testing.assert_equal(simulation._replicon_lengths(np.zeros(6, dtype=int), np.arange(6.), [0], .5, 10), [3])


def test_replicon_retention_is_bounded_without_changing_ensemble_statistics():
    full = ensemble(time_stats_xtQ=False)
    limited = ensemble(time_stats_xtQ=False, retain_replicon_lengths_limit=3)
    assert limited["replicon_length_count"] > 3
    assert limited["replicon_length_count"] == full["replicon_length_count"]
    assert limited["replicon_lengths_truncated"] is True
    np.testing.assert_equal(limited["replicon_lengths"], full["replicon_lengths"][:3])
    np.testing.assert_equal(limited["replication_timing"], full["replication_timing"])
    np.testing.assert_equal(limited["origin_firing"], full["origin_firing"])
    zero = ensemble(time_stats_xtQ=False, retain_replicon_lengths_limit=0)
    assert zero["replicon_length_count"] == full["replicon_length_count"]
    assert not zero["replicon_lengths"].size


@pytest.mark.parametrize("detailed", [False, True])
def test_time_tracking_gates_origin_firing_and_replicon_pool_independently_of_xt(detailed):
    result = ensemble(time_statsQ=False, time_stats_xtQ=detailed)
    assert result["origin_firing"] is None
    assert result["replicon_lengths"] is None
    assert result["replicon_length_count"] == 0
    assert result["replicon_lengths_truncated"] is False
    assert (result["fork_density_xt"] is not None) is detailed


def test_origin_firing_respects_the_stopping_time_and_nonuniform_time_bins(monkeypatch):
    events = []
    original = simulation.simulate_replication

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        events.extend(result[2][result[2] <= 1.5])
        return result

    monkeypatch.setattr(simulation, "simulate_replication", capture)
    result = ensemble(max_rep_time=1.5)
    expected = np.bincount(np.searchsorted(result["time_grid"], events), minlength=len(result["time_grid"])) / 5
    np.testing.assert_allclose(result["origin_firing"], expected)
    np.testing.assert_allclose(result["initiation_events_xt"].sum(axis=0), expected, atol=1e-6)
    assert np.all(result["origin_firing"][5:] == 0)
