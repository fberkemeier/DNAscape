"""Fork counts include unfinished pauses and terminate at recovery/rescue."""
import asyncio

import numpy as np
import pytest

from dnascape.simulation import rsim, simulate_replication
from dnascape.workflows.simulation import ConstantSimulationRequest, run_constant_simulation


class PrescribedDraws:
    """Select explicit events in the existing sampling/propagation model."""

    def __init__(self, forward, backward=None, firing=(0.,), pause=5.):
        self.masks = iter((np.asarray(forward), np.asarray(backward if backward is not None else forward)))
        self.firing = np.asarray(firing)
        self.pause = pause

    def random(self, size):
        mask = next(self.masks)
        assert mask.shape == (size,)
        return mask

    def exponential(self, scale, size=None):
        if size is None:
            assert np.asarray(scale).size == self.firing.size
            return self.firing.copy()
        return np.full(size, self.pause)


def run_path(n, origins, forward, *, backward=None, recovery=0., times=None,
             periodic=False, shared=False, cutoff=100., pause=5.):
    initiation = np.zeros(n)
    initiation[origins] = 1.
    return simulate_replication(
        initiation_rate=initiation, fork_speed=1., stall_rate_per_kb=1.,
        stall_recovery_rate=recovery, stall_shared_directions=shared,
        rng=PrescribedDraws(forward, backward, firing=np.zeros(len(origins)), pause=pause),
        perQ=periodic, max_rep_time=cutoff, time_statsQ=True,
        time_grid=np.asarray(times if times is not None else [0., .5, 1., 2., 3., 10.]),
    )


def test_permanent_stall_is_counted_through_the_cutoff_and_last_recorded_time():
    result = run_path(3, [0], [.9, 0., .9], backward=[.9] * 3,
                      times=[0., .5, 1., 2., 10., 11.], cutoff=10.)
    stats = result[5]
    np.testing.assert_array_equal(stats["stalled_forks"], [0, 0, 1, 1, 1, 0])
    np.testing.assert_array_equal(stats["total_forks"], [1, 1, 1, 1, 1, 0])
    np.testing.assert_array_equal(stats["active_forks"], [1, 1, 0, 0, 0, 0])
    assert np.isinf(result[3][1:]).all()


@pytest.mark.parametrize("cutoff", [2., 100.])
def test_finite_pause_continuing_past_the_recording_window_is_counted(cutoff):
    result = run_path(3, [0], [.9, 0., .9], backward=[.9] * 3,
                      recovery=.2, times=[0., .5, 1., 2.], cutoff=cutoff)
    np.testing.assert_array_equal(result[5]["stalled_forks"], [0, 0, 1, 1])
    assert result[5]["total_forks"][-1] == 1


def test_fork_recovers_without_double_counting_at_bin_boundaries():
    result = run_path(3, [0], [.9, 0., .9], backward=[.9] * 3, recovery=.2,
                      times=[0., .5, 1., 5., 6., 6.5, 7., 8.])
    np.testing.assert_array_equal(result[3], [0., 6., 7.])
    np.testing.assert_array_equal(result[5]["stalled_forks"], [0, 0, 1, 1, 0, 0, 0, 0])
    np.testing.assert_array_equal(result[5]["total_forks"], [1, 1, 1, 1, 1, 1, 0, 0])
    np.testing.assert_array_equal(result[5]["active_forks"], [1, 1, 0, 0, 1, 1, 0, 0])


def test_opposing_fork_rescues_a_permanent_pause():
    result = run_path(5, [0, 4], [.9, 0., .9, .9, .9], backward=[.9] * 5,
                      times=[0., .5, 1., 2.5, 3., 4.])
    np.testing.assert_array_equal(result[3], [0., 3., 2., 1., 0.])
    np.testing.assert_array_equal(result[5]["stalled_forks"], [0, 0, 1, 1, 0, 0])
    np.testing.assert_array_equal(result[5]["total_forks"], [2, 2, 2, 2, 0, 0])


@pytest.mark.parametrize("shared", [False, True])
def test_periodic_pause_ends_on_rescue_or_blocks_both_directions(shared):
    result = run_path(4, [0], [.9, 0., .9, .9], backward=[.9] * 4,
                      periodic=True, shared=shared, times=[0., 1., 2., 3., 4.])
    np.testing.assert_array_equal(result[5]["stalled_forks"],
                                  [0, 1, 1, 2, 2] if shared else [0, 1, 1, 0, 0])
    assert np.isinf(result[3][1]) if shared else result[3][1] == 3.


@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("shared", [False, True])
def test_two_directional_permanent_stalls_are_counted_independently(periodic, shared):
    forward = [.9, 0. if shared else .9, .9, 0., .9]
    backward = [.9, 0., .9, .9, .9]
    result = run_path(5, [2], forward, backward=backward, shared=shared,
                      periodic=periodic, times=[0., .5, 1., 10.])
    np.testing.assert_array_equal(result[5]["stalled_forks"], [0, 0, 2, 2])
    np.testing.assert_array_equal(result[5]["total_forks"], [2, 2, 2, 2])


@pytest.mark.parametrize("form", ["scalar", "x", "t", "xt"])
@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("rate", [0., .1])
def test_tracking_preserves_stochastic_speed_samples_and_rng_state(form, periodic, shared, rate):
    n = 31
    speeds = {"scalar": 1.4, "x": np.linspace(1., 2., n),
              "t": np.array([[1., 1.4, 2., .8]]),
              "xt": np.linspace(1., 2., n)[:, None] * np.array([1., .8, 1.4, 1.])}
    arguments = dict(initiation_rate=np.linspace(.01, .04, n), fork_speed=speeds[form],
                     resolution_t=5., perQ=periodic, fork_speed_std=.3,
                     stall_rate_per_kb=.1, stall_recovery_rate=rate,
                     stall_shared_directions=shared, max_rep_time=80.,
                     time_grid=np.arange(0., 41., 2.))
    first_rng, second_rng = np.random.default_rng(71), np.random.default_rng(71)
    plain = simulate_replication(**arguments, rng=first_rng)
    tracked = simulate_replication(**arguments, rng=second_rng, time_statsQ=True)
    for index in (0, 2, 3, 6):
        np.testing.assert_array_equal(plain[index], tracked[index])
    assert first_rng.bit_generator.state == second_rng.bit_generator.state


@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("stalls,variation", [(False, False), (True, False), (False, True), (True, True)])
def test_time_counts_agree_with_directional_density_fields(periodic, stalls, variation):
    n = 41
    result = rsim(initiation_rate=np.full(n, .03), fork_speed=1.4, sim_number=5,
                  stall_rate_per_kb=np.linspace(0., .3, n) if stalls else 0.,
                  stall_recovery_rate=np.tile([0., .1, .3], 14)[:n],
                  stall_shared_directions=True, fork_speed_std=.3 if variation else 0.,
                  geometry="periodic" if periodic else "linear", seed=19,
                  time_grid=np.arange(0., 51., .5), max_rep_time=40.,
                  time_statsQ=True, time_stats_xtQ=True, verbose=False)
    np.testing.assert_allclose(result["fork_density_xt"].sum(axis=0), result["total_forks"], atol=2e-6)
    np.testing.assert_allclose(result["active_forks"] + result["stalled_forks"], result["total_forks"])
    assert np.all(result["active_forks"] >= 0)
    assert np.all(result["stalled_forks"] >= 0)
    if not stalls:
        assert not np.any(result["stalled_forks"])


def test_workflow_preserves_permanent_stall_counts():
    result = run_constant_simulation(ConstantSimulationRequest(
        initiation_rate=np.array([0., 0., 1e9, 0., 0.]), fork_speed=1.,
        resolution_x=1., resolution_t=1., region_start_kb=0., region_end_kb=5.,
        replicates=3, geometry="linear", time_window=10., max_replication_time=10.,
        stall_rate_per_kb=1e6, stall_recovery_rate=0., seed=3,
        track_time_outputs=True, track_detailed_outputs=True,
    ))
    assert result.outputs["stalled_forks"][-1] == 2.
    assert result.outputs["active_forks"][-1] == 0.
    assert result.outputs["total_forks"][-1] == 2.
    assert result.incomplete_simulations == 3


def test_live_stalled_forks_preview_receives_nonzero_counts():
    from nicegui.elements.upload import Upload
    from nicegui.testing import user_simulation
    from app.main import index

    async def scenario():
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-simulation").click()
            await user.should_see(marker="action-simulate", retries=60)

            def control(marker):
                return next(item for item in user.client.elements.values()
                            if marker in item._markers and not item.is_deleted)

            async def value(marker, selected):
                with user.client:
                    control(marker).set_value(selected)
                for _ in range(4):
                    await asyncio.sleep(0)

            await value("source-initiation-rate-mode", "profile")
            await control("source-initiation-rate-upload").handle_uploads([
                Upload.SmallFileUpload("initiation.tsv", "text/tab-separated-values",
                                       b"initiation_rate\n0\n0\n1000000000\n0\n0\n")])
            await user.should_see(marker="source-initiation-rate-detected-data", retries=60)
            user.find("Simulation settings").click()
            await value("simulation-region-end", 5.)
            await value("simulation-replicates", 3)
            await value("simulation-track-time-outputs", True)
            await value("simulation-fork-stalling", True)
            await value("simulation-stall-rate-per-kb", 1e6)
            user.find(marker="action-simulate").click()
            await user.should_see(marker="simulation-result-plot", retries=120)
            await value("simulation-output", "Stalled forks (time)")
            plotted = control("simulation-result-plot").options
            assert plotted["yAxis"]["max"] >= 2.
            points = plotted["series"][0]["data"]
            assert points[-1] == [300., 2.]
            assert not user.notify.messages, user.notify.messages
    asyncio.run(scenario())
