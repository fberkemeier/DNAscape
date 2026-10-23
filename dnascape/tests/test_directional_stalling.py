"""Shared genomic obstacles versus independently sampled directional stalls."""

import numpy as np
import pytest

from dnascape import rsim
from dnascape import simulation as engine


class FixedStalls:
    """Left-moving/right-moving samples differ unless explicitly shared."""

    def __init__(self):
        self.mask_count = 0
        self.delay_count = 0

    def random(self, size):
        values = np.ones(size)
        values[1 if self.mask_count == 0 else 2] = 0
        self.mask_count += 1
        return values

    def exponential(self, scale, size=None):
        if size is None:
            return np.zeros(np.asarray(scale).shape)
        self.delay_count += 1
        return np.full(size, 2. if self.delay_count == 1 else 7.)


@pytest.mark.parametrize("shared", [False, True])
def test_shared_stalls_use_the_same_positions_and_recovery_delays_in_both_directions(shared):
    results = []
    for origin in (0, 3):
        firing = np.zeros(4)
        firing[origin] = 1.
        results.append(engine.simulate_replication(
            firing_rate=firing, fork_speed=1., rng=FixedStalls(), stall_rate=.5,
            tau=2., stall_shared_directions=shared, max_rep_time=None,
        )[3])
    np.testing.assert_array_equal(results[0], [0, 3, 4, 5])
    np.testing.assert_array_equal(results[1], [5, 4, 1, 0] if shared else [10, 9, 8, 0])


def test_shared_permanent_obstacle_blocks_either_direction_at_the_same_position():
    for origin, expected in ((0, [0, np.inf, np.inf, np.inf]), (3, [np.inf, np.inf, 1, 0])):
        firing = np.zeros(4)
        firing[origin] = 1.
        result = engine.simulate_replication(
            firing_rate=firing, fork_speed=1., rng=FixedStalls(), stall_rate=.5,
            tau=np.inf, stall_shared_directions=True, max_rep_time=None,
        )
        np.testing.assert_array_equal(result[3], expected)


@pytest.mark.parametrize("geometry", ["linear", "periodic"])
def test_shared_stalls_are_reproducible_and_support_structured_and_physical_inputs(geometry):
    source = dict(firing_rate=np.linspace(.02, .1, 20), fork_speed=1.4)
    assumptions = dict(sim_number=4, seed=64, geometry=geometry, tau=2., verbose=False,
                       stall_shared_directions=True, stall_rate_per_kb=.17, resolution_x=2.)
    first = rsim(source=source, assumptions=assumptions)
    repeated = rsim(**source, **assumptions)
    np.testing.assert_array_equal(first["rep_times_per_sim"], repeated["rep_times_per_sim"])
    probability = engine.stall_probability_per_bin(.17, 2.)
    legacy = rsim(**source, **{**assumptions, "stall_rate_per_kb": None, "stall_rate": probability})
    np.testing.assert_array_equal(first["rep_times_per_sim"], legacy["rep_times_per_sim"])


def test_shared_switch_has_no_effect_without_stalling_and_default_keeps_independent_stalls():
    arguments = dict(firing_rate=.03, fork_speed=1.4, n_bins=20, sim_number=4, seed=123, verbose=False)
    baseline = rsim(**arguments)
    shared = rsim(**arguments, stall_shared_directions=True)
    np.testing.assert_array_equal(baseline["rep_times_per_sim"], shared["rep_times_per_sim"])
    defaults = rsim(**arguments, stall_rate_per_kb=.2, tau=2.)
    explicit = rsim(**arguments, stall_rate_per_kb=.2, tau=2., stall_shared_directions=False)
    np.testing.assert_array_equal(defaults["rep_times_per_sim"], explicit["rep_times_per_sim"])


@pytest.mark.parametrize("invalid", ["shared", None, 1])
def test_invalid_directional_stalling_mode_has_a_clear_error(invalid):
    with pytest.raises(ValueError, match="stall_shared_directions"):
        rsim(firing_rate=.1, n_bins=4, verbose=False, stall_shared_directions=invalid)
