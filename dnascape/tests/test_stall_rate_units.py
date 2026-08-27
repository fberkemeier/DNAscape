import math
import unittest
from unittest.mock import patch

import numpy as np

from dnascape import fitting, rmap, rsim, stall_probability_per_bin


class StallRateUnitTests(unittest.TestCase):
    @staticmethod
    def _simulation_inputs():
        return {
            "firing_rate": np.linspace(0.004, 0.012, 18),
            "fork_speed": 1.4,
            "sim_number": 3,
            "resolution_space": 5.0,
            "tau": 2.5,
            "seed": 9182,
            "verbose": False,
        }

    @staticmethod
    def _assert_same_seeded_summary(expected, actual):
        for key in (
            "replication_timing",
            "replication_timing_std",
            "rep_times_per_sim",
            "s_phase_duration",
            "inter_origin_distances",
            "efficiency",
            "fork_directionality",
        ):
            np.testing.assert_array_equal(expected[key], actual[key])
        assert expected["num_oris"] == actual["num_oris"]

    def test_density_conversion_uses_poisson_encounter_probability(self):
        rate_per_kb = 0.17
        resolution_kb = 5.0

        probability = stall_probability_per_bin(rate_per_kb, resolution_kb)

        self.assertAlmostEqual(
            probability,
            1.0 - math.exp(-(rate_per_kb * resolution_kb)),
            places=15,
        )
        self.assertAlmostEqual(probability, 0.5725850680512734, places=15)
        self.assertEqual(stall_probability_per_bin(0.0, resolution_kb), 0.0)

    def test_density_conversion_rejects_invalid_rate_or_resolution(self):
        for invalid_rate in (-0.01, np.nan, np.inf, -np.inf):
            with self.subTest(rate=invalid_rate):
                with self.assertRaisesRegex(ValueError, "stall_rate_per_kb"):
                    stall_probability_per_bin(invalid_rate, 5.0)

        for invalid_resolution in (0.0, -1.0, np.nan, np.inf, -np.inf):
            with self.subTest(resolution=invalid_resolution):
                with self.assertRaisesRegex(ValueError, "resolution_x"):
                    stall_probability_per_bin(0.17, invalid_resolution)

    def test_seeded_density_and_equivalent_legacy_probability_are_identical(self):
        rate_per_kb = 0.17
        resolution_kb = 5.0
        probability = stall_probability_per_bin(rate_per_kb, resolution_kb)
        common = self._simulation_inputs()

        legacy = rsim(**common, stall_rate=probability)
        physical = rsim(**common, stall_rate_per_kb=rate_per_kb)

        self._assert_same_seeded_summary(legacy, physical)

    def test_nonzero_legacy_probability_conflicts_with_density(self):
        common = self._simulation_inputs()

        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            rsim(
                **common,
                stall_rate=0.1,
                stall_rate_per_kb=0.17,
            )

    def test_structured_density_aliases_match_the_canonical_name(self):
        source = {
            "firing_rate": np.linspace(0.004, 0.012, 18),
            "fork_speed": 1.4,
        }
        common_assumptions = {
            "sim_number": 3,
            "resolution": 5.0,
            "tau": 2.5,
            "seed": 9182,
            "verbose": False,
        }
        canonical = rsim(
            source=source,
            assumptions={
                **common_assumptions,
                "stall_rate_per_kb": 0.17,
            },
        )

        for alias in ("stall_density", "stall_density_per_kb", "stalls_per_kb"):
            with self.subTest(alias=alias):
                aliased = rsim(
                    source=source,
                    assumptions={
                        **common_assumptions,
                        alias: 0.17,
                    },
                )
                self._assert_same_seeded_summary(canonical, aliased)

    def test_positive_density_selects_and_matches_rmap_simulation_route(self):
        firing_rate = np.linspace(0.004, 0.012, 18)
        assumptions = {
            "resolution": 5.0,
            "stall_rate_per_kb": 0.17,
            "tau": 2.5,
            "sim_number": 2,
            "seed": 651,
            "verbose": False,
        }

        mapped = rmap(
            source={"firing_rate": firing_rate, "fork_speed": 1.4},
            target="replication_timing",
            assumptions=assumptions,
            return_info=True,
        )
        direct = rsim(
            firing_rate=firing_rate,
            fork_speed=1.4,
            **assumptions,
        )["replication_timing"]

        self.assertEqual(mapped["method"], "firing_and_speed_to_timing_simulation")
        self.assertEqual(mapped["assumptions"]["stall_rate_per_kb"], 0.17)
        np.testing.assert_array_equal(mapped["values"], direct)

    def test_weibull_firing_targets_reject_positive_density(self):
        timing = np.linspace(15.0, 35.0, 12)
        for target_form in ("time", "spacetime"):
            with self.subTest(target_form=target_form):
                with self.assertRaisesRegex(
                    ValueError,
                    "Stalling is not supported.*target_shape='weibull'",
                ):
                    rmap(
                        source={
                            "replication_timing": timing,
                            "fork_speed": 1.4,
                        },
                        target="firing_rate",
                        assumptions={
                            "target_form": target_form,
                            "target_shape": "weibull",
                            "resolution": 5.0,
                            "stall_rate_per_kb": 0.17,
                            "verbose": False,
                        },
                    )

    def test_free_firing_refinements_forward_density_to_rsim(self):
        observed = np.linspace(10.0, 25.0, 8)
        captured = []

        def fake_rsim(**kwargs):
            captured.append(kwargs)
            return {"replication_timing": observed.copy()}

        with patch.object(fitting, "rsim", side_effect=fake_rsim):
            fitting._refine_temporal_firing(
                np.full((1, 5), 0.01),
                observed,
                fork_speed=1.4,
                resolution=5.0,
                resolution_time=5.0,
                temporal_maxiter=1,
                temporal_sim_number=2,
                stall_rate_per_kb=0.17,
                tau=2.5,
            )
            fitting._refine_spacetime_firing(
                np.full((5, observed.size), 0.01),
                observed,
                fork_speed=1.4,
                resolution=5.0,
                resolution_time=5.0,
                refine_maxiter=1,
                spacetime_sim_number=2,
                stall_rate_per_kb=0.17,
                tau=2.5,
            )

        self.assertEqual(len(captured), 2)
        for call in captured:
            self.assertEqual(call["stall_rate"], 0.0)
            self.assertEqual(call["stall_rate_per_kb"], 0.17)
            self.assertEqual(call["resolution_space"], 5.0)
            self.assertEqual(call["tau"], 2.5)


if __name__ == "__main__":
    unittest.main()
