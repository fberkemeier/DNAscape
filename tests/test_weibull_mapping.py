import unittest

import numpy as np

from dnascape import rmap
from dnascape.fitting import _weibull_survival_fields


class WeibullMappingTests(unittest.TestCase):
    def test_timing_is_survival_integral(self):
        n = 18
        alpha = np.full(n, 2.5e-5)
        time_grid = np.arange(0.0, 160.0, 4.0)

        fields = _weibull_survival_fields(
            alpha,
            fork_speed=1.2,
            shape=2.0,
            resolution=1.0,
            resolution_time=4.0,
            time_grid=time_grid,
        )

        survival_timing = np.trapezoid(
            fields["survival"],
            x=time_grid,
            axis=0,
        )
        replicated_integral = np.trapezoid(
            1.0 - fields["survival"],
            x=time_grid,
            axis=0,
        )

        self.assertTrue(np.allclose(fields["predicted_timing"], survival_timing))
        self.assertFalse(np.allclose(fields["predicted_timing"], replicated_integral))

    def test_spacetime_weibull_fit_recovers_synthetic_timing(self):
        n = 24
        positions = np.arange(n, dtype=float)
        fork_speed = 1.3
        alpha = 2e-5 * (1.0 + 0.35 * np.sin(2.0 * np.pi * positions / n))
        time_grid = np.arange(0.0, 180.0, 3.0)
        synthetic = _weibull_survival_fields(
            alpha,
            fork_speed=fork_speed,
            shape=2.0,
            resolution=1.0,
            resolution_time=3.0,
            time_grid=time_grid,
        )

        fit = rmap(
            source={
                "replication_timing": synthetic["predicted_timing"],
                "fork_speed": fork_speed,
            },
            target="firing_rate",
            assumptions={
                "target_form": "spacetime",
                "target_shape": "weibull",
                "resolution_t": 3.0,
                "time_grid": time_grid,
                "maxiter": 30,
                "verbose": False,
            },
            return_info=True,
        )

        self.assertEqual(fit["values"].shape, (time_grid.size, n))
        self.assertTrue(np.all(fit["alpha"] > 0))
        self.assertLess(float(np.nanmean(np.abs(fit["residual"]))), 0.5)

    def test_alpha_speed_and_delay_monotonicity(self):
        n = 20
        time_grid = np.arange(0.0, 180.0, 3.0)
        base_alpha = np.full(n, 1.8e-5)

        slower_alpha = _weibull_survival_fields(
            base_alpha * 0.6,
            fork_speed=1.2,
            shape=2.0,
            resolution=1.0,
            resolution_time=3.0,
            time_grid=time_grid,
        )["predicted_timing"]
        faster_alpha = _weibull_survival_fields(
            base_alpha * 1.8,
            fork_speed=1.2,
            shape=2.0,
            resolution=1.0,
            resolution_time=3.0,
            time_grid=time_grid,
        )["predicted_timing"]
        slow_forks = _weibull_survival_fields(
            base_alpha,
            fork_speed=0.8,
            shape=2.0,
            resolution=1.0,
            resolution_time=3.0,
            time_grid=time_grid,
        )["predicted_timing"]
        fast_forks = _weibull_survival_fields(
            base_alpha,
            fork_speed=1.6,
            shape=2.0,
            resolution=1.0,
            resolution_time=3.0,
            time_grid=time_grid,
        )["predicted_timing"]
        delayed = _weibull_survival_fields(
            base_alpha,
            fork_speed=1.2,
            shape=2.0,
            resolution=1.0,
            resolution_time=3.0,
            time_grid=time_grid,
            delay=12.0,
        )["predicted_timing"]
        undelayed = _weibull_survival_fields(
            base_alpha,
            fork_speed=1.2,
            shape=2.0,
            resolution=1.0,
            resolution_time=3.0,
            time_grid=time_grid,
        )["predicted_timing"]

        self.assertLess(float(np.nanmean(faster_alpha)), float(np.nanmean(slower_alpha)))
        self.assertGreater(float(np.nanmean(slow_forks)), float(np.nanmean(fast_forks)))
        self.assertGreater(float(np.nanmean(delayed)), float(np.nanmean(undelayed)))


if __name__ == "__main__":
    unittest.main()
