import unittest

import numpy as np

from dnascape import rmap


class ForkSpeedShapeTests(unittest.TestCase):
    def test_spatial_fork_speed_exponential_shape(self):
        n = 12
        timing = np.linspace(20.0, 45.0, n)
        firing_rate = np.linspace(1e-5, 3e-5, n)

        fit = rmap(
            source={
                "replication_timing": timing,
                "firing_rate": firing_rate,
            },
            target="fork_speed",
            assumptions={
                "target_form": "space",
                "target_shape": "exponential",
                "maxiter": 0,
                "verbose": False,
            },
            return_info=True,
        )

        self.assertEqual(fit["values"].shape, (n,))
        self.assertEqual(fit["assumptions"]["target_shape"], "exponential")
        self.assertEqual(fit["target_shape"], "exponential")
        self.assertGreater(float(fit["fork_speed_v0"]), 0.0)
        self.assertTrue(np.isfinite(fit["fork_speed_amplitude_x"]))
        self.assertTrue(np.all(fit["values"] > 0))

    def test_time_fork_speed_exponential_shape(self):
        n = 12
        time_grid = np.arange(0.0, 80.0, 5.0)
        timing = np.linspace(20.0, 45.0, n)
        firing_rate = np.linspace(1e-5, 3e-5, n)

        fit = rmap(
            source={
                "replication_timing": timing,
                "firing_rate": firing_rate,
            },
            target="fork_speed",
            assumptions={
                "target_form": "time",
                "target_shape": "exponential",
                "time_grid": time_grid,
                "resolution_t": 5.0,
                "maxiter": 0,
                "verbose": False,
            },
            return_info=True,
        )

        self.assertEqual(fit["values"].shape, (1, time_grid.size))
        self.assertEqual(fit["assumptions"]["target_shape"], "exponential")
        self.assertEqual(fit["target_shape"], "exponential")
        self.assertGreater(float(fit["fork_speed_v0"]), 0.0)
        self.assertTrue(np.all(fit["values"] > 0))

    def test_spacetime_fork_speed_exponential_shape(self):
        n = 10
        time_grid = np.arange(0.0, 60.0, 5.0)
        timing = np.linspace(18.0, 40.0, n)
        firing_rate = np.tile(np.linspace(1e-5, 2e-5, n)[:, None], (1, time_grid.size))

        fit = rmap(
            source={
                "replication_timing": timing,
                "firing_rate": firing_rate,
            },
            target="fork_speed",
            assumptions={
                "target_form": "spacetime",
                "target_shape": "exponential",
                "time_grid": time_grid,
                "resolution_t": 5.0,
                "maxiter": 0,
                "verbose": False,
            },
            return_info=True,
        )

        self.assertEqual(fit["values"].shape, (n, time_grid.size))
        self.assertEqual(fit["assumptions"]["target_shape"], "exponential")
        self.assertEqual(fit["target_shape"], "exponential")
        self.assertTrue(np.isfinite(fit["fork_speed_amplitude_x"]))
        self.assertTrue(np.isfinite(fit["fork_speed_amplitude_t"]))
        self.assertTrue(np.all(fit["values"] > 0))

    def test_fork_speed_shape_defaults_to_free(self):
        n = 8
        time_grid = np.arange(0.0, 40.0, 5.0)
        timing = np.linspace(15.0, 30.0, n)
        firing_rate = np.linspace(1e-5, 2e-5, n)

        fit = rmap(
            source={
                "replication_timing": timing,
                "firing_rate": firing_rate,
            },
            target="fork_speed",
            assumptions={
                "target_form": "time",
                "time_grid": time_grid,
                "resolution_t": 5.0,
                "maxiter": 0,
                "verbose": False,
            },
            return_info=True,
        )

        self.assertEqual(fit["assumptions"]["target_shape"], "free")
        self.assertEqual(fit["values"].shape, (1, time_grid.size))


if __name__ == "__main__":
    unittest.main()
