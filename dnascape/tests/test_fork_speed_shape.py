import unittest
from unittest.mock import patch

import numpy as np

from dnascape import rmap
from dnascape import fitting


class ForkSpeedShapeTests(unittest.TestCase):
    def test_constant_fork_speed_target_shape_defaults_to_best(self):
        n = 10
        timing = np.linspace(18.0, 40.0, n)
        firing_rate = np.linspace(2e-4, 5e-4, n)

        fit = rmap(
            source={
                "replication_timing": timing,
                "firing_rate": firing_rate,
            },
            target="fork_speed",
            assumptions={
                "target_form": "constant",
                "maxiter": 0,
                "verbose": False,
            },
            return_info=True,
        )

        self.assertEqual(fit["assumptions"]["target_shape"], "best")
        self.assertGreater(float(fit["values"]), 0.0)

    def test_fork_speed_target_shape_defaults_to_best(self):
        n = 10
        timing = np.linspace(18.0, 40.0, n)
        firing_rate = np.linspace(2e-4, 5e-4, n)

        fit = rmap(
            source={
                "replication_timing": timing,
                "firing_rate": firing_rate,
            },
            target="fork_speed",
            assumptions={
                "target_form": "space",
                "maxiter": 0,
                "verbose": False,
            },
            return_info=True,
        )

        self.assertEqual(fit["assumptions"]["target_shape"], "best")
        self.assertEqual(fit["target_shape"], "best")
        self.assertEqual(fit["values"].shape, (n,))
        self.assertEqual(fit["fork_speed_shape"], "site-specific v(x)")
        self.assertTrue(np.all(fit["values"] > 0))

    def test_removed_fork_speed_target_shapes_are_rejected(self):
        for target_shape in ("free", "exponential", "weibull"):
            with self.subTest(target_shape=target_shape):
                with self.assertRaises(ValueError):
                    fitting._canonical_speed_target_shape(target_shape)

    def test_time_fork_speed_best_shape_uses_temporal_component(self):
        n = 12
        time_grid = np.arange(0.0, 80.0, 5.0)
        timing = np.linspace(20.0, 45.0, n)
        firing_rate = np.linspace(2e-4, 5e-4, n)

        with patch("dnascape.fitting.map_timing_firing_spatial_speed") as spatial_mock:
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
                    "maxiter": 1,
                    "verbose": False,
                },
                return_info=True,
            )

        self.assertFalse(spatial_mock.called)
        self.assertEqual(fit["values"].shape, (1, time_grid.size))
        self.assertEqual(fit["assumptions"]["target_shape"], "best")
        self.assertEqual(fit["target_shape"], "best")
        self.assertEqual(fit["fork_speed_component_t"], "v0 * exp(b_t * T)")
        self.assertTrue(np.all(fit["values"] > 0))

    def test_spacetime_fork_speed_best_shape_is_hybrid(self):
        n = 10
        time_grid = np.arange(0.0, 60.0, 5.0)
        timing = np.linspace(18.0, 40.0, n)
        firing_rate = np.tile(np.linspace(2e-4, 5e-4, n)[:, None], (1, time_grid.size))

        with patch("dnascape.fitting._refine_spacetime_speed") as refine_mock:
            fit = rmap(
                source={
                    "replication_timing": timing,
                    "firing_rate": firing_rate,
                },
                target="fork_speed",
                assumptions={
                    "target_form": "spacetime",
                    "time_grid": time_grid,
                    "resolution_t": 5.0,
                    "maxiter": 0,
                    "verbose": False,
                },
                return_info=True,
            )

        self.assertFalse(refine_mock.called)
        self.assertEqual(fit["values"].shape, (n, time_grid.size))
        self.assertEqual(fit["assumptions"]["target_shape"], "best")
        self.assertEqual(fit["target_shape"], "best")
        self.assertEqual(fit["fork_speed_component_x"], "site-specific v_x(x)")
        self.assertEqual(fit["fork_speed_component_t"], "smooth multiplicative q(t)")
        self.assertTrue(np.isfinite(fit["fit_mse"]))
        self.assertTrue(np.all(fit["values"] > 0))

    def test_spacetime_firing_to_fork_speed_uses_deterministic_routes(self):
        n = 8
        time_grid = np.arange(0.0, 35.0, 5.0)
        timing = np.linspace(18.0, 38.0, n)
        firing_rate = np.tile(np.linspace(2e-4, 5e-4, n)[:, None], (1, time_grid.size))

        with patch("dnascape.fitting.rsim", side_effect=AssertionError("rsim should not be called")):
            constant_fit = rmap(
                source={"replication_timing": timing, "firing_rate": firing_rate},
                target="fork_speed",
                assumptions={
                    "target_form": "constant",
                    "resolution_t": 5.0,
                    "maxiter": 1,
                    "verbose": False,
                },
                return_info=True,
            )
            spatial_fit = rmap(
                source={"replication_timing": timing, "firing_rate": firing_rate},
                target="fork_speed",
                assumptions={
                    "target_form": "space",
                    "resolution_t": 5.0,
                    "maxiter": 1,
                    "verbose": False,
                },
                return_info=True,
            )
            temporal_fit = rmap(
                source={"replication_timing": timing, "firing_rate": firing_rate},
                target="fork_speed",
                assumptions={
                    "target_form": "time",
                    "resolution_t": 5.0,
                    "maxiter": 1,
                    "verbose": False,
                },
                return_info=True,
            )
            spacetime_fit = rmap(
                source={"replication_timing": timing, "firing_rate": firing_rate},
                target="fork_speed",
                assumptions={
                    "target_form": "spacetime",
                    "resolution_t": 5.0,
                    "maxiter": 1,
                    "verbose": False,
                },
                return_info=True,
            )

        self.assertGreater(float(constant_fit["values"]), 0.0)
        self.assertEqual(spatial_fit["values"].shape, (n,))
        self.assertEqual(temporal_fit["values"].shape, (1, time_grid.size))
        self.assertEqual(spacetime_fit["values"].shape, (n, time_grid.size))
        self.assertTrue(np.all(spatial_fit["values"] > 0))
        self.assertTrue(np.all(temporal_fit["values"] > 0))
        self.assertTrue(np.all(spacetime_fit["values"] > 0))

    def test_fork_speed_target_forms_cover_all_firing_forms(self):
        n = 9
        time_grid = np.arange(0.0, 45.0, 5.0)
        timing = np.linspace(18.0, 36.0, n)
        firing_forms = {
            "constant": 3e-4,
            "space": np.linspace(2e-4, 5e-4, n),
            "time": np.linspace(2e-4, 5e-4, time_grid.size).reshape(1, -1),
            "spacetime": np.tile(np.linspace(2e-4, 5e-4, n)[:, None], (1, time_grid.size)),
        }
        expected_shapes = {
            "constant": (),
            "space": (n,),
            "time": (1, time_grid.size),
            "spacetime": (n, time_grid.size),
        }

        with patch("dnascape.fitting.rsim", side_effect=AssertionError("rsim should not be called")):
            for firing_label, firing_rate in firing_forms.items():
                for target_form, expected_shape in expected_shapes.items():
                    with self.subTest(firing=firing_label, target_form=target_form):
                        fit = rmap(
                            source={
                                "replication_timing": timing,
                                "firing_rate": firing_rate,
                            },
                            target="fork_speed",
                            assumptions={
                                "target_form": target_form,
                                "time_grid": time_grid,
                                "resolution_t": 5.0,
                                "maxiter": 0,
                                "verbose": False,
                            },
                            return_info=True,
                        )

                        values = np.asarray(fit["values"], dtype=float)
                        self.assertEqual(values.shape, expected_shape)
                        self.assertTrue(np.all(np.isfinite(values)))
                        self.assertTrue(np.all(values > 0))
                        self.assertEqual(fit["assumptions"]["target_shape"], "best")

    def test_spacetime_fork_speed_runs_space_then_time_components(self):
        n = 8
        time_grid = np.arange(0.0, 40.0, 5.0)
        timing = np.linspace(15.0, 30.0, n)
        firing_rate = np.tile(np.linspace(1e-5, 2e-5, n)[:, None], (1, time_grid.size))
        calls = []
        spatial_component = np.linspace(1.0, 1.4, n)
        temporal_component = np.linspace(0.9, 1.1, time_grid.size)

        def fake_space(*args, **kwargs):
            calls.append(("space", np.asarray(args[2], dtype=float).copy(), kwargs.get("maxiter"), kwargs.get("fit_step")))
            return spatial_component

        def fake_time(*args, **kwargs):
            calls.append(("time", np.asarray(args[2], dtype=float).copy(), kwargs.get("maxiter"), kwargs.get("fit_step")))
            return temporal_component

        with patch("dnascape.fitting._fit_spatial_component_for_separable_speed", side_effect=fake_space), \
                patch("dnascape.fitting._fit_temporal_factor_for_separable_speed", side_effect=fake_time):
            values = fitting.map_timing_firing_spacetime_speed(
                {
                    "replication_timing": timing,
                    "firing_rate": firing_rate,
                },
                time_grid=time_grid,
                maxiter=1,
                verbose=False,
            )

        self.assertEqual(values.shape, (n, time_grid.size))
        self.assertEqual(len(calls), 2)
        self.assertEqual([call[0] for call in calls], ["space", "time"])
        self.assertTrue(np.allclose(calls[0][1], np.ones(time_grid.size)))
        self.assertTrue(np.allclose(calls[1][1], spatial_component))
        self.assertEqual(calls[0][2], 1)
        self.assertEqual(calls[1][2], 1)
        self.assertGreater(calls[0][3], 0.0)
        self.assertGreater(calls[1][3], 0.0)


if __name__ == "__main__":
    unittest.main()
