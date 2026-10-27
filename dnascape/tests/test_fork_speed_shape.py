import unittest
from unittest.mock import patch

import numpy as np

from dnascape import fitting, rmap


class ForkSpeedShapeTests(unittest.TestCase):
    def test_exponential_speed_large_diagnostic_uses_retained_fit_prediction(self):
        n = 1601
        timing = np.linspace(20.0, 40.0, n)
        time_grid = np.asarray([0.0, 5.0, 10.0])
        evaluated_sizes = []
        source_sizes = []

        def fake_prediction(source_x, _cdf, target_x, _axis, _speed):
            source_sizes.append(len(source_x))
            evaluated_sizes.append(len(target_x))
            return np.full(len(target_x), time_grid[-1]), np.ones(len(target_x))

        with patch.object(fitting, "_FULL_TIMING_DIAGNOSTIC_CELL_CAP", 0), patch(
            "dnascape.mapping._static_firing_timing_time_speed",
            side_effect=fake_prediction,
        ):
            speed = fitting._fit_exponential_speed_shape(
                timing,
                3e-4,
                "time",
                time_axis=time_grid,
                refine_edge_kb=0,
                maxiter=0,
            )

        info = speed._rmap_info
        self.assertTrue(evaluated_sizes)
        self.assertTrue(all(size < n for size in evaluated_sizes))
        self.assertTrue(all(size == n for size in source_sizes))
        self.assertTrue(info["replication_timing_fit_interpolated"])
        self.assertEqual(np.asarray(info["replication_timing_fit"]).shape, (n,))
        self.assertEqual(np.asarray(info["residual"]).shape, (n,))
        self.assertTrue(np.all(np.isfinite(info["replication_timing_fit"])))
        self.assertTrue(np.all(np.isfinite(info["residual"])))
        self.assertTrue(np.isfinite(info["fit_mse"]))

    def test_constant_speed_large_diagnostic_uses_sampled_mse_and_prediction(self):
        n = 1001
        timing = np.linspace(18.0, 38.0, n)
        time_grid = np.asarray([0.0, 5.0, 10.0])
        evaluated_sizes = []

        def fake_predict(
            _initiation_rate,
            _spatial_speed,
            _temporal_factor,
            observed_timing,
            *_args,
            **_kwargs,
        ):
            current = np.asarray(observed_timing, dtype=float)
            evaluated_sizes.append(current.size)
            return current + 2.0

        with patch.object(fitting, "_FULL_TIMING_DIAGNOSTIC_CELL_CAP", 0), patch.object(
            fitting,
            "_predict_timing_separable_speed",
            side_effect=fake_predict,
        ):
            speed = fitting._fit_constant_speed_with_general_firing(
                timing,
                np.linspace(2e-4, 5e-4, n),
                time_grid=time_grid,
                refine_edge_kb=0,
                maxiter=0,
                verbose=False,
            )

        info = speed._rmap_info
        self.assertTrue(evaluated_sizes)
        self.assertTrue(all(size < n for size in evaluated_sizes))
        self.assertAlmostEqual(info["fit_mse"], 4.0)
        self.assertTrue(info["replication_timing_fit_interpolated"])
        self.assertEqual(np.asarray(info["replication_timing_fit"]).shape, (n,))
        self.assertEqual(np.asarray(info["residual"]).shape, (n,))
        self.assertTrue(np.all(np.isfinite(info["replication_timing_fit"])))
        self.assertTrue(np.all(np.isfinite(info["residual"])))

    def test_spatial_speed_large_diagnostic_uses_sampled_mse_and_prediction(self):
        n = 1001
        timing = np.linspace(18.0, 38.0, n)
        time_grid = np.asarray([0.0, 5.0, 10.0])
        evaluated_sizes = []

        def fake_predict(
            _initiation_rate,
            _spatial_speed,
            _temporal_factor,
            observed_timing,
            *_args,
            **_kwargs,
        ):
            current = np.asarray(observed_timing, dtype=float)
            evaluated_sizes.append(current.size)
            return current + 3.0

        with patch.object(fitting, "_FULL_TIMING_DIAGNOSTIC_CELL_CAP", 0), patch.object(
            fitting,
            "_predict_timing_separable_speed",
            side_effect=fake_predict,
        ):
            speed = fitting._fit_spatial_speed_with_general_firing(
                timing,
                np.linspace(2e-4, 5e-4, n),
                time_grid=time_grid,
                refine_edge_kb=0,
                smoothw=None,
                maxiter=0,
                verbose=False,
            )

        info = speed._rmap_info
        self.assertTrue(evaluated_sizes)
        self.assertTrue(all(size < n for size in evaluated_sizes))
        self.assertAlmostEqual(info["fit_mse"], 9.0)
        self.assertTrue(info["replication_timing_fit_interpolated"])
        self.assertEqual(np.asarray(info["replication_timing_fit"]).shape, (n,))
        self.assertEqual(np.asarray(info["residual"]).shape, (n,))
        self.assertTrue(np.all(np.isfinite(info["replication_timing_fit"])))
        self.assertTrue(np.all(np.isfinite(info["residual"])))

    def test_failed_exact_diagnostic_falls_back_without_failing_fit(self):
        n = 1001
        timing = np.linspace(18.0, 38.0, n)
        evaluated_sizes = []

        def fake_predict(
            _initiation_rate,
            _spatial_speed,
            _temporal_factor,
            observed_timing,
            *_args,
            **_kwargs,
        ):
            current = np.asarray(observed_timing, dtype=float)
            evaluated_sizes.append(current.size)
            if current.size == n:
                raise MemoryError("synthetic full-grid allocation failure")
            return current + 1.0

        with patch.object(
            fitting,
            "_predict_timing_separable_speed",
            side_effect=fake_predict,
        ):
            speed = fitting._fit_constant_speed_with_general_firing(
                timing,
                np.linspace(2e-4, 5e-4, n),
                time_grid=np.asarray([0.0, 5.0, 10.0]),
                refine_edge_kb=0,
                maxiter=0,
                verbose=False,
            )

        info = speed._rmap_info
        self.assertIn(n, evaluated_sizes)
        self.assertTrue(info["replication_timing_fit_interpolated"])
        self.assertEqual(np.asarray(info["replication_timing_fit"]).shape, (n,))
        self.assertTrue(np.all(np.isfinite(info["replication_timing_fit"])))

    def test_constant_fork_speed_target_shape_defaults_to_best(self):
        n = 10
        timing = np.linspace(18.0, 40.0, n)
        initiation_rate = np.linspace(2e-4, 5e-4, n)

        fit = rmap(
            source={
                "replication_timing": timing,
                "initiation_rate": initiation_rate,
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
        initiation_rate = np.linspace(2e-4, 5e-4, n)

        fit = rmap(
            source={
                "replication_timing": timing,
                "initiation_rate": initiation_rate,
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
        self.assertFalse(fit["converged"])
        self.assertIn("Initialization only", fit["convergence_message"])
        self.assertEqual(fit["optimizer_evaluations"], 1)
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
        initiation_rate = np.linspace(2e-4, 5e-4, n)

        with patch("dnascape.fitting.map_timing_firing_spatial_speed") as spatial_mock:
            fit = rmap(
                source={
                    "replication_timing": timing,
                    "initiation_rate": initiation_rate,
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
        self.assertEqual(fit["fork_speed_shape"], "exponential v(t)")
        self.assertEqual(fit["fork_speed_component_t"], "v0 * exp(b_t * T)")
        self.assertTrue(np.all(fit["values"] > 0))
        self.assertEqual(np.asarray(fit["replication_timing_fit"]).shape, (n,))
        self.assertEqual(np.asarray(fit["residual"]).shape, (n,))
        self.assertTrue(np.all(np.isfinite(fit["replication_timing_fit"])))
        self.assertTrue(np.all(np.isfinite(fit["residual"])))
        self.assertTrue(np.isfinite(fit["fit_mse"]))

    def test_spacetime_fork_speed_best_shape_uses_factorized_components(self):
        n = 10
        time_grid = np.arange(0.0, 60.0, 5.0)
        timing = np.linspace(18.0, 40.0, n)
        initiation_rate = np.tile(np.linspace(2e-4, 5e-4, n)[:, None], (1, time_grid.size))

        with patch("dnascape.fitting._refine_spacetime_speed") as refine_mock:
            fit = rmap(
                source={
                    "replication_timing": timing,
                    "initiation_rate": initiation_rate,
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
        self.assertEqual(fit["fork_speed_shape"], "crossed factorized v_x(x) * q(t)")
        self.assertEqual(fit["spacetime_order"], "crossed")
        self.assertEqual(fit["fit_history"], [])
        self.assertTrue(np.isfinite(fit["fit_mse"]))
        self.assertTrue(np.all(fit["values"] > 0))

    def test_spacetime_firing_to_fork_speed_uses_deterministic_routes(self):
        n = 8
        time_grid = np.arange(0.0, 35.0, 5.0)
        timing = np.linspace(18.0, 38.0, n)
        initiation_rate = np.tile(np.linspace(2e-4, 5e-4, n)[:, None], (1, time_grid.size))

        with patch("dnascape.fitting.rsim", side_effect=AssertionError("rsim should not be called")):
            constant_fit = rmap(
                source={"replication_timing": timing, "initiation_rate": initiation_rate},
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
                source={"replication_timing": timing, "initiation_rate": initiation_rate},
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
                source={"replication_timing": timing, "initiation_rate": initiation_rate},
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
                source={"replication_timing": timing, "initiation_rate": initiation_rate},
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
        np.testing.assert_array_equal(temporal_fit["time_grid"], time_grid)
        np.testing.assert_array_equal(spacetime_fit["time_grid"], time_grid)
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
            for firing_label, initiation_rate in firing_forms.items():
                for target_form, expected_shape in expected_shapes.items():
                    with self.subTest(firing=firing_label, target_form=target_form):
                        fit = rmap(
                            source={
                                "replication_timing": timing,
                                "initiation_rate": initiation_rate,
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
                        if target_form == "spacetime":
                            self.assertEqual(fit["fork_speed_shape"], "crossed factorized v_x(x) * q(t)")
                            self.assertEqual(fit["spacetime_order"], "crossed")
                            self.assertEqual(fit["fit_history"], [])

    def test_spacetime_fork_speed_default_crossed_updates_space_and_time(self):
        n = 8
        time_grid = np.arange(0.0, 40.0, 5.0)
        timing = np.linspace(15.0, 30.0, n)
        firing_space = np.linspace(1e-5, 2e-5, n)
        firing_time = np.linspace(0.8, 1.2, time_grid.size)
        initiation_rate = np.outer(firing_space, firing_time)

        def fake_predict(_initiation_rate, _spatial_speed, temporal_factor, observed_timing, *_args, **_kwargs):
            slope = float(temporal_factor[-1] - temporal_factor[0])
            space_gain = float(np.nanmean(_spatial_speed) - 1.4)
            return np.asarray(observed_timing, dtype=float) + 0.4 - 2.0 * space_gain - 4.0 * slope

        with patch("dnascape.fitting._predict_timing_separable_speed", side_effect=fake_predict), \
                patch("dnascape.fitting._spacetime_speed_timing_fit_info", return_value={}):
            fit = fitting.map_timing_firing_spacetime_speed(
                {
                    "replication_timing": timing,
                    "initiation_rate": initiation_rate,
                },
                time_grid=time_grid,
                maxiter=2,
                temporal_fit_step=0.1,
                verbose=False,
            )

        values = np.asarray(fit, dtype=float)
        info = getattr(fit, "_rmap_info", {})
        self.assertEqual(values.shape, (n, time_grid.size))
        self.assertGreater(np.ptp(np.nanmean(values, axis=1)), 0.0)
        self.assertGreater(np.ptp(np.nanmean(values, axis=0)), 0.0)
        self.assertEqual(info["spacetime_order"], "crossed")
        self.assertGreater(info["fork_speed_crossed_space_steps"], 0)
        self.assertGreater(info["fork_speed_crossed_time_steps"], 0)


if __name__ == "__main__":
    unittest.main()
