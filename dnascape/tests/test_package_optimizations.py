import ast
import gc
import gzip
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import dnascape.io as io_module
import dnascape.routing as routing_module
from dnascape.plotting import mplotf, ploth
from dnascape.simulation import (
    _as_inverse_speed_field,
    _as_rate_field,
    rsim,
    simulate_replication,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _run_clean_python(source, *, cwd=None):
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(REPOSITORY_ROOT), existing) if part
    )
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [sys.executable, "-c", source],
        cwd=cwd,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


class LazyImportTests(unittest.TestCase):
    def test_package_root_does_not_eagerly_import_scientific_stack(self):
        _run_clean_python(
            "import sys; import dnascape; "
            "assert not {'numpy','pandas','matplotlib','scipy'} & set(sys.modules)"
        )

    def test_io_import_does_not_create_working_directories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _run_clean_python(
                "from pathlib import Path; import dnascape.io; "
                "assert not list(Path.cwd().iterdir())",
                cwd=tmpdir,
            )

    def test_plotting_import_keeps_specialist_dependencies_lazy(self):
        _run_clean_python(
            "import sys; import dnascape.plotting; "
            "assert 'scipy' not in sys.modules; "
            "assert 'pandas' not in sys.modules; "
            "assert 'seaborn' not in sys.modules"
        )


class SelectiveLoadingTests(unittest.TestCase):
    def test_selected_table_column_uses_usecols_and_explicit_separator(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "wide.csv"
            path.write_text(
                "chr1,chr2,annotation\n1,3,a\n2,4,b\n",
                encoding="utf-8",
            )
            original = io_module.pd.read_csv
            with patch.object(io_module.pd, "read_csv", wraps=original) as reader:
                values = io_module.rload(path, chromosome="chr2")

            np.testing.assert_array_equal(values, np.array([3.0, 4.0]))
            kwargs = reader.call_args.kwargs
            self.assertTrue(callable(kwargs["usecols"]))
            self.assertEqual(kwargs["sep"], ",")

    def test_selected_columns_support_gzip_and_delimited_txt(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gz_path = Path(tmpdir) / "wide.tsv.gz"
            with gzip.open(gz_path, "wt", encoding="utf-8") as handle:
                handle.write("chr1\tchr2\tnote\n1\t3\ta\n2\t4\tb\n")
            np.testing.assert_array_equal(
                io_module.rload(gz_path, chromosome="chr1"),
                np.array([1.0, 2.0]),
            )

            txt_path = Path(tmpdir) / "wide.txt"
            txt_path.write_text(
                "first|second|note\n5|8|a\n6|9|b\n",
                encoding="utf-8",
            )
            np.testing.assert_array_equal(
                io_module.rload(txt_path, value_column="second"),
                np.array([8.0, 9.0]),
            )

    def test_bigwig_partial_bin_uses_exact_true_boundaries(self):
        class FakeHandle:
            def __init__(self):
                self.calls = []

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def chroms(self):
                return {"chr1": 25}

            def stats(self, chrom, start, end, *, nBins, type, exact):
                self.calls.append((chrom, start, end, nBins, type, exact))
                value = 2.0 if type == "mean" else 0.5
                return [value] * nBins

        class FakeBigWig:
            def __init__(self):
                self.handle = None

            def open(self, _path):
                self.handle = FakeHandle()
                return self.handle

        fake = FakeBigWig()
        with patch("dnascape.io._require_pybigwig", return_value=fake):
            table = io_module._bin_bigwig_chromosome(
                "unused.bw",
                "chr1",
                0,
                25,
                10,
                "sum",
                "value",
            )

        np.testing.assert_array_equal(table["end_bp"], np.array([10, 20, 25]))
        np.testing.assert_allclose(table["value"], np.array([10.0, 10.0, 5.0]))
        self.assertIn(("chr1", 0, 20, 2, "mean", True), fake.handle.calls)
        self.assertIn(("chr1", 20, 25, 1, "mean", True), fake.handle.calls)
        self.assertIn(("chr1", 20, 25, 1, "coverage", True), fake.handle.calls)


class SimulationOptimizationTests(unittest.TestCase):
    def test_retention_controls_preserve_seeded_summaries(self):
        kwargs = {
            "firing_rate": np.linspace(0.001, 0.003, 24),
            "fork_speed": 1.4,
            "sim_number": 5,
            "resolution_space": 1.0,
            "seed": 9182,
            "verbose": False,
        }
        full = rsim(**kwargs)
        lean = rsim(
            **kwargs,
            retain_rep_times_per_sim=False,
            retain_inter_origin_distances=False,
        )

        for key in (
            "replication_timing",
            "replication_timing_std",
            "s_phase_duration",
            "efficiency",
            "fork_directionality",
            "num_oris",
        ):
            np.testing.assert_array_equal(full[key], lean[key])
        np.testing.assert_allclose(
            full["replication_timing_std"],
            np.std(full["rep_times_per_sim"], axis=0, ddof=0),
        )
        self.assertIsNone(lean["rep_times_per_sim"])
        self.assertIsNone(lean["inter_origin_distances"])
        self.assertIs(lean["num_oris"], lean["num_origins"])
        self.assertIs(
            lean["inter_origin_distances"],
            lean["inter_origin_distance"],
        )
        self.assertEqual(
            full["inter_origin_distance_count"],
            full["inter_origin_distances"].size,
        )
        self.assertEqual(
            lean["inter_origin_distance_count"],
            full["inter_origin_distance_count"],
        )
        self.assertFalse(full["inter_origin_distances_truncated"])
        self.assertFalse(lean["inter_origin_distances_truncated"])

    def test_replication_profile_retention_is_bounded_without_sampling_summaries(self):
        kwargs = {
            "firing_rate": np.linspace(0.001, 0.003, 18),
            "fork_speed": 1.4,
            "sim_number": 6,
            "resolution_space": 1.0,
            "seed": 193,
            "verbose": False,
            "retain_inter_origin_distances": False,
        }
        full = rsim(**kwargs)
        bounded = rsim(**kwargs, retain_rep_times_limit=2)
        oversized_limit = rsim(**kwargs, retain_rep_times_limit=50)

        self.assertEqual(bounded["rep_times_per_sim"].shape, (2, 18))
        np.testing.assert_array_equal(
            bounded["rep_times_per_sim"],
            full["rep_times_per_sim"][:2],
        )
        self.assertEqual(oversized_limit["rep_times_per_sim"].shape, (6, 18))
        np.testing.assert_array_equal(
            bounded["replication_timing"],
            full["replication_timing"],
        )
        np.testing.assert_array_equal(
            bounded["replication_timing_std"],
            full["replication_timing_std"],
        )

    def test_replication_timing_std_uses_full_finite_population(self):
        profiles = np.asarray(
            [
                [1.0, np.inf, 3.0, np.inf],
                [3.0, 5.0, np.nan, np.nan],
                [5.0, 7.0, 9.0, np.inf],
            ]
        )
        profile_iterator = iter(profiles)

        def fake_simulate_replication(**kwargs):
            n_bins = int(kwargs["n_bins"])
            replication_time = next(profile_iterator).copy()
            return (
                np.asarray([0], dtype=np.int64),
                np.asarray([0.0]),
                np.asarray([0.0]),
                replication_time,
                np.asarray([], dtype=float),
                None,
                np.zeros(n_bins, dtype=float),
                np.full(n_bins, np.inf),
                np.zeros(n_bins, dtype=np.int64),
                np.full(n_bins, -1, dtype=np.int64),
            )

        with patch("dnascape.simulation.simulate_replication", fake_simulate_replication):
            result = rsim(
                firing_rate=np.full(4, 0.01),
                sim_number=3,
                retain_rep_times_limit=2,
                retain_inter_origin_distances=False,
                verbose=False,
            )

        np.testing.assert_array_equal(result["rep_times_per_sim"], profiles[:2])
        np.testing.assert_allclose(
            result["replication_timing"],
            np.asarray([3.0, 6.0, 6.0, np.inf]),
        )
        np.testing.assert_allclose(
            result["replication_timing_std"][:3],
            np.asarray([np.sqrt(8.0 / 3.0), 1.0, 3.0]),
        )
        self.assertTrue(np.isnan(result["replication_timing_std"][3]))

    def test_replication_profile_limit_validation_is_conditional_on_retention(self):
        base = {
            "firing_rate": 0.001,
            "n_bins": 8,
            "sim_number": 2,
            "retain_inter_origin_distances": False,
            "verbose": False,
        }
        for invalid in (-1, 1.5, np.inf, "many"):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "retain_rep_times_limit"):
                    rsim(**base, retain_rep_times_limit=invalid)

        lean = rsim(
            **base,
            retain_rep_times_per_sim=False,
            retain_rep_times_limit="ignored",
        )
        self.assertIsNone(lean["rep_times_per_sim"])
        empty = rsim(**base, retain_rep_times_limit=0)
        self.assertEqual(empty["rep_times_per_sim"].shape, (0, 8))

    def test_lazy_seed_spawning_matches_eager_child_sequence_order(self):
        seed = 8204
        sim_number = 7
        expected = np.asarray(
            [
                np.random.default_rng(child).random()
                for child in np.random.SeedSequence(seed).spawn(sim_number)
            ]
        )

        def fake_simulate_replication(**kwargs):
            value = kwargs["rng"].random()
            n_bins = int(kwargs["n_bins"])
            return (
                np.asarray([0], dtype=np.int64),
                np.asarray([0.0]),
                np.asarray([value]),
                np.full(n_bins, value),
                np.asarray([], dtype=float),
                None,
                np.zeros(n_bins, dtype=float),
                np.full(n_bins, np.inf),
                np.zeros(n_bins, dtype=np.int64),
                np.full(n_bins, -1, dtype=np.int64),
            )

        with patch("dnascape.simulation.simulate_replication", fake_simulate_replication):
            result = rsim(
                firing_rate=0.01,
                n_bins=1,
                sim_number=sim_number,
                seed=seed,
                retain_rep_times_per_sim=False,
                retain_inter_origin_distances=False,
                verbose=False,
            )

        np.testing.assert_array_equal(result["completion_time"], expected)
        np.testing.assert_array_equal(result["s_phase_duration"], np.zeros(sim_number))

    def test_s_phase_duration_excludes_wait_before_first_realised_initiation(self):
        realizations = iter(
            (
                (
                    np.asarray([0, 2], dtype=np.int64),
                    np.asarray([3.0, 5.0]),
                    np.asarray([3.0, 7.0, 11.0, 9.0]),
                ),
                (
                    np.asarray([1], dtype=np.int64),
                    np.asarray([4.0]),
                    np.asarray([7.0, 4.0, 9.0, 8.0]),
                ),
                (
                    np.asarray([], dtype=np.int64),
                    np.asarray([], dtype=float),
                    np.full(4, 1200.0),
                ),
            )
        )

        def fake_simulate_replication(**kwargs):
            fired_idx, fired_times, replication_time = next(realizations)
            n_bins = int(kwargs["n_bins"])
            return (
                fired_idx,
                fired_idx.astype(float),
                fired_times,
                replication_time,
                np.asarray([], dtype=float),
                None,
                np.zeros(n_bins, dtype=float),
                np.full(n_bins, np.inf),
                np.zeros(n_bins, dtype=np.int64),
                np.full(n_bins, -1, dtype=np.int64),
            )

        with patch("dnascape.simulation.simulate_replication", fake_simulate_replication):
            result = rsim(
                firing_rate=0.01,
                n_bins=4,
                sim_number=3,
                retain_rep_times_per_sim=False,
                retain_inter_origin_distances=False,
                verbose=False,
            )

        np.testing.assert_allclose(
            result["completion_time"][:2],
            np.asarray([11.0, 9.0]),
        )
        np.testing.assert_allclose(
            result["s_phase_duration"][:2],
            np.asarray([8.0, 5.0]),
        )
        self.assertTrue(np.isnan(result["completion_time"][2]))
        self.assertTrue(np.isnan(result["s_phase_duration"][2]))

    def test_inter_origin_retention_limit_preserves_prefix_and_full_count(self):
        iod_values = (
            np.asarray([10.0, 20.0]),
            np.asarray([30.0, 40.0, 50.0]),
            np.asarray([60.0]),
        )
        fired_counts = (3, 4, 2)

        def run(*, retain=True, limit=None):
            calls = iter(zip(fired_counts, iod_values))
            compute_flags = []

            def fake_simulate_replication(**kwargs):
                fired_count, distances = next(calls)
                compute = bool(kwargs["_compute_inter_origin_distances"])
                compute_flags.append(compute)
                n_bins = int(kwargs["n_bins"])
                return (
                    np.arange(fired_count, dtype=np.int64),
                    np.arange(fired_count, dtype=float),
                    np.zeros(fired_count),
                    np.ones(n_bins),
                    distances.copy() if compute else np.asarray([], dtype=float),
                    None,
                    np.zeros(n_bins, dtype=float),
                    np.full(n_bins, np.inf),
                    np.zeros(n_bins, dtype=np.int64),
                    np.full(n_bins, -1, dtype=np.int64),
                )

            with patch("dnascape.simulation.simulate_replication", fake_simulate_replication):
                result = rsim(
                    firing_rate=0.01,
                    n_bins=5,
                    sim_number=3,
                    retain_rep_times_per_sim=False,
                    retain_inter_origin_distances=retain,
                    retain_inter_origin_distances_limit=limit,
                    verbose=False,
                )
            return result, compute_flags

        bounded, bounded_flags = run(limit=4)
        np.testing.assert_array_equal(
            bounded["inter_origin_distances"],
            np.asarray([10.0, 20.0, 30.0, 40.0]),
        )
        self.assertEqual(bounded["inter_origin_distance_count"], 6)
        self.assertTrue(bounded["inter_origin_distances_truncated"])
        self.assertEqual(bounded_flags, [True, True, False])

        empty, empty_flags = run(limit=0)
        self.assertEqual(empty["inter_origin_distances"].size, 0)
        self.assertEqual(empty["inter_origin_distance_count"], 6)
        self.assertTrue(empty["inter_origin_distances_truncated"])
        self.assertEqual(empty_flags, [False, False, False])

        disabled, disabled_flags = run(retain=False, limit="ignored")
        self.assertIsNone(disabled["inter_origin_distances"])
        self.assertEqual(disabled["inter_origin_distance_count"], 6)
        self.assertFalse(disabled["inter_origin_distances_truncated"])
        self.assertEqual(disabled_flags, [False, False, False])

        complete, complete_flags = run()
        np.testing.assert_array_equal(
            complete["inter_origin_distances"],
            np.concatenate(iod_values),
        )
        self.assertEqual(complete["inter_origin_distance_count"], 6)
        self.assertFalse(complete["inter_origin_distances_truncated"])
        self.assertEqual(complete_flags, [True, True, True])

    def test_inter_origin_limit_validation_is_conditional_on_retention(self):
        base = {
            "firing_rate": 0.01,
            "n_bins": 8,
            "sim_number": 2,
            "retain_rep_times_per_sim": False,
            "verbose": False,
        }
        for invalid in (-1, 1.5, np.inf, "many"):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(
                    ValueError,
                    "retain_inter_origin_distances_limit",
                ):
                    rsim(**base, retain_inter_origin_distances_limit=invalid)

        result = rsim(
            **base,
            retain_inter_origin_distances=False,
            retain_inter_origin_distances_limit="ignored",
        )
        self.assertIsNone(result["inter_origin_distances"])

    def test_periodic_inter_origin_count_includes_the_wrap_interval(self):
        calls = iter((3, 1))

        def fake_simulate_replication(**kwargs):
            fired_count = next(calls)
            compute = bool(kwargs["_compute_inter_origin_distances"])
            n_bins = int(kwargs["n_bins"])
            distances = (
                np.asarray([1.0, 1.0, 3.0])
                if compute and fired_count == 3
                else np.asarray([], dtype=float)
            )
            return (
                np.arange(fired_count, dtype=np.int64),
                np.arange(fired_count, dtype=float),
                np.zeros(fired_count),
                np.ones(n_bins),
                distances,
                None,
                np.zeros(n_bins, dtype=float),
                np.full(n_bins, np.inf),
                np.zeros(n_bins, dtype=np.int64),
                np.full(n_bins, -1, dtype=np.int64),
            )

        with patch("dnascape.simulation.simulate_replication", fake_simulate_replication):
            result = rsim(
                firing_rate=0.01,
                n_bins=5,
                sim_number=2,
                perQ=True,
                retain_rep_times_per_sim=False,
                retain_inter_origin_distances_limit=2,
                verbose=False,
            )

        np.testing.assert_array_equal(result["inter_origin_distances"], [1.0, 1.0])
        self.assertEqual(result["inter_origin_distance_count"], 3)
        self.assertTrue(result["inter_origin_distances_truncated"])

    def test_inter_origin_distribution_is_independent_of_time_statistics(self):
        common = {
            "firing_rate": 100.0,
            "fork_speed": 1.4,
            "n_bins": 12,
            "sim_number": 4,
            "seed": 771,
            "verbose": False,
            "retain_rep_times_per_sim": False,
        }
        distances = rsim(
            **common,
            time_statsQ=False,
            time_stats_xtQ=False,
            retain_inter_origin_distances=True,
        )
        time_only = rsim(
            **common,
            time_statsQ=True,
            time_grid=np.arange(0.0, 6.0, 1.0),
            retain_inter_origin_distances=False,
        )

        self.assertGreater(distances["inter_origin_distances"].size, 0)
        self.assertIsNone(distances["time_grid"])
        self.assertIsNone(time_only["inter_origin_distances"])
        self.assertIsNotNone(time_only["replicated_fraction"])
        self.assertEqual(
            distances["inter_origin_distance_count"],
            time_only["inter_origin_distance_count"],
        )
        for key in (
            "replication_timing",
            "replication_timing_std",
            "s_phase_duration",
            "completion_time",
            "num_origins",
            "origin_efficiency",
            "fork_directionality",
        ):
            np.testing.assert_array_equal(distances[key], time_only[key])

    def test_prepared_fields_are_bitwise_seed_equivalent(self):
        n_bins = 14
        resolution_time = 2.0
        firing_rate = np.outer(
            np.linspace(0.001, 0.003, n_bins),
            np.linspace(0.8, 1.2, 12),
        )
        fork_speed = np.outer(
            np.linspace(1.2, 1.6, n_bins),
            np.linspace(0.9, 1.1, 12),
        )
        prepared_firing = _as_rate_field(firing_rate, "firing_rate", n_bins)
        prepared_inverse_speed = _as_inverse_speed_field(
            fork_speed,
            "fork_speed",
            n_bins,
        )
        cumulative_hazard = np.cumsum(
            prepared_firing[1] * resolution_time,
            axis=0,
        )
        common = {
            "firing_rate": firing_rate,
            "fork_speed": fork_speed,
            "n_bins": n_bins,
            "resolution_time": resolution_time,
        }
        reference = simulate_replication(
            **common,
            rng=np.random.default_rng(51),
        )
        prepared = simulate_replication(
            **common,
            rng=np.random.default_rng(51),
            _firing_cumulative_hazard=cumulative_hazard,
            _prepared_firing_field=prepared_firing,
            _prepared_inverse_speed_field=prepared_inverse_speed,
        )

        for expected, actual in zip(reference, prepared):
            if expected is None:
                self.assertIsNone(actual)
            else:
                np.testing.assert_array_equal(expected, actual)

    def test_rsim_rejects_invalid_execution_parameters(self):
        base = {"firing_rate": 0.001, "n_bins": 8, "verbose": False}
        with self.assertRaisesRegex(ValueError, "sim_number"):
            rsim(**base, sim_number=1.5)
        with self.assertRaisesRegex(ValueError, "tau"):
            rsim(**base, tau=np.nan)
        with self.assertRaisesRegex(ValueError, "ffiring_recycle"):
            rsim(**base, ffiring_nr=4, ffiring_recycle=-0.1)
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            rsim(
                **base,
                time_statsQ=True,
                time_grid=np.array([0.0, 2.0, 1.0]),
            )

    def test_space_time_memmaps_are_removed_after_last_view(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = rsim(
                firing_rate=np.full(10, 0.01),
                fork_speed=1.4,
                sim_number=1,
                time_stats_xtQ=True,
                time_grid=np.arange(0.0, 21.0, 5.0),
                dens_memmap_dir=tmpdir,
                retain_rep_times_per_sim=False,
                retain_inter_origin_distances=False,
                seed=7,
                verbose=False,
            )
            self.assertEqual(len(list(Path(tmpdir).glob("*.dat"))), 6)
            self.assertIsInstance(result["replicated_fraction_xt"], np.memmap)
            del result
            gc.collect()
            self.assertEqual(list(Path(tmpdir).glob("*.dat")), [])


class MappingSimulationRetentionTests(unittest.TestCase):
    def test_duration_and_absolute_completion_are_distinct_rmap_features(self):
        def fake_rsim(**_kwargs):
            return {
                "s_phase_duration": np.asarray([7.0, 8.0]),
                "completion_time": np.asarray([10.0, 12.0]),
            }

        self.assertIn("s_phase_duration", routing_module.rmap.features())
        self.assertIn("completion_time", routing_module.rmap.features())
        with patch.object(routing_module, "rsim", side_effect=fake_rsim):
            duration = routing_module.rmap(
                source={"firing_rate": 0.01, "fork_speed": 1.4},
                target="s_phase_duration",
            )
            completion = routing_module.rmap(
                source={"firing_rate": 0.01, "fork_speed": 1.4},
                target="completion_time",
            )

        np.testing.assert_array_equal(duration, [7.0, 8.0])
        np.testing.assert_array_equal(completion, [10.0, 12.0])

    def test_summary_route_does_not_retain_unused_ensemble_arrays(self):
        captured = {}

        def fake_rsim(**kwargs):
            captured.update(kwargs)
            return {"replication_timing": np.arange(4.0)}

        mapper = routing_module._simulation_source_feature_mapper("replication_timing")
        with patch.object(routing_module, "rsim", side_effect=fake_rsim):
            mapped = mapper({"firing_rate": 0.01, "fork_speed": 1.4})

        np.testing.assert_array_equal(mapped, np.arange(4.0))
        self.assertIs(captured["retain_rep_times_per_sim"], False)
        self.assertIs(captured["retain_inter_origin_distances"], False)

    def test_inter_origin_route_retains_only_the_requested_distribution(self):
        captured = {}

        def fake_rsim(**kwargs):
            captured.update(kwargs)
            return {
                "inter_origin_distance": np.asarray([2.0, 3.0]),
                "inter_origin_distance_count": 9,
                "inter_origin_distances_truncated": True,
            }

        mapper = routing_module._simulation_feature_mapper("inter_origin_distance")
        with patch.object(routing_module, "rsim", side_effect=fake_rsim):
            mapped = mapper(0.01, fork_speed=1.4)

        np.testing.assert_array_equal(mapped, [2.0, 3.0])
        self.assertEqual(mapped._rmap_info["inter_origin_distance_count"], 9)
        self.assertIs(mapped._rmap_info["inter_origin_distances_truncated"], True)
        self.assertIs(captured["retain_rep_times_per_sim"], False)
        self.assertIs(captured["retain_inter_origin_distances"], True)

    def test_explicit_route_retention_options_override_memory_safe_defaults(self):
        captured = {}

        def fake_rsim(**kwargs):
            captured.update(kwargs)
            return {"replication_timing": np.arange(2.0)}

        mapper = routing_module._simulation_source_feature_mapper("replication_timing")
        with patch.object(routing_module, "rsim", side_effect=fake_rsim):
            mapper(
                {"firing_rate": 0.01, "fork_speed": 1.4},
                retain_rep_times_per_sim=True,
                retain_inter_origin_distances=True,
            )

        self.assertIs(captured["retain_rep_times_per_sim"], True)
        self.assertIs(captured["retain_inter_origin_distances"], True)

    def test_internal_fits_disable_both_unused_rsim_retentions(self):
        source = (REPOSITORY_ROOT / "dnascape" / "fitting.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "rsim"
        ]

        self.assertGreater(len(calls), 0)
        for call in calls:
            keywords = {item.arg: item.value for item in call.keywords if item.arg is not None}
            with self.subTest(line=call.lineno):
                for name in (
                    "retain_rep_times_per_sim",
                    "retain_inter_origin_distances",
                ):
                    self.assertIn(name, keywords)
                    self.assertIsInstance(keywords[name], ast.Constant)
                    self.assertIs(keywords[name].value, False)


class PlottingOptimizationTests(unittest.TestCase):
    def test_mplotf_and_ploth_can_return_handles_without_showing(self):
        with patch("dnascape.plotting.plt.show") as show:
            figure, axes = mplotf(
                [np.arange(8.0)],
                showQ=False,
                return_handles=True,
            )
            self.assertEqual(len(axes), 1)
            show.assert_not_called()
            plt.close(figure)

        with patch("dnascape.plotting.plt.show") as show:
            figure, axis = ploth(
                np.arange(8.0),
                showQ=False,
                return_handles=True,
            )
            self.assertIs(axis, figure.axes[0])
            show.assert_not_called()
            plt.close(figure)


if __name__ == "__main__":
    unittest.main()
