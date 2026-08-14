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
            "s_phase_duration",
            "efficiency",
            "fork_directionality",
            "num_oris",
        ):
            np.testing.assert_array_equal(full[key], lean[key])
        self.assertIsNone(lean["rep_times_per_sim"])
        self.assertIsNone(lean["inter_origin_distances"])
        self.assertIs(lean["num_oris"], lean["num_origins"])
        self.assertIs(
            lean["inter_origin_distances"],
            lean["inter_origin_distance"],
        )

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
            self.assertEqual(len(list(Path(tmpdir).glob("*.dat"))), 5)
            self.assertIsInstance(result["replicated_fraction_xt"], np.memmap)
            del result
            gc.collect()
            self.assertEqual(list(Path(tmpdir).glob("*.dat")), [])


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
