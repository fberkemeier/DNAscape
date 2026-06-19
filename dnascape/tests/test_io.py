import tempfile
import unittest
from pathlib import Path

import numpy as np

from dnascape.io import csv_to_tsv, rdata, rload, rsave


class IOTests(unittest.TestCase):
    def test_csv_to_tsv_defaults_to_tsv_output_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "table.csv"
            path.write_text("a,b\n1.234,2\n", encoding="utf-8")

            output_path = csv_to_tsv(path)

            self.assertEqual(output_path, Path(tmpdir) / "table.tsv")
            self.assertEqual(path.read_text(encoding="utf-8"), "a,b\n1.234,2\n")
            self.assertEqual(output_path.read_text(encoding="utf-8"), "a\tb\n1.23\t2.00\n")

    def test_csv_to_tsv_uses_output_path_when_provided(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "table.csv"
            output_path = Path(tmpdir) / "table.tsv"
            input_path.write_text('a,b,label\n1.23456,2,"two, words"\n', encoding="utf-8")

            result_path = csv_to_tsv(input_path, output_path, decimal_places=3)

            self.assertEqual(result_path, output_path)
            self.assertEqual(output_path.read_text(encoding="utf-8"), "a\tb\tlabel\n1.235\t2.000\ttwo, words\n")
            self.assertEqual(input_path.read_text(encoding="utf-8"), 'a,b,label\n1.23456,2,"two, words"\n')

    def test_csv_to_tsv_detects_tab_delimited_csv_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "table.csv"
            input_path.write_text("chr1\tchr2\n206.0462\t253.2694\n", encoding="utf-8")

            output_path = csv_to_tsv(input_path)

            self.assertEqual(output_path.read_text(encoding="utf-8"), "chr1\tchr2\n206.05\t253.27\n")

    def test_rdata_preserves_chromosome_labels_when_converting_tables(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "timing.csv"
            input_path.write_text("chr1\tchr2\n1.2345\t3.4567\n2.3456\t4.5678\n", encoding="utf-8")

            output_path = rdata(input_path, target="tsv", decimal_places=2)

            self.assertEqual(output_path, Path(tmpdir) / "timing.tsv")
            self.assertEqual(
                output_path.read_text(encoding="utf-8"),
                "chr1\tchr2\n1.23\t3.46\n2.35\t4.57\n",
            )

    def test_rdata_extracts_chromosome_column_to_txt_vector(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "timing.tsv"
            output_path = Path(tmpdir) / "chr2.txt"
            input_path.write_text("chr1\tchr2\n1\t3\n2\t4\n", encoding="utf-8")

            rdata(input_path, target=output_path, chromosome="chr2")

            values = np.loadtxt(output_path)
            np.testing.assert_allclose(values, np.array([3.0, 4.0]))

    def test_rdata_writes_temporal_vectors_as_transposed_arrays(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "temporal.npv"

            result = rdata(
                np.array([1.0, 2.0, 3.0]),
                target="npv",
                output_path=output_path,
                data_form="time",
                return_data=True,
            )

            loaded = np.load(output_path, allow_pickle=False)
            self.assertEqual(result["data_form"], "time")
            self.assertEqual(loaded.shape, (1, 3))
            np.testing.assert_allclose(loaded, np.array([[1.0, 2.0, 3.0]]))

    def test_rdata_roundtrips_npz_metadata_to_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            npz_path = Path(tmpdir) / "field.npz"
            csv_path = Path(tmpdir) / "field.csv"
            np.savez(
                npz_path,
                values=np.array([[1.0, 2.0], [3.0, 4.0]]),
                columns=np.array(["t0", "t1"]),
                data_form=np.array("spacetime"),
            )

            result = rdata(npz_path, target=csv_path, return_data=True)

            self.assertEqual(result["data_form"], "spacetime")
            self.assertEqual(csv_path.read_text(encoding="utf-8"), "t0,t1\n1.0,2.0\n3.0,4.0\n")

    def test_rdata_reports_route_progress(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "timing.csv"
            output_path = Path(tmpdir) / "timing.tsv"
            input_path.write_text("chr1,chr2\n1,3\n2,4\n", encoding="utf-8")
            events = []

            rdata(input_path, target=output_path, progress=events.append)

            self.assertEqual(events[0]["route"], "csv->tsv")
            self.assertEqual(events[0]["stage"], "load")
            self.assertEqual(events[-1]["stage"], "complete")
            self.assertEqual(events[-1]["status"], "done")
            self.assertEqual(events[-1]["progress"], 1.0)
            self.assertEqual(events[-1]["output_path"], str(output_path))

    def test_rload_extracts_chromosome_as_dnascape_vector(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "timing.csv"
            input_path.write_text("chr1,chr2\n1,3\n2,4\n", encoding="utf-8")

            values = rload(input_path, chromosome="chr1")

            self.assertEqual(values.shape, (2,))
            np.testing.assert_allclose(values, np.array([1.0, 2.0]))

    def test_rload_can_enforce_time_row_array(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "time.txt"
            input_path.write_text("1\n2\n3\n", encoding="utf-8")

            values = rload(input_path, data_form="time")

            self.assertEqual(values.shape, (1, 3))
            np.testing.assert_allclose(values, np.array([[1.0, 2.0, 3.0]]))

    def test_rsave_infers_format_and_preserves_spacetime_labels(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "field.tsv"

            result_path = rsave(
                np.array([[1.0, 2.0], [3.0, 4.0]]),
                output_path,
                data_form="spacetime",
                labels=["t0", "t1"],
                decimal_places=1,
            )

            self.assertEqual(result_path, output_path)
            self.assertEqual(output_path.read_text(encoding="utf-8"), "t0\tt1\n1.0\t2.0\n3.0\t4.0\n")

    def test_rsave_time_form_writes_transposed_npv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "time.npv"

            rsave(np.array([1.0, 2.0, 3.0]), output_path, data_form="time")

            values = np.load(output_path, allow_pickle=False)
            self.assertEqual(values.shape, (1, 3))
            np.testing.assert_allclose(values, np.array([[1.0, 2.0, 3.0]]))

    def test_rload_and_rsave_are_public_api(self):
        from dnascape import rload as public_rload, rsave as public_rsave

        self.assertIs(public_rload, rload)
        self.assertIs(public_rsave, rsave)


if __name__ == "__main__":
    unittest.main()
