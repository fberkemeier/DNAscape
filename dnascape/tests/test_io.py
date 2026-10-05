import gzip
import tempfile
import unittest
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd

from dnascape.io import csv_to_tsv, inspect_genomic_track, rdata, rload, rsave, rtrack


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

    def test_rtrack_bins_bedgraph_as_space_array(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "signal.bedgraph"
            input_path.write_text("chr1\t0\t10\t1\nchr1\t20\t30\t3\n", encoding="utf-8")

            values = rtrack(
                input_path,
                chromosome="chr1",
                start=0,
                end=30,
                bin_size=10,
            )

            np.testing.assert_allclose(values, np.array([1.0, np.nan, 3.0]), equal_nan=True)

    def test_rtrack_bins_three_column_bed_intervals_as_presence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "sns_origins.bed"
            input_path.write_text(
                "chr1\t0\t500\nchr1\t1000\t1500\n",
                encoding="utf-8",
            )

            values = rtrack(
                input_path,
                source_format="bedgraph",
                chromosome="chr1",
                start=0,
                end=2000,
                bin_size=1000,
                aggregation="coverage",
            )

            np.testing.assert_allclose(values, np.array([0.5, 0.5]))

    def test_rtrack_interpolates_and_normalizes_bedgraph(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "signal.bdg"
            input_path.write_text("chr1\t0\t10\t1\nchr1\t20\t30\t3\n", encoding="utf-8")

            values = rtrack(
                input_path,
                chromosome="chr1",
                start=0,
                end=30,
                bin_size=10,
                interpolation="linear",
                normalize="minmax",
            )

            np.testing.assert_allclose(values, np.array([0.0, 0.5, 1.0]))

    def test_rtrack_combines_tracks_as_spacetime_matrix(self):
        first = np.array(
            [
                ["chr1", 0, 10, 1.0],
                ["chr1", 10, 20, 2.0],
            ],
            dtype=object,
        )
        second = np.array(
            [
                ["chr1", 0, 10, 3.0],
                ["chr1", 10, 20, 4.0],
            ],
            dtype=object,
        )

        result = rtrack(
            [
                pd.DataFrame(
                    {"chrom": first[:, 0], "start": first[:, 1], "end": first[:, 2], "signal": first[:, 3]}
                ),
                pd.DataFrame(
                    {"chrom": second[:, 0], "start": second[:, 1], "end": second[:, 2], "signal": second[:, 3]}
                ),
            ],
            source_format="bedgraph",
            chromosome="chr1",
            start=0,
            end=20,
            bin_size=10,
            labels=["early", "late"],
            return_metadata=True,
        )

        self.assertEqual(result["data_form"], "spacetime")
        self.assertEqual(result["labels"], ["early", "late"])
        np.testing.assert_allclose(result["values"], np.array([[1.0, 3.0], [2.0, 4.0]]))

    def test_inspect_genomic_track_summarizes_bedgraph_chromosomes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "signal.bedgraph"
            input_path.write_text("chr2\t0\t5\t1\nchr1\t0\t10\t2\nchr1\t10\t20\t3\n", encoding="utf-8")

            summary = inspect_genomic_track(input_path)

            self.assertEqual(summary["format"], "bedgraph")
            self.assertEqual(summary["chromosome_names"], ["chr1", "chr2"])
            self.assertEqual(summary["chromosomes"]["chr1"], 20)

    def test_inspect_genomic_track_skips_ucsc_bedgraph_preamble(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            payload = (
                '\ufefftrack type=bedGraph name="SNS signal" visibility=full\n'
                "chr1\t0\t10\t24\n"
                "  browser position chr1:1-20\n"
                "  # an interleaved comment\n"
                "chr1\t10\t20\t4\n"
            )
            paths = (
                Path(tmpdir) / "signal.bedgraph",
                Path(tmpdir) / "signal.bedgraph.gz",
            )
            paths[0].write_text(payload, encoding="utf-8")
            with gzip.open(paths[1], "wt", encoding="utf-8") as handle:
                handle.write(payload)

            for input_path in paths:
                with self.subTest(input_path=input_path.name):
                    summary = inspect_genomic_track(input_path)
                    result = rtrack(
                        input_path,
                        chromosome="chr1",
                        start=0,
                        end=20,
                        bin_size=10,
                    )

                    self.assertEqual(summary["chromosome_names"], ["chr1"])
                    self.assertEqual(summary["chromosomes"]["chr1"], 20)
                    self.assertEqual(summary["table"]["intervals"].tolist(), [2])
                    np.testing.assert_allclose(result, np.array([24.0, 4.0]))

    def test_rdata_converts_bedgraph_to_coordinate_table(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "signal.bedgraph"
            output_path = Path(tmpdir) / "signal.tsv"
            input_path.write_text("chr1\t0\t10\t1\nchr1\t20\t30\t3\n", encoding="utf-8")

            result = rdata(
                input_path,
                target=output_path,
                chromosome="chr1",
                start=0,
                end=30,
                bin_size=10,
                missing="zero",
                return_data=True,
            )

            self.assertEqual(result["source_format"], "bedgraph")
            self.assertEqual(result["data_form"], "space")
            self.assertEqual(
                output_path.read_text(encoding="utf-8"),
                "chrom\tstart_bp\tend_bp\tposition_bp\tposition_kb\tvalue\n"
                "chr1\t0\t10\t5.0\t0.005\t1.0\n"
                "chr1\t10\t20\t15.0\t0.015\t0.0\n"
                "chr1\t20\t30\t25.0\t0.025\t3.0\n",
            )

    def test_rdata_converts_compressed_bedgraph_to_txt_vector(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "signal.bedgraph.gz"
            output_path = Path(tmpdir) / "signal.txt"
            with gzip.open(input_path, "wt", encoding="utf-8") as handle:
                handle.write("chr1\t0\t10\t1\nchr1\t20\t30\t3\n")

            rdata(
                input_path,
                target=output_path,
                chromosome="chr1",
                start=0,
                end=30,
                bin_size=10,
                interpolation="linear",
            )

            np.testing.assert_allclose(np.loadtxt(output_path), np.array([1.0, 2.0, 3.0]))

    def test_rdata_can_write_gzip_text_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "timing.csv"
            output_path = Path(tmpdir) / "timing.tsv.gz"
            input_path.write_text("chr1,chr2\n1,3\n2,4\n", encoding="utf-8")

            result_path = rdata(input_path, target=output_path)

            self.assertEqual(result_path, output_path)
            with gzip.open(output_path, "rt", encoding="utf-8") as handle:
                self.assertEqual(handle.read(), "chr1\tchr2\n1\t3\n2\t4\n")

    def test_rdata_can_write_gzip_binary_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "values.npy.gz"

            result_path = rdata(
                np.array([1.0, 2.0, 3.0]),
                target=output_path,
                data_form="space",
            )

            self.assertEqual(result_path, output_path)
            with gzip.open(output_path, "rb") as handle:
                loaded = np.load(BytesIO(handle.read()), allow_pickle=False)
            np.testing.assert_allclose(loaded, np.array([1.0, 2.0, 3.0]))

    def test_rload_extracts_chromosome_as_dnascape_vector(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "timing.csv"
            input_path.write_text("chr1,chr2\n1,3\n2,4\n", encoding="utf-8")

            values = rload(input_path, chromosome="chr1")

            self.assertEqual(values.shape, (2,))
            np.testing.assert_allclose(values, np.array([1.0, 2.0]))

    def test_rload_preserves_missing_bins_in_a_selected_chromosome(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "timing.csv"
            input_path.write_text("chr1,chr2\n1,3\n,4\n2,5\n,6\n", encoding="utf-8")

            values = rload(input_path, chromosome="chr1")

            # The internal gap remains a genomic bin; only the structural
            # blank used to pad the shorter chromosome at the tail is removed.
            self.assertEqual(values.shape, (3,))
            self.assertTrue(np.isnan(values[1]))
            np.testing.assert_allclose(values[[0, 2]], np.array([1.0, 2.0]))

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
        from dnascape import rload as public_rload
        from dnascape import rsave as public_rsave
        from dnascape import rtrack as public_rtrack

        self.assertIs(public_rload, rload)
        self.assertIs(public_rsave, rsave)
        self.assertIs(public_rtrack, rtrack)


if __name__ == "__main__":
    unittest.main()
