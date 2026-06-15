import tempfile
import unittest
from pathlib import Path

from dnascape.io import csv_to_tsv


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


if __name__ == "__main__":
    unittest.main()
