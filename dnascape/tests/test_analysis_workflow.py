"""Numerical analysis preserves source data and pairs matching coordinates."""

import csv
import io
from concurrent.futures import CancelledError

import numpy as np
import pytest
from scipy import stats

from dnascape.workflows.analysis import (
    AnalysisDataset, SmoothingSettings, analyse_profiles, export_correlation_report, smooth_profile,
)


def test_full_symmetric_matrices_match_scipy_including_rank_ties() -> None:
    values = [np.array([1, 2, 2, 4, 5.]), np.array([5, 2, 3, 1, 4.]), np.array([2, 4, 4, 8, 10.])]
    datasets = [AnalysisDataset(name, array) for name, array in zip(["A", "B", "C"], values)]
    report = analyse_profiles(datasets)
    functions = {"spearman": stats.spearmanr, "pearson": stats.pearsonr, "kendall": stats.kendalltau}
    assert report.names == ("A", "B", "C")
    for matrix in report.matrices:
        assert matrix.coefficients.shape == (3, 3)
        np.testing.assert_array_equal(matrix.sample_counts, np.full((3, 3), 5))
        np.testing.assert_allclose(matrix.coefficients, matrix.coefficients.T)
        np.testing.assert_allclose(np.diag(matrix.coefficients), 1)
        for i in range(3):
            for j in range(i + 1, 3):
                assert matrix.coefficients[i, j] == pytest.approx(functions[matrix.method](values[i], values[j]).statistic)


def test_missing_pairs_and_unequal_lengths_are_filtered_without_shifting_indices() -> None:
    report = analyse_profiles([
        AnalysisDataset("A", np.array([1, np.nan, 3, 4, 9.])),
        AnalysisDataset("B", np.array([2, 100, 6, np.inf])),
    ])
    for matrix in report.matrices:
        assert matrix.sample_counts[0, 1] == 2
        assert matrix.coefficients[0, 1] == pytest.approx(1)


def test_constant_and_insufficient_pairs_are_undefined_and_dependences_are_not_mixed() -> None:
    datasets = [AnalysisDataset("Constant", np.ones(4)), AnalysisDataset("Single", np.array([1.])),
                AnalysisDataset("Time", np.arange(4), "t"), AnalysisDataset("Space", np.arange(4), "x")]
    report = analyse_profiles(datasets)
    for matrix in report.matrices:
        assert np.isnan(matrix.coefficients[:2]).all()
        assert np.isnan(matrix.coefficients[2, 3])
        assert matrix.sample_counts[2, 3] == 0


def test_spacetime_alignment_crops_each_dimension_before_flattening() -> None:
    first = np.arange(12).reshape(3, 4)
    second = np.array([[0, 2], [8, 10]])
    report = analyse_profiles([AnalysisDataset("A", first, "xt"), AnalysisDataset("B", second, "xt")])
    for matrix in report.matrices:
        assert matrix.sample_counts[0, 1] == 4
        assert matrix.coefficients[0, 1] == pytest.approx(1)


def test_independent_smoothing_preserves_missing_samples_grid_and_original_values() -> None:
    original = np.array([1., 8., np.nan, 3., 5.])
    copy = original.copy()
    averaged = smooth_profile(original, SmoothingSettings("moving_average", 3))
    np.testing.assert_allclose(averaged, [4.5, 4.5, np.nan, 4, 4], equal_nan=True)
    gaussian = smooth_profile(original, SmoothingSettings("gaussian", sigma=1))
    assert gaussian.shape == original.shape and np.isnan(gaussian[2])
    np.testing.assert_allclose(original, copy, equal_nan=True)
    np.testing.assert_allclose(smooth_profile(original, SmoothingSettings()), original, equal_nan=True)


def test_smoothing_time_profiles_and_fields_keeps_axes_independent() -> None:
    row = np.array([[0., 9., 0.]])
    np.testing.assert_allclose(smooth_profile(row, SmoothingSettings("moving_average", 3), dependence="t"), [[4.5, 3, 4.5]])
    field = np.column_stack((row[0], row[0] * 10))
    np.testing.assert_allclose(smooth_profile(field, SmoothingSettings("moving_average", 3), dependence="xt"),
                               [[4.5, 45], [3, 30], [4.5, 45]])


@pytest.mark.parametrize("file_format,delimiter", [("csv", ","), ("tsv", "\t")])
def test_report_exports_all_coefficients_and_sample_counts_in_one_file(file_format, delimiter) -> None:
    report = analyse_profiles([AnalysisDataset("A, first", np.arange(4)), AnalysisDataset("B", -np.arange(4))])
    rows = list(csv.reader(io.StringIO(export_correlation_report(report, file_format=file_format).decode()), delimiter=delimiter))
    assert rows[0] == ["Correlation", "Statistic", "Dataset", "A, first", "B"]
    assert len(rows) == 13 and rows[1][3:] == ["1", "-1"]
    assert {row[0] for row in rows[1:]} == {"Spearman", "Pearson", "Kendall tau"}
    assert rows[3] == ["Spearman", "Paired samples", "A, first", "4", "4"]
    individual = list(csv.reader(io.StringIO(export_correlation_report(report, file_format=file_format, method="pearson").decode()), delimiter=delimiter))
    assert individual[0] == ["Statistic", "Dataset", "A, first", "B"]
    assert len(individual) == 5


def test_correlation_calculation_can_be_cancelled() -> None:
    def cancel():
        raise CancelledError

    with pytest.raises(CancelledError):
        analyse_profiles([AnalysisDataset("A", np.arange(5))], cancel=cancel)
