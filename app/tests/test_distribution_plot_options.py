"""Combined distributions, bounded KDE, and independently scaled axes."""
from __future__ import annotations

import numpy as np
import pytest

from dnascape.workflows.preview import PreviewLine, _kde_data, build_preview_plot_options


def options(plot_type: str, **overrides: object) -> dict:
    settings = dict(
        plot_type=plot_type, title="", x_label="Value", y_label="Count",
        color="#b9340b", palette="viridis", histogram_bins=4, density=False,
        show_legend=True, show_colorbar=False, log_y=False, invert_y=False,
        y_limits=None, series_label="First",
        lines=(
            PreviewLine("First", np.arange(5), np.array([1, 2, 3, 4, 5]), "#b9340b"),
            PreviewLine("Second", np.arange(5), np.array([100, 200, 300, 400, 500]), "#2596be"),
        ),
    )
    settings.update(overrides)
    return build_preview_plot_options(**settings)


def test_combined_histograms_share_bins_but_keep_each_datasets_counts() -> None:
    chart = options("Histogram")
    first, second = chart["series"]
    assert first["z"] > second["z"]
    assert [item["name"] for item in chart["series"]] == ["First", "Second"]
    assert [item["itemStyle"]["color"] for item in chart["series"]] == ["#b9340b", "#2596be"]
    np.testing.assert_allclose(np.asarray(first["data"])[:, :2], np.asarray(second["data"])[:, :2])
    assert [sum(row[2] for row in item["data"]) for item in chart["series"]] == [5, 5]
    assert [row[2] for row in first["data"]] != [row[2] for row in second["data"]]
    assert all(item["itemStyle"]["opacity"] < 1 for item in chart["series"])


def test_combined_boxplots_use_distinct_numeric_positions_and_preserve_quartiles() -> None:
    chart = options("Boxplot")
    assert chart["xAxis"]["type"] == "value"
    assert chart["xAxis"]["axisLabel"]["customValues"] == [1, 2]
    assert [item["data"][0] for item in chart["series"]] == [
        [1, 1, 2, 3, 4, 5], [2, 100, 200, 300, 400, 500],
    ]
    assert all(item["encode"]["x"] == 0 for item in chart["series"])
    assert chart["legend"]["data"] == ["First", "Second"]


@pytest.mark.parametrize("plot_type,kde", [("Line", False), ("Histogram", False), ("Histogram", True), ("Boxplot", False)])
def test_all_combined_views_support_independent_log_invert_and_bounds(plot_type: str, kde: bool) -> None:
    chart = options(
        plot_type, density=kde,
        secondary_y_label="Second scale", secondary_x_label="Second values",
        log_x=False, invert_x=True, log_y=False, invert_y=False,
        secondary_log_x=True, secondary_invert_x=False,
        secondary_log_y=True, secondary_invert_y=True,
        x_limits=(0.5, 5.5), histogram_x_limits=(0.5, 5.5),
        secondary_x_limits=((0.5, 2.5) if plot_type == "Boxplot" else (0.5, 550)),
        y_limits=(0, 10), secondary_y_limits=(0.00001, 600),
    )
    assert [axis["type"] for axis in chart["xAxis"]] == ["value", "log"]
    assert [axis["type"] for axis in chart["yAxis"]] == ["value", "log"]
    assert [axis["inverse"] for axis in chart["xAxis"]] == [True, False]
    assert [axis["inverse"] for axis in chart["yAxis"]] == [False, True]
    assert [axis["position"] for axis in chart["xAxis"]] == ["bottom", "top"]
    assert [axis["position"] for axis in chart["yAxis"]] == ["left", "right"]
    assert [(item["xAxisIndex"], item["yAxisIndex"]) for item in chart["series"]] == [(0, 0), (1, 1)]
    assert [(axis["min"], axis["max"]) for axis in chart["yAxis"]] == [(0, 10), (0.00001, 600)]
    assert chart["grid"]["top"] >= chart["xAxis"][1]["nameGap"]


def test_combined_kde_contains_only_two_smooth_curves() -> None:
    chart = options("Histogram", density=True)
    assert [item["type"] for item in chart["series"]] == ["line", "line"]
    assert all(len(item["data"]) == 256 for item in chart["series"])
    assert chart["series"][0]["z"] > chart["series"][1]["z"]


def test_kde_bottom_statistics_leave_room_for_the_x_axis_label() -> None:
    chart = options("Histogram", density=True, show_histogram_statistics=True,
                    histogram_statistics_position="Bottom right")
    assert chart["grid"]["bottom"] >= 100
    assert "Mean" in chart["graphic"][0]["style"]["text"]


@pytest.mark.parametrize("samples", [[2], [2, 2, 2], [np.nan, 1, 2, 3, np.inf]])
def test_kde_is_finite_and_integrates_to_one_on_its_automatic_domain(samples: list) -> None:
    x, y = _kde_data(samples)
    assert len(x) == len(y) == 256
    assert np.isfinite(y).all() and np.all(y >= 0)
    assert np.trapezoid(y, x) == pytest.approx(1, abs=0.003)


def test_kde_filters_nonpositive_samples_for_log_x_and_bounds_large_inputs(monkeypatch) -> None:
    import scipy.stats

    original = scipy.stats.gaussian_kde
    sizes = []

    def tracked(samples, **kwargs):
        sizes.append(len(samples))
        return original(samples, **kwargs)

    monkeypatch.setattr(scipy.stats, "gaussian_kde", tracked)
    x, y = _kde_data(np.arange(-10, 50000), log_x=True)
    assert x[0] > 0 and len(x) == len(y) == 256
    assert sizes == [20000]
    with pytest.raises(ValueError, match="positive"):
        _kde_data([-2, 0, np.nan], log_x=True)
