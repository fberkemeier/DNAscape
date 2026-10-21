from __future__ import annotations

import asyncio
import base64
import io
import re
from typing import Any

import numpy as np
import pytest

from app.preview_plotting import (
    PreviewBand,
    PreviewChartHandle,
    PreviewLine,
    PreviewSettingsControls,
    approximate_histogram_bin_count,
    approximate_histogram_bin_size,
    automatic_y_limits,
    build_preview_plot_options,
    histogram_domain,
    nice_linear_limits,
    preview_settings_markers,
)


class _CaptureChart:
    def __init__(self, options: dict[str, Any], response: str) -> None:
        self.options = options
        self.response = response
        self.update_calls = 0
        self.method_calls: list[tuple[str, tuple[Any, ...], float]] = []

    def update(self) -> None:
        self.update_calls += 1

    async def run_chart_method(
        self,
        name: str,
        *args: Any,
        timeout: float = 1,
    ) -> str:
        self.method_calls.append((name, args, timeout))
        await asyncio.sleep(0)
        return self.response


def test_preview_setting_markers_cover_every_shared_control_consistently() -> None:
    expected = {name for name in PreviewSettingsControls.__dataclass_fields__ if not name.startswith("_")}.difference(
        {"primary_color_surface", "secondary_color_surface", "title"}
    )
    expected.add("container")

    mapping = preview_settings_markers(
        "mapping",
        overrides={"primary_color": "mapping-line-or-bar-color"},
    )
    simulation = preview_settings_markers("simulation")
    separate = preview_settings_markers(
        "plotting",
        qualifier="separate-2",
        overrides={"container": "plotting-settings"},
    )

    assert set(mapping) == set(simulation) == set(separate) == expected
    assert mapping["primary_color"] == "mapping-line-or-bar-color"
    assert simulation["show_legend"] == "simulation-show-legend"
    assert separate["container"] == "plotting-settings-separate-2"
    assert separate["histogram_bins"] == "plotting-histogram-bins-separate-2"


def test_preview_setting_markers_reject_unknown_control_names() -> None:
    with pytest.raises(KeyError, match="not_a_control"):
        preview_settings_markers(
            "mapping",
            overrides={"not_a_control": "mapping-unknown"},
        )


def test_vector_capture_is_reused_until_the_visible_chart_changes() -> None:
    async def exercise() -> None:
        nested_data = [[0.0, 1.0], [1.0, 2.0]]
        canvas = _CaptureChart(
            {"animation": False, "series": [{"type": "line", "data": nested_data}]},
            "canvas-data",
        )
        vector = _CaptureChart({"animation": False}, "svg-data")
        handle = PreviewChartHandle(canvas=canvas, vector=vector)

        first, second = await asyncio.gather(
            handle.run_chart_method("getDataURL", {"type": "svg"}, timeout=10),
            handle.run_chart_method("getDataURL", {"type": "svg"}, timeout=10),
        )

        assert first == second == "svg-data"
        assert vector.update_calls == 1
        assert len(vector.method_calls) == 1
        assert vector.options == canvas.options
        assert vector.options is not canvas.options
        assert vector.options["series"] is canvas.options["series"]

        await handle.run_chart_method(
            "getDataURL",
            {"type": "svg", "backgroundColor": "#ffffff"},
            timeout=10,
        )
        assert vector.update_calls == 1
        assert len(vector.method_calls) == 2

        canvas.options["series"][0]["data"] = [[0.0, 3.0], [1.0, 4.0]]
        handle.update()
        refreshed = await handle.run_chart_method(
            "getDataURL",
            {"type": "svg"},
            timeout=10,
        )

        assert refreshed == "svg-data"
        assert canvas.update_calls == 1
        assert vector.update_calls == 2
        assert len(vector.method_calls) == 3
        assert vector.options == canvas.options

        # A browser theme change can alter colours without changing the data
        # or Python options. It must also invalidate an already captured SVG.
        handle.invalidate_export()
        await handle.run_chart_method("getDataURL", {"type": "svg"}, timeout=10)
        assert canvas.update_calls == 1
        assert vector.update_calls == 3
        assert len(vector.method_calls) == 4

    asyncio.run(exercise())


def _histogram_options(
    values: object,
    **overrides: object,
) -> dict[str, Any]:
    settings: dict[str, object] = {
        "plot_type": "Histogram",
        "title": "Distribution",
        "x_label": "Value",
        "y_label": "Count",
        "color": "#2c7f77",
        "palette": "viridis",
        "histogram_bins": 4,
        "density": False,
        "show_legend": False,
        "show_colorbar": False,
        "log_y": False,
        "invert_y": False,
        "y_limits": None,
        "series_label": "Values",
        "values": values,
    }
    settings.update(overrides)
    return build_preview_plot_options(**settings)  # type: ignore[arg-type]


def test_histogram_bin_count_and_size_are_data_derived() -> None:
    values = np.asarray([0.0, 2.0, 6.0, 10.0])

    assert approximate_histogram_bin_size(values, 4) == 2.5
    assert approximate_histogram_bin_count(values, 2.0) == 5


def test_histogram_bin_helpers_ignore_nonfinite_values_and_bound_counts() -> None:
    values = np.asarray([np.nan, -np.inf, 0.0, 10.0, np.inf])

    assert approximate_histogram_bin_size(values, 4) == 2.5
    assert approximate_histogram_bin_count(values, 0.1) == 100
    assert approximate_histogram_bin_count([0.0, 10_000.0], 0.1) == 2_000
    assert approximate_histogram_bin_count(values, None) is None
    assert approximate_histogram_bin_count(values, 0.0) is None
    assert approximate_histogram_bin_size(values, None) is None


def test_constant_histogram_domains_match_numpy_and_stay_positive_on_log_x() -> None:
    linear = histogram_domain([4.0, 4.0])
    assert linear is not None
    np.testing.assert_allclose(linear[1], (3.5, 4.5))
    assert approximate_histogram_bin_size([4.0, 4.0], 5) == 0.2

    logarithmic = histogram_domain([-2.0, 0.0, 0.1, 0.1], log_x=True)
    assert logarithmic is not None
    assert logarithmic[0].tolist() == [0.1, 0.1]
    assert 0.0 < logarithmic[1][0] < 0.1 < logarithmic[1][1]


def test_log_histogram_synchronization_uses_only_positive_values() -> None:
    values = np.asarray([-100.0, 0.0, 1.0, 11.0, np.nan])

    assert approximate_histogram_bin_size(values, 5, log_x=True) == 2.0
    assert approximate_histogram_bin_count(values, 2.0, log_x=True) == 5
    assert approximate_histogram_bin_count([-2.0, 0.0], 1.0, log_x=True) is None


def test_histogram_x_limits_drive_exact_widths_and_bin_counts() -> None:
    values = np.asarray([0.0, 2.0, 4.0, 6.0, 8.0])

    assert approximate_histogram_bin_size(values, 4, x_limits=(2.0, 6.0)) == 1.0
    assert (
        approximate_histogram_bin_count(
            values,
            0.3,
            x_limits=(2.0, 6.0),
        )
        == 14
    )


def test_histogram_renderer_respects_custom_x_range_and_arbitrary_bin_size() -> None:
    options = _histogram_options(
        [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        histogram_bin_size=0.125,
        histogram_x_limits=(0.25, 0.75),
        show_histogram_statistics=True,
    )

    assert options["xAxis"]["min"] == pytest.approx(0.25)
    assert options["xAxis"]["max"] == pytest.approx(0.75)
    assert len(options["series"][0]["data"]) == 4
    assert options["graphic"][0]["style"]["text"] == (
        "Mean       0.5\nMedian    0.5\nStd dev   0.1"
    )


def test_histogram_statistics_are_optional_and_positioned_like_a_legend() -> None:
    assert _histogram_options([1.0, 2.0, 3.0, 4.0])["graphic"] == []

    options = _histogram_options(
        [1.0, 2.0, 3.0, 4.0],
        show_histogram_statistics=True,
        legend_position="Bottom left",
    )
    graphic = options["graphic"]
    assert isinstance(graphic, list)
    summary = graphic[0]
    assert summary["left"] == options["grid"]["left"] + 8
    assert summary["bottom"] == options["grid"]["bottom"] + 8
    assert "top" not in summary
    assert "right" not in summary
    assert summary["style"]["text"] == "Mean       2.5\nMedian    2.5\nStd dev   1.118"
    assert summary["z"] == 10_000
    assert summary["z2"] == 10_000
    assert summary["zlevel"] == 20


def test_shared_grid_keeps_axis_titles_inside_every_cartesian_plot() -> None:
    common = {
        "title": "Preview",
        "x_label": "A deliberately descriptive horizontal axis",
        "y_label": "A deliberately descriptive vertical axis",
        "color": "#2c7f77",
        "palette": "viridis",
        "histogram_bins": 4,
        "density": False,
        "show_legend": False,
        "show_colorbar": False,
        "log_y": False,
        "invert_y": False,
        "y_limits": None,
        "series_label": "Values",
    }
    options = [
        build_preview_plot_options(
            plot_type="Line",
            values=np.asarray([1.0, 2.0, 3.0]),
            x_axis=np.asarray([0.0, 1.0, 2.0]),
            **common,
        ),
        build_preview_plot_options(
            plot_type="Histogram",
            values=np.asarray([1.0, 2.0, 3.0]),
            **common,
        ),
        build_preview_plot_options(
            plot_type="Boxplot",
            values=np.asarray([1.0, 2.0, 3.0]),
            **common,
        ),
    ]

    for plot in options:
        # The complete gutter is calculated explicitly; containLabel would
        # reserve the tick width for a second time and create excess padding.
        assert plot["grid"]["containLabel"] is False
        assert plot["grid"]["left"] >= plot["yAxis"]["nameGap"]
        assert plot["grid"]["bottom"] > plot["xAxis"]["nameGap"]
        assert plot["xAxis"]["nameTextStyle"]["fontSize"] == 11
        assert plot["yAxis"]["nameTextStyle"]["fontSize"] == 11
        assert plot["yAxis"]["axisLabel"]["fontSize"] == 10
        assert plot["yAxis"]["axisLabel"]["margin"] == 7

    # These explicit defaults clear values left by ECharts' live option merge
    # when a numeric preview is replaced by the categorical boxplot view.
    boxplot = options[-1]
    assert boxplot["xAxis"]["axisTick"]["customValues"] is None
    assert boxplot["xAxis"]["axisLabel"]["customValues"] is None
    assert boxplot["xAxis"]["splitLine"]["showMinLine"] is True
    assert boxplot["xAxis"]["splitLine"]["showMaxLine"] is True


def test_heatmap_scale_is_compact_and_labels_its_vertical_bar_cleanly() -> None:
    options = build_preview_plot_options(
        plot_type="Heatmap",
        title="Kinetic field",
        x_label="Position (kb)",
        y_label="Time (min)",
        color="#2c7f77",
        palette="viridis",
        histogram_bins=4,
        density=False,
        show_legend=False,
        show_colorbar=True,
        log_y=False,
        invert_y=False,
        y_limits=None,
        series_label="Firing rate",
        values=np.asarray([[1.0, 2.0], [3.0, 4.0]]),
        x_axis=np.asarray([0.0, 1.0]),
        y_axis=np.asarray([0.0, 5.0]),
    )

    scale = options["visualMap"]
    assert scale["orient"] == "vertical"
    assert scale["text"] == ["4", "0"]
    assert scale["textStyle"]["align"] == "center"
    assert scale["textGap"] == 5
    assert options["grid"]["right"] - scale["right"] - scale["itemWidth"] <= 50
    assert options["graphic"][0]["rotation"] == pytest.approx(np.pi / 2)
    assert options["graphic"][0]["right"] < scale["right"]


def _heatmap_options(**overrides: Any) -> dict[str, Any]:
    settings = {
        "plot_type": "Heatmap",
        "title": "Replicated fraction",
        "x_label": "Position (kb)",
        "y_label": "Time (min)",
        "color": "#2c7f77",
        "palette": "viridis",
        "histogram_bins": 20,
        "density": False,
        "show_legend": False,
        "show_colorbar": False,
        "log_y": False,
        "invert_y": False,
        "y_limits": None,
        "series_label": "Replicated fraction",
        "values": np.asarray([[0.0, 0.2, 0.1], [1.0, 1.0, 1.0]]),
        "x_axis": np.asarray([0.0, 5_000.0, 9_999.0]),
        "y_axis": np.asarray([0.0, 1_000.0]),
    }
    return build_preview_plot_options(**{**settings, **overrides})


@pytest.mark.parametrize("invert", (False, True))
@pytest.mark.parametrize("logarithmic", (False, True))
def test_heatmap_automatic_axes_meet_the_raster_edges_without_margins(
    invert: bool,
    logarithmic: bool,
) -> None:
    x_axis = np.asarray([10.0, 1_000.0, 9_999.0])
    y_axis = np.asarray([10.0, 1_000.0])
    limits = automatic_y_limits(
        [],
        plot_type="Heatmap",
        histogram_bins=20,
        density=False,
        log_y=logarithmic,
        y_axis=y_axis,
    )
    for y_limits in (None, limits):
        options = _heatmap_options(
            x_axis=x_axis,
            y_axis=y_axis,
            log_x=logarithmic,
            log_y=logarithmic,
            invert_x=invert,
            invert_y=invert,
            y_limits=y_limits,
        )
        assert [options["xAxis"]["min"], options["xAxis"]["max"]] == [10.0, 9_999.0]
        assert [options["yAxis"]["min"], options["yAxis"]["max"]] == [10.0, 1_000.0]
        assert options["series"][0]["data"] == [[10.0, 10.0, 9_999.0, 1_000.0]]


@pytest.mark.parametrize(
    "requested",
    (
        (-100.0, 1_100.0),
        (100.0, 900.0),
        (-100.0, 900.0),
        (100.0, 1_100.0),
    ),
)
def test_heatmap_explicit_limits_crop_without_rescaling_the_raster(
    requested: tuple[float, float],
) -> None:
    options = _heatmap_options(y_limits=requested)
    assert (options["yAxis"]["min"], options["yAxis"]["max"]) == requested
    # The raster and hover coordinates retain their original physical extent;
    # ECharts clips it at the selected domain rather than stretching rows.
    assert options["series"][0]["data"] == [[0.0, 0.0, 9_999.0, 1_000.0]]
    assert options["series"][0]["clip"] is True
    assert max(cell[6] for cell in options["series"][1]["data"]) == 1_000.0


@pytest.mark.parametrize("limits", ((1_100.0, 1_200.0), (-200.0, -100.0), (1_000.0, 1_100.0)))
def test_heatmap_limits_outside_the_data_preserve_the_requested_view(limits: tuple[float, float]) -> None:
    options = _heatmap_options(y_limits=limits)
    assert (options["yAxis"]["min"], options["yAxis"]["max"]) == limits
    assert options["series"][0]["data"] == [[0.0, 0.0, 9_999.0, 1_000.0]]
    assert options["series"][0]["clip"] is True


def test_heatmap_downsampling_keeps_full_original_axis_extents_and_missing_values() -> None:
    from PIL import Image

    values = np.ones((401, 1_001), dtype=float)
    values[:20, :20] = np.nan
    x_axis = np.linspace(10.0, 10_010.0, values.shape[1])
    y_axis = np.linspace(10.0, 1_000.0, values.shape[0])
    options = _heatmap_options(values=values, x_axis=x_axis, y_axis=y_axis)
    assert options["series"][0]["data"] == [[10.0, 10.0, 10_010.0, 1_000.0]]
    assert options["xAxis"]["max"] == 10_010.0
    assert options["yAxis"]["max"] == 1_000.0
    payload = re.search(r"data:image/png;base64,([A-Za-z0-9+/=]+)", options["series"][0][":renderItem"])
    assert payload is not None
    raster = np.asarray(Image.open(io.BytesIO(base64.b64decode(payload[1]))))
    assert raster.shape[0] < values.shape[0]
    assert raster.shape[1] < values.shape[1]
    assert np.all(raster[0, :, 3] == 255)
    assert np.all(raster[:, -1, 3] == 255)
    assert raster[-1, 0, 3] == 0  # genuine NaNs remain transparent, including at an edge


@pytest.mark.parametrize("log_z", (False, True))
def test_heatmap_sampling_averages_each_block_without_losing_missing_cells(log_z: bool) -> None:
    from dnascape.workflows.preview import _sample_heatmap

    values = np.arange(1.0, 64.0).reshape(7, 9)
    values[:3, :4] = np.nan
    values[3, 4] = np.inf
    values[4, 5] = 0.0
    values[5, 6] = -1.0
    x_axis, y_axis = np.arange(9.0), np.arange(7.0)
    x, y, sampled, _, _ = _sample_heatmap(values, x_axis, y_axis, limit=4, log_z=log_z)
    assert sampled.shape == (1, 2)
    edges = np.linspace(0, 9, 3, dtype=int)
    for column, (lower, upper) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        block = values[:, lower:upper]
        finite = block[np.isfinite(block) & ((block > 0) if log_z else True)]
        expected = np.mean(np.log10(finite) if log_z else finite)
        assert sampled[0, column] == pytest.approx(expected)
    np.testing.assert_array_equal(x, x_axis)
    np.testing.assert_array_equal(y, y_axis)


@pytest.mark.parametrize("shape", ((1000, 1), (1, 1000), (301, 721)))
def test_heatmap_sampling_handles_narrow_inputs_with_a_one_cell_budget(shape: tuple[int, int]) -> None:
    from dnascape.workflows.preview import _sample_heatmap

    values = np.ones(shape)
    _, _, sampled, _, _ = _sample_heatmap(values, np.arange(shape[1]), np.arange(shape[0]), limit=1)
    assert sampled.shape == (1, 1)
    assert sampled[0, 0] == 1.0


@pytest.mark.parametrize("log_z", (False, True))
def test_heatmap_sampling_has_bounded_working_memory_for_large_views(log_z: bool) -> None:
    import tracemalloc
    from dnascape.workflows.preview import _sample_heatmap, MAX_HEATMAP_CELLS

    # A temporal profile broadcast across a long chromosome represents 160 MB
    # of heatmap data without allocating it. Sampling must preserve that saving.
    values = np.broadcast_to(np.linspace(1.0, 10.0, 4000)[:, None], (4000, 5000))
    tracemalloc.start()
    try:
        _, _, sampled, _, _ = _sample_heatmap(values, np.arange(5000), np.arange(4000), log_z=log_z)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert sampled.size <= MAX_HEATMAP_CELLS
    assert peak < 20_000_000
    assert np.all(np.isfinite(sampled))


def test_heatmap_sampling_averages_cells_across_work_tile_boundaries() -> None:
    from dnascape.workflows.preview import _sample_heatmap, MAX_HEATMAP_WORK_CELLS

    width = MAX_HEATMAP_WORK_CELLS + 1000
    values = np.arange(width, dtype=np.float32)[None, :]
    _, _, sampled, _, _ = _sample_heatmap(values, np.arange(width), [0.0])
    edges = np.linspace(0, width, sampled.shape[1] + 1, dtype=int)
    expected = (edges[:-1] + edges[1:] - 1) / 2.0
    np.testing.assert_allclose(sampled[0], expected)


def test_heatmap_custom_xyz_ranges_preserve_data_coordinates_and_clip_colours() -> None:
    options = _heatmap_options(x_limits=(-1000.0, 12_000.0), y_limits=(-20.0, 1200.0), z_limits=(0.2, 0.8))
    assert (options["xAxis"]["min"], options["xAxis"]["max"]) == (-1000.0, 12_000.0)
    assert (options["yAxis"]["min"], options["yAxis"]["max"]) == (-20.0, 1200.0)
    assert (options["visualMap"]["min"], options["visualMap"]["max"]) == (0.2, 0.8)
    assert options["series"][0]["data"] == [[0.0, 0.0, 9999.0, 1000.0]]


def test_heatmap_log_z_limits_are_entered_as_original_values() -> None:
    options = _heatmap_options(log_z=True, z_limits=(0.01, 100.0))
    assert (options["visualMap"]["min"], options["visualMap"]["max"]) == (-2.0, 2.0)
    with pytest.raises(ValueError, match="Log Z"):
        _heatmap_options(log_z=True, z_limits=(0.0, 100.0))


def test_heatmap_narrow_nonzero_domains_are_not_expanded_as_singletons() -> None:
    x_axis = np.asarray([1_000_000.0, 1_000_000.00001, 1_000_000.00002])
    y_axis = np.asarray([500.0, 500.00000001])
    options = _heatmap_options(x_axis=x_axis, y_axis=y_axis)
    assert (options["xAxis"]["min"], options["xAxis"]["max"]) == (x_axis[0], x_axis[-1])
    assert (options["yAxis"]["min"], options["yAxis"]["max"]) == (y_axis[0], y_axis[-1])


def test_singleton_heatmap_axes_share_the_same_nonzero_image_and_display_extent() -> None:
    options = _heatmap_options(values=np.asarray([[0.5]]), x_axis=[10.0], y_axis=[100.0])
    assert options["xAxis"]["min"] < options["xAxis"]["max"]
    assert options["yAxis"]["min"] < options["yAxis"]["max"]
    assert options["series"][0]["data"] == [[
        options["xAxis"]["min"], options["yAxis"]["min"],
        options["xAxis"]["max"], options["yAxis"]["max"],
    ]]


def test_log_histogram_statistics_match_the_values_that_are_actually_plotted() -> None:
    options = _histogram_options(
        [-100.0, 0.0, 1.0, 3.0, np.nan],
        log_x=True,
        show_histogram_statistics=True,
    )

    assert options["graphic"][0]["style"]["text"] == ("Mean       2\nMedian    2\nStd dev   1")


def test_histogram_boundary_mode_moves_exact_internal_edges_without_losing_endpoints() -> None:
    settings = {
        "histogram_bins": 2,
        "histogram_x_limits": (0.0, 2.0),
    }

    left = _histogram_options([0.0, 1.0, 2.0], **settings)
    right = _histogram_options(
        [0.0, 1.0, 2.0],
        histogram_boundary="right",
        **settings,
    )

    assert [item[2] for item in left["series"][0]["data"]] == [1.0, 2.0]
    assert [item[2] for item in right["series"][0]["data"]] == [2.0, 1.0]
    assert sum(item[2] for item in left["series"][0]["data"]) == 3.0
    assert sum(item[2] for item in right["series"][0]["data"]) == 3.0


@pytest.mark.parametrize("boundary", ("left", "right"))
def test_histogram_kde_uses_scott_bandwidth_instead_of_density_bars(boundary: str) -> None:
    from scipy.stats import gaussian_kde

    samples = [0.0, 0.5, 1.0, 1.5, 2.0]
    options = _histogram_options(
        samples,
        histogram_bins=2,
        histogram_x_limits=(0.0, 2.0),
        histogram_boundary=boundary,
        density=True,
    )

    assert len(options["series"]) == 1
    assert options["series"][0]["type"] == "line"
    data = np.asarray(options["series"][0]["data"])
    assert data.shape == (256, 2)
    assert (data[0, 0], data[-1, 0]) == (0.0, 2.0)
    np.testing.assert_allclose(data[:, 1], gaussian_kde(samples, bw_method="scott")(data[:, 0]))


def test_histogram_custom_rectangles_preserve_exact_edges_and_spacing_choice() -> None:
    touching = _histogram_options(
        [0.0, 0.4, 0.8, 1.0],
        histogram_bin_size=0.4,
        histogram_x_limits=(0.0, 1.0),
        histogram_touching=True,
    )
    spaced = _histogram_options(
        [0.0, 0.4, 0.8, 1.0],
        histogram_bin_size=0.4,
        histogram_x_limits=(0.0, 1.0),
        histogram_touching=False,
    )

    touching_series = touching["series"][0]
    assert touching_series["type"] == "custom"
    assert [item[:2] for item in touching_series["data"]] == [
        [0.0, 0.4],
        [0.4, 0.8],
        [0.8, 1.0],
    ]
    assert "* 0" in touching_series[":renderItem"]
    assert "* 0.14" in spaced["series"][0][":renderItem"]


def test_default_line_domains_use_exact_dynamic_padding_not_rounded_ticks() -> None:
    options = build_preview_plot_options(
        plot_type="Line",
        title="Measured profile",
        x_label="Position",
        y_label="Signal",
        color="#2c7f77",
        palette="viridis",
        histogram_bins=20,
        density=False,
        show_legend=False,
        show_colorbar=False,
        log_y=False,
        invert_y=False,
        y_limits=None,
        series_label="Measured",
        values=np.asarray([11.2, 12.5, 14.8]),
        x_axis=np.asarray([2.3, 5.1, 9.7]),
    )

    assert options["xAxis"]["min"] == pytest.approx(2.3 - 7.4 * 0.04)
    assert options["xAxis"]["max"] == pytest.approx(9.7 + 7.4 * 0.04)
    assert options["yAxis"]["min"] == pytest.approx(11.2 - 3.6 * 0.04)
    assert options["yAxis"]["max"] == pytest.approx(14.8 + 3.6 * 0.04)
    assert options["xAxis"]["min"] != pytest.approx(2.0)
    assert options["yAxis"]["max"] != pytest.approx(15.0)
    assert options["xAxis"]["axisLabel"]["customValues"] == [4.0, 6.0, 8.0]
    assert options["yAxis"]["axisLabel"]["customValues"] == [12.0, 13.0, 14.0]
    assert options["xAxis"]["axisLabel"]["showMinLabel"] is True
    assert options["xAxis"]["axisLabel"]["showMaxLabel"] is True
    assert options["yAxis"]["axisLabel"]["showMinLabel"] is True
    assert options["yAxis"]["axisLabel"]["showMaxLabel"] is True
    assert options["xAxis"]["splitLine"]["showMinLine"] is True
    assert options["xAxis"]["splitLine"]["showMaxLine"] is True
    assert (
        options["xAxis"]["splitLine"]["customValues"]
        == options["xAxis"]["axisTick"]["customValues"]
    )
    assert 2.3 not in options["xAxis"]["axisTick"]["customValues"]
    assert 9.7 not in options["xAxis"]["axisTick"]["customValues"]


def test_dynamic_axis_margin_scales_with_the_data_span() -> None:
    small = nice_linear_limits(10.0, 20.0)
    large = nice_linear_limits(100.0, 200.0)

    assert 10.0 - small[0] == pytest.approx(0.4)
    assert small[1] - 20.0 == pytest.approx(0.4)
    assert 100.0 - large[0] == pytest.approx(4.0)
    assert large[1] - 200.0 == pytest.approx(4.0)


def test_real_zero_endpoints_remain_labeled_while_only_outer_margin_is_hidden() -> None:
    options = build_preview_plot_options(
        plot_type="Line",
        title="Zero based",
        x_label="Position",
        y_label="Signal",
        color="#2c7f77",
        palette="viridis",
        histogram_bins=20,
        density=False,
        show_legend=False,
        show_colorbar=False,
        log_y=False,
        invert_y=False,
        y_limits=None,
        series_label="Signal",
        values=np.asarray([0.0, 0.5, 1.0]),
        x_axis=np.asarray([0.0, 5.0, 10.0]),
    )

    assert options["xAxis"]["min"] == 0.0
    assert options["yAxis"]["min"] == 0.0
    assert options["xAxis"]["axisLabel"]["showMinLabel"] is True
    assert options["yAxis"]["axisLabel"]["showMinLabel"] is True
    assert options["xAxis"]["axisLabel"]["showMaxLabel"] is True
    assert options["yAxis"]["axisLabel"]["showMaxLabel"] is True


def test_line_grid_keeps_real_endpoint_ticks_without_gridding_padded_limits() -> None:
    options = build_preview_plot_options(
        plot_type="Line",
        title="Replication timing",
        x_label="Position (kb)",
        y_label="Replication timing",
        color="#2c7f77",
        palette="viridis",
        histogram_bins=20,
        density=False,
        show_legend=True,
        show_colorbar=False,
        log_y=False,
        invert_y=False,
        y_limits=None,
        series_label="Replication timing",
        values=np.asarray([18.0, 35.0, 23.0]),
        x_axis=np.asarray([0.0, 500.0, 1_000.0]),
    )

    # The exact data range plus dynamic whitespace remains untouched.
    assert options["xAxis"]["min"] == 0.0
    assert options["xAxis"]["max"] == pytest.approx(1_040.0)
    assert options["yAxis"]["min"] == pytest.approx(17.32)
    assert options["yAxis"]["max"] == pytest.approx(35.68)

    # Labels and ticks are globally aligned rather than offset by the padded
    # minimum. Real endpoint ticks retain their grid lines; the padded 1,040
    # boundary is absent from customValues and therefore has no grid line.
    assert options["xAxis"]["axisTick"]["customValues"] == [
        0.0,
        200.0,
        400.0,
        600.0,
        800.0,
        1_000.0,
    ]
    assert (
        options["xAxis"]["axisLabel"]["customValues"]
        == options["xAxis"]["axisTick"]["customValues"]
    )
    assert options["xAxis"]["splitLine"]["showMinLine"] is True
    assert options["xAxis"]["splitLine"]["showMaxLine"] is True
    assert options["yAxis"]["axisTick"]["customValues"] == [20.0, 25.0, 30.0, 35.0]
    assert options["yAxis"]["splitLine"]["showMinLine"] is True
    assert options["yAxis"]["splitLine"]["showMaxLine"] is True


def test_histogram_grid_keeps_aligned_maximum_tick_but_not_padded_limit() -> None:
    options = build_preview_plot_options(
        plot_type="Histogram",
        title="Origins",
        x_label="Origins",
        y_label="Count",
        color="#2c7f77",
        palette="viridis",
        histogram_bins=5,
        density=False,
        show_legend=False,
        show_colorbar=False,
        log_y=False,
        invert_y=False,
        y_limits=None,
        series_label="Origins",
        values=np.r_[np.repeat(1.0, 12), 2.0, 3.0, 4.0, 5.0],
    )

    assert options["yAxis"]["max"] == pytest.approx(12.48)
    assert options["yAxis"]["axisTick"]["customValues"] == [
        0.0,
        2.0,
        4.0,
        6.0,
        8.0,
        10.0,
        12.0,
    ]
    assert 12.48 not in options["yAxis"]["axisTick"]["customValues"]
    assert (
        options["yAxis"]["splitLine"]["customValues"]
        == options["yAxis"]["axisTick"]["customValues"]
    )
    assert options["yAxis"]["splitLine"]["showMaxLine"] is True


def test_explicit_and_fixed_scientific_y_domains_are_not_padded_or_hidden() -> None:
    common = {
        "plot_type": "Line",
        "title": "Directionality",
        "x_label": "Position",
        "y_label": "Fork directionality",
        "color": "#2c7f77",
        "palette": "viridis",
        "histogram_bins": 20,
        "density": False,
        "show_legend": False,
        "show_colorbar": False,
        "log_y": False,
        "invert_y": False,
        "series_label": "Directionality",
        "values": np.asarray([-0.35, 0.1, 0.72]),
        "x_axis": np.asarray([0.0, 1.0, 2.0]),
    }
    fixed = build_preview_plot_options(y_limits=(-1.0, 1.0), **common)
    custom = build_preview_plot_options(y_limits=(-0.5, 0.8), **common)

    assert (fixed["yAxis"]["min"], fixed["yAxis"]["max"]) == (-1.0, 1.0)
    assert fixed["yAxis"]["axisLabel"]["showMinLabel"] is True
    assert fixed["yAxis"]["axisLabel"]["showMaxLabel"] is True
    assert (custom["yAxis"]["min"], custom["yAxis"]["max"]) == (-0.5, 0.8)
    assert custom["yAxis"]["axisLabel"]["showMinLabel"] is True


def test_y_axis_name_gap_tracks_tick_label_width_without_excess_spacing() -> None:
    common = {
        "plot_type": "Line",
        "title": "Signal",
        "x_label": "Position",
        "y_label": "Signal",
        "color": "#2c7f77",
        "palette": "viridis",
        "histogram_bins": 20,
        "density": False,
        "show_legend": False,
        "show_colorbar": False,
        "log_y": False,
        "invert_y": False,
        "y_limits": None,
        "series_label": "Signal",
        "x_axis": np.asarray([0.0, 1.0, 2.0]),
    }
    compact = build_preview_plot_options(values=np.asarray([1.0, 2.0, 3.0]), **common)
    scientific = build_preview_plot_options(
        values=np.asarray([1.2e-7, 5.0e-7, 9.8e-7]),
        **common,
    )

    assert compact["yAxis"]["nameGap"] < 64
    assert scientific["yAxis"]["nameGap"] > compact["yAxis"]["nameGap"]
    assert compact["grid"]["left"] >= compact["yAxis"]["nameGap"]
    assert scientific["grid"]["left"] >= scientific["yAxis"]["nameGap"]


def test_line_layout_uses_actual_aligned_ticks_without_double_reserving_space() -> None:
    """Match the scale and compact left gutter of the reported timing plot."""

    options = build_preview_plot_options(
        plot_type="Line",
        title="Replication timing",
        x_label="Position (kb)",
        y_label="Replication timing",
        color="#2c7f77",
        palette="viridis",
        histogram_bins=20,
        density=False,
        show_legend=True,
        show_colorbar=False,
        log_y=False,
        invert_y=False,
        y_limits=None,
        series_label="Replication timing",
        values=np.asarray([18.0, 35.0, 23.0]),
        x_axis=np.asarray([0.0, 500.0, 1_000.0]),
    )

    # The domain keeps its exact margin, while explicit clean ticks avoid both
    # the old 22.32-style labels and their unnecessary extra left padding.
    assert options["yAxis"]["min"] == pytest.approx(17.32)
    assert options["yAxis"]["axisLabel"]["customValues"] == [20.0, 25.0, 30.0, 35.0]
    assert options["yAxis"]["nameGap"] <= 30
    assert options["grid"]["left"] >= options["yAxis"]["nameGap"] + 9
    assert options["grid"]["left"] <= 47
    assert options["grid"]["containLabel"] is False
    assert options["grid"]["outerBoundsMode"] == "none"


def test_five_character_y_ticks_receive_extra_title_clearance() -> None:
    options = build_preview_plot_options(
        plot_type="Line",
        title="Small signal",
        x_label="Position",
        y_label="Signal",
        color="#2c7f77",
        palette="viridis",
        histogram_bins=20,
        density=False,
        show_legend=False,
        show_colorbar=False,
        log_y=False,
        invert_y=False,
        y_limits=None,
        series_label="Signal",
        values=np.asarray([0.001, 0.009, 0.004]),
        x_axis=np.asarray([0.0, 1.0, 2.0]),
    )

    assert "0.002" in [f"{value:.5g}" for value in options["yAxis"]["axisLabel"]["customValues"]]
    assert 42 <= options["yAxis"]["nameGap"] < 48
    assert options["grid"]["left"] >= options["yAxis"]["nameGap"] + 9
    assert options["grid"]["left"] <= 64


@pytest.mark.parametrize("side", ("left", "right"))
def test_bottom_line_legend_stays_inside_axes_without_changing_plot_margins(side: str) -> None:
    common = {
        "plot_type": "Line",
        "title": "Curves",
        "x_label": "Position",
        "y_label": "Signal",
        "color": "#2c7f77",
        "palette": "viridis",
        "histogram_bins": 20,
        "density": False,
        "show_legend": True,
        "show_colorbar": False,
        "log_y": False,
        "invert_y": False,
        "y_limits": None,
        "series_label": "Signal",
        "values": np.asarray([1.0, 2.0, 3.0]),
        "x_axis": np.asarray([0.0, 1.0, 2.0]),
    }
    top = build_preview_plot_options(legend_position="Top right", **common)
    bottom = build_preview_plot_options(legend_position=f"Bottom {side}", **common)

    assert bottom["legend"]["bottom"] > bottom["grid"]["bottom"]
    assert "top" not in bottom["legend"]
    assert side in bottom["legend"]
    assert bottom["legend"]["orient"] == top["legend"]["orient"] == "vertical"
    assert bottom["grid"] == top["grid"]


def test_every_interactive_plot_uses_two_decimal_tooltip_formatting() -> None:
    common = {
        "title": "Preview",
        "x_label": "X",
        "y_label": "Y",
        "color": "#2c7f77",
        "palette": "viridis",
        "histogram_bins": 4,
        "density": False,
        "show_legend": False,
        "show_colorbar": False,
        "log_y": False,
        "invert_y": False,
        "y_limits": None,
        "series_label": "Values",
        "interactive": True,
    }
    plots = (
        build_preview_plot_options(
            plot_type="Line",
            values=np.asarray([1.234, 2.345]),
            x_axis=np.asarray([0.123, 1.234]),
            **common,
        ),
        build_preview_plot_options(
            plot_type="Histogram",
            values=np.asarray([1.234, 2.345, 3.456]),
            **common,
        ),
        build_preview_plot_options(
            plot_type="Boxplot",
            values=np.asarray([1.234, 2.345, 3.456]),
            **common,
        ),
        build_preview_plot_options(
            plot_type="Heatmap",
            values=np.asarray([[1.234, 2.345], [3.456, 4.567]]),
            x_axis=np.asarray([0.123, 1.234]),
            y_axis=np.asarray([5.678, 6.789]),
            **common,
        ),
    )

    for plot in plots:
        assert "toFixed(2)" in plot["tooltip"][":valueFormatter"]
        assert "toFixed(2)" in plot["tooltip"]["axisPointer"]["label"][":formatter"]
    assert len(plots[-1]["series"]) == 2
    assert len(plots[-1]["series"][1]["data"]) <= 2_400


def test_dense_curve_overlays_disable_hover_but_keep_legend_selection() -> None:
    axis = np.asarray([0.0, 1.0, 2.0])

    def options(curve_count: int) -> dict[str, Any]:
        return build_preview_plot_options(
            plot_type="Line",
            title="Profiles",
            x_label="Position",
            y_label="Value",
            color="#2c7f77",
            palette="viridis",
            histogram_bins=20,
            density=False,
            show_legend=True,
            show_colorbar=False,
            log_y=False,
            invert_y=False,
            y_limits=None,
            series_label="Profiles",
            lines=tuple(
                PreviewLine(
                    f"Curve {index + 1}",
                    axis,
                    axis + index,
                    "#2c7f77",
                )
                for index in range(curve_count)
            ),
        )

    six = options(6)
    seven = options(7)

    assert six["tooltip"]["show"] is True
    assert all(item["silent"] is False for item in six["series"])
    assert seven["tooltip"]["show"] is False
    assert seven["tooltip"]["axisPointer"]["label"]["show"] is False
    assert all(item["silent"] is True for item in seven["series"])
    assert all(item["emphasis"]["disabled"] is True for item in seven["series"])
    assert seven["legend"]["selectedMode"] is True


def test_line_renderer_supports_grouped_individual_curves_and_a_filled_band() -> None:
    axis = np.asarray([0.0, 1.0, 2.0])
    options = build_preview_plot_options(
        plot_type="Line",
        title="Replication timing",
        x_label="Position (kb)",
        y_label="Replication timing",
        color="#2c7f77",
        palette="viridis",
        histogram_bins=20,
        density=False,
        show_legend=True,
        show_colorbar=False,
        log_y=False,
        invert_y=False,
        y_limits=None,
        series_label="Mean",
        lines=(
            PreviewLine(
                "Individual simulations",
                axis,
                np.asarray([1.0, 2.0, 3.0]),
                "#2c7f77",
                opacity=0.2,
                width=0.8,
            ),
            PreviewLine("Mean", axis, np.asarray([2.0, 3.0, 4.0]), "#2c7f77"),
        ),
        bands=(
            PreviewBand(
                "Mean ± SD",
                axis,
                np.asarray([1.5, 2.5, 3.5]),
                np.asarray([2.5, 3.5, 4.5]),
                "#2c7f77",
            ),
        ),
    )

    assert options["series"][0]["type"] == "custom"
    assert options["series"][0]["name"] == "Mean ± SD"
    assert options["series"][1]["lineStyle"]["opacity"] == pytest.approx(0.2)
    assert options["series"][1]["lineStyle"]["width"] == pytest.approx(0.8)
    assert options["legend"]["data"] == [
        "Individual simulations",
        "Mean",
        "Mean ± SD",
    ]
