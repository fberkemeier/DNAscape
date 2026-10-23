"""Histogram controls, selectable guides, and detail refinement."""

import asyncio
import base64
import gzip
import io
import re
from dataclasses import replace

import numpy as np
import pytest
from nicegui.elements.upload import Upload
from nicegui.testing import user_simulation
from PIL import Image

from app.main import index
from app.pages.data_management import _converted_preview_data
from app.preview_plotting import PreviewBand, PreviewChartHandle, PreviewLine
from app.simulation_exports import ExportArtifact
from app.tests.test_plot_navigation import CaptureChart, navigation_options
from dnascape.workflows.conversion import DataConversionResult, DataPreview
from dnascape.workflows.preview import _kde_data


def test_kde_smoothness_changes_bandwidth_including_constant_data():
    values = np.r_[np.linspace(-0.2, 0.2, 100), np.linspace(1.8, 2.2, 100)]
    x, narrow = _kde_data(values, x_limits=(-3, 5), bandwidth_scale=0.4)
    _, smooth = _kde_data(values, x_limits=(-3, 5), bandwidth_scale=2)
    assert narrow.max() > smooth.max() * 2
    assert np.trapezoid(smooth, x) == pytest.approx(1, abs=0.002)
    _, constant_narrow = _kde_data([2, 2], x_limits=(0, 4), bandwidth_scale=0.5)
    _, constant_smooth = _kde_data([2, 2], x_limits=(0, 4), bandwidth_scale=2)
    assert constant_narrow.max() > constant_smooth.max() * 3
    with pytest.raises(ValueError, match="KDE smoothness"):
        _kde_data(values, bandwidth_scale=0)
    x, detail = _kde_data([2, 2], x_limits=(0, 10), evaluation_limits=(1.9, 2.1))
    expected = np.exp(-0.5 * ((x - 2) / 0.5) ** 2) / (0.5 * np.sqrt(2 * np.pi))
    np.testing.assert_allclose(detail, expected)  # Zoom must preserve the bandwidth.


def test_combined_histograms_have_one_selectable_panel_and_shared_bin_slots():
    lines = (PreviewLine("First", np.arange(5.0), np.arange(1.0, 6.0), "#b03412"),
             PreviewLine("Second", np.arange(5.0), np.arange(6.0, 11.0), "#2596be"))
    options = navigation_options("Histogram", lines=lines, histogram_x_limits=(1, 10),
                                 show_histogram_statistics=True, show_legend=True, histogram_grouped=True)
    assert options["legend"]["show"] is False
    assert len(options["graphic"]) == 1
    assert len(options["dnascapeHistogramGuides"]["choices"]) == 2
    assert "First" in options["graphic"][0]["style"]["text"]
    assert "Mean       3" in options["graphic"][0]["style"]["text"]
    assert "Mean       8" in options["dnascapeHistogramGuides"]["choices"][1]["graphic"]["style"]["text"]
    assert "*0/2" in options["series"][0][":renderItem"]
    assert "*1/2" in options["series"][1][":renderItem"]
    overlay = navigation_options("Histogram", lines=lines, histogram_x_limits=(1, 10))
    for index in range(2):
        assert options["series"][index]["data"] == overlay["series"][index]["data"]
    handle = PreviewChartHandle(CaptureChart(options), CaptureChart({}))
    handle._apply_navigation()
    handle.select_histogram_dataset({"index": 1})
    assert "Second" in options["graphic"][0]["style"]["text"]


def test_line_zoom_refines_from_native_points_and_can_restore_full_view():
    axis = np.arange(250_000.0)
    values = np.sin(axis / 100)
    values[12_031] = 50
    options = navigation_options(values=values, x_axis=axis)
    initial = list(options["series"][0]["data"])
    assert not any(point[0] == 12_031 for point in initial)
    handle = PreviewChartHandle(CaptureChart(options), CaptureChart({}))
    handle._apply_navigation()
    handle.accept_navigation({"context": handle._navigation.context, "ranges": {"x": [12000, 12080]}})
    detail = options["series"][0]["data"]
    assert [12031.0, 50.0] in detail
    assert len(initial) == 1000
    visible = [row for row in detail if 12000 <= row[0] <= 12080]
    assert len(visible) <= 1000
    assert len(visible) > sum(12000 <= row[0] <= 12080 for row in initial)
    assert detail[0][0] == axis[0] and detail[-1][0] == axis[-1]
    assert len(detail) <= 2000
    assert list(detail.refine(None, None)["data"]) == initial


@pytest.mark.parametrize("descending", [False, True])
def test_overview_detail_samples_remain_bounded_and_preserve_native_nan_gaps(descending):
    axis = np.arange(100_001.)
    values = np.sin(axis / 50)
    values[12340:12350] = np.nan
    if descending:
        axis, values = axis[::-1], values[::-1]
    options = navigation_options(values=values, x_axis=axis)
    handle = PreviewChartHandle(CaptureChart(options), CaptureChart({}))
    handle._apply_navigation()
    for lower, upper in ((12330, 12360), (90000, 90500), (0, 50000), (20, 25)):
        handle.accept_navigation(dict(context=handle._navigation.context, ranges={"x": (lower, upper)}))
        data = options["series"][0]["data"]
        visible = [row for row in data if lower <= row[0] <= upper]
        assert len(visible) <= 1000 and len(data) <= 2000
        assert data[0][0] == axis[0] and data[-1][0] == axis[-1]
        if lower == 12330:
            assert [row[0] for row in visible if row[1] is None] == list(np.arange(12340, 12350)[::-1 if descending else 1])


def test_uncertainty_band_keeps_a_coarse_outer_profile_and_a_native_inner_gap():
    axis = np.arange(100_001.)
    lower, upper = np.full(axis.size, 3.), np.full(axis.size, 5.)
    lower[12340:12350] = np.nan
    options = navigation_options(values=np.full(axis.size, 4.), x_axis=axis,
                                 bands=(PreviewBand("SD", axis, lower, upper, "#b03412"),))
    handle = PreviewChartHandle(CaptureChart(options), CaptureChart({}))
    handle._apply_navigation()
    handle.accept_navigation(dict(context=handle._navigation.context, ranges={"x": (12330, 12360)}))
    polygons = options["series"][0]["data"]
    assert len(polygons) < 2000
    assert polygons[0][0] == 0 and polygons[-1][3] == 100000
    assert not any(row[0] < 12350 and row[3] > 12339 for row in polygons)


def test_kde_refinement_retains_its_full_support_without_adding_legend_entries():
    options = navigation_options("Histogram", density=True, values=np.linspace(1, 10, 1001))
    initial = list(options["series"][0]["data"])
    legend = list(options["legend"]["data"])
    handle = PreviewChartHandle(CaptureChart(options), CaptureChart({}))
    handle._apply_navigation()
    handle.accept_navigation(dict(context=handle._navigation.context, ranges={"x": (4.2, 4.4)}))
    data = options["series"][0]["data"]
    assert data[0] == initial[0] and data[-1] == initial[-1]
    assert len([row for row in data if 4.2 <= row[0] <= 4.4]) == 1000
    assert options["legend"]["data"] == legend
    assert len(options["series"]) == 1


def test_heatmap_zoom_reveals_native_cells_without_changing_colour_scale():
    values = np.zeros((601, 1201))
    values[300, 600] = 1
    options = navigation_options("Heatmap", values=values, x_axis=np.arange(1201.0), y_axis=np.arange(601.0))
    original = options["series"][0][":renderItem"]
    color_scale = dict(options["visualMap"])
    handle = PreviewChartHandle(CaptureChart(options), CaptureChart({}))
    handle._apply_navigation()
    handle.accept_navigation({"context": handle._navigation.context, "ranges": {"x": [598, 602], "y": [298, 302]}})
    renderer = options["series"][0][":renderItem"]
    def pixels(source, index=0):
        encoded = re.findall(r"data:image/png;base64,([A-Za-z0-9+/=]+)", source)[index]
        return Image.open(io.BytesIO(base64.b64decode(encoded)))
    assert pixels(original).width == 720
    assert pixels(renderer).width == 720  # The full-region overview is retained.
    assert pixels(renderer, -1).width == 7
    assert options["series"][0]["data"] == [[0, 0, 1200, 600]]
    assert options["visualMap"] == color_scale
    assert any(row[:3] == [600, 300, 1] for row in options["series"][1]["data"])


@pytest.mark.parametrize("compressed", (False, True))
@pytest.mark.parametrize("format_name", ("tsv", "csv", "txt", "npz", "npz-fortran", "npz-regions", "txt-row", "npz-row"))
def test_data_manager_zoom_streams_converted_profile_bytes(compressed, format_name):
    values = np.zeros(15000)
    values[12031] = 99
    sampled = np.linspace(0, values.size - 1, 500, dtype=int)
    preview = DataPreview(sampled.astype(float), values[sampled], "Position Kb", "signal", (values.size,), values.size, values.size, 0, 0.0, 99.0)
    if format_name.endswith("-row"):
        preview = replace(preview, source_shape=(1, values.size))
    target_format = format_name.split("-")[0]
    labels = ("signal",)
    if target_format == "npz":
        output = io.BytesIO()
        if format_name == "npz-fortran":
            np.savez(output, values=np.asfortranarray(np.c_[np.arange(values.size) + 0.5, values]), columns=np.array(["position_kb", "signal"]))
        elif format_name == "npz-regions":
            np.savez(output, chr1=values, chr2=values * 2)
            labels = ("chr1", "chr2")
        else:
            np.savez(output, values=values[None, :] if format_name.endswith("-row") else values)
        payload = output.getvalue()
    elif target_format == "txt":
        payload = (" " if format_name.endswith("-row") else "\n").join(str(value) for value in values).encode()
    else:
        delimiter = "," if target_format == "csv" else "\t"
        payload = (f"position_kb{delimiter}signal\n" + "\n".join(f"{i + 0.5}{delimiter}{v}" for i, v in enumerate(values))).encode()
    if compressed:
        payload = gzip.compress(payload)
    result = DataConversionResult(ExportArtifact(payload, "profile.gz" if compressed else "profile", "text/plain"), "tsv", target_format, "x", (values.size,), labels, preview=preview)
    data = _converted_preview_data(result)
    assert [12031.0, 99.0] not in data
    refined = data.refine((12000, 12080), None)["data"]
    assert [12031.0, 99.0] in refined
    assert len([row for row in refined if 12000 <= row[0] <= 12080]) < 100
    assert refined[0][0] == 0 and refined[-1][0] == values.size - 1
    assert data.refine((0, 2), None)["data"][0] == [0.0, 0.0]
    broad = data.refine((1000, 14000), None)["data"]
    assert len([row for row in broad if 999 <= row[0] <= 14001]) == 1000
    assert broad[0][0] == 0 and broad[-1][0] == values.size - 1


def test_histogram_controls_select_guides_and_clear_axis_labels(monkeypatch):
    handles = []
    attach = PreviewChartHandle.attach_settings
    def capture(handle, controls):
        attach(handle, controls)
        handles.append(handle)
    monkeypatch.setattr(PreviewChartHandle, "attach_settings", capture)
    async def scenario():
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-plotting").click()
            def control(marker):
                return next(item for item in user.client.elements.values() if f"plotting-{marker}" in item._markers and not item.is_deleted)
            await control("data-upload").handle_uploads([
                Upload.SmallFileUpload("first.csv", "text/csv", b"signal\n1\n2\n3\n4\n5\n"),
                Upload.SmallFileUpload("second.csv", "text/csv", b"signal\n6\n7\n8\n9\n10\n"),
            ])
            await user.should_see("second.csv", retries=60)
            await asyncio.sleep(0.1)
            user.find(marker="action-plot").click()
            await user.should_see(marker="plotting-result-plot", retries=60)
            assert not control("curve-type").visible
            assert all(control(f"curve-type-{i}").visible for i in (1, 2))
            with user.client:
                control("view").set_value("Histogram")
                control("side-by-side-bins").set_value(True)
            assert control("legend-position").value == "Top right"
            assert control("legend-position")._props["label"] == "Panel position"
            assert not control("histogram-statistics-position").visible
            with user.client:
                handles[-1].select_histogram_dataset({"index": 1})
            assert "signal 2" in control("result-plot").options["graphic"][0]["style"]["text"]
            with user.client:
                control("density").set_value(True)
                control("kde-smoothness").set_value(2)
            assert control("kde-smoothness").visible
            assert not control("side-by-side-bins").visible
            user.find(marker="plotting-dual-y-axes").click()
            with user.client:
                for field in ("x-axis-label", "y-axis-label", "x2-axis-label", "y2-axis-label"):
                    assert control(field)._props["clearable"] is True
                    control(field).set_value(None)
                control("show-hover").set_value(False)
            options = control("result-plot").options
            assert all(axis["name"] == "" for axis in options["xAxis"] + options["yAxis"])
            assert options["tooltip"]["show"] is False
            assert options["tooltip"]["axisPointer"]["label"]["show"] is False
            assert options["axisPointer"]["triggerOn"] == "none"
            assert all(series["emphasis"]["disabled"] and series["symbolSize"] == 0 for series in options["series"])
            with user.client:
                control("show-hover").set_value(True)
            assert control("result-plot").options["tooltip"]["show"] is True
            assert all(series["symbolSize"] == 6 for series in control("result-plot").options["series"])
    asyncio.run(scenario())
