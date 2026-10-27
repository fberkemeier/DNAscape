"""Plot gestures retain bounded viewports without modifying plotted data."""

import asyncio
import copy
import html
import json
import re
import shutil
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import nicegui
from nicegui.elements.upload import Upload
from nicegui.testing import user_simulation

from app.main import index
from app.preview_plotting import PreviewBand, PreviewChartHandle, PreviewLine, build_preview_plot_options
from dnascape.workflows.plotting import prepare_echarts_runtime


def navigation_options(plot_type="Line", **overrides):
    arguments = dict(
        plot_type=plot_type, title="Profile", x_label="Position", y_label="Value",
        color="#b03412", palette="viridis", histogram_bins=5, density=False,
        show_legend=False, show_colorbar=False, log_y=False, invert_y=False,
        y_limits=None, series_label="Profile", values=np.arange(1.0, 11.0),
        x_axis=np.arange(10.0),
    )
    if plot_type == "Heatmap":
        arguments.update(values=np.arange(1.0, 101.0).reshape(10, 10), y_axis=np.arange(10.0))
    if plot_type == "Histogram":
        arguments.update(histogram_x_limits=(1.0, 10.0))
    arguments.update(overrides)
    return build_preview_plot_options(**arguments)


class CaptureChart:
    def __init__(self, options):
        self.options = options
        self.updates = 0

    def update(self):
        self.updates += 1


@pytest.mark.parametrize("axis,log_x", [
    ([0, 124000, 249003.1415926], False),
    ([40000, 40450, 40999.75], False),
    ([.13, .47, .81], False),
    ([np.sqrt(2), np.sqrt(5), np.sqrt(11)], False),
    ([-9.7, -5.1, -2.3], False),
    ([.13, .47, .81], True),
    ([10, 1000, 9999], True),
])
@pytest.mark.parametrize("invert_x", [False, True])
def test_native_horizontal_bounds_are_independent_of_tick_labels(axis, log_x, invert_x):
    options = navigation_options(values=[1, 2, 3], x_axis=axis, log_x=log_x, invert_x=invert_x)
    assert [options["xAxis"]["min"], options["xAxis"]["max"]] == [axis[0], axis[-1]]
    assert options["dnascapeDefaultAxes"]["x"] == [axis[0], axis[-1]]
    ticks = options["xAxis"]["axisLabel"]["customValues"]
    if ticks:
        assert all(axis[0] <= tick <= axis[-1] for tick in ticks)
        assert axis[-1] not in ticks
    custom = navigation_options(values=[1, 2, 3], x_axis=axis, log_x=log_x, x_limits=(1, 12))
    assert (custom["xAxis"]["min"], custom["xAxis"]["max"]) == (1, 12)


def test_navigation_preserves_exact_viewport_edges_without_forcing_endpoint_labels():
    options = navigation_options()
    handle = PreviewChartHandle(CaptureChart(options), CaptureChart({}))
    handle._apply_navigation()
    handle.accept_navigation(dict(context=handle._navigation.context, ranges={"x": (2.13, 7.91)}))
    assert (options["xAxis"]["min"], options["xAxis"]["max"]) == (2.13, 7.91)
    for section in ("axisLabel", "axisTick", "splitLine"):
        assert options["xAxis"][section]["customValues"] == [3, 4, 5, 6, 7]


def test_collision_spacing_survives_backend_redraws_without_accumulating_or_leaking():
    options = navigation_options()
    handle = PreviewChartHandle(CaptureChart(options), CaptureChart({}))
    handle._apply_navigation()
    gap, left = options["yAxis"]["nameGap"], options["grid"]["left"]
    handle.accept_navigation(dict(
        context=handle._navigation.context, ranges={"x": (2, 4), "y": (3, 6)},
        titleGaps={"y": 60, "x": 100},
    ))
    assert options["yAxis"]["nameGap"] == 60
    assert options["grid"]["left"] == left + 60 - gap
    assert options["xAxis"]["nameGap"] == 28
    enlarged = dict(options["grid"])
    handle.accept_navigation(dict(
        context=handle._navigation.context, ranges={"y": (0, 10)}, titleGaps={"y": 28},
    ))
    assert options["grid"] == enlarged
    assert options["yAxis"]["nameGap"] == 60
    replacement = navigation_options(y_limits=(3, 6))
    expected_left = replacement["grid"]["left"] + 60 - replacement["yAxis"]["nameGap"]
    options.clear()
    options.update(replacement)
    handle.update()
    assert options["yAxis"]["nameGap"] == 60
    assert options["grid"]["left"] == expected_left
    handle.update()
    assert options["grid"]["left"] == expected_left
    replacement = navigation_options(values=np.arange(100., 110.))
    expected_grid = dict(replacement["grid"])
    options.clear()
    options.update(replacement)
    handle.update()
    assert options["grid"] == expected_grid
    assert options["yAxis"]["nameGap"] < 60


@pytest.mark.parametrize("plot_type", ("Line", "Heatmap", "Histogram", "Boxplot"))
def test_navigation_defaults_survive_explicit_axis_limits(plot_type):
    default = navigation_options(plot_type)
    zoom = navigation_options(plot_type, x_limits=(2.0, 5.0), y_limits=(2.0, 4.0))
    assert zoom["dnascapeDefaultAxes"] == default["dnascapeDefaultAxes"]
    assert zoom["yAxis"]["min"] == 2
    assert zoom["yAxis"]["max"] == 4


@pytest.mark.parametrize("plot_type", ("Line", "Histogram", "Boxplot"))
def test_independent_axes_keep_their_own_navigation_bounds(plot_type):
    lines = (
        PreviewLine("First", np.arange(10.0), np.arange(1.0, 11.0), "#b03412"),
        PreviewLine("Second", np.arange(10.0), np.arange(1.0, 11.0) * 100, "#2596be"),
    )
    options = navigation_options(plot_type, lines=lines, secondary_y_label="Second", secondary_x_label="Position")
    defaults = options["dnascapeDefaultAxes"]
    assert set(defaults) == {"x", "y", "secondary_x", "secondary_y"}
    dimension = "x" if plot_type == "Histogram" else "y"
    assert defaults[dimension][1] < defaults[f"secondary_{dimension}"][0]


@pytest.mark.parametrize("logarithmic", (False, True))
def test_navigation_clamps_and_preserves_data_and_rejects_stale_events(logarithmic):
    options = navigation_options(log_y=logarithmic)
    series = copy.deepcopy(options["series"])
    handle = PreviewChartHandle(CaptureChart(options), CaptureChart({}))
    handle._apply_navigation()
    context = options["dnascapeNavigation"]["context"]
    lower, upper = options["dnascapeDefaultAxes"]["y"]
    requested = (upper * 2, upper * 3) if logarithmic else (upper + 10, upper + 11)
    handle.accept_navigation(dict(context=context, ranges={"x": (2, 4), "y": requested}))
    assert (options["xAxis"]["min"], options["xAxis"]["max"]) == (2, 4)
    assert options["yAxis"]["max"] == pytest.approx(upper)
    assert options["yAxis"]["min"] >= lower
    if logarithmic:
        assert options["yAxis"]["max"] / options["yAxis"]["min"] == pytest.approx(1.5)
    else:
        assert options["yAxis"]["max"] - options["yAxis"]["min"] == pytest.approx(1)
    # Navigation refines only the displayed sample; the source data remains available.
    refined = options["series"][0]["data"]
    assert len(refined) <= len(series[0]["data"])
    assert list(refined.refine(None, None)["data"]) == series[0]["data"]
    handle.accept_navigation(dict(context="old plot", ranges={"x": (5, 6)}))
    assert options["xAxis"]["min"] == 2
    options.clear()
    options.update(navigation_options(values=np.arange(11.0, 21.0)))
    handle.update()
    assert not handle._navigation.ranges
    assert options["xAxis"]["min"] == options["dnascapeDefaultAxes"]["x"][0]


def test_automatic_zero_floor_survives_zoomed_redraws():
    original = navigation_options()
    upper = original["dnascapeDefaultAxes"]["y"][1]
    options = navigation_options(y_limits=(0, upper))
    handle = PreviewChartHandle(CaptureChart(options), CaptureChart({}))
    handle._apply_navigation()
    handle.accept_navigation(dict(context=handle._navigation.context, ranges={"y": (3, 6)}))
    options.clear()
    options.update(navigation_options(y_limits=(3, 6)))
    handle.update()
    assert options["dnascapeNavigation"]["axes"]["y"]["min"] == 0
    assert options["yAxis"]["min"] == 3
    assert options["yAxis"]["max"] == 6


def test_fixed_fraction_bounds_survive_zoomed_redraws():
    values = np.linspace(0.2, 0.8, 10)
    options = navigation_options(values=values, y_limits=(0, 1))
    handle = PreviewChartHandle(CaptureChart(options), CaptureChart({}))
    handle._apply_navigation()
    handle.accept_navigation(dict(context=handle._navigation.context, ranges={"y": (0.3, 0.5)}))
    options.clear()
    options.update(navigation_options(values=values, y_limits=(0.3, 0.5)))
    handle.update()
    assert options["dnascapeNavigation"]["axes"]["y"]["min"] == 0
    assert options["dnascapeNavigation"]["axes"]["y"]["max"] == 1
    assert options["yAxis"]["min"] == 0.3
    assert options["yAxis"]["max"] == 0.5


@pytest.mark.parametrize("histogram", (False, True))
def test_navigation_updates_fields_and_survives_styles_without_rebinning(monkeypatch, histogram):
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
                Upload.SmallFileUpload("first.csv", "text/csv", b"signal\n1\n2\n3\n4\n5\n6\n7\n8\n9\n10\n"),
            ])
            await user.should_see("first.csv", retries=60)
            await user.should_see("Array size: 10", retries=60)
            await asyncio.sleep(0.05)
            user.find(marker="action-plot").click()
            await user.should_see(marker="plotting-result-plot", retries=60)
            if histogram:
                with user.client:
                    control("view").set_value("Histogram")
            handle = handles[-1]
            original_series = copy.deepcopy(handle.options["series"])
            original_bounds = copy.deepcopy(handle.options["dnascapeDefaultAxes"])
            x_key = "histogram-x" if histogram else "x"
            with user.client:
                handle.accept_navigation(dict(context=handle._navigation.context, ranges={"x": (3, 6), "y": (0.5, 0.8) if histogram else (3, 6)}))
            assert control(f"{x_key}-minimum").value == 3
            assert control(f"{x_key}-maximum").value == 6
            assert control("y-minimum").value == (0.5 if histogram else 3)
            with user.client:
                control("line-width").set_value(3.2)
                control("show-grid").set_value(True)
            options = control("result-plot").options
            assert (options["xAxis"]["min"], options["xAxis"]["max"]) == (3, 6)
            if histogram:
                assert options["series"][0]["data"] == original_series[0]["data"]
            else:
                assert options["series"][0]["lineStyle"]["width"] == pytest.approx(3.2)
            assert options["dnascapeDefaultAxes"] == original_bounds
            user.find(marker=f"plotting-reset-{'histogram-x' if histogram else 'x'}-limits").click()
            await asyncio.sleep(0)
            assert control("result-plot").options["xAxis"]["min"] < 3
            assert control("result-plot").options["xAxis"]["max"] > 6
            user.find(marker="plotting-reset-y-limits").click()
            await asyncio.sleep(0)
            assert control("y-minimum").value == pytest.approx(original_bounds["y"][0])
    asyncio.run(scenario())


@pytest.mark.parametrize("data_manager", [False, True])
def test_real_nicegui_chart_retains_native_resolution_across_zoom_pan_styles_and_reset(monkeypatch, data_manager):
    handles = []
    attach = PreviewChartHandle.attach_settings

    def capture(handle, controls):
        attach(handle, controls)
        handles.append(handle)

    monkeypatch.setattr(PreviewChartHandle, "attach_settings", capture)
    values = np.sin(np.arange(10_001) / 50)
    values[3334] = 17.0  # A native peak absent from the initial preview.
    payload = ("signal\n" + "\n".join(map(str, values))).encode()

    async def scenario():
        async with user_simulation(index) as user:
            await user.open("/")
            name = "data-management" if data_manager else "plotting"
            prefix = "data-manager" if data_manager else "plotting"
            user.find(marker=f"nav-{name}").click()

            def control(marker):
                return next(item for item in user.client.elements.values()
                            if f"{prefix}-{marker}" in item._markers and not item.is_deleted)

            await control("upload" if data_manager else "data-upload").handle_uploads([
                Upload.SmallFileUpload("signal.tsv", "text/tab-separated-values", payload),
            ])
            await user.should_see("Array size: 10001", retries=80)
            user.find(marker="action-convert" if data_manager else "action-plot").click()
            await user.should_see(marker=f"{prefix}-{'preview-chart' if data_manager else 'result-plot'}", retries=100)
            handle = handles[-1]
            assert handle.options is not handle.canvas.options
            assert callable(handle.options["series"][0]["data"].refine)
            assert not hasattr(handle.canvas.options["series"][0]["data"], "refine")
            assert len(handle.canvas.options["series"][0]["data"]) == 1000
            assert not any(point[0] == 3334 for point in handle.canvas.options["series"][0]["data"])

            def navigate(lower, upper):
                with user.client:
                    handle.accept_navigation(dict(context=handle._navigation.context, ranges={"x": (lower, upper)}))
                return np.asarray(handle.canvas.options["series"][0]["data"])

            detailed = navigate(3320, 3340)
            np.testing.assert_equal(detailed[(detailed[:, 0] >= 3319) & (detailed[:, 0] <= 3341), 0], np.arange(3319, 3342))
            assert detailed[0, 0] == 0 and detailed[-1, 0] == 10_000
            assert detailed[detailed[:, 0] == 3334, 1] == 17
            with user.client:
                control("plot-line-width" if data_manager else "line-width").set_value(3)
                handle.open_zoom()
            np.testing.assert_equal(handle.zoom.options["series"][0]["data"], detailed)
            broad = navigate(1000, 6000)
            assert len(broad[(broad[:, 0] >= 999) & (broad[:, 0] <= 6001)]) == 1000
            assert broad[0, 0] == 0 and broad[-1, 0] == 10_000
            assert len(broad) <= 4000
            panned = navigate(7000, 7020)
            np.testing.assert_equal(panned[(panned[:, 0] >= 6999) & (panned[:, 0] <= 7021), 0], np.arange(6999, 7022))
            user.find(marker=f"{prefix}-{'plot-' if data_manager else ''}reset-x-limits").click()
            reset = np.asarray(handle.canvas.options["series"][0]["data"])
            assert len(reset) == 1000 and reset[0, 0] == 0 and reset[-1, 0] == 10_000

    asyncio.run(scenario())


@pytest.mark.parametrize("logarithmic,descending", [(False, False), (False, True), (True, False), (True, True)])
def test_buffered_detail_keeps_the_visible_budget_and_refines_on_zoom(logarithmic, descending):
    axis = np.geomspace(1, 1e6, 100_001) if logarithmic else np.arange(100_001.)
    if descending:
        axis = axis[::-1]
    options = navigation_options(x_axis=axis, values=np.sin(np.arange(axis.size) / 10), log_x=logarithmic)
    handle = PreviewChartHandle(CaptureChart(options), CaptureChart({}))
    handle._apply_navigation()
    window = (1000, 2000) if logarithmic else (40_000, 41_000)
    handle.accept_navigation(dict(context=handle._navigation.context, ranges={"x": window}))
    data = options["series"][0]["data"]
    assert data.coverage["x"][0] < window[0] < window[1] < data.coverage["x"][1]
    coordinates = np.asarray(data)[:, 0]
    inside = coordinates[(coordinates >= window[0]) & (coordinates <= window[1])]
    assert 995 <= inside.size <= 1000
    assert len(data) <= 4000
    assert coordinates.min() == axis.min() and coordinates.max() == axis.max()
    small_pan = (1050, 2050) if logarithmic else (40_100, 41_100)
    handle.accept_navigation(dict(context=handle._navigation.context, ranges={"x": small_pan}))
    assert options["series"][0]["data"] is data
    close = (1200, 1210) if logarithmic else (40_400, 40_410)
    handle.accept_navigation(dict(context=handle._navigation.context, ranges={"x": close}))
    assert options["series"][0]["data"] is not data
    coordinates = np.asarray(options["series"][0]["data"])[:, 0]
    expected = axis[(axis >= close[0]) & (axis <= close[1])]
    np.testing.assert_equal(coordinates[(coordinates >= close[0]) & (coordinates <= close[1])], expected)


def test_navigation_sequences_and_reset_tokens_reject_obsolete_events():
    options = navigation_options()
    handle = PreviewChartHandle(CaptureChart(options), CaptureChart({}))
    handle._apply_navigation()
    payload = dict(context=handle._navigation.context, token=handle._navigation.token)
    handle.accept_navigation(dict(**payload, sequence=3, ranges={"x": (4, 6)}))
    for sequence in (1, 2, 3, -1, True, float("nan"), "4"):
        handle.accept_navigation(dict(**payload, sequence=sequence, ranges={"x": (1, 2)}))
        assert options["xAxis"]["min"] == 4
    handle._navigation.clear()
    options["xAxis"]["min"], options["xAxis"]["max"] = options["dnascapeDefaultAxes"]["x"]
    handle.update()
    handle.accept_navigation(dict(**payload, sequence=4, ranges={"x": (1, 2)}))
    assert options["xAxis"]["min"] == options["dnascapeDefaultAxes"]["x"][0]


@pytest.mark.parametrize("reset_during_read", (False, True))
def test_async_detail_reads_coalesce_without_blocking_and_only_install_the_latest(monkeypatch, reset_during_read):
    from app import preview_plotting

    monkeypatch.setattr(preview_plotting.run, "io_bound", asyncio.to_thread)
    options = navigation_options(x_axis=np.arange(20_001.), values=np.arange(20_001.) % 17)
    handle = PreviewChartHandle(CaptureChart(options), CaptureChart({}))
    handle._apply_navigation()
    source = options["series"][0]["data"]
    original = source.refine
    entered, release = threading.Event(), threading.Event()
    calls = []
    def read(x, y):
        calls.append(x)
        if len(calls) == 1:
            entered.set()
            assert release.wait(3), "The UI did not stay responsive while reading detail."
        return original(x, y)
    source.refine = read

    async def scenario():
        metadata = dict(context=handle._navigation.context, token=handle._navigation.token)
        first = asyncio.create_task(handle.accept_navigation_async(dict(**metadata, sequence=1, ranges={"x": (100, 200)})))
        for _ in range(100):
            if entered.is_set():
                break
            await asyncio.sleep(.005)
        assert entered.is_set()
        await handle.accept_navigation_async(dict(**metadata, sequence=2, ranges={"x": (1000, 1100)}))
        if reset_during_read:
            handle._navigation.clear()
            metadata["token"] = handle._navigation.token
        await handle.accept_navigation_async(dict(**metadata, sequence=3, ranges={"x": (9000, 9100)}))
        assert not first.done()
        assert options["xAxis"]["min"] == 9000
        release.set()
        await first
        assert calls == [(100., 200.), (9000., 9100.)]
        assert options["dnascapeNavigation"]["sequence"] == 3
        assert options["series"][0]["data"].window[0] == (9000., 9100.)
        coordinates = np.asarray(options["series"][0]["data"])[:, 0]
        np.testing.assert_equal(coordinates[(coordinates >= 9000) & (coordinates <= 9100)], np.arange(9000, 9101))
        assert handle.canvas.updates == 1
        await handle.accept_navigation_async(dict(**metadata, sequence=4, ranges={"x": (9010, 9110)}))
        assert len(calls) == 2 and handle.canvas.updates == 1
    try:
        asyncio.run(scenario())
    finally:
        release.set()


@pytest.mark.parametrize("file_type", ("tsv", "txt", "csv", "npz"))
@pytest.mark.parametrize("compressed", (False, True))
def test_converted_detail_formats_keep_native_positions_missing_values_and_buffer(file_type, compressed):
    from app.data_conversion_service import DataConversionRequest, convert_data
    from app.mapping_uploads import inspect_upload
    from app.pages.data_management import _converted_preview_data

    values = np.arange(10_001, dtype=float)
    values[3334] = np.nan
    inspection = inspect_upload("source.tsv", ("signal\n" + "\n".join(map(str, values))).encode())
    result = convert_data(DataConversionRequest(
        inspection=inspection, input_form="x", value_column="signal",
        missing_policy="ignore", target_format=file_type, gzip_output=compressed,
    ))
    original_bytes = result.artifact.data
    data = _converted_preview_data(result)
    refined = data.refine((3320, 3340), None)["data"]
    native = {row[0]: row[1] for row in refined}
    for x in range(3319, 3342):
        assert native[x] == (None if x == 3334 else x)
    assert refined.coverage["x"] == (3300, 3360)
    assert min(native) == 0 and max(native) == 10_000
    assert len(refined) <= 4000
    assert result.artifact.data == original_bytes


def test_vector_export_uses_latest_cached_pan_without_requiring_a_canvas_redraw(monkeypatch):
    from app import preview_plotting
    monkeypatch.setattr(preview_plotting.run, "io_bound", asyncio.to_thread)
    class ExportChart(CaptureChart):
        async def run_chart_method(self, *_args, **_kwargs):
            return "svg"
    options = navigation_options(x_axis=np.arange(10_001.), values=np.sin(np.arange(10_001.) / 20))
    canvas, vector = ExportChart(copy.deepcopy(options)), ExportChart({})
    handle = PreviewChartHandle(canvas, vector, source_options=options)
    handle._apply_navigation()
    async def scenario():
        metadata = dict(context=handle._navigation.context, token=handle._navigation.token)
        await handle.accept_navigation_async(dict(**metadata, sequence=1, ranges={"x": (4000, 4200)}))
        updates = canvas.updates
        await handle.accept_navigation_async(dict(**metadata, sequence=2, ranges={"x": (4050, 4250)}))
        assert canvas.updates == updates
        assert canvas.options["xAxis"]["min"] == 4000
        assert handle.options["xAxis"]["min"] == 4050
        await handle.run_chart_method("getDataURL", {"type": "svg"})
        assert vector.options["xAxis"]["min"] == 4050
        assert vector.options["xAxis"]["max"] == 4250
    asyncio.run(scenario())


def test_mouse_gestures_in_real_echarts_previews(tmp_path):
    chrome = shutil.which("chromium") or shutil.which("google-chrome") or shutil.which("chrome")
    windows_chrome = Path("C:/Program Files/Google/Chrome/Application/chrome.exe")
    if chrome is None and windows_chrome.exists():
        chrome = str(windows_chrome)
    if chrome is None:
        pytest.skip("Chrome/Chromium is required for real chart interaction checks")
    root = Path(__file__).resolve().parents[2]
    cases = []

    def add(name, *, hover=True, refinement=None, **kwargs):
        options = navigation_options(**kwargs)
        handle = PreviewChartHandle(CaptureChart(options), CaptureChart({}))
        handle._apply_navigation()
        if refinement is not None:
            handle.accept_navigation(dict(context=handle._navigation.context, ranges=refinement))
        if not hover:
            handle.settings = SimpleNamespace(show_hover=SimpleNamespace(value=False))
        handle._apply_guides()
        cases.append(dict(name=name, options=options))

    add("line")
    add("wide zoomed ticks")
    add("unrounded endpoint", x_axis=np.linspace(0, 249003.1415926, 1001), values=np.linspace(1, 10, 1001))
    add("line hover off", hover=False)
    add("line hover again")
    add("line with legend", show_legend=True)
    add("inverted line", invert_x=True, invert_y=True)
    add("log line", x_axis=np.arange(1.0, 11.0), log_x=True, log_y=True)
    add("heatmap", plot_type="Heatmap")
    add("heatmap hover off", plot_type="Heatmap", hover=False)
    add("inverted log heatmap", plot_type="Heatmap", x_axis=np.arange(1.0, 11.0),
        y_axis=np.arange(1.0, 11.0), log_x=True, log_y=True, invert_x=True, invert_y=True)
    add("histogram", plot_type="Histogram")
    add("histogram hover off", plot_type="Histogram", hover=False)
    add("separated histogram", plot_type="Histogram", histogram_touching=False)
    add("inverted histogram", plot_type="Histogram", invert_x=True, invert_y=True)
    add("fine histogram", plot_type="Histogram", histogram_bins=30,
        values=np.random.default_rng(42).normal(6, .8, 1000), histogram_x_limits=(3, 9))
    add("log histogram", plot_type="Histogram", log_x=True, log_y=True)
    add("KDE", plot_type="Histogram", density=True)
    add("boxplot", plot_type="Boxplot")
    long_axis = np.arange(10_001.)
    add("retained curve overview", x_axis=long_axis, values=np.full(long_axis.size, 4.),
        refinement={"x": (4200, 4400)}, show_grid=False)
    add("retained inverted overview", x_axis=long_axis, values=np.full(long_axis.size, 4.),
        refinement={"x": (4200, 4400)}, invert_x=True, show_grid=False)
    add("retained band overview", x_axis=long_axis, values=np.full(long_axis.size, 4.),
        bands=(PreviewBand("SD", long_axis, np.full(long_axis.size, 3.), np.full(long_axis.size, 5.), "#b03412"),),
        refinement={"x": (4200, 4400)}, show_grid=False)
    heatmap = np.ones((601, 1201))
    heatmap[300, 600] = np.nan
    add("retained heatmap overview", plot_type="Heatmap", values=heatmap,
        x_axis=np.arange(1201.), y_axis=np.arange(601.), show_grid=False,
        refinement={"x": (598, 602), "y": (298, 302)})
    add("retained inverted heatmap overview", plot_type="Heatmap", values=heatmap,
        x_axis=np.arange(1201.), y_axis=np.arange(601.), show_grid=False, invert_x=True, invert_y=True,
        refinement={"x": (598, 602), "y": (298, 302)})
    lines = (
        PreviewLine("First", np.arange(1.0, 11.0), np.arange(1.0, 11.0), "#b03412"),
        PreviewLine("Second", np.arange(1.0, 11.0), np.arange(1.0, 11.0) * 100, "#2596be"),
    )
    add("hovered curves", lines=lines)
    add("grouped histogram guides", plot_type="Histogram", lines=(lines[0],
        PreviewLine("Second", np.arange(1.0, 11.0), np.arange(2.0, 12.0), "#2596be")),
        show_legend=True, show_histogram_statistics=True, histogram_grouped=True)
    add("overlaid histogram", plot_type="Histogram", lines=(lines[0],
        PreviewLine("Second", np.arange(1.0, 11.0), np.arange(2.0, 12.0), "#2596be")))
    for plot_type in ("Line", "Histogram", "Boxplot"):
        add(f"independent {plot_type}", plot_type=plot_type, lines=lines,
            secondary_y_label="Second", secondary_x_label="Second X",
            secondary_log_y=True, secondary_invert_y=True, secondary_invert_x=True)
    runtime = Path(nicegui.__file__).parent / "elements/echart/dist/index.js"
    (tmp_path / "echarts.js").write_text(prepare_echarts_runtime(runtime), encoding="utf-8")
    document = tmp_path / "navigation.html"
    document.write_text(
        '<!doctype html><html><head><meta charset="utf-8">'
        f'<link rel="stylesheet" href="{(root / "app/static/styles.css").as_uri()}">'
        '</head><body class="dn-body"><div id="c1" class="nicegui-echart dn-plot-navigation" '
        'style="width:820px;height:360px"></div><pre id="result">pending</pre>'
        '<script src="echarts.js"></script>'
        f'<script>window.navigationCases={json.dumps(cases)};</script>'
        f'<script src="{(root / "app/static/brand_theme.js").as_uri()}"></script>'
        f'<script src="{(root / "dnascape/workflows/plot_navigation.js").as_uri()}"></script>'
        f'<script src="{(root / "app/tests/fixtures/plot_navigation_probe.js").as_uri()}"></script>'
        '</body></html>', encoding="utf-8",
    )
    completed = subprocess.run([
        chrome, "--headless", "--disable-gpu", "--no-first-run", "--no-default-browser-check",
        "--allow-file-access-from-files", "--window-size=1000,600", "--virtual-time-budget=18000",
        f"--user-data-dir={tmp_path / 'browser'}", "--dump-dom", document.as_uri(),
    ], capture_output=True, text=True, encoding="utf-8", timeout=30, check=True)
    match = re.search(r'<pre id="result">([^<]+)</pre>', completed.stdout)
    assert match is not None and match[1] != "pending", completed.stderr[-1500:]
    result = json.loads(html.unescape(match[1]))
    assert not result["errors"]
    assert len(result["results"]) == len(cases)
    assert all(item["passed"] for item in result["results"]), json.dumps(result["results"], indent=2)
