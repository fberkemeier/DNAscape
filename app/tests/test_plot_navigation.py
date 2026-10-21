"""Plot gestures retain bounded viewports without modifying plotted data."""

import asyncio
import copy
import html
import json
import re
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import nicegui
from nicegui.elements.upload import Upload
from nicegui.testing import user_simulation

from app.main import index
from app.preview_plotting import PreviewChartHandle, PreviewLine, build_preview_plot_options
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


def test_mouse_gestures_in_real_echarts_previews(tmp_path):
    chrome = shutil.which("chromium") or shutil.which("google-chrome") or shutil.which("chrome")
    windows_chrome = Path("C:/Program Files/Google/Chrome/Application/chrome.exe")
    if chrome is None and windows_chrome.exists():
        chrome = str(windows_chrome)
    if chrome is None:
        pytest.skip("Chrome/Chromium is required for real chart interaction checks")
    root = Path(__file__).resolve().parents[2]
    cases = []

    def add(name, *, hover=True, **kwargs):
        options = navigation_options(**kwargs)
        handle = PreviewChartHandle(CaptureChart(options), CaptureChart({}))
        handle._apply_navigation()
        if not hover:
            handle.settings = SimpleNamespace(show_hover=SimpleNamespace(value=False))
        handle._apply_guides()
        cases.append(dict(name=name, options=options))

    add("line")
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
    add("log histogram", plot_type="Histogram", log_x=True, log_y=True)
    add("KDE", plot_type="Histogram", density=True)
    add("boxplot", plot_type="Boxplot")
    lines = (
        PreviewLine("First", np.arange(1.0, 11.0), np.arange(1.0, 11.0), "#b03412"),
        PreviewLine("Second", np.arange(1.0, 11.0), np.arange(1.0, 11.0) * 100, "#2596be"),
    )
    add("hovered curves", lines=lines)
    add("grouped histogram guides", plot_type="Histogram", lines=(lines[0],
        PreviewLine("Second", np.arange(1.0, 11.0), np.arange(2.0, 12.0), "#2596be")),
        show_legend=True, show_histogram_statistics=True, histogram_grouped=True)
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
        f'<script src="{(root / "app/static/plot_navigation.js").as_uri()}"></script>'
        f'<script src="{(root / "app/tests/fixtures/plot_navigation_probe.js").as_uri()}"></script>'
        '</body></html>', encoding="utf-8",
    )
    completed = subprocess.run([
        chrome, "--headless", "--disable-gpu", "--no-first-run", "--no-default-browser-check",
        "--allow-file-access-from-files", "--window-size=1000,600", "--virtual-time-budget=8000",
        f"--user-data-dir={tmp_path / 'browser'}", "--dump-dom", document.as_uri(),
    ], capture_output=True, text=True, encoding="utf-8", timeout=30, check=True)
    match = re.search(r'<pre id="result">([^<]+)</pre>', completed.stdout)
    assert match is not None and match[1] != "pending", completed.stderr[-1500:]
    result = json.loads(html.unescape(match[1]))
    assert not result["errors"]
    assert len(result["results"]) == len(cases)
    assert all(item["passed"] for item in result["results"]), json.dumps(result["results"], indent=2)
