"""Mapping plot parity and executable portable HTML exports."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from dataclasses import replace
from html.parser import HTMLParser
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from xml.etree import ElementTree

import numpy as np
import pytest

from dnascape.workflows.mapping import MappingResult
from dnascape.workflows.plotting import (
    combined_mapping_plot_options,
    mapping_plot_options,
    prepare_echarts_runtime,
    write_plot_html,
)
from dnascape.workflows.recipe_exports import plot_mapping_outputs


def _result() -> MappingResult:
    return MappingResult(
        target="replication_timing",
        target_key="replication_timing",
        target_label="Replication timing",
        method="test-profile",
        output_form="x",
        output_dependence="space",
        primary_output="replication_timing",
        outputs={"replication_timing": np.asarray([2.0, 4.0, 3.0])},
        output_forms={"replication_timing": "x"},
        x_axis=np.asarray([10.0, 12.0, 14.0]),
        time_axis=np.asarray([], dtype=float),
        region_labels=(),
        elapsed_seconds=0.1,
        assumptions={"geometry": "periodic"},
    )


def _settings() -> dict[str, Any]:
    return dict(
        output_key="replication_timing",
        plot_type="Line",
        normalization="None",
        title="Timing </script> & example",
        x_label="Position (kb)",
        y_label="Time (min)",
        color="#2596be",
        secondary_color="#b03412",
        palette="viridis",
        histogram_bins=4,
        line_width=3.0,
        show_markers=False,
        show_colorbar=True,
        log_y=False,
        invert_y=False,
        log_z=False,
        series_label="Mapped timing",
        density=False,
        show_legend=True,
        legend_position="Bottom left",
        show_grid=False,
        y_limits=(1.0, 5.0),
        log_x=False,
        invert_x=True,
        invert_z=False,
        show_histogram_statistics=True,
        histogram_statistics_position="Top left",
        histogram_bin_size=1.0,
        histogram_x_limits=(1.0, 5.0),
        histogram_touching=False,
        histogram_boundary="right",
    )


class _Scripts(HTMLParser):
    def __init__(self, html: str) -> None:
        super().__init__()
        self.scripts: list[str] = []
        self.in_script = False
        self.feed(html)

    def handle_starttag(self, tag: str, _attrs: Any) -> None:
        if tag == "script":
            self.in_script = True
            self.scripts.append("")

    def handle_endtag(self, tag: str) -> None:
        if tag == "script":
            self.in_script = False

    def handle_data(self, data: str) -> None:
        if self.in_script:
            self.scripts[-1] += data


def _embedded_options(path: Path) -> dict[str, Any]:
    match = re.search(r"const specification = (.+);\n", path.read_text(encoding="utf-8"))
    assert match is not None
    return json.loads(match.group(1))


@pytest.mark.parametrize("plot_type", ("Line", "Histogram", "Heatmap"))
def test_mapping_package_and_app_share_identical_options(plot_type: str) -> None:
    from app.pages.mapping import _mapping_plot_options

    result = _result()
    settings = {**_settings(), "plot_type": plot_type}
    if plot_type == "Heatmap":
        result = replace(
            result,
            output_form="xt",
            output_dependence="spacetime",
            outputs={"replication_timing": np.asarray([[2.0, 3.0], [4.0, 5.0], [3.0, 4.0]])},
            output_forms={"replication_timing": "xt"},
            time_axis=np.asarray([0.0, 5.0]),
        )
        settings["y_limits"] = None
    assert mapping_plot_options(result, **settings) == _mapping_plot_options(result, **settings)


def test_optional_mapping_html_keeps_the_exact_chart_specification(tmp_path: Path) -> None:
    result = _result()
    named = SimpleNamespace(key="mapping-case-1", name="Full name", short_name="Short", result=result)
    runtime = tmp_path / "echarts.js"
    runtime.write_text("/* bundled offline runtime */", encoding="utf-8")
    settings = _settings()
    paths = plot_mapping_outputs(
        (named,),
        {"output": "primary", "view": "Default", "settings": settings, "plot_export_dpi": 300},
        directory=tmp_path / "plots",
        echarts_path=runtime,
    )
    assert len(paths) == 1
    assert _embedded_options(paths[0]) == mapping_plot_options(result, **settings)
    html = paths[0].read_text(encoding="utf-8")
    assert "bundled offline runtime" in html
    assert 'src="https://' not in html
    assert "Timing &lt;/script&gt; &amp; example" in html


def test_combined_diagnostic_plots_keep_the_custom_curve_colors(tmp_path: Path) -> None:
    first = _result()
    result = replace(
        first,
        outputs={**first.outputs, "residual": np.asarray([0.1, 0.2, 0.15])},
        output_forms={**first.output_forms, "residual": "x"},
    )
    named = tuple(
        SimpleNamespace(key=f"mapping-case-{index}", name=f"Dataset {index}", short_name=f"D{index}", result=result)
        for index in (1, 2)
    )
    colors = ("#aabbcc", "#ddffee")
    paths = plot_mapping_outputs(
        named,
        {
            "output": "fit_diagnostic",
            "view": "Default",
            "layout": "Combined",
            "settings": {**_settings(), "output_key": "fit_diagnostic"},
            "colors": [
                {"dataset": dataset.key, "output": "fit_diagnostic", "series": 0, "color": color}
                for dataset, color in zip(named, colors, strict=True)
            ],
        },
        directory=tmp_path,
    )
    options = _embedded_options(paths[0])
    assert [series["lineStyle"]["color"] for series in options["series"]] == list(colors)


@pytest.mark.parametrize("dpi,expected_ratio", ((48, 1.0), (150, 150 / 96), (600, 4.0)))
def test_exported_browser_script_revives_callbacks_and_downloads_raster_and_vector(
    tmp_path: Path, dpi: int, expected_ratio: float
) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is needed to execute the emitted browser callbacks.")
    path = write_plot_html(
        mapping_plot_options(_result(), **_settings()), tmp_path / "mapped.html", dpi=dpi
    )
    scripts = _Scripts(path.read_text(encoding="utf-8")).scripts
    assert len(scripts) == 2  # Escaped title/data cannot inject additional script elements.
    emitted = tmp_path / "emitted.js"
    emitted.write_text(scripts[-1], encoding="utf-8")
    runner = tmp_path / "verify.js"
    runner.write_text(
        """const fs = require('node:fs');
const vm = require('node:vm');
const charts = [], downloads = [], handlers = {};
const document = {
  body: {appendChild() {}},
  getElementById(id) {return {addEventListener(_event, callback) {handlers[id] = callback;}};},
  createElement(tag) {
    return {style: {}, remove() {}, click() {downloads.push(this.download);}};
  },
};
const echarts = {init(_surface, _theme, initOptions) {
  const chart = {
    renderer: initOptions.renderer,
    setOption(options) {this.options = options;},
    getDataURL(options) {this.capture = options; return 'data:image/mock';},
    getWidth() {return 800;}, resize() {}, dispose() {this.disposed = true;},
  };
  charts.push(chart);
  return chart;
}};
vm.runInNewContext(fs.readFileSync(process.argv[2], 'utf8'), {
  document, echarts, window: {addEventListener() {}},
});
handlers['download-png']();
handlers['download-svg']();
console.log(JSON.stringify({
  axisLabel: charts[0].options.xAxis.axisLabel.formatter(12.5),
  tooltip: charts[0].options.tooltip.valueFormatter(1.234),
  renderers: charts.map(chart => chart.renderer),
  captures: charts.map(chart => chart.capture),
  sharedOptions: charts[0].options === charts[1].options,
  vectorDisposed: charts[1].disposed,
  downloads,
}));
""",
        encoding="utf-8",
    )
    completed = subprocess.run([node, str(runner), str(emitted)], capture_output=True, text=True, check=True)
    observed = json.loads(completed.stdout)
    assert observed["axisLabel"] == "12.5"
    assert observed["tooltip"] == "1.23"
    assert observed["renderers"] == ["canvas", "svg"]
    assert observed["captures"][0] == {"type": "png", "pixelRatio": expected_ratio, "backgroundColor": "#fff"}
    assert observed["captures"][1] == {"type": "svg", "backgroundColor": "#fff"}
    assert observed["sharedOptions"] is True
    assert observed["vectorDisposed"] is True
    assert observed["downloads"] == ["mapped.png", "mapped.svg"]


def test_optional_plot_workflow_does_not_import_app_or_nicegui(tmp_path: Path) -> None:
    script = """import builtins, sys
from pathlib import Path
original_import = builtins.__import__
def headless_import(name, *args, **kwargs):
    if name.split('.')[0] in {'app', 'nicegui'}:
        raise ImportError('UI import attempted: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = headless_import
from dnascape.workflows.plotting import write_plot_html
from dnascape.workflows.recipe_exports import plot_mapping_outputs
from types import SimpleNamespace
import json
import numpy as np
write_plot_html({'series': []}, Path(sys.argv[1]) / 'headless.html')
result = SimpleNamespace(
    primary_output='replication_timing', output_forms={'replication_timing': 'x'},
    output_dependence='space', outputs={'replication_timing': np.asarray([2., 4., 3.])},
    x_axis=np.asarray([10., 12., 14.]), assumptions={}, diagnostics={},
)
named = SimpleNamespace(key='one', name='Mapped profile', short_name='Profile', result=result)
assert len(plot_mapping_outputs((named,), {
    'output': 'primary', 'view': 'Default', 'settings': json.loads(sys.argv[2]),
}, directory=sys.argv[1])) == 1
assert not any(name.split('.')[0] in {'app', 'nicegui'} for name in sys.modules)
"""
    subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), json.dumps(_settings())],
        capture_output=True, text=True, check=True,
    )
    assert (tmp_path / "headless.html").is_file()
    assert (tmp_path / "mapping_one.html").is_file()


def test_bundled_echarts_runtime_is_valid_javascript_and_renders_svg(tmp_path: Path) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is needed to validate the bundled renderer.")
    import nicegui

    entry = Path(nicegui.__file__).resolve().parent / "elements" / "echart" / "dist" / "index.js"
    runtime = tmp_path / "echarts.js"
    runtime.write_text(prepare_echarts_runtime(entry), encoding="utf-8")
    subprocess.run([node, "--check", str(runtime)], capture_output=True, text=True, check=True)
    renderer = tmp_path / "render.js"
    renderer.write_text(
        """const fs = require('node:fs');
const vm = require('node:vm');
const context = {setTimeout, clearTimeout};
vm.runInNewContext(fs.readFileSync(process.argv[2], 'utf8'), context);
const chart = context.echarts.init(null, null, {renderer: 'svg', ssr: true, width: 800, height: 360});
chart.setOption({animation: false, xAxis: {type: 'value'}, yAxis: {type: 'value'}, series: [{type: 'line', data: [[0, 1], [1, 2]]}]});
const svg = chart.renderToSVGString();
chart.dispose();
console.log(JSON.stringify({version: context.echarts.version, valid: svg.includes('<svg') && svg.includes('<path')}));
""",
        encoding="utf-8",
    )
    completed = subprocess.run([node, str(renderer), str(runtime)], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr[-2500:]
    observed = json.loads(completed.stdout)
    assert observed["version"]
    assert observed["valid"] is True


@pytest.mark.parametrize("side", ("left", "right"))
@pytest.mark.parametrize("kind", ("single", "combined", "histogram"))
def test_rendered_bottom_legends_stay_inside_axes_after_resize(
    tmp_path: Path, side: str, kind: str,
) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is needed to exercise the app's ECharts renderer.")
    import nicegui

    settings = {
        **_settings(), "legend_position": f"Bottom {side}",
        "plot_type": "Histogram" if kind == "histogram" else "Line",
        "show_histogram_statistics": False,
    }
    result = _result()
    if kind == "single":
        options = mapping_plot_options(result, **settings)
    else:
        named = tuple(
            SimpleNamespace(key=f"case-{index}", name=f"Dataset {index}", result=result)
            for index in (1, 2)
        )
        options = combined_mapping_plot_options(named, **settings)
    entry = Path(nicegui.__file__).resolve().parent / "elements" / "echart" / "dist" / "index.js"
    runtime = tmp_path / "echarts.js"
    runtime.write_text(prepare_echarts_runtime(entry), encoding="utf-8")
    specification = tmp_path / "specification.json"
    specification.write_text(json.dumps(options), encoding="utf-8")
    renderer = tmp_path / "legend.cjs"
    renderer.write_text(
        """const fs = require('node:fs');
const vm = require('node:vm');
const context = {setTimeout, clearTimeout};
vm.runInNewContext(fs.readFileSync(process.argv[2], 'utf8'), context);
function revive(value) {
  if (Array.isArray(value)) return value.map(revive);
  if (!value || typeof value !== 'object') return value;
  return Object.fromEntries(Object.entries(value).map(([key, item]) =>
    key.startsWith(':') ? [key.slice(1), vm.runInNewContext('(' + item + ')', context)] : [key, revive(item)]));
}
const chart = context.echarts.init(null, null, {renderer: 'svg', ssr: true, width: 800, height: 360});
chart.setOption(revive(JSON.parse(fs.readFileSync(process.argv[3], 'utf8'))));
const layouts = [];
for (const [width, height] of [[800, 360], [360, 240], [1200, 540]]) {
  chart.resize({width, height});
  chart.renderToSVGString();
  const model = chart.getModel();
  const grid = model.getComponent('grid').coordinateSystem.getRect();
  const group = chart.getViewOfComponentModel(model.getComponent('legend')).group;
  const rect = group.getBoundingRect();
  const [x, y] = group.transformCoordToGlobal(rect.x, rect.y);
  layouts.push({grid, legend: {x, y, width: rect.width, height: rect.height}});
}
console.log(JSON.stringify(layouts));
chart.dispose();
""",
        encoding="utf-8",
    )
    completed = subprocess.run(
        [node, str(renderer), str(runtime), str(specification)],
        capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stderr[-2500:]
    for layout in json.loads(completed.stdout):
        grid, legend = layout["grid"], layout["legend"]
        assert legend["y"] >= grid["y"]
        assert legend["y"] + legend["height"] < grid["y"] + grid["height"]
        assert legend["x"] >= grid["x"]
        assert legend["x"] + legend["width"] <= grid["x"] + grid["width"]
        grid_center = grid["x"] + grid["width"] / 2
        legend_center = legend["x"] + legend["width"] / 2
        assert (legend_center < grid_center) if side == "left" else (legend_center > grid_center)


@pytest.mark.parametrize("show_grid", (False, True))
def test_heatmap_renderer_keeps_grid_above_raster_and_exports_vector_axes(
    tmp_path: Path, show_grid: bool,
) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is needed to exercise the app's ECharts renderer.")
    import nicegui
    import base64
    from dnascape.workflows.exports import build_chart_capture_export

    result = replace(
        _result(), output_form="xt", output_dependence="spacetime",
        outputs={"replication_timing": np.asarray([[2.0, 3.0], [4.0, 5.0], [3.0, 4.0]])},
        output_forms={"replication_timing": "xt"}, time_axis=np.asarray([0.0, 5.0]),
    )
    options = mapping_plot_options(result, **{
        **_settings(), "plot_type": "Heatmap", "y_limits": None,
        "show_grid": show_grid, "show_colorbar": False,
    })
    entry = Path(nicegui.__file__).resolve().parent / "elements" / "echart" / "dist" / "index.js"
    runtime = tmp_path / "echarts.js"
    runtime.write_text(prepare_echarts_runtime(entry), encoding="utf-8")
    specification = tmp_path / "specification.json"
    specification.write_text(json.dumps(options), encoding="utf-8")
    renderer = tmp_path / "heatmap.js"
    renderer.write_text(
        """const fs = require('node:fs');
const vm = require('node:vm');
const context = {setTimeout, clearTimeout};
vm.runInNewContext(fs.readFileSync(process.argv[2], 'utf8'), context);
function revive(value) {
  if (Array.isArray(value)) return value.map(revive);
  if (!value || typeof value !== 'object') return value;
  return Object.fromEntries(Object.entries(value).map(([key, item]) =>
    key.startsWith(':') ? [key.slice(1), vm.runInNewContext('(' + item + ')', context)] : [key, revive(item)]));
}
const chart = context.echarts.init(null, null, {renderer: 'svg', ssr: true, width: 800, height: 360});
chart.setOption(revive(JSON.parse(fs.readFileSync(process.argv[3], 'utf8'))));
fs.writeFileSync(process.argv[4], chart.renderToSVGString());
chart.dispose();
""",
        encoding="utf-8",
    )
    svg_path = tmp_path / "heatmap.svg"
    completed = subprocess.run(
        [node, str(renderer), str(runtime), str(specification), str(svg_path)],
        capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stderr[-2500:]
    svg = svg_path.read_bytes()
    elements = list(ElementTree.fromstring(svg).iter())
    images = [element for element in elements if element.tag.endswith("}image")]
    assert len(images) == 1
    assert any(value.startswith("data:image/png;base64,") for value in images[0].attrib.values())
    assert any(element.tag.endswith("}text") and element.text == "Position (kb)" for element in elements)
    grid_lines = [element for element in elements if element.attrib.get("stroke") == "#e8e3de"]
    if show_grid:
        assert grid_lines
        assert all(elements.index(line) > elements.index(images[0]) for line in grid_lines)
    else:
        assert not grid_lines
    data_uri = "data:image/svg+xml;base64," + base64.b64encode(svg).decode("ascii")
    exported = build_chart_capture_export(data_uri, "pdf")
    assert exported.data.startswith(b"%PDF")


def test_large_temporal_mapping_heatmap_is_bounded_before_and_during_sampling() -> None:
    import tracemalloc

    result = replace(
        _result(), output_form="t", output_dependence="time",
        outputs={"replication_timing": np.linspace(1.0, 10.0, 4000)},
        output_forms={"replication_timing": "t"},
        x_axis=np.arange(5000.0), time_axis=np.arange(4000.0), assumptions={},
    )
    tracemalloc.start()
    try:
        options = mapping_plot_options(result, **{
            **_settings(), "plot_type": "Heatmap", "y_limits": None,
            "log_z": True,
        })
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 40_000_000  # Broadcasting would otherwise allocate 160 MB.
    assert options["series"][0]["data"] == [[0.0, 0.0, 4999.0, 3999.0]]
    assert len(options["series"]) == 1  # Temporal fields do not need a hover grid.
