"""Histogram hover follows rendered bars and their actual bin intervals."""

import json
import re
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dnascape.workflows.plotting import prepare_echarts_runtime, write_plot_html
from dnascape.workflows.preview import PreviewLine, build_preview_plot_options


def histogram_options(**overrides):
    settings = dict(
        plot_type="Histogram", title="Replicon length", x_label="Length (kb)",
        y_label="Count", color="#b03412", palette="viridis", histogram_bins=5,
        density=False, show_legend=False, show_colorbar=False, log_y=False,
        invert_y=False, y_limits=None, series_label="Replicon length",
        values=np.linspace(180., 185., 101), histogram_x_limits=(180., 185.),
    )
    settings.update(overrides)
    return build_preview_plot_options(**settings)


@pytest.mark.parametrize("grouped", [False, True])
def test_histograms_use_single_item_hover_including_combined_datasets(grouped):
    lines = tuple(PreviewLine(name, np.arange(101.), np.linspace(180., 185., 101), color)
                  for name, color in (("First", "#b03412"), ("Second", "#2596be")))
    options = histogram_options(lines=lines, histogram_grouped=grouped)
    assert options["tooltip"]["trigger"] == "item"
    assert options["tooltip"]["axisPointer"]["type"] == "none"
    assert options["axisPointer"]["triggerEmphasis"] is False
    assert all(len(row) == 4 for series in options["series"] for row in series["data"])


def test_kde_keeps_curve_hover_and_noninteractive_histograms_remain_disabled():
    kde = histogram_options(density=True)
    assert kde["tooltip"]["trigger"] == "axis"
    assert kde["tooltip"]["axisPointer"]["type"] == "line"
    assert histogram_options(interactive=False)["tooltip"]["show"] is False


def test_real_browser_histogram_guides_ranges_and_single_count(tmp_path):
    playwright = pytest.importorskip("playwright.sync_api")
    nicegui = pytest.importorskip("nicegui")
    chrome = shutil.which("google-chrome") or shutil.which("chromium")
    windows_chrome = Path("C:/Program Files/Google/Chrome/Application/chrome.exe")
    if chrome is None and windows_chrome.exists():
        chrome = str(windows_chrome)
    if chrome is None:
        pytest.skip("Chrome/Chromium is required for real histogram hover checks")
    runtime = prepare_echarts_runtime(Path(nicegui.__file__).parent / "elements/echart/dist/index.js")
    document = write_plot_html(histogram_options(), tmp_path / "histogram.html", echarts_source=runtime)
    lines = tuple(PreviewLine(name, np.arange(101.), np.linspace(180., end, 101), color)
                  for name, end, color in (("First", 185., "#b03412"), ("Second", 184.5, "#2596be")))
    cases = {
        "integer bins": histogram_options(),
        "more bins": histogram_options(histogram_bins=10),
        "custom width and short final bin": histogram_options(histogram_bin_size=.7),
        "separated bars": histogram_options(histogram_touching=False),
        "reversed axes": histogram_options(invert_x=True, invert_y=True),
        "logarithmic axes": histogram_options(log_x=True, log_y=True),
        "overlaid datasets": histogram_options(lines=lines),
        "grouped datasets": histogram_options(lines=lines, histogram_grouped=True),
        "independent axes": histogram_options(lines=lines, secondary_x_label="Second X",
                                               secondary_y_label="Second Y", secondary_invert_x=True),
        "fractional bins": histogram_options(values=np.linspace(.1, 1., 101),
                                              histogram_x_limits=(.1, 1.), histogram_bin_size=.04),
        "tiny bins": histogram_options(values=np.linspace(1e-7, 6e-7, 101),
                                       histogram_x_limits=(1e-7, 6e-7), histogram_bin_size=1e-7),
        "negative bins": histogram_options(values=np.linspace(-5., 0., 101),
                                           histogram_x_limits=(-5., 0.)),
    }
    from app.preview_plotting import PreviewChartHandle
    app_options = histogram_options()
    handle = PreviewChartHandle(SimpleNamespace(options=app_options), SimpleNamespace(options={}))
    handle._apply_navigation()
    handle._apply_guides()
    cases["app navigation and guides"] = app_options
    clipped = histogram_options()
    clipped["xAxis"].update(min=180.4, max=184.8)
    cases["clipped rectangles"] = clipped
    probe = r"""async cases => {
      const pause=ms=>new Promise(resolve=>setTimeout(resolve,ms)), results=[];
      const chart=echarts.getInstanceByDom(document.getElementById('chart'));
      const element=chart.getDom(),component=getElement();
      const convert=value=>{
        if(Array.isArray(value))return value.map(convert);
        if(value&&typeof value==='object')return Object.fromEntries(Object.entries(value).map(([k,v])=>
          k.startsWith(':')?[k.slice(1),Function('return ('+v+')')()]:[k,convert(v)]));
        return value;
      };
      const move=p=>{
        const canvas=element.querySelector('canvas'),box=element.getBoundingClientRect();
        for(const type of ['pointermove','mousemove'])canvas.dispatchEvent(new MouseEvent(type,{
          clientX:box.left+p[0],clientY:box.top+p[1],bubbles:true}));
      };
      const visible=selector=>{const e=element.querySelector(selector);return e&&e.style.display!=='none';};
      for(const [name,spec] of Object.entries(cases)){
        for(const key of Object.keys(component.options))delete component.options[key];
        Object.assign(component.options,convert(spec));
        chart.setOption(component.options,{notMerge:true});
        move([1,1]);await pause(25);
        if(visible('.dn-histogram-hover-guide'))throw Error(name+': guide survived redraw');
        const bars=[];
        chart.getModel().eachSeries(series=>{
          const data=series.getData(),count=data.count();
          for(const index of new Set([0,Math.floor(count/2),count-1])){
            const bar=data.getItemGraphicEl(index);
            if(bar?.shape?.width>1&&bar.shape.height>1)bars.push({series,index,bar});
          }
        });
        for(const target of bars)for(const fraction of [.15,.85]){
          const s=target.bar.shape,p=target.bar.transformCoordToGlobal(s.x+s.width*fraction,s.y+s.height*.75);
          const hit=chart.getZr().handler.findHover(...p).target;
          let hovered;
          chart.getModel().eachSeries(series=>series.getData().each(index=>{
            if(series.getData().getItemGraphicEl(index)===hit)hovered={series,index,bar:hit};
          }));
          if(!hovered)throw Error(name+': test missed rectangle');
          move(p);await pause(25);
          const guide=element.querySelector('.dn-histogram-hover-guide');
          const label=element.querySelector('.dn-histogram-hover-position');
          if(!visible('.dn-histogram-hover-guide'))throw Error(name+': guide missing');
          const shape=hovered.bar.shape,expected=hovered.bar.transformCoordToGlobal(shape.x+shape.width/2,shape.y)[0];
          const line=guide.querySelector('line');
          if(Math.abs(Number(line.getAttribute('x1'))-expected)>1e-6||line.getAttribute('x1')!==line.getAttribute('x2'))
            throw Error(name+': guide not through midpoint');
          if(!line.getAttribute('stroke-dasharray'))throw Error(name+': guide not dashed');
          const nativeGuides=chart.getZr().storage.getDisplayList(true).filter(e=>
            e.type==='line'&&e.style.stroke==='#b8b0aa'&&e.style.lineDash?.length);
          if(nativeGuides.length)throw Error(name+': edge guide still visible');
          const row=hovered.series.getData().getRawDataItem(hovered.index);
          const counts=[...element.querySelectorAll('b')].filter(e=>getComputedStyle(e).visibility==='visible');
          if(counts.length!==1||Number(counts[0].textContent)!==row[2])throw Error(name+': wrong tooltip '+counts.map(e=>e.textContent));
          results.push({name,label:label.textContent,left:row[0],right:row[1]});
        }
        move([1,1]);await pause(25);
        if(visible('.dn-histogram-hover-guide')||visible('.dn-histogram-hover-position'))throw Error(name+': hover stuck after leaving bar');
      }
      for(const disabled of [false,true]){
        const spec=convert(cases['integer bins']);spec.tooltip.show=!disabled;
        for(const s of spec.series){s.silent=disabled;s.emphasis.disabled=disabled;}
        Object.assign(component.options,spec);chart.setOption(component.options,{notMerge:true});
        move([1,1]);await pause(25);
        const s=chart.getModel().getSeriesByIndex(0).getData().getItemGraphicEl(1).shape;
        move([s.x+s.width*.2,s.y+s.height*.5]);await pause(25);
        if(visible('.dn-histogram-hover-guide')===disabled)throw Error('Hover checkbox ignored');
      }
      return results;
    }"""
    with playwright.sync_playwright() as driver:
        browser = driver.chromium.launch(executable_path=chrome, headless=True)
        try:
            page = browser.new_page(viewport={"width": 1000, "height": 650})
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(document.as_uri())
            page.wait_for_function("window.__dnascapePlotNavigation")
            page.wait_for_timeout(75)
            results = page.evaluate(probe, cases)
            combined = histogram_options(lines=lines, histogram_grouped=True,
                                         show_legend=True, show_histogram_statistics=True)
            combined_document = write_plot_html(combined, tmp_path / "combined.html", echarts_source=runtime)
            page.goto(combined_document.as_uri())
            page.wait_for_timeout(75)
            selected = page.evaluate("""() => {
              const chart=echarts.getInstanceByDom(document.getElementById('chart'));
              chart.trigger('click',{componentType:'series',seriesIndex:1});
              return chart.getOption().graphic[0].elements.some(e=>e.style?.text?.includes('Second'));
            }""")
            assert selected, "Downloaded histograms must retain dataset selection"
            assert not errors
        finally:
            browser.close()
    assert {item["name"] for item in results} == set(cases)
    assert "180-181" in {item["label"] for item in results if item["name"] == "integer bins"}
    number = r"(-?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?)"
    for item in results:
        match = re.fullmatch(number + "-" + number, item["label"])
        assert match, json.dumps(item)
        width = item["right"] - item["left"]
        assert float(match[1]) == pytest.approx(item["left"], rel=0, abs=width*.001)
        assert float(match[2]) == pytest.approx(item["right"], rel=0, abs=width*.001)
