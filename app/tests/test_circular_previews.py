"""Circular rendering, bounded navigation and shared style controls."""

import asyncio
import base64
import html
import json
import re
import shutil
import subprocess
from pathlib import Path

import nicegui
import numpy as np
import pytest
from nicegui.elements.upload import Upload
from nicegui.testing import user_simulation

from app.main import index
from app.preview_plotting import PreviewChartHandle, PreviewLine, build_preview_plot_options
from dnascape.workflows.circular import circular_preview_options
from dnascape.workflows.plotting import prepare_echarts_runtime


def options(kind="Line", **overrides):
    x = np.linspace(0., 10_000., 1001)
    arguments = dict(plot_type=kind, title="Profiles", x_label="Position (kb)", y_label="Timing (min)",
        color="#b03412", palette="viridis", histogram_bins=12, density=False,
        show_legend=True, show_colorbar=True, log_y=False, invert_y=False, y_limits=None,
        values=2.+np.sin(x/1000.), x_axis=x, series_label="Profile")
    if kind == "Line":
        arguments["lines"] = (PreviewLine("Observed", x, 2.+np.sin(x/1000.), "#2596be"),
                              PreviewLine("Fitted", x, 2.1+np.sin(x/1000.), "#b03412"))
    elif kind == "Heatmap":
        arguments.update(values=np.arange(800.).reshape(20,40), x_axis=np.arange(40.), y_axis=np.arange(20.)*5)
    arguments.update(overrides)
    result = build_preview_plot_options(**arguments)
    class Chart:
        def __init__(self, spec): self.options = spec
    handle = PreviewChartHandle(Chart(result), Chart({}))
    handle._apply_navigation()
    return result


def test_circular_tracks_keep_data_and_square_geometry():
    result = options()
    original = [s["data"] for s in result["series"]]
    circular_preview_options(result)
    assert result["dnascapeAspectRatio"] == 1.
    assert len(result["polar"]) == 2
    assert all(s["coordinateSystem"] == "polar" for s in result["series"])
    assert all(s["dnascapeCircularSourceData"] is source for s,source in zip(result["series"],original))
    result["xAxis"].update(min=4200.,max=4400.)
    circular_preview_options(result)
    assert all(4200 <= p[0] <= 4400 for s in result["series"] for p in s["data"])
    assert all(s["dnascapeCircularSourceData"] is source for s,source in zip(result["series"],original))
    assert result["angleAxis"][0]["min"] == 4200
    circular_preview_options(result, layout="Overlay")
    assert len(result["polar"]) == 1
    with pytest.raises(ValueError, match="heatmaps"):
        circular_preview_options(options("Heatmap"))


def test_shared_circular_style_and_defaults_in_analysis():
    async def scenario():
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-plotting").click()
            await user.should_see(marker="plotting-data-upload", retries=60)
            upload = next(iter(user.find(marker="plotting-data-upload").elements))
            await upload.handle_uploads([Upload.SmallFileUpload("profile.tsv","text/tab-separated-values",b"signal\n1\n2\n3\n4\n5\n")])
            await user.should_see("profile.tsv",retries=60)
            await user.should_see("Array size: 5",retries=60)
            user.find(marker="action-plot").click()
            await user.should_see(marker="plotting-result-plot",retries=60)
            def control(suffix):
                return next(e for e in user.client.elements.values() if f"plotting-{suffix}" in e._markers and not e.is_deleted)
            assert control("aspect-ratio").value == 2.5
            assert control("show-grid").value is False
            original_names = [s["name"] for s in control("result-plot").options["series"]]
            with user.client:
                control("reverse-order").set_value(True)
            await asyncio.sleep(0)
            assert [s["name"] for s in control("result-plot").options["series"]] == original_names[::-1]
            with user.client:
                control("view").set_value("Circular")
            for _ in range(6): await asyncio.sleep(0)
            chart = control("result-plot")
            assert chart.options["dnascapeCircular"]
            assert chart.options["dnascapeAspectRatio"] == 1.
            assert control("aspect-ratio").enabled is False
            with user.client:
                control("view").set_value("Default")
            for _ in range(6): await asyncio.sleep(0)
            assert control("aspect-ratio").enabled
            assert control("aspect-ratio").value == 2.5
    asyncio.run(scenario())


def test_mapping_stacked_quantities_and_hidden_plot_settings(monkeypatch):
    from app.mapping_service import MappingComparison, MappingResult
    from app.pages import mapping

    def mapped(request, report, cancel):
        observed, fitted = np.array([10., 20., 30.]), np.array([11., 19., 30.])
        return MappingResult(
            target="initiation_rate", target_key="initiation_rate", target_label="Initiation rate",
            method=request.route.method, output_form="x", output_dependence="space",
            primary_output="initiation_rate", outputs={"initiation_rate": np.array([.01, .02, .03]),
                "input:replication_timing": observed, "input:fork_speed": np.ones(3)*2,
                "replication_timing_fit": fitted},
            output_forms={"initiation_rate": "x", "input:replication_timing": "x",
                          "input:fork_speed": "x", "replication_timing_fit": "x"},
            x_axis=np.arange(3.), time_axis=np.array([]), region_labels=(), elapsed_seconds=.01,
            assumptions={}, comparisons={"replication_timing_fit": MappingComparison(
                "replication_timing", "replication_timing_fit", "x", observed, fitted)},
        )
    monkeypatch.setattr(mapping, "run_mapping", mapped)

    async def scenario():
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-mapping").click()
            upload = next(iter(user.find(marker="source-replication-timing-upload").elements))
            await upload.handle_uploads([Upload.SmallFileUpload("timing.csv", "text/csv", b"timing\n10\n20\n30\n")])
            await user.should_see(marker="mapping-input-replication-timing-detected-data", retries=60)
            user.find(marker="action-map").click()
            await user.should_see(marker="mapping-result-plot", retries=60)
            def control(suffix):
                return next(e for e in user.client.elements.values() if f"mapping-{suffix}" in e._markers and not e.is_deleted)
            choices = list(control("preview-output").options)
            assert choices.index("input:fork_speed") < choices.index("replication_timing_fit")
            with user.client:
                control("preview-output").set_value("all_plots")
            await user.should_see(marker="mapping-all-plot-2", retries=60)
            charts = [control("result-plot"), control("all-plot-1"), control("all-plot-2")]
            assert [len(c.options["series"]) for c in charts] == [1, 1, 2]
            assert "Initiation rate" in charts[0].options["yAxis"]["name"]
            assert "Fork speed" in charts[1].options["yAxis"]["name"]
            with user.client:
                control("view").set_value("Circular")
            await asyncio.sleep(0)
            charts = [control("result-plot"), control("all-plot-1"), control("all-plot-2")]
            assert all(c.options["dnascapeCircular"] and c.options["dnascapeAspectRatio"] == 1 for c in charts)
            assert [len(c.options["polar"]) for c in charts] == [1, 1, 2]
            with user.client:
                control("show-plot").set_value(False)
            await asyncio.sleep(0)
            await user.should_not_see(marker="mapping-result-plot")
            await user.should_not_see(marker="mapping-plot-settings")
    asyncio.run(scenario())


def test_circular_and_heatmap_gestures_in_browser(tmp_path):
    chrome = shutil.which("chrome") or shutil.which("chromium") or shutil.which("google-chrome")
    candidate = Path("C:/Program Files/Google/Chrome/Application/chrome.exe")
    if not chrome and candidate.exists(): chrome = str(candidate)
    if not chrome: pytest.skip("Chrome is required")
    root = Path(__file__).resolve().parents[2]
    specs = []
    for index, kind in enumerate(("Line", "Histogram", "Heatmap", "Line")):
        spec = options(kind, **({"secondary_y_label": "Second scale", "secondary_x_label": "Position"} if index == 3 else {}))
        if index == 0:
            spec["series"][0]["markArea"] = {"data": [[{"xAxis": 2000., "name": "Filter"}, {"xAxis": 3500.}]]}
            spec["series"][0]["areaStyle"] = {"opacity": .15, "origin": "start"}
        if kind != "Heatmap":
            circular_preview_options(spec)
            spec["dnascapeNavigation"]["polar"] = True
        specs.append(spec)
    runtime = Path(nicegui.__file__).parent/"elements/echart/dist/index.js"
    (tmp_path/"echarts.js").write_text(prepare_echarts_runtime(runtime),encoding="utf-8")
    probe = r'''
    (async()=>{
      const errors=[];window.addEventListener('error',e=>errors.push(e.message));
      const wait=ms=>new Promise(r=>setTimeout(r,ms));
      window.requestAnimationFrame=f=>setTimeout(()=>f(performance.now()),16);window.cancelAnimationFrame=clearTimeout;
      const element=document.getElementById('c1'),chart=echarts.init(element,null,{renderer:'canvas'});
      const convert=o=>{if(!o||typeof o!=='object')return;for(const k of Object.keys(o))
        if(k.startsWith(':')){o[k.slice(1)]=Function('return ('+o[k]+')')();delete o[k];}else convert(o[k]);};
      const component={chart,options:{},$emit:()=>{},update_chart:()=>chart.setOption(component.options,{notMerge:true})};
      window.getElement=()=>component;
      const key=(on)=>window.dispatchEvent(new KeyboardEvent(on?'keydown':'keyup',{code:'ControlLeft',ctrlKey:on,bubbles:true}));
      const mouse=(type,x,y)=>{const box=element.getBoundingClientRect();
        (type==='mousedown'?element:window).dispatchEvent(new MouseEvent(type,{button:0,buttons:type==='mouseup'?0:1,
        clientX:box.left+x,clientY:box.top+y,ctrlKey:true,bubbles:true,cancelable:true}));};
      const assert=(ok,msg)=>{if(!ok)throw new Error(msg);};
      const results=[],images=[];
      try {
        for(const [index,spec] of window.specs.entries()){
          convert(spec);component.options=spec;component.update_chart();await wait(100);
          if(index<2)images.push(chart.getDataURL({type:'png',pixelRatio:1,backgroundColor:'#fff'}));
          key(true);
          if(index!==2){
            const p=chart.getModel().getComponent('polar',spec.polar.length-1).coordinateSystem;
            const r=Math.max(...p.getRadiusAxis().getExtent()),x=p.cx+r*.7,y=p.cy;
            const before=chart.getOption().xAxis[0];const old=[before.min,before.max];
            const box=element.getBoundingClientRect();element.dispatchEvent(new WheelEvent('wheel',{
              clientX:box.left+x,clientY:box.top+y,deltaY:-250,ctrlKey:true,bubbles:true,cancelable:true}));
            await wait(50);const after=chart.getOption().xAxis[0];
            assert(after.max-after.min<old[1]-old[0],'Circular zoom did not narrow the genomic window');
            assert(after.min>=spec.dnascapeNavigation.axes.x.min&&after.max<=spec.dnascapeNavigation.axes.x.max,'Circular zoom escaped bounds');
            assert(chart.getZr().storage.getDisplayList().length>10,'Circular chart is empty');
            const data=chart.getOption().series[0].data;
            assert(data.every(v=>v[0]>=after.min&&v[0]<=after.max),'Out-of-window data wrapped around the circle');
            if(index===3)assert(chart.getOption().radiusAxis[1].min===chart.getOption().yAxis[1].min,'Secondary circular scale did not zoom');
            mouse('mousedown',p.cx,p.cy-r*.6);mouse('mousemove',p.cx+r*.5,p.cy-r*.3);
            mouse('mouseup',p.cx+r*.5,p.cy-r*.3);await wait(40);
          }else{
            const rect=chart.getModel().getComponent('grid',0).coordinateSystem.getRect();
            const before=chart.getOption(),xs=before.xAxis[0].max-before.xAxis[0].min,ys=before.yAxis[0].max-before.yAxis[0].min;
            mouse('mousedown',rect.x+rect.width*.2,rect.y+rect.height*.2);
            mouse('mousemove',rect.x+rect.width*.8,rect.y+rect.height*.8);
            const overlay=element.querySelector('.dn-plot-range-selection');
            assert(parseFloat(overlay.style.height)>rect.height*.5,'Heatmap brush is not a box');
            mouse('mouseup',rect.x+rect.width*.8,rect.y+rect.height*.8);await wait(40);
            const after=chart.getOption();assert(after.xAxis[0].max-after.xAxis[0].min<xs,'Box did not zoom X');
            assert(after.yAxis[0].max-after.yAxis[0].min<ys,'Box did not zoom Y');
          }
          key(false);results.push(index);
        }
      }catch(e){errors.push(e.stack||e.message);}
      document.getElementById('result').textContent=JSON.stringify({errors,results,images});
    })();'''
    document=tmp_path/"circular.html"
    document.write_text('<!doctype html><meta charset="utf-8"><body><div id="c1" class="nicegui-echart dn-plot-navigation" '
        'style="width:640px;height:640px"></div><pre id="result">pending</pre>'
        '<script src="echarts.js"></script><script>window.specs='+json.dumps(specs)+';</script>'
        f'<script src="{(root/"app/static/plot_navigation.js").as_uri()}"></script>'
        '<script>'+probe+'</script></body>',encoding="utf-8")
    screenshot=root/".tmp-polish/circular-browser.png"
    run=subprocess.run([chrome,"--headless","--disable-gpu","--no-first-run","--no-default-browser-check",
        "--allow-file-access-from-files","--window-size=850,850","--virtual-time-budget=6000",
        f"--user-data-dir={tmp_path/'browser'}",f"--screenshot={screenshot}","--dump-dom",document.as_uri()],
        capture_output=True,text=True,encoding="utf-8",timeout=30,check=True)
    found=re.search(r'<pre id="result">([^<]+)</pre>',run.stdout)
    assert found and found[1]!="pending",run.stderr[-1000:]
    result=json.loads(html.unescape(found[1]))
    for index, data in enumerate(result.pop("images", [])):
        (root/f".tmp-polish/circular-{index}.png").write_bytes(base64.b64decode(data.split(",", 1)[1]))
    (root/".tmp-polish/circular-errors.json").write_text(json.dumps(result),encoding="utf-8")
    assert not result["errors"],result
    assert result["results"]==[0,1,2,3]
