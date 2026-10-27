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
    radii = [list(map(lambda v: float(v.rstrip("%")), track["radius"])) for track in result["polar"]]
    assert all(outer-inner == 7. and inner >= 64 for inner,outer in radii)
    assert all(s["coordinateSystem"] == "polar" for s in result["series"])
    assert all(s["dnascapeCircularSourceData"] is source for s,source in zip(result["series"],original))
    result["xAxis"].update(min=4200.,max=4400.)
    circular_preview_options(result)
    assert all(s["data"] == source for s, source in zip(result["series"], original))
    assert all(s["dnascapeCircularSourceData"] is source for s,source in zip(result["series"],original))
    assert result["angleAxis"][0]["min"] == 0
    assert result["angleAxis"][0]["max"] == 10000
    assert result["polar"] != result["dnascapeCircularBasePolar"]
    circular_preview_options(result, layout="Overlay")
    assert len(result["polar"]) == 1
    circular_preview_options(result, layout="Overlay", track_height=16.)
    assert [float(v.rstrip("%")) for v in result["dnascapeCircularBasePolar"][0]["radius"]] == [64.,80.]
    circular_preview_options(result, track_height=1.)
    assert all(float(track["radius"][1][:-1])-float(track["radius"][0][:-1]) == 1.
               for track in result["dnascapeCircularBasePolar"])
    with pytest.raises(ValueError, match="Track height"):
        circular_preview_options(result, track_height=.99)
    circular_preview_options(result, inner_radius=0.)
    assert result["dnascapeCircularBasePolar"][0]["radius"][0] == "0.0%"
    with pytest.raises(ValueError, match="Inner radius"):
        circular_preview_options(result, inner_radius=79.)
    with pytest.raises(ValueError, match="heatmaps"):
        circular_preview_options(options("Heatmap"))


def test_single_point_profile_still_uses_circular_coordinates():
    spec=options(values=[2.],x_axis=[0.],lines=())
    circular_preview_options(spec)
    assert spec['dnascapeCircular']
    assert len(spec['polar'])==1
    assert spec['series'][0]['coordinateSystem']=='polar'
    assert spec['series'][0]['data']==[[0.,2.]]


@pytest.mark.parametrize("kind", ["Line", "Histogram", "ensemble"])
def test_circular_zoom_keeps_surrounding_curve_samples_to_the_frame_edges(kind):
    spec = ensemble_options() if kind == "ensemble" else options(kind)
    circular_preview_options(spec)
    original = [list(series["data"]) for series in spec["series"]]
    axes = spec["xAxis"] if isinstance(spec["xAxis"], list) else [spec["xAxis"]]
    for axis in axes:
        span = axis["max"] - axis["min"]
        lower = axis["min"]
        axis.update(min=lower + .19 * span, max=lower + .28 * span)
    circular_preview_options(spec)
    for series, data in zip(spec["series"], original, strict=True):
        assert series["data"] == data
    assert float(spec["polar"][0]["radius"][1].rstrip("%")) > 100.


@pytest.mark.parametrize("logarithmic", [False, True])
@pytest.mark.parametrize("with_band", [False, True])
def test_seam_crossing_window_keeps_both_ends_and_refines_native_positions(logarithmic, with_band):
    from dnascape.workflows.circular import circular_window_parts
    from dnascape.workflows.preview import PreviewBand
    x = np.geomspace(1., 10000., 10001) if logarithmic else np.arange(10001.)
    values = 9. + np.sin(np.arange(x.size) / 83.)
    spec = options(x_axis=x, values=values, lines=(), log_x=logarithmic,
                   bands=(PreviewBand("SD", x, values-.1, values+.1, "#b03412"),) if with_band else ())
    circular_preview_options(spec)

    class Chart:
        def __init__(self, options): self.options = options

    handle = PreviewChartHandle(Chart(spec), Chart({}))
    handle._apply_navigation()
    axis = handle._navigation.axes["x"]
    convert, restore = (np.log, np.exp) if logarithmic else (float, float)
    low, high = convert(axis["min"]), convert(axis["max"])
    window = (restore(low + .98 * (high - low)), restore(low + 1.02 * (high - low)))
    assert handle._commit_navigation(dict(context=handle._navigation.context, token=handle._navigation.token,
        sequence=1, ranges={"x": window}))
    handle._apply_navigation()
    np.testing.assert_allclose(handle._navigation.ranges["x"], window)
    parts = circular_window_parts(axis, window)
    assert len(parts) == 2
    jobs = handle._refinement_plan()
    assert jobs and len(jobs[0][3].parts) == 2
    assert handle._install_refinements(handle._compute_refinements(jobs))
    circular_preview_options(spec)
    line = next(series for series in spec["series"] if series["type"] == "line")
    points = np.asarray(line["data"])
    assert np.count_nonzero(points[:, 0] >= parts[0][0]) >= 180
    assert np.count_nonzero(points[:, 0] <= parts[1][1]) >= 180
    assert float(spec["polar"][0]["radius"][1].rstrip("%")) > 100.
    assert spec["graphic"][-1]["style"]["font"] == "11px sans-serif"
    if with_band:
        band = next(series for series in spec["series"] if series["type"] == "custom")
        assert all(a[3] <= b[0] for a, b in zip(band["data"], band["data"][1:]))


def test_shared_circular_style_and_defaults_in_analysis(monkeypatch):
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
            assert control("circular-track-height").value == 8.
            assert control("circular-inner-radius").value == 64.
            assert not control("circular-combine").visible
            with user.client:
                control("circular-inner-radius").set_value(45.)
            await asyncio.sleep(0)
            assert control("result-plot").options["dnascapeCircularInnerRadius"] == 45.
            with user.client:
                control("circular-track-height").set_value(12.)
            await asyncio.sleep(0)
            chart = control("result-plot")
            assert chart.options["dnascapeCircularTrackHeight"] == 12.
            handle = chart._dn_preview_handle
            legend = dict(chart.options["legend"])
            monkeypatch.setattr(handle, "_refinement_plan", lambda **_kwargs: [])
            with user.client:
                await handle.accept_navigation_async({"context":handle._navigation.context,
                    "token":handle._navigation.token,"ranges":{"x":(1.,2.)}})
            assert handle.shell._style["max-width"] == "30rem"
            assert (chart.options["xAxis"]["min"],chart.options["xAxis"]["max"]) == (1., 2.)
            assert (chart.options["angleAxis"][0]["min"],chart.options["angleAxis"][0]["max"]) == (0., 4.)
            assert chart.options["polar"] != chart.options["dnascapeCircularBasePolar"]
            assert all(1 <= tick <= 2 for tick in chart.options["angleAxis"][0]["axisLabel"]["customValues"])
            assert chart.options["legend"] == legend
            assert control("x-minimum").value == 1.
            assert control("x-maximum").value == 2.
            assert control("legend-position").visible
            assert control("legend-position").value == "Top right"
            for position, horizontal, vertical in (("Top left", "left", "top"), ("Bottom left", "left", "bottom"), ("Bottom right", "right", "bottom"), ("Top right", "right", "top")):
                with user.client:
                    control("legend-position").set_value(position)
                for _ in range(6): await asyncio.sleep(0)
                placed = control("result-plot").options["legend"]
                assert placed[horizontal] == 8 and placed[vertical] == 2
                assert placed["right" if horizontal == "left" else "left"] is None
                assert placed["bottom" if vertical == "top" else "top"] is None
            with user.client:
                control("show-legend").set_value(False)
            for _ in range(6): await asyncio.sleep(0)
            assert not control("legend-position").visible
            with user.client:
                control("show-legend").set_value(True)
            for _ in range(6): await asyncio.sleep(0)
            assert control("legend-position").visible
            with user.client:
                control("circular-track-height").set_value(1.)
            await asyncio.sleep(0)
            assert control("result-plot").options["dnascapeCircularTrackHeight"] == 1.
            assert control("result-plot")._dn_preview_handle.shell._style["max-width"] == "30rem"
            with user.client:
                control("show-hover").set_value(False)
            await asyncio.sleep(0)
            chart = control("result-plot")
            assert chart.options["tooltip"]["show"] is False
            assert all(axis["axisPointer"]["show"] is False for axis in chart.options["angleAxis"])
            with user.client:
                control("show-hover").set_value(True)
            await asyncio.sleep(0)
            assert control("result-plot").options["tooltip"]["show"] is True
            with user.client:
                control("view").set_value("Default")
            for _ in range(6): await asyncio.sleep(0)
            assert control("aspect-ratio").enabled
            assert control("aspect-ratio").value == 2.5
    asyncio.run(scenario())


def test_multiple_datasets_offer_combine_and_keep_inner_radius_in_both_layouts():
    async def scenario():
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-plotting").click()
            def control(suffix):
                return next(e for e in user.client.elements.values() if f"plotting-{suffix}" in e._markers and not e.is_deleted)
            await control("data-upload").handle_uploads([
                Upload.SmallFileUpload("one.tsv","text/plain",b"A\n1\n2\n3\n"),
                Upload.SmallFileUpload("two.tsv","text/plain",b"B\n3\n2\n1\n"),
            ])
            await user.should_see(marker="plotting-detected-data-2",retries=60)
            user.find(marker="action-plot").click()
            await user.should_see(marker="plotting-result-plot",retries=60)
            with user.client:
                control("view").set_value("Circular")
            for _ in range(8):await asyncio.sleep(0)
            assert control("circular-combine").visible
            assert not control("circular-combine").value
            assert len(control("result-plot").options["polar"]) == 2
            height=control("circular-track-height")
            radius=control("circular-inner-radius")
            assert radius.parent_slot is height.parent_slot
            assert height.parent_slot.children.index(height)<height.parent_slot.children.index(radius)
            with user.client:
                radius.set_value(35.)
                control("circular-combine").set_value(True)
            for _ in range(8):await asyncio.sleep(0)
            assert len(control("result-plot").options["polar"]) == 1
            assert control("result-plot").options["dnascapeCircularInnerRadius"] == 35.
            with user.client:
                control("circular-combine").set_value(False)
            for _ in range(8):await asyncio.sleep(0)
            assert len(control("result-plot").options["polar"]) == 2
            assert control("result-plot").options["dnascapeCircularInnerRadius"] == 35.
    asyncio.run(scenario())


def ensemble_options():
    from app.pages.simulation import _result_plot_options
    from dnascape.workflows.simulation import ConstantSimulationResult
    x = np.arange(160.)
    mean = 8.+np.sin(x/15.)
    profiles = np.stack([mean+1.5*np.cos(x/8.),mean-1.5*np.cos(x/8.),mean])
    result = ConstantSimulationResult(position_kb=x, replication_timing=mean,
        elapsed_seconds=.01,replicates=3,geometry="periodic",time_grid=np.array([]),
        outputs={"replication_timing":mean},replication_time_profiles=profiles,
        replication_timing_std=np.std(profiles,axis=0))
    return _result_plot_options(result,output_label="Replication timing",title="",series_label="",
        x_axis_label="Position (kb)",y_axis_label="Time (min)",color="#b03412",palette="viridis",
        view_mode="Circular",histogram_bins=12,density=False,show_legend=True,show_colorbar=False,
        log_y=False,invert_y=False,y_limits=None,show_individual_profiles=True,
        individual_profile_count=2,show_standard_deviation=True)


def test_circular_ensemble_shares_the_mean_track_and_keeps_uncertainty_values():
    spec = ensemble_options()
    original = [list(s["data"]) for s in spec["series"]]
    circular_preview_options(spec)
    assert spec["dnascapeCircularDatasetCount"] == 1
    assert len(spec["polar"]) == 1
    assert len(spec["series"]) == 4  # Mean, two retained profiles, and the SD band.
    assert {s["polarIndex"] for s in spec["series"]} == {0}
    assert [list(s["data"]) for s in spec["series"]] == original
    assert spec["series"][0]["silent"]
    assert not spec["series"][-1]["silent"]
    assert spec["tooltip"]["show"]


def test_simulation_distributions_do_not_offer_circular_view():
    from app.pages import simulation
    async def scenario():
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-simulation").click()
            def control(suffix):
                return next(e for e in user.client.elements.values() if f"simulation-{suffix}" in e._markers and not e.is_deleted)
            with user.client:
                control("view").set_value("Circular")
                control("track-time-outputs").set_value(True)
            assert "Circular" in control("view").options
            for label in {**simulation._DISTRIBUTION_OUTPUTS, **simulation._TIME_DISTRIBUTION_OUTPUTS}:
                with user.client:
                    control("output").set_value(label)
                await asyncio.sleep(0)
                assert "Circular" not in control("view").options
                assert control("view").value == "Default"
            with user.client:
                control("output").set_value("Replication timing")
            assert "Circular" in control("view").options
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
    spec = ensemble_options()
    class Chart:
        def __init__(self, spec): self.options=spec
    handle = PreviewChartHandle(Chart(spec),Chart({}))
    handle._apply_navigation()
    circular_preview_options(spec)
    spec["dnascapeNavigation"]["polar"] = True
    specs.append(spec)
    spec=options(values=[2.],x_axis=[0.],lines=())
    circular_preview_options(spec)
    spec["dnascapeNavigation"]["polar"]=True
    specs.append(spec)
    spec=options(x_axis=np.geomspace(1.,10000.,1001), log_x=True, log_y=True,
                 invert_x=True, invert_y=True, lines=())
    circular_preview_options(spec, layout="Overlay", track_height=1.)
    spec["dnascapeNavigation"]["polar"]=True
    specs.append(spec)
    spec=options(interactive=False)
    circular_preview_options(spec)
    spec["dnascapeNavigation"]["polar"]=True
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
        (type==='mouseup'?window:element).dispatchEvent(new MouseEvent(type,{button:0,buttons:type==='mouseup'?0:1,
        clientX:box.left+x,clientY:box.top+y,ctrlKey:true,bubbles:true,cancelable:true}));};
      const assert=(ok,msg)=>{if(!ok)throw new Error(msg);};
      const results=[],images=[];
      try {
        for(const [index,spec] of window.specs.entries()){
          if(index===4)document.body.style.setProperty('--dn-accent','#2a7b62');
          convert(spec);component.options=spec;component.update_chart();await wait(100);
          if(index<2||index===4||index===6)images.push(chart.getDataURL({type:'png',pixelRatio:1,backgroundColor:'#fff'}));
          key(true);
          if(index!==2){
            const p=chart.getModel().getComponent('polar',spec.polar.length-1).coordinateSystem;
            const extent=p.getRadiusAxis().getExtent(),r=(extent[0]+extent[1])/2,x=p.cx+r,y=p.cy;
            const before=chart.getOption().xAxis[0];const old=[before.min,before.max];
            const spanOf=axis=>spec.dnascapeNavigation.axes.x.log?Math.log(axis.max/axis.min):axis.max-axis.min;
            const oldWidth=chart.getWidth();
            const curveStyles=()=>JSON.stringify(chart.getModel().getSeries().map(series=>({
              name:series.get('name'),line:series.get('lineStyle'),item:series.get('itemStyle'),area:series.get('areaStyle'),
              visual:series.getData().getVisual('style'),legend:series.getData().getVisual('legendLineStyle')})));
            const originalStyles=curveStyles();
            const axisLabel=()=>chart.getZr().storage.getDisplayList().filter(e=>String(e.style?.text||'').includes('Position (kb)'))
              .map(e=>[e.style.text,e.style.font,Array.from(e.getComputedTransform()||[])]);
            const originalLabel=JSON.stringify(axisLabel());
            assert(axisLabel().length>0,'Circular axis title did not render');
            const legendBox=()=>{const model=chart.getModel().getComponent('legend');
              const group=chart.getViewOfComponentModel(model).group;
              const rect=group.getBoundingRect().clone();rect.applyTransform(group.getComputedTransform());
              return [rect.x,rect.y,rect.width,rect.height];};
            if(index===0){
              for(const size of [240,320,480,640]){
                element.style.width=size+'px';element.style.height=size+'px';chart.resize();await wait(30);
                const circle=chart.getModel().getComponent('polar',spec.polar.length-1).coordinateSystem;
                const box=legendBox();
                assert(box[0]>=-1&&box[0]+box[2]<=size+1,'Circular legend escaped the preview: '+JSON.stringify([size,box]));
                assert(box[1]+box[3]<circle.cy-Math.max(...circle.getRadiusAxis().getExtent()),'Small circular legend overlaps the circle');
              }
            }
            const legend=legendBox();
            const legendStable=()=>legendBox().every((value,i)=>Math.abs(value-legend[i])<.5);
            assert(legendBox()[1]+legendBox()[3]<p.cy-Math.max(...extent),'Legend overlaps the circle');
            assert(p.cx===chart.getWidth()/2&&p.cy===chart.getHeight()/2,'Circle is not centred');
            if(index===0||index===3||index===4||index===6||index===7){
              const series=chart.getModel().getSeries().find(s=>s.get('type')==='line'&&(index===7||!s.get('silent')));
              const data=series.getData(),sampleIndex=Math.floor(data.count()/3),sample=data.getRawDataItem(sampleIndex);
              const point=series.coordinateSystem.dataToPoint([sample[1],sample[0]]);
              mouse('mousemove',point[0],point[1]);
              await wait(40);
              const tooltip=element.querySelector('.dn-circular-tooltip');
              if(index===7){
                assert(!tooltip||tooltip.style.display==='none','Disabled circular hover still shows readouts');
                assert(chart.getModel().getComponent('angleAxis',0).get('axisPointer.show')===false,'Disabled circular hover still shows guides');
              }else{
                assert(tooltip&&tooltip.style.display!=='none'&&tooltip.querySelector('b'),'Circular hover values are missing');
                const guide=element.querySelector('.dn-circular-guide');
                assert(guide,'Circular hover guide is missing');
                const readout=element.querySelector('.dn-circular-position');
                assert(readout&&readout.style.display!=='none'&&element.querySelectorAll('.dn-circular-position').length===1,'Circular position readout is missing or duplicated');
                assert(getComputedStyle(readout).fontSize==='11px'&&getComputedStyle(readout).transform==='none','Circular position text is scaled');
                const box=readout.getBoundingClientRect(),origin=element.getBoundingClientRect();
                assert(Math.hypot(box.left+box.width/2-origin.left-point[0],box.top+box.height/2-origin.top-point[1])<80,'Circular readout is not next to the dot');
                const line=guide.querySelector('line');
                assert(+line.getAttribute('x1')===p.cx&&+line.getAttribute('y1')===p.cy,'The dashed guide does not start at the centre');
                assert(getComputedStyle(guide).zIndex==='1000'&&line.getAttribute('stroke-dasharray')==='4 4','The dashed guide is covered by the curve');
                assert(guide.getBoundingClientRect().width===chart.getWidth(),'The guide is scaled differently from the circle');
                assert(guide.getBoundingClientRect().height<=chart.getHeight()-24,'The guide can draw across the fixed axis title');
                assert(guide.querySelector('circle'),'Circular hover points are missing');
                mouse('mousemove',p.cx+(point[0]-p.cx)*.2,p.cy+(point[1]-p.cy)*.2);await wait(40);
                assert(element.querySelector('.dn-circular-tooltip')?.style.display!=='none','Inner-circle hover is missing');
                const valueBox=element.querySelector('.dn-circular-tooltip').getBoundingClientRect();
                assert(valueBox.bottom<=origin.top+chart.getHeight()-26,'The floating readout overlaps the fixed axis title');
                assert(chart.getOption().angleAxis.every(a=>a.axisPointer.show===false),'Native angle pointers add duplicate labels');
                assert(chart.getOption().xAxis.every(a=>a.axisPointer.show===false)&&chart.getOption().yAxis.every(a=>a.axisPointer.show===false),'Hidden Cartesian pointers add duplicate labels');
              }
            }
            const box=element.getBoundingClientRect();element.dispatchEvent(new WheelEvent('wheel',{
              clientX:box.left+x,clientY:box.top+y,deltaY:-250,ctrlKey:true,bubbles:true,cancelable:true}));
            await wait(50);const after=chart.getOption().xAxis[0];
            assert(after.max-after.min<old[1]-old[0],'Circular wheel zoom did not refine the genomic window');
            assert(curveStyles()===originalStyles,'Wheel zoom temporarily changed the curve style: '+JSON.stringify([originalStyles,curveStyles()]));
            assert(chart.getWidth()===oldWidth,'Circular zoom resized the diagram');
            assert(JSON.stringify(axisLabel())===originalLabel,'Circular axis title changed font or transform during zoom');
            assert(legendStable(),'Circular zoom moved the legend: '+JSON.stringify([legend,legendBox()]));
            assert(after.min>=spec.dnascapeNavigation.axes.x.min&&after.max<=spec.dnascapeNavigation.axes.x.max,'Circular zoom escaped bounds');
            assert(chart.getZr().storage.getDisplayList().length>10,'Circular chart is empty');
            const data=chart.getOption().series[0].data;
            assert(data.length===spec.series[0].dnascapeCircularSourceData.length,'Zoom cropped the surrounding circular curve');
            for(const series of chart.getModel().getSeries()){
              const clip=chart.getViewOfSeriesModel(series).group.getClipPath();
              assert(clip?.type==='rect'&&clip.shape.width===chart.getWidth()&&clip.shape.height===chart.getHeight()-28,
                'Circular data is not clipped to the plot frame above the fixed caption');
            }
            if(index===3)assert(chart.getOption().radiusAxis[1].min===chart.getOption().yAxis[1].min,'Secondary circular scale did not zoom');
            const beforeBrush=chart.getOption(),span=spanOf(beforeBrush.xAxis[0]);
            const valueRanges=JSON.stringify(beforeBrush.yAxis.map(axis=>[axis.min,axis.max]));
            const camera=chart.getModel().getComponent('polar',0).coordinateSystem;
            const ring=camera.getRadiusAxis().getExtent(),middle=(ring[0]+ring[1])/2;
            const full=spec.dnascapeNavigation.axes.x,log=full.log;
            const forward=v=>log?Math.log(v):v;
            const low=forward(beforeBrush.xAxis[0].min),high=forward(beforeBrush.xAxis[0].max);
            const onCircle=f=>{
              let angle=(low+f*(high-low)-forward(full.min))/(forward(full.max)-forward(full.min));
              if(full.inverse)angle=1-angle;angle*=2*Math.PI;
              return [camera.cx+Math.sin(angle)*middle,camera.cy-Math.cos(angle)*middle];};
            const start=onCircle(.2),end=onCircle(.8);
            mouse('mousedown',...start);mouse('mousemove',...end);
            const overlay=element.querySelector('.dn-plot-range-selection');
            assert(overlay&&overlay.style.background.includes('conic-gradient'),'Circular selection is not an arc');
            mouse('mouseup',...end);await wait(50);
            const brush=chart.getOption().xAxis[0];
            assert(Math.abs(spanOf(brush)/span-.6)<.002,'Selected angular window is incorrect: '+spanOf(brush)/span);
            assert(valueRanges===JSON.stringify(chart.getOption().yAxis.map(axis=>[axis.min,axis.max])),'Arc selection changed radial ranges');
            assert(chart.getOption().angleAxis[0].min===full.min&&chart.getOption().angleAxis[0].max===full.max,'Zoom remapped the slice around the full circle');
            assert(parseFloat(chart.getOption().polar[0].radius[1])>parseFloat(spec.dnascapeCircularBasePolar[0].radius[1]),'Arc selection did not magnify the original circle');
            assert(chart.getOption().series.every((series,i)=>!spec.series[i].dnascapeCircularSourceData||series.data.length===spec.series[i].dnascapeCircularSourceData.length),
              'Selected arc endpoints cropped a curve or uncertainty band inside the frame');
            const scalesMatch=()=>chart.getOption().radiusAxis.every(axis=>{
              const source=chart.getOption().yAxis[axis.dnascapeSourceAxis==='secondary_y'?1:0];
              return axis.min===source.min&&axis.max===source.max;});
            assert(scalesMatch(),'Arc selection reset the radial zoom');
            assert(legendStable(),'Arc selection moved the legend: '+JSON.stringify([legend,legendBox()]));
            assert(!element.querySelector('.dn-plot-range-selection'),'Arc selection was not cleared');
            if(index===0||index===6)images.push(chart.getDataURL({type:'png',pixelRatio:1,backgroundColor:'#fff'}));
            // A delayed detail response must preserve the latest viewport and
            // complete polar series definitions, including masks and SD bands.
            component.update_chart();await wait(30);
            assert(chart.getOption().xAxis[0].min===brush.min,'Stale response reset the circular zoom');
            assert(chart.getOption().series.every(s=>s.coordinateSystem==='polar'),'Stale response dropped polar series styles');
            assert(scalesMatch(),'Stale response reset the radial zoom');
            key(false);
            mouse('mousedown',chart.getWidth()/2,chart.getHeight()/2);mouse('mousemove',chart.getWidth()/2+12,chart.getHeight()/2+8);await wait(30);
            assert(curveStyles()===originalStyles,'Dragging changed the curve colour, thickness or fill before mouse release');
            mouse('mouseup',chart.getWidth()/2+12,chart.getHeight()/2+8);await wait(30);
            const panned=chart.getOption().xAxis[0];
            assert(Math.abs(spanOf(panned)/spanOf(brush)-1)<1e-8,'Panning changed the angular span');
            key(true);
            for(let repeat=0;repeat<5;repeat++){
              const current=chart.getModel().getComponent('polar',0).coordinateSystem;
              const bounds=element.getBoundingClientRect();
              element.dispatchEvent(new WheelEvent('wheel',{clientX:bounds.left+chart.getWidth()/2,clientY:bounds.top+chart.getHeight()/2,
                deltaY:1000,ctrlKey:true,bubbles:true,cancelable:true}));
            }
            await wait(40);
            assert(chart.getWidth()===oldWidth,'Circular zoom-out resized the diagram');
            assert(Math.abs(chart.getOption().xAxis[0].min-old[0])<1e-10&&Math.abs(chart.getOption().xAxis[0].max-old[1])<1e-10,
              'Circular zoom-out did not restore the full window: '+JSON.stringify([index,old,chart.getOption().xAxis[0].min,chart.getOption().xAxis[0].max]));
            if(index===0||index===6){
              for(const [first,last,expected] of [[.97,.03,.06],[.03,.97,.06],[.1,.9,.2],[.9,.1,.2]]){
                spec.dnascapeNavigation.token='seam-'+index+'-'+first;
                component.update_chart();await wait(30);
                const circle=chart.getModel().getComponent('polar',0).coordinateSystem;
                const extent=circle.getRadiusAxis().getExtent(),radius=(extent[0]+extent[1])/2;
                const point=f=>[circle.cx+Math.sin(f*2*Math.PI)*radius,circle.cy-Math.cos(f*2*Math.PI)*radius];
                const a=point(first),b=point(last);
                mouse('mousedown',...a);mouse('mousemove',...b);
                const brush=element.querySelector('.dn-plot-range-selection');
                const degrees=[...brush.style.background.matchAll(/(-?[\d.]+)deg/g)].map(m=>+m[1]);
                assert(Math.abs(degrees[2]-expected*360)<1,'The shaded selection uses the larger arc: '+JSON.stringify([brush.style.background,degrees,expected]));
                mouse('mouseup',...b);await wait(40);
                const selected=chart.getOption().xAxis[0];
                assert(Math.abs(spanOf(selected)/spanOf({min:full.min,max:full.max})-expected)<.003,'Seam-crossing selection used the larger arc');
                assert(selected.max>full.max,'The selection was shifted away from the seam');
                assert(chart.getOption().series[0].data.some(p=>p[0]===full.min)&&chart.getOption().series[0].data.some(p=>p[0]===full.max),'The selected seam lost one end of the curve');
                assert(JSON.stringify(axisLabel())===originalLabel,'Seam zoom changed the circular title font or position');
                assert(legendStable(),'Seam zoom moved the legend');
              }
            }
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
        f'<script src="{(root/"app/static/brand_theme.js").as_uri()}"></script>'
        f'<script src="{(root/"dnascape/workflows/plot_navigation.js").as_uri()}"></script>'
        '<script>'+probe+'</script></body>',encoding="utf-8")
    screenshot=tmp_path/"circular-browser.png"
    run=subprocess.run([chrome,"--headless","--disable-gpu","--no-first-run","--no-default-browser-check",
        "--allow-file-access-from-files","--window-size=850,850","--virtual-time-budget=6000",
        f"--user-data-dir={tmp_path/'browser'}",f"--screenshot={screenshot}","--dump-dom",document.as_uri()],
        capture_output=True,text=True,encoding="utf-8",timeout=30,check=True)
    found=re.search(r'<pre id="result">([^<]+)</pre>',run.stdout)
    assert found and found[1]!="pending",run.stderr[-1000:]
    result=json.loads(html.unescape(found[1]))
    for index, data in enumerate(result.pop("images", [])):
        (tmp_path/f"circular-{index}.png").write_bytes(base64.b64decode(data.split(",", 1)[1]))
    (tmp_path/"circular-errors.json").write_text(json.dumps(result),encoding="utf-8")
    assert not result["errors"],result
    assert result["results"]==list(range(8))
