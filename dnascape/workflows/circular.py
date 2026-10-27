"""Circular presentation of bounded profile previews, without changing data."""

from copy import deepcopy


def _sector_renderer(inverse):
    return (
        "(p,a)=>{const c=p.coordSys,q0=a.coord([a.value(3),a.value(0)]),q1=a.coord([a.value(2),a.value(1)]);"
        "let s=Math.atan2(q0[1]-c.cy,q0[0]-c.cx),e=Math.atan2(q1[1]-c.cy,q1[0]-c.cx);const clockwise="
        + ("false" if inverse else "true") + ";if(Math.abs(e-s)<1e-9&&a.value(1)>a.value(0))e=s+(clockwise?2:-2)*Math.PI;"
        "else{if(clockwise&&e<s)e+=2*Math.PI;if(!clockwise&&e>s)e-=2*Math.PI;}"
        "const r0=Math.max(c.r0,Math.min(c.r,Math.hypot(q0[0]-c.cx,q0[1]-c.cy))),"
        "r=Math.max(c.r0,Math.min(c.r,Math.hypot(q1[0]-c.cx,q1[1]-c.cy)));"
        "return {type:'sector',shape:{cx:c.cx,cy:c.cy,r0:Math.min(r0,r),r:Math.max(r0,r),"
        "startAngle:s,endAngle:e,clockwise},style:a.style()};}"
    )


def circular_preview_options(options, *, layout="Rings"):
    if options.get("dnascapePlotType") == "Heatmap":
        raise ValueError("Circular view is available for profiles, not heatmaps.")
    x_axes = options.get("xAxis", {})
    y_axes = options.get("yAxis", {})
    x = x_axes[0] if isinstance(x_axes, list) else x_axes
    y = y_axes[0] if isinstance(y_axes, list) else y_axes
    histogram = options.get("dnascapePlotType") == "Histogram"
    options["series"] = [s for s in options.get("series", []) if not s.get("dnascapeCircularMask")]
    curves = [s for s in options.get("series", [])
              if (s.get("type") == "line" or (histogram and s.get("type") == "custom" and not s.get("silent")))
              and len(s.get("dnascapeCircularSourceData", s.get("data", [])))]
    if not curves:
        return options
    options.setdefault("dnascapeCircularLabels", [x.get("name", ""), y.get("name", "")])
    styles = options.setdefault("dnascapeCircularAxisStyles", {
        "x": deepcopy(x_axes if isinstance(x_axes, list) else [x_axes]),
        "y": deepcopy(y_axes if isinstance(y_axes, list) else [y_axes]),
    })
    options["dnascapeCircular"] = True
    options["dnascapeAspectRatio"] = 1.
    options["dataZoom"] = []
    options["tooltip"] = {**options.get("tooltip", {}), "trigger": "item", "axisPointer": {"show": False, "type": "none"}}
    options["axisPointer"] = {"show": False, "link": []}
    rings = layout == "Rings" and len(curves) > 1
    independent_axes = len(styles["x"]) > 1 or len(styles["y"]) > 1
    tracks = len(curves) if rings or independent_axes else 1
    options["polar"], options["angleAxis"], options["radiusAxis"] = [], [], []
    for index in range(tracks):
        gap = min(3., 20./tracks)
        width = 70./tracks
        inner = 9. + index*width if rings else 8.
        outer = 9. + (index+1)*width - gap if rings else 78.
        options["polar"].append({"center": ["50%", "50%"], "radius": [f"{inner}%", f"{outer}%"]})
        curve = curves[index] if tracks > 1 else curves[0]
        xi, yi = curve.get("xAxisIndex", 0), curve.get("yAxisIndex", 0)
        current_x = x_axes[xi] if isinstance(x_axes, list) else x_axes
        current_y = y_axes[yi] if isinstance(y_axes, list) else y_axes
        angle = deepcopy(styles["x"][xi])
        angle.update(min=current_x.get("min"), max=current_x.get("max"), interval=None)
        angle["dnascapeSourceAxis"] = "x" if xi == 0 else "secondary_x"
        angle.update(polarIndex=index, startAngle=90, clockwise=True, name="", show=True)
        angle.pop("position", None)
        angle["axisLabel"] = {**angle.get("axisLabel", {}), "show": index == tracks-1,
                              "showMaxLabel": False, "hideOverlap": True, "fontSize": 10}
        angle["axisLine"] = {**angle.get("axisLine", {}), "show": True}
        angle["axisPointer"] = {"show": False}
        radius = deepcopy(styles["y"][yi])
        radius.update(min=current_y.get("min"), max=current_y.get("max"), interval=None)
        radius["dnascapeSourceAxis"] = "y" if yi == 0 else "secondary_y"
        radius.update(polarIndex=index, name="", show=True, splitNumber=2)
        radius.pop("position", None)
        radius["axisPointer"] = {"show": False}
        radius["axisLabel"] = {**radius.get("axisLabel", {}), "show": True,
                               "hideOverlap": True, "fontSize": 9}
        if rings:
            radius["axisLabel"]["color"] = curves[index].get("lineStyle", {}).get("color", "#777")
        options["angleAxis"].append(angle)
        options["radiusAxis"].append(radius)
    for index, series in enumerate(curves):
        polar_index = index if tracks > 1 else 0
        current_x, current_y = options["angleAxis"][polar_index], options["radiusAxis"][polar_index]
        series.update(coordinateSystem="polar", polarIndex=polar_index,
                      encode={"angle": [0, 1], "radius": [2, 3]} if series.get("type") == "custom" else {"angle": 0, "radius": 1}, clip=True)
        source = series.get("dnascapeCircularSourceData", series.get("data", []))
        series["dnascapeCircularSourceData"] = source
        # Angular axes wrap out-of-view coordinates around the circle. Filter
        # only the rendered samples; retain the source for panning/refinement.
        lower, upper = current_x.get("min"), current_x.get("max")
        if series.get("type") == "custom":
            series["data"] = [[max(point[0], lower), min(point[1], upper), *point[2:]]
                              for point in source if point[1] > lower and point[0] < upper]
            series[":renderItem"] = _sector_renderer(bool(current_x.get("inverse")))
        else:
            series["data"] = [point for point in source if isinstance(point, (list, tuple)) and len(point) >= 2
                              and point[0] is not None and lower <= point[0] <= upper]
            series.update(showSymbol=True, symbolSize=2, emphasis={**series.get("emphasis", {}), "scale": 3})
        if series.get("markArea"):
            source = series.setdefault("dnascapeCircularMaskSource", series["markArea"].get("data", []))
            mask, mask_source = [], []
            for interval in source:
                start, end = interval[0].get("xAxis"), interval[1].get("xAxis")
                if start is not None and end is not None:
                    mask_source.append([start, end, current_y["max"], current_y["min"]])
                    if end > lower and start < upper:
                        mask.append([max(start, lower), min(end, upper), current_y["max"], current_y["min"]])
            series["markArea"] = {**series["markArea"], "data": []}
            if mask_source:
                options["series"].append({"type": "custom", "coordinateSystem": "polar",
                    "polarIndex": polar_index, "name": interval[0].get("name", "Filter"),
                    "dnascapeCircularMask": True, "silent": True, "z": 0,
                    "dnascapeCircularMaskSourceData": mask_source,
                    "encode": {"angle": [0, 1], "radius": [2, 3]}, "data": mask,
                    "itemStyle": {"color": "rgba(128,128,128,.15)"},
                    ":renderItem": _sector_renderer(bool(current_x.get("inverse"))),
                })
    for series in options.get("series", []):
        if series.get("type") != "custom" or not series.get("silent"):
            continue
        source = series.get("dnascapeCircularSourceData", series.get("data", []))
        if not source or not isinstance(source[0], (list, tuple)) or len(source[0]) != 6:
            continue
        series.update(coordinateSystem="polar", polarIndex=0,
                      encode={"angle": [0, 3], "radius": [1, 2, 4, 5]})
        series["dnascapeCircularSourceData"] = source
        series["data"] = [segment for segment in source if x["min"] <= segment[0] and segment[3] <= x["max"]]
        series[":renderItem"] = (
            "(params,api)=>({type:'polygon',shape:{points:["
            "api.coord([api.value(1),api.value(0)]),api.coord([api.value(4),api.value(3)]),"
            "api.coord([api.value(5),api.value(3)]),api.coord([api.value(2),api.value(0)])]},style:api.style()})"
        )
    for axes in (x_axes, y_axes):
        for axis in axes if isinstance(axes, list) else [axes]:
            axis.update(show=False, name="")
            for part in ("axisLine", "axisTick", "axisLabel", "splitLine"):
                axis[part] = {**axis.get(part, {}), "show": False}
    label = " · ".join(text for text in options["dnascapeCircularLabels"] if text)
    graphic = options.get("graphic", [])
    graphic = graphic if isinstance(graphic, list) else [graphic]
    options["graphic"] = [g for g in graphic if g.get("id") != "dn-circular-axis-label"] + [{
        "id": "dn-circular-axis-label", "type": "text", "left": "center", "bottom": 10,
        "silent": True, "style": {"text": label, "fontSize": 11, "fill": "#777"},
    }]
    return options
