"""Circular presentation of bounded profile previews, without changing data."""

from copy import deepcopy
import math
import json

DEFAULT_OUTER_RADIUS = 72.
DEFAULT_INNER_RADIUS = 64.


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


def circular_window_parts(axis, window):
    """Split an unwrapped circular viewport at its original coordinate seam."""
    convert = math.log if axis.get("log", axis.get("type") == "log") else float
    restore = math.exp if axis.get("log", axis.get("type") == "log") else float
    lower, upper = convert(axis["min"]), convert(axis["max"])
    start, end = map(convert, window)
    period = upper - lower
    if end - start >= period * (1 - 1e-9):
        return [(axis["min"], axis["max"])]
    first = lower + (start - lower) % period
    last = first + end - start
    if last <= upper + period * 1e-12:
        return [(restore(first), restore(min(last, upper)))]
    return [(restore(first), axis["max"]), (axis["min"], restore(lower + last - upper))]


def circular_camera(polars, axis, window):
    """Magnify an arc in place, keeping its full-circle angular coordinates."""
    base = deepcopy(polars)
    lower, upper = axis["min"], axis["max"]
    convert = math.log if axis.get("type") == "log" else float
    full_lower, full_upper = convert(lower), convert(upper)
    first, last = (convert(value) for value in window)
    if math.isclose(last-first, full_upper-full_lower, rel_tol=1e-9):
        return base
    fractions = [(first-full_lower)/(full_upper-full_lower), (last-full_lower)/(full_upper-full_lower)]
    if axis.get("inverse"):
        fractions = [1-value for value in fractions[::-1]]
    start, end = fractions
    samples = [start, end] + [value/4 for value in range(math.floor(start*4), math.ceil(end*4)+1) if start <= value/4 <= end]
    radii = [float(value.rstrip("%"))/100 for polar in base for value in polar["radius"]]
    points = [(radius*math.sin(2*math.pi*f), -radius*math.cos(2*math.pi*f)) for radius in (min(radii),max(radii)) for f in samples]
    xs, ys = zip(*points)
    scale = min(200., max(1., min(1.72/max(max(xs)-min(xs),1e-9), 1.6/max(max(ys)-min(ys),1e-9))))
    center = [50-25*(min(xs)+max(xs))*scale, 50-25*(min(ys)+max(ys))*scale]
    for polar in base:
        polar["center"] = [f"{value:.12g}%" for value in center]
        polar["radius"] = [f"{float(value.rstrip('%'))*scale:.12g}%" for value in polar["radius"]]
    return base


def circular_preview_options(options, *, layout="Rings", outer_radius=None, inner_radius=DEFAULT_INNER_RADIUS, track_height=None):
    if options.get("dnascapePlotType") == "Heatmap":
        raise ValueError("Circular view is available for profiles, not heatmaps.")
    legacy_height = track_height is not None and outer_radius is None
    if legacy_height:
        track_height = float(track_height)
        if not math.isfinite(track_height) or not 1 <= track_height <= 24:
            raise ValueError("Track height must be between 1 and 24 percent.")
    inner_radius = float(inner_radius)
    outer_radius = float(DEFAULT_OUTER_RADIUS if outer_radius is None else outer_radius)
    if not math.isfinite(outer_radius) or not 0 < outer_radius <= 100:
        raise ValueError("Outer radius must be greater than 0 and at most 100 percent.")
    if not math.isfinite(inner_radius) or not 0 <= inner_radius < (80. if legacy_height else outer_radius):
        raise ValueError("Inner radius must be non-negative and less than the outer radius.")
    x_axes = options.get("xAxis", {})
    y_axes = options.get("yAxis", {})
    x = x_axes[0] if isinstance(x_axes, list) else x_axes
    y = y_axes[0] if isinstance(y_axes, list) else y_axes
    histogram = options.get("dnascapePlotType") == "Histogram"
    options["series"] = [s for s in options.get("series", []) if not s.get("dnascapeCircularMask")]
    curves = [s for s in options.get("series", [])
              if (s.get("type") in {"line", "scatter"} or (histogram and s.get("type") == "custom" and not s.get("silent")))
              and len(s.get("dnascapeCircularSourceData", s.get("data", [])))]
    if not curves:
        return options
    options.setdefault("dnascapeCircularLabels", [x.get("name", ""), y.get("name", "")])
    styles = options.setdefault("dnascapeCircularAxisStyles", {
        "x": deepcopy(x_axes if isinstance(x_axes, list) else [x_axes]),
        "y": deepcopy(y_axes if isinstance(y_axes, list) else [y_axes]),
    })
    if "dnascapeNavigation" not in options:
        axes = {}
        for dimension, specifications in (("x", x_axes), ("y", y_axes)):
            for index, axis in enumerate(specifications if isinstance(specifications, list) else [specifications]):
                axes[("secondary_" if index else "") + dimension] = dict(
                    dimension=dimension, index=index, min=axis["min"], max=axis["max"],
                    log=axis.get("type") == "log", inverse=bool(axis.get("inverse")),
                )
        options["dnascapeNavigation"] = dict(context=json.dumps(axes, sort_keys=True), axes=axes, token=0, sequence=0, polar=True, heatmap=False)
    options["dnascapeCircular"] = True
    for axis in options["dnascapeNavigation"]["axes"].values():
        if axis["dimension"] == "x":
            axis["circular"] = True
    options["dnascapeAspectRatio"] = 1.
    options["dataZoom"] = []
    tooltip = options.get("tooltip", {})
    hover = bool(tooltip.get("show"))
    pointer = {**tooltip.get("axisPointer", {}), "show": False, "type": "none", "snap": True}
    options["tooltip"] = {**tooltip, "trigger": "item", "triggerOn": "none", "axisPointer": pointer}
    options["axisPointer"] = {"show": False, "triggerOn": "none"}
    options["dnascapeCircularDatasetCount"] = len({(curve.get("dnascapeCircularGroup", index), curve.get("xAxisIndex", 0), curve.get("yAxisIndex", 0)) for index,curve in enumerate(curves)})
    rings = layout == "Rings" and len(curves) > 1
    groups, group_indices, curve_tracks = [], {}, []
    for index, curve in enumerate(curves):
        key = (curve.get("dnascapeCircularGroup", index) if rings else "overlay",
               curve.get("xAxisIndex", 0), curve.get("yAxisIndex", 0))
        if key not in group_indices:
            group_indices[key] = len(groups)
            groups.append(curve)
        curve_tracks.append(group_indices[key])
    tracks = len(groups)
    radial_tracks = tracks if rings else 1
    if legacy_height:
        # Previously exported scripts specify thickness rather than an outer
        # radius. Preserve their geometry while new controls use both radii.
        gap = min(2., 10./radial_tracks, (80.-inner_radius)/(2*radial_tracks))
        height = min(track_height, (80.-inner_radius-gap*(radial_tracks-1))/radial_tracks)
        outer_radius = inner_radius + radial_tracks*height + gap*(radial_tracks-1)
    else:
        span = outer_radius - inner_radius
        gap = min(2., 10./radial_tracks, span/(2*radial_tracks))
        height = (span-gap*(radial_tracks-1))/radial_tracks
    options["dnascapeCircularTrackHeight"] = height
    options["dnascapeCircularOuterRadius"] = outer_radius
    options["dnascapeCircularInnerRadius"] = inner_radius
    legend = options.get("legend")
    if isinstance(legend, dict) and legend.get("show"):
        # Preserve the selected corner independently of the circular viewport.
        # Keep the compact scrolling row and the circle's centred geometry.
        left = legend.get("left") is not None
        bottom = legend.get("bottom") is not None
        legend.update(type="scroll", orient="horizontal", left=8 if left else None, right=None if left else 8,
                      top=None if bottom else 2, bottom=2 if bottom else None, width=None, height=None, padding=[2, 4], itemWidth=18, itemHeight=10,
                      tooltip={"show": True},
                      textStyle={**legend.get("textStyle", {}), "width": 110, "overflow": "truncate"})
    options["polar"], options["angleAxis"], options["radiusAxis"] = [], [], []
    for index in range(tracks):
        inner = inner_radius + index*(height+gap) if rings else inner_radius
        outer = inner+height
        options["polar"].append({"center": ["50%", "50%"], "radius": [f"{inner}%", f"{outer}%"]})
        curve = groups[index]
        xi, yi = curve.get("xAxisIndex", 0), curve.get("yAxisIndex", 0)
        current_x = x_axes[xi] if isinstance(x_axes, list) else x_axes
        current_y = y_axes[yi] if isinstance(y_axes, list) else y_axes
        angle = deepcopy(styles["x"][xi])
        source_key = "x" if xi == 0 else "secondary_x"
        domain = options.get("dnascapeNavigation", {}).get("axes", {}).get(source_key, {})
        angle.update(min=domain.get("min", styles["x"][xi].get("min")), max=domain.get("max", styles["x"][xi].get("max")), interval=None)
        angle["inverse"] = bool(current_x.get("inverse"))
        angle["dnascapeSourceAxis"] = "x" if xi == 0 else "secondary_x"
        angle.update(polarIndex=index, startAngle=90, clockwise=True, name="", show=True)
        angle.pop("position", None)
        angle["axisLabel"] = {**angle.get("axisLabel", {}), "show": index == tracks-1,
                              "showMaxLabel": False, "hideOverlap": True, "fontSize": 10}
        angle["axisLine"] = {**angle.get("axisLine", {}), "show": True}
        ticks = None
        if angle.get("type") != "log" and (current_x["min"] != angle["min"] or current_x["max"] != angle["max"]):
            from .preview import _aligned_axis_ticks
            ticks = sorted({tick for part in circular_window_parts(angle, [current_x["min"], current_x["max"]])
                            for tick in _aligned_axis_ticks(*part, log_axis=False)})
        for part in ("axisLabel", "axisTick", "splitLine"):
            angle[part] = {**angle.get(part, {}), "customValues": ticks}
        angle["axisPointer"] = {**deepcopy(pointer), "show": False}
        radius = deepcopy(styles["y"][yi])
        radius.update(min=current_y.get("min"), max=current_y.get("max"), interval=None)
        radius["inverse"] = bool(current_y.get("inverse"))
        radius["dnascapeSourceAxis"] = "y" if yi == 0 else "secondary_y"
        radius.update(polarIndex=index, name="", show=True, splitNumber=1)
        radius.pop("position", None)
        radius["axisPointer"] = {"show": False}
        radius["axisLabel"] = {**radius.get("axisLabel", {}), "show": True,
                               "hideOverlap": True, "fontSize": 8, "margin": 3}
        for part in ("axisLabel", "axisTick", "splitLine"):
            radius[part] = {**radius.get(part, {}), "customValues": None}
        if rings:
            radius["axisLabel"]["color"] = curve.get("lineStyle", {}).get("color", "#777")
        options["angleAxis"].append(angle)
        options["radiusAxis"].append(radius)
    for index, series in enumerate(curves):
        polar_index = curve_tracks[index]
        current_x, current_y = options["angleAxis"][polar_index], options["radiusAxis"][polar_index]
        series.update(coordinateSystem="polar", polarIndex=polar_index,
                      encode={"angle": [0, 1], "radius": [2, 3]} if series.get("type") == "custom" else {"angle": 0, "radius": 1}, clip=True)
        source = series.get("dnascapeCircularSourceData", series.get("data", []))
        series["dnascapeCircularSourceData"] = source
        # The angular domain stays fixed while the camera magnifies the arc.
        # Keep surrounding samples so curves continue to the frame edges,
        # rather than stopping at the narrower selected angular interval.
        if series.get("type") == "custom":
            series["data"] = list(source)
            series[":renderItem"] = _sector_renderer(bool(current_x.get("inverse")))
        else:
            series["data"] = [point for point in source if isinstance(point, (list, tuple)) and len(point) >= 2
                              and point[0] is not None]
            series.update(showSymbol=series.get("type") == "scatter",
                          symbolSize=6,
                          emphasis={**series.get("emphasis", {}), "disabled": True})
    # Named mask guides have no curve points. Convert them separately so they
    # shade the existing tracks without creating extra radial tracks.
    for series in tuple(options.get("series", [])):
        if series.get("markArea"):
            source = series.setdefault("dnascapeCircularMaskSource", series["markArea"].get("data", []))
            series["markArea"] = {**series["markArea"], "data": []}
            for polar_index in ([series["polarIndex"]] if "polarIndex" in series else range(tracks)):
                current_x, current_y = options["angleAxis"][polar_index], options["radiusAxis"][polar_index]
                mask_source = [[interval[0]["xAxis"], interval[1]["xAxis"], current_y["max"], current_y["min"]]
                               for interval in source if "xAxis" in interval[0] and "xAxis" in interval[1]]
                if mask_source:
                    options["series"].append({"type": "custom", "coordinateSystem": "polar",
                        "polarIndex": polar_index, "name": series.get("name", "Mask"),
                        "dnascapeCircularMask": True, "silent": True, "z": 0,
                        "dnascapeCircularMaskSourceData": mask_source,
                        "encode": {"angle": [0, 1], "radius": [2, 3]}, "data": list(mask_source),
                        "itemStyle": series["markArea"].get("itemStyle", {"color": "rgba(128,128,128,.15)"}),
                        ":renderItem": _sector_renderer(bool(current_x.get("inverse"))),
                    })
    for series in options.get("series", []):
        if series.get("type") != "custom" or not series.get("silent"):
            continue
        source = series.get("dnascapeCircularSourceData", series.get("data", []))
        if not source or not isinstance(source[0], (list, tuple)) or len(source[0]) != 6:
            continue
        key = (series.get("dnascapeCircularGroup") if rings else "overlay",
               series.get("xAxisIndex", 0), series.get("yAxisIndex", 0))
        track = group_indices.get(key, 0)
        angle = options["angleAxis"][track]
        series.update(coordinateSystem="polar", polarIndex=track, clip=True,
                      encode={"angle": [0, 3], "radius": [1, 2, 4, 5]})
        series["dnascapeCircularSourceData"] = source
        series["data"] = [segment for segment in source if angle["min"] <= segment[0] and segment[3] <= angle["max"]]
        series[":renderItem"] = (
            "(params,api)=>({type:'polygon',shape:{points:["
            "api.coord([api.value(1),api.value(0)]),api.coord([api.value(4),api.value(3)]),"
            "api.coord([api.value(5),api.value(3)]),api.coord([api.value(2),api.value(0)])]},style:api.style()})"
        )
    for axes in (x_axes, y_axes):
        for axis in axes if isinstance(axes, list) else [axes]:
            axis.update(show=False, name="", axisPointer={"show": False, "label": {"show": False}})
            for part in ("axisLine", "axisTick", "axisLabel", "splitLine"):
                axis[part] = {**axis.get(part, {}), "show": False}
    options["dnascapeCircularBasePolar"] = deepcopy(options["polar"])
    options["polar"] = circular_camera(options["polar"], options["angleAxis"][0], [x["min"], x["max"]])
    label = " · ".join(text for text in options["dnascapeCircularLabels"] if text)
    graphic = options.get("graphic", [])
    graphic = graphic if isinstance(graphic, list) else [graphic]
    options["graphic"] = [g for g in graphic if g.get("id") not in {"dn-circular-axis-label", "dn-circular-frame"}] + [{
        "id": "dn-circular-frame", "type": "rect", "invisible": True, "silent": True,
        "shape": {"x": 0, "y": 0, "width": 0, "height": 0, "dnascapeCircularFrame": True},
    }, {
        "id": "dn-circular-axis-label", "type": "text", "left": "center", "bottom": 10,
        "silent": True, "scaleX": 1, "scaleY": 1, "rotation": 0,
        "style": {"text": label, "font": "11px sans-serif", "fontSize": 11,
                  "fontFamily": "sans-serif", "fontWeight": "normal", "fontStyle": "normal", "fill": "#777"},
    }]
    return options
