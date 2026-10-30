"""Reproduce Analysis data preparation and Plotting without the web application."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from .analysis import AnalysisDataset, SmoothingSettings, analyse_profiles, smooth_profile
from .code_recipes import restore_settings, write_artifact


def prepare_analysis_recipe(recipe, *, input_directory="inputs"):
    from .analysis_transforms import analysis_values, transform_profiles
    from .uploads import materialize_upload

    profiles = restore_settings(recipe["profiles"], input_directory=input_directory)
    transforms = restore_settings(recipe.get("transformations", {}), input_directory=input_directory)
    data_transforms = restore_settings(recipe.get("data_transforms", {}), input_directory=input_directory)
    methods = recipe.get("methods")
    region_enabled = bool(transforms.get("settings") is not None and transforms["settings"].enabled)
    selected = []
    for profile in profiles:
        identity = tuple(profile["id"])
        inspection = replace(profile["source"], expected_form=profile["form"])
        values = materialize_upload(inspection, column=profile.get("column"), array_key=profile.get("array_key"),
                                    missing_policy="ignore" if methods is not None or region_enabled or data_transforms.get("enabled") else "error")
        values = smooth_profile(values, profile.get("smoothing", SmoothingSettings()) if methods is not None else SmoothingSettings(),
                                dependence=profile["form"])
        selected.append(dict(profile, id=identity, values=values, transformed=False))
    selected = transform_profiles(selected, data_transforms=data_transforms, region_transforms=transforms)
    if not selected:
        raise ValueError("Select at least one profile for Plotting.")
    from collections import Counter
    counts = Counter(item["name"] for item in selected)
    occurrences = Counter()
    for item in selected:
        name = item["name"]
        occurrences[name] += 1
        if counts[name] > 1:
            item["name"] = f"{name} {occurrences[name]}"
    report = (analyse_profiles([AnalysisDataset(item["name"], analysis_values(item), item["form"]) for item in selected], methods=methods)
              if methods is not None else None)
    if report is not None and recipe.get("report_names"):
        report = replace(report, names=tuple(recipe["report_names"].get(f"{item['id'][0]}:{item['id'][1]}", item["name"])
                                            for item in selected))
    return tuple(selected), report


def build_profile_recipe_plot(profiles, plot):
    from .preview import PreviewLine, build_preview_plot_options

    selected = [next(item for item in profiles if tuple(item["id"]) == tuple(key)) for key in plot["profiles"]]
    arguments = dict(plot["arguments"])
    if arguments["plot_type"] in {"Histogram", "Boxplot"}:
        from .analysis_transforms import analysis_values
        selected = [dict(item, values=analysis_values(item)) for item in selected]
    profile = selected[0]
    if arguments["plot_type"] == "Heatmap":
        from .temporal_preview import profile_heatmap_data
        values, positions, times = profile_heatmap_data(profile)
        arguments.update(values=values, x_axis=positions, y_axis=times)
    else:
        curves = plot.get("curves", [{}] * len(selected))
        arguments["lines"] = tuple(PreviewLine(
            x=np.arange(np.asarray(item["values"]).size, dtype=float),
            y=np.asarray(item["values"]).reshape(-1), **curve,
        ) for item, curve in zip(selected, curves, strict=True))
        arguments["values"] = profile["values"]
    options = build_preview_plot_options(**arguments)
    return apply_profile_presentation(options, plot.get("presentation", {}))


def apply_profile_presentation(options, settings):
    """Apply the display-only controls used after constructing a preview."""
    heatmap = options.get("dnascapePlotType") == "Heatmap"
    series = options.get("series", [])
    pointer = options.get("tooltip", {}).get("axisPointer", {})
    if settings.get("theme_color") and isinstance(pointer.get("label"), dict):
        pointer["label"]["backgroundColor"] = settings["theme_color"]
    if not heatmap:
        reverse = bool(settings.get("reverse_order"))
        for index, item in enumerate(series):
            item.setdefault("dnascapePresentationOrder", index)
            item.setdefault("dnascapePresentationZ", item.get("z", 2))
        options["series"] = sorted(series, key=lambda item: (
            item.get("type") != "custom", (-1 if reverse else 1) * item["dnascapePresentationOrder"],
        ))
        for kind in {item.get("type") for item in series}:
            layer = [item for item in options["series"] if item.get("type") == kind and not item.get("silent")]
            for item, z in zip(layer, sorted((item["dnascapePresentationZ"] for item in layer), reverse=True)):
                item["z"] = z
        legend = options.get("legend", {})
        if isinstance(legend, dict) and isinstance(legend.get("data"), list):
            options.setdefault("dnascapeLegendOrder", list(legend["data"]))
            legend["data"] = list(reversed(options["dnascapeLegendOrder"])) if reverse else list(options["dnascapeLegendOrder"])
        if settings.get("view") == "Circular":
            from .circular import circular_preview_options
            radii = dict(inner_radius=settings.get('circular_inner_radius', 64.))
            if 'circular_outer_radius' in settings:
                radii['outer_radius'] = settings['circular_outer_radius']
            elif 'circular_track_height' in settings:
                radii['track_height'] = settings['circular_track_height']
            circular_preview_options(options, layout=("Overlay" if settings["circular_combine"] else "Rings") if "circular_combine" in settings else settings.get("circular_layout", "Rings"), **radii)
    else:
        from .preview import set_heatmap_colorbar_label
        set_heatmap_colorbar_label(options, settings.get("z_axis_label"))
    axes = options.get("yAxis", {})
    axes = axes if isinstance(axes, list) else [axes]
    for item in options.get("series", []):
        if item.get("type") != "line":
            continue
        axis = axes[min(int(item.get("yAxisIndex", 0)), len(axes) - 1)]
        style = item.get("lineStyle", {})
        item["areaStyle"] = ({"color": style.get("color", item.get("itemStyle", {}).get("color")),
                              "opacity": .15 * float(style.get("opacity", 1.)),
                              "origin": "end" if (settings.get("fill_side") == "above") != bool(axis.get("inverse")) else "start"}
                             if settings.get("show_fill") else None)
    if not settings.get("show_hover", True):
        options["axisPointer"] = {**options.get("axisPointer", {}), "show": False, "triggerOn": "none"}
        tooltip = options.setdefault("tooltip", {})
        pointer = tooltip.get("axisPointer", {})
        tooltip.update(show=False, triggerOn="none", axisPointer={**pointer, "show": False, "type": "none",
            "label": {**pointer.get("label", {}), "show": False}})
        for item in options.get("series", []):
            item["emphasis"] = {**item.get("emphasis", {}), "disabled": True}
            if item.get("type") == "line":
                item["symbolSize"] = 0
    guide = options.get("dnascapeHistogramGuides")
    if guide and guide.get("choices"):
        selected = max(0, min(int(settings.get("histogram_dataset", 0)), len(guide["choices"]) - 1))
        guide["selected"] = selected
        graphic = [item for item in options.get("graphic", []) if item.get("id") != "dn-histogram-guide"]
        options["graphic"] = graphic + ([guide["choices"][selected]["graphic"]] if guide["visible"] else [])
    return options


def run_plotting_recipe(recipe, *, input_directory="inputs", output_directory="outputs"):
    from .export_settings import get_export_settings
    settings = get_export_settings()
    previous = settings.preview_points
    try:
        settings.preview_points = int(recipe.get("preview_points", previous))
        return _run_plotting_recipe(recipe, input_directory=input_directory, output_directory=output_directory)
    finally:
        settings.preview_points = previous


def _run_plotting_recipe(recipe, *, input_directory="inputs", output_directory="outputs"):
    from .analysis import export_correlation_report, export_profile_data
    from .exports import ExportArtifact, build_histogram_statistics_export, safe_export_filename
    from .plotting import write_plot_html
    from .result_naming import compact_export_stem

    profiles, report = prepare_analysis_recipe(recipe, input_directory=input_directory)
    paths = []
    for item in profiles:
        key = f"{item['id'][0]}:{item['id'][1]}"
        preference = recipe.get("data_exports", {}).get(key, {})
        artifact = export_profile_data(item["values"], dependence=item["form"],
            file_format=preference.get("type", "TSV"),
            filename=preference.get("name") or compact_export_stem(item["name"], fallback="profile"),
            compress=bool(preference.get("compress", False)))
        write_artifact(artifact, output_directory, paths)
    if report is not None and report.matrices:
        preference = recipe.get("report", {})
        file_format = preference.get("file_format", "tsv")
        data = export_correlation_report(report, file_format=file_format,
            include_p_values=bool(preference.get("include_p_values", False)), combined=True)
        name = safe_export_filename(preference.get("filename", "analysis_report"), file_format)
        write_artifact(ExportArtifact(data, name, "text/plain"), output_directory, paths)
    if recipe.get("mask_exports"):
        from .analysis_transforms import analysis_mask_artifact, profile_key
        for item in profiles:
            preference = recipe["mask_exports"].get(profile_key(item["id"]))
            if preference is not None and item.get("mask") is not None:
                write_artifact(analysis_mask_artifact(item, **preference), output_directory, paths)
    for index, plot in enumerate(recipe.get("plots", []), 1):
        options = build_profile_recipe_plot(profiles, plot)
        filename = Path(str(plot.get("filename") or f"plot_{index}")).name
        path = Path(output_directory) / (filename.removesuffix(".html") + ".html")
        if path in paths:
            raise ValueError("Several plots use the same filename.")
        write_plot_html(options, path, title=plot.get("arguments", {}).get("title", "Profiles"),
                        dpi=recipe.get("plot_dpi", 150))
        paths.append(path)
        if plot.get("statistics"):
            artifact = build_histogram_statistics_export(statistics=options.get("dnascapeHistogramStatistics", []),
                                                         **plot["statistics"])
            write_artifact(artifact, output_directory, paths)
    return (profiles, report), tuple(paths)
