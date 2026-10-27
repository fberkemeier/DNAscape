"""Raw exports and optional exact ECharts plots for Mapping recipes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def _output_key(result: Any, selected: str) -> str:
    if selected in {"", "primary", "all_plots"}:
        return result.primary_output
    # The Mapping preview exposes these aliases for fitting diagnostics.
    from .plotting import resolved_output_key
    return resolved_output_key(result, selected)


def save_mapping_outputs(
    results: Sequence[Any],
    options: Mapping[str, Any],
    *,
    directory: str | Path = "outputs",
) -> tuple[Path, ...]:
    """Use the app's exact serializers, including per-dataset raw controls."""
    from .mapping_exports import build_mapping_comparison_raw_export, build_mapping_raw_export, mapping_output_dependence_suffix
    from .result_naming import compact_export_stem, output_file_token

    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    overrides = options.get("per_dataset", {})
    for named in results:
        settings = {**options, **overrides.get(named.key, {})}
        selected = str(settings.get("output", "primary"))
        filename = str(settings.get("filename") or "").strip()
        if not filename:
            stem = named.export_stem if settings.get("name_style") == "full" else (named.short_export_stem or named.export_stem)
            filename = f"rmap_{stem[:120].rstrip('-') or 'mapping-result'}"
            actual_key = _output_key(named.result, selected)
            comparison = selected.startswith("comparison:")
            suffix = mapping_output_dependence_suffix(named.result, selected if comparison else actual_key)
            if comparison or actual_key != named.result.primary_output:
                token = (
                    f"{output_file_token(selected.removeprefix('comparison:').removesuffix('_fit'))}-cmp"
                    if comparison else output_file_token(actual_key)
                )
                if suffix and not comparison:
                    token = token.removesuffix(f"-{suffix}")
                filename += "_" + compact_export_stem(token, fallback="mapping-result-1", maximum=150)
            if suffix:
                filename += f"_{suffix}"
        elif len(results) > 1 and named.key not in overrides:
            filename = f"{filename}_{named.key}"
        compression = str(settings.get("compression", "gzip")) if settings.get("compress", False) else "none"
        if selected.startswith("comparison:"):
            artifact = build_mapping_comparison_raw_export(
                named.result, selected.removeprefix("comparison:"),
                str(settings.get("file_format", "TSV")), filename=filename,
                compression=compression,
            )
        else:
            artifact = build_mapping_raw_export(
                named.result, _output_key(named.result, selected),
                str(settings.get("file_format", "TSV")), filename=filename,
                compression=compression,
                filter_layout=str(settings.get("filter_layout", "positions")),
            )
        path = folder / artifact.filename
        if path in paths:
            raise ValueError("Several Mapping outputs use the same filename.")
        path.write_bytes(artifact.data)
        paths.append(path)
        if "filter_positions" in named.result.outputs and selected != "filter_positions":
            options_filter = dict(options.get("filter_exports", {}).get(named.key, {}))
            stem = named.export_stem if options.get("name_style") == "full" else (named.short_export_stem or named.export_stem)
            filter_name = options_filter.get("filename") or f"rmap_{stem}_filter_positions"
            artifact = build_mapping_raw_export(named.result, "filter_positions", options_filter.get("file_format", "TSV"),
                filename=filter_name, filter_layout=options_filter.get("layout", "positions"),
                compression=options_filter.get("compression", "gzip") if options_filter.get("compress") else "none")
            from .code_recipes import write_artifact
            write_artifact(artifact, folder, paths)
    statistics = options.get("statistics")
    if statistics and results:
        from .code_recipes import save_recipe_statistics
        from .mapping_exports import select_mapping_output
        selected_results = results
        if statistics.get("dataset_index") is not None:
            selected_results = (results[min(max(0, int(statistics["dataset_index"])), len(results) - 1)],)
        datasets = [(named.name, select_mapping_output(named.result, _output_key(named.result, str(options.get("output", "primary")))).values)
                    for named in selected_results if not statistics.get("dataset") or named.key == statistics["dataset"]]
        save_recipe_statistics(datasets, statistics, folder, paths)
    return tuple(paths)


def plot_mapping_outputs(
    results: Sequence[Any],
    options: Mapping[str, Any],
    *,
    directory: str | Path = "outputs",
    echarts_path: str | Path | None = None,
) -> tuple[Path, ...]:
    """Write optional plots using the same chart specification as Mapping."""
    from .plotting import combined_mapping_plot_options, mapping_plot_options, write_plot_html
    from .preview_inputs import input_preview_result

    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    settings = dict(options.get("settings", {}))
    selected = str(options.get("output", "primary"))
    if results:
        settings["output_key"] = results[0].result.primary_output if selected in {"", "primary"} else selected
        if "view" in options:
            actual_key = _output_key(results[0].result, selected)
            preview_result = input_preview_result(results[0].result, actual_key)
            output_form = preview_result.output_forms.get(actual_key)
            requested_view = str(options["view"] or "Default")
            settings["plot_type"] = (
                "Line" if selected.startswith("comparison:") else
                "Heatmap" if output_form in {"t", "xt"} else
                requested_view if requested_view in {"Histogram", "Boxplot"} else
                "Histogram" if preview_result.output_dependence == "ensemble" else "Line"
            )
    colors = {
        (item["dataset"], item["output"], int(item["series"])): item["color"]
        for item in options.get("colors", ())
    }
    line_types = {
        (item["dataset"], item["output"], int(item["series"])): item["line_type"]
        for item in options.get("line_types", ())
    }
    curve_names = {
        (item["dataset"], item["output"], int(item["series"])): item["name"]
        for item in options.get("curve_names", ())
    }
    display_names = {
        named.key: getattr(named, "curve_label", "") or (
            named.name if options.get("name_style") == "full" else (named.short_name or named.name)
        )
        for named in results
    }
    runtime = Path(echarts_path).read_text(encoding="utf-8") if echarts_path and Path(echarts_path).is_file() else None
    combined = len(results) > 1 and str(options.get("layout", "Combined")) == "Combined"
    charts = (
        [("mapping_combined.html", combined_mapping_plot_options(results, colors=colors, line_types=line_types, curve_names=curve_names, display_names=display_names, **settings))]
        if combined else
        [(f"mapping_{named.key}.html", mapping_plot_options(named.result, **settings)) for named in results]
    )
    paths = []
    for filename, specification in charts:
        path = folder / filename
        write_plot_html(specification, path, title=str(settings.get("title", "Mapping")), echarts_source=runtime,
                        dpi=int(options.get("plot_export_dpi", 150)))
        paths.append(path)
    return tuple(paths)
