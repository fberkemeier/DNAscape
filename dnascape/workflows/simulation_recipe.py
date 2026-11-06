"""Headless Simulation recipes with the page's shared batch planner."""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np

from .batch import (BatchSourceSummary, CombinationSource, combination_names,
                    plan_batch_sources, plan_combination_indices, unique_dataset_names)
from .result_naming import ResultName, unique_compact_export_stems

_MAX_SIMULATION_CASES = 50_000


def prepare_simulation_recipe(recipe, *, input_directory="inputs"):
    from .code_recipes import restore_settings
    from .simulation import ConstantSimulationRequest, validate_constant_simulation
    from .uploads import materialize_upload

    settings = restore_settings(recipe["settings"], input_directory=input_directory)
    sources = []
    for key, label, units in (("initiation_rate", "Initiation rate", "min⁻¹ per bin"),
                              ("fork_speed", "Fork speed", "kb/min")):
        source = recipe["inputs"][key]
        form = source["form"]
        files = []
        if source["mode"] == "profile":
            files = [restore_settings(item["source"], input_directory=input_directory) for item in source["files"]]
            values = tuple(materialize_upload(
                replace(inspection, expected_form=form), column=item.get("column") if form in {"x", "t"} else None,
                array_key=item.get("array_key"),
                missing_policy="ignore" if key in settings.get("data_filters", {}) else settings.get("missing_values", "error"),
            ) for inspection, item in zip(files, source["files"], strict=True))
        else:
            values = tuple(source["values"])
        if not values:
            raise ValueError(f"Add at least one {label.lower()} source.")
        components = (unique_dataset_names(tuple(item.name for item in files)) if files else
                      tuple(_format_simulation_component(value, units=units) for value in values))
        summary = (BatchSourceSummary(label, tuple(item.name for item in files),
                                     tuple(form for _item in files), tuple(np.shape(value) for value in values))
                   if files else None)
        sources.append(_PreparedSimulationSource(label, values, components, summary))
    # Movement profiles are shared by the existing initiation/speed batch. They
    # use the same upload selections and native grid as the live page.
    for key in ("stall_rate_per_kb", "stall_duration", "stall_recovery_rate"):
        source = recipe["inputs"].get(key)
        if source is None:
            continue
        if source["mode"] == "profile":
            if len(source["files"]) != 1:
                raise ValueError("Choose one spatial profile per fork-movement input.")
            item = source["files"][0]
            inspection = restore_settings(item["source"], input_directory=input_directory)
            if inspection.inferred_form == "t":
                raise ValueError(f"{key} requires a spatial profile.")
            value = materialize_upload(replace(inspection, expected_form="x"),
                                       column=item.get("column"), array_key=item.get("array_key"),
                                       missing_policy=settings.get("missing_values", "error"))
        else:
            if len(source["values"]) != 1:
                raise ValueError("Choose one constant per fork-movement input.")
            value = source["values"][0]
        settings[key] = value
    cases = _plan_simulation_cases(sources, combination_mode=recipe.get("combination_mode", "pairwise"))
    configured_end = float(recipe["region_end_kb"])
    start = float(settings["region_start_kb"])
    use_selected_end = configured_end != recipe.get("automatic_region_end", configured_end) or start != 0.
    prepared = []
    for case in cases:
        fields = [source.values[index] for source, index in zip(sources, case.item_indices, strict=True)]
        grid_fields = fields + [settings[key] for key in ("stall_rate_per_kb", "stall_duration", "stall_recovery_rate")
                                if settings.get(key) is not None]
        sizes = {np.asarray(value).shape[0] for value in grid_fields
                 if np.asarray(value).ndim == 2 and np.asarray(value).shape[0] > 1}
        sizes.update(np.asarray(value).size for value in grid_fields if np.asarray(value).ndim == 1 and np.asarray(value).size > 1)
        if len(sizes) > 1:
            raise ValueError("Spatial kinetic input axes disagree.")
        native_spacing = float(settings.get("source_resolution_x") or settings["resolution_x"])
        end = start + next(iter(sizes)) * native_spacing if sizes and not use_selected_end else configured_end
        request = ConstantSimulationRequest(initiation_rate=fields[0], fork_speed=fields[1],
                                            region_end_kb=end, **settings)
        validated = validate_constant_simulation(request)
        prepared.append((case, validated.request if request.data_filters else request))
    return tuple(prepared)


def run_simulation_recipe(recipe, *, input_directory="inputs", output_directory="outputs"):
    from .simulation import run_constant_simulation
    from .preview_inputs import add_simulation_input_previews

    results = []
    paths = []
    prepared = prepare_simulation_recipe(recipe, input_directory=input_directory)
    export_options = {**recipe.get("export", {}), "multiple_datasets": len(prepared) > 1}
    for case, request in prepared:
        result = run_constant_simulation(request)
        result = add_simulation_input_previews(result, request)
        results.append((case, result))
        _save_simulation_result(case, result, export_options, output_directory, paths)
    if export_options.get("statistics"):
        from .code_recipes import save_recipe_statistics
        from .exports import select_result_output
        settings = export_options["statistics"]
        selected = export_options.get("output", "replication_timing")
        selected_results = results
        if settings.get("dataset_index") is not None:
            selected_results = (results[min(max(0, int(settings["dataset_index"])), len(results) - 1)],)
        datasets = [(case.name, select_result_output(result, selected).values) for case, result in selected_results
                    if not settings.get("dataset") or settings["dataset"] == case.key]
        save_recipe_statistics(datasets, settings, output_directory, paths)
    return tuple(results), tuple(paths)


def _save_simulation_result(case, result, options, directory, paths):
    from .code_recipes import write_artifact
    from .exports import ExportArtifact, build_raw_export
    from .result_naming import output_file_token
    from .simulation_filters import build_simulation_filter_export
    import io
    import zipfile

    selected = options.get("output", "replication_timing")
    stem = case.full_export_stem if options.get("name_style") == "full" else case.short_export_stem
    suffix = "" if stem == "dataset" else "_" + stem
    defaults = {"file_type": options.get("file_type", "TSV"), "compress": options.get("compress", False)}

    def preference(key, name):
        return {**defaults, "file_name": name, **options.get("per_dataset", {}).get(case.key, {}).get(key, {})}

    def compression(settings):
        return options.get("compression", "gzip") if settings.get("compress") else "none"

    keys = tuple(result.outputs) if selected == "all" else (selected,)
    for key in keys:
        settings = preference(key, f"rsim_{output_file_token(key)}{suffix}")
        if options.get("filename") and len(keys) == 1:
            settings["file_name"] = options["filename"] + (suffix if options.get("multiple_datasets") else "")
        artifact = build_raw_export(result, key, settings["file_type"], filename=settings["file_name"],
                                    compression=compression(settings))
        write_artifact(artifact, directory, paths)
    if options.get("standard_deviation") and result.replication_timing_std is not None:
        settings = preference("standard_deviation", f"rsim_rt_standard_deviation{suffix}")
        selected_result = replace(result, outputs={"replication_timing_std": result.replication_timing_std})
        write_artifact(build_raw_export(selected_result, "replication_timing_std", settings["file_type"],
                       filename=settings["file_name"], compression=compression(settings), axis_kind="space"), directory, paths)
    if options.get("individual_profiles") and result.replication_time_profiles is not None:
        count = min(int(options.get("profile_count", 10)), len(result.replication_time_profiles))
        settings = preference("profiles", f"rsim_rt_individual_profiles{suffix}")
        import re
        name = re.sub(r"_\d+_replicates?$", "", settings["file_name"], flags=re.I) + f"_{count}_replicates.zip"
        data = io.BytesIO()
        with zipfile.ZipFile(data, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for index, values in enumerate(result.replication_time_profiles[:count], 1):
                selected_result = replace(result, replication_timing=values, outputs={"replication_timing": values})
                artifact = build_raw_export(selected_result, "replication_timing", settings["file_type"],
                                           filename=f"replicate_{index:03d}", axis_kind="space")
                archive.writestr(artifact.filename, artifact.data)
        write_artifact(ExportArtifact(data.getvalue(), name, "application/zip"), directory, paths)
    if result.filter_state is not None:
        for key in sorted(result.filter_state.exports):
            settings = preference(key + "_filter", f"rsim_{output_file_token(key + '_filter')}{suffix}")
            if key not in result.filter_state.selections:
                continue
            artifact = build_simulation_filter_export(result.filter_state, key, settings["file_type"],
                filename=settings["file_name"], layout=settings.get("layout", "positions"), compression=compression(settings))
            write_artifact(artifact, directory, paths)

@dataclass(frozen=True, slots=True)
class _PreparedSimulationSource:
    """One fully materialized Simulation factor and its naming metadata."""

    label: str
    values: tuple[float | np.ndarray, ...]
    components: tuple[str, ...]
    summary: BatchSourceSummary | None = None


@dataclass(frozen=True, slots=True)
class _SimulationBatchCase:
    """One explicit pairwise or Cartesian Simulation request."""

    key: str
    identity: ResultName
    full_export_stem: str
    short_export_stem: str
    item_indices: tuple[int, ...]

    @property
    def name(self) -> str:
        """Retain the fully descriptive legacy planner interface."""

        return self.identity.full

    @property
    def export_stem(self) -> str:
        """Retain the fully descriptive legacy export-stem interface."""

        return self.full_export_stem


def _format_simulation_component(value: float, *, units: str = "") -> str:
    """Format a constant compactly for descriptive batch names."""

    numeric = f"{float(value):.10g}"
    return f"{numeric} {units}" if units else numeric


def _plan_simulation_cases(
    inputs: Sequence[_PreparedSimulationSource],
    *,
    combination_mode: str = "pairwise",
) -> tuple[_SimulationBatchCase, ...]:
    """Resolve explicit pairwise or Cartesian cases before starting rsim."""

    for item in inputs:
        if not item.values:
            raise ValueError(f"{item.label} must contain at least one value or file.")
        if item.summary is not None:
            plan_batch_sources((item.summary,), require_matching_shapes=False)
        if len(item.components) != len(item.values):
            raise ValueError(f"{item.label} naming metadata are incomplete.")

    mode = "all" if combination_mode == "all" else "pairwise"
    source_sizes = tuple(len(item.values) for item in inputs)
    projected = (
        math.prod(source_sizes)
        if mode == "all"
        else max(source_sizes, default=1)
    )
    if projected > _MAX_SIMULATION_CASES:
        raise ValueError(
            f"These inputs would create {projected:,} simulation runs. "
            f"Reduce the inputs or use at most {_MAX_SIMULATION_CASES:,} combinations."
        )
    combinations = plan_combination_indices(source_sizes, mode=mode)
    identities = combination_names(
        "Simulation",
        tuple(
            CombinationSource(
                item.label,
                item.components,
                include_singleton_in_short=item.summary is not None,
            )
            for item in inputs
        ),
        combinations,
        include_singletons=True,
    )
    full_stems = unique_compact_export_stems(
        tuple(_simulation_case_label(identity.full) for identity in identities),
        fallback="dataset",
        maximum=110,
    )
    short_stems = unique_compact_export_stems(
        tuple(_simulation_case_label(identity.short) for identity in identities),
        fallback="dataset",
        maximum=110,
    )
    # A single run driven entirely by fixed scalar assumptions has no useful
    # case identity in an exported filename. Keep its full label descriptive
    # in the UI, but retain the concise workflow/output filename convention.
    if len(combinations) == 1 and not any(item.summary is not None for item in inputs):
        full_stems = ("dataset",)
        short_stems = ("dataset",)
    cases: list[_SimulationBatchCase] = []
    for identity, full_stem, short_stem, indices in zip(
        identities,
        full_stems,
        short_stems,
        combinations,
        strict=True,
    ):
        cases.append(
            _SimulationBatchCase(
                key=f"simulation-case-{full_stem}",
                identity=identity,
                full_export_stem=full_stem,
                short_export_stem=short_stem,
                item_indices=tuple(indices),
            )
        )
    return tuple(cases)


def _simulation_case_label(display_name: str) -> str:
    """Remove the workflow prefix before constructing a case-only stem."""

    candidate = str(display_name).strip()
    prefix = "Simulation · "
    if candidate.startswith(prefix):
        return candidate.removeprefix(prefix)
    return "dataset" if candidate == "Simulation" else candidate

