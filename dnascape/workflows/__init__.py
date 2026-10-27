"""Headless workflows shared by the DNAscape interface and exported scripts.

These functions apply the same preparation, validation, execution and export
rules as the interactive pages. Numerical and plotting dependencies load only
when a workflow is first requested; importing this namespace needs no UI.
The page workflows require Python 3.10 or newer.
"""

from __future__ import annotations

from importlib import import_module

_PUBLIC_BY_MODULE = {
    "mapping": (
        "MappingComparison",
        "MappingRequest",
        "MappingResult",
        "materialize_many_tracks",
        "materialize_track_upload",
        "run_mapping",
        "validate_mapping_request",
    ),
    "mapping_catalog": (
        "MappingCatalog",
        "MappingRoute",
        "get_mapping_catalog",
        "mapping_route",
        "mapping_routes",
        "mapping_target",
    ),
    "simulation": (
        "ConstantSimulationRequest",
        "ConstantSimulationResult",
        "run_constant_simulation",
        "validate_constant_simulation",
    ),
    "conversion": (
        "DataConversionRequest",
        "DataConversionResult",
        "PreparedData",
        "convert_data",
        "convert_combined_data",
        "inspect_data_source",
        "prepare_data",
    ),
    "filtering": ("DataFilterSettings", "FilterProfile", "apply_data_filter", "parse_position_ranges"),
    "modulation": ("ProfileModulationSettings", "ModulationProfile"),
    "formulas": ("evaluate_formula",),
    "uploads": (
        "UploadInspection",
        "inspect_upload",
        "materialize_many",
        "materialize_upload",
    ),
    "mapping_exports": (
        "build_mapping_comparison_raw_export",
        "build_mapping_raw_export",
    ),
    "exports": (
        "ExportArtifact",
        "build_raw_export",
        "compressed_export_artifact",
    ),
    "mapping_recipe": (
        "prepare_mapping_recipe",
        "run_mapping_recipe",
    ),
    "recipe_exports": (
        "plot_mapping_outputs",
        "save_mapping_outputs",
    ),
    "code_recipes": ("run_data_manager_recipe",),
    "simulation_recipe": ("prepare_simulation_recipe", "run_simulation_recipe"),
    "analysis_recipe": ("prepare_analysis_recipe", "run_plotting_recipe"),
    "plotting": (
        "combined_mapping_plot_options",
        "mapping_plot_options",
        "write_plot_html",
    ),
}

_SUBMODULES = (
    "batch",
    "code_recipes",
    "simulation_recipe",
    "analysis_recipe",
    "bigwig",
    "conversion",
    "exports",
    "export_settings",
    "filtering",
    "modulation",
    "mapping",
    "mapping_catalog",
    "mapping_defaults",
    "mapping_exports",
    "mapping_recipe",
    "plotting",
    "preview",
    "recipe_exports",
    "result_naming",
    "simulation",
    "uploads",
)

_MODULE_FOR_NAME = {
    name: module_name
    for module_name, names in _PUBLIC_BY_MODULE.items()
    for name in names
}
__all__ = tuple(_MODULE_FOR_NAME)


def __getattr__(name: str):
    module_name = _MODULE_FOR_NAME.get(name)
    if module_name is not None:
        value = getattr(import_module(f".{module_name}", __name__), name)
    elif name in _SUBMODULES:
        value = import_module(f".{name}", __name__)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__) | set(_SUBMODULES))
