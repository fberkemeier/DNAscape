"""DNAscape public API.

The package root exposes the established convenience API lazily.  Importing
``dnascape`` therefore does not import plotting, SciPy, pandas, or the route
registry until one of their public objects is requested.  Code that
intentionally needs the historical eager namespace can still import
``dnascape.core``.
"""

from __future__ import annotations

from importlib import import_module

_PUBLIC_BY_MODULE = {
    "constants": (
        "NameMapping",
        "mapt",
    ),
    "io": (
        "load",
        "loadtxt",
        "loadcsv",
        "savetxt",
        "savecsv",
        "csv_to_tsv",
        "infer_rdata_source_format",
        "inspect_genomic_track",
        "rload",
        "rsave",
        "rtrack",
        "rdata",
        "compress_to_gz",
        "load_gene_models_from_gtf_gz",
    ),
    "utils": (
        "logistic",
        "interp_nans",
        "smoothf",
        "rescale",
        "fast_shift_add",
        "match_yaxis_scales",
        "mean_squared_error",
        "hms",
        "arrint",
        "arrcomp",
        "etaf",
        "etaferr",
    ),
    "plotting": (
        "plot_gene_track",
        "plotf",
        "mplotf",
        "ploth",
        "plotb",
        "plotd",
        "plotw",
        "plothm",
    ),
    "mapping": (
        "map_replicated_fraction_xt",
        "map_replicated_fraction",
        "map_fork_densities_xt",
        "map_fork_densities_directionality",
        "map_initiation_events_xt",
        "map_initiation_events",
        "map_initiation_events_profile",
        "map_probability_density_xt",
        "map_probability_density_profile",
        "map_origin_efficiency",
        "map_coalescence_events_xt",
        "map_coalescence_events_profile",
        "bwmap",
        "map_firing_timing",
        "map_timing_directionality",
        "map_timing_directionality_fork_speed",
        "map_directionality_timing",
    ),
    "simulation": (
        "simulate_replication",
        "rsim",
        "stall_probability_per_bin",
    ),
    "fitting": (
        "select_profile_region",
        "map_timing_firing",
        "map_timing_firing_time",
        "map_timing_firing_spacetime",
        "map_timing_constant_firing",
        "map_timing_firing_constant_speed",
        "map_timing_firing_spatial_speed",
        "map_timing_firing_temporal_speed",
        "map_timing_firing_spacetime_speed",
        "map_timing_firing_constant_speed_simulated",
        "map_timing_firing_spatial_speed_simulated",
        "map_timing_firing_temporal_speed_simulated",
        "map_timing_firing_spacetime_speed_simulated",
        "rfit",
    ),
    "routing": (
        "MappingSpec",
        "DataTypeMapper",
        "rmap",
    ),
}

_SUBMODULES = (
    "constants",
    "io",
    "utils",
    "plotting",
    "mapping",
    "simulation",
    "fitting",
    "routing",
    "core",
    "workflows",
)

_PUBLIC_MODULE_FOR_NAME = {
    name: module_name
    for module_name, names in _PUBLIC_BY_MODULE.items()
    for name in names
}

__all__ = tuple(
    name
    for names in _PUBLIC_BY_MODULE.values()
    for name in names
)


def __getattr__(name: str):
    """Load a public object or submodule on first access."""
    module_name = _PUBLIC_MODULE_FOR_NAME.get(name)
    if module_name is not None:
        value = getattr(import_module(f".{module_name}", __name__), name)
        globals()[name] = value
        return value
    if name in _SUBMODULES:
        value = import_module(f".{name}", __name__)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__) | set(_SUBMODULES))
