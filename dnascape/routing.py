"""High-level route registry for DNAscape feature mapping."""

import inspect
from dataclasses import dataclass

import numpy as np

from .fitting import (
    map_timing_constant_firing,
    map_timing_firing,
    map_timing_firing_constant_speed,
    map_timing_firing_spatial_speed,
    map_timing_firing_constant_speed_simulated,
    map_timing_firing_spatial_speed_simulated,
    map_timing_firing_temporal_speed_simulated,
    map_timing_firing_spacetime_speed_simulated,
    map_timing_firing_spacetime_speed,
    map_timing_firing_spacetime,
    map_timing_firing_temporal_speed,
    map_timing_firing_time,
    map_timing_joint_spacetime_kinetics,
)
from .mapping import map_directionality_timing, map_firing_timing, map_timing_directionality
from .simulation import rsim

__all__ = ["MappingSpec", "DataTypeMapper", "rmap"]

_FEATURE_ALIASES = {
    "fr": "firing_rate",
    "firing": "firing_rate",
    "origin_activity": "firing_rate",
    "initiation": "firing_rate",
    "initiation_rate": "firing_rate",
    "rt": "replication_timing",
    "timing": "replication_timing",
    "replication_time": "replication_timing",
    "rfd": "fork_directionality",
    "directionality": "fork_directionality",
    "fork_direction": "fork_directionality",
    "fs": "fork_speed",
    "fss": "fork_speed",
    "v": "fork_speed",
    "speed": "fork_speed",
    "fork_speeds": "fork_speed",
    "ori_rate": "firing_rate",
    "origin_rate": "firing_rate",
    "kinetics": "kinetic_fields",
    "replication_kinetics": "kinetic_fields",
    "kinetic_profile": "kinetic_fields",
    "kinetic_profiles": "kinetic_fields",
    "firing_and_fork_speed": "kinetic_fields",
    "firing_rate_and_fork_speed": "kinetic_fields",
    "initiation_and_fork_speed": "kinetic_fields",
    "eff": "origin_efficiency",
    "efficiency": "origin_efficiency",
    "iod": "inter_origin_distance",
    "inter_origin_distances": "inter_origin_distance",
    "s_phase_duration": "completion_time",
    "nori": "num_origins",
    "num_oris": "num_origins",
    "number_of_origins": "num_origins",
    "right_moving_fork_density": "right_fork_density",
    "left_moving_fork_density": "left_fork_density",
    "total_fork_density": "fork_density",
    "total_fork_density_xt": "fork_density_xt",
}

_CANONICAL_FEATURES = {
    "firing_rate",
    "fork_speed",
    "kinetic_fields",
    "replication_timing",
    "fork_directionality",
    "replicated_fraction",
    "fork_density",
    "fork_density_xt",
    "right_fork_density",
    "left_fork_density",
    "origin_efficiency",
    "inter_origin_distance",
    "completion_time",
    "num_origins",
    "initiation_events",
    "coalescence_events",
    "replicated_fraction_xt",
    "initiation_events_xt",
    "coalescence_events_xt",
    "right_fork_density_xt",
    "left_fork_density_xt",
}

_FEATURE_DESCRIPTIONS = {
    "firing_rate": "Origin firing activity to infer or supply.",
    "fork_speed": "Fork progression speed to infer or supply.",
    "kinetic_fields": "Jointly fitted firing-rate and fork-speed fields.",
    "replication_timing": "Replication timing along genomic position.",
    "fork_directionality": "Net fork movement directionality along position.",
    "replicated_fraction": "Fraction of replicated DNA over time.",
    "replicated_fraction_xt": "Space-time replicated-fraction field.",
    "fork_density": "Active fork density over time.",
    "fork_density_xt": "Space-time active fork-density field.",
    "right_fork_density": "Right-moving fork density over time.",
    "left_fork_density": "Left-moving fork density over time.",
    "right_fork_density_xt": "Space-time right-moving fork-density field.",
    "left_fork_density_xt": "Space-time left-moving fork-density field.",
    "origin_efficiency": "Origin efficiency estimated from simulated initiation usage.",
    "inter_origin_distance": "Distances between neighbouring fired origins.",
    "completion_time": "Replication completion time across simulated S phases.",
    "num_origins": "Number of origins fired across simulated S phases.",
    "initiation_events": "Initiation-event summary generated from simulation.",
    "coalescence_events": "Coalescence-event summary generated from simulation.",
    "initiation_events_xt": "Space-time initiation-event field.",
    "coalescence_events_xt": "Space-time coalescence-event field.",
}

_ASSUMPTION_DESCRIPTIONS = {
    "target_form": "Requested output form: scalar (constant in space and time), array (space-dependent), transposed array (time-dependent), or matrix (space-time dependent).",
    "resolution": "Spatial resolution along the genome, in kb.",
    "resolution_time": "Temporal resolution, in minutes.",
    "time_grid": "Explicit time axis to use for time-dependent fields.",
    "T0": "Reference start time used when converting timing values to a relative time axis.",
    "neighbourhood_range": "Spatial range, in kb, used by deterministic neighbourhood mappings.",
    "fit_step": "Global multiplicative update size. For x,t targets it sets fit_step_x and fit_step_t unless those are supplied explicitly.",
    "maxiter": "Global iteration count. For x,t targets it overrides maxiter_x and maxiter_t, setting both to the same value.",
    "sim_number": "Global number of stochastic simulations used by the active fitting phase when a route calls rsim.",
    "err_threshold": "Lower numerical floor for fitted firing-rate values.",
    "deviation_threshold": "Tolerance below which a spatial firing update can be left unchanged.",
    "rate_bounds": "Allowed lower and upper bounds for scalar firing-rate fitting.",
    "speed_bounds": "Allowed lower and upper bounds for fork-speed fitting.",
    "initial_fork_speed": "Starting fork-speed value or profile for fork-speed fitting.",
    "initial_firing_rate": "Starting firing-rate value or field for joint kinetic fitting.",
    "initial_kinetics": "Dictionary with initial firing_rate and/or fork_speed fields.",
    "space_fit_step": "Spatial x-phase update size; overrides fit_step for x-only initialization or refinement.",
    "space_maxiter": "Number of x-phase iterations; for mixed x,t fits it contributes to the mixed iteration count.",
    "temporal_smoothw": "Optional smoothing window for t-dependent profiles.",
    "temporal_fit_step": "Time-profile t-phase update size; controls t targets and factorized x,t targets.",
    "temporal_maxiter": "Number of t-phase refinement iterations; for mixed x,t fits it contributes to the mixed iteration count.",
    "temporal_sim_number": "Number of simulations per t-phase refinement iteration.",
    "spacetime_fit_step": "Mixed local x,t update size; used only when spacetime_update='mixed'.",
    "spacetime_space_fit_step": "Legacy internal x component for factorized x,t corrections.",
    "spacetime_time_fit_step": "Legacy internal t component for factorized x,t corrections.",
    "spacetime_sim_number": "Number of simulations per mixed x,t refinement iteration.",
    "spacetime_update": "Space-time update rule: 'factorized' separates x and t phases; 'mixed' fits local x,t corrections.",
    "spacetime_order": "Order of factorized x,t phases: 'space_then_time' by default, or 'time_then_space'.",
    "spacetime_kernel_t": "Temporal kernel width, in time bins, for mixed x,t corrections.",
    "refine_ratio_bounds": "Bounds applied to timing-ratio updates during simulation refinement.",
    "refine_edge_kb": "Non-periodic chromosome edge width, in kb, ignored during refinement.",
    "refine_seed": "Seed offset for simulation-refined fitting routes.",
    "refine_verbose": "Whether refinement sub-loops print ETA and fitting error.",
    "floor": "Lower numerical floor for fitted firing-rate fields.",
    "smoothw": "Spatial smoothing window for fitted profiles.",
    "stall_rate": "Fork-stalling rate used when fitting with stochastic simulation.",
    "tau": "Fork-stall lifetime parameter used by stochastic simulation.",
    "perQ": "Whether the genome is treated as periodic/circular.",
    "genome_length": "Genome or interval length, in kb, used to infer n_bins.",
    "interval_kb": "Genomic interval, in kb, used to infer or select a region.",
    "space_init_fit_step": "x-phase update size for the initial joint-kinetics firing fit.",
    "space_init_maxiter": "Iterations for the initial joint-kinetics x fit.",
    "space_init_sim_number": "Simulation count for the initial joint-kinetics x fit.",
    "joint_maxiter": "Iterations for simultaneous firing-rate and fork-speed fitting.",
    "joint_sim_number": "Simulations per joint fitting iteration.",
    "firing_space_fit_step": "x update size for firing rate in joint fitting.",
    "firing_time_fit_step": "t update size for firing rate in joint fitting.",
    "speed_space_fit_step": "x update size for fork speed in joint fitting.",
    "speed_time_fit_step": "t update size for fork speed in joint fitting.",
    "firing_bounds": "Allowed lower and upper bounds for firing-rate fields.",
    "firing_floor": "Lower numerical floor for joint-fitted firing-rate fields.",
    "spatial_smoothw": "x smoothing window used in joint kinetic fitting.",
}

_ASSUMPTION_ORDER = (
    "target_form",
    "resolution",
    "resolution_time",
    "time_grid",
    "T0",
    "interval_kb",
    "genome_length",
    "perQ",
    "neighbourhood_range",
    "initial_firing_rate",
    "initial_fork_speed",
    "initial_kinetics",
    "rate_bounds",
    "speed_bounds",
    "firing_bounds",
    "fit_step",
    "maxiter",
    "sim_number",
    "space_fit_step",
    "space_maxiter",
    "temporal_fit_step",
    "temporal_maxiter",
    "temporal_sim_number",
    "temporal_smoothw",
    "spacetime_fit_step",
    "spacetime_sim_number",
    "spacetime_update",
    "spacetime_order",
    "spacetime_kernel_t",
    "space_init_fit_step",
    "space_init_maxiter",
    "space_init_sim_number",
    "joint_maxiter",
    "joint_sim_number",
    "firing_space_fit_step",
    "firing_time_fit_step",
    "speed_space_fit_step",
    "speed_time_fit_step",
    "err_threshold",
    "deviation_threshold",
    "floor",
    "firing_floor",
    "smoothw",
    "spatial_smoothw",
    "refine_ratio_bounds",
    "refine_edge_kb",
    "refine_seed",
    "refine_verbose",
    "stall_rate",
    "tau",
)

_ASSUMPTION_FALLBACK_DEFAULTS = {
    "temporal_fit_step": 0.35,
    "temporal_maxiter": 4,
    "temporal_sim_number": 100,
    "spacetime_fit_step": 0.25,
    "spacetime_sim_number": 100,
    "spacetime_update": "factorized",
    "spacetime_order": "space_then_time",
    "spacetime_kernel_t": 1.0,
    "sim_number": 100,
    "space_init_sim_number": 100,
    "joint_sim_number": 100,
}

_FORM_DESCRIPTIONS = {
    "scalar": "scalar (constant in space and time)",
    "constant": "scalar (constant in space and time)",
    "x": "array (space-dependent)",
    "space": "array (space-dependent)",
    "t": "transposed array (time-dependent)",
    "time": "transposed array (time-dependent)",
    "xt": "matrix (space-time dependent)",
    "spacetime": "matrix (space-time dependent)",
    "supplied": "supplied data",
}

_PUBLIC_MAPPING_TARGETS = (
    "firing_rate",
    "fork_speed",
    "kinetic_fields",
    "replication_timing",
    "fork_directionality",
)

_ASSUMPTION_ALIASES = {
    "resolution_x": "resolution",
    "resolution_x_kb": "resolution",
    "resolution_space": "resolution",
    "resolution_space_kb": "resolution",
    "bin_size": "resolution",
    "resolution_kb": "resolution",
    "dx": "resolution",
    "resolution_t": "resolution_time",
    "resolution_t_min": "resolution_time",
    "dt": "resolution_time",
    "fit_step_x": "space_fit_step",
    "x_fit_step": "space_fit_step",
    "space_fit_step": "space_fit_step",
    "maxiter_x": "space_maxiter",
    "x_maxiter": "space_maxiter",
    "space_maxiter": "space_maxiter",
    "fit_step_t": "temporal_fit_step",
    "t_fit_step": "temporal_fit_step",
    "time_fit_step": "temporal_fit_step",
    "temporal_fit_step": "temporal_fit_step",
    "maxiter_t": "temporal_maxiter",
    "t_maxiter": "temporal_maxiter",
    "time_maxiter": "temporal_maxiter",
    "temporal_maxiter": "temporal_maxiter",
    "sim_number_t": "temporal_sim_number",
    "t_sim_number": "temporal_sim_number",
    "time_sim_number": "temporal_sim_number",
    "temporal_sim_number": "temporal_sim_number",
    "smoothw_t": "temporal_smoothw",
    "t_smoothw": "temporal_smoothw",
    "temporal_smoothw": "temporal_smoothw",
    "fit_step_xt": "spacetime_fit_step",
    "xt_fit_step": "spacetime_fit_step",
    "spacetime_fit_step": "spacetime_fit_step",
    "fit_step_xt_x": "spacetime_space_fit_step",
    "xt_x_fit_step": "spacetime_space_fit_step",
    "spacetime_space_fit_step": "spacetime_space_fit_step",
    "fit_step_xt_t": "spacetime_time_fit_step",
    "xt_t_fit_step": "spacetime_time_fit_step",
    "spacetime_time_fit_step": "spacetime_time_fit_step",
    "sim_number_xt": "spacetime_sim_number",
    "xt_sim_number": "spacetime_sim_number",
    "spacetime_sim_number": "spacetime_sim_number",
    "spacetime_update": "spacetime_update",
    "update_xt": "spacetime_update",
    "xt_update": "spacetime_update",
    "spacetime_order": "spacetime_order",
    "fit_order": "spacetime_order",
    "spacetime_fit_order": "spacetime_order",
    "factorized_order": "spacetime_order",
    "factorised_order": "spacetime_order",
    "kernel_xt_t": "spacetime_kernel_t",
    "xt_t_kernel": "spacetime_kernel_t",
    "spacetime_kernel_t": "spacetime_kernel_t",
    "fit_step_init_x": "space_init_fit_step",
    "maxiter_init_x": "space_init_maxiter",
    "sim_number_init_x": "space_init_sim_number",
    "maxiter_joint": "joint_maxiter",
    "sim_number_joint": "joint_sim_number",
    "fit_step_firing_x": "firing_space_fit_step",
    "fit_step_firing_t": "firing_time_fit_step",
    "fit_step_speed_x": "speed_space_fit_step",
    "fit_step_speed_t": "speed_time_fit_step",
    "smoothw_x": "spatial_smoothw",
    "periodic": "perQ",
    "perq": "perQ",
}


def _canonical_feature(name):
    text = str(name).strip().replace("-", "_")
    key = text.lower()
    if key in _FEATURE_ALIASES:
        return _FEATURE_ALIASES[key]
    if key in _CANONICAL_FEATURES:
        return key
    return text


def _canonical_assumption(name):
    text = str(name).strip().replace("-", "_")
    key = text.lower()
    if key in _ASSUMPTION_ALIASES:
        return _ASSUMPTION_ALIASES[key]
    return _canonical_feature(text)


def _geometry_to_perq(value):
    if isinstance(value, str):
        key = value.strip().lower()
        if key in {"periodic", "circular", "circle"}:
            return True
        if key in {"linear", "chromosome", "open"}:
            return False
    return None


def _field_kind(value):
    """Return the modelling form used for route selection.

    Scalars are constant in space and time. One-dimensional arrays are
    interpreted as spatial profiles. Two-dimensional row vectors are treated
    as time-only profiles, while other two-dimensional arrays are treated as
    space-time fields.
    """
    arr = np.asarray(value)
    if arr.size == 1:
        return "scalar"
    if arr.ndim == 0:
        return "scalar"
    if arr.ndim == 1:
        return "x"
    if arr.ndim == 2:
        if arr.shape[0] == 1 and arr.shape[1] > 1:
            return "t"
        if arr.shape[1] == 1 and arr.shape[0] > 1:
            return "x"
        return "xt"
    return f"{arr.ndim}d"


def _canonical_form(value):
    if value is None:
        return None
    text = str(value).strip().lower().replace(" ", "")
    aliases = {
        "0": "constant",
        "scalar": "constant",
        "constant": "constant",
        "const": "constant",
        "i0": "constant",
        "v0": "constant",
        "x": "space",
        "array": "space",
        "1d": "space",
        "profile": "space",
        "spatial": "space",
        "space": "space",
        "i(x)": "space",
        "v(x)": "space",
        "t": "time",
        "transposedarray": "time",
        "transposed_array": "time",
        "rowvector": "time",
        "row_vector": "time",
        "temporal": "time",
        "time": "time",
        "i(t)": "time",
        "v(t)": "time",
        "xt": "spacetime",
        "matrix": "spacetime",
        "2d": "spacetime",
        "field": "spacetime",
        "x,t": "spacetime",
        "(x,t)": "spacetime",
        "space-time": "spacetime",
        "space_time": "spacetime",
        "spacetime": "spacetime",
        "i(x,t)": "spacetime",
        "v(x,t)": "spacetime",
    }
    return aliases.get(text, text)


def _canonical_spacetime_update(value):
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "factorised": "factorized",
        "factorized": "factorized",
        "separable": "factorized",
        "row_column": "factorized",
        "mixed": "mixed",
        "cellwise": "mixed",
        "cell_wise": "mixed",
        "local": "mixed",
    }
    return aliases.get(text, text)


@dataclass(frozen=True)
class MappingSpec:
    source: tuple
    target: str
    method: str
    func: object = None
    implemented: bool = True
    required_assumptions: tuple = ()
    optional_assumptions: tuple = ()
    default_assumptions: tuple = ()
    source_kinds: tuple = ()
    assumption_kinds: tuple = ()
    assumption_values: tuple = ()
    description: str = ""

    def defaults_dict(self):
        return dict(self.default_assumptions)

    def assumptions_with_defaults(self, assumptions):
        out = self.defaults_dict()
        out.update(assumptions)
        return out

    def missing_required(self, assumptions):
        merged = self.assumptions_with_defaults(assumptions)
        return tuple(name for name in self.required_assumptions if name not in merged)

    def _source_kind_matches(self, source_data):
        for name, kinds in self.source_kinds:
            cname = _canonical_feature(name)
            if cname not in source_data or source_data[cname] is None:
                continue
            kind = _field_kind(source_data[cname])
            if kind not in kinds:
                return False
        return True

    def _assumption_value_matches(self, assumptions):
        merged = self.assumptions_with_defaults(assumptions)
        for name, allowed in self.assumption_values:
            cname = _canonical_assumption(name)
            allowed_forms = tuple(_canonical_form(value) for value in allowed)
            if cname not in merged:
                if None in allowed_forms:
                    continue
                return False
            value = _canonical_form(merged[cname])
            if value not in allowed_forms:
                return False
        return True

    def accepts(self, source_data, assumptions):
        if not self._source_kind_matches(source_data):
            return False
        if not self._assumption_value_matches(assumptions):
            return False
        if self.missing_required(assumptions):
            return False
        merged = self.assumptions_with_defaults(assumptions)
        for name, kinds in self.assumption_kinds:
            cname = _canonical_assumption(name)
            if cname not in merged:
                continue
            kind = _field_kind(merged[cname])
            if kind not in kinds:
                return False
        return True

    def implemented_status(self):
        return "implemented" if self.implemented else "needs method"


_RSIM_OPTION_NAMES = tuple(
    name for name in inspect.signature(rsim).parameters
    if name not in {"ori_rate", "firing_rate", "initiation_rate"}
)
_RSIM_ASSUMPTION_NAMES = tuple(
    name for name in _RSIM_OPTION_NAMES
    if name != "fork_speed"
)


def _simulation_feature_mapper(output_name, default_options=None):
    def mapper(firing_rate, **kwargs):
        options = dict(default_options or {})
        options.update({
            key: value
            for key, value in kwargs.items()
            if key in _RSIM_OPTION_NAMES
        })
        return rsim(firing_rate=firing_rate, **options)[output_name]

    mapper.__name__ = f"map_firing_to_{output_name}_simulation"
    return mapper


def _simulation_source_feature_mapper(output_name, default_options=None):
    def mapper(source, **kwargs):
        options = dict(default_options or {})
        options.update({
            key: value
            for key, value in kwargs.items()
            if key in _RSIM_ASSUMPTION_NAMES
        })
        options["fork_speed"] = source["fork_speed"]
        return rsim(firing_rate=source["firing_rate"], **options)[output_name]

    mapper.__name__ = f"map_firing_and_speed_to_{output_name}_simulation"
    return mapper


def map_firing_to_fork_density_xt_simulation(firing_rate, **kwargs):
    options = {"time_stats_xtQ": True}
    options.update({
        key: value
        for key, value in kwargs.items()
        if key in _RSIM_OPTION_NAMES
    })
    result = rsim(firing_rate=firing_rate, **options)
    return result["right_fork_density_xt"] + result["left_fork_density_xt"]


def map_firing_and_speed_to_fork_density_xt_simulation(source, **kwargs):
    options = {"time_stats_xtQ": True}
    options.update({
        key: value
        for key, value in kwargs.items()
        if key in _RSIM_ASSUMPTION_NAMES
    })
    options["fork_speed"] = source["fork_speed"]
    result = rsim(firing_rate=source["firing_rate"], **options)
    return result["right_fork_density_xt"] + result["left_fork_density_xt"]


_FORWARD_TIMING_OPTIONS = {
    "resolution",
    "neighbourhood_range",
    "perQ",
    "n_bins",
    "genome_length",
    "interval_kb",
}

_TIMING_TO_CONSTANT_FIRING_OPTIONS = {
    "resolution",
    "resolution_time",
    "neighbourhood_range",
    "fit_step",
    "maxiter",
    "sim_number",
    "err_threshold",
    "rate_bounds",
    "refine_seed",
    "verbose",
    "print_every",
    "perQ",
    "n_bins",
}

_TIMING_TO_SPATIAL_FIRING_OPTIONS = {
    "resolution",
    "neighbourhood_range",
    "fit_step",
    "maxiter",
    "sim_number",
    "err_threshold",
    "deviation_threshold",
    "space_fit_step",
    "space_maxiter",
    "verbose",
    "print_every",
    "perQ",
    "stall_rate",
    "tau",
}

_TIMING_TO_GENERAL_FIRING_OPTIONS = _TIMING_TO_SPATIAL_FIRING_OPTIONS | {
    "resolution_time",
    "time_grid",
    "T0",
    "temporal_smoothw",
    "space_fit_step",
    "space_maxiter",
    "temporal_fit_step",
    "temporal_maxiter",
    "temporal_sim_number",
    "spacetime_fit_step",
    "spacetime_sim_number",
    "spacetime_update",
    "spacetime_order",
    "spacetime_kernel_t",
    "refine_ratio_bounds",
    "refine_edge_kb",
    "refine_seed",
    "refine_verbose",
    "floor",
}

_TIMING_TO_CONSTANT_SPEED_OPTIONS = {
    "resolution",
    "resolution_time",
    "neighbourhood_range",
    "fit_step",
    "maxiter",
    "sim_number",
    "speed_bounds",
    "initial_fork_speed",
    "refine_seed",
    "verbose",
    "print_every",
    "perQ",
    "n_bins",
}

_TIMING_TO_SPATIAL_SPEED_OPTIONS = {
    "resolution",
    "resolution_time",
    "neighbourhood_range",
    "fit_step",
    "maxiter",
    "space_fit_step",
    "space_maxiter",
    "sim_number",
    "speed_bounds",
    "initial_fork_speed",
    "smoothw",
    "refine_seed",
    "verbose",
    "print_every",
    "perQ",
    "n_bins",
}

_TIMING_TO_GENERAL_SPEED_OPTIONS = _TIMING_TO_SPATIAL_SPEED_OPTIONS | {
    "resolution_time",
    "time_grid",
    "T0",
    "temporal_smoothw",
    "temporal_fit_step",
    "temporal_maxiter",
    "temporal_sim_number",
    "spacetime_fit_step",
    "spacetime_sim_number",
    "spacetime_update",
    "spacetime_order",
    "spacetime_kernel_t",
    "refine_ratio_bounds",
    "refine_edge_kb",
    "refine_seed",
    "refine_verbose",
}

_TIMING_TO_JOINT_KINETICS_OPTIONS = {
    "resolution",
    "resolution_time",
    "time_grid",
    "T0",
    "initial_firing_rate",
    "initial_fork_speed",
    "initial_kinetics",
    "target_form",
    "fit_step",
    "maxiter",
    "sim_number",
    "space_init_fit_step",
    "space_init_maxiter",
    "space_init_sim_number",
    "joint_maxiter",
    "joint_sim_number",
    "firing_space_fit_step",
    "firing_time_fit_step",
    "speed_space_fit_step",
    "speed_time_fit_step",
    "firing_bounds",
    "speed_bounds",
    "firing_floor",
    "refine_ratio_bounds",
    "refine_edge_kb",
    "refine_seed",
    "spatial_smoothw",
    "temporal_smoothw",
    "perQ",
    "verbose",
    "print_every",
}

_DIRECTIONALITY_OPTIONS = {
    "resolution",
    "T0",
    "smoothw",
    "perQ",
}


def _options(kwargs, names):
    return {key: value for key, value in kwargs.items() if key in names}


def _apply_global_fit_controls(options, target_form=None):
    """Map global fit controls to the active target phase.

    For t targets, ``maxiter`` tunes the temporal phase. For x,t targets,
    ``maxiter`` overrides the phase-specific counts by setting both
    ``maxiter_x`` and ``maxiter_t`` to the same value.
    """
    out = dict(options)
    form = _canonical_form(target_form if target_form is not None else out.get("target_form"))
    spacetime_update = _canonical_spacetime_update(out.get("spacetime_update", "factorized"))
    if form == "time":
        if "maxiter" in out:
            out["temporal_maxiter"] = out.pop("maxiter")
        if "fit_step" in out:
            out.setdefault("temporal_fit_step", out.pop("fit_step"))
        if "sim_number" in out:
            out.setdefault("temporal_sim_number", out.pop("sim_number"))
    elif form == "spacetime":
        if "maxiter" in out:
            maxiter = out.pop("maxiter")
            out["space_maxiter"] = maxiter
            out["temporal_maxiter"] = maxiter
        if "fit_step" in out:
            fit_step = out.pop("fit_step")
            out.setdefault("space_fit_step", fit_step)
            out.setdefault("temporal_fit_step", fit_step)
            if spacetime_update == "mixed":
                out.setdefault("spacetime_fit_step", fit_step)
        if "sim_number" in out:
            sim_number = out.pop("sim_number")
            out.setdefault("temporal_sim_number", sim_number)
            if spacetime_update == "mixed":
                out.setdefault("spacetime_sim_number", sim_number)
    return out


def map_firing_and_speed_to_timing(source, **kwargs):
    return map_firing_timing(
        source["firing_rate"],
        fork_speed=source["fork_speed"],
        **_options(kwargs, _FORWARD_TIMING_OPTIONS),
    )


def map_timing_and_speed_to_constant_firing(source, **kwargs):
    return map_timing_constant_firing(
        source["replication_timing"],
        fork_speed=source["fork_speed"],
        **_options(kwargs, _TIMING_TO_CONSTANT_FIRING_OPTIONS),
    )


def map_timing_and_speed_to_firing(source, **kwargs):
    return map_timing_firing(
        source["replication_timing"],
        fork_speed=source["fork_speed"],
        **_options(kwargs, _TIMING_TO_SPATIAL_FIRING_OPTIONS),
    )


def map_timing_and_speed_to_general_firing(source, **kwargs):
    target_form = _canonical_form(kwargs.get("target_form", "space"))
    options = _options(kwargs, _TIMING_TO_GENERAL_FIRING_OPTIONS)
    options = _apply_global_fit_controls(options, target_form=target_form)
    if target_form == "time":
        return map_timing_firing_time(
            source["replication_timing"],
            fork_speed=source["fork_speed"],
            **options,
        )
    if target_form == "spacetime":
        return map_timing_firing_spacetime(
            source["replication_timing"],
            fork_speed=source["fork_speed"],
            **options,
        )
    raise ValueError("target_form must be 'time' or 'spacetime' for this route.")


def map_timing_and_speed_to_directionality(source, **kwargs):
    return map_timing_directionality(
        source["replication_timing"],
        fork_speed=source["fork_speed"],
        **_options(kwargs, _DIRECTIONALITY_OPTIONS),
    )


def map_directionality_and_speed_to_timing(source, **kwargs):
    return map_directionality_timing(
        source["fork_directionality"],
        fork_speed=source["fork_speed"],
        **_options(kwargs, _DIRECTIONALITY_OPTIONS),
    )


def map_timing_and_firing_to_spatial_speed(source, **kwargs):
    return map_timing_firing_spatial_speed(
        source,
        **_options(kwargs, _TIMING_TO_SPATIAL_SPEED_OPTIONS),
    )


def map_timing_and_firing_to_general_speed(source, **kwargs):
    target_form = _canonical_form(kwargs.get("target_form", "space"))
    options = _options(kwargs, _TIMING_TO_GENERAL_SPEED_OPTIONS)
    options = _apply_global_fit_controls(options, target_form=target_form)
    if target_form == "time":
        return map_timing_firing_temporal_speed(source, **options)
    if target_form == "spacetime":
        return map_timing_firing_spacetime_speed(source, **options)
    raise ValueError("target_form must be 'time' or 'spacetime' for this route.")


def map_timing_to_kinetic_fields(timing, **kwargs):
    options = _options(kwargs, _TIMING_TO_JOINT_KINETICS_OPTIONS)
    if "maxiter" in options:
        options.setdefault("joint_maxiter", options.pop("maxiter"))
    if "sim_number" in options:
        options.setdefault("joint_sim_number", options.pop("sim_number"))
    if "fit_step" in options:
        fit_step = options.pop("fit_step")
        options.setdefault("firing_space_fit_step", fit_step)
        options.setdefault("firing_time_fit_step", fit_step)
        options.setdefault("speed_space_fit_step", fit_step)
        options.setdefault("speed_time_fit_step", fit_step)
    return map_timing_joint_spacetime_kinetics(
        timing,
        **options,
    )


class DataTypeMapper:
    """Route replication feature mappings and fits.

    The mapper is intentionally broader than a direct function lookup. It
    records which feature-to-feature routes are implemented now and which
    routes still need a dedicated fitting method.
    """

    def __init__(self):
        self._specs = []

    @staticmethod
    def canonical(name):
        return _canonical_feature(name)

    def __repr__(self):
        n_routes = len(self._specs)
        n_implemented = sum(spec.implemented for spec in self._specs)
        targets = ", ".join(sorted({spec.target for spec in self._specs}))
        return (
            "DNAscape rmap route registry\n"
            f"  routes: {n_implemented} implemented / {n_routes} registered\n"
            f"  targets: {targets}\n"
            "  use rmap.available() for a user-facing table and "
            "rmap.development() for the full internal registry"
        )

    def register(
        self,
        from_type,
        to_type,
        func,
        method=None,
        required_assumptions=(),
        optional_assumptions=(),
        default_assumptions=None,
        source_kinds=(),
        assumption_kinds=(),
        assumption_values=(),
        implemented=True,
        description="",
    ):
        source = tuple(_canonical_feature(x) for x in (
            from_type if isinstance(from_type, (tuple, list)) else (from_type,)
        ))
        target = _canonical_feature(to_type)
        if method is None:
            method = f"{'_'.join(source)}_to_{target}"
        spec = MappingSpec(
            source=source,
            target=target,
            method=method,
            func=func,
            implemented=implemented,
            required_assumptions=tuple(
                _canonical_assumption(name)
                for name in required_assumptions
            ),
            optional_assumptions=tuple(
                _canonical_assumption(name)
                for name in optional_assumptions
            ),
            default_assumptions=tuple(
                (_canonical_assumption(name), value)
                for name, value in (default_assumptions or {}).items()
            ),
            source_kinds=tuple(
                (_canonical_feature(name), (kinds,) if isinstance(kinds, str) else tuple(kinds))
                for name, kinds in source_kinds
            ),
            assumption_kinds=tuple(
                (_canonical_assumption(name), (kinds,) if isinstance(kinds, str) else tuple(kinds))
                for name, kinds in assumption_kinds
            ),
            assumption_values=tuple(
                (_canonical_assumption(name), (values,) if isinstance(values, str) else tuple(values))
                for name, values in assumption_values
            ),
            description=description,
        )
        self._specs.append(spec)
        return spec

    def register_placeholder(
        self,
        from_type,
        to_type,
        method,
        required_assumptions=(),
        optional_assumptions=(),
        default_assumptions=None,
        source_kinds=(),
        assumption_kinds=(),
        assumption_values=(),
        description="",
    ):
        return self.register(
            from_type=from_type,
            to_type=to_type,
            func=None,
            method=method,
            required_assumptions=required_assumptions,
            optional_assumptions=optional_assumptions,
            default_assumptions=default_assumptions,
            source_kinds=source_kinds,
            assumption_kinds=assumption_kinds,
            assumption_values=assumption_values,
            implemented=False,
            description=description,
        )

    def _normalise_assumptions(self, assumptions, extra):
        out = {}
        if assumptions:
            out.update({_canonical_assumption(k): v for k, v in assumptions.items()})
        out.update({_canonical_assumption(k): v for k, v in extra.items()})
        perq = _geometry_to_perq(out.get("geometry"))
        if perq is not None and "perQ" not in out:
            out["perQ"] = perq
        return out

    def _normalise_source(self, source, data):
        if isinstance(source, dict):
            if data is not None:
                raise ValueError("Pass either source={...} or source='feature' with data=..., not both.")
            return {_canonical_feature(k): v for k, v in source.items()}

        if source is None:
            raise ValueError("rmap requires a source feature or source={feature: data}.")
        if data is None:
            raise ValueError("rmap requires data when source is passed as a feature name.")
        return {_canonical_feature(source): data}

    def _candidate_specs(self, source_names, target, method):
        source_set = set(source_names)
        out = []
        for spec in self._specs:
            if set(spec.source) != source_set or spec.target != target:
                continue
            if method != "auto" and spec.method != method:
                continue
            out.append(spec)
        return out

    def _select_spec(self, source_data, target, assumptions, method):
        candidates = self._candidate_specs(source_data.keys(), target, method)
        if not candidates:
            src = ", ".join(source_data.keys())
            raise KeyError(f"No mapping route registered from [{src}] to '{target}'.")

        kind_matches = [spec for spec in candidates if spec.accepts(source_data, assumptions)]
        if kind_matches:
            spec = kind_matches[0]
            return spec, spec.assumptions_with_defaults(assumptions)

        details = []
        for spec in candidates:
            missing = spec.missing_required(assumptions)
            req = ", ".join(
                f"{name}: {'/'.join(kinds)}"
                for name, kinds in spec.assumption_kinds
            )
            value_req = ", ".join(
                f"{name}: {'/'.join(str(_canonical_form(value)) for value in values if value is not None)}"
                for name, values in spec.assumption_values
            )
            src_req = ", ".join(
                f"{name}: {'/'.join(kinds)}"
                for name, kinds in spec.source_kinds
            )
            missing_msg = f"; missing {', '.join(missing)}" if missing else ""
            expected = []
            if src_req:
                expected.append(f"source {src_req}")
            if req:
                expected.append(f"assumptions {req}")
            if value_req:
                expected.append(f"assumption values {value_req}")
            details.append(
                f"{spec.method} expects {', '.join(expected) or 'no specific field kinds'}{missing_msg}"
            )
        raise ValueError(
            "No registered mapping route matches the supplied source or assumption field kinds. "
            + "; ".join(details)
        )

    def _function_kwargs(self, spec, assumptions):
        if spec.func is None:
            return assumptions

        sig = inspect.signature(spec.func)
        params = sig.parameters
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
            return assumptions
        return {
            key: value
            for key, value in assumptions.items()
            if key in params
        }

    def _not_implemented_message(self, spec):
        src = ", ".join(spec.source)
        msg = (
            f"Mapping route '{spec.method}' from [{src}] to '{spec.target}' is registered "
            "but not implemented yet."
        )
        if spec.description:
            msg += f" {spec.description}"
        return msg

    def __call__(
        self,
        source=None,
        target=None,
        data=None,
        assumptions=None,
        method="auto",
        return_info=False,
        **kwargs,
    ):
        if target is None:
            raise ValueError("rmap requires a target feature.")

        target = _canonical_feature(target)
        source_data = self._normalise_source(source, data)
        assumptions = self._normalise_assumptions(assumptions, kwargs)
        spec, assumptions = self._select_spec(source_data, target, assumptions, method)

        if not spec.implemented:
            raise NotImplementedError(self._not_implemented_message(spec))

        payload = (
            source_data[spec.source[0]]
            if len(spec.source) == 1
            else dict(source_data)
        )
        values = spec.func(payload, **self._function_kwargs(spec, assumptions))

        if return_info:
            return {
                "target": spec.target,
                "values": values,
                "method": spec.method,
                "source": dict(source_data),
                "assumptions": dict(assumptions),
                "implemented": spec.implemented,
                "required_assumptions": spec.required_assumptions,
                "optional_assumptions": spec.optional_assumptions,
                "source_kinds": spec.source_kinds,
                "assumption_values": spec.assumption_values,
                "description": spec.description,
            }
        return values

    def get(self, from_type, to_type, method="auto", assumptions=None, **kwargs):
        source_data = {_canonical_feature(from_type): None}
        assumptions = self._normalise_assumptions(assumptions, kwargs)
        spec, _ = self._select_spec(source_data, _canonical_feature(to_type), assumptions, method)
        if not spec.implemented:
            raise NotImplementedError(self._not_implemented_message(spec))
        return spec.func

    def map(self, from_type, to_type, data, method="auto", assumptions=None, **kwargs):
        """Backward-compatible feature mapping API."""
        return self(
            source=from_type,
            target=to_type,
            data=data,
            assumptions=assumptions,
            method=method,
            **kwargs,
        )

    @staticmethod
    def _join(values):
        values = tuple(value for value in values if value)
        return ", ".join(values) if values else "-"

    @staticmethod
    def _display_assumption_name(name):
        display_names = {
            "resolution": "resolution_x",
            "resolution_time": "resolution_t",
            "space_fit_step": "fit_step_x",
            "space_maxiter": "maxiter_x",
            "temporal_fit_step": "fit_step_t",
            "temporal_maxiter": "maxiter_t",
            "temporal_sim_number": "sim_number_t",
            "temporal_smoothw": "smoothw_t",
            "spacetime_fit_step": "fit_step_xt",
            "spacetime_space_fit_step": "fit_step_xt_x",
            "spacetime_time_fit_step": "fit_step_xt_t",
            "spacetime_sim_number": "sim_number_xt",
            "spacetime_kernel_t": "kernel_xt_t",
            "space_init_fit_step": "fit_step_init_x",
            "space_init_maxiter": "maxiter_init_x",
            "space_init_sim_number": "sim_number_init_x",
            "joint_maxiter": "maxiter_joint",
            "joint_sim_number": "sim_number_joint",
            "firing_space_fit_step": "fit_step_firing_x",
            "firing_time_fit_step": "fit_step_firing_t",
            "speed_space_fit_step": "fit_step_speed_x",
            "speed_time_fit_step": "fit_step_speed_t",
            "spatial_smoothw": "smoothw_x",
        }
        return display_names.get(name, name)

    @classmethod
    def _join_assumption_names(cls, values):
        values = tuple(cls._display_assumption_name(value) for value in values if value)
        return cls._join(values)

    @staticmethod
    def _unique(values):
        out = []
        for value in values:
            if value and value not in out:
                out.append(value)
        return tuple(out)

    @staticmethod
    def _join_pairs(pairs):
        pairs = tuple(pairs)
        if not pairs:
            return "-"
        return "; ".join(
            f"{name}: {'/'.join(str(value) for value in values)}"
            for name, values in pairs
        )

    @staticmethod
    def _join_assumption_values(pairs):
        pairs = tuple(pairs)
        if not pairs:
            return "-"
        formatted = []
        for name, values in pairs:
            forms = []
            for value in values:
                form = _canonical_form(value)
                if form is not None and form not in forms:
                    forms.append(form)
            formatted.append(f"{name}: {'/'.join(forms)}")
        return "; ".join(formatted)

    @classmethod
    def _order_assumption_names(cls, names):
        names = cls._unique(names)
        order = {name: idx for idx, name in enumerate(_ASSUMPTION_ORDER)}
        return tuple(
            sorted(
                names,
                key=lambda name: (
                    order.get(name, len(order)),
                    cls._display_assumption_name(name),
                ),
            )
        )

    @classmethod
    def _assumption_name_tuple(cls, specs):
        omitted = {"verbose", "print_every", "n_bins"}
        names = []
        for spec in specs:
            for name in spec.required_assumptions + spec.optional_assumptions:
                if name not in omitted and name not in _CANONICAL_FEATURES:
                    names.append(name)
            for pairs in (spec.assumption_kinds, spec.assumption_values):
                for name, _ in pairs:
                    if name not in omitted and name not in _CANONICAL_FEATURES:
                        names.append(name)
        return cls._order_assumption_names(names)

    @classmethod
    def _assumption_names(cls, specs):
        return cls._join_assumption_names(cls._assumption_name_tuple(specs))

    @classmethod
    def _required_data(cls, spec):
        names = list(spec.source)
        for name in spec.required_assumptions:
            if name in _CANONICAL_FEATURES:
                names.append(name)
        for name, _ in spec.assumption_kinds:
            if name in _CANONICAL_FEATURES:
                names.append(name)
        return cls._unique(names)

    @staticmethod
    def _target_description(target):
        return _FEATURE_DESCRIPTIONS.get(target, "Registered DNAscape replication feature.")

    @classmethod
    def _form_label(cls, form):
        return _FORM_DESCRIPTIONS.get(form, str(form))

    @classmethod
    def _source_data_form_records(cls, specs):
        names = cls._unique(name for spec in specs for name in cls._required_data(spec))
        forms_by_name = {name: [] for name in names}
        for spec in specs:
            for pairs in (spec.source_kinds, spec.assumption_kinds):
                for name, kinds in pairs:
                    if name not in forms_by_name:
                        continue
                    for kind in kinds:
                        if kind not in forms_by_name[name]:
                            forms_by_name[name].append(kind)
        records = []
        for name in names:
            forms = forms_by_name[name] or ["supplied"]
            for form in forms:
                records.append(
                    {
                        "role": "source",
                        "quantity": name,
                        "form": cls._form_label(form),
                        "target_form": "-",
                    }
                )
        return records

    @classmethod
    def _target_form_records(cls, specs):
        forms = []
        for spec in specs:
            for name, values in spec.assumption_values:
                if name != "target_form":
                    continue
                for value in values:
                    form = _canonical_form(value)
                    if form is not None and form not in forms:
                        forms.append(form)
        if not forms:
            forms = ["space"]
        return [
            {
                "role": "target",
                "quantity": specs[0].target,
                "form": cls._form_label(form),
                "target_form": form,
            }
            for form in forms
        ]

    @classmethod
    def _data_form_records(cls, grouped_specs):
        records = []
        seen = set()
        for specs in grouped_specs:
            for record in cls._source_data_form_records(specs) + cls._target_form_records(specs):
                key = (record["role"], record["quantity"], record["form"], record["target_form"])
                if key in seen:
                    continue
                seen.add(key)
                records.append(record)
        return records

    @staticmethod
    def _format_default_value(value):
        if isinstance(value, str):
            return value
        if value is None:
            return "None"
        return repr(value)

    @classmethod
    def _assumption_default_value(cls, name, specs):
        values = []
        missing = False
        for spec in specs:
            defaults = spec.defaults_dict()
            if name in defaults:
                values.append(cls._format_default_value(defaults[name]))
                continue
            if spec.func is not None:
                try:
                    params = inspect.signature(spec.func).parameters
                except (TypeError, ValueError):
                    params = {}
                param = params.get(name)
                if param is not None and param.default is not inspect._empty:
                    values.append(cls._format_default_value(param.default))
                    continue
            missing = True
        values = cls._unique(values)
        fallback = _ASSUMPTION_FALLBACK_DEFAULTS.get(name)
        if fallback is not None and not values:
            return cls._format_default_value(fallback)
        if fallback is not None and "None" in values:
            fallback_value = cls._format_default_value(fallback)
            values = tuple(fallback_value if value == "None" else value for value in values)
            values = cls._unique(values)
        if not values:
            return "None"
        if name == "target_form" and "space" in values:
            return "space"
        if len(values) == 1:
            return values[0]
        prefix = "route-dependent: " if missing else ""
        return prefix + cls._join(values)

    @classmethod
    def _assumption_detail_records(cls, grouped_specs):
        all_specs = tuple(spec for specs in grouped_specs for spec in specs)
        names = cls._order_assumption_names(
            name
            for specs in grouped_specs
            for name in cls._assumption_name_tuple(specs)
        )
        records = []
        for name in names:
            related_specs = tuple(
                spec
                for spec in all_specs
                if name in cls._assumption_name_tuple((spec,))
            )
            records.append(
                {
                    "assumption": cls._display_assumption_name(name),
                    "default": cls._assumption_default_value(name, related_specs),
                    "description": _ASSUMPTION_DESCRIPTIONS.get(
                        name,
                        "Route-specific option passed to the selected mapping method.",
                    ),
                }
            )
        return records

    @staticmethod
    def _table(records, columns, transpose=False):
        try:
            import pandas as pd
        except ImportError:
            return records
        pd.set_option("display.max_colwidth", None)
        table = pd.DataFrame.from_records(records, columns=columns)
        if transpose:
            table = table.T
            table.columns = [""] * table.shape[1]
        return table

    def _filtered_specs(self, target=None, source=None, implemented=None):
        if target is None:
            target_set = None
        else:
            target_names = target if isinstance(target, (tuple, list, set)) else (target,)
            target_set = {_canonical_feature(name) for name in target_names}

        if source is None:
            source_set = None
        else:
            source_names = source if isinstance(source, (tuple, list, set)) else (source,)
            source_set = {_canonical_feature(name) for name in source_names}

        specs = []
        for spec in self._specs:
            if target_set is not None and spec.target not in target_set:
                continue
            if source_set is not None and set(spec.source) != source_set:
                continue
            if implemented is not None and spec.implemented != bool(implemented):
                continue
            specs.append(spec)
        return specs

    def _grouped_specs(self, target=None, source=None, implemented=None):
        groups = {}
        for spec in self._filtered_specs(target=target, source=source, implemented=implemented):
            key = (spec.target, self._required_data(spec))
            groups.setdefault(key, []).append(spec)
        return tuple(
            groups[key]
            for key in sorted(groups, key=lambda item: (item[0], item[1]))
        )

    def available(self, target=None, source=None, implemented=None):
        """Return a compact catalogue of available mapping targets.

        By default the table shows implemented core mapping targets. Use
        ``requirements`` for the source data and assumptions needed for a
        specific target. Parameters can narrow or expand the catalogue, e.g.
        ``rmap.available(target="firing_rate")`` or
        ``rmap.available(target=("replicated_fraction", "fork_density"))``.
        """
        if implemented is None:
            implemented = True
        target_names = _PUBLIC_MAPPING_TARGETS if target is None else target
        target_names = target_names if isinstance(target_names, (tuple, list, set)) else (target_names,)
        target_order = tuple(_canonical_feature(name) for name in target_names)
        columns = ("target", "description")
        filtered = self._filtered_specs(target=target_order, source=source, implemented=implemented)
        available_targets = {spec.target for spec in filtered}
        records = []
        for target_name in target_order:
            if target_name not in available_targets:
                continue
            records.append(
                {
                    "target": target_name,
                    "description": self._target_description(target_name),
                }
            )
        return self._table(records, columns)

    def requirements(
        self,
        target=None,
        source=None,
        implemented=None,
        show_data=False,
        explain_assumptions=False,
        detail=False,
    ):
        """Return the user-facing data requirements for mapping routes.

        By default only implemented routes are shown. ``show_data=True``
        returns only the source/target data forms.
        ``explain_assumptions=True`` returns only the assumptions and their
        shared definitions. ``detail=True`` returns both tables in a
        dictionary.
        """
        if implemented is None:
            implemented = True
        groups = self._grouped_specs(target=target, source=source, implemented=implemented)
        if detail:
            return {
                "data": self._table(
                    self._data_form_records(groups),
                    ("role", "quantity", "form", "target_form"),
                ),
                "assumptions": self._table(
                    self._assumption_detail_records(groups),
                    ("assumption", "default", "description"),
                ),
            }
        if show_data:
            return self._table(
                self._data_form_records(groups),
                ("role", "quantity", "form", "target_form"),
            )
        if explain_assumptions:
            return self._table(
                self._assumption_detail_records(groups),
                ("assumption", "default", "description"),
            )
        columns = ["target", "source (required data)", "assumptions", "target description"]
        records = [
            {
                "target": specs[0].target,
                "source (required data)": self._join(self._required_data(specs[0])),
                "assumptions": self._assumption_names(specs),
                "target description": self._target_description(specs[0].target),
            }
            for specs in groups
        ]
        return self._table(records, columns, transpose=True)

    def development(self, target=None, source=None, implemented=None):
        """Return the full per-route registry for internal development."""
        columns = (
            "target",
            "required data",
            "data forms",
            "required assumptions",
            "optional assumptions",
            "assumption forms",
            "assumption values",
            "method",
            "status",
            "description",
        )
        records = [
            {
                "target": spec.target,
                "required data": self._join(spec.source),
                "data forms": self._join_pairs(spec.source_kinds),
                "required assumptions": self._join_assumption_names(spec.required_assumptions),
                "optional assumptions": self._join_assumption_names(spec.optional_assumptions),
                "assumption forms": self._join_pairs(spec.assumption_kinds),
                "assumption values": self._join_assumption_values(spec.assumption_values),
                "method": spec.method,
                "status": spec.implemented_status(),
                "description": spec.description,
            }
            for spec in self._filtered_specs(target=target, source=source, implemented=implemented)
        ]
        return self._table(records, columns)

    def describe(self):
        print("Available replication feature mappings:")
        for spec in self._specs:
            status = spec.implemented_status()
            src = ", ".join(spec.source)
            required = ", ".join(spec.required_assumptions)
            src_req = ", ".join(
                f"{name}: {'/'.join(kinds)}"
                for name, kinds in spec.source_kinds
            )
            req = ", ".join(
                f"{name}: {'/'.join(kinds)}"
                for name, kinds in spec.assumption_kinds
            )
            value_req = ", ".join(
                f"{name}: {'/'.join(str(_canonical_form(value)) for value in values if value is not None)}"
                for name, values in spec.assumption_values
            )
            details = []
            if required:
                details.append(f"requires {required}")
            if src_req:
                details.append(f"source {src_req}")
            if req:
                details.append(f"supports {req}")
            if value_req:
                details.append(f"values {value_req}")
            suffix = f" [{'; '.join(details)}]" if details else ""
            print(f"  {src} -> {spec.target} via {spec.method} ({status}){suffix}")

    def features(self):
        return tuple(sorted(_CANONICAL_FEATURES))

    def routes(self):
        return tuple(self._specs)


rmap = DataTypeMapper()
rmap.register(
    ("firing_rate", "fork_speed"),
    "replication_timing",
    map_firing_and_speed_to_timing,
    method="firing_and_speed_to_timing_deterministic",
    optional_assumptions=(
        "resolution",
        "neighbourhood_range",
        "perQ",
        "n_bins",
        "genome_length",
        "interval_kb",
    ),
    source_kinds=(
        ("firing_rate", ("scalar", "x")),
        ("fork_speed", ("scalar", "x")),
    ),
    description="Deterministic mapping for I0 or I(x) with v0 or spatial effective v(x).",
)
rmap.register_placeholder(
    ("firing_rate", "fork_speed"),
    "replication_timing",
    method="firing_and_variable_speed_to_timing",
    optional_assumptions=(
        "resolution",
        "neighbourhood_range",
        "perQ",
        "n_bins",
        "genome_length",
        "interval_kb",
    ),
    source_kinds=(
        ("firing_rate", ("scalar", "x", "t", "xt")),
        ("fork_speed", ("t", "xt")),
    ),
    description="Forward timing maps with time-only or space-time fork speed need dedicated methods.",
)
rmap.register(
    ("replication_timing", "fork_speed"),
    "firing_rate",
    map_timing_and_speed_to_constant_firing,
    method="timing_and_speed_to_constant_firing_rate",
    required_assumptions=("target_form",),
    optional_assumptions=(
        "resolution",
        "neighbourhood_range",
        "fit_step",
        "maxiter",
        "err_threshold",
        "rate_bounds",
        "verbose",
        "print_every",
        "perQ",
        "n_bins",
    ),
    default_assumptions={"verbose": True},
    source_kinds=(
        ("replication_timing", ("x",)),
        ("fork_speed", ("scalar",)),
    ),
    assumption_values=(("target_form", ("constant",)),),
    description="Scalar fit of I0 from timing and constant fork speed.",
)
rmap.register(
    ("replication_timing", "fork_speed"),
    "firing_rate",
    map_timing_and_speed_to_firing,
    method="timing_and_speed_to_firing_rate",
    optional_assumptions=(
        "resolution",
        "neighbourhood_range",
        "fit_step",
        "maxiter",
        "sim_number",
        "err_threshold",
        "deviation_threshold",
        "space_fit_step",
        "space_maxiter",
        "verbose",
        "print_every",
        "perQ",
        "stall_rate",
        "tau",
        "target_form",
    ),
    default_assumptions={"verbose": True, "target_form": "space"},
    source_kinds=(
        ("replication_timing", ("x",)),
        ("fork_speed", ("scalar",)),
    ),
    assumption_values=(("target_form", ("space",)),),
    description="Iterative fit of I(x) from timing and constant fork speed.",
)
rmap.register(
    ("replication_timing", "fork_speed"),
    "firing_rate",
    map_timing_and_speed_to_constant_firing,
    method="timing_and_variable_speed_to_firing_rate",
    required_assumptions=("target_form",),
    optional_assumptions=(
        "resolution",
        "resolution_time",
        "neighbourhood_range",
        "fit_step",
        "maxiter",
        "sim_number",
        "err_threshold",
        "rate_bounds",
        "refine_seed",
        "verbose",
        "print_every",
        "perQ",
        "n_bins",
    ),
    default_assumptions={"verbose": True},
    source_kinds=(
        ("replication_timing", ("x",)),
        ("fork_speed", ("x", "t", "xt")),
    ),
    assumption_values=(("target_form", ("constant",)),),
    description="Scalar effective fit of I0 from timing with variable fork speed.",
)
rmap.register(
    ("replication_timing", "fork_speed"),
    "firing_rate",
    map_timing_and_speed_to_firing,
    method="timing_and_variable_speed_to_spatial_firing_rate",
    optional_assumptions=(
        "resolution",
        "neighbourhood_range",
        "fit_step",
        "maxiter",
        "sim_number",
        "err_threshold",
        "deviation_threshold",
        "space_fit_step",
        "space_maxiter",
        "verbose",
        "print_every",
        "perQ",
        "stall_rate",
        "tau",
        "target_form",
    ),
    default_assumptions={"verbose": True, "target_form": "space"},
    source_kinds=(
        ("replication_timing", ("x",)),
        ("fork_speed", ("x", "t", "xt")),
    ),
    assumption_values=(("target_form", ("space",)),),
    description="Space-dependent effective fit of I(x) from timing with variable fork speed.",
)
rmap.register(
    ("replication_timing", "firing_rate"),
    "fork_speed",
    map_timing_firing_constant_speed,
    method="timing_and_firing_to_constant_fork_speed",
    optional_assumptions=(
        "resolution",
        "neighbourhood_range",
        "fit_step",
        "maxiter",
        "speed_bounds",
        "initial_fork_speed",
        "verbose",
        "print_every",
        "perQ",
        "n_bins",
        "target_form",
    ),
    default_assumptions={"verbose": True},
    source_kinds=(
        ("replication_timing", ("x",)),
        ("firing_rate", ("scalar", "x")),
    ),
    assumption_values=(("target_form", ("constant",)),),
    description="Scalar fit of v0 from timing and firing rate.",
)
rmap.register(
    ("replication_timing", "firing_rate"),
    "fork_speed",
    map_timing_firing_constant_speed_simulated,
    method="timing_and_time_firing_to_constant_fork_speed",
    required_assumptions=("target_form",),
    optional_assumptions=(
        "resolution",
        "resolution_time",
        "fit_step",
        "maxiter",
        "sim_number",
        "speed_bounds",
        "initial_fork_speed",
        "refine_seed",
        "verbose",
        "print_every",
        "perQ",
        "n_bins",
        "target_form",
    ),
    default_assumptions={"verbose": True},
    source_kinds=(
        ("replication_timing", ("x",)),
        ("firing_rate", ("t", "xt")),
    ),
    assumption_values=(("target_form", ("constant",)),),
    description="Scalar effective fit of v0 from timing and time-dependent firing rate.",
)
rmap.register(
    ("replication_timing", "firing_rate"),
    "fork_speed",
    map_timing_and_firing_to_spatial_speed,
    method="timing_and_firing_to_spatial_fork_speed",
    optional_assumptions=(
        "resolution",
        "neighbourhood_range",
        "fit_step",
        "maxiter",
        "space_fit_step",
        "space_maxiter",
        "speed_bounds",
        "initial_fork_speed",
        "smoothw",
        "verbose",
        "print_every",
        "perQ",
        "n_bins",
        "target_form",
    ),
    default_assumptions={"verbose": True, "target_form": "space"},
    source_kinds=(
        ("replication_timing", ("x",)),
        ("firing_rate", ("scalar", "x")),
    ),
    assumption_values=(("target_form", ("space",)),),
    description="Spatial effective fit of v(x) from timing and firing rate.",
)
rmap.register(
    ("replication_timing", "firing_rate"),
    "fork_speed",
    map_timing_firing_spatial_speed_simulated,
    method="timing_and_time_firing_to_spatial_fork_speed",
    optional_assumptions=(
        "resolution",
        "resolution_time",
        "fit_step",
        "maxiter",
        "space_fit_step",
        "space_maxiter",
        "sim_number",
        "speed_bounds",
        "initial_fork_speed",
        "smoothw",
        "refine_seed",
        "verbose",
        "print_every",
        "perQ",
        "n_bins",
        "target_form",
    ),
    default_assumptions={"verbose": True, "target_form": "space"},
    source_kinds=(
        ("replication_timing", ("x",)),
        ("firing_rate", ("t", "xt")),
    ),
    assumption_values=(("target_form", ("space",)),),
    description="Spatial effective fit of v(x) from timing and time-dependent firing rate.",
)
rmap.register(
    ("replication_timing", "firing_rate"),
    "fork_speed",
    map_timing_and_firing_to_general_speed,
    method="timing_and_firing_to_time_fork_speed",
    optional_assumptions=(
        "resolution",
        "resolution_time",
        "neighbourhood_range",
        "fit_step",
        "maxiter",
        "space_fit_step",
        "space_maxiter",
        "speed_bounds",
        "initial_fork_speed",
        "smoothw",
        "temporal_smoothw",
        "temporal_fit_step",
        "temporal_maxiter",
        "temporal_sim_number",
        "refine_ratio_bounds",
        "refine_edge_kb",
        "refine_seed",
        "refine_verbose",
        "verbose",
        "print_every",
        "perQ",
        "n_bins",
        "target_form",
        "time_grid",
        "T0",
    ),
    default_assumptions={"verbose": True, "target_form": "time"},
    source_kinds=(
        ("replication_timing", ("x",)),
        ("firing_rate", ("scalar", "x")),
    ),
    assumption_values=(("target_form", ("time",)),),
    description="Time-only effective fit of v(t) from timing and spatial firing rate.",
)
rmap.register(
    ("replication_timing", "firing_rate"),
    "fork_speed",
    map_timing_firing_temporal_speed_simulated,
    method="timing_and_time_firing_to_time_fork_speed",
    optional_assumptions=(
        "resolution",
        "resolution_time",
        "time_grid",
        "T0",
        "temporal_smoothw",
        "fit_step",
        "maxiter",
        "temporal_fit_step",
        "temporal_maxiter",
        "temporal_sim_number",
        "speed_bounds",
        "initial_fork_speed",
        "refine_ratio_bounds",
        "refine_edge_kb",
        "refine_seed",
        "refine_verbose",
        "verbose",
        "print_every",
        "perQ",
        "n_bins",
        "target_form",
    ),
    default_assumptions={"verbose": True, "target_form": "time"},
    source_kinds=(
        ("replication_timing", ("x",)),
        ("firing_rate", ("t", "xt")),
    ),
    assumption_values=(("target_form", ("time",)),),
    description="Time-only effective fit of v(t) from timing and time-dependent firing rate.",
)
rmap.register(
    ("replication_timing", "firing_rate"),
    "fork_speed",
    map_timing_and_firing_to_general_speed,
    method="timing_and_firing_to_spacetime_fork_speed",
    optional_assumptions=(
        "resolution",
        "resolution_time",
        "neighbourhood_range",
        "fit_step",
        "maxiter",
        "space_fit_step",
        "space_maxiter",
        "speed_bounds",
        "initial_fork_speed",
        "smoothw",
        "temporal_smoothw",
        "temporal_fit_step",
        "temporal_maxiter",
        "temporal_sim_number",
        "spacetime_fit_step",
        "spacetime_sim_number",
        "spacetime_update",
        "spacetime_order",
        "spacetime_kernel_t",
        "refine_ratio_bounds",
        "refine_edge_kb",
        "refine_seed",
        "refine_verbose",
        "verbose",
        "print_every",
        "perQ",
        "n_bins",
        "target_form",
        "time_grid",
        "T0",
    ),
    default_assumptions={"verbose": True, "target_form": "spacetime"},
    source_kinds=(
        ("replication_timing", ("x",)),
        ("firing_rate", ("xt",)),
    ),
    assumption_values=(("target_form", ("spacetime",)),),
    description="Space-time effective fit of v(x,t) from timing and space-time firing rate.",
)
rmap.register(
    ("replication_timing", "firing_rate"),
    "fork_speed",
    map_timing_firing_spacetime_speed_simulated,
    method="timing_and_general_firing_to_spacetime_fork_speed",
    optional_assumptions=(
        "resolution",
        "resolution_time",
        "time_grid",
        "T0",
        "fit_step",
        "maxiter",
        "space_fit_step",
        "space_maxiter",
        "temporal_fit_step",
        "temporal_maxiter",
        "temporal_sim_number",
        "spacetime_fit_step",
        "spacetime_sim_number",
        "spacetime_update",
        "spacetime_order",
        "spacetime_kernel_t",
        "speed_bounds",
        "initial_fork_speed",
        "refine_ratio_bounds",
        "refine_edge_kb",
        "refine_seed",
        "refine_verbose",
        "verbose",
        "print_every",
        "perQ",
        "n_bins",
        "target_form",
    ),
    default_assumptions={"verbose": True, "target_form": "spacetime"},
    source_kinds=(
        ("replication_timing", ("x",)),
        ("firing_rate", ("scalar", "x", "t")),
    ),
    assumption_values=(("target_form", ("spacetime",)),),
    description="Space-time effective fit of v(x,t) from timing and general firing rate.",
)
rmap.register(
    ("replication_timing", "fork_speed"),
    "fork_directionality",
    map_timing_and_speed_to_directionality,
    method="timing_and_speed_to_directionality",
    optional_assumptions=("resolution", "T0", "smoothw", "perQ"),
    source_kinds=(
        ("replication_timing", ("x",)),
        ("fork_speed", ("scalar",)),
    ),
    description="Gradient-based fork-directionality mapping.",
)
rmap.register(
    ("fork_directionality", "fork_speed"),
    "replication_timing",
    map_directionality_and_speed_to_timing,
    method="directionality_and_speed_to_timing",
    optional_assumptions=("resolution", "T0", "smoothw", "perQ"),
    source_kinds=(
        ("fork_directionality", ("x",)),
        ("fork_speed", ("scalar",)),
    ),
    description="Integration of fork directionality to timing.",
)
rmap.register(
    "firing_rate",
    "replication_timing",
    map_firing_timing,
    method="firing_to_timing_deterministic",
    required_assumptions=("fork_speed",),
    optional_assumptions=(
        "resolution",
        "neighbourhood_range",
        "perQ",
        "n_bins",
        "genome_length",
        "interval_kb",
    ),
    default_assumptions={"fork_speed": 1.4},
    source_kinds=(("firing_rate", ("scalar", "x")),),
    assumption_kinds=(("fork_speed", ("scalar", "x")),),
    description="Deterministic mapping for I0 or I(x) with v0 or spatial effective v(x).",
)
rmap.register_placeholder(
    "firing_rate",
    "replication_timing",
    method="firing_to_timing_variable_fork_speed",
    required_assumptions=("fork_speed",),
    optional_assumptions=(
        "resolution",
        "neighbourhood_range",
        "perQ",
        "n_bins",
        "genome_length",
        "interval_kb",
    ),
    source_kinds=(("firing_rate", ("scalar", "x")),),
    assumption_kinds=(("fork_speed", ("t", "xt")),),
    description="Time-only or space-time fork-speed timing maps need dedicated methods.",
)
rmap.register_placeholder(
    "firing_rate",
    "replication_timing",
    method="firing_to_timing_general_firing_form",
    required_assumptions=("fork_speed",),
    optional_assumptions=("resolution", "neighbourhood_range", "perQ"),
    source_kinds=(("firing_rate", ("t", "xt")),),
    assumption_kinds=(("fork_speed", ("scalar", "x", "t", "xt")),),
    description="Forward timing maps for time-only or space-time firing need dedicated methods.",
)
rmap.register(
    "replication_timing",
    "firing_rate",
    map_timing_constant_firing,
    method="timing_to_constant_firing_rate_constant_fork_speed",
    required_assumptions=("fork_speed", "target_form"),
    optional_assumptions=(
        "resolution",
        "neighbourhood_range",
        "fit_step",
        "maxiter",
        "err_threshold",
        "rate_bounds",
        "verbose",
        "print_every",
        "perQ",
        "n_bins",
    ),
    default_assumptions={"fork_speed": 1.4, "verbose": True},
    source_kinds=(("replication_timing", ("x",)),),
    assumption_kinds=(("fork_speed", ("scalar",)),),
    assumption_values=(("target_form", ("constant",)),),
    description="Scalar fit of I0 by optimising the deterministic timing mismatch.",
)
rmap.register(
    "replication_timing",
    "firing_rate",
    map_timing_firing,
    method="timing_to_firing_constant_fork_speed",
    required_assumptions=("fork_speed",),
    optional_assumptions=(
        "resolution",
        "neighbourhood_range",
        "fit_step",
        "maxiter",
        "sim_number",
        "err_threshold",
        "deviation_threshold",
        "space_fit_step",
        "space_maxiter",
        "verbose",
        "print_every",
        "perQ",
        "stall_rate",
        "tau",
        "target_form",
    ),
    default_assumptions={"fork_speed": 1.4, "verbose": True, "target_form": "space"},
    source_kinds=(("replication_timing", ("x",)),),
    assumption_kinds=(("fork_speed", ("scalar",)),),
    assumption_values=(("target_form", ("space",)),),
    description="Iterative fit of a space-dependent firing rate with constant fork speed.",
)
rmap.register(
    "replication_timing",
    "firing_rate",
    map_timing_constant_firing,
    method="timing_to_constant_firing_rate_variable_fork_speed",
    required_assumptions=("fork_speed", "target_form"),
    optional_assumptions=(
        "resolution",
        "resolution_time",
        "neighbourhood_range",
        "fit_step",
        "maxiter",
        "sim_number",
        "err_threshold",
        "rate_bounds",
        "refine_seed",
        "verbose",
        "print_every",
        "perQ",
        "n_bins",
        "target_form",
    ),
    default_assumptions={"verbose": True},
    source_kinds=(("replication_timing", ("x",)),),
    assumption_kinds=(("fork_speed", ("x", "t", "xt")),),
    assumption_values=(("target_form", ("constant",)),),
    description="Scalar effective fit of I0 from timing with variable fork speed.",
)
rmap.register(
    "replication_timing",
    "firing_rate",
    map_timing_firing,
    method="timing_to_spatial_firing_rate_variable_fork_speed",
    required_assumptions=("fork_speed",),
    optional_assumptions=(
        "resolution",
        "neighbourhood_range",
        "fit_step",
        "maxiter",
        "sim_number",
        "err_threshold",
        "deviation_threshold",
        "space_fit_step",
        "space_maxiter",
        "verbose",
        "print_every",
        "perQ",
        "stall_rate",
        "tau",
        "target_form",
    ),
    default_assumptions={"verbose": True, "target_form": "space"},
    source_kinds=(("replication_timing", ("x",)),),
    assumption_kinds=(("fork_speed", ("x", "t", "xt")),),
    assumption_values=(("target_form", ("space",)),),
    description="Space-dependent effective fit of I(x) from timing with variable fork speed.",
)
rmap.register(
    ("replication_timing", "fork_speed"),
    "firing_rate",
    map_timing_and_speed_to_general_firing,
    method="timing_and_speed_to_time_firing_rate",
    optional_assumptions=(
        "resolution",
        "resolution_time",
        "neighbourhood_range",
        "fit_step",
        "maxiter",
        "sim_number",
        "err_threshold",
        "deviation_threshold",
        "space_fit_step",
        "space_maxiter",
        "verbose",
        "print_every",
        "perQ",
        "stall_rate",
        "tau",
        "target_form",
        "time_grid",
        "T0",
        "temporal_smoothw",
        "temporal_fit_step",
        "temporal_maxiter",
        "temporal_sim_number",
        "refine_ratio_bounds",
        "refine_edge_kb",
        "refine_seed",
        "refine_verbose",
        "floor",
    ),
    default_assumptions={"verbose": True, "target_form": "time"},
    source_kinds=(
        ("replication_timing", ("x",)),
        ("fork_speed", ("scalar", "x", "t", "xt")),
    ),
    assumption_values=(("target_form", ("time",)),),
    description="Time-only firing-rate target lifted from the fitted spatial firing profile.",
)
rmap.register(
    ("replication_timing", "fork_speed"),
    "firing_rate",
    map_timing_and_speed_to_general_firing,
    method="timing_and_speed_to_spacetime_firing_rate",
    optional_assumptions=(
        "resolution",
        "resolution_time",
        "neighbourhood_range",
        "fit_step",
        "maxiter",
        "sim_number",
        "err_threshold",
        "deviation_threshold",
        "space_fit_step",
        "space_maxiter",
        "verbose",
        "print_every",
        "perQ",
        "stall_rate",
        "tau",
        "target_form",
        "time_grid",
        "T0",
        "temporal_smoothw",
        "temporal_fit_step",
        "temporal_maxiter",
        "temporal_sim_number",
        "spacetime_fit_step",
        "spacetime_sim_number",
        "spacetime_update",
        "spacetime_order",
        "spacetime_kernel_t",
        "refine_ratio_bounds",
        "refine_edge_kb",
        "refine_seed",
        "refine_verbose",
        "floor",
    ),
    default_assumptions={"verbose": True, "target_form": "spacetime"},
    source_kinds=(
        ("replication_timing", ("x",)),
        ("fork_speed", ("scalar", "x", "t", "xt")),
    ),
    assumption_values=(("target_form", ("spacetime",)),),
    description="Space-time firing-rate target lifted from the fitted spatial firing profile.",
)
rmap.register(
    "replication_timing",
    "kinetic_fields",
    map_timing_to_kinetic_fields,
    method="timing_to_joint_spacetime_kinetic_fields",
    optional_assumptions=(
        "resolution",
        "resolution_time",
        "time_grid",
        "T0",
        "initial_firing_rate",
        "initial_fork_speed",
        "initial_kinetics",
        "target_form",
        "maxiter",
        "space_init_fit_step",
        "space_init_maxiter",
        "space_init_sim_number",
        "joint_maxiter",
        "joint_sim_number",
        "firing_space_fit_step",
        "firing_time_fit_step",
        "speed_space_fit_step",
        "speed_time_fit_step",
        "firing_bounds",
        "speed_bounds",
        "firing_floor",
        "refine_ratio_bounds",
        "refine_edge_kb",
        "refine_seed",
        "spatial_smoothw",
        "temporal_smoothw",
        "verbose",
        "print_every",
        "perQ",
    ),
    default_assumptions={"verbose": True, "target_form": "spacetime"},
    source_kinds=(("replication_timing", ("x",)),),
    assumption_values=(("target_form", ("spacetime",)),),
    description="Simultaneous effective fit of I(x,t) and v(x,t) from replication timing.",
)
rmap.register(
    "replication_timing",
    "fork_directionality",
    map_timing_directionality,
    method="timing_to_directionality",
    required_assumptions=("fork_speed",),
    optional_assumptions=("resolution", "T0", "smoothw", "perQ"),
    default_assumptions={"fork_speed": 1.4},
    source_kinds=(("replication_timing", ("x",)),),
    assumption_kinds=(("fork_speed", ("scalar",)),),
    description="Gradient-based fork-directionality mapping.",
)
rmap.register(
    "fork_directionality",
    "replication_timing",
    map_directionality_timing,
    method="directionality_to_timing",
    required_assumptions=("fork_speed",),
    optional_assumptions=("resolution", "T0", "smoothw", "perQ"),
    default_assumptions={"fork_speed": 1.4},
    source_kinds=(("fork_directionality", ("x",)),),
    assumption_kinds=(("fork_speed", ("scalar",)),),
    description="Integration of fork directionality to timing.",
)

_SIMULATION_DERIVED_FEATURES = {
    "origin_efficiency": {},
    "inter_origin_distance": {},
    "completion_time": {},
    "num_origins": {},
    "fork_directionality": {},
    "replicated_fraction": {"time_statsQ": True},
    "replicated_fraction_xt": {"time_stats_xtQ": True},
    "initiation_events_xt": {"time_stats_xtQ": True},
    "coalescence_events_xt": {"time_stats_xtQ": True},
    "right_fork_density_xt": {"time_stats_xtQ": True},
    "left_fork_density_xt": {"time_stats_xtQ": True},
}

for _target, _defaults in _SIMULATION_DERIVED_FEATURES.items():
    rmap.register(
        ("firing_rate", "fork_speed"),
        _target,
        _simulation_source_feature_mapper(_target, _defaults),
        method=f"firing_and_speed_to_{_target}_simulation",
        optional_assumptions=_RSIM_ASSUMPTION_NAMES,
        default_assumptions={
            "verbose": False,
            **_defaults,
        },
        source_kinds=(
            ("firing_rate", ("scalar", "x", "t", "xt")),
            ("fork_speed", ("scalar", "x", "t", "xt")),
        ),
        description="Simulation-derived feature generated with rsim.",
    )
    rmap.register(
        "firing_rate",
        _target,
        _simulation_feature_mapper(_target, _defaults),
        method=f"firing_to_{_target}_simulation",
        required_assumptions=("fork_speed",),
        optional_assumptions=_RSIM_OPTION_NAMES,
        default_assumptions={
            "fork_speed": 1.4,
            "verbose": False,
            **_defaults,
        },
        source_kinds=(("firing_rate", ("scalar", "x", "t", "xt")),),
        assumption_kinds=(("fork_speed", ("scalar", "x", "t", "xt")),),
        description="Simulation-derived feature generated with rsim.",
    )

rmap.register(
    ("firing_rate", "fork_speed"),
    "fork_density_xt",
    map_firing_and_speed_to_fork_density_xt_simulation,
    method="firing_and_speed_to_fork_density_xt_simulation",
    optional_assumptions=_RSIM_ASSUMPTION_NAMES,
    default_assumptions={
        "time_stats_xtQ": True,
        "verbose": False,
    },
    source_kinds=(
        ("firing_rate", ("scalar", "x", "t", "xt")),
        ("fork_speed", ("scalar", "x", "t", "xt")),
    ),
    description="Simulation-derived total fork-density field generated with rsim.",
)

rmap.register(
    "firing_rate",
    "fork_density_xt",
    map_firing_to_fork_density_xt_simulation,
    method="firing_to_fork_density_xt_simulation",
    required_assumptions=("fork_speed",),
    optional_assumptions=_RSIM_OPTION_NAMES,
    default_assumptions={
        "fork_speed": 1.4,
        "time_stats_xtQ": True,
        "verbose": False,
    },
    source_kinds=(("firing_rate", ("scalar", "x", "t", "xt")),),
    assumption_kinds=(("fork_speed", ("scalar", "x", "t", "xt")),),
    description="Simulation-derived total fork-density field generated with rsim.",
)

