"""High-level route registry for DNAscape feature mapping.

The public ``rmap`` initiation-rate contract is initiation density in
``min**-1 kb**-1`` for every scalar, spatial, temporal and space-time form.
Legacy static timing and stochastic simulators use a per-bin hazard internally;
their route adapters multiply the density by the resolved bin width in kb.
Mean-field/light-cone routes already use densities and require no conversion.
Direct legacy numerical functions and ``rsim`` retain their original units.
"""

import inspect
from dataclasses import dataclass, replace

import numpy as np

from . import conversions as _conversions
from . import fitting as _fitting
from .fitting import (
    map_timing_constant_firing,
    map_timing_firing,
    map_timing_firing_spacetime,
    map_timing_firing_time,
)
from .inverse import map_timing_and_firing_to_speed
from .mapping import (
    map_coalescence_events_profile,
    map_coalescence_events_xt,
    map_directionality_timing,
    map_firing_timing,
    map_fork_densities_directionality,
    map_fork_densities_xt,
    map_initiation_events,
    map_initiation_events_profile,
    map_initiation_events_xt,
    map_origin_efficiency,
    map_probability_density_profile,
    map_probability_density_xt,
    map_timing_directionality,
    map_timing_directionality_fork_speed,
)
from .simulation import rsim
from .utils import mapping_progress, smoothf

__all__ = ["MappingSpec", "DataTypeMapper", "rmap"]

_FEATURE_ALIASES = {
    "ir": "initiation_rate",
    "firing": "initiation_rate",
    "origin_activity": "initiation_rate",
    "initiation": "initiation_rate",
    "initiation_rate": "initiation_rate",
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
    "ori_rate": "initiation_rate",
    "origin_rate": "initiation_rate",
    "eff": "origin_efficiency",
    "efficiency": "origin_efficiency",
    "origin_usage": "origin_efficiency",
    "iod": "inter_origin_distance",
    "inter_origin_distances": "inter_origin_distance",
    "nori": "num_origins",
    "num_oris": "num_origins",
    "number_of_origins": "num_origins",
    "realised_initiation": "initiation_events",
    "realized_initiation": "initiation_events",
    "realised_initiation_profile": "initiation_events",
    "realized_initiation_profile": "initiation_events",
    "realised_initiation_xt": "initiation_events_xt",
    "realized_initiation_xt": "initiation_events_xt",
    "initiation_event_density": "initiation_events_xt",
    "initiation_pdf": "initiation_pdf",
    "initiation_pdf_xt": "initiation_pdf_xt",
    "phi_i": "initiation_pdf_xt",
    "phi_i_xt": "initiation_pdf_xt",
    "phi_i_x": "initiation_pdf",
    "termination": "coalescence_events",
    "termination_profile": "coalescence_events",
    "termination_events": "coalescence_events",
    "termination_density": "coalescence_events_xt",
    "termination_events_xt": "coalescence_events_xt",
    "coalescence_density": "coalescence_events_xt",
    "coalescence_pdf": "coalescence_pdf",
    "coalescence_pdf_xt": "coalescence_pdf_xt",
    "phi_c": "coalescence_pdf_xt",
    "phi_c_xt": "coalescence_pdf_xt",
    "phi_c_x": "coalescence_pdf",
    "right_moving_fork_density": "right_fork_density",
    "left_moving_fork_density": "left_fork_density",
    "total_fork_density_xt": "fork_density_xt",
    "directional_fork_densities": "fork_densities_xt",
    "directional_fork_densities_xt": "fork_densities_xt",
}

_CANONICAL_FEATURES = {
    "initiation_rate",
    "fork_speed",
    "replication_timing",
    "fork_directionality",
    "replicated_fraction",
    "fork_density_xt",
    "fork_densities_xt",
    "right_fork_density",
    "left_fork_density",
    "origin_efficiency",
    "inter_origin_distance",
    "s_phase_duration",
    "completion_time",
    "num_origins",
    "initiation_events",
    "coalescence_events",
    "initiation_pdf",
    "coalescence_pdf",
    "replicated_fraction_xt",
    "initiation_events_xt",
    "coalescence_events_xt",
    "initiation_pdf_xt",
    "coalescence_pdf_xt",
    "right_fork_density_xt",
    "left_fork_density_xt",
}

_FEATURE_DESCRIPTIONS = {
    "initiation_rate": "Initiation-rate density to infer or supply, in min^-1 kb^-1.",
    "fork_speed": "Fork progression speed to infer or supply.",
    "replication_timing": "Replication timing along genomic position.",
    "fork_directionality": "Net fork movement directionality along position.",
    "replicated_fraction": "Fraction of replicated DNA over time.",
    "replicated_fraction_xt": "Space-time replicated-fraction field.",
    "fork_density_xt": "Space-time active fork-density field.",
    "fork_densities_xt": "Right- and left-moving space-time fork-density fields.",
    "right_fork_density": "Spatial-mean right-moving density over time, or time-integrated density along position.",
    "left_fork_density": "Spatial-mean left-moving density over time, or time-integrated density along position.",
    "right_fork_density_xt": "Space-time right-moving fork-density field.",
    "left_fork_density_xt": "Space-time left-moving fork-density field.",
    "origin_efficiency": "Expected initiation-event count or efficiency over user-defined regions.",
    "inter_origin_distance": "Distances between neighbouring fired origins.",
    "s_phase_duration": (
        "Elapsed time from the first realised initiation to replication completion."
    ),
    "completion_time": "Absolute replication-completion time on the simulation clock.",
    "num_origins": "Number of origins fired across simulated S phases.",
    "initiation_events": "Time-integrated observed initiation-event density.",
    "coalescence_events": "Time-integrated observed fork-coalescence-event density.",
    "initiation_pdf": "Time-integrated probability density of observing initiation.",
    "coalescence_pdf": "Time-integrated probability density of observing fork coalescence.",
    "initiation_events_xt": "Space-time observed initiation-event density.",
    "coalescence_events_xt": "Space-time observed fork-coalescence-event density.",
    "initiation_pdf_xt": "Space-time probability density of observing initiation.",
    "coalescence_pdf_xt": "Space-time probability density of observing fork coalescence.",
}

_ASSUMPTION_DESCRIPTIONS = {
    "target_form": "Requested output form: scalar (constant in space and time), array (space-dependent), transposed array (time-dependent), or matrix (space-time dependent, positions by time).",
    "target_shape": "Optional parametric shape for the target field. For initiation-rate targets I(t) and I(x,t), rmap defaults to 'weibull'; use 'free' for the unconstrained fit. Fork-speed targets currently use only 'best': v(x) is site-specific, v(t) uses the current temporal fit, and v(x,t) is factorized as v_x(x) q(t).",
    "weibull_shape": "Weibull shape parameter k used when target_shape='weibull'. k=2 gives a linearly increasing hazard; k=1 recovers the exponential case.",
    "weibull_delay": "Optional activation delay, in minutes, for Weibull initiation-rate target shapes.",
    "fit_weibull_shape": "Whether to optimise the Weibull shape parameter k in addition to alpha(x). By default k is fixed.",
    "alpha_smooth": "Smoothness regularisation strength applied to log(alpha) for Weibull initiation-rate fits.",
    "alpha_prior": "Prior regularisation strength keeping log(alpha) close to its local uniform initialisation.",
    "resolution": "Spatial resolution along the genome, in kb.",
    "resolution_time": "Temporal resolution, in minutes.",
    "time_grid": "Explicit time axis to use for time-dependent fields.",
    "T0": "Reference start time used when converting timing values to a relative time axis.",
    "neighbourhood_range": "Spatial range, in kb, used by deterministic neighbourhood mappings.",
    "fit_step": "Global multiplicative update size. For x,t targets it sets fit_step_x and fit_step_t unless those are supplied explicitly.",
    "maxiter": "Global iteration count. For x,t targets it overrides maxiter_x and maxiter_t, setting both to the same value.",
    "error": "Optional MSE threshold for early stopping during fitting. The default None leaves early stopping off.",
    "sim_number": "Global number of stochastic simulations used by the active fitting phase when a route calls rsim.",
    "err_threshold": "Lower numerical floor for fitted initiation-rate values.",
    "deviation_threshold": "Tolerance below which a spatial firing update can be left unchanged.",
    "rate_bounds": "Optional lower and upper initiation-rate bounds; Weibull fits bound amplitude. Each side can be automatic.",
    "initial_initiation_rate": "Starting initiation rate, or starting amplitude for Weibull fits. None uses the native initializer.",
    "speed_bounds": "Allowed lower and upper bounds for fork-speed fitting.",
    "speed_regularization": (
        "Noise-adaptive smoothness penalty on site speeds in the integrated "
        "timing fit using supplied fork directionality."
    ),
    "initial_fork_speed": "Starting fork-speed value or profile for fork-speed fitting.",
    "space_fit_step": "Spatial x-phase update size; overrides fit_step for x-only initialization or refinement.",
    "space_maxiter": "Number of x-phase iterations; for mixed x,t fits it contributes to the mixed iteration count.",
    "temporal_smoothw": "Optional smoothing window for t-dependent profiles.",
    "temporal_fit_step": "Time-profile t-phase update size; controls t targets and factorized x,t targets.",
    "temporal_maxiter": "Number of t-phase refinement iterations; for mixed x,t fits it contributes to the mixed iteration count.",
    "temporal_sim_number": "Number of simulations per t-phase refinement iteration.",
    "spacetime_fit_step": "Mixed local x,t update size; used only when spacetime_update='mixed'.",
    "spacetime_space_fit_step": "x-phase update size for factorized v(x,t) or I(x,t) refinement.",
    "spacetime_time_fit_step": "t-phase update size for factorized v(x,t) or I(x,t) refinement.",
    "spacetime_sim_number": "Number of simulations per mixed x,t refinement iteration.",
    "spacetime_update": "Space-time update rule: 'factorized' separates x and t components; 'mixed' fits local x,t corrections.",
    "spacetime_order": "Order of factorized x,t phases. For fork-speed v(x,t), 'crossed' alternates one accepted/rejected x step with one accepted/rejected t step; 'space_then_time' and 'time_then_space' are sequential alternatives.",
    "spacetime_kernel_t": "Temporal kernel width, in time bins, for mixed x,t corrections.",
    "refine_ratio_bounds": "Bounds applied to timing-ratio updates during simulation refinement.",
    "refine_edge_kb": (
        "Maximum non-periodic endpoint context, in kb, used for reflected forward "
        "prediction and a narrow update taper; set to zero when the supplied "
        "endpoints are true chromosome ends."
    ),
    "refine_seed": "Seed offset for simulation-refined fitting routes.",
    "refine_verbose": "Whether refinement sub-loops print ETA and fitting error.",
    "floor": "Lower numerical floor for fitted initiation-rate fields.",
    "smoothw": "Spatial smoothing window for fitted profiles.",
    "stall_rate": (
        "Legacy probability that a fork stalls while crossing one spatial bin."
    ),
    "stall_rate_per_kb": (
        "Physical stall density in stalls/kb. Stochastic routes convert it to "
        "1 - exp(-stall_rate_per_kb * resolution)."
    ),
    "tau": "Fork-stall lifetime parameter used by stochastic simulation.",
    "perQ": "Whether the genome is treated as periodic/circular.",
    "cfl": "Maximum transport Courant number used for automatic mean-field integration substeps.",
    "regions": "Genomic regions used for origin-efficiency or regional event-count summaries.",
    "region_names": "Labels for the regions used in regional summaries.",
    "x_array": "Explicit spatial coordinate axis for profile integration.",
    "unreplicated_floor": "Numerical floor for conditioning coalescence on unreplicated DNA.",
    "genome_length": "Genome or interval length, in kb, used to infer n_bins.",
    "interval_kb": "Genomic interval, in kb, used to infer or select a region.",
    "spatial_average": "Spatial averaging: bins weights every supplied bin equally; trapezoid integrates between supplied positions and divides by the sampled span.",
    "tail_policy": "Timing beyond the observation window: require_complete rejects unfinished replication, truncate reports a restricted mean, exponential extrapolates the final survival decay.",
    "tail_tolerance": "Maximum residual unreplicated fraction treated as completed when integrating timing.",
    "tail_fit_points": "Number of final time samples used for the log-linear exponential survival-tail fit.",
    "initial_policy": "Require initially unreplicated DNA, or calculate timing conditional on being unreplicated at the first supplied time.",
    "monotonic_policy": "Reject decreasing fraction curves, or project noisy observations onto their cumulative nondecreasing upper envelope.",
    "density_floor": "Fork-density values at or below this threshold cannot identify local fork speed.",
    "derivative_method": "Temporal derivative on the supplied time grid: local polynomial gradient or forward interval slopes with the final slope repeated.",
    "invalid_policy": "Unidentifiable inverse values: mask with NaN, fill with zero, or raise an error. Validity masks and counts accompany the result metadata.",
    "spatial_knots": "Number of control points for spatial variation in constrained kinetic fitting.",
    "regularization": "Smoothness penalty on logarithmic kinetic-field control points.",
    "tolerance": "Positive convergence tolerance used by the constrained least-squares optimizer.",
}

_ASSUMPTION_ORDER = (
    "target_form",
    "target_shape",
    "weibull_shape",
    "weibull_delay",
    "fit_weibull_shape",
    "resolution",
    "resolution_time",
    "time_grid",
    "T0",
    "interval_kb",
    "genome_length",
    "perQ",
    "cfl",
    "x_array",
    "regions",
    "region_names",
    "neighbourhood_range",
    "initial_fork_speed",
    "rate_bounds",
    "speed_bounds",
    "speed_regularization",
    "alpha_smooth",
    "alpha_prior",
    "fit_step",
    "maxiter",
    "error",
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
    "err_threshold",
    "initial_initiation_rate",
    "deviation_threshold",
    "rate_bounds",
    "floor",
    "unreplicated_floor",
    "smoothw",
    "refine_ratio_bounds",
    "refine_edge_kb",
    "refine_seed",
    "refine_verbose",
    "stall_rate",
    "stall_rate_per_kb",
    "tau",
)

_ASSUMPTION_FALLBACK_DEFAULTS = {
    "target_shape": "free",
    "error": None,
    "maxiter": 10,
    "weibull_shape": 2.0,
    "weibull_delay": 0.0,
    "fit_weibull_shape": False,
    "alpha_smooth": 1e-2,
    "alpha_prior": 0.0,
    "space_maxiter": 10,
    "temporal_fit_step": 0.35,
    "temporal_maxiter": 10,
    "temporal_sim_number": 100,
    "spacetime_fit_step": 0.25,
    "spacetime_sim_number": 100,
    "spacetime_update": "factorized",
    "spacetime_order": "crossed",
    "spacetime_kernel_t": 1.0,
    "sim_number": 100,
    "refine_edge_kb": 1000,
    "stall_rate_per_kb": 0.0,
}

_FORM_DESCRIPTIONS = {
    "scalar": "scalar (constant in space and time)",
    "constant": "scalar (constant in space and time)",
    "x": "array (space-dependent)",
    "space": "array (space-dependent)",
    "t": "transposed array (time-dependent)",
    "time": "transposed array (time-dependent)",
    "xt": "matrix (space-time dependent; positions by time)",
    "spacetime": "matrix (space-time dependent; positions by time)",
    "supplied": "supplied data",
}

_PUBLIC_MAPPING_TARGETS = (
    "initiation_rate",
    "fork_speed",
    "replication_timing",
    "replicated_fraction_xt",
    "replicated_fraction",
    "fork_directionality",
    "fork_densities_xt",
    "right_fork_density_xt",
    "left_fork_density_xt",
    "right_fork_density",
    "left_fork_density",
    "initiation_events_xt",
    "initiation_pdf_xt",
    "initiation_events",
    "initiation_pdf",
    "origin_efficiency",
    "coalescence_events_xt",
    "coalescence_pdf_xt",
    "coalescence_events",
    "coalescence_pdf",
    "inter_origin_distance",
    "s_phase_duration",
    "completion_time",
    "num_origins",
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
    "target_distribution": "target_shape",
    "target_model": "target_shape",
    "weibull_k": "weibull_shape",
    "k_weibull": "weibull_shape",
    "activation_delay": "weibull_delay",
    "firing_delay": "weibull_delay",
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
    "periodic": "perQ",
    "perq": "perQ",
    "positions": "x_array",
    "positions_kb": "x_array",
    "origin_regions": "regions",
    "origin_windows": "regions",
    "region_kb": "regions",
    "regions_kb": "regions",
    "origin_names": "region_names",
    "stall_density": "stall_rate_per_kb",
    "stall_density_per_kb": "stall_rate_per_kb",
    "stalls_per_kb": "stall_rate_per_kb",
    "stall_probability": "stall_rate",
    "stall_probability_per_bin": "stall_rate",
}


def _canonical_feature(name):
    from ._terminology import canonical_rate_key
    text = str(name).strip().replace("-", "_")
    key = canonical_rate_key(text.lower())
    if key in _FEATURE_ALIASES:
        return _FEATURE_ALIASES[key]
    if key in _CANONICAL_FEATURES:
        return key
    return canonical_rate_key(text)


def _canonical_assumption(name):
    from ._terminology import canonical_rate_key
    text = str(name).strip().replace("-", "_")
    key = canonical_rate_key(text.lower())
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
    position-by-time space-time fields.
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
            # An explicitly named space-time observable retains both semantic
            # axes even when one has length one. Kinetic inputs still use the
            # ordinary scalar/x/t convention because their names have no _xt.
            if cname.endswith("_xt") and np.asarray(source_data[cname]).ndim == 2:
                kind = "xt"
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
    if name not in {"ori_rate", "initiation_rate", "initiation_rate"}
)
_RSIM_ASSUMPTION_NAMES = tuple(
    name for name in _RSIM_OPTION_NAMES
    if name != "fork_speed"
)


class _SimulationFeatureArray(np.ndarray):
    """Numeric rmap result carrying optional rsim provenance metadata."""

    def __new__(cls, values, info=None):
        obj = np.asarray(values).view(cls)
        obj._rmap_info = dict(info or {})
        return obj

    def __array_finalize__(self, obj):
        if obj is not None:
            self._rmap_info = getattr(obj, "_rmap_info", {})


def _simulation_feature_result(result, output_name):
    values = result[output_name]
    if output_name not in {"inter_origin_distance", "inter_origin_distances"}:
        return values
    if values is None:
        return None
    return _SimulationFeatureArray(
        values,
        {
            "inter_origin_distance_count": result.get(
                "inter_origin_distance_count",
                np.asarray(values).size,
            ),
            "inter_origin_distances_truncated": result.get(
                "inter_origin_distances_truncated",
                False,
            ),
        },
    )


def _rmap_spatial_resolution(options):
    """Resolve the kb-per-bin width with the legacy simulator's precedence."""
    resolution = options.get("resolution_space", 1.0)
    if options.get("resolution") is not None:
        resolution = options["resolution"]
    if options.get("resolution_x") is not None:
        resolution = options["resolution_x"]
    resolution = float(resolution)
    if not np.isfinite(resolution) or resolution <= 0.0:
        raise ValueError("Spatial resolution must be positive and finite.")
    return resolution


def _firing_density_to_bin_hazard(initiation_rate, options):
    """Convert public rmap density into a legacy per-bin initiation hazard."""
    return np.asarray(initiation_rate, dtype=float) * _rmap_spatial_resolution(options)


def _simulation_feature_mapper(output_name, default_options=None):
    def mapper(initiation_rate, **kwargs):
        options = dict(default_options or {})
        # A mapping route returns one selected feature, so retaining every
        # per-replicate profile would be pure hidden memory overhead. Pooled
        # inter-origin distances are needed only by the IOD routes. Keep these
        # as defaults rather than forced values so direct Python callers can
        # still request the auxiliary outputs explicitly.
        options.setdefault("retain_rep_times_per_sim", False)
        options.setdefault(
            "retain_inter_origin_distances",
            output_name in {"inter_origin_distance", "inter_origin_distances"},
        )
        options.update({
            key: value
            for key, value in kwargs.items()
            if key in _RSIM_OPTION_NAMES
        })
        hazard = _firing_density_to_bin_hazard(initiation_rate, options)
        deterministic = _deterministic_mapping_feature(output_name, hazard, options)
        if deterministic is not None:
            return deterministic
        return _simulation_feature_result(rsim(initiation_rate=hazard, **options), output_name)

    mapper.__name__ = f"map_firing_to_{output_name}_simulation"
    return mapper


def _simulation_source_feature_mapper(output_name, default_options=None):
    def mapper(source, **kwargs):
        options = dict(default_options or {})
        options.setdefault("retain_rep_times_per_sim", False)
        options.setdefault(
            "retain_inter_origin_distances",
            output_name in {"inter_origin_distance", "inter_origin_distances"},
        )
        options.update({
            key: value
            for key, value in kwargs.items()
            if key in _RSIM_ASSUMPTION_NAMES
        })
        options["fork_speed"] = source["fork_speed"]
        hazard = _firing_density_to_bin_hazard(source["initiation_rate"], options)
        deterministic = _deterministic_mapping_feature(output_name, hazard, options)
        if deterministic is not None:
            return deterministic
        return _simulation_feature_result(
            rsim(
                initiation_rate=hazard,
                **options,
            ),
            output_name,
        )

    mapper.__name__ = f"map_firing_and_speed_to_{output_name}_simulation"
    return mapper


def _deterministic_mapping_feature(output_name, hazard, options):
    """Timing and directionality have deterministic no-stalling forward maps."""
    resolution = _rmap_spatial_resolution(options)
    if _fitting._stochastic_mapping_model(options, resolution) or output_name not in {"replication_timing", "fork_directionality"}:
        return None
    if output_name == "replication_timing":
        return _fitting._mapping_prediction(initiation_rate=hazard, **options)["replication_timing"]
    density = np.asarray(hazard, dtype=float) / resolution
    selected = {key: value for key, value in options.items() if key in _FORK_DENSITY_OPTIONS}
    selected["resolution"] = resolution
    selected["fork_speed"] = options.get("fork_speed", 1.4)
    if selected.get("time_grid") is None and selected.get("max_time") is None and selected.get("max_rep_time") is None:
        selected["max_time"] = 1000.
    if options.get("geometry") in {"linear", "periodic"}:
        selected["perQ"] = options["geometry"] == "periodic"
    from .inverse import _compact_space_time_field, _mean_field_timing
    from .mapping import _fraction_time_axis, _resolve_n_bins_from_fields
    speed = selected.get("fork_speed", 1.4)
    n = _resolve_n_bins_from_fields(density, speed, n_bins=selected.get("n_bins"), resolution_x=resolution,
                                  genome_length=selected.get("genome_length"), interval_kb=selected.get("interval_kb"))
    if n is None:
        raise ValueError("Provide a spatial profile or region length for directionality.")
    dt = selected.get("resolution_time", 5.)
    t = _fraction_time_axis((density, speed), n, resolution_time=dt, time_grid=selected.get("time_grid"),
                            max_time=selected.get("max_time") or selected.get("max_rep_time"))
    i = _compact_space_time_field(density, len(t), n, t, dt, "initiation_rate")
    v = _compact_space_time_field(speed, len(t), n, t, dt, "fork_speed", positive=True)
    integrals = (np.zeros(n), np.zeros(n))
    _mean_field_timing(i, v, t, resolution, selected.get("perQ", False), fork_integrals=integrals)
    return map_fork_densities_directionality(*integrals)


_REPLICATED_FRACTION_OPTIONS = (
    "resolution",
    "resolution_time",
    "time_grid",
    "max_time",
    "max_rep_time",
    "n_bins",
    "genome_length",
    "interval_kb",
    "perQ",
)


def map_firing_and_speed_to_replicated_fraction_xt(source, **kwargs):
    from ._forward_mapping import fraction_field
    clock, field = fraction_field(source["initiation_rate"], source["fork_speed"],
                                  **_options(kwargs, _REPLICATED_FRACTION_OPTIONS))
    return _SimulationFeatureArray(field, {"time_grid": clock, "method": "deterministic"})


def map_firing_to_replicated_fraction_xt(initiation_rate, **kwargs):
    return map_firing_and_speed_to_replicated_fraction_xt(
        {"initiation_rate": initiation_rate, "fork_speed": kwargs.get("fork_speed", 1.4)}, **kwargs)


def map_firing_and_speed_to_replicated_fraction(source, **kwargs):
    field = map_firing_and_speed_to_replicated_fraction_xt(source, **kwargs)
    return _SimulationFeatureArray(np.mean(field, axis=0), field._rmap_info)


def map_firing_to_replicated_fraction(initiation_rate, **kwargs):
    return map_firing_and_speed_to_replicated_fraction(
        {"initiation_rate": initiation_rate, "fork_speed": kwargs.get("fork_speed", 1.4)}, **kwargs)


_INITIATION_EVENT_OPTIONS = ("resolution_time", "time_grid")
_INITIATION_EVENT_DEFAULTS = {
    "resolution_time": 5.0,
    "time_grid": None,
}
_INITIATION_FORWARD_OPTIONS = tuple(dict.fromkeys(
    _REPLICATED_FRACTION_OPTIONS + _INITIATION_EVENT_OPTIONS
))
_INITIATION_FORWARD_DEFAULTS = {
    "resolution": 1.0,
    "resolution_time": 5.0,
    "time_grid": None,
    "max_time": None,
    "max_rep_time": None,
    "n_bins": None,
    "genome_length": None,
    "interval_kb": None,
    "perQ": False,
}
_PDF_OPTIONS = ("resolution", "resolution_time", "time_grid", "x_array")
_PDF_DEFAULTS = {
    "resolution": 1.0,
    "resolution_time": 5.0,
    "time_grid": None,
    "x_array": None,
}
_INITIATION_PDF_FORWARD_OPTIONS = tuple(dict.fromkeys(
    _INITIATION_FORWARD_OPTIONS + _PDF_OPTIONS
))
_INITIATION_PDF_FORWARD_DEFAULTS = dict(_INITIATION_FORWARD_DEFAULTS)
_INITIATION_PDF_FORWARD_DEFAULTS.update(_PDF_DEFAULTS)


def map_firing_and_fraction_to_initiation_events_xt(source, **kwargs):
    return map_initiation_events_xt(
        source["initiation_rate"],
        source["replicated_fraction_xt"],
        **_options(kwargs, _INITIATION_EVENT_OPTIONS),
    )


def map_firing_and_fraction_to_initiation_events(source, **kwargs):
    return map_initiation_events(
        source["initiation_rate"],
        source["replicated_fraction_xt"],
        **_options(kwargs, _INITIATION_EVENT_OPTIONS),
    )


def map_firing_and_fraction_to_initiation_pdf_xt(source, **kwargs):
    initiation_xt = map_firing_and_fraction_to_initiation_events_xt(source, **kwargs)
    return map_event_density_to_pdf_xt(initiation_xt, **kwargs)


def map_firing_and_fraction_to_initiation_pdf(source, **kwargs):
    pdf_xt = map_firing_and_fraction_to_initiation_pdf_xt(source, **kwargs)
    return map_probability_density_profile(
        pdf_xt,
        **_options(kwargs, _INITIATION_EVENT_OPTIONS),
    )


def map_initiation_events_xt_to_initiation_events(initiation_events_xt, **kwargs):
    return map_initiation_events_profile(
        initiation_events_xt,
        **_options(kwargs, _INITIATION_EVENT_OPTIONS),
    )


def map_firing_and_speed_to_initiation_events_xt(source, **kwargs):
    return _kinetic_event_result(source, "initiation_events_xt", **kwargs)


def map_firing_and_speed_to_initiation_events(source, **kwargs):
    return _kinetic_event_result(source, "initiation_events", **kwargs)


def _kinetic_event_result(source, output, **kwargs):
    from ._event_densities import kinetic_event_densities
    options = _options(kwargs, _REPLICATED_FRACTION_OPTIONS + ("cfl", "unreplicated_floor"))
    fields = kinetic_event_densities(source["initiation_rate"], source["fork_speed"], **options)
    return _SimulationFeatureArray(fields[output], {
        "time_grid": fields["time_grid"],
        "event_density_units": "events/(kb min)" if output.endswith("_xt") else "events/kb",
        "event_time_integration": "adaptive",
    })


def map_firing_and_speed_to_initiation_pdf_xt(source, **kwargs):
    initiation_xt = map_firing_and_speed_to_initiation_events_xt(source, **kwargs)
    return map_event_density_to_pdf_xt(initiation_xt, **kwargs)


def map_firing_and_speed_to_initiation_pdf(source, **kwargs):
    pdf_xt = map_firing_and_speed_to_initiation_pdf_xt(source, **kwargs)
    return map_probability_density_profile(
        pdf_xt,
        **_options(kwargs, _INITIATION_EVENT_OPTIONS),
    )


def map_event_density_to_pdf_xt(event_density_xt, **kwargs):
    return map_probability_density_xt(
        event_density_xt,
        **_options(kwargs, _PDF_OPTIONS),
    )


def map_event_density_to_pdf(event_density_xt, **kwargs):
    pdf_xt = map_event_density_to_pdf_xt(event_density_xt, **kwargs)
    return map_probability_density_profile(
        pdf_xt,
        **_options(kwargs, _INITIATION_EVENT_OPTIONS),
    )


def map_pdf_xt_to_pdf(pdf_xt, **kwargs):
    return map_probability_density_profile(
        pdf_xt,
        **_options(kwargs, _INITIATION_EVENT_OPTIONS),
    )


_ORIGIN_EFFICIENCY_OPTIONS = (
    "resolution",
    "resolution_time",
    "time_grid",
    "x_array",
    "regions",
    "region_names",
)
_ORIGIN_EFFICIENCY_DEFAULTS = {
    "resolution": 1.0,
    "resolution_time": 5.0,
    "time_grid": None,
    "x_array": None,
    "region_names": None,
}


def map_initiation_events_to_origin_efficiency(initiation_events, **kwargs):
    return map_origin_efficiency(
        initiation_events,
        **_options(kwargs, _ORIGIN_EFFICIENCY_OPTIONS),
    )


_COALESCENCE_EVENT_OPTIONS = (
    "resolution_time",
    "time_grid",
    "unreplicated_floor",
)
_COALESCENCE_EVENT_DEFAULTS = {
    "resolution_time": 5.0,
    "time_grid": None,
    "unreplicated_floor": 1e-12,
}
_COALESCENCE_PROFILE_OPTIONS = ("resolution_time", "time_grid")
_COALESCENCE_PROFILE_DEFAULTS = {
    "resolution_time": 5.0,
    "time_grid": None,
}
_COALESCENCE_PDF_OPTIONS = tuple(dict.fromkeys(
    _COALESCENCE_EVENT_OPTIONS + _PDF_OPTIONS
))
_COALESCENCE_PDF_DEFAULTS = dict(_COALESCENCE_EVENT_DEFAULTS)
_COALESCENCE_PDF_DEFAULTS.update(_PDF_DEFAULTS)


def map_fork_density_bundle_to_coalescence_events_xt(source, **kwargs):
    fork_densities = source["fork_densities_xt"]
    if not isinstance(fork_densities, dict):
        raise ValueError("fork_densities_xt must be the dictionary returned by rmap(..., target='fork_densities_xt').")
    return map_coalescence_events_xt(
        source["fork_speed"],
        fork_densities["replicated_fraction_xt"],
        fork_densities["right_fork_density_xt"],
        fork_densities["left_fork_density_xt"],
        **_options(kwargs, _COALESCENCE_EVENT_OPTIONS),
    )


def map_fork_density_bundle_to_coalescence_pdf_xt(source, **kwargs):
    coalescence_xt = map_fork_density_bundle_to_coalescence_events_xt(source, **kwargs)
    return map_event_density_to_pdf_xt(coalescence_xt, **kwargs)


def map_fork_density_bundle_to_coalescence_events(source, **kwargs):
    coalescence_xt = map_fork_density_bundle_to_coalescence_events_xt(source, **kwargs)
    return map_coalescence_events_xt_to_coalescence_events(coalescence_xt, **kwargs)


def map_fork_density_fields_to_coalescence_events_xt(source, **kwargs):
    return map_coalescence_events_xt(
        source["fork_speed"],
        source["replicated_fraction_xt"],
        source["right_fork_density_xt"],
        source["left_fork_density_xt"],
        **_options(kwargs, _COALESCENCE_EVENT_OPTIONS),
    )


def map_fork_density_fields_to_coalescence_pdf_xt(source, **kwargs):
    coalescence_xt = map_fork_density_fields_to_coalescence_events_xt(source, **kwargs)
    return map_event_density_to_pdf_xt(coalescence_xt, **kwargs)


def map_fork_density_fields_to_coalescence_events(source, **kwargs):
    coalescence_xt = map_fork_density_fields_to_coalescence_events_xt(source, **kwargs)
    return map_coalescence_events_xt_to_coalescence_events(coalescence_xt, **kwargs)


def map_firing_and_speed_to_coalescence_events_xt(source, **kwargs):
    return _kinetic_event_result(source, "coalescence_events_xt", **kwargs)


def map_firing_and_speed_to_coalescence_pdf_xt(source, **kwargs):
    coalescence_xt = map_firing_and_speed_to_coalescence_events_xt(source, **kwargs)
    return map_event_density_to_pdf_xt(coalescence_xt, **kwargs)


def map_coalescence_events_xt_to_coalescence_events(coalescence_events_xt, **kwargs):
    return map_coalescence_events_profile(
        coalescence_events_xt,
        **_options(kwargs, _COALESCENCE_PROFILE_OPTIONS),
    )


def map_firing_and_speed_to_coalescence_events(source, **kwargs):
    return _kinetic_event_result(source, "coalescence_events", **kwargs)


def map_firing_and_speed_to_coalescence_pdf(source, **kwargs):
    pdf_xt = map_firing_and_speed_to_coalescence_pdf_xt(source, **kwargs)
    return map_probability_density_profile(
        pdf_xt,
        **_options(kwargs, _COALESCENCE_PROFILE_OPTIONS),
    )


_FORK_DENSITY_OPTIONS = _REPLICATED_FRACTION_OPTIONS + ("cfl",)
_COALESCENCE_FORWARD_OPTIONS = tuple(dict.fromkeys(
    _FORK_DENSITY_OPTIONS + _COALESCENCE_EVENT_OPTIONS
))
_COALESCENCE_PDF_FORWARD_OPTIONS = tuple(dict.fromkeys(
    _COALESCENCE_FORWARD_OPTIONS + _PDF_OPTIONS
))
_FORK_DENSITY_DIRECTIONALITY_OPTIONS = ("time_grid", "resolution_time")
_FORK_DENSITY_DIRECTIONALITY_DEFAULTS = {
    "time_grid": None,
    "resolution_time": 5.0,
}


def map_firing_and_speed_to_fork_densities_xt(source, **kwargs):
    options = _options(kwargs, _FORK_DENSITY_OPTIONS)
    if all(options.get(key) is None for key in ("time_grid", "max_time", "max_rep_time")):
        from ._forward_mapping import fraction_field
        clock, _ = fraction_field(source["initiation_rate"], source["fork_speed"],
                                  **_options(options, _REPLICATED_FRACTION_OPTIONS))
        options["time_grid"] = clock
    result = map_fork_densities_xt(
        source["initiation_rate"],
        fork_speed=source["fork_speed"],
        **options,
    )
    from .mapping import _fraction_time_axis
    n = result["replicated_fraction_xt"].shape[0]
    result["time_grid"] = _fraction_time_axis(
        (source["initiation_rate"], source["fork_speed"]), n,
        resolution_time=options.get("resolution_time", 5.), time_grid=options.get("time_grid"),
        max_time=options.get("max_time") or options.get("max_rep_time"))
    return result


def map_firing_and_speed_to_right_fork_density_xt(source, **kwargs):
    fields = map_firing_and_speed_to_fork_densities_xt(source, **kwargs)
    return _SimulationFeatureArray(fields["right_fork_density_xt"], {"time_grid": fields["time_grid"]})


def map_firing_and_speed_to_left_fork_density_xt(source, **kwargs):
    fields = map_firing_and_speed_to_fork_densities_xt(source, **kwargs)
    return _SimulationFeatureArray(fields["left_fork_density_xt"], {"time_grid": fields["time_grid"]})


def map_fork_density_bundle_to_directionality(fork_densities, **kwargs):
    return map_fork_densities_directionality(
        fork_densities["right_fork_density_xt"],
        fork_densities["left_fork_density_xt"],
        **_options(kwargs, _FORK_DENSITY_DIRECTIONALITY_OPTIONS),
    )


def map_fork_density_fields_to_directionality(source, **kwargs):
    return map_fork_densities_directionality(
        source["right_fork_density_xt"],
        source["left_fork_density_xt"],
        **_options(kwargs, _FORK_DENSITY_DIRECTIONALITY_OPTIONS),
    )


def map_fork_density_profiles_to_directionality(source, **kwargs):
    return map_fork_densities_directionality(
        source["right_fork_density"],
        source["left_fork_density"],
        **_options(kwargs, _FORK_DENSITY_DIRECTIONALITY_OPTIONS),
    )


_FORWARD_TIMING_OPTIONS = {
    "resolution",
    "neighbourhood_range",
    "perQ",
    "n_bins",
    "genome_length",
    "interval_kb",
    "resolution_time", "time_grid", "max_time", "max_rep_time",
}

_TIMING_TO_CONSTANT_FIRING_OPTIONS = {
    "resolution",
    "resolution_time",
    "time_grid",
    "neighbourhood_range",
    "fit_step",
    "maxiter",
    "error",
    "sim_number",
    "err_threshold",
    "initial_initiation_rate",
    "rate_bounds",
    "refine_seed",
    "verbose",
    "print_every",
    "perQ",
    "n_bins",
}

_TIMING_TO_SPATIAL_FIRING_OPTIONS = {
    "resolution",
    "resolution_time",
    "time_grid",
    "refine_seed",
    "neighbourhood_range",
    "fit_step",
    "maxiter",
    "error",
    "sim_number",
    "err_threshold",
    "initial_initiation_rate",
    "deviation_threshold",
    "rate_bounds",
    "space_fit_step",
    "space_maxiter",
    "verbose",
    "print_every",
    "perQ",
    "refine_edge_kb",
    "stall_rate",
    "stall_rate_per_kb",
    "tau",
}

_TIMING_TO_GENERAL_FIRING_OPTIONS = _TIMING_TO_SPATIAL_FIRING_OPTIONS | {
    "target_shape",
    "weibull_shape",
    "weibull_delay",
    "fit_weibull_shape",
    "alpha_smooth",
    "alpha_prior",
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
    "spacetime_space_fit_step",
    "spacetime_time_fit_step",
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

_TIMING_TO_SPATIAL_SPEED_OPTIONS = {
    "target_shape",
    "resolution",
    "resolution_time",
    "time_grid",
    "T0",
    "neighbourhood_range",
    "fit_step",
    "maxiter",
    "error",
    "space_fit_step",
    "space_maxiter",
    "sim_number",
    "speed_bounds",
    "initial_fork_speed",
    "smoothw",
    "refine_edge_kb",
    "refine_seed",
    "verbose",
    "print_every",
    "perQ",
    "n_bins",
}

_TIMING_TO_GENERAL_SPEED_OPTIONS = _TIMING_TO_SPATIAL_SPEED_OPTIONS | {
    "target_shape",
    "resolution_time",
    "time_grid",
    "T0",
    "temporal_smoothw",
    "temporal_fit_step",
    "temporal_maxiter",
    "temporal_sim_number",
    "spacetime_fit_step",
    "spacetime_space_fit_step",
    "spacetime_time_fit_step",
    "spacetime_sim_number",
    "spacetime_update",
    "spacetime_order",
    "spacetime_kernel_t",
    "refine_ratio_bounds",
    "refine_edge_kb",
    "refine_seed",
    "refine_verbose",
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


def map_firing_density_to_timing(
    initiation_rate, fork_speed=1.4, resolution=1.0, resolution_x=None,
    neighbourhood_range=2000, perQ=False, n_bins=None, genome_length=None,
    interval_kb=None, resolution_time=5.0, time_grid=None, max_time=None, max_rep_time=None,
):
    """Adapt density-valued public firing inputs to the legacy timing solver."""
    dx = _rmap_spatial_resolution({"resolution": resolution, "resolution_x": resolution_x})
    from .mapping import _field_time_length, _resolve_n_bins_from_fields
    n = _resolve_n_bins_from_fields(initiation_rate, fork_speed, n_bins=n_bins,
        resolution_x=dx, genome_length=genome_length, interval_kb=interval_kb)
    if any(_field_time_length(field, n) is not None for field in (initiation_rate, fork_speed)):
        from ._forward_mapping import fraction_field, timing_from_fraction
        clock, fraction = fraction_field(initiation_rate, fork_speed, resolution=dx,
            resolution_time=resolution_time, time_grid=time_grid, max_time=max_time,
            max_rep_time=max_rep_time, n_bins=n, interval_kb=interval_kb, perQ=perQ)
        complete = time_grid is None and max_time is None and max_rep_time is None
        return _SimulationFeatureArray(timing_from_fraction(clock, fraction, require_complete=complete),
                                       {"time_grid": clock, "method": "deterministic"})
    return map_firing_timing(
        np.asarray(initiation_rate, dtype=float) * dx, fork_speed=fork_speed,
        resolution=dx, neighbourhood_range=neighbourhood_range, perQ=perQ,
        n_bins=n_bins, genome_length=genome_length, interval_kb=interval_kb,
    )


def map_firing_and_speed_to_timing(source, **kwargs):
    return map_firing_density_to_timing(
        source["initiation_rate"], fork_speed=source["fork_speed"],
        **_options(kwargs, _FORWARD_TIMING_OPTIONS),
    )


def map_firing_and_speed_to_directionality(source, **kwargs):
    """Compute directionality through deterministic replication timing.

    A temporal velocity weights the survival gradient at each time, which is
    the timing-gradient identity before time integration. A static velocity
    uses the existing timing-to-directionality route directly.
    """
    from .mapping import _expand_space_time_field, _field_time_length, _resolve_n_bins_from_fields
    options = _options(kwargs, _FORWARD_TIMING_OPTIONS)
    dx = _rmap_spatial_resolution(options)
    firing, speed = source["initiation_rate"], source["fork_speed"]
    n = _resolve_n_bins_from_fields(firing, speed, n_bins=options.get("n_bins"),
        resolution_x=dx, genome_length=options.get("genome_length"), interval_kb=options.get("interval_kb"))
    if _field_time_length(speed, n) is None:
        timing = map_firing_and_speed_to_timing(source, **options)
        result = map_timing_directionality(timing, speed, resolution=dx,
            perQ=bool(options.get("perQ", False)), T0=kwargs.get("T0"), smoothw=kwargs.get("smoothw"))
        return _SimulationFeatureArray(result, {**getattr(timing, "_rmap_info", {}), "method": "deterministic"})
    from ._forward_mapping import fraction_field, timing_from_fraction
    clock, fraction = fraction_field(firing, speed, **_options(options, _REPLICATED_FRACTION_OPTIONS))
    complete = all(options.get(key) is None for key in ("time_grid", "max_time", "max_rep_time"))
    timing_from_fraction(clock, fraction, require_complete=complete)
    survival = 1. - fraction
    velocity = _expand_space_time_field(speed, len(clock), n, clock,
        options.get("resolution_time", 5.), "fork_speed", positive=True).T
    difference = survival - np.roll(survival, 1, axis=0)
    edge_velocity = .5 * (velocity + np.roll(velocity, 1, axis=0))
    if not options.get("perQ", False):
        difference[0] = 0.
    result = np.trapezoid(edge_velocity * difference, clock, axis=1) / dx
    result = np.clip(smoothf(result, kwargs.get("smoothw")), -1., 1.)
    return _SimulationFeatureArray(result, {"time_grid": clock, "method": "deterministic"})


def map_firing_to_directionality(initiation_rate, **kwargs):
    return map_firing_and_speed_to_directionality(
        {"initiation_rate": initiation_rate, "fork_speed": kwargs.get("fork_speed", 1.4)}, **kwargs)


def _legacy_firing_options(options, resolution):
    from ._fit_bounds import scale_bounds
    options = dict(options)
    if options.get("rate_bounds") is not None:
        options["rate_bounds"] = scale_bounds(options["rate_bounds"], resolution)
    if options.get("initial_initiation_rate") is not None:
        options["initial_initiation_rate"] = float(options["initial_initiation_rate"]) * resolution
    if options.get("floor") is not None:
        options["floor"] = float(options["floor"]) * resolution
    options["err_threshold"] = float(options.get("err_threshold", 15.0)) - np.log10(resolution)
    return options


def _firing_density_result(result, resolution):
    values = np.asarray(result, dtype=float) / resolution
    info = dict(getattr(result, "_rmap_info", {}))
    info["initiation_rate_units"] = "min^-1 kb^-1"
    return (_fitting._RmapInfoFloat(float(values), info) if values.ndim == 0
            else _fitting._RmapInfoArray(values, info))


def map_timing_to_constant_firing_density(observed_timing, fork_speed=1.4, **kwargs):
    return map_timing_and_speed_to_constant_firing(
        {"replication_timing": observed_timing, "fork_speed": fork_speed}, **kwargs)


def map_timing_to_spatial_firing_density(observed_timing, fork_speed=1.4, **kwargs):
    return map_timing_and_speed_to_firing(
        {"replication_timing": observed_timing, "fork_speed": fork_speed}, **kwargs)


def map_timing_and_speed_to_constant_firing(source, **kwargs):
    resolution = _rmap_spatial_resolution(kwargs)
    result = map_timing_constant_firing(
        source["replication_timing"], fork_speed=source["fork_speed"],
        **_legacy_firing_options(_options(kwargs, _TIMING_TO_CONSTANT_FIRING_OPTIONS), resolution),
    )
    return _firing_density_result(result, resolution)


def map_timing_and_speed_to_firing(source, **kwargs):
    resolution = _rmap_spatial_resolution(kwargs)
    options = _legacy_firing_options(_options(kwargs, _TIMING_TO_SPATIAL_FIRING_OPTIONS), resolution)
    # The paper's squared-ratio update converges faster for the exact model.
    # Simulation noise from stalled forks needs the original smaller step.
    if "fit_step" not in options:
        stalled = _fitting._resolve_stall_probability(
            options.get("stall_rate", 0.), options.get("stall_rate_per_kb"), resolution,
        ) > 0.
        options["fit_step"] = .6 if stalled else 2.
    result = map_timing_firing(
        source["replication_timing"], fork_speed=source["fork_speed"],
        **options,
    )
    return _firing_density_result(result, resolution)


def map_timing_and_speed_to_general_firing(source, **kwargs):
    target_form = _canonical_form(kwargs.get("target_form", "space"))
    options = _options(kwargs, _TIMING_TO_GENERAL_FIRING_OPTIONS)
    options = _apply_global_fit_controls(options, target_form=target_form)
    free = options.get("target_shape", "weibull") == "free"
    resolution = _rmap_spatial_resolution(kwargs)
    fork_speed = source["fork_speed"]
    original_axis = options.get("time_grid")
    uniform_axis = None
    if free:
        options = _legacy_firing_options(options, resolution)
        fork_speed, uniform_axis = _fitting._uniform_speed_for_simulation(
            fork_speed, original_axis, options.get("resolution_time", 5.0))
        if uniform_axis is not None:
            options["time_grid"] = uniform_axis
    if target_form == "time":
        result = map_timing_firing_time(
            source["replication_timing"], fork_speed=fork_speed, **options)
    elif target_form == "spacetime":
        result = map_timing_firing_spacetime(
            source["replication_timing"], fork_speed=fork_speed, **options)
    else:
        raise ValueError("target_form must be 'time' or 'spacetime' for this route.")
    if not free:
        return result
    info = dict(getattr(result, "_rmap_info", {}))
    simulation_rates = np.asarray(result, dtype=float)
    if uniform_axis is not None and not np.array_equal(original_axis, uniform_axis):
        result = np.stack([np.interp(original_axis, uniform_axis, row) for row in simulation_rates])
        simulation_rates = np.stack([np.interp(uniform_axis, original_axis, row) for row in result])
        # A returned coarse field has different interpolation from its fine fit.
        info.pop("replication_timing_fit", None)
    if info.get("replication_timing_fit") is None:
        info["replication_timing_fit"] = _fitting._mapping_prediction(
            timing_reference=source["replication_timing"],
            initiation_rate=simulation_rates, fork_speed=fork_speed,
            resolution_space=resolution, resolution_time=options.get("resolution_time", 5.0),
            n_bins=len(source["replication_timing"]),
            sim_number=options.get("spacetime_sim_number") or options.get("temporal_sim_number", 100),
            perQ=options.get("perQ", False), seed=options.get("refine_seed"),
            stall_rate=options.get("stall_rate", 0.0),
            stall_rate_per_kb=options.get("stall_rate_per_kb"), tau=options.get("tau", np.inf),
            verbose=False, retain_rep_times_per_sim=False, retain_inter_origin_distances=False,
        )["replication_timing"]
    residual = np.asarray(info["replication_timing_fit"]) - source["replication_timing"]
    info["residual"] = residual
    info["fit_mse"] = float(np.mean(residual[np.isfinite(residual)]**2))
    return _firing_density_result(_fitting._RmapInfoArray(result, info), resolution)


def map_timing_and_speed_to_directionality(source, **kwargs):
    return map_timing_directionality(
        source["replication_timing"],
        fork_speed=source["fork_speed"],
        **_options(kwargs, _DIRECTIONALITY_OPTIONS),
    )


def map_timing_and_spatial_speed_to_directionality(source, **kwargs):
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


def map_timing_and_directionality_to_fork_speed(source, **kwargs):
    """Infer fork speed and retain the corresponding timing reconstruction."""

    options = _options(
        kwargs,
        (
            "resolution",
            "T0",
            "smoothw",
            "perQ",
            "target_form",
            "speed_bounds",
            "speed_regularization",
        ),
    )
    speed = map_timing_directionality_fork_speed(
        source["replication_timing"],
        source["fork_directionality"],
        **options,
    )
    timing = np.asarray(source["replication_timing"], dtype=float).reshape(-1)
    directionality = np.asarray(source["fork_directionality"], dtype=float).reshape(-1)
    perq = bool(options.get("perQ", False))
    smoothw = options.get("smoothw")
    supplied_t0 = options.get("T0")
    anchor = 0.0 if supplied_t0 is None else float(supplied_t0)
    timing_fit = np.asarray(
        map_directionality_timing(
            directionality,
            fork_speed=speed,
            resolution=options.get("resolution", 1.0),
            T0=anchor,
            smoothw=smoothw,
            perQ=perq,
        ),
        dtype=float,
    )
    if perq or supplied_t0 is None:
        timing_fit = timing_fit + float(np.mean(timing - timing_fit))
    residual = timing_fit - timing
    finite = np.isfinite(residual)
    fit_mse = float(np.mean(np.square(residual[finite]))) if np.any(finite) else None
    info = {
        "target_shape": "best",
        "replication_timing_fit": timing_fit,
        "residual": residual,
        "fit_mse": fit_mse,
        "fit_mse_description": (
            "Timing reconstruction MSE from the inferred fork speed and supplied "
            "fork directionality."
        ),
    }
    if np.asarray(speed).size == 1:
        info["fork_speed_shape"] = "constant v0"
        return _fitting._RmapInfoFloat(float(np.asarray(speed).reshape(-1)[0]), info)
    info["fork_speed_shape"] = "regularized site-specific v(x)"
    return _fitting._RmapInfoArray(np.asarray(speed, dtype=float), info=info)


def map_timing_and_firing_to_spatial_speed(source, **kwargs):
    options = _options(kwargs, _TIMING_TO_SPATIAL_SPEED_OPTIONS)
    source = dict(source)
    source["initiation_rate"] = _firing_density_to_bin_hazard(source["initiation_rate"], options)
    return _fitting.map_timing_firing_spatial_speed(
        source,
        **options,
    )


def map_timing_and_firing_to_general_speed(source, **kwargs):
    target_form = _canonical_form(kwargs.get("target_form", "space"))
    options = _options(kwargs, _TIMING_TO_GENERAL_SPEED_OPTIONS)
    options = _apply_global_fit_controls(options, target_form=target_form)
    if target_form == "time":
        return _fitting.map_timing_firing_temporal_speed(source, _initiation_rate_is_density=True, **options)
    if target_form == "spacetime":
        return _fitting.map_timing_firing_spacetime_speed(source, _initiation_rate_is_density=True, **options)
    raise ValueError("target_form must be 'time' or 'spacetime' for this route.")


def map_timing_and_time_firing_to_temporal_speed(source, **kwargs):
    options = _apply_global_fit_controls(
        _options(kwargs, _TIMING_TO_GENERAL_SPEED_OPTIONS), target_form="time",
    )
    return _fitting.map_timing_firing_temporal_speed_simulated(
        source, _initiation_rate_is_density=True, **options,
    )


def map_timing_and_general_firing_to_spacetime_speed(source, **kwargs):
    options = _apply_global_fit_controls(
        _options(kwargs, _TIMING_TO_GENERAL_SPEED_OPTIONS), target_form="spacetime",
    )
    return _fitting.map_timing_firing_spacetime_speed_simulated(
        source, _initiation_rate_is_density=True, **options,
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
            "  use rmap.targets() for a user-facing target table and "
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

    def _ensure_forward_timing_routes(self):
        """Keep autoreloaded rmap instances aligned with timing forward routes."""
        stale_methods = {
            "firing_and_variable_speed_to_timing",
            "firing_to_timing_variable_fork_speed",
            "firing_to_timing_general_firing_form",
        }
        self._specs = [
            spec for spec in self._specs
            if spec.method not in stale_methods
        ]
        methods = {spec.method for spec in self._specs}

        if "firing_and_speed_to_timing_simulation" not in methods:
            self.register(
                ("initiation_rate", "fork_speed"),
                "replication_timing",
                _simulation_source_feature_mapper("replication_timing"),
                method="firing_and_speed_to_timing_simulation",
                optional_assumptions=_RSIM_ASSUMPTION_NAMES,
                default_assumptions={"verbose": False},
                source_kinds=(
                    ("initiation_rate", ("scalar", "x", "t", "xt")),
                    ("fork_speed", ("scalar", "x", "t", "xt")),
                ),
                description="Simulation-backed timing map for kinetic fields with time dependence.",
            )

        if "firing_to_timing_simulation" not in methods:
            self.register(
                "initiation_rate",
                "replication_timing",
                _simulation_feature_mapper("replication_timing"),
                method="firing_to_timing_simulation",
                required_assumptions=("fork_speed",),
                optional_assumptions=_RSIM_OPTION_NAMES,
                default_assumptions={"fork_speed": 1.4, "verbose": False},
                source_kinds=(("initiation_rate", ("scalar", "x", "t", "xt")),),
                assumption_kinds=(("fork_speed", ("scalar", "x", "t", "xt")),),
                description="Simulation-backed timing map for kinetic fields with time dependence.",
            )

    def _ensure_fork_speed_shape_routes(self):
        """Keep autoreloaded rmap instances aligned with fork-speed target shapes."""
        updated = []
        changed = False
        for spec in self._specs:
            if spec.target != "fork_speed":
                updated.append(spec)
                continue

            assumption_values = []
            spec_changed = False
            has_target_shape = False
            for name, values in spec.assumption_values:
                if _canonical_assumption(name) != "target_shape":
                    assumption_values.append((name, values))
                    continue
                has_target_shape = True
                if tuple(values) != ("best",):
                    spec_changed = True
                assumption_values.append((name, ("best",)))
            if not has_target_shape:
                assumption_values.append(("target_shape", ("best",)))
                spec_changed = True

            optional = tuple(
                name for name in spec.optional_assumptions
                if name not in {"weibull_shape", "weibull_delay"}
            )
            if optional != tuple(spec.optional_assumptions):
                spec_changed = True
            if "target_shape" not in optional:
                optional = (*optional, "target_shape")
                spec_changed = True

            defaults = dict(spec.default_assumptions)
            if defaults.get("target_shape") != "best":
                defaults["target_shape"] = "best"
                spec_changed = True

            required = tuple(
                name for name in spec.required_assumptions
                if name != "target_shape"
            )
            if required != tuple(spec.required_assumptions):
                spec_changed = True

            if spec_changed:
                spec = replace(
                    spec,
                    required_assumptions=required,
                    optional_assumptions=optional,
                    default_assumptions=tuple(defaults.items()),
                    assumption_values=tuple(assumption_values),
                )
                changed = True
            updated.append(spec)

        if changed:
            self._specs = updated

    def _ensure_directionality_routes(self):
        """Keep autoreloaded rmap instances aligned with new directionality routes."""
        methods = {spec.method for spec in self._specs}

        for method, source, func in (
            ("firing_and_speed_to_directionality_deterministic", ("initiation_rate", "fork_speed"), map_firing_and_speed_to_directionality),
            ("firing_to_directionality_deterministic", "initiation_rate", map_firing_to_directionality),
        ):
            if method not in methods:
                paired = isinstance(source, tuple)
                self.register(source, "fork_directionality", func, method=method,
                    required_assumptions=() if paired else ("fork_speed",),
                    optional_assumptions=tuple(sorted(_FORWARD_TIMING_OPTIONS)) + ("T0", "smoothw"),
                    default_assumptions={} if paired else {"fork_speed": 1.4},
                    source_kinds=tuple((key, ("scalar", "x", "t", "xt"))
                                       for key in (source if paired else (source,))),
                    assumption_kinds=() if paired else (("fork_speed", ("scalar", "x", "t", "xt")),),
                    description="Deterministic directionality through replication timing, without simulations.")

        if "timing_and_spatial_speed_to_directionality" not in methods:
            self.register(
                ("replication_timing", "fork_speed"),
                "fork_directionality",
                map_timing_and_spatial_speed_to_directionality,
                method="timing_and_spatial_speed_to_directionality",
                optional_assumptions=("resolution", "T0", "smoothw", "perQ"),
                source_kinds=(
                    ("replication_timing", ("x",)),
                    ("fork_speed", ("x",)),
                ),
                description="Gradient-based fork-directionality mapping with spatial fork speed.",
            )

        if "timing_to_directionality_spatial_speed" not in methods:
            self.register(
                "replication_timing",
                "fork_directionality",
                map_timing_directionality,
                method="timing_to_directionality_spatial_speed",
                required_assumptions=("fork_speed",),
                optional_assumptions=("resolution", "T0", "smoothw", "perQ"),
                source_kinds=(("replication_timing", ("x",)),),
                assumption_kinds=(("fork_speed", ("x",)),),
                description="Gradient-based fork-directionality mapping with spatial fork speed.",
            )

        if "fork_density_bundle_to_directionality_time_integrated" not in methods:
            self.register(
                "fork_densities_xt",
                "fork_directionality",
                map_fork_density_bundle_to_directionality,
                method="fork_density_bundle_to_directionality_time_integrated",
                optional_assumptions=_FORK_DENSITY_DIRECTIONALITY_OPTIONS,
                default_assumptions=_FORK_DENSITY_DIRECTIONALITY_DEFAULTS,
                description="Time-integrated fork directionality from right- and left-moving fork-density fields.",
            )

        if "fork_density_fields_to_directionality_time_integrated" not in methods:
            self.register(
                ("right_fork_density_xt", "left_fork_density_xt"),
                "fork_directionality",
                map_fork_density_fields_to_directionality,
                method="fork_density_fields_to_directionality_time_integrated",
                optional_assumptions=_FORK_DENSITY_DIRECTIONALITY_OPTIONS,
                default_assumptions=_FORK_DENSITY_DIRECTIONALITY_DEFAULTS,
                source_kinds=(
                    ("right_fork_density_xt", ("xt",)),
                    ("left_fork_density_xt", ("xt",)),
                ),
                description="Time-integrated fork directionality from right- and left-moving fork-density fields.",
            )

        if "fork_density_profiles_to_directionality" not in methods:
            self.register(
                ("right_fork_density", "left_fork_density"),
                "fork_directionality",
                map_fork_density_profiles_to_directionality,
                method="fork_density_profiles_to_directionality",
                optional_assumptions=_FORK_DENSITY_DIRECTIONALITY_OPTIONS,
                default_assumptions=_FORK_DENSITY_DIRECTIONALITY_DEFAULTS,
                source_kinds=(
                    ("right_fork_density", ("x",)),
                    ("left_fork_density", ("x",)),
                ),
                description="Fork directionality from time-integrated right- and left-moving fork-density profiles.",
            )

    def _ensure_replicated_fraction_routes(self):
        self._specs = [
            spec for spec in self._specs
            if not (
                spec.target in {"replicated_fraction", "replicated_fraction_xt"}
                and spec.method.endswith("_simulation")
            )
        ]
        methods = {spec.method for spec in self._specs}

        if "firing_and_speed_to_replicated_fraction_xt_lightcone" not in methods:
            self.register(
                ("initiation_rate", "fork_speed"),
                "replicated_fraction_xt",
                map_firing_and_speed_to_replicated_fraction_xt,
                method="firing_and_speed_to_replicated_fraction_xt_lightcone",
                optional_assumptions=_REPLICATED_FRACTION_OPTIONS,
                source_kinds=(
                    ("initiation_rate", ("scalar", "x", "t", "xt")),
                    ("fork_speed", ("scalar", "x", "t", "xt")),
                ),
                description="Analytical light-cone map of f(x,t) from initiation rate and fork speed.",
            )
        if "firing_to_replicated_fraction_xt_lightcone" not in methods:
            self.register(
                "initiation_rate",
                "replicated_fraction_xt",
                map_firing_to_replicated_fraction_xt,
                method="firing_to_replicated_fraction_xt_lightcone",
                required_assumptions=("fork_speed",),
                optional_assumptions=_REPLICATED_FRACTION_OPTIONS,
                default_assumptions={"fork_speed": 1.4},
                source_kinds=(("initiation_rate", ("scalar", "x", "t", "xt")),),
                assumption_kinds=(("fork_speed", ("scalar", "x", "t", "xt")),),
                description="Analytical light-cone map of f(x,t) from initiation rate and fork speed.",
            )
        if "firing_and_speed_to_replicated_fraction_lightcone" not in methods:
            self.register(
                ("initiation_rate", "fork_speed"),
                "replicated_fraction",
                map_firing_and_speed_to_replicated_fraction,
                method="firing_and_speed_to_replicated_fraction_lightcone",
                optional_assumptions=_REPLICATED_FRACTION_OPTIONS,
                source_kinds=(
                    ("initiation_rate", ("scalar", "x", "t", "xt")),
                    ("fork_speed", ("scalar", "x", "t", "xt")),
                ),
                description="Analytical light-cone map of the spatial mean replicated fraction.",
            )
        if "firing_to_replicated_fraction_lightcone" not in methods:
            self.register(
                "initiation_rate",
                "replicated_fraction",
                map_firing_to_replicated_fraction,
                method="firing_to_replicated_fraction_lightcone",
                required_assumptions=("fork_speed",),
                optional_assumptions=_REPLICATED_FRACTION_OPTIONS,
                default_assumptions={"fork_speed": 1.4},
                source_kinds=(("initiation_rate", ("scalar", "x", "t", "xt")),),
                assumption_kinds=(("fork_speed", ("scalar", "x", "t", "xt")),),
                description="Analytical light-cone map of the spatial mean replicated fraction.",
            )

    def _ensure_initiation_event_routes(self):
        targets = {"initiation_events", "initiation_events_xt"}
        self._specs = [
            spec for spec in self._specs
            if not (
                spec.target in targets
                and spec.method.endswith("_simulation")
            )
        ]
        methods = {spec.method for spec in self._specs}
        if "firing_and_speed_to_initiation_events_xt_mean_field" not in methods:
            self.register(
                ("initiation_rate", "fork_speed"),
                "initiation_events_xt",
                map_firing_and_speed_to_initiation_events_xt,
                method="firing_and_speed_to_initiation_events_xt_mean_field",
                optional_assumptions=_INITIATION_FORWARD_OPTIONS,
                default_assumptions=_INITIATION_FORWARD_DEFAULTS,
                source_kinds=(
                    ("initiation_rate", ("scalar", "x", "t", "xt")),
                    ("fork_speed", ("scalar", "x", "t", "xt")),
                ),
                description="Mean-field observed initiation-event density I(x,t)[1-f(x,t)] from initiation rate and fork speed.",
            )
        if "firing_and_speed_to_initiation_pdf_xt_mean_field" not in methods:
            self.register(
                ("initiation_rate", "fork_speed"),
                "initiation_pdf_xt",
                map_firing_and_speed_to_initiation_pdf_xt,
                method="firing_and_speed_to_initiation_pdf_xt_mean_field",
                optional_assumptions=_INITIATION_PDF_FORWARD_OPTIONS,
                default_assumptions=_INITIATION_PDF_FORWARD_DEFAULTS,
                source_kinds=(
                    ("initiation_rate", ("scalar", "x", "t", "xt")),
                    ("fork_speed", ("scalar", "x", "t", "xt")),
                ),
                description="Normalized probability density phi_i(x,t) from initiation rate and fork speed.",
            )
        if "firing_and_speed_to_initiation_events_mean_field" not in methods:
            self.register(
                ("initiation_rate", "fork_speed"),
                "initiation_events",
                map_firing_and_speed_to_initiation_events,
                method="firing_and_speed_to_initiation_events_mean_field",
                optional_assumptions=_INITIATION_FORWARD_OPTIONS,
                default_assumptions=_INITIATION_FORWARD_DEFAULTS,
                source_kinds=(
                    ("initiation_rate", ("scalar", "x", "t", "xt")),
                    ("fork_speed", ("scalar", "x", "t", "xt")),
                ),
                description="Time-integrated initiation-observation profile from initiation rate and fork speed.",
            )
        if "firing_and_fraction_to_initiation_events_xt_mean_field" not in methods:
            self.register(
                ("initiation_rate", "replicated_fraction_xt"),
                "initiation_events_xt",
                map_firing_and_fraction_to_initiation_events_xt,
                method="firing_and_fraction_to_initiation_events_xt_mean_field",
                optional_assumptions=_INITIATION_EVENT_OPTIONS,
                default_assumptions=_INITIATION_EVENT_DEFAULTS,
                source_kinds=(
                    ("initiation_rate", ("scalar", "x", "t", "xt")),
                    ("replicated_fraction_xt", ("xt",)),
                ),
                description="Mean-field observed initiation-event density I(x,t)[1-f(x,t)].",
            )
        if "firing_and_fraction_to_initiation_pdf_xt_normalized" not in methods:
            self.register(
                ("initiation_rate", "replicated_fraction_xt"),
                "initiation_pdf_xt",
                map_firing_and_fraction_to_initiation_pdf_xt,
                method="firing_and_fraction_to_initiation_pdf_xt_normalized",
                optional_assumptions=_INITIATION_PDF_FORWARD_OPTIONS,
                default_assumptions=_INITIATION_PDF_FORWARD_DEFAULTS,
                source_kinds=(
                    ("initiation_rate", ("scalar", "x", "t", "xt")),
                    ("replicated_fraction_xt", ("xt",)),
                ),
                description="Normalized probability density phi_i(x,t) from initiation rate and replicated fraction.",
            )
        if "firing_and_fraction_to_initiation_events_mean_field" not in methods:
            self.register(
                ("initiation_rate", "replicated_fraction_xt"),
                "initiation_events",
                map_firing_and_fraction_to_initiation_events,
                method="firing_and_fraction_to_initiation_events_mean_field",
                optional_assumptions=_INITIATION_EVENT_OPTIONS,
                default_assumptions=_INITIATION_EVENT_DEFAULTS,
                source_kinds=(
                    ("initiation_rate", ("scalar", "x", "t", "xt")),
                    ("replicated_fraction_xt", ("xt",)),
                ),
                description="Time-integrated initiation-observation profile from I(x,t) and f(x,t).",
            )
        if "firing_and_fraction_to_initiation_pdf_normalized" not in methods:
            self.register(
                ("initiation_rate", "replicated_fraction_xt"),
                "initiation_pdf",
                map_firing_and_fraction_to_initiation_pdf,
                method="firing_and_fraction_to_initiation_pdf_normalized",
                optional_assumptions=_INITIATION_PDF_FORWARD_OPTIONS,
                default_assumptions=_INITIATION_PDF_FORWARD_DEFAULTS,
                source_kinds=(
                    ("initiation_rate", ("scalar", "x", "t", "xt")),
                    ("replicated_fraction_xt", ("xt",)),
                ),
                description="Time-integrated normalized probability density phi_i(x).",
            )
        if "initiation_events_xt_to_initiation_events_time_integrated" not in methods:
            self.register(
                "initiation_events_xt",
                "initiation_events",
                map_initiation_events_xt_to_initiation_events,
                method="initiation_events_xt_to_initiation_events_time_integrated",
                optional_assumptions=_INITIATION_EVENT_OPTIONS,
                default_assumptions=_INITIATION_EVENT_DEFAULTS,
                source_kinds=(("initiation_events_xt", ("xt",)),),
                description="Time-integrated observed initiation-event density profile from I(x,t)[1-f(x,t)].",
            )
        if "initiation_events_xt_to_initiation_pdf_xt_normalized" not in methods:
            self.register(
                "initiation_events_xt",
                "initiation_pdf_xt",
                map_event_density_to_pdf_xt,
                method="initiation_events_xt_to_initiation_pdf_xt_normalized",
                optional_assumptions=_PDF_OPTIONS,
                default_assumptions=_PDF_DEFAULTS,
                source_kinds=(("initiation_events_xt", ("xt",)),),
                description="Normalized probability density phi_i(x,t) from the observed initiation-event density.",
            )
        if "initiation_events_xt_to_initiation_pdf_normalized" not in methods:
            self.register(
                "initiation_events_xt",
                "initiation_pdf",
                map_event_density_to_pdf,
                method="initiation_events_xt_to_initiation_pdf_normalized",
                optional_assumptions=_PDF_OPTIONS,
                default_assumptions=_PDF_DEFAULTS,
                source_kinds=(("initiation_events_xt", ("xt",)),),
                description="Time-integrated normalized probability density phi_i(x).",
            )
        if "initiation_pdf_xt_to_initiation_pdf_time_integrated" not in methods:
            self.register(
                "initiation_pdf_xt",
                "initiation_pdf",
                map_pdf_xt_to_pdf,
                method="initiation_pdf_xt_to_initiation_pdf_time_integrated",
                optional_assumptions=_INITIATION_EVENT_OPTIONS,
                default_assumptions=_INITIATION_EVENT_DEFAULTS,
                source_kinds=(("initiation_pdf_xt", ("xt",)),),
                description="Time-integrated normalized probability density phi_i(x).",
            )

    def _ensure_origin_efficiency_routes(self):
        methods = {spec.method for spec in self._specs}
        defaults = dict(_ORIGIN_EFFICIENCY_DEFAULTS)
        if "initiation_events_to_origin_efficiency_region_integral" not in methods:
            self.register(
                "initiation_events",
                "origin_efficiency",
                map_initiation_events_to_origin_efficiency,
                method="initiation_events_to_origin_efficiency_region_integral",
                required_assumptions=("regions",),
                optional_assumptions=_ORIGIN_EFFICIENCY_OPTIONS,
                default_assumptions=defaults,
                source_kinds=(("initiation_events", ("x",)),),
                description="Region-wise integral of the observed initiation-event density over user-defined origin or initiation-zone windows.",
            )
        if "initiation_events_xt_to_origin_efficiency_region_integral" not in methods:
            self.register(
                "initiation_events_xt",
                "origin_efficiency",
                map_initiation_events_to_origin_efficiency,
                method="initiation_events_xt_to_origin_efficiency_region_integral",
                required_assumptions=("regions",),
                optional_assumptions=_ORIGIN_EFFICIENCY_OPTIONS,
                default_assumptions=defaults,
                source_kinds=(("initiation_events_xt", ("xt",)),),
                description="Region-wise integral of I(x,t)[1-f(x,t)] over time and user-defined genomic windows.",
            )
        if "initiation_pdf_to_origin_efficiency_region_integral" not in methods:
            self.register(
                "initiation_pdf",
                "origin_efficiency",
                map_initiation_events_to_origin_efficiency,
                method="initiation_pdf_to_origin_efficiency_region_integral",
                required_assumptions=("regions",),
                optional_assumptions=_ORIGIN_EFFICIENCY_OPTIONS,
                default_assumptions=defaults,
                source_kinds=(("initiation_pdf", ("x",)),),
                description="Region-wise integral of the normalized initiation probability density over user-defined genomic windows.",
            )
        if "initiation_pdf_xt_to_origin_efficiency_region_integral" not in methods:
            self.register(
                "initiation_pdf_xt",
                "origin_efficiency",
                map_initiation_events_to_origin_efficiency,
                method="initiation_pdf_xt_to_origin_efficiency_region_integral",
                required_assumptions=("regions",),
                optional_assumptions=_ORIGIN_EFFICIENCY_OPTIONS,
                default_assumptions=defaults,
                source_kinds=(("initiation_pdf_xt", ("xt",)),),
                description="Region-wise integral of phi_i(x,t) over time and user-defined genomic windows.",
            )

    def _ensure_coalescence_event_routes(self):
        targets = {"coalescence_events", "coalescence_events_xt"}
        self._specs = [
            spec for spec in self._specs
            if not (
                spec.target in targets
                and spec.method.endswith("_simulation")
            )
        ]
        methods = {spec.method for spec in self._specs}
        forward_defaults = {
            "resolution": 1.0,
            "resolution_time": 5.0,
            "time_grid": None,
            "max_time": None,
            "max_rep_time": None,
            "n_bins": None,
            "genome_length": None,
            "interval_kb": None,
            "perQ": False,
            "cfl": 0.8,
            "unreplicated_floor": 1e-12,
        }

        if "fork_density_bundle_to_coalescence_events_xt_mean_field" not in methods:
            self.register(
                ("fork_densities_xt", "fork_speed"),
                "coalescence_events_xt",
                map_fork_density_bundle_to_coalescence_events_xt,
                method="fork_density_bundle_to_coalescence_events_xt_mean_field",
                optional_assumptions=_COALESCENCE_EVENT_OPTIONS,
                default_assumptions=_COALESCENCE_EVENT_DEFAULTS,
                source_kinds=(("fork_speed", ("scalar", "x", "t", "xt")),),
                description="Mean-field observed fork-coalescence-event density from directional fork densities.",
            )
        if "fork_density_bundle_to_coalescence_pdf_xt_mean_field" not in methods:
            self.register(
                ("fork_densities_xt", "fork_speed"),
                "coalescence_pdf_xt",
                map_fork_density_bundle_to_coalescence_pdf_xt,
                method="fork_density_bundle_to_coalescence_pdf_xt_mean_field",
                optional_assumptions=_COALESCENCE_PDF_OPTIONS,
                default_assumptions=_COALESCENCE_PDF_DEFAULTS,
                source_kinds=(("fork_speed", ("scalar", "x", "t", "xt")),),
                description="Normalized probability density phi_c(x,t) from directional fork densities.",
            )
        if "fork_density_bundle_to_coalescence_events_mean_field" not in methods:
            self.register(
                ("fork_densities_xt", "fork_speed"),
                "coalescence_events",
                map_fork_density_bundle_to_coalescence_events,
                method="fork_density_bundle_to_coalescence_events_mean_field",
                optional_assumptions=_COALESCENCE_EVENT_OPTIONS,
                default_assumptions=_COALESCENCE_EVENT_DEFAULTS,
                source_kinds=(("fork_speed", ("scalar", "x", "t", "xt")),),
                description="Time-integrated coalescence-observation profile from directional fork densities.",
            )

        if "fork_density_fields_to_coalescence_events_xt_mean_field" not in methods:
            self.register(
                (
                    "right_fork_density_xt",
                    "left_fork_density_xt",
                    "fork_speed",
                    "replicated_fraction_xt",
                ),
                "coalescence_events_xt",
                map_fork_density_fields_to_coalescence_events_xt,
                method="fork_density_fields_to_coalescence_events_xt_mean_field",
                optional_assumptions=_COALESCENCE_EVENT_OPTIONS,
                default_assumptions=_COALESCENCE_EVENT_DEFAULTS,
                source_kinds=(
                    ("right_fork_density_xt", ("xt",)),
                    ("left_fork_density_xt", ("xt",)),
                    ("fork_speed", ("scalar", "x", "t", "xt")),
                    ("replicated_fraction_xt", ("xt",)),
                ),
                description="Mean-field observed fork-coalescence-event density from explicit fork-density fields.",
            )
        if "fork_density_fields_to_coalescence_pdf_xt_mean_field" not in methods:
            self.register(
                (
                    "right_fork_density_xt",
                    "left_fork_density_xt",
                    "fork_speed",
                    "replicated_fraction_xt",
                ),
                "coalescence_pdf_xt",
                map_fork_density_fields_to_coalescence_pdf_xt,
                method="fork_density_fields_to_coalescence_pdf_xt_mean_field",
                optional_assumptions=_COALESCENCE_PDF_OPTIONS,
                default_assumptions=_COALESCENCE_PDF_DEFAULTS,
                source_kinds=(
                    ("right_fork_density_xt", ("xt",)),
                    ("left_fork_density_xt", ("xt",)),
                    ("fork_speed", ("scalar", "x", "t", "xt")),
                    ("replicated_fraction_xt", ("xt",)),
                ),
                description="Normalized probability density phi_c(x,t) from explicit fork-density fields.",
            )
        if "fork_density_fields_to_coalescence_events_mean_field" not in methods:
            self.register(
                (
                    "right_fork_density_xt",
                    "left_fork_density_xt",
                    "fork_speed",
                    "replicated_fraction_xt",
                ),
                "coalescence_events",
                map_fork_density_fields_to_coalescence_events,
                method="fork_density_fields_to_coalescence_events_mean_field",
                optional_assumptions=_COALESCENCE_EVENT_OPTIONS,
                default_assumptions=_COALESCENCE_EVENT_DEFAULTS,
                source_kinds=(
                    ("right_fork_density_xt", ("xt",)),
                    ("left_fork_density_xt", ("xt",)),
                    ("fork_speed", ("scalar", "x", "t", "xt")),
                    ("replicated_fraction_xt", ("xt",)),
                ),
                description="Time-integrated coalescence-observation profile from explicit fork-density fields.",
            )

        if "firing_and_speed_to_coalescence_events_xt_mean_field" not in methods:
            self.register(
                ("initiation_rate", "fork_speed"),
                "coalescence_events_xt",
                map_firing_and_speed_to_coalescence_events_xt,
                method="firing_and_speed_to_coalescence_events_xt_mean_field",
                optional_assumptions=_COALESCENCE_FORWARD_OPTIONS,
                default_assumptions=forward_defaults,
                source_kinds=(
                    ("initiation_rate", ("scalar", "x", "t", "xt")),
                    ("fork_speed", ("scalar", "x", "t", "xt")),
                ),
                description="Mean-field observed fork-coalescence-event density from initiation rate and fork speed.",
            )
        if "firing_and_speed_to_coalescence_pdf_xt_mean_field" not in methods:
            pdf_forward_defaults = dict(forward_defaults)
            pdf_forward_defaults.update(_PDF_DEFAULTS)
            self.register(
                ("initiation_rate", "fork_speed"),
                "coalescence_pdf_xt",
                map_firing_and_speed_to_coalescence_pdf_xt,
                method="firing_and_speed_to_coalescence_pdf_xt_mean_field",
                optional_assumptions=_COALESCENCE_PDF_FORWARD_OPTIONS,
                default_assumptions=pdf_forward_defaults,
                source_kinds=(
                    ("initiation_rate", ("scalar", "x", "t", "xt")),
                    ("fork_speed", ("scalar", "x", "t", "xt")),
                ),
                description="Normalized probability density phi_c(x,t) from initiation rate and fork speed.",
            )

        if "coalescence_events_xt_to_coalescence_events_time_integrated" not in methods:
            self.register(
                "coalescence_events_xt",
                "coalescence_events",
                map_coalescence_events_xt_to_coalescence_events,
                method="coalescence_events_xt_to_coalescence_events_time_integrated",
                optional_assumptions=_COALESCENCE_PROFILE_OPTIONS,
                default_assumptions=_COALESCENCE_PROFILE_DEFAULTS,
                source_kinds=(("coalescence_events_xt", ("xt",)),),
                description="Time-integrated observed fork-coalescence-event density profile from the coalescence sink term.",
            )
        if "coalescence_events_xt_to_coalescence_pdf_xt_normalized" not in methods:
            self.register(
                "coalescence_events_xt",
                "coalescence_pdf_xt",
                map_event_density_to_pdf_xt,
                method="coalescence_events_xt_to_coalescence_pdf_xt_normalized",
                optional_assumptions=_PDF_OPTIONS,
                default_assumptions=_PDF_DEFAULTS,
                source_kinds=(("coalescence_events_xt", ("xt",)),),
                description="Normalized probability density phi_c(x,t) from the observed fork-coalescence-event density.",
            )
        if "coalescence_events_xt_to_coalescence_pdf_normalized" not in methods:
            self.register(
                "coalescence_events_xt",
                "coalescence_pdf",
                map_event_density_to_pdf,
                method="coalescence_events_xt_to_coalescence_pdf_normalized",
                optional_assumptions=_PDF_OPTIONS,
                default_assumptions=_PDF_DEFAULTS,
                source_kinds=(("coalescence_events_xt", ("xt",)),),
                description="Time-integrated normalized probability density phi_c(x).",
            )
        if "coalescence_pdf_xt_to_coalescence_pdf_time_integrated" not in methods:
            self.register(
                "coalescence_pdf_xt",
                "coalescence_pdf",
                map_pdf_xt_to_pdf,
                method="coalescence_pdf_xt_to_coalescence_pdf_time_integrated",
                optional_assumptions=_COALESCENCE_PROFILE_OPTIONS,
                default_assumptions=_COALESCENCE_PROFILE_DEFAULTS,
                source_kinds=(("coalescence_pdf_xt", ("xt",)),),
                description="Time-integrated normalized probability density phi_c(x).",
            )

        if "firing_and_speed_to_coalescence_events_mean_field" not in methods:
            self.register(
                ("initiation_rate", "fork_speed"),
                "coalescence_events",
                map_firing_and_speed_to_coalescence_events,
                method="firing_and_speed_to_coalescence_events_mean_field",
                optional_assumptions=_COALESCENCE_FORWARD_OPTIONS,
                default_assumptions=forward_defaults,
                source_kinds=(
                    ("initiation_rate", ("scalar", "x", "t", "xt")),
                    ("fork_speed", ("scalar", "x", "t", "xt")),
                ),
                description="Time-integrated coalescence-observation profile from initiation rate and fork speed.",
            )
        if "firing_and_speed_to_coalescence_pdf_mean_field" not in methods:
            pdf_forward_defaults = dict(forward_defaults)
            pdf_forward_defaults.update(_PDF_DEFAULTS)
            self.register(
                ("initiation_rate", "fork_speed"),
                "coalescence_pdf",
                map_firing_and_speed_to_coalescence_pdf,
                method="firing_and_speed_to_coalescence_pdf_mean_field",
                optional_assumptions=_COALESCENCE_PDF_FORWARD_OPTIONS,
                default_assumptions=pdf_forward_defaults,
                source_kinds=(
                    ("initiation_rate", ("scalar", "x", "t", "xt")),
                    ("fork_speed", ("scalar", "x", "t", "xt")),
                ),
                description="Time-integrated normalized probability density phi_c(x).",
            )

    def _ensure_fork_density_routes(self):
        targets = {
            "fork_densities_xt",
            "right_fork_density_xt",
            "left_fork_density_xt",
        }
        self._specs = [
            spec for spec in self._specs
            if not (
                spec.target in targets
                and spec.method.endswith("_simulation")
            )
        ]
        methods = {spec.method for spec in self._specs}
        defaults = {
            "resolution": 1.0,
            "resolution_time": 5.0,
            "time_grid": None,
            "max_time": None,
            "max_rep_time": None,
            "n_bins": None,
            "genome_length": None,
            "interval_kb": None,
            "perQ": False,
            "cfl": 0.8,
        }
        routes = (
            (
                "fork_densities_xt",
                map_firing_and_speed_to_fork_densities_xt,
                "firing_and_speed_to_directional_fork_densities_xt_mean_field",
                "Mean-field map of right- and left-moving fork-density fields.",
            ),
            (
                "right_fork_density_xt",
                map_firing_and_speed_to_right_fork_density_xt,
                "firing_and_speed_to_right_fork_density_xt_mean_field",
                "Mean-field map of the right-moving fork-density field.",
            ),
            (
                "left_fork_density_xt",
                map_firing_and_speed_to_left_fork_density_xt,
                "firing_and_speed_to_left_fork_density_xt_mean_field",
                "Mean-field map of the left-moving fork-density field.",
            ),
        )
        for target, func, method, description in routes:
            if method in methods:
                continue
            self.register(
                ("initiation_rate", "fork_speed"),
                target,
                func,
                method=method,
                optional_assumptions=_FORK_DENSITY_OPTIONS,
                default_assumptions=defaults,
                source_kinds=(
                    ("initiation_rate", ("scalar", "x", "t", "xt")),
                    ("fork_speed", ("scalar", "x", "t", "xt")),
                ),
                description=description,
            )

    def _candidate_specs(self, source_names, target, method):
        from ._terminology import canonical_rate_key
        method = canonical_rate_key(method)
        target = _canonical_feature(target)
        source_set = {_canonical_feature(name) for name in source_names}
        if target == "replication_timing" and "initiation_rate" in source_set:
            self._ensure_forward_timing_routes()
        if target == "fork_speed":
            self._ensure_fork_speed_shape_routes()
        if target == "fork_directionality":
            self._ensure_directionality_routes()
        if target in {"replicated_fraction", "replicated_fraction_xt"}:
            self._ensure_replicated_fraction_routes()
        if target in {"initiation_events", "initiation_events_xt"}:
            self._ensure_initiation_event_routes()
        if target in {"initiation_pdf", "initiation_pdf_xt"}:
            self._ensure_initiation_event_routes()
        if target == "origin_efficiency":
            self._ensure_origin_efficiency_routes()
        if target in {"coalescence_events", "coalescence_events_xt"}:
            self._ensure_coalescence_event_routes()
        if target in {"coalescence_pdf", "coalescence_pdf_xt"}:
            self._ensure_coalescence_event_routes()
        if target in {
            "fork_densities_xt",
            "right_fork_density_xt",
            "left_fork_density_xt",
        }:
            self._ensure_fork_density_routes()
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
        # A non-zero (or invalid, so it can be validated downstream) stalling
        # input must not be silently discarded by an earlier deterministic
        # route. Zero remains equivalent to no stalling and therefore keeps the
        # existing fast automatic route selection.
        stalling_requested = False
        for name in ("stall_rate_per_kb", "stall_rate"):
            if name not in assumptions or assumptions[name] is None:
                continue
            try:
                value = float(assumptions[name])
                stalling_requested = not np.isfinite(value) or value != 0.0
            except (TypeError, ValueError):
                stalling_requested = True
            if stalling_requested:
                break
        if kind_matches and stalling_requested:
            stall_matches = [
                spec
                for spec in kind_matches
                if {"stall_rate", "stall_rate_per_kb"}
                & set(spec.required_assumptions + spec.optional_assumptions)
            ]
            if stall_matches:
                kind_matches = stall_matches
            else:
                route_label = (
                    f"method '{method}'"
                    if method != "auto"
                    else "the requested source and target"
                )
                raise ValueError(
                    f"No mapping route for {route_label} supports fork stalling; "
                    "choose a stochastic simulation route or set the stall rate to zero."
                )
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
        with mapping_progress(assumptions):
            values = spec.func(payload, **self._function_kwargs(spec, assumptions))

        if return_info:
            info = {
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
            rmap_info = getattr(values, "_rmap_info", None)
            if rmap_info:
                info.update(rmap_info)
            return info
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

    def targets(self, target=None, source=None, implemented=None):
        """Return a compact catalogue of mapping targets.

        By default the table shows implemented core mapping targets. Use
        ``requirements`` for the source data and assumptions needed for a
        specific target. Parameters can narrow or expand the catalogue, e.g.
        ``rmap.targets(target="initiation_rate")`` or
        ``rmap.targets(target=("replicated_fraction", "fork_density"))``.
        """
        if implemented is None:
            implemented = True
        target_names = _PUBLIC_MAPPING_TARGETS if target is None else target
        target_names = target_names if isinstance(target_names, (tuple, list, set)) else (target_names,)
        target_order = tuple(_canonical_feature(name) for name in target_names)
        columns = ("target", "description")
        filtered = self._filtered_specs(
            target=target_order if target is not None else None,
            source=source,
            implemented=implemented,
        )
        available_targets = {spec.target for spec in filtered}
        if target is None:
            extra_targets = tuple(sorted(available_targets.difference(target_order)))
            target_order = target_order + extra_targets
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

    def available(self, target=None, source=None, implemented=None):
        """Backward-compatible alias for ``rmap.targets()``."""
        return self.targets(target=target, source=source, implemented=implemented)

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
    ("initiation_rate", "fork_speed"),
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
        "resolution_time", "time_grid", "max_time", "max_rep_time",
    ),
    source_kinds=(
        ("initiation_rate", ("scalar", "x", "t", "xt")),
        ("fork_speed", ("scalar", "x", "t", "xt")),
    ),
    description="Deterministic timing from constant, spatial, temporal or space-time kinetics.",
)
rmap.register(
    ("initiation_rate", "fork_speed"),
    "replication_timing",
    _simulation_source_feature_mapper("replication_timing"),
    method="firing_and_speed_to_timing_simulation",
    optional_assumptions=_RSIM_ASSUMPTION_NAMES,
    default_assumptions={"verbose": False},
    source_kinds=(
        ("initiation_rate", ("scalar", "x", "t", "xt")),
        ("fork_speed", ("scalar", "x", "t", "xt")),
    ),
    description="Simulation-backed timing map for kinetic fields with time dependence.",
)
rmap.register(
    ("replication_timing", "fork_speed"),
    "initiation_rate",
    map_timing_and_speed_to_constant_firing,
    method="timing_and_speed_to_constant_initiation_rate",
    required_assumptions=("target_form",),
    optional_assumptions=(
        "resolution",
        "neighbourhood_range",
        "fit_step",
        "maxiter",
        "error",
        "err_threshold",
        "initial_initiation_rate",
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
    "initiation_rate",
    map_timing_and_speed_to_firing,
    method="timing_and_speed_to_initiation_rate",
    optional_assumptions=(
        "resolution",
        "neighbourhood_range",
        "fit_step",
        "maxiter",
        "error",
        "sim_number",
        "err_threshold",
        "initial_initiation_rate",
        "deviation_threshold",
        "rate_bounds",
        "space_fit_step",
        "space_maxiter",
        "verbose",
        "print_every",
        "perQ",
        "refine_edge_kb",
        "stall_rate",
        "stall_rate_per_kb",
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
    "initiation_rate",
    map_timing_and_speed_to_constant_firing,
    method="timing_and_variable_speed_to_initiation_rate",
    required_assumptions=("target_form",),
    optional_assumptions=(
        "resolution",
        "resolution_time",
        "time_grid",
        "neighbourhood_range",
        "fit_step",
        "maxiter",
        "error",
        "sim_number",
        "err_threshold",
        "initial_initiation_rate",
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
    "initiation_rate",
    map_timing_and_speed_to_firing,
    method="timing_and_variable_speed_to_spatial_initiation_rate",
    optional_assumptions=(
        "resolution",
        "resolution_time",
        "time_grid",
        "neighbourhood_range",
        "fit_step",
        "maxiter",
        "error",
        "sim_number",
        "err_threshold",
        "initial_initiation_rate",
        "deviation_threshold",
        "rate_bounds",
        "space_fit_step",
        "space_maxiter",
        "verbose",
        "print_every",
        "perQ",
        "refine_edge_kb",
        "refine_seed",
        "stall_rate",
        "stall_rate_per_kb",
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
# Static firing and spatial speed retain the inexpensive local ratio fitter.
# Its public firing density is converted to a per-bin hazard in the wrapper.
rmap.register(
    ("replication_timing", "initiation_rate"),
    "fork_speed",
    map_timing_and_firing_to_spatial_speed,
    method="timing_and_firing_to_spatial_fork_speed",
    optional_assumptions=(
        "resolution", "T0", "neighbourhood_range", "fit_step", "maxiter",
        "speed_bounds", "initial_fork_speed", "smoothw", "refine_edge_kb", "error", "perQ",
        "n_bins", "verbose", "print_every", "target_form", "target_shape",
    ),
    default_assumptions={"verbose": False, "target_form": "space", "target_shape": "best"},
    source_kinds=(("replication_timing", ("x",)), ("initiation_rate", ("scalar", "x"))),
    assumption_values=(("target_form", ("space",)), ("target_shape", ("best",))),
    description="Iterative spatial fork-speed fitting from timing and static initiation rate.",
)
# Constant speed and spatial speed with time-varying firing retain the inverse.
_SPEED_INVERSE_OPTIONS = (
    "resolution", "resolution_time", "time_grid", "T0", "initial_fork_speed",
    "speed_bounds", "maxiter", "tolerance", "perQ", "verbose",
    "target_form", "target_shape",
)
for _speed_method, _speed_form, _firing_kinds in (
    ("timing_and_firing_to_constant_fork_speed", "constant", ("scalar", "x")),
    ("timing_and_time_firing_to_constant_fork_speed", "constant", ("t", "xt")),
    ("timing_and_time_firing_to_spatial_fork_speed", "space", ("t", "xt")),
):
    _speed_knots = ()
    if _speed_form in {"space", "spacetime"}:
        _speed_knots += ("spatial_knots",)
    if _speed_form in {"time", "spacetime"}:
        _speed_knots += ("temporal_knots",)
    if _speed_form != "constant":
        _speed_knots += ("regularization",)
    rmap.register(
        ("replication_timing", "initiation_rate"),
        "fork_speed",
        map_timing_and_firing_to_speed,
        method=_speed_method,
        required_assumptions=(
            ("target_form",)
            if _speed_method == "timing_and_time_firing_to_constant_fork_speed" else ()
        ),
        optional_assumptions=_SPEED_INVERSE_OPTIONS + _speed_knots,
        default_assumptions={
            "verbose": False, "target_shape": "best",
            **({} if _speed_method == "timing_and_time_firing_to_constant_fork_speed"
               else {"target_form": _speed_form}),
        },
        source_kinds=(
            ("replication_timing", ("x",)),
            ("initiation_rate", _firing_kinds),
        ),
        assumption_values=(("target_form", (_speed_form,)), ("target_shape", ("best",))),
        description=(
            f"Bounded {_speed_form} fork-speed inference from timing and supplied "
            "initiation rate using the mean-field replication model."
        ),
    )
# Restore the earlier temporal and alternating spatial/temporal fits. Their
# deterministic forward models consume density-valued firing directly; only
# legacy spatial seeds convert to per-bin hazards inside the fitter.
_RESTORED_SPEED_OPTIONS = (
    "resolution", "resolution_time", "time_grid", "T0", "initial_fork_speed",
    "speed_bounds", "temporal_fit_step", "temporal_maxiter", "refine_edge_kb",
    "error", "perQ", "verbose", "print_every", "refine_verbose", "n_bins",
    "target_form", "target_shape", "fit_step", "maxiter",
)
for _speed_method, _speed_func, _speed_form, _firing_kinds in (
    ("timing_and_firing_to_time_fork_speed", map_timing_and_firing_to_general_speed,
     "time", ("scalar", "x")),
    ("timing_and_time_firing_to_time_fork_speed", map_timing_and_time_firing_to_temporal_speed,
     "time", ("t", "xt")),
    ("timing_and_firing_to_spacetime_fork_speed", map_timing_and_firing_to_general_speed,
     "spacetime", ("xt",)),
    ("timing_and_general_firing_to_spacetime_fork_speed", map_timing_and_general_firing_to_spacetime_speed,
     "spacetime", ("scalar", "x", "t")),
):
    _is_spacetime = _speed_form == "spacetime"
    rmap.register(
        ("replication_timing", "initiation_rate"), "fork_speed", _speed_func,
        method=_speed_method,
        optional_assumptions=_RESTORED_SPEED_OPTIONS + (
            ("space_fit_step", "space_maxiter", "smoothw", "refine_ratio_bounds",
             "neighbourhood_range", "spacetime_order") if _is_spacetime else ()
        ),
        default_assumptions={
            "verbose": False, "target_form": _speed_form, "target_shape": "best",
            "temporal_fit_step": .003, "temporal_maxiter": 10,
            **({"space_fit_step": .4, "space_maxiter": 10, "spacetime_order": "crossed"}
               if _is_spacetime else {}),
        },
        source_kinds=(("replication_timing", ("x",)), ("initiation_rate", _firing_kinds)),
        assumption_values=(
            ("target_form", (_speed_form,)), ("target_shape", ("best",)),
        ) + ((("spacetime_order", ("crossed", "space_then_time", "time_then_space")),)
             if _is_spacetime else ()),
        description=(
            "Alternating spatial and temporal fork-speed fitting from timing and supplied firing."
            if _is_spacetime else
            "Compact temporal fork-speed fitting from timing and supplied firing."
        ),
    )
rmap.register(
    ("replication_timing", "fork_directionality"),
    "fork_speed",
    map_timing_and_directionality_to_fork_speed,
    method="timing_and_directionality_to_constant_fork_speed",
    optional_assumptions=(
        "resolution",
        "T0",
        "smoothw",
        "speed_bounds",
        "perQ",
        "target_form",
        "target_shape",
    ),
    default_assumptions={
        "T0": None,
        "target_form": "constant",
        "target_shape": "best",
    },
    source_kinds=(
        ("replication_timing", ("x",)),
        ("fork_directionality", ("x",)),
    ),
    assumption_values=(("target_form", ("constant",)), ("target_shape", ("best",))),
    description=(
        "Constant fork-speed reconstruction from replication timing and fork "
        "directionality."
    ),
)
rmap.register(
    ("replication_timing", "fork_directionality"),
    "fork_speed",
    map_timing_and_directionality_to_fork_speed,
    method="timing_and_directionality_to_spatial_fork_speed",
    optional_assumptions=(
        "resolution",
        "T0",
        "smoothw",
        "speed_bounds",
        "speed_regularization",
        "perQ",
        "target_form",
        "target_shape",
    ),
    default_assumptions={
        "T0": None,
        "target_form": "space",
        "target_shape": "best",
        "speed_regularization": 4.0,
    },
    source_kinds=(
        ("replication_timing", ("x",)),
        ("fork_directionality", ("x",)),
    ),
    assumption_values=(("target_form", ("space",)), ("target_shape", ("best",))),
    description=(
        "Regularized spatial fork-speed reconstruction from replication timing "
        "and fork directionality."
    ),
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
    description="Gradient-based fork-directionality mapping with constant fork speed.",
)
rmap.register(
    ("replication_timing", "fork_speed"),
    "fork_directionality",
    map_timing_and_spatial_speed_to_directionality,
    method="timing_and_spatial_speed_to_directionality",
    optional_assumptions=("resolution", "T0", "smoothw", "perQ"),
    source_kinds=(
        ("replication_timing", ("x",)),
        ("fork_speed", ("x",)),
    ),
    description="Gradient-based fork-directionality mapping with spatial fork speed.",
)
rmap.register(
    ("fork_directionality", "fork_speed"),
    "replication_timing",
    map_directionality_and_speed_to_timing,
    method="directionality_and_speed_to_timing",
    optional_assumptions=("resolution", "T0", "smoothw", "perQ"),
    source_kinds=(
        ("fork_directionality", ("x",)),
        ("fork_speed", ("scalar", "x")),
    ),
    description="Integration of fork directionality to timing with constant or spatial fork speed.",
)
rmap.register(
    "initiation_rate",
    "replication_timing",
    map_firing_density_to_timing,
    method="firing_to_timing_deterministic",
    required_assumptions=("fork_speed",),
    optional_assumptions=(
        "resolution",
        "neighbourhood_range",
        "perQ",
        "n_bins",
        "genome_length",
        "interval_kb",
        "resolution_time", "time_grid", "max_time", "max_rep_time",
    ),
    default_assumptions={"fork_speed": 1.4},
    source_kinds=(("initiation_rate", ("scalar", "x", "t", "xt")),),
    assumption_kinds=(("fork_speed", ("scalar", "x", "t", "xt")),),
    description="Deterministic timing from constant, spatial, temporal or space-time kinetics.",
)
rmap.register(
    "initiation_rate",
    "replication_timing",
    _simulation_feature_mapper("replication_timing"),
    method="firing_to_timing_simulation",
    required_assumptions=("fork_speed",),
    optional_assumptions=_RSIM_OPTION_NAMES,
    default_assumptions={"fork_speed": 1.4, "verbose": False},
    source_kinds=(("initiation_rate", ("scalar", "x", "t", "xt")),),
    assumption_kinds=(("fork_speed", ("scalar", "x", "t", "xt")),),
    description="Simulation-backed timing map for kinetic fields with time dependence.",
)
rmap.register(
    "replication_timing",
    "initiation_rate",
    map_timing_to_constant_firing_density,
    method="timing_to_constant_initiation_rate_constant_fork_speed",
    required_assumptions=("fork_speed", "target_form"),
    optional_assumptions=(
        "resolution",
        "neighbourhood_range",
        "fit_step",
        "maxiter",
        "error",
        "err_threshold",
        "initial_initiation_rate",
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
    "initiation_rate",
    map_timing_to_spatial_firing_density,
    method="timing_to_firing_constant_fork_speed",
    required_assumptions=("fork_speed",),
    optional_assumptions=(
        "resolution",
        "neighbourhood_range",
        "fit_step",
        "maxiter",
        "error",
        "sim_number",
        "err_threshold",
        "initial_initiation_rate",
        "deviation_threshold",
        "rate_bounds",
        "space_fit_step",
        "space_maxiter",
        "verbose",
        "print_every",
        "perQ",
        "stall_rate",
        "stall_rate_per_kb",
        "tau",
        "target_form",
    ),
    default_assumptions={"fork_speed": 1.4, "verbose": True, "target_form": "space"},
    source_kinds=(("replication_timing", ("x",)),),
    assumption_kinds=(("fork_speed", ("scalar",)),),
    assumption_values=(("target_form", ("space",)),),
    description="Iterative fit of a space-dependent initiation rate with constant fork speed.",
)
rmap.register(
    "replication_timing",
    "initiation_rate",
    map_timing_to_constant_firing_density,
    method="timing_to_constant_initiation_rate_variable_fork_speed",
    required_assumptions=("fork_speed", "target_form"),
    optional_assumptions=(
        "resolution",
        "resolution_time",
        "time_grid",
        "neighbourhood_range",
        "fit_step",
        "maxiter",
        "error",
        "sim_number",
        "err_threshold",
        "initial_initiation_rate",
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
    "initiation_rate",
    map_timing_to_spatial_firing_density,
    method="timing_to_spatial_initiation_rate_variable_fork_speed",
    required_assumptions=("fork_speed",),
    optional_assumptions=(
        "resolution",
        "resolution_time",
        "time_grid",
        "neighbourhood_range",
        "fit_step",
        "maxiter",
        "error",
        "sim_number",
        "err_threshold",
        "initial_initiation_rate",
        "deviation_threshold",
        "rate_bounds",
        "space_fit_step",
        "space_maxiter",
        "verbose",
        "print_every",
        "perQ",
        "refine_seed",
        "stall_rate",
        "stall_rate_per_kb",
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
    "initiation_rate",
    map_timing_and_speed_to_general_firing,
    method="timing_and_speed_to_time_initiation_rate",
    optional_assumptions=(
        "resolution",
        "resolution_time",
        "neighbourhood_range",
        "fit_step",
        "maxiter",
        "error",
        "sim_number",
        "err_threshold",
        "initial_initiation_rate",
        "deviation_threshold",
        "rate_bounds",
        "space_fit_step",
        "space_maxiter",
        "verbose",
        "print_every",
        "perQ",
        "stall_rate",
        "stall_rate_per_kb",
        "tau",
        "target_form",
        "target_shape",
        "weibull_shape",
        "weibull_delay",
        "fit_weibull_shape",
        "alpha_smooth",
        "alpha_prior",
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
    default_assumptions={
        "verbose": True,
        "target_form": "time",
        "target_shape": "weibull",
        "weibull_shape": 2.0,
        "weibull_delay": 0.0,
        "fit_weibull_shape": False,
        "alpha_smooth": 1e-2,
        "alpha_prior": 0.0,
    },
    source_kinds=(
        ("replication_timing", ("x",)),
        ("fork_speed", ("scalar", "x", "t", "xt")),
    ),
    assumption_values=(("target_form", ("time",)), ("target_shape", ("free", "weibull"))),
    description="Time-only initiation-rate target; defaults to a Weibull firing-time model.",
)
rmap.register(
    ("replication_timing", "fork_speed"),
    "initiation_rate",
    map_timing_and_speed_to_general_firing,
    method="timing_and_speed_to_spacetime_initiation_rate",
    optional_assumptions=(
        "resolution",
        "resolution_time",
        "neighbourhood_range",
        "fit_step",
        "maxiter",
        "error",
        "sim_number",
        "err_threshold",
        "initial_initiation_rate",
        "deviation_threshold",
        "rate_bounds",
        "space_fit_step",
        "space_maxiter",
        "verbose",
        "print_every",
        "perQ",
        "stall_rate",
        "stall_rate_per_kb",
        "tau",
        "target_form",
        "target_shape",
        "weibull_shape",
        "weibull_delay",
        "fit_weibull_shape",
        "alpha_smooth",
        "alpha_prior",
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
    default_assumptions={
        "verbose": True,
        "target_form": "spacetime",
        "target_shape": "weibull",
        "weibull_shape": 2.0,
        "weibull_delay": 0.0,
        "fit_weibull_shape": False,
        "alpha_smooth": 1e-2,
        "alpha_prior": 0.0,
    },
    source_kinds=(
        ("replication_timing", ("x",)),
        ("fork_speed", ("scalar", "x", "t", "xt")),
    ),
    assumption_values=(("target_form", ("spacetime",)), ("target_shape", ("free", "weibull"))),
    description="Space-time initiation-rate target; defaults to a Weibull firing-time model.",
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
    description="Gradient-based fork-directionality mapping with constant fork speed.",
)
rmap.register(
    "replication_timing",
    "fork_directionality",
    map_timing_directionality,
    method="timing_to_directionality_spatial_speed",
    required_assumptions=("fork_speed",),
    optional_assumptions=("resolution", "T0", "smoothw", "perQ"),
    source_kinds=(("replication_timing", ("x",)),),
    assumption_kinds=(("fork_speed", ("x",)),),
    description="Gradient-based fork-directionality mapping with spatial fork speed.",
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
    assumption_kinds=(("fork_speed", ("scalar", "x")),),
    description="Integration of fork directionality to timing with constant or spatial fork speed.",
)

rmap.register(
    "fork_densities_xt",
    "fork_directionality",
    map_fork_density_bundle_to_directionality,
    method="fork_density_bundle_to_directionality_time_integrated",
    optional_assumptions=_FORK_DENSITY_DIRECTIONALITY_OPTIONS,
    default_assumptions=_FORK_DENSITY_DIRECTIONALITY_DEFAULTS,
    description="Time-integrated fork directionality from right- and left-moving fork-density fields.",
)
rmap.register(
    ("right_fork_density_xt", "left_fork_density_xt"),
    "fork_directionality",
    map_fork_density_fields_to_directionality,
    method="fork_density_fields_to_directionality_time_integrated",
    optional_assumptions=_FORK_DENSITY_DIRECTIONALITY_OPTIONS,
    default_assumptions=_FORK_DENSITY_DIRECTIONALITY_DEFAULTS,
    source_kinds=(
        ("right_fork_density_xt", ("xt",)),
        ("left_fork_density_xt", ("xt",)),
    ),
    description="Time-integrated fork directionality from right- and left-moving fork-density fields.",
)
rmap.register(
    ("right_fork_density", "left_fork_density"),
    "fork_directionality",
    map_fork_density_profiles_to_directionality,
    method="fork_density_profiles_to_directionality",
    optional_assumptions=_FORK_DENSITY_DIRECTIONALITY_OPTIONS,
    default_assumptions=_FORK_DENSITY_DIRECTIONALITY_DEFAULTS,
    source_kinds=(
        ("right_fork_density", ("x",)),
        ("left_fork_density", ("x",)),
    ),
    description="Fork directionality from time-integrated right- and left-moving fork-density profiles.",
)

rmap.register(
    ("initiation_rate", "fork_speed"),
    "replicated_fraction_xt",
    map_firing_and_speed_to_replicated_fraction_xt,
    method="firing_and_speed_to_replicated_fraction_xt_lightcone",
    optional_assumptions=_REPLICATED_FRACTION_OPTIONS,
    source_kinds=(
        ("initiation_rate", ("scalar", "x", "t", "xt")),
        ("fork_speed", ("scalar", "x", "t", "xt")),
    ),
    description="Analytical light-cone map of f(x,t) from initiation rate and fork speed.",
)
rmap.register(
    "initiation_rate",
    "replicated_fraction_xt",
    map_firing_to_replicated_fraction_xt,
    method="firing_to_replicated_fraction_xt_lightcone",
    required_assumptions=("fork_speed",),
    optional_assumptions=_REPLICATED_FRACTION_OPTIONS,
    default_assumptions={"fork_speed": 1.4},
    source_kinds=(("initiation_rate", ("scalar", "x", "t", "xt")),),
    assumption_kinds=(("fork_speed", ("scalar", "x", "t", "xt")),),
    description="Analytical light-cone map of f(x,t) from initiation rate and fork speed.",
)
rmap.register(
    ("initiation_rate", "fork_speed"),
    "replicated_fraction",
    map_firing_and_speed_to_replicated_fraction,
    method="firing_and_speed_to_replicated_fraction_lightcone",
    optional_assumptions=_REPLICATED_FRACTION_OPTIONS,
    source_kinds=(
        ("initiation_rate", ("scalar", "x", "t", "xt")),
        ("fork_speed", ("scalar", "x", "t", "xt")),
    ),
    description="Analytical light-cone map of the spatial mean replicated fraction.",
)
rmap.register(
    "initiation_rate",
    "replicated_fraction",
    map_firing_to_replicated_fraction,
    method="firing_to_replicated_fraction_lightcone",
    required_assumptions=("fork_speed",),
    optional_assumptions=_REPLICATED_FRACTION_OPTIONS,
    default_assumptions={"fork_speed": 1.4},
    source_kinds=(("initiation_rate", ("scalar", "x", "t", "xt")),),
    assumption_kinds=(("fork_speed", ("scalar", "x", "t", "xt")),),
    description="Analytical light-cone map of the spatial mean replicated fraction.",
)

rmap.register(
    ("initiation_rate", "fork_speed"),
    "initiation_events_xt",
    map_firing_and_speed_to_initiation_events_xt,
    method="firing_and_speed_to_initiation_events_xt_mean_field",
    optional_assumptions=_INITIATION_FORWARD_OPTIONS,
    default_assumptions=_INITIATION_FORWARD_DEFAULTS,
    source_kinds=(
        ("initiation_rate", ("scalar", "x", "t", "xt")),
        ("fork_speed", ("scalar", "x", "t", "xt")),
    ),
    description="Mean-field observed initiation-event density I(x,t)[1-f(x,t)] from initiation rate and fork speed.",
)
rmap.register(
    ("initiation_rate", "fork_speed"),
    "initiation_pdf_xt",
    map_firing_and_speed_to_initiation_pdf_xt,
    method="firing_and_speed_to_initiation_pdf_xt_mean_field",
    optional_assumptions=_INITIATION_PDF_FORWARD_OPTIONS,
    default_assumptions=_INITIATION_PDF_FORWARD_DEFAULTS,
    source_kinds=(
        ("initiation_rate", ("scalar", "x", "t", "xt")),
        ("fork_speed", ("scalar", "x", "t", "xt")),
    ),
    description="Normalized probability density phi_i(x,t) from initiation rate and fork speed.",
)
rmap.register(
    ("initiation_rate", "fork_speed"),
    "initiation_events",
    map_firing_and_speed_to_initiation_events,
    method="firing_and_speed_to_initiation_events_mean_field",
    optional_assumptions=_INITIATION_FORWARD_OPTIONS,
    default_assumptions=_INITIATION_FORWARD_DEFAULTS,
    source_kinds=(
        ("initiation_rate", ("scalar", "x", "t", "xt")),
        ("fork_speed", ("scalar", "x", "t", "xt")),
    ),
    description="Time-integrated initiation-observation profile from initiation rate and fork speed.",
)
rmap.register(
    ("initiation_rate", "fork_speed"),
    "initiation_pdf",
    map_firing_and_speed_to_initiation_pdf,
    method="firing_and_speed_to_initiation_pdf_mean_field",
    optional_assumptions=_INITIATION_PDF_FORWARD_OPTIONS,
    default_assumptions=_INITIATION_PDF_FORWARD_DEFAULTS,
    source_kinds=(
        ("initiation_rate", ("scalar", "x", "t", "xt")),
        ("fork_speed", ("scalar", "x", "t", "xt")),
    ),
    description="Time-integrated normalized initiation probability density from initiation rate and fork speed.",
)

rmap.register(
    ("initiation_rate", "replicated_fraction_xt"),
    "initiation_events_xt",
    map_firing_and_fraction_to_initiation_events_xt,
    method="firing_and_fraction_to_initiation_events_xt_mean_field",
    optional_assumptions=_INITIATION_EVENT_OPTIONS,
    default_assumptions=_INITIATION_EVENT_DEFAULTS,
    source_kinds=(
        ("initiation_rate", ("scalar", "x", "t", "xt")),
        ("replicated_fraction_xt", ("xt",)),
    ),
    description="Mean-field observed initiation-event density I(x,t)[1-f(x,t)].",
)
rmap.register(
    ("initiation_rate", "replicated_fraction_xt"),
    "initiation_pdf_xt",
    map_firing_and_fraction_to_initiation_pdf_xt,
    method="firing_and_fraction_to_initiation_pdf_xt_normalized",
    optional_assumptions=_INITIATION_PDF_FORWARD_OPTIONS,
    default_assumptions=_INITIATION_PDF_FORWARD_DEFAULTS,
    source_kinds=(
        ("initiation_rate", ("scalar", "x", "t", "xt")),
        ("replicated_fraction_xt", ("xt",)),
    ),
    description="Normalized probability density phi_i(x,t) from initiation rate and replicated fraction.",
)
rmap.register(
    ("initiation_rate", "replicated_fraction_xt"),
    "initiation_events",
    map_firing_and_fraction_to_initiation_events,
    method="firing_and_fraction_to_initiation_events_mean_field",
    optional_assumptions=_INITIATION_EVENT_OPTIONS,
    default_assumptions=_INITIATION_EVENT_DEFAULTS,
    source_kinds=(
        ("initiation_rate", ("scalar", "x", "t", "xt")),
        ("replicated_fraction_xt", ("xt",)),
    ),
    description="Time-integrated initiation-observation profile from I(x,t) and f(x,t).",
)
rmap.register(
    ("initiation_rate", "replicated_fraction_xt"),
    "initiation_pdf",
    map_firing_and_fraction_to_initiation_pdf,
    method="firing_and_fraction_to_initiation_pdf_normalized",
    optional_assumptions=_INITIATION_PDF_FORWARD_OPTIONS,
    default_assumptions=_INITIATION_PDF_FORWARD_DEFAULTS,
    source_kinds=(
        ("initiation_rate", ("scalar", "x", "t", "xt")),
        ("replicated_fraction_xt", ("xt",)),
    ),
    description="Time-integrated normalized probability density phi_i(x).",
)
rmap.register(
    "initiation_events_xt",
    "initiation_events",
    map_initiation_events_xt_to_initiation_events,
    method="initiation_events_xt_to_initiation_events_time_integrated",
    optional_assumptions=_INITIATION_EVENT_OPTIONS,
    default_assumptions=_INITIATION_EVENT_DEFAULTS,
    source_kinds=(("initiation_events_xt", ("xt",)),),
    description="Time-integrated observed initiation-event density profile from I(x,t)[1-f(x,t)].",
)
rmap.register(
    "initiation_events_xt",
    "initiation_pdf_xt",
    map_event_density_to_pdf_xt,
    method="initiation_events_xt_to_initiation_pdf_xt_normalized",
    optional_assumptions=_PDF_OPTIONS,
    default_assumptions=_PDF_DEFAULTS,
    source_kinds=(("initiation_events_xt", ("xt",)),),
    description="Normalized probability density phi_i(x,t) from the observed initiation-event density.",
)
rmap.register(
    "initiation_events_xt",
    "initiation_pdf",
    map_event_density_to_pdf,
    method="initiation_events_xt_to_initiation_pdf_normalized",
    optional_assumptions=_PDF_OPTIONS,
    default_assumptions=_PDF_DEFAULTS,
    source_kinds=(("initiation_events_xt", ("xt",)),),
    description="Time-integrated normalized probability density phi_i(x).",
)
rmap.register(
    "initiation_pdf_xt",
    "initiation_pdf",
    map_pdf_xt_to_pdf,
    method="initiation_pdf_xt_to_initiation_pdf_time_integrated",
    optional_assumptions=_INITIATION_EVENT_OPTIONS,
    default_assumptions=_INITIATION_EVENT_DEFAULTS,
    source_kinds=(("initiation_pdf_xt", ("xt",)),),
    description="Time-integrated normalized probability density phi_i(x).",
)

_COALESCENCE_FORWARD_DEFAULTS = {
    "resolution": 1.0,
    "resolution_time": 5.0,
    "time_grid": None,
    "max_time": None,
    "max_rep_time": None,
    "n_bins": None,
    "genome_length": None,
    "interval_kb": None,
    "perQ": False,
    "cfl": 0.8,
    "unreplicated_floor": 1e-12,
}
_COALESCENCE_PDF_FORWARD_DEFAULTS = dict(_COALESCENCE_FORWARD_DEFAULTS)
_COALESCENCE_PDF_FORWARD_DEFAULTS.update(_PDF_DEFAULTS)

rmap.register(
    "initiation_events",
    "origin_efficiency",
    map_initiation_events_to_origin_efficiency,
    method="initiation_events_to_origin_efficiency_region_integral",
    required_assumptions=("regions",),
    optional_assumptions=_ORIGIN_EFFICIENCY_OPTIONS,
    default_assumptions=_ORIGIN_EFFICIENCY_DEFAULTS,
    source_kinds=(("initiation_events", ("x",)),),
    description="Region-wise integral of the observed initiation-event density over user-defined origin or initiation-zone windows.",
)
rmap.register(
    "initiation_events_xt",
    "origin_efficiency",
    map_initiation_events_to_origin_efficiency,
    method="initiation_events_xt_to_origin_efficiency_region_integral",
    required_assumptions=("regions",),
    optional_assumptions=_ORIGIN_EFFICIENCY_OPTIONS,
    default_assumptions=_ORIGIN_EFFICIENCY_DEFAULTS,
    source_kinds=(("initiation_events_xt", ("xt",)),),
    description="Region-wise integral of I(x,t)[1-f(x,t)] over time and user-defined genomic windows.",
)
rmap.register(
    "initiation_pdf",
    "origin_efficiency",
    map_initiation_events_to_origin_efficiency,
    method="initiation_pdf_to_origin_efficiency_region_integral",
    required_assumptions=("regions",),
    optional_assumptions=_ORIGIN_EFFICIENCY_OPTIONS,
    default_assumptions=_ORIGIN_EFFICIENCY_DEFAULTS,
    source_kinds=(("initiation_pdf", ("x",)),),
    description="Region-wise integral of the normalized initiation probability density over user-defined genomic windows.",
)
rmap.register(
    "initiation_pdf_xt",
    "origin_efficiency",
    map_initiation_events_to_origin_efficiency,
    method="initiation_pdf_xt_to_origin_efficiency_region_integral",
    required_assumptions=("regions",),
    optional_assumptions=_ORIGIN_EFFICIENCY_OPTIONS,
    default_assumptions=_ORIGIN_EFFICIENCY_DEFAULTS,
    source_kinds=(("initiation_pdf_xt", ("xt",)),),
    description="Region-wise integral of phi_i(x,t) over time and user-defined genomic windows.",
)

rmap.register(
    ("fork_densities_xt", "fork_speed"),
    "coalescence_events_xt",
    map_fork_density_bundle_to_coalescence_events_xt,
    method="fork_density_bundle_to_coalescence_events_xt_mean_field",
    optional_assumptions=_COALESCENCE_EVENT_OPTIONS,
    default_assumptions=_COALESCENCE_EVENT_DEFAULTS,
    source_kinds=(("fork_speed", ("scalar", "x", "t", "xt")),),
    description="Mean-field observed fork-coalescence-event density from directional fork densities.",
)
rmap.register(
    ("fork_densities_xt", "fork_speed"),
    "coalescence_pdf_xt",
    map_fork_density_bundle_to_coalescence_pdf_xt,
    method="fork_density_bundle_to_coalescence_pdf_xt_mean_field",
    optional_assumptions=_COALESCENCE_PDF_OPTIONS,
    default_assumptions=_COALESCENCE_PDF_DEFAULTS,
    source_kinds=(("fork_speed", ("scalar", "x", "t", "xt")),),
    description="Normalized probability density phi_c(x,t) from directional fork densities.",
)
rmap.register(
    ("fork_densities_xt", "fork_speed"),
    "coalescence_events",
    map_fork_density_bundle_to_coalescence_events,
    method="fork_density_bundle_to_coalescence_events_mean_field",
    optional_assumptions=_COALESCENCE_EVENT_OPTIONS,
    default_assumptions=_COALESCENCE_EVENT_DEFAULTS,
    source_kinds=(("fork_speed", ("scalar", "x", "t", "xt")),),
    description="Time-integrated coalescence-observation profile from directional fork densities.",
)
rmap.register(
    (
        "right_fork_density_xt",
        "left_fork_density_xt",
        "fork_speed",
        "replicated_fraction_xt",
    ),
    "coalescence_events_xt",
    map_fork_density_fields_to_coalescence_events_xt,
    method="fork_density_fields_to_coalescence_events_xt_mean_field",
    optional_assumptions=_COALESCENCE_EVENT_OPTIONS,
    default_assumptions=_COALESCENCE_EVENT_DEFAULTS,
    source_kinds=(
        ("right_fork_density_xt", ("xt",)),
        ("left_fork_density_xt", ("xt",)),
        ("fork_speed", ("scalar", "x", "t", "xt")),
        ("replicated_fraction_xt", ("xt",)),
    ),
    description="Mean-field observed fork-coalescence-event density from explicit fork-density fields.",
)
rmap.register(
    (
        "right_fork_density_xt",
        "left_fork_density_xt",
        "fork_speed",
        "replicated_fraction_xt",
    ),
    "coalescence_pdf_xt",
    map_fork_density_fields_to_coalescence_pdf_xt,
    method="fork_density_fields_to_coalescence_pdf_xt_mean_field",
    optional_assumptions=_COALESCENCE_PDF_OPTIONS,
    default_assumptions=_COALESCENCE_PDF_DEFAULTS,
    source_kinds=(
        ("right_fork_density_xt", ("xt",)),
        ("left_fork_density_xt", ("xt",)),
        ("fork_speed", ("scalar", "x", "t", "xt")),
        ("replicated_fraction_xt", ("xt",)),
    ),
    description="Normalized probability density phi_c(x,t) from explicit fork-density fields.",
)
rmap.register(
    (
        "right_fork_density_xt",
        "left_fork_density_xt",
        "fork_speed",
        "replicated_fraction_xt",
    ),
    "coalescence_events",
    map_fork_density_fields_to_coalescence_events,
    method="fork_density_fields_to_coalescence_events_mean_field",
    optional_assumptions=_COALESCENCE_EVENT_OPTIONS,
    default_assumptions=_COALESCENCE_EVENT_DEFAULTS,
    source_kinds=(
        ("right_fork_density_xt", ("xt",)),
        ("left_fork_density_xt", ("xt",)),
        ("fork_speed", ("scalar", "x", "t", "xt")),
        ("replicated_fraction_xt", ("xt",)),
    ),
    description="Time-integrated coalescence-observation profile from explicit fork-density fields.",
)
rmap.register(
    ("initiation_rate", "fork_speed"),
    "coalescence_events_xt",
    map_firing_and_speed_to_coalescence_events_xt,
    method="firing_and_speed_to_coalescence_events_xt_mean_field",
    optional_assumptions=_COALESCENCE_FORWARD_OPTIONS,
    default_assumptions=_COALESCENCE_FORWARD_DEFAULTS,
    source_kinds=(
        ("initiation_rate", ("scalar", "x", "t", "xt")),
        ("fork_speed", ("scalar", "x", "t", "xt")),
    ),
    description="Mean-field observed fork-coalescence-event density from initiation rate and fork speed.",
)
rmap.register(
    ("initiation_rate", "fork_speed"),
    "coalescence_pdf_xt",
    map_firing_and_speed_to_coalescence_pdf_xt,
    method="firing_and_speed_to_coalescence_pdf_xt_mean_field",
    optional_assumptions=_COALESCENCE_PDF_FORWARD_OPTIONS,
    default_assumptions=_COALESCENCE_PDF_FORWARD_DEFAULTS,
    source_kinds=(
        ("initiation_rate", ("scalar", "x", "t", "xt")),
        ("fork_speed", ("scalar", "x", "t", "xt")),
    ),
    description="Normalized probability density phi_c(x,t) from initiation rate and fork speed.",
)
rmap.register(
    "coalescence_events_xt",
    "coalescence_events",
    map_coalescence_events_xt_to_coalescence_events,
    method="coalescence_events_xt_to_coalescence_events_time_integrated",
    optional_assumptions=_COALESCENCE_PROFILE_OPTIONS,
    default_assumptions=_COALESCENCE_PROFILE_DEFAULTS,
    source_kinds=(("coalescence_events_xt", ("xt",)),),
    description="Time-integrated observed fork-coalescence-event density profile from the coalescence sink term.",
)
rmap.register(
    "coalescence_events_xt",
    "coalescence_pdf_xt",
    map_event_density_to_pdf_xt,
    method="coalescence_events_xt_to_coalescence_pdf_xt_normalized",
    optional_assumptions=_PDF_OPTIONS,
    default_assumptions=_PDF_DEFAULTS,
    source_kinds=(("coalescence_events_xt", ("xt",)),),
    description="Normalized probability density phi_c(x,t) from the observed fork-coalescence-event density.",
)
rmap.register(
    "coalescence_events_xt",
    "coalescence_pdf",
    map_event_density_to_pdf,
    method="coalescence_events_xt_to_coalescence_pdf_normalized",
    optional_assumptions=_PDF_OPTIONS,
    default_assumptions=_PDF_DEFAULTS,
    source_kinds=(("coalescence_events_xt", ("xt",)),),
    description="Time-integrated normalized probability density phi_c(x).",
)
rmap.register(
    "coalescence_pdf_xt",
    "coalescence_pdf",
    map_pdf_xt_to_pdf,
    method="coalescence_pdf_xt_to_coalescence_pdf_time_integrated",
    optional_assumptions=_COALESCENCE_PROFILE_OPTIONS,
    default_assumptions=_COALESCENCE_PROFILE_DEFAULTS,
    source_kinds=(("coalescence_pdf_xt", ("xt",)),),
    description="Time-integrated normalized probability density phi_c(x).",
)
rmap.register(
    ("initiation_rate", "fork_speed"),
    "coalescence_events",
    map_firing_and_speed_to_coalescence_events,
    method="firing_and_speed_to_coalescence_events_mean_field",
    optional_assumptions=_COALESCENCE_FORWARD_OPTIONS,
    default_assumptions=_COALESCENCE_FORWARD_DEFAULTS,
    source_kinds=(
        ("initiation_rate", ("scalar", "x", "t", "xt")),
        ("fork_speed", ("scalar", "x", "t", "xt")),
    ),
    description="Time-integrated coalescence-observation profile from initiation rate and fork speed.",
)
rmap.register(
    ("initiation_rate", "fork_speed"),
    "coalescence_pdf",
    map_firing_and_speed_to_coalescence_pdf,
    method="firing_and_speed_to_coalescence_pdf_mean_field",
    optional_assumptions=_COALESCENCE_PDF_FORWARD_OPTIONS,
    default_assumptions=_COALESCENCE_PDF_FORWARD_DEFAULTS,
    source_kinds=(
        ("initiation_rate", ("scalar", "x", "t", "xt")),
        ("fork_speed", ("scalar", "x", "t", "xt")),
    ),
    description="Time-integrated normalized probability density phi_c(x).",
)

_SIMULATION_DERIVED_FEATURES = {
    "origin_efficiency": {},
    "inter_origin_distance": {},
    "s_phase_duration": {},
    "completion_time": {},
    "num_origins": {},
    "fork_directionality": {},
}

for _target, _defaults in _SIMULATION_DERIVED_FEATURES.items():
    rmap.register(
        ("initiation_rate", "fork_speed"),
        _target,
        _simulation_source_feature_mapper(_target, _defaults),
        method=f"firing_and_speed_to_{_target}_simulation",
        optional_assumptions=_RSIM_ASSUMPTION_NAMES,
        default_assumptions={
            "verbose": False,
            **_defaults,
        },
        source_kinds=(
            ("initiation_rate", ("scalar", "x", "t", "xt")),
            ("fork_speed", ("scalar", "x", "t", "xt")),
        ),
        description="Simulation-derived feature generated with rsim.",
    )
    rmap.register(
        "initiation_rate",
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
        source_kinds=(("initiation_rate", ("scalar", "x", "t", "xt")),),
        assumption_kinds=(("fork_speed", ("scalar", "x", "t", "xt")),),
        description="Simulation-derived feature generated with rsim.",
    )

_FORK_DENSITY_DEFAULTS = {
    "resolution": 1.0,
    "resolution_time": 5.0,
    "time_grid": None,
    "max_time": None,
    "max_rep_time": None,
    "n_bins": None,
    "genome_length": None,
    "interval_kb": None,
    "perQ": False,
    "cfl": 0.8,
}

for _target, _func, _method, _description in (
    (
        "fork_densities_xt",
        map_firing_and_speed_to_fork_densities_xt,
        "firing_and_speed_to_directional_fork_densities_xt_mean_field",
        "Mean-field map of right- and left-moving fork-density fields.",
    ),
    (
        "right_fork_density_xt",
        map_firing_and_speed_to_right_fork_density_xt,
        "firing_and_speed_to_right_fork_density_xt_mean_field",
        "Mean-field map of the right-moving fork-density field.",
    ),
    (
        "left_fork_density_xt",
        map_firing_and_speed_to_left_fork_density_xt,
        "firing_and_speed_to_left_fork_density_xt_mean_field",
        "Mean-field map of the left-moving fork-density field.",
    ),
):
    rmap.register(
        ("initiation_rate", "fork_speed"),
        _target,
        _func,
        method=_method,
        optional_assumptions=_FORK_DENSITY_OPTIONS,
        default_assumptions=_FORK_DENSITY_DEFAULTS,
        source_kinds=(
            ("initiation_rate", ("scalar", "x", "t", "xt")),
            ("fork_speed", ("scalar", "x", "t", "xt")),
        ),
        description=_description,
    )


# Supplied-field reductions and identifiable local inverses.  Each biological
# input combination is registered once per output dependence; defaults are
# settings, never duplicate single-source routes hiding a required factor.
_SPATIAL_REDUCTION_DEFAULTS = {"resolution": 1.0, "x_array": None, "spatial_average": "bins"}
_TIME_REDUCTION_DEFAULTS = {"resolution_time": 5.0, "time_grid": None}

rmap.register(
    "replicated_fraction_xt", "replicated_fraction", _conversions.map_fraction_to_mean_fraction,
    method="replicated_fraction_xt_to_replicated_fraction_spatial_mean",
    optional_assumptions=tuple(_SPATIAL_REDUCTION_DEFAULTS),
    default_assumptions=_SPATIAL_REDUCTION_DEFAULTS,
    source_kinds=(("replicated_fraction_xt", ("xt",)),),
    assumption_values=(("spatial_average", ("bins", "trapezoid")),),
    description="Spatial average of an already supplied replicated-fraction field.",
)
_TIMING_REDUCTION_DEFAULTS = {
    **_TIME_REDUCTION_DEFAULTS, "tail_policy": "require_complete", "tail_tolerance": 1e-6,
    "tail_fit_points": 3, "initial_policy": "require_unreplicated", "monotonic_policy": "raise",
}
rmap.register(
    "replicated_fraction_xt", "replication_timing", _conversions.map_fraction_to_timing,
    method="replicated_fraction_xt_to_timing_survival_integral",
    optional_assumptions=tuple(_TIMING_REDUCTION_DEFAULTS),
    default_assumptions=_TIMING_REDUCTION_DEFAULTS,
    source_kinds=(("replicated_fraction_xt", ("xt",)),),
    assumption_values=(
        ("tail_policy", ("require_complete", "truncate", "exponential")),
        ("initial_policy", ("require_unreplicated", "conditional")),
        ("monotonic_policy", ("raise", "project")),
    ),
    description="Survival integral with explicit initial conditioning and finite-window tail policy.",
)

def _bundle_extraction_mapper(extract):
    def mapper(bundle, **kwargs):
        info = {key: bundle[key] for key in ("time_grid", "x_array") if key in bundle}
        return _SimulationFeatureArray(extract(bundle), info)
    return mapper


for _target, _func in (
    ("replicated_fraction_xt", _conversions.map_bundle_to_fraction),
    ("right_fork_density_xt", _conversions.map_bundle_to_right),
    ("left_fork_density_xt", _conversions.map_bundle_to_left),
):
    rmap.register(
        "fork_densities_xt", _target, _bundle_extraction_mapper(_func),
        method=f"fork_density_bundle_to_{_target}_extraction",
        optional_assumptions=("resolution", "resolution_time", "time_grid", "x_array"),
        default_assumptions={"resolution": 1., "resolution_time": 5., "time_grid": None, "x_array": None},
        description="Extract a validated field from an existing fork-density bundle.",
    )


def _density_summary_mapper(member, source_kind, dependence):
    def mapper(source, **kwargs):
        if source_kind == "forward":
            bundle = map_firing_and_speed_to_fork_densities_xt(source, **kwargs)
            field = _conversions._bundle_field(bundle, member)
        elif source_kind == "bundle":
            bundle = source
            field = _conversions._bundle_field(bundle, member)
        else:
            bundle = {}
            field = source
        if dependence == "time":
            options = _options(kwargs, _SPATIAL_REDUCTION_DEFAULTS)
            if options.get("x_array") is None and "x_array" in bundle:
                options["x_array"] = bundle["x_array"]
            return _conversions.map_density_to_temporal(field, **options)
        options = _options(kwargs, _TIME_REDUCTION_DEFAULTS)
        if options.get("time_grid") is None and "time_grid" in bundle:
            options["time_grid"] = bundle["time_grid"]
        return _conversions.map_density_to_spatial(field, **options)

    mapper.__name__ = f"map_{source_kind}_to_{member.removesuffix('_xt')}_{dependence}"
    return mapper


for _target in ("right_fork_density", "left_fork_density"):
    _member = f"{_target}_xt"
    for _dependence in ("time", "space"):
        _reduction_defaults = (
            _SPATIAL_REDUCTION_DEFAULTS if _dependence == "time" else _TIME_REDUCTION_DEFAULTS
        )
        for _source_kind, _source, _kinds in (
            ("field", _member, ((_member, ("xt",)),)),
            ("bundle", "fork_densities_xt", ()),
            ("forward", ("initiation_rate", "fork_speed"), (
                ("initiation_rate", ("scalar", "x", "t", "xt")),
                ("fork_speed", ("scalar", "x", "t", "xt")),
            )),
        ):
            _defaults = {
                **(_FORK_DENSITY_DEFAULTS if _source_kind == "forward" else {}),
                **_reduction_defaults, "target_form": _dependence,
            }
            rmap.register(
                _source, _target, _density_summary_mapper(_member, _source_kind, _dependence),
                method=f"{_source_kind}_to_{_target}_{_dependence}_summary",
                optional_assumptions=tuple(_defaults), default_assumptions=_defaults,
                source_kinds=_kinds,
                assumption_values=(
                    (("target_form", (_dependence,)), ("spatial_average", ("bins", "trapezoid")))
                    if _dependence == "time" else (("target_form", (_dependence,)),)
                ),
                description=("Spatial-mean fork density over time (forks/kb)." if _dependence == "time"
                             else "Time-integrated spatial fork density (fork-minutes/kb)."),
            )

for _event, _pdf in (("initiation_events", "initiation_pdf"), ("coalescence_events", "coalescence_pdf")):
    rmap.register(
        _event, _pdf, _conversions.map_spatial_events_to_pdf,
        method=f"{_event}_to_{_pdf}_spatial_normalization",
        optional_assumptions=("resolution", "x_array"),
        default_assumptions={"resolution": 1.0, "x_array": None},
        source_kinds=((_event, ("x",)),),
        description="Normalize an already time-integrated spatial event profile.",
    )

for _source, _func, _method, _kinds in (
    (("fork_densities_xt", "fork_speed"), _conversions.map_density_bundle_to_coalescence_pdf,
     "fork_density_bundle_to_coalescence_pdf_spatial_normalization",
     (("fork_speed", ("scalar", "x", "t", "xt")),)),
    (("right_fork_density_xt", "left_fork_density_xt", "fork_speed", "replicated_fraction_xt"),
     _conversions.map_density_fields_to_coalescence_pdf,
     "fork_density_fields_to_coalescence_pdf_spatial_normalization",
     (("right_fork_density_xt", ("xt",)), ("left_fork_density_xt", ("xt",)),
      ("fork_speed", ("scalar", "x", "t", "xt")), ("replicated_fraction_xt", ("xt",)))),
):
    rmap.register(
        _source, "coalescence_pdf", _func, method=_method,
        optional_assumptions=_COALESCENCE_PDF_OPTIONS,
        default_assumptions=_COALESCENCE_PDF_DEFAULTS, source_kinds=_kinds,
        description="Normalize time-integrated coalescence events from supplied directional fields.",
    )

_EVENT_INVERSE_DEFAULTS = {"unreplicated_floor": 1e-8, "invalid_policy": "mask", "target_form": "spacetime"}
rmap.register(
    ("initiation_events_xt", "replicated_fraction_xt"), "initiation_rate",
    _conversions.map_events_and_fraction_to_firing,
    method="initiation_events_and_fraction_to_initiation_rate_local_inverse",
    optional_assumptions=tuple(_EVENT_INVERSE_DEFAULTS), default_assumptions=_EVENT_INVERSE_DEFAULTS,
    source_kinds=(("initiation_events_xt", ("xt",)), ("replicated_fraction_xt", ("xt",))),
    assumption_values=(("target_form", ("spacetime",)), ("invalid_policy", ("mask", "zero", "raise"))),
    description="Local E_i/(1-f) inversion; completed DNA is explicitly marked unidentifiable.",
)
_SPEED_INVERSE_DEFAULTS = {
    **_TIME_REDUCTION_DEFAULTS, "density_floor": 1e-10, "temporal_smoothw": 1,
    "unreplicated_floor": 0.001,
    "monotonic_policy": "raise", "derivative_method": "gradient", "invalid_policy": "mask",
    "target_form": "spacetime", "target_shape": "best",
}
rmap.register(
    ("replicated_fraction_xt", "fork_density_xt"), "fork_speed",
    _conversions.map_fraction_and_density_to_speed,
    method="replicated_fraction_and_fork_density_to_speed_local_inverse",
    optional_assumptions=tuple(_SPEED_INVERSE_DEFAULTS), default_assumptions=_SPEED_INVERSE_DEFAULTS,
    source_kinds=(("replicated_fraction_xt", ("xt",)), ("fork_density_xt", ("xt",))),
    assumption_values=(
        ("target_form", ("spacetime",)), ("target_shape", ("best",)),
        ("monotonic_policy", ("raise", "project")),
        ("derivative_method", ("gradient", "forward")), ("invalid_policy", ("mask", "zero", "raise")),
    ),
    description="Symmetric fork-speed reconstruction from df/dt divided by total active fork density.",
)

# Include the deterministic kinetic directionality methods in first-use
# discovery, in addition to the autoreload hook used when invoking rmap.
rmap._ensure_directionality_routes()
