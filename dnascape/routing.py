"""High-level route registry for DNAscape feature mapping."""

import inspect
from dataclasses import dataclass

import numpy as np

from .fitting import (
    map_timing_constant_firing,
    map_timing_firing,
    map_timing_firing_constant_speed,
    map_timing_firing_spatial_speed,
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

_ASSUMPTION_ALIASES = {
    "resolution_space": "resolution",
    "bin_size": "resolution",
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
        "0": "scalar",
        "scalar": "scalar",
        "constant": "scalar",
        "const": "scalar",
        "i0": "scalar",
        "v0": "scalar",
        "x": "x",
        "spatial": "x",
        "space": "x",
        "i(x)": "x",
        "v(x)": "x",
        "t": "t",
        "temporal": "t",
        "time": "t",
        "i(t)": "t",
        "v(t)": "t",
        "xt": "xt",
        "x,t": "xt",
        "(x,t)": "xt",
        "space-time": "xt",
        "spacetime": "xt",
        "i(x,t)": "xt",
        "v(x,t)": "xt",
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
}

_TIMING_TO_CONSTANT_FIRING_OPTIONS = {
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
}

_TIMING_TO_SPATIAL_FIRING_OPTIONS = {
    "resolution",
    "neighbourhood_range",
    "fit_step",
    "maxiter",
    "sim_number",
    "err_threshold",
    "deviation_threshold",
    "verbose",
    "print_every",
    "perQ",
    "stall_rate",
    "tau",
}

_TIMING_TO_CONSTANT_SPEED_OPTIONS = {
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
}

_TIMING_TO_SPATIAL_SPEED_OPTIONS = {
    "resolution",
    "neighbourhood_range",
    "fit_step",
    "maxiter",
    "speed_bounds",
    "initial_fork_speed",
    "smoothw",
    "verbose",
    "print_every",
    "perQ",
    "n_bins",
}

_DIRECTIONALITY_OPTIONS = {
    "resolution",
    "T0",
    "smoothw",
    "perQ",
}


def _options(kwargs, names):
    return {key: value for key, value in kwargs.items() if key in names}


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
    optional_assumptions=("resolution", "neighbourhood_range", "perQ", "n_bins"),
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
    optional_assumptions=("resolution", "neighbourhood_range", "perQ", "n_bins"),
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
    assumption_values=(("target_form", ("scalar", "I0", "constant")),),
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
        "verbose",
        "print_every",
        "perQ",
        "stall_rate",
        "tau",
        "target_form",
    ),
    default_assumptions={"verbose": True},
    source_kinds=(
        ("replication_timing", ("x",)),
        ("fork_speed", ("scalar",)),
    ),
    assumption_values=(("target_form", (None, "x", "I(x)", "spatial")),),
    description="Iterative fit of I(x) from timing and constant fork speed.",
)
rmap.register_placeholder(
    ("replication_timing", "fork_speed"),
    "firing_rate",
    method="timing_and_variable_speed_to_firing_rate",
    optional_assumptions=(
        "resolution",
        "neighbourhood_range",
        "fit_step",
        "maxiter",
        "sim_number",
        "verbose",
        "print_every",
        "perQ",
        "target_form",
    ),
    source_kinds=(
        ("replication_timing", ("x",)),
        ("fork_speed", ("x", "t", "xt")),
    ),
    description="Fitting firing rate from timing with variable fork speed needs a dedicated method.",
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
    assumption_values=(("target_form", (None, "scalar", "v0", "constant")),),
    description="Scalar fit of v0 from timing and firing rate.",
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
        "speed_bounds",
        "initial_fork_speed",
        "smoothw",
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
    assumption_values=(("target_form", ("x", "v(x)", "spatial")),),
    description="Spatial effective fit of v(x) from timing and firing rate.",
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
    optional_assumptions=("resolution", "neighbourhood_range", "perQ", "n_bins"),
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
    optional_assumptions=("resolution", "neighbourhood_range", "perQ", "n_bins"),
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
    assumption_values=(("target_form", ("scalar", "I0", "constant")),),
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
        "verbose",
        "print_every",
        "perQ",
        "stall_rate",
        "tau",
        "target_form",
    ),
    default_assumptions={"fork_speed": 1.4, "verbose": True},
    source_kinds=(("replication_timing", ("x",)),),
    assumption_kinds=(("fork_speed", ("scalar",)),),
    assumption_values=(("target_form", (None, "x", "I(x)", "spatial")),),
    description="Iterative fit of a space-dependent firing rate with constant fork speed.",
)
rmap.register_placeholder(
    "replication_timing",
    "firing_rate",
    method="timing_to_firing_variable_fork_speed",
    required_assumptions=("fork_speed",),
    optional_assumptions=(
        "resolution",
        "neighbourhood_range",
        "fit_step",
        "maxiter",
        "sim_number",
        "verbose",
        "print_every",
        "perQ",
    ),
    source_kinds=(("replication_timing", ("x",)),),
    assumption_kinds=(("fork_speed", ("x", "t", "xt")),),
    description="Fitting firing rate from timing with variable fork speed needs a dedicated method.",
)
rmap.register_placeholder(
    ("replication_timing", "fork_speed"),
    "firing_rate",
    method="timing_and_speed_to_general_firing_rate",
    optional_assumptions=("resolution", "geometry", "verbose", "print_every", "perQ", "target_form"),
    source_kinds=(
        ("replication_timing", ("x",)),
        ("fork_speed", ("scalar", "x", "t", "xt")),
    ),
    assumption_values=(("target_form", ("t", "I(t)", "xt", "I(x,t)")),),
    description="Time-only or space-time firing-rate targets need dedicated methods.",
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

rmap.register_placeholder(
    ("replication_timing", "firing_rate"),
    "fork_speed",
    method="timing_and_firing_to_fork_speed",
    optional_assumptions=("resolution", "geometry", "perQ"),
    source_kinds=(
        ("replication_timing", ("x",)),
        ("firing_rate", ("scalar", "x", "t", "xt")),
    ),
    description="Variable fork-speed targets need dedicated optimisation methods.",
)
rmap.register_placeholder(
    "replication_timing",
    "fork_speed",
    method="timing_to_fork_speed_with_fixed_firing",
    required_assumptions=("firing_rate",),
    optional_assumptions=("resolution", "geometry", "perQ"),
    source_kinds=(("replication_timing", ("x",)),),
    assumption_kinds=(("firing_rate", ("scalar", "x", "t", "xt")),),
    description="Pass firing_rate as an assumption once this fitting route is implemented.",
)

