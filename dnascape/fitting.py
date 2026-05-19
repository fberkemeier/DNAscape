"""High-level fitting wrappers and mapper registry."""

import os
import math
import time
import inspect
from dataclasses import dataclass

import numpy as np

from .constants import mapt
from .io import _resolve_data_path, savecsv
from .mapping import map_directionality_timing, map_firing_timing, map_timing_directionality
from .simulation import rsim
from .utils import etaferr, mean_squared_error

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
}

_CANONICAL_FEATURES = {
    "firing_rate",
    "fork_speed",
    "replication_timing",
    "fork_directionality",
    "replicated_fraction",
    "fork_density",
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
    arr = np.asarray(value)
    if arr.ndim == 0:
        return "scalar"
    if arr.ndim == 1:
        return "x"
    if arr.ndim == 2:
        return "xt"
    return f"{arr.ndim}d"


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
    assumption_kinds: tuple = ()
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

    def accepts(self, assumptions):
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
        return "implemented" if self.implemented else "planned"


def select_profile_region(
    profile,
    raw_resolution_kb=1.0,
    interval_kb=None,
    output_resolution_kb=None,
    resample_method="sample",
    input_resolution_kb=None,
):
    """Select a genomic interval from a regular spatial profile.

    Parameters
    ----------
    profile : array-like
        One-dimensional profile sampled at a regular genomic spacing.
    raw_resolution_kb : float, default 1.0
        Genomic spacing, in kb, between neighbouring values in ``profile``.
    interval_kb : tuple, optional
        ``(start_kb, end_kb)`` interval to select from the raw profile.
    output_resolution_kb : float, optional
        Spacing, in kb, of the returned profile. Defaults to
        ``raw_resolution_kb``.
    resample_method : {"sample", "nearest", "linear", "interpolate", "mean", "average"}
        Rule used to move from the raw profile to the requested output
        positions. The default, ``"sample"``, picks the nearest raw value.
    input_resolution_kb : float, optional
        Deprecated alias for ``raw_resolution_kb``.

    Returns
    -------
    selected : ndarray
        Profile values over the selected interval at ``output_resolution_kb``.
    positions_kb : ndarray
        Genomic positions, in kb, corresponding to ``selected``.
    """
    arr = np.asarray(profile)
    if arr.ndim != 1:
        raise ValueError("profile must be a one-dimensional array.")
    if arr.size == 0:
        raise ValueError("profile must contain at least one value.")

    if input_resolution_kb is not None:
        raw_resolution_kb = input_resolution_kb

    raw_resolution_kb = float(raw_resolution_kb)
    if not np.isfinite(raw_resolution_kb) or raw_resolution_kb <= 0:
        raise ValueError("raw_resolution_kb must be a finite positive value.")

    if output_resolution_kb is None:
        output_resolution_kb = raw_resolution_kb
    output_resolution_kb = float(output_resolution_kb)
    if not np.isfinite(output_resolution_kb) or output_resolution_kb <= 0:
        raise ValueError("output_resolution_kb must be a finite positive value.")

    method = str(resample_method).strip().lower().replace("-", "_")
    method_aliases = {
        "sample": "sample",
        "nearest": "sample",
        "pick": "sample",
        "linear": "linear",
        "interpolate": "linear",
        "interp": "linear",
        "mean": "mean",
        "average": "mean",
        "avg": "mean",
    }
    if method not in method_aliases:
        raise ValueError(
            "resample_method must be one of 'sample', 'linear', or 'mean'."
        )
    method = method_aliases[method]
    if method == "mean" and output_resolution_kb < raw_resolution_kb:
        raise ValueError(
            "resample_method='mean' requires output_resolution_kb >= raw_resolution_kb. "
            "Use resample_method='sample' or 'linear' for finer output positions."
        )

    genome_end_kb = arr.size * raw_resolution_kb
    if interval_kb is None:
        start_kb, end_kb = 0.0, genome_end_kb
    else:
        if len(interval_kb) != 2:
            raise ValueError("interval_kb must be a (start_kb, end_kb) pair.")
        start_kb, end_kb = map(float, interval_kb)

    if not (np.isfinite(start_kb) and np.isfinite(end_kb)):
        raise ValueError("interval_kb values must be finite.")
    if start_kb < 0 or end_kb <= start_kb:
        raise ValueError("interval_kb must satisfy 0 <= start_kb < end_kb.")
    if end_kb > genome_end_kb + 1e-9:
        raise ValueError("interval_kb extends beyond the profile length.")

    positions_kb = np.arange(start_kb, end_kb, output_resolution_kb, dtype=float)
    positions_kb = positions_kb[positions_kb < end_kb - 1e-12]
    if positions_kb.size == 0:
        return np.asarray([], dtype=arr.dtype), positions_kb

    if method == "sample":
        indices = np.floor(positions_kb / raw_resolution_kb + 0.5).astype(int)
        indices = np.clip(indices, 0, arr.size - 1)
        selected = arr[indices]
    elif method == "linear":
        raw_positions_kb = np.arange(arr.size, dtype=float) * raw_resolution_kb
        selected = np.interp(positions_kb, raw_positions_kb, arr.astype(float))
    else:
        selected = []
        for start in positions_kb:
            stop = min(start + output_resolution_kb, end_kb)
            start_idx = int(np.ceil(start / raw_resolution_kb))
            stop_idx = int(np.floor((stop - 1e-12) / raw_resolution_kb)) + 1
            start_idx = max(start_idx, 0)
            stop_idx = min(stop_idx, arr.size)
            if stop_idx <= start_idx:
                nearest = int(
                    np.clip(
                        np.floor(start / raw_resolution_kb + 0.5),
                        0,
                        arr.size - 1,
                    )
                )
                selected.append(arr[nearest])
            else:
                selected.append(np.mean(arr[start_idx:stop_idx]))
        selected = np.asarray(selected)

    return selected, positions_kb


def map_timing_firing(
    observed_timing,
    fork_speed=1.4,
    fork_speeds=None,
    stall_rate=0.,
    tau=np.inf,
    resolution=1.0,
    neighbourhood_range=2000,
    fit_step=0.6,
    maxiter=10,
    sim_number=1000,
    err_threshold=15.0,
    deviation_threshold=None,
    verbose=False,
    perQ=False,
    chr_number=1,
    save_errQ=False
):

    mse_list = []

    T = np.asarray(observed_timing, dtype=float)
    n = len(T)

    fork_speed_per_bin = fork_speed / resolution

    # initial guess for firing rate
    x0 = (math.pi / (4.0 * fork_speed_per_bin)) * np.maximum(T, 1e-80) ** (-2)
    x = x0

    lm = int(1000 / resolution)
    if perQ:
        crop = slice(None)
    else:
        crop = slice(lm, -lm) if 2 * lm < n else slice(None)
    floor = 10.0 ** (-err_threshold)

    def update_firing_rate(observed, predicted, current):
        t_safe = np.maximum(observed, 1e-80)
        p_safe = np.maximum(predicted, 1e-80)
        ratio  = p_safe / t_safe

        updated = current * (ratio ** fit_step)
        updated = np.maximum(updated, floor)

        if deviation_threshold is not None:
            mse = np.abs(predicted - observed)
            close = mse <= deviation_threshold
            updated[close] = current[close]

        return updated

    # initial prediction
    y = map_firing_timing(
        x,
        fork_speed=fork_speed,
        resolution=resolution,
        neighbourhood_range=neighbourhood_range,
        perQ=perQ,
    )
    mse = mean_squared_error(T[crop], y[crop])

    if stall_rate > 0. or fork_speeds is not None:
        maxiter0 = 0
    else:
        maxiter0 = maxiter

    start_time = time.time()

    for it in range(1, maxiter0 + 1):
        x = update_firing_rate(T, y, x)
        y = map_firing_timing(
            x,
            fork_speed=fork_speed,
            resolution=resolution,
            neighbourhood_range=neighbourhood_range,
            perQ=perQ
        )
        mse = mean_squared_error(T[crop], y[crop])

        if verbose:
            etaferr(it=it, maxiter=maxiter0, start_time=start_time, err=mse)
            mse_list.append(mse)
            

    # Stalling models
    if stall_rate > 0. or fork_speeds is not None:

        alpha = fit_step
        iterations = maxiter
    
        rate_min = 1e-9
        rate_max = None
        T_target = np.asarray(observed_timing, float)
        
        rates = np.clip(np.asarray(x, float), rate_min, None)
        #eff_f = 0.17
        #rates = (math.pi / (4.0 * 10 * fork_speed_per_bin)) * np.maximum(T, 1e-80) ** (-2)
        
        if rate_max is None:
            pos = rates[rates > 0]
            rate_max = (10.0 * np.median(pos)) if pos.size else 1.0
    
        hist_mse = []
        T_pred = None
        start_time = time.time()
    
        for k in range(iterations):
            simres = rsim(
                rates,
                sim_number=sim_number,
                fork_speed=fork_speeds if fork_speeds is not None else fork_speed,
                stall_rate=stall_rate,
                tau=tau,
                resolution_space=resolution,
                verbose=False,
                perQ=perQ
            )
            T_pred = simres['replication_timing']
    
            eps = 1e-12
            ratio = (T_pred + eps) / (T_target + eps)
            rates *= np.power(ratio, alpha)
            np.clip(rates, rate_min, rate_max, out=rates)
    
            mse = T_pred - T_target
            mse = float(np.mean(mse**2))
            hist_mse.append(mse)
    
            if verbose:
                etaferr(it=k+1, maxiter=iterations, start_time=start_time, err=mse)
                mse_list.append(mse)

        x = rates

    #np.savetxt(f"data/fit_error/mse_chr{chr_number}.txt", mse_list, fmt="%.6f")

    if save_errQ:
        savecsv(arr=mse_list, chr_number=chr_number, stitle=f'fit_error', outpath="fit_error")
        
    return x

class DataTypeMapper:
    """Route replication feature mappings and fits.

    The mapper is intentionally broader than a direct function lookup. It
    records which feature-to-feature routes are implemented now and which
    routes are planned but still need a dedicated fitting method.
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
        assumption_kinds=(),
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
            assumption_kinds=tuple(
                (_canonical_assumption(name), (kinds,) if isinstance(kinds, str) else tuple(kinds))
                for name, kinds in assumption_kinds
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
        assumption_kinds=(),
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
            assumption_kinds=assumption_kinds,
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

        kind_matches = [spec for spec in candidates if spec.accepts(assumptions)]
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
            missing_msg = f"; missing {', '.join(missing)}" if missing else ""
            details.append(f"{spec.method} expects {req or 'no specific field kinds'}{missing_msg}")
        raise ValueError(
            "No registered mapping route matches the supplied assumption field kinds. "
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
            req = ", ".join(
                f"{name}: {'/'.join(kinds)}"
                for name, kinds in spec.assumption_kinds
            )
            details = []
            if required:
                details.append(f"requires {required}")
            if req:
                details.append(f"supports {req}")
            suffix = f" [{'; '.join(details)}]" if details else ""
            print(f"  {src} -> {spec.target} via {spec.method} ({status}){suffix}")

    def features(self):
        return tuple(sorted(_CANONICAL_FEATURES))

    def routes(self):
        return tuple(self._specs)


rmap = DataTypeMapper()
rmap.register(
    "firing_rate",
    "replication_timing",
    map_firing_timing,
    method="firing_to_timing_deterministic",
    required_assumptions=("fork_speed",),
    optional_assumptions=("resolution", "neighbourhood_range", "perQ"),
    default_assumptions={"fork_speed": 1.4},
    assumption_kinds=(("fork_speed", ("scalar",)),),
    description="Deterministic mapping with constant fork speed.",
)
rmap.register_placeholder(
    "firing_rate",
    "replication_timing",
    method="firing_to_timing_variable_fork_speed",
    required_assumptions=("fork_speed",),
    optional_assumptions=("resolution", "neighbourhood_range", "perQ"),
    assumption_kinds=(("fork_speed", ("x", "xt")),),
    description="Variable fork-speed deterministic mapping needs a dedicated solver.",
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
        "perQ",
        "stall_rate",
        "tau",
    ),
    default_assumptions={"fork_speed": 1.4},
    assumption_kinds=(("fork_speed", ("scalar",)),),
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
        "perQ",
    ),
    assumption_kinds=(("fork_speed", ("x", "xt")),),
    description="Fitting firing rate from timing with variable fork speed is planned.",
)
rmap.register_placeholder(
    ("replication_timing", "fork_speed"),
    "firing_rate",
    method="timing_and_fork_speed_to_firing",
    optional_assumptions=("resolution", "geometry", "perQ"),
    description="Use assumptions={'fork_speed': scalar} for the implemented constant-speed route.",
)
rmap.register(
    "replication_timing",
    "fork_directionality",
    map_timing_directionality,
    method="timing_to_directionality",
    required_assumptions=("fork_speed",),
    optional_assumptions=("resolution", "T0", "smoothw", "perQ"),
    default_assumptions={"fork_speed": 1.4},
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
    assumption_kinds=(("fork_speed", ("scalar",)),),
    description="Integration of fork directionality to timing.",
)
rmap.register_placeholder(
    ("replication_timing", "firing_rate"),
    "fork_speed",
    method="timing_and_firing_to_fork_speed",
    optional_assumptions=("resolution", "geometry", "perQ"),
    description="Fitting fork speed from timing and firing rate needs a dedicated optimisation method.",
)
rmap.register_placeholder(
    "replication_timing",
    "fork_speed",
    method="timing_to_fork_speed_with_fixed_firing",
    required_assumptions=("firing_rate",),
    optional_assumptions=("resolution", "geometry", "perQ"),
    assumption_kinds=(("firing_rate", ("scalar", "x", "xt")),),
    description="Pass firing_rate as an assumption once this fitting route is implemented.",
)

def rfit(
    d1='firing_rate',
    d2='replication_timing',
    cell_line='H1', chr_number=1,
    fork_speed=1.4, resolution=1.,
    source=[],
    saveQ=False,
    **kwargs
):

    if any(source):
        data1 = source
    else:
        data1 = np.loadtxt(_resolve_data_path(f"{mapt[d1]}/{mapt[d1]}_{cell_line}_chr{chr_number}.txt"), dtype=float)
    data2 = rmap.map(d1, d2, data1, fork_speed=fork_speed, resolution=resolution, **kwargs)

    if saveQ:
        out = _resolve_data_path(f"{mapt[d2]}/{mapt[d2]}_{cell_line}_chr{chr_number}.txt", for_write=True)
        os.makedirs(out.parent, exist_ok=True)
        np.savetxt(out, data2)

    return data2
