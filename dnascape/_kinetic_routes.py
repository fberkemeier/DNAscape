"""One deterministic inverse contract behind the public kinetic route names.

Legacy numerical functions remain available for Python callers.  The registry
uses a common hazard fit for timing/fraction plus the supplied kinetic field.
"""

from dataclasses import replace


_DOMAIN_DEFAULTS = {
    "resolution": 1.0, "resolution_time": 5.0, "time_grid": None,
    "n_bins": None, "T0": None, "neighbourhood_range": 2000.0,
    "perQ": False, "error": None, "verbose": False,
}
_SPACE_DEFAULTS = {"space_maxiter": 10, "space_fit_step": 1.0}
_TIME_DEFAULTS = {
    "temporal_maxiter": 10, "temporal_fit_step": 1.0,
    "weibull_shape": 2.0, "weibull_delay": 0.0,
    "weibull_onset": 1.0,
    "fit_weibull_shape": True,
}


def _defaults(target, form):
    defaults = dict(_DOMAIN_DEFAULTS)
    defaults.update(alpha_smooth=0.001, alpha_prior=0.0,
                    adaptive_fitting=True, relative_tolerance=.001)
    if form in {"constant", "space"}:
        defaults.update(maxiter=10, fit_step=1.0)
    else:
        defaults.update(_TIME_DEFAULTS)
        if form == "spacetime":
            defaults.update(_SPACE_DEFAULTS, spacetime_order="crossed")
    if target == "initiation_rate":
        defaults.update(rate_bounds=(None, None), initial_initiation_rate=None, floor=1e-15)
    else:
        defaults.update(speed_bounds=(0.05, 10.0), initial_fork_speed=1.4)
    defaults.update(target_form=form, target_shape="weibull" if form in {"time", "spacetime"} else "best")
    return defaults


def _mapper(target, observation, source_keys, legacy=None):
    supplied_key = "fork_speed" if target == "initiation_rate" else "initiation_rate"

    def fit(source, **options):
        from ._kinetic_fit import fit_kinetic

        data = source if isinstance(source, dict) else {source_keys[0]: source}
        observed_key = next(key for key in source_keys if key != supplied_key)
        supplied = data.get(supplied_key, options.get(supplied_key))
        # Existing stalled-fork routes retain their stochastic model. A
        # deterministic light cone assumes a prescribed propagation velocity.
        stalled = any(float(options.get(key) or 0.0) > 0.0
                      for key in ("stall_rate", "stall_rate_per_kb"))
        if stalled:
            if legacy is None:
                raise ValueError("Stalled-fork fits require timing observations; use Simulation for fraction dynamics with stalling.")
            return legacy(source, **options)
        options = {key: value for key, value in options.items()
                   if key not in {supplied_key, "target_shape", "sim_number", "stall_rate",
                                  "stall_rate_per_kb", "tau", "refine_seed", "print_every"}}
        return fit_kinetic(data[observed_key], supplied, target=target,
                           observation=observation, **options)

    fit.__name__ = f"fit_{target}_from_{observation}"
    fit._unified_kinetic = True
    return fit


def install_kinetic_routes(mapper):
    """Preserve existing method names and add fraction-based kinetic fits."""
    updated = []
    for spec in mapper._specs:
        supplied = "fork_speed" if spec.target == "initiation_rate" else "initiation_rate"
        compatible = (
            spec.target in {"initiation_rate", "fork_speed"}
            and "replication_timing" in spec.source
            and set(spec.source) <= {"replication_timing", supplied}
            and (supplied in spec.source or supplied in spec.required_assumptions)
        )
        if not compatible:
            updated.append(spec)
            continue
        old_defaults = dict(spec.default_assumptions)
        form = old_defaults.get("target_form")
        if form is None:
            form = next((values[0] for name, values in spec.assumption_values
                         if name == "target_form"), "space")
        defaults = _defaults(spec.target, form)
        if supplied in old_defaults:
            defaults[supplied] = old_defaults[supplied]
        # Compatibility aliases stay callable, while the app renders only
        # controls consumed by the selected spatial/temporal phase.
        optional = tuple(dict.fromkeys((*defaults, *spec.optional_assumptions,
                                       "space_maxiter", "space_fit_step", "temporal_maxiter",
                                       "temporal_fit_step", "fit_step", "maxiter")))
        values = (("target_form", (form,)),)
        if form in {"time", "spacetime"}:
            values += (("target_shape", ("weibull", "best")),)
        if form == "spacetime":
            values += (("spacetime_order", ("crossed", "space_then_time", "time_then_space")),)
        updated.append(replace(
            spec, func=_mapper(spec.target, "timing", spec.source, legacy=spec.func),
            optional_assumptions=optional, default_assumptions=tuple(defaults.items()),
            assumption_values=values,
            description="Fit the kinetic target to timing through the shared finite-influence replication hazard.",
        ))
    mapper._specs = updated

    for target, supplied in (("initiation_rate", "fork_speed"), ("fork_speed", "initiation_rate")):
        for form, name in (("constant", "constant"), ("space", "spatial"),
                           ("time", "temporal"), ("spacetime", "spacetime")):
            for observed_key, observed_form in (("replicated_fraction_xt", "xt"),
                                                 ("replicated_fraction", "t")):
                sources = (observed_key, supplied)
                defaults = _defaults(target, form)
                values = (("target_form", (form,)),)
                if form in {"time", "spacetime"}:
                    values += (("target_shape", ("weibull", "best")),)
                if form == "spacetime":
                    values += (("spacetime_order", ("crossed", "space_then_time", "time_then_space")),)
                mapper.register(
                    sources, target, _mapper(target, "fraction", sources),
                    method=f"{observed_key}_and_{supplied}_to_{name}_{target}",
                    optional_assumptions=tuple(defaults), default_assumptions=defaults,
                    source_kinds=((observed_key, (observed_form,)),
                                  (supplied, ("scalar", "x", "t", "xt"))),
                    assumption_values=values,
                    description="Fit the kinetic target to replicated-fraction observations through the shared replication hazard.",
                )
