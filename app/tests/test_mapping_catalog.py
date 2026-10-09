"""Scientific drift tests for the Mapping page catalogue boundary."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import replace
from itertools import product
from typing import Any

import pytest

from app import mapping_catalog as catalog_module
from app.mapping_catalog import (
    DATA_FORMS,
    RDATA_FORM_HINT,
    RDATA_SOURCE_FORMAT_KEYS,
    RDATA_TARGET_FORMAT_KEYS,
    RDATA_UPLOAD_ACCEPT,
    RDATA_UPLOAD_HINT,
    DataForm,
    MappingInput,
    MappingRouteFamily,
    get_mapping_catalog,
    mapping_route,
    mapping_route_families,
    mapping_routes,
    mapping_target,
    mapping_targets,
    rdata_formats,
)
from dnascape import rmap

EXPECTED_PUBLIC_TARGET_COUNT = 27
EXPECTED_IMPLEMENTED_METHOD_COUNT = 114
EXPECTED_FRIENDLY_TARGET_COUNT = 19


def _records(table: Any) -> tuple[dict[str, Any], ...]:
    if hasattr(table, "to_dict"):
        return tuple(table.to_dict(orient="records"))
    return tuple(dict(record) for record in table)


def _unique(values: Iterable[str]) -> tuple[str, ...]:
    result: list[str] = []
    for value in values:
        if value not in result:
            result.append(value)
    return tuple(result)


def _normalise_kind(kind: object) -> str:
    value = str(kind).strip().lower()
    return value if value in DATA_FORMS else "supplied"


def _expected_forms(spec: Any, name: str) -> tuple[str, ...]:
    for pairs in (spec.source_kinds, spec.assumption_kinds):
        for quantity, kinds in pairs:
            if quantity == name:
                return _unique(_normalise_kind(kind) for kind in kinds)
    if name == "fork_densities_xt":
        return ("bundle",)
    if name == "regions":
        return ("regions",)
    return ("supplied",)


def _expected_inputs(spec: Any) -> tuple[tuple[str, str, tuple[str, ...]], ...]:
    features = set(rmap.features())
    inputs: list[tuple[str, str, tuple[str, ...]]] = [
        (name, "source", _expected_forms(spec, name)) for name in spec.source
    ]
    present = set(spec.source)
    assumption_names = _unique(
        list(spec.required_assumptions) + [name for name, _kinds in spec.assumption_kinds]
    )
    for name in assumption_names:
        if name in present:
            continue
        if name == "regions":
            inputs.append((name, "regions", ("regions",)))
        elif name in features:
            inputs.append((name, "feature_assumption", _expected_forms(spec, name)))
    return tuple(inputs)


def _input_signature(item: MappingInput) -> tuple[str, str, tuple[str, ...]]:
    return item.key, item.role, item.forms


def _family_route_accepts_forms(
    family: MappingRouteFamily,
    route: Any,
    selections: dict[str, DataForm],
) -> bool:
    """Compare a friendly/canonical family input with an exact route parameter."""

    return all(form in family.variant_input(route, key).forms for key, form in selections.items())


def _candidate_form_selections(
    family: MappingRouteFamily,
    *,
    partial: bool,
) -> Iterable[dict[str, DataForm]]:
    """Enumerate every full or partial selection exposed by a route family."""

    domains: list[tuple[DataForm | None, ...]] = []
    for key in family.input_keys:
        forms = family.allowed_forms(key)
        can_be_omitted = partial or key in family.optional_input_keys
        domains.append((None, *forms) if can_be_omitted else forms)
    for values in product(*domains):
        yield {
            key: value
            for key, value in zip(family.input_keys, values, strict=True)
            if value is not None
        }


def test_catalog_is_an_exact_snapshot_of_the_public_implemented_rmap_registry() -> None:
    catalog = get_mapping_catalog()
    records = _records(rmap.targets(implemented=True))
    public_targets = tuple(str(record["target"]) for record in records)
    public_target_set = set(public_targets)
    implemented_specs = tuple(
        spec for spec in rmap.routes() if spec.implemented and spec.target in public_target_set
    )
    methods = tuple(spec.method for spec in implemented_specs)

    # These explicit counts intentionally make scientific registry drift visible.
    assert len(public_targets) == EXPECTED_PUBLIC_TARGET_COUNT
    assert len(implemented_specs) == EXPECTED_IMPLEMENTED_METHOD_COUNT
    assert len(methods) == len(set(methods))

    assert catalog.internal_target_count == EXPECTED_PUBLIC_TARGET_COUNT
    assert catalog.route_count == EXPECTED_IMPLEMENTED_METHOD_COUNT
    assert catalog.target_count == EXPECTED_FRIENDLY_TARGET_COUNT
    assert catalog.signature() == (public_targets, methods)
    assert catalog.public_target_ids == public_targets
    assert catalog.methods == methods


def test_route_families_are_factor_only_and_partition_all_exact_methods() -> None:
    """The route picker describes biology while retaining every rmap method."""

    catalog = get_mapping_catalog()
    represented_methods = tuple(
        method for family in catalog.route_families for method in family.methods
    )

    assert catalog.route_family_count > 0
    assert catalog.route_family_count == 84
    assert len(represented_methods) == EXPECTED_IMPLEMENTED_METHOD_COUNT
    assert len(represented_methods) == len(set(represented_methods))
    assert set(represented_methods) == set(catalog.methods)

    for family in catalog.route_families:
        assert family.variants
        assert family.factor_keys
        # An assumed speed is still a scientific factor, with its default
        # prefilled in the same family as explicitly supplied speed.
        assert family.factor_keys == family.input_keys
        assert all(item.required for item in family.inputs)
        assert family.input_keys == tuple(item.key for item in family.inputs)
        assert family.factor_keys == tuple(
            item.key for item in family.inputs if item.key not in family.optional_input_keys
        )
        assert not set(family.factor_keys) & set(family.optional_input_keys)
        assert set(family.input_keys) == set(family.factor_keys) | set(family.optional_input_keys)
        assert family.label == " + ".join(family.input(key).label for key in family.factor_keys)
        assert "_" not in family.label
        assert family.methods == tuple(route.method for route in family.variants)
        assert all(route.target == family.target for route in family.variants)
        assert all(route.target_key == family.target_key for route in family.variants)
        assert all(route.output_dependence == family.output_dependence for route in family.variants)
        for route in family.variants:
            assert (
                tuple(family.exact_input_key(route, key) for key in family.input_keys)
                == route.source_keys
            )

    for target in catalog.targets:
        for dependence in target.dependences:
            labels = tuple(
                family.label for family in catalog.route_families_for(target.key, dependence)
            )
            assert len(labels) == len(set(labels))


def test_same_feature_reductions_name_the_source_dependence_and_never_offer_identity() -> None:
    catalog = get_mapping_catalog()
    reductions = []
    for family in catalog.route_families:
        assert family.label != catalog.target(family.target_key).label
        if family.input_keys != (family.target_key,):
            continue
        reductions.append(family)
        assert family.input(family.target_key).forms == ("xt",)
        assert family.output_form in {"x", "t"}
        assert family.label.endswith(" (space-time)")
        for route in family.variants:
            assert route.output_form not in route.inputs[0].forms

    # These are eleven actual field reductions, not same-form copies.
    assert len(reductions) == 11
    assert {family.target_key for family in reductions} == {
        "initiation_events", "initiation_pdf", "coalescence_events", "coalescence_pdf",
        "replicated_fraction", "fork_density", "right_fork_density", "left_fork_density",
    }


@pytest.mark.parametrize("input_key", ["initiation_events", "initiation_events_xt"])
@pytest.mark.parametrize("forms", [("x",), ("x", "xt")])
def test_catalog_filters_same_form_identity_but_preserves_other_input_forms(
    input_key: str, forms: tuple[str, ...],
) -> None:
    prototype = next(
        spec for spec in rmap.routes()
        if spec.method == "initiation_events_xt_to_initiation_events_time_integrated"
    )
    candidate = replace(
        prototype, method="test_same_quantity_conversion", source=(input_key,),
        source_kinds=((input_key, forms),),
    )
    routes = catalog_module._build_routes((candidate,), {})
    if forms == ("x",):
        assert routes == ()
    else:
        assert len(routes) == 1
        source = routes[0].input(input_key)
        assert source.forms == ("xt",)
        assert tuple(form for form, _ in source.form_formats) == ("xt",)
        assert tuple(form for form, _ in source.form_accepts) == ("xt",)
        assert tuple(form for form, _ in source.form_multiples) == ("xt",)
        assert tuple(form for form, _ in source.form_track_modes) == ("xt",)
        family = catalog_module._build_route_families(routes)[0]
        assert family.allowed_forms("initiation_events") == ("xt",)
        with pytest.raises(ValueError, match="No implemented rmap method"):
            family.resolve({"initiation_events": "x"})


def test_identity_guard_does_not_remove_a_target_used_with_an_independent_measurement() -> None:
    prototype = next(
        spec for spec in rmap.routes()
        if spec.method == "timing_and_firing_to_spatial_fork_speed"
    )
    # A future prior-based inverse still has independent measured timing.
    candidate = replace(
        prototype, method="test_timing_and_speed_prior_to_speed",
        source=("replication_timing", "fork_speed"),
        source_kinds=(("replication_timing", ("x",)), ("fork_speed", ("x",))),
    )
    assert len(catalog_module._build_routes((candidate,), {})) == 1


def test_fork_speed_catalog_has_timing_directionality_routes_for_constant_and_space() -> None:
    expected = {
        "constant": "timing_and_directionality_to_constant_fork_speed",
        "space": "timing_and_directionality_to_spatial_fork_speed",
    }
    for dependence, method in expected.items():
        families = mapping_route_families("fork_speed", dependence)
        family = next(
            item
            for item in families
            if item.factor_keys == ("replication_timing", "fork_directionality")
        )
        assert family.methods == (method,)
        assert family.input("replication_timing").forms == ("x",)
        assert family.input("fork_directionality").forms == ("x",)
        route = family.resolve({"replication_timing": "x", "fork_directionality": "x"})
        assert route.output_dependence == dependence
        assert route.output_form == ("scalar" if dependence == "constant" else "x")
        assert route.default_target_shape == "best"
        t0 = route.setting("T0")
        assert t0.nullable is True
        assert t0.default is None
        if dependence == "space":
            regularization = route.setting("speed_regularization")
            assert regularization.label == "Speed regularization"
            assert regularization.kind == "number"
            assert regularization.default == pytest.approx(4.0)
            assert regularization.min_value is not None
            assert regularization.min_value == 0
            assert "smooth" in regularization.description
        else:
            with pytest.raises(KeyError):
                route.setting("speed_regularization")


def test_route_family_form_constraints_are_the_exact_union_of_rmap_variants() -> None:
    """Every partial input selection constrains all peers without inventing tuples."""

    catalog = get_mapping_catalog()
    for family in catalog.route_families:
        for selections in _candidate_form_selections(family, partial=True):
            expected_compatible = tuple(
                route
                for route in family.variants
                if _family_route_accepts_forms(family, route, selections)
            )
            assert family.compatible_routes(selections) == expected_compatible

            for input_key in family.input_keys:
                peer_selections: dict[str, DataForm] = {
                    key: form for key, form in selections.items() if key != input_key
                }
                peer_compatible = tuple(
                    route
                    for route in family.variants
                    if _family_route_accepts_forms(family, route, peer_selections)
                )
                expected_forms = tuple(
                    form
                    for form in DATA_FORMS
                    if any(
                        form in family.variant_input(route, input_key).forms
                        for route in peer_compatible
                    )
                )
                actual_forms = family.allowed_forms(input_key, selections)
                assert actual_forms == expected_forms

        assert family.compatible_routes({"not_an_input": "x"}) == ()
        with pytest.raises(KeyError):
            family.allowed_forms("not_an_input")


def test_route_family_partial_selection_can_mutually_constrain_peer_forms() -> None:
    """The family abstraction remains safe if rmap adds non-Cartesian routes."""

    catalog = get_mapping_catalog()
    base_family = next(family for family in catalog.route_families if len(family.inputs) == 2)
    base_route = base_family.variants[0]
    left, right = base_route.inputs
    left_scalar = replace(left, forms=("scalar",))
    left_space = replace(left, forms=("x",))
    right_scalar = replace(right, forms=("scalar",))
    right_space = replace(right, forms=("x",))
    scalar_space = replace(
        base_route, method="test_scalar_space", inputs=(left_scalar, right_space)
    )
    space_scalar = replace(
        base_route, method="test_space_scalar", inputs=(left_space, right_scalar)
    )
    synthetic = replace(
        base_family,
        key="test_mutual_constraints",
        inputs=(replace(left, forms=("scalar", "x")), replace(right, forms=("scalar", "x"))),
        variants=(scalar_space, space_scalar),
    )

    assert synthetic.allowed_forms(right.key) == ("scalar", "x")
    assert synthetic.allowed_forms(right.key, {left.key: "scalar"}) == ("x",)
    assert synthetic.allowed_forms(right.key, {left.key: "x"}) == ("scalar",)
    assert synthetic.allowed_forms(left.key, {right.key: "scalar"}) == ("x",)
    assert synthetic.allowed_forms(left.key, {right.key: "x"}) == ("scalar",)
    assert synthetic.resolve({left.key: "scalar", right.key: "x"}).method == "test_scalar_space"
    with pytest.raises(ValueError, match="No implemented rmap method"):
        synthetic.resolve({left.key: "scalar", right.key: "scalar"})


def test_canonical_profile_and_spacetime_families_resolve_exact_parameter_names() -> None:
    """X and XT spellings are one visible factor family, not duplicate routes."""

    catalog = get_mapping_catalog()
    density_family = next(
        family
        for family in catalog.route_families_for("fork_directionality", "space")
        if family.factor_keys == ("right_fork_density", "left_fork_density")
    )
    assert density_family.allowed_forms("left_fork_density", {"right_fork_density": "x"}) == ("x",)
    assert density_family.allowed_forms("left_fork_density", {"right_fork_density": "xt"}) == (
        "xt",
    )
    profile = density_family.resolve({"right_fork_density": "x", "left_fork_density": "x"})
    spacetime = density_family.resolve({"right_fork_density": "xt", "left_fork_density": "xt"})
    assert profile.method == "fork_density_profiles_to_directionality"
    assert spacetime.method == "fork_density_fields_to_directionality_time_integrated"
    assert density_family.exact_input_key(profile, "right_fork_density") == "right_fork_density"
    assert (
        density_family.exact_input_key(spacetime, "right_fork_density") == "right_fork_density_xt"
    )

    regional_families = catalog.route_families_for("origin_efficiency", "regions")
    for factor, x_method, xt_method in (
        (
            "initiation_events",
            "initiation_events_to_origin_efficiency_region_integral",
            "initiation_events_xt_to_origin_efficiency_region_integral",
        ),
        (
            "initiation_pdf",
            "initiation_pdf_to_origin_efficiency_region_integral",
            "initiation_pdf_xt_to_origin_efficiency_region_integral",
        ),
    ):
        family = next(item for item in regional_families if factor in item.factor_keys)
        assert family.resolve({factor: "x", "regions": "regions"}).method == x_method
        resolved_xt = family.resolve({factor: "xt", "regions": "regions"})
        assert resolved_xt.method == xt_method
        assert family.exact_input_key(resolved_xt, factor) == f"{factor}_xt"


def test_complete_family_form_choices_resolve_deterministically_without_gaps() -> None:
    """Every selectable tuple resolves to an exact method; invalid tuples fail closed."""

    catalog = get_mapping_catalog()
    resolved_methods: set[str] = set()

    for family in catalog.route_families:
        for selections in _candidate_form_selections(family, partial=False):
            compatible = tuple(
                route
                for route in family.variants
                if _family_route_accepts_forms(family, route, selections)
            )
            if compatible:
                resolved = family.resolve(selections)  # type: ignore[arg-type]
                assert resolved in compatible
                assert family.resolve(selections) is resolved  # type: ignore[arg-type]
                resolved_methods.add(resolved.method)
            else:
                with pytest.raises(ValueError, match="No implemented rmap method"):
                    family.resolve(selections)  # type: ignore[arg-type]

        if len(family.factor_keys) > 1:
            incomplete = {family.factor_keys[0]: family.allowed_forms(family.factor_keys[0])[0]}
            with pytest.raises(ValueError, match="Incomplete route-form selection"):
                family.resolve(incomplete)  # type: ignore[arg-type]

    # Exact methods may overlap on the same tuple, but every method remains
    # represented in a selectable family and every family has a resolvable tuple.
    assert resolved_methods
    assert all(
        any(
            family.resolve(selections).method in family.methods  # type: ignore[arg-type]
            for selections in _candidate_form_selections(family, partial=False)
            if family.compatible_routes(selections)  # type: ignore[arg-type]
        )
        for family in catalog.route_families
    )


def test_every_catalog_route_preserves_rmap_inputs_forms_and_constraints() -> None:
    catalog = get_mapping_catalog()
    authoritative = {
        spec.method: spec
        for spec in rmap.routes()
        if spec.implemented and spec.target in set(catalog.public_target_ids)
    }

    assert set(catalog.methods) == set(authoritative)
    for route in catalog.routes:
        spec = authoritative[route.method]
        assert route.target == spec.target
        assert route.required_assumptions == tuple(spec.required_assumptions)
        assert route.optional_assumptions == tuple(spec.optional_assumptions)
        assert tuple(name for name, _value in route.defaults) == tuple(
            name for name, _value in spec.default_assumptions
        )
        assert route.assumption_values == tuple(spec.assumption_values)
        assert tuple(_input_signature(item) for item in route.inputs) == _expected_inputs(spec)
        assert route.output_form in DATA_FORMS
        assert route.output_dependence
        assert route.label
        assert route.description
        assert len({setting.key for setting in route.settings}) == len(route.settings)
        assert not set(route.source_keys) & {setting.key for setting in route.settings}
        assert set(route.setting_defaults_dict) == {
            setting.key for setting in route.settings if setting.has_default
        }
        assert all(setting.location != "derived" for setting in route.user_settings)

        for item in route.inputs:
            if item.role == "feature_assumption" and item.has_default:
                assert item.required is False
            else:
                assert item.required is True
            assert item.forms
            assert set(item.forms) <= set(DATA_FORMS)
            assert item.label
            assert item.description
            assert item.hint
            if item.key == "fork_densities_xt":
                assert item.forms == ("bundle",)
                assert item.prepared_result_only is True
                assert item.allows_upload is False
            elif item.key == "regions":
                assert item.forms == ("regions",)
                assert item.role == "regions"

            plain_formats = tuple(
                data_format.key
                for data_format in rdata_formats(source=True, target=False)
                if not data_format.genomic_track
            )
            expected_by_form = {
                "scalar": (),
                "x": RDATA_SOURCE_FORMAT_KEYS,
                "t": plain_formats,
                "xt": RDATA_SOURCE_FORMAT_KEYS,
                "supplied": plain_formats,
                "bundle": (),
                "regions": plain_formats,
            }
            assert item.form_formats == tuple((form, expected_by_form[form]) for form in item.forms)
            for form in item.forms:
                assert item.formats_for(form) == expected_by_form[form]
                if form in {"scalar", "bundle"}:
                    assert item.accept_for(form) == ""
                elif form in {"x", "xt"}:
                    assert ".bedgraph" in item.accept_for(form)
                    assert ".bw" in item.accept_for(form)
                    assert ".gz" in item.accept_for(form)
                else:
                    assert ".bedgraph" not in item.accept_for(form)
                    assert ".bw" not in item.accept_for(form)
                assert item.multiple_for(form) is (form == "xt")
                expected_track_mode = {"x": "single", "xt": "series"}.get(form, "unsupported")
                assert item.track_mode_for(form) == expected_track_mode


def test_every_friendly_target_variant_has_routes_and_consistent_forms() -> None:
    catalog = get_mapping_catalog()
    assert mapping_targets() == catalog.targets

    for target in catalog.targets:
        assert target.key
        assert target.label
        assert target.description
        assert target.variants
        assert target.default_dependence in target.dependences
        assert mapping_target(target.key) == target

        for variant in target.variants:
            assert variant.internal_target in catalog.public_target_ids
            assert variant.dependence
            assert variant.output_form in DATA_FORMS
            assert variant.methods
            routes = mapping_routes(target.key, variant.dependence)
            assert routes
            assert tuple(route.method for route in routes) == variant.methods
            assert all(route.target_key == target.key for route in routes)
            assert all(route.target == variant.internal_target for route in routes)
            assert all(route.output_dependence == variant.dependence for route in routes)
            assert all(mapping_route(route.method) == route for route in routes)


def test_dependence_shape_region_and_bundle_semantics_are_not_generalized() -> None:
    expected_variable_dependences = ("constant", "space", "time", "spacetime")
    firing = mapping_target("firing_rate")
    speed = mapping_target("fork_speed")
    timing = mapping_target("replication_timing")
    directionality = mapping_target("fork_directionality")

    assert firing.dependences == expected_variable_dependences
    assert speed.dependences == expected_variable_dependences
    assert timing.dependences == ("space",)
    assert directionality.dependences == ("space",)
    assert "constant" not in timing.dependences
    assert "constant" not in directionality.dependences

    for dependence in ("constant", "space"):
        assert all(not route.target_shapes for route in mapping_routes("firing_rate", dependence))
    for dependence in ("time", "spacetime"):
        routes = tuple(route for route in mapping_routes("firing_rate", dependence)
                       if "replication_timing" in route.source_keys)
        assert routes
        assert all(route.target_shapes == ("weibull",) for route in routes)
        assert all(route.default_target_shape == "weibull" for route in routes)
    assert all(route.target_shapes == ("best",) for route in mapping_routes("fork_speed"))

    regional_routes = tuple(
        route
        for route in mapping_routes("origin_efficiency", "regions")
        if any(item.key == "regions" for item in route.inputs)
    )
    assert len(regional_routes) == 4
    assert all(route.input("regions").forms == ("regions",) for route in regional_routes)
    origin = mapping_target("origin_efficiency")
    assert origin.dependences == ("space", "regions")
    simulation_routes = mapping_routes("origin_efficiency", "space")
    assert len(simulation_routes) == 2
    assert all("simulation" in route.method for route in simulation_routes)
    assert all(route.output_form == "x" for route in simulation_routes)

    bundle_consumers = tuple(
        route
        for route in get_mapping_catalog().routes
        if any(item.key == "fork_densities_xt" for item in route.inputs)
    )
    assert bundle_consumers
    assert all(route.input("fork_densities_xt").prepared_result_only for route in bundle_consumers)
    assert all(not route.input("fork_densities_xt").accepted_formats for route in bundle_consumers)


def test_route_settings_are_exact_assumptions_for_each_scientific_family() -> None:
    """The page must never show a generic setting which the selected route rejects."""

    representatives = {
        "firing_and_speed_to_timing_simulation": {
            "present": {"sim_number", "stall_rate_per_kb", "seed", "geometry"},
            "absent": {"fit_step", "maxiter", "cfl", "region_names"},
        },
        "timing_and_speed_to_firing_rate": {
            "present": {"fit_step", "maxiter", "neighbourhood_range", "err_threshold"},
            "absent": {"seed", "geometry", "cfl", "region_names"},
        },
        "firing_and_speed_to_replicated_fraction_xt_lightcone": {
            "present": {"resolution", "resolution_time", "time_grid", "max_time"},
            "absent": {"sim_number", "fit_step", "cfl", "region_names"},
        },
        "firing_and_speed_to_coalescence_events_xt_mean_field": {
            "present": {"resolution", "resolution_time", "cfl", "unreplicated_floor"},
            "absent": {"sim_number", "fit_step", "region_names"},
        },
        "initiation_events_to_origin_efficiency_region_integral": {
            "present": {"resolution", "resolution_time", "time_grid", "region_names"},
            "absent": {"sim_number", "fit_step", "cfl", "seed"},
        },
        "fork_density_fields_to_directionality_time_integrated": {
            "present": {"resolution_time", "time_grid"},
            "absent": {"sim_number", "fit_step", "cfl", "region_names"},
        },
    }

    for method, expectations in representatives.items():
        route = mapping_route(method)
        registry_assumptions = set(route.required_assumptions) | set(route.optional_assumptions)
        setting_keys = {setting.key for setting in route.settings}
        assert setting_keys <= registry_assumptions
        assert expectations["present"] <= setting_keys
        assert not expectations["absent"] & setting_keys
        assert not {"target_form", "target_shape", *route.source_keys} & setting_keys
        assert all(route.setting(setting.key) is setting for setting in route.settings)

    simulation = mapping_route("firing_and_speed_to_timing_simulation")
    assert {item.key for item in simulation.settings_for("simulation")} >= {
        "sim_number",
        "stall_rate_per_kb",
        "seed",
    }
    assert simulation.setting("geometry").location == "region"
    assert simulation.setting("geometry").allowed_values == ("linear", "periodic")
    assert simulation.setting("sim_number").max_value == 50_000

    deterministic = mapping_route("firing_and_speed_to_timing_deterministic")
    assert deterministic.setting("perQ").label == "Periodic"

    fit = mapping_route("timing_and_speed_to_firing_rate")
    assert fit.setting("maxiter").category == "fitting"
    assert fit.setting("neighbourhood_range").category == "numerics"
    assert fit.setting("maxiter").max_value is None

    lightcone = mapping_route("firing_and_speed_to_replicated_fraction_xt_lightcone")
    assert all(setting.location in {"region", "derived"} for setting in lightcone.settings)

    mean_field = mapping_route("firing_and_speed_to_coalescence_events_xt_mean_field")
    assert mean_field.setting("cfl").default == 0.8
    assert mean_field.setting("unreplicated_floor").default == 1e-12

    integration = mapping_route("initiation_events_to_origin_efficiency_region_integral")
    assert integration.setting("region_names").category == "integration"
    assert integration.input("regions").key not in {item.key for item in integration.settings}

    # Signature inference must not materialize huge arrays into an input field.
    for route in get_mapping_catalog().routes:
        for setting in route.settings:
            if setting.kind in {"number_list", "number_pair"} and setting.has_default:
                assert setting.default is None or len(setting.default) <= 20


def test_defaulted_feature_assumptions_remain_optional_overrides() -> None:
    defaulted = mapping_route("firing_to_timing_simulation").input("fork_speed")
    assert defaulted.role == "feature_assumption"
    assert defaulted.has_default is True
    assert defaulted.default == 1.4
    assert defaulted.required is False

    explicit = mapping_route("firing_and_speed_to_timing_simulation").input("fork_speed")
    assert explicit.role == "source"
    assert explicit.has_default is False
    assert explicit.required is True

    variable = mapping_route("timing_to_spatial_firing_rate_variable_fork_speed").input(
        "fork_speed"
    )
    assert variable.role == "feature_assumption"
    assert variable.has_default is False
    assert variable.required is True


def test_numeric_setting_defaults_are_finite_and_stall_recovery_is_opt_in() -> None:
    for route in get_mapping_catalog().routes:
        for setting in route.user_settings:
            if (
                setting.kind in {"integer", "number"}
                and setting.has_default
                and isinstance(setting.default, (int, float))
                and not isinstance(setting.default, bool)
            ):
                assert math.isfinite(float(setting.default)), (route.method, setting.key)

    tau = mapping_route("firing_and_speed_to_timing_simulation").setting("tau")
    assert tau.required is False
    assert tau.has_default is True
    assert tau.default is None
    assert tau.nullable is True
    assert tau.min_value is not None and tau.min_value > 0


def test_setting_defaults_follow_exact_delegated_optimizer_and_preserve_inheritance() -> None:
    spatial = mapping_route("timing_and_firing_to_spatial_fork_speed")
    assert spatial.setting("resolution").default == 1.0
    assert spatial.setting("maxiter").default == 10
    assert spatial.setting("initial_fork_speed").default == 1.4
    assert spatial.setting("fit_step").default == .4
    assert spatial.setting("smoothw").default == 100
    assert spatial.setting("neighbourhood_range").default == 2000
    edge_context = spatial.setting("refine_edge_kb")
    assert edge_context.default == mapping_route("timing_and_speed_to_firing_rate").setting("refine_edge_kb").default == 1000
    assert edge_context.label == "Maximum edge context"
    assert edge_context.unit == "kb"
    assert spatial.setting("perQ").default is False
    assert not {"regularization", "tolerance", "spatial_knots", "time_grid", "resolution_time"} & {
        setting.key for setting in spatial.active_settings()
    }

    # A helper's .25 step must not replace the historical temporal wrapper's .003.
    for method in (
        "timing_and_firing_to_time_fork_speed",
        "timing_and_time_firing_to_time_fork_speed",
        "timing_and_firing_to_spacetime_fork_speed",
        "timing_and_general_firing_to_spacetime_fork_speed",
    ):
        speed_route = mapping_route(method)
        assert speed_route.setting("temporal_fit_step").default == .003
        assert speed_route.setting("temporal_maxiter").default == 10
        assert speed_route.setting("initial_fork_speed").default == 1.4
        if speed_route.output_dependence == "spacetime":
            assert speed_route.setting("space_fit_step").default == .4
            assert speed_route.setting("space_maxiter").default == 10
            assert speed_route.setting("spacetime_order").default == "crossed"

    # The XT branch must not inherit the T branch's .35 update or 100 simulations.
    spacetime = mapping_route("timing_and_speed_to_spacetime_firing_rate")
    for key in ("temporal_fit_step", "temporal_maxiter", "temporal_sim_number"):
        assert spacetime.setting(key).default is None
    assert spacetime.setting("spacetime_order").default == "space_then_time"
    projected_fraction = mapping_route("firing_and_speed_to_replicated_fraction_lightcone")
    assert projected_fraction.setting("resolution").default == 1.0
    assert projected_fraction.setting("resolution_time").default == 5.0
    assert all(setting.has_default for setting in projected_fraction.active_settings())


def test_active_settings_follow_shape_and_implemented_scientific_branch() -> None:
    firing = mapping_route("timing_and_speed_to_spacetime_firing_rate")
    weibull = {item.key for item in firing.active_settings("weibull")}
    free = {item.key for item in firing.active_settings("free")}
    assert {"alpha_smooth", "alpha_prior", "weibull_shape"} <= weibull
    assert not {"alpha_smooth", "alpha_prior", "weibull_shape"} & free
    assert (
        not {"temporal_sim_number", "fit_step", "spacetime_update"}
        & weibull
    )
    assert {"temporal_sim_number", "spacetime_update", "tau", "perQ"} <= free
    assert not {"maxiter", "fit_step", "sim_number"} & free
    assert not {"space_maxiter", "temporal_maxiter"} & weibull
    assert (
        next(item for item in firing.active_settings("weibull") if item.key == "maxiter").default
        == 20
    )
    assert next(
        item for item in firing.active_settings("free") if item.key == "spacetime_order"
    ).allowed_values == ("space_then_time", "time_then_space")
    assert firing.shapes_for({"fork_speed": "x"}) == ("weibull",)
    assert firing.shapes_for({"fork_speed": "t"}) == ("weibull",)
    assert firing.shapes_for({"fork_speed": "xt"}) == ("weibull",)

    temporal = mapping_route("timing_and_firing_to_time_fork_speed")
    active = {item.key for item in temporal.active_settings()}
    assert {"temporal_fit_step", "temporal_maxiter", "speed_bounds", "time_grid", "T0"} <= active
    assert (
        not {
            "space_fit_step",
            "smoothw",
            "temporal_sim_number",
            "refine_seed",
            "fit_step",
            "maxiter",
            "tolerance",
            "temporal_knots",
            "regularization",
        }
        & active
    )

    for method in (
        "timing_and_firing_to_spacetime_fork_speed",
        "timing_and_general_firing_to_spacetime_fork_speed",
    ):
        spacetime = mapping_route(method)
        settings = {item.key: item for item in spacetime.active_settings()}
        assert {"space_fit_step", "space_maxiter", "temporal_fit_step", "temporal_maxiter"} <= settings.keys()
        assert settings["spacetime_order"].allowed_values == (
            "crossed", "space_then_time", "time_then_space",
        )
        assert not {
            "maxiter", "fit_step", "sim_number", "temporal_sim_number", "spacetime_sim_number",
            "spacetime_fit_step", "spacetime_space_fit_step", "spacetime_time_fit_step",
            "spacetime_update", "spacetime_kernel_t", "temporal_smoothw", "refine_seed",
            "tolerance", "regularization", "spatial_knots", "temporal_knots",
        } & settings.keys()

    pdf = mapping_route("firing_and_fraction_to_initiation_pdf_xt_normalized")
    assert {item.key for item in pdf.active_settings()} == {
        "resolution",
        "resolution_time",
        "time_grid",
        "x_array",
    }
    profiles = mapping_route("fork_density_profiles_to_directionality")
    assert profiles.active_settings() == ()


def test_periodic_controls_match_actual_forward_and_inverse_geometry_support() -> None:
    for method in (
        "firing_and_speed_to_replicated_fraction_xt_lightcone",
        "firing_and_speed_to_initiation_events_xt_mean_field",
        "timing_and_firing_to_constant_fork_speed",
        "timing_and_time_firing_to_spatial_fork_speed",
        "timing_and_firing_to_spacetime_fork_speed",
    ):
        route = mapping_route(method)
        assert route.supports_periodic() is True
        assert "perQ" in {item.key for item in route.active_settings()}
    for method in (
        "firing_and_speed_to_timing_simulation",
        "firing_and_speed_to_coalescence_events_xt_mean_field",
        "timing_and_firing_to_spatial_fork_speed",
        "timing_and_directionality_to_spatial_fork_speed",
    ):
        assert mapping_route(method).supports_periodic() is True


def test_bundle_axis_settings_remain_validated_but_are_derived_from_stored_coordinates() -> None:
    axis_keys = {"resolution", "resolution_time", "time_grid", "x_array"}
    for route in get_mapping_catalog().routes:
        if not any("bundle" in item.forms for item in route.inputs):
            continue
        active = {item.key: item for item in route.active_settings()}
        original = {item.key: item for item in route.settings if item.key in axis_keys}
        assert set(original) <= set(active)
        for key, setting in original.items():
            assert active[key] == replace(setting, location="derived")
    pdf = mapping_route("fork_density_bundle_to_coalescence_pdf_xt_mean_field")
    assert {item.key for item in pdf.active_settings() if item.location == "derived"} >= axis_keys


def test_rdata_format_and_shape_guidance_matches_the_public_io_contract() -> None:
    expected_source = ("csv", "tsv", "txt", "npy", "npv", "npz", "bedgraph", "bigwig")
    expected_target = ("csv", "tsv", "txt", "npy", "npv", "npz")
    source_formats = rdata_formats(source=True, target=False)
    target_formats = rdata_formats(source=False, target=True)

    assert RDATA_SOURCE_FORMAT_KEYS == expected_source
    assert RDATA_TARGET_FORMAT_KEYS == expected_target
    assert tuple(item.key for item in source_formats) == expected_source
    assert tuple(item.key for item in target_formats) == expected_target
    assert {extension for item in source_formats for extension in item.extensions} == {
        ".csv",
        ".tsv",
        ".txt",
        ".npy",
        ".npv",
        ".npz",
        ".bedgraph",
        ".bdg",
        ".bigwig",
        ".bw",
    }
    assert all(
        extension in RDATA_UPLOAD_ACCEPT for item in source_formats for extension in item.extensions
    )
    assert ".gz" in RDATA_UPLOAD_ACCEPT
    assert "NPY/NPV" in RDATA_UPLOAD_HINT
    assert "bedGraph" in RDATA_UPLOAD_HINT
    assert "BigWig" in RDATA_UPLOAD_HINT
    assert "temporal profiles as row arrays (1 x time)" in RDATA_FORM_HINT
    assert "space-time fields as position-by-time matrices" in RDATA_FORM_HINT
