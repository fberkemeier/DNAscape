"""Scientific drift tests for the Mapping page catalogue boundary."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import replace
from itertools import product
from typing import Any

import pytest

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
    mapping_routes,
    mapping_target,
    mapping_targets,
    rdata_formats,
)
from dnascape import rmap

EXPECTED_PUBLIC_TARGET_COUNT = 24
EXPECTED_IMPLEMENTED_METHOD_COUNT = 81
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
    assert catalog.route_family_count == 66
    assert len(represented_methods) == EXPECTED_IMPLEMENTED_METHOD_COUNT
    assert len(represented_methods) == len(set(represented_methods))
    assert set(represented_methods) == set(catalog.methods)

    for family in catalog.route_families:
        assert family.variants
        assert family.factor_keys
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
        routes = mapping_routes("firing_rate", dependence)
        assert routes
        assert all(route.target_shapes == ("free", "weibull") for route in routes)
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
