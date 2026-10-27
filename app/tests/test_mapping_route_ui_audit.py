"""Walk the live Mapping surface across every available route family and form."""

from __future__ import annotations

import asyncio
import itertools
from typing import Any, cast

import pytest
from nicegui.elements.select import Select
from nicegui.testing import user_simulation

from app.main import index
from app.mapping_catalog import DataForm, MappingRoute, MappingRouteFamily, get_mapping_catalog
from app.pages import mapping as mapping_page


def _slug(value: str) -> str:
    return "-".join(value.casefold().replace("_", " ").split())


def _element(user: Any, marker: str) -> Any:
    matches = [
        element for element in user.client.elements.values()
        if marker in element._markers and not element.is_deleted
    ]
    assert len(matches) == 1, (marker, len(matches))
    return matches[0]


async def _set(user: Any, marker: str, value: Any) -> None:
    control = _element(user, marker)
    with user.client:
        control.set_value(value)
    # Refreshable inputs and both settings panels settle on following loop turns.
    for _ in range(4):
        await asyncio.sleep(0)


async def _set_input_form(user: Any, key: str, form: DataForm) -> None:
    # Users choose Profile before upload detection supplies x/t/xt. Changing
    # the hidden detected-form control directly from Constant would bypass that
    # sequence and race the input-mode callback's initial profile choice.
    mode = "profile" if form in {"x", "t", "xt"} else form
    await _set(user, f"mapping-input-{_slug(key)}-type", mode)
    await _set(user, f"mapping-input-{_slug(key)}-form", form)


def _accepted_form_combinations(family: MappingRouteFamily) -> list[dict[str, DataForm]]:
    combinations: dict[tuple[DataForm, ...], dict[str, DataForm]] = {}
    for variant in family.variants:
        for forms in itertools.product(*(
            family.variant_input(variant, key).forms for key in family.input_keys
        )):
            combinations.setdefault(forms, dict(zip(family.input_keys, forms, strict=True)))
    return list(combinations.values())


def _expected_calculations(
    family: MappingRouteFamily, forms: dict[str, DataForm],
) -> dict[str, MappingRoute]:
    # Calculate expected dispatch directly from declared variants, without using
    # the page's calculation selector implementation.
    accepted = [
        route for route in family.variants
        if all(form in family.variant_input(route, key).forms for key, form in forms.items())
    ]
    choices: dict[str, MappingRoute] = {}
    for route in accepted:
        if family.target_key == "replication_timing":
            solver = "simulation" if route.method.endswith("_simulation") else "deterministic"
        else:
            solver = "automatic"
        choices.setdefault(solver, route)
    return choices


def test_all_mapping_targets_families_forms_and_shapes_render_exact_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise real selector callbacks, refreshables, and their settings widgets."""

    rendered: dict[str, tuple[MappingRoute, dict[str, Any]]] = {}
    original_render = mapping_page._render_route_setting_groups

    def observe_render(route: MappingRoute, **kwargs: Any) -> dict[str, Any]:
        controls = original_render(route, **kwargs)
        rendered[kwargs["location"]] = route, controls
        return controls

    monkeypatch.setattr(mapping_page, "_render_route_setting_groups", observe_render)
    catalogue = get_mapping_catalog()
    visited_families: set[str] = set()
    visited_targets: set[str] = set()
    visited_cases: set[tuple[str, tuple[tuple[str, DataForm], ...], str | None]] = set()
    expected_cases: set[tuple[str, tuple[tuple[str, DataForm], ...], str | None]] = set()

    async def assert_surface(
        user: Any, family: MappingRouteFamily, route: MappingRoute,
        forms: dict[str, DataForm], shape: str | None,
    ) -> None:
        assert _element(user, "mapping-route").value == family.key
        grid = _element(user, "mapping-required-inputs")
        cards = [child for child in grid.default_slot.children if not child.is_deleted]
        assert len(cards) == len(family.inputs), (family.key, forms)
        for key, form in forms.items():
            _element(user, f"mapping-input-{_slug(key)}")
            control = cast(Select, _element(user, f"mapping-input-{_slug(key)}-form"))
            assert control.value == form, (family.key, key, forms)
            assert tuple(control.options) == family.allowed_forms(key)

        expected_settings = {
            setting.key: setting for setting in route.active_settings(shape, forms)
            if setting.location in {"region", "settings"}
        }
        if route.target == "initiation_rate" and shape == "weibull":
            assert "fit_step" not in expected_settings
            if "replication_timing" in family.factor_keys:
                assert {"stall_rate_per_kb", "tau", "perQ"} <= expected_settings.keys()
            if route.output_dependence == "time":
                assert "space_maxiter" not in expected_settings
        if route.target == "fork_speed" and route.output_dependence == "time":
            assert "refine_seed" not in expected_settings
        if route.target == "initiation_rate" and "spacetime_order" in expected_settings:
            assert expected_settings["spacetime_order"].allowed_values == (
                "space_then_time", "time_then_space",
            )
        actual_controls: dict[str, Any] = {}
        for location in ("region", "settings"):
            actual_route, controls = rendered[location]
            assert actual_route.method == route.method, (family.key, forms, location)
            assert set(controls) == {
                key for key, setting in expected_settings.items() if setting.location == location
            }, (route.method, shape, location)
            actual_controls.update(controls)

        expected_markers = {f"mapping-setting-{_slug(key)}" for key in expected_settings}
        assert not route.single_phase_fit_aliases.keys() & actual_controls.keys()
        if route.target == "initiation_rate" and route.output_dependence in {"constant", "space"}:
            key = "deviation_threshold" if route.output_dependence == "space" else "neighbourhood_range"
            if key in actual_controls:
                expected_active = forms.get("fork_speed") not in {"x", "t", "xt"}
                assert actual_controls[key].active is expected_active, (route.method, forms, key)
                assert actual_controls[key].holder.visible is expected_active, (route.method, forms, key)
        if "refine_edge_kb" in actual_controls and "perQ" in actual_controls:
            await _set(user, "mapping-setting-perq", True)
            edge = actual_controls["refine_edge_kb"]
            assert edge.active is False, (route.method, forms, shape)
            assert edge.holder.visible is False, (route.method, forms, shape)
            with pytest.raises(KeyError):
                mapping_page._read_setting(edge)
            await _set(user, "mapping-setting-perq", False)
            assert edge.active is True
            assert edge.holder.visible is True
        for key, live in actual_controls.items():
            assert live.setting == expected_settings[key]
            assert not live.control.is_deleted
            if live.override is not None:
                expected_markers.add(f"mapping-setting-{_slug(key)}-override")
            if live.setting.location == "settings":
                # Every dynamic route shares the same three-section layout;
                # numerical options must never return as a separate section.
                ancestor = live.holder
                section_markers: set[str] = set()
                while ancestor is not None:
                    section_markers.update(ancestor._markers)
                    ancestor = ancestor.parent_slot.parent if ancestor.parent_slot else None
                assert section_markers & {
                    "mapping-settings-fitting", "mapping-settings-simulation", "mapping-settings-advanced",
                }, (route.method, key)
                if "mapping-settings-advanced" in section_markers:
                    assert (live.override is not None) == (key == "tau"), (route.method, key)
                if key.endswith("sim_number"):
                    assert live.control.value == 100, (route.method, key)
            if live.automatic:
                with pytest.raises(KeyError):
                    mapping_page._read_setting(live)
        assert not any(
            marker in {"mapping-settings-numerics", "mapping-settings-integration"}
            for element in user.client.elements.values() if not element.is_deleted
            for marker in element._markers
        )
        live_markers = [
            marker for element in user.client.elements.values() if not element.is_deleted
            for marker in element._markers if marker.startswith("mapping-setting-")
        ]
        assert set(live_markers) == expected_markers, (route.method, shape)
        assert len(live_markers) == len(set(live_markers)), "Stale duplicate settings survived refresh"

    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-mapping").click()
            await user.should_see(marker="mapping-target", retries=60)
            target_options = _element(user, "mapping-target").options
            assert set(target_options) == {
                target.key for target in catalogue.targets
                if target.key not in {"left_fork_density", "right_fork_density"}
            }
            for target_key in target_options:
                target = catalogue.target(target_key)
                await _set(user, "mapping-target", target.key)
                visited_targets.add(target.key)
                assert tuple(_element(user, "mapping-target-form").options) == target.dependences
                for dependence in target.dependences:
                    await _set(user, "mapping-target-form", dependence)
                    families = catalogue.route_families_for(target.key, dependence)
                    route_options = _element(user, "mapping-route").options
                    assert tuple(route_options) == tuple(family.key for family in families)
                    assert len(set(route_options.values())) == len(route_options)
                    for family in families:
                        await _set(user, "mapping-route", family.key)
                        visited_families.add(family.key)
                        for forms in _accepted_form_combinations(family):
                            for key, form in forms.items():
                                await _set_input_form(user, key, form)
                            calculations = _expected_calculations(family, forms)
                            calculation_control = _element(user, "mapping-calculation")
                            assert tuple(calculation_control.options) == tuple(calculations)
                            assert calculation_control.visible == (len(calculations) > 1)
                            for calculation, route in calculations.items():
                                await _set(user, "mapping-calculation", calculation)
                                shapes = route.shapes_for(forms)
                                shape_control = _element(user, "mapping-target-shape")
                                assert tuple(shape_control.options) == shapes, (route.method, forms)
                                assert _element(user, "mapping-target-shape-field").visible == (len(shapes) > 1)
                                for shape in shapes or (None,):
                                    if shape is not None:
                                        await _set(user, "mapping-target-shape", shape)
                                    expected_case = route.method, tuple(forms.items()), shape
                                    expected_cases.add(expected_case)
                                    await assert_surface(user, family, route, forms, shape)
                                    visited_cases.add(expected_case)

    asyncio.run(scenario())
    expected_targets = {
        target.key for target in catalogue.targets
        if target.key not in {"left_fork_density", "right_fork_density"}
    }
    assert visited_targets == expected_targets
    assert visited_families == {
        family.key for family in catalogue.route_families
        if family.target_key in expected_targets
    }
    assert visited_cases == expected_cases
    assert len(visited_cases) > sum(
        route.target_key in expected_targets for route in catalogue.routes
    )


def test_mapping_keeps_user_target_shape_when_input_form_still_supports_it() -> None:
    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-mapping").click()
            await user.should_see(marker="mapping-target", retries=60)
            await _set(user, "mapping-target-form", "spacetime")
            await _set(user, "mapping-target-shape", "weibull")
            await _set_input_form(user, "fork_speed", "x")
            assert _element(user, "mapping-target-shape").value == "weibull"
            assert _element(user, "mapping-setting-stall-rate-per-kb").visible
            await _set_input_form(user, "fork_speed", "xt")
            assert _element(user, "mapping-target-shape").value == "weibull"
            await _set_input_form(user, "fork_speed", "scalar")
            assert _element(user, "mapping-target-shape").value == "weibull"

    asyncio.run(scenario())
