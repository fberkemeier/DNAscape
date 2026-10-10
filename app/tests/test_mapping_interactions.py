"""Simulated-user coverage for the registry-driven Mapping controls."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, cast

from nicegui.elements.input import Input
from nicegui.elements.label import Label
from nicegui.elements.number import Number
from nicegui.elements.select import Select
from nicegui.elements.upload import Upload
from nicegui.testing import user_simulation

from app.main import index
from app.mapping_catalog import (
    MappingRoute,
    MappingRouteFamily,
    get_mapping_catalog,
    mapping_route,
    mapping_route_families,
    mapping_targets,
)
from app.pages.mapping import _mapping_route_families_for_ui, _preview_outputs, _setting_value


def _slug(value: str) -> str:
    return "-".join(value.casefold().replace("_", " ").split())


def _select(user: Any, marker: str) -> Select:
    assert user.client is not None
    return cast(
        Select,
        next(
            element
            for element in user.client.elements.values()
            if marker in element._markers and not element.is_deleted
        ),
    )


def _option_keys(control: Select) -> tuple[str, ...]:
    assert isinstance(control.options, dict)
    return tuple(str(value) for value in control.options)


def _family_for_method(method: str) -> MappingRouteFamily:
    return next(
        family for family in get_mapping_catalog().route_families if method in family.methods
    )


async def _set_select(user: Any, marker: str, value: str) -> Select:
    control = _select(user, marker)
    assert user.client is not None
    with user.client:
        control.set_value(value)
    # Mapping inputs live in NiceGUI refreshables. Their fire-and-forget rebuild starts
    # on the next event-loop turn after the selector's synchronous value callback.
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    return control


async def _assert_family_surface(user: Any, family: MappingRouteFamily) -> MappingRoute:
    """Assert the visible factors and default forms for one route family."""

    route_control = _select(user, "mapping-route")
    assert route_control.value == family.key

    await user.should_see(marker="mapping-required-inputs")
    input_grid = next(
        element
        for element in user.find(marker="mapping-required-inputs").elements
        if not element.is_deleted
    )
    live_children = [child for child in input_grid.default_slot.children if not child.is_deleted]
    assert len(live_children) == len(family.inputs)

    selections: dict[str, Any] = {}
    for key in family.factor_keys:
        selections[key] = _select(user, f"mapping-input-{_slug(key)}-form").value
    route = family.resolve(selections)

    for input_spec in family.inputs:
        marker = _slug(input_spec.key)
        await user.should_see(marker=f"mapping-input-{marker}")
        if input_spec.key in family.optional_input_keys:
            await user.should_see(marker=f"mapping-input-{marker}-override")
            await user.should_not_see(marker=f"mapping-input-{marker}-form")
            continue
        form_control = _select(user, f"mapping-input-{marker}-form")
        allowed = family.allowed_forms(input_spec.key)
        assert _option_keys(form_control) == allowed
        assert form_control.value == selections[input_spec.key]
        await user.should_not_see(marker=f"mapping-input-{marker}-form")
        input_type = _select(user, f"mapping-input-{marker}-type")
        expected_types = ({"scalar": "Constant"} if "scalar" in allowed else {}) | (
            {"profile": "Profile"} if set(allowed) & {"x", "t", "xt"} else {}
        )
        if "scalar" in allowed and input_spec.key.removesuffix("_xt") in {
            "firing_rate",
            "fork_speed",
        }:
            expected_types = {
                "scalar": "Constant",
                "constants": "Multiple constants",
                **({"profile": "Profile"} if set(allowed) & {"x", "t", "xt"} else {}),
            }
        if not expected_types:
            expected_types = {form: form.replace("_", " ").title() for form in allowed}
        assert _option_keys(input_type) == tuple(expected_types)

    return route


async def _assert_setting_present(user: Any, route: MappingRoute, key: str) -> None:
    """Assert a route setting is rendered with its exact nullable/default semantics."""

    setting = route.setting(key)
    marker = f"mapping-setting-{_slug(key)}"
    needs_override = (not setting.required) and (
        not setting.has_default or (setting.nullable and setting.default is None)
    )
    if needs_override:
        await user.should_see(marker=f"{marker}-override")
        await user.should_not_see(marker=marker)
    else:
        await user.should_see(marker=marker)


def test_expensive_mapping_settings_start_with_interactive_effort() -> None:
    route = mapping_route("timing_to_joint_spacetime_kinetic_fields")
    defaults = {item.key: _setting_value(item) for item in route.active_settings()}
    assert defaults["maxiter"] == 20
    assert defaults["spatial_knots"] == 12
    assert not {"space_init_sim_number", "joint_sim_number"} & defaults.keys()

    single_simulation = mapping_route("firing_and_speed_to_timing_simulation")
    assert next(_setting_value(item) for item in single_simulation.active_settings()
                if item.key == "sim_number") == 128


def test_spatial_kinetic_targets_only_show_complete_factor_routes() -> None:
    firing = _mapping_route_families_for_ui("firing_rate", "space")
    speed = _mapping_route_families_for_ui("fork_speed", "space")

    assert tuple(family.factor_keys for family in firing) == (
        ("replication_timing", "fork_speed"),
    )
    assert tuple(family.factor_keys for family in speed) == (
        ("replication_timing", "firing_rate"),
        ("replication_timing", "fork_directionality"),
    )
    # The convenience wrapper shares the complete visible factor family.
    assert any(
        "timing_to_firing_constant_fork_speed" in family.methods
        for family in mapping_route_families("firing_rate", "space")
    )


def test_every_fork_speed_route_promises_the_same_fit_diagnostics_as_firing_rate() -> None:
    expected_tail = (
        "replication_timing_fit",
        "comparison:replication_timing_fit",
        "fit_diagnostic",
        "mse_diagnostic",
    )
    for dependence in ("constant", "space", "time", "spacetime"):
        for family in _mapping_route_families_for_ui("fork_speed", dependence):
            if "replication_timing" not in family.factor_keys:
                continue  # A local inverse has no timing observation to compare.
            options = _preview_outputs(family.variants[0])
            assert tuple(options)[-4:] == expected_tail


def test_live_spatial_kinetic_route_picker_keeps_every_complete_route() -> None:
    """Exercise the real target/dependence callbacks with hard-coded UI expectations."""

    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-mapping").click()
            await user.should_see(marker="mapping-route", retries=60)

            async def assert_route(expected_key: str, expected_label: str) -> None:
                route = _select(user, "mapping-route")
                assert route.options == {expected_key: expected_label}
                assert route.value == expected_key

            firing_key = "firing_rate--space--replication_timing-and-fork_speed"
            await assert_route(firing_key, "Replication timing + Fork speed")

            # Move away from and back to Space so the dependence callback, rather
            # than only the initial page construction, must replace every option.
            await _set_select(user, "mapping-target-form", "time")
            await _set_select(user, "mapping-target-form", "space")
            await assert_route(firing_key, "Replication timing + Fork speed")
            await user.should_see(marker="mapping-input-replication-timing")
            await user.should_see(marker="mapping-input-fork-speed")
            await user.should_not_see(marker="mapping-input-firing-rate")

            speed_key = "fork_speed--space--replication_timing-and-firing_rate"
            directionality_key = (
                "fork_speed--space--replication_timing-and-fork_directionality"
            )
            await _set_select(user, "mapping-target", "fork_speed")
            route = _select(user, "mapping-route")
            assert route.options == {
                speed_key: "Replication timing + Firing rate",
                directionality_key: "Replication timing + Fork directionality",
            }
            assert route.value == speed_key
            await _set_select(user, "mapping-target-form", "time")
            await _set_select(user, "mapping-target-form", "space")
            route = _select(user, "mapping-route")
            assert route.options == {
                speed_key: "Replication timing + Firing rate",
                directionality_key: "Replication timing + Fork directionality",
            }
            assert route.value == speed_key
            await user.should_see(marker="mapping-input-replication-timing")
            await user.should_see(marker="mapping-input-firing-rate")
            await user.should_not_see(marker="mapping-input-fork-speed")

            await _set_select(user, "mapping-route", directionality_key)
            await user.should_see(marker="mapping-input-replication-timing")
            await user.should_see(marker="mapping-input-fork-directionality")
            await user.should_not_see(marker="mapping-input-firing-rate")
            await _assert_setting_present(
                user,
                mapping_route("timing_and_directionality_to_spatial_fork_speed"),
                "speed_regularization",
            )

            await _set_select(user, "mapping-target-form", "constant")
            constant_directionality_key = (
                "fork_speed--constant--replication_timing-and-fork_directionality"
            )
            route = _select(user, "mapping-route")
            assert constant_directionality_key in route.options
            await _set_select(user, "mapping-route", constant_directionality_key)
            await user.should_see(marker="mapping-input-fork-directionality")
            await user.should_not_see(marker="mapping-setting-speed-regularization")

    asyncio.run(scenario())


def test_mapping_target_dependence_route_and_form_selectors_follow_the_catalog() -> None:
    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-mapping").click()
            await user.should_see(marker="mapping-target", retries=60)

            target_control = _select(user, "mapping-target")
            preferred_targets = (
                "firing_rate", "fork_speed", "replication_timing",
                "fork_directionality", "fork_densities",
            )
            assert _option_keys(target_control) == (
                *preferred_targets,
                *(
                    target.key for target in mapping_targets()
                    if target.key not in {
                        *preferred_targets, "left_fork_density", "right_fork_density",
                    }
                ),
            )
            assert tuple(target_control.options.values())[:5] == (
                "Firing rate", "Fork speed", "Replication timing",
                "Fork directionality", "Fork densities",
            )
            assert target_control.value == "firing_rate"
            initial_family = _mapping_route_families_for_ui("firing_rate", "space")[0]
            await _assert_family_surface(user, initial_family)

            with user.scope(marker="scientific-action-dock"):
                await user.should_see(marker="action-map")

            for marker, title in (
                ("mapping-heading-target", "Target"),
                ("mapping-heading-dependence", "Dependence"),
                ("mapping-heading-route", "Route"),
                ("mapping-heading-inputs", "Inputs"),
            ):
                heading = cast(Label, next(iter(user.find(marker=marker).elements)))
                assert heading.text == title
                assert "dn-setting-group-title" in heading._classes

            variable_dependences = ("constant", "space", "time", "spacetime")
            for target_key in ("firing_rate", "fork_speed"):
                await _set_select(user, "mapping-target", target_key)
                dependence_control = _select(user, "mapping-target-form")
                assert _option_keys(dependence_control) == variable_dependences
                assert dependence_control.enabled is True

                for dependence in variable_dependences:
                    await _set_select(user, "mapping-target-form", dependence)
                    families = _mapping_route_families_for_ui(target_key, dependence)
                    route_control = _select(user, "mapping-route")
                    route_options = cast(dict[str, str], route_control.options)
                    assert _option_keys(route_control) == tuple(family.key for family in families)
                    assert tuple(route_options.values()) == tuple(
                        family.label for family in families
                    )
                    assert route_control.value == families[0].key
                    assert all(
                        label == " + ".join(family.input(key).label for key in family.factor_keys)
                        for label, family in zip(route_options.values(), families, strict=True)
                    )
                    await _assert_family_surface(user, families[0])

                    if target_key == "firing_rate" and dependence in {"time", "spacetime"}:
                        await user.should_not_see(marker="mapping-target-shape-field")
                        shape_control = _select(user, "mapping-target-shape")
                        assert _option_keys(shape_control) == ("weibull",)
                        assert shape_control.value == "weibull"
                    else:
                        await user.should_not_see(marker="mapping-target-shape-field")

            # Exercise the polymorphic input itself, not merely the catalogue metadata.
            await _set_select(user, "mapping-target", "firing_rate")
            await _set_select(user, "mapping-target-form", "time")
            speed_form = _select(user, "mapping-input-fork-speed-form")
            assert _option_keys(speed_form) == ("scalar", "x", "t", "xt")

            await _set_select(user, "mapping-input-fork-speed-form", "t")
            await user.should_not_see(marker="mapping-input-fork-speed-shape-guidance")
            temporal_upload = next(iter(user.find(marker="source-fork-speed-upload").elements))
            assert ".gz" in temporal_upload._props["accept"]
            assert ".bedgraph" in temporal_upload._props["accept"]
            assert ".bw" in temporal_upload._props["accept"]
            await user.should_not_see("Accepted: CSV, TSV, TXT, NPY, NPV, NPZ, GZ")

            await _set_select(user, "mapping-input-fork-speed-form", "xt")
            await user.should_not_see(marker="mapping-input-fork-speed-shape-guidance")

            # Scientifically fixed spatial targets must never offer Constant.
            for target_key in ("replication_timing", "fork_directionality"):
                await _set_select(user, "mapping-target", target_key)
                dependence_control = _select(user, "mapping-target-form")
                assert _option_keys(dependence_control) == ("space",)
                assert "constant" not in dependence_control.options
                assert dependence_control.value == "space"
                assert dependence_control.enabled is False
                first_family = _mapping_route_families_for_ui(target_key, "space")[0]
                await _assert_family_surface(user, first_family)

    asyncio.run(scenario())


def test_mapping_route_switches_remove_stale_inputs_and_render_special_sources(
    caplog: Any,
) -> None:
    async def scenario() -> None:
        caplog.set_level(logging.ERROR, logger="nicegui")
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-mapping").click()
            await user.should_see(marker="mapping-target", retries=60)

            # A factor-only timing route exposes its two biological inputs.
            await _set_select(user, "mapping-target", "replication_timing")
            timing_method = "directionality_and_speed_to_timing"
            timing_family = _family_for_method(timing_method)
            await _set_select(user, "mapping-route", timing_family.key)
            await _assert_family_surface(user, timing_family)
            await user.should_see(marker="mapping-input-fork-directionality")
            await user.should_see(marker="mapping-input-fork-speed")
            await user.should_not_see(marker="mapping-input-replication-timing")
            directionality_form = _select(user, "mapping-input-fork-directionality-form")
            assert _option_keys(directionality_form) == ("x",)
            assert "scalar" not in directionality_form.options

            # A prepared bundle is one canonical biological factor, not an _xt route label.
            await _set_select(user, "mapping-target", "fork_directionality")
            bundle_method = "fork_density_bundle_to_directionality_time_integrated"
            bundle_family = _family_for_method(bundle_method)
            await _set_select(user, "mapping-route", bundle_family.key)
            await _assert_family_surface(user, bundle_family)
            await user.should_see(marker="mapping-input-fork-densities-prepared")
            await user.should_see(marker="mapping-input-fork-densities-prepared-select")
            await user.should_not_see(marker="source-fork-densities-upload")
            await user.should_not_see(marker="mapping-input-fork-directionality")
            await user.should_not_see(marker="mapping-input-fork-speed")

            # X and XT directional densities share a family and mutually constrain forms.
            field_method = "fork_density_fields_to_directionality_time_integrated"
            density_family = _family_for_method(field_method)
            await _set_select(user, "mapping-route", density_family.key)
            await _assert_family_surface(user, density_family)
            await user.should_not_see(marker="mapping-input-fork-densities")
            for marker in ("right-fork-density", "left-fork-density"):
                await user.should_see(marker=f"mapping-input-{marker}")
                assert _select(user, f"mapping-input-{marker}-form").value == "x"

            await _set_select(user, "mapping-input-right-fork-density-form", "xt")
            left_form = _select(user, "mapping-input-left-fork-density-form")
            assert _option_keys(left_form) == ("x", "xt")
            assert left_form.value == "xt"
            for marker in ("right-fork-density", "left-fork-density"):
                await user.should_not_see(marker=f"mapping-input-{marker}-shape-guidance")

            await _set_select(user, "mapping-input-left-fork-density-form", "x")
            right_form = _select(user, "mapping-input-right-fork-density-form")
            assert _option_keys(right_form) == ("x", "xt")
            assert right_form.value == "x"

            # Target/dependence changes replace the whole family state without stale cards.
            await _set_select(user, "mapping-target", "origin_efficiency")
            await _set_select(user, "mapping-target-form", "regions")
            region_method = "initiation_events_to_origin_efficiency_region_integral"
            region_family = _family_for_method(region_method)
            assert _select(user, "mapping-route").value == region_family.key
            await _assert_family_surface(user, region_family)
            await user.should_see(marker="mapping-input-initiation-events")
            await user.should_see(marker="mapping-input-regions")
            await user.should_see(marker="mapping-regions-source")
            await user.should_see(marker="mapping-regions-editor")
            await user.should_not_see(marker="mapping-input-right-fork-density-xt")
            await user.should_not_see(marker="mapping-input-left-fork-density-xt")

            await _set_select(user, "mapping-regions-source", "upload")
            await user.should_see(marker="mapping-regions-upload")
            region_upload = next(iter(user.find(marker="mapping-regions-upload").elements))
            assert ".npv" in region_upload._props["accept"]
            assert ".gz" in region_upload._props["accept"]
            await user.should_not_see("Accepted: CSV, TSV, TXT, NPY, NPV, NPZ, GZ")

            await _set_select(user, "mapping-input-initiation-events-form", "xt")
            initiation_form = _select(user, "mapping-input-initiation-events-form")
            assert initiation_form.value == "xt"
            assert (
                _family_for_method("initiation_events_xt_to_origin_efficiency_region_integral").key
                == region_family.key
            )

        mapping_errors = [
            record
            for record in caplog.records
            if record.levelno >= logging.ERROR
            and (
                "mapping route family" in record.getMessage().casefold()
                or "mapping" in record.pathname.casefold()
            )
        ]
        assert mapping_errors == []

    asyncio.run(scenario())


def test_mapping_import_controls_follow_rdata_forms_and_track_semantics() -> None:
    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-mapping").click()
            await user.should_see(marker="mapping-target", retries=60)

            # The default timing profile is spatial: one table/array or one genomic track.
            await user.should_not_see(marker="mapping-input-replication-timing-interpretation")
            spatial_upload = next(
                iter(user.find(marker="source-replication-timing-upload").elements)
            )
            assert ".bedgraph" in spatial_upload._props["accept"]
            assert ".bw" in spatial_upload._props["accept"]
            assert spatial_upload._props["multiple"] is True
            await user.should_not_see(marker="mapping-input-replication-timing-track-options")

            spatial_upload = cast(Upload, spatial_upload)
            await spatial_upload.handle_uploads(
                [
                    Upload.SmallFileUpload(
                        "timing.bedgraph",
                        "text/plain",
                        b"chr1 0 1000 1.0\nchr1 1000 2000 2.0\n",
                    )
                ]
            )
            await user.should_see("timing.bedgraph", retries=40)
            await user.should_see(marker="mapping-input-replication-timing-track-options")
            await user.should_not_see(marker="mapping-input-replication-timing-import-options")
            for suffix in (
                "chromosome",
                "start",
                "end",
                "bin-size",
                "aggregation",
                "value-column",
                "preprocessing",
                "missing",
                "interpolation",
                "smooth-window",
                "normalize",
            ):
                await user.should_see(marker=f"mapping-input-replication-timing-track-{suffix}")
            bin_size = cast(
                Number,
                next(
                    iter(
                        user.find(marker="mapping-input-replication-timing-track-bin-size").elements
                    )
                ),
            )
            assert bin_size.value == 1000
            assert bin_size._props["label"] == "Bin size (bp)"
            assert (
                next(
                    iter(user.find(marker="mapping-input-replication-timing-track-start").elements)
                )._props["label"]
                == "Start (bp, inclusive; optional)"
            )
            aggregation = _select(
                user,
                "mapping-input-replication-timing-track-aggregation",
            )
            assert _option_keys(aggregation) == (
                "mean",
                "sum",
                "max",
                "min",
                "coverage",
                "std",
            )
            missing = _select(user, "mapping-input-replication-timing-track-missing")
            assert _option_keys(missing) == ("nan", "zero", "fill")
            await user.should_not_see(marker="mapping-input-replication-timing-track-fill-value")
            await _set_select(
                user,
                "mapping-input-replication-timing-track-missing",
                "fill",
            )
            await user.should_see(marker="mapping-input-replication-timing-track-fill-value")

            # Profile uploads are auto-detected, so the surface accepts every
            # profile format available to the route family.
            await _set_select(user, "mapping-target", "firing_rate")
            await _set_select(user, "mapping-target-form", "time")
            await _set_select(user, "mapping-input-fork-speed-form", "t")
            await user.should_not_see(marker="mapping-input-fork-speed-interpretation")
            temporal_upload = next(iter(user.find(marker="source-fork-speed-upload").elements))
            assert ".bedgraph" in temporal_upload._props["accept"]
            assert ".bw" in temporal_upload._props["accept"]
            assert "multiple" in temporal_upload._props
            await user.should_not_see(marker="mapping-input-fork-speed-shape-guidance")

            # Space-time accepts one complete matrix per independent dataset.
            await _set_select(user, "mapping-input-fork-speed-form", "xt")
            spacetime_upload = next(iter(user.find(marker="source-fork-speed-upload").elements))
            assert ".bedgraph" in spacetime_upload._props["accept"]
            assert ".bw" in spacetime_upload._props["accept"]
            assert "multiple" in spacetime_upload._props
            await user.should_not_see(marker="mapping-input-fork-speed-interpretation")
            await user.should_not_see(marker="mapping-input-fork-speed-multiple-track-guidance")

            # Every XT file may expose all signal columns as one matrix or a
            # specific signal selection, independently of the other datasets.
            spacetime_upload = cast(Upload, spacetime_upload)
            await spacetime_upload.handle_uploads(
                [
                    Upload.SmallFileUpload(
                        "speed_t0.csv",
                        "text/csv",
                        b"position,t0,t1\n0,1,10\n1,2,20\n",
                    )
                ]
            )
            await user.should_see(marker="mapping-input-fork-speed-detected-data")
            value_selectors = [
                cast(Select, element)
                for element in user.find(marker="mapping-input-fork-speed-value-column").elements
                if not element.is_deleted
            ]
            assert len(value_selectors) == 1
            assert value_selectors[0].value == "__all__"
            assert user.client is not None
            with user.client:
                value_selectors[0].set_value("t1")

            await spacetime_upload.handle_uploads(
                [
                    Upload.SmallFileUpload(
                        "speed_t1.csv",
                        "text/csv",
                        b"position,t0,t1\n0,3,30\n1,4,40\n",
                    )
                ]
            )
            value_selectors: list[Select] = []
            for _ in range(100):
                value_selectors = [
                    cast(Select, element)
                    for element in user.find(
                        marker="mapping-input-fork-speed-value-column"
                    ).elements
                    if not element.is_deleted
                ]
                if len(value_selectors) == 2:
                    break
                await asyncio.sleep(0.01)
            assert len(value_selectors) == 2
            selections_by_file = {
                str(selector._props["data-upload-name"]): selector.value
                for selector in value_selectors
            }
            assert selections_by_file == {
                "speed_t0.csv": "t1",
                "speed_t1.csv": "__all__",
            }
            for selector in value_selectors:
                options = (
                    tuple(selector.options)
                    if isinstance(selector.options, list)
                    else tuple(selector.options.keys())
                )
                assert "__all__" in options

            # Even a one-factor route accepts several complete matrix datasets.
            await _set_select(user, "mapping-target", "initiation_pdf")
            await _set_select(user, "mapping-target-form", "spacetime")
            single_input_family = _family_for_method(
                "initiation_events_xt_to_initiation_pdf_xt_normalized"
            )
            await _set_select(user, "mapping-route", single_input_family.key)
            single_upload = next(
                element
                for element in user.find(marker="source-initiation-events-upload").elements
                if not element.is_deleted
            )
            assert "max-files" not in single_upload._props
            assert single_upload._props["multiple"] is True
            assert ".bedgraph" not in single_upload._props["accept"]
            assert ".bw" not in single_upload._props["accept"]
            await user.should_not_see(
                marker="mapping-input-initiation-events-multiple-track-guidance"
            )

    asyncio.run(scenario())


def test_compatible_target_and_output_changes_retain_uploaded_profiles() -> None:
    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-mapping").click()
            await user.should_see(marker="source-replication-timing-upload", retries=60)
            uploader = cast(
                Upload,
                next(iter(user.find(marker="source-replication-timing-upload").elements)),
            )
            await uploader.handle_uploads(
                [
                    Upload.SmallFileUpload(
                        "timing.csv",
                        "text/csv",
                        b"timing\n8\n7\n9\n",
                    )
                ]
            )
            await user.should_see("timing.csv", retries=20)
            assert _select(user, "mapping-input-replication-timing-form").value == "x"

            # All firing-rate output forms use the same timing profile.
            await _set_select(user, "mapping-target-form", "constant")
            await user.should_see("timing.csv", retries=20)
            await user.should_see(
                marker="mapping-input-replication-timing-detected-data",
                retries=20,
            )

            # Spatial fork-speed fitting also consumes that timing profile.
            await _set_select(user, "mapping-target", "fork_speed")
            await user.should_see("timing.csv", retries=20)
            assert _select(user, "mapping-input-replication-timing-form").value == "x"
            assert _select(user, "mapping-target-form").value == "space"

    asyncio.run(scenario())


def test_mapping_settings_refresh_to_only_assumptions_accepted_by_the_route() -> None:
    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-mapping").click()
            await user.should_see(marker="mapping-target", retries=60)
            user.find("Region").click()
            user.find("Mapping settings").click()

            # Default route is a fit: expose fit controls, not unrelated PDE/simulation ones.
            route = mapping_route("timing_and_speed_to_firing_rate")
            for key in ("fit_step", "maxiter", "neighbourhood_range", "err_threshold"):
                await _assert_setting_present(user, route, key)
            for marker in ("seed", "geometry", "cfl", "region-names"):
                await user.should_not_see(marker=f"mapping-setting-{marker}")
            for category, title in (
                ("numerics", "Numerics"),
                ("fitting", "Fitting"),
                ("simulation", "Simulation"),
            ):
                group = next(iter(user.find(marker=f"mapping-settings-{category}").elements))
                header = cast(Label, group.default_slot.children[0])
                assert header.text == title
                assert "dn-setting-group-title" in header._classes

            await _set_select(user, "mapping-target", "replication_timing")
            timing_family = _family_for_method("firing_and_speed_to_timing_simulation")
            assert _select(user, "mapping-route").value == timing_family.key
            # Scalar inputs resolve to the deterministic implementation; a temporal
            # firing rate can only resolve to the registered simulation method.
            await user.should_see(marker="mapping-setting-neighbourhood-range")
            await user.should_not_see(marker="mapping-setting-seed")
            await _set_select(user, "mapping-input-firing-rate-form", "t")
            route = mapping_route("firing_and_speed_to_timing_simulation")
            for key in ("sim_number", "stall_rate_per_kb", "seed", "geometry"):
                await _assert_setting_present(user, route, key)
            await _assert_setting_present(user, route, "tau")
            await user.should_see("Allow stall recovery")
            await user.should_not_see("Infinity")
            for marker in ("fit-step", "maxiter", "cfl", "region-names"):
                await user.should_not_see(marker=f"mapping-setting-{marker}")

            await _set_select(user, "mapping-target", "replicated_fraction")
            await _set_select(user, "mapping-target-form", "spacetime")
            fraction_family = _family_for_method(
                "firing_and_speed_to_replicated_fraction_xt_lightcone"
            )
            assert _select(user, "mapping-route").value == fraction_family.key
            route = mapping_route("firing_and_speed_to_replicated_fraction_xt_lightcone")
            for key in ("resolution", "resolution_time", "time_grid", "max_time"):
                await _assert_setting_present(user, route, key)
            for marker in ("sim-number", "fit-step", "cfl", "region-names"):
                await user.should_not_see(marker=f"mapping-setting-{marker}")

            await _set_select(user, "mapping-target", "coalescence_events")
            await _set_select(user, "mapping-target-form", "spacetime")
            coalescence_family = _family_for_method(
                "firing_and_speed_to_coalescence_events_xt_mean_field"
            )
            assert _select(user, "mapping-route").value != coalescence_family.key
            await _set_select(user, "mapping-route", coalescence_family.key)
            route = mapping_route("firing_and_speed_to_coalescence_events_xt_mean_field")
            await _assert_setting_present(user, route, "cfl")
            await _assert_setting_present(user, route, "unreplicated_floor")
            await user.should_not_see(marker="mapping-setting-fit-step")

            await _set_select(user, "mapping-target", "origin_efficiency")
            await _set_select(user, "mapping-target-form", "regions")
            region_family = _family_for_method(
                "initiation_events_to_origin_efficiency_region_integral"
            )
            assert _select(user, "mapping-route").value == region_family.key
            route = mapping_route("initiation_events_to_origin_efficiency_region_integral")
            await _assert_setting_present(user, route, "region_names")
            await user.should_not_see(marker="mapping-setting-cfl")
            await user.should_not_see(marker="mapping-setting-fit-step")

            await _set_select(user, "mapping-target", "fork_directionality")
            density_family = _family_for_method(
                "fork_density_fields_to_directionality_time_integrated"
            )
            await _set_select(user, "mapping-route", density_family.key)
            await _set_select(user, "mapping-input-right-fork-density-form", "xt")
            route = mapping_route("fork_density_fields_to_directionality_time_integrated")
            await _assert_setting_present(user, route, "resolution_time")
            await _assert_setting_present(user, route, "time_grid")
            for marker in ("sim-number", "fit-step", "cfl", "region-names"):
                await user.should_not_see(marker=f"mapping-setting-{marker}")

    asyncio.run(scenario())


def test_defaulted_speed_is_visible_in_one_complete_route_in_the_ui() -> None:
    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-mapping").click()
            await user.should_see(marker="mapping-target", retries=60)
            await _set_select(user, "mapping-target", "replication_timing")
            family = _family_for_method("firing_to_timing_simulation")
            assert family is _family_for_method("firing_and_speed_to_timing_simulation")
            await _set_select(user, "mapping-route", family.key)

            await user.should_not_see(marker="mapping-input-fork-speed-override")
            await user.should_see(marker="mapping-input-fork-speed-type")
            await user.should_not_see(marker="mapping-input-fork-speed-form")
            await user.should_see(marker="mapping-input-fork-speed-constant")
            constant = cast(
                Number,
                next(iter(user.find(marker="mapping-input-fork-speed-constant").elements)),
            )
            assert constant.value == 1.4

            calculation = _select(user, "mapping-calculation")
            assert calculation.options == {
                "deterministic": "Deterministic", "simulation": "Simulation",
            }
            assert calculation.value == "deterministic"
            await user.should_not_see(marker="mapping-setting-sim-number")
            await _set_select(user, "mapping-calculation", "simulation")
            await _assert_setting_present(user, mapping_route("firing_and_speed_to_timing_simulation"), "sim_number")
            await _set_select(user, "mapping-calculation", "deterministic")
            await user.should_not_see(marker="mapping-setting-sim-number")

    asyncio.run(scenario())


def test_firing_target_shape_only_exposes_compatible_weibull_settings() -> None:
    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-mapping").click()
            await user.should_see(marker="mapping-target", retries=60)
            await _set_select(user, "mapping-target", "firing_rate")
            await _set_select(user, "mapping-target-form", "time")
            family = _family_for_method("timing_and_speed_to_time_firing_rate")
            await _set_select(user, "mapping-route", family.key)
            user.find("Mapping settings").click()

            shape = _select(user, "mapping-target-shape")
            assert shape.value == "weibull"
            for marker in ("weibull-shape", "weibull-delay", "fit-weibull-shape"):
                await user.should_see(marker=f"mapping-setting-{marker}")

            await user.should_not_see(marker="mapping-target-shape-field")

            for form in ("t", "xt"):
                await _set_select(user, "mapping-input-fork-speed-form", form)
                assert shape.options == {"weibull": "Weibull"}
                assert shape.value == "weibull"
                await user.should_see(marker="mapping-setting-weibull-shape")
            await _set_select(user, "mapping-input-fork-speed-form", "x")
            assert shape.options == {"weibull": "Weibull"}

    asyncio.run(scenario())


def test_mapping_preview_and_raw_filename_follow_the_selected_output() -> None:
    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-mapping").click()
            await user.should_see(marker="mapping-preview-output", retries=60)

            output = _select(user, "mapping-preview-output")
            assert user.client is not None
            raw_name = cast(
                Input,
                next(
                    element
                    for element in user.client.elements.values()
                    if "mapping-raw-filename" in element._markers
                    and not element.is_deleted
                ),
            )
            await user.should_not_see(marker="raw-output-row")
            await user.should_not_see("Title")
            y_label = cast(
                Input,
                next(iter(user.find(marker="mapping-preview-y-label").elements)),
            )
            x_label = cast(
                Input,
                next(iter(user.find(marker="mapping-preview-x-label").elements)),
            )
            view = _select(user, "mapping-view")
            assert output.value == "firing_rate"
            assert output.enabled is True
            assert isinstance(output.options, dict)
            assert output.options["firing_rate"] == "Firing rate (target)"
            assert _option_keys(output) == (
                "firing_rate",
                "replication_timing_fit",
                "comparison:replication_timing_fit",
                "fit_diagnostic",
                "mse_diagnostic",
            )
            assert raw_name.value == "rmap_fr"

            await _set_select(
                user,
                "mapping-preview-output",
                "comparison:replication_timing_fit",
            )
            assert view.enabled is False
            assert x_label.value == "Position (kb)"
            assert y_label.value == "Replication timing"
            assert raw_name.value == "rmap_rt-cmp"

            await _set_select(user, "mapping-target", "replicated_fraction")
            await _set_select(user, "mapping-target-form", "spacetime")
            assert output.value == "replicated_fraction_xt"
            assert raw_name.value == "rmap_rf-xt"
            assert x_label.value == "Position (kb)"
            assert y_label.value == "Time (min)"
            assert view.value == "Default"

            await _set_select(user, "mapping-target-form", "time")
            assert output.value == "replicated_fraction"
            assert raw_name.value == "rmap_rf"
            assert x_label.value == "Position (kb)"
            assert y_label.value == "Time (min)"
            assert view.value == "Default"

            await _set_select(user, "mapping-target", "kinetic_fields")
            assert _option_keys(output) == (
                "firing_rate",
                "fork_speed",
                "replication_timing_fit",
                "comparison:replication_timing_fit",
                "fit_diagnostic",
                "mse_diagnostic",
            )
            assert output.enabled is True
            await _set_select(user, "mapping-preview-output", "fit_diagnostic")
            assert raw_name.value == "rmap_res"
            assert view.value == "Default"

            await _set_select(user, "mapping-preview-output", "mse_diagnostic")
            assert raw_name.value == "rmap_mse"

    asyncio.run(scenario())
