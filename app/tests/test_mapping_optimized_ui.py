"""Live UI checks for new constrained inverse and observable conversions."""

from __future__ import annotations

import asyncio
from itertools import product
from typing import Any

import numpy as np
import pytest
from nicegui.testing import user_simulation

from app.main import index
from app.mapping_catalog import get_mapping_catalog
from app.mapping_service import MappingRequest, MappingResult, run_mapping
from app.pages import mapping as mapping_page


def _element(user: Any, marker: str) -> Any:
    matches = [
        item for item in user.client.elements.values()
        if marker in item._markers and not item.is_deleted
    ]
    assert len(matches) == 1, (marker, len(matches))
    return matches[0]


async def _set(user: Any, marker: str, value: Any) -> None:
    with user.client:
        _element(user, marker).set_value(value)
    for _ in range(4):
        await asyncio.sleep(0)


def _record_settings(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    latest: dict[str, Any] = {}
    original = mapping_page._render_route_setting_groups

    def recording(route: Any, **kwargs: Any) -> Any:
        controls = original(route, **kwargs)
        latest[kwargs["location"]] = controls
        return controls

    monkeypatch.setattr(mapping_page, "_render_route_setting_groups", recording)
    return latest


def test_static_firing_step_adapts_to_stalling_and_preserves_user_choice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    latest = _record_settings(monkeypatch)

    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-mapping").click()
            await user.should_see(marker="mapping-target", retries=60)
            await _set(user, "mapping-target", "firing_rate")
            await _set(user, "mapping-target-form", "space")
            await _set(user, "mapping-route", "firing_rate--space--replication_timing-and-fork_speed")
            controls = latest["settings"]
            assert mapping_page._read_setting(controls["fit_step"]) == 2.
            await _set(user, "mapping-setting-stall-rate-per-kb", .001)
            assert mapping_page._read_setting(controls["fit_step"]) == .6
            await _set(user, "mapping-setting-stall-rate-per-kb", 0.)
            assert mapping_page._read_setting(controls["fit_step"]) == 2.
            await _set(user, "mapping-setting-fit-step", 1.)
            await _set(user, "mapping-setting-stall-rate-per-kb", .001)
            assert mapping_page._read_setting(controls["fit_step"]) == 1.
            await _set(user, "mapping-setting-stall-rate-per-kb", 0.)
            assert mapping_page._read_setting(controls["fit_step"]) == 1.

    asyncio.run(scenario())


def test_joint_kinetic_forms_keep_one_family_and_only_read_applicable_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    latest = _record_settings(monkeypatch)
    catalog = get_mapping_catalog()
    families = catalog.route_families_for("kinetic_fields", "spacetime")
    assert len(families) == 1

    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-mapping").click()
            await user.should_see(marker="mapping-target", retries=60)
            await _set(user, "mapping-target", "kinetic_fields")
            assert tuple(_element(user, "mapping-route").options) == (families[0].key,)
            for firing_form, speed_form in product(
                ("constant", "space", "time", "spacetime"), repeat=2,
            ):
                await _set(user, "mapping-setting-firing-form", firing_form)
                await _set(user, "mapping-setting-speed-form", speed_form)
                controls = latest["settings"] | latest["region"]
                assert "initial_kinetics" not in controls
                forms = {firing_form, speed_form}
                applicable = {
                    "spatial_knots": bool(forms & {"space", "spacetime"}),
                    "temporal_knots": bool(forms & {"time", "spacetime"}),
                    "regularization": forms != {"constant"},
                }
                read = {}
                for key, control in controls.items():
                    try:
                        read[key] = mapping_page._read_setting(control)
                    except KeyError:
                        pass
                assert read["firing_form"] == firing_form
                assert read["speed_form"] == speed_form
                for key, active in applicable.items():
                    assert controls[key].active == active, (forms, key)
                    assert controls[key].holder.visible == active, (forms, key)
                    assert (key in read) == active, (forms, key)
                    if not active:
                        with pytest.raises(KeyError):
                            mapping_page._read_setting(controls[key])
                assert _element(user, "mapping-route").value == families[0].key

    asyncio.run(scenario())


def test_free_spacetime_firing_update_mode_reads_only_its_visible_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    latest = _record_settings(monkeypatch)
    mixed_values = {
        "spacetime_fit_step": .2, "spacetime_sim_number": 32,
        "spacetime_kernel_t": 2.,
    }
    factorized_values = {"spacetime_order": "space_then_time", "temporal_sim_number": 16}

    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-mapping").click()
            await user.should_see(marker="mapping-target", retries=60)
            await _set(user, "mapping-target-form", "spacetime")
            await _set(user, "mapping-target-shape", "free")
            for mode in ("mixed", "factorized", "mixed", "factorized"):
                await _set(user, "mapping-setting-spacetime-update", mode)
                controls = latest["settings"] | latest["region"]
                assert not {"spacetime_space_fit_step", "spacetime_time_fit_step"} & controls.keys()
                active_values = mixed_values if mode == "mixed" else factorized_values
                for key in mixed_values | factorized_values:
                    active = key in active_values
                    live = controls[key]
                    assert live.active == active, (mode, key)
                    assert live.holder.visible == active, (mode, key)
                    if active:
                        if live.override is not None:
                            await _set(user, f"mapping-setting-{key.replace('_', '-')}-override", True)
                        await _set(user, f"mapping-setting-{key.replace('_', '-')}", active_values[key])
                    else:
                        with pytest.raises(KeyError):
                            mapping_page._read_setting(live)
                # Parameters entered during the previous mode remain in their
                # widgets, but must not leak into the submitted settings.
                read = {}
                for key, live in controls.items():
                    try:
                        read[key] = mapping_page._read_setting(live)
                    except KeyError:
                        pass
                assert read["spacetime_update"] == mode
                assert read["space_maxiter"] == 30
                assert read["temporal_maxiter"] == 30
                for key in mixed_values | factorized_values:
                    assert (key in read) == (key in active_values), (mode, key)
                    if key in active_values:
                        assert read[key] == active_values[key]

            await _set(user, "mapping-target-shape", "weibull")
            assert mapping_page._read_setting(latest["settings"]["maxiter"]) == 20
            assert mapping_page._read_setting(latest["settings"]["alpha_smooth"]) == .1
            await _set(user, "mapping-target-form", "time")
            assert mapping_page._read_setting(latest["settings"]["maxiter"]) == 10

    asyncio.run(scenario())


def test_supplied_fraction_timing_exposes_and_reads_explicit_tail_policies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    latest = _record_settings(monkeypatch)

    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-mapping").click()
            await user.should_see(marker="mapping-target", retries=60)
            await _set(user, "mapping-target", "replication_timing")
            await _set(user, "mapping-route", "replication_timing--space--replicated_fraction")
            tail = _element(user, "mapping-setting-tail-policy")
            assert tuple(tail.options) == ("require_complete", "truncate", "exponential")
            assert tail.value == "require_complete"
            for policy in tail.options:
                await _set(user, "mapping-setting-tail-policy", policy)
                controls = latest["settings"] | latest["region"]
                assert mapping_page._read_setting(controls["tail_policy"]) == policy
                assert {"initial_policy", "monotonic_policy", "tail_tolerance"} <= controls.keys()
                assert not {"fork_speed", "sim_number", "fit_step", "maxiter"} & controls.keys()

    asyncio.run(scenario())


def test_fully_unidentifiable_inverse_maps_and_previews_without_fabricated_points(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results: list[MappingResult] = []
    notifications: list[str] = []

    async def supplied_profiles(key: str, live: Any, **kwargs: Any) -> Any:
        # Supply already parsed fixtures at the upload boundary; the actual
        # request, scientific router, output service and preview remain active.
        values = np.ones((4, 5)) if key == "replicated_fraction" else np.zeros((4, 5))
        return mapping_page._PreparedBatchInput(
            key=key, label=live.spec.label, form="xt", values=(values,),
            prewindowed=(False,), filenames=(f"{key}.npy",),
            detected_forms=("xt",),
        )

    def recording_run(request: MappingRequest, report: Any, cancel: Any) -> MappingResult:
        result = run_mapping(request, report, cancel)
        results.append(result)
        return result

    monkeypatch.setattr(mapping_page, "_read_mapping_input_batch_async", supplied_profiles)
    monkeypatch.setattr(mapping_page, "run_mapping", recording_run)
    monkeypatch.setattr(mapping_page.ui, "notify", lambda message, **kwargs: notifications.append(str(message)))

    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-mapping").click()
            await user.should_see(marker="mapping-target", retries=60)
            await _set(user, "mapping-target-form", "spacetime")
            await _set(user, "mapping-route", "firing_rate--spacetime--initiation_events-and-replicated_fraction")
            await _set(user, "mapping-region-start", 0.)
            await _set(user, "mapping-region-end", 4.)
            assert _element(user, "mapping-setting-invalid-policy").value == "mask"
            user.find(marker="action-map").click()
            for _ in range(200):
                if results and _element(user, "action-map").enabled:
                    break
                await asyncio.sleep(.02)
            await user.should_not_see(marker="mapping-error")
            assert len(results) == 1, (notifications, _element(user, "mapping-status").text, _element(user, "action-map").text)
            assert results[0].diagnostics["unidentifiable_count"] == 20
            assert np.isnan(results[0].outputs["firing_rate"]).all()
            await user.should_see(marker="mapping-result-plot", retries=30)
            options = _element(user, "mapping-result-plot").options
            assert options["series"] == []
            assert options["graphic"][0]["style"]["text"] == "No identifiable values in this interval."

    asyncio.run(scenario())
