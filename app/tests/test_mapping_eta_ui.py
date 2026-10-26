"""Live Mapping ETA remains visible independently of worker callbacks."""

from __future__ import annotations

import asyncio
import time
from threading import Event
from typing import Any

import numpy as np
import pytest
from nicegui.testing import user_simulation

from app.main import index
from app.mapping_eta import MappingEta
from app.mapping_service import MappingResult, run_mapping
from app.pages import mapping as mapping_page


def _element(user: Any, marker: str) -> Any:
    return next(
        element for element in user.client.elements.values()
        if marker in element._markers and not element.is_deleted
    )


async def _until(predicate: Any) -> None:
    for _ in range(300):
        if predicate():
            return
        await asyncio.sleep(.02)
    raise AssertionError("Mapping did not reach the expected state")


@pytest.mark.parametrize("mode", ["quiet", "progress", "mse", "batch", "stop", "error", "disconnect"])
def test_eta_lifecycle_through_real_mapping_action(monkeypatch, mode):
    """Pause the actual conversion worker to exercise long callbacks reliably."""
    clock = [0.0]
    allow_preparation = Event()
    release = [Event(), Event()]
    entered: list[int] = []
    results: list[MappingResult] = []
    count = 2 if mode == "batch" else 1

    async def supplied_profiles(key: str, live: Any, **_kwargs: Any) -> Any:
        while not allow_preparation.is_set():
            await asyncio.sleep(.01)
        values = np.full((4, 5), .5 if key == "replicated_fraction" else .01)
        return mapping_page._PreparedBatchInput(
            key=key, label=live.spec.label, form="xt", values=(values,) * count,
            prewindowed=(False,) * count,
            filenames=tuple(f"{key}_{index}.npy" for index in range(count)),
            detected_forms=("xt",) * count,
        )

    def paused_mapping(request, report, cancel):
        case_index = len(entered)
        paused = False

        def delayed_report(value, message):
            nonlocal paused
            report(value, message)
            if value == .04 and not paused:
                paused = True
                entered.append(case_index)
                if mode == "progress":
                    report(.49, "Mapping... 1/2 - ETA 00:00:20 - MSE 1e-2")
                elif mode == "mse":
                    report(.49, "Mapping... 1/2 - ETA 00:00:20 - MSE 1.234e-02")
                    report(.50, "Fitting fork speed... iteration 1/2")
                deadline = time.monotonic() + 15
                while not release[case_index].wait(.01):
                    cancel()
                    if time.monotonic() > deadline:
                        raise TimeoutError("test did not release Mapping")
                if mode == "error":
                    raise ValueError("deliberate worker failure")

        result = run_mapping(request, delayed_report, cancel)
        results.append(result)
        return result

    monkeypatch.setattr(mapping_page, "MappingEta", lambda: MappingEta(clock=lambda: clock[0]))
    monkeypatch.setattr(mapping_page, "_read_mapping_input_batch_async", supplied_profiles)
    monkeypatch.setattr(mapping_page, "run_mapping", paused_mapping)

    async def scenario():
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-mapping").click()
            await user.should_see(marker="mapping-target", retries=60)
            with user.client:
                _element(user, "mapping-target-form").set_value("spacetime")
            await asyncio.sleep(0)
            with user.client:
                _element(user, "mapping-route").set_value(
                    "firing_rate--spacetime--initiation_events-and-replicated_fraction"
                )
            await asyncio.sleep(0)
            with user.client:
                _element(user, "mapping-region-start").set_value(0.)
                _element(user, "mapping-region-end").set_value(4.)
            user.find(marker="action-map").click()
            await _until(lambda: _element(user, "mapping-eta").visible)
            label = _element(user, "mapping-eta")
            line = _element(user, "mapping-status-line")
            assert line.default_slot.children == [_element(user, "mapping-status"), label]
            timer = _element(user, "mapping-eta-heartbeat")
            assert "ETA estimating" in label.text
            clock[0] = 10
            await _until(lambda: "Elapsed 00:00:10" in label.text)
            assert "ETA estimating" in label.text
            allow_preparation.set()
            await _until(lambda: len(entered) == 1)
            if mode == "progress":
                await _until(lambda: "00:00:20" in label.text)
                clock[0] = 15
                await _until(lambda: "00:00:15" in label.text)
            elif mode == "mse":
                await _until(lambda: "iteration 1/2 · MSE 1.234e-02" in _element(user, "mapping-status").text)
            else:
                clock[0] = 20
                await _until(lambda: "Elapsed 00:00:20" in label.text)
                assert "ETA estimating" in label.text
            assert "ETA" not in _element(user, "mapping-status").text
            if mode == "stop":
                user.find(marker="action-map").click()
            elif mode == "disconnect":
                # NiceGUI invokes these same normalized handlers on disconnect.
                for handler in user.client.disconnect_handlers:
                    user.client.safe_invoke(handler)
                await _until(lambda: timer.is_deleted or timer._is_canceled)
                # ETA cleanup must not cancel a scientific job merely because
                # the browser connection was interrupted.
                release[0].set()
            else:
                release[0].set()
            if mode == "batch":
                await _until(lambda: len(entered) == 2)
                clock[0] = 22
                await _until(lambda: "00:00:08 (batch)" in label.text)
                release[1].set()
            await _until(lambda: _element(user, "action-map").text == "Map")
            assert label.visible
            expected = (
                "stopped" if mode == "stop"
                else "unavailable" if mode == "error" else "00:00:00"
            )
            assert label.text.startswith(f"ETA {expected}")
            assert timer.is_deleted or timer._is_canceled
            final_text = label.text
            clock[0] += 100
            await asyncio.sleep(.25)
            assert label.text == final_text
            assert len(results) == (0 if mode in {"stop", "error"} else count)

    try:
        asyncio.run(scenario())
    finally:
        allow_preparation.set()
        for event in release:
            event.set()


def test_spacetime_speed_sequential_settings_follow_order(monkeypatch):
    rendered = {}
    original = mapping_page._render_route_setting_groups

    def record_controls(route, **kwargs):
        controls = original(route, **kwargs)
        rendered[kwargs["location"]] = controls
        return controls

    monkeypatch.setattr(mapping_page, "_render_route_setting_groups", record_controls)

    async def scenario():
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-mapping").click()
            await user.should_see(marker="mapping-target", retries=60)
            with user.client:
                _element(user, "mapping-target").set_value("fork_speed")
            await asyncio.sleep(0)
            with user.client:
                _element(user, "mapping-target-form").set_value("spacetime")
            await asyncio.sleep(0)
            for order in ("crossed", "space_then_time", "time_then_space", "crossed"):
                with user.client:
                    _element(user, "mapping-setting-spacetime-order").set_value(order)
                await asyncio.sleep(0)
                controls = rendered["settings"]
                for key in ("neighbourhood_range", "refine_verbose"):
                    if key not in controls:
                        continue
                    active = order != "crossed"
                    assert controls[key].active == active
                    assert controls[key].holder.visible == active
                    if not active:
                        with pytest.raises(KeyError):
                            mapping_page._read_setting(controls[key])

    asyncio.run(scenario())
