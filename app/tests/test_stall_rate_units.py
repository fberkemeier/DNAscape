"""Focused app coverage for resolution-aware physical stall rates."""

from __future__ import annotations

import asyncio
import math
from dataclasses import replace
from typing import Any, cast

import numpy as np
import pytest
from nicegui.elements.number import Number
from nicegui.testing import user_simulation

import dnascape
from app.main import index
from app.mapping_catalog import get_mapping_catalog, mapping_route
from app.simulation_service import (
    ConstantSimulationRequest,
    run_constant_simulation,
    validate_constant_simulation,
)


def _request(**changes: Any) -> ConstantSimulationRequest:
    baseline = ConstantSimulationRequest(
        initiation_rate=0.5,
        fork_speed=2.0,
        resolution_x=5.0,
        resolution_t=1.0,
        region_start_kb=0.0,
        region_end_kb=20.0,
        replicates=2,
        geometry="linear",
        time_window=10.0,
        max_replication_time=100.0,
        seed=1,
    )
    return replace(baseline, **changes)


def test_physical_stall_rate_validation_accepts_density_and_rejects_ambiguity() -> None:
    validated = validate_constant_simulation(_request(stall_rate_per_kb=0.17))
    assert validated.request.stall_rate_per_kb == pytest.approx(0.17)

    # A density is not a probability and therefore has no upper bound of one.
    validate_constant_simulation(_request(stall_rate_per_kb=2.5))

    for value in (-0.01, math.inf, math.nan):
        with pytest.raises(ValueError, match="Stall rate"):
            validate_constant_simulation(_request(stall_rate_per_kb=value))

    with pytest.raises(ValueError, match="cannot be combined"):
        validate_constant_simulation(
            _request(stall_rate=0.2, stall_rate_per_kb=0.17)
        )


def test_physical_stall_rate_is_forwarded_to_rsim_without_legacy_probability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assumptions_seen: dict[str, Any] = {}

    def inspect_rsim(*, source: dict[str, Any], assumptions: dict[str, Any]) -> dict[str, Any]:
        del source
        assumptions_seen.update(assumptions)
        return {
            "replication_timing": np.asarray([4.0, 3.0, 2.0, 1.0]),
            "replication_timing_std": np.zeros(4),
            "rep_times_per_sim": np.ones((2, 4)),
            "origin_efficiency": np.asarray([0.1, 0.2, 0.3, 0.4]),
            "fork_directionality": np.asarray([-0.5, -0.2, 0.2, 0.5]),
            "s_phase_duration": np.asarray([3.0, 4.0]),
            "completion_time": np.asarray([4.0, 5.0]),
            "num_origins": np.asarray([2.0, 3.0]),
            "inter_origin_distances": np.asarray([1.0, 2.0]),
        }

    monkeypatch.setattr(dnascape, "rsim", inspect_rsim)
    run_constant_simulation(_request(stall_rate_per_kb=0.17))

    assert assumptions_seen["stall_rate_per_kb"] == pytest.approx(0.17)
    assert assumptions_seen["resolution_x"] == pytest.approx(5.0)
    assert "stall_rate" not in assumptions_seen


def test_mapping_catalog_exposes_only_the_physical_stall_rate_control() -> None:
    simulation = mapping_route("timing_and_speed_to_initiation_rate")
    setting_keys = {setting.key for setting in simulation.user_settings}

    assert "stall_rate_per_kb" in simulation.optional_assumptions
    assert "stall_rate" in simulation.optional_assumptions  # Python compatibility only.
    assert "stall_rate_per_kb" in setting_keys
    assert "stall_rate" not in setting_keys

    setting = simulation.setting("stall_rate_per_kb")
    assert setting.label == "Stall rate"
    assert setting.unit == "stalls/kb"
    assert setting.category == "simulation"
    assert setting.has_default is True
    assert setting.default == pytest.approx(0.0)
    assert setting.min_value == pytest.approx(0.0)
    assert setting.max_value is None
    assert "selected spatial resolution" in setting.description
    assert "1 - exp" in setting.description

    # Every mapping route backed by the stochastic engine follows the same UI
    # contract, even though the legacy keyword remains accepted by rmap.
    for route in get_mapping_catalog().routes:
        assumptions = set(route.required_assumptions) | set(route.optional_assumptions)
        if "stall_rate" not in assumptions:
            continue
        route_setting_keys = {item.key for item in route.user_settings}
        assert "stall_rate_per_kb" in assumptions, route.method
        assert "stall_rate_per_kb" in route_setting_keys, route.method
        assert "stall_rate" not in route_setting_keys, route.method


def test_simulation_ui_hides_stalling_until_enabled_and_resets_when_disabled() -> None:
    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-simulation").click()
            await user.should_see("Simulate DNA replication")
            user.find("Simulation settings").click()
            await user.should_not_see(marker="simulation-stall-rate-per-kb")
            await user.should_not_see(marker="simulation-stall-probability")
            user.find(marker="simulation-fork-stalling").click()
            await user.should_see(marker="simulation-stall-rate-per-kb")

            resolution = cast(
                Number,
                next(iter(user.find(marker="simulation-resolution-x").elements)),
            )
            rate = cast(
                Number,
                next(iter(user.find(marker="simulation-stall-rate-per-kb").elements)),
            )
            assert rate.value == pytest.approx(0.0)
            assert rate._props.get("max") is None

            assert user.client is not None
            with user.client:
                resolution.set_value(5.0)
                rate.set_value(0.17)

            await user.should_not_see(marker="simulation-stall-probability")
            user.find(marker="simulation-fork-stalling").click()
            await user.should_not_see(marker="simulation-stall-rate-per-kb")
            assert rate.value == 0

    asyncio.run(scenario())
