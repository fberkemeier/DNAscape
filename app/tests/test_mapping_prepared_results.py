"""Live reuse of a mapped directional fork-density bundle."""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np
import pytest
from nicegui.testing import user_simulation

from app.main import index
from app.mapping_service import MappingRequest, MappingResult, run_mapping


def test_mapping_reuses_prepared_bundle_across_downstream_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.pages import mapping as mapping_page

    requests: list[MappingRequest] = []
    results: list[MappingResult] = []

    def recording_run(request: MappingRequest, report: Any, cancel: Any) -> MappingResult:
        requests.append(request)
        result = run_mapping(request, report, cancel)
        results.append(result)
        return result

    monkeypatch.setattr(mapping_page, "run_mapping", recording_run)

    async def scenario() -> None:
        async with user_simulation(index) as user:
            def control(marker: str) -> Any:
                return next(iter(user.find(marker=marker).elements))

            async def set_value(marker: str, value: Any) -> None:
                assert user.client is not None
                with user.client:
                    control(marker).set_value(value)
                await asyncio.sleep(0)
                await asyncio.sleep(0)

            async def map_and_wait(count: int) -> None:
                user.find(marker="action-map").click()
                for _ in range(200):
                    if (
                        len(results) == count
                        and control("action-map").enabled
                        and control("action-map").text == "Map"
                    ):
                        break
                    await asyncio.sleep(0.02)
                await user.should_not_see(marker="mapping-error")
                assert len(results) == count
                await user.should_see(marker="mapping-result-plot", retries=30)

            await user.open("/")
            user.find(marker="nav-mapping").click()
            await user.should_see(marker="mapping-target", retries=60)
            await set_value("mapping-target", "fork_densities")
            density_outputs = (
                "fork_density_xt", "left_fork_density_xt", "right_fork_density_xt",
            )
            assert tuple(control("mapping-preview-output").options)[:3] == density_outputs
            assert control("mapping-preview-output").value == "fork_density_xt"
            await set_value("mapping-region-start", 20.0)
            await set_value("mapping-region-end", 36.0)
            await set_value("mapping-setting-resolution", 2.0)
            await set_value("mapping-setting-perq", False)
            await set_value("mapping-input-initiation-rate-constant", 0.02)
            await set_value("mapping-setting-time-grid-override", True)
            await set_value("mapping-setting-time-grid", "0, 1, 3, 6, 10")
            await map_and_wait(1)
            assert results[0].target == "fork_densities_xt"
            with user.scope(marker="mapping-stats"):
                await user.should_see("Dependence")
                await user.should_not_see("Form")
            assert tuple(control("mapping-preview-output").options)[:3] == density_outputs
            assert control("mapping-preview-output").value == "fork_density_xt"
            np.testing.assert_allclose(
                results[0].outputs["fork_density_xt"],
                results[0].outputs["left_fork_density_xt"]
                + results[0].outputs["right_fork_density_xt"],
            )
            raw_names: set[str] = set()
            for output_key in density_outputs:
                await set_value("mapping-preview-output", output_key)
                chart = control("mapping-result-plot")
                assert chart.options["xAxis"]["name"] == "Position (kb)"
                assert chart.options["yAxis"]["name"] == "Time (min)"
                cells = chart.options["series"][-1]["data"]
                np.testing.assert_allclose(
                    [cell[2] for cell in cells],
                    results[0].outputs[output_key].T.reshape(-1),
                )
                raw_names.add(control("mapping-raw-filename").value)
            assert len(raw_names) == 3

            await set_value("mapping-target", "replicated_fraction")
            await set_value("mapping-target-form", "spacetime")
            await set_value("mapping-route", "replicated_fraction--spacetime--fork_densities")
            prepared = control("mapping-input-fork-densities-prepared-select")
            assert prepared.enabled
            assert len(prepared.options) == 1
            await set_value("mapping-whole-region", True)
            assert control("mapping-region-start").value == 0.0
            assert control("mapping-region-end").value == 36.0
            await set_value("mapping-whole-region", False)
            await set_value("mapping-region-start", 20.0)
            await set_value("mapping-region-end", 36.0)
            await map_and_wait(2)
            bundle = requests[1].inputs["fork_densities"]
            assert requests[1].input_forms["fork_densities"] == "bundle"
            assert bundle["right_fork_density_xt"] is results[0].outputs["right_fork_density_xt"]
            np.testing.assert_array_equal(bundle["time_grid"], [0, 1, 3, 6, 10])
            assert requests[1].resolution_x == 2.0
            np.testing.assert_array_equal(results[1].x_axis, np.arange(20.0, 36.0, 2.0))

            # The first downstream result replaced the visible batch, but its
            # retained source remains available to another bundle consumer.
            await set_value("mapping-target", "coalescence_events")
            await set_value("mapping-target-form", "spacetime")
            await set_value(
                "mapping-route",
                "coalescence_events--spacetime--fork_densities-and-fork_speed",
            )
            assert control("mapping-input-fork-densities-prepared-select").enabled
            await map_and_wait(3)
            assert requests[2].route.method == "fork_density_bundle_to_coalescence_events_xt_mean_field"
            np.testing.assert_array_equal(results[2].time_axis, results[0].time_axis)
            np.testing.assert_array_equal(results[2].x_axis, results[0].x_axis)
            assert np.isfinite(results[2].outputs["coalescence_events_xt"]).all()

    asyncio.run(scenario())
