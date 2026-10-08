"""Simulation-page coverage for flexible profile and constant batches."""

from __future__ import annotations

import asyncio
import io
from typing import Any, cast

import numpy as np
import pytest
from nicegui.elements.echart import EChart
from nicegui.elements.number import Number
from nicegui.elements.select import Select
from nicegui.elements.upload import Upload
from nicegui.testing import user_simulation

from app.main import index
from app.simulation_service import ConstantSimulationRequest, ConstantSimulationResult


def _result_for(request: ConstantSimulationRequest) -> ConstantSimulationResult:
    bins = int(
        np.ceil((request.region_end_kb - request.region_start_kb) / request.resolution_x)
    )
    position = request.region_start_kb + np.arange(bins, dtype=float) * request.resolution_x
    firing = np.asarray(request.firing_rate, dtype=float).reshape(-1)
    timing = (
        np.full(bins, float(firing[0]) * 10.0)
        if firing.size == 1
        else firing * 10.0
    )
    return ConstantSimulationResult(
        position_kb=position,
        replication_timing=timing,
        elapsed_seconds=0.01,
        replicates=request.replicates,
        geometry=request.geometry,
        time_grid=np.asarray([], dtype=float),
        outputs={"replication_timing": timing},
    )


def _select(user: Any, marker: str) -> Select:
    return cast(Select, next(iter(user.find(marker=marker).elements)))


def _upload(user: Any, marker: str) -> Upload:
    return cast(Upload, next(iter(user.find(marker=marker).elements)))


def test_simulation_accepts_different_spatial_lengths_per_paired_dataset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each pair derives its own region while still supporting one overlay."""

    from app.pages import simulation as simulation_page

    requests: list[ConstantSimulationRequest] = []

    def fake_run(request: ConstantSimulationRequest, report: Any, _cancel: Any):
        requests.append(request)
        report(0.8, "Running variable-length dataset...")
        return _result_for(request)

    monkeypatch.setattr(simulation_page, "run_constant_simulation", fake_run)

    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-simulation").click()
            await user.should_see("Simulate DNA replication")
            _select(user, "source-firing-rate-mode").set_value("profile")
            _select(user, "source-fork-speed-mode").set_value("profile")

            firing_upload = _upload(user, "source-firing-rate-upload")
            speed_upload = _upload(user, "source-fork-speed-upload")
            await firing_upload.handle_uploads(
                [
                    Upload.SmallFileUpload(
                        "short.csv",
                        "text/csv",
                        b"firing_rate\n0.1\n0.2\n0.3\n",
                    ),
                    Upload.SmallFileUpload(
                        "long.csv",
                        "text/csv",
                        b"firing_rate\n0.1\n0.2\n0.3\n0.4\n0.5\n",
                    ),
                ]
            )
            await speed_upload.handle_uploads(
                [
                    Upload.SmallFileUpload(
                        "short.csv",
                        "text/csv",
                        b"fork_speed\n1.1\n1.2\n1.3\n",
                    ),
                    Upload.SmallFileUpload(
                        "long.csv",
                        "text/csv",
                        b"fork_speed\n1.1\n1.2\n1.3\n1.4\n1.5\n",
                    ),
                ]
            )
            for factor in ("firing-rate", "fork-speed"):
                marker = f"source-{factor}-detected-data"
                await user.should_see(marker=marker, retries=60)
                with user.scope(marker=marker):
                    await user.should_see("Files: 2", retries=60)
            # A sole unambiguous signal is selected automatically and needs no
            # redundant value-column field.
            await user.should_not_see(marker="source-firing-rate-value-column")
            await user.should_not_see(marker="source-fork-speed-value-column")

            user.find(marker="action-simulate").click()
            await user.should_see("2 datasets complete", retries=60)

            assert [request.region_end_kb for request in requests] == pytest.approx([3.0, 5.0])
            assert [np.asarray(request.firing_rate).size for request in requests] == [3, 5]
            assert [np.asarray(request.fork_speed).size for request in requests] == [3, 5]
            layout = _select(user, "simulation-presentation")
            assert layout.options == ["Combined", "Individual"]
            assert layout.value == "Combined"
            chart = cast(
                EChart,
                next(iter(user.find(marker="simulation-result-plot").elements)),
            )
            assert [len(series["data"]) for series in chart.options["series"]] == [3, 5]
            assert float(chart.options["xAxis"]["max"]) >= 4.0

    asyncio.run(scenario())


def test_simulation_pairs_and_broadcasts_multiple_constants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.pages import simulation as simulation_page

    requests: list[ConstantSimulationRequest] = []

    def fake_run(request: ConstantSimulationRequest, report: Any, _cancel: Any):
        requests.append(request)
        report(0.8, "Running constant dataset...")
        return _result_for(request)

    monkeypatch.setattr(simulation_page, "run_constant_simulation", fake_run)

    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-simulation").click()
            await user.should_see("Simulate DNA replication")

            firing_mode = _select(user, "source-firing-rate-mode")
            firing_mode.set_value("constants")
            await user.should_see(marker="source-firing-rate-constant-2")
            user.find(marker="source-firing-rate-constant-add").click()
            await user.should_see(marker="source-firing-rate-constant-3")
            for constant_index, value in enumerate(
                (0.01, 0.02, 0.03),
                start=1,
            ):
                next(
                    iter(
                        user.find(
                            marker=f"source-firing-rate-constant-{constant_index}"
                        ).elements
                    )
                ).set_value(value)
            next(
                iter(user.find(marker="source-fork-speed-constant").elements)
            ).set_value(1.4)

            user.find(marker="action-simulate").click()
            await user.should_see("3 datasets complete", retries=60)

            assert [request.firing_rate for request in requests] == pytest.approx(
                [0.01, 0.02, 0.03]
            )
            assert [request.fork_speed for request in requests] == pytest.approx(
                [1.4, 1.4, 1.4]
            )
            layout = _select(user, "simulation-presentation")
            layout.set_value("Individual")
            await user.should_see(marker="simulation-dataset-select")
            dataset_select = _select(user, "simulation-dataset-select")
            assert list(dataset_select.options.values()) == [
                "Simulation · 0.01",
                "Simulation · 0.02",
                "Simulation · 0.03",
            ]
            assert len(set(dataset_select.options)) == 3
            assert all(
                str(key).startswith("simulation-case-")
                for key in dataset_select.options
            )

    asyncio.run(scenario())


def test_simulation_rejects_a_spatial_mismatch_inside_one_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.pages import simulation as simulation_page

    requests: list[ConstantSimulationRequest] = []

    def fake_run(request: ConstantSimulationRequest, report: Any, _cancel: Any):
        requests.append(request)
        return _result_for(request)

    monkeypatch.setattr(simulation_page, "run_constant_simulation", fake_run)

    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-simulation").click()
            await user.should_see("Simulate DNA replication")
            _select(user, "source-firing-rate-mode").set_value("profile")
            _select(user, "source-fork-speed-mode").set_value("profile")
            await _upload(user, "source-firing-rate-upload").handle_uploads(
                [
                    Upload.SmallFileUpload(
                        "sample.csv", "text/csv", b"firing_rate\n0.1\n0.2\n0.3\n"
                    )
                ]
            )
            await _upload(user, "source-fork-speed-upload").handle_uploads(
                [
                    Upload.SmallFileUpload(
                        "sample.csv", "text/csv", b"fork_speed\n1.1\n1.2\n1.3\n1.4\n"
                    )
                ]
            )
            await asyncio.sleep(0.25)

            user.find(marker="action-simulate").click()
            await user.should_see("axes disagree", retries=40)
            assert requests == []

    asyncio.run(scenario())


@pytest.mark.parametrize("spacetime", [False, True])
def test_simulation_preserves_edited_end_and_trims_spatial_profiles(
    monkeypatch: pytest.MonkeyPatch, spacetime: bool,
) -> None:
    from app.pages import simulation as simulation_page

    requests: list[ConstantSimulationRequest] = []
    real_run = simulation_page.run_constant_simulation
    firing = np.full((6, 4), 0.1) if spacetime else np.arange(1, 7, dtype=float) / 10
    speed = np.full((1, 4), 1.4) if spacetime else np.arange(1, 7, dtype=float)

    def record_run(request: ConstantSimulationRequest, report: Any, cancel: Any):
        requests.append(request)
        return real_run(request, report, cancel)

    monkeypatch.setattr(simulation_page, "run_constant_simulation", record_run)

    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-simulation").click()
            await user.should_see("Simulate DNA replication", retries=60)
            assert user.client is not None
            for factor, values in (("firing-rate", firing), ("fork-speed", speed)):
                with user.client:
                    _select(user, f"source-{factor}-mode").set_value("profile")
                await user.should_see(marker=f"source-{factor}-upload")
                payload = io.BytesIO()
                np.save(payload, values, allow_pickle=False)
                await _upload(user, f"source-{factor}-upload").handle_uploads([
                    Upload.SmallFileUpload(f"{factor}.npy", "application/octet-stream", payload.getvalue())
                ])
                await user.should_see(marker=f"source-{factor}-upload-file", retries=60)
            resolution = cast(Number, next(iter(user.find(marker="simulation-resolution-x").elements)))
            end = cast(Number, next(iter(user.find(marker="simulation-region-end").elements)))
            with user.client:
                resolution.set_value(0.1)
            assert end.value == pytest.approx(0.6)
            with user.client:
                end.set_value(0.3)

            user.find(marker="action-simulate").click()
            await user.should_see(marker="simulation-result-plot", retries=60)
            await user.should_not_see(marker="simulation-error")
            assert end.value == pytest.approx(0.3)
            assert len(requests) == 1
            assert requests[0].region_end_kb == pytest.approx(0.3)
            np.testing.assert_array_equal(requests[0].firing_rate, firing[:3])
            np.testing.assert_array_equal(requests[0].fork_speed, speed if spacetime else speed[:3])
            chart = cast(EChart, next(iter(user.find(marker="simulation-result-plot").elements)))
            assert len(chart.options["series"][0]["data"]) == 3

    asyncio.run(scenario())
