"""Background jobs retain their pages and report independent sidebar status."""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any, cast

import numpy as np
import pytest
from nicegui.elements.upload import Upload
from nicegui.testing import user_simulation

from app.export_settings import get_export_settings
from app.main import index
from app.mapping_service import MappingResult
from app.page_activity import PageActivity, current_page_activity
from app.pages import data_management, mapping, plotting, simulation
from app.simulation_service import ConstantSimulationResult

_WORKFLOWS = {
    "mapping": (mapping, "run_mapping", "map", "Mapping stopped", "mapping-result-plot"),
    "simulation": (
        simulation, "run_constant_simulation", "simulate", "Simulation stopped",
        "simulation-result-plot",
    ),
    "data-management": (
        data_management, "convert_data", "convert", "Conversion stopped", "data-manager-result-1",
    ),
}


class _BlockedWorker:
    def __init__(self, monkeypatch: pytest.MonkeyPatch, slug: str) -> None:
        self.slug = slug
        self.started = threading.Event()
        self.release = threading.Event()
        module, function, *_ = _WORKFLOWS[slug]
        self.original = getattr(module, function)
        monkeypatch.setattr(module, function, self.run)

    def run(self, request: Any, report: Any, cancel: Any) -> Any:
        self.started.set()
        deadline = time.monotonic() + 10
        while not self.release.wait(0.01):
            cancel()
            if time.monotonic() > deadline:
                raise TimeoutError("test did not release the background worker")
        cancel()
        if self.slug == "data-management":
            return self.original(request, report, cancel)
        values = np.asarray([1.0, 2.0, 3.0])
        if self.slug == "mapping":
            return MappingResult(
                target=request.route.target, target_key=request.route.target_key,
                target_label="Firing rate", method=request.route.method,
                output_form="x", output_dependence="space", primary_output="firing_rate",
                outputs={"firing_rate": values / 10}, output_forms={"firing_rate": "x"},
                x_axis=np.arange(3.0), time_axis=np.asarray([]), region_labels=(),
                elapsed_seconds=0.01, assumptions={},
            )
        return ConstantSimulationResult(
            position_kb=np.arange(3.0), replication_timing=values,
            elapsed_seconds=0.01, replicates=request.replicates, geometry=request.geometry,
            time_grid=np.asarray([]), outputs={"replication_timing": values},
        )


async def _prepare(user: Any, slug: str) -> None:
    user.find(marker=f"nav-{slug}").click()
    if slug == "simulation":
        return
    upload_marker = (
        "source-replication-timing-upload" if slug == "mapping" else "data-manager-upload"
    )
    uploader = cast(Upload, next(iter(user.find(marker=upload_marker).elements)))
    await uploader.handle_uploads(
        [Upload.SmallFileUpload("timing.csv", "text/csv", b"timing\n1\n2\n3\n")]
    )
    if slug == "mapping":
        await user.should_see(marker="mapping-input-replication-timing-detected-data", retries=60)
    else:
        await user.should_see("Array size: 3", retries=60)


@pytest.mark.parametrize("preserve_page_state", (True, False))
@pytest.mark.parametrize("finish_on_page", (True, False))
@pytest.mark.parametrize("slug", tuple(_WORKFLOWS))
def test_background_workflow_finishes_and_stops_across_page_switches(
    monkeypatch: pytest.MonkeyPatch, slug: str, preserve_page_state: bool,
    finish_on_page: bool,
) -> None:
    worker = _BlockedWorker(monkeypatch, slug)
    _, _, action, stopped_message, result_marker = _WORKFLOWS[slug]
    settings = get_export_settings()
    previous = settings.preserve_page_state
    settings.preserve_page_state = preserve_page_state

    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            await _prepare(user, slug)
            button = next(iter(user.find(marker=f"action-{action}").elements))
            user.find(marker=f"action-{action}").click()
            assert await asyncio.to_thread(worker.started.wait, 3)
            await user.should_see(marker=f"nav-{slug}-running", retries=30)

            user.find(marker="nav-overview").click()
            await user.should_see("Build quantitative models of DNA replication")
            await user.should_see(marker=f"nav-{slug}-running")
            user.find(marker=f"nav-{slug}").click()
            assert next(iter(user.find(marker=f"action-{action}").elements)) is button
            await user.should_see(marker=f"nav-{slug}-running")
            if not finish_on_page:
                user.find(marker="nav-overview").click()
            worker.release.set()
            if not finish_on_page:
                await user.should_see(marker=f"nav-{slug}-complete", retries=60)
                user.find(marker=f"nav-{slug}").click()
            await user.should_not_see(marker=f"nav-{slug}-running", retries=60)
            await user.should_not_see(marker=f"nav-{slug}-complete")
            await user.should_see(marker=result_marker, retries=60)
            assert next(iter(user.find(marker=f"action-{action}").elements)) is button

            worker.release.clear()
            worker.started.clear()
            user.find(marker=f"action-{action}").click()
            assert await asyncio.to_thread(worker.started.wait, 3)
            await user.should_see(marker=f"nav-{slug}-running", retries=30)
            user.find(marker="nav-overview").click()
            user.find(marker=f"nav-{slug}").click()
            user.find(marker=f"action-{action}").click()
            await user.should_not_see(marker=f"nav-{slug}-running", retries=30)
            await user.should_not_see(marker=f"nav-{slug}-complete")
            await user.should_see(stopped_message, retries=60)
            await user.should_not_see(marker=f"nav-{slug}-complete")

    try:
        asyncio.run(scenario())
    finally:
        worker.release.set()
        settings.preserve_page_state = previous


@pytest.mark.parametrize(
    ("source", "message"),
    (
        ("firing-rate", "Upload firing rate data."),
        ("fork-speed", "Upload fork speed data."),
    ),
)
def test_missing_simulation_profile_clears_activity(
    monkeypatch: pytest.MonkeyPatch, source: str, message: str,
) -> None:
    def unexpected_worker(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("A simulation worker must not start without its input profile.")

    monkeypatch.setattr(simulation, "run_constant_simulation", unexpected_worker)

    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-simulation").click()
            mode = next(iter(user.find(marker=f"source-{source}-mode").elements))
            with user.client:
                mode.set_value("profile")
            user.find(marker="action-simulate").click()

            await user.should_see(message, retries=60)
            await user.should_see("Check the settings")
            await user.should_not_see(marker="nav-simulation-running", retries=30)
            await user.should_not_see(marker="nav-simulation-complete")
            await user.should_not_see(marker="simulation-progress")
            button = next(iter(user.find(marker="action-simulate").elements))
            assert button.text == "Simulate"
            assert button.enabled
            assert mode.enabled
            with user.client:
                assert not current_page_activity().retain_page("simulation")

    asyncio.run(scenario())


def test_mapping_and_simulation_can_run_together_with_independent_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mapping_worker = _BlockedWorker(monkeypatch, "mapping")
    simulation_worker = _BlockedWorker(monkeypatch, "simulation")

    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            await _prepare(user, "mapping")
            user.find(marker="action-map").click()
            assert await asyncio.to_thread(mapping_worker.started.wait, 3)
            await _prepare(user, "simulation")
            user.find(marker="action-simulate").click()
            assert await asyncio.to_thread(simulation_worker.started.wait, 3)
            await user.should_see(marker="nav-mapping-running")
            await user.should_see(marker="nav-simulation-running")
            user.find(marker="nav-plotting").click()

            mapping_worker.release.set()
            await user.should_see(marker="nav-mapping-complete", retries=60)
            await user.should_see(marker="nav-simulation-running")
            user.find(marker="nav-mapping").click()
            await user.should_not_see(marker="nav-mapping-complete")
            await user.should_see(marker="mapping-result-plot")
            simulation_worker.release.set()
            await user.should_see(marker="nav-simulation-complete", retries=60)
            user.find(marker="nav-simulation").click()
            await user.should_not_see(marker="nav-simulation-complete")
            await user.should_see(marker="simulation-result-plot")

    try:
        asyncio.run(scenario())
    finally:
        mapping_worker.release.set()
        simulation_worker.release.set()


def test_stopping_and_resetting_suppress_late_completion() -> None:
    activity = PageActivity()
    cancelled = threading.Event()
    stopped = activity.begin("mapping", cancel=cancelled.set)
    activity.stop("mapping")
    assert cancelled.is_set()
    assert activity.indicator("mapping") is None
    assert activity.retain_page("mapping")
    activity.finish("mapping", stopped, success=True)
    assert activity.indicator("mapping") is None
    assert not activity.retain_page("mapping")

    old = activity.begin("simulation", cancel=cancelled.set)
    activity.reset()
    current = activity.begin("simulation")
    activity.finish("simulation", old, success=True)
    assert activity.indicator("simulation") == "running"
    activity.finish("simulation", current, success=False)
    assert activity.indicator("simulation") is None


@pytest.mark.parametrize("finish_on_page", (True, False))
def test_plotting_completion_with_page_preservation_off(
    monkeypatch: pytest.MonkeyPatch, finish_on_page: bool,
) -> None:
    started, release = threading.Event(), threading.Event()
    collect = plotting._collect_plot_series

    def blocked_collect(sources: Any) -> Any:
        started.set()
        if not release.wait(10):
            raise TimeoutError("test did not release plot preparation")
        return collect(sources)

    monkeypatch.setattr(plotting, "_collect_plot_series", blocked_collect)
    settings = get_export_settings()
    previous = settings.preserve_page_state
    settings.preserve_page_state = False

    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-plotting").click()
            uploader = cast(
                Upload, next(iter(user.find(marker="plotting-data-upload").elements)),
            )
            await uploader.handle_uploads(
                [Upload.SmallFileUpload("signal.csv", "text/csv", b"signal\n1\n2\n3\n")]
            )
            await user.should_see("Array size: 3", retries=60)
            user.find(marker="action-plot").click()
            assert await asyncio.to_thread(started.wait, 3)
            await user.should_see(marker="nav-plotting-running", retries=30)
            if not finish_on_page:
                user.find(marker="nav-overview").click()
            await user.should_see(marker="nav-plotting-running")
            release.set()
            if not finish_on_page:
                await user.should_see(marker="nav-plotting-complete", retries=60)
                user.find(marker="nav-plotting").click()
            await user.should_not_see(marker="nav-plotting-running", retries=60)
            await user.should_not_see(marker="nav-plotting-complete")
            await user.should_see(marker="plotting-result-plot", retries=60)
            assert next(iter(user.find(marker="plotting-data-upload").elements)) is uploader

    try:
        asyncio.run(scenario())
    finally:
        release.set()
        settings.preserve_page_state = previous


def test_activity_is_isolated_between_browser_clients() -> None:
    async def scenario() -> None:
        async with user_simulation(index) as first, user_simulation(index) as second:
            await first.open("/")
            await second.open("/")
            assert first.client is not None and second.client is not None
            with first.client:
                first_activity = current_page_activity()
                run = first_activity.begin("mapping")
            await first.should_see(marker="nav-mapping-running", retries=30)
            await second.should_not_see(marker="nav-mapping-running")
            with second.client:
                assert current_page_activity() is not first_activity
            with first.client:
                first_activity.finish("mapping", run, success=True)
            await first.should_see(marker="nav-mapping-complete", retries=30)
            await second.should_not_see(marker="nav-mapping-complete")
            first.find(marker="nav-mapping").click()
            await first.should_not_see(marker="nav-mapping-complete")
            with first.client:
                run = first_activity.begin("mapping")
                first_activity.finish("mapping", run, success=True)
            await first.should_not_see(marker="nav-mapping-running", retries=30)
            await first.should_not_see(marker="nav-mapping-complete")

    asyncio.run(scenario())
