"""Fast simulated-user coverage for the presentation-only NiceGUI prototype."""

from __future__ import annotations

import asyncio
import threading
from typing import cast

import numpy as np
import pytest
from nicegui.elements.button import Button
from nicegui.elements.drawer import LeftDrawer
from nicegui.elements.echart import EChart
from nicegui.elements.expansion import Expansion
from nicegui.elements.html import Html
from nicegui.elements.icon import Icon
from nicegui.elements.select import Select
from nicegui.testing import user_simulation

from app.main import index


def test_navigation_reaches_every_page_and_prototype_actions_remain_responsive() -> None:
    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            await user.should_see("Build quantitative models of DNA replication")

            destinations = (
                ("data-management", "Convert genomic data files", "convert"),
                ("mapping", "Map replication kinetic landscapes", "map"),
                # Simulation is now live and is exercised through an injected worker below.
                ("simulation", "Simulate DNA replication", None),
                ("plotting", "Visualize DNAscape outputs", "plot"),
                ("cached-files", "Manage generated files", "clear-generated-cache"),
                ("resources", "Project links and citation guidance", "copy-citation"),
                ("overview", "Build quantitative models of DNA replication", None),
            )
            for slug, heading, action in destinations:
                user.find(marker=f"nav-{slug}").click()
                await user.should_see(heading)
                if action is not None:
                    user.find(marker=f"action-{action}").click()
                    await user.should_see(heading)

    asyncio.run(scenario())


def test_sidebar_logo_returns_to_overview() -> None:
    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-data-management").click()
            await user.should_see("Data Manager")
            await user.should_see("Convert genomic data files")

            user.find(marker="nav-logo-overview").click()
            await user.should_see("Build quantitative models of DNA replication")

    asyncio.run(scenario())


def test_navigation_drawer_can_be_collapsed_and_expanded() -> None:
    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            await user.should_see(marker="navigation-drawer")
            drawer = cast(
                LeftDrawer,
                next(iter(user.find(marker="navigation-drawer").elements)),
            )
            assert drawer.value is True
            toggle = next(iter(user.find(marker="drawer-toggle").elements))
            assert toggle._props["icon"] == "first_page"

            user.find(marker="drawer-toggle").click()
            assert drawer.value is False
            toggle = next(iter(user.find(marker="drawer-toggle").elements))
            assert toggle._props["icon"] == "last_page"

            user.find(marker="drawer-toggle").click()
            assert drawer.value is True
            user.find(marker="nav-data-management").click()
            await user.should_see("Convert genomic data files")

    asyncio.run(scenario())


def test_application_menu_updates_theme_action_and_exposes_project_links() -> None:
    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="app-menu-toggle").click()
            await user.should_see("Dark theme")
            for marker in ("menu-documentation", "menu-github", "menu-issue"):
                await user.should_see(marker=marker)
            with user.scope(marker="menu-issue"):
                issue_icon = next(iter(user.find(kind=Icon).elements))
                assert issue_icon._props["name"] == "lightbulb"
            theme_icon = next(iter(user.find(marker="theme-action-icon").elements))
            assert theme_icon._props["name"] == "dark_mode"

            user.find(marker="menu-theme").click()
            user.find(marker="app-menu-toggle").click()
            await user.should_see("Light theme")
            theme_icon = next(iter(user.find(marker="theme-action-icon").elements))
            assert theme_icon._props["name"] == "light_mode"

    asyncio.run(scenario())


def test_scientific_pages_share_expansion_and_action_dock_hooks() -> None:
    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            destinations = (
                ("mapping", "Map replication kinetic landscapes", 4, "action-map"),
                ("simulation", "Simulate DNA replication", 4, "action-simulate"),
                ("plotting", "Visualize DNAscape outputs", 3, "action-plot"),
            )
            for slug, heading, minimum_sections, action_marker in destinations:
                user.find(marker=f"nav-{slug}").click()
                await user.should_see(heading)
                expansions = user.find(kind=Expansion).elements
                assert len(expansions) >= minimum_sections
                assert all("dn-expansion" in expansion._classes for expansion in expansions)

                dock = next(iter(user.find(marker="scientific-action-dock").elements))
                assert "dn-action-dock" in dock._classes
                with user.scope(marker="scientific-action-dock"):
                    await user.should_see(marker=action_marker)

    asyncio.run(scenario())


def test_plotting_uses_the_same_axis_and_line_symbol_in_navigation_and_action() -> None:
    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            with user.scope(marker="nav-plotting"):
                navigation_icon = next(iter(user.find(kind=Html).elements))
                assert "dn-plot-icon-axis" in navigation_icon.content
                assert "dn-plot-icon-line" in navigation_icon.content

            user.find(marker="nav-plotting").click()
            await user.should_see("Visualize DNAscape outputs")
            with user.scope(marker="action-plot"):
                action_icon = next(iter(user.find(kind=Html).elements))
                assert "dn-plot-icon-axis" in action_icon.content
                assert "dn-plot-icon-line" in action_icon.content

    asyncio.run(scenario())


def test_upload_fields_share_clear_copy_and_profile_defaults() -> None:
    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-data-management").click()
            await user.should_see(marker="data-manager-upload")
            await user.should_see("Accepted: CSV, TSV, TXT, NumPy, BigWig, bedGraph, GZ")

            user.find(marker="nav-mapping").click()
            await user.should_see(marker="source-replication-timing-upload")
            await user.should_see("Upload a profile")

            user.find(marker="nav-plotting").click()
            await user.should_see("Visualize DNAscape outputs")
            user.find(marker="plotting-source-mode").click()
            user.find("Upload file").click()
            await user.should_see(marker="plotting-data-upload")
            await user.should_see("Accepted: CSV, TSV, TXT, NPY, NPZ, GZ")

    asyncio.run(scenario())


def test_scientific_previews_share_the_floating_toolbar_and_raw_data_section() -> None:
    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            for slug in ("mapping", "simulation", "plotting"):
                user.find(marker=f"nav-{slug}").click()
                await user.should_see("Raw data")
                user.find(marker="action-fit-plot-to-window").click()
                await user.should_see("Preview")
                user.find(marker="action-download-plot").click()
                for file_format in ("png", "svg", "pdf"):
                    await user.should_see(marker=f"download-plot-{file_format}")

    asyncio.run(scenario())


def test_scientific_controls_use_compact_mode_selects_and_keep_primary_actions_docked() -> None:
    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-mapping").click()
            await user.should_see(kind=Select, marker="source-replication-timing-mode")
            await user.should_see(marker="source-replication-timing-profile")
            await user.should_not_see(marker="source-replication-timing-constant")
            user.find(marker="source-replication-timing-mode").click()
            user.find("Constant").click()
            await user.should_not_see(marker="source-replication-timing-profile")
            await user.should_see(marker="source-replication-timing-constant")
            for section in ("Region", "Mapping settings", "Plotting settings"):
                user.find(section).click()
            await user.should_see(marker="scientific-action-dock")
            with user.scope(marker="scientific-action-dock"):
                await user.should_see(marker="action-map")

        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-simulation").click()
            await user.should_see(kind=Select, marker="source-firing-rate-mode")
            await user.should_not_see(marker="source-firing-rate-profile")
            await user.should_see(marker="source-firing-rate-constant")
            user.find(marker="source-firing-rate-mode").click()
            user.find("Profile").click()
            await user.should_see(marker="source-firing-rate-profile")
            await user.should_not_see(marker="source-firing-rate-constant")
            user.find(marker="source-firing-rate-mode").click()
            user.find("Constant").click()
            await user.should_see(marker="source-firing-rate-constant")
            for section in ("Region", "Simulation settings", "Plotting settings"):
                user.find(section).click()
            await user.should_see(marker="scientific-action-dock")
            with user.scope(marker="scientific-action-dock"):
                await user.should_see(marker="action-simulate")

        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-plotting").click()
            await user.should_see(kind=Select, marker="plotting-source-mode")
            await user.should_see(marker="plotting-prepared-fields")
            await user.should_not_see(marker="plotting-upload-fields")
            user.find(marker="plotting-source-mode").click()
            user.find("Upload file").click()
            await user.should_not_see(marker="plotting-prepared-fields")
            await user.should_see(marker="plotting-upload-fields")
            for section in ("Region", "Plotting settings"):
                user.find(section).click()
            await user.should_see(marker="scientific-action-dock")
            with user.scope(marker="scientific-action-dock"):
                await user.should_see(marker="action-plot")

    asyncio.run(scenario())


def test_plot_controls_only_show_options_relevant_to_the_selected_view() -> None:
    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-mapping").click()
            await user.should_see("Map replication kinetic landscapes")
            user.find("Plotting settings").click()
            for marker in ("mapping-heatmap-palette", "mapping-colorbar", "mapping-histogram-bins"):
                await user.should_not_see(marker=marker)

            user.find(marker="mapping-plot-type").click()
            user.find("Heatmap").click()
            for marker in ("mapping-heatmap-palette", "mapping-colorbar"):
                await user.should_see(marker=marker)
            await user.should_not_see(marker="mapping-histogram-bins")

            user.find(marker="mapping-plot-type").click()
            user.find("Histogram").click()
            for marker in ("mapping-heatmap-palette", "mapping-colorbar"):
                await user.should_not_see(marker=marker)
            await user.should_see(marker="mapping-histogram-bins")

        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-plotting").click()
            await user.should_see("Visualize DNAscape outputs")
            user.find("Plotting settings").click()
            await user.should_not_see(marker="plotting-comparison-color")
            await user.should_see(marker="plotting-line-width")
            for marker in (
                "plotting-heatmap-palette",
                "plotting-colorbar",
                "plotting-colorbar-label",
                "plotting-heatmap-minimum",
                "plotting-heatmap-maximum",
                "plotting-histogram-bins",
            ):
                await user.should_not_see(marker=marker)

            with user.scope("Plotting settings"):
                user.find(marker="plotting-comparison-source").click()
                user.find("Simulation result").click()
            await user.should_see(marker="plotting-comparison-color")

            user.find(marker="plotting-plot-type").click()
            user.find("Heatmap").click()
            await user.should_not_see(marker="plotting-line-width")
            for marker in (
                "plotting-heatmap-palette",
                "plotting-colorbar",
                "plotting-colorbar-label",
                "plotting-heatmap-minimum",
                "plotting-heatmap-maximum",
            ):
                await user.should_see(marker=marker)
            await user.should_not_see(marker="plotting-histogram-bins")

            user.find(marker="plotting-plot-type").click()
            user.find("Histogram").click()
            for marker in (
                "plotting-heatmap-palette",
                "plotting-colorbar",
                "plotting-colorbar-label",
                "plotting-heatmap-minimum",
                "plotting-heatmap-maximum",
            ):
                await user.should_not_see(marker=marker)
            await user.should_see(marker="plotting-histogram-bins")

        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-simulation").click()
            await user.should_see("Simulate DNA replication")
            user.find("Plotting settings").click()
            for marker in (
                "simulation-histogram-bins",
                "simulation-density",
                "simulation-heatmap-palette",
                "simulation-colorbar",
            ):
                await user.should_not_see(marker=marker)
            await user.should_see(marker="simulation-line-or-bar-color")

            user.find(marker="simulation-view").click()
            user.find("Histogram").click()
            for marker in ("simulation-histogram-bins", "simulation-density"):
                await user.should_see(marker=marker)
            for marker in ("simulation-heatmap-palette", "simulation-colorbar"):
                await user.should_not_see(marker=marker)

            user.find(marker="simulation-view").click()
            user.find("Time array").click()
            for marker in ("simulation-histogram-bins", "simulation-density"):
                await user.should_not_see(marker=marker)
            for marker in ("simulation-heatmap-palette", "simulation-colorbar"):
                await user.should_see(marker=marker)
            await user.should_not_see(marker="simulation-line-or-bar-color")

    asyncio.run(scenario())


def test_dependent_simulation_controls_follow_optional_toggles() -> None:
    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-simulation").click()
            await user.should_see("Simulate DNA replication")
            user.find("Simulation settings").click()
            await user.should_see(marker="simulation-seed")
            user.find("Fixed seed").click()
            await user.should_not_see(marker="simulation-seed")

            await user.should_not_see(marker="simulation-stall-duration")
            user.find("Non-recovering stalls").click()
            await user.should_see(marker="simulation-stall-duration")

            await user.should_not_see(marker="simulation-pool-fields")
            user.find("Finite firing-factor pool").click()
            await user.should_see(marker="simulation-pool-fields")

    asyncio.run(scenario())


def test_constant_simulation_reports_progress_and_renders_worker_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.pages import simulation as simulation_page
    from app.simulation_service import ConstantSimulationRequest, ConstantSimulationResult

    started = threading.Event()
    release = threading.Event()
    requests: list[ConstantSimulationRequest] = []

    def fake_run(
        request: ConstantSimulationRequest,
        report,
    ) -> ConstantSimulationResult:
        requests.append(request)
        report(0.45, "Running test ensemble...")
        started.set()
        if not release.wait(timeout=5):
            raise TimeoutError("test did not release the simulation worker")
        position = np.array([0.0, 1.0, 2.0])
        timing = np.array([18.0, 14.0, 20.0])
        return ConstantSimulationResult(
            position_kb=position,
            replication_timing=timing,
            elapsed_seconds=0.12,
            replicates=request.replicates,
            geometry=request.geometry,
            time_grid=np.array([0.0, 5.0, 10.0]),
            outputs={"replication_timing": timing},
        )

    monkeypatch.setattr(simulation_page, "run_constant_simulation", fake_run)

    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-simulation").click()
            await user.should_see("Simulate DNA replication")
            await user.should_not_see(marker="simulation-progress")
            await user.should_not_see(marker="simulation-result-plot")

            user.find(marker="action-simulate").click()
            await user.should_see(marker="simulation-progress", retries=20)
            assert await asyncio.to_thread(started.wait, 2)
            await user.should_see("Running test ensemble...", retries=20)
            assert requests
            request = requests[0]
            assert request.firing_rate == pytest.approx(0.01)
            assert request.fork_speed == pytest.approx(1.4)
            assert request.region_start_kb == pytest.approx(0.0)
            assert request.region_end_kb == pytest.approx(1000.0)
            assert request.replicates == 20
            assert request.geometry == "linear"
            assert request.track_detailed_outputs is False

            release.set()
            await user.should_see(marker="simulation-result-plot", retries=30)
            await user.should_see("Complete in 0.12 s", retries=10)
            await user.should_not_see(marker="simulation-error")

    try:
        asyncio.run(scenario())
    finally:
        release.set()


def test_detailed_simulation_switch_exposes_xt_plot_and_live_downloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.pages import simulation as simulation_page
    from app.simulation_exports import ExportArtifact
    from app.simulation_service import ConstantSimulationRequest, ConstantSimulationResult

    requests: list[ConstantSimulationRequest] = []
    plot_exports: list[tuple[str, str]] = []
    raw_exports: list[tuple[str, str, bool]] = []

    def fake_run(request: ConstantSimulationRequest, report) -> ConstantSimulationResult:
        requests.append(request)
        report(0.65, "Collecting detailed outputs...")
        position = np.array([0.0, 1.0, 2.0])
        time = np.array([0.0, 5.0])
        timing = np.array([18.0, 14.0, 20.0])
        return ConstantSimulationResult(
            position_kb=position,
            replication_timing=timing,
            elapsed_seconds=0.2,
            replicates=request.replicates,
            geometry=request.geometry,
            time_grid=time,
            outputs={
                "replication_timing": timing,
                "origin_efficiency": np.array([0.2, 0.6, 0.2]),
                "fork_directionality": np.array([-0.5, 0.0, 0.5]),
                "replicated_fraction": np.array([0.0, 0.8]),
                "replicated_fraction_xt": np.array(
                    [[0.0, 0.8], [0.0, 0.7], [0.0, 0.9]]
                ),
            },
            detailed_outputs=True,
        )

    def fake_plot_export(result, output_key: str, file_format: str, **_kwargs):
        assert result.detailed_outputs
        plot_exports.append((output_key, file_format))
        return ExportArtifact(b"plot", f"plot.{file_format.lower()}", "image/test")

    def fake_raw_export(
        result,
        output_key: str,
        file_format: str,
        *,
        gzip_output: bool = False,
        **_kwargs,
    ):
        assert result.detailed_outputs
        raw_exports.append((output_key, file_format, gzip_output))
        return ExportArtifact(b"raw", "simulation.npz", "application/octet-stream")

    monkeypatch.setattr(simulation_page, "run_constant_simulation", fake_run)
    monkeypatch.setattr(simulation_page, "build_plot_export", fake_plot_export)
    monkeypatch.setattr(simulation_page, "build_raw_export", fake_raw_export)

    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-simulation").click()
            await user.should_see("Simulate DNA replication")
            user.find("Simulation settings").click()
            user.find(marker="simulation-track-detailed-outputs").click()
            user.find(marker="action-simulate").click()
            await user.should_see(marker="simulation-result-plot", retries=30)
            assert requests and requests[0].track_detailed_outputs is True

            user.find("Plotting settings").click()
            user.find(marker="simulation-output").click()
            user.find("Replicated fraction (space-time)").click()
            await user.should_see(marker="simulation-result-plot", retries=10)
            chart = cast(
                EChart,
                next(iter(user.find(marker="simulation-result-plot").elements)),
            )
            assert chart.options["series"][0]["type"] == "heatmap"

            user.find(marker="action-download-plot").click()
            user.find(marker="download-plot-svg").click()
            for _ in range(30):
                if plot_exports:
                    break
                await asyncio.sleep(0.02)
            assert plot_exports == [("replicated_fraction_xt", "SVG")]

            user.find("Raw data").click()
            user.find(marker="action-download-raw-data").click()
            for _ in range(30):
                if raw_exports:
                    break
                await asyncio.sleep(0.02)
            assert raw_exports == [("replicated_fraction_xt", "NPZ", False)]

    asyncio.run(scenario())


def test_real_constant_simulation_replaces_placeholder_with_result_plot() -> None:
    """Exercise the default UI controls through the real DNAscape worker."""

    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-simulation").click()
            await user.should_see("Simulate DNA replication")

            user.find(marker="action-simulate").click()
            await user.should_see(marker="simulation-result-plot", retries=60)
            await user.should_see("Complete in", retries=20)
            await user.should_not_see(marker="simulation-error")

    asyncio.run(scenario())


def test_profile_mode_is_rejected_without_starting_simulation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.pages import simulation as simulation_page

    called = False

    def unexpected_run(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("profile validation must happen before starting the worker")

    monkeypatch.setattr(simulation_page, "run_constant_simulation", unexpected_run)

    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-simulation").click()
            await user.should_see(kind=Select, marker="source-firing-rate-mode")
            user.find(marker="source-firing-rate-mode").click()
            user.find("Profile").click()
            await user.should_see(marker="source-firing-rate-profile")

            user.find(marker="action-simulate").click()
            await user.should_see(marker="simulation-error")
            await user.should_see("This first integration supports Constant inputs only.")
            await user.should_see("Choose Constant inputs to run")
            await user.should_not_see(marker="simulation-progress")
            await user.should_not_see(marker="simulation-result-plot")
            assert called is False

    asyncio.run(scenario())


def test_simulation_worker_failure_stays_inside_preview(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.pages import simulation as simulation_page

    def failing_run(*_args, **_kwargs):
        raise ValueError("Synthetic simulation failure")

    monkeypatch.setattr(simulation_page, "run_constant_simulation", failing_run)

    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-simulation").click()
            await user.should_see("Simulate DNA replication")
            user.find(marker="action-simulate").click()

            await user.should_see("Synthetic simulation failure", retries=20)
            await user.should_see("Simulation could not be completed")
            await user.should_see(marker="simulation-error")
            await user.should_not_see(marker="simulation-progress")
            await user.should_not_see(marker="simulation-result-plot")
            simulate_button = cast(
                Button,
                next(iter(user.find(marker="action-simulate").elements)),
            )
            assert simulate_button.enabled

    asyncio.run(scenario())
