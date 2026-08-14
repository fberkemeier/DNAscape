"""Fast simulated-user coverage for the presentation-only NiceGUI prototype."""

from __future__ import annotations

import asyncio
from typing import cast

from nicegui.elements.drawer import LeftDrawer
from nicegui.testing import user_simulation

from app.main import index


def test_navigation_reaches_every_page_and_primary_actions_are_inert() -> None:
    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            await user.should_see("Build quantitative models of DNA replication")

            destinations = (
                ("data-management", "Convert genomic data files", "convert"),
                ("mapping", "Map replication kinetic landscapes", "map"),
                ("simulation", "Simulate DNA replication", "simulate"),
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

            user.find(marker="drawer-toggle").click()
            assert drawer.value is False

            user.find(marker="drawer-toggle").click()
            assert drawer.value is True
            user.find(marker="nav-data-management").click()
            await user.should_see("Convert genomic data files")

    asyncio.run(scenario())


def test_scientific_previews_share_the_floating_toolbar_and_raw_data_section() -> None:
    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            for slug in ("mapping", "simulation", "plotting"):
                user.find(marker=f"nav-{slug}").click()
                await user.should_see("Raw data")
                for action in ("fit-plot-to-window", "download-plot", "clear-plot"):
                    user.find(marker=f"action-{action}").click()
                    await user.should_see("Preview")

    asyncio.run(scenario())


def test_dependent_controls_follow_source_and_simulation_toggles() -> None:
    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-mapping").click()
            await user.should_see(marker="source-replication-timing-profile")
            await user.should_not_see(marker="source-replication-timing-constant")
            with user.scope(marker="source-replication-timing"):
                user.find("Constant").click()
            await user.should_not_see(marker="source-replication-timing-profile")
            await user.should_see(marker="source-replication-timing-constant")

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
