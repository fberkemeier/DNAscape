"""Fast simulated-user coverage for the presentation-only NiceGUI prototype."""

from __future__ import annotations

from app.pages.common import workflow_action_icon

import asyncio
import gzip
import io
import threading
import time
from typing import Any, cast

import numpy as np
import pytest
from nicegui.elements.button import Button
from nicegui.elements.checkbox import Checkbox
from nicegui.elements.dark_mode import DarkMode
from nicegui.elements.drawer import LeftDrawer
from nicegui.elements.echart import EChart
from nicegui.elements.expansion import Expansion
from nicegui.elements.html import Html
from nicegui.elements.icon import Icon
from nicegui.elements.input import Input
from nicegui.elements.label import Label
from nicegui.elements.number import Number
from nicegui.elements.select import Select
from nicegui.elements.tabs import TabPanels
from nicegui.elements.upload import Upload
from nicegui.testing import user_simulation
from nicegui.testing.user import UserInteraction

from app.icons import DATA_ICON_SVG
from app.main import index
from app.mapping_catalog import mapping_route_families
from app.mapping_service import MappingResult
from app.page_activity import current_page_activity
from app.session_cache import current_session_cache
from app.simulation_exports import ExportArtifact
from app.simulation_service import ConstantSimulationResult


def _select_by_marker(user: object, marker: str) -> Select:
    client = getattr(user, "client", None)
    assert client is not None
    return cast(
        Select,
        next(
            element
            for element in client.elements.values()
            if marker in element._markers and not element.is_deleted
        ),
    )


def test_navigation_reaches_every_page_and_prototype_actions_remain_responsive() -> None:
    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            await user.should_see("Build quantitative models of DNA replication")

            destinations = (
                ("data-management", "Convert genomic data files", "convert"),
                # Scientific actions are live async workflows and have dedicated tests below.
                ("mapping", "Map replication kinetic landscapes", None),
                ("simulation", "Simulate DNA replication", None),
                ("plotting", "Explore DNAscape data", None),
                ("cached-files", "Session files and results", "clear-generated-cache"),
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


def test_cached_files_inventory_filters_removes_and_clears_real_session_items() -> None:
    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            assert user.client is not None
            with user.client:
                cache = current_session_cache()
                raw = cache.add_artifact(
                    ExportArtifact(b"raw", "timing.npy.gz", "application/gzip"),
                    source="Simulation",
                    kind="raw",
                    detail="Replication timing",
                )
                plot = cache.add_artifact(
                    ExportArtifact(b"plot", "timing.png", "image/png"),
                    source="Simulation",
                    kind="plot",
                    detail="Replication timing",
                )
                upload = cache.add_artifact(
                    ExportArtifact(b"position,value", "profile.csv", "text/csv"),
                    category="uploads",
                    source="Mapping",
                    kind="upload",
                    detail="CSV | Space | 2",
                )
                workflow = cache.add_reference(
                    name="Simulation result",
                    source="Simulation",
                    detail="2 replicates | 0-2 kb | Linear",
                    payload=ConstantSimulationResult(
                        position_kb=np.asarray([0.0, 1.0, 2.0]),
                        replication_timing=np.asarray([3.0, 2.0, 4.0]),
                        elapsed_seconds=0.1,
                        replicates=2,
                        geometry="linear",
                        time_grid=np.asarray([0.0, 1.0]),
                        outputs={"replication_timing": np.asarray([3.0, 2.0, 4.0])},
                    ),
                    navigate_to="simulation",
                )

            user.find(marker="nav-cached-files").click()
            await user.should_see("Session files and results")
            all_panel = next(iter(user.find(marker="cache-panel-all").elements))
            assert all_panel.parent_slot is not None
            assert isinstance(all_panel.parent_slot.parent, TabPanels)
            await user.should_see("timing.npy.gz")
            await user.should_see("timing.png")
            await user.should_see(marker="cache-search")
            await user.should_see(marker="cache-kind-filter")
            await user.should_see(marker="cache-sort")
            await user.should_see("Simulation result")
            await user.should_see("profile.csv")
            raw_keep = cast(
                Checkbox,
                next(iter(user.find(marker=f"cache-keep-{raw.id}").elements)),
            )
            upload_keep = cast(
                Checkbox,
                next(iter(user.find(marker=f"cache-keep-{upload.id}").elements)),
            )
            workflow_keep = cast(
                Checkbox,
                next(iter(user.find(marker=f"cache-keep-{workflow.id}").elements)),
            )
            assert raw_keep.value is True
            assert upload_keep.value is False
            assert workflow_keep.value is True
            with user.client:
                upload_keep.set_value(True)
            await asyncio.sleep(0)
            retained_upload = cache.get(upload.id)
            assert retained_upload is not None
            assert retained_upload.keep is True

            kind_filter = cast(
                Select,
                next(iter(user.find(marker="cache-kind-filter").elements)),
            )
            with user.client:
                kind_filter.set_value("plot")
            await asyncio.sleep(0)
            await user.should_see("timing.png")
            await user.should_not_see("timing.npy.gz")

            user.find(marker=f"cache-remove-{plot.id}").click()
            await user.should_not_see(marker=f"cache-entry-{plot.id}")
            assert cache.get(plot.id) is None

            with user.client:
                kind_filter.set_value("all")
            await asyncio.sleep(0)
            await user.should_see("timing.npy.gz")
            user.find("Workflow results").click()
            await user.should_see("Simulation result")
            await user.should_see(marker=f"cache-download-{workflow.id}")
            user.find(marker=f"cache-open-{workflow.id}").click()
            await user.should_see("Simulate DNA replication")
            await user.should_see("Restored from Cached Files")
            await user.should_see(marker="simulation-result-plot")

            replicates = cast(
                Number,
                next(iter(user.find(marker="simulation-replicates").elements)),
            )
            with user.client:
                replicates.set_value(7)
            user.find(marker="nav-cached-files").click()
            user.find(marker="clear-all-cache").click()
            await user.should_see("Clear all cached files?")
            user.find(marker="confirm-clear-all-cache").click()
            assert cache.count() == 0
            assert cache.get(raw.id) is None
            assert cache.get(upload.id) is None
            assert cache.get(workflow.id) is None

            user.find(marker="nav-simulation").click()
            await user.should_not_see(marker="simulation-result-plot")
            await user.should_see("No result yet")
            restored_replicates = cast(
                Number,
                next(iter(user.find(marker="simulation-replicates").elements)),
            )
            assert restored_replicates.value == 7

    asyncio.run(scenario())


def test_cached_inventory_refreshes_after_results_are_added_on_another_page() -> None:
    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-cached-files").click()
            await user.should_see("No cached items yet")
            user.find(marker="nav-overview").click()
            assert user.client is not None
            with user.client:
                current_session_cache().add_artifact(
                    ExportArtifact(b"values", "new-result.tsv.gz", "application/gzip"),
                    source="Simulation",
                    kind="raw",
                )
            user.find(marker="nav-cached-files").click()
            await user.should_see("new-result.tsv.gz")

    asyncio.run(scenario())


def test_constant_simulation_is_added_to_the_default_cache_inventory() -> None:
    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-simulation").click()
            await user.should_see("Simulate DNA replication")
            assert user.client is not None

            region_end = cast(
                Number,
                next(iter(user.find(marker="simulation-region-end").elements)),
            )
            replicates = cast(
                Number,
                next(iter(user.find(marker="simulation-replicates").elements)),
            )
            initiation_rate = cast(
                Number,
                next(iter(user.find(marker="source-initiation-rate-constant").elements)),
            )
            with user.client:
                region_end.set_value(20.0)
                replicates.set_value(1)
                initiation_rate.set_value(0.1)

            user.find(marker="action-simulate").click()
            await user.should_see(marker="simulation-result-plot", retries=120)
            await user.should_see("Complete in", retries=30)
            with user.client:
                cache = current_session_cache()
                assert cache.count("workflows") == 1

            user.find(marker="nav-cached-files").click()
            await user.should_see("Session files and results")
            await user.should_see("Simulation result")
            await user.should_see("1 replicates")

    asyncio.run(scenario())


def test_cached_mapping_result_reopens_in_the_mapping_preview() -> None:
    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            assert user.client is not None
            user.find(marker="nav-mapping").click()
            result = MappingResult(
                target="initiation_rate",
                target_key="initiation_rate",
                target_label="Initiation rate",
                method="cached_test",
                output_form="x",
                output_dependence="space",
                primary_output="initiation_rate",
                outputs={"initiation_rate": np.asarray([0.1, 0.3, 0.2])},
                output_forms={"initiation_rate": "x"},
                x_axis=np.asarray([10.0, 11.0, 12.0]),
                time_axis=np.asarray([], dtype=float),
                region_labels=(),
                elapsed_seconds=0.1,
                assumptions={},
            )
            with user.client:
                cache = current_session_cache()
                workflow = cache.add_reference(
                    name="Mapping: Initiation rate",
                    source="Mapping",
                    detail="Space | 3 points | Cached Test",
                    payload=result,
                    navigate_to="mapping",
                )

            user.find(marker="nav-cached-files").click()
            await user.should_see("Session files and results")
            with user.client:
                activity = current_page_activity()
                completed_run = activity.begin("mapping")
                activity.finish("mapping", completed_run, success=True)
            await user.should_see(marker="nav-mapping-complete", retries=30)
            user.find("Workflow results").click()
            await user.should_see("Mapping: Initiation rate")
            user.find(marker=f"cache-open-{workflow.id}").click()
            await user.should_see("Map replication kinetic landscapes")
            await user.should_see("Restored from Cached Files")
            await user.should_not_see(marker="nav-mapping-complete")
            await user.should_see(marker="mapping-result-plot")
            chart = cast(
                EChart,
                next(iter(user.find(marker="mapping-result-plot").elements)),
            )
            assert chart.options["series"][0]["data"] == [
                [10.0, 0.1],
                [11.0, 0.3],
                [12.0, 0.2],
            ]

    asyncio.run(scenario())


def test_scientific_data_sections_share_the_outline_data_symbol() -> None:
    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            for slug, heading, expected_icons in (
                ("mapping", "Map replication kinetic landscapes", 2),
                ("simulation", "Simulate DNA replication", 2),
                ("plotting", "Explore DNAscape data", 2),
            ):
                user.find(marker=f"nav-{slug}").click()
                await user.should_see(heading)
                icons = [
                    cast(Html, icon) for icon in user.find(marker="data-outline-icon").elements
                ]
                assert len(icons) == expected_icons
                assert all(icon.content == DATA_ICON_SVG for icon in icons)

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
            assert "mini" not in drawer._props
            toggle = next(iter(user.find(marker="drawer-toggle").elements))
            assert toggle._props["icon"] == "first_page"

            user.find(marker="drawer-toggle").click()
            assert drawer.value is False
            assert toggle.visible is False
            expand = next(iter(user.find(marker="drawer-expand").elements))
            assert expand.visible is True
            assert expand._props["icon"] == "last_page"
            assert "mini" not in drawer._props

            user.find(marker="drawer-expand").click()
            assert drawer.value is True
            assert toggle.visible is True
            assert expand.visible is False
            assert "mini" not in drawer._props
            user.find(marker="nav-data-management").click()
            await user.should_see("Convert genomic data files")

    asyncio.run(scenario())


def test_application_menu_updates_theme_action_and_exposes_project_links() -> None:
    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            dark_mode = cast(DarkMode, next(iter(user.find(kind=DarkMode).elements)))
            assert dark_mode.value is None
            user.find(marker="app-menu-toggle").click()
            await user.should_see("System theme")
            for marker in ("menu-documentation", "menu-github", "menu-issue"):
                await user.should_see(marker=marker)
            with user.scope(marker="menu-issue"):
                issue_icon = next(iter(user.find(kind=Icon).elements))
                assert issue_icon._props["name"] == "lightbulb"
            theme_icon = next(iter(user.find(marker="theme-action-icon").elements))
            assert theme_icon._props["name"] == "brightness_auto"

            user.find(marker="menu-theme").click()
            user.find(marker="app-menu-toggle").click()
            await user.should_see("Light theme")
            assert dark_mode.value is False
            theme_icon = next(iter(user.find(marker="theme-action-icon").elements))
            assert theme_icon._props["name"] == "light_mode"

            user.find(marker="menu-theme").click()
            user.find(marker="app-menu-toggle").click()
            await user.should_see("Dark theme")
            assert dark_mode.value is True
            theme_icon = next(iter(user.find(marker="theme-action-icon").elements))
            assert theme_icon._props["name"] == "dark_mode"

            user.find(marker="menu-theme").click()
            user.find(marker="app-menu-toggle").click()
            await user.should_see("System theme")
            assert dark_mode.value is None
            theme_icon = next(iter(user.find(marker="theme-action-icon").elements))
            assert theme_icon._props["name"] == "brightness_auto"

    asyncio.run(scenario())


def test_scientific_pages_share_expansion_and_action_dock_hooks() -> None:
    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            runnable_destinations = (
                ("mapping", "Map replication kinetic landscapes", 4, "action-map"),
                ("simulation", "Simulate DNA replication", 4, "action-simulate"),
            )
            for slug, heading, minimum_sections, action_marker in runnable_destinations:
                user.find(marker=f"nav-{slug}").click()
                await user.should_see(heading)
                expansions = user.find(kind=Expansion).elements
                assert len(expansions) >= minimum_sections
                assert all("dn-expansion" in expansion._classes for expansion in expansions)
                assert all("expand-separator" not in expansion._props for expansion in expansions)
                expansion_by_label = {
                    expansion._props.get("label"): expansion for expansion in expansions
                }
                plotting_settings = expansion_by_label["Plot settings"]
                raw_data = expansion_by_label["Raw data"]
                assert "dn-preview-settings-section" in plotting_settings._classes
                assert "dn-output-section" in raw_data._classes
                assert "dn-expansion" in plotting_settings._classes
                assert "dn-expansion" in raw_data._classes

                dock = next(iter(user.find(marker="scientific-action-dock").elements))
                assert "dn-action-dock" in dock._classes
                with user.scope(marker="scientific-action-dock"):
                    await user.should_see(marker=action_marker)

            user.find(marker="nav-plotting").click()
            await user.should_see("Explore DNAscape data")
            expansions = user.find(kind=Expansion).elements
            expansion_by_label = {
                expansion._props.get("label"): expansion for expansion in expansions
            }
            assert set(expansion_by_label) == {"Data", "Analysis", "Data transformations", "Plot settings", "Raw data"}
            plotting_settings = expansion_by_label["Plot settings"]
            assert "dn-preview-settings-section" in plotting_settings._classes
            assert "dn-expansion" in plotting_settings._classes
            await user.should_see(marker="scientific-action-dock")
            with user.scope(marker="scientific-action-dock"):
                await user.should_see(marker="action-plot")

    asyncio.run(scenario())


def test_plotting_uses_the_axis_and_line_symbol_in_navigation() -> None:
    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            with user.scope(marker="nav-plotting"):
                navigation_icon = next(iter(user.find(kind=Html).elements))
                assert "dn-plot-icon-axis" in navigation_icon.content
                assert "dn-plot-icon-line" in navigation_icon.content

            user.find(marker="nav-plotting").click()
            await user.should_see("Explore DNAscape data")
            await user.should_see(marker="action-plot")
            await user.should_see(marker="plotting-data-upload")

    asyncio.run(scenario())


def test_upload_fields_share_clear_copy_and_route_aware_mapping_defaults() -> None:
    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-data-management").click()
            await user.should_see(marker="data-manager-upload")
            await user.should_not_see(
                "Accepted: CSV, TSV, TXT, NPY, NPV, NPZ, bedGraph, BigWig, GZ"
            )

            user.find(marker="nav-mapping").click()
            await user.should_see(marker="source-replication-timing-upload")
            await user.should_see("Choose file or drag it here")
            await user.should_not_see(
                "Accepted: CSV, TSV, TXT, NPY, NPV, NPZ, bedGraph, BigWig, GZ"
            )
            await user.should_not_see(marker="mapping-input-replication-timing-shape-guidance")

            user.find(marker="nav-plotting").click()
            await user.should_see("Explore DNAscape data")
            await user.should_see(marker="plotting-data-upload")
            await user.should_not_see("Accepted: CSV, TSV, TXT, NPY, NPV, NPZ, GZ")

    asyncio.run(scenario())


def test_data_manager_preserves_data_and_queues_page_local_converted_files() -> None:
    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-data-management").click()
            await user.should_see("Convert genomic data files")
            uploader = cast(
                Upload,
                next(iter(user.find(marker="data-manager-upload").elements)),
            )
            await uploader.handle_uploads(
                [
                    Upload.SmallFileUpload(
                        "signal.csv",
                        "text/csv",
                        b"position,signal\n0,1\n1,3\n2,2\n",
                    )
                ]
            )

            await user.should_see("Dependence: Space", retries=30)
            await user.should_see("Array size: 3")
            await user.should_see(marker="data-manager-upload")
            assert uploader._props.get("multiple") is True
            assert uploader.visible is True
            await user.should_not_see(marker="data-manager-output-form")
            await user.should_not_see(marker="data-manager-missing-values")
            await user.should_not_see(marker="data-manager-gzip")

            await user.should_not_see(marker="data-manager-result-plot")
            await user.should_see(marker="data-manager-empty-result")
            user.find(marker="action-convert").click()
            await user.should_see(marker="data-manager-result-1", retries=60)
            await user.should_see(marker="data-manager-result-preview", retries=60)
            await user.should_see("Complete in", retries=60)
            await user.should_not_see(marker="data-manager-result-plot")
            await user.should_not_see(marker="data-manager-empty-result")

            user.find(marker="data-manager-upload-remove").click()
            await user.should_see(marker="data-manager-upload")
            await user.should_not_see(marker="data-manager-result-plot")
            await user.should_see(marker="action-download-converted-data")

            second_uploader = cast(
                Upload,
                next(iter(user.find(marker="data-manager-upload").elements)),
            )
            await second_uploader.handle_uploads(
                [
                    Upload.SmallFileUpload(
                        "second.csv",
                        "text/csv",
                        b"position,signal\n0,5\n1,7\n",
                    )
                ]
            )
            await user.should_see("Array size: 2", retries=30)
            user.find(marker="action-convert").click()
            await user.should_see(marker="data-manager-result-2", retries=60)
            await user.should_see(marker="data-manager-result-1")
            await user.should_see(marker="delete-all-converted-data")

            user.find(marker="remove-converted-data-1").click()
            await user.should_not_see(marker="data-manager-result-1")
            await user.should_see(marker="data-manager-result-2")
            user.find(marker="delete-all-converted-data").click()
            await user.should_see(marker="data-manager-empty-result")

            user.find(marker="nav-mapping").click()
            await user.should_see("Map replication kinetic landscapes")
            user.find(marker="nav-data-management").click()
            await user.should_see(marker="data-manager-empty-result")

    asyncio.run(scenario())


def test_data_manager_uses_dnascape_archive_dependence_as_its_output_default() -> None:
    payload = io.BytesIO()
    np.savez(
        payload,
        values=np.asarray([1.0, 2.0, 3.0]),
        data_form=np.asarray("time"),
    )

    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-data-management").click()
            await user.should_see(marker="data-manager-upload")
            uploader = cast(
                Upload,
                next(iter(user.find(marker="data-manager-upload").elements)),
            )
            await uploader.handle_uploads(
                [
                    Upload.SmallFileUpload(
                        "timing.npz",
                        "application/zip",
                        payload.getvalue(),
                    )
                ]
            )

            await user.should_see("Dependence: Time", retries=30)
            await user.should_not_see(marker="data-manager-output-form")
            user.find(marker="action-convert").click()
            await user.should_see(marker="data-manager-result-1", retries=60)
            await user.should_see("Time")

    asyncio.run(scenario())


def test_plotting_commits_data_once_then_updates_plot_settings_live() -> None:
    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-plotting").click()
            await user.should_see("Explore DNAscape data")
            uploader = cast(
                Upload,
                next(iter(user.find(marker="plotting-data-upload").elements)),
            )

            await uploader.handle_uploads(
                [
                    Upload.SmallFileUpload(
                        "signal.csv",
                        "text/csv",
                        b"position,signal\n0,1\n1,3\n2,2\n",
                    )
                ]
            )
            await user.should_see("Dependence: Space", retries=30)
            await user.should_see("Array size: 3")
            await user.should_see(marker="plotting-data-upload")
            assert uploader._props.get("multiple") is True
            await user.should_not_see(marker="action-add-plot-profile")
            await user.should_not_see(marker="plotting-profile-selection-1")
            await user.should_not_see(marker="plotting-result-plot")
            await user.should_see("No plot yet")

            user.find(marker="action-plot").click()
            await user.should_see(marker="plotting-result-plot", retries=30)

            def chart() -> EChart:
                return cast(
                    EChart,
                    next(iter(user.find(marker="plotting-result-plot").elements)),
                )

            async def wait_for_chart(predicate: object) -> None:
                for _ in range(60):
                    if cast(Any, predicate)(chart().options):
                        return
                    await asyncio.sleep(0.1)
                raise AssertionError("plot did not reach the expected committed state")

            assert chart().options["series"][0]["type"] == "line"
            assert chart().options["series"][0]["name"] == "signal"
            assert chart().options["xAxis"]["name"] == "Position (kb)"

            user.find("Plot settings").click()
            await user.should_not_see("Curves")
            await user.should_see(marker="plotting-primary-color")
            await user.should_see(marker="plotting-line-width")
            assert user.client is not None
            title = cast(Input, next(iter(user.find(marker="plotting-x-axis-label").elements)))
            thickness = cast(
                Number,
                next(iter(user.find(marker="plotting-line-width").elements)),
            )
            curve_color = cast(
                Input,
                next(iter(user.find(marker="plotting-primary-color").elements)),
            )
            grid = cast(
                Checkbox,
                next(iter(user.find(marker="plotting-show-grid").elements)),
            )
            result_container = next(
                iter(user.find(marker="plotting-result-container").elements)
            )
            shared_settings = next(
                iter(user.find(marker="plotting-shared-settings").elements)
            )
            assert result_container.parent_slot is not None
            assert shared_settings.parent_slot is not None
            preview_card = result_container.parent_slot.parent
            assert shared_settings.parent_slot.parent is preview_card
            assert preview_card.default_slot.children.index(
                result_container
            ) < preview_card.default_slot.children.index(shared_settings)
            assert thickness.parent_slot is not None
            assert curve_color.parent_slot is not None
            color_surface = curve_color.parent_slot.parent
            assert color_surface.parent_slot is not None
            color_grid = color_surface.parent_slot.parent
            assert color_grid.parent_slot is not None
            style_group = color_grid.parent_slot.parent
            assert thickness.parent_slot.parent is color_grid
            assert any(
                isinstance(child, Label) and child.text == "Style"
                for child in style_group.default_slot.children
            )
            with user.client:
                title.set_value("Edited profile")
                thickness.set_value(3.2)
                curve_color.set_value("#8b6f37")
                grid.set_value(False)
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            await wait_for_chart(lambda options: options["xAxis"]["name"] == "Edited profile")
            assert chart().options["xAxis"]["name"] == "Edited profile"
            assert chart().options["series"][0]["lineStyle"]["width"] == pytest.approx(3.2)
            assert chart().options["series"][0]["lineStyle"]["color"] == "#8b6f37"
            assert chart().options["xAxis"]["splitLine"]["show"] is False

            legend_position = cast(
                Select,
                next(iter(user.find(marker="plotting-legend-position").elements)),
            )
            original_grid = dict(chart().options["grid"])
            for side in ("left", "right"):
                with user.client:
                    legend_position.set_value(f"Bottom {side}")
                await wait_for_chart(
                    lambda options: "bottom" in options["legend"] and side in options["legend"]
                )
                assert chart().options["legend"]["bottom"] > chart().options["grid"]["bottom"]
                assert chart().options["grid"] == original_grid
            with user.client:
                legend_position.set_value("Top right")
            await wait_for_chart(lambda options: "top" in options["legend"])

            profile_view = cast(
                Select,
                next(iter(user.find(marker="plotting-view").elements)),
            )
            with user.client:
                profile_view.set_value("Histogram")
            await asyncio.sleep(0)
            histogram_bins = cast(
                Number,
                next(iter(user.find(marker="plotting-histogram-bins").elements)),
            )
            histogram_bin_size = cast(
                Number,
                next(iter(user.find(marker="plotting-histogram-bin-size").elements)),
            )
            with user.client:
                histogram_bins.set_value(4)
            await asyncio.sleep(0)
            assert histogram_bin_size.value == pytest.approx(0.5)
            await wait_for_chart(
                lambda options: options["series"][0]["type"] == "custom"
                and len(options["series"][0]["data"]) == 4
            )
            statistics_position = cast(
                Select,
                next(
                    iter(
                        user.find(marker="plotting-legend-position").elements
                    )
                ),
            )
            with user.client:
                statistics_position.set_value("Bottom left")
            await asyncio.sleep(0)
            await wait_for_chart(
                lambda options: bool(options["graphic"])
                and options["graphic"][0].get("left") == options["grid"]["left"] + 8
                and options["graphic"][0].get("bottom") == options["grid"]["bottom"] + 8
            )
            summary = chart().options["graphic"][0]
            assert summary["left"] == chart().options["grid"]["left"] + 8
            assert summary["bottom"] == chart().options["grid"]["bottom"] + 8
            assert "Mean" in summary["style"]["text"]
            assert "Median" in summary["style"]["text"]
            assert "Std dev" in summary["style"]["text"]
            with user.client:
                histogram_bin_size.set_value(None)
            await asyncio.sleep(0)
            await asyncio.sleep(0.1)
            assert len(chart().options["series"][0]["data"]) == 4
            await user.should_not_see("Plot unavailable")

            with user.client:
                histogram_bin_size.set_value(1.0)
            await asyncio.sleep(0)
            assert histogram_bins.value == 2
            await wait_for_chart(lambda options: len(options["series"][0]["data"]) == 2)

            x_minimum = cast(
                Number,
                next(iter(user.find(marker="plotting-histogram-x-minimum").elements)),
            )
            x_maximum = cast(
                Number,
                next(iter(user.find(marker="plotting-histogram-x-maximum").elements)),
            )
            with user.client:
                histogram_bin_size.set_value(0.333333)
            await asyncio.sleep(0)
            assert histogram_bin_size.value == pytest.approx(0.333333)
            assert histogram_bins.value == 7
            with user.client:
                x_minimum.set_value(1.25)
                x_maximum.set_value(2.75)
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            assert histogram_bins.value == 5
            assert histogram_bin_size.value == pytest.approx(0.333333)
            await wait_for_chart(
                lambda options: len(options["series"][0]["data"]) == 5
                and options["xAxis"]["min"] == pytest.approx(1.25)
                and options["xAxis"]["max"] == pytest.approx(2.75)
            )
            summary_text = chart().options["graphic"][0]["style"]["text"]
            assert "Mean       2" in summary_text
            user.find(marker="plotting-reset-histogram-x-limits").click()
            await asyncio.sleep(0)
            assert x_minimum.value is not None
            assert x_maximum.value is not None
            assert float(x_minimum.value) == pytest.approx(1.0)
            assert float(x_maximum.value) == pytest.approx(3.0)
            assert histogram_bins.value == 7

            with user.client:
                profile_view.set_value("Default")
            await wait_for_chart(lambda options: options["series"][0]["type"] == "line")
            assert chart().options["series"][0]["lineStyle"]["color"] == "#8b6f37"
            assert chart().options["series"][0]["lineStyle"]["width"] == pytest.approx(3.2)

            user.find(marker="plotting-data-upload-remove").click()
            await user.should_not_see("signal.csv")
            await user.should_see(marker="plotting-result-plot")
            await user.should_see(marker="plotting-data-upload")

            uploader = cast(
                Upload,
                next(iter(user.find(marker="plotting-data-upload").elements)),
            )
            await uploader.handle_uploads(
                [
                    Upload.SmallFileUpload(
                        "space_time.csv",
                        "text/csv",
                        b"position,t0,t1\n0,0.1,0.2\n1,0.3,0.4\n",
                    )
                ]
            )
            await user.should_see("Dependence: Space-time", retries=30)
            await user.should_see("Array size: 2 x 2")
            assert chart().options["series"][0]["type"] == "line"
            # Plot controls follow the committed chart, as they do in Mapping;
            # merely staging a heatmap must not relabel the visible line plot.
            await user.should_not_see(marker="plotting-heatmap-palette")
            await user.should_not_see(marker="plotting-colorbar")
            await user.should_not_see("Curves")
            await user.should_see(marker="plotting-primary-color")
            await user.should_see(marker="plotting-line-width")

            user.find(marker="action-plot").click()
            await wait_for_chart(lambda options: options["series"][0]["type"] == "custom")
            assert chart().options["series"][0]["type"] == "custom"
            assert chart().options["series"][0]["silent"] is True
            assert chart().options["tooltip"]["show"] is True
            await user.should_see(marker="plotting-heatmap-palette")
            await user.should_see(marker="plotting-colorbar")
            await user.should_see(marker="plotting-log-z")
            await user.should_not_see(marker="plotting-line-width")
            view = cast(Select, next(iter(user.find(marker="plotting-view").elements)))
            assert view.options == ["Default"]

            profiles = cast(
                Select,
                next(
                    control
                    for control in user.find(kind=Select).elements
                    if control._props.get("data-upload-name") == "space_time.csv"
                ),
            )
            with user.client:
                profiles.set_value(["t0", "t1"])
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            await user.should_see("Dependence: Space")
            assert chart().options["series"][0]["type"] == "custom"

            user.find(marker="action-plot").click()
            await wait_for_chart(lambda options: len(options["series"]) == 2 and all(
                series["type"] == "line" for series in options["series"]
            ))
            user.find("Curves").click()
            curves_expansion = next(
                expansion
                for expansion in user.find(kind=Expansion).elements
                if expansion._props.get("label") == "Curves"
            )
            assert "dn-preview-settings-section" in curves_expansion._classes
            assert "dn-plot-curve-settings" in curves_expansion._classes
            await user.should_see(marker="plotting-curve-color-1")
            await user.should_see(marker="plotting-curve-color-2")
            assert [series["name"] for series in chart().options["series"]] == ["t0", "t1"]
            assert [series["type"] for series in chart().options["series"]] == ["line", "line"]
            assert chart().options["series"][0]["lineStyle"]["color"] == "#b03412"
            assert chart().options["series"][1]["lineStyle"]["color"] == "#2596be"

            assert chart().options["series"][0]["z"] > chart().options["series"][1]["z"]
            name = next(iter(user.find(marker="plotting-curve-name-1").elements))
            with user.client:
                name.set_value("First profile")
            await wait_for_chart(lambda options: options["series"][0]["name"] == "First profile")
            assert chart().options["legend"]["data"][0] == "First profile"
            with user.client:
                name.set_value("")
            await wait_for_chart(lambda options: options["series"][0]["name"] == "t0")

            await uploader.handle_uploads(
                [
                    Upload.SmallFileUpload(
                        "other.csv",
                        "text/csv",
                        b"position,other\n0,4\n1,5\n",
                    )
                ]
            )
            await user.should_see("other.csv", retries=30)
            assert len(chart().options["series"]) == 2

            # Each file's controls and metadata belong directly below its row,
            # with the bulk removal action after the final file's details.
            for name, source_id in (("space_time.csv", 2), ("other.csv", 3)):
                file_row = next(
                    row for row in user.find(marker="plotting-data-upload-file").elements
                    if row._props.get("title") == name
                )
                details = next(iter(user.find(
                    marker=f"plotting-profile-source-{source_id}"
                ).elements))
                assert file_row.parent_slot is not None
                assert details.parent_slot is file_row.parent_slot
                children = file_row.parent_slot.children
                assert children[children.index(file_row) + 1] is details
            delete_all = next(iter(user.find(
                marker="plotting-data-upload-delete-all"
            ).elements))
            assert delete_all.parent_slot is not None
            assert children[-1] is delete_all.parent_slot.parent
            with user.scope(marker="plotting-profile-source-3"):
                await user.should_see("Dependence: Space")
                await user.should_see("Array size: 2")

            user.find(marker="action-plot").click()
            await wait_for_chart(lambda options: len(options["series"]) == 3)
            assert [series["name"] for series in chart().options["series"]] == [
                "t0",
                "t1",
                "other",
            ]

            # Dropping a same-name table replaces its retained values without
            # adding another row or disturbing the source order.
            await uploader.handle_uploads(
                [
                    Upload.SmallFileUpload(
                        "other.csv",
                        "text/csv",
                        b"position,other\n0,8\n1,9\n2,10\n",
                    )
                ]
            )
            await user.should_see("Array size: 3", retries=30)
            assert len(user.find(marker="plotting-data-upload-file").elements) == 2
            with user.scope(marker="plotting-profile-source-3"):
                await user.should_see("Array size: 3")
            user.find(marker="action-plot").click()
            await wait_for_chart(
                lambda options: options["series"][2]["data"]
                == [[0.0, 8.0], [1.0, 9.0], [2.0, 10.0]]
            )

            remove_button = next(
                button
                for button in user.find(
                    marker="plotting-data-upload-remove"
                ).elements
                if button._props.get("data-upload-name") == "other.csv"
            )
            assert len(user.find(marker="plotting-data-upload-remove").elements) == 2
            UserInteraction(user, {remove_button}, None).click()
            await user.should_not_see("other.csv", retries=30)
            await user.should_not_see(marker="plotting-profile-source-3")
            assert len(chart().options["series"]) == 3

    asyncio.run(scenario())


def test_plotting_preserves_styles_when_switching_between_one_and_multiple_profiles() -> None:
    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-plotting").click()
            uploader = cast(
                Upload,
                next(iter(user.find(marker="plotting-data-upload").elements)),
            )
            await uploader.handle_uploads(
                [
                    Upload.SmallFileUpload(
                        "first.csv",
                        "text/csv",
                        b"position,first\n0,1\n1,3\n2,2\n",
                    )
                ]
            )
            await user.should_see("Dependence: Space", retries=30)
            user.find("Plot settings").click()
            color = cast(
                Input,
                next(iter(user.find(marker="plotting-primary-color").elements)),
            )
            thickness = cast(
                Number,
                next(iter(user.find(marker="plotting-line-width").elements)),
            )
            assert color.value == "#b03412"
            assert user.client is not None
            with user.client:
                color.set_value("#8b6f37")
                thickness.set_value(3.2)
            user.find(marker="action-plot").click()
            await user.should_see(marker="plotting-result-plot", retries=30)
            await user.should_not_see("Curves")

            def chart() -> EChart:
                return cast(
                    EChart,
                    next(iter(user.find(marker="plotting-result-plot").elements)),
                )

            async def wait_for_chart(predicate: Any) -> None:
                for _ in range(60):
                    if predicate(chart().options):
                        return
                    await asyncio.sleep(0.1)
                raise AssertionError("plot did not reach the expected committed state")

            assert chart().options["series"][0]["lineStyle"]["color"] == "#8b6f37"
            assert chart().options["series"][0]["lineStyle"]["width"] == pytest.approx(3.2)
            # Live shared Style edits must also survive adding another curve.
            with user.client:
                thickness.set_value(3.6)
            await wait_for_chart(
                lambda options: options["series"][0]["lineStyle"]["width"] == 3.6
            )
            await uploader.handle_uploads(
                [
                    Upload.SmallFileUpload(
                        "second.csv",
                        "text/csv",
                        b"position,second\n0,4\n1,6\n2,5\n",
                    )
                ]
            )
            await user.should_see("second.csv", retries=30)
            await user.should_not_see("Curves")
            user.find(marker="action-plot").click()
            await wait_for_chart(lambda options: len(options["series"]) == 2)
            user.find("Curves").click()
            await user.should_not_see(marker="plotting-primary-color")
            await user.should_not_see(marker="plotting-line-width")
            first_color = cast(
                Input,
                next(iter(user.find(marker="plotting-curve-color-1").elements)),
            )
            first_thickness = cast(
                Number,
                next(iter(user.find(marker="plotting-curve-width-1").elements)),
            )
            assert first_color.value == "#8b6f37"
            assert first_thickness.value == pytest.approx(3.6)
            second_color = cast(
                Input,
                next(iter(user.find(marker="plotting-curve-color-2").elements)),
            )
            second_thickness = cast(
                Number,
                next(iter(user.find(marker="plotting-curve-width-2").elements)),
            )
            with user.client:
                second_color.set_value("#6f5b8f")
                second_thickness.set_value(4.4)
            await wait_for_chart(
                lambda options: options["series"][1]["lineStyle"]["color"] == "#6f5b8f"
                and options["series"][1]["lineStyle"]["width"] == 4.4
            )
            remove_first = next(
                button
                for button in user.find(marker="plotting-data-upload-remove").elements
                if button._props.get("data-upload-name") == "first.csv"
            )
            UserInteraction(user, {remove_first}, None).click()
            await user.should_not_see("first.csv", retries=30)
            # Removing a staged upload does not change the committed controls.
            await user.should_see("Curves")
            user.find(marker="action-plot").click()
            await wait_for_chart(lambda options: len(options["series"]) == 1)
            await user.should_not_see("Curves")
            await user.should_see(marker="plotting-primary-color")
            await user.should_see(marker="plotting-line-width")
            assert color.value == "#6f5b8f"
            assert thickness.value == pytest.approx(4.4)
            assert chart().options["series"][0]["name"] == "second"
            assert chart().options["series"][0]["lineStyle"]["color"] == "#6f5b8f"
            assert chart().options["series"][0]["lineStyle"]["width"] == pytest.approx(4.4)

    asyncio.run(scenario())


def test_plotting_separates_spatial_curves_and_heatmaps_and_blocks_time_heatmaps() -> None:
    async def separate_space_and_heatmap() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-plotting").click()
            first_uploader = cast(
                Upload,
                next(iter(user.find(marker="plotting-data-upload").elements)),
            )
            await first_uploader.handle_uploads(
                [
                    Upload.SmallFileUpload(
                        "space.csv",
                        "text/csv",
                        b"position,signal\n0,1\n1,3\n2,2\n",
                    )
                ]
            )
            await user.should_see("space.csv", retries=30)
            await first_uploader.handle_uploads(
                [
                    Upload.SmallFileUpload(
                        "map.csv",
                        "text/csv",
                        b"position,t0,t1\n0,0.1,0.2\n1,0.3,0.4\n",
                    )
                ]
            )
            await user.should_see("map.csv", retries=30)
            assert len(user.find(marker="plotting-data-upload-file").elements) == 2
            await user.should_see(marker="plotting-presentation")

            presentation = cast(
                Select,
                next(iter(user.find(marker="plotting-presentation").elements)),
            )
            assert presentation.value == "Separate"
            assert presentation.options == ["Separate"]
            user.find(marker="action-plot").click()
            await user.should_see(marker="plotting-result-plot", retries=30)
            await user.should_see(marker="plotting-result-plot-2", retries=30)
            first_chart = cast(
                EChart,
                next(iter(user.find(marker="plotting-result-plot").elements)),
            )
            second_chart = cast(
                EChart,
                next(iter(user.find(marker="plotting-result-plot-2").elements)),
            )
            first_settings = next(
                iter(user.find(marker="plotting-separate-settings-surface-1").elements)
            )
            second_settings = next(
                iter(user.find(marker="plotting-separate-settings-surface-2").elements)
            )
            assert first_chart.parent_slot is not None
            assert second_chart.parent_slot is not None
            first_shell = first_chart.parent_slot.parent
            second_shell = second_chart.parent_slot.parent
            assert first_shell.parent_slot is not None
            assert second_shell.parent_slot is not None
            first_panel = first_shell.parent_slot.parent
            second_panel = second_shell.parent_slot.parent
            assert first_settings.parent_slot is not None
            assert second_settings.parent_slot is not None
            assert first_settings.parent_slot.parent is first_panel
            assert second_settings.parent_slot.parent is second_panel
            assert first_panel.default_slot.children.index(
                first_shell
            ) < first_panel.default_slot.children.index(first_settings)
            assert second_panel.default_slot.children.index(
                second_shell
            ) < second_panel.default_slot.children.index(second_settings)
            assert first_chart.options["series"][0]["type"] == "line"
            assert second_chart.options["series"][0]["type"] == "custom"

            first_title = cast(
                Input,
                next(iter(user.find(marker="plotting-x-axis-label-separate-1").elements)),
            )
            first_color = cast(
                Input,
                next(
                    iter(user.find(marker="plotting-primary-color-separate-1").elements)
                ),
            )
            second_title = cast(
                Input,
                next(iter(user.find(marker="plotting-x-axis-label-separate-2").elements)),
            )
            second_palette = cast(
                Select,
                next(
                    iter(user.find(marker="plotting-heatmap-palette-separate-2").elements)
                ),
            )
            first_palette_before = list(
                second_chart.options["visualMap"]["inRange"]["color"]
            )
            assert user.client is not None
            with user.client:
                first_title.set_value("Edited spatial profile")
                first_color.set_value("#8b6f37")
                second_title.set_value("Edited space-time map")
                second_palette.set_value("magma")
            for _ in range(40):
                if (
                    first_chart.options["xAxis"]["name"] == "Edited spatial profile"
                    and second_chart.options["xAxis"]["name"] == "Edited space-time map"
                    and first_chart.options["series"][0]["lineStyle"]["color"]
                    == "#8b6f37"
                    and second_chart.options["visualMap"]["inRange"]["color"]
                    != first_palette_before
                ):
                    break
                await asyncio.sleep(0.1)
            else:
                raise AssertionError("separate plotting controls did not update independently")

            # Each independent plot owns editable axes and resets, including
            # the colour range of its heatmap.
            for axis, limits in (("x", (-1.0, 3.0)), ("y", (-2.0, 4.0)), ("z", (0.0, 1.0))):
                lower = cast(Number, next(iter(user.find(marker=f"plotting-{axis}-minimum-separate-2").elements)))
                upper = cast(Number, next(iter(user.find(marker=f"plotting-{axis}-maximum-separate-2").elements)))
                reset = next(iter(user.find(marker=f"plotting-reset-{axis}-limits-separate-2").elements))
                assert lower.enabled and upper.enabled and reset.enabled
                key = "visualMap" if axis == "z" else f"{axis}Axis"
                original = (second_chart.options[key]["min"], second_chart.options[key]["max"])
                with user.client:
                    lower.set_value(limits[0])
                    upper.set_value(limits[1])
                await asyncio.sleep(0)
                assert (second_chart.options[key]["min"], second_chart.options[key]["max"]) == limits
                with user.client:
                    upper.set_value(None)
                user.find(marker=f"plotting-reset-{axis}-limits-separate-2").click()
                await asyncio.sleep(0)
                assert (second_chart.options[key]["min"], second_chart.options[key]["max"]) == original

            user.find(marker="plotting-data-upload-delete-all").click()
            await user.should_see(marker="plotting-data-upload")
            await user.should_not_see("space.csv")
            await user.should_not_see("map.csv")
            # Removing staged inputs follows Mapping's behavior: the last
            # committed preview remains available until it is replaced or the
            # application is reset.
            await user.should_see(marker="plotting-result-plot")
            await user.should_see(marker="plotting-result-plot-2")
            await user.should_see(marker="plotting-separate-settings-surface-1")
            await user.should_see(marker="plotting-separate-settings-surface-2")
            assert presentation.value == "Separate"
            assert presentation.visible is True

    async def reject_time_and_heatmap() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-plotting").click()
            first_uploader = cast(
                Upload,
                next(iter(user.find(marker="plotting-data-upload").elements)),
            )
            await first_uploader.handle_uploads(
                [
                    Upload.SmallFileUpload(
                        "time.csv",
                        "text/csv",
                        b"t0,t1,t2\n0.1,0.3,0.8\n",
                    )
                ]
            )
            await user.should_see("time.csv", retries=30)
            await first_uploader.handle_uploads(
                [
                    Upload.SmallFileUpload(
                        "map.csv",
                        "text/csv",
                        b"position,t0,t1\n0,0.1,0.2\n1,0.3,0.4\n",
                    )
                ]
            )
            await user.should_see("map.csv", retries=30)

            await user.should_see(
                "Time profiles cannot be mixed with space-time heatmaps."
            )
            action = cast(
                Button,
                next(iter(user.find(marker="action-plot").elements)),
            )
            assert action._props.get("disable") is True

    asyncio.run(separate_space_and_heatmap())
    asyncio.run(reject_time_and_heatmap())


def test_plotting_preserves_independent_settings_across_layout_toggles() -> None:
    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-plotting").click()
            first_uploader = cast(
                Upload,
                next(iter(user.find(marker="plotting-data-upload").elements)),
            )
            await first_uploader.handle_uploads(
                [
                    Upload.SmallFileUpload(
                        "first.csv",
                        "text/csv",
                        b"position,signal\n0,1\n1,3\n2,2\n",
                    )
                ]
            )
            await user.should_see("first.csv", retries=30)
            await first_uploader.handle_uploads(
                [
                    Upload.SmallFileUpload(
                        "second.csv",
                        "text/csv",
                        b"position,signal\n0,4\n1,6\n2,5\n",
                    )
                ]
            )
            await user.should_see("second.csv", retries=30)
            user.find(marker="action-plot").click()
            await user.should_see(marker="plotting-result-plot", retries=30)

            presentation = cast(
                Select,
                next(iter(user.find(marker="plotting-presentation").elements)),
            )
            assert presentation.options == ["Combined", "Separate"]
            assert user.client is not None
            with user.client:
                presentation.set_value("Separate")
            await user.should_see(marker="plotting-separate-settings-surface-1", retries=30)
            await user.should_see(marker="plotting-separate-settings-surface-2", retries=30)

            first_title = cast(
                Input,
                next(iter(user.find(marker="plotting-x-axis-label-separate-1").elements)),
            )
            first_color = cast(
                Input,
                next(iter(user.find(marker="plotting-primary-color-separate-1").elements)),
            )
            second_title = cast(
                Input,
                next(iter(user.find(marker="plotting-x-axis-label-separate-2").elements)),
            )
            second_color = cast(
                Input,
                next(iter(user.find(marker="plotting-primary-color-separate-2").elements)),
            )
            second_view = cast(
                Select,
                next(iter(user.find(marker="plotting-view-separate-2").elements)),
            )
            first_fill = next(iter(user.find(marker="plotting-fill-separate-1").elements))
            first_fill_side = next(item for item in user.client.elements.values()
                                   if "plotting-fill-side-separate-1" in item._markers and not item.is_deleted)
            with user.client:
                first_title.set_value("First custom plot")
                first_color.set_value("#8b6f37")
                first_fill.set_value(True)
                first_fill_side.set_value("above")
                second_title.set_value("Second distribution")
                second_color.set_value("#6f5b8f")
                second_view.set_value("Histogram")
            await asyncio.sleep(0)
            with user.client:
                second_title.set_value("Second distribution")
            second_bins = cast(
                Number,
                next(iter(user.find(marker="plotting-histogram-bins-separate-2").elements)),
            )
            with user.client:
                second_bins.set_value(3)
            await asyncio.sleep(0)

            with user.client:
                presentation.set_value("Combined")
            await user.should_not_see(marker="plotting-separate-settings-surface-1")
            with user.client:
                presentation.set_value("Separate")
            await user.should_see(marker="plotting-separate-settings-surface-1", retries=30)
            await user.should_see(marker="plotting-separate-settings-surface-2", retries=30)

            restored_first_title = cast(
                Input,
                next(iter(user.find(marker="plotting-x-axis-label-separate-1").elements)),
            )
            restored_first_color = cast(
                Input,
                next(iter(user.find(marker="plotting-primary-color-separate-1").elements)),
            )
            restored_second_title = cast(
                Input,
                next(iter(user.find(marker="plotting-x-axis-label-separate-2").elements)),
            )
            restored_second_color = cast(
                Input,
                next(iter(user.find(marker="plotting-primary-color-separate-2").elements)),
            )
            restored_second_view = cast(
                Select,
                next(iter(user.find(marker="plotting-view-separate-2").elements)),
            )
            restored_second_bins = cast(
                Number,
                next(iter(user.find(marker="plotting-histogram-bins-separate-2").elements)),
            )
            assert restored_first_title.value == "First custom plot"
            assert restored_first_color.value == "#8b6f37"
            assert restored_second_title.value == "Second distribution"
            assert restored_second_color.value == "#6f5b8f"
            assert restored_second_view.value == "Histogram"
            assert restored_second_bins.value == 3
            assert next(iter(user.find(marker="plotting-fill-separate-1").elements)).value is True
            assert next(iter(user.find(marker="plotting-fill-side-separate-1").elements)).value == "above"

            charts = [
                cast(EChart, next(iter(user.find(marker=marker).elements)))
                for marker in ("plotting-result-plot", "plotting-result-plot-2")
            ]
            assert charts[0].options["xAxis"]["name"] == "First custom plot"
            assert charts[0].options["series"][0]["lineStyle"]["color"] == "#8b6f37"
            assert charts[0].options["series"][0]["areaStyle"] == {
                "origin": "end", "color": "#8b6f37", "opacity": .15,
            }
            assert charts[1].options["xAxis"]["name"] == "Second distribution"
            assert charts[1].options["series"][0]["type"] == "custom"
            assert len(charts[1].options["series"][0]["data"]) == 3

    asyncio.run(scenario())


def test_scientific_previews_start_with_guidance_and_share_download_sections() -> None:
    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            for slug in ("mapping", "simulation"):
                user.find(marker=f"nav-{slug}").click()
                await user.should_see("Raw data")
                await user.should_see(marker="preview-guidance")
                if slug == "simulation":
                    await user.should_see("No result yet")
                    await user.should_not_see("Choose inputs and settings")
                await user.should_not_see(marker="plot-toolbar")
                user.find("Raw data").click()
                if slug == "mapping":
                    await user.should_not_see(marker="raw-output-row")
                    await user.should_not_see(marker="action-download-raw-data")
                else:
                    await user.should_see("Download")

            user.find(marker="nav-plotting").click()
            await user.should_see(marker="preview-guidance")
            await user.should_see("No plot yet")
            assert any(expansion._props.get("label") == "Raw data" for expansion in user.find(kind=Expansion).elements)
            await user.should_not_see(marker="plot-toolbar")

    asyncio.run(scenario())


def test_raw_output_fields_and_download_share_one_ordered_row() -> None:
    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            for slug in ("mapping", "simulation"):
                user.find(marker=f"nav-{slug}").click()
                await user.should_see("Raw data")
                user.find("Raw data").click()
                if slug == "mapping":
                    await user.should_not_see(marker="raw-output-row")
                else:
                    await user.should_see(marker="raw-output-row")
                assert user.client is not None
                if slug == "mapping":
                    raw_name = next(
                        element
                        for element in user.client.elements.values()
                        if "mapping-raw-filename" in element._markers
                        and not element.is_deleted
                    )
                    assert raw_name.parent_slot is not None
                    fields = raw_name.parent_slot.parent
                    assert fields.parent_slot is not None
                    row = fields.parent_slot.parent
                else:
                    row = next(iter(user.find(marker="raw-output-row").elements))
                children = row.default_slot.children
                download = cast(
                    Button,
                    next(
                        child
                        for child in children
                        if "action-download-raw-data" in child._markers
                    ),
                )
                assert "dn-download-file-row" in row._classes
                fields = next(
                    child for child in children if "dn-raw-output-fields" in child._classes
                )
                assert [
                    next(
                        class_name
                        for class_name in ("dn-raw-output-name", "dn-raw-output-type")
                        if class_name in child._classes
                    )
                    for child in fields.default_slot.children
                ] == ["dn-raw-output-name", "dn-raw-output-type"]
                assert any("dn-raw-output-compress" in child._classes for child in children)
                assert download in children
                assert download.parent_slot is not None
                assert download.parent_slot.parent is row
                assert download.text == ""
                file_type = cast(
                    Select,
                    next(
                        child
                        for child in fields.default_slot.children
                        if "dn-raw-output-type" in child._classes
                    ),
                )
                assert file_type.options == ["TSV", "TXT", "CSV", "NPZ"]
                assert file_type.value == "TSV"
                compression = cast(
                    Checkbox,
                    next(
                        child
                        for child in children
                        if "raw-output-compression" in child._markers
                    ),
                )
                assert compression.value is False

    asyncio.run(scenario())


def test_simulation_preview_header_and_output_summary_have_stable_structure() -> None:
    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-simulation").click()
            await user.should_see(marker="simulation-preview-header-row")

            preview_header = next(iter(user.find(marker="simulation-preview-header-row").elements))
            summary = next(iter(user.find(marker="simulation-preview-output-bar").elements))
            output_stat = next(iter(user.find(marker="simulation-output-stat").elements))
            output = cast(
                Select,
                next(iter(user.find(marker="simulation-output").elements)),
            )
            controls_title = next(
                element for element in user.find("Controls").elements
                if "dn-panel-title" in element._classes
            )

            assert "dn-preview-heading-row" in preview_header._classes
            assert preview_header.parent_slot is not None
            preview_card = preview_header.parent_slot.parent
            assert "dn-preview-card" in preview_card._classes
            assert preview_card.default_slot.children[:2] == [preview_header, summary]
            assert len(preview_header.default_slot.children) == 1
            preview_title = cast(Label, preview_header.default_slot.children[0])
            assert preview_title.text == "Preview"
            assert "dn-panel-title" in preview_title._classes
            assert controls_title.parent_slot is not None
            controls_card = controls_title.parent_slot.parent
            assert "dn-control-card" in controls_card._classes
            assert controls_card.default_slot.children[0] is controls_title
            assert "dn-panel-title" in controls_title._classes

            assert "dn-stat-grid" in summary._classes
            assert "dn-simulation-summary-grid" in summary._classes
            run_stats = summary.default_slot.children[1]
            assert summary.default_slot.children[:2] == [output_stat, run_stats]
            assert "simulation-stats" in run_stats._markers
            assert "dn-stat" in output_stat._classes
            assert output.parent_slot is not None
            assert output.parent_slot.parent is output_stat
            assert output.value == "Replication timing"

            run_stat_grid = run_stats.default_slot.children[0]
            stat_labels = [
                cast(Label, stat.default_slot.children[0]).text
                for stat in run_stat_grid.default_slot.children
            ]
            assert stat_labels == ["Simulations (replicates)", "Resolution", "Region", "Geometry"]
            await user.should_see(marker="simulation-output")
            await user.should_not_see(marker="simulation-stats")

    asyncio.run(scenario())


def test_scientific_controls_use_compact_mode_selects_and_keep_primary_actions_docked() -> None:
    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-mapping").click()
            await user.should_see("Map replication kinetic landscapes", retries=60)
            await user.should_see(marker="mapping-target", retries=60)
            target = cast(Select, next(iter(user.find(marker="mapping-target").elements)))
            dependence = cast(
                Select,
                next(iter(user.find(marker="mapping-target-form").elements)),
            )
            route = cast(Select, next(iter(user.find(marker="mapping-route").elements)))
            timing_type = cast(
                Select,
                next(iter(user.find(marker="mapping-input-replication-timing-type").elements)),
            )
            timing_form = _select_by_marker(
                user,
                "mapping-input-replication-timing-form",
            )
            speed_type = cast(
                Select,
                next(iter(user.find(marker="mapping-input-fork-speed-type").elements)),
            )
            speed_form = _select_by_marker(user, "mapping-input-fork-speed-form")
            assert target.value == "initiation_rate"
            assert dependence.value == "space"
            assert route.value == mapping_route_families("initiation_rate", "space")[0].key
            assert timing_type.options == {"profile": "Profile"}
            assert timing_type.value == "profile"
            assert timing_type._props["disable"] is True
            assert timing_form.options == {"x": "Space"}
            assert timing_form.value == "x"
            assert not timing_form.visible
            assert speed_type.options == {
                "scalar": "Constant",
                "constants": "Multiple constants",
                "profile": "Profile",
            }
            assert speed_type.value == "scalar"
            assert speed_form.options == {
                "scalar": "Constant",
                "x": "Space",
                "t": "Time",
                "xt": "Space–time",
            }
            assert speed_form.value == "scalar"
            assert not speed_form.visible
            await user.should_see(marker="mapping-input-fork-speed-constant")
            for section in ("Region", "Mapping settings"):
                user.find(section).click()
            await user.should_see(marker="scientific-action-dock")
            with user.scope(marker="scientific-action-dock"):
                await user.should_see(marker="action-map")

        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-simulation").click()
            await user.should_see(kind=Select, marker="source-initiation-rate-mode")
            simulation_mode = cast(
                Select,
                next(iter(user.find(marker="source-initiation-rate-mode").elements)),
            )
            simulation_form = _select_by_marker(user, "source-initiation-rate-form")
            assert simulation_mode.options == {
                "scalar": "Constant",
                "constants": "Multiple constants",
                "profile": "Profile",
            }
            assert simulation_mode.value == "scalar"
            assert simulation_mode._props["label"] == "Dependence"
            assert _select_by_marker(user, "source-fork-speed-mode")._props["label"] == "Dependence"
            assert simulation_form.options == {
                "x": "Space",
                "t": "Time",
                "xt": "Space-time",
            }
            assert not simulation_form.visible
            await user.should_not_see(marker="source-initiation-rate-profile")
            await user.should_see(marker="source-initiation-rate-constant")
            user.find(marker="source-initiation-rate-mode").click()
            user.find("Profile").click()
            await user.should_see(marker="source-initiation-rate-profile")
            await user.should_not_see(marker="source-initiation-rate-constant")
            user.find(marker="source-initiation-rate-mode").click()
            user.find("Constant").click()
            await user.should_see(marker="source-initiation-rate-constant")
            for section in ("Region", "Simulation settings"):
                user.find(section).click()
            replicates = cast(
                Number,
                next(iter(user.find(marker="simulation-replicates").elements)),
            )
            assert replicates._props["max"] == 50_000
            await user.should_see(marker="simulation-track-time-outputs")
            await user.should_see(marker="simulation-track-detailed-outputs")
            await user.should_see(marker="scientific-action-dock")
            with user.scope(marker="scientific-action-dock"):
                await user.should_see(marker="action-simulate")

        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-plotting").click()
            await user.should_see(marker="plotting-data-upload")
            await user.should_not_see(marker="plotting-profile-source-1")
            plotting_uploader = cast(
                Upload,
                next(iter(user.find(marker="plotting-data-upload").elements)),
            )
            assert plotting_uploader._props.get("multiple") is True
            await user.should_see(marker="plotting-settings")
            await user.should_not_see("Region")
            assert any(expansion._props.get("label") == "Raw data" for expansion in user.find(kind=Expansion).elements)
            await user.should_see(marker="scientific-action-dock")
            await user.should_see(marker="action-plot")

    asyncio.run(scenario())


def test_plot_controls_only_show_options_relevant_to_the_selected_view() -> None:
    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-mapping").click()
            await user.should_see("Map replication kinetic landscapes")
            user.find("Plot settings").click()
            for marker in (
                "mapping-heatmap-palette",
                "mapping-colorbar",
                "mapping-log-z",
                "mapping-histogram-bins",
                "mapping-histogram-bin-size",
                "mapping-histogram-touching",
                "mapping-histogram-boundary",
                "mapping-histogram-x-minimum",
                "mapping-histogram-x-maximum",
                "mapping-reset-histogram-x-limits",
                "mapping-show-histogram-statistics",
                "mapping-histogram-statistics-position",
            ):
                await user.should_not_see(marker=marker)
            for marker in ("mapping-line-width", "mapping-show-grid"):
                await user.should_see(marker=marker)
            await user.should_not_see(marker="mapping-show-markers")

            user.find(marker="mapping-view").click()
            user.find("Histogram").click()
            for marker in ("mapping-heatmap-palette", "mapping-colorbar", "mapping-log-z"):
                await user.should_not_see(marker=marker)
            await user.should_see(marker="mapping-histogram-bins")
            await user.should_see(marker="mapping-histogram-bin-size")
            await user.should_see(marker="mapping-histogram-touching")
            await user.should_see(marker="mapping-histogram-boundary")
            await user.should_see(marker="mapping-histogram-x-minimum")
            await user.should_see(marker="mapping-histogram-x-maximum")
            await user.should_see(marker="mapping-reset-histogram-x-limits")
            await user.should_see(marker="mapping-show-histogram-statistics")
            await user.should_not_see(marker="mapping-histogram-statistics-position")
            await user.should_see(marker="mapping-legend-position")
            mapping_statistics = cast(
                Checkbox,
                next(iter(user.find(marker="mapping-show-histogram-statistics").elements)),
            )
            assert mapping_statistics.value is True

            assert user.client is not None
            target = cast(Select, next(iter(user.find(marker="mapping-target").elements)))
            with user.client:
                target.set_value("replicated_fraction")
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            dependence = cast(
                Select,
                next(iter(user.find(marker="mapping-target-form").elements)),
            )
            assert dependence.value == "time"
            for marker in ("mapping-heatmap-palette", "mapping-colorbar", "mapping-log-z"):
                await user.should_see(marker=marker)
            await user.should_not_see(marker="mapping-histogram-bins")
            await user.should_not_see(marker="mapping-histogram-bin-size")
            await user.should_not_see(marker="mapping-histogram-x-minimum")
            await user.should_not_see(marker="mapping-histogram-x-maximum")
            await user.should_not_see(marker="mapping-line-width")
            with user.client:
                dependence.set_value("spacetime")
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            for marker in ("mapping-heatmap-palette", "mapping-colorbar", "mapping-log-z"):
                await user.should_see(marker=marker)
            await user.should_not_see(marker="mapping-histogram-bins")
            await user.should_not_see(marker="mapping-histogram-bin-size")

        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-plotting").click()
            await user.should_see("Explore DNAscape data")
            user.find("Plot settings").click()
            await user.should_see(marker="plotting-line-width")
            for marker in (
                "plotting-heatmap-palette",
                "plotting-colorbar",
                "plotting-log-z",
                "plotting-histogram-bins",
                "plotting-histogram-bin-size",
                "plotting-histogram-touching",
                "plotting-histogram-boundary",
                "plotting-histogram-x-minimum",
                "plotting-histogram-x-maximum",
                "plotting-reset-histogram-x-limits",
                "plotting-show-histogram-statistics",
                "plotting-histogram-statistics-position",
            ):
                await user.should_not_see(marker=marker)
            view = cast(Select, next(iter(user.find(marker="plotting-view").elements)))
            assert view.options == ["Default", "Circular", "Histogram", "Boxplot"]
            assert user.client is not None
            with user.client:
                view.set_value("Histogram")
            await asyncio.sleep(0)
            await user.should_see(marker="plotting-histogram-bins")
            await user.should_see(marker="plotting-histogram-bin-size")
            await user.should_see(marker="plotting-histogram-touching")
            await user.should_see(marker="plotting-histogram-boundary")
            await user.should_see(marker="plotting-histogram-x-minimum")
            await user.should_see(marker="plotting-histogram-x-maximum")
            await user.should_see(marker="plotting-reset-histogram-x-limits")
            await user.should_see(marker="plotting-show-histogram-statistics")
            await user.should_not_see(marker="plotting-histogram-statistics-position")
            await user.should_see(marker="plotting-legend-position")
            plotting_statistics = cast(
                Checkbox,
                next(iter(user.find(marker="plotting-show-histogram-statistics").elements)),
            )
            assert plotting_statistics.value is True
            for marker in (
                "plotting-y-minimum",
                "plotting-y-maximum",
                "plotting-reset-y-limits",
                "plotting-show-grid",
            ):
                await user.should_see(marker=marker)

        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-simulation").click()
            await user.should_see("Simulate DNA replication")
            user.find("Plot settings").click()
            for marker in (
                "simulation-y-minimum",
                "simulation-y-maximum",
                "simulation-reset-y-limits",
            ):
                await user.should_see(marker=marker)
            for marker in (
                "simulation-histogram-bins",
                "simulation-histogram-bin-size",
                "simulation-histogram-touching",
                "simulation-histogram-boundary",
                "simulation-histogram-x-minimum",
                "simulation-histogram-x-maximum",
                "simulation-reset-histogram-x-limits",
                "simulation-density",
                "simulation-show-histogram-statistics",
                "simulation-histogram-statistics-position",
                "simulation-heatmap-palette",
                "simulation-colorbar",
                "simulation-log-z",
            ):
                await user.should_not_see(marker=marker)
            await user.should_see(marker="simulation-line-or-bar-color")
            await user.should_see(marker="simulation-ensemble-profile-options")
            for marker in (
                "simulation-line-width",
                "simulation-show-grid",
            ):
                await user.should_see(marker=marker)
            await user.should_not_see(marker="simulation-show-markers")

            user.find(marker="simulation-view").click()
            user.find("Histogram").click()
            for marker in (
                "simulation-histogram-bins",
                "simulation-histogram-bin-size",
                "simulation-histogram-touching",
                "simulation-histogram-boundary",
                "simulation-histogram-x-minimum",
                "simulation-histogram-x-maximum",
                "simulation-reset-histogram-x-limits",
                "simulation-density",
                "simulation-show-histogram-statistics",
            ):
                await user.should_see(marker=marker)
            await user.should_not_see(marker="simulation-histogram-statistics-position")
            await user.should_see(marker="simulation-legend-position")
            simulation_statistics = cast(
                Checkbox,
                next(iter(user.find(marker="simulation-show-histogram-statistics").elements)),
            )
            assert simulation_statistics.value is True
            for marker in (
                "simulation-heatmap-palette",
                "simulation-colorbar",
                "simulation-log-z",
            ):
                await user.should_not_see(marker=marker)
            x_axis_label = cast(
                Input,
                next(iter(user.find(marker="simulation-x-axis-label").elements)),
            )
            y_axis_label = cast(
                Input,
                next(iter(user.find(marker="simulation-y-axis-label").elements)),
            )
            assert x_axis_label.value == "Replication timing"
            assert y_axis_label.value == "Count"
            user.find(marker="simulation-density").click()
            assert y_axis_label.value == "Density"

            view = cast(
                Select,
                next(iter(user.find(marker="simulation-view").elements)),
            )
            assert view.options == ["Default", "Circular", "Histogram"]
            assert "Time array" not in view.options

    asyncio.run(scenario())


def test_dependent_simulation_controls_follow_optional_toggles() -> None:
    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-simulation").click()
            await user.should_see("Simulate DNA replication")
            user.find("Simulation settings").click()
            await user.should_not_see(marker="simulation-seed")
            user.find("Fix seed").click()
            await user.should_see(marker="simulation-seed")

            await user.should_not_see(marker="simulation-stall-duration")
            user.find(marker="simulation-fork-stalling").click()
            user.find("Stall recovery").click()
            await user.should_see(marker="simulation-stall-duration")

            await user.should_not_see(marker="simulation-pool-fields")
            user.find(marker="simulation-finite-pool").click()
            await user.should_see(marker="simulation-pool-fields")

    asyncio.run(scenario())


def test_constant_simulation_reports_progress_and_renders_worker_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.pages import simulation as simulation_page
    from app.simulation_service import ConstantSimulationRequest

    started = threading.Event()
    release = threading.Event()
    requests: list[ConstantSimulationRequest] = []

    def fake_run(
        request: ConstantSimulationRequest,
        report,
        _cancel,
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
            outputs={
                "replication_timing": timing,
                "origin_efficiency": np.ones(3),
                "fork_directionality": np.zeros(3),
                "s_phase_duration": np.asarray([17.0] * request.replicates),
                "completion_time": np.asarray([18.0] * request.replicates),
                "num_origins": np.asarray([2.0] * request.replicates),
                "inter_origin_distances": np.asarray([1.0, 2.0]),
            },
            replication_time_profiles=np.tile(timing, (request.replicates, 1)),
            replication_timing_std=np.zeros(3),
        )

    monkeypatch.setattr(simulation_page, "run_constant_simulation", fake_run)

    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-simulation").click()
            await user.should_see("Simulate DNA replication")
            await user.should_not_see(marker="simulation-progress")
            await user.should_not_see(marker="simulation-result-plot")
            await user.should_not_see(marker="simulation-stats")
            await user.should_see(marker="simulation-output")
            output_before_run = cast(
                Select,
                next(iter(user.find(marker="simulation-output").elements)),
            )
            assert output_before_run.value == "Replication timing"

            user.find(marker="action-simulate").click()
            await user.should_see(marker="simulation-progress", retries=20)
            assert await asyncio.to_thread(started.wait, 2)
            await user.should_see("Running test ensemble...", retries=20)
            status = next(iter(user.find(marker="simulation-status").elements))
            assert status.text == "Running test ensemble..."
            assert requests
            request = requests[0]
            assert request.initiation_rate == pytest.approx(0.01)
            assert request.fork_speed == pytest.approx(1.4)
            assert request.region_start_kb == pytest.approx(0.0)
            assert request.region_end_kb == pytest.approx(1_000.0)
            assert request.replicates == 20
            assert request.geometry == "linear"
            assert request.track_detailed_outputs is False

            release.set()
            await user.should_see(marker="simulation-result-plot", retries=30)
            await user.should_see(marker="simulation-stats", retries=10)
            await user.should_see("Complete in 0.12 s", retries=10)
            await user.should_not_see(marker="simulation-error")
            await user.should_see(marker="simulation-output")
            output_after_run = cast(
                Select,
                next(iter(user.find(marker="simulation-output").elements)),
            )
            assert output_after_run is output_before_run
            assert output_after_run.value == "Replication timing"
            assert "Inter-origin distance" in output_after_run.options
            assert "Replicated fraction (time)" not in output_after_run.options
            assert "Replicated fraction (space-time)" not in output_after_run.options
            user.find("Simulation settings").click()
            user.find(marker="simulation-track-time-outputs").click()
            user.find(marker="simulation-track-detailed-outputs").click()
            assert "Replicated fraction (time)" not in output_after_run.options
            assert "Replicated fraction (space-time)" not in output_after_run.options

    try:
        asyncio.run(scenario())
    finally:
        release.set()


def test_simulate_button_becomes_stop_and_cancels_the_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.pages import simulation as simulation_page
    from app.simulation_service import ConstantSimulationRequest

    started = threading.Event()

    def cancellable_run(
        request: ConstantSimulationRequest,
        report,
        cancel,
    ) -> ConstantSimulationResult:
        del request
        report(0.35, "Simulating test data...")
        started.set()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            cancel()
            time.sleep(0.01)
        raise TimeoutError("simulation cancellation was not delivered")

    monkeypatch.setattr(simulation_page, "run_constant_simulation", cancellable_run)

    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-simulation").click()
            await user.should_see(marker="action-simulate", retries=60)
            button = cast(
                Button,
                next(iter(user.find(marker="action-simulate").elements)),
            )

            user.find(marker="action-simulate").click()
            assert await asyncio.to_thread(started.wait, 2)
            for _ in range(20):
                if button.text == "Stop":
                    break
                await asyncio.sleep(0.05)
            assert button.text == "Stop"
            assert button.icon == "stop"
            assert button.enabled

            user.find(marker="action-simulate").click()
            await user.should_see("Simulation stopped", retries=40)
            assert button.text == "Simulate"
            assert button.icon == workflow_action_icon("simulation")
            assert button.enabled
            await user.should_not_see(marker="simulation-error")

    asyncio.run(scenario())


def test_detailed_simulation_switch_exposes_xt_plot_and_live_downloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.pages import simulation as simulation_page
    from app.simulation_exports import ExportArtifact
    from app.simulation_exports import build_raw_export as real_build_raw_export
    from app.simulation_service import ConstantSimulationRequest, ConstantSimulationResult

    requests: list[ConstantSimulationRequest] = []
    chart_method_calls: list[tuple[EChart, str, dict[str, object], float]] = []
    capture_exports: list[tuple[str, str, dict[str, object]]] = []
    raw_exports: list[tuple[str, str, str, str | None, ExportArtifact]] = []
    space_time_fraction = np.array(
        [
            [0.0, 0.2, 0.45, 0.7, 0.88, 0.97, 1.0],
            [0.0, 0.15, 0.4, 0.68, 0.86, 0.96, 1.0],
            [0.0, 0.18, 0.43, 0.72, 0.9, 0.98, 1.0],
        ]
    )

    def fake_run(request: ConstantSimulationRequest, report, _cancel) -> ConstantSimulationResult:
        requests.append(request)
        report(0.65, "Collecting detailed outputs...")
        position = np.array([0.0, 1.0, 2.0])
        time = np.arange(0.0, 35.0, 5.0)
        timing = np.array([18.0, 14.0, 20.0])
        timing_profiles = np.array(
            [
                [17.0, 13.0, 19.0],
                [18.0, 14.0, 20.0],
                [19.0, 15.0, 21.0],
            ]
        )
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
                "replicated_fraction": np.array([0.0, 0.2, 0.45, 0.7, 0.88, 0.97, 1.0]),
                "s_phase_duration": np.array([16.0, 18.0]),
                "completion_time": np.array([17.0, 19.0]),
                "inter_origin_distances": np.array([42.0, 58.0, 75.0, 91.0]),
                "replicated_fraction_xt": space_time_fraction,
            },
            time_outputs=True,
            detailed_outputs=True,
            replication_time_profiles=timing_profiles,
            replication_timing_std=np.std(timing_profiles, axis=0),
        )

    async def fake_run_chart_method(
        chart: EChart,
        method: str,
        arguments: dict[str, object],
        *,
        timeout: float,
    ) -> str:
        chart_method_calls.append((chart, method, arguments, timeout))
        if str(arguments.get("type", "png")) == "svg":
            return "data:image/svg+xml;base64,PHN2Zy8+"
        return "data:image/png;base64,cHJldmlldy1jYXB0dXJl"

    def fake_capture_export(data_uri: str, file_format: str, **kwargs):
        capture_exports.append((data_uri, file_format, kwargs))
        return ExportArtifact(b"plot", f"plot.{file_format.lower()}", "image/test")

    def fake_raw_export(
        result,
        output_key: str,
        file_format: str,
        *,
        filename: str | None = None,
        compression: str = "none",
        **_kwargs,
    ):
        assert result.detailed_outputs
        artifact = real_build_raw_export(
            result,
            output_key,
            file_format,
            filename=filename,
            compression=compression,
            **_kwargs,
        )
        raw_exports.append((output_key, file_format, compression, filename, artifact))
        return artifact

    monkeypatch.setattr(simulation_page, "run_constant_simulation", fake_run)
    monkeypatch.setattr(EChart, "run_chart_method", fake_run_chart_method)
    monkeypatch.setattr(simulation_page, "build_chart_capture_export", fake_capture_export)
    monkeypatch.setattr(simulation_page, "build_raw_export", fake_raw_export)

    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-simulation").click()
            await user.should_see("Simulate DNA replication")
            await user.should_not_see(marker="simulation-stats")
            with user.scope(marker="simulation-preview-output-bar"):
                await user.should_see(marker="simulation-output")
            user.find("Simulation settings").click()
            user.find(marker="simulation-track-time-outputs").click()
            user.find(marker="simulation-track-detailed-outputs").click()
            user.find(marker="action-simulate").click()
            await user.should_see(marker="simulation-result-plot", retries=30)
            await user.should_see(marker="simulation-stats", retries=10)
            assert requests and requests[0].track_time_outputs is True
            assert requests[0].track_detailed_outputs is True
            available_outputs = cast(
                Select,
                next(iter(user.find(marker="simulation-output").elements)),
            ).options
            assert "Inter-origin distance" in available_outputs
            assert "Replicated fraction (time)" in available_outputs
            assert "Replicated fraction (space-time)" in available_outputs

            user.find("Plot settings").click()
            with user.scope("Plot settings"):
                await user.should_not_see(marker="simulation-output")

            y_minimum = cast(
                Number,
                next(iter(user.find(marker="simulation-y-minimum").elements)),
            )
            y_maximum = cast(
                Number,
                next(iter(user.find(marker="simulation-y-maximum").elements)),
            )
            assert y_minimum.value is not None
            assert y_maximum.value is not None
            automatic_limits = (float(y_minimum.value), float(y_maximum.value))
            assert 0.0 < automatic_limits[0] < 14.0
            assert automatic_limits[1] > 20.0
            assert automatic_limits == pytest.approx((13.76, 20.24))
            y_minimum.set_value(-100.0)
            y_maximum.set_value(100.0)
            user.find(marker="simulation-reset-y-limits").click()
            await asyncio.sleep(0)
            assert y_minimum.value is not None
            assert y_maximum.value is not None
            assert float(y_minimum.value) == pytest.approx(automatic_limits[0])
            assert float(y_maximum.value) == pytest.approx(automatic_limits[1])

            line_chart = cast(
                EChart,
                next(iter(user.find(marker="simulation-result-plot").elements)),
            )
            assert line_chart.options["series"][0]["type"] == "line"
            assert line_chart.options["yAxis"]["min"] == pytest.approx(automatic_limits[0])
            assert line_chart.options["yAxis"]["max"] == pytest.approx(automatic_limits[1])
            assert line_chart.options["xAxis"]["position"] == "bottom"
            assert line_chart.options["series"][0]["lineStyle"]["width"] == pytest.approx(1.8)
            assert line_chart.options["series"][0]["symbol"] == "circle"
            assert line_chart.options["series"][0]["showSymbol"] is False
            assert line_chart.options["series"][0]["showSymbol"] is False
            assert line_chart.options["xAxis"]["splitLine"]["show"] is False
            assert line_chart.options["legend"]["right"] == 28
            assert line_chart.options["grid"]["right"] == 24
            assert line_chart.options["grid"]["left"] >= line_chart.options["yAxis"][
                "nameGap"
            ]
            assert line_chart.options["grid"]["containLabel"] is False
            assert line_chart.options["xAxis"]["nameGap"] == 28
            assert 28 <= line_chart.options["yAxis"]["nameGap"] <= 82
            assert line_chart.options["title"]["show"] is False
            view = cast(
                Select,
                next(iter(user.find(marker="simulation-view").elements)),
            )
            assert view.options == ["Default", "Circular", "Histogram"]
            assert view.value == "Default"
            user.find(marker="action-download-plot").click()
            user.find(marker="download-plot-svg").click()
            for _ in range(30):
                if capture_exports:
                    break
                await asyncio.sleep(0.02)
            assert len(chart_method_calls) == 1
            captured_chart, method, arguments, timeout = chart_method_calls[0]
            assert captured_chart._props["renderer"] == "svg"
            assert captured_chart.options == line_chart.options
            assert method == "getDataURL"
            assert arguments == {"type": "svg"}
            assert timeout == 10
            assert capture_exports == [
                (
                    "data:image/svg+xml;base64,PHN2Zy8+",
                    "SVG",
                    {
                            "filename": "rsim_rt_plot",
                        "dpi": 150,
                    },
                )
            ]

            curve_thickness = cast(
                Number,
                next(iter(user.find(marker="simulation-line-width").elements)),
            )
            curve_thickness.set_value(3.2)
            user.find(marker="simulation-show-grid").click()
            await asyncio.sleep(0)
            styled_chart = cast(
                EChart,
                next(iter(user.find(marker="simulation-result-plot").elements)),
            )
            assert styled_chart.options["series"][0]["lineStyle"]["width"] == pytest.approx(3.2)
            assert styled_chart.options["series"][0]["symbol"] == "circle"
            assert styled_chart.options["series"][0]["showSymbol"] is False
            assert styled_chart.options["series"][0]["showSymbol"] is False
            assert styled_chart.options["xAxis"]["splitLine"]["show"] is True
            assert styled_chart.options["yAxis"]["splitLine"]["show"] is True

            await user.should_see(marker="simulation-ensemble-profile-options")
            user.find(marker="simulation-show-individual-profiles").click()
            await user.should_see(marker="simulation-individual-profile-count")
            profile_count = cast(
                Number,
                next(iter(user.find(marker="simulation-individual-profile-count").elements)),
            )
            assert profile_count.value == 3
            user.find(marker="simulation-show-standard-deviation").click()
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            ensemble_chart = cast(
                EChart,
                next(iter(user.find(marker="simulation-result-plot").elements)),
            )
            assert [series["type"] for series in ensemble_chart.options["series"]] == [
                "custom",
                "line",
                "line",
                "line",
                "line",
            ]
            assert ensemble_chart.options["legend"]["data"] == [
                "Individual simulations",
                "Mean",
                "Mean ± SD",
            ]
            profile_count.set_value(2)
            await asyncio.sleep(0)
            limited_chart = cast(
                EChart,
                next(iter(user.find(marker="simulation-result-plot").elements)),
            )
            assert len(limited_chart.options["series"]) == 4

            user.find(marker="simulation-view").click()
            user.find("Histogram").click()
            assert view.value == "Histogram"
            await user.should_not_see(marker="simulation-ensemble-profile-options")
            user.find(marker="simulation-output").click()
            user.find("Fork directionality").click()
            await user.should_see(marker="simulation-result-plot", retries=10)
            assert view.value == "Default"
            assert float(y_minimum.value) == pytest.approx(-1.0)
            assert float(y_maximum.value) == pytest.approx(1.0)
            direction_chart = cast(
                EChart,
                next(iter(user.find(marker="simulation-result-plot").elements)),
            )
            assert direction_chart.options["series"][0]["type"] == "line"
            assert direction_chart.options["yAxis"]["min"] == -1.0
            assert direction_chart.options["yAxis"]["max"] == 1.0

            user.find(marker="simulation-invert-y").click()
            await asyncio.sleep(0)
            inverted_chart = cast(
                EChart,
                next(iter(user.find(marker="simulation-result-plot").elements)),
            )
            assert inverted_chart.options["yAxis"]["inverse"] is True
            assert inverted_chart.options["xAxis"]["position"] == "bottom"

            y_minimum.set_value(-0.25)
            y_maximum.set_value(0.75)
            await asyncio.sleep(0)
            custom_chart = cast(
                EChart,
                next(iter(user.find(marker="simulation-result-plot").elements)),
            )
            assert custom_chart.options["yAxis"]["min"] == pytest.approx(-0.25)
            assert custom_chart.options["yAxis"]["max"] == pytest.approx(0.75)
            user.find(marker="simulation-reset-y-limits").click()
            await asyncio.sleep(0)
            assert float(y_minimum.value) == pytest.approx(-1.0)
            assert float(y_maximum.value) == pytest.approx(1.0)

            user.find(marker="simulation-output").click()
            user.find("S-phase duration").click()
            await user.should_see(marker="simulation-result-plot", retries=10)
            assert view.value == "Default"
            distribution_chart = cast(
                EChart,
                next(iter(user.find(marker="simulation-result-plot").elements)),
            )
            assert distribution_chart.options["series"][0]["type"] == "custom"
            assert distribution_chart.options["xAxis"]["position"] == "bottom"

            output_control = cast(
                Select,
                next(iter(user.find(marker="simulation-output").elements)),
            )
            assert "Inter-origin distance" in output_control.options
            user.find(marker="simulation-output").click()
            user.find("Inter-origin distance").click()
            await user.should_see(marker="simulation-result-plot", retries=10)
            assert view.value == "Default"
            inter_origin_histogram = cast(
                EChart,
                next(iter(user.find(marker="simulation-result-plot").elements)),
            )
            assert inter_origin_histogram.options["series"][0]["type"] == "custom"
            assert inter_origin_histogram.options["xAxis"]["name"] == "Distance (kb)"
            assert inter_origin_histogram.options["yAxis"]["name"] == "Count"
            assert len(requests) == 1
            simulation_client = user.client
            assert simulation_client is not None

            statistics = cast(
                Checkbox,
                next(
                    iter(
                        user.find(marker="simulation-show-histogram-statistics").elements
                    )
                ),
            )
            with simulation_client:
                statistics.set_value(True)
            await asyncio.sleep(0)
            statistics_position = cast(
                Select,
                next(
                    iter(
                        user.find(marker="simulation-legend-position").elements
                    )
                ),
            )
            with simulation_client:
                statistics_position.set_value("Top left")
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            statistics_chart = cast(
                EChart,
                next(iter(user.find(marker="simulation-result-plot").elements)),
            )
            summary = statistics_chart.options["graphic"][0]
            assert summary["left"] == statistics_chart.options["grid"]["left"] + 8
            assert summary["top"] == statistics_chart.options["grid"]["top"] + 8
            assert "Mean" in summary["style"]["text"]
            assert "Median" in summary["style"]["text"]
            assert "Std dev" in summary["style"]["text"]
            assert len(requests) == 1

            histogram_bins = cast(
                Number,
                next(iter(user.find(marker="simulation-histogram-bins").elements)),
            )
            histogram_bin_size = cast(
                Number,
                next(iter(user.find(marker="simulation-histogram-bin-size").elements)),
            )
            with simulation_client:
                histogram_bins.set_value(7)
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            assert histogram_bin_size.value == pytest.approx(7.0)
            assert len(
                cast(
                    EChart,
                    next(iter(user.find(marker="simulation-result-plot").elements)),
                ).options["series"][0]["data"]
            ) == 7

            with simulation_client:
                histogram_bin_size.set_value(10.0)
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            assert histogram_bins.value == 5
            assert histogram_bin_size.value == pytest.approx(10.0)

            histogram_x_minimum = cast(
                Number,
                next(
                    iter(user.find(marker="simulation-histogram-x-minimum").elements)
                ),
            )
            histogram_x_maximum = cast(
                Number,
                next(
                    iter(user.find(marker="simulation-histogram-x-maximum").elements)
                ),
            )
            assert histogram_x_minimum.value is not None
            assert histogram_x_maximum.value is not None
            original_x_minimum = float(histogram_x_minimum.value)
            original_x_maximum = float(histogram_x_maximum.value)
            custom_x_maximum = original_x_minimum + (
                original_x_maximum - original_x_minimum
            ) * 0.55
            with simulation_client:
                histogram_x_maximum.set_value(custom_x_maximum)
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            assert histogram_bins.value == int(
                np.ceil((custom_x_maximum - original_x_minimum) / 10.0 - 1e-12)
            )
            limited_histogram = cast(
                EChart,
                next(iter(user.find(marker="simulation-result-plot").elements)),
            )
            assert limited_histogram.options["xAxis"]["min"] == pytest.approx(
                original_x_minimum
            )
            assert limited_histogram.options["xAxis"]["max"] == pytest.approx(
                custom_x_maximum
            )
            user.find(marker="simulation-reset-histogram-x-limits").click()
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            assert histogram_x_minimum.value is not None
            assert histogram_x_maximum.value is not None
            assert float(histogram_x_minimum.value) == pytest.approx(original_x_minimum)
            assert float(histogram_x_maximum.value) == pytest.approx(original_x_maximum)
            assert histogram_bins.value == 5
            preserved_histogram = cast(
                EChart,
                next(iter(user.find(marker="simulation-result-plot").elements)),
            )
            with simulation_client:
                histogram_bin_size.set_value(None)
            await asyncio.sleep(0)
            assert len(preserved_histogram.options["series"][0]["data"]) == 5
            await user.should_not_see("Output unavailable")
            with simulation_client:
                histogram_bin_size.set_value(10.0)
            await asyncio.sleep(0)

            user.find(marker="simulation-view").click()
            user.find("Boxplot").click()
            await asyncio.sleep(0)
            inter_origin_boxplot = cast(
                EChart,
                next(iter(user.find(marker="simulation-result-plot").elements)),
            )
            assert view.value == "Boxplot"
            assert inter_origin_boxplot.options["series"][0]["type"] == "boxplot"

            user.find(marker="simulation-output").click()
            user.find("Replicated fraction (space-time)").click()
            await user.should_see(marker="simulation-result-plot", retries=10)
            assert len(requests) == 1  # switching Preview output is a local redraw
            assert view.value == "Default"
            chart = cast(
                EChart,
                next(iter(user.find(marker="simulation-result-plot").elements)),
            )
            assert chart.options["series"][0]["type"] == "custom"
            assert chart.options["xAxis"]["name"] == "Position (kb)"
            assert chart.options["yAxis"]["name"] == "Time (min)"
            assert float(y_minimum.value) == pytest.approx(0.0)
            assert float(y_maximum.value) == pytest.approx(20.0)
            assert chart.options["yAxis"]["min"] == pytest.approx(0.0)
            assert chart.options["yAxis"]["max"] == pytest.approx(20.0)
            assert chart.options["series"][0]["silent"] is True
            assert chart.options["tooltip"]["show"] is True
            assert chart.options["xAxis"]["position"] == "bottom"

            for axis, limits in (("x", (-1.0, 3.0)), ("y", (-10.0, 40.0)), ("z", (0.2, 0.8))):
                lower = cast(Number, next(iter(user.find(marker=f"simulation-{axis}-minimum").elements)))
                upper = cast(Number, next(iter(user.find(marker=f"simulation-{axis}-maximum").elements)))
                reset = next(iter(user.find(marker=f"simulation-reset-{axis}-limits").elements))
                assert lower.enabled and upper.enabled and reset.enabled
                key = "visualMap" if axis == "z" else f"{axis}Axis"
                original = (chart.options[key]["min"], chart.options[key]["max"])
                with simulation_client:
                    lower.set_value(limits[0])
                    upper.set_value(limits[1])
                await asyncio.sleep(0)
                current = cast(EChart, next(iter(user.find(marker="simulation-result-plot").elements)))
                assert (current.options[key]["min"], current.options[key]["max"]) == limits
                with simulation_client:
                    upper.set_value(None)
                user.find(marker=f"simulation-reset-{axis}-limits").click()
                await asyncio.sleep(0)
                current = cast(EChart, next(iter(user.find(marker="simulation-result-plot").elements)))
                assert (current.options[key]["min"], current.options[key]["max"]) == original
            assert len(requests) == 1

            # Axis edits rebuild the chart; exports must match the current one.
            chart = cast(EChart, next(iter(user.find(marker="simulation-result-plot").elements)))
            user.find(marker="action-download-plot").click()
            user.find(marker="download-plot-svg").click()
            for _ in range(30):
                if len(capture_exports) >= 2:
                    break
                await asyncio.sleep(0.02)
            assert len(chart_method_calls) == 2
            captured_chart, method, arguments, timeout = chart_method_calls[1]
            assert captured_chart._props["renderer"] == "svg"
            assert captured_chart.options == chart.options
            assert method == "getDataURL"
            assert arguments == {"type": "svg"}
            assert timeout == 10
            assert capture_exports[1] == (
                "data:image/svg+xml;base64,PHN2Zy8+",
                "SVG",
                {
                        "filename": "rsim_rf-xt_plot",
                    "dpi": 150,
                },
            )

            user.find("Raw data").click()
            user.find(marker="action-download-raw-data").click()
            for _ in range(30):
                if raw_exports:
                    break
                await asyncio.sleep(0.02)
            assert len(raw_exports) == 1
            output_key, file_format, compression, requested_name, artifact = raw_exports[0]
            assert output_key == "replicated_fraction_xt"
            assert file_format == "TSV"
            assert compression == "none"
            assert requested_name == "rsim_rf-xt"
            assert artifact.filename == "rsim_rf-xt.tsv"
            assert artifact.data
            values = np.loadtxt(io.BytesIO(artifact.data), delimiter="\t", skiprows=1)
            np.testing.assert_array_equal(
                values,
                space_time_fraction,
            )

            user.find(marker="raw-output-compression").click()
            user.find(marker="action-download-raw-data").click()
            for _ in range(30):
                if len(raw_exports) >= 2:
                    break
                await asyncio.sleep(0.02)
            assert raw_exports[1][2] == "gzip"
            assert raw_exports[1][4].filename.endswith(".tsv.gz")

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


def test_uploaded_initiation_rate_and_fork_speed_reach_simulation_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.pages import simulation as simulation_page
    from app.simulation_service import ConstantSimulationRequest, ConstantSimulationResult

    requests: list[ConstantSimulationRequest] = []

    def fake_run(request: ConstantSimulationRequest, report, _cancel) -> ConstantSimulationResult:
        requests.append(request)
        report(0.7, "Running uploaded fields...")
        position = np.arange(3, dtype=float)
        timing = np.asarray([8.0, 9.0, 10.0])
        return ConstantSimulationResult(
            position_kb=position,
            replication_timing=timing,
            elapsed_seconds=0.04,
            replicates=request.replicates,
            geometry=request.geometry,
            time_grid=np.asarray([], dtype=float),
            outputs={"replication_timing": timing},
        )

    monkeypatch.setattr(simulation_page, "run_constant_simulation", fake_run)

    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-simulation").click()
            await user.should_see("Simulate DNA replication", retries=60)
            firing_mode = cast(
                Select,
                next(iter(user.find(marker="source-initiation-rate-mode").elements)),
            )
            speed_mode = cast(
                Select,
                next(iter(user.find(marker="source-fork-speed-mode").elements)),
            )
            firing_mode.set_value("profile")
            speed_mode.set_value("profile")
            await user.should_see(marker="source-initiation-rate-profile")
            await user.should_see(marker="source-fork-speed-profile")

            firing_upload = cast(
                Upload,
                next(iter(user.find(marker="source-initiation-rate-upload").elements)),
            )
            speed_upload = cast(
                Upload,
                next(iter(user.find(marker="source-fork-speed-upload").elements)),
            )
            await firing_upload.handle_uploads(
                [
                    Upload.SmallFileUpload(
                        "firing.csv",
                        "text/csv",
                        b"position,initiation_rate\n0,0.1\n1,0.2\n2,0.3\n",
                    )
                ]
            )
            await speed_upload.handle_uploads(
                [
                    Upload.SmallFileUpload(
                        "speed.csv",
                        "text/csv",
                        b"t0,t1,t2\n1.1,1.2,1.3\n",
                    )
                ]
            )
            await user.should_see(marker="source-initiation-rate-detected-data", retries=30)
            await user.should_see(marker="source-fork-speed-detected-data")
            await user.should_see(marker="source-initiation-rate-upload-file")
            await user.should_see(marker="source-fork-speed-upload-file")
            firing_form = _select_by_marker(user, "source-initiation-rate-form")
            speed_form = _select_by_marker(user, "source-fork-speed-form")
            assert firing_form.value == "x"
            assert speed_form.value == "t"
            await user.should_not_see("Accepted:")
            # Profile uploaders remain available so another file can extend
            # the batch or a same-name upload can replace retained bytes.
            assert firing_upload.visible
            assert speed_upload.visible
            region_start = cast(
                Number,
                next(iter(user.find(marker="simulation-region-start").elements)),
            )
            region_end = cast(
                Number,
                next(iter(user.find(marker="simulation-region-end").elements)),
            )
            assert region_start.value == pytest.approx(0.0)
            assert region_end.value == pytest.approx(3.0)

            user.find(marker="action-simulate").click()
            await user.should_see(marker="simulation-result-plot", retries=30)
            await user.should_not_see(marker="simulation-error")
            assert requests
            assert requests[0].region_end_kb == pytest.approx(3.0)
            np.testing.assert_array_equal(requests[0].initiation_rate, [0.1, 0.2, 0.3])
            np.testing.assert_array_equal(requests[0].fork_speed, [[1.1, 1.2, 1.3]])
            assert np.asarray(requests[0].initiation_rate).shape == (3,)
            assert np.asarray(requests[0].fork_speed).shape == (1, 3)

            output = _select_by_marker(user, "simulation-output")
            assert "Initiation rate (input)" in output.options
            assert "Fork speed (input)" in output.options
            for label, expected in (
                ("Initiation rate (input)", [0.1, 0.2, 0.3]),
                ("Fork speed (input)", [1.1, 1.2, 1.3]),
            ):
                with user.client:
                    output.set_value(label)
                await asyncio.sleep(0)
                chart = next(iter(user.find(marker="simulation-result-plot").elements))
                np.testing.assert_allclose(
                    [point[1] for point in chart.options["series"][0]["data"]], expected,
                )

    asyncio.run(scenario())


def test_simulation_batch_runs_all_profiles_and_keeps_atomic_preview(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.pages import simulation as simulation_page
    from app.simulation_service import ConstantSimulationRequest

    requests: list[ConstantSimulationRequest] = []
    archived_counts: list[int] = []
    real_archive = simulation_page.build_artifact_archive

    def fake_run(
        request: ConstantSimulationRequest,
        report: Any,
        _cancel: Any,
    ) -> ConstantSimulationResult:
        requests.append(request)
        report(0.6, "Simulating batch dataset...")
        timing = np.asarray(request.initiation_rate, dtype=float).reshape(-1) * 10.0
        position = np.arange(timing.size, dtype=float)
        return ConstantSimulationResult(
            position_kb=position,
            replication_timing=timing,
            elapsed_seconds=0.04,
            replicates=request.replicates,
            geometry=request.geometry,
            time_grid=np.asarray([], dtype=float),
            outputs={"replication_timing": timing},
        )

    def track_archive(artifacts: Any, **kwargs: Any) -> Any:
        retained = tuple(artifacts)
        archived_counts.append(len(retained))
        return real_archive(retained, **kwargs)

    monkeypatch.setattr(simulation_page, "run_constant_simulation", fake_run)
    monkeypatch.setattr(simulation_page, "build_artifact_archive", track_archive)

    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-simulation").click()
            await user.should_see("Simulate DNA replication", retries=60)
            firing_mode = cast(
                Select,
                next(iter(user.find(marker="source-initiation-rate-mode").elements)),
            )
            assert user.client is not None
            with user.client:
                firing_mode.set_value("profile")
            uploader = cast(
                Upload,
                next(iter(user.find(marker="source-initiation-rate-upload").elements)),
            )
            assert uploader._props["multiple"] is True
            await uploader.handle_uploads(
                [
                    Upload.SmallFileUpload(
                        "sample_a.csv",
                        "text/csv",
                        b"initiation_rate\n0.1\n0.2\n0.3\n",
                    ),
                    Upload.SmallFileUpload(
                        "sample_b.csv",
                        "text/csv",
                        b"initiation_rate\n0.4\n0.5\n0.6\n",
                    ),
                ]
            )
            await user.should_see("Files: 2", retries=40)

            user.find(marker="action-simulate").click()
            await user.should_see("2 datasets complete", retries=60)
            assert len(requests) == 2
            np.testing.assert_array_equal(requests[0].initiation_rate, [0.1, 0.2, 0.3])
            np.testing.assert_array_equal(requests[1].initiation_rate, [0.4, 0.5, 0.6])
            assert requests[0].fork_speed == pytest.approx(1.4)
            assert requests[1].fork_speed == pytest.approx(1.4)

            layout = cast(
                Select,
                next(iter(user.find(marker="simulation-presentation").elements)),
            )
            assert layout.value == "Combined"
            assert layout.options == ["Combined", "Individual"]
            chart = cast(
                EChart,
                next(iter(user.find(marker="simulation-result-plot").elements)),
            )
            series_names = [series["name"] for series in chart.options["series"]]
            assert series_names == [
                "Simulation · sample_a",
                "Simulation · sample_b",
            ]
            # Status and chart updates can reach the simulated client one
            # outbox tick before the refreshable curve controls.
            await user.should_see(marker="simulation-dataset-color-1", retries=30)
            await user.should_see(marker="simulation-dataset-color-2", retries=30)
            name = next(iter(user.find(marker="simulation-dataset-name-1").elements))
            with user.client:
                name.set_value("First simulation")
            await asyncio.sleep(0)
            renamed = next(iter(user.find(marker="simulation-result-plot").elements))
            assert renamed.options["series"][0]["name"] == "First simulation"
            with user.client:
                name.set_value("")
            await asyncio.sleep(0)
            await user.should_see(marker="simulation-batch-raw-surface")
            user.find("Raw data").click()
            await user.should_see(marker="simulation-batch-raw-1")
            await user.should_see(marker="simulation-batch-raw-2")
            user.find(marker="action-download-all-simulation-raw").click()
            for _ in range(40):
                if archived_counts:
                    break
                await asyncio.sleep(0.02)
            assert archived_counts == [2]

            with user.client:
                layout.set_value("Individual")
            await user.should_see(marker="simulation-dataset-select", retries=20)
            dataset_select = _select_by_marker(user, "simulation-dataset-select")
            assert list(dataset_select.options.values()) == [
                "Simulation · sample_a",
                "Simulation · sample_b",
            ]
            assert all(
                str(key).startswith("simulation-case-")
                for key in dataset_select.options
            )
            assert dataset_select.value == next(iter(dataset_select.options))

            # A same-name replacement remains in its original batch slot.
            await uploader.handle_uploads(
                [
                    Upload.SmallFileUpload(
                        "sample_a.csv",
                        "text/csv",
                        b"initiation_rate\n0.7\n0.8\n0.9\n",
                    )
                ]
            )
            await asyncio.sleep(0.25)
            user.find(marker="action-simulate").click()
            for _ in range(80):
                if len(requests) >= 4:
                    break
                await asyncio.sleep(0.02)
            assert len(requests) == 4
            np.testing.assert_array_equal(requests[2].initiation_rate, [0.7, 0.8, 0.9])
            np.testing.assert_array_equal(requests[3].initiation_rate, [0.4, 0.5, 0.6])

            # Replacing one same-name upload must retain the other dataset. The
            # ambiguous, differently named signals are rejected before any
            # further worker starts, and the last complete preview remains.
            await uploader.handle_uploads(
                [
                    Upload.SmallFileUpload(
                        "sample_b.csv",
                        "text/csv",
                        b"other_a,other_b\n0.4,1.4\n0.5,1.5\n0.6,1.6\n",
                    )
                ]
            )
            await asyncio.sleep(0.25)
            await user.should_see("Files: 2", retries=20)
            user.find(marker="action-simulate").click()
            await user.should_see("Check the settings", retries=40)
            assert len(requests) == 4
            await user.should_see(marker="simulation-result-plot")
            await user.should_see(marker="simulation-batch-raw-surface")

    asyncio.run(scenario())


def test_simulation_profile_headers_select_arrays_and_resize_region() -> None:
    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-simulation").click()
            await user.should_see("Simulate DNA replication")
            mode = cast(
                Select,
                next(iter(user.find(marker="source-initiation-rate-mode").elements)),
            )
            mode.set_value("profile")
            uploader = cast(
                Upload,
                next(iter(user.find(marker="source-initiation-rate-upload").elements)),
            )
            await uploader.handle_uploads(
                [
                    Upload.SmallFileUpload(
                        "chromosomes.csv",
                        "text/csv",
                        b"chr1,chr2\n0.1,0.3\n0.2,0.4\n0.3,\n",
                    )
                ]
            )
            await user.should_see(marker="source-initiation-rate-detected-data", retries=30)
            column = cast(
                Select,
                next(iter(user.find(marker="source-initiation-rate-value-column").elements)),
            )
            region_end = cast(
                Number,
                next(iter(user.find(marker="simulation-region-end").elements)),
            )
            assert column.options == ["chr1", "chr2"]
            assert column.value == "chr1"
            assert region_end.value == pytest.approx(3.0)

            column.set_value("chr2")
            for _ in range(40):
                if region_end.value == 2.0:
                    break
                await asyncio.sleep(0.1)
            assert region_end.value == pytest.approx(2.0)

            user.find(marker="source-initiation-rate-upload-remove").click()
            await user.should_not_see(
                marker="source-initiation-rate-upload-file",
                retries=20,
            )
            await uploader.handle_uploads(
                [
                    Upload.SmallFileUpload(
                        "named_profiles.csv",
                        "text/csv",
                        b"sample_a,sample_b\n0.1,0.3\n0.2,0.4\n0.3,\n",
                    )
                ]
            )
            await user.should_see(marker="source-initiation-rate-detected-data", retries=30)
            column = cast(
                Select,
                next(iter(user.find(marker="source-initiation-rate-value-column").elements)),
            )
            assert column.options == {
                "__all_columns__": "All columns (space-time)",
                "sample_a": "sample_a",
                "sample_b": "sample_b",
            }
            column.set_value("sample_b")
            for _ in range(40):
                if region_end.value == 3.0:
                    break
                await asyncio.sleep(0.1)
            # Ordinary profile tables preserve missing bins at their genomic
            # positions; only padding in chromosome tables is trimmed.
            assert region_end.value == pytest.approx(3.0)

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
            await user.should_not_see(marker="simulation-error")
            await user.should_not_see(marker="simulation-progress")
            await user.should_not_see(marker="simulation-result-plot")
            simulate_button = cast(
                Button,
                next(iter(user.find(marker="action-simulate").elements)),
            )
            assert simulate_button.enabled

    asyncio.run(scenario())
