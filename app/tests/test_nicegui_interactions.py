"""Fast simulated-user coverage for the presentation-only NiceGUI prototype."""

from __future__ import annotations

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
from nicegui.elements.drawer import LeftDrawer
from nicegui.elements.echart import EChart
from nicegui.elements.expansion import Expansion
from nicegui.elements.html import Html
from nicegui.elements.icon import Icon
from nicegui.elements.input import Input
from nicegui.elements.label import Label
from nicegui.elements.number import Number
from nicegui.elements.select import Select
from nicegui.elements.upload import Upload
from nicegui.testing import user_simulation

from app.icons import DATA_ICON_SVG
from app.main import index
from app.mapping_catalog import mapping_route_families
from app.mapping_service import MappingResult
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
                ("mapping", "Map replication kinetic landscapes", "map"),
                # Simulation is now live and is exercised through an injected worker below.
                ("simulation", "Simulate DNA replication", None),
                ("plotting", "Visualize DNAscape outputs", "plot"),
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
            await user.should_see("timing.npy.gz")
            await user.should_see("timing.png")
            await user.should_see(marker="cache-search")
            await user.should_see(marker="cache-kind-filter")
            await user.should_see(marker="cache-sort")
            await user.should_see("Simulation result")
            await user.should_see("profile.csv")

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
            user.find(marker="clear-generated-cache").click()
            await user.should_see("Clear generated files?")
            user.find(marker="confirm-clear-generated-cache").click()
            await user.should_not_see(marker=f"cache-entry-{raw.id}")
            assert cache.get(raw.id) is None

            user.find("Uploads").click()
            await user.should_see("profile.csv")
            assert cache.get(upload.id) is not None

            user.find("Workflow results").click()
            await user.should_see("Simulation result")
            user.find(marker=f"cache-open-{workflow.id}").click()
            await user.should_see("Simulate DNA replication")
            await user.should_see("Restored from Cached Files")
            await user.should_see(marker="simulation-result-plot")

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
            firing_rate = cast(
                Number,
                next(iter(user.find(marker="source-firing-rate-constant").elements)),
            )
            with user.client:
                region_end.set_value(20.0)
                replicates.set_value(1)
                firing_rate.set_value(0.1)

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
            result = MappingResult(
                target="firing_rate",
                target_key="firing_rate",
                target_label="Firing rate",
                method="cached_test",
                output_form="x",
                output_dependence="space",
                primary_output="firing_rate",
                outputs={"firing_rate": np.asarray([0.1, 0.3, 0.2])},
                output_forms={"firing_rate": "x"},
                x_axis=np.asarray([10.0, 11.0, 12.0]),
                time_axis=np.asarray([], dtype=float),
                region_labels=(),
                elapsed_seconds=0.1,
                assumptions={},
            )
            with user.client:
                cache = current_session_cache()
                workflow = cache.add_reference(
                    name="Mapping: Firing rate",
                    source="Mapping",
                    detail="Space | 3 points | Cached Test",
                    payload=result,
                    navigate_to="mapping",
                )

            user.find(marker="nav-cached-files").click()
            await user.should_see("Session files and results")
            user.find("Workflow results").click()
            await user.should_see("Mapping: Firing rate")
            user.find(marker=f"cache-open-{workflow.id}").click()
            await user.should_see("Map replication kinetic landscapes")
            await user.should_see("Restored from Cached Files")
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
                ("plotting", "Visualize DNAscape outputs", 1),
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
                expansion_by_label = {
                    expansion._props.get("label"): expansion for expansion in expansions
                }
                plotting_settings = expansion_by_label["Plotting settings"]
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
            await user.should_see("Visualize DNAscape outputs")
            expansions = user.find(kind=Expansion).elements
            expansion_by_label = {
                expansion._props.get("label"): expansion for expansion in expansions
            }
            assert set(expansion_by_label) == {"Data", "Plotting settings"}
            plotting_settings = expansion_by_label["Plotting settings"]
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
            await user.should_see("Visualize DNAscape outputs")
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
            await user.should_see("Choose a file or drag it here")
            await user.should_not_see(
                "Accepted: CSV, TSV, TXT, NPY, NPV, NPZ, bedGraph, BigWig, GZ"
            )
            await user.should_not_see(marker="mapping-input-replication-timing-shape-guidance")

            user.find(marker="nav-plotting").click()
            await user.should_see("Visualize DNAscape outputs")
            await user.should_see(marker="plotting-data-upload")
            await user.should_not_see("Accepted: CSV, TSV, TXT, NPY, NPV, NPZ, GZ")

    asyncio.run(scenario())


def test_data_manager_converts_between_forms_and_preserves_the_last_result() -> None:
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
            await user.should_not_see(marker="data-manager-upload")
            output_form = cast(
                Select,
                next(iter(user.find(marker="data-manager-output-form").elements)),
            )
            assert output_form.options == {
                "space": "Space",
                "time": "Time",
                "table": "Table",
            }
            gzip_output = cast(
                Checkbox,
                next(iter(user.find(marker="data-manager-gzip").elements)),
            )
            assert gzip_output.value is True
            missing_values = cast(
                Select,
                next(iter(user.find(marker="data-manager-missing-values").elements)),
            )
            assert missing_values.options == {
                "error": "Error",
                "ignore": "Ignore",
                "interpolate": "Interpolate",
                "zero": "Fill with zero",
            }
            assert user.client is not None
            with user.client:
                output_form.set_value("time")

            await user.should_not_see(marker="data-manager-result-plot")
            await user.should_see(marker="data-manager-empty-result")
            user.find(marker="action-convert").click()
            await user.should_see(marker="action-download-converted-data", retries=60)
            await user.should_not_see(marker="data-manager-result-plot")
            await user.should_not_see(marker="data-manager-empty-result")
            await user.should_see("1 x 3")

            user.find(marker="data-manager-upload-remove").click()
            await user.should_see(marker="data-manager-upload")
            await user.should_not_see(marker="data-manager-result-plot")
            await user.should_see(marker="action-download-converted-data")

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
            output_form = cast(
                Select,
                next(iter(user.find(marker="data-manager-output-form").elements)),
            )
            assert output_form.value == "time"

    asyncio.run(scenario())


def test_plotting_commits_single_and_multiple_profiles_only_when_plot_is_clicked() -> None:
    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-plotting").click()
            await user.should_see("Visualize DNAscape outputs")
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
            await user.should_see("Dependence: Space")
            await user.should_see("Array size: 3")
            await user.should_not_see(marker="plotting-data-upload")
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
            assert chart().options["xAxis"]["name"] == "Position (bin)"

            user.find("Plotting settings").click()
            assert user.client is not None
            title = cast(Input, next(iter(user.find(marker="plotting-title").elements)))
            thickness = cast(
                Number,
                next(iter(user.find(marker="plotting-line-width").elements)),
            )
            grid = cast(
                Checkbox,
                next(iter(user.find(marker="plotting-show-grid").elements)),
            )
            with user.client:
                title.set_value("Edited profile")
                thickness.set_value(3.2)
                grid.set_value(False)
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            assert chart().options["title"]["text"] == "signal"
            assert chart().options["series"][0]["lineStyle"]["width"] == pytest.approx(1.8)
            assert chart().options["xAxis"]["splitLine"]["show"] is True

            user.find(marker="action-plot").click()
            await wait_for_chart(lambda options: options["title"]["text"] == "Edited profile")
            assert chart().options["title"]["text"] == "Edited profile"
            assert chart().options["series"][0]["lineStyle"]["width"] == pytest.approx(3.2)
            assert chart().options["xAxis"]["splitLine"]["show"] is False

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

            user.find(marker="action-plot").click()
            await wait_for_chart(lambda options: options["series"][0]["type"] == "custom")
            assert chart().options["series"][0]["type"] == "custom"
            assert chart().options["series"][0]["silent"] is True
            assert chart().options["tooltip"]["show"] is False
            await user.should_see(marker="plotting-heatmap-palette")
            await user.should_see(marker="plotting-colorbar")
            await user.should_see(marker="plotting-log-z")
            await user.should_not_see(marker="plotting-line-width")
            view = cast(Select, next(iter(user.find(marker="plotting-view").elements)))
            assert view.options == ["Default"]

            profiles = cast(
                Select,
                next(iter(user.find(marker="plotting-profile-selection-1").elements)),
            )
            with user.client:
                profiles.set_value(["t0", "t1"])
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            await user.should_see("Dependence: Space")
            assert chart().options["series"][0]["type"] == "custom"
            await user.should_see(marker="plotting-secondary-color")

            user.find(marker="action-plot").click()
            await wait_for_chart(lambda options: len(options["series"]) == 2)
            assert [series["name"] for series in chart().options["series"]] == ["t0", "t1"]
            assert [series["type"] for series in chart().options["series"]] == ["line", "line"]
            assert chart().options["series"][0]["lineStyle"]["color"] == "#2c7f77"
            assert chart().options["series"][1]["lineStyle"]["color"] == "#b03412"

            user.find(marker="action-add-plot-profile").click()
            await user.should_see(marker="plotting-data-upload-2")
            second_uploader = cast(
                Upload,
                next(iter(user.find(marker="plotting-data-upload-2").elements)),
            )
            await second_uploader.handle_uploads(
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

            user.find(marker="action-plot").click()
            await wait_for_chart(lambda options: len(options["series"]) == 3)
            assert [series["name"] for series in chart().options["series"]] == [
                "t0",
                "t1",
                "other",
            ]

            user.find(marker="plotting-remove-profile-2").click()
            await user.should_not_see("other.csv")
            assert len(chart().options["series"]) == 3

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
                await user.should_see("Download")

            user.find(marker="nav-plotting").click()
            await user.should_see(marker="preview-guidance")
            await user.should_see("No plot yet")
            assert all(
                expansion._props.get("label") != "Raw data"
                for expansion in user.find(kind=Expansion).elements
            )
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
                await user.should_see(marker="raw-output-row")

                row = next(iter(user.find(marker="raw-output-row").elements))
                download = cast(
                    Button,
                    next(iter(user.find(marker="action-download-raw-data").elements)),
                )
                children = row.default_slot.children

                assert [
                    next(
                        class_name
                        for class_name in (
                            "dn-raw-output-name",
                            "dn-raw-output-type",
                            "dn-raw-output-compress",
                            "dn-raw-output-download",
                        )
                        if class_name in child._classes
                    )
                    for child in children
                ] == [
                    "dn-raw-output-name",
                    "dn-raw-output-type",
                    "dn-raw-output-compress",
                    "dn-raw-output-download",
                ]
                assert children[-1] is download
                assert download.parent_slot is not None
                assert download.parent_slot.parent is row
                assert download.text == "Download"
                file_type = cast(
                    Select,
                    next(child for child in children if "dn-raw-output-type" in child._classes),
                )
                assert file_type.options == ["NPY", "NPV", "NPZ", "TXT", "TSV", "CSV"]
                assert file_type.value == "NPY"
                compression = cast(
                    Checkbox,
                    next(iter(user.find(marker="raw-output-compression").elements)),
                )
                assert compression.value is True

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
            controls_title = next(iter(user.find("Controls").elements))

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
            assert stat_labels == ["Replicates", "Resolution", "Region", "Geometry"]
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
            assert target.value == "firing_rate"
            assert dependence.value == "space"
            assert route.value == mapping_route_families("firing_rate", "space")[0].key
            assert timing_type.options == {"profile": "Profile"}
            assert timing_type.value == "profile"
            assert timing_type._props["disable"] is True
            assert timing_form.options == {"x": "Space"}
            assert timing_form.value == "x"
            assert not timing_form.visible
            assert speed_type.options == {
                "scalar": "Constant",
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
            await user.should_see(kind=Select, marker="source-firing-rate-mode")
            simulation_mode = cast(
                Select,
                next(iter(user.find(marker="source-firing-rate-mode").elements)),
            )
            simulation_form = _select_by_marker(user, "source-firing-rate-form")
            assert simulation_mode.options == {
                "scalar": "Constant",
                "profile": "Profile",
            }
            assert simulation_mode.value == "scalar"
            assert simulation_form.options == {
                "x": "Space",
                "t": "Time",
                "xt": "Space-time",
            }
            assert not simulation_form.visible
            await user.should_not_see(marker="source-firing-rate-profile")
            await user.should_see(marker="source-firing-rate-constant")
            user.find(marker="source-firing-rate-mode").click()
            user.find("Profile").click()
            await user.should_see(marker="source-firing-rate-profile")
            await user.should_not_see(marker="source-firing-rate-constant")
            user.find(marker="source-firing-rate-mode").click()
            user.find("Constant").click()
            await user.should_see(marker="source-firing-rate-constant")
            for section in ("Region", "Simulation settings"):
                user.find(section).click()
            await user.should_see(marker="scientific-action-dock")
            with user.scope(marker="scientific-action-dock"):
                await user.should_see(marker="action-simulate")

        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-plotting").click()
            await user.should_see(marker="plotting-profile-source-1")
            await user.should_see(marker="plotting-data-upload")
            await user.should_see(marker="plotting-settings")
            await user.should_not_see("Region")
            assert all(
                expansion._props.get("label") != "Raw data"
                for expansion in user.find(kind=Expansion).elements
            )
            await user.should_see(marker="scientific-action-dock")
            await user.should_see(marker="action-plot")

    asyncio.run(scenario())


def test_plot_controls_only_show_options_relevant_to_the_selected_view() -> None:
    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-mapping").click()
            await user.should_see("Map replication kinetic landscapes")
            user.find("Plotting settings").click()
            for marker in (
                "mapping-heatmap-palette",
                "mapping-colorbar",
                "mapping-log-z",
                "mapping-histogram-bins",
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
            await user.should_not_see(marker="mapping-line-width")
            with user.client:
                dependence.set_value("spacetime")
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            for marker in ("mapping-heatmap-palette", "mapping-colorbar", "mapping-log-z"):
                await user.should_see(marker=marker)
            await user.should_not_see(marker="mapping-histogram-bins")

        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-plotting").click()
            await user.should_see("Visualize DNAscape outputs")
            user.find("Plotting settings").click()
            await user.should_see(marker="plotting-line-width")
            for marker in (
                "plotting-heatmap-palette",
                "plotting-colorbar",
                "plotting-log-z",
                "plotting-histogram-bins",
            ):
                await user.should_not_see(marker=marker)
            view = cast(Select, next(iter(user.find(marker="plotting-view").elements)))
            assert view.options == ["Default", "Histogram", "Boxplot"]
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
            user.find("Plotting settings").click()
            for marker in (
                "simulation-y-minimum",
                "simulation-y-maximum",
                "simulation-reset-y-limits",
            ):
                await user.should_see(marker=marker)
            for marker in (
                "simulation-histogram-bins",
                "simulation-density",
                "simulation-heatmap-palette",
                "simulation-colorbar",
                "simulation-log-z",
            ):
                await user.should_not_see(marker=marker)
            await user.should_see(marker="simulation-line-or-bar-color")
            for marker in (
                "simulation-line-width",
                "simulation-show-grid",
            ):
                await user.should_see(marker=marker)
            await user.should_not_see(marker="simulation-show-markers")

            user.find(marker="simulation-view").click()
            user.find("Histogram").click()
            for marker in ("simulation-histogram-bins", "simulation-density"):
                await user.should_see(marker=marker)
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
            assert view.options == ["Default", "Histogram"]
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
            user.find("Fixed seed").click()
            await user.should_see(marker="simulation-seed")

            await user.should_not_see(marker="simulation-stall-duration")
            user.find("Non-recovering").click()
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
            assert button.icon == "play_arrow"
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

    def fake_run(request: ConstantSimulationRequest, report, _cancel) -> ConstantSimulationResult:
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
                "completion_time": np.array([17.0, 19.0]),
                "replicated_fraction_xt": np.array([[0.0, 0.8], [0.0, 0.7], [0.0, 0.9]]),
            },
            detailed_outputs=True,
        )

    async def fake_run_chart_method(
        chart: EChart,
        method: str,
        arguments: dict[str, object],
        *,
        timeout: float,
    ) -> str:
        chart_method_calls.append((chart, method, arguments, timeout))
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
            user.find(marker="simulation-track-detailed-outputs").click()
            user.find(marker="action-simulate").click()
            await user.should_see(marker="simulation-result-plot", retries=30)
            await user.should_see(marker="simulation-stats", retries=10)
            assert requests and requests[0].track_detailed_outputs is True

            user.find("Plotting settings").click()
            with user.scope("Plotting settings"):
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
            assert all(value.is_integer() for value in automatic_limits)
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
            assert line_chart.options["series"][0]["symbol"] == "none"
            assert line_chart.options["xAxis"]["splitLine"]["show"] is True
            assert line_chart.options["legend"]["right"] == 28
            assert line_chart.options["grid"]["right"] == 18
            assert line_chart.options["grid"]["containLabel"] is True
            assert line_chart.options["xAxis"]["nameGap"] == 28
            assert line_chart.options["yAxis"]["nameGap"] == 36
            view = cast(
                Select,
                next(iter(user.find(marker="simulation-view").elements)),
            )
            assert view.options == ["Default", "Histogram"]
            assert view.value == "Default"
            user.find(marker="action-download-plot").click()
            user.find(marker="download-plot-svg").click()
            for _ in range(30):
                if capture_exports:
                    break
                await asyncio.sleep(0.02)
            assert chart_method_calls == [
                (
                    line_chart,
                    "getDataURL",
                    {"type": "png", "pixelRatio": pytest.approx(150 / 96)},
                    10,
                )
            ]
            assert capture_exports == [
                (
                    "data:image/png;base64,cHJldmlldy1jYXB0dXJl",
                    "SVG",
                    {
                        "filename": "dnascape_replication_timing_plot",
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
            assert styled_chart.options["series"][0]["symbol"] == "none"
            assert styled_chart.options["xAxis"]["splitLine"]["show"] is False
            assert styled_chart.options["yAxis"]["splitLine"]["show"] is False

            user.find(marker="simulation-view").click()
            user.find("Histogram").click()
            assert view.value == "Histogram"
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

            user.find("Invert Y").click()
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
            assert distribution_chart.options["series"][0]["type"] == "bar"
            assert distribution_chart.options["xAxis"]["position"] == "bottom"

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
            assert chart.options["series"][0]["silent"] is True
            assert chart.options["tooltip"]["show"] is False
            assert chart.options["xAxis"]["position"] == "bottom"

            user.find(marker="action-download-plot").click()
            user.find(marker="download-plot-svg").click()
            for _ in range(30):
                if len(capture_exports) >= 2:
                    break
                await asyncio.sleep(0.02)
            assert len(chart_method_calls) == 2
            captured_chart, method, arguments, timeout = chart_method_calls[1]
            assert captured_chart is chart
            assert method == "getDataURL"
            assert arguments == {"type": "png", "pixelRatio": pytest.approx(150 / 96)}
            assert timeout == 10
            assert capture_exports[1] == (
                "data:image/png;base64,cHJldmlldy1jYXB0dXJl",
                "SVG",
                {
                    "filename": "dnascape_replicated_fraction_xt_plot",
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
            assert file_format == "NPY"
            assert compression == "gzip"
            assert requested_name == "dnascape_simulation_replicated_fraction_xt"
            assert artifact.filename == "dnascape_simulation_replicated_fraction_xt.npy.gz"
            assert artifact.data
            values = np.load(
                io.BytesIO(gzip.decompress(artifact.data)),
                allow_pickle=False,
            )
            np.testing.assert_array_equal(
                values,
                np.array([[0.0, 0.8], [0.0, 0.7], [0.0, 0.9]]),
            )

            user.find(marker="raw-output-compression").click()
            user.find(marker="action-download-raw-data").click()
            for _ in range(30):
                if len(raw_exports) >= 2:
                    break
                await asyncio.sleep(0.02)
            assert raw_exports[1][2] == "none"
            assert raw_exports[1][4].filename.endswith(".npy")

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


def test_uploaded_firing_rate_and_fork_speed_reach_simulation_worker(
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
                next(iter(user.find(marker="source-firing-rate-mode").elements)),
            )
            speed_mode = cast(
                Select,
                next(iter(user.find(marker="source-fork-speed-mode").elements)),
            )
            firing_mode.set_value("profile")
            speed_mode.set_value("profile")
            await user.should_see(marker="source-firing-rate-profile")
            await user.should_see(marker="source-fork-speed-profile")

            firing_upload = cast(
                Upload,
                next(iter(user.find(marker="source-firing-rate-upload").elements)),
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
                        b"position,firing_rate\n0,0.1\n1,0.2\n2,0.3\n",
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
            await user.should_see(marker="source-firing-rate-detected-data")
            await user.should_see(marker="source-fork-speed-detected-data")
            await user.should_see(marker="source-firing-rate-upload-file")
            await user.should_see(marker="source-fork-speed-upload-file")
            firing_form = _select_by_marker(user, "source-firing-rate-form")
            speed_form = _select_by_marker(user, "source-fork-speed-form")
            assert firing_form.value == "x"
            assert speed_form.value == "t"
            await user.should_not_see("Accepted:")
            assert not firing_upload.visible
            assert not speed_upload.visible
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
            np.testing.assert_array_equal(requests[0].firing_rate, [0.1, 0.2, 0.3])
            np.testing.assert_array_equal(requests[0].fork_speed, [[1.1, 1.2, 1.3]])
            assert np.asarray(requests[0].firing_rate).shape == (3,)
            assert np.asarray(requests[0].fork_speed).shape == (1, 3)

    asyncio.run(scenario())


def test_simulation_profile_headers_select_arrays_and_resize_region() -> None:
    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-simulation").click()
            await user.should_see("Simulate DNA replication")
            mode = cast(
                Select,
                next(iter(user.find(marker="source-firing-rate-mode").elements)),
            )
            mode.set_value("profile")
            uploader = cast(
                Upload,
                next(iter(user.find(marker="source-firing-rate-upload").elements)),
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
            await user.should_see(marker="source-firing-rate-detected-data")
            column = cast(
                Select,
                next(iter(user.find(marker="source-firing-rate-value-column").elements)),
            )
            region_end = cast(
                Number,
                next(iter(user.find(marker="simulation-region-end").elements)),
            )
            assert column.options == ["chr1", "chr2"]
            assert column.value == "chr1"
            assert region_end.value == pytest.approx(3.0)

            column.set_value("chr2")
            await asyncio.sleep(0)
            assert region_end.value == pytest.approx(2.0)

            user.find(marker="source-firing-rate-upload-remove").click()
            await user.should_see(marker="source-firing-rate-upload")
            await uploader.handle_uploads(
                [
                    Upload.SmallFileUpload(
                        "named_profiles.csv",
                        "text/csv",
                        b"sample_a,sample_b\n0.1,0.3\n0.2,0.4\n0.3,\n",
                    )
                ]
            )
            await user.should_see(marker="source-firing-rate-detected-data")
            column = cast(
                Select,
                next(iter(user.find(marker="source-firing-rate-value-column").elements)),
            )
            assert column.options == {
                "__all_columns__": "All columns (space-time)",
                "sample_a": "sample_a",
                "sample_b": "sample_b",
            }
            column.set_value("sample_b")
            await asyncio.sleep(0)
            assert region_end.value == pytest.approx(2.0)

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
