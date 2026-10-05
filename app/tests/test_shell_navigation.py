"""Focused regression coverage for the responsive application shell."""

from __future__ import annotations

import asyncio
from typing import cast

import pytest
from nicegui.elements.button import Button
from nicegui.elements.checkbox import Checkbox
from nicegui.elements.drawer import LeftDrawer
from nicegui.elements.number import Number
from nicegui.elements.select import Select
from nicegui.testing import user_simulation

from app.export_settings import (
    CACHE_FILE_FORMAT_OPTIONS,
    COMPRESSION_OPTIONS,
    ExportSettings,
    get_export_settings,
)
from app.main import index
from app.result_naming import NAME_STYLE_OPTIONS
from app.session_cache import current_session_cache
from app.simulation_exports import ExportArtifact


def test_drawer_fully_closes_and_uses_an_external_control_to_reopen() -> None:
    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            drawer = cast(
                LeftDrawer,
                next(iter(user.find(marker="navigation-drawer").elements)),
            )
            collapse = cast(
                Button,
                next(iter(user.find(marker="drawer-toggle").elements)),
            )
            assert drawer.value is True
            assert collapse.visible is True
            assert "mini" not in drawer._props

            user.find(marker="drawer-toggle").click()
            expand = cast(
                Button,
                next(iter(user.find(marker="drawer-expand").elements)),
            )
            assert drawer.value is False
            assert collapse.visible is False
            assert expand.visible is True
            assert expand._props["aria-expanded"] == "false"
            assert "mini" not in drawer._props

            user.find(marker="drawer-expand").click()
            assert drawer.value is True
            assert collapse.visible is True
            assert expand.visible is False
            assert collapse._props["aria-expanded"] == "true"

    asyncio.run(scenario())


def test_database_icon_is_explicitly_stroke_only() -> None:
    from app.icons import DATA_ICON_SVG

    assert "<ellipse" in DATA_ICON_SVG
    assert 'fill="none"' in DATA_ICON_SVG
    assert 'stroke="currentColor"' in DATA_ICON_SVG


def test_export_preferences_live_in_the_application_settings_menu() -> None:
    settings = get_export_settings()
    previous = (
        settings.plot_dpi,
        settings.raw_compression,
        settings.cached_file_format,
        settings.cached_compression,
        settings.result_name_style,
        settings.preserve_page_state,
        settings.show_examples,
        settings.show_help,
    )

    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="app-menu-toggle").click()
            user.find(marker="menu-settings").click()
            await user.should_see("Settings")

            dpi = cast(Number, next(iter(user.find(marker="settings-plot-dpi").elements)))
            compression = cast(
                Select,
                next(iter(user.find(marker="settings-raw-compression").elements)),
            )
            cached_format = cast(
                Select,
                next(iter(user.find(marker="settings-cached-file-format").elements)),
            )
            cached_compression = cast(
                Select,
                next(iter(user.find(marker="settings-cached-compression").elements)),
            )
            result_name_style = cast(
                Select,
                next(iter(user.find(marker="settings-result-name-style").elements)),
            )
            preserve_page_state = cast(
                Checkbox,
                next(iter(user.find(marker="settings-preserve-page-state").elements)),
            )
            show_examples = cast(
                Checkbox,
                next(iter(user.find(marker="settings-show-examples").elements)),
            )
            show_help = cast(
                Checkbox,
                next(iter(user.find(marker="settings-show-help").elements)),
            )
            assert dpi.value == previous[0]
            assert compression.options == COMPRESSION_OPTIONS
            assert compression.value == previous[1]
            assert cached_format.options == CACHE_FILE_FORMAT_OPTIONS
            assert cached_format.value == previous[2]
            assert cached_compression.options == COMPRESSION_OPTIONS
            assert cached_compression.value == previous[3]
            assert result_name_style.options == NAME_STYLE_OPTIONS
            assert result_name_style.value == previous[4]
            assert preserve_page_state.value == previous[5]
            assert show_examples.value == previous[6]
            assert show_help.value == previous[7]
            assert user.client is not None
            with user.client:
                dpi.set_value(240)
                compression.set_value("zip")
                cached_format.set_value("csv")
                cached_compression.set_value("xz")
                result_name_style.set_value("full")
                preserve_page_state.set_value(False)
                show_examples.set_value(False)
                show_help.set_value(True)
            await asyncio.sleep(0)
            assert settings.plot_dpi == 240
            assert settings.raw_compression == "zip"
            assert settings.cached_file_format == "csv"
            assert settings.cached_compression == "xz"
            assert settings.result_name_style == "full"
            assert settings.preserve_page_state is False
            assert settings.show_examples is False
            assert settings.show_help is True
            with user.client:
                result_name_style.set_value("short")
            await asyncio.sleep(0)
            assert settings.result_name_style == "short"

    try:
        asyncio.run(scenario())
    finally:
        (
            settings.plot_dpi,
            settings.raw_compression,
            settings.cached_file_format,
            settings.cached_compression,
            settings.result_name_style,
            settings.preserve_page_state,
            settings.show_examples,
            settings.show_help,
        ) = previous


def test_page_controls_are_retained_until_page_preservation_is_disabled() -> None:
    settings = get_export_settings()
    previous = settings.preserve_page_state
    settings.preserve_page_state = True

    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-simulation").click()
            end = cast(
                Number,
                next(iter(user.find(marker="simulation-region-end").elements)),
            )
            assert user.client is not None
            with user.client:
                end.set_value(321)

            user.find(marker="nav-mapping").click()
            user.find(marker="nav-simulation").click()
            retained = cast(
                Number,
                next(iter(user.find(marker="simulation-region-end").elements)),
            )
            assert retained.value == 321

            user.find(marker="app-menu-toggle").click()
            user.find(marker="menu-settings").click()
            preserve = cast(
                Checkbox,
                next(iter(user.find(marker="settings-preserve-page-state").elements)),
            )
            with user.client:
                preserve.set_value(False)
            user.find(marker="nav-mapping").click()
            user.find(marker="nav-simulation").click()
            reset = cast(
                Number,
                next(iter(user.find(marker="simulation-region-end").elements)),
            )
            assert reset.value == 1_000

    try:
        asyncio.run(scenario())
    finally:
        settings.preserve_page_state = previous


def test_about_menu_identifies_dnascape_and_its_author() -> None:
    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="app-menu-toggle").click()
            user.find(marker="menu-about").click()
            await user.should_see(marker="about-dialog")
            await user.should_see(marker="about-logo")
            await user.should_see("DNAscape builds quantitative models of DNA replication")
            await user.should_see("Francisco Berkemeier")
            email = next(iter(user.find(marker="about-email").elements))
            assert email._props["href"] == "mailto:fp409@cam.ac.uk"
            documentation = next(iter(user.find(marker="about-documentation").elements))
            assert documentation._props["href"].startswith("https://dnascape.readthedocs.io")
            github = next(iter(user.find(marker="about-github").elements))
            assert github._props["href"] == "https://github.com/fberkemeier/DNAscape"

    asyncio.run(scenario())


def test_reset_menu_restores_defaults_and_clears_the_client_workspace() -> None:
    settings = get_export_settings()
    previous = (
        settings.plot_dpi,
        settings.raw_compression,
        settings.cached_file_format,
        settings.cached_compression,
        settings.result_name_style,
        settings.preserve_page_state,
        settings.show_examples,
        settings.show_help,
    )
    settings.plot_dpi = 288
    settings.raw_compression = "zip"
    settings.cached_file_format = "csv"
    settings.cached_compression = "xz"
    settings.result_name_style = "full"
    settings.preserve_page_state = False
    settings.show_examples = False
    settings.show_help = True

    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            assert user.client is not None
            with user.client:
                current_session_cache().add_artifact(
                    ExportArtifact(b"values", "result.npy", "application/octet-stream"),
                    source="Simulation",
                    kind="raw",
                )
            user.find(marker="nav-simulation").click()
            await user.should_see(marker="simulation-region-end")
            user.find(marker="app-menu-toggle").click()
            user.find(marker="menu-reset").click()
            await user.should_see(marker="reset-dialog")
            user.find(marker="confirm-reset-everything").click()
            await user.should_see("Build quantitative models of DNA replication")

            defaults = ExportSettings()
            assert settings == defaults
            assert settings.result_name_style == "short"
            assert settings.show_examples is True
            assert settings.show_help is False
            name_style = cast(
                Select,
                next(iter(user.find(marker="settings-result-name-style").elements)),
            )
            show_examples = cast(
                Checkbox,
                next(iter(user.find(marker="settings-show-examples").elements)),
            )
            show_help = cast(
                Checkbox,
                next(iter(user.find(marker="settings-show-help").elements)),
            )
            assert name_style.value == "short"
            assert show_examples.value is True
            assert show_help.value is False
            with user.client:
                assert current_session_cache().count() == 0

    try:
        asyncio.run(scenario())
    finally:
        (
            settings.plot_dpi,
            settings.raw_compression,
            settings.cached_file_format,
            settings.cached_compression,
            settings.result_name_style,
            settings.preserve_page_state,
            settings.show_examples,
            settings.show_help,
        ) = previous


def test_browser_launch_is_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.main import _show_browser

    monkeypatch.delenv("DNASCAPE_APP_SHOW", raising=False)
    assert _show_browser() is False
    monkeypatch.setenv("DNASCAPE_APP_SHOW", "true")
    assert _show_browser() is True
