"""Focused regression coverage for the responsive application shell."""

from __future__ import annotations

import asyncio
from typing import cast

import pytest
from nicegui.elements.button import Button
from nicegui.elements.drawer import LeftDrawer
from nicegui.elements.number import Number
from nicegui.elements.select import Select
from nicegui.testing import user_simulation

from app.export_settings import COMPRESSION_OPTIONS, get_export_settings
from app.main import index


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
    previous = (settings.plot_dpi, settings.raw_compression)

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
            assert dpi.value == previous[0]
            assert compression.options == COMPRESSION_OPTIONS
            assert compression.value == previous[1]
            assert user.client is not None
            with user.client:
                dpi.set_value(240)
                compression.set_value("zip")
            await asyncio.sleep(0)
            assert settings.plot_dpi == 240
            assert settings.raw_compression == "zip"

    try:
        asyncio.run(scenario())
    finally:
        settings.plot_dpi, settings.raw_compression = previous


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


def test_browser_launch_is_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.main import _show_browser

    monkeypatch.delenv("DNASCAPE_APP_SHOW", raising=False)
    assert _show_browser() is False
    monkeypatch.setenv("DNASCAPE_APP_SHOW", "true")
    assert _show_browser() is True
