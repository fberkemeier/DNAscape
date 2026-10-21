"""Preference drafts are applied atomically and defaults preserve the workspace."""
import asyncio
from dataclasses import asdict, replace

import pytest

from nicegui.elements.switch import Switch
from nicegui.testing import user_simulation

from app.export_settings import ExportSettings, get_export_settings
from app.main import index
from app.theme import DEFAULT_THEME_COLOR, theme_css


def test_settings_drafts_apply_cancel_and_restore_defaults() -> None:
    settings = get_export_settings()
    previous = replace(settings)

    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-simulation").click()
            end = next(iter(user.find(marker="simulation-region-end").elements))
            end.set_value(321)

            def open_settings() -> None:
                user.find(marker="app-menu-toggle").click()
                user.find(marker="menu-settings").click()

            def control(marker: str):
                return next(iter(user.find(marker=f"settings-{marker}").elements))

            open_settings()
            dialog = control("dialog")
            confirmation = control("unsaved-dialog")
            apply = control("apply")
            assert user.client is not None
            body = next(
                element for element in user.client.elements.values()
                if element._props.get("selector") == "body"
            )
            assert control("control-font-size").value == previous.control_font_size_px
            assert body._props["style"]["--dn-control-font-size"] == f"{previous.control_font_size_px}px"
            assert apply.enabled is False
            restore = control("restore-defaults")
            assert restore.parent_slot is apply.parent_slot
            assert apply.parent_slot is not None
            assert apply.parent_slot.children == [restore, apply]
            for marker in ("preserve-page-state", "show-examples", "show-help", "dev-mode"):
                assert isinstance(control(marker), Switch)
            assert control("dev-mode").text == "Developer mode"
            assert control("dev-mode").parent_slot is control("show-help").parent_slot
            control("dev-mode").set_value(not previous.dev_mode)
            await asyncio.sleep(0)
            assert apply.enabled is True
            control("dev-mode").set_value(previous.dev_mode)
            await asyncio.sleep(0)
            assert apply.enabled is False
            control("dev-mode").set_value(not previous.dev_mode)
            control("plot-dpi").set_value(240)
            control("control-font-size").set_value(16)
            await asyncio.sleep(0)
            assert settings == previous
            assert body._props["style"]["--dn-control-font-size"] == f"{previous.control_font_size_px}px"
            close = user.find(marker="settings-close")
            close.click()
            await user.should_see("Changes have been made but not applied.")
            assert confirmation.value is True
            user.find(marker="settings-keep-editing").click()
            assert confirmation.value is False and dialog.value is True
            assert control("plot-dpi").value == 240
            # Quasar emits shake when a persistent dialog's backdrop is clicked.
            user.find(marker="settings-dialog").trigger("shake")
            await user.should_see("Changes have been made but not applied.")
            assert confirmation.value is True
            assert settings == previous
            user.find(marker="settings-discard-changes").click()
            assert confirmation.value is False and dialog.value is False
            open_settings()
            assert control("dev-mode").value == previous.dev_mode
            assert control("plot-dpi").value == previous.plot_dpi
            assert control("control-font-size").value == previous.control_font_size_px
            control("dev-mode").set_value(True)
            control("show-help").set_value(True)
            control("control-font-size").set_value(16)
            user.find(marker="settings-dialog").trigger("shake")
            await user.should_see(marker="settings-unsaved-apply")
            user.find(marker="settings-unsaved-apply").click()
            await asyncio.sleep(0)
            assert settings.dev_mode is True and settings.show_help is True
            assert settings.control_font_size_px == 16
            assert body._props["style"]["--dn-control-font-size"] == "16px"
            assert dialog.value is False and confirmation.value is False
            await user.should_see(marker="action-code")
            assert end.value == 321
            open_settings()
            user.find(marker="settings-restore-defaults").click()
            assert control("dev-mode").value is False
            assert control("control-font-size").value == ExportSettings().control_font_size_px
            assert body._props["style"]["--dn-control-font-size"] == "16px"
            assert settings.dev_mode is True
            assert end.value == 321
            user.find(marker="settings-apply").click()
            await asyncio.sleep(0)
            assert settings == ExportSettings()
            assert body._props["style"]["--dn-control-font-size"] == "14px"
            assert body._props["style"]["--dn-control-font-scale"] == "1"
            assert end.value == 321
            await user.should_not_see(marker="action-code")
            open_settings()
            user.find(marker="settings-restore-defaults").click()
            assert control("apply").enabled is False
            user.find(marker="settings-dialog").trigger("shake")
            assert confirmation.value is False and dialog.value is False

    try:
        asyncio.run(scenario())
    finally:
        for key, value in asdict(previous).items():
            setattr(settings, key, value)


@pytest.mark.parametrize("size", [None, 11, 19, 14.5])
def test_invalid_control_font_sizes_cannot_partially_apply_settings(size) -> None:
    settings = get_export_settings()
    previous = replace(settings)

    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="app-menu-toggle").click()
            user.find(marker="menu-settings").click()

            def control(marker: str):
                return next(iter(user.find(marker=f"settings-{marker}").elements))

            control("plot-dpi").set_value(240 if previous.plot_dpi != 240 else 150)
            control("control-font-size").set_value(size)
            user.find(marker="settings-dialog").trigger("shake")
            await user.should_see(marker="settings-unsaved-apply")
            user.find(marker="settings-unsaved-apply").click()
            await user.should_see("Enter a valid controls font size.")
            assert settings == previous
            assert control("dialog").value is True
            assert control("unsaved-dialog").value is True
            user.find(marker="settings-keep-editing").click()
            control("control-font-size").set_value(18)
            user.find(marker="settings-apply").click()
            assert settings.control_font_size_px == 18
            assert control("dialog").value is False

    try:
        asyncio.run(scenario())
    finally:
        for key, value in asdict(previous).items():
            setattr(settings, key, value)


def test_theme_applies_atomically_to_existing_logos_and_restores_defaults() -> None:
    settings = get_export_settings()
    previous = replace(settings)

    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            assert user.client is not None
            body = next(element for element in user.client.elements.values() if element._props.get("selector") == "body")
            logos = [element for element in user.client.elements.values() if element._props.get("data-dn-logo")]
            assert len(logos) == 3
            before = [logo._props["data-dn-logo-light"] for logo in logos]

            def open_settings():
                user.find(marker="app-menu-toggle").click()
                user.find(marker="menu-settings").click()
                return next(iter(user.find(marker="settings-theme-color").elements))

            color = open_settings()
            assert color.value == previous.theme_color
            color.set_value("#2563eb")
            assert settings == previous
            assert [logo._props["data-dn-logo-light"] for logo in logos] == before
            user.find(marker="settings-close").click()
            await user.should_see("Changes have been made but not applied.")
            user.find(marker="settings-discard-changes").click()
            assert settings == previous
            color = open_settings()
            color.set_value("#2563eb")
            user.find(marker="settings-apply").click()
            assert settings.theme_color == "#2563eb"
            assert body._props["style"]["--dn-theme-base"] == "#2563eb"
            assert body._props["style"]["--dn-theme-dark"] == theme_css("#2563eb")["--dn-theme-dark"]
            assert all(logo._props["data-dn-logo-light"] != value for logo, value in zip(logos, before))
            assert body._props["props"]["data-dn-favicon-light"].startswith("data:image/svg+xml;base64,")
            color = open_settings()
            user.find(marker="settings-restore-defaults").click()
            assert color.value == DEFAULT_THEME_COLOR
            assert settings.theme_color == "#2563eb"
            user.find(marker="settings-apply").click()
            assert settings.theme_color == DEFAULT_THEME_COLOR
            assert body._props["style"]["--dn-theme-base"] == DEFAULT_THEME_COLOR

    try:
        asyncio.run(scenario())
    finally:
        for key, value in asdict(previous).items():
            setattr(settings, key, value)


@pytest.mark.parametrize("invalid", (None, "", "not a colour", "#12345678"))
def test_invalid_theme_cannot_partially_apply_other_settings(invalid) -> None:
    settings = get_export_settings()
    previous = replace(settings)

    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="app-menu-toggle").click()
            user.find(marker="menu-settings").click()
            next(iter(user.find(marker="settings-theme-color").elements)).set_value(invalid)
            next(iter(user.find(marker="settings-plot-dpi").elements)).set_value(240)
            user.find(marker="settings-apply").click()
            await user.should_see("Choose a valid theme colour")
            assert settings == previous

    try:
        asyncio.run(scenario())
    finally:
        for key, value in asdict(previous).items():
            setattr(settings, key, value)
