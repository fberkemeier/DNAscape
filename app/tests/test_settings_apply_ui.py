"""Preference drafts are applied atomically and defaults preserve the workspace."""
import asyncio
from dataclasses import asdict, replace

from nicegui.elements.switch import Switch
from nicegui.testing import user_simulation

from app.export_settings import ExportSettings, get_export_settings
from app.main import index


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
            assert apply.enabled is False
            restore = control("restore-defaults")
            assert restore.parent_slot is apply.parent_slot
            assert apply.parent_slot is not None
            assert apply.parent_slot.children == [restore, apply]
            for marker in ("preserve-page-state", "show-examples", "show-help", "dev-mode"):
                assert isinstance(control(marker), Switch)
            assert control("dev-mode").text == "Developed mode"
            assert control("dev-mode").parent_slot is control("show-help").parent_slot
            control("dev-mode").set_value(not previous.dev_mode)
            await asyncio.sleep(0)
            assert apply.enabled is True
            control("dev-mode").set_value(previous.dev_mode)
            await asyncio.sleep(0)
            assert apply.enabled is False
            control("dev-mode").set_value(not previous.dev_mode)
            control("plot-dpi").set_value(240)
            await asyncio.sleep(0)
            assert settings == previous
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
            control("dev-mode").set_value(True)
            control("show-help").set_value(True)
            user.find(marker="settings-apply").click()
            await asyncio.sleep(0)
            assert settings.dev_mode is True and settings.show_help is True
            await user.should_see(marker="action-code")
            assert end.value == 321
            open_settings()
            user.find(marker="settings-restore-defaults").click()
            assert control("dev-mode").value is False
            assert settings.dev_mode is True
            assert end.value == 321
            user.find(marker="settings-apply").click()
            await asyncio.sleep(0)
            assert settings == ExportSettings()
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
