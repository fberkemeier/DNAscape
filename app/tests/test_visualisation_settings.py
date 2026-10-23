"""Global point budgets update mounted previews while preserving their viewports."""

import asyncio
from dataclasses import asdict, replace

import numpy as np
from nicegui.elements.upload import Upload
from nicegui.testing import user_simulation

from app.export_settings import get_export_settings
from app.main import index
from app.preview_plotting import PreviewChartHandle


def test_visualisation_settings_update_existing_preview_and_zoom(monkeypatch):
    preferences = get_export_settings()
    previous = replace(preferences)
    handles = []
    attach = PreviewChartHandle.attach_settings
    def capture(handle, controls):
        attach(handle, controls)
        handles.append(handle)
    monkeypatch.setattr(PreviewChartHandle, "attach_settings", capture)

    async def scenario():
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-plotting").click()
            def control(marker):
                return next(item for item in user.client.elements.values() if marker in item._markers and not item.is_deleted)
            assert control("plotting-aspect-ratio").value == 3
            values = np.sin(np.arange(10001.) / 100)
            payload = ("signal\n" + "\n".join(map(str, values))).encode()
            await control("plotting-data-upload").handle_uploads([Upload.SmallFileUpload("profile.tsv", "text/plain", payload)])
            await user.should_see(marker="plotting-detected-data-1", retries=80)
            user.find(marker="action-plot").click()
            await user.should_see(marker="plotting-result-plot", retries=80)
            handle = handles[-1]
            assert len(handle.options["series"][0]["data"]) == 1000

            async def set_budget(limit):
                user.find(marker="app-menu-toggle").click()
                user.find(marker="menu-settings").click()
                visualisation = next(item for item in user.client.elements.values() if getattr(item, "text", None) == "Visualisation")
                section = visualisation.parent_slot.parent
                for marker in ("settings-theme-color", "settings-control-font-size", "settings-preview-points"):
                    assert control(marker).parent_slot.parent.parent_slot.parent is section
                with user.client:
                    control("settings-preview-points").set_value(limit)
                user.find(marker="settings-apply").click()
                await asyncio.sleep(.05)
            await set_budget(2000)
            assert preferences.preview_points == 2000
            assert len(handle.options["series"][0]["data"]) == 2000
            with user.client:
                handle.accept_navigation(dict(context=handle._navigation.context, ranges={"x": (2000, 4000)}))
            await set_budget(400)
            data = np.asarray(handle.options["series"][0]["data"])
            assert 398 <= len(data[(data[:, 0] >= 2000) & (data[:, 0] <= 4000)]) <= 400
            assert handle.options["xAxis"]["min"] == 2000
            assert handle.options["xAxis"]["max"] == 4000
            assert control("plotting-show-hover").text == "Hover"
            assert control("plotting-show-hover").parent_slot is control("plotting-show-grid").parent_slot
            await set_budget(0)
            assert preferences.preview_points == 400
            assert control("settings-dialog").value
            user.find(marker="settings-restore-defaults").click()
            assert control("settings-preview-points").value == 1000
            user.find(marker="settings-apply").click()
            await asyncio.sleep(.05)
            assert preferences.preview_points == 1000
            assert handle.options["xAxis"]["min"] == 2000
            assert handle.options["xAxis"]["max"] == 4000

    try:
        asyncio.run(scenario())
    finally:
        for name, value in asdict(previous).items():
            setattr(preferences, name, value)
