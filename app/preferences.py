"""Client-owned server preferences, persisted only in this browser's localStorage."""

import json
from dataclasses import asdict, fields
from pathlib import Path

from nicegui import context, ui
from nicegui.slot import Slot

from dnascape.workflows.export_settings import ExportSettings, load_export_settings

STORAGE_KEY = "dnascape.preferences.v1"
CLIENT_SETTINGS_KEY = "dnascape.preferences"


def client_settings() -> ExportSettings | None:
    # Do not let NiceGUI create its script-mode pseudo client during imports
    # or headless work. Only an already-entered UI slot owns preferences.
    if not Slot.get_stack():
        return None
    try:
        client = context.client
    except RuntimeError:
        return None
    settings = client.storage.get(CLIENT_SETTINGS_KEY)
    if settings is None:
        settings = ExportSettings()
        client.storage[CLIENT_SETTINGS_KEY] = settings
    return settings


class BrowserPreferences:
    def __init__(self, on_load):
        self.client = context.client
        self.settings = client_settings()
        assert self.settings is not None
        self.loaded = False
        self.on_load = on_load
        self.client.storage["dnascape.preferences_bridge"] = self
        ui.add_body_html("<script>" + Path(__file__).with_name("static").joinpath(
            "preferences.js").read_text(encoding="utf-8") + "</script>")
        with ui.element("div").classes("dn-preferences-loading").mark("preferences-loading") as self.loading:
            ui.spinner(size="sm")
            ui.label("Loading your preferences…")
        ui.on("dnascapePreferences", self.receive)
        ui.on("dnascapePreferenceStorageError", self.storage_error)
        self.client.on_connect(lambda: ui.run_javascript("window.dnascapePreferences.load()"))

    def receive(self, event):
        payload = event.args if isinstance(event.args, dict) else {}
        saved = payload.get("preferences")
        values = saved.get("settings") if isinstance(saved, dict) and saved.get("version") == 1 else {}
        normalized = load_export_settings(values)
        for item in fields(normalized):
            setattr(self.settings, item.name, getattr(normalized, item.name))
        self.loaded = True
        self.on_load()
        self.loading.set_visibility(False)
        if payload.get("available") is False:
            self.storage_error()

    def save(self, *, reset=False):
        # Never overwrite saved values while the initial browser read is pending.
        if not self.loaded:
            return
        if reset:
            ui.run_javascript("window.dnascapePreferences.reset()")
        else:
            payload = json.dumps({"version": 1, "settings": asdict(self.settings)})
            ui.run_javascript(f"window.dnascapePreferences.save({payload})")

    def storage_error(self, _event=None):
        from app.notifications import notify
        notify("Your preferences apply to this tab, but browser storage is unavailable. "
               "They cannot be remembered after a reload.", type="warning")
