"""Field hover hints stay brief and independent of detailed help and plot hover."""

import asyncio
import html
import json
import re
import shutil
import subprocess
from dataclasses import asdict, replace
from pathlib import Path

import pytest
from nicegui.elements.switch import Switch
from nicegui.testing import user_simulation

from app.export_settings import ExportSettings, get_export_settings
from app.field_hints import FIELD_HINTS, MAPPING_SETTING_HINTS
from app.main import index
from dnascape.workflows.mapping_catalog import mapping_routes


def test_hover_text_is_brief_and_covers_all_mapping_settings():
    settings = {setting.key for route in mapping_routes() for setting in route.user_settings}
    assert not settings.difference(MAPPING_SETTING_HINTS)
    for text in (*FIELD_HINTS.values(), *MAPPING_SETTING_HINTS.values()):
        assert 2 <= len(text.split()) <= 12, text
        assert len(text) <= 100, text
        assert "\n" not in text
    assert ExportSettings().show_field_hints is True
    assert ExportSettings().show_help is False


@pytest.mark.parametrize("initial", [True, False])
def test_hover_preference_drafts_apply_and_defaults_are_independent(initial):
    settings = get_export_settings()
    previous = replace(settings)
    settings.show_field_hints = initial
    settings.show_help = False

    async def scenario():
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-simulation").click()
            end = next(iter(user.find(marker="simulation-region-end").elements))
            with user.client:
                end.set_value(97)
            body = next(item for item in user.client.elements.values() if item._props.get("selector") == "body")
            assert ("dn-field-hints-disabled" in body._props["classes"]) is not initial

            def open_settings():
                user.find(marker="app-menu-toggle").click()
                user.find(marker="menu-settings").click()

            def control(marker):
                return next(iter(user.find(marker=f"settings-{marker}").elements))

            open_settings()
            hover = control("show-field-hints")
            assert isinstance(hover, Switch)
            assert hover.value is initial
            assert hover.parent_slot is control("show-help").parent_slot
            with user.client:
                hover.set_value(not initial)
            assert settings.show_field_hints is initial
            assert ("dn-field-hints-disabled" in body._props["classes"]) is not initial
            user.find(marker="settings-close").click()
            user.find(marker="settings-discard-changes").click()
            assert settings.show_field_hints is initial

            open_settings()
            with user.client:
                control("show-field-hints").set_value(False)
                control("show-help").set_value(True)
            user.find(marker="settings-apply").click()
            await asyncio.sleep(0)
            assert settings.show_field_hints is False and settings.show_help is True
            assert "dn-field-hints-disabled" in body._props["classes"]
            assert "dn-field-help-enabled" in body._props["classes"]
            assert end.value == 97

            open_settings()
            user.find(marker="settings-restore-defaults").click()
            assert control("show-field-hints").value is True
            assert control("show-help").value is False
            assert "dn-field-hints-disabled" in body._props["classes"]
            user.find(marker="settings-apply").click()
            await asyncio.sleep(0)
            assert settings.show_field_hints is True and settings.show_help is False
            assert "dn-field-hints-disabled" not in body._props["classes"]
            assert "dn-field-help-enabled" not in body._props["classes"]
            assert end.value == 97
    try:
        asyncio.run(scenario())
    finally:
        for key, value in asdict(previous).items():
            setattr(settings, key, value)


def test_hover_behavior_in_real_browser(tmp_path):
    chrome = shutil.which("chromium") or shutil.which("google-chrome") or shutil.which("chrome")
    windows_chrome = Path("C:/Program Files/Google/Chrome/Application/chrome.exe")
    if chrome is None and windows_chrome.exists():
        chrome = str(windows_chrome)
    if chrome is None:
        pytest.skip("Chrome/Chromium is required for the field hover check")
    root = Path(__file__).resolve().parents[2]
    document = tmp_path / "hints.html"
    document.write_text(
        '<!doctype html><html><head><meta charset="utf-8">'
        f'<link rel="stylesheet" href="{(root / "app/static/styles.css").as_uri()}">'
        '<style>body{margin:0}.probe{position:absolute;width:150px;height:44px}'
        '.probe input{width:100px}.q-field__label{display:block}</style>'
        '</head><body class="dn-body" style="--dn-text:#403932;--dn-panel:white">'
        '<pre id="result">pending</pre>'
        f'<script>window.dnascapeFieldHintLabels={json.dumps(FIELD_HINTS)};</script>'
        f'<script src="{(root / "app/static/field_hints.js").as_uri()}"></script>'
        f'<script src="{(root / "app/tests/fixtures/field_hints_probe.js").as_uri()}"></script>'
        '</body></html>', encoding="utf-8",
    )
    completed = subprocess.run([
        chrome, "--headless", "--disable-gpu", "--no-first-run", "--no-default-browser-check",
        "--allow-file-access-from-files", "--window-size=700,600", "--virtual-time-budget=10000",
        f"--user-data-dir={tmp_path / 'browser'}", "--dump-dom", document.as_uri(),
    ], capture_output=True, text=True, encoding="utf-8", timeout=30, check=True)
    match = re.search(r'<pre id="result">([^<]+)</pre>', completed.stdout)
    assert match is not None and match[1] != "pending", completed.stderr[-1000:]
    result = json.loads(html.unescape(match[1]))
    assert result["passed"], result
    assert not result["errors"]
