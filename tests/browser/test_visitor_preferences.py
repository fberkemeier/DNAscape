"""Real Chromium contexts against the app, with separate localStorage and tabs.

Run with DNASCAPE_BROWSER_TESTS=1 and DNASCAPE_TEST_PYTHON pointing to the app
environment; install the browser-test extra and `playwright install chromium`.
"""

import gzip
import os
import socket
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import pytest

if os.environ.get("DNASCAPE_BROWSER_TESTS") != "1":
    pytest.skip("Opt-in real-browser regression", allow_module_level=True)

playwright = pytest.importorskip("playwright.sync_api")
ROOT = Path(__file__).resolve().parents[2]
KEY = "dnascape.preferences.v1"


@pytest.fixture
def server(tmp_path):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = {**os.environ, "RENDER": "true", "DNASCAPE_APP_HOST": "127.0.0.1",
           "DNASCAPE_APP_PORT": str(port), "DNASCAPE_APP_SHOW": "false", "PYTHONPATH": str(ROOT)}
    env.pop("PYTEST_CURRENT_TEST", None)
    env.pop("NICEGUI_USER_SIMULATION", None)
    log = (tmp_path / "server.log").open("w", encoding="utf-8")
    process = subprocess.Popen([os.environ.get("DNASCAPE_TEST_PYTHON", sys.executable), "-m", "app.main"],
        cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    url = f"http://127.0.0.1:{port}"
    try:
        for _ in range(400):
            if process.poll() is not None:
                pytest.fail((tmp_path / "server.log").read_text(encoding="utf-8"))
            try:
                urllib.request.urlopen(url, timeout=1).close()
                break
            except (OSError, TimeoutError):
                time.sleep(.15)
        else:
            pytest.fail("Test server did not start")
        yield url
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
        log.close()


def ready(page, url):
    page.goto(url)
    page.locator(".dn-preferences-loading").wait_for(state="hidden")


def settings(page):
    page.get_by_role("button", name="Open application menu", exact=True).click()
    page.get_by_label("Open settings", exact=True).click()
    return page.locator(".dn-settings-dialog").filter(has=page.get_by_text("Interface", exact=True))


def choose(page, label, option):
    page.get_by_label(label, exact=True).click()
    page.get_by_role("option", name=option, exact=True).click()


def apply(page, color, font, points, fmt, compression):
    dialog = settings(page)
    page.get_by_label("DNAscape theme", exact=True).fill(color)
    page.get_by_label("Controls font size (px)", exact=True).fill(str(font))
    page.get_by_label("Preview points", exact=True).fill(str(points))
    page.get_by_label("Plot export DPI", exact=True).fill("240")
    choose(page, "Cached file type", fmt)
    choose(page, "Cached compression", compression)
    dialog.get_by_role("button", name="Apply", exact=True).click()
    dialog.wait_for(state="hidden")
    page.wait_for_function("([key,color]) => JSON.parse(localStorage.getItem(key))?.settings.theme_color === color", arg=[KEY, color])


def color(page):
    return page.evaluate("document.body.style.getPropertyValue('--dn-theme-base')")


def enable_history(page):
    page.get_by_role("button", name="Cached Files", exact=True).click()
    page.get_by_label("Open cache settings", exact=True).click()
    checkbox = page.get_by_role("checkbox", name="Workflow results", exact=True)
    checkbox.click()
    playwright.expect(checkbox).to_have_attribute("aria-checked", "true")
    page.get_by_label("Close cache settings", exact=True).click()


def simulate(page, bins):
    page.get_by_role("button", name="Simulation", exact=True).click()
    workspace = page.locator(".dn-page-view:visible")
    workspace.get_by_text("Region", exact=True).first.click()
    workspace.get_by_label("End (kb)", exact=True).fill(str(bins))
    workspace.get_by_text("Simulation settings", exact=True).click()
    workspace.get_by_label("Simulations (replicates)", exact=True).fill("2")
    page.get_by_role("button", name="Simulate", exact=True).click()
    page.locator(".dn-run-status-line").filter(has_text="Complete").wait_for(timeout=15000)


def test_browser_isolation_reload_exports_and_reset(server, tmp_path):
    with playwright.sync_playwright() as driver:
        browser = driver.chromium.launch(headless=True)
        try:
            first, second = browser.new_context(accept_downloads=True), browser.new_context(accept_downloads=True)
            alice, bob = first.new_page(), second.new_page()
            ready(alice, server)
            ready(bob, server)
            assert color(alice) == color(bob) == "#b03412"
            alice.evaluate("localStorage.setItem('unrelated.preferences', 'keep')")
            apply(alice, "#2563eb", 16, 200, "CSV", "GZIP (.gz)")
            assert color(alice) == "#2563eb" and color(bob) == "#b03412"
            apply(bob, "#0f766e", 18, 800, "NPZ", "None")
            bob_saved = bob.evaluate("localStorage.getItem('dnascape.preferences.v1')")
            alice.get_by_role("button", name="Open application menu", exact=True).click()
            alice.get_by_label("System theme; switch to light theme", exact=True).click()
            alice.keyboard.press("Escape")
            alice.wait_for_function("JSON.parse(localStorage.getItem('dnascape.preferences.v1')).settings.theme_mode === 'light'")
            # Reload and a fresh page in this browser recover only preferences.
            ready(alice, server)
            assert color(alice) == "#2563eb"
            assert alice.locator("body").get_attribute("class").find("body--light") >= 0
            assert alice.evaluate("document.body.style.getPropertyValue('--dn-control-font-size')") == "16px"
            another_tab = first.new_page()
            ready(another_tab, server)
            assert color(another_tab) == "#2563eb"
            for page, name in ((alice, "alice-timing.csv"), (bob, "bob-timing.csv")):
                page.get_by_role("button", name="Mapping", exact=True).click()
                page.locator('input[type="file"]').first.set_input_files({
                    "name": name, "mimeType": "text/csv", "buffer": b"timing\n100\n110\n120\n"})
                playwright.expect(page.locator(".dn-upload-file-name").get_by_text(name, exact=True)).to_be_visible()
            assert alice.get_by_text("bob-timing.csv", exact=True).count() == 0
            assert bob.get_by_text("alice-timing.csv", exact=True).count() == 0
            another_tab.get_by_role("button", name="Mapping", exact=True).click()
            assert another_tab.get_by_text("alice-timing.csv", exact=True).count() == 0
            enable_history(alice)
            enable_history(bob)
            simulate(alice, 20)
            bob.get_by_role("button", name="Cached Files", exact=True).click()
            assert bob.get_by_text("Simulation result", exact=True).count() == 0
            another_tab.get_by_role("button", name="Cached Files", exact=True).click()
            assert another_tab.get_by_text("Simulation result", exact=True).count() == 0
            alice.get_by_role("button", name="Cached Files", exact=True).click()
            with alice.expect_download() as first_download:
                alice.get_by_label("Download cached file", exact=True).click()
            download = first_download.value
            assert download.suggested_filename.endswith(".csv.gz")
            assert len(gzip.decompress(Path(download.path()).read_bytes()).splitlines()) == 21
            simulate(bob, 30)
            bob.get_by_role("button", name="Cached Files", exact=True).click()
            with bob.expect_download() as second_download:
                bob.get_by_label("Download cached file", exact=True).click()
            download = second_download.value
            assert download.suggested_filename.endswith(".npz")
            with zipfile.ZipFile(download.path()) as archive:
                assert any(name.endswith(".npy") for name in archive.namelist())
            # Reset affects this browser's preferences and only this tab's workspace.
            alice.get_by_role("button", name="Open application menu", exact=True).click()
            alice.get_by_label("Reset everything to default", exact=True).click()
            alice.get_by_role("button", name="Reset everything", exact=True).click()
            alice.wait_for_function("localStorage.getItem('dnascape.preferences.v1') === null")
            assert alice.evaluate("localStorage.getItem('unrelated.preferences')") == "keep"
            assert bob.evaluate("localStorage.getItem('dnascape.preferences.v1')") == bob_saved
            assert color(bob) == "#0f766e"
            assert bob.get_by_text("Simulation result", exact=True).count() == 1
            ready(alice, server)
            assert color(alice) == "#b03412"
            ready(bob, server)
            assert color(bob) == "#0f766e"
        finally:
            browser.close()


def test_unavailable_and_corrupt_storage_keep_visitors_isolated(server):
    with playwright.sync_playwright() as driver:
        browser = driver.chromium.launch(headless=True)
        try:
            blocked = browser.new_context()
            blocked.add_init_script("Object.defineProperty(window, 'localStorage', {get() {throw new DOMException('Blocked', 'SecurityError')}})")
            page = blocked.new_page()
            ready(page, server)
            playwright.expect(page.get_by_text("Your preferences apply to this tab", exact=False)).to_be_visible()
            dialog = settings(page)
            page.get_by_label("DNAscape theme", exact=True).fill("#be185d")
            dialog.get_by_role("button", name="Apply", exact=True).click()
            page.wait_for_function("document.body.style.getPropertyValue('--dn-theme-base') === '#be185d'")
            corrupt = browser.new_context()
            corrupt.add_init_script("localStorage.setItem('dnascape.preferences.v1', JSON.stringify({version:1,settings:{theme_color:'bad',preview_points:-100,control_font_size_px:1000}}))")
            other = corrupt.new_page()
            ready(other, server)
            assert color(other) == "#b03412"
            assert other.evaluate("document.body.style.getPropertyValue('--dn-control-font-size')") == "14px"
        finally:
            browser.close()
