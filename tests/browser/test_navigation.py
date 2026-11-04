"""First visits and repeat navigation on a real hosted-mode shell."""

import json
import os
import time

import pytest

if os.environ.get("DNASCAPE_BROWSER_TESTS") != "1":
    pytest.skip("Opt-in real-browser regression", allow_module_level=True)

from test_visitor_preferences import playwright, ready, server  # noqa: F401


def test_first_menu_visits_and_repeat_navigation(request):
    url = request.getfixturevalue("server")
    timings = {}
    pages = (
        ("Mapping", "Map replication kinetic landscapes"),
        ("Data Manager", "Convert genomic data files"),
        ("Simulation", "Simulate DNA replication"),
        ("Analysis", "Explore DNAscape data"),
        ("Cached Files", "Session files and results"),
        ("Resources", "Project links and citation guidance"),
    )
    with playwright.sync_playwright() as driver:
        browser = driver.chromium.launch(headless=True)
        try:
            page = browser.new_page()
            ready(page, url)
            for name, heading in pages:
                started = time.perf_counter()
                page.get_by_role("button", name=name, exact=True).click()
                page.locator(".dn-page-view:visible").get_by_text(heading, exact=True).wait_for()
                timings[name] = round(time.perf_counter() - started, 3)
            for name, heading in (pages[0], pages[2], pages[0]):
                page.get_by_role("button", name=name, exact=True).click()
                page.locator(".dn-page-view:visible").get_by_text(heading, exact=True).wait_for()
            assert len(page.locator(".dn-page-view:visible").all()) == 1
            print("First-click seconds:", json.dumps(timings))
        finally:
            browser.close()
