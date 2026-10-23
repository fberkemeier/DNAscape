"""Source labels, inspection state, and configuration follow their source."""

import asyncio

import numpy as np
from nicegui.elements.upload import Upload
from nicegui.testing import user_simulation
from nicegui.testing.user import UserInteraction

from app.main import index
from app.pages import data_management as page
from app.remote_sources import RemoteSource


def test_each_upload_keeps_its_configuration_and_bypass_preserves_values(monkeypatch):
    requests, results = [], []
    original = page.convert_data
    def capture(request, *args, **kwargs):
        result = original(request, *args, **kwargs)
        requests.append(request)
        results.append(result)
        return result
    monkeypatch.setattr(page, "convert_data", capture)

    async def scenario():
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-data-management").click()
            def control(name):
                return next(item for item in user.client.elements.values()
                            if f"data-manager-{name}" in item._markers and not item.is_deleted)
            async def select_source(key):
                previous = control("apply-configuration")
                with user.client:
                    control("configuration-source").set_value(key)
                for _ in range(100):
                    await asyncio.sleep(.01)
                    if previous.is_deleted:
                        break
                assert previous.is_deleted
            async def convert(count):
                user.find(marker="action-convert").click()
                for _ in range(200):
                    await asyncio.sleep(.02)
                    if len(results) == count and control("apply-configuration").enabled:
                        break
                assert len(results) == count
            upload = control("upload")
            await upload.handle_uploads([Upload.SmallFileUpload("first.tsv", "text/plain", b"signal\n1\n2\n3\n")])
            await user.should_see("Array size: 3", retries=60)
            await user.should_see(marker="data-manager-upload-source-label-1", retries=60)
            await user.should_see(marker="data-manager-apply-configuration", retries=60)
            assert control("upload-source-label-1").text == "Source"
            assert control("apply-configuration").value is True
            await user.should_not_see(marker="data-manager-source-selection")
            await user.should_not_see(marker="data-manager-configuration-source")
            await upload.handle_uploads([Upload.SmallFileUpload("second.tsv", "text/plain", b"signal\n4\nNaN\n6\n")])
            await user.should_see(marker="data-manager-configuration-source", retries=60)
            await user.should_see(marker="data-manager-upload-source-label-2", retries=60)
            assert control("upload-source-label-1").text == "Source 1"
            assert control("upload-source-label-2").text == "Source 2"
            keys = list(control("configuration-source").options)
            assert list(control("configuration-source").options.values()) == ["Source 1 · first", "Source 2 · second"]
            await select_source(keys[0])
            with user.client:
                control("value-scaling").set_value("multiplier")
                control("scale-multiplier").set_value(2)
            await select_source(keys[1])
            assert control("value-scaling").value == "none"
            with user.client:
                control("value-scaling").set_value("multiplier")
                control("scale-multiplier").set_value(3)
            await select_source(keys[0])
            assert control("scale-multiplier").value == 2
            await convert(2)
            np.testing.assert_equal(results[-2].preview.values, [2, 4, 6])
            np.testing.assert_equal(results[-1].preview.values, [12, 0, 18])
            await select_source(keys[1])
            assert control("scale-multiplier").value == 3
            with user.client:
                control("apply-configuration").set_value(False)
                control("decimal-places").set_value(0)
            await convert(4)
            np.testing.assert_equal(results[-2].preview.values, [2, 4, 6])
            np.testing.assert_equal(results[-1].preview.values, [4, np.nan, 6])
            assert requests[-1].missing_policy == "ignore"
            assert requests[-1].decimal_places is None
            assert requests[-1].scale == 1
            await select_source(keys[0])
            assert control("apply-configuration").value is True
            await select_source(keys[1])
            assert control("apply-configuration").value is False
            assert control("scale-multiplier").value == 3
            remove = next(item for item in user.find(marker="data-manager-upload-remove").elements
                          if item._props.get("data-upload-name") == "first.tsv")
            UserInteraction(user, {remove}, None).click()
            await user.should_not_see("first.tsv", retries=60)
            assert control("upload-source-label-1").text == "Source"
            await user.should_not_see(marker="data-manager-configuration-source")
            assert control("apply-configuration").value is False
    asyncio.run(scenario())


def test_link_field_labels_follow_visible_rows_and_keep_configuration(monkeypatch):
    def fetch(url):
        return RemoteSource(requested_url=url, final_url=url, filename=url.rsplit("/", 1)[-1],
                            content=b"signal\n1\n2\n3\n", content_type="text/plain",
                            etag=None, last_modified=None, accept_ranges=False, redirects=())
    monkeypatch.setattr(page, "fetch_remote_source", fetch)
    async def scenario():
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-data-management").click()
            def control(name):
                return next(item for item in user.client.elements.values()
                            if f"data-manager-{name}" in item._markers and not item.is_deleted)
            async def inspect(row_id):
                action = next(iter(user.find(marker=f"action-inspect-data-link-{row_id}").elements))
                user.find(marker=f"action-inspect-data-link-{row_id}").click()
                for _ in range(100):
                    await asyncio.sleep(.01)
                    if action.text == "✓" and action.enabled:
                        break
                assert action.text == "✓"
                assert action.enabled
                assert action.icon is None
            with user.client:
                control("source-tabs").set_value("url")
            await user.should_see(marker="data-manager-source-url-1")
            assert control("source-url-1").label == "Source (public file link)"
            assert not any("data-manager-link-source-label-1" in item._markers
                           for item in user.client.elements.values())
            with user.client:
                control("source-url-1").set_value("https://example.org/one.tsv")
            await inspect(1)
            with user.client:
                control("apply-configuration").set_value(False)
            user.find(marker="data-manager-add-link").click()
            await user.should_see(marker="data-manager-source-url-2")
            assert control("source-url-1").label == "Source 1 (public file link)"
            assert control("source-url-2").label == "Source 2 (public file link)"
            assert next(iter(user.find(marker="action-inspect-data-link-1").elements)).text == "✓"
            assert next(iter(user.find(marker="action-inspect-data-link-2").elements)).text == "Inspect"
            with user.client:
                control("source-url-2").set_value("https://example.org/two.tsv")
            await inspect(2)
            assert control("apply-configuration").value is True
            keys = list(control("configuration-source").options)
            with user.client:
                control("apply-configuration").set_value(True)
                control("configuration-source").set_value(keys[0])
            for _ in range(60):
                await asyncio.sleep(.01)
                if not control("apply-configuration").value:
                    break
            assert control("apply-configuration").value is False
            user.find(marker="remove-data-link-1").click()
            await user.should_not_see(marker="data-manager-source-url-1", retries=60)
            assert control("source-url-2").label == "Source (public file link)"
            assert next(iter(user.find(marker="action-inspect-data-link-2").elements)).text == "✓"
    asyncio.run(scenario())


def test_link_numbers_follow_row_order_when_inspected_in_reverse(monkeypatch):
    def fetch(url):
        if url.endswith("/bad.tsv"):
            raise ValueError("The remote file could not be inspected.")
        return RemoteSource(requested_url=url, final_url=url, filename=url.rsplit("/", 1)[-1],
                            content=b"signal\n1\n2\n3\n", content_type="text/plain",
                            etag=None, last_modified=None, accept_ranges=False, redirects=())
    monkeypatch.setattr(page, "fetch_remote_source", fetch)
    async def scenario():
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-data-management").click()
            def control(name):
                return next(item for item in user.client.elements.values()
                            if f"data-manager-{name}" in item._markers and not item.is_deleted)
            def action(row_id):
                return next(iter(user.find(marker=f"action-inspect-data-link-{row_id}").elements))
            async def inspect(row_id):
                user.find(marker=f"action-inspect-data-link-{row_id}").click()
                for _ in range(100):
                    await asyncio.sleep(.01)
                    if action(row_id).text == "✓" and action(row_id).enabled:
                        break
                assert action(row_id).text == "✓"
                assert action(row_id).enabled
            with user.client:
                control("source-tabs").set_value("url")
            await user.should_see(marker="data-manager-source-url-1")
            user.find(marker="data-manager-add-link").click()
            await user.should_see(marker="data-manager-source-url-2")
            with user.client:
                control("source-url-1").set_value("https://example.org/one.tsv")
                control("source-url-2").set_value("https://example.org/two.tsv")
            await inspect(2)
            assert control("source-url-2").label == "Source 2 (public file link)"
            await inspect(1)
            assert list(control("configuration-source").options) == ["url:1", "url:2"]
            assert control("source-url-1").label == "Source 1 (public file link)"
            assert control("source-url-2").label == "Source 2 (public file link)"
            with user.client:
                control("source-url-1").set_value("https://example.org/one.tsv ")
            await user.should_not_see(marker="data-manager-configuration-source", retries=60)
            assert action(1).text == "Inspect"
            assert action(1)._props["aria-label"] == "Inspect public link"
            assert action(2).text == "✓"
            await inspect(1)
            with user.client:
                control("source-url-1").set_value("https://example.org/bad.tsv")
            await user.should_not_see(marker="data-manager-configuration-source", retries=60)
            user.find(marker="action-inspect-data-link-1").click()
            await user.should_see("The remote file could not be inspected.", retries=60)
            assert action(1).text == "Inspect"
            with user.client:
                control("source-url-2").set_value("")
            await user.should_see("Add a source to reveal only the controls that apply to it.", retries=60)
            assert action(2).text == "Inspect"
    asyncio.run(scenario())
