"""Upload and inspected links share region settings for reusable DNAscape arrays."""

import asyncio
import io

import numpy as np
import pytest
from nicegui.elements.upload import Upload
from nicegui.testing import user_simulation

from app import preview_plotting
from app.main import index
from app.pages import data_management as page
from app.remote_sources import RemoteSource


@pytest.mark.parametrize("mode", ["upload", "url"])
@pytest.mark.parametrize("layout", ["npz", "rows"])
def test_native_chromosome_selection_region_binning_and_configuration_bypass(monkeypatch, mode, layout):
    buffer = io.BytesIO()
    np.savez(buffer, chr1=np.arange(6.), chr2=np.arange(1., 7.))
    payload = buffer.getvalue()
    filename = "native.npz"
    if layout == "rows":
        payload = ("chromosome,position_bp,signal\n" + "\n".join(
            f"chr{chromosome},{index * 1000},{index + chromosome - 1}"
            for chromosome in (1, 2) for index in range(6))).encode()
        filename = "native.csv"
    monkeypatch.setattr(page, "fetch_remote_source", lambda url: RemoteSource(
        requested_url=url, final_url=url, filename=filename, content=payload,
        content_type="application/octet-stream", etag=None, last_modified=None, accept_ranges=False, redirects=()))
    results, messages = [], []
    original = page.convert_data
    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        results.append(result)
        return result
    monkeypatch.setattr(page, "convert_data", capture)
    monkeypatch.setattr(page, "notify", lambda text, **kwargs: messages.append((text, kwargs.get("type"))))

    async def scenario():
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-data-management").click()
            def control(name):
                return next(item for item in user.client.elements.values()
                            if f"data-manager-{name}" in item._markers and not item.is_deleted)
            async def wait(predicate):
                for _ in range(200):
                    if predicate():
                        return
                    await asyncio.sleep(.01)
                raise AssertionError(messages)
            if mode == "upload":
                await control("upload").handle_uploads([Upload.SmallFileUpload(filename, "application/octet-stream", payload)])
            else:
                with user.client:
                    control("source-tabs").set_value("url")
                await user.should_see(marker="data-manager-source-url-1")
                with user.client:
                    control("source-url-1").set_value(f"https://example.org/{filename}")
                user.find(marker="action-inspect-data-link-1").click()
            marker = "data-manager-source-selection" if layout == "npz" else "data-manager-chromosome"
            await wait(lambda: any(marker in item._markers and not item.is_deleted for item in user.client.elements.values()))
            with user.client:
                select = control("source-selection" if layout == "npz" else "chromosome")
                old_start = control("start")
                select.set_value(next(key for key, label in select.options.items() if label == "chr2") if layout == "npz" else "chr2")
            await wait(lambda: old_start.is_deleted)
            assert control("start").value == 0 and control("end").value == 6000
            assert control("bin-size").value == 1000
            with user.client:
                control("start").set_value(1000)
                control("end").set_value(5000)
                control("bin-size").set_value(2000)
            user.find(marker="action-convert").click()
            await wait(lambda: len(results) == 1 and control("apply-configuration").enabled)
            np.testing.assert_equal(results[-1].preview.values, [2.5, 4.5])
            with user.client:
                control("apply-configuration").set_value(False)
            user.find(marker="action-convert").click()
            await wait(lambda: len(results) == 2 and control("apply-configuration").enabled)
            np.testing.assert_equal(results[-1].preview.values, np.arange(1, 7) if layout == "npz" else np.r_[np.arange(6), np.arange(1, 7)])
            assert not any(kind == "negative" for _text, kind in messages), messages
    asyncio.run(scenario())


def test_different_source_lengths_keep_their_full_region_by_default(monkeypatch):
    results, messages = [], []
    original = page.convert_data
    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        results.append(result)
        return result
    monkeypatch.setattr(page, "convert_data", capture)
    monkeypatch.setattr(page, "notify", lambda text, **kwargs: messages.append((text, kwargs.get("type"))))
    async def scenario():
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-data-management").click()
            upload = next(iter(user.find(marker="data-manager-upload").elements))
            await upload.handle_uploads([
                Upload.SmallFileUpload("short.tsv", "text/plain", b"signal\n1\n2\n"),
                Upload.SmallFileUpload("long.tsv", "text/plain", b"signal\n1\n2\n3\n4\n5\n6\n"),
            ])
            await user.should_see("Array size: 6", retries=80)
            user.find(marker="action-convert").click()
            for _ in range(200):
                await asyncio.sleep(.02)
                if len(results) == 2:
                    break
            assert [result.shape for result in results] == [(2,), (6,)], messages
    asyncio.run(scenario())


def test_data_manager_log_views_use_native_positive_minima_and_report_invalid_edits(monkeypatch):
    messages = []
    def notice(text, **kwargs):
        messages.append((text, kwargs.get("type")))
    monkeypatch.setattr(page, "notify", notice)
    monkeypatch.setattr(preview_plotting, "notify", notice)
    values = np.ones(5001)
    values[0], values[2345] = 0, .03
    payload = ("signal\n" + "\n".join(map(str, values))).encode()
    async def scenario():
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-data-management").click()
            def control(name):
                return next(item for item in user.client.elements.values()
                            if f"data-manager-{name}" in item._markers and not item.is_deleted)
            await control("upload").handle_uploads([Upload.SmallFileUpload("zeros.tsv", "text/plain", payload)])
            await user.should_see("Array size: 5001", retries=80)
            user.find(marker="action-convert").click()
            await user.should_see(marker="data-manager-preview-chart", retries=150)
            with user.client:
                control("plot-log-y").set_value(True)
                control("plot-log-x").set_value(True)
            assert control("preview-chart").options["yAxis"]["min"] == .03, messages
            assert control("preview-chart").options["xAxis"]["min"] == 1, messages
            assert not any(kind == "negative" for _text, kind in messages), messages
            with user.client:
                control("plot-y-minimum").set_value(0)
            assert messages[-1] == ("Log Y minimum must be greater than zero.", "negative")
            with user.client:
                control("plot-y-minimum").set_value(.03)
                control("plot-x-minimum").set_value(-1)
            assert messages[-1] == ("Log X minimum must be greater than zero.", "negative")
            with user.client:
                control("plot-x-minimum").set_value(1)
                control("plot-log-y").set_value(False)
                control("plot-y-minimum").set_value(0)
                control("plot-log-y").set_value(True)
            assert messages[-1] == ("Log Y minimum must be greater than zero.", "negative")
            with user.client:
                control("plot-y-minimum").set_value(.03)
                control("plot-view").set_value("Histogram")
            assert control("plot-histogram-x-minimum").value == .03
            with user.client:
                control("plot-histogram-x-minimum").set_value(0)
            assert messages[-1] == ("Log X minimum must be greater than zero.", "negative")
    asyncio.run(scenario())
