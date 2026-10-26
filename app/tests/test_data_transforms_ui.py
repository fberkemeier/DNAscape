"""Data Manager formulas keep per-upload state and combine before Filter."""

import asyncio

import numpy as np
from nicegui.elements.switch import Switch
from nicegui.elements.upload import Upload
from nicegui.testing import user_simulation

from app.main import index
from app.pages import data_management as page


def test_data_transforms_per_source_combined_output_and_filter_context(monkeypatch):
    results, messages = [], []
    original, combined = page.convert_data, page.convert_combined_data
    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        results.append(result)
        return result
    def capture_combined(*args, **kwargs):
        result = combined(*args, **kwargs)
        results.append(result)
        return result
    monkeypatch.setattr(page, "convert_data", capture)
    monkeypatch.setattr(page, "convert_combined_data", capture_combined)
    monkeypatch.setattr(page, "notify", lambda message, **_kwargs: messages.append(message))

    async def scenario():
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-data-management").click()
            def control(name):
                return next(item for item in user.client.elements.values()
                            if f"data-manager-{name}" in item._markers and not item.is_deleted)
            async def convert(count):
                user.find(marker="action-convert").click()
                for _ in range(160):
                    await asyncio.sleep(.02)
                    if len(results) >= count and control("transform-enabled").enabled:
                        break
                assert len(results) == count, messages
            assert control("transform-formula-1").label == "Source transform formula"
            assert control("transform-formula-1").value == "source"
            assert not control("transform-combine").visible
            assert not control("transform").value
            user.find(marker="data-manager-transform").click()
            with user.client:
                control("transform-formula-1").set_value("source * 2")
            await control("upload").handle_uploads([Upload.SmallFileUpload("first.tsv", "text/plain", b"signal\n1\n2\n3\n")])
            await user.should_see("Array size: 3", retries=60)
            await user.should_see(marker="data-manager-transform-formula-1", retries=60)
            assert control("transform-formula-1").label == "Source transform formula"
            assert isinstance(control("transform-combine"), Switch)
            assert not control("transform-enabled").value
            assert not control("transform-combine").visible
            assert control("transform-formula-1").value == "source * 2"
            await convert(1)
            np.testing.assert_equal(results[-1].preview.values, [1, 2, 3])
            user.find(marker="data-manager-transform-enabled").click()
            await convert(2)
            np.testing.assert_equal(results[-1].preview.values, [2, 4, 6])
            await control("upload").handle_uploads([Upload.SmallFileUpload("second.tsv", "text/plain", b"signal\n3\n1\n2\n")])
            await user.should_see(marker="data-manager-transform-formula-2", retries=60)
            assert control("transform-formula-1").label == "Source 1 transform formula"
            assert control("transform-formula-2").label == "Source 2 transform formula"
            assert control("transform-combine").visible
            assert control("transform-formula-1").value == "source * 2"
            with user.client:
                control("transform-formula-2").set_value("source + 1")
            await convert(4)
            np.testing.assert_equal(results[-2].preview.values, [2, 4, 6])
            np.testing.assert_equal(results[-1].preview.values, [4, 2, 3])
            user.find(marker="data-manager-transform-combine").click()
            await user.should_see(marker="data-manager-transform-combined-formula")
            assert control("transform-combined-formula").label == "Combined transform formula"
            assert not any(
                "dn-transform-source-name" in item._classes and not item.is_deleted
                for item in user.client.elements.values()
            )
            with user.client:
                control("transform-combined-formula").set_value("source1 - source2")
            user.find(marker="data-manager-filter").click()
            user.find(marker="data-manager-filter-enabled").click()
            with user.client:
                control("filter-action").set_value("modulate")
            await control("filter-modulation-upload").handle_uploads([
                Upload.SmallFileUpload("profile.tsv", "text/plain", b"signal\n10\n20\n30\n")
            ])
            await user.should_see("profile.tsv", retries=60)
            assert control("filter-modulation-expression").value == "source * profile"
            await convert(5)
            await asyncio.sleep(.05)
            np.testing.assert_equal(results[-1].preview.values, [-20, 1, 1])
            files = [item for item in user.client.elements.values()
                     if any(mark.startswith("data-manager-result-") and mark.removeprefix("data-manager-result-").isdigit()
                            for mark in item._markers) and not item.is_deleted]
            assert len(files) == 1, [item._markers for item in files]
            retained = control("preview-chart")
            with user.client:
                control("transform-combined-formula").set_value("source1[1:]")
            await convert(5)
            assert any("source shape" in message for message in messages), messages
            assert control("preview-chart") is retained
            user.find(marker="data-manager-transform-combine").click()
            await user.should_see(marker="data-manager-transform-formula-2")
            assert control("transform-formula-1").value == "source * 2"
            assert control("transform-formula-2").value == "source + 1"
    asyncio.run(scenario())
