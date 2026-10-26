"""Filter position outputs follow the applied result and raw export controls."""

import asyncio
import gzip
import io
import zipfile

import numpy as np
from nicegui.elements.upload import Upload
from nicegui.testing import user_simulation

from app.main import index
from app.pages import data_management as page


def test_export_filter_is_independent_of_shading_and_downloads_native_positions(monkeypatch):
    downloads = []
    messages = []
    monkeypatch.setattr(page, "notify", lambda message, **_kwargs: messages.append(message))

    async def scenario():
        async with user_simulation(index) as user:
            await user.open("/")
            monkeypatch.setattr(type(user.download), "__call__", lambda _self, data, filename=None, media_type=None: downloads.append((data, filename)))
            user.find(marker="nav-data-management").click()

            def control(name):
                return next(item for item in user.client.elements.values()
                            if f"data-manager-{name}" in item._markers and not item.is_deleted)

            assert not control("filter-export").value
            assert not control("filter-export").enabled
            assert control("filter-export").visible
            assert control("filter-export").parent_slot is control("filter-enabled").parent_slot
            assert not any("data-manager-plot-export-filter" in item._markers
                           for item in user.client.elements.values())
            assert not control("plot-show-filter").visible
            await control("upload").handle_uploads([Upload.SmallFileUpload("signal.tsv", "text/plain", b"signal\n1\n-2\n3\n-4\n")])
            await user.should_see("Array size: 4", retries=60)
            user.find(marker="data-manager-filter").click()
            user.find(marker="data-manager-filter-enabled").click()
            user.find(marker="action-convert").click()
            await user.should_see(marker="data-manager-result-name-1", retries=80)
            assert not control("plot-show-filter").value
            assert control("filter-export").visible
            assert control("filter-export").parent_slot is control("filter-enabled").parent_slot
            with user.client:
                control("filter-export").set_value(True)
            await user.should_see(marker="data-manager-filter-result-1")
            with user.client:
                control("filter-result-type-1").set_value("NPZ")
                control("filter-result-compression-1").set_value(True)
                control("filter-result-name-1").set_value("selected_positions")
            user.find(marker="download-filter-positions-1").click()
            assert next(iter(user.find(marker="download-filter-positions-1").elements)).enabled
            for _ in range(100):
                await asyncio.sleep(.02)
                if downloads:
                    break
            assert downloads, messages
            payload, filename = downloads[-1]
            assert filename == "selected_positions.npz.gz"
            with np.load(io.BytesIO(gzip.decompress(payload)), allow_pickle=False) as archive:
                assert archive.files == ["positions"]
                np.testing.assert_equal(archive["positions"], [1000, 3000])
            user.find(marker="download-all-converted-data").click()
            for _ in range(100):
                await asyncio.sleep(.02)
                if len(downloads) == 2:
                    break
            with zipfile.ZipFile(io.BytesIO(downloads[-1][0])) as archive:
                assert "selected_positions.npz.gz" in archive.namelist()
                assert len(archive.namelist()) == 2
            with user.client:
                control("filter-enabled").set_value(False)
            await user.should_not_see(marker="data-manager-filter-result-1")
            assert not control("filter-export").enabled
            assert control("filter-export").visible
            assert not control("plot-show-filter").visible

    asyncio.run(scenario())


def test_only_selected_file_has_one_filter_export_and_retains_its_layout(monkeypatch):
    downloads = []

    async def scenario():
        async with user_simulation(index) as user:
            await user.open("/")
            monkeypatch.setattr(
                type(user.download), "__call__",
                lambda _self, data, filename=None, media_type=None: downloads.append((data, filename)),
            )
            user.find(marker="nav-data-management").click()

            def control(name):
                return next(item for item in user.client.elements.values()
                            if f"data-manager-{name}" in item._markers and not item.is_deleted)

            def filter_rows():
                return [item for item in user.client.elements.values()
                        if not item.is_deleted and any(
                            marker.startswith("data-manager-filter-result-") and marker.rsplit("-", 1)[-1].isdigit()
                            and marker.removeprefix("data-manager-filter-result-").isdigit()
                            for marker in item._markers)]

            await control("upload").handle_uploads([
                Upload.SmallFileUpload("one.tsv", "text/plain", b"signal\n1\n-2\n-3\n4\n"),
                Upload.SmallFileUpload("two.tsv", "text/plain", b"signal\n-1\n2\n3\n-4\n"),
            ])
            await user.should_see(marker="data-manager-upload-source-label-2", retries=60)
            user.find(marker="data-manager-filter").click()
            user.find(marker="data-manager-filter-enabled").click()
            user.find(marker="action-convert").click()
            await user.should_see(marker="data-manager-result-name-2", retries=80)
            with user.client:
                control("filter-export").set_value(True)
            await user.should_see(marker="data-manager-filter-result-2", retries=60)
            assert len(filter_rows()) == 1
            for result_id in (1, 2, 1):
                user.find(marker=f"data-manager-select-result-{result_id}").click()
                await user.should_see(marker=f"data-manager-filter-result-{result_id}", retries=60)
                assert len(filter_rows()) == 1
            fields = control("filter-result-name-1").parent_slot.parent
            assert fields.default_slot.children == [
                control("filter-result-name-1"), control("filter-result-layout-1"), control("filter-result-type-1"),
            ]
            assert control("filter-result-row-1").parent_slot.parent.parent_slot.parent.parent_slot.parent is control("result-1")
            with user.client:
                control("filter-result-name-1").set_value("first_selection")
                control("filter-result-layout-1").set_value("intervals")
                control("filter-result-type-1").set_value("CSV")
            user.find(marker="download-filter-positions-1").click()
            for _ in range(100):
                await asyncio.sleep(.02)
                if downloads:
                    break
            assert downloads[-1] == (b"1000,3000\n", "first_selection.csv")
            user.find(marker="data-manager-select-result-2").click()
            await user.should_see(marker="data-manager-filter-result-2", retries=60)
            assert control("filter-result-layout-2").value == "positions"
            user.find(marker="data-manager-select-result-1").click()
            await user.should_see(marker="data-manager-filter-result-1", retries=60)
            assert control("filter-result-layout-1").value == "intervals"
            assert control("filter-result-name-1").value == "first_selection"
            with user.client:
                control("filter-result-type-1").set_value("NPZ")
            user.find(marker="download-filter-positions-1").click()
            for _ in range(100):
                await asyncio.sleep(.02)
                if len(downloads) == 2:
                    break
            with np.load(io.BytesIO(downloads[-1][0]), allow_pickle=False) as archive:
                assert archive.files == ["intervals"]
                np.testing.assert_equal(archive["intervals"], [[1000, 3000]])
            with user.client:
                control("filter-export").set_value(False)
            assert not filter_rows()
            with user.client:
                control("filter-export").set_value(True)
            assert len(filter_rows()) == 1

    asyncio.run(scenario())
