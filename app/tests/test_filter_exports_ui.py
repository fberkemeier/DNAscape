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

            assert not control("plot-export-filter").value
            assert not control("plot-export-filter").enabled
            assert not control("plot-export-filter").visible
            assert not control("plot-show-filter").visible
            await control("upload").handle_uploads([Upload.SmallFileUpload("signal.tsv", "text/plain", b"signal\n1\n-2\n3\n-4\n")])
            await user.should_see("Array size: 4", retries=60)
            user.find(marker="data-manager-filter").click()
            user.find(marker="data-manager-filter-enabled").click()
            user.find(marker="action-convert").click()
            await user.should_see(marker="data-manager-result-name-1", retries=80)
            assert not control("plot-show-filter").value
            assert control("plot-export-filter").visible
            assert control("plot-show-filter").parent_slot is control("plot-export-filter").parent_slot
            with user.client:
                control("plot-export-filter").set_value(True)
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
                np.testing.assert_equal(archive["positions"], [1, 3])
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
            assert not control("plot-export-filter").enabled
            assert not control("plot-export-filter").visible
            assert not control("plot-show-filter").visible

    asyncio.run(scenario())
