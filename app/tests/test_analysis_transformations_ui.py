"""Analysis transformations are independent of correlations and transactional."""

import asyncio
import io

import numpy as np
from nicegui.elements.upload import Upload
from nicegui.testing import user_simulation

from app.main import index
from app.pages import plotting as page


def test_transformations_interpolate_combine_keep_originals_export_and_preserve_last_valid_plot(monkeypatch):
    messages, downloads = [], []
    monkeypatch.setattr(page, "notify", lambda message, **_kwargs: messages.append(message))
    async def scenario():
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-plotting").click()
            monkeypatch.setattr(type(user.download), "__call__", lambda _self, data, filename=None, media_type=None: downloads.append((data, filename)))
            def control(marker):
                return next(item for item in user.client.elements.values() if marker in item._markers and not item.is_deleted)
            def plotted():
                return [np.asarray(item["data"], dtype=float)[:, 1] for item in control("plotting-result-plot").options["series"]]
            async def commit(expected):
                user.find(marker="action-plot").click()
                await user.should_see(marker="plotting-result-plot", retries=80)
                for _ in range(100):
                    await asyncio.sleep(.02)
                    if control("analysis-action-label").text in {"Plot", "Analyse"}:
                        break
                assert len(plotted()) == len(expected), messages
                for actual, values in zip(plotted(), expected):
                    np.testing.assert_equal(actual, values)
            assert control("analysis-transform").value is False
            assert control("analysis-transform-enabled").value is False
            await control("plotting-data-upload").handle_uploads([
                Upload.SmallFileUpload("a.csv", "text/csv", b"A\n1\nnan\n3\n"),
                Upload.SmallFileUpload("b.csv", "text/csv", b"B\n10\n20\n30\n"),
            ])
            await user.should_see(marker="plotting-detected-data-2", retries=80)
            user.find(marker="analysis-transform").click()
            assert control("analysis-transform-selection").value == "threshold"
            assert control("analysis-transform-action").value == "replace"
            with user.client:
                control("analysis-transform-selection").set_value("nan")
                control("analysis-transform-action").set_value("interpolate")
            with user.client:
                control("analysis-transform-target").set_value("1:A")
                control("analysis-transform-enabled").set_value(True)
            await commit([[1, 2, 3], [10, 20, 30]])
            assert not control("analysis-enabled").value
            await user.should_see(marker="analysis-transform-result-1")
            with user.client:
                control("analysis-enabled").set_value(True)
                control("analysis-transform-keep-originals").set_value(True)
            await commit([[1, np.nan, 3], [1, 2, 3], [10, 20, 30]])
            await user.should_see(marker="analysis-report-table")
            with user.client:
                control("analysis-transform-format-1").set_value("NPZ")
            user.find(marker="analysis-transform-download-1").click()
            for _ in range(100):
                await asyncio.sleep(.02)
                if downloads:
                    break
            assert downloads
            with np.load(io.BytesIO(downloads[-1][0]), allow_pickle=False) as archive:
                np.testing.assert_equal(archive["values"], [1, 2, 3])
            await control("plotting-data-upload").handle_uploads([
                Upload.SmallFileUpload("a.csv", "text/csv", b"A\n1\n2\n3\n"),
            ])
            with user.client:
                control("analysis-transform-selection").set_value("all")
                control("analysis-transform-action").set_value("modulate")
            await control("analysis-transform-modulation-upload").handle_uploads([
                Upload.SmallFileUpload("extra.tsv", "text/plain", b"assay\n2\n2\n2\n"),
            ])
            await user.should_see("Profile shape: 3", retries=80)
            await commit([[1, 2, 3], [2, 4, 6], [10, 20, 30]])
            await control("analysis-transform-modulation-upload").handle_uploads([
                Upload.SmallFileUpload("short.tsv", "text/plain", b"assay\n2\n2\n"),
            ])
            await user.should_see("Profile shape: 2", retries=80)
            await commit([[1, 2, 3], [2, 4, 3], [10, 20, 30]])
            with user.client:
                control("analysis-transform-modulation-expression").set_value("source[:-1]")
            user.find(marker="action-plot").click()
            for _ in range(100):
                await asyncio.sleep(.02)
                if any("source shape" in message for message in messages) and control("analysis-action-label").text == "Analyse":
                    break
            assert any("source shape" in message for message in messages)
            np.testing.assert_equal(plotted()[1], [2, 4, 3])
            with user.client:
                control("analysis-transform-enabled").set_value(False)
            await commit([[1, 2, 3], [10, 20, 30]])
            await user.should_not_see(marker="analysis-transform-result-1")
    asyncio.run(scenario())
