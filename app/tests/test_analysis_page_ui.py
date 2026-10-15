"""Analysis matrices, report downloads, and per-dataset smoothing in the UI."""

import asyncio
import csv
import io
from typing import Any

import numpy as np
from nicegui.elements.upload import Upload
from nicegui.testing import user_simulation
from scipy.stats import pearsonr

from app.main import index
from app.pages import plotting as page


def test_analysis_switch_smoothing_matrices_and_report_downloads(monkeypatch) -> None:
    artifacts = []
    monkeypatch.setattr(page, "cache_artifact", lambda artifact, **kwargs: artifacts.append(artifact))

    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            with user.scope(marker="nav-plotting"):
                await user.should_see("Analysis")
            user.find(marker="nav-plotting").click()
            await user.should_see("Explore DNAscape data")

            def control(marker: str) -> Any:
                return next(iter(user.find(marker=marker).elements))

            assert control("analysis-enabled").value is False
            assert control("analysis-action-label").text == "Plot"
            await user.should_not_see("Analysis report")
            upload = control("plotting-data-upload")
            await upload.handle_uploads([
                Upload.SmallFileUpload("a.csv", "text/csv", b"position,A\n0,1\n1,3\n2,2\n3,5\n4,4\n"),
                Upload.SmallFileUpload("b.csv", "text/csv", b"position,B\n0,2\n1,6\n2,4\n3,10\n4,8\n"),
                Upload.SmallFileUpload("c.csv", "text/csv", b"position,C\n0,5\n1,3\n2,4\n3,1\n4,2\n"),
            ])
            await user.should_see(marker="analysis-smoothing-method-3", retries=60)
            assert all(control(f"analysis-smoothing-method-{index}").value == "none" for index in (1, 2, 3))
            user.find(marker="action-plot").click()
            await user.should_see(marker="plotting-result-plot", retries=60)
            await user.should_not_see("Analysis report")
            assert user.client is not None
            with user.client:
                control("analysis-enabled").set_value(True)
            assert control("analysis-action-label").text == "Analyse"
            user.find(marker="action-analyse").click()
            await user.should_see(marker="analysis-table-kendall", retries=60)
            for method in ("spearman", "pearson", "kendall"):
                table = control(f"analysis-table-{method}")
                assert [column["label"] for column in table.columns] == ["Dataset", "A", "B", "C"]
                assert [row["dataset"] for row in table.rows] == ["A", "B", "C"]
                assert table.rows[0]["d1"] == "1.000" and table.rows[0]["d2"] == "-1.000"
            assert control("analysis-report-format").value == "csv"
            assert control("analysis-download-report").id > control("analysis-table-kendall").id
            for method in ("spearman", "pearson", "kendall"):
                await user.should_not_see(marker=f"analysis-download-{method}")
            user.find(marker="analysis-download-report").click()
            await asyncio.sleep(0)
            rows = list(csv.reader(io.StringIO(artifacts[-1].data.decode())))
            assert rows[0] == ["Correlation", "Statistic", "Dataset", "A", "B", "C"]
            assert len(rows) == 19 and artifacts[-1].filename == "analysis-report.csv"
            assert rows[4] == ["Spearman", "Paired samples", "A", "5", "5", "5"]
            with user.client:
                control("analysis-report-format").set_value("tsv")
            user.find(marker="analysis-download-report").click()
            await asyncio.sleep(0)
            rows = list(csv.reader(io.StringIO(artifacts[-1].data.decode()), delimiter="\t"))
            assert rows[0] == ["Correlation", "Statistic", "Dataset", "A", "B", "C"]
            assert len(rows) == 19 and artifacts[-1].filename == "analysis-report.tsv"
            with user.client:
                control("analysis-kendall").set_value(False)
                control("analysis-smoothing-method-1").set_value("moving_average")
                control("analysis-smoothing-window-1").set_value(3)
            await user.should_not_see(marker="analysis-table-pearson")
            assert control("analysis-smoothing-method-2").value == "none"
            user.find(marker="action-analyse").click()
            await user.should_see(marker="analysis-table-pearson", retries=60)
            await user.should_not_see(marker="analysis-table-kendall")
            smoothed = np.array([2, 2, 10 / 3, 11 / 3, 4.5])
            chart = control("plotting-result-plot").options
            np.testing.assert_allclose(np.asarray(chart["series"][0]["data"])[:, 1], smoothed)
            np.testing.assert_allclose(np.asarray(chart["series"][1]["data"])[:, 1], [2, 6, 4, 10, 8])
            expected = pearsonr(smoothed, np.array([2, 6, 4, 10, 8])).statistic
            assert control("analysis-table-pearson").rows[0]["d1"] == f"{expected:.3f}"
            with user.client:
                control("analysis-enabled").set_value(False)
            assert control("analysis-action-label").text == "Plot"
            await user.should_not_see("Analysis report")
            await user.should_not_see(marker="analysis-table-pearson")
            previous_chart_id = control("plotting-result-plot").id
            user.find(marker="action-plot").click()
            for _ in range(60):
                if control("plotting-result-plot").id != previous_chart_id and control("analysis-action-label").text == "Plot":
                    break
                await asyncio.sleep(0.02)
            assert control("plotting-result-plot").id != previous_chart_id
            assert control("analysis-action-label").text == "Plot"
            np.testing.assert_allclose(np.asarray(control("plotting-result-plot").options["series"][0]["data"])[:, 1], smoothed)
            await user.should_not_see("Analysis report")

    asyncio.run(scenario())


def test_analysis_can_stop_and_clear_navigation_activity(monkeypatch) -> None:
    import threading
    import time

    from app.page_activity import current_page_activity

    started = threading.Event()

    def cancellable_prepare(_sources, _smoothing, _methods, cancel):
        started.set()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            cancel()
            time.sleep(0.005)
        raise TimeoutError("Cancellation was not delivered")

    monkeypatch.setattr(page, "_prepare_analysis_profiles", cancellable_prepare)

    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-plotting").click()
            upload = next(iter(user.find(marker="plotting-data-upload").elements))
            await upload.handle_uploads([Upload.SmallFileUpload("data.csv", "text/csv", b"value\n1\n2\n3\n")])
            await user.should_see(marker="analysis-smoothing-method-1", retries=60)
            enabled = next(iter(user.find(marker="analysis-enabled").elements))
            assert user.client is not None
            with user.client:
                enabled.set_value(True)
            user.find(marker="action-analyse").click()
            assert await asyncio.to_thread(started.wait, 2)
            label = next(iter(user.find(marker="analysis-action-label").elements))
            assert label.text == "Stop" and enabled.enabled is False
            with user.client:
                assert current_page_activity().indicator("plotting") == "running"
            user.find(marker="action-analyse").click()
            for _ in range(60):
                if label.text == "Analyse":
                    break
                await asyncio.sleep(0.02)
            assert label.text == "Analyse" and enabled.enabled is True
            with user.client:
                assert current_page_activity().indicator("plotting") is None
            await user.should_not_see(marker="analysis-table-pearson")

    asyncio.run(scenario())
