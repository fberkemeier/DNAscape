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


def test_analysis_switch_combined_report_p_values_download_and_clear(monkeypatch) -> None:
    artifacts = []
    messages = []
    monkeypatch.setattr(page, "cache_artifact", lambda artifact, **kwargs: artifacts.append(artifact))
    monkeypatch.setattr(page, "notify", lambda message, **kwargs: messages.append(message))

    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-plotting").click()
            await user.should_see("Explore DNAscape data")

            def control(marker: str) -> Any:
                return next(iter(user.find(marker=marker).elements))

            assert control("analysis-enabled").value is False
            assert control("analysis-action-label").text == "Plot"
            await user.should_not_see("Analysis report")
            for marker in ("analysis-spearman", "analysis-show-p-values", "analysis-smoothing-method-1"):
                await user.should_not_see(marker=marker)
            await control("plotting-data-upload").handle_uploads([
                Upload.SmallFileUpload("a.csv", "text/csv", b"position,A\n0,1\n1,3\n2,2\n3,5\n4,4\n"),
                Upload.SmallFileUpload("b.csv", "text/csv", b"position,B\n0,2\n1,6\n2,4\n3,10\n4,8\n"),
                Upload.SmallFileUpload("c.csv", "text/csv", b"position,C\n0,5\n1,3\n2,4\n3,1\n4,2\n"),
            ])
            await user.should_see(marker="plotting-detected-data-3", retries=60)
            await user.should_not_see(marker="analysis-smoothing-method-1")
            user.find(marker="action-plot").click()
            try:
                await user.should_see(marker="plotting-result-plot", retries=60)
            except AssertionError as exc:
                raise AssertionError(f"Plot failed: {messages}") from exc
            await user.should_not_see("Analysis report")
            assert user.client is not None
            with user.client:
                control("analysis-enabled").set_value(True)
            await user.should_see(marker="analysis-smoothing-method-3", retries=60)
            assert all(control(f"analysis-smoothing-method-{index}").value == "none" for index in (1, 2, 3))
            assert control("analysis-show-p-values").value is False
            assert control("analysis-action-label").text == "Analyse"
            await user.should_not_see("Apply to each dataset when you click Plot or Analyse.")
            await user.should_not_see("Upload data to choose smoothing for each dataset.")
            user.find(marker="action-analyse").click()
            await user.should_see(marker="analysis-report-table", retries=60)
            table = control("analysis-report-table")
            assert len(table.columns) == 10 and len(table.rows) == 3
            assert [column["group"] for column in table.columns if column.get("groupStart")] == ["Spearman", "Pearson", "Kendall tau"]
            assert [row["dataset"] for row in table.rows] == ["A", "B", "C"]
            for method in ("spearman", "pearson", "kendall"):
                assert table.rows[0][f"{method}_1"] == "1.000"
                assert table.rows[0][f"{method}_2"] == "-1.000"
                assert table.rows[0]["counts"][f"{method}_1"] == 5
            assert all(not column["showPValues"] for column in table.columns[1:])
            await user.should_not_see("Matched indices over the shared span; finite pairs only.")
            assert control("analysis-report-format").value == "csv"
            assert control("analysis-report-name").value == "analysis-report"
            assert control("analysis-download-report")._props["icon"] == "download"
            assert control("analysis-clear-preview")._props["icon"] == "close"
            assert control("analysis-report-name").id < control("analysis-report-format").id < control("analysis-download-report").id < control("analysis-clear-preview").id
            assert control("analysis-download-report").id > table.id
            with user.client:
                control("analysis-report-name").set_value("My comparison")
            user.find(marker="analysis-download-report").click()
            await asyncio.sleep(0)
            rows = list(csv.reader(io.StringIO(artifacts[-1].data.decode())))
            assert len(rows) == 4 and len(rows[0]) == 19
            assert rows[0][0] == "Dataset" and artifacts[-1].filename == "My_comparison.csv"
            assert rows[1][rows[0].index("Spearman: B (coefficient)")] == "1"
            assert rows[1][rows[0].index("Spearman: B (paired samples)")] == "5"
            assert not any("(p-value)" in column for column in rows[0])
            with user.client:
                control("analysis-report-format").set_value("tsv")
                control("analysis-show-p-values").set_value(True)
            for _ in range(60):
                if all(column["showPValues"] for column in control("analysis-report-table").columns[1:]):
                    break
                await asyncio.sleep(0.02)
            table = control("analysis-report-table")
            assert all(column["showPValues"] for column in table.columns[1:])
            assert table.rows[0]["p_values"]["pearson_0"] == "—"
            assert control("analysis-report-name").value == "My comparison"
            assert control("analysis-report-format").value == "tsv"
            user.find(marker="analysis-download-report").click()
            await asyncio.sleep(0)
            rows = list(csv.reader(io.StringIO(artifacts[-1].data.decode()), delimiter="\t"))
            assert len(rows) == 4 and len(rows[0]) == 28
            assert artifacts[-1].filename == "My_comparison.tsv"
            np.testing.assert_allclose(float(rows[1][rows[0].index("Pearson: B (p-value)")]),
                                       pearsonr([1, 3, 2, 5, 4], [2, 6, 4, 10, 8]).pvalue, rtol=1e-10, atol=0)
            with user.client:
                control("analysis-kendall").set_value(False)
                control("analysis-smoothing-method-1").set_value("moving_average")
                control("analysis-smoothing-window-1").set_value(3)
            await user.should_not_see(marker="analysis-report-table")
            assert control("analysis-smoothing-method-2").value == "none"
            user.find(marker="action-analyse").click()
            await user.should_see(marker="analysis-report-table", retries=60)
            table = control("analysis-report-table")
            assert len(table.columns) == 7
            smoothed = np.array([2, 2, 10 / 3, 11 / 3, 4.5])
            chart = control("plotting-result-plot").options
            np.testing.assert_allclose(np.asarray(chart["series"][0]["data"])[:, 1], smoothed)
            np.testing.assert_allclose(np.asarray(chart["series"][1]["data"])[:, 1], [2, 6, 4, 10, 8])
            expected = pearsonr(smoothed, np.array([2, 6, 4, 10, 8])).statistic
            assert table.rows[0]["pearson_1"] == f"{expected:.3f}"
            with user.client:
                control("analysis-enabled").set_value(False)
            assert control("analysis-action-label").text == "Plot"
            for marker in ("analysis-report-table", "analysis-spearman", "analysis-smoothing-method-1", "analysis-show-p-values"):
                await user.should_not_see(marker=marker)
            previous_chart_id = control("plotting-result-plot").id
            user.find(marker="action-plot").click()
            for _ in range(60):
                if control("plotting-result-plot").id != previous_chart_id and control("analysis-action-label").text == "Plot":
                    break
                await asyncio.sleep(0.02)
            np.testing.assert_allclose(np.asarray(control("plotting-result-plot").options["series"][0]["data"])[:, 1], [1, 3, 2, 5, 4])
            with user.client:
                control("analysis-enabled").set_value(True)
            assert control("analysis-smoothing-method-1").value == "moving_average"
            user.find(marker="action-analyse").click()
            await user.should_see(marker="analysis-clear-preview", retries=60)
            user.find(marker="analysis-clear-preview").click()
            await user.should_not_see(marker="analysis-report-table")
            await user.should_not_see(marker="plotting-result-plot")
            await user.should_see("No plot yet")
            assert control("analysis-smoothing-method-1").value == "moving_average"

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
            await user.should_not_see(marker="analysis-smoothing-method-1")
            enabled = next(iter(user.find(marker="analysis-enabled").elements))
            assert user.client is not None
            with user.client:
                enabled.set_value(True)
            await user.should_see(marker="analysis-smoothing-method-1", retries=60)
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
            await user.should_not_see(marker="analysis-report-table")

    asyncio.run(scenario())
