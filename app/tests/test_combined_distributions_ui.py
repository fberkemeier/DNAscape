"""Plotting layout switches retain combined distributions and axis controls."""
import asyncio
from typing import Any

import numpy as np
import pytest
from nicegui.elements.upload import Upload
from nicegui.testing import user_simulation

from app.main import index


@pytest.mark.parametrize("view", ["Histogram", "Boxplot"])
def test_combined_distributions_kde_and_independent_axes(view: str) -> None:
    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-plotting").click()
            upload = next(iter(user.find(marker="plotting-data-upload").elements))
            await upload.handle_uploads([
                Upload.SmallFileUpload("first.csv", "text/csv", b"position,value\n0,1\n1,2\n2,3\n3,4\n"),
                Upload.SmallFileUpload("second.csv", "text/csv", b"position,value\n0,1000\n1,2000\n2,3000\n3,4000\n"),
            ])
            await user.should_see("second.csv", retries=60)
            user.find(marker="action-plot").click()
            await user.should_see(marker="plotting-result-plot", retries=60)
            user.find("Plot settings").click()

            def control(marker: str) -> Any:
                return next(iter(user.find(marker=f"plotting-{marker}").elements))

            def chart() -> dict:
                return control("result-plot").options

            assert {"Histogram", "Boxplot"} <= set(control("view").options)
            assert user.client is not None
            with user.client:
                control("view").set_value(view)
            await asyncio.sleep(0)
            assert control("presentation").value == "Combined"
            expected_type = "custom" if view == "Histogram" else "boxplot"
            assert [item["type"] for item in chart()["series"]] == [expected_type, expected_type]
            if view == "Boxplot":
                assert [item["data"][0][0] for item in chart()["series"]] == [1, 2]
                await user.should_see(marker="plotting-x-axis-options")
            else:
                assert [sum(row[2] for row in item["data"]) for item in chart()["series"]] == [4, 4]
            user.find(marker="plotting-dual-y-axes").click()
            await asyncio.sleep(0)
            assert len(chart()["xAxis"]) == len(chart()["yAxis"]) == 2
            assert [item["yAxisIndex"] for item in chart()["series"]] == [0, 1]
            assert control("x2-minimum").value == pytest.approx(chart()["xAxis"][1]["min"])
            assert control("x2-maximum").value == pytest.approx(chart()["xAxis"][1]["max"])
            if view == "Histogram":
                assert [len(item["data"]) for item in chart()["series"]] == [control("histogram-bins").value] * 2
                assert [sum(row[2] for row in item["data"]) for item in chart()["series"]] == [4, 4]
            with user.client:
                control("log-x2").set_value(True)
                control("log-y2").set_value(True)
                control("invert-x2").set_value(True)
                control("invert-y2").set_value(True)
            await asyncio.sleep(0)
            assert [axis["type"] for axis in chart()["xAxis"]] == ["value", "log"]
            assert [axis["type"] for axis in chart()["yAxis"]] == ["value", "log"]
            assert [axis["inverse"] for axis in chart()["xAxis"]] == [False, True]
            assert [axis["inverse"] for axis in chart()["yAxis"]] == [False, True]
            if view == "Histogram":
                await user.should_see("KDE")
                with user.client:
                    control("density").set_value(True)
                await asyncio.sleep(0)
                assert [item["type"] for item in chart()["series"]] == ["line", "line"]
                assert all(len(item["data"]) == 256 for item in chart()["series"])
                kde_x = np.asarray(chart()["series"][1]["data"])[:, 0]
                assert kde_x.max() > 4000
                assert control("x2-maximum").value == pytest.approx(chart()["xAxis"][1]["max"])
                await user.should_not_see(marker="plotting-histogram-bins")
                await user.should_not_see(marker="plotting-curve-type")
            user.find(marker="plotting-dual-y-axes").click()
            await asyncio.sleep(0)
            assert isinstance(chart()["xAxis"], dict) and isinstance(chart()["yAxis"], dict)
            assert len(chart()["series"]) == 2
            await user.should_not_see(marker="plotting-log-x2")

    asyncio.run(scenario())
