"""Automatic axis fields stay readable and reset across live changes."""
import asyncio
import io
from typing import Any

import numpy as np
import pytest
from nicegui.elements.upload import Upload
from nicegui.testing import user_simulation

from app.main import index
from app.mapping_service import MappingResult
from app.pages import mapping


@pytest.mark.parametrize("form", ("t", "xt"))
def test_displayed_axis_defaults_custom_limits_and_reset(form: str) -> None:
    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-plotting").click()
            values = np.asarray([[0.01, 0.04, 0.08, 0.12]])
            if form == "xt":
                values = np.concatenate((values, values * 2, values * 3))
            buffer = io.BytesIO()
            np.save(buffer, values)
            upload = next(iter(user.find(marker="plotting-data-upload").elements))
            await upload.handle_uploads([Upload.SmallFileUpload("signal.npy", "application/octet-stream", buffer.getvalue())])
            await user.should_see("Array size: " + " x ".join(map(str, values.shape)), retries=60)
            user.find(marker="action-plot").click()
            await user.should_see(marker="plotting-result-plot", retries=60)
            user.find("Plot settings").click()
            await user.should_not_see("Title")

            def control(marker: str) -> Any:
                return next(iter(user.find(marker=f"plotting-{marker}").elements))

            def options() -> dict[str, Any]:
                return control("result-plot").options

            assert control("x-minimum").value == pytest.approx(options()["xAxis"]["min"])
            assert control("x-maximum").value == pytest.approx(options()["xAxis"]["max"])
            if form == "t":
                assert options()["series"][0]["type"] == "line"
            else:
                assert control("z-minimum").value == pytest.approx(options()["visualMap"]["min"])
                assert control("z-maximum").value == pytest.approx(options()["visualMap"]["max"])
            assert user.client is not None
            original_x = (control("x-minimum").value, control("x-maximum").value)
            assert control("aspect-ratio").value == 3.0
            assert control("aspect-ratio")._props["stack-label"] is True
            assert control("x-axis-label")._props["stack-label"] is True
            with user.client:
                control("aspect-ratio").set_value(2.0)
            await asyncio.sleep(0)
            assert options()["dnascapeAspectRatio"] == 2.0
            chart_styles = control("result-plot")._style
            assert chart_styles["width"] == "100%"
            assert chart_styles["aspect-ratio"] == "2.0"
            assert chart_styles["height"] == "auto"
            user.find(marker="plotting-reset-aspect-ratio").click()
            await asyncio.sleep(0)
            assert options()["dnascapeAspectRatio"] == 3.0
            assert control("aspect-ratio")._props["stack-label"] is True
            assert control("result-plot")._style["height"] == "auto"
            with user.client:
                control("x-minimum").set_value(0.25)
                control("x-maximum").set_value(1.5)
            await asyncio.sleep(0)
            assert options()["xAxis"]["min"] == 0.25
            assert options()["xAxis"]["max"] == 1.5
            user.find(marker="plotting-reset-x-limits").click()
            await asyncio.sleep(0)
            assert (control("x-minimum").value, control("x-maximum").value) == original_x
            with user.client:
                control("invert-x").set_value(True)
            await asyncio.sleep(0)
            assert options()["xAxis"]["inverse"] is True
            assert options()["yAxis"]["axisLine"]["onZero"] is False
            if form == "xt":
                with user.client:
                    control("log-z").set_value(True)
                await asyncio.sleep(0)
                assert control("z-minimum").value == pytest.approx(10 ** options()["visualMap"]["min"])
                assert control("z-maximum").value == pytest.approx(10 ** options()["visualMap"]["max"])
                original_z = control("z-maximum").value
                with user.client:
                    control("z-maximum").set_value(1.0)
                await asyncio.sleep(0)
                assert options()["visualMap"]["max"] == pytest.approx(0.0)
                user.find(marker="plotting-reset-z-limits").click()
                await asyncio.sleep(0)
                assert control("z-maximum").value == pytest.approx(original_z)

    asyncio.run(scenario())


def test_mapping_time_profile_uses_position_time_heatmap_defaults(monkeypatch) -> None:
    def mapped(request, report, cancel):
        cancel()
        return MappingResult(
            target="firing_rate", target_key="firing_rate", target_label="Firing rate",
            method=request.route.method, output_form="t", output_dependence="time",
            primary_output="firing_rate", outputs={"firing_rate": np.asarray([[0.01, 0.02, 0.03]])},
            output_forms={"firing_rate": "t"}, x_axis=np.arange(3.0),
            time_axis=np.asarray([0.0, 5.0, 10.0]), region_labels=(),
            elapsed_seconds=0.01, assumptions={},
        )

    monkeypatch.setattr(mapping, "run_mapping", mapped)

    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-mapping").click()
            assert user.client is not None
            with user.client:
                next(iter(user.find(marker="mapping-target-form").elements)).set_value("time")
            await asyncio.sleep(0)
            await user.should_not_see(marker="mapping-target-shape-field")
            upload = next(iter(user.find(marker="source-replication-timing-upload").elements))
            await upload.handle_uploads([Upload.SmallFileUpload("timing.csv", "text/csv", b"timing\n10\n20\n30\n")])
            await user.should_see(marker="mapping-input-replication-timing-detected-data", retries=60)
            user.find(marker="action-map").click()
            await user.should_see(marker="mapping-result-plot", retries=60)
            options = next(iter(user.find(marker="mapping-result-plot").elements)).options
            assert options["series"][0]["type"] == "custom"
            assert options["xAxis"]["name"] == "Position (kb)"
            assert options["yAxis"]["name"] == "Time (min)"
            assert options["visualMap"]["min"] <= 0.01
            assert options["visualMap"]["max"] >= 0.03
            assert np.asarray(options["series"][1]["data"])[:, 2].min() == pytest.approx(0.01)
            assert np.asarray(options["series"][1]["data"])[:, 0].max() == pytest.approx(2.0)
            assert np.asarray(options["series"][1]["data"])[:, 1].max() == pytest.approx(10.0)
            user.find("Plot settings").click()
            await user.should_not_see("Title")
            for endpoint in ("min", "max"):
                suffix = "minimum" if endpoint == "min" else "maximum"
                value = next(iter(user.find(marker=f"mapping-x-{suffix}").elements)).value
                assert value == pytest.approx(options["xAxis"][endpoint])

    asyncio.run(scenario())


def test_two_dataset_plot_has_optional_independent_y_axes_and_curve_patterns() -> None:
    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-plotting").click()
            user.find("Plot settings").click()
            await user.should_not_see(marker="plotting-dual-y-axes")
            upload = next(iter(user.find(marker="plotting-data-upload").elements))
            await upload.handle_uploads([
                Upload.SmallFileUpload("first.csv", "text/csv", b"position,first\n0,1\n1,3\n2,2\n"),
                Upload.SmallFileUpload("second.csv", "text/csv", b"position,second\n0,1000\n1,3000\n2,2000\n"),
            ])
            await user.should_see("second.csv", retries=60)
            user.find(marker="action-plot").click()
            await user.should_see(marker="plotting-result-plot", retries=60)
            def control(marker: str) -> Any:
                return next(iter(user.find(marker=f"plotting-{marker}").elements))

            def options() -> dict[str, Any]:
                return control("result-plot").options

            await user.should_see(marker="plotting-dual-y-axes")
            assert control("dual-y-axes").value is False
            assert isinstance(options()["yAxis"], dict)
            user.find(marker="plotting-dual-y-axes").click()
            await asyncio.sleep(0)
            axes = options()["yAxis"]
            assert [axis["position"] for axis in axes] == ["left", "right"]
            assert [series["yAxisIndex"] for series in options()["series"]] == [0, 1]
            assert axes[0]["max"] < 10
            assert axes[1]["max"] > 3000
            assert control("y-minimum")._props["label"] == "Y1 min"
            assert control("x-minimum")._props["label"] == "X1 min"
            assert [axis["position"] for axis in options()["xAxis"]] == ["bottom", "top"]
            assert control("y-maximum")._props["label"] == "Y1 max"
            await user.should_see(marker="plotting-y2-minimum")
            assert control("y2-maximum").value == pytest.approx(axes[1]["max"])
            await user.should_not_see(marker="plotting-curve-type")
            user.find("Curves").click()
            assert user.client is not None
            with user.client:
                control("y-minimum").set_value(0)
                control("y-maximum").set_value(8)
                control("y2-minimum").set_value(500)
                control("y2-maximum").set_value(4000)
                control("invert-x").set_value(True)
                control("curve-type-1").set_value("dashed")
                control("curve-type-2").set_value("dashed")
            await asyncio.sleep(0)
            assert [(axis["min"], axis["max"]) for axis in options()["yAxis"]] == [(0, 8), (500, 4000)]
            assert options()["xAxis"][0]["inverse"] is True
            assert options()["xAxis"][1]["inverse"] is False
            assert all(axis["axisLine"]["onZero"] is False for axis in options()["yAxis"])
            assert all(series["lineStyle"]["type"] == "dashed" for series in options()["series"])
            assert "default" not in control("curve-type-2").options
            assert control("curve-type-2").value == "dashed"
            with user.client:
                control("curve-type-2").set_value("dotted")
            await asyncio.sleep(0)
            assert [series["lineStyle"]["type"] for series in options()["series"]] == ["dashed", "dotted"]
            user.find(marker="plotting-reset-y2-limits").click()
            await asyncio.sleep(0)
            assert options()["yAxis"][0]["max"] == 8
            assert 3000 < options()["yAxis"][1]["max"] < 4000
            user.find(marker="plotting-reset-y-limits").click()
            await asyncio.sleep(0)
            assert options()["yAxis"][0]["max"] < 8
            with user.client:
                control("log-x2").set_value(True)
                control("log-y2").set_value(True)
                control("invert-y2").set_value(True)
                control("invert-x2").set_value(True)
                control("x2-minimum").set_value(0.5)
                control("x2-maximum").set_value(10)
            await asyncio.sleep(0)
            assert [axis["type"] for axis in options()["xAxis"]] == ["value", "log"]
            assert [axis["type"] for axis in options()["yAxis"]] == ["value", "log"]
            assert [axis["inverse"] for axis in options()["yAxis"]] == [False, True]
            assert [axis["inverse"] for axis in options()["xAxis"]] == [True, True]
            assert options()["xAxis"][1]["max"] == 10
            user.find(marker="plotting-reset-x2-limits").click()
            await asyncio.sleep(0)
            assert options()["xAxis"][1]["max"] < 10
            user.find(marker="plotting-dual-y-axes").click()
            await asyncio.sleep(0)
            assert isinstance(options()["yAxis"], dict)
            assert isinstance(options()["xAxis"], dict)
            assert "yAxisIndex" not in options()["series"][1]
            assert options()["yAxis"]["max"] > 3000
            assert control("y-minimum")._props["label"] == "Y min"
            await user.should_not_see(marker="plotting-y2-minimum")
            await upload.handle_uploads([
                Upload.SmallFileUpload("third.csv", "text/csv", b"position,third\n0,5\n1,7\n2,6\n"),
            ])
            await user.should_see("third.csv", retries=60)
            user.find(marker="action-plot").click()
            for _ in range(60):
                if len(options()["series"]) == 3:
                    break
                await asyncio.sleep(.01)
            assert len(options()["series"]) == 3
            await user.should_not_see(marker="plotting-dual-y-axes")

    asyncio.run(scenario())
