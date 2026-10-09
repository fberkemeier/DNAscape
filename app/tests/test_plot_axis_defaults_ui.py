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
            user.find("Plotting settings").click()
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


def test_mapping_time_profile_uses_shared_curve_defaults(monkeypatch) -> None:
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
            assert options["series"][0]["type"] == "line"
            assert options["xAxis"]["name"] == "Time (min)"
            assert options["yAxis"]["name"] == "Firing rate"
            assert options["series"][0]["lineStyle"]["color"] == "#b03412"
            user.find("Plotting settings").click()
            await user.should_not_see("Title")
            for endpoint in ("min", "max"):
                suffix = "minimum" if endpoint == "min" else "maximum"
                value = next(iter(user.find(marker=f"mapping-x-{suffix}").elements)).value
                assert value == pytest.approx(options["xAxis"][endpoint])

    asyncio.run(scenario())
