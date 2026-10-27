"""Fitting rows configure native solvers while supplied inputs remain fixed."""

import asyncio

import numpy as np
from nicegui.elements.upload import Upload
from nicegui.testing import user_simulation

from app.main import index
from app.pages import mapping as page


def test_fit_rows_keep_supplied_values_fixed_and_apply_bounds_and_guesses(monkeypatch):
    requests, results = [], []
    original = page.run_mapping

    def record(request, *args):
        result = original(request, *args)
        requests.append(request)
        results.append(result)
        return result

    monkeypatch.setattr(page, "run_mapping", record)

    async def scenario():
        async with user_simulation(index) as user:
            def control(marker):
                return next(element for element in user.client.elements.values()
                            if marker in element._markers and not element.is_deleted)

            async def set_value(marker, value):
                with user.client:
                    control(marker).set_value(value)
                await asyncio.sleep(0)

            await user.open("/")
            user.find(marker="nav-mapping").click()
            await user.should_see(marker="mapping-fitting-parameters", retries=60)
            assert control("mapping-fit-initiation-rate-lower").value == "1e-15"
            assert control("mapping-fit-initiation-rate-upper").value is None
            assert control("mapping-fit-initiation-rate-initial").value is None
            await user.should_not_see(marker="mapping-fit-replication-timing")
            await user.should_not_see(marker="mapping-fit-fork-speed")
            await set_value("mapping-input-fork-speed-constant", 2.)
            await control("source-replication-timing-upload").handle_uploads([
                Upload.SmallFileUpload("timing.tsv", "text/plain", b"timing\n20\n20\n20\n20\n20\n20\n20\n20\n"),
            ])
            await user.should_see("Array size: 8 x 1", retries=60)
            await set_value("mapping-fit-initiation-rate-lower", "0.002")
            await set_value("mapping-fit-initiation-rate-upper", "0.004")
            await set_value("mapping-fit-initiation-rate-upper", None)
            await set_value("mapping-fit-initiation-rate-lower", "1e-15")
            assert control("mapping-setting-rate-bounds").value is None
            await set_value("mapping-fit-initiation-rate-lower", "0.002")
            await set_value("mapping-fit-initiation-rate-upper", "0.004")
            await set_value("mapping-fit-initiation-rate-initial", .003)
            await set_value("mapping-setting-maxiter", 1)
            user.find(marker="action-map").click()
            for _ in range(300):
                await asyncio.sleep(.02)
                if results and control("action-map").enabled:
                    break
            assert results
            assert requests[-1].settings["rate_bounds"] == (.002, .004)
            assert requests[-1].settings["initial_initiation_rate"] == .003
            np.testing.assert_array_equal(requests[-1].inputs["replication_timing"], np.full(8, 20.))
            assert np.all(results[-1].outputs["initiation_rate"] >= .002)
            assert np.all(results[-1].outputs["initiation_rate"] <= .004)
            await set_value("mapping-target", "fork_speed")
            await set_value("mapping-target-form", "constant")
            assert control("mapping-fit-fork-speed-lower").value == "0.2"
            assert control("mapping-fit-fork-speed-upper").value == "6"
            assert control("mapping-fit-fork-speed-initial").value == 1.4
            assert control("mapping-fit-fork-speed-initial").enabled
            await user.should_not_see(marker="mapping-fit-initiation-rate")

    asyncio.run(scenario())
