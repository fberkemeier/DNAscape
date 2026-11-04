"""A time series is one kinetic input, not hundreds of independent fits."""

import asyncio

import numpy as np
import pytest
from nicegui.elements.upload import Upload
from nicegui.testing import user_simulation

from app.main import index
from app.mapping_service import MappingResult


@pytest.mark.parametrize("timing_matrix", [False, True])
def test_spacetime_speed_upload_runs_one_fit(monkeypatch, timing_matrix):
    from app.pages import mapping as page

    requests = []
    n, nt = 8, 212
    initiation = 1e-5 * (1 + np.arange(n)[:, None] + np.arange(nt)[None, :] / nt)
    timing = 20 + np.arange(n, dtype=float)

    def fake_run(request, report, cancel):
        requests.append(request)
        values = np.ones((n, 2))
        report(1., "Complete")
        return MappingResult(
            target=request.route.target, target_key="fork_speed", target_label="Fork speed",
            method=request.route.method, output_form="xt", output_dependence="spacetime",
            primary_output="fork_speed", outputs={"fork_speed": values},
            output_forms={"fork_speed": "xt"}, x_axis=np.arange(n, dtype=float),
            time_axis=np.array([0., 1.]), region_labels=(), elapsed_seconds=.01,
            assumptions={},
        )

    monkeypatch.setattr(page, "run_mapping", fake_run)

    async def scenario():
        async with user_simulation(index) as user:
            def control(marker):
                return next(element for element in user.client.elements.values()
                            if marker in element._markers and not element.is_deleted)

            async def select(marker, value):
                with user.client:
                    control(marker).set_value(value)
                await asyncio.sleep(0)
                await asyncio.sleep(0)

            async def upload(key, content):
                await user.should_see(marker=f"source-{key}-upload", retries=60)
                uploader = control(f"source-{key}-upload")
                await uploader.handle_uploads([
                    Upload.SmallFileUpload(f"{key}.csv", "text/csv", content.encode())
                ])
                await user.should_see(marker=f"mapping-input-{key}-detected-data", retries=60)
                await asyncio.sleep(0)
                await asyncio.sleep(0)

            await user.open("/")
            user.find(marker="nav-mapping").click()
            await user.should_see(marker="mapping-target", retries=60)
            await select("mapping-target", "fork_speed")
            await select("mapping-target-form", "spacetime")
            await select("mapping-input-initiation-rate-type", "profile")
            matrix_csv = "position," + ",".join(f"t{i}" for i in range(nt)) + "\n"
            matrix_csv += "\n".join(
                str(i) + "," + ",".join(map(str, row)) for i, row in enumerate(initiation)
            ) + "\n"
            await upload("initiation-rate", matrix_csv)
            assert control("mapping-input-initiation-rate-form").value == "xt"
            assert control("mapping-input-initiation-rate-value-column").value == "__all__"

            if timing_matrix:
                # A table uploaded into the timing-only input must not start
                # 212 fits just because it has 212 numeric columns.
                timing_csv = ",".join(f"t{i}" for i in range(nt)) + "\n"
                timing_csv += "\n".join(
                    ",".join(str(value + j) for j in range(nt)) for value in timing
                ) + "\n"
            else:
                timing_csv = "timing\n" + "\n".join(map(str, timing)) + "\n"
            await upload("replication-timing", timing_csv)
            if timing_matrix:
                assert control("mapping-input-replication-timing-value-column").value == "t0"
            user.find(marker="action-map").click()
            await user.should_see(marker="mapping-result-plot", retries=100)
            assert len(requests) == 1
            np.testing.assert_allclose(requests[0].inputs["initiation_rate"], initiation)
            np.testing.assert_allclose(requests[0].inputs["replication_timing"], timing)

    asyncio.run(scenario())
