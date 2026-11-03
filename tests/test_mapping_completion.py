"""Completed XT mappings remain available while their preview is prepared."""

import asyncio

import numpy as np
import pytest
from nicegui.elements.upload import Upload
from nicegui.testing import user_simulation

from app.main import index
from app.mapping_service import MappingResult
from app.pages import mapping as page
from dnascape.workflows import preview


@pytest.mark.parametrize("hosted", [False, True])
def test_xt_heatmap_finishes_without_materializing_hidden_histogram(monkeypatch, hosted):
    if hosted:
        monkeypatch.setenv("RENDER", "true")
    else:
        monkeypatch.delenv("RENDER", raising=False)
    notifications = []
    monkeypatch.setattr(page, "notify", lambda text, **kwargs: notifications.append(text))
    original_domain = preview.histogram_domain

    def domain(values, **kwargs):
        assert np.asarray(values).ndim != 2, "Hidden histogram scanned the entire XT target"
        return original_domain(values, **kwargs)

    monkeypatch.setattr(preview, "histogram_domain", domain)

    def mapped(request, report, cancel):
        cancel()
        report(.98, "Preparing mapped preview...")
        n, nt = 12, 41
        return MappingResult(
            target="initiation_rate", target_key="initiation_rate", target_label="Initiation rate",
            method=request.route.method, output_form="xt", output_dependence="spacetime",
            primary_output="initiation_rate",
            outputs={"initiation_rate": np.linspace(.001, .003, n)[:, None] * np.linspace(1., 2., nt)},
            output_forms={"initiation_rate": "xt"}, x_axis=np.arange(n), time_axis=np.arange(nt)*5.,
            region_labels=(), elapsed_seconds=.1, assumptions=request.settings,
        )

    monkeypatch.setattr(page, "run_mapping", mapped)

    async def scenario():
        async with user_simulation(index) as user:
            def element(marker):
                return next(e for e in user.client.elements.values()
                            if marker in e._markers and not e.is_deleted)

            async def value(marker, selected):
                with user.client:
                    element(marker).set_value(selected)
                for _ in range(4):
                    await asyncio.sleep(0)

            await user.open("/")
            user.find(marker="nav-mapping").click()
            await value("mapping-target-form", "spacetime")
            await value("mapping-route", "initiation_rate--spacetime--replication_timing-and-fork_speed")
            await element("source-replication-timing-upload").handle_uploads([
                Upload.SmallFileUpload("timing.csv", "text/csv", b"timing\n" + b"100\n"*12)])
            await user.should_see(marker="mapping-input-replication-timing-detected-data", retries=60)
            await value("mapping-region-start", 0.)
            await value("mapping-region-end", 12.)
            user.find(marker="action-map").click()
            for _ in range(200):
                if notifications or element("mapping-status").text.startswith("Complete"):
                    break
                await asyncio.sleep(.01)
            assert not notifications, notifications
            await user.should_see(marker="mapping-result-plot", retries=100)
            for _ in range(100):
                if element("action-map").text == "Map":
                    break
                await asyncio.sleep(.01)
            assert element("mapping-status").text.startswith("Complete")
            assert element("action-map").text == "Map"
            assert element("mapping-result-plot").options["dnascapePlotType"] == "Heatmap"
            assert not notifications

    asyncio.run(scenario())
