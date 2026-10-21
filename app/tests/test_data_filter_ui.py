"""The optional Filter controls drive the converted preview and raw output."""

import asyncio
import io

import numpy as np
import pandas as pd
from nicegui.elements.upload import Upload
from nicegui.testing import user_simulation

from app.main import index
from app.pages import data_management as page
from app.session_cache import current_session_cache


def test_filter_defaults_visibility_uploaded_mask_and_post_configure_results(monkeypatch):
    requests, results = [], []
    messages = []
    monkeypatch.setattr(page, "notify", lambda message, **_kwargs: messages.append(message))
    original = page.convert_data
    def capture(request, *args, **kwargs):
        result = original(request, *args, **kwargs)
        requests.append(request)
        results.append(result)
        return result
    monkeypatch.setattr(page, "convert_data", capture)

    async def scenario():
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-data-management").click()
            def control(name):
                return next(item for item in user.client.elements.values() if f"data-manager-{name}" in item._markers and not item.is_deleted)
            async def converted(expected):
                count = len(results)
                user.find(marker="action-convert").click()
                for _ in range(150):
                    await asyncio.sleep(.02)
                    if len(results) > count and control("filter-enabled").enabled:
                        break
                assert len(results) == count + 1, messages
                np.testing.assert_equal(results[-1].preview.values, expected)
                np.testing.assert_equal(pd.read_csv(io.BytesIO(results[-1].artifact.data), sep="\t").iloc[:, -1].to_numpy(), expected)

            assert control("filter").value is False
            user.find(marker="data-manager-filter").click()
            assert control("filter-enabled").value is False
            await user.should_not_see(marker="data-manager-filter-selection")
            await control("upload").handle_uploads([
                Upload.SmallFileUpload("signal.csv", "text/csv", b"position,value\n0,-2\n1,1\n2,3\n"),
            ])
            await user.should_see("Array size: 3", retries=60)
            with user.client:
                control("value-scaling").set_value("multiplier")
                control("scale-multiplier").set_value(-2)
            await converted([4, -2, -6])
            assert requests[-1].data_filter.enabled is False
            user.find(marker="data-manager-filter-enabled").click()
            assert control("filter-selection").value == "threshold"
            assert control("filter-comparison").value == "below"
            assert control("filter-threshold").value == 0
            assert control("filter-action").value == "replace"
            assert control("filter-replace-value").value == 0
            await converted([4, 0, 0])
            with user.client:
                control("filter-selection").set_value("between")
                control("filter-lower").set_value(-6)
                control("filter-upper").set_value(-1)
                control("filter-action").set_value("interpolate")
            await user.should_see(marker="data-manager-filter-interpolation")
            await user.should_not_see(marker="data-manager-filter-threshold")
            await user.should_not_see(marker="data-manager-filter-standard-deviation")
            await converted([4, 4, 4])
            with user.client:
                control("filter-selection").set_value("profile")
                control("filter-action").set_value("replace")
                control("filter-replace-value").set_value(7)
            # No uploaded mask is deliberately inert.
            await converted([4, -2, -6])
            with user.client:
                before = current_session_cache().count()
            await control("filter-upload").handle_uploads([
                Upload.SmallFileUpload("mask.tsv", "text/tab-separated-values", b"mask\n0\n1\n0\n"),
            ])
            await user.should_see("mask.tsv", retries=60)
            with user.client:
                assert current_session_cache().count() == before
            await converted([4, 7, -6])
            user.find(marker="data-manager-filter-upload-remove").click()
            await user.should_not_see("mask.tsv")
            await converted([4, -2, -6])
            await control("filter-upload").handle_uploads([
                Upload.SmallFileUpload("positions.tsv", "text/tab-separated-values", b"position_bp\n1000\n"),
            ])
            await user.should_see("positions.tsv", retries=60)
            assert control("filter-profile-layout").value == "positions"
            await user.should_see(marker="data-manager-filter-profile-units")
            await converted([4, 7, -6])
            with user.client:
                control("filter-selection").set_value("all")
                control("filter-action").set_value("nan")
            await converted([np.nan] * 3)
            await user.should_see("No finite values to plot. The raw data remain available below.")
            await user.should_see(marker="data-manager-result-name-1")
            user.find(marker="data-manager-filter-enabled").click()
            await converted([4, -2, -6])
            await user.should_see(marker="data-manager-preview-chart")
    asyncio.run(scenario())


def test_filter_parameters_survive_configure_refresh_and_match_active_controls(monkeypatch):
    requests = []
    original = page.convert_data
    def capture(request, *args, **kwargs):
        requests.append(request)
        return original(request, *args, **kwargs)
    monkeypatch.setattr(page, "convert_data", capture)
    async def scenario():
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-data-management").click()
            def control(name):
                return next(item for item in user.client.elements.values() if f"data-manager-{name}" in item._markers and not item.is_deleted)
            await control("upload").handle_uploads([Upload.SmallFileUpload("source.tsv", "text/plain", b"value\n1\n2\n3\n")])
            await user.should_see("Array size: 3", retries=60)
            user.find(marker="data-manager-filter").click()
            user.find(marker="data-manager-filter-enabled").click()
            with user.client:
                control("filter-selection").set_value("maximum")
                control("filter-comparison").set_value("above")
                control("filter-threshold").set_value(.5)
                control("filter-action").set_value("noise")
                control("filter-standard-deviation").set_value(.2)
                control("filter-seed").set_value(42)
                control("feature").set_value("firing_rate")
            await user.should_see(marker="data-manager-filter-standard-deviation")
            await user.should_not_see(marker="data-manager-filter-replace-value")
            assert control("filter-selection").value == "maximum"
            user.find(marker="action-convert").click()
            for _ in range(150):
                await asyncio.sleep(.02)
                if requests and control("filter-enabled").enabled:
                    break
            config = requests[-1].data_filter
            assert config.enabled and config.selection == "maximum" and config.comparison == "above"
            assert config.threshold == .5 and config.standard_deviation == .2 and config.seed == 42
            assert control("filter-enabled").enabled and control("filter-upload").enabled
    asyncio.run(scenario())
