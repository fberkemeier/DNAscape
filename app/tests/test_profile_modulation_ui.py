"""Modulation formulas preserve conversion settings and selected-region semantics."""

import asyncio
import io

import numpy as np
import pandas as pd
from nicegui.elements.upload import Upload
from nicegui.testing import user_simulation

from app.main import index
from app.pages import data_management as page
from app.session_cache import current_session_cache


def test_profile_modulation_formulas_shape_errors_and_existing_output(monkeypatch):
    results, messages, kinds = [], [], []
    original = page.convert_data
    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        results.append(result)
        return result
    monkeypatch.setattr(page, "convert_data", capture)
    def notice(message, **kwargs):
        messages.append(message)
        kinds.append(kwargs.get("type"))
    monkeypatch.setattr(page, "notify", notice)

    async def scenario():
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-data-management").click()
            def control(name):
                return next(item for item in user.client.elements.values()
                            if f"data-manager-{name}" in item._markers and not item.is_deleted)
            async def convert(expected=None, error=None):
                count = len(results)
                messages.clear()
                user.find(marker="action-convert").click()
                for _ in range(180):
                    await asyncio.sleep(.02)
                    if control("filter-enabled").enabled and (len(results) > count or messages):
                        break
                if error:
                    assert any(error in message for message in messages), messages
                    assert len(results) == count
                else:
                    assert len(results) == count + 1, messages
                    np.testing.assert_allclose(results[-1].preview.values, expected, equal_nan=True)
                    table = pd.read_csv(io.BytesIO(results[-1].artifact.data), sep="\t")
                    np.testing.assert_allclose(table.iloc[:, -1], expected, equal_nan=True)
                assert control("filter-enabled").enabled and control("filter-modulation-upload").enabled
            async def assay(payload, name="assay.tsv"):
                with user.client:
                    count = current_session_cache().count()
                await control("filter-modulation-upload").handle_uploads([Upload.SmallFileUpload(name, "text/plain", payload)])
                await user.should_see(name, retries=60)
                for _ in range(120):
                    await asyncio.sleep(.01)
                    if control("filter-modulation-upload").enabled and next(iter(user.find(marker="action-convert").elements)).enabled:
                        break
                with user.client:
                    assert current_session_cache().count() == count
            await control("upload").handle_uploads([Upload.SmallFileUpload("source.tsv", "text/plain", b"signal\n1\n2\n3\n")])
            await user.should_see("Array size: 3", retries=60)
            with user.client:
                control("value-scaling").set_value("multiplier")
                control("scale-multiplier").set_value(2)
            user.find(marker="data-manager-filter").click()
            user.find(marker="data-manager-filter-enabled").click()
            with user.client:
                control("filter-selection").set_value("all")
                control("filter-action").set_value("modulate")
            await user.should_see(marker="data-manager-filter-modulation-upload")
            await user.should_see("Arrays align from the first sample; only their overlap is combined.")
            assert control("filter-modulation-expression").value == "source * profile"
            await user.should_not_see("Operator")
            await convert(error="Upload a modulation profile")
            await assay(b"atac\n1\n3\n2\n")
            await convert([2, 12, 12])
            with user.client:
                control("filter-modulation-expression").set_value("source * profile / np.mean(profile)")
            await convert([1, 6, 6])
            with user.client:
                control("filter-selection").set_value("threshold")
                control("filter-comparison").set_value("above")
                control("filter-threshold").set_value(3)
            await convert([2, 6, 6])
            with user.client:
                control("filter-selection").set_value("all")
                control("filter-modulation-expression").set_value("source / profile")
            await assay(b"atac\n1\n0\n2\n")
            await convert(error="infinite")
            with user.client:
                control("filter-modulation-expression").set_value("np.where(profile != 0, source / profile, np.nan)")
            await convert([2, np.nan, 3])
            with user.client:
                control("filter-modulation-expression").set_value("0.75 * source + 0.25 * profile")
            await convert([1.75, 3, 5])
            with user.client:
                control("filter-modulation-expression").set_value("np.log2(source) * profile**2")
            await convert([1, 0, np.log2(6) * 4])
            await assay(b"atac\n1\n2\n", "short.tsv")
            await convert([1, 8, 6])
            assert any("first sample" in message for message in messages)
            assert kinds[-1] == "warning"
            user.find(marker="data-manager-filter-modulation-upload-remove").click()
            await convert(error="Upload a modulation profile")
            with user.client:
                control("filter-action").set_value("replace")
            await convert([0, 0, 0])
    asyncio.run(scenario())


def test_convolution_formula_survives_configure_changes(monkeypatch):
    results = []
    original = page.convert_data
    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        results.append(result)
        return result
    monkeypatch.setattr(page, "convert_data", capture)
    async def scenario():
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-data-management").click()
            def control(name):
                return next(item for item in user.client.elements.values()
                            if f"data-manager-{name}" in item._markers and not item.is_deleted)
            await control("upload").handle_uploads([Upload.SmallFileUpload("source.tsv", "text/plain", b"signal\n2\n4\n6\n")])
            await user.should_see("Array size: 3", retries=60)
            user.find(marker="data-manager-filter").click()
            user.find(marker="data-manager-filter-enabled").click()
            formula = "convolve(source, profile / profile.sum())"
            with user.client:
                control("filter-selection").set_value("all")
                control("filter-action").set_value("modulate")
                control("filter-modulation-expression").set_value(formula)
            await control("filter-modulation-upload").handle_uploads([
                Upload.SmallFileUpload("kernel.tsv", "text/plain", b"weight\n1\n1\n1\n")
            ])
            await user.should_see("kernel.tsv", retries=60)
            with user.client:
                control("feature").set_value("firing_rate")
            assert control("filter-modulation-expression").value == formula
            user.find(marker="action-convert").click()
            for _ in range(180):
                await asyncio.sleep(.02)
                if results and control("filter-enabled").enabled:
                    break
            assert results
            np.testing.assert_allclose(results[-1].preview.values, np.convolve([2, 4, 6], [1/3] * 3, "same"))
            assert dict(results[-1].metadata)["modulation_formula"] == formula
    asyncio.run(scenario())
