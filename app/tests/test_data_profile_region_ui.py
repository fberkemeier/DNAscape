"""Uploaded profiles retain their native extent when cropped, reset and rebinned."""

import asyncio
import io

import numpy as np
import pytest
from nicegui.elements.upload import Upload
from nicegui.testing import user_simulation

from app.main import index
from app.pages import data_management as page


def _capture_conversions(monkeypatch):
    conversions, messages = [], []
    original = page.convert_data

    def capture(request, *args, **kwargs):
        result = original(request, *args, **kwargs)
        conversions.append((request, result))
        return result

    monkeypatch.setattr(page, "convert_data", capture)
    monkeypatch.setattr(page, "notify", lambda message, **_kwargs: messages.append(message))
    return conversions, messages


def _control(user, name):
    return next(element for element in user.client.elements.values()
                if f"data-manager-{name}" in element._markers and not element.is_deleted)


async def _convert(user, conversions, messages):
    count = len(conversions)
    user.find(marker="action-convert").click()
    for _ in range(150):
        await asyncio.sleep(.02)
        if len(conversions) > count and _control(user, "filter-enabled").enabled:
            break
    assert len(conversions) == count + 1, messages
    return conversions[-1]


def test_plain_upload_full_range_reset_preserves_bin_size_and_uses_source_resolution(monkeypatch):
    conversions, messages = _capture_conversions(monkeypatch)

    async def scenario():
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-data-management").click()
            control = lambda name: _control(user, name)
            await control("upload").handle_uploads([
                Upload.SmallFileUpload("signal.tsv", "text/tab-separated-values", b"signal\n1\n2\n3\n4\n5\n6\n"),
            ])
            await user.should_see(marker="data-manager-reset-region", retries=60)
            assert (control("start").value, control("end").value) == (0, 6000)
            assert control("source-bin-size").value == 1000
            assert control("source-bin-size").enabled
            siblings = control("source-bin-size").parent_slot.children
            assert siblings[siblings.index(control("source-bin-size")) + 1] is control("bin-size")
            await user.should_not_see(marker="data-manager-source-start")
            await user.should_not_see(marker="data-manager-aggregation")
            with user.client:
                control("bin-size").set_value(2000)
                control("start").set_value(1000)
                control("end").set_value(5000)
            user.find(marker="data-manager-reset-region").click()
            assert (control("start").value, control("end").value) == (0, 6000)
            assert control("bin-size").value == 2000
            request, result = await _convert(user, conversions, messages)
            assert (request.start, request.end, request.bin_size, request.aggregation) == (0, 6000, 2000, "mean")
            np.testing.assert_equal(result.preview.values, [1.5, 3.5, 5.5])

            # Changing the assumed source spacing updates its full extent.
            with user.client:
                control("source-bin-size").set_value(500)
            for _ in range(60):
                await asyncio.sleep(.02)
                if control("end").value == 3000:
                    break
            assert (control("start").value, control("end").value) == (0, 3000)
            with user.client:
                control("bin-size").set_value(1000)
                control("start").set_value(500)
                control("end").set_value(2500)
            user.find(marker="data-manager-reset-region").click()
            assert (control("start").value, control("end").value) == (0, 3000)
            request, result = await _convert(user, conversions, messages)
            assert request.source_bin_size == 500
            np.testing.assert_equal(result.preview.values, [1.5, 3.5, 5.5])
            np.testing.assert_equal(result.preview.x, [0, 1, 2])

    asyncio.run(scenario())


def test_profile_reset_uses_the_selected_chromosome_extent(monkeypatch):
    conversions, messages = _capture_conversions(monkeypatch)

    async def scenario():
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-data-management").click()
            control = lambda name: _control(user, name)
            await control("upload").handle_uploads([
                Upload.SmallFileUpload("chromosomes.tsv", "text/tab-separated-values",
                    b"chrom\tposition_kb\tsignal\nchr1\t10\t1\nchr1\t11\t2\nchr1\t12\t3\nchr2\t20\t4\nchr2\t22\t6\n"),
            ])
            await user.should_see(marker="data-manager-chromosome", retries=60)
            assert (control("start").value, control("end").value) == (10000, 13000)
            with user.client:
                control("chromosome").set_value("chr2")
            for _ in range(60):
                await asyncio.sleep(.02)
                if control("start").value == 20000:
                    break
            assert (control("start").value, control("end").value) == (20000, 24000)
            assert control("source-bin-size").value == 2000
            with user.client:
                control("bin-size").set_value(4000)
                control("start").set_value(22000)
            user.find(marker="data-manager-reset-region").click()
            assert (control("start").value, control("end").value) == (20000, 24000)
            assert control("bin-size").value == 4000
            request, result = await _convert(user, conversions, messages)
            assert request.chromosome == "chr2"
            np.testing.assert_equal(result.preview.values, [5])
            np.testing.assert_equal(result.preview.x, [20])

    asyncio.run(scenario())


@pytest.mark.parametrize("file_type", ["tsv", "csv", "txt", "npz", "intervals"])
def test_coordinate_upload_reset_retains_nonzero_native_extent_and_converts(monkeypatch, file_type):
    conversions, messages = _capture_conversions(monkeypatch)
    if file_type == "npz":
        stream = io.BytesIO()
        np.savez_compressed(stream, values=np.array([1., 2., 3., 4.]),
                            position_kb=np.array([10., 11., 12., 13.]),
                            data_form=np.array("space"), columns=np.array(["signal"]))
        filename, payload = "profile.npz", stream.getvalue()
    elif file_type == "intervals":
        filename = "profile.csv"
        payload = b"start_bp,end_bp,signal\n10000,11000,1\n11000,12000,2\n12000,13000,3\n13000,14000,4\n"
    else:
        separator = "," if file_type == "csv" else "\t"
        filename = f"profile.{file_type}"
        payload = (f"position_kb{separator}signal\n" + "".join(
            f"{position}{separator}{value}\n" for position, value in zip(range(10, 14), range(1, 5))
        )).encode()

    async def scenario():
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-data-management").click()
            control = lambda name: _control(user, name)
            await control("upload").handle_uploads([
                Upload.SmallFileUpload(filename, "application/octet-stream", payload),
            ])
            await user.should_see(marker="data-manager-reset-region", retries=60)
            assert (control("start").value, control("end").value) == (10000, 14000)
            assert control("source-bin-size").value == 1000
            assert not control("source-bin-size").enabled
            await user.should_not_see(marker="data-manager-source-start")
            await user.should_not_see(marker="data-manager-aggregation")
            with user.client:
                control("start").set_value(11000)
                control("end").set_value(13000)
                control("bin-size").set_value(2000)
            request, result = await _convert(user, conversions, messages)
            assert (request.start, request.end, request.bin_size) == (11000, 13000, 2000)
            np.testing.assert_equal(result.preview.values, [2.5])
            user.find(marker="data-manager-reset-region").click()
            assert (control("start").value, control("end").value) == (10000, 14000)
            assert control("bin-size").value == 2000
            request, result = await _convert(user, conversions, messages)
            assert (request.start, request.end) == (10000, 14000)
            np.testing.assert_equal(result.preview.values, [1.5, 3.5])
            np.testing.assert_equal(result.preview.x, [10, 12])

    asyncio.run(scenario())
