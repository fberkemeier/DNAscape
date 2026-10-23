"""Curve shading follows live controls, viewport edges, and per-plot state."""

import asyncio
import io
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest
from nicegui.elements.upload import Upload
from nicegui.testing import user_simulation

from app.main import index
from app.preview_plotting import PreviewChartHandle, PreviewLine
from app.tests.test_plot_navigation import CaptureChart, navigation_options


def test_combined_curves_keep_their_colours_and_fill_each_axis_at_its_displayed_edge():
    lines = (
        PreviewLine("First", np.arange(3.), np.array([-2., 3., 1.]), "#123456"),
        PreviewLine("Second", np.arange(3.), np.array([10., 30., 20.]), "#abcdef", opacity=.4),
    )
    options = navigation_options(lines=lines, secondary_y_label="Second", secondary_invert_y=True)
    chart = CaptureChart(options)
    handle = PreviewChartHandle(chart, CaptureChart({}))
    handle.settings = SimpleNamespace(show_fill=SimpleNamespace(value=True), fill_side=SimpleNamespace(value="below"))
    handle._apply_curve_fill()
    assert [series["areaStyle"] for series in options["series"]] == [
        {"color": "#123456", "opacity": .15, "origin": "start"},
        {"color": "#abcdef", "opacity": .06, "origin": "end"},
    ]
    assert len(options["series"]) == 2


def test_curve_fill_updates_both_sides_inversion_and_export_without_changing_data():
    async def scenario():
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-plotting").click()

            def control(name):
                return next(item for item in user.client.elements.values()
                            if f"plotting-{name}" in item._markers and not item.is_deleted)

            stream = io.BytesIO()
            np.savez(stream, values=[2., 4., 6., 3., 8.])
            await control("data-upload").handle_uploads([
                Upload.SmallFileUpload("signal.npz", "application/octet-stream", stream.getvalue()),
            ])
            await user.should_see("Array size: 5", retries=60)
            user.find(marker="action-plot").click()
            await user.should_see(marker="plotting-result-plot", retries=60)
            user.find("Plot settings").click()
            chart = control("result-plot")
            handle = chart._dn_preview_handle
            original = list(chart.options["series"][0]["data"])
            assert not control("fill").value
            assert not control("fill-side").visible
            assert chart.options["series"][0]["areaStyle"] is None
            assert control("view").parent_slot.parent is control("aspect-ratio").parent_slot.parent.parent_slot.parent

            user.find(marker="plotting-fill").click()
            assert control("fill-side").visible
            assert chart.options["series"][0]["areaStyle"]["origin"] == "start"
            with user.client:
                control("fill-side").set_value("above")
                control("primary-color").set_value("#123456")
            assert chart.options["series"][0]["areaStyle"] == {
                "origin": "end", "color": "#123456", "opacity": .15,
            }
            with user.client:
                control("invert-y").set_value(True)
            assert chart.options["series"][0]["areaStyle"]["origin"] == "start"
            with user.client:
                control("log-y").set_value(True)
            handle.accept_navigation(dict(context=handle._navigation.context, ranges={"x": (1, 3), "y": (3, 6)}))
            assert chart.options["series"][0]["areaStyle"]["origin"] == "start"
            assert (chart.options["yAxis"]["min"], chart.options["yAxis"]["max"]) == pytest.approx((3, 6))
            assert len(chart.options["series"]) == 1
            assert list(chart.options["series"][0]["data"]) == original
            handle.vector.run_chart_method = AsyncMock(return_value="data:image/svg+xml;base64,PHN2Zy8+")
            await handle.run_chart_method("getDataURL", {"type": "svg"})
            assert handle.vector.options["series"][0]["areaStyle"] == chart.options["series"][0]["areaStyle"]

            with user.client:
                control("view").set_value("Histogram")
            assert not control("fill-options").visible
            with user.client:
                control("density").set_value(True)
            assert control("fill-options").visible
            assert chart.options["series"][0]["areaStyle"]["opacity"] == .15
            user.find(marker="plotting-fill").click()
            assert chart.options["series"][0]["areaStyle"] is None
            assert not control("fill-side").visible

    asyncio.run(scenario())
