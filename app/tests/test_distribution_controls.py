"""Combined distributions expose all legends and editable, centered boxes."""

import asyncio

import numpy as np
import pytest
from nicegui.elements.upload import Upload
from nicegui.testing import user_simulation

from app.main import index
from app.preview_plotting import PreviewChartHandle, PreviewLine
from app.tests.test_plot_navigation import CaptureChart, navigation_options


def distributions(**overrides):
    return (
        PreviewLine("First", np.arange(5.), np.arange(1., 6.), "#b03412", **overrides),
        PreviewLine("Second", np.arange(5.), np.arange(6., 11.), "#2596be"),
    )


@pytest.mark.parametrize("density", [False, True])
def test_histogram_legend_lists_every_dataset_without_statistics(density):
    options = navigation_options("Histogram", lines=distributions(), density=density,
                                 show_legend=True, show_histogram_statistics=False)
    assert options["legend"]["show"]
    assert options["legend"]["data"] == ["First", "Second"]
    assert not options["graphic"]
    assert not options["dnascapeHistogramGuides"]["selectable"]
    handle = PreviewChartHandle(CaptureChart(options), CaptureChart({}))
    handle.select_histogram_dataset({"index": 1})
    assert not options["graphic"]
    stats = navigation_options("Histogram", lines=distributions(), density=density,
                               show_legend=True, show_histogram_statistics=True)
    handle = PreviewChartHandle(CaptureChart(stats), CaptureChart({}))
    handle.select_histogram_dataset({"index": 1})
    assert "Second" in stats["graphic"][0]["style"]["text"]
    stats.clear()
    stats.update(navigation_options("Histogram", lines=distributions(), density=density,
                                    show_legend=True, show_histogram_statistics=False))
    handle.update()
    assert stats["graphic"] == [{"id": "dn-histogram-guide", "$action": "remove"}]
    single = navigation_options("Histogram", show_legend=True, show_histogram_statistics=True)
    assert not single["dnascapeHistogramGuides"]["selectable"]


def test_boxes_have_explicit_positions_labels_and_preserve_them_when_zoomed():
    lines = (PreviewLine("First", np.arange(5.), np.arange(1., 6.), "#b03412", box_position=2.5, box_label="Control", width=3),
             PreviewLine("Second", np.arange(5.), np.arange(6., 11.), "#2596be", box_position=8, box_label="Treatment"))
    options = navigation_options("Boxplot", lines=lines, show_legend=True)
    assert [series["data"][0][0] for series in options["series"]] == [2.5, 8]
    assert options["xAxis"]["axisTick"]["customValues"] == [2.5, 8]
    assert '"Control", "Treatment"' in options["xAxis"]["axisLabel"][":formatter"]
    assert options["series"][0]["data"][0][1:] == [1, 2, 3, 4, 5]
    assert all(series["type"] == "custom" and series["clip"] for series in options["series"])
    assert "lineWidth:3.0" in options["series"][0][":renderItem"]
    handle = PreviewChartHandle(CaptureChart(options), CaptureChart({}))
    handle._apply_navigation()
    handle.accept_navigation(dict(context=handle._navigation.context, ranges={"x": (2, 4)}))
    assert options["xAxis"]["axisLabel"]["customValues"] == [2.5]
    assert '"Control", "Treatment"' in options["xAxis"]["axisLabel"][":formatter"]
    with pytest.raises(ValueError, match="distinct"):
        navigation_options("Boxplot", lines=distributions(box_position=2))
    with pytest.raises(ValueError, match="positive"):
        navigation_options("Boxplot", lines=distributions(box_position=-1), log_x=True)


def test_distribution_controls_edit_styles_positions_and_follow_combined_layout():
    async def scenario():
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-plotting").click()
            def control(marker):
                return next(element for element in user.client.elements.values()
                            if f"plotting-{marker}" in element._markers and not element.is_deleted)
            def options():
                return control("result-plot").options
            await control("data-upload").handle_uploads([
                Upload.SmallFileUpload("first.tsv", "text/plain", b"First\n1\n2\n3\n4\n5\n"),
                Upload.SmallFileUpload("second.tsv", "text/plain", b"Second\n6\n7\n8\n9\n10\n"),
            ])
            await user.should_see(marker="plotting-detected-data-2", retries=80)
            await asyncio.sleep(.05)
            user.find(marker="action-plot").click()
            await user.should_see(marker="plotting-result-plot", retries=80)
            user.find("Curves").click()
            with user.client:
                control("view").set_value("Histogram")
                control("show-histogram-statistics").set_value(False)
            await user.should_not_see("Curves")
            user.find("Histograms").click()
            with user.client:
                control("histogram-name-1").set_value("Control")
                control("histogram-color-2").set_value("#123456")
                control("histogram-opacity-2").set_value(.25)
            assert options()["legend"]["data"] == ["Control", "Second"]
            assert options()["legend"]["show"]
            assert control("legend-position")._props["label"] == "Legend position"
            assert options()["series"][1]["itemStyle"]["color"] == "#123456"
            assert options()["series"][1]["itemStyle"]["opacity"] == .25
            with user.client:
                control("density").set_value(True)
            await user.should_see(marker="plotting-histogram-width-2", retries=40)
            with user.client:
                control("histogram-width-2").set_value(3)
                control("histogram-type-2").set_value("dashed")
            assert options()["series"][1]["lineStyle"]["width"] == 3
            assert options()["series"][1]["lineStyle"]["type"] == "dashed"
            assert options()["series"][1]["lineStyle"]["opacity"] == .25
            with user.client:
                control("view").set_value("Boxplot")
            await user.should_not_see("Histograms")
            user.find("Boxes").click()
            with user.client:
                control("box-position-1").set_value(3.5)
                control("box-label-1").set_value("Untreated")
                control("box-width-1").set_value(2.5)
            assert options()["series"][0]["data"][0][0] == 3.5
            assert options()["xAxis"]["axisTick"]["customValues"] == [3.5, 2]
            assert "Untreated" in options()["xAxis"]["axisLabel"][":formatter"]
            assert "lineWidth:2.5" in options()["series"][0][":renderItem"]
            with user.client:
                control("presentation").set_value("Separate")
            await user.should_not_see("Curves")
            await user.should_not_see("Boxes")
            await user.should_not_see("Histograms")
            with user.client:
                control("presentation").set_value("Combined")
            await user.should_see(marker="plotting-box-position-1", retries=40)
            assert control("box-position-1").value == 3.5
            with user.client:
                control("box-label-1").set_value(None)
            assert '"3.5"' in options()["xAxis"]["axisLabel"][":formatter"]
    asyncio.run(scenario())
