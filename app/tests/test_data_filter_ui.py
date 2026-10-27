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
from app.preview_plotting import PreviewChartHandle
from dnascape.workflows.conversion import DataConversionRequest, convert_data
from dnascape.workflows.filtering import FilterSelection
from dnascape.workflows.uploads import inspect_upload


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
                previous_chart = next((item for item in user.client.elements.values()
                                       if "data-manager-preview-chart" in item._markers and not item.is_deleted), None)
                user.find(marker="action-convert").click()
                for _ in range(150):
                    await asyncio.sleep(.02)
                    chart_ready = not np.isfinite(expected).any() or any(
                        "data-manager-preview-chart" in item._markers and not item.is_deleted
                        and item is not previous_chart for item in user.client.elements.values()
                    )
                    if len(results) > count and control("filter-enabled").enabled and chart_ready:
                        break
                assert len(results) == count + 1, messages
                np.testing.assert_equal(results[-1].preview.values, expected)
                np.testing.assert_equal(pd.read_csv(io.BytesIO(results[-1].artifact.data), sep="\t").iloc[:, -1].to_numpy(), expected)
                if np.isfinite(expected).any():
                    await user.should_see(marker="data-manager-preview-chart", retries=80)

            assert control("filter").value is False
            user.find(marker="data-manager-filter").click()
            assert control("filter-enabled").value is False
            from nicegui.elements.switch import Switch
            assert isinstance(control("filter-export"), Switch)
            assert isinstance(control("filter-complement"), Switch)
            assert control("filter-complement").value is False
            actions = control("filter-export").parent_slot.children
            assert actions.index(control("filter-complement")) == actions.index(control("filter-export")) + 1
            assert control("filter-profile-units").value == "bp"
            await user.should_see(marker="data-manager-filter-selection")
            assert control("filter-enabled").__class__.__name__ == "Switch"
            await user.should_see(marker="data-manager-filter-replace-value")
            await control("upload").handle_uploads([
                Upload.SmallFileUpload("signal.csv", "text/csv", b"position,value\n0,-2\n1,1\n2,3\n"),
            ])
            await user.should_see("Array size: 3", retries=60)
            with user.client:
                control("value-scaling").set_value("multiplier")
                control("scale-multiplier").set_value(-2)
            await converted([4, -2, -6])
            assert requests[-1].data_filter.enabled is False
            chart = control("preview-chart")
            assert "markArea" not in chart.options["series"][0]
            assert control("plot-show-filter").value is False
            assert not control("plot-show-filter").enabled
            user.find(marker="data-manager-filter-enabled").click()
            assert control("filter-selection").value == "threshold"
            assert control("filter-comparison").value == "below"
            assert control("filter-threshold").value == 0
            assert control("filter-action").value == "replace"
            assert control("filter-replace-value").value == 0
            await converted([4, 0, 0])
            with user.client:
                control("plot-show-filter").set_value(True)
                control("plot-show-legend").set_value(True)
                control("plot-legend-name").set_value("Processed signal")
                control("plot-filter-name").set_value("Corrected regions")
            areas = control("preview-chart").options["series"][1]["markArea"]
            assert areas["silent"] and areas["data"] == [[{"xAxis": .5}, {"xAxis": 2.5}]]
            assert control("preview-chart").options["legend"]["data"] == [
                "Processed signal", {"name": "Corrected regions", "icon": "roundRect"},
            ]
            assert len(results) == 2  # Display options never reconvert the output.
            # The original negative bins remain highlighted after replacement with zero.
            assert control("preview-chart").options["series"][1]["markArea"]["data"] == areas["data"]
            with user.client:
                control("filter-complement").set_value(True)
            await converted([0, -2, -6])
            assert requests[-1].data_filter.complement is True
            assert control("preview-chart").options["series"][1]["markArea"]["data"] == [
                [{"xAxis": -.5}, {"xAxis": .5}],
            ]
            with user.client:
                control("filter-complement").set_value(False)
            await converted([4, 0, 0])
            with user.client:
                control("filter-selection").set_value("between")
                control("filter-lower").set_value(-6)
                control("filter-upper").set_value(-1)
                control("filter-action").set_value("interpolate")
            await user.should_see(marker="data-manager-filter-interpolation")
            await user.should_not_see(marker="data-manager-filter-threshold")
            await user.should_not_see(marker="data-manager-filter-standard-deviation")
            assert control("filter-between-mode").value == "within"
            await converted([4, 4, 4])
            assert control("preview-chart").options["series"][1]["markArea"]["data"] == areas["data"]
            with user.client:
                control("filter-between-mode").set_value("outside")
                control("filter-action").set_value("replace")
                control("filter-replace-value").set_value(8)
            await converted([8, -2, -6])
            assert requests[-1].data_filter.between_mode == "outside"
            with user.client:
                control("filter-selection").set_value("profile")
                control("filter-action").set_value("replace")
                control("filter-replace-value").set_value(7)
            # No uploaded mask is deliberately inert.
            await converted([4, -2, -6])
            with user.client:
                before = current_session_cache().count()
            await control("filter-upload").handle_uploads([
                Upload.SmallFileUpload("mask.tsv", "text/tab-separated-values", b"position_kb\n1\n"),
            ])
            await user.should_see("mask.tsv", retries=60)
            assert control("filter-profile-units").value == "kb"
            assert list(control("filter-profile-units").options) == ["kb", "bp"]
            await user.should_not_see(marker="data-manager-filter-profile-layout")
            await user.should_not_see(marker="data-manager-filter-profile-column")
            await user.should_not_see(marker="data-manager-filter-profile-array-key")
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
            assert control("filter-profile-units").value == "bp"
            await user.should_see(marker="data-manager-filter-profile-units")
            await converted([4, 7, -6])
            assert requests[-1].data_filter.profile.layout == "positions"
            user.find(marker="data-manager-filter-upload-remove").click()
            await user.should_not_see("positions.tsv")
            await control("filter-upload").handle_uploads([
                Upload.SmallFileUpload("regions.tsv", "text/tab-separated-values", b"start_kb\tend_kb\n0\t2\n"),
            ])
            await user.should_see("regions.tsv", retries=60)
            assert control("filter-profile-units").value == "kb"
            await converted([7, 7, -6])
            assert requests[-1].data_filter.profile.layout == "intervals"
            with user.client:
                control("filter-selection").set_value("all")
                control("filter-action").set_value("nan")
            await converted([np.nan] * 3)
            await user.should_see("No finite values to plot. The raw data remain available below.")
            await user.should_see(marker="data-manager-result-name-1")
            user.find(marker="data-manager-filter-enabled").click()
            assert not control("plot-show-filter").enabled
            await converted([4, -2, -6])
            await user.should_see(marker="data-manager-preview-chart")
            assert len(control("preview-chart").options["series"]) == 1
    asyncio.run(scenario())


def test_filter_shading_follows_selected_generated_file_and_guide_visibility():
    async def scenario():
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-data-management").click()

            def control(name):
                return next(item for item in user.client.elements.values()
                            if f"data-manager-{name}" in item._markers and not item.is_deleted)

            await control("upload").handle_uploads([
                Upload.SmallFileUpload("first.tsv", "text/plain", b"value\n1\n-2\n3\n"),
                Upload.SmallFileUpload("second.tsv", "text/plain", b"value\n1\n2\n-3\n"),
            ])
            await user.should_see(marker="data-manager-upload-source-label-2", retries=80)
            user.find(marker="data-manager-filter").click()
            user.find(marker="data-manager-filter-enabled").click()
            user.find(marker="action-convert").click()
            await user.should_see(marker="data-manager-select-result-2", retries=80)
            await user.should_see(marker="data-manager-preview-chart", retries=80)
            with user.client:
                control("plot-show-filter").set_value(True)
                control("plot-show-legend").set_value(True)
            assert control("plot-legend-name").visible
            assert control("plot-filter-name").visible
            assert control("preview-chart").options["series"][1]["markArea"]["data"] == [[{"xAxis": 1.5}, {"xAxis": 2.5}]]
            user.find(marker="data-manager-select-result-1").click()
            for _ in range(80):
                await asyncio.sleep(.01)
                if control("select-result-1")._props["aria-pressed"] == "true":
                    break
            series = control("preview-chart").options["series"]
            assert series[1]["markArea"]["data"] == [[{"xAxis": .5}, {"xAxis": 1.5}]]
            np.testing.assert_equal([row[1] for row in series[0]["data"]], [1, 0, 3])
            assert control("filter-enabled").value is True
            with user.client:
                control("plot-show-legend").set_value(False)
            assert not control("plot-legend-name").visible
            assert not control("plot-filter-name").visible
            assert len(control("preview-chart").options["series"]) == 2
            with user.client:
                control("plot-view").set_value("Histogram")
            assert not control("plot-show-filter").visible
            assert not control("plot-filter-name").visible
            with user.client:
                control("plot-view").set_value("Line")
                control("filter-enabled").set_value(False)
            assert len(control("preview-chart").options["series"]) == 1
            assert not control("plot-show-filter").enabled

    asyncio.run(scenario())


def test_filter_shading_refines_with_plot_navigation_and_resets_to_full_view():
    result = convert_data(DataConversionRequest(
        inspection=inspect_upload("source.tsv", b"value\n1\n2\n3\n"), input_form="x",
    ))
    options = page._preview_options(result.preview)
    mask = np.zeros(100_000, dtype=bool)
    mask[12_345] = True
    selection = FilterSelection.from_mask(mask, np.linspace(0, 2, len(mask)))
    options["series"][0]["markArea"] = {"data": page._FilterPreviewRegions(selection)}

    class Chart:
        def __init__(self, options):
            self.options = options

        def update(self):
            pass

    handle = PreviewChartHandle(Chart(options), Chart({}))
    handle._apply_navigation()
    full_view = list(options["series"][0]["markArea"]["data"])
    handle.accept_navigation(dict(context=handle._navigation.context, ranges={"x": (.2468, .247)}))
    refined = options["series"][0]["markArea"]["data"]
    np.testing.assert_allclose([[edge["xAxis"] for edge in area] for area in refined],
                               np.array([[12_344.5, 12_345.5]]) * selection.step)
    assert refined != full_view
    handle._navigation.ranges.clear()
    options["xAxis"].update(min=0, max=2)
    handle._refine_data()
    assert options["series"][0]["markArea"]["data"] == full_view


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
                control("feature").set_value("initiation_rate")
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
