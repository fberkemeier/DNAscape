"""End-to-end NiceGUI coverage for the live Mapping action."""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import replace
from typing import Any, cast

import numpy as np
import pytest
from nicegui.elements.button import Button
from nicegui.elements.checkbox import Checkbox
from nicegui.elements.echart import EChart
from nicegui.elements.input import Input
from nicegui.elements.number import Number
from nicegui.elements.progress import LinearProgress
from nicegui.elements.select import Select
from nicegui.elements.upload import Upload
from nicegui.testing import user_simulation

from app.main import index
from app.mapping_service import MappingComparison, MappingRequest, MappingResult


def _mapped_result(request: MappingRequest, values: np.ndarray) -> MappingResult:
    return MappingResult(
        target=request.route.target,
        target_key=request.route.target_key,
        target_label="Firing rate",
        method=request.route.method,
        output_form="x",
        output_dependence="space",
        primary_output="firing_rate",
        outputs={"firing_rate": values},
        output_forms={"firing_rate": "x"},
        x_axis=np.arange(values.size, dtype=float),
        time_axis=np.asarray([], dtype=float),
        region_labels=(),
        elapsed_seconds=0.08,
        assumptions={},
    )


def test_csv_upload_auto_selects_chr1_and_map_renders_progress_and_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.pages import mapping as mapping_page

    started = threading.Event()
    release = threading.Event()
    requests: list[MappingRequest] = []

    def fake_run(request: MappingRequest, report: Any, _cancel: Any) -> MappingResult:
        requests.append(request)
        report(0.55, "Fitting target profile...")
        started.set()
        if not release.wait(timeout=5):
            raise TimeoutError("test did not release the mapping worker")
        return _mapped_result(request, np.asarray([0.2, 0.4, 0.6]))

    monkeypatch.setattr(mapping_page, "run_mapping", fake_run)

    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-mapping").click()
            await user.should_see("Map replication kinetic landscapes", retries=60)
            uploader = cast(
                Upload,
                next(iter(user.find(marker="source-replication-timing-upload").elements)),
            )
            await uploader.handle_uploads(
                [
                    Upload.SmallFileUpload(
                        "timing.csv",
                        "text/csv",
                        b"position,chr2,chr1\n0,9,1\n1,8,\n2,7,3\n",
                    )
                ]
            )
            await user.should_see(marker="mapping-input-replication-timing-detected-data")
            column = cast(
                Select,
                next(
                    iter(user.find(marker="mapping-input-replication-timing-value-column").elements)
                ),
            )
            assert column.value == "chr1"
            missing_policy = cast(
                Select,
                next(iter(user.find(marker="mapping-region-missing").elements)),
            )
            missing_policy.set_value("interpolate")
            remove_button = cast(
                Button,
                next(iter(user.find(marker="source-replication-timing-upload-remove").elements)),
            )
            assert remove_button.icon == "close"

            user.find(marker="action-map").click()
            await user.should_see(marker="mapping-progress", retries=20)
            assert await asyncio.to_thread(started.wait, 2)
            await user.should_see("Fitting target profile...", retries=20)
            await asyncio.sleep(0.05)
            assert not uploader.enabled
            assert not remove_button.enabled
            assert requests
            np.testing.assert_array_equal(requests[0].inputs["replication_timing"], [1, 2, 3])

            release.set()
            await user.should_see(marker="mapping-result-plot", retries=30)
            await user.should_see(marker="mapping-stats", retries=10)
            await user.should_see("Complete in 0.08 s", retries=10)
            await user.should_not_see(marker="mapping-error")
            assert uploader.enabled
            assert remove_button.enabled

            user.find(marker="source-replication-timing-upload-remove").click()
            await user.should_see(marker="mapping-result-plot", retries=20)
            await user.should_see("Complete in 0.08 s", retries=20)

    try:
        asyncio.run(scenario())
    finally:
        release.set()


def test_mapping_plot_settings_redraw_the_completed_preview(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.pages import mapping as mapping_page

    def fake_run(request: MappingRequest, report: Any, _cancel: Any) -> MappingResult:
        report(0.8, "Preparing mapped preview...")
        return _mapped_result(request, np.asarray([0.2, 0.4, 0.6, 0.8]))

    monkeypatch.setattr(mapping_page, "run_mapping", fake_run)

    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-mapping").click()
            await user.should_see(marker="source-replication-timing-upload", retries=60)
            uploader = cast(
                Upload,
                next(iter(user.find(marker="source-replication-timing-upload").elements)),
            )
            await uploader.handle_uploads(
                [
                    Upload.SmallFileUpload(
                        "timing.csv",
                        "text/csv",
                        b"timing\n8\n7\n9\n10\n",
                    )
                ]
            )
            await user.should_see(
                marker="mapping-input-replication-timing-detected-data",
                retries=20,
            )
            user.find(marker="action-map").click()
            await user.should_see(marker="mapping-result-plot", retries=30)
            user.find("Plotting settings").click()

            def chart() -> EChart:
                return cast(
                    EChart,
                    next(iter(user.find(marker="mapping-result-plot").elements)),
                )

            title = cast(
                Input,
                next(iter(user.find(marker="mapping-preview-title").elements)),
            )
            await user.should_not_see("Series label")
            x_label = cast(
                Input,
                next(iter(user.find(marker="mapping-preview-x-label").elements)),
            )
            y_label = cast(
                Input,
                next(iter(user.find(marker="mapping-preview-y-label").elements)),
            )
            color = cast(
                Input,
                next(iter(user.find(marker="mapping-line-or-bar-color").elements)),
            )
            assert user.client is not None
            with user.client:
                title.set_value("Updated mapping")
                x_label.set_value("Genomic position")
                y_label.set_value("Rate")
                color.set_value("#c43b22")
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            updated = chart()
            assert updated.options["title"]["text"] == "Updated mapping"
            assert updated.options["series"][0]["lineStyle"]["color"] == "#c43b22"
            assert updated.options["xAxis"]["name"] == "Genomic position"
            assert updated.options["yAxis"]["name"] == "Rate"

            line_width = cast(
                Number,
                next(iter(user.find(marker="mapping-line-width").elements)),
            )
            with user.client:
                line_width.set_value(None)
            await asyncio.sleep(0)
            assert chart().options["title"]["text"] == "Updated mapping"
            await user.should_not_see("Output unavailable")
            with user.client:
                line_width.set_value(1.8)
            await asyncio.sleep(0)

            invert = cast(
                Checkbox,
                next(iter(user.find(marker="mapping-invert-y").elements)),
            )
            with user.client:
                invert.set_value(True)
            await asyncio.sleep(0)
            assert chart().options["yAxis"]["inverse"] is True
            assert chart().options["xAxis"]["position"] == "bottom"

            y_minimum = cast(
                Number,
                next(iter(user.find(marker="mapping-y-minimum").elements)),
            )
            y_maximum = cast(
                Number,
                next(iter(user.find(marker="mapping-y-maximum").elements)),
            )
            assert float(y_minimum.value) == pytest.approx(0.176)
            assert float(y_maximum.value) == pytest.approx(0.824)
            assert chart().options["yAxis"]["min"] == pytest.approx(0.176)
            assert chart().options["yAxis"]["max"] == pytest.approx(0.824)
            with user.client:
                y_maximum.set_value(1.0)
                y_minimum.set_value(0.0)
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            assert chart().options["yAxis"]["min"] == 0.0
            assert chart().options["yAxis"]["max"] == 1.0

            view = cast(
                Select,
                next(iter(user.find(marker="mapping-view").elements)),
            )
            with user.client:
                view.set_value("Histogram")
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            assert chart().options["series"][0]["type"] == "custom"
            await user.should_see(marker="mapping-histogram-bins")

            touching = cast(
                Checkbox,
                next(iter(user.find(marker="mapping-histogram-touching").elements)),
            )
            boundary = cast(
                Select,
                next(iter(user.find(marker="mapping-histogram-boundary").elements)),
            )
            assert touching.value is True
            assert boundary.value == "left"
            assert "* 0" in chart().options["series"][0][":renderItem"]
            with user.client:
                touching.set_value(False)
            await asyncio.sleep(0)
            assert "* 0.14" in chart().options["series"][0][":renderItem"]
            with user.client:
                boundary.set_value("right")
            await asyncio.sleep(0)
            assert boundary.value == "right"

            bins = cast(
                Number,
                next(iter(user.find(marker="mapping-histogram-bins").elements)),
            )
            bin_size = cast(
                Number,
                next(iter(user.find(marker="mapping-histogram-bin-size").elements)),
            )
            density = cast(
                Checkbox,
                next(iter(user.find(marker="mapping-density").elements)),
            )
            with user.client:
                bins.set_value(2)
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            histogram = chart()
            assert len(histogram.options["series"][0]["data"]) == 2
            assert bin_size.value == pytest.approx(0.3)

            with user.client:
                bin_size.set_value(0.2)
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            assert bins.value == 3
            assert len(chart().options["series"][0]["data"]) == 3

            with user.client:
                bin_size.set_value(None)
            await asyncio.sleep(0)
            assert len(chart().options["series"][0]["data"]) == 3
            await user.should_not_see("Output unavailable")

            with user.client:
                bin_size.set_value(0.3)
                density.set_value(True)
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            histogram = chart()
            assert bins.value == 2
            assert len(histogram.options["series"][0]["data"]) == 2
            assert histogram.options["yAxis"]["name"] == "Density"

            x_minimum = cast(
                Number,
                next(iter(user.find(marker="mapping-histogram-x-minimum").elements)),
            )
            x_maximum = cast(
                Number,
                next(iter(user.find(marker="mapping-histogram-x-maximum").elements)),
            )
            with user.client:
                bin_size.set_value(0.125)
                x_minimum.set_value(0.3)
                x_maximum.set_value(0.7)
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            assert bin_size.value == pytest.approx(0.125)
            assert bins.value == 4
            histogram = chart()
            assert len(histogram.options["series"][0]["data"]) == 4
            assert histogram.options["xAxis"]["min"] == pytest.approx(0.3)
            assert histogram.options["xAxis"]["max"] == pytest.approx(0.7)
            assert "Mean       0.5" in histogram.options["graphic"][0]["style"]["text"]

            user.find(marker="mapping-reset-histogram-x-limits").click()
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            assert x_minimum.value is not None
            assert x_maximum.value is not None
            assert float(x_minimum.value) == pytest.approx(0.2)
            assert float(x_maximum.value) == pytest.approx(0.8)
            assert bins.value == 5

            statistics = cast(
                Checkbox,
                next(iter(user.find(marker="mapping-show-histogram-statistics").elements)),
            )
            with user.client:
                statistics.set_value(True)
            await asyncio.sleep(0)
            statistics_position = cast(
                Select,
                next(iter(user.find(marker="mapping-histogram-statistics-position").elements)),
            )
            with user.client:
                statistics_position.set_value("Bottom right")
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            summary = chart().options["graphic"][0]
            assert summary["right"] == 28
            assert summary["bottom"] == 8
            assert chart().options["grid"]["bottom"] >= 100
            assert "Mean" in summary["style"]["text"]
            assert "Median" in summary["style"]["text"]
            assert "Std dev" in summary["style"]["text"]

    asyncio.run(scenario())


def test_mapping_reports_mse_and_orders_residual_before_mse_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.pages import mapping as mapping_page

    def fake_run(request: MappingRequest, report: Any, _cancel: Any) -> MappingResult:
        report(0.8, "Preparing mapped preview...")
        result = _mapped_result(request, np.asarray([0.2, 0.4, 0.6]))
        return replace(
            result,
            outputs={
                "firing_rate": np.asarray([0.2, 0.4, 0.6]),
                "residual": np.asarray([0.1, -0.2, 0.05]),
                "fit_history": np.asarray([0.4, 0.2, 0.0525]),
            },
            output_forms={
                "firing_rate": "x",
                "residual": "x",
                "fit_history": "supplied",
            },
            fit_mse=0.0525,
        )

    monkeypatch.setattr(mapping_page, "run_mapping", fake_run)

    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-mapping").click()
            await user.should_see(marker="source-replication-timing-upload", retries=60)
            uploader = cast(
                Upload,
                next(iter(user.find(marker="source-replication-timing-upload").elements)),
            )
            await uploader.handle_uploads(
                [Upload.SmallFileUpload("timing.csv", "text/csv", b"timing\n8\n7\n9\n")]
            )
            await user.should_see("timing.csv", retries=20)
            user.find(marker="action-map").click()
            await user.should_see("MSE 0.0525", retries=30)

            output = cast(
                Select,
                next(iter(user.find(marker="mapping-preview-output").elements)),
            )
            assert tuple(output.options)[-2:] == ("residual", "fit_history")
            assert output.options["residual"] == "Residual"
            assert output.options["fit_history"] == "MSE"
            assert user.client is not None
            with user.client:
                output.set_value("fit_history")
            await asyncio.sleep(0)
            raw_name = cast(
                Input,
                next(iter(user.find(marker="mapping-raw-filename").elements)),
            )
            x_label = cast(
                Input,
                next(iter(user.find(marker="mapping-preview-x-label").elements)),
            )
            y_label = cast(
                Input,
                next(iter(user.find(marker="mapping-preview-y-label").elements)),
            )
            assert x_label.value == "Iteration"
            assert y_label.value == "MSE"
            assert raw_name.value == "dnascape_mapping_mse"
            chart = cast(
                EChart,
                next(iter(user.find(marker="mapping-result-plot").elements)),
            )
            assert chart.options["series"][0]["data"] == [
                [1.0, 0.4],
                [2.0, 0.2],
                [3.0, 0.0525],
            ]

    asyncio.run(scenario())


def test_mapping_validation_and_worker_failures_reenable_map_button(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.pages import mapping as mapping_page

    def failing_run(*_args: Any, **_kwargs: Any) -> MappingResult:
        raise ValueError("Synthetic mapping failure")

    monkeypatch.setattr(mapping_page, "run_mapping", failing_run)

    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-mapping").click()
            await user.should_see("Map replication kinetic landscapes", retries=60)
            button = cast(
                Button,
                next(iter(user.find(marker="action-map").elements)),
            )

            user.find(marker="action-map").click()
            await user.should_see("Upload replication timing data", retries=20)
            await asyncio.sleep(0.1)
            assert button.enabled

            uploader = cast(
                Upload,
                next(iter(user.find(marker="source-replication-timing-upload").elements)),
            )
            await uploader.handle_uploads(
                [Upload.SmallFileUpload("timing.csv", "text/csv", b"chr1\n1\n2\n")]
            )
            await user.should_see(
                marker="mapping-input-replication-timing-detected-data",
                retries=20,
            )
            await asyncio.sleep(0.05)
            user.find(marker="action-map").click()
            await user.should_see("Synthetic mapping failure", retries=30)
            await user.should_see("Mapping could not be completed")
            await user.should_not_see(marker="mapping-progress")
            await user.should_not_see(marker="mapping-result-plot")
            assert button.enabled

    asyncio.run(scenario())


def test_map_button_becomes_stop_and_cancels_the_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.pages import mapping as mapping_page

    started = threading.Event()

    def cancellable_run(request: MappingRequest, report: Any, cancel: Any) -> MappingResult:
        del request
        report(0.4, "Mapping test data...")
        started.set()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            cancel()
            time.sleep(0.01)
        raise TimeoutError("mapping cancellation was not delivered")

    monkeypatch.setattr(mapping_page, "run_mapping", cancellable_run)

    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-mapping").click()
            await user.should_see(marker="source-replication-timing-upload", retries=60)
            uploader = cast(
                Upload,
                next(iter(user.find(marker="source-replication-timing-upload").elements)),
            )
            await uploader.handle_uploads(
                [Upload.SmallFileUpload("timing.csv", "text/csv", b"timing\n1\n2\n3\n")]
            )
            await user.should_see(
                marker="mapping-input-replication-timing-detected-data",
                retries=20,
            )
            button = cast(Button, next(iter(user.find(marker="action-map").elements)))

            user.find(marker="action-map").click()
            assert await asyncio.to_thread(started.wait, 2)
            for _ in range(20):
                if button.text == "Stop":
                    break
                await asyncio.sleep(0.05)
            assert button.text == "Stop"
            assert button.icon == "stop"
            assert button.enabled

            user.find(marker="action-map").click()
            await user.should_see("Mapping stopped", retries=40)
            assert button.text == "Map"
            assert button.icon == "map"
            assert button.enabled
            await user.should_not_see(marker="mapping-error")

    asyncio.run(scenario())


def test_stop_interrupts_input_preparation_without_waiting_for_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.pages import mapping as mapping_page

    started = threading.Event()

    async def slow_read(*_args: Any, **_kwargs: Any) -> tuple[np.ndarray, bool]:
        started.set()
        await asyncio.sleep(10)
        return np.ones(3), False

    def unexpected_mapping(*_args: Any, **_kwargs: Any) -> MappingResult:
        raise AssertionError("rmap must not start after preparation is stopped")

    monkeypatch.setattr(mapping_page, "_read_mapping_input_async", slow_read)
    monkeypatch.setattr(mapping_page, "run_mapping", unexpected_mapping)

    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-mapping").click()
            await user.should_see(marker="action-map", retries=60)
            button = cast(Button, next(iter(user.find(marker="action-map").elements)))

            user.find(marker="action-map").click()
            assert await asyncio.to_thread(started.wait, 2)
            assert button.text == "Stop"
            stopped_at = time.monotonic()
            user.find(marker="action-map").click()
            await user.should_see("Mapping stopped", retries=30)
            assert time.monotonic() - stopped_at < 1.5
            assert button.text == "Map"
            assert button.enabled

    asyncio.run(scenario())


def test_whole_region_toggle_disables_region_and_progress_has_no_value_label() -> None:
    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-mapping").click()
            await user.should_see(marker="mapping-whole-region", retries=60)
            whole_region = cast(
                Checkbox,
                next(iter(user.find(marker="mapping-whole-region").elements)),
            )
            start = cast(Number, next(iter(user.find(marker="mapping-region-start").elements)))
            end = cast(Number, next(iter(user.find(marker="mapping-region-end").elements)))
            assert user.client is not None
            progress = cast(
                LinearProgress,
                next(
                    element
                    for element in user.client.elements.values()
                    if "mapping-progress" in element._markers
                ),
            )

            assert whole_region.value is False
            assert start.enabled and end.enabled
            assert progress._props["size"] == "4px"
            assert not progress.default_slot.children

            with user.client:
                whole_region.set_value(True)
            await asyncio.sleep(0)
            assert not start.enabled and not end.enabled

            with user.client:
                whole_region.set_value(False)
            await asyncio.sleep(0)
            assert start.enabled and end.enabled

    asyncio.run(scenario())


def test_scalar_mapping_preview_uses_a_visible_point() -> None:
    from app.pages.mapping import _mapping_plot_options

    result = MappingResult(
        target="completion_time",
        target_key="completion_time",
        target_label="Completion time",
        method="test",
        output_form="scalar",
        output_dependence="constant",
        primary_output="completion_time",
        outputs={"completion_time": np.asarray([12.5])},
        output_forms={"completion_time": "scalar"},
        x_axis=np.asarray([], dtype=float),
        time_axis=np.asarray([], dtype=float),
        region_labels=(),
        elapsed_seconds=0.01,
        assumptions={},
    )
    options = _mapping_plot_options(
        result,
        output_key="completion_time",
        plot_type="Line",
        normalization="None",
        title="Completion time",
        x_label="Output",
        y_label="Completion time",
        color="#2c7f77",
        secondary_color="#b03412",
        palette="viridis",
        histogram_bins=20,
        line_width=1.8,
        show_markers=False,
        show_colorbar=True,
        log_y=False,
        invert_y=False,
    )

    series = options["series"][0]
    assert series["type"] == "scatter"
    assert series["showSymbol"] is True
    assert series["symbolSize"] >= 8
    assert series["data"] == [[0.0, 12.5]]


def test_periodic_spatial_preview_closes_the_ring_at_the_next_bin() -> None:
    from app.pages.mapping import _mapping_plot_options

    result = MappingResult(
        target="firing_rate",
        target_key="firing_rate",
        target_label="Firing rate",
        method="test",
        output_form="x",
        output_dependence="space",
        primary_output="firing_rate",
        outputs={"firing_rate": np.asarray([0.1, 0.3, 0.2])},
        output_forms={"firing_rate": "x"},
        x_axis=np.asarray([10.0, 11.0, 12.0]),
        time_axis=np.asarray([], dtype=float),
        region_labels=(),
        elapsed_seconds=0.01,
        assumptions={"perQ": True},
    )
    options = _mapping_plot_options(
        result,
        output_key="firing_rate",
        plot_type="Line",
        normalization="None",
        title="Firing rate",
        x_label="Position (kb)",
        y_label="Firing rate",
        color="#2c7f77",
        palette="viridis",
        histogram_bins=20,
        line_width=1.8,
        show_markers=False,
        show_colorbar=True,
        log_y=False,
        invert_y=False,
    )

    assert options["series"][0]["data"] == [
        [10.0, 0.1],
        [11.0, 0.3],
        [12.0, 0.2],
        [13.0, 0.1],
    ]


def test_temporal_mapping_preview_spans_position_x_and_time_y() -> None:
    from app.pages.mapping import _mapping_plot_options

    result = MappingResult(
        target="replicated_fraction",
        target_key="replicated_fraction",
        target_label="Replicated fraction",
        method="test",
        output_form="t",
        output_dependence="time",
        primary_output="replicated_fraction",
        outputs={"replicated_fraction": np.asarray([0.1, 0.4, 0.9])},
        output_forms={"replicated_fraction": "t"},
        x_axis=np.asarray([10.0, 11.0]),
        time_axis=np.asarray([0.0, 5.0, 10.0]),
        region_labels=(),
        elapsed_seconds=0.01,
        assumptions={},
    )
    options = _mapping_plot_options(
        result,
        output_key="replicated_fraction",
        plot_type="Heatmap",
        normalization="None",
        title="Replicated fraction",
        x_label="Chromosome position (kb)",
        y_label="Time (min)",
        color="#2c7f77",
        palette="viridis",
        histogram_bins=20,
        line_width=1.8,
        show_markers=False,
        show_colorbar=True,
        log_y=False,
        invert_y=False,
        y_limits=(0.0, 10.0),
    )

    assert options["series"][0]["type"] == "custom"
    assert options["xAxis"]["name"] == "Chromosome position (kb)"
    assert options["yAxis"]["name"] == "Time (min)"
    assert options["xAxis"]["type"] == "value"
    assert options["yAxis"]["type"] == "value"
    assert options["xAxis"]["min"] == pytest.approx(9.96)
    assert options["xAxis"]["max"] == pytest.approx(11.04)
    assert options["yAxis"]["min"] == pytest.approx(0.0)
    assert options["yAxis"]["max"] == pytest.approx(10.0)
    assert options["xAxis"]["axisLabel"]["customValues"] == pytest.approx(
        [10.0, 10.2, 10.4, 10.6, 10.8, 11.0]
    )
    assert options["xAxis"]["splitLine"]["showMinLine"] is True
    assert options["xAxis"]["splitLine"]["showMaxLine"] is True
    assert options["series"][0]["data"] == [
        [10.0, 0.0, 11.0, 10.0],
    ]
    assert "data:image/png;base64," in options["series"][0][":renderItem"]
    assert options["series"][0]["silent"] is True
    assert options["tooltip"]["show"] is False
    assert options["visualMap"]["text"] == ["1", "0"]
    assert options["graphic"][0]["style"]["text"] == "Replicated fraction"
    assert options["title"]["show"] is False


@pytest.mark.parametrize(
    ("output_key", "values", "expected"),
    (
        ("fork_directionality", [-0.45, 0.0, 0.6], (-1.0, 1.0)),
        ("origin_efficiency", [0.2, 0.7, 0.4], (0.0, 1.0)),
        ("replicated_fraction", [0.1, 0.5, 0.9], (0.0, 1.0)),
    ),
)
def test_mapping_line_defaults_preserve_fixed_scientific_domains(
    output_key: str,
    values: list[float],
    expected: tuple[float, float],
) -> None:
    from app.pages.mapping import _mapping_automatic_y_limits, _mapping_plot_options

    profile = np.asarray(values, dtype=float)
    result = MappingResult(
        target=output_key,
        target_key=output_key,
        target_label=output_key.replace("_", " ").title(),
        method="test",
        output_form="x",
        output_dependence="space",
        primary_output=output_key,
        outputs={output_key: profile},
        output_forms={output_key: "x"},
        x_axis=np.arange(profile.size, dtype=float),
        time_axis=np.asarray([], dtype=float),
        region_labels=(),
        elapsed_seconds=0.01,
        assumptions={},
    )

    limits = _mapping_automatic_y_limits(
        result,
        output_key=output_key,
        plot_type="Line",
        histogram_bins=20,
        density=False,
        log_y=False,
    )
    assert limits == expected
    histogram_limits = _mapping_automatic_y_limits(
        result,
        output_key=output_key,
        plot_type="Histogram",
        histogram_bins=3,
        density=False,
        log_y=False,
    )
    assert histogram_limits[0] == 0.0
    assert histogram_limits[1] > 0.0
    assert histogram_limits != expected

    options = _mapping_plot_options(
        result,
        output_key=output_key,
        plot_type="Line",
        normalization="None",
        title=result.target_label,
        x_label="Position (kb)",
        y_label=result.target_label,
        color="#2c7f77",
        palette="viridis",
        histogram_bins=20,
        line_width=1.8,
        show_markers=False,
        show_colorbar=False,
        log_y=False,
        invert_y=False,
        y_limits=limits,
    )
    assert options["yAxis"]["min"] == expected[0]
    assert options["yAxis"]["max"] == expected[1]
    assert options["yAxis"]["axisLabel"]["showMinLabel"] is True
    assert options["yAxis"]["axisLabel"]["showMaxLabel"] is True


def test_mapping_dynamic_line_limits_follow_data_not_convenient_ticks_or_zero() -> None:
    from app.pages.mapping import _mapping_automatic_y_limits

    values = np.asarray([6.4, 7.3, 10.2])
    result = MappingResult(
        target="replication_timing",
        target_key="replication_timing",
        target_label="Replication timing",
        method="test",
        output_form="x",
        output_dependence="space",
        primary_output="replication_timing",
        outputs={"replication_timing": values},
        output_forms={"replication_timing": "x"},
        x_axis=np.arange(values.size, dtype=float),
        time_axis=np.asarray([], dtype=float),
        region_labels=(),
        elapsed_seconds=0.01,
        assumptions={},
    )

    limits = _mapping_automatic_y_limits(
        result,
        output_key="replication_timing",
        plot_type="Line",
        histogram_bins=20,
        density=False,
        log_y=False,
    )
    assert limits == pytest.approx((6.248, 10.352))


def test_square_space_time_heatmap_uses_position_x_and_time_y() -> None:
    from app.pages.mapping import _mapping_plot_options

    result = MappingResult(
        target="replicated_fraction_xt",
        target_key="replicated_fraction",
        target_label="Replicated fraction",
        method="test",
        output_form="xt",
        output_dependence="spacetime",
        primary_output="replicated_fraction_xt",
        outputs={"replicated_fraction_xt": np.asarray([[1.0, 2.0], [3.0, 4.0]])},
        output_forms={"replicated_fraction_xt": "xt"},
        x_axis=np.asarray([10.0, 11.0]),
        time_axis=np.asarray([0.0, 5.0]),
        region_labels=(),
        elapsed_seconds=0.01,
        assumptions={},
    )
    options = _mapping_plot_options(
        result,
        output_key="replicated_fraction_xt",
        plot_type="Heatmap",
        normalization="None",
        title="Replicated fraction",
        x_label="Chromosome position (kb)",
        y_label="Time (min)",
        color="#2c7f77",
        palette="viridis",
        histogram_bins=20,
        line_width=1.8,
        show_markers=False,
        show_colorbar=True,
        log_y=False,
        invert_y=False,
    )

    assert options["series"][0]["type"] == "custom"
    assert options["xAxis"]["name"] == "Chromosome position (kb)"
    assert options["yAxis"]["name"] == "Time (min)"
    assert options["xAxis"]["type"] == "value"
    assert options["yAxis"]["type"] == "value"
    assert options["series"][0]["data"] == [
        [10.0, 0.0, 11.0, 5.0],
    ]
    assert options["series"][0]["silent"] is True
    assert options["tooltip"]["show"] is False


def test_log_z_uses_logarithmic_colours_and_masks_nonpositive_values() -> None:
    from app.preview_plotting import build_preview_plot_options

    options = build_preview_plot_options(
        plot_type="Heatmap",
        title="Kinetic field",
        x_label="Chromosome position (kb)",
        y_label="Time (min)",
        color="#2c7f77",
        palette="viridis",
        histogram_bins=60,
        density=False,
        show_legend=False,
        show_colorbar=True,
        log_y=False,
        log_z=True,
        invert_y=False,
        y_limits=None,
        series_label="Firing rate",
        values=np.asarray([[0.0, 1.0], [10.0, 100.0]]),
        x_axis=np.asarray([10.0, 11.0]),
        y_axis=np.asarray([0.0, 5.0]),
    )

    assert options["series"][0]["type"] == "custom"
    assert "data:image/png;base64," in options["series"][0][":renderItem"]
    assert options["visualMap"]["min"] == pytest.approx(0.0)
    assert options["visualMap"]["max"] == pytest.approx(2.0)
    assert options["visualMap"]["text"] == ["100", "1"]
    assert options["graphic"][0]["style"]["text"] == "Firing rate (log scale)"
    assert ":formatter" in options["visualMap"]


def test_heatmap_frame_ends_at_the_last_time_without_clipping_labels() -> None:
    from app.preview_plotting import build_preview_plot_options

    time_axis = np.arange(0.0, 1_380.0, 5.0)
    options = build_preview_plot_options(
        plot_type="Heatmap",
        title="Firing rate",
        x_label="Chromosome position (kb)",
        y_label="Time (min)",
        color="#2c7f77",
        palette="viridis",
        histogram_bins=60,
        density=False,
        show_legend=False,
        show_colorbar=True,
        log_y=False,
        invert_y=False,
        y_limits=None,
        series_label="Firing rate",
        values=np.ones((time_axis.size, 10)),
        x_axis=np.arange(10, dtype=float),
        y_axis=time_axis,
    )

    assert options["yAxis"]["min"] == pytest.approx(time_axis[0])
    assert options["yAxis"]["max"] == pytest.approx(time_axis[-1] * 1.04)
    assert options["series"][0]["data"][0][3] == pytest.approx(time_axis[-1])
    assert options["yAxis"]["axisLabel"]["showMinLabel"] is True
    assert options["yAxis"]["axisLabel"]["showMaxLabel"] is True
    assert options["yAxis"]["splitLine"]["showMinLine"] is True
    assert options["yAxis"]["splitLine"]["showMaxLine"] is True
    assert options["xAxis"]["axisTick"]["show"] is True
    assert options["yAxis"]["axisTick"]["show"] is True
    assert options["grid"]["bottom"] >= 40
    assert options["grid"]["left"] >= options["yAxis"]["nameGap"]
    assert options["grid"]["containLabel"] is False
    assert options["grid"]["right"] == 86
    assert options["visualMap"]["right"] == 24
    assert options["visualMap"]["textStyle"]["align"] == "center"
    assert options["graphic"][0]["right"] == 2
    assert options["graphic"][0]["rotation"] == pytest.approx(np.pi / 2)


def test_axis_log_and_inversion_options_apply_to_each_numeric_dimension() -> None:
    from app.preview_plotting import build_preview_plot_options

    line = build_preview_plot_options(
        plot_type="Line",
        title="Profile",
        x_label="Position",
        y_label="Value",
        color="#2c7f77",
        palette="viridis",
        histogram_bins=60,
        density=False,
        show_legend=False,
        show_colorbar=True,
        log_y=True,
        invert_y=True,
        y_limits=None,
        series_label="Profile",
        values=np.asarray([0.0, 10.0, 100.0]),
        x_axis=np.asarray([0.0, 1.0, 10.0]),
        log_x=True,
        invert_x=True,
    )
    assert line["xAxis"]["type"] == "log"
    assert line["xAxis"]["min"] < 1.0
    assert line["xAxis"]["max"] > 10.0
    assert line["xAxis"]["inverse"] is True
    assert line["yAxis"]["type"] == "log"
    assert line["yAxis"]["inverse"] is True

    heatmap = build_preview_plot_options(
        plot_type="Heatmap",
        title="Field",
        x_label="Position",
        y_label="Time",
        color="#2c7f77",
        palette="viridis",
        histogram_bins=60,
        density=False,
        show_legend=False,
        show_colorbar=True,
        log_y=False,
        invert_y=False,
        y_limits=None,
        series_label="Field",
        values=np.asarray([[1.0, 2.0], [3.0, 4.0]]),
        x_axis=np.asarray([1.0, 10.0]),
        y_axis=np.asarray([1.0, 10.0]),
        invert_z=True,
    )
    assert heatmap["visualMap"]["inRange"]["color"] == list(
        reversed(heatmap_palette := ["#440154", "#31688e", "#35b779", "#fde725"])
    )
    assert heatmap_palette[0] == "#440154"


def test_large_heatmap_is_block_averaged_to_a_bounded_static_raster() -> None:
    from app.preview_plotting import (
        MAX_HEATMAP_CELLS,
        MAX_HEATMAP_COLUMNS,
        MAX_HEATMAP_ROWS,
        _sample_heatmap,
        build_preview_plot_options,
    )

    values = np.arange(400 * 600, dtype=float).reshape(400, 600)
    options = build_preview_plot_options(
        plot_type="Heatmap",
        title="Kinetic field",
        x_label="Chromosome position (kb)",
        y_label="Time (min)",
        color="#2c7f77",
        palette="viridis",
        histogram_bins=60,
        density=False,
        show_legend=False,
        show_colorbar=True,
        log_y=False,
        invert_y=False,
        y_limits=None,
        series_label="Firing rate",
        values=values,
        x_axis=np.arange(values.shape[1], dtype=float),
        y_axis=np.arange(values.shape[0], dtype=float),
        interactive=False,
    )

    _, _, raster, _, _ = _sample_heatmap(
        values,
        np.arange(values.shape[1], dtype=float),
        np.arange(values.shape[0], dtype=float),
    )
    assert raster.shape[0] <= MAX_HEATMAP_ROWS
    assert raster.shape[1] <= MAX_HEATMAP_COLUMNS
    assert raster.size <= MAX_HEATMAP_CELLS
    assert options["series"][0]["type"] == "custom"
    assert len(options["series"][0]["data"]) == 1
    assert "data:image/png;base64," in options["series"][0][":renderItem"]
    assert options["series"][0]["silent"] is True
    assert options["tooltip"]["show"] is False


def test_numeric_plot_axes_use_exact_padded_limits_and_compact_tick_formatters() -> None:
    from app.preview_plotting import build_preview_plot_options

    options = build_preview_plot_options(
        plot_type="Line",
        title="Profile",
        x_label="Position (kb)",
        y_label="Value",
        color="#2c7f77",
        palette="viridis",
        histogram_bins=60,
        density=False,
        show_legend=False,
        show_colorbar=True,
        log_y=False,
        invert_y=False,
        y_limits=None,
        series_label="Profile",
        values=np.asarray([0.19, 0.27, 0.36]),
        x_axis=np.asarray([0.13, 0.47, 0.81]),
    )

    assert options["xAxis"]["min"] == pytest.approx(0.1028)
    assert options["xAxis"]["max"] == pytest.approx(0.8372)
    assert options["xAxis"]["interval"] is None
    assert options["yAxis"]["min"] == pytest.approx(0.1832)
    assert options["yAxis"]["max"] == pytest.approx(0.3668)
    assert options["yAxis"]["interval"] is None
    assert options["xAxis"]["axisLabel"]["customValues"] == pytest.approx([0.2, 0.4, 0.6, 0.8])
    assert options["yAxis"]["axisLabel"]["customValues"] == pytest.approx([0.2, 0.25, 0.3, 0.35])
    assert options["xAxis"]["axisLabel"]["showMinLabel"] is True
    assert options["xAxis"]["axisLabel"]["showMaxLabel"] is True
    assert options["yAxis"]["axisLabel"]["showMinLabel"] is True
    assert options["yAxis"]["axisLabel"]["showMaxLabel"] is True
    assert ":formatter" in options["xAxis"]["axisLabel"]
    assert ":formatter" in options["yAxis"]["axisLabel"]


def test_fit_comparison_preview_overlays_observed_and_fitted_profiles() -> None:
    from app.pages.mapping import _mapping_plot_options

    result = MappingResult(
        target="firing_rate",
        target_key="firing_rate",
        target_label="Firing rate",
        method="test_fit",
        output_form="scalar",
        output_dependence="constant",
        primary_output="firing_rate",
        outputs={
            "firing_rate": np.asarray([0.001]),
            "replication_timing_fit": np.asarray([8.2, 7.1, 9.1]),
        },
        output_forms={"firing_rate": "scalar", "replication_timing_fit": "x"},
        x_axis=np.asarray([10.0, 11.0, 12.0]),
        time_axis=np.asarray([], dtype=float),
        region_labels=(),
        elapsed_seconds=0.01,
        assumptions={},
        comparisons={
            "replication_timing_fit": MappingComparison(
                source_key="replication_timing",
                fitted_key="replication_timing_fit",
                form="x",
                observed=np.asarray([8.0, 7.0, 9.0]),
                fitted=np.asarray([8.2, 7.1, 9.1]),
            )
        },
    )
    options = _mapping_plot_options(
        result,
        output_key="comparison:replication_timing_fit",
        plot_type="Line",
        normalization="None",
        title="Replication timing: observed vs fitted",
        x_label="Position (kb)",
        y_label="Replication timing",
        color="#2c7f77",
        palette="viridis",
        histogram_bins=20,
        line_width=1.8,
        show_markers=False,
        show_colorbar=True,
        log_y=False,
        invert_y=False,
    )

    assert options["legend"]["data"] == ["Observed", "Fitted"]
    assert [series["name"] for series in options["series"]] == ["Observed", "Fitted"]
    assert options["series"][0]["lineStyle"]["color"] == "#2c7f77"
    assert options["series"][1]["lineStyle"]["color"] == "#b03412"
    assert options["series"][0]["data"] == [[10.0, 8.0], [11.0, 7.0], [12.0, 9.0]]
    assert options["series"][1]["data"] == [[10.0, 8.2], [11.0, 7.1], [12.0, 9.1]]


def test_fit_comparison_is_selectable_in_the_live_mapping_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.pages import mapping as mapping_page

    def fitted_result(request: MappingRequest, report: Any, _cancel: Any) -> MappingResult:
        report(0.9, "Preparing mapped preview...")
        observed = np.asarray([8.0, 7.0, 9.0])
        fitted = np.asarray([8.2, 7.1, 9.1])
        return MappingResult(
            target=request.route.target,
            target_key=request.route.target_key,
            target_label="Firing rate",
            method=request.route.method,
            output_form="x",
            output_dependence="space",
            primary_output="firing_rate",
            outputs={
                "firing_rate": np.asarray([0.1, 0.2, 0.3]),
                "replication_timing_fit": fitted,
            },
            output_forms={"firing_rate": "x", "replication_timing_fit": "x"},
            x_axis=np.asarray([0.0, 1.0, 2.0]),
            time_axis=np.asarray([], dtype=float),
            region_labels=(),
            elapsed_seconds=0.03,
            assumptions={},
            comparisons={
                "replication_timing_fit": MappingComparison(
                    source_key="replication_timing",
                    fitted_key="replication_timing_fit",
                    form="x",
                    observed=observed,
                    fitted=fitted,
                )
            },
        )

    monkeypatch.setattr(mapping_page, "run_mapping", fitted_result)

    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-mapping").click()
            await user.should_see(marker="source-replication-timing-upload", retries=60)
            uploader = cast(
                Upload,
                next(iter(user.find(marker="source-replication-timing-upload").elements)),
            )
            await uploader.handle_uploads(
                [Upload.SmallFileUpload("timing.csv", "text/csv", b"timing\n8\n7\n9\n")]
            )
            await user.should_see("timing.csv", retries=20)
            output = cast(
                Select,
                next(iter(user.find(marker="mapping-preview-output").elements)),
            )
            comparison_key = "comparison:replication_timing_fit"
            assert comparison_key in output.options
            assert user.client is not None
            with user.client:
                output.set_value(comparison_key)
            await asyncio.sleep(0)
            user.find(marker="action-map").click()
            await user.should_see(marker="mapping-result-plot", retries=30)

            assert output.value == comparison_key

            view = cast(
                Select,
                next(iter(user.find(marker="mapping-view").elements)),
            )
            chart = cast(
                EChart,
                next(iter(user.find(marker="mapping-result-plot").elements)),
            )
            assert view.value == "Default"
            assert not view.enabled
            assert [series["name"] for series in chart.options["series"]] == [
                "Observed",
                "Fitted",
            ]
            first_color = cast(
                Input,
                next(iter(user.find(marker="mapping-line-or-bar-color").elements)),
            )
            second_color = cast(
                Input,
                next(iter(user.find(marker="mapping-second-curve-color").elements)),
            )
            assert first_color.visible
            assert second_color.visible
            with user.client:
                first_color.set_value("#2467a6")
                second_color.set_value("#c43b22")
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            chart = cast(
                EChart,
                next(iter(user.find(marker="mapping-result-plot").elements)),
            )
            assert chart.options["series"][0]["lineStyle"]["color"] == "#2467a6"
            assert chart.options["series"][1]["lineStyle"]["color"] == "#c43b22"

    asyncio.run(scenario())
