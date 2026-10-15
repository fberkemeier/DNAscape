"""End-to-end NiceGUI coverage for the live Mapping action."""

from __future__ import annotations

from app.pages.common import workflow_action_icon

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


def test_whole_region_end_uses_binned_track_extent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.pages import mapping as mapping_page

    requests: list[MappingRequest] = []

    def fake_run(request: MappingRequest, _report: Any, _cancel: Any) -> MappingResult:
        requests.append(request)
        values = np.asarray(request.inputs["replication_timing"])
        assert values.size == 6
        return _mapped_result(request, np.full(values.size, 0.001))

    monkeypatch.setattr(mapping_page, "run_mapping", fake_run)

    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-mapping").click()
            await user.should_see(marker="mapping-target", retries=60)
            uploader = cast(
                Upload,
                next(iter(user.find(marker="source-replication-timing-upload").elements)),
            )
            await uploader.handle_uploads([
                Upload.SmallFileUpload(
                    "timing.bedgraph", "text/plain",
                    b"chr1 0 2000 8.0\nchr1 2000 6000 6.0\n",
                ),
            ])
            await user.should_see(
                marker="mapping-input-replication-timing-track-options", retries=40,
            )
            user.find("Region").click()
            end = cast(Number, next(iter(user.find(marker="mapping-region-end").elements)))
            with user.client:
                end.set_value(12.0)
            user.find(marker="mapping-whole-region").click()
            assert end.value == 12.0  # Two track records do not mean two bins.
            user.find(marker="action-map").click()
            await user.should_see(marker="mapping-result-plot", retries=60)
            assert end.value == 6.0
            assert len(requests) == 1
            assert requests[0].whole_genome is True

    asyncio.run(scenario())


def test_combined_mapping_colors_every_curve_and_limits_dense_hover(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.pages import mapping as mapping_page

    def fake_options(_result: Any, **_kwargs: Any) -> dict[str, Any]:
        return {
            "xAxis": {"min": 0.0, "max": 2.0},
            "tooltip": {
                "show": True,
                "axisPointer": {"label": {"show": True}},
            },
            "legend": {"selectedMode": True},
            "series": [
                {
                    "type": "line",
                    "name": role,
                    "lineStyle": {},
                    "itemStyle": {},
                    "silent": False,
                    "emphasis": {"disabled": False},
                    "data": [[0.0, 1.0], [1.0, 2.0]],
                }
                for role in ("Observed", "Fitted")
            ],
        }

    monkeypatch.setattr(mapping_page, "_mapping_plot_options", fake_options)
    named = tuple(
        mapping_page._NamedMappingResult(
            f"mapping-case-{index}",
            f"Firing rate · Input: sample_{index}",
            f"firing-rate-sample-{index}",
            cast(MappingResult, object()),
            f"Firing rate · sample_{index}",
            f"firing-rate-sample-{index}",
        )
        for index in range(1, 8)
    )
    color_keys = tuple(
        (item.key, "firing_rate", series_index)
        for item in named
        for series_index in range(2)
    )
    colors = {
        color_key: f"#{index:06x}"
        for index, color_key in enumerate(color_keys, start=1)
    }

    options = mapping_page._combined_mapping_plot_options(
        named,
        colors=colors,
        output_key="firing_rate",
    )

    assert [item["lineStyle"]["color"] for item in options["series"]] == list(
        colors.values()
    )
    assert options["tooltip"]["show"] is False
    assert all(item["silent"] is True for item in options["series"])
    assert options["legend"]["selectedMode"] is True


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


def test_mapping_rejects_empty_rows_then_fills_them_without_moving_bins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.pages import mapping as mapping_page

    requests: list[MappingRequest] = []

    def fake_run(request: MappingRequest, _report: Any, _cancel: Any) -> MappingResult:
        requests.append(request)
        values = np.asarray(request.inputs["replication_timing"], dtype=float)
        return _mapped_result(request, values)

    monkeypatch.setattr(mapping_page, "run_mapping", fake_run)

    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-mapping").click()
            await user.should_see(marker="source-replication-timing-upload", retries=60)
            uploader = cast(
                Upload, next(iter(user.find(marker="source-replication-timing-upload").elements))
            )
            await uploader.handle_uploads(
                [
                    Upload.SmallFileUpload(
                        "timing.tsv", "text/tab-separated-values", b'chr1\n""\n100\n""\n200\n""\n'
                    )
                ]
            )
            await user.should_see(marker="mapping-input-replication-timing-detected-data")
            missing = cast(
                Select, next(iter(user.find(marker="mapping-region-missing").elements))
            )
            assert missing.value == "error"
            user.find(marker="action-map").click()
            await user.should_see("The selected data contain missing values", retries=60)
            assert requests == []

            missing.set_value("zero")
            await asyncio.sleep(0)
            user.find(marker="action-map").click()
            await user.should_see(marker="mapping-result-plot", retries=60)
            assert len(requests) == 1
            np.testing.assert_array_equal(
                requests[0].inputs["replication_timing"], [0, 100, 0, 200, 0]
            )

    asyncio.run(scenario())


def test_mapping_all_value_columns_expands_one_table_into_profile_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.pages import mapping as mapping_page

    requests: list[MappingRequest] = []

    def fake_run(request: MappingRequest, report: Any, _cancel: Any) -> MappingResult:
        requests.append(request)
        values = np.asarray(request.inputs["replication_timing"], dtype=float)
        report(1.0, "Mapped profile")
        return _mapped_result(request, values)

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
                        "timing.tsv",
                        "text/tab-separated-values",
                        b"chr1\tchr2\n1\t9\n2\t8\n3\t7\n",
                    )
                ]
            )
            await user.should_see(
                marker="mapping-input-replication-timing-detected-data",
                retries=40,
            )
            await user.should_see(
                marker="mapping-input-replication-timing-value-column",
                retries=40,
            )
            column = cast(
                Select,
                next(
                    iter(user.find(marker="mapping-input-replication-timing-value-column").elements)
                ),
            )
            assert tuple(column.options) == ("__all__", "chr1", "chr2")
            assert column.options["__all__"] == "All"
            assert user.client is not None
            with user.client:
                column.set_value("__all__")
            user.find(marker="action-map").click()
            await user.should_see("Complete · 2 datasets", retries=60)

            assert len(requests) == 2
            np.testing.assert_array_equal(
                requests[0].inputs["replication_timing"],
                [1.0, 2.0, 3.0],
            )
            np.testing.assert_array_equal(
                requests[1].inputs["replication_timing"],
                [9.0, 8.0, 7.0],
            )

    asyncio.run(scenario())


def test_mapping_batch_runs_all_files_tabs_exports_and_preserves_previous_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.pages import mapping as mapping_page

    requests: list[MappingRequest] = []
    archived_counts: list[int] = []
    real_archive = mapping_page.build_artifact_archive

    def fake_run(request: MappingRequest, report: Any, _cancel: Any) -> MappingResult:
        requests.append(request)
        report(0.6, "Fitting batch dataset...")
        values = np.asarray(request.inputs["replication_timing"], dtype=float)
        result = _mapped_result(request, values / 10.0)
        unique_key = "first_only" if len(requests) == 1 else "second_only"
        return replace(
            result,
            outputs={**result.outputs, unique_key: values},
            output_forms={**result.output_forms, unique_key: "x"},
        )

    def track_archive(artifacts: Any, **kwargs: Any) -> Any:
        archived_counts.append(len(artifacts))
        return real_archive(artifacts, **kwargs)

    monkeypatch.setattr(mapping_page, "run_mapping", fake_run)
    monkeypatch.setattr(mapping_page, "build_artifact_archive", track_archive)

    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-mapping").click()
            await user.should_see(marker="source-replication-timing-upload", retries=60)
            await user.should_not_see(marker="mapping-batch-datasets")

            uploader = cast(
                Upload,
                next(iter(user.find(marker="source-replication-timing-upload").elements)),
            )
            assert uploader._props["multiple"] is True
            await uploader.handle_uploads(
                [
                    Upload.SmallFileUpload(
                        "timing_a.csv",
                        "text/csv",
                        b"timing\n8\n7\n9\n",
                    ),
                    Upload.SmallFileUpload(
                        "timing_b.csv",
                        "text/csv",
                        b"timing\n6\n5\n7\n8\n",
                    ),
                ]
            )
            await user.should_see("Datasets: 2", retries=40)

            user.find(marker="action-map").click()
            await user.should_see("Complete · 2 datasets", retries=60)
            assert len(requests) == 2
            np.testing.assert_array_equal(
                requests[0].inputs["replication_timing"],
                [8.0, 7.0, 9.0],
            )
            np.testing.assert_array_equal(
                requests[1].inputs["replication_timing"],
                [6.0, 5.0, 7.0, 8.0],
            )
            await user.should_see(marker="mapping-batch-raw-files")
            await user.should_see(marker="mapping-download-all-raw")

            layout = cast(
                Select,
                next(iter(user.find(marker="mapping-preview-layout").elements)),
            )
            output = cast(
                Select,
                next(iter(user.find(marker="mapping-preview-output").elements)),
            )
            assert layout.value == "Combined"
            assert tuple(output.options) == ("firing_rate", "input:replication_timing")

            # Histograms share one binning configuration and may be overlaid
            # just like line profiles. Individual selection remains available.
            view = cast(
                Select,
                next(iter(user.find(marker="mapping-view").elements)),
            )
            with user.client:
                view.set_value("Histogram")
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            assert layout.value == "Combined"
            assert list(layout.options) == ["Combined", "Individual"]
            await user.should_not_see(marker="mapping-dataset-select")
            histogram_chart: EChart | None = None
            for _ in range(40):
                histogram_chart = cast(
                    EChart,
                    next(iter(user.find(marker="mapping-result-plot").elements)),
                )
                if all(
                    item.get("type") == "custom"
                    for item in histogram_chart.options.get("series", ())
                ):
                    break
                await asyncio.sleep(0.01)
            assert histogram_chart is not None
            assert len(histogram_chart.options["series"]) == 2
            assert all(
                item["type"] == "custom" for item in histogram_chart.options["series"]
            ), (view.value, histogram_chart.options["series"])
            assert histogram_chart.options["legend"]["show"] is True
            density = cast(
                Checkbox,
                next(iter(user.find(marker="mapping-density").elements)),
            )
            touching = cast(
                Checkbox,
                next(iter(user.find(marker="mapping-histogram-touching").elements)),
            )
            bins = cast(
                Number,
                next(iter(user.find(marker="mapping-histogram-bins").elements)),
            )
            with user.client:
                density.set_value(True)
                touching.set_value(False)
                bins.set_value(2)
            for _ in range(40):
                histogram_chart = cast(
                    EChart,
                    next(iter(user.find(marker="mapping-result-plot").elements)),
                )
                if (
                    histogram_chart.options.get("yAxis", {}).get("name") == "Density"
                    and all(
                        item.get("type") == "line"
                        and len(item.get("data", ())) == 256
                        for item in histogram_chart.options.get("series", ())
                    )
                ):
                    break
                await asyncio.sleep(0.01)
            assert histogram_chart.options["yAxis"]["name"] == "Density"
            assert all(
                item["type"] == "line" and len(item["data"]) == 256
                for item in histogram_chart.options["series"]
            )
            with user.client:
                view.set_value("Default")
            await asyncio.sleep(0)
            assert list(layout.options) == ["Combined", "Individual"]
            with user.client:
                layout.set_value("Individual")
            await user.should_see(marker="mapping-dataset-select", retries=20)
            dataset_select = cast(
                Select,
                next(iter(user.find(marker="mapping-dataset-select").elements)),
            )
            assert tuple(dataset_select.options) == (
                "mapping-case-1",
                "mapping-case-2",
            )
            assert tuple(dataset_select.options.values()) == (
                "Firing rate · timing_a",
                "Firing rate · timing_b",
            )
            with user.client:
                dataset_select.set_value("mapping-case-2")
            await asyncio.sleep(0)
            # Dataset selection must retain the submit-time output
            # intersection; result-specific outputs cannot leak into a batch.
            assert tuple(output.options) == ("firing_rate", "input:replication_timing")

            user.find(marker="mapping-download-all-raw").click()
            for _ in range(40):
                if archived_counts:
                    break
                await asyncio.sleep(0.02)
            assert archived_counts == [2]

            # Same-name replacement keeps its original pairing position.
            await uploader.handle_uploads(
                [
                    Upload.SmallFileUpload(
                        "timing_a.csv",
                        "text/csv",
                        b"timing\n10\n11\n12\n",
                    )
                ]
            )
            await asyncio.sleep(0.25)
            user.find(marker="action-map").click()
            for _ in range(80):
                if len(requests) >= 4:
                    break
                await asyncio.sleep(0.02)
            assert len(requests) == 4
            np.testing.assert_array_equal(
                requests[2].inputs["replication_timing"],
                [10.0, 11.0, 12.0],
            )
            np.testing.assert_array_equal(
                requests[3].inputs["replication_timing"],
                [6.0, 5.0, 7.0, 8.0],
            )

            # Replace one retained dataset with an incompatible matrix. The
            # click-time consistency check must stop before another worker and
            # leave the already published two-dataset preview intact.
            await uploader.handle_uploads(
                [
                    Upload.SmallFileUpload(
                        "timing_a.csv",
                        "text/csv",
                        b"position,t0,t1\n0,1,2\n1,3,4\n2,5,6\n",
                    )
                ]
            )
            await asyncio.sleep(0.1)
            user.find(marker="action-map").click()
            await asyncio.sleep(0.1)
            await user.should_see("Mapping could not be completed", retries=40)
            assert len(requests) == 4
            await user.should_see(marker="mapping-result-plot")
            await user.should_see(marker="mapping-batch-raw-files")

    asyncio.run(scenario())


def test_mapping_batch_submit_plan_rejects_mismatched_counts_and_forms() -> None:
    from app.pages.mapping import (
        _plan_mapping_batch,
        _PreparedBatchInput,
        _validate_paired_batch_axes,
    )

    def prepared(
        label: str,
        filenames: tuple[str, ...],
        forms: tuple[str | None, ...],
    ) -> _PreparedBatchInput:
        return _PreparedBatchInput(
            key=label.casefold().replace(" ", "_"),
            label=label,
            form="x",
            values=tuple(np.arange(index + 2, dtype=float) for index in range(len(filenames))),
            prewindowed=(False,) * len(filenames),
            filenames=filenames,
            detected_forms=forms,
        )

    with pytest.raises(ValueError, match="needs either one shared file or 3 paired files"):
        _plan_mapping_batch(
            (
                prepared("Timing", ("a.csv", "b.csv"), ("x", "x")),
                prepared(
                    "Speed",
                    ("a.csv", "b.csv", "c.csv"),
                    ("x", "x", "x"),
                ),
            )
        )

    with pytest.raises(ValueError, match="do not have one consistent data type"):
        _plan_mapping_batch(
            (
                prepared(
                    "Timing",
                    ("timing_a.csv", "timing_b.csv"),
                    ("x", "xt"),
                ),
            )
        )

    constants = _PreparedBatchInput(
        key="fork_speed",
        label="Fork speed",
        form="scalar",
        values=(1.0, 1.2, 1.4),
        prewindowed=(False, False, False),
    )
    plan = _plan_mapping_batch((constants,))
    assert plan.size == 3
    assert plan.names == ("Dataset 1", "Dataset 2", "Dataset 3")

    with pytest.raises(ValueError, match="one shared value/file or 3 paired"):
        _plan_mapping_batch(
            (
                constants,
                replace(constants, key="firing_rate", label="Firing rate", values=(0.1, 0.2)),
            )
        )

    # Spatial factors may differ within a case. The Mapping service selects
    # their shared length after applying the requested genomic interval.
    timing = _PreparedBatchInput(
        key="replication_timing",
        label="Replication timing",
        form="x",
        values=(np.arange(3.0), np.arange(5.0)),
        prewindowed=(False, False),
        filenames=("short.tsv", "long.tsv"),
        detected_forms=("x", "x"),
    )
    speed = _PreparedBatchInput(
        key="fork_speed",
        label="Fork speed",
        form="x",
        values=(np.arange(3.0), np.arange(5.0)),
        prewindowed=(False, False),
        filenames=("short_speed.tsv", "long_speed.tsv"),
        detected_forms=("x", "x"),
    )
    assert _plan_mapping_batch((timing, speed)).size == 2
    _validate_paired_batch_axes((timing, speed), 0, "short")
    _validate_paired_batch_axes((timing, speed), 1, "long")
    _validate_paired_batch_axes(
        (timing, replace(speed, values=(np.arange(3.0), np.arange(4.0)))),
        1,
        "long",
    )


@pytest.mark.parametrize("firing_form", ["scalar", "x"])
def test_spatial_speed_submit_aligns_profiles_and_keeps_selected_boundaries(
    monkeypatch: pytest.MonkeyPatch,
    firing_form: str,
) -> None:
    import dnascape
    from app.mapping_service import run_mapping
    from app.pages import mapping as mapping_page

    requests: list[MappingRequest] = []
    calls: list[dict[str, Any]] = []
    results: list[MappingResult] = []
    timing = 10.0 + np.arange(30, dtype=float) / 10.0
    local_firing = np.full(10, 0.01)
    whole_finished = threading.Event()
    selected_end = 36.0 if firing_form == "scalar" else 44.0
    expected_bins = 6 if firing_form == "scalar" else 10

    def fake_rmap(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {"values": np.full(kwargs["source"]["replication_timing"].size, 1.4)}

    def recording_run(request: MappingRequest, report: Any, cancel: Any) -> MappingResult:
        requests.append(request)
        result = run_mapping(request, report, cancel)
        results.append(result)
        if len(results) == 2:
            whole_finished.set()
        return result

    monkeypatch.setattr(dnascape, "rmap", fake_rmap)
    monkeypatch.setattr(mapping_page, "run_mapping", recording_run)

    async def scenario() -> None:
        async with user_simulation(index) as user:
            def control(marker: str) -> Any:
                return next(iter(user.find(marker=marker).elements))

            async def set_value(marker: str, value: Any) -> None:
                assert user.client is not None
                with user.client:
                    control(marker).set_value(value)
                await asyncio.sleep(0)
                await asyncio.sleep(0)

            async def upload(marker: str, name: str, values: np.ndarray) -> None:
                content = "value\n" + "\n".join(str(value) for value in values) + "\n"
                await cast(Upload, control(marker)).handle_uploads(
                    [Upload.SmallFileUpload(name, "text/csv", content.encode("utf-8"))]
                )
                input_marker = marker.removeprefix("source-").removesuffix("-upload")
                await user.should_see(
                    marker=f"mapping-input-{input_marker}-detected-data", retries=40
                )

            await user.open("/")
            user.find(marker="nav-mapping").click()
            await user.should_see(marker="mapping-target", retries=60)
            await set_value("mapping-target", "fork_speed")
            await set_value("mapping-target-form", "space")
            assert control("mapping-route").value == (
                "fork_speed--space--replication_timing-and-firing_rate"
            )
            await set_value("mapping-region-start", 24.0)
            await set_value("mapping-region-end", selected_end)
            await set_value("mapping-setting-resolution", 2.0)
            await upload("source-replication-timing-upload", "timing.csv", timing)
            if firing_form == "x":
                await set_value("mapping-input-firing-rate-type", "profile")
                await upload("source-firing-rate-upload", "firing.csv", local_firing)
                await set_value("mapping-region-end", 56.0)
                user.find(marker="action-map").click()
                await user.should_see("choose a width of at most 20 kb", retries=60)
                assert requests == calls == results == []
                assert control("mapping-region-start").value == 24.0
                assert control("mapping-region-end").value == 56.0
                assert control("action-map").enabled
                await set_value("mapping-region-end", selected_end)
            else:
                await set_value("mapping-input-firing-rate-constant", 0.01)

            user.find(marker="action-map").click()
            await user.should_see(marker="mapping-result-plot", retries=60)
            await user.should_not_see(marker="mapping-error")
            assert len(requests) == len(calls) == len(results) == 1
            assert requests[0].route.method == "timing_and_firing_to_spatial_fork_speed"
            assert requests[0].region_start_kb == 24.0
            assert requests[0].region_end_kb == selected_end
            assert requests[0].resolution_x == 2.0
            np.testing.assert_allclose(
                calls[0]["source"]["replication_timing"], timing[12 : 12 + expected_bins]
            )
            if firing_form == "x":
                np.testing.assert_array_equal(calls[0]["source"]["firing_rate"], local_firing)
            else:
                assert calls[0]["source"]["firing_rate"] == 0.01
            np.testing.assert_array_equal(
                results[0].x_axis, 24.0 + np.arange(expected_bins) * 2.0
            )
            assert results[0].outputs[results[0].primary_output].size == expected_bins

            if firing_form == "x":
                await set_value("mapping-whole-region", True)
                assert control("mapping-region-start").value == 0.0
                assert control("mapping-region-end").value == 20.0
                user.find(marker="action-map").click()
                assert await asyncio.to_thread(whole_finished.wait, 5)
                await user.should_see("Complete in", retries=60)
                await user.should_not_see(marker="mapping-error")
                assert len(requests) == len(calls) == len(results) == 2
                assert requests[1].whole_genome is True
                np.testing.assert_array_equal(calls[1]["source"]["replication_timing"], timing[:10])
                np.testing.assert_array_equal(calls[1]["source"]["firing_rate"], local_firing)
                np.testing.assert_array_equal(results[1].x_axis, np.arange(10) * 2.0)
                assert control("mapping-region-start").value == 0.0
                assert control("mapping-region-end").value == 20.0

    asyncio.run(scenario())


def test_mapping_multiple_constants_pair_with_independent_profile_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.pages import mapping as mapping_page

    requests: list[MappingRequest] = []

    def fake_run(request: MappingRequest, report: Any, _cancel: Any) -> MappingResult:
        requests.append(request)
        report(0.8, "Mapping paired dataset...")
        values = np.asarray(request.inputs["replication_timing"], dtype=float)
        return _mapped_result(request, values)

    monkeypatch.setattr(mapping_page, "run_mapping", fake_run)

    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-mapping").click()
            await user.should_see(marker="mapping-route", retries=60)
            assert user.client is not None
            route = cast(
                Select,
                next(iter(user.find(marker="mapping-route").elements)),
            )
            with user.client:
                route.set_value("firing_rate--space--replication_timing-and-fork_speed")
            await asyncio.sleep(0)
            await asyncio.sleep(0)

            input_type = cast(
                Select,
                next(iter(user.find(marker="mapping-input-fork-speed-type").elements)),
            )
            assert tuple(input_type.options) == ("scalar", "constants", "profile")
            with user.client:
                input_type.set_value("constants")
            await asyncio.sleep(0)
            matching = cast(
                Select,
                next(
                    iter(
                        user.find(
                            marker="mapping-input-fork-speed-combination-mode"
                        ).elements
                    )
                ),
            )
            assert matching.value == "pairwise"
            constant_1 = cast(
                Number,
                next(
                    iter(user.find(marker="mapping-input-fork-speed-constant-1").elements)
                ),
            )
            constant_2 = cast(
                Number,
                next(
                    iter(user.find(marker="mapping-input-fork-speed-constant-2").elements)
                ),
            )
            with user.client:
                constant_1.set_value(1.2)
                constant_2.set_value(1.8)

            uploader = cast(
                Upload,
                next(iter(user.find(marker="source-replication-timing-upload").elements)),
            )
            await uploader.handle_uploads(
                [
                    Upload.SmallFileUpload("short.tsv", "text/tab-separated-values", b"value\n8\n7\n9\n"),
                    Upload.SmallFileUpload(
                        "long.tsv",
                        "text/tab-separated-values",
                        b"value\n6\n5\n7\n8\n",
                    ),
                ]
            )
            await user.should_see("Datasets: 2", retries=30)
            await user.should_see("short.tsv")
            await user.should_see("long.tsv")
            constant_1 = cast(
                Number,
                next(
                    iter(user.find(marker="mapping-input-fork-speed-constant-1").elements)
                ),
            )
            constant_2 = cast(
                Number,
                next(
                    iter(user.find(marker="mapping-input-fork-speed-constant-2").elements)
                ),
            )
            assert constant_1.value == pytest.approx(1.2)
            assert constant_2.value == pytest.approx(1.8)
            # A single signal column is automatic and should not consume UI space.
            await user.should_not_see(
                marker="mapping-input-replication-timing-value-column"
            )

            user.find(marker="action-map").click()
            await user.should_see(marker="mapping-result-plot", retries=60)
            assert len(requests) == 2
            assert requests[0].inputs["fork_speed"] == pytest.approx(1.2)
            assert requests[1].inputs["fork_speed"] == pytest.approx(1.8)
            assert np.asarray(requests[0].inputs["replication_timing"]).size == 3
            assert np.asarray(requests[1].inputs["replication_timing"]).size == 4
            for marker in (
                "mapping-batch-raw-type-mapping-case-1",
                "mapping-batch-raw-compression-mapping-case-1",
                "mapping-download-raw-mapping-case-1",
                "mapping-batch-raw-type-mapping-case-2",
                "mapping-batch-raw-compression-mapping-case-2",
                "mapping-download-raw-mapping-case-2",
                "mapping-download-all-raw",
            ):
                assert user.find(marker=marker).elements

    asyncio.run(scenario())


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

            await user.should_not_see("Title")
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
                x_label.set_value("Genomic position")
                y_label.set_value("Rate")
                color.set_value("#c43b22")
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            updated = chart()
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
            assert chart().options["xAxis"]["name"] == "Genomic position"
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
            assert len(histogram.options["series"][0]["data"]) == 256
            assert histogram.options["series"][0]["type"] == "line"
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
            assert len(histogram.options["series"][0]["data"]) == 256
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


def test_mapping_show_plot_can_skip_rendering_and_restore_without_remapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.pages import mapping as mapping_page

    runs: list[MappingRequest] = []
    renders: list[MappingResult] = []
    original_options = mapping_page._mapping_plot_options

    def fake_run(request: MappingRequest, _report: Any, _cancel: Any) -> MappingResult:
        runs.append(request)
        return _mapped_result(request, np.asarray([0.2, 0.4, 0.6]))

    def record_options(result: MappingResult, **kwargs: Any) -> dict[str, Any]:
        renders.append(result)
        return original_options(result, **kwargs)

    monkeypatch.setattr(mapping_page, "run_mapping", fake_run)
    monkeypatch.setattr(mapping_page, "_mapping_plot_options", record_options)

    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-mapping").click()
            await user.should_see(marker="source-replication-timing-upload", retries=60)
            show_plot = cast(Checkbox, next(iter(user.find(marker="mapping-show-plot").elements)))
            assert show_plot.value is True
            assert user.client is not None
            with user.client:
                show_plot.set_value(False)
            with user.client:
                show_plot.set_value(True)
            await user.should_see("No result")
            with user.client:
                show_plot.set_value(False)
            uploader = cast(Upload, next(iter(user.find(marker="source-replication-timing-upload").elements)))
            await uploader.handle_uploads([
                Upload.SmallFileUpload("timing.csv", "text/csv", b"timing\n8\n7\n9\n")
            ])
            await user.should_see(marker="mapping-input-replication-timing-detected-data", retries=20)
            user.find(marker="action-map").click()
            await user.should_see("Complete in 0.08 s", retries=30)
            await user.should_see(marker="mapping-stats")
            await user.should_see(marker="mapping-raw-file-type")
            assert next(iter(user.find(marker="action-download-raw-data").elements)).enabled
            await user.should_not_see(marker="mapping-result-plot")
            assert len(runs) == 1
            assert not renders
            with user.client:
                show_plot.set_value(True)
            await user.should_see(marker="mapping-result-plot", retries=20)
            assert len(runs) == 1
            assert renders
            with user.client:
                show_plot.set_value(False)
            await user.should_not_see(marker="mapping-result-plot")
            await user.should_see("Complete in 0.08 s")

    asyncio.run(scenario())


@pytest.mark.parametrize("heatmap", (False, True))
def test_mapping_axis_limits_remain_editable_and_resettable(
    monkeypatch: pytest.MonkeyPatch, heatmap: bool,
) -> None:
    from app.pages import mapping as mapping_page

    def fake_run(request: MappingRequest, _report: Any, _cancel: Any) -> MappingResult:
        result = _mapped_result(request, np.asarray([0.2, 0.4, 0.6, 0.8]))
        if heatmap:
            return replace(
                result, output_form="xt", output_dependence="spacetime",
                outputs={"firing_rate": np.arange(12.0).reshape(4, 3)},
                output_forms={"firing_rate": "xt"}, time_axis=np.asarray([0.0, 10.0, 20.0]),
            )
        return result

    monkeypatch.setattr(mapping_page, "run_mapping", fake_run)

    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-mapping").click()
            await user.should_see(marker="source-replication-timing-upload", retries=60)
            uploader = cast(Upload, next(iter(user.find(marker="source-replication-timing-upload").elements)))
            await uploader.handle_uploads([
                Upload.SmallFileUpload("timing.csv", "text/csv", b"timing\n8\n7\n9\n10\n")
            ])
            await user.should_see(marker="mapping-input-replication-timing-detected-data", retries=20)
            user.find(marker="action-map").click()
            try:
                await user.should_see(marker="mapping-result-plot", retries=30)
            except AssertionError:
                raise AssertionError(user.notify.messages) from None
            user.find("Plotting settings").click()

            def chart() -> EChart:
                return cast(EChart, next(iter(user.find(marker="mapping-result-plot").elements)))

            automatic = chart().options
            assert user.client is not None
            axes = [("x", (-1.0, 5.0)), ("y", (-10.0, 40.0))]
            if heatmap:
                axes.append(("z", (2.0, 8.0)))
            for axis, limits in axes:
                lower = cast(Number, next(iter(user.find(marker=f"mapping-{axis}-minimum").elements)))
                upper = cast(Number, next(iter(user.find(marker=f"mapping-{axis}-maximum").elements)))
                reset = cast(Button, next(iter(user.find(marker=f"mapping-reset-{axis}-limits").elements)))
                assert lower.enabled and upper.enabled and reset.enabled
                with user.client:
                    lower.set_value(limits[0])
                    upper.set_value(limits[1])
                await asyncio.sleep(0)
                displayed = chart().options["visualMap" if axis == "z" else f"{axis}Axis"]
                assert (displayed["min"], displayed["max"]) == limits
                # An incomplete/invalid edit must not prevent using Reset.
                with user.client:
                    upper.set_value(None)
                user.find(marker=f"mapping-reset-{axis}-limits").click()
                await asyncio.sleep(0)
                restored = chart().options["visualMap" if axis == "z" else f"{axis}Axis"]
                original = automatic["visualMap" if axis == "z" else f"{axis}Axis"]
                assert (restored["min"], restored["max"]) == (original["min"], original["max"])
            await user.should_not_see("Output unavailable")

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
            await user.should_see("MSE 5.250e-02", retries=30)

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
            assert raw_name.value == "rmap_fr-timing_mse"
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
            assert button.icon == workflow_action_icon("mapping")
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
    assert options["xAxis"]["min"] == pytest.approx(10.0)
    assert options["xAxis"]["max"] == pytest.approx(11.0)
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
    assert options["tooltip"]["show"] is True
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
    assert options["tooltip"]["show"] is True


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
    assert options["yAxis"]["max"] == pytest.approx(time_axis[-1])
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
    assert 2 < options["graphic"][0]["right"] < options["visualMap"]["right"]
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
        color="#b03412",
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
    assert options["series"][0]["lineStyle"]["color"] == "#b03412"
    assert options["series"][1]["lineStyle"]["color"] == "#2596be"
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
