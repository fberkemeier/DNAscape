"""Focused simulated-user coverage for the Data Manager workflow shell."""

from __future__ import annotations

from app.pages.common import workflow_action_icon

import asyncio
import gzip
import io
import threading
import time
import zipfile
from typing import Any, cast

import numpy as np
import pytest
from nicegui.elements.button import Button
from nicegui.elements.checkbox import Checkbox
from nicegui.elements.input import Input
from nicegui.elements.label import Label
from nicegui.elements.number import Number
from nicegui.elements.progress import LinearProgress
from nicegui.elements.select import Select
from nicegui.elements.tabs import Tabs
from nicegui.elements.upload import Upload
from nicegui.testing import user_simulation

from app.data_conversion_service import (
    DataConversionRequest, DataConversionResult, TrackInspection, TrackRegion, inspect_track,
)
from app.export_settings import get_export_settings
from app.main import index
from app.mapping_uploads import inspect_upload
from app.pages.data_management import (
    _MATRIX,
    _apply_feature_defaults,
    _automatic_output_name,
    _build_generated_files_archive,
    _feature_settings_compatible,
    _generated_download_artifact,
    _resolved_feature_bounds,
    _selected_data,
    _selection_options,
    _track_aggregation_options,
    _unique_conversion_output_names,
)
from app.remote_sources import RemoteSource
from app.simulation_exports import ExportArtifact


async def _wait_for(predicate: Any, *, attempts: int = 100) -> None:
    for _ in range(attempts):
        if predicate():
            return
        await asyncio.sleep(0.02)
    raise AssertionError("Timed out while waiting for the Data Manager action.")


def _converted_result(request: DataConversionRequest) -> DataConversionResult:
    payload = b"value\n1\n2\n"
    filename = f"{request.output_name}.{request.target_format}"
    if request.gzip_output:
        payload = gzip.compress(payload, mtime=0)
        filename += ".gz"
    return DataConversionResult(
        artifact=ExportArtifact(
            payload,
            filename,
            "application/gzip" if request.gzip_output else "text/plain",
        ),
        source_format=request.inspection.source_format,
        target_format=request.target_format,
        data_form=request.input_form,
        shape=request.inspection.shape,
        labels=("value",),
    )


def test_generated_file_selector_switches_preview_and_export_and_preserves_preferences(monkeypatch) -> None:
    from app.pages import data_management as page
    from app.preview_plotting import PreviewChartHandle
    exports = []

    async def capture_chart(*_args, **_kwargs):
        return "data:image/png;base64,cG5n"

    def build_plot(_data_uri, _file_format, *, filename, dpi):
        exports.append(filename)
        return ExportArtifact(b"png", f"{filename}.png", "image/png")

    monkeypatch.setattr(PreviewChartHandle, "run_chart_method", capture_chart)
    monkeypatch.setattr(page, "build_chart_capture_export", build_plot)

    async def scenario():
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-data-management").click()

            def control(marker):
                return next(item for item in user.client.elements.values() if marker in item._markers and not item.is_deleted)

            def plotted_values():
                return np.asarray(control("data-manager-preview-chart").options["series"][0]["data"])[:, 1]

            await control("data-manager-upload").handle_uploads([
                Upload.SmallFileUpload("first.csv", "text/csv", b"position,signal\n0,1\n1,3\n2,2\n"),
                Upload.SmallFileUpload("second.csv", "text/csv", b"position,signal\n0,5\n1,7\n2,6\n"),
            ])
            await user.should_see("Datasets: 2", retries=60)
            user.find(marker="action-convert").click()
            await user.should_see(marker="data-manager-select-result-2", retries=80)
            assert control("data-manager-select-result-2")._props["aria-pressed"] == "true"
            assert control("data-manager-select-result-1")._props["aria-pressed"] == "false"
            np.testing.assert_array_equal(plotted_values(), [5, 7, 6])
            with user.client:
                control("data-manager-result-name-1").set_value("custom-first")
                control("data-manager-result-type-1").set_value("CSV")
            user.find(marker="data-manager-select-result-1").click()
            await _wait_for(lambda: control("data-manager-select-result-1")._props["aria-pressed"] == "true")
            np.testing.assert_array_equal(plotted_values(), [1, 3, 2])
            assert control("data-manager-result-name-1").value == "custom-first"
            assert control("data-manager-result-type-1").value == "CSV"

            user.find(marker="data-manager-preview-toggle").click()
            await user.should_not_see(marker="data-manager-preview-chart")
            user.find(marker="data-manager-preview-toggle").click()
            await user.should_see(marker="data-manager-preview-chart")
            np.testing.assert_array_equal(plotted_values(), [1, 3, 2])
            user.find(marker="action-download-plot").click()
            user.find(marker="download-plot-png").click()
            await _wait_for(lambda: bool(exports))
            assert "first" in exports[0] and "second" not in exports[0]

            await control("data-manager-upload").handle_uploads([
                Upload.SmallFileUpload("third.csv", "text/csv", b"position,signal\n0,9\n1,11\n2,10\n"),
            ])
            await user.should_see("Datasets: 3", retries=60)
            user.find(marker="action-convert").click()
            await user.should_see(marker="data-manager-select-result-3", retries=80)
            assert control("data-manager-select-result-3")._props["aria-pressed"] == "true"
            np.testing.assert_array_equal(plotted_values(), [9, 11, 10])
            assert control("data-manager-result-name-1").value == "custom-first"
            assert control("data-manager-result-type-1").value == "CSV"
            user.find(marker="remove-converted-data-3").click()
            await user.should_not_see(marker="data-manager-select-result-3")
            assert control("data-manager-select-result-2")._props["aria-pressed"] == "true"
            np.testing.assert_array_equal(plotted_values(), [5, 7, 6])
            user.find(marker="remove-converted-data-1").click()
            assert control("data-manager-select-result-2")._props["aria-pressed"] == "true"
            user.find(marker="delete-all-converted-data").click()
            await user.should_see(marker="data-manager-empty-result")
            await user.should_not_see(marker="data-manager-preview-chart")
    asyncio.run(scenario())


def test_generated_plot_settings_update_preview_and_export_without_changing_data(monkeypatch) -> None:
    import copy
    from app.pages import data_management as page
    from app.preview_plotting import PreviewChartHandle

    captures = []

    async def capture_chart(chart, *_args, **_kwargs):
        captures.append(copy.deepcopy(chart.options))
        return "data:image/png;base64,cG5n"

    monkeypatch.setattr(PreviewChartHandle, "run_chart_method", capture_chart)
    monkeypatch.setattr(page, "build_chart_capture_export", lambda *args, **kwargs: ExportArtifact(b"png", "plot.png", "image/png"))

    async def scenario():
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-data-management").click()

            def control(marker):
                return next(item for item in user.client.elements.values() if marker in item._markers and not item.is_deleted)

            def options():
                return control("data-manager-preview-chart").options

            await user.should_not_see("Plot settings")
            await control("data-manager-upload").handle_uploads([
                Upload.SmallFileUpload("first.csv", "text/csv", b"position,signal\n0,1\n1,3\n2,2\n"),
                Upload.SmallFileUpload("second.csv", "text/csv", b"position,signal\n0,5\n1,7\n2,6\n"),
            ])
            await user.should_see("Datasets: 2", retries=60)
            user.find(marker="action-convert").click()
            await user.should_see(marker="data-manager-preview-chart", retries=80)
            await user.should_see("Plot settings")
            await user.should_see("Raw data")
            assert control("data-manager-plot-aspect-ratio").value == 5.0
            assert options()["dnascapeAspectRatio"] == 5.0
            with user.client:
                control("data-manager-plot-aspect-ratio").set_value(2.0)
            assert options()["dnascapeAspectRatio"] == 2.0
            user.find(marker="data-manager-plot-reset-aspect-ratio").click()
            assert options()["dnascapeAspectRatio"] == 5.0
            settings = control("data-manager-plot-preview-settings")
            # Shared settings follow the plot and precede every generated row.
            container = control("data-manager-result-container")
            ancestors = []
            ancestor = settings
            while ancestor.parent_slot is not None:
                ancestor = ancestor.parent_slot.parent
                ancestors.append(ancestor)
            assert control("data-manager-result-preview") in ancestors
            assert container.default_slot.children.index(control("data-manager-result-preview")) == 0
            assert options()["yAxis"]["min"] == control("data-manager-plot-y-minimum").value
            with user.client:
                control("data-manager-plot-primary-color").set_value("#8b6f37")
                control("data-manager-plot-line-width").set_value(3.2)
                control("data-manager-plot-curve-type").set_value("dashed")
                control("data-manager-plot-x-axis-label").set_value("Genomic position")
                control("data-manager-plot-y-minimum").set_value(4)
                control("data-manager-plot-y-maximum").set_value(8)
            user.find(marker="data-manager-plot-invert-y").click()
            assert options()["series"][0]["lineStyle"]["color"] == "#8b6f37"
            assert options()["series"][0]["lineStyle"]["width"] == pytest.approx(3.2)
            assert options()["series"][0]["lineStyle"]["type"] == "dashed"
            assert options()["xAxis"]["name"] == "Genomic position"
            assert options()["yAxis"]["inverse"] is True
            assert (options()["yAxis"]["min"], options()["yAxis"]["max"]) == (4, 8)
            user.find(marker="data-manager-plot-reset-y-limits").click()
            assert options()["yAxis"]["min"] < 5 < 7 < options()["yAxis"]["max"]
            user.find(marker="data-manager-select-result-1").click()
            await _wait_for(lambda: np.asarray(options()["series"][0]["data"])[0, 1] == 1)
            np.testing.assert_array_equal(np.asarray(options()["series"][0]["data"])[:, 1], [1, 3, 2])
            assert options()["series"][0]["lineStyle"]["color"] == "#8b6f37"
            assert options()["yAxis"]["min"] < 1 < 3 < options()["yAxis"]["max"]
            with user.client:
                control("data-manager-plot-view").set_value("Histogram")
                control("data-manager-plot-histogram-bins").set_value(3)
            assert options()["series"][0]["type"] == "custom"
            assert len(options()["series"][0]["data"]) == 3
            user.find(marker="data-manager-plot-density").click()
            assert options()["series"][0]["type"] == "line"
            assert options()["yAxis"]["name"] == "Density"
            assert not control("data-manager-plot-histogram-bins").visible
            user.find(marker="data-manager-preview-toggle").click()
            await user.should_not_see("Plot settings")
            user.find(marker="data-manager-preview-toggle").click()
            await user.should_see("Plot settings")
            assert control("data-manager-plot-density").value is True
            expected = copy.deepcopy(options())
            user.find(marker="action-download-plot").click()
            user.find(marker="download-plot-png").click()
            await _wait_for(lambda: bool(captures))
            assert captures[0] == expected
            user.find(marker="delete-all-converted-data").click()
            await user.should_not_see("Plot settings")
            await user.should_not_see(marker="data-manager-preview-chart")
    asyncio.run(scenario())


def test_download_all_archive_preserves_every_file_with_safe_unique_names() -> None:
    bundled = _build_generated_files_archive(
        (
            ExportArtifact(b"first", "nested/profile.tsv", "text/tab-separated-values"),
            ExportArtifact(b"second", r"C:\temp\PROFILE.tsv", "text/tab-separated-values"),
            ExportArtifact(b"third", "values.tsv.gz", "application/gzip"),
            ExportArtifact(b"fourth", "values.tsv.gz", "application/gzip"),
        )
    )

    assert bundled.filename == "rdata_outputs.zip"
    assert bundled.media_type == "application/zip"
    with zipfile.ZipFile(io.BytesIO(bundled.data)) as archive:
        assert archive.namelist() == [
            "profile.tsv",
            "PROFILE_2.tsv",
            "values.tsv.gz",
            "values_2.tsv.gz",
        ]
        assert [archive.read(name) for name in archive.namelist()] == [
            b"first",
            b"second",
            b"third",
            b"fourth",
        ]


def test_generated_file_type_transcodes_retained_tsv_bytes() -> None:
    result = DataConversionResult(
        artifact=ExportArtifact(
            b"value\n1.25\n2.5\n",
            "profile.tsv",
            "text/tab-separated-values",
        ),
        source_format="csv",
        target_format="tsv",
        data_form="space",
        shape=(2,),
        labels=("value",),
    )

    artifact = _generated_download_artifact(
        result,
        file_name="profile",
        file_type="NPZ",
        compress=False,
    )

    assert artifact.filename == "profile.npz"
    assert artifact.media_type == "application/zip"
    with np.load(io.BytesIO(artifact.data), allow_pickle=False) as archive:
        np.testing.assert_allclose(archive["values"], [1.25, 2.5])


def test_automatic_data_manager_names_use_output_then_concise_source() -> None:
    assert _automatic_output_name("signal.tsv", "generic") == "rdata_signal"
    assert _automatic_output_name(
        "Replication Timing H1.tsv.gz",
        "replication_timing",
    ) == "rdata_rt_h1"


def test_batch_output_names_remain_unique_after_export_sanitization() -> None:
    assert _unique_conversion_output_names(
        ("nested/profile", r"C:\other\profile"),
        target_format="tsv",
        gzip_output=True,
    ) == ("profile", "profile_2")


def test_gwseq_backend_hides_known_broken_minimum_and_maximum_aggregations() -> None:
    assert tuple(_track_aggregation_options("gwseq_io_pp")) == (
        "mean",
        "sum",
        "coverage",
        "std",
    )
    assert "min" in _track_aggregation_options("pyBigWig")
    assert "max" in _track_aggregation_options("pyBigWig")


def test_aggregation_choices_follow_peak_and_presence_inputs(sns_peak_payload: bytes) -> None:
    peaks = inspect_upload("SNS.bed", sns_peak_payload)
    presence = inspect_upload("origins.bed", b"chr1\t100\t200\n")
    continuous = inspect_upload("signal.bedgraph", b"chr1\t100\t200\t10\n")
    assert tuple(_track_aggregation_options("pandas", peaks)) == ("max", "mean", "sum", "coverage")
    assert tuple(_track_aggregation_options("pandas", presence)) == ("max", "sum", "coverage")
    assert tuple(_track_aggregation_options("pandas", continuous)) == (
        "mean", "sum", "min", "max", "coverage", "std",
    )


def test_sns_firing_preset_and_switch_back_to_repliseq_bigwig(
    sns_peak_payload: bytes, monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[DataConversionRequest] = []

    def fake_fetch(url: str) -> RemoteSource:
        bigwig = url.endswith((".bigWig", ".bw"))
        okseq = "K562_merged_hmm_RFD" in url
        return RemoteSource(
            requested_url=url, final_url=url,
            filename="K562_RFD.bw" if okseq else "RepliSeq.bigWig" if bigwig else "K562_Peaks_FDR15.bed.gz",
            content=b"\x26\xfc\x8f\x88fake" if bigwig else gzip.compress(sns_peak_payload),
            content_type="application/octet-stream", etag=None, last_modified=None,
            accept_ranges=False, redirects=(),
        )

    def fake_inspect(inspection: Any) -> TrackInspection:
        if inspection.source_format == "bigwig":
            return TrackInspection(
                source_format="bigwig", regions=(TrackRegion("chr1", 0, 3000),),
                value_columns=("value",), size_bytes=len(inspection.content), backend="pyBigWig",
            )
        return inspect_track(inspection)

    def fake_convert(request: DataConversionRequest, *_args: Any, **_kwargs: Any) -> DataConversionResult:
        requests.append(request)
        return _converted_result(request)

    monkeypatch.setattr("app.pages.data_management.fetch_remote_source", fake_fetch)
    monkeypatch.setattr("app.pages.data_management.inspect_track", fake_inspect)
    monkeypatch.setattr("app.pages.data_management.convert_data", fake_convert)

    async def scenario() -> None:
        async with user_simulation(index) as user:
            def control(marker: str) -> Any:
                return next(iter(user.find(marker=marker).elements))

            await user.open("/")
            user.find(marker="nav-data-management").click()
            control("data-manager-source-tabs").set_value("url")
            await user.should_see(marker="use-bedgraph-data-example")
            user.find(marker="use-bedgraph-data-example").click()
            await asyncio.sleep(0.1)
            user.find(marker="action-inspect-data-link-1").click()
            await asyncio.sleep(0)
            await _wait_for(lambda: control("action-inspect-data-link-1").enabled)
            await user.should_see("The source is ready.", retries=80)
            assert control("data-manager-feature").value == "firing_rate"
            assert control("data-manager-aggregation").value == "max"
            assert control("data-manager-bin-size").value == 1000
            column = control("data-manager-bedgraph-value-column")
            assert column.value == "column:enrichment_ratio"
            assert tuple(column.options) == ("column:enrichment_ratio",)
            assert column.enabled is False
            assert control("data-manager-track-missing").value == "zero"
            assert tuple(control("data-manager-interpolation").options) == ("none",)
            assert control("data-manager-interpolation").enabled is False
            assert control("data-manager-smoothing").value == "none"
            assert "zscore" not in control("data-manager-value-scaling").options

            # Selecting Firing rate also repairs a manually changed generic preset.
            control("data-manager-feature").set_value("generic")
            await asyncio.sleep(0.1)
            control("data-manager-aggregation").set_value("mean")
            control("data-manager-smoothing").set_value("moving_average")
            control("data-manager-value-scaling").set_value("zscore")
            control("data-manager-feature").set_value("firing_rate")
            await asyncio.sleep(0.1)
            assert control("data-manager-aggregation").value == "max"
            assert control("data-manager-smoothing").value == "none"
            assert control("data-manager-value-scaling").value == "none"
            user.find(marker="action-convert").click()
            await _wait_for(lambda: len(requests) == 1)
            await user.should_see(marker="data-manager-result-1", retries=80)
            assert requests[0].value_column == "enrichment_ratio"
            assert requests[0].aggregation == "max"
            assert requests[0].smooth_window == 1
            assert requests[0].interpolation == "none"
            assert requests[0].biological_role == "firing_rate"

            user.find(marker="use-encode-data-example").click()
            await asyncio.sleep(0.1)
            user.find(marker="action-inspect-data-link-1").click()
            await asyncio.sleep(0)
            await _wait_for(lambda: control("action-inspect-data-link-1").enabled)
            await user.should_see("The source is ready.", retries=80)
            assert control("data-manager-feature").value == "replication_timing"
            aggregation = control("data-manager-aggregation")
            assert aggregation.value == "mean"
            assert tuple(aggregation.options) == ("mean", "sum", "min", "max", "coverage", "std")
            assert tuple(control("data-manager-interpolation").options) == ("none", "linear", "ffill", "bfill")
            assert control("data-manager-interpolation").enabled is True
            assert "zscore" in control("data-manager-value-scaling").options
            await user.should_not_see(marker="data-manager-bedgraph-value-column")
            user.find(marker="action-convert").click()
            await _wait_for(lambda: len(requests) == 2)
            assert requests[1].inspection.source_format == "bigwig"
            assert requests[1].aggregation == "mean"
            assert requests[1].biological_role == "replication_timing"

            await user.should_see(marker="data-manager-result-2", retries=80)
            user.find(marker="use-okseq-data-example").click()
            await asyncio.sleep(0.1)
            assert control("data-manager-source-url").value == (
                "https://raw.githubusercontent.com/CL-CHEN-Lab/OK-Seq/master/"
                "published_results/K562/K562_merged_hmm_RFD_cutoff10_bs1kb_sorted.bw"
            )
            user.find(marker="action-inspect-data-link-1").click()
            await asyncio.sleep(0)
            await _wait_for(lambda: control("action-inspect-data-link-1").enabled)
            await user.should_see("The source is ready.", retries=80)
            assert control("data-manager-feature").value == "fork_directionality"
            assert control("data-manager-aggregation").value == "mean"
            assert control("data-manager-track-missing").value == "zero"
            user.find(marker="action-convert").click()
            await _wait_for(lambda: len(requests) == 3)
            assert requests[2].inspection.source_format == "bigwig"
            assert requests[2].biological_role == "fork_directionality"
            assert requests[2].aggregation == "mean"
            assert requests[2].track_missing == "zero"
            assert requests[2].threshold_min == -1.0
            assert requests[2].threshold_max == 1.0

    asyncio.run(scenario())


def test_all_wide_chromosome_headers_remain_spatial_profiles() -> None:
    inspection = inspect_upload(
        "chromosomes.tsv",
        b"chr1\tchr2\n1\t4\n2\t5\n3\t6\n",
    )

    assert inspection.inferred_form == "x"
    assert _selection_options(inspection)[_MATRIX] == "All"
    selected = _selected_data(inspection, _MATRIX)
    assert selected.form == "x"
    # rdata uses a matrix internally to retain every independently spatial
    # header, without changing the dependence displayed to the user.
    assert selected.request_form == "xt"
    assert selected.shape == (3, 2)


@pytest.mark.parametrize(
    ("feature", "expected"),
    (
        ("generic", (False, None, None)),
        ("replication_timing", (True, 0.0, None)),
        ("firing_rate", (True, 0.0, None)),
        ("fork_speed", (True, 0.0, None)),
        ("fork_directionality", (True, -1.0, 1.0)),
        ("replicated_fraction", (True, 0.0, 1.0)),
    ),
)
def test_prepare_as_applies_value_domain_defaults(
    feature: str,
    expected: tuple[bool, float | None, float | None],
) -> None:
    settings: dict[str, Any] = {}

    _apply_feature_defaults(settings, feature)

    assert (
        settings["threshold_enabled"],
        settings["threshold_min"],
        settings["threshold_max"],
    ) == expected


@pytest.mark.parametrize(
    ("feature", "minimum", "maximum", "expected"),
    (
        ("generic", None, None, (False, None, None)),
        ("replication_timing", None, None, (True, 0.0, None)),
        ("fork_directionality", None, None, (True, -1.0, 1.0)),
        ("fork_directionality", -0.5, None, (True, -0.5, 1.0)),
        ("replicated_fraction", None, 0.8, (True, 0.0, 0.8)),
    ),
)
def test_blank_value_bounds_fall_back_to_prepare_as_defaults(
    feature: str,
    minimum: float | None,
    maximum: float | None,
    expected: tuple[bool, float | None, float | None],
) -> None:
    assert _resolved_feature_bounds(feature, minimum, maximum) == expected


@pytest.mark.parametrize(
    ("feature", "edits", "expected"),
    (
        ("replication_timing", {"threshold_min": -1}, False),
        ("replication_timing", {"value_scaling": "interval", "rescale_min": 30, "rescale_max": 500}, True),
        ("firing_rate", {"value_scaling": "multiplier", "scale": 2}, True),
        ("firing_rate", {"value_scaling": "multiplier", "scale": -1}, False),
        ("fork_speed", {"threshold_min": .5, "threshold_max": 5}, True),
        ("fork_speed", {"value_scaling": "center"}, False),
        ("fork_directionality", {"threshold_min": -.5, "threshold_max": .5}, True),
        ("fork_directionality", {"threshold_max": 2}, False),
        ("fork_directionality", {"value_scaling": "multiplier", "scale": -1}, True),
        ("fork_directionality", {"value_scaling": "multiplier", "scale": 2}, False),
        ("replicated_fraction", {"value_scaling": "interval", "rescale_min": 0, "rescale_max": 2}, False),
        ("replicated_fraction", {"missing_policy": "fill", "fill_value": 2}, False),
        ("replicated_fraction", {"track_missing": "fill", "fill_value": 2}, True),
        ("replicated_fraction", {"threshold_min": None, "threshold_max": None}, True),
        ("fork_speed", {"decimal_places": 2, "smoothing_mode": "moving_average", "bin_size": 500}, True),
        ("generic", {"threshold_min": -10, "threshold_max": 10}, True),
    ),
)
def test_biological_profile_names_follow_the_configured_value_domain(
    feature: str, edits: dict[str, Any], expected: bool,
) -> None:
    settings: dict[str, Any] = {"value_scaling": "none", "missing_policy": "zero"}
    _apply_feature_defaults(settings, feature)
    settings.update(edits)
    assert _feature_settings_compatible(feature, settings) is expected


def test_data_manager_starts_with_auto_detection_and_an_explicit_url_source() -> None:
    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-data-management").click()

            feature = cast(
                Select,
                next(iter(user.find(marker="data-manager-feature").elements)),
            )
            await user.should_not_see(marker="data-manager-source-format")
            assert feature.value == "generic"
            await user.should_see("Generated files")
            await user.should_see(marker="data-manager-empty-result")
            await user.should_not_see(marker="download-all-converted-data")

            tabs = cast(
                Tabs,
                next(iter(user.find(marker="data-manager-source-tabs").elements)),
            )
            tabs.set_value("url")
            await asyncio.sleep(0.1)
            await user.should_see(marker="data-manager-source-url")
            await user.should_see(marker="use-bedgraph-data-example")
            inspect = cast(
                Button,
                next(iter(user.find(marker="action-inspect-data-link").elements)),
            )
            assert inspect.text == "Inspect"
            assert inspect.icon is None
            assert user.client is not None
            progress = cast(
                LinearProgress,
                next(
                    element
                    for element in user.client.elements.values()
                    if "data-manager-url-progress-1" in element._markers
                ),
            )
            assert progress._props["size"] == "4px"
            assert not progress.default_slot.children
            encode_example = cast(
                Button,
                next(
                    iter(user.find(marker="use-encode-data-example").elements)
                ),
            )
            bedgraph_example = cast(
                Button,
                next(
                    iter(user.find(marker="use-bedgraph-data-example").elements)
                ),
            )
            assert encode_example.text == (
                "Example: ENCODE Repli-seq (bigWig) - K562"
            )
            assert bedgraph_example.text == (
                "Example: NCBI GEO SNS-seq (BED) - K562"
            )
            assert encode_example.icon == bedgraph_example.icon == "science"
            user.find(marker="use-encode-data-example").click()
            await asyncio.sleep(0.1)

            url = cast(
                Input,
                next(iter(user.find(marker="data-manager-source-url").elements)),
            )
            assert url.value == (
                "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/encodeDCC/"
                "wgEncodeUwRepliSeq/wgEncodeUwRepliSeqK562WaveSignalRep1.bigWig"
            )
            feature = cast(
                Select,
                next(iter(user.find(marker="data-manager-feature").elements)),
            )
            assert feature.value == "replication_timing"
            await user.should_not_see("earlier = larger")
            # Filling the curated example is intentionally not a network action.
            await user.should_see(marker="data-manager-empty-result")

    asyncio.run(scenario())


def test_empty_bin_fills_have_their_requested_order_in_data_manager(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.data_conversion_service import convert_data

    results: list[DataConversionResult] = []

    def convert_and_capture(*args: Any, **kwargs: Any) -> DataConversionResult:
        result = convert_data(*args, **kwargs)
        results.append(result)
        return result

    monkeypatch.setattr("app.pages.data_management.convert_data", convert_and_capture)

    async def scenario() -> None:
        async with user_simulation(index) as user:
            def control(marker: str) -> Any:
                return next(iter(user.find(marker=marker).elements))

            await user.open("/")
            user.find(marker="nav-data-management").click()
            uploader = cast(Upload, control("data-manager-upload"))
            await uploader.handle_uploads([Upload.SmallFileUpload(
                "signal.bedgraph", "text/plain", b"chr1\t0\t10\t10\nchr1\t20\t30\t20\n",
            )])
            await user.should_see(marker="data-manager-track-missing", retries=60)
            missing = cast(Select, control("data-manager-track-missing"))
            assert missing.options["min"] == "Fill with minimum"
            assert missing.options["max"] == "Fill with maximum"
            assert list(missing.options.values()) == [
                "Keep empty values", "Fill with zero", "Fill with a value",
                "Fill with minimum", "Fill with maximum", "Report error",
            ]
            assert missing.value == "zero"
            control("data-manager-bin-size").set_value(10)
            control("data-manager-value-scaling").set_value("interval")
            await user.should_see(marker="data-manager-target-minimum")
            control("data-manager-target-minimum").set_value(2)
            control("data-manager-target-maximum").set_value(4)
            for expected_count, (policy, expected) in enumerate((
                ("min", [2, 2, 4]), ("max", [2, 4, 4]),
                ("zero", [2, 0, 4]), ("fill", [2, 7, 4]),
            ), start=1):
                missing.set_value(policy)
                if policy == "fill":
                    await user.should_see(marker="data-manager-fill-value")
                    control("data-manager-fill-value").set_value(7)
                else:
                    await user.should_not_see(marker="data-manager-fill-value")
                user.find(marker="action-convert").click()
                await _wait_for(lambda: len(results) == expected_count)
                await _wait_for(lambda: control("action-convert").text == "Convert")
                payload = results[-1].artifact.data.decode("utf-8").splitlines()
                assert payload[0] == "chr1"
                np.testing.assert_allclose([float(value) for value in payload[1:]], expected)
            missing.set_value("error")
            user.find(marker="action-convert").click()
            await user.should_see("contain missing values", retries=60)
            await _wait_for(lambda: control("action-convert").text == "Convert")
            assert len(results) == 4
            await user.should_not_see(marker="data-manager-result-5")

    asyncio.run(scenario())


def test_bedgraph_reveals_track_polishing_controls_and_defaults_to_plain_tsv() -> None:
    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-data-management").click()
            uploader = cast(
                Upload,
                next(iter(user.find(marker="data-manager-upload").elements)),
            )
            await uploader.handle_uploads(
                [
                    Upload.SmallFileUpload(
                        "signal.bedgraph",
                        "text/plain",
                        b"chr2\t0\t5\t4\nchr1\t0\t10\t1\nchr1\t20\t30\t3\n",
                    )
                ]
            )

            await user.should_see("Format: bedGraph", retries=30)
            await user.should_see("Chromosomes: 2")
            missing = cast(
                Select, next(iter(user.find(marker="data-manager-track-missing").elements))
            )
            assert missing.value == "zero"
            for marker in (
                "data-manager-chromosome-scope",
                "data-manager-chromosome",
                "data-manager-bin-size",
                "data-manager-aggregation",
                "data-manager-bedgraph-value-column",
                "data-manager-track-missing",
                "data-manager-interpolation",
                "data-manager-smoothing",
                "data-manager-value-scaling",
                "data-manager-threshold-min",
                "data-manager-threshold-max",
            ):
                await user.should_see(marker=marker)
            await user.should_not_see(marker="data-manager-smooth-window")
            await user.should_not_see(marker="data-manager-multiplier-fields")
            await user.should_not_see(marker="data-manager-target-interval")
            await user.should_not_see(marker="data-manager-offset")
            await user.should_not_see(marker="data-manager-signal-transform")
            await user.should_see(marker="data-manager-threshold-fields")
            await user.should_not_see(marker="data-manager-threshold-enabled")

            smoothing = cast(
                Select,
                next(iter(user.find(marker="data-manager-smoothing").elements)),
            )
            interpolation = cast(
                Select,
                next(iter(user.find(marker="data-manager-interpolation").elements)),
            )
            smoothing_column = next(
                iter(user.find(marker="data-manager-smoothing-column").elements)
            )
            missing_column = next(
                iter(user.find(marker="data-manager-missing-column").elements)
            )
            scaling_column = next(
                iter(user.find(marker="data-manager-scaling-column").elements)
            )
            values_grid = next(
                iter(user.find(marker="data-manager-polishing-grid").elements)
            )
            value_scaling = cast(
                Select,
                next(iter(user.find(marker="data-manager-value-scaling").elements)),
            )
            assert smoothing.parent_slot is not None
            assert smoothing.parent_slot.parent is smoothing_column
            assert interpolation.parent_slot is not None
            assert interpolation.parent_slot.parent is smoothing_column
            assert value_scaling.parent_slot is not None
            assert value_scaling.parent_slot.parent is scaling_column
            assert values_grid.default_slot.children[:3] == [
                missing_column,
                smoothing_column,
                scaling_column,
            ]
            smoothing.set_value("moving_average")
            await user.should_see(marker="data-manager-smooth-window")
            smooth_window = cast(
                Number,
                next(iter(user.find(marker="data-manager-smooth-window").elements)),
            )
            assert smooth_window.value == 3
            assert smooth_window.parent_slot is not None
            assert smooth_window.parent_slot.parent is smoothing_column

            assert tuple(value_scaling.options) == (
                "none",
                "multiplier",
                "interval",
                "center",
                "zscore",
                "maxabs",
            )
            value_scaling.set_value("multiplier")
            await user.should_see(marker="data-manager-multiplier-fields")
            await user.should_see(marker="data-manager-scale-multiplier")
            await user.should_not_see(marker="data-manager-target-interval")
            multiplier_fields = next(
                iter(user.find(marker="data-manager-multiplier-fields").elements)
            )
            assert multiplier_fields.parent_slot is not None
            assert multiplier_fields.parent_slot.parent is scaling_column
            value_scaling.set_value("interval")
            await user.should_not_see(marker="data-manager-multiplier-fields")
            await user.should_see(marker="data-manager-target-interval")
            await user.should_see(marker="data-manager-target-minimum")
            await user.should_see(marker="data-manager-target-maximum")
            await user.should_see(marker="data-manager-interval-invert")
            interval_fields = next(
                iter(user.find(marker="data-manager-target-interval").elements)
            )
            assert interval_fields.parent_slot is not None
            assert interval_fields.parent_slot.parent is scaling_column

            threshold_max = cast(
                Number,
                next(iter(user.find(marker="data-manager-threshold-max").elements)),
            )
            threshold_fields = next(
                iter(user.find(marker="data-manager-threshold-fields").elements)
            )
            assert threshold_fields.parent_slot is not None
            assert threshold_fields.parent_slot.parent is missing_column
            threshold_max.set_value(10)

            bin_size = cast(
                Number,
                next(iter(user.find(marker="data-manager-bin-size").elements)),
            )
            assert bin_size.value == 1_000

            scope = cast(
                Select,
                next(iter(user.find(marker="data-manager-chromosome-scope").elements)),
            )
            assert scope.value == "single"
            assert tuple(scope.options) == ("single", "all")

            await user.should_not_see(marker="data-manager-output-format")
            await user.should_not_see("Export settings")
            decimal_places = cast(
                Number,
                next(iter(user.find(marker="data-manager-decimal-places").elements)),
            )
            assert decimal_places.value == 6
            assert decimal_places.parent_slot.parent is missing_column
            assert missing_column.default_slot.children[0] is decimal_places
            assert missing_column.default_slot.children[1] is missing
            assert missing.parent_slot.children.index(missing) < missing.parent_slot.children.index(threshold_fields)
            await user.should_not_see(marker="data-manager-gzip")

            preview_toggle = cast(
                Checkbox,
                next(iter(user.find(marker="data-manager-preview-toggle").elements)),
            )
            assert preview_toggle.value is True

            chromosome = next(iter(user.find(marker="data-manager-chromosome").elements))
            start = next(iter(user.find(marker="data-manager-start").elements))
            end = next(iter(user.find(marker="data-manager-end").elements))
            chromosome.set_value("chr1")
            start.set_value(5)
            end.set_value(20)
            user.find(marker="data-manager-reset-region").click()
            assert (start.value, end.value) == (0, 30)
            chromosome.set_value("chr2")
            start.set_value(1)
            end.set_value(4)
            user.find(marker="data-manager-reset-region").click()
            assert (start.value, end.value) == (0, 5)
            assert end._props["max"] == "5"

            user.find(marker="action-convert").click()
            await user.should_see(marker="data-manager-result-1", retries=60)
            await user.should_see(marker="download-all-converted-data")
            await user.should_see(marker="data-manager-result-preview", retries=60)

            preview_toggle.set_value(False)
            await user.should_not_see(marker="data-manager-result-preview")
            preview_toggle.set_value(True)
            await user.should_see(marker="data-manager-result-preview", retries=60)
            await user.should_see(marker="data-manager-preview-chart", retries=60)

            scope.set_value("all")
            await user.should_see(marker="data-manager-all-scope-note")
            await user.should_not_see(marker="data-manager-single-region")
            await user.should_see("Dependence: Space")

            user.find(marker="remove-converted-data-1").click()
            await user.should_see(marker="data-manager-empty-result")

    asyncio.run(scenario())


def test_track_conversion_can_be_retried_after_increasing_settings_memory_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app import data_conversion_service

    monkeypatch.setattr(data_conversion_service, "_TRACK_ESTIMATED_BYTES_PER_BIN", 4 * 1024**2)
    settings = get_export_settings()
    previous_limit = settings.conversion_memory_limit_mb
    settings.conversion_memory_limit_mb = 64

    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-data-management").click()
            uploader = cast(
                Upload, next(iter(user.find(marker="data-manager-upload").elements))
            )
            await uploader.handle_uploads(
                [
                    Upload.SmallFileUpload(
                        "signal.bedgraph",
                        "text/plain",
                        b"chr1\t0\t30000\t1\nchr2\t0\t20000\t2\n",
                    )
                ]
            )
            await user.should_see("Chromosomes: 2", retries=30)
            scope = cast(
                Select, next(iter(user.find(marker="data-manager-chromosome-scope").elements))
            )
            scope.set_value("all")
            user.find(marker="action-convert").click()
            await user.should_see("conversion memory limit is 64 MB", retries=60)
            await user.should_not_see(marker="data-manager-result-1")

            user.find(marker="app-menu-toggle").click()
            user.find(marker="menu-settings").click()
            memory_limit = cast(
                Number, next(iter(user.find(marker="settings-conversion-memory-limit").elements))
            )
            memory_limit.set_value(512)
            user.find(marker="settings-apply").click()
            await asyncio.sleep(0)
            assert settings.conversion_memory_limit_mb == 512
            assert scope.value == "all"
            user.find(marker="action-convert").click()
            await user.should_see(marker="data-manager-result-1", retries=60)
            await user.should_see(marker="download-all-converted-data")

    try:
        asyncio.run(scenario())
    finally:
        settings.conversion_memory_limit_mb = previous_limit


def test_plain_profile_reveals_the_same_polishing_controls() -> None:
    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-data-management").click()
            uploader = cast(
                Upload,
                next(iter(user.find(marker="data-manager-upload").elements)),
            )
            await uploader.handle_uploads(
                [
                    Upload.SmallFileUpload(
                        "signal.tsv",
                        "text/tab-separated-values",
                        b"position\tsignal\n0\t-2\n1\t1\n2\t4\n",
                    )
                ]
            )

            await user.should_see("Dependence: Space", retries=30)
            for marker in (
                "data-manager-smoothing",
                "data-manager-value-scaling",
                "data-manager-threshold-min",
                "data-manager-threshold-max",
            ):
                await user.should_see(marker=marker)
            await user.should_not_see(marker="data-manager-threshold-enabled")

            value_scaling = cast(
                Select,
                next(iter(user.find(marker="data-manager-value-scaling").elements)),
            )
            value_scaling.set_value("interval")
            await user.should_see(marker="data-manager-target-interval")
            rescale_min = cast(
                Number,
                next(iter(user.find(marker="data-manager-target-minimum").elements)),
            )
            rescale_max = cast(
                Number,
                next(iter(user.find(marker="data-manager-target-maximum").elements)),
            )
            rescale_min.set_value(3)
            rescale_max.set_value(1)
            value_scaling.set_value("none")
            await user.should_not_see(marker="data-manager-target-interval")

            lower = cast(
                Number,
                next(iter(user.find(marker="data-manager-threshold-min").elements)),
            )
            upper = cast(
                Number,
                next(iter(user.find(marker="data-manager-threshold-max").elements)),
            )
            lower.set_value(5)
            upper.set_value(1)
            lower.set_value(None)
            upper.set_value(None)
            await user.should_see(marker="data-manager-threshold-fields")

            user.find(marker="action-convert").click()
            await user.should_see(marker="data-manager-result-1", retries=60)

    asyncio.run(scenario())


def test_configure_starts_with_prepare_as_and_preserves_uploaded_filename_tooltip() -> None:
    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-data-management").click()
            uploader = cast(
                Upload,
                next(iter(user.find(marker="data-manager-upload").elements)),
            )
            long_name = "replication_timing_profile_with_a_full_name.tsv"
            await uploader.handle_uploads(
                [
                    Upload.SmallFileUpload(
                        long_name,
                        "text/tab-separated-values",
                        b"position\tsignal\n0\t1\n1\t2\n",
                    )
                ]
            )

            await user.should_see(marker="data-manager-upload-file", retries=30)
            await user.should_see(marker="data-manager-config-grid", retries=30)
            config_grid = next(
                iter(user.find(marker="data-manager-config-grid").elements)
            )
            feature = cast(
                Select,
                next(iter(user.find(marker="data-manager-feature").elements)),
            )
            assert feature.parent_slot is not None
            assert feature.parent_slot.parent is config_grid
            assert config_grid.default_slot.children[0] is feature
            preview = next(iter(user.find(marker="data-manager-preview-toggle").elements))
            assert config_grid.default_slot.children[1] is preview
            assert preview.text == "Show plot"
            await user.should_not_see("Output", kind=Label)
            await user.should_not_see(marker="data-manager-genome-build")
            await user.should_not_see(marker="data-manager-units")
            await user.should_not_see(marker="data-manager-value-name")
            await user.should_not_see("Polish values")

            upload_row = next(
                iter(user.find(marker="data-manager-upload-file").elements)
            )
            assert upload_row._props["title"] == long_name
            assert "Drop another file" not in str(upload_row._props.get("title"))

            feature.set_value("fork_directionality")
            await asyncio.sleep(0.1)
            lower = cast(
                Number,
                next(iter(user.find(marker="data-manager-threshold-min").elements)),
            )
            upper = cast(
                Number,
                next(iter(user.find(marker="data-manager-threshold-max").elements)),
            )
            await user.should_not_see(marker="data-manager-threshold-enabled")
            assert lower.value == -1.0
            assert upper.value == 1.0

    asyncio.run(scenario())


def test_out_of_domain_edits_switch_to_generic_without_resetting_the_conversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[DataConversionRequest] = []

    def capture(request: DataConversionRequest, *_args: Any) -> DataConversionResult:
        requests.append(request)
        return _converted_result(request)

    monkeypatch.setattr("app.pages.data_management.convert_data", capture)

    async def scenario() -> None:
        async with user_simulation(index) as user:
            def control(suffix: str) -> Any:
                return next(iter(user.find(marker=f"data-manager-{suffix}").elements))

            await user.open("/")
            user.find(marker="nav-data-management").click()
            await control("upload").handle_uploads([Upload.SmallFileUpload(
                "signal.csv", "text/csv", b"value\n-0.2\n0.4\n0.1\n",
            )])
            await user.should_see(marker="data-manager-threshold-min", retries=60)
            control("feature").set_value("fork_directionality")
            for _ in range(4):
                await asyncio.sleep(0)
            control("decimal-places").set_value(2)
            control("smoothing").set_value("moving_average")
            control("threshold-min").set_value(-.5)
            control("threshold-max").set_value(.5)
            assert control("feature").value == "fork_directionality"
            control("value-scaling").set_value("interval")
            control("target-minimum").set_value(-.2)
            control("target-maximum").set_value(.8)
            assert control("feature").value == "fork_directionality"
            control("target-maximum").set_value(2)
            assert control("feature").value == "generic"
            assert control("value-scaling").value == "interval"
            assert control("threshold-min").value == -.5
            assert control("threshold-max").value == .5
            assert control("decimal-places").value == 2
            assert control("smoothing").value == "moving_average"
            assert "zscore" in control("value-scaling").options
            user.find(marker="action-convert").click()
            await _wait_for(lambda: bool(requests))

    asyncio.run(scenario())
    assert requests[0].scaling_mode == "interval"
    assert (requests[0].rescale_min, requests[0].rescale_max) == (-.2, 2)
    assert (requests[0].threshold_min, requests[0].threshold_max) == (-.5, .5)
    assert requests[0].decimal_places == 2
    assert requests[0].smooth_window == 3
    assert requests[0].output_name == "rdata_signal"


def test_blank_bounds_are_resolved_in_the_conversion_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.pages import data_management as data_manager_page

    captured: list[DataConversionRequest] = []

    def fake_convert(
        request: DataConversionRequest,
        report: Any,
        cancel: Any,
    ) -> DataConversionResult:
        captured.append(request)
        cancel()
        return _converted_result(request)

    monkeypatch.setattr(data_manager_page, "convert_data", fake_convert)
    monkeypatch.setattr(data_manager_page, "cache_dataset", lambda *_args: None)

    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-data-management").click()
            uploader = cast(
                Upload,
                next(iter(user.find(marker="data-manager-upload").elements)),
            )
            await uploader.handle_uploads(
                [
                    Upload.SmallFileUpload(
                        "directionality.tsv",
                        "text/tab-separated-values",
                        b"position\tdirectionality\n0\t-0.2\n1\t0.4\n",
                    )
                ]
            )
            await user.should_see(marker="data-manager-threshold-min", retries=30)
            feature = cast(
                Select,
                next(iter(user.find(marker="data-manager-feature").elements)),
            )
            feature.set_value("fork_directionality")
            await asyncio.sleep(0.1)
            lower = cast(
                Number,
                next(iter(user.find(marker="data-manager-threshold-min").elements)),
            )
            upper = cast(
                Number,
                next(iter(user.find(marker="data-manager-threshold-max").elements)),
            )
            lower.set_value(None)
            upper.set_value(None)

            user.find(marker="action-convert").click()
            await _wait_for(lambda: bool(captured))

    asyncio.run(scenario())
    request = captured[0]
    assert request.threshold_enabled is True
    assert request.threshold_min == -1.0
    assert request.threshold_max == 1.0
    assert request.genome_build is None
    assert request.units is None


def test_selecting_all_chromosome_headers_does_not_change_space_dependence() -> None:
    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-data-management").click()
            uploader = cast(
                Upload,
                next(iter(user.find(marker="data-manager-upload").elements)),
            )
            await uploader.handle_uploads(
                [
                    Upload.SmallFileUpload(
                        "chromosomes.tsv",
                        "text/tab-separated-values",
                        b"chr1\tchr2\n1\t4\n2\t5\n3\t6\n",
                    )
                ]
            )

            await user.should_see("Dependence: Space", retries=30)
            profiles = cast(
                Select,
                next(
                    iter(
                        user.find(marker="data-manager-source-selection").elements
                    )
                ),
            )
            assert profiles.options[_MATRIX] == "All"
            profiles.set_value(_MATRIX)
            await asyncio.sleep(0.1)
            await user.should_see("Dependence: Space", retries=30)
            await user.should_not_see("Dependence: Space-time")

    asyncio.run(scenario())


def test_data_manager_retains_multiple_uploads_converts_the_batch_and_deletes_all() -> None:
    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-data-management").click()
            uploader = cast(
                Upload,
                next(iter(user.find(marker="data-manager-upload").elements)),
            )

            assert uploader._props.get("multiple") is True
            assert uploader.visible is True
            await uploader.handle_uploads(
                [
                    Upload.SmallFileUpload(
                        "first.csv",
                        "text/csv",
                        b"position,signal\n0,1\n1,3\n2,2\n",
                    ),
                    Upload.SmallFileUpload(
                        "second.csv",
                        "text/csv",
                        b"position,signal\n0,5\n1,7\n2,6\n",
                    ),
                ]
            )

            await user.should_see("first.csv", retries=30)
            await user.should_see("second.csv", retries=30)
            await user.should_see("Datasets: 2", retries=30)
            await user.should_see(marker="data-manager-source-dataset")
            await user.should_see(marker="data-manager-upload-delete-all")
            assert uploader.visible is True
            assert len(user.find(marker="data-manager-upload-file").elements) == 2

            # A same-name drop replaces retained bytes in place without removing
            # the row or changing the batch order.
            await uploader.handle_uploads(
                [
                    Upload.SmallFileUpload(
                        "first.csv",
                        "text/csv",
                        b"position,signal\n0,8\n1,9\n",
                    )
                ]
            )
            await user.should_see("Array size: 2", retries=30)
            assert len(user.find(marker="data-manager-upload-file").elements) == 2

            user.find(marker="action-convert").click()
            await user.should_see(marker="data-manager-result-1", retries=80)
            await user.should_see(marker="data-manager-result-2", retries=80)
            await user.should_see(marker="download-all-converted-data")
            assert len(user.find(marker="data-manager-result-1").elements) == 1
            assert len(user.find(marker="data-manager-result-2").elements) == 1

            user.find(marker="data-manager-upload-delete-all").click()
            await user.should_not_see(marker="data-manager-upload-file", retries=30)
            await user.should_not_see(marker="data-manager-upload-delete-all")
            await user.should_not_see(marker="data-manager-source-dataset")
            assert uploader.visible is True
            convert = next(iter(user.find(marker="action-convert").elements))
            assert convert.enabled is False
            # Removing inputs must not silently discard already generated files.
            await user.should_see(marker="data-manager-result-1")
            await user.should_see(marker="data-manager-result-2")

    asyncio.run(scenario())


def test_data_manager_adds_and_inspects_multiple_public_link_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_fetch(url: str) -> RemoteSource:
        name = "first.csv" if url.endswith("first.csv") else "second.csv"
        values = b"value\n1\n2\n" if name == "first.csv" else b"value\n3\n4\n"
        return RemoteSource(
            requested_url=url,
            final_url=url,
            filename=name,
            content=values,
            content_type="text/csv",
            etag=None,
            last_modified=None,
            accept_ranges=False,
            redirects=(),
        )

    monkeypatch.setattr("app.pages.data_management.fetch_remote_source", fake_fetch)

    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-data-management").click()
            tabs = cast(
                Tabs,
                next(iter(user.find(marker="data-manager-source-tabs").elements)),
            )
            tabs.set_value("url")
            await user.should_see(marker="data-manager-source-url-1")

            user.find(marker="data-manager-add-link").click()
            await user.should_see(marker="data-manager-source-url-2")
            assert len(user.find(marker="data-manager-source-url").elements) == 2
            assert len(user.find(marker="action-inspect-data-link").elements) == 2

            first_url = cast(
                Input,
                next(iter(user.find(marker="data-manager-source-url-1").elements)),
            )
            second_url = cast(
                Input,
                next(iter(user.find(marker="data-manager-source-url-2").elements)),
            )
            second_url.set_value("https://example.org/second.csv")
            user.find(marker="use-bedgraph-data-example").click()
            await asyncio.sleep(0.1)
            assert first_url.value == (
                "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM716nnn/"
                "GSM716435/suppl/GSM716435_K562_Peaks_FDR15.bed.gz"
            )
            assert second_url.value == "https://example.org/second.csv"
            user.find(marker="use-encode-data-example").click()
            await asyncio.sleep(0.1)
            assert first_url.value.endswith("WaveSignalRep1.bigWig")
            assert second_url.value == "https://example.org/second.csv"

            first_url.set_value("https://example.org/first.csv")
            second_url.set_value("https://example.org/second.csv")

            user.find(marker="action-inspect-data-link-1").click()
            await user.should_see("The source is ready.", retries=80)
            convert = next(iter(user.find(marker="action-convert").elements))
            assert convert.enabled is False
            user.find(marker="action-inspect-data-link-2").click()
            await user.should_see("The source is ready.", retries=80)
            await user.should_see("Datasets: 2", retries=80)
            await user.should_see(marker="data-manager-source-dataset")
            assert len(user.find(marker="data-manager-source-url").elements) == 2
            assert convert.enabled is True

            user.find(marker="action-convert").click()
            await user.should_see(marker="data-manager-result-1", retries=80)
            await user.should_see(marker="data-manager-result-2", retries=80)

            # Replacing an already inspected first link invalidates only that
            # retained source, so stale bytes cannot be converted under the new URL.
            user.find(marker="use-bedgraph-data-example").click()
            await asyncio.sleep(0.1)
            assert first_url.value.endswith(
                "GSM716435_K562_Peaks_FDR15.bed.gz"
            )
            assert second_url.value == "https://example.org/second.csv"
            assert convert.enabled is False

            user.find(marker="remove-data-link-1").click()
            await user.should_not_see(marker="data-manager-source-url-1", retries=30)
            await user.should_see(marker="data-manager-source-url-2")
            await user.should_not_see(marker="data-manager-source-dataset")

    asyncio.run(scenario())


def test_data_manager_examples_follow_the_global_visibility_setting() -> None:
    settings = get_export_settings()
    previous = settings.show_examples
    settings.show_examples = True

    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-data-management").click()
            tabs = cast(
                Tabs,
                next(iter(user.find(marker="data-manager-source-tabs").elements)),
            )
            tabs.set_value("url")
            await asyncio.sleep(0.1)
            await user.should_see(marker="data-manager-link-examples")

            user.find(marker="app-menu-toggle").click()
            user.find(marker="menu-settings").click()
            show_examples = cast(
                Checkbox,
                next(iter(user.find(marker="settings-show-examples").elements)),
            )
            assert show_examples.value is True
            show_examples.set_value(False)
            user.find(marker="settings-apply").click()
            await user.should_not_see(marker="data-manager-link-examples")
            assert settings.show_examples is False
            user.find(marker="app-menu-toggle").click()
            user.find(marker="menu-settings").click()
            show_examples.set_value(True)
            user.find(marker="settings-apply").click()
            await user.should_see(marker="data-manager-link-examples")
            assert settings.show_examples is True

    try:
        asyncio.run(scenario())
    finally:
        settings.show_examples = previous


def test_generated_files_use_raw_output_rows_and_keep_export_preferences(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.pages import data_management as data_manager_page

    prepared_artifacts: list[tuple[str, str, bool, ExportArtifact]] = []
    archives: list[tuple[ExportArtifact, ...]] = []
    real_prepare_artifact = data_manager_page._generated_download_artifact
    real_build_archive = data_manager_page._build_generated_files_archive

    def fake_convert(
        request: DataConversionRequest,
        report: Any,
        cancel: Any,
    ) -> DataConversionResult:
        cancel()
        report({"stage": "prepare", "status": "done", "progress": 0.8})
        return _converted_result(request)

    def record_artifact(
        result: DataConversionResult,
        *,
        file_name: str,
        file_type: str | None = None,
        compress: bool,
    ) -> ExportArtifact:
        artifact = real_prepare_artifact(
            result,
            file_name=file_name,
            file_type=file_type,
            compress=compress,
        )
        prepared_artifacts.append((file_name, str(file_type), compress, artifact))
        return artifact

    def record_archive(artifacts: Any) -> ExportArtifact:
        retained = tuple(artifacts)
        archives.append(retained)
        return real_build_archive(retained)

    monkeypatch.setattr(data_manager_page, "convert_data", fake_convert)
    monkeypatch.setattr(
        data_manager_page,
        "_generated_download_artifact",
        record_artifact,
    )
    monkeypatch.setattr(
        data_manager_page,
        "_build_generated_files_archive",
        record_archive,
    )
    monkeypatch.setattr(data_manager_page, "cache_dataset", lambda *_args: None)

    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-data-management").click()

            assert user.client is not None
            progress = cast(
                LinearProgress,
                next(
                    element
                    for element in user.client.elements.values()
                    if "data-manager-progress" in element._markers
                ),
            )
            status = next(iter(user.find(marker="data-manager-status").elements))
            generated_panel = progress.parent_slot.parent
            assert generated_panel is not None
            assert "dn-data-manager-output-card" in generated_panel._classes
            assert status.parent_slot.parent is generated_panel

            uploader = cast(
                Upload,
                next(iter(user.find(marker="data-manager-upload").elements)),
            )
            await uploader.handle_uploads(
                [
                    Upload.SmallFileUpload(
                        "first.csv",
                        "text/csv",
                        b"value\n1\n2\n",
                    ),
                    Upload.SmallFileUpload(
                        "second.csv",
                        "text/csv",
                        b"value\n3\n4\n",
                    ),
                ]
            )
            await user.should_see("Datasets: 2", retries=40)
            user.find(marker="action-convert").click()
            await user.should_see(marker="data-manager-result-2", retries=60)

            first_row = next(
                iter(user.find(marker="data-manager-result-row-1").elements)
            )
            assert "dn-download-file-row" in first_row._classes
            assert "dn-raw-output-row" in first_row._classes
            first_card = next(
                iter(user.find(marker="data-manager-result-1").elements)
            )
            assert not any(
                "dn-mapping-batch-raw-title" in child._classes
                for child in first_card.default_slot.children
            )
            first_name = cast(
                Input,
                next(
                    iter(user.find(marker="data-manager-result-name-1").elements)
                ),
            )
            first_type = cast(
                Select,
                next(
                    iter(user.find(marker="data-manager-result-type-1").elements)
                ),
            )
            first_compress = cast(
                Checkbox,
                next(
                    iter(
                        user.find(
                            marker="data-manager-result-compression-1"
                        ).elements
                    )
                ),
            )
            second_name = cast(
                Input,
                next(
                    iter(user.find(marker="data-manager-result-name-2").elements)
                ),
            )
            second_compress = cast(
                Checkbox,
                next(
                    iter(
                        user.find(
                            marker="data-manager-result-compression-2"
                        ).elements
                    )
                ),
            )
            assert first_type.value == "TSV"
            assert first_type.enabled is True
            assert tuple(first_type.options) == (
                "TSV",
                "TXT",
                "CSV",
                "NPZ",
            )
            await user.should_see(marker="remove-converted-data-1")
            await user.should_see(marker="remove-converted-data-2")
            download_all = next(
                iter(user.find(marker="download-all-converted-data").elements)
            )
            delete_all = next(
                iter(user.find(marker="delete-all-converted-data").elements)
            )
            assert download_all.parent_slot is delete_all.parent_slot
            assert download_all.parent_slot is not None
            bulk_children = download_all.parent_slot.children
            assert bulk_children.index(download_all) < bulk_children.index(delete_all)

            with user.client:
                first_name.set_value("renamed_first")
                first_type.set_value("CSV")
                first_compress.set_value(True)
                second_name.set_value("renamed_second")
                second_compress.set_value(False)
            await asyncio.sleep(0)

            user.find(marker="action-download-converted-data-1").click()
            await _wait_for(lambda: len(prepared_artifacts) == 1)
            file_name, file_type, compress, artifact = prepared_artifacts[0]
            assert (file_name, file_type, compress) == (
                "renamed_first",
                "CSV",
                True,
            )
            assert artifact.filename == "renamed_first.csv.gz"
            assert artifact.media_type == "application/gzip"
            assert gzip.decompress(artifact.data) == b"value\n1.0\n2.0\n"

            user.find(marker="download-all-converted-data").click()
            await _wait_for(lambda: bool(archives))
            assert [artifact.filename for artifact in archives[0]] == [
                "renamed_first.csv.gz",
                "renamed_second.tsv",
            ]
            bundled = real_build_archive(archives[0])
            with zipfile.ZipFile(io.BytesIO(bundled.data)) as archive:
                assert archive.namelist() == [
                    "renamed_first.csv.gz",
                    "renamed_second.tsv",
                ]
                assert gzip.decompress(archive.read("renamed_first.csv.gz")) == (
                    b"value\n1.0\n2.0\n"
                )
                assert archive.read("renamed_second.tsv") == b"value\n1\n2\n"

            user.find(marker="remove-converted-data-1").click()
            await user.should_not_see(marker="data-manager-result-1")
            await user.should_see(marker="data-manager-result-2")

    asyncio.run(scenario())


def test_data_manager_stop_cancels_batch_without_publishing_staged_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.pages import data_management as data_manager_page

    second_started = threading.Event()
    calls = 0

    def cancellable_convert(
        request: DataConversionRequest,
        report: Any,
        cancel: Any,
    ) -> DataConversionResult:
        nonlocal calls
        calls += 1
        report({"stage": "prepare", "status": "start", "progress": 0.4})
        cancel()
        if calls == 1:
            return _converted_result(request)
        second_started.set()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            cancel()
            time.sleep(0.01)
        raise TimeoutError("conversion cancellation was not delivered")

    monkeypatch.setattr(data_manager_page, "convert_data", cancellable_convert)
    monkeypatch.setattr(data_manager_page, "cache_dataset", lambda *_args: None)

    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-data-management").click()
            uploader = cast(
                Upload,
                next(iter(user.find(marker="data-manager-upload").elements)),
            )
            await uploader.handle_uploads(
                [
                    Upload.SmallFileUpload(
                        "first.csv",
                        "text/csv",
                        b"value\n1\n2\n",
                    ),
                    Upload.SmallFileUpload(
                        "second.csv",
                        "text/csv",
                        b"value\n3\n4\n",
                    ),
                ]
            )
            await user.should_see("Datasets: 2", retries=40)
            button = cast(
                Button,
                next(iter(user.find(marker="action-convert").elements)),
            )
            assert user.client is not None
            progress = cast(
                LinearProgress,
                next(
                    element
                    for element in user.client.elements.values()
                    if "data-manager-progress" in element._markers
                ),
            )

            user.find(marker="action-convert").click()
            assert await asyncio.to_thread(second_started.wait, 2)
            assert button.text == "Stop"
            assert button.icon == "stop"
            assert button.enabled
            assert progress.visible
            await user.should_see("2/2 - second.csv", retries=30)

            user.find(marker="action-convert").click()
            await user.should_see("Conversion stopped", retries=40)
            assert button.text == "Convert"
            assert button.icon == workflow_action_icon("data")
            assert button.enabled
            assert progress.visible is False
            await user.should_see(marker="data-manager-empty-result")
            await user.should_not_see(marker="data-manager-result-1")
            await user.should_not_see(marker="data-manager-result-2")

    asyncio.run(scenario())
