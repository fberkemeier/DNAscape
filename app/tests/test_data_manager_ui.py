"""Focused simulated-user coverage for the Data Manager workflow shell."""

from __future__ import annotations

import asyncio
import io
import zipfile
from typing import cast

from nicegui.elements.checkbox import Checkbox
from nicegui.elements.input import Input
from nicegui.elements.number import Number
from nicegui.elements.select import Select
from nicegui.elements.tabs import Tabs
from nicegui.elements.upload import Upload
from nicegui.testing import user_simulation

from app.main import index
from app.pages.data_management import (
    _automatic_output_name,
    _build_generated_files_archive,
    _track_aggregation_options,
)
from app.simulation_exports import ExportArtifact


def test_download_all_archive_preserves_every_file_with_safe_unique_names() -> None:
    bundled = _build_generated_files_archive(
        (
            ExportArtifact(b"first", "nested/profile.tsv", "text/tab-separated-values"),
            ExportArtifact(b"second", r"C:\temp\PROFILE.tsv", "text/tab-separated-values"),
            ExportArtifact(b"third", "values.tsv.gz", "application/gzip"),
            ExportArtifact(b"fourth", "values.tsv.gz", "application/gzip"),
        )
    )

    assert bundled.filename == "dnascape_data_outputs.zip"
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


def test_automatic_data_manager_names_use_output_then_concise_source() -> None:
    assert _automatic_output_name(
        "Replication Timing H1.tsv.gz",
        "replication_timing",
    ) == "dnascape_data_replication-timing_h1"


def test_gwseq_backend_hides_known_broken_minimum_and_maximum_aggregations() -> None:
    assert tuple(_track_aggregation_options("gwseq_io_pp")) == (
        "mean",
        "sum",
        "coverage",
        "std",
    )
    assert "min" in _track_aggregation_options("pyBigWig")
    assert "max" in _track_aggregation_options("pyBigWig")


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
            await user.should_see(marker="data-manager-source-url")
            user.find(marker="use-encode-data-example").click()

            url = cast(
                Input,
                next(iter(user.find(marker="data-manager-source-url").elements)),
            )
            assert url.value == (
                "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/encodeDCC/"
                "wgEncodeUwRepliSeq/wgEncodeUwRepliSeqK562WaveSignalRep1.bigWig"
            )
            assert feature.value == "replication_timing"
            await user.should_not_see("earlier = larger")
            # Filling the curated example is intentionally not a network action.
            await user.should_see(marker="data-manager-empty-result")

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
            for marker in (
                "data-manager-chromosome-scope",
                "data-manager-chromosome",
                "data-manager-bin-size",
                "data-manager-aggregation",
                "data-manager-track-missing",
                "data-manager-interpolation",
                "data-manager-smoothing",
                "data-manager-value-scaling",
                "data-manager-threshold-enabled",
            ):
                await user.should_see(marker=marker)
            await user.should_not_see(marker="data-manager-smooth-window")
            await user.should_not_see(marker="data-manager-multiplier-fields")
            await user.should_not_see(marker="data-manager-target-interval")
            await user.should_not_see(marker="data-manager-offset")
            await user.should_not_see(marker="data-manager-signal-transform")
            await user.should_not_see(marker="data-manager-threshold-fields")

            smoothing = cast(
                Select,
                next(iter(user.find(marker="data-manager-smoothing").elements)),
            )
            smoothing.set_value("moving_average")
            await user.should_see(marker="data-manager-smooth-window")
            smooth_window = cast(
                Number,
                next(iter(user.find(marker="data-manager-smooth-window").elements)),
            )
            assert smooth_window.value == 3

            value_scaling = cast(
                Select,
                next(iter(user.find(marker="data-manager-value-scaling").elements)),
            )
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
            value_scaling.set_value("interval")
            await user.should_not_see(marker="data-manager-multiplier-fields")
            await user.should_see(marker="data-manager-target-interval")
            await user.should_see(marker="data-manager-target-minimum")
            await user.should_see(marker="data-manager-target-maximum")
            await user.should_see(marker="data-manager-interval-invert")

            thresholds = cast(
                Checkbox,
                next(iter(user.find(marker="data-manager-threshold-enabled").elements)),
            )
            thresholds.set_value(True)
            await user.should_see(marker="data-manager-threshold-fields")
            await user.should_see(marker="data-manager-threshold-min")
            await user.should_see(marker="data-manager-threshold-max")
            threshold_max = cast(
                Number,
                next(iter(user.find(marker="data-manager-threshold-max").elements)),
            )
            threshold_max.set_value(10)

            bin_size = cast(
                Number,
                next(iter(user.find(marker="data-manager-bin-size").elements)),
            )
            assert bin_size.value == 10_000

            scope = cast(
                Select,
                next(iter(user.find(marker="data-manager-chromosome-scope").elements)),
            )
            assert scope.value == "single"
            assert tuple(scope.options) == ("single", "all")

            output_format = cast(
                Select,
                list(user.find(marker="data-manager-output-format").elements)[-1],
            )
            compression = cast(
                Checkbox,
                next(iter(user.find(marker="data-manager-gzip").elements)),
            )
            assert output_format.value == "tsv"
            assert tuple(output_format.options) == (
                "tsv",
                "csv",
                "txt",
                "npy",
                "npv",
                "npz",
            )
            assert compression.value is False

            preview_toggle = cast(
                Checkbox,
                next(iter(user.find(marker="data-manager-preview-toggle").elements)),
            )
            assert preview_toggle.value is False

            user.find(marker="action-convert").click()
            await user.should_see(marker="data-manager-result-1", retries=60)
            await user.should_see(marker="download-all-converted-data")
            await user.should_not_see(marker="data-manager-result-preview")

            preview_toggle.set_value(True)
            await user.should_see(marker="data-manager-result-preview", retries=60)
            await user.should_see(marker="data-manager-preview-chart", retries=60)

            scope.set_value("all")
            await user.should_see(marker="data-manager-all-scope-note")
            await user.should_not_see(marker="data-manager-single-region")
            await asyncio.sleep(0.1)
            refreshed_formats = {
                tuple(cast(Select, element).options)
                for element in user.find(marker="data-manager-output-format").elements
            }
            assert ("tsv", "csv", "npz") in refreshed_formats

            user.find(marker="remove-converted-data-1").click()
            await user.should_see(marker="data-manager-empty-result")

    asyncio.run(scenario())


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
                "data-manager-threshold-enabled",
            ):
                await user.should_see(marker=marker)

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

            thresholds = cast(
                Checkbox,
                next(iter(user.find(marker="data-manager-threshold-enabled").elements)),
            )
            thresholds.set_value(True)
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
            thresholds.set_value(False)
            await user.should_not_see(marker="data-manager-threshold-fields")

            user.find(marker="action-convert").click()
            await user.should_see(marker="data-manager-result-1", retries=60)

    asyncio.run(scenario())
