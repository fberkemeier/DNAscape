"""Interaction coverage for independent Mapping batch raw-data exports."""

from __future__ import annotations

import asyncio
from typing import Any, cast

import numpy as np
import pytest
from nicegui.elements.button import Button
from nicegui.elements.checkbox import Checkbox
from nicegui.elements.input import Input
from nicegui.elements.select import Select
from nicegui.elements.upload import Upload
from nicegui.testing import user_simulation

from app.main import index
from app.mapping_service import MappingRequest, MappingResult
from app.simulation_exports import ExportArtifact


def _mapped_result(request: MappingRequest) -> MappingResult:
    timing = np.asarray(request.inputs["replication_timing"], dtype=float)
    mapped = timing / 10.0
    return MappingResult(
        target=request.route.target,
        target_key=request.route.target_key,
        target_label="Firing rate",
        method=request.route.method,
        output_form="x",
        output_dependence="space",
        primary_output="firing_rate",
        outputs={
            "firing_rate": mapped,
            "replication_timing_fit": timing,
        },
        output_forms={
            "firing_rate": "x",
            "replication_timing_fit": "x",
        },
        x_axis=np.arange(timing.size, dtype=float),
        time_axis=np.asarray([], dtype=float),
        region_labels=(),
        elapsed_seconds=0.02,
        assumptions={},
    )


async def _wait_for(predicate: Any, *, attempts: int = 80) -> None:
    for _ in range(attempts):
        if predicate():
            return
        await asyncio.sleep(0.02)
    raise AssertionError("Timed out while waiting for the Mapping download action.")


def test_mapping_batch_raw_rows_are_bounded_for_large_runs() -> None:
    from app.pages.mapping import _mapping_raw_page_bounds

    page, page_count, start, end = _mapping_raw_page_bounds(50_000, 10_000)

    assert (page, page_count) == (1_000, 1_000)
    assert (start, end) == (49_950, 50_000)
    assert end - start == 50


def test_mapping_batch_raw_rows_are_independent_and_bulk_uses_each_preference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.pages import mapping as mapping_page

    export_calls: list[dict[str, Any]] = []
    archive_calls: list[tuple[str, ...]] = []

    def fake_run(
        request: MappingRequest,
        report: Any,
        _cancel: Any,
    ) -> MappingResult:
        report(0.7, "Mapping batch data...")
        return _mapped_result(request)

    def fake_raw_export(
        result: MappingResult,
        output_key: str,
        file_format: str,
        *,
        filename: str | None = None,
        compression: str = "none",
        **_kwargs: Any,
    ) -> ExportArtifact:
        call = {
            "result": result,
            "output_key": output_key,
            "file_format": file_format,
            "filename": filename,
            "compression": compression,
        }
        export_calls.append(call)
        suffix = file_format.casefold()
        compressed_suffix = ".gz" if compression == "gzip" else ""
        return ExportArtifact(
            data=f"{filename}:{output_key}".encode(),
            filename=f"{filename}.{suffix}{compressed_suffix}",
            media_type="application/octet-stream",
        )

    def fake_archive(
        artifacts: Any,
        *,
        filename: str,
        **_kwargs: Any,
    ) -> ExportArtifact:
        names = tuple(artifact.filename for artifact in artifacts)
        archive_calls.append(names)
        return ExportArtifact(b"zip", filename, "application/zip")

    monkeypatch.setattr(mapping_page, "run_mapping", fake_run)
    monkeypatch.setattr(mapping_page, "build_mapping_raw_export", fake_raw_export)
    monkeypatch.setattr(mapping_page, "build_artifact_archive", fake_archive)
    monkeypatch.setattr(mapping_page, "cache_artifact", lambda *_args, **_kwargs: None)

    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-mapping").click()
            await user.should_see(marker="source-replication-timing-upload", retries=60)
            uploader = cast(
                Upload,
                next(
                    iter(
                        user.find(
                            marker="source-replication-timing-upload"
                        ).elements
                    )
                ),
            )
            await uploader.handle_uploads(
                [
                    Upload.SmallFileUpload(
                        "timing_a.tsv",
                        "text/tab-separated-values",
                        b"timing\n8\n7\n9\n",
                    ),
                    Upload.SmallFileUpload(
                        "timing_b.tsv",
                        "text/tab-separated-values",
                        b"timing\n6\n5\n7\n",
                    ),
                ]
            )
            await user.should_see("Datasets: 2", retries=40)
            user.find(marker="action-map").click()
            await user.should_see(marker="mapping-batch-raw-files", retries=60)
            assert user.client is not None
            user.find("Plot settings").click()
            shared_type = next(iter(user.find(marker="mapping-curve-type").elements))
            with user.client:
                shared_type.set_value("dashed")
            await asyncio.sleep(0)
            user.find("Curves").click()
            individual_type = next(iter(user.find(marker="mapping-dataset-curve-type-2").elements))
            assert "default" not in individual_type.options
            assert individual_type.value == "dashed"
            with user.client:
                individual_type.set_value("dotted")
            await asyncio.sleep(0)
            chart = next(iter(user.find(marker="mapping-result-plot").elements))
            assert [series["lineStyle"]["type"] for series in chart.options["series"]] == ["dashed", "dotted"]
            stats = next(iter(user.find(marker="mapping-stats").elements))

            first_row = next(
                iter(
                    user.find(
                        marker="mapping-batch-raw-row-mapping-case-1"
                    ).elements
                )
            )
            second_row = next(
                iter(
                    user.find(
                        marker="mapping-batch-raw-row-mapping-case-2"
                    ).elements
                )
            )
            assert first_row is not second_row

            def controls(
                case: int,
            ) -> tuple[Input, Select, Checkbox, Button, Button]:
                key = f"mapping-case-{case}"
                return (
                    cast(
                        Input,
                        next(
                            iter(
                                user.find(
                                    marker=f"mapping-batch-raw-name-{key}"
                                ).elements
                            )
                        ),
                    ),
                    cast(
                        Select,
                        next(
                            iter(
                                user.find(
                                    marker=f"mapping-batch-raw-type-{key}"
                                ).elements
                            )
                        ),
                    ),
                    cast(
                        Checkbox,
                        next(
                            iter(
                                user.find(
                                    marker=f"mapping-batch-raw-compression-{key}"
                                ).elements
                            )
                        ),
                    ),
                    cast(
                        Button,
                        next(
                            iter(
                                user.find(
                                    marker=f"mapping-download-raw-{key}"
                                ).elements
                            )
                        ),
                    ),
                    cast(
                        Button,
                        next(
                            iter(
                                user.find(
                                    marker=f"mapping-remove-raw-{key}"
                                ).elements
                            )
                        ),
                    ),
                )

            (
                first_name,
                first_type,
                first_compress,
                first_download,
                first_remove,
            ) = controls(1)
            (
                second_name,
                second_type,
                second_compress,
                second_download,
                second_remove,
            ) = controls(2)

            # Every result owns a distinct set of controls. The generated
            # filenames retain target-first case identity instead of reverting
            # to one shared batch filename.
            assert first_name is not second_name
            assert first_type is not second_type
            assert first_compress is not second_compress
            assert first_download is not second_download
            assert first_remove is not second_remove
            assert str(first_name.value).startswith("rmap_fr-")
            assert str(second_name.value).startswith("rmap_fr-")
            assert "timing-a" in str(first_name.value)
            assert "timing-b" in str(second_name.value)
            assert first_name.value != second_name.value
            assert first_type.value == second_type.value == "TSV"
            assert first_type.options[0] == second_type.options[0] == "TSV"
            assert first_compress.value is second_compress.value is False

            with user.client:
                first_name.set_value("first_custom")
                first_type.set_value("TSV")
                first_compress.set_value(False)
                second_name.set_value("second_custom")
                second_type.set_value("CSV")
                second_compress.set_value(True)
            assert first_name.value == "first_custom"
            assert second_name.value == "second_custom"
            assert first_type.value == "TSV"
            assert second_type.value == "CSV"
            assert first_compress.value is False
            assert second_compress.value is True

            # Refreshing the rows for another output and returning must restore
            # the independent preferences for the original output key.
            output = cast(
                Select,
                next(iter(user.find(marker="mapping-preview-output").elements)),
            )
            with user.client:
                output.set_value("replication_timing_fit")
            await asyncio.sleep(0)
            fit_first_name, *_ = controls(1)
            assert fit_first_name.value != "first_custom"
            with user.client:
                output.set_value("firing_rate")
            await asyncio.sleep(0)
            (
                first_name,
                first_type,
                first_compress,
                first_download,
                first_remove,
            ) = controls(1)
            (
                second_name,
                second_type,
                second_compress,
                second_download,
                second_remove,
            ) = controls(2)
            assert first_name.value == "first_custom"
            assert first_type.value == "TSV"
            assert first_compress.value is False
            assert second_name.value == "second_custom"
            assert second_type.value == "CSV"
            assert second_compress.value is True

            user.find(marker="mapping-download-raw-mapping-case-1").click()
            await _wait_for(lambda: len(export_calls) >= 1)
            user.find(marker="mapping-download-raw-mapping-case-2").click()
            await _wait_for(lambda: len(export_calls) >= 2)
            assert [
                (
                    call["output_key"],
                    call["file_format"],
                    call["filename"],
                    call["compression"],
                )
                for call in export_calls[:2]
            ] == [
                ("firing_rate", "TSV", "first_custom", "none"),
                ("firing_rate", "CSV", "second_custom", "gzip"),
            ]

            user.find(marker="mapping-download-all-raw").click()
            await _wait_for(lambda: bool(archive_calls))
            assert [
                (
                    call["output_key"],
                    call["file_format"],
                    call["filename"],
                    call["compression"],
                )
                for call in export_calls[2:]
            ] == [
                ("firing_rate", "TSV", "first_custom", "none"),
                ("firing_rate", "CSV", "second_custom", "gzip"),
            ]
            assert archive_calls == [
                ("first_custom.tsv", "second_custom.csv.gz")
            ]

            # Removing one batch row keeps the other result and reconciles the
            # raw-data surface back to the shared single-file controls.
            user.find(marker="mapping-remove-raw-mapping-case-1").click()
            await user.should_see(marker="raw-output-row", retries=40)
            await user.should_not_see(marker="mapping-batch-raw-files")
            single_row = next(iter(user.find(marker="raw-output-row").elements))
            single_remove = cast(
                Button,
                next(iter(user.find(marker="action-remove-raw-data").elements)),
            )
            single_name = cast(
                Input,
                next(iter(user.find(marker="mapping-raw-filename").elements)),
            )
            assert single_row.visible
            assert single_remove.visible
            assert "timing-b" in str(single_name.value)

            # The same close control clears the final result, returning the
            # preview and raw-data surface to their empty state.
            user.find(marker="action-remove-raw-data").click()
            await user.should_see("No result yet", retries=40)
            await user.should_not_see(marker="raw-output-row")
            await user.should_not_see(marker="mapping-result-plot")
            assert not stats.visible

            # A fresh batch can also be discarded in one operation. Delete all
            # resets the same plot, statistics, and raw-data state as removing
            # the final row individually.
            user.find(marker="action-map").click()
            await user.should_see(marker="mapping-batch-raw-files", retries=60)
            await user.should_see(marker="mapping-delete-all-raw")
            user.find(marker="mapping-delete-all-raw").click()
            await user.should_see("No result yet", retries=40)
            await user.should_not_see(marker="mapping-batch-raw-files")
            await user.should_not_see(marker="raw-output-row")
            await user.should_not_see(marker="mapping-result-plot")
            assert not stats.visible

    asyncio.run(scenario())
