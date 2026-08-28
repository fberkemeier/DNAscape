"""Interaction coverage for independent Simulation batch raw exports."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
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
from app.simulation_exports import ExportArtifact
from app.simulation_service import ConstantSimulationRequest, ConstantSimulationResult


def _result_for(request: ConstantSimulationRequest) -> ConstantSimulationResult:
    firing = np.asarray(request.firing_rate, dtype=float).reshape(-1)
    timing = firing * 10.0
    return ConstantSimulationResult(
        position_kb=np.arange(timing.size, dtype=float),
        replication_timing=timing,
        elapsed_seconds=0.01,
        replicates=request.replicates,
        geometry=request.geometry,
        time_grid=np.asarray([], dtype=float),
        outputs={
            "replication_timing": timing,
            "origin_efficiency": firing,
        },
    )


async def _wait_for(predicate: Any, *, attempts: int = 80) -> None:
    for _ in range(attempts):
        if predicate():
            return
        await asyncio.sleep(0.02)
    raise AssertionError("Timed out while waiting for the Simulation download action.")


def test_simulation_bulk_clear_retains_an_upload_completed_after_its_snapshot() -> None:
    from app.pages.simulation import _remove_simulation_uploads

    inspections: list[Any] = [
        SimpleNamespace(name="old_a.csv"),
        SimpleNamespace(name="old_b.csv"),
        SimpleNamespace(name="late.csv"),
    ]
    selections = {
        name: {"column": "signal", "array_key": None}
        for name in ("old_a.csv", "old_b.csv", "late.csv")
    }
    _remove_simulation_uploads(
        inspections,
        selections,
        SimpleNamespace(name="", names=("old_a.csv", "old_b.csv")),
    )

    assert [inspection.name for inspection in inspections] == ["late.csv"]
    assert set(selections) == {"late.csv"}


def test_simulation_batch_raw_rows_are_bounded_for_large_runs() -> None:
    from app.pages.simulation import _simulation_raw_page_bounds

    page, page_count, start, end = _simulation_raw_page_bounds(50_000, 10_000)

    assert (page, page_count) == (1_000, 1_000)
    assert (start, end) == (49_950, 50_000)
    assert end - start == 50


def test_simulation_batch_raw_rows_are_independent_and_bulk_uses_each_preference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.pages import simulation as simulation_page

    export_calls: list[dict[str, Any]] = []
    archive_calls: list[tuple[str, ...]] = []

    def fake_run(
        request: ConstantSimulationRequest,
        report: Any,
        _cancel: Any,
    ) -> ConstantSimulationResult:
        report(0.8, "Simulating batch data...")
        return _result_for(request)

    def fake_raw_export(
        result: ConstantSimulationResult,
        output_key: str,
        file_format: str,
        *,
        filename: str | None = None,
        compression: str = "none",
        **_kwargs: Any,
    ) -> ExportArtifact:
        export_calls.append(
            {
                "result": result,
                "output_key": output_key,
                "file_format": file_format,
                "filename": filename,
                "compression": compression,
            }
        )
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

    monkeypatch.setattr(simulation_page, "run_constant_simulation", fake_run)
    monkeypatch.setattr(simulation_page, "build_raw_export", fake_raw_export)
    monkeypatch.setattr(simulation_page, "build_artifact_archive", fake_archive)
    monkeypatch.setattr(simulation_page, "cache_artifact", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        simulation_page,
        "get_export_settings",
        lambda: SimpleNamespace(raw_compression="gzip"),
    )

    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-simulation").click()
            await user.should_see("Simulate DNA replication", retries=60)
            firing_mode = cast(
                Select,
                next(iter(user.find(marker="source-firing-rate-mode").elements)),
            )
            assert user.client is not None
            with user.client:
                firing_mode.set_value("profile")
            uploader = cast(
                Upload,
                next(iter(user.find(marker="source-firing-rate-upload").elements)),
            )
            await uploader.handle_uploads(
                [
                    Upload.SmallFileUpload(
                        "sample_a.csv",
                        "text/csv",
                        b"firing_rate\n0.1\n0.2\n0.3\n",
                    ),
                    Upload.SmallFileUpload(
                        "sample_b.csv",
                        "text/csv",
                        b"firing_rate\n0.4\n0.5\n0.6\n",
                    ),
                ]
            )
            await user.should_see("Files: 2", retries=40)
            user.find(marker="action-simulate").click()
            await user.should_see("2 datasets complete", retries=60)
            user.find("Raw data").click()
            await user.should_see(marker="simulation-batch-raw-surface")

            def controls(index: int) -> tuple[Input, Select, Checkbox, Button]:
                return (
                    cast(
                        Input,
                        next(
                            iter(
                                user.find(
                                    marker=f"simulation-batch-raw-name-{index}"
                                ).elements
                            )
                        ),
                    ),
                    cast(
                        Select,
                        next(
                            iter(
                                user.find(
                                    marker=f"simulation-batch-raw-type-{index}"
                                ).elements
                            )
                        ),
                    ),
                    cast(
                        Checkbox,
                        next(
                            iter(
                                user.find(
                                    marker=f"simulation-batch-raw-compression-{index}"
                                ).elements
                            )
                        ),
                    ),
                    cast(
                        Button,
                        next(
                            iter(
                                user.find(
                                    marker=f"action-download-simulation-raw-{index}"
                                ).elements
                            )
                        ),
                    ),
                )

            first_name, first_type, first_compress, first_download = controls(1)
            second_name, second_type, second_compress, second_download = controls(2)
            assert first_name is not second_name
            assert first_type is not second_type
            assert first_compress is not second_compress
            assert first_download is not second_download
            assert first_name.value == (
                "dnascape_simulation_replication_timing_sample-a"
            )
            assert second_name.value == (
                "dnascape_simulation_replication_timing_sample-b"
            )

            with user.client:
                first_name.set_value("first_custom")
                first_type.set_value("TSV")
                first_compress.set_value(False)
                second_name.set_value("second_custom")
                second_type.set_value("CSV")
                second_compress.set_value(True)

            # Preferences belong to one output. Switching away and back
            # reconstructs the controls while restoring each edited row.
            output = cast(
                Select,
                next(iter(user.find(marker="simulation-output").elements)),
            )
            with user.client:
                output.set_value("Origin efficiency")
            await asyncio.sleep(0)
            efficiency_name, *_ = controls(1)
            assert efficiency_name.value.startswith(
                "dnascape_simulation_origin_efficiency_"
            )
            with user.client:
                output.set_value("Replication timing")
            await asyncio.sleep(0)
            first_name, first_type, first_compress, _ = controls(1)
            second_name, second_type, second_compress, _ = controls(2)
            assert (first_name.value, first_type.value, first_compress.value) == (
                "first_custom",
                "TSV",
                False,
            )
            assert (second_name.value, second_type.value, second_compress.value) == (
                "second_custom",
                "CSV",
                True,
            )

            user.find(marker="action-download-simulation-raw-1").click()
            await _wait_for(lambda: len(export_calls) >= 1)
            user.find(marker="action-download-simulation-raw-2").click()
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
                ("replication_timing", "TSV", "first_custom", "none"),
                ("replication_timing", "CSV", "second_custom", "gzip"),
            ]

            user.find(marker="action-download-all-simulation-raw").click()
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
                ("replication_timing", "TSV", "first_custom", "none"),
                ("replication_timing", "CSV", "second_custom", "gzip"),
            ]
            assert archive_calls == [("first_custom.tsv", "second_custom.csv.gz")]

    asyncio.run(scenario())
