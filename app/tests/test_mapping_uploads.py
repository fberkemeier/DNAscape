"""Focused coverage for Mapping upload inspection and immediate-upload UX."""

from __future__ import annotations

import asyncio
import gzip
import io
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from nicegui.elements.upload import Upload
from nicegui.testing import user_simulation

import app.mapping_uploads as uploads
from app.mapping_service import materialize_track_upload
from app.mapping_uploads import (
    MAX_UPLOAD_BYTES,
    UploadInspectionError,
    UploadTooLargeError,
    inspect_upload,
    inspect_upload_event,
    materialize_many,
    materialize_upload,
)
from app.pages.common import (
    UploadClearEventArguments,
    data_upload,
    set_data_upload_locked,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_csv_header_detection_prefers_chr1_over_position_and_other_chromosomes() -> None:
    inspection = inspect_upload(
        "timing.csv",
        b"position,chr2,chr1\n0,2.0,1.0\n1,2.5,1.5\n",
        expected_form="x",
    )

    assert inspection.has_header is True
    assert inspection.delimiter == ","
    assert inspection.columns == ("position", "chr2", "chr1")
    assert inspection.numeric_columns == ("chr2", "chr1")
    assert inspection.default_column == "chr1"
    np.testing.assert_array_equal(materialize_upload(inspection), [1.0, 1.5])


def test_large_repository_csv_is_inspected_without_losing_ragged_chromosome_tails() -> None:
    source = REPOSITORY_ROOT / "data" / "replication_timing" / "replication_timing_H1.csv"
    if not source.is_file():
        pytest.skip("requires the optional local large-table benchmark fixture")
    assert source.stat().st_size == 53_806_241
    assert source.stat().st_size <= MAX_UPLOAD_BYTES

    inspection = inspect_upload(source.name, source.read_bytes(), expected_form="x")

    chromosomes = tuple(f"chr{index}" for index in range(1, 23))
    assert inspection.delimiter == "\t"
    assert inspection.shape == (249_000, 22)
    assert inspection.shape is not None
    assert inspection.shape[0] * inspection.shape[1] > 4_000_000
    assert inspection.columns == chromosomes
    assert inspection.numeric_columns == chromosomes
    assert inspection.default_column == "chr1"
    assert inspection.missing_columns == chromosomes[1:]
    assert inspection.missing_count == 2_602_550

    chr1 = materialize_upload(inspection)
    assert chr1.shape == (249_000,)
    assert chr1[0] == pytest.approx(206.04622947236015)
    assert chr1[-1] == pytest.approx(248.4035679038704)

    chr2 = materialize_upload(inspection, column="chr2")
    assert chr2.shape == (242_200,)
    assert chr2[0] == pytest.approx(253.2694993937086)
    assert chr2[-1] == pytest.approx(441.6402767830693)


def test_streaming_table_limit_is_enforced_before_numeric_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = b"chr1\tchr2\n1\t2\n3\t4\n5\t6\n"
    monkeypatch.setattr(uploads, "MAX_TABLE_CELLS", 5)

    with pytest.raises(UploadTooLargeError, match="cannot be kept safely"):
        inspect_upload("too_many_values.tsv", table, expected_form="x")


def test_pathological_single_row_width_is_rejected_before_csv_expansion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(uploads, "MAX_TABLE_COLUMNS", 3)

    with pytest.raises(UploadTooLargeError, match="too many columns"):
        inspect_upload(
            "too_wide.csv",
            b"chr1,chr2,chr3,chr4\n1,2,3,4\n",
            expected_form="x",
        )


def test_nonnumeric_signal_late_in_stream_is_not_missed_by_header_sample() -> None:
    rows = ["position,chr1"]
    rows.extend(f"{index},{index / 10}" for index in range(80))
    rows[-1] = "79,typo"

    with pytest.raises(UploadInspectionError, match="'chr1'.*nonnumeric"):
        inspect_upload("late_typo.csv", "\n".join(rows).encode(), expected_form="x")


def test_headerless_numeric_text_is_detected_without_losing_its_first_row() -> None:
    inspection = inspect_upload("field.txt", b"1 2\n3 4\n", expected_form="xt")

    assert inspection.has_header is False
    assert inspection.delimiter is None
    assert inspection.columns == ("Column 1", "Column 2")
    assert inspection.shape == (2, 2)
    np.testing.assert_array_equal(materialize_upload(inspection), [[1.0, 2.0], [3.0, 4.0]])


def test_one_row_table_uses_rdata_time_profile_convention() -> None:
    inspection = inspect_upload(
        "speed.csv",
        b"t0,t1,t2\n1.0,1.2,1.1\n",
        expected_form="auto",
    )

    assert inspection.inferred_form == "t"
    temporal = materialize_upload(replace(inspection, expected_form="t"))
    assert temporal.shape == (1, 3)
    np.testing.assert_array_equal(temporal, [[1.0, 1.2, 1.1]])


def test_xt_table_uses_all_non_coordinate_columns_by_default() -> None:
    inspection = inspect_upload(
        "space_time.csv",
        b"position,t0,t1\n0,0.1,0.2\n1,0.3,0.4\n",
        expected_form="xt",
    )

    assert inspection.default_column == "t0"
    np.testing.assert_allclose(
        materialize_upload(inspection, column=None),
        [[0.1, 0.2], [0.3, 0.4]],
    )
    with pytest.raises(UploadInspectionError, match="position-by-time matrix"):
        materialize_upload(inspection, column="t0")

    with pytest.raises(UploadInspectionError, match="not a selectable numeric signal"):
        materialize_upload(inspection, column="position")


def test_bp_coordinate_headers_are_not_offered_as_signal_profiles() -> None:
    inspection = inspect_upload(
        "legacy_profile.tsv",
        (
            b"chrom\tstart_bp\tend_bp\tposition_bp\treplication_timing\n"
            b"chr1\t0\t10\t5\t1.0\n"
            b"chr1\t10\t20\t15\t2.0\n"
        ),
        expected_form="x",
    )

    assert inspection.numeric_columns == ("replication_timing",)
    assert inspection.default_column == "replication_timing"
    np.testing.assert_array_equal(materialize_upload(inspection), [1.0, 2.0])


def test_missing_value_policies_preserve_vector_rows_and_matrix_shape() -> None:
    vector = inspect_upload(
        "profile.csv",
        b"position,chr1\n0,1\n1,\n2,3\n",
        expected_form="x",
    )
    assert vector.shape == (3, 2)
    assert vector.missing_columns == ("chr1",)
    assert vector.missing_count == 1
    with pytest.raises(UploadInspectionError, match="contain missing values"):
        materialize_upload(vector)
    np.testing.assert_array_equal(
        materialize_upload(vector, missing_policy="zero"),
        [1.0, 0.0, 3.0],
    )
    np.testing.assert_array_equal(
        materialize_upload(vector, missing_policy="interpolate"),
        [1.0, 2.0, 3.0],
    )
    ignored = materialize_upload(vector, missing_policy="ignore")
    assert np.isnan(ignored[1])
    np.testing.assert_array_equal(ignored[[0, 2]], [1.0, 3.0])

    matrix = inspect_upload(
        "field.csv",
        b"position,t0,t1\n0,0,0\n1,,2\n2,2,\n",
        expected_form="xt",
    )
    zeroed = materialize_upload(matrix, missing_policy="zero")
    interpolated = materialize_upload(matrix, missing_policy="interpolate")
    assert zeroed.shape == (3, 2)
    assert interpolated.shape == (3, 2)
    np.testing.assert_array_equal(zeroed, [[0.0, 0.0], [0.0, 2.0], [2.0, 0.0]])
    np.testing.assert_array_equal(interpolated, [[0.0, 0.0], [1.0, 2.0], [2.0, 2.0]])

    empty_slice = inspect_upload(
        "empty_slice.csv",
        b"position,t0,t1\n0,1,\n1,2,\n2,3,\n",
        expected_form="xt",
    )
    recovered = materialize_upload(empty_slice, missing_policy="interpolate")
    assert recovered.shape == (3, 2)
    np.testing.assert_array_equal(recovered, [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])


def test_nonnumeric_signal_typo_never_falls_back_to_position_column() -> None:
    with pytest.raises(UploadInspectionError, match="'chr1'.*nonnumeric"):
        inspect_upload(
            "broken.csv",
            b"position,chr1\n0,1\n1,typo\n2,3\n",
            expected_form="x",
        )

    with pytest.raises(UploadInspectionError, match="'t1'.*nonnumeric"):
        inspect_upload(
            "broken_matrix.csv",
            b"position,t0,t1\n0,1,2\n1,3,typo\n",
            expected_form="xt",
        )


def test_npz_metadata_and_array_key_are_detected_without_pickle() -> None:
    buffer = io.BytesIO()
    np.savez(
        buffer,
        values=np.asarray([[1.0, 2.0], [3.0, 4.0]]),
        columns=np.asarray(["chr1", "chr2"]),
        data_form=np.asarray(["xt"]),
    )

    inspection = inspect_upload("profiles.npz", buffer.getvalue(), expected_form="auto")

    assert inspection.array_keys == ("values",)
    assert inspection.default_array_key == "values"
    assert inspection.columns == ("chr1", "chr2")
    assert inspection.default_column == "chr1"
    assert inspection.inferred_form == "xt"
    np.testing.assert_array_equal(
        materialize_upload(inspection, array_key="values"),
        [[1.0, 2.0], [3.0, 4.0]],
    )


def test_npz_semantic_and_axis_metadata_are_not_offered_as_data_arrays() -> None:
    buffer = io.BytesIO()
    np.savez(
        buffer,
        values=np.asarray([[10.0, 20.0], [30.0, 40.0]]),
        data_form=np.asarray(["space"]),
        semantic_form=np.asarray(["regions"]),
        region_names=np.asarray(["origin_a", "origin_b"]),
        position_kb=np.asarray([0.0, 1.0]),
        time_min=np.asarray([0.0, 5.0]),
        replicate=np.asarray([0, 1]),
    )

    inspection = inspect_upload("regions.npz", buffer.getvalue(), expected_form="auto")

    assert inspection.array_keys == ("values",)
    assert inspection.semantic_form == "regions"
    assert inspection.region_names == ("origin_a", "origin_b")
    assert inspection.inferred_form == "regions"


def test_multiple_profile_uploads_stack_in_time_order() -> None:
    first = inspect_upload("t0.csv", b"chr1\n1\n2\n", expected_form="xt")
    second = inspect_upload("t1.csv", b"chr1\n3\n4\n", expected_form="xt")

    np.testing.assert_array_equal(
        materialize_many([first, second], columns=["chr1", "chr1"]),
        [[1.0, 3.0], [2.0, 4.0]],
    )


def test_non_finite_arrays_and_bounded_gzip_expansion_are_rejected(monkeypatch: Any) -> None:
    buffer = io.BytesIO()
    np.save(buffer, np.asarray([1.0, np.inf]), allow_pickle=False)
    with pytest.raises(UploadInspectionError, match="infinite"):
        inspect_upload("bad.npy", buffer.getvalue(), expected_form="x")

    missing_buffer = io.BytesIO()
    np.save(missing_buffer, np.asarray([1.0, np.nan, 3.0]), allow_pickle=False)
    missing = inspect_upload("missing.npy", missing_buffer.getvalue(), expected_form="x")
    assert missing.missing_count == 1
    with pytest.raises(UploadInspectionError, match="contain missing values"):
        materialize_upload(missing)
    np.testing.assert_array_equal(
        materialize_upload(missing, missing_policy="interpolate"),
        [1.0, 2.0, 3.0],
    )

    monkeypatch.setattr(uploads, "MAX_DECOMPRESSED_BYTES", 32)
    compressed = gzip.compress(b"1\n" * 17)
    with pytest.raises(UploadTooLargeError, match="expands beyond"):
        inspect_upload("large.txt.gz", compressed, expected_form="x")


def test_bedgraph_uses_public_track_binning_and_validates_finite_output() -> None:
    inspection = inspect_upload(
        "signal.bedgraph",
        b"chr1 0 1000 1.0\nchr1 1000 2000 3.0\n",
        expected_form="x",
    )

    assert inspection.inferred_form == "track"
    assert inspection.default_column == "value"
    np.testing.assert_allclose(
        materialize_track_upload(
            inspection,
            chromosome="chr1",
            start=0,
            end=2000,
            bin_size=1000,
        ),
        [1.0, 3.0],
    )


def test_bedgraph_accepts_standard_ucsc_comment_track_and_browser_lines() -> None:
    inspection = inspect_upload(
        "ucsc.bedgraph",
        (
            b"# zero-based, half-open coords\n"
            b'track name="BedGraph Format" color=200,100,0\n'
            b"browser position chr19:49302000-49302600\n"
            b"chr19\t49302000\t49302300\t-1.0\n"
            b"chr19\t49302300\t49302600\t-0.75\n"
        ),
        expected_form="x",
    )

    assert inspection.inferred_form == "track"
    assert inspection.shape == (2, 4)
    assert inspection.default_column == "value"
    np.testing.assert_allclose(
        materialize_track_upload(
            inspection,
            chromosome="chr19",
            start=49_302_000,
            end=49_302_600,
            bin_size=300,
        ),
        [-1.0, -0.75],
    )


def test_three_column_bed_is_a_unit_valued_genomic_track() -> None:
    inspection = inspect_upload(
        "sns_origins.bed.gz",
        gzip.compress(
            b"chr1\t0\t500\nchr1\t1000\t1500\nchr2\t0\t1000\n"
        ),
        expected_form="x",
    )

    assert inspection.source_format == "bedgraph"
    assert inspection.inferred_form == "track"
    assert inspection.columns == ("chromosome", "start", "end")
    assert inspection.numeric_columns == ("value",)
    assert inspection.default_column == "value"
    np.testing.assert_allclose(
        materialize_track_upload(
            inspection,
            chromosome="chr1",
            start=0,
            end=2000,
            bin_size=1000,
            aggregation="coverage",
        ),
        [0.5, 0.5],
    )


def test_bed9_sns_peaks_use_the_quantitative_fourth_column() -> None:
    inspection = inspect_upload(
        "K562_Peaks_FDR15.bed.gz",
        gzip.compress(
            b"# Cell line = K562\n"
            b"chr1\t756600\t756699\t4.27529\t427\tna\t756600\t756700\t8,119,38\n"
        ),
        expected_form="x",
    )

    assert inspection.source_format == "bedgraph"
    assert inspection.columns[:4] == ("chromosome", "start", "end", "value")
    assert inspection.default_column == "value"
    np.testing.assert_allclose(
        materialize_track_upload(
            inspection,
            chromosome="chr1",
            start=756_600,
            end=756_700,
            bin_size=100,
            aggregation="mean",
        ),
        [4.27529],
    )


def test_auto_upload_replacement_and_cross_remove_clear_retained_bytes() -> None:
    retained: dict[str, uploads.UploadInspection] = {}
    removed: list[str] = []

    async def receive(event: Any) -> bool:
        try:
            inspection = await inspect_upload_event(event, expected_form="x")
        except (OSError, TypeError, ValueError):
            return False
        retained.clear()
        retained[event.file.name] = inspection
        return True

    def clear(event: UploadClearEventArguments) -> None:
        removed.append(event.name)
        retained.pop(event.name, None)

    def page() -> None:
        data_upload(
            accept=".csv",
            accepted_formats="CSV",
            marker="test-upload",
            show_accepted_formats=False,
            on_upload=receive,
            on_clear=clear,
            hide_when_loaded=True,
        )

    async def wait_for(predicate: Any) -> None:
        for _ in range(200):
            if predicate():
                return
            await asyncio.sleep(0.01)
        raise AssertionError("Timed out waiting for upload state.")

    async def scenario() -> None:
        async with user_simulation(page) as user:
            await user.open("/")
            uploader = next(iter(user.find(kind=Upload).elements))
            assert uploader._props["auto-upload"] is True
            assert uploader._props["accept"] == ".csv,text/csv"

            await uploader.handle_uploads(
                [Upload.SmallFileUpload("same.csv", "text/csv", b"chr1\n1\n2\n")]
            )
            await wait_for(lambda: retained.get("same.csv") is not None)
            await user.should_see("same.csv")
            await user.should_not_see("0 B")
            await user.should_not_see("100%")
            assert uploader.visible is False
            await user.should_see(marker="test-upload-file")
            completed_row = next(iter(user.find(marker="test-upload-file").elements))
            assert completed_row._props["data-upload-drop-target"] == "true"
            drop_listener = next(
                listener
                for listener in completed_row._event_listeners.values()
                if listener.type == "drop"
            )
            assert drop_listener.js_handler is not None
            assert "q.reset()" in drop_listener.js_handler
            assert "q.addFiles(files.slice(0, 1))" in drop_listener.js_handler
            assert "new DataTransfer()" not in drop_listener.js_handler
            finish_listener = next(
                listener
                for listener in uploader._event_listeners.values()
                if listener.type == "finish"
            )
            assert finish_listener.js_handler is not None
            assert "'reset'" in finish_listener.js_handler

            await uploader.handle_uploads(
                [Upload.SmallFileUpload("same.csv", "text/csv", b"chr1\n8\n9\n")]
            )
            await wait_for(
                lambda: bool(
                    retained.get("same.csv") and retained["same.csv"].content.endswith(b"8\n9\n")
                )
            )
            assert removed == []
            await user.should_see("same.csv")
            await user.should_not_see("Inspecting…")
            assert len(user.find(marker="test-upload-file").elements) == 1

            await uploader.handle_uploads(
                [Upload.SmallFileUpload("bad.csv", "text/csv", b"label\nnot-a-number\n")]
            )
            await asyncio.sleep(0.05)
            assert retained["same.csv"].content.endswith(b"8\n9\n")
            await user.should_see("same.csv")
            await user.should_not_see("bad.csv")

            user.find(marker="test-upload-remove").click()
            await user.should_not_see("same.csv")
            await wait_for(lambda: not retained)
            assert removed == ["same.csv"]
            assert uploader.visible is True

    asyncio.run(scenario())


def test_invalid_upload_never_gets_a_completed_file_row() -> None:
    def page() -> None:
        async def reject_invalid(event: Any) -> bool:
            try:
                await inspect_upload_event(event, expected_form="x")
            except UploadInspectionError:
                return False
            return True

        data_upload(
            accept=".csv",
            accepted_formats="CSV",
            marker="invalid-upload",
            show_accepted_formats=False,
            on_upload=reject_invalid,
        )

    async def scenario() -> None:
        async with user_simulation(page) as user:
            await user.open("/")
            uploader = next(iter(user.find(kind=Upload).elements))
            await uploader.handle_uploads(
                [Upload.SmallFileUpload("invalid.csv", "text/csv", b"name\nalice\n")]
            )
            await asyncio.sleep(0.05)
            await user.should_not_see("invalid.csv")

    asyncio.run(scenario())


def test_multiple_auto_uploads_keep_selection_order_and_individual_remove_rows() -> None:
    received: list[str] = []
    release_first = asyncio.Event()

    async def receive(event: Any) -> bool:
        # Make the first item observably slower: the shared uploader must still
        # serialize callbacks in selection order for XT track stacking.
        if event.file.name == "t0.csv":
            await release_first.wait()
        received.append(event.file.name)
        return True

    def page() -> None:
        data_upload(
            accept=".csv",
            accepted_formats="CSV",
            marker="multi-upload",
            show_accepted_formats=False,
            on_upload=receive,
            multiple=True,
        )

    async def scenario() -> None:
        async with user_simulation(page) as user:
            await user.open("/")
            uploader = next(iter(user.find(kind=Upload).elements))
            assert uploader._props["multiple"] is True
            await uploader.handle_uploads(
                [
                    Upload.SmallFileUpload("t0.csv", "text/csv", b"chr1\n1\n"),
                    Upload.SmallFileUpload("t1.csv", "text/csv", b"chr1\n2\n"),
                ]
            )
            await user.should_see("Inspecting…")
            release_first.set()
            await user.should_see("t0.csv")
            await user.should_see("t1.csv")
            await user.should_not_see("Inspecting…")
            assert received == ["t0.csv", "t1.csv"]
            assert len(user.find(marker="multi-upload-file").elements) == 2

    asyncio.run(scenario())


def test_multiple_upload_delete_all_is_scoped_ordered_and_lock_safe() -> None:
    cleared: list[tuple[str, tuple[str, ...]]] = []
    received: list[str] = []

    async def receive(event: Any) -> bool:
        received.append(event.file.name)
        return True

    def clear(event: UploadClearEventArguments) -> None:
        cleared.append((event.name, event.names))

    def page() -> None:
        data_upload(
            accept=".csv",
            accepted_formats="CSV",
            marker="bulk-upload",
            show_accepted_formats=False,
            on_upload=receive,
            on_clear=clear,
            multiple=True,
        )

    async def wait_for(predicate: Any) -> None:
        for _ in range(200):
            if predicate():
                return
            await asyncio.sleep(0.01)
        raise AssertionError("Timed out waiting for upload state.")

    async def scenario() -> None:
        async with user_simulation(page) as user:
            await user.open("/")
            uploader = next(iter(user.find(kind=Upload).elements))
            await user.should_not_see(marker="bulk-upload-delete-all")

            await uploader.handle_uploads(
                [Upload.SmallFileUpload("first.csv", "text/csv", b"chr1\n1\n")]
            )
            await wait_for(lambda: received == ["first.csv"])
            await user.should_see("first.csv")
            await user.should_not_see("Inspectingâ€¦")
            await user.should_not_see(marker="bulk-upload-delete-all")

            await uploader.handle_uploads(
                [Upload.SmallFileUpload("second.csv", "text/csv", b"chr1\n2\n")]
            )
            await wait_for(lambda: received == ["first.csv", "second.csv"])
            await user.should_see(marker="bulk-upload-delete-all")
            await user.should_not_see("Inspectingâ€¦")
            assert len(user.find(marker="bulk-upload-file").elements) == 2

            delete_all = next(iter(user.find(marker="bulk-upload-delete-all").elements))
            set_data_upload_locked(uploader, True)
            assert delete_all.enabled is False
            user.find(marker="bulk-upload-delete-all").click()
            await asyncio.sleep(0.05)
            assert cleared == []
            assert len(user.find(marker="bulk-upload-file").elements) == 2

            set_data_upload_locked(uploader, False)
            assert delete_all.enabled is True
            user.find(marker="bulk-upload-delete-all").click()
            await wait_for(lambda: len(cleared) == 1)
            assert cleared == [("", ("first.csv", "second.csv"))]
            await user.should_not_see(marker="bulk-upload-file")
            await user.should_not_see(marker="bulk-upload-delete-all")

            await uploader.handle_uploads(
                [Upload.SmallFileUpload("same.csv", "text/csv", b"chr1\n3\n")]
            )
            await wait_for(lambda: received[-1:] == ["same.csv"])
            await user.should_see("same.csv")
            await user.should_not_see("Inspectingâ€¦")
            await uploader.handle_uploads(
                [Upload.SmallFileUpload("same.csv", "text/csv", b"chr1\n4\n")]
            )
            await wait_for(lambda: received.count("same.csv") == 2)
            await user.should_not_see("Inspectingâ€¦")
            assert len(user.find(marker="bulk-upload-file").elements) == 1
            await user.should_not_see(marker="bulk-upload-delete-all")
            assert cleared == [("", ("first.csv", "second.csv"))]

    asyncio.run(scenario())


def test_multiple_upload_removals_wait_for_inflight_inspection() -> None:
    entered = asyncio.Event()
    release = asyncio.Event()
    cleared: list[tuple[str, tuple[str, ...]]] = []

    async def receive(_event: Any) -> bool:
        entered.set()
        await release.wait()
        return True

    def clear(event: UploadClearEventArguments) -> None:
        cleared.append((event.name, event.names))

    def page() -> None:
        data_upload(
            accept=".csv",
            accepted_formats="CSV",
            marker="inspection-locked-upload",
            show_accepted_formats=False,
            on_upload=receive,
            on_clear=clear,
            multiple=True,
            initial_names=("first.csv", "second.csv"),
        )

    async def scenario() -> None:
        async with user_simulation(page) as user:
            await user.open("/")
            uploader = next(iter(user.find(kind=Upload).elements))
            pending = asyncio.create_task(
                uploader.handle_uploads(
                    [Upload.SmallFileUpload("third.csv", "text/csv", b"chr1\n3\n")]
                )
            )
            await asyncio.wait_for(entered.wait(), timeout=1)
            delete_all = next(
                iter(user.find(marker="inspection-locked-upload-delete-all").elements)
            )
            assert delete_all.enabled is False
            user.find(marker="inspection-locked-upload-delete-all").click()
            await asyncio.sleep(0.05)
            assert cleared == []

            release.set()
            await pending
            await user.should_not_see(
                marker="inspection-locked-upload-inspecting",
                retries=80,
            )
            await user.should_see("third.csv")
            delete_all = next(
                iter(user.find(marker="inspection-locked-upload-delete-all").elements)
            )
            assert delete_all.enabled is True
            assert len(user.find(marker="inspection-locked-upload-file").elements) == 3

    asyncio.run(scenario())
