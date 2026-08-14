"""Adversarial coverage for retained state and non-scientific app pages."""

from __future__ import annotations

import gzip
import importlib
from pathlib import Path

import numpy as np
from streamlit.testing.v1 import AppTest

from app_streamlit.app_shell import GENERATED_ARTIFACTS_KEY
from app_streamlit.data_service import DatasetRecord, SourceInspection
from app_streamlit.state import (
    DATASETS_KEY,
    INSPECTIONS_KEY,
    UPLOADS_KEY,
    WORKFLOWS_KEY,
    StoredWorkflowResult,
    UploadRecord,
)

APP_PATH = Path(__file__).resolve().parents[1] / "streamlit_app.py"


def _app() -> AppTest:
    return AppTest.from_file(str(APP_PATH), default_timeout=60)


def _assert_clean(app: AppTest) -> None:
    assert not app.exception, [str(item) for item in app.exception]
    assert not app.error, [item.value for item in app.error]


def _button(app: AppTest, key: str):
    return next(button for button in app.button if button.key == key)


def test_router_callbacks_preserve_prepared_data_and_workflow_results() -> None:
    payload = b"position_kb,value\n0,1\n1,2\n2,3\n"
    upload = UploadRecord("source-digest", "profile.csv", payload, "text/csv")
    inspection = SourceInspection(
        digest=upload.digest,
        name=upload.name,
        source_format="csv",
        size_bytes=len(payload),
        columns=("position_kb", "value"),
        shape=(3, 2),
        dtype="float64",
        delimiter=",",
    )
    workflow_payload = {
        "output_forms": {"replication_timing": "space"},
        "replication_timing": np.linspace(0.0, 1.0, 5),
        "position_kb": np.arange(5, dtype=float),
    }

    app = _app().run()
    _assert_clean(app)
    app.session_state[UPLOADS_KEY] = {upload.digest: upload}
    app.session_state[INSPECTIONS_KEY] = {upload.digest: inspection}
    app.session_state[WORKFLOWS_KEY] = {
        "mapping": {
            "result": StoredWorkflowResult(workflow_payload, "mapping-signature")
        }
    }

    app = _button(app, "workflow-card-button-data-management").click().run()
    _assert_clean(app)
    convert = _button(app, "data_management_convert_button")
    assert not convert.disabled
    app = convert.click().run()
    _assert_clean(app)
    dataset_ids = tuple(app.session_state[DATASETS_KEY])
    assert len(dataset_ids) == 1

    for page in ("resources", "cached-files", "plotting", "data-management"):
        app = _button(app, f"sidebar-nav-{page}").click().run()
        _assert_clean(app)
        assert tuple(app.session_state[DATASETS_KEY]) == dataset_ids
        assert "mapping" in app.session_state[WORKFLOWS_KEY]

    app = _button(app, "sidebar-nav-plotting").click().run()
    _assert_clean(app)
    source = next(item for item in app.selectbox if item.label == "Data source")
    assert any("profile.csv" in option for option in source.options)
    assert any("Mapping - replication timing" in option for option in source.options)


def test_data_management_repairs_malformed_retained_widget_values() -> None:
    upload = UploadRecord(
        "track-digest",
        "profile.bedgraph",
        b"chr1\t0\t100\t1\n",
        "text/plain",
    )
    inspection = SourceInspection(
        digest=upload.digest,
        name=upload.name,
        source_format="bedgraph",
        size_bytes=upload.size_bytes,
        chromosomes=("chr1",),
        chromosome_bounds={"chr1": (0, 100)},
    )
    app = _app()
    app.session_state["dnascape_active_page"] = "data-management"
    app.session_state[UPLOADS_KEY] = {upload.digest: upload}
    app.session_state[INSPECTIONS_KEY] = {upload.digest: inspection}
    app.session_state["data_management_track_chromosome"] = np.array(["stale"])
    app.session_state["data_management_track_bin_size"] = "not-a-number"
    app.session_state["data_management_track_fill_value"] = {}
    app.session_state["data_management_track_smooth_window"] = float("nan")

    app = app.run()
    _assert_clean(app)
    values = {item.label: item.value for item in app.number_input}
    assert values["Bin size (bp)"] == 1_000
    assert values["Custom fill value"] == 0.0
    assert values["Centered smoothing window (bins)"] == 1
    chromosome = next(item for item in app.selectbox if item.label == "Chromosome")
    assert chromosome.value == "chr1"


def test_cached_retained_data_open_and_remove_actions() -> None:
    upload = UploadRecord("source", "source.npy", b"payload")
    dataset = DatasetRecord(
        "dataset",
        "source.npy",
        upload.digest,
        "npy",
        "space",
        np.arange(8, dtype=float),
    )
    app = _app()
    app.session_state["dnascape_active_page"] = "cached-files"
    app.session_state[UPLOADS_KEY] = {upload.digest: upload}
    app.session_state[DATASETS_KEY] = {dataset.dataset_id: dataset}
    app.session_state[WORKFLOWS_KEY] = {
        "simulation": {
            "result": StoredWorkflowResult(
                {"replication_timing": np.arange(8, dtype=float)},
                "simulation-signature",
            )
        }
    }

    app = app.run()
    _assert_clean(app)
    app = _button(app, "cached_files_plot_dataset").click().run()
    _assert_clean(app)
    assert app.session_state["dnascape_active_page"] == "plotting"
    assert app.session_state["plotting_data_view"] == "dataset:dataset"

    app = _button(app, "sidebar-nav-cached-files").click().run()
    app = _button(app, "cached_files_upload_remove_source").click().run()
    _assert_clean(app)
    assert app.session_state[UPLOADS_KEY] == {}
    assert "dataset" in app.session_state[DATASETS_KEY]

    app = _button(app, "cached_files_workflow_remove_simulation").click().run()
    _assert_clean(app)
    assert "result" not in app.session_state[WORKFLOWS_KEY]["simulation"]

    app = _button(app, "cached_files_dataset_remove_dataset").click().run()
    _assert_clean(app)
    assert app.session_state[DATASETS_KEY] == {}


def test_cached_files_repairs_legacy_entries_and_clear_remove_paths() -> None:
    app = _app()
    app.session_state["dnascape_active_page"] = "cached-files"
    app.session_state[GENERATED_ARTIFACTS_KEY] = [
        None,
        {"id": "broken", "variants": {"CSV": {"payload": "not bytes"}}},
        {"id": "same", "payload": b"first", "created_at": 1e300},
        {"id": "same", "payload": bytearray(b"second"), "created_at": 2.0},
    ]
    app.session_state["cached_files_generated_sort"] = np.array(["stale"])
    app.session_state["cached_files_generated_descending"] = {}
    app.session_state["cached_files_limit"] = "invalid"
    app.session_state["cached_files_keep_raw"] = {"invalid": True}
    app.session_state["cached_gz_same"] = "invalid"

    app = app.run()
    _assert_clean(app)
    retained = app.session_state[GENERATED_ARTIFACTS_KEY]
    assert [item["id"] for item in retained] == ["same", "same-2"]
    assert all(isinstance(item["payload"], bytes) for item in retained)
    assert app.warning

    app = _button(app, "cached_remove_same").click().run()
    _assert_clean(app)
    assert [item["id"] for item in app.session_state[GENERATED_ARTIFACTS_KEY]] == [
        "same-2"
    ]
    app = _button(app, "cached_files_clear_generated").click().run()
    _assert_clean(app)
    assert app.session_state[GENERATED_ARTIFACTS_KEY] == []


def test_cached_gzip_preparation_round_trips_bytes() -> None:
    cached_files = importlib.import_module("app_streamlit.pages.5_Cached_Files")

    raw = b"position,value\n0,1\n"
    variant = {"payload": memoryview(raw), "file_name": "result.csv", "mime": "text/csv"}
    payload, name, mime = cached_files.selected_download_payload(variant, compress=True)
    assert gzip.decompress(payload) == raw
    assert name == "result.csv.gz"
    assert mime == "application/gzip"
