"""Regression tests for responsive state, jobs, and workflow controls."""

from __future__ import annotations

import importlib
import time
from io import BytesIO
from threading import Thread
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from app_streamlit.data_service import UploadRecord, inspect_source
from app_streamlit.exports import ExportPayload
from app_streamlit.jobs import (
    MAX_PENDING_JOB_MESSAGES,
    BackgroundJob,
    poll_background_job,
    start_background_job,
)
from app_streamlit.workflow_models import (
    MAX_TIME_GRID_POINTS,
    SimulationConfig,
    build_time_grid,
)

mapping_page = importlib.import_module("app_streamlit.pages.2_Mapping")
simulation_page = importlib.import_module("app_streamlit.pages.3_Simulation")


def _poll_until_terminal(job: BackgroundJob, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while job.running and time.monotonic() < deadline:
        poll_background_job(job)
        time.sleep(0.001)
    poll_background_job(job)
    assert not job.running


def test_chatty_background_job_has_bounded_queue_and_keeps_terminal_result() -> None:
    def worker(*, job_context):
        for index in range(20_000):
            job_context.report(index / 20_000, f"update {index}")
        return {"ok": True}

    job = start_background_job("chatty", "signature", worker)
    assert job._thread is not None
    job._thread.join(timeout=3.0)
    assert job._messages.qsize() <= MAX_PENDING_JOB_MESSAGES
    _poll_until_terminal(job)
    assert job.status == "complete"
    assert job.result == {"ok": True}


def test_dead_worker_cannot_leave_job_running_forever() -> None:
    job = BackgroundJob("dead", "signature")
    job._thread = Thread(target=lambda: None)
    job._thread.start()
    job._thread.join(timeout=1.0)

    poll_background_job(job)

    assert job.status == "error"
    assert job.error_type == "WorkerExited"


def test_pathological_time_grid_is_rejected_before_allocation() -> None:
    with patch("app_streamlit.workflow_models.np.arange") as arange:
        with pytest.raises(ValueError, match="app limit"):
            build_time_grid(float(MAX_TIME_GRID_POINTS), 0.5)
    arange.assert_not_called()


def test_simulation_estimate_counts_grid_without_materializing_it() -> None:
    config = SimulationConfig(
        resolution_x=1.0,
        resolution_t=1.0,
        region_start_kb=0.0,
        region_end_kb=100.0,
        time_max=100.0,
        time_stats=True,
    )
    with patch("app_streamlit.workflow_models.np.arange") as arange:
        estimate = config.estimate()
    arange.assert_not_called()
    assert estimate.time_points == 101


def test_npy_inspection_reads_header_without_loading_array() -> None:
    buffer = BytesIO()
    np.save(buffer, np.zeros((12, 7), dtype=np.float32), allow_pickle=False)
    upload = UploadRecord("digest", "profile.npy", buffer.getvalue())

    with patch("app_streamlit.data_service.np.load", side_effect=AssertionError("eager load")):
        inspection = inspect_source(upload)

    assert inspection.shape == (12, 7)
    assert inspection.dtype == "float32"


def test_simulation_requested_output_enables_only_needed_optional_arrays() -> None:
    config = SimulationConfig(
        resolution_x=1.0,
        resolution_t=5.0,
        region_start_kb=0.0,
        region_end_kb=20.0,
    )
    time_config = simulation_page._config_for_requested_output(
        config, {"output": "replicated_fraction"}
    )
    xt_config = simulation_page._config_for_requested_output(
        config, {"output": "replicated_fraction_xt"}
    )

    assert time_config.time_stats and not time_config.spacetime_stats
    assert xt_config.spacetime_stats and not xt_config.time_stats


def test_simulation_preview_accepts_settings_mapping_and_encodes_png_once() -> None:
    state: dict[str, object] = {}
    stored = SimpleNamespace(payload={"outputs": {}}, input_signature="result")
    settings = {"output": "replication_timing", "signature": "plot-signature"}
    fake_streamlit = SimpleNamespace(session_state=state)

    with (
        patch.object(simulation_page, "st", fake_streamlit),
        patch.object(simulation_page, "_build_result_figure", return_value=object()),
        patch.object(
            simulation_page,
            "serialize_figure",
            return_value=ExportPayload(b"png", "png", "image/png"),
        ) as serialize,
        patch.object(simulation_page, "release_figure"),
    ):
        simulation_page._store_plot_preview(stored, settings)

    serialize.assert_called_once()
    assert serialize.call_args.args[1] == "png"
    preview = state[simulation_page.PLOT_KEY]
    assert isinstance(preview, dict)
    assert preview["signature"] == "plot-signature"


def test_mapping_preview_encodes_only_bounded_png() -> None:
    state: dict[str, object] = {}
    stored = SimpleNamespace(payload={"outputs": {}}, input_signature="result")
    settings = ("target", "Title", "#123456", "viridis", "")
    fake_streamlit = SimpleNamespace(session_state=state)

    with (
        patch.object(mapping_page, "st", fake_streamlit),
        patch.object(mapping_page, "_build_result_figure", return_value=object()),
        patch.object(
            mapping_page,
            "serialize_figure",
            return_value=ExportPayload(b"png", "png", "image/png"),
        ) as serialize,
        patch.object(mapping_page, "release_figure"),
    ):
        mapping_page._store_plot_preview(stored, settings)

    serialize.assert_called_once()
    assert serialize.call_args.args[1] == "png"
    preview = state[mapping_page.PLOT_KEY]
    assert isinstance(preview, dict)
    assert "exports" not in preview
