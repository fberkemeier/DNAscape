"""End-to-end smoke tests for the Streamlit application shell.

These tests deliberately exercise the rendered widget tree.  Pure package
tests cannot detect stale Streamlit callback signatures or session-state
regressions that only appear after a background result becomes available.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parents[1] / "streamlit_app.py"
PAGE_SLUGS = (
    "overview",
    "data-management",
    "mapping",
    "simulation",
    "plotting",
    "cached-files",
    "resources",
)


def _app() -> AppTest:
    return AppTest.from_file(str(APP_PATH), default_timeout=60)


def _assert_clean_render(app: AppTest) -> None:
    assert not app.exception, [str(item) for item in app.exception]
    assert not app.error, [item.value for item in app.error]


def _assert_no_widget_state_conflicts(caplog) -> None:
    conflicts = [
        record.getMessage()
        for record in caplog.records
        if "created with a default value" in record.getMessage()
    ]
    assert not conflicts


def _button(app: AppTest, label: str):
    return next(button for button in app.button if button.label == label)


def _wait_for_job(app: AppTest, key: str, timeout: float = 20.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = app.session_state[key]
        if not job.running:
            return app, job
        time.sleep(0.05)
        app = app.run()
    raise AssertionError(f"Background job {key!r} did not finish within {timeout:g} s.")


def test_every_page_renders_cleanly_during_repeated_navigation(caplog) -> None:
    app = _app()
    for _ in range(2):
        for slug in PAGE_SLUGS:
            app.session_state["dnascape_active_page"] = slug
            app = app.run()
            _assert_clean_render(app)
    _assert_no_widget_state_conflicts(caplog)


def test_map_button_completes_and_renders_a_preview(caplog) -> None:
    app = _app()
    app.session_state["dnascape_active_page"] = "mapping"
    app.session_state["mapping_route"] = "predict_timing"
    app.session_state["mapping_firing_rate_input_mode"] = "Constant"
    app.session_state["mapping_firing_rate_constant"] = 0.02
    app.session_state["mapping_fork_speed_input_mode"] = "Constant"
    app.session_state["mapping_fork_speed_constant"] = 1.4
    app = app.run()
    _assert_clean_render(app)

    # Set these after the sources have initialized their inferred region.
    app.session_state["mapping_region_start"] = 0.0
    app.session_state["mapping_region_end"] = 20.0
    app = app.run()
    action = _button(app, "Map")
    assert not action.disabled

    app = action.click().run()
    app, job = _wait_for_job(app, "mapping_background_job_v2")
    assert job.status == "complete", (job.error_type, job.error_message)
    _assert_clean_render(app)
    preview = app.session_state["mapping_plot_preview_v2"]
    assert isinstance(preview["payload"], bytes)
    assert preview["payload"]
    _assert_no_widget_state_conflicts(caplog)


def test_simulate_button_completes_and_renders_a_preview(caplog) -> None:
    app = _app()
    app.session_state["dnascape_active_page"] = "simulation"
    app.session_state["simulation_firing_rate_input_mode"] = "Constant"
    app.session_state["simulation_firing_rate_constant"] = 0.02
    app.session_state["simulation_fork_speed_input_mode"] = "Constant"
    app.session_state["simulation_fork_speed_constant"] = 1.4
    app = app.run()
    _assert_clean_render(app)

    # Keep the integration check fast while still using the actual simulator.
    app.session_state["simulation_region_start"] = 0.0
    app.session_state["simulation_region_end"] = 20.0
    app.session_state["simulation_realizations"] = 2
    app.session_state["simulation_time_max"] = 30.0
    app.session_state["simulation_max_rep_time"] = 60.0
    app = app.run()
    action = _button(app, "Simulate")
    assert not action.disabled

    app = action.click().run()
    app, job = _wait_for_job(app, "simulation_background_job_v2")
    assert job.status == "complete", (job.error_type, job.error_message)
    _assert_clean_render(app)
    preview = app.session_state["simulation_plot_preview_v2"]
    assert isinstance(preview["payload"], bytes)
    assert preview["payload"]
    _assert_no_widget_state_conflicts(caplog)


@pytest.mark.parametrize(
    ("output_label", "output_key"),
    (
        ("Replicated fraction", "replicated_fraction"),
        ("Replicated fraction (space-time)", "replicated_fraction_xt"),
    ),
)
def test_simulation_requests_selected_optional_output(
    output_label: str, output_key: str
) -> None:
    app = _app()
    app.session_state["dnascape_active_page"] = "simulation"
    app.session_state["simulation_firing_rate_input_mode"] = "Constant"
    app.session_state["simulation_firing_rate_constant"] = 0.02
    app.session_state["simulation_fork_speed_input_mode"] = "Constant"
    app.session_state["simulation_fork_speed_constant"] = 1.4
    app.session_state["simulation_preview_output"] = output_label
    app = app.run()

    app.session_state["simulation_region_start"] = 0.0
    app.session_state["simulation_region_end"] = 20.0
    app.session_state["simulation_realizations"] = 2
    app.session_state["simulation_time_max"] = 20.0
    app.session_state["simulation_max_rep_time"] = 60.0
    app = app.run()
    action = _button(app, "Simulate")
    assert not action.disabled

    app = action.click().run()
    app, job = _wait_for_job(app, "simulation_background_job_v2")
    assert job.status == "complete", (job.error_type, job.error_message)
    _assert_clean_render(app)
    workflows = app.session_state["dnascape_workflow_state"]
    result = workflows["simulation"]["result"]
    assert output_key in result.payload["outputs"]
