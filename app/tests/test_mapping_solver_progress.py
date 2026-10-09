"""Real constrained-route progress and cancellation across the app boundary."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

import app.mapping_service as mapping_service
import dnascape
import dnascape.inverse as inverse
from app.mapping_catalog import mapping_route
from app.mapping_service import MappingRequest, run_mapping
from dnascape.utils import check_cancelled, progress_context


def _spatial_speed_request() -> MappingRequest:
    return MappingRequest(
        route=mapping_route("timing_and_time_firing_to_spatial_fork_speed"),
        inputs={"replication_timing": np.linspace(10.0, 15.0, 8), "firing_rate": np.full((1, 3), 0.001)},
        input_forms={"replication_timing": "x", "firing_rate": "t"},
        settings={"maxiter": 1, "spatial_knots": 4, "time_grid": [0.0, 20.0, 40.0]},
        resolution_x=1.0,
    )


@pytest.mark.parametrize("entry_point", ["rmap", "app"])
def test_real_speed_route_reports_before_its_first_forward_evaluation(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    entry_point: str,
) -> None:
    request = _spatial_speed_request()
    progress: list[tuple[float, str]] = []
    calls = 0

    def forward(
        _firing: np.ndarray,
        _speed: np.ndarray,
        _time: np.ndarray,
        _resolution: float,
        _periodic: bool,
        _fields: Any,
        parameters: np.ndarray,
        *,
        progress: Callable[[float], None],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        nonlocal calls
        calls += 1
        assert messages[-1] == "Fitting fork speed... evaluation 1/1 (0%)"
        assert "Preparing kinetic fields..." in messages
        progress(0.5)
        return np.asarray(request.inputs["replication_timing"]).copy(), np.zeros((8, parameters.size)), np.ones(8)

    messages: list[str] = []

    def report(value: float, message: str) -> None:
        progress.append((value, message))
        messages.append(message)

    monkeypatch.setattr(inverse, "_mean_field_timing", forward)
    if entry_point == "rmap":
        # The numerical route must publish its stage even with its ordinary
        # quiet default; the browser's verbose override cannot hide a defect.
        assert dict(request.route.defaults)["verbose"] is False
        with progress_context(report=report, suppress_output=True):
            dnascape.rmap(
                source=request.inputs,
                target="fork_speed",
                method=request.route.method,
                assumptions=dict(request.settings),
            )
    else:
        run_mapping(request, report=report)
        assert messages.index("Mapping fork speed...") < messages.index("Preparing kinetic fields...")
        assert messages[-1] == "Mapping complete"

    assert calls == 1
    assert "Fitting fork speed... evaluation 1/1 (50%)" in messages
    assert any("MSE" in message for message in messages)
    assert [value for value, _message in progress] == sorted(value for value, _message in progress)
    assert capsys.readouterr().out == ""


def test_directionality_speed_reports_mse_before_the_optimizer_returns(monkeypatch, capsys):
    from dnascape import directionality_inverse
    from dnascape.mapping import map_timing_directionality

    n = 24
    timing = 7. + np.cumsum(np.linspace(.03, .09, n))
    speed = 1.1 + .25 * np.sin(np.linspace(0., np.pi, n)) ** 2
    directionality = map_timing_directionality(timing, speed, resolution=2., T0=6.)
    original = directionality_inverse.least_squares
    returned = False
    reported = []

    def record_return(*args, **kwargs):
        nonlocal returned
        result = original(*args, **kwargs)
        returned = True
        return result

    def report(value, message):
        if "MSE" in message:
            reported.append((returned, value, message))

    monkeypatch.setattr(directionality_inverse, "least_squares", record_return)
    result = run_mapping(MappingRequest(
        route=mapping_route("timing_and_directionality_to_spatial_fork_speed"),
        inputs={"replication_timing": timing, "fork_directionality": directionality},
        input_forms={"replication_timing": "x", "fork_directionality": "x"},
        settings={"T0": 6.}, resolution_x=2.,
    ), report=report)
    assert any(not already_returned for already_returned, _, _ in reported)
    assert reported[-1][0] is True
    assert result.fit_mse == pytest.approx(result.outputs["fit_history"][-1])
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("cancel_stage", ["preparing", "evaluation_start", "evaluation_progress"])
def test_cancellation_during_first_speed_evaluation_releases_app_lock(
    monkeypatch: pytest.MonkeyPatch,
    cancel_stage: str,
) -> None:
    class StoppedMapping(Exception):
        pass

    cancellation_requested = False
    forward_calls = 0
    forward_completions = 0
    messages: list[str] = []

    def report(_value: float, message: str) -> None:
        nonlocal cancellation_requested
        messages.append(message)
        if (
            (cancel_stage == "preparing" and message == "Preparing kinetic fields...")
            or (cancel_stage == "evaluation_start" and message.endswith("1/1 (0%)"))
            or (cancel_stage == "evaluation_progress" and message.endswith("1/1 (50%)"))
        ):
            cancellation_requested = True

    def cancel() -> None:
        if cancellation_requested:
            raise StoppedMapping("Cancelled during first fit evaluation")

    def forward(
        _firing: np.ndarray,
        _speed: np.ndarray,
        _time: np.ndarray,
        _resolution: float,
        _periodic: bool,
        _fields: Any,
        parameters: np.ndarray,
        *,
        progress: Callable[[float], None],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        nonlocal forward_calls, forward_completions
        forward_calls += 1
        check_cancelled()
        progress(0.5)
        check_cancelled()
        forward_completions += 1
        return np.linspace(10.0, 15.0, 8), np.zeros((8, parameters.size)), np.ones(8)

    monkeypatch.setattr(inverse, "_mean_field_timing", forward)
    with pytest.raises(StoppedMapping, match="Cancelled during first fit evaluation"):
        run_mapping(_spatial_speed_request(), report=report, cancel=cancel)

    assert cancellation_requested
    assert forward_calls == (0 if cancel_stage == "preparing" else 1)
    assert forward_completions == 0
    assert "Mapping complete" not in messages
    assert mapping_service._RUN_LOCK.acquire(blocking=False)
    mapping_service._RUN_LOCK.release()
    # Both the app lock and thread-local cancellation context must be restored
    # so another request on the same worker can finish normally.
    result = run_mapping(_spatial_speed_request())
    assert result.primary_output == "fork_speed"
    assert forward_completions == 1
