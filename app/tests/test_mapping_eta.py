"""ETA estimates use measured work and remain honest when work stalls."""

import pytest

from app.mapping_eta import MappingEta


@pytest.fixture
def clock():
    return [0.0]


def test_preparation_never_fabricates_a_numeric_eta(clock):
    eta = MappingEta(clock=lambda: clock[0])
    eta.start_case(1)
    eta.report(.02, "Preparing mapping...")
    clock[0] = 100
    assert eta.text() == "ETA estimating… · Elapsed 00:01:40"
    eta.report(.04, "Mapping fork speed...")
    clock[0] = 110
    assert eta.text() == "ETA estimating… · Elapsed 00:01:50"


def test_explicit_eta_counts_down_and_overrun_returns_to_estimating(clock):
    eta = MappingEta(clock=lambda: clock[0])
    eta.start_case(1)
    clock[0] = 2
    eta.report(.49, "Mapping... 1/2 - ETA 00:00:20 - MSE 1.00e-2")
    clock[0] = 7
    assert eta.text() == "ETA ≈ 00:00:15 · Elapsed 00:00:07"
    clock[0] = 25
    assert eta.text() == "ETA estimating… · Elapsed 00:00:25"
    eta.report(.94, "Mapping... 2/2 - ETA 00:00:00")
    assert "estimating" in eta.text()


def test_generic_solver_progress_estimates_stage_and_preview_resets_it(clock):
    eta = MappingEta(clock=lambda: clock[0])
    eta.start_case(1)
    eta.report(.04, "Mapping...")
    clock[0] = 10
    eta.report(.49, "Fitting target profile...")
    assert eta.text() == "ETA ≈ 00:00:10 · Elapsed 00:00:10"
    eta.report(.98, "Preparing mapped preview...")
    assert "estimating" in eta.text()


def test_completed_quiet_cases_inform_the_batch_without_leaking_stage_eta(clock):
    eta = MappingEta(clock=lambda: clock[0])
    eta.start_case(3)
    clock[0] = 10
    eta.complete_case()
    eta.start_case(3)
    clock[0] = 12
    assert eta.text() == "ETA ≈ 00:00:18 (batch) · Elapsed 00:00:12"
    eta.report(.49, "Fitting... ETA 00:00:04")
    assert "00:00:04" in eta.text()
    clock[0] = 15
    eta.complete_case()
    eta.start_case(3)
    assert "00:00:08 (batch)" in eta.text()
    clock[0] = 100
    assert "estimating" in eta.text()


@pytest.mark.parametrize("outcome, expected", [
    ("complete", "00:00:00"), ("stopped", "stopped"), ("unavailable", "unavailable"),
])
def test_terminal_state_freezes_elapsed_and_clears_estimate(clock, outcome, expected):
    eta = MappingEta(clock=lambda: clock[0])
    clock[0] = 5
    eta.finish(outcome)
    clock[0] = 100
    assert eta.text() == f"ETA {expected} · Elapsed 00:00:05"


def test_nonfinite_progress_is_ignored(clock):
    eta = MappingEta(clock=lambda: clock[0])
    eta.start_case(1)
    eta.report(float("nan"), "Fitting...")
    eta.report(float("inf"), "Fitting...")
    assert "estimating" in eta.text()
