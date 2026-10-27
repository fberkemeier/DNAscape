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


def test_first_forward_prediction_reports_eta_before_an_iteration_finishes(monkeypatch):
    from dnascape import utils
    now = [0.]
    updates = []
    monkeypatch.setattr(utils.time, "perf_counter", lambda: now[0])
    with utils.progress_context(report=lambda fraction, text: updates.append((fraction, text))):
        with utils.measured_work(total=11):
            now[0] = 1.
            utils.report_measured_work(.1)
            assert updates and "initial prediction 10%" in updates[-1][1]
            assert "ETA 00:01:49" in updates[-1][1]
            assert .04 < updates[-1][0] < .05
            eta = MappingEta(clock=lambda: now[0])
            eta.start_case(1)
            eta.report(*updates[-1])
            assert "ETA ≈ 00:01:49" in eta.text()
        before = len(updates)
        utils.report_measured_work(.5)
        assert len(updates) == before


def test_earlier_eta_leaves_the_fit_numerically_identical(monkeypatch):
    import numpy as np
    from dnascape import fitting, utils
    observed = 20. + np.sin(np.arange(64.) / 7.)
    expected = fitting.map_timing_firing(observed, maxiter=1, neighbourhood_range=128, refine_edge_kb=0, verbose=False)
    now = [0.]
    def tick():
        now[0] += .3
        return now[0]
    monkeypatch.setattr(utils.time, "perf_counter", tick)
    messages = []
    with utils.progress_context(report=lambda _fraction, text: messages.append(text), suppress_output=True):
        actual = fitting.map_timing_firing(observed, maxiter=1, neighbourhood_range=128, refine_edge_kb=0, verbose=False)
    assert any("initial prediction" in message and "ETA" in message for message in messages)
    np.testing.assert_array_equal(actual, expected)


def test_stochastic_prediction_progress_preserves_the_seeded_ensemble(monkeypatch):
    import numpy as np
    from dnascape import rsim, utils
    options = dict(initiation_rate=.1, fork_speed=1.4, n_bins=8, sim_number=3, seed=123, verbose=False)
    expected = rsim(**options)
    now = [0.]
    def tick():
        now[0] += .3
        return now[0]
    monkeypatch.setattr(utils.time, "perf_counter", tick)
    messages = []
    with utils.progress_context(report=lambda _fraction, text: messages.append(text), suppress_output=True):
        with utils.measured_work(total=5):
            actual = rsim(**options)
    assert any("initial prediction" in text for text in messages)
    for key in ("replication_timing", "origin_efficiency", "fork_directionality"):
        np.testing.assert_array_equal(actual[key], expected[key])


@pytest.mark.parametrize("target", ["time", "spacetime"])
@pytest.mark.parametrize("speed_form", ["constant", "space", "time", "spacetime"])
def test_first_weibull_evaluation_reports_eta_for_every_dependence(monkeypatch, target, speed_form):
    import numpy as np
    from dnascape import fitting, utils
    n = 18
    t = np.arange(0., 61., 5.)
    speeds = {"constant": 1.4, "space": 1.4 + .1 * np.sin(np.arange(n)),
              "time": (1.4 + t / 1000.)[None, :],
              "spacetime": np.outer(1.4 + .1 * np.sin(np.arange(n)), 1. + t / 1000.)}
    options = dict(fork_speed=speeds[speed_form], target_form=target, time_grid=t,
                   perQ=True, maxiter=2, refine_edge_kb=0, refine_verbose=False)
    observed = 20. + np.sin(np.arange(n) / 4.)
    expected = fitting._fit_weibull_survival_firing(observed, **options)
    now = [0.]
    def tick():
        now[0] += .3
        return now[0]
    monkeypatch.setattr(utils.time, "perf_counter", tick)
    updates = []
    with utils.progress_context(report=lambda fraction, text: updates.append((fraction, text)), suppress_output=True):
        actual = fitting._fit_weibull_survival_firing(observed, **options)
    assert any("initial prediction" in text and "ETA" in text for _, text in updates)
    np.testing.assert_array_equal(actual, expected)
    assert updates[-1][1].split(" - ")[1] == "ETA 00:00:00"


def test_weibull_eta_counts_accepted_iterations_instead_of_line_search_trials(monkeypatch):
    from types import SimpleNamespace
    import numpy as np
    from dnascape import fitting, utils
    updates = []

    def optimizer(function, initial, *, callback, **_kwargs):
        for _ in range(8):
            function(initial)
        assert updates and all(value < .94 for value, _text in updates)
        assert all("4/4" not in text for _value, text in updates)
        callback(initial)
        assert "2/4" in updates[-1][1]
        return SimpleNamespace(success=True, message="Test optimizer stopped after one accepted step.")

    monkeypatch.setattr(fitting, "minimize", optimizer)
    with utils.progress_context(report=lambda value, text: updates.append((value, text)), suppress_output=True):
        fitting._fit_weibull_survival_firing(np.full(18, 20.), 1.4, target_form="spacetime",
            maxiter=3, perQ=True, time_grid=np.arange(0., 61., 5.))
    assert "4/4" in updates[-1][1]
