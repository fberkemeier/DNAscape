"""Adaptive ETA follows measured phases and work moved between grids."""

from collections import Counter

import numpy as np
import pytest

from dnascape import _kinetic_progress as progress


@pytest.fixture
def clock(monkeypatch):
    now = [0.]
    monkeypatch.setattr(progress.time, "perf_counter", lambda: now[0])
    return now


def ready(tracker, clock, *, points=1024, phases=("space",) * 3, duration=1., nt=10, backend="travel"):
    stage = progress.FitStage(tracker, points, phases=phases)
    with stage.prediction():
        clock[0] += duration
    stage.ready(time_points=nt, backend=backend)
    return stage


def step(stage, clock, phase, duration):
    clock[0] += duration
    stage.update(phase)


def test_first_measured_iteration_replaces_expensive_initialization(clock):
    tracker = progress.FitProgress(4)
    stage = ready(tracker, clock, phases=("space",) * 4, duration=10.)
    step(stage, clock, "space", 2.)
    # Three further 2-second steps remain; the 10-second initialization is over.
    assert stage.eta(3) == pytest.approx(6.)


def test_spatial_and_temporal_steps_keep_separate_costs(clock):
    stage = ready(progress.FitProgress(6), clock, phases=["space"] * 4 + ["time"] * 2)
    step(stage, clock, "space", 2.)
    step(stage, clock, "time", 12.)
    assert stage.eta(4) == pytest.approx(3 * 2. + 12.)
    step(stage, clock, "space", 2.)
    assert stage.eta(3) == pytest.approx(2 * 2. + 12.)


def test_future_grid_initialization_is_not_charged_as_a_full_line_search(clock):
    tracker = progress.FitProgress(5)
    tracker.reserve(4096, 2, phases=["space"] * 2)
    stage = ready(tracker, clock, duration=5.)
    step(stage, clock, "space", 2.)
    # Current: 2*2 seconds. Fine grid: initialization 20, two steps at 8 each.
    assert stage.eta(2) == pytest.approx(40.)


def test_fft_refinement_accounts_for_transform_growth(clock):
    tracker = progress.FitProgress(4)
    tracker.reserve(4096, 2, phases=["space"] * 2)
    stage = ready(tracker, clock, phases=["space"] * 2, backend="stationary_source_convolution")
    step(stage, clock, "space", 2.)
    # Four times as many positions, with log2(N) growing from 10 to 12.
    ratio = 4 * 12 / 10
    assert stage.eta(1) == pytest.approx(2. + ratio * (1. + 2 * 2.))


def test_new_grid_reuses_measurements_then_adapts_to_its_actual_cost(clock):
    tracker = progress.FitProgress(4)
    coarse = ready(tracker, clock, phases=["space"] * 2)
    step(coarse, clock, "space", 2.)
    native = ready(tracker, clock, points=4096, phases=["space"] * 2, duration=100., nt=20)
    # Both the spatial and time grids grew. Reuse the coarse step timing,
    # rather than multiplying the new one-off initialization by the budget.
    assert native.eta(2) == pytest.approx(2 * 2. * 4 * 2)
    step(native, clock, "space", 6.)
    assert native.eta(1) == pytest.approx(6.)


def test_costs_from_different_backends_are_not_reused(clock):
    tracker = progress.FitProgress(4)
    coarse = ready(tracker, clock)
    step(coarse, clock, "space", 20.)
    native = ready(tracker, clock, backend="exact_cone_convolution", duration=1.)
    assert native.eta(3) == pytest.approx(6.)


def test_repeated_clock_extensions_do_not_inflate_every_later_step(clock):
    stage = progress.FitStage(progress.FitProgress(3), 1024, phases=["space"] * 3)
    with stage.prediction():
        clock[0] += 10.
    with stage.prediction():
        clock[0] += 2.
    stage.ready()
    assert stage.eta(3) == pytest.approx(12.)


def test_unused_coarse_iterations_move_to_the_finer_grid(clock):
    tracker = progress.FitProgress(7)
    tracker.reserve(4096, 2, phases=["space", "time"])
    stage = ready(tracker, clock, phases=["space"] * 3 + ["time"] * 2)
    step(stage, clock, "space", 2.)
    step(stage, clock, "time", 3.)
    stage.finish(transfer=True)
    assert tracker.pending[-1] == (4096, Counter(space=3, time=2))
    # Fine initialization + three spatial and two temporal iterations.
    assert stage.eta(0) == pytest.approx(4 * (1. + 3 * 2. + 2 * 3.))
    tracker.release()
    assert stage.eta(0) == 0.


def test_nested_refinement_moves_work_to_the_next_grid_only(clock):
    tracker = progress.FitProgress(8)
    tracker.reserve(16384, 1, phases=["space"])
    tracker.reserve(4096, 1, phases=["space"])
    stage = ready(tracker, clock, phases=["space"] * 6)
    step(stage, clock, "space", 2.)
    stage.finish(transfer=True)
    assert tracker.pending == [(16384, Counter(space=1)), (4096, Counter(space=6))]
    tracker.release()
    middle = ready(tracker, clock, points=4096, phases=["space"] * 6)
    step(middle, clock, "space", 4.)
    middle.finish(transfer=True)
    assert tracker.pending == [(16384, Counter(space=6))]


@pytest.mark.parametrize("target", ["initiation_rate", "fork_speed"])
def test_adaptive_eta_does_not_change_fitted_values_or_iteration_history(monkeypatch, target):
    from dnascape._kinetic_fit import fit_kinetic
    from dnascape.utils import mapping_progress, progress_context

    x = np.arange(4096)
    observed = 100. + 15 * np.sin(x / 200.)
    supplied = (1.4 if target == "initiation_rate" else .00006) * (1. + .1 * np.cos(x / 300.))
    options = dict(target=target, target_form="x", maxiter=3, relative_tolerance=0.)
    reference = fit_kinetic(observed, supplied, **options)
    assert reference._rmap_info["adaptive_fitting"]
    monkeypatch.setattr(progress.FitStage, "eta", lambda _self, _remaining: 0.)
    updates = []
    with progress_context(report=lambda fraction, text: updates.append(text), suppress_output=True), mapping_progress(options):
        actual = fit_kinetic(observed, supplied, **options)
    np.testing.assert_array_equal(actual, reference)
    assert actual._rmap_info["phase_history"] == reference._rmap_info["phase_history"]
    assert any("ETA" in text for text in updates)
