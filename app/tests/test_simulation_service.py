"""Fast checks for the framework-independent NiceGUI simulation bridge."""

from __future__ import annotations

import math
from collections.abc import Callable
from concurrent.futures import CancelledError
from dataclasses import replace
from typing import Any, cast

import numpy as np
import pytest

import app.simulation_service as simulation_service
import dnascape
from app.simulation_service import (
    CORE_OUTPUT_KEYS,
    DISTRIBUTION_OUTPUT_KEYS,
    MAX_DETAILED_XT_CELLS,
    MAX_INTER_ORIGIN_SAMPLES,
    MAX_REPLICATES,
    MAX_RETAINED_REPLICATION_PROFILE_BYTES,
    MAX_RETAINED_REPLICATION_PROFILES,
    MAX_SPATIAL_BINS,
    MAX_TIME_POINTS,
    POOLED_DISTRIBUTION_OUTPUT_KEYS,
    REPLICATE_OUTPUT_KEYS,
    TIME_OUTPUT_KEYS,
    XT_OUTPUT_KEYS,
    ConstantSimulationRequest,
    run_constant_simulation,
    validate_constant_simulation,
)


def _request(**changes: Any) -> ConstantSimulationRequest:
    baseline = ConstantSimulationRequest(
        firing_rate=0.5,
        fork_speed=2.0,
        resolution_x=1.0,
        resolution_t=1.0,
        region_start_kb=0.0,
        region_end_kb=4.0,
        replicates=2,
        geometry="linear",
        time_window=10.0,
        max_replication_time=100.0,
        seed=1,
    )
    return replace(baseline, **changes)


def _fake_detailed_output(
    *,
    profile_shape: tuple[int, ...] = (2, 4),
    std_shape: tuple[int, ...] = (4,),
) -> dict[str, Any]:
    n_bins = 4
    n_times = 11
    return {
        "replication_timing": np.ones(n_bins),
        "replication_timing_std": np.zeros(std_shape),
        "rep_times_per_sim": np.ones(profile_shape),
        "origin_efficiency": np.ones(n_bins),
        "fork_directionality": np.zeros(n_bins),
        "replicated_fraction": np.zeros(n_times),
        "total_forks": np.zeros(n_times),
        "active_forks": np.zeros(n_times),
        "stalled_forks": np.zeros(n_times),
        "s_phase_duration": np.asarray([4.0, 5.0]),
        "completion_time": np.asarray([5.0, 6.0]),
        "num_origins": np.asarray([3.0, 4.0]),
        "inter_origin_distances": np.asarray([1.0, 2.0, 1.0]),
        **{
            name: np.zeros((n_bins, n_times), dtype=np.float32)
            for name in XT_OUTPUT_KEYS
        },
    }


def test_tiny_constant_ensemble_runs_real_dnascape_and_reports_progress() -> None:
    progress: list[tuple[float, str]] = []

    result = run_constant_simulation(
        _request(),
        lambda value, message: progress.append((value, message)),
    )

    np.testing.assert_array_equal(result.position_kb, [0.0, 1.0, 2.0, 3.0])
    assert result.replication_timing.shape == (4,)
    assert np.all(np.isfinite(result.replication_timing))
    assert result.replicates == 2
    assert result.geometry == "linear"
    assert result.elapsed_seconds >= 0
    assert result.outputs["replication_timing"].shape == (4,)
    assert result.time_outputs is False
    assert result.detailed_outputs is False
    assert result.replication_time_profiles is not None
    assert result.replication_time_profiles.shape == (2, 4)
    assert result.replication_timing_std is not None
    assert result.replication_timing_std.shape == (4,)
    assert result.time_grid.size == 0
    assert set(result.outputs) == set(CORE_OUTPUT_KEYS) | set(DISTRIBUTION_OUTPUT_KEYS)
    assert result.outputs["s_phase_duration"].shape == (2,)
    assert result.outputs["completion_time"].shape == (2,)
    assert np.all(
        result.outputs["completion_time"] >= result.outputs["s_phase_duration"]
    )
    assert np.any(
        result.outputs["completion_time"] > result.outputs["s_phase_duration"]
    )
    assert set(TIME_OUTPUT_KEYS).isdisjoint(result.outputs)
    assert set(XT_OUTPUT_KEYS).isdisjoint(result.outputs)
    assert progress[0] == (0.02, "Preparing simulation...")
    assert progress[-1] == (1.0, "Simulation complete")
    assert [value for value, _message in progress] == sorted(value for value, _message in progress)
    for array in (
        result.position_kb,
        result.replication_timing,
        result.time_grid,
        result.replication_timing_std,
        *result.outputs.values(),
    ):
        assert array.flags.writeable is False


def test_cancellation_is_checked_inside_the_simulation_ensemble() -> None:
    checks = 0

    def cancel() -> None:
        nonlocal checks
        checks += 1
        if checks >= 3:
            raise CancelledError

    with pytest.raises(CancelledError):
        run_constant_simulation(_request(replicates=20), cancel=cancel)

    assert checks >= 3


def test_time_and_space_time_modes_collect_their_independent_outputs() -> None:
    result = run_constant_simulation(
        _request(
            track_detailed_outputs=True,
            track_time_outputs=True,
            firing_factors=10.0,
        )
    )

    assert result.detailed_outputs is True
    assert result.time_outputs is True
    assert result.replication_time_profiles is not None
    assert result.replication_time_profiles.shape == (2, 4)
    assert result.replication_timing_std is not None
    assert result.replication_timing_std.shape == (4,)
    np.testing.assert_allclose(
        result.replication_timing_std,
        np.std(result.replication_time_profiles, axis=0, ddof=0),
    )
    np.testing.assert_array_equal(result.time_grid, np.arange(11, dtype=float))
    assert set(CORE_OUTPUT_KEYS) <= result.outputs.keys()
    assert set(DISTRIBUTION_OUTPUT_KEYS) <= result.outputs.keys()
    assert set(TIME_OUTPUT_KEYS) <= result.outputs.keys()
    assert set(XT_OUTPUT_KEYS) <= result.outputs.keys()
    for name in TIME_OUTPUT_KEYS:
        assert result.outputs[name].shape == (11,), name
    for name in REPLICATE_OUTPUT_KEYS:
        assert result.outputs[name].shape == (2,), name
    for name in POOLED_DISTRIBUTION_OUTPUT_KEYS:
        values = result.outputs[name]
        assert values.ndim == 1, name
        assert np.all(np.isfinite(values)), name
        assert np.all(values > 0), name
    for name in XT_OUTPUT_KEYS:
        assert result.outputs[name].shape == (4, 11), name
        assert result.outputs[name].dtype == np.float32, name
    for array in result.outputs.values():
        assert array.flags.writeable is False
    assert result.replication_time_profiles.flags.writeable is False
    assert result.replication_timing_std.flags.writeable is False


@pytest.mark.parametrize("time_outputs", (False, True))
@pytest.mark.parametrize("detailed", (False, True))
def test_rsim_statistics_flags_are_independent_and_summaries_are_always_retained(
    monkeypatch: pytest.MonkeyPatch,
    time_outputs: bool,
    detailed: bool,
) -> None:
    assumptions_seen: dict[str, Any] = {}
    real_rsim = cast(Callable[..., dict[str, Any]], dnascape.rsim)

    def inspect_rsim(*, source: dict[str, Any], assumptions: dict[str, Any]) -> dict[str, Any]:
        assumptions_seen.update(assumptions)
        return real_rsim(source=source, assumptions=assumptions)

    monkeypatch.setattr(dnascape, "rsim", inspect_rsim)
    run_constant_simulation(
        _request(
            track_time_outputs=time_outputs,
            track_detailed_outputs=detailed,
        )
    )

    assert assumptions_seen["time_statsQ"] is time_outputs
    assert assumptions_seen["time_stats_xtQ"] is detailed
    assert assumptions_seen["retain_rep_times_per_sim"] is True
    assert assumptions_seen["retain_rep_times_limit"] == 2
    assert assumptions_seen["retain_inter_origin_distances"] is True
    assert assumptions_seen["retain_inter_origin_distances_limit"] == MAX_INTER_ORIGIN_SAMPLES
    assert ("time_grid" in assumptions_seen) is (time_outputs or detailed)


def test_large_ensemble_uses_bounded_progress_cadence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assumptions_seen: dict[str, Any] = {}

    def inspect_rsim(*, source: dict[str, Any], assumptions: dict[str, Any]) -> dict[str, Any]:
        del source
        assumptions_seen.update(assumptions)
        n_bins = 4
        replicates = MAX_REPLICATES
        return {
            "replication_timing": np.ones(n_bins),
            "replication_timing_std": np.zeros(n_bins),
            "rep_times_per_sim": np.ones((MAX_RETAINED_REPLICATION_PROFILES, n_bins)),
            "origin_efficiency": np.ones(n_bins),
            "fork_directionality": np.zeros(n_bins),
            "s_phase_duration": np.ones(replicates),
            "completion_time": np.ones(replicates),
            "num_origins": np.ones(replicates),
            "inter_origin_distances": np.asarray([1.0]),
        }

    monkeypatch.setattr(dnascape, "rsim", inspect_rsim)
    run_constant_simulation(_request(replicates=MAX_REPLICATES))

    assert assumptions_seen["print_every"] == 100


def test_inter_origin_distances_are_exact_pooled_rsim_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The app must not mistake pooled intervals for one value per replicate."""

    def inspect_rsim(*, source: dict[str, Any], assumptions: dict[str, Any]) -> dict[str, Any]:
        del source
        assert assumptions["retain_inter_origin_distances"] is True
        n_bins = 4
        n_times = 11
        return {
            "replication_timing": np.ones(n_bins),
            "replication_timing_std": np.zeros(n_bins),
            "rep_times_per_sim": np.ones((2, n_bins)),
            "origin_efficiency": np.ones(n_bins),
            "fork_directionality": np.zeros(n_bins),
            "replicated_fraction": np.zeros(n_times),
            "total_forks": np.zeros(n_times),
            "active_forks": np.zeros(n_times),
            "stalled_forks": np.zeros(n_times),
            "s_phase_duration": np.asarray([4.0, 5.0]),
            "completion_time": np.asarray([5.0, 6.0]),
            "num_origins": np.asarray([3.0, 4.0]),
            # Seven intervals from two realizations: deliberately not replicates=2.
            "inter_origin_distances": np.asarray([1.0, 2.0, 1.0, 3.0, 1.0, 2.0, 4.0]),
            **{
                name: np.zeros((n_bins, n_times), dtype=np.float32)
                for name in XT_OUTPUT_KEYS
            },
        }

    monkeypatch.setattr(dnascape, "rsim", inspect_rsim)
    result = run_constant_simulation(_request())

    np.testing.assert_array_equal(
        result.outputs["inter_origin_distances"],
        [1.0, 2.0, 1.0, 3.0, 1.0, 2.0, 4.0],
    )
    assert result.outputs["inter_origin_distances"].flags.writeable is False


@pytest.mark.parametrize(
    ("profile_shape", "std_shape", "message"),
    (
        ((1, 4), (4,), "profiles with shape"),
        ((2, 4), (3,), "replication_timing_std.*shape"),
    ),
)
def test_standard_profile_statistics_reject_shape_mismatches(
    monkeypatch: pytest.MonkeyPatch,
    profile_shape: tuple[int, ...],
    std_shape: tuple[int, ...],
    message: str,
) -> None:
    def inspect_rsim(*, source: dict[str, Any], assumptions: dict[str, Any]) -> dict[str, Any]:
        del source, assumptions
        return _fake_detailed_output(
            profile_shape=profile_shape,
            std_shape=std_shape,
        )

    monkeypatch.setattr(dnascape, "rsim", inspect_rsim)
    with pytest.raises(RuntimeError, match=message):
        run_constant_simulation(_request())


def test_inter_origin_distance_retention_metadata_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def inspect_rsim(*, source: dict[str, Any], assumptions: dict[str, Any]) -> dict[str, Any]:
        del source
        assert assumptions["retain_inter_origin_distances_limit"] == MAX_INTER_ORIGIN_SAMPLES
        raw = _fake_detailed_output()
        raw["inter_origin_distances"] = np.asarray([1.0, 2.0])
        raw["inter_origin_distance_count"] = 7
        raw["inter_origin_distances_truncated"] = True
        return raw

    monkeypatch.setattr(dnascape, "rsim", inspect_rsim)
    result = run_constant_simulation(_request())

    assert result.inter_origin_distance_count == 7
    assert result.inter_origin_distances_truncated is True
    np.testing.assert_array_equal(result.outputs["inter_origin_distances"], [1.0, 2.0])


def test_uploaded_kinetic_fields_are_passed_to_rsim_with_native_shapes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_seen: dict[str, Any] = {}
    firing_rate = np.asarray([0.1, 0.2, 0.3, 0.4])
    fork_speed = np.asarray([[1.0, 1.5, 2.0]])

    def inspect_rsim(*, source: dict[str, Any], assumptions: dict[str, Any]) -> dict[str, Any]:
        del assumptions
        source_seen.update(source)
        return {
            "replication_timing": np.asarray([4.0, 3.0, 2.0, 1.0]),
            "replication_timing_std": np.zeros(4),
            "rep_times_per_sim": np.ones((2, 4)),
            "origin_efficiency": np.asarray([0.1, 0.2, 0.3, 0.4]),
            "fork_directionality": np.asarray([-0.5, -0.2, 0.2, 0.5]),
            "s_phase_duration": np.asarray([3.0, 4.0]),
            "completion_time": np.asarray([4.0, 5.0]),
            "num_origins": np.asarray([2.0, 3.0]),
            "inter_origin_distances": np.asarray([1.0, 2.0]),
        }

    monkeypatch.setattr(dnascape, "rsim", inspect_rsim)
    result = run_constant_simulation(_request(firing_rate=firing_rate, fork_speed=fork_speed))

    np.testing.assert_array_equal(source_seen["firing_rate"], firing_rate)
    np.testing.assert_array_equal(source_seen["fork_speed"], fork_speed)
    assert np.asarray(source_seen["firing_rate"]).shape == (4,)
    assert np.asarray(source_seen["fork_speed"]).shape == (1, 3)
    assert np.asarray(source_seen["firing_rate"]).flags.writeable is False
    assert np.asarray(source_seen["fork_speed"]).flags.writeable is False
    assert firing_rate.flags.writeable is True
    assert fork_speed.flags.writeable is True
    np.testing.assert_array_equal(result.replication_timing, [4.0, 3.0, 2.0, 1.0])


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"firing_rate": np.asarray([0.1, 0.2, 0.3])}, "must contain 4 values"),
        ({"fork_speed": np.ones((3, 2))}, "must contain 4 rows"),
        ({"firing_rate": np.asarray([0.1, -0.2, 0.3, 0.4])}, "non-negative"),
        ({"fork_speed": np.asarray([1.0, 0.0, 1.0, 1.0])}, "greater than zero"),
        ({"firing_rate": np.ones((2, 2, 2))}, "must be scalar"),
    ),
)
def test_uploaded_kinetic_field_validation_matches_rsim(
    changes: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_constant_simulation(_request(**changes))


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"firing_rate": math.nan}, "Firing rate must be finite"),
        ({"fork_speed": 0.0}, "Fork speed must be finite and greater than zero"),
        ({"resolution_x": math.inf}, "Spatial resolution must be finite"),
        ({"region_end_kb": 0.0}, "Region end must be greater than region start"),
        ({"replicates": 1.5}, "Replicates must be a positive integer"),
        ({"geometry": "circular"}, "Geometry must be 'linear' or 'periodic'"),
        ({"missing_values": "ignore"}, "Missing values must be"),
        ({"time_window": 0.0}, "Time window must be finite and greater than zero"),
        ({"track_time_outputs": "yes"}, "Time-output tracking must be"),
        ({"track_detailed_outputs": "yes"}, "Space-time-output tracking must be"),
    ),
)
def test_scalar_validation_rejects_nonfinite_and_invalid_settings(
    changes: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_constant_simulation(_request(**changes))


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        (
            {"region_end_kb": float(MAX_SPATIAL_BINS + 1)},
            "local app limit",
        ),
        (
            {
                "time_window": float(MAX_TIME_POINTS),
                "resolution_t": 1.0,
                "track_time_outputs": True,
            },
            "time grid contains",
        ),
    ),
)
def test_grid_limits_fail_before_numpy_allocation(
    monkeypatch: pytest.MonkeyPatch,
    changes: dict[str, Any],
    message: str,
) -> None:
    def unexpected_allocation(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("validation should reject this request before allocating a grid")

    monkeypatch.setattr(simulation_service.np, "arange", unexpected_allocation)
    with pytest.raises(ValueError, match=message):
        validate_constant_simulation(_request(**changes))


def test_summary_mode_does_not_limit_or_allocate_an_unused_time_grid() -> None:
    validated = validate_constant_simulation(
        _request(
            time_window=float(MAX_TIME_POINTS),
            resolution_t=1.0,
        )
    )
    assert validated.time_grid.size == 0


def test_fractional_resolution_does_not_create_a_floating_point_extra_bin() -> None:
    validated = validate_constant_simulation(
        _request(
            firing_rate=np.asarray([0.1, 0.2, 0.3]),
            fork_speed=np.asarray([1.1, 1.2, 1.3]),
            resolution_x=0.1,
            region_end_kb=3 * 0.1,
        )
    )

    assert validated.n_bins == 3


def test_summary_mode_does_not_compute_an_extreme_unused_time_ratio() -> None:
    validated = validate_constant_simulation(
        _request(
            time_window=float(np.finfo(float).max),
            resolution_t=float(np.finfo(float).tiny),
        )
    )
    assert validated.time_points == 0
    assert validated.time_grid.size == 0

    with pytest.raises(ValueError, match="time grid is not finite"):
        validate_constant_simulation(
            _request(
                time_window=float(np.finfo(float).max),
                resolution_t=float(np.finfo(float).tiny),
                track_time_outputs=True,
            )
        )


def test_replicate_limit_allows_fifty_thousand_and_rejects_more() -> None:
    validated = validate_constant_simulation(_request(replicates=MAX_REPLICATES))
    assert validated.request.replicates == 50_000

    with pytest.raises(ValueError, match="50,000"):
        validate_constant_simulation(_request(replicates=MAX_REPLICATES + 1))


def test_retained_profile_count_is_bounded_by_memory() -> None:
    n_bins = 1_000_000
    validated = validate_constant_simulation(
        _request(
            region_end_kb=float(n_bins),
            replicates=MAX_RETAINED_REPLICATION_PROFILES,
        )
    )
    expected = MAX_RETAINED_REPLICATION_PROFILE_BYTES // (
        n_bins * np.dtype(float).itemsize
    )
    assert validated.retained_profile_count == expected


def test_detailed_preallocation_limit_fails_before_time_grid_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    time_points = 1_001
    n_bins = MAX_DETAILED_XT_CELLS // time_points + 1

    def unexpected_allocation(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("oversized detailed output reached time-grid allocation")

    monkeypatch.setattr(simulation_service.np, "arange", unexpected_allocation)
    with pytest.raises(ValueError, match="Detailed space-time outputs would preallocate"):
        validate_constant_simulation(
            _request(
                region_end_kb=float(n_bins),
                time_window=float(time_points - 1),
                track_detailed_outputs=True,
            )
        )
