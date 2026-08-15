"""Fast checks for the framework-independent NiceGUI simulation bridge."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import replace
from typing import Any, cast

import numpy as np
import pytest

import app.simulation_service as simulation_service
import dnascape
from app.simulation_service import (
    CORE_OUTPUT_KEYS,
    DISTRIBUTION_OUTPUT_KEYS,
    MAX_DETAILED_WORK_CELLS,
    MAX_DETAILED_XT_CELLS,
    MAX_SPATIAL_BINS,
    MAX_TIME_POINTS,
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
    assert result.detailed_outputs is False
    assert result.time_grid.size == 0
    assert set(result.outputs) == set(CORE_OUTPUT_KEYS)
    assert set(DISTRIBUTION_OUTPUT_KEYS).isdisjoint(result.outputs)
    assert set(TIME_OUTPUT_KEYS).isdisjoint(result.outputs)
    assert set(XT_OUTPUT_KEYS).isdisjoint(result.outputs)
    assert progress[0] == (0.02, "Preparing simulation...")
    assert progress[-1] == (1.0, "Simulation complete")
    assert [value for value, _message in progress] == sorted(value for value, _message in progress)
    for array in (
        result.position_kb,
        result.replication_timing,
        result.time_grid,
        *result.outputs.values(),
    ):
        assert array.flags.writeable is False


def test_detailed_mode_collects_time_and_space_time_outputs() -> None:
    result = run_constant_simulation(
        _request(
            track_detailed_outputs=True,
            firing_factors=10.0,
        )
    )

    assert result.detailed_outputs is True
    np.testing.assert_array_equal(result.time_grid, np.arange(11, dtype=float))
    assert set(CORE_OUTPUT_KEYS) <= result.outputs.keys()
    assert set(DISTRIBUTION_OUTPUT_KEYS) <= result.outputs.keys()
    assert set(TIME_OUTPUT_KEYS) <= result.outputs.keys()
    assert set(XT_OUTPUT_KEYS) <= result.outputs.keys()
    for name in TIME_OUTPUT_KEYS:
        assert result.outputs[name].shape == (11,), name
    for name in DISTRIBUTION_OUTPUT_KEYS:
        assert result.outputs[name].shape == (2,), name
    for name in XT_OUTPUT_KEYS:
        assert result.outputs[name].shape == (4, 11), name
        assert result.outputs[name].dtype == np.float32, name
    for array in result.outputs.values():
        assert array.flags.writeable is False


@pytest.mark.parametrize("detailed", (False, True))
def test_rsim_statistics_flags_follow_explicit_detailed_mode(
    monkeypatch: pytest.MonkeyPatch,
    detailed: bool,
) -> None:
    assumptions_seen: dict[str, Any] = {}
    real_rsim = cast(Callable[..., dict[str, Any]], dnascape.rsim)

    def inspect_rsim(*, source: dict[str, Any], assumptions: dict[str, Any]) -> dict[str, Any]:
        assumptions_seen.update(assumptions)
        return real_rsim(source=source, assumptions=assumptions)

    monkeypatch.setattr(dnascape, "rsim", inspect_rsim)
    run_constant_simulation(_request(track_detailed_outputs=detailed))

    assert assumptions_seen["time_statsQ"] is detailed
    assert assumptions_seen["time_stats_xtQ"] is detailed
    assert ("time_grid" in assumptions_seen) is detailed


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
        ({"track_detailed_outputs": "yes"}, "Detailed-output tracking must be"),
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
            {"time_window": float(MAX_TIME_POINTS), "resolution_t": 1.0},
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


def test_oversized_bin_replicate_product_is_rejected_before_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_allocation(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("oversized ensemble reached time-grid allocation")

    monkeypatch.setattr(simulation_service.np, "arange", unexpected_allocation)
    with pytest.raises(ValueError, match="ensemble is too large for an interactive run"):
        validate_constant_simulation(_request(region_end_kb=10_001.0, replicates=2_000))


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


def test_detailed_replicate_scaled_work_limit_is_rejected() -> None:
    time_points = 1_001
    n_bins = 1_000
    replicates = MAX_DETAILED_WORK_CELLS // (n_bins * time_points) + 1

    with pytest.raises(ValueError, match="detailed space-time workload is too large"):
        validate_constant_simulation(
            _request(
                region_end_kb=float(n_bins),
                time_window=float(time_points - 1),
                replicates=replicates,
                track_detailed_outputs=True,
            )
        )
