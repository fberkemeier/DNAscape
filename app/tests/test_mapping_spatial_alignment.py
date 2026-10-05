"""Spatial alignment and genomic bounds at the Mapping-to-rmap boundary."""

from typing import Any

import numpy as np
import pytest

import dnascape
from app.mapping_catalog import mapping_route
from app.mapping_service import MappingRequest, run_mapping, validate_mapping_request


@pytest.mark.parametrize("start,end,expected_start,expected_size", [
    (0, 248_441, 0, 10_000),
    (100_000, 120_000, 100_000, 10_000),
    (100_000, 100_128, 100_000, 128),
    (248_400, 260_000, 248_400, 41),
])
def test_chromosome_and_local_profile_use_shortest_available_window(
    start: int, end: int, expected_start: int, expected_size: int,
) -> None:
    timing = np.arange(248_441, dtype=float) + 1
    firing = np.linspace(0.001, 0.002, 10_000)
    prepared = validate_mapping_request(MappingRequest(
        route=mapping_route("timing_and_firing_to_spatial_fork_speed"),
        inputs={"replication_timing": timing, "firing_rate": firing},
        input_forms={"replication_timing": "x", "firing_rate": "x"},
        region_start_kb=start, region_end_kb=end, resolution_x=1,
    ))
    np.testing.assert_array_equal(
        prepared.source["replication_timing"],
        timing[expected_start:expected_start + expected_size],
    )
    np.testing.assert_array_equal(prepared.source["firing_rate"], firing[:expected_size])
    assert prepared.spatial_size == expected_size
    assert prepared.start_kb == expected_start


def test_spacetime_firing_is_cropped_in_space_without_changing_its_time_axis() -> None:
    timing = np.arange(100, dtype=float) + 1
    firing = np.arange(60, dtype=float).reshape(20, 3) * 1e-5
    prepared = validate_mapping_request(MappingRequest(
        route=mapping_route("timing_and_time_firing_to_spatial_fork_speed"),
        inputs={"replication_timing": timing, "firing_rate": firing},
        input_forms={"replication_timing": "x", "firing_rate": "xt"},
        region_start_kb=50, region_end_kb=62, resolution_x=1,
        settings={"time_grid": [0, 2, 5]},
    ))
    np.testing.assert_array_equal(prepared.source["replication_timing"], timing[50:62])
    np.testing.assert_array_equal(prepared.source["firing_rate"], firing[:12, :])
    np.testing.assert_array_equal(prepared.assumptions["time_grid"], [0, 2, 5])


@pytest.mark.parametrize("resolution,start,end", [
    (1.0, 100_000.0, 100_128.0),
    (0.5, 50_000.0, 50_032.0),
    (2.0, 200_000.0, 200_256.0),
])
def test_constant_firing_spatial_speed_sends_only_selected_timing_to_rmap(
    monkeypatch: pytest.MonkeyPatch, resolution: float, start: float, end: float,
) -> None:
    timing = np.arange(248_441, dtype=float) + 1
    first, stop = int(start / resolution), int(end / resolution)
    observed = timing[first:stop]
    captured: dict[str, Any] = {}

    def record_rmap(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {
            "values": np.full(observed.size, 1.4),
            "replication_timing_fit": observed,
        }

    monkeypatch.setattr(dnascape, "rmap", record_rmap)
    result = run_mapping(MappingRequest(
        route=mapping_route("timing_and_firing_to_spatial_fork_speed"),
        inputs={"replication_timing": timing, "firing_rate": 0.001},
        input_forms={"replication_timing": "x", "firing_rate": "scalar"},
        region_start_kb=start, region_end_kb=end, resolution_x=resolution,
    ))
    np.testing.assert_array_equal(captured["source"]["replication_timing"], observed)
    assert captured["source"]["firing_rate"] == 0.001
    assert captured["assumptions"]["resolution"] == resolution
    assert captured["assumptions"]["target_form"] == "space"
    assert result.outputs["fork_speed"].size == stop - first
    np.testing.assert_array_equal(result.x_axis, np.arange(first, stop) * resolution)
    np.testing.assert_array_equal(
        result.comparisons["replication_timing_fit"].observed, observed,
    )
    assert result.x_axis[-1] < end
