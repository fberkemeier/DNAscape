"""Spatial alignment and genomic bounds at the Mapping-to-rmap boundary."""

from typing import Any

import numpy as np
import pytest

import dnascape
from app.mapping_catalog import mapping_route
from app.mapping_service import MappingRequest, run_mapping, validate_mapping_request


@pytest.mark.parametrize("start,end,expected_start,expected_size", [
    (0, 10_000, 0, 10_000),
    (100_000, 110_000, 100_000, 10_000),
    (100_000, 100_128, 100_000, 128),
    (30_000, 35_000, 30_000, 5_000),
    (248_400, 248_441, 248_400, 41),
])
def test_chromosome_and_local_profile_use_the_complete_selected_window(
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


@pytest.mark.parametrize("whole_region", [False, True])
def test_shorter_prepared_data_aligns_automatically_without_changing_bin_width(whole_region):
    bundle = {
        "x_array": np.arange(20., 60., 2.),
        "time_grid": np.asarray([0., 1., 3.]),
        "replicated_fraction_xt": np.full((20, 3), .5),
        "right_fork_density_xt": np.full((20, 3), .01),
        "left_fork_density_xt": np.full((20, 3), .02),
    }
    speed = np.linspace(1., 2., 100)
    parameters = dict(
        route=mapping_route("fork_density_bundle_to_coalescence_events_xt_mean_field"),
        inputs={"fork_densities": bundle, "fork_speed": speed},
        input_forms={"fork_densities": "bundle", "fork_speed": "x"},
        region_start_kb=100., region_end_kb=110., resolution_x=2.,
        whole_genome=whole_region,
    )
    request = MappingRequest(**parameters)
    prepared = validate_mapping_request(request)
    supplied = {**prepared.source, **prepared.assumptions}
    bundle_key = next(item.key for item in parameters["route"].inputs if item.prepared_result_only)
    selected_bundle = supplied[bundle_key]
    expected_axis = np.arange(0., 40., 2.) if whole_region else np.arange(100., 110., 2.)
    size = 20 if whole_region else 5
    expected_speed = speed[:20] if whole_region else speed[50:55]
    np.testing.assert_array_equal(selected_bundle["x_array"], expected_axis)
    np.testing.assert_array_equal(selected_bundle["time_grid"], bundle["time_grid"])
    np.testing.assert_array_equal(selected_bundle["right_fork_density_xt"], bundle["right_fork_density_xt"][:size])
    np.testing.assert_array_equal(supplied["fork_speed"], expected_speed)
    np.testing.assert_array_equal(bundle["x_array"], np.arange(20., 60., 2.))
    result = run_mapping(request)
    np.testing.assert_array_equal(result.x_axis, expected_axis)
    np.testing.assert_array_equal(result.time_axis, bundle["time_grid"])


@pytest.mark.parametrize("resolution,start,end", [
    (1., 30_000., 45_000.),
    (1., 0., 10_001.),
    (.5, 30_000., 35_001.),
    (2., 30_000., 50_001.),
])
def test_oversized_width_is_rejected_before_mapping(monkeypatch, resolution, start, end):
    def forbidden(**_kwargs):
        pytest.fail("Invalid regions must not enter the numerical mapper.")

    monkeypatch.setattr(dnascape, "rmap", forbidden)
    request = MappingRequest(
        route=mapping_route("timing_and_firing_to_spatial_fork_speed"),
        inputs={"replication_timing": np.ones(100_000), "firing_rate": np.full(10_000, .001)},
        input_forms={"replication_timing": "x", "firing_rate": "x"},
        region_start_kb=start, region_end_kb=end, resolution_x=resolution,
    )
    with pytest.raises(ValueError, match=r"Firing rate, the shortest spatial profile.*10,000 bins"):
        run_mapping(request)
    assert request.region_start_kb == start
    assert request.region_end_kb == end


def test_end_outside_the_largest_profile_is_rejected():
    with pytest.raises(ValueError, match=r"Region end.*outside.*longest uploaded spatial input"):
        validate_mapping_request(MappingRequest(
            route=mapping_route("timing_and_firing_to_spatial_fork_speed"),
            inputs={"replication_timing": np.ones(100), "firing_rate": np.full(20, .001)},
            input_forms={"replication_timing": "x", "firing_rate": "x"},
            region_start_kb=90., region_end_kb=105., resolution_x=1.,
        ))


@pytest.mark.parametrize("start,end", [(30_000., 45_000.), (-1., -10.), (None, None)])
def test_whole_region_uses_zero_to_the_shortest_length_and_ignores_entered_bounds(start, end):
    timing = np.arange(100, dtype=float) + 1
    firing = np.linspace(.001, .002, 20)
    prepared = validate_mapping_request(MappingRequest(
        route=mapping_route("timing_and_firing_to_spatial_fork_speed"),
        inputs={"replication_timing": timing, "firing_rate": firing},
        input_forms={"replication_timing": "x", "firing_rate": "x"},
        region_start_kb=start, region_end_kb=end, resolution_x=2., whole_genome=True,
    ))
    np.testing.assert_array_equal(prepared.source["replication_timing"], timing[:20])
    np.testing.assert_array_equal(prepared.source["firing_rate"], firing)
    assert prepared.start_kb == 0.
    assert prepared.spatial_size == 20


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
