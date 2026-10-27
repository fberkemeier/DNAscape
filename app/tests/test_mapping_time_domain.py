"""Forward mappings always receive a usable clock, including static inputs."""

from dataclasses import replace

import numpy as np
import pytest

from dnascape.workflows.mapping import MappingRequest, run_mapping, validate_mapping_request
from dnascape.workflows.mapping_catalog import mapping_route, mapping_routes
from dnascape.workflows.mapping_recipe import run_mapping_recipe


FORWARD_METHODS = (
    "firing_and_speed_to_replicated_fraction_xt_lightcone",
    "firing_and_speed_to_directional_fork_densities_xt_mean_field",
)


def _request(method: str, **changes) -> MappingRequest:
    values = {
        "route": mapping_route(method),
        "inputs": {"initiation_rate": 0.001, "fork_speed": 1.4},
        "input_forms": {"initiation_rate": "scalar", "fork_speed": "scalar"},
        "region_start_kb": 0.0,
        "region_end_kb": 4.0,
        "resolution_x": 1.0,
    }
    values.update(changes)
    return MappingRequest(**values)


@pytest.mark.parametrize("method", FORWARD_METHODS)
@pytest.mark.parametrize("firing_form", ["scalar", "x"])
@pytest.mark.parametrize("speed_form", ["scalar", "x"])
def test_static_forward_mapping_defaults_have_a_finite_time_grid(
    method: str, firing_form: str, speed_form: str,
) -> None:
    request = _request(method, inputs={
        "initiation_rate": 0.001 if firing_form == "scalar" else np.linspace(0.001, 0.002, 4),
        "fork_speed": 1.4 if speed_form == "scalar" else np.linspace(1.0, 1.4, 4),
    }, input_forms={"initiation_rate": firing_form, "fork_speed": speed_form})
    result = run_mapping(request)
    expected_times = result.time_axis
    assert expected_times[0] == 0.
    assert np.all(np.diff(expected_times) == 5.)
    assert result.assumptions.get("max_time") is None
    if "lightcone" in method:
        assert np.min(result.outputs["replicated_fraction_xt"][:, -1]) > 1. - 1e-7
    np.testing.assert_array_equal(result.x_axis, [0.0, 1.0, 2.0, 3.0])
    assert result.time_axis.flags.writeable is False
    assert result.outputs[result.primary_output].shape == (4, expected_times.size)
    for key, field in result.outputs.items():
        if result.output_forms[key] == "xt":
            assert field.shape == (4, expected_times.size)
        assert np.isfinite(field).all()


@pytest.mark.parametrize("method", FORWARD_METHODS)
@pytest.mark.parametrize("input_key", ["initiation_rate", "fork_speed"])
@pytest.mark.parametrize("form", ["t", "xt"])
def test_temporal_uploads_extend_their_clock_at_the_native_step(
    method: str, input_key: str, form: str,
) -> None:
    request = _request(method)
    inputs = dict(request.inputs)
    forms = dict(request.input_forms)
    inputs[input_key] = np.full((1 if form == "t" else 4, 5), inputs[input_key])
    forms[input_key] = form
    result = run_mapping(replace(request, inputs=inputs, input_forms=forms, resolution_t=3.0))
    np.testing.assert_array_equal(result.time_axis[:5], [0.0, 3.0, 6.0, 9.0, 12.0])
    assert np.all(np.diff(result.time_axis) == 3.)
    assert result.time_axis[-1] > 12.
    assert result.assumptions.get("max_time") is None
    assert result.outputs[result.primary_output].shape == (4, result.time_axis.size)


@pytest.mark.parametrize("method", FORWARD_METHODS)
def test_custom_maximum_ends_the_grid_exactly_at_the_limit(method: str) -> None:
    result = run_mapping(_request(method, settings={"max_time": 11.0, "resolution_time": 3.0}))
    np.testing.assert_array_equal(result.time_axis, [0.0, 3.0, 6.0, 9.0, 11.0])
    assert result.assumptions["max_time"] == 11.0


@pytest.mark.parametrize("method", FORWARD_METHODS)
def test_explicit_time_grid_overrides_the_automatic_grid(method: str) -> None:
    result = run_mapping(_request(method, settings={"time_grid": [0.0, 2.0, 7.0]}))
    np.testing.assert_array_equal(result.time_axis, [0.0, 2.0, 7.0])
    assert result.assumptions.get("max_time") is None


def test_explicit_maximum_replication_time_alias_is_preserved() -> None:
    result = run_mapping(_request(FORWARD_METHODS[0], settings={"max_rep_time": 10.0}))
    np.testing.assert_array_equal(result.time_axis, [0.0, 5.0, 10.0])
    assert result.assumptions["max_time"] == 10.0


@pytest.mark.parametrize("maximum", [1e12, 1e308])
def test_oversized_default_grid_is_rejected_before_allocation(maximum: float) -> None:
    with pytest.raises(ValueError, match="time grid exceeds the local limit"):
        validate_mapping_request(_request(FORWARD_METHODS[0], settings={
            "max_time": maximum, "resolution_time": 1e-6,
        }))


def test_all_forward_routes_derive_the_maximum_time() -> None:
    covered = set()
    for route in mapping_routes():
        settings = {item.key: item for item in route.active_settings()}
        if {"time_grid", "max_time"} <= settings.keys():
            assert settings["max_time"].default is None
            assert settings["max_time"].location == "derived"
            covered.add(route.method)
    assert set(FORWARD_METHODS) <= covered


@pytest.mark.parametrize("method", FORWARD_METHODS)
def test_downloadable_recipe_uses_the_same_forward_clock(method: str) -> None:
    recipe = {
        "method": method,
        "settings": {"resolution": 1.0},
        "region": {"start_kb": 20.0, "end_kb": 24.0, "whole_genome": False},
        "inputs": {
            "initiation_rate": {"form": "scalar", "mode": "constant", "values": 0.001},
            "fork_speed": {"form": "scalar", "mode": "constant", "values": 1.4},
        },
    }
    (completed,) = run_mapping_recipe(recipe)
    assert completed.result.time_axis[0] == 0.0
    assert completed.result.time_axis[-1] > 0.
    assert completed.result.assumptions.get("max_time") is None
    np.testing.assert_array_equal(completed.result.x_axis, [20.0, 21.0, 22.0, 23.0])
