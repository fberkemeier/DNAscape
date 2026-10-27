"""Exercise the public mapping catalogue against the actual scientific router."""

from __future__ import annotations

from dataclasses import replace
from itertools import product
from typing import Any, cast

import numpy as np
import pytest

from app.mapping_catalog import DataForm, MappingRoute, mapping_route, mapping_routes
from app.mapping_service import MappingRequest, run_mapping
from dnascape import rmap
from dnascape.mapping import map_timing_directionality

mapper = cast(Any, rmap)


def _small_input(key: str, form: DataForm) -> Any:
    if form == "regions":
        return {"left": (0.0, 4.0), "right": (4.0, 8.0)}
    if form == "bundle":
        return {
            "replicated_fraction_xt": np.tile(np.linspace(0, 1, 7), (8, 1)),
            "right_fork_density_xt": np.full((8, 7), 0.03),
            "left_fork_density_xt": np.full((8, 7), 0.02),
            "fork_density_xt": np.full((8, 7), 0.05),
        }
    if key == "replication_timing":
        return 10.0 + np.arange(8) * 0.05
    if key == "replicated_fraction_xt":
        return np.tile(np.linspace(0, 1, 7), (8, 1))
    value = {
        "fork_speed": 1.4,
        "fork_directionality": 0.07,
        "replicated_fraction_xt": 0.5,
    }.get(key, 0.02)
    if form == "scalar":
        return value
    if form == "x":
        return np.full(8, value)
    if form == "t":
        return np.full(7, value)
    if form == "xt":
        return np.full((8, 7), value)
    raise AssertionError(f"Uncovered input form: {key} / {form}")


def _small_request(
    route: MappingRoute,
    forms: dict[str, DataForm] | None = None,
    target_shape: str | None = None,
) -> MappingRequest:
    if forms is None:
        forms = {item.key: item.forms[0] for item in route.inputs if item.required}
    settings: dict[str, Any] = {}
    for setting in route.active_settings(target_shape or route.default_target_shape, forms):
        if setting.key.endswith("maxiter"):
            settings[setting.key] = 1
        elif setting.key.endswith("sim_number"):
            settings[setting.key] = 2
        elif setting.key == "time_grid":
            settings[setting.key] = np.arange(7, dtype=float) * 5.0
        elif setting.key == "refine_edge_kb":
            settings[setting.key] = 0.0
        elif setting.key == "max_time":
            settings[setting.key] = 30.0
        elif setting.key == "seed":
            settings[setting.key] = 42
    return MappingRequest(
        route=route,
        inputs={key: _small_input(key, form) for key, form in forms.items()},
        input_forms=forms,
        settings=settings,
        region_start_kb=0.0,
        region_end_kb=8.0,
        resolution_x=1.0,
        resolution_t=5.0,
        target_shape=target_shape,
    )


@pytest.mark.parametrize("route", mapping_routes(), ids=lambda route: route.method)
def test_every_catalog_route_executes_real_science(route: MappingRoute) -> None:
    result = run_mapping(_small_request(route))

    assert result.method == route.method
    assert result.target == route.target
    primary = result.outputs[result.primary_output]
    assert primary.size or route.target == "inter_origin_distance"
    assert np.isfinite(primary).sum() + result.diagnostics.get("unidentifiable_count", 0) == primary.size
    assert not primary.flags.writeable
    form = result.output_forms[result.primary_output]
    if form == "x":
        assert primary.shape == result.x_axis.shape == (8,)
    elif form == "t":
        assert primary.shape == result.time_axis.shape
    elif form == "xt":
        assert primary.shape == (result.x_axis.size, result.time_axis.size)
    elif form == "regions":
        assert primary.size == len(result.region_labels) == 2


def _additional_route_cases() -> list[tuple[MappingRoute, dict[str, DataForm], str | None]]:
    cases = []
    for route in mapping_routes():
        base_forms = {item.key: item.forms[0] for item in route.inputs if item.required}
        for selected_forms in product(*(item.forms for item in route.inputs)):
            forms = dict(zip((item.key for item in route.inputs), selected_forms, strict=True))
            for shape in route.shapes_for(forms) or (None,):
                if forms == base_forms and shape == route.default_target_shape:
                    continue
                cases.append((route, forms, shape))
    return cases


@pytest.mark.parametrize(
    ("route", "forms", "shape"),
    _additional_route_cases(),
    ids=[
        f"{route.method}-{'-'.join(forms.values())}-{shape}"
        for route, forms, shape in _additional_route_cases()
    ],
)
def test_supported_input_forms_and_target_shapes_execute(
    route: MappingRoute, forms: dict[str, DataForm], shape: str | None
) -> None:
    result = run_mapping(_small_request(route, forms, shape))
    values = result.outputs[result.primary_output]
    assert np.isfinite(values).sum() + result.diagnostics.get("unidentifiable_count", 0) == values.size
    assert result.method == route.method


@pytest.mark.parametrize("method", ["directionality_and_speed_to_timing", "directionality_to_timing"])
def test_spatial_speed_directionality_route_recovers_timing(method: str) -> None:
    timing = 7.0 + np.cumsum(np.linspace(0.01, 0.09, 8))
    speed = np.linspace(0.8, 1.9, 8)
    directionality = cast(Any, map_timing_directionality)(timing, fork_speed=speed, T0=7.0)
    result = run_mapping(
        MappingRequest(
            route=mapping_route(method),
            inputs={"fork_directionality": directionality, "fork_speed": speed},
            input_forms={"fork_directionality": "x", "fork_speed": "x"},
            settings={"T0": 7.0},
            region_start_kb=0.0,
            region_end_kb=8.0,
            resolution_x=1.0,
        )
    )
    np.testing.assert_allclose(result.outputs["replication_timing"], timing)


def test_simulation_only_outputs_are_absent_from_mapping() -> None:
    for method in ("firing_and_speed_to_initiation_pdf_mean_field",
                   "firing_and_speed_to_inter_origin_distance_simulation"):
        with pytest.raises(KeyError):
            mapping_route(method)


@pytest.mark.parametrize("target_form", ["time", "spacetime"])
@pytest.mark.parametrize("time_points", [7, 8])
def test_weibull_firing_fit_accepts_temporal_speed_without_reinterpreting_it(
    target_form: str, time_points: int
) -> None:
    # Cover both unequal axes and the formerly silent X == T ambiguity.
    result = mapper(
            source={
                "replication_timing": 10.0 + np.arange(8) * 0.05,
                "fork_speed": np.full((1, time_points), 1.4),
            },
            target="initiation_rate",
            assumptions={
                "target_form": target_form,
                "target_shape": "weibull",
                "time_grid": np.arange(time_points) * 5.0,
                "maxiter": 1,
                "verbose": False,
            },
    )
    assert np.asarray(result).shape == (1 if target_form == "time" else 8, time_points)
    assert np.isfinite(result).all()
