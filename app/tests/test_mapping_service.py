"""Focused checks for the framework-independent rmap execution bridge."""

from __future__ import annotations

import gzip
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

import app.mapping_service as mapping_service
import dnascape
from app.mapping_catalog import DataForm, MappingRoute, mapping_route, mapping_routes
from app.mapping_exports import (
    build_mapping_comparison_raw_export,
    build_mapping_raw_export,
)
from app.mapping_service import (
    MappingRequest,
    materialize_many_tracks,
    materialize_track_upload,
    run_mapping,
    validate_mapping_request,
)
from dnascape.mapping import map_timing_directionality


@dataclass(frozen=True)
class _TrackUpload:
    name: str
    content: bytes
    source_format: str = "bedgraph"
    columns: tuple[str, ...] = ("chromosome", "start", "end", "value")


def _deterministic_request(**changes: Any) -> MappingRequest:
    values: dict[str, Any] = {
        "route": mapping_route("firing_and_speed_to_timing_deterministic"),
        "inputs": {"firing_rate": 0.001, "fork_speed": 1.4},
        "input_forms": {"firing_rate": "scalar", "fork_speed": "scalar"},
        "region_start_kb": 20.0,
        "region_end_kb": 24.0,
        "resolution_x": 1.0,
    }
    values.update(changes)
    return MappingRequest(**values)


def test_real_deterministic_map_runs_with_axes_progress_and_read_only_outputs() -> None:
    progress: list[tuple[float, str]] = []
    result = run_mapping(
        _deterministic_request(),
        lambda value, message: progress.append((value, message)),
    )

    assert result.target == "replication_timing"
    assert result.target_label == "Replication timing"
    assert result.method == "firing_and_speed_to_timing_deterministic"
    assert result.primary_output == "replication_timing"
    assert result.output_form == "x"
    assert result.output_dependence == "space"
    assert result.outputs["replication_timing"].shape == (4,)
    np.testing.assert_array_equal(result.x_axis, [20.0, 21.0, 22.0, 23.0])
    assert result.time_axis.size == 0
    assert result.elapsed_seconds >= 0
    assert progress[0] == (0.02, "Preparing mapping...")
    assert progress[-1] == (1.0, "Mapping complete")
    assert [value for value, _message in progress] == sorted(value for value, _message in progress)
    assert result.x_axis.flags.writeable is False
    assert result.outputs["replication_timing"].flags.writeable is False
    with pytest.raises(TypeError):
        result.outputs["new"] = np.ones(1)  # type: ignore[index]


def test_directionality_speed_route_exposes_complete_preview_and_exports() -> None:
    from app.pages.mapping import _result_preview_outputs

    route = mapping_route("timing_and_directionality_to_spatial_fork_speed")
    count = 24
    axis = np.linspace(0.0, 1.0, count)
    timing = 7.0 + np.cumsum(np.linspace(0.03, 0.09, count))
    expected_speed = 1.1 + 0.25 * np.sin(np.pi * axis) ** 2
    directionality = map_timing_directionality(
        timing,
        fork_speed=expected_speed,
        resolution=2.0,
        T0=6.0,
    )

    result = run_mapping(
        MappingRequest(
            route=route,
            inputs={
                "replication_timing": timing,
                "fork_directionality": directionality,
            },
            input_forms={
                "replication_timing": "x",
                "fork_directionality": "x",
            },
            settings={"T0": 6.0},
            region_start_kb=0.0,
            region_end_kb=48.0,
            resolution_x=2.0,
        )
    )

    assert result.primary_output == "fork_speed"
    assert result.output_forms["fork_speed"] == "x"
    assert result.output_forms["replication_timing_fit"] == "x"
    assert result.output_forms["residual"] == "x"
    assert result.output_forms["fit_history"] == "supplied"
    assert result.fit_mse is not None and np.isfinite(result.fit_mse)
    assert np.isfinite(result.outputs["replication_timing_fit"]).all()
    assert np.isfinite(result.outputs["residual"]).all()
    comparison = result.comparisons["replication_timing_fit"]
    np.testing.assert_array_equal(comparison.observed, timing)
    np.testing.assert_array_equal(comparison.fitted, result.outputs["replication_timing_fit"])
    assert tuple(_result_preview_outputs(result)) == (
        "fork_speed",
        "replication_timing_fit",
        "input:replication_timing",
        "input:fork_directionality",
        "comparison:replication_timing_fit",
        "residual",
        "fit_history",
    )

    target_export = build_mapping_raw_export(result, "fork_speed", "tsv")
    assert target_export.filename == "rmap_fs.tsv"
    assert target_export.data.decode().splitlines()[0] == "fork_speed"
    comparison_export = build_mapping_comparison_raw_export(
        result,
        "replication_timing_fit",
        "csv",
    )
    assert comparison_export.filename == "rmap_rt-cmp.csv"
    assert comparison_export.data.decode().splitlines()[0] == "position_kb,observed,fitted"


def test_time_target_keeps_the_uploaded_chromosome_axis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route = mapping_route("timing_and_firing_to_time_fork_speed")

    def fake_rmap(**_kwargs: Any) -> dict[str, Any]:
        return {
            "values": np.asarray([1.2, 1.4, 1.6]),
            "time_grid": np.asarray([0.0, 5.0, 10.0]),
        }

    monkeypatch.setattr(dnascape, "rmap", fake_rmap)
    result = run_mapping(
        MappingRequest(
            route=route,
            inputs={
                "replication_timing": np.arange(14, dtype=float),
                "firing_rate": 0.001,
            },
            input_forms={"replication_timing": "x", "firing_rate": "scalar"},
            region_start_kb=10.0,
            region_end_kb=14.0,
            resolution_x=1.0,
            resolution_t=5.0,
        )
    )

    assert result.output_form == "t"
    np.testing.assert_array_equal(result.x_axis, [10.0, 11.0, 12.0, 13.0])
    np.testing.assert_array_equal(result.time_axis, [0.0, 5.0, 10.0])


def test_optional_feature_assumption_is_omitted_until_overridden() -> None:
    route = mapping_route("firing_to_timing_deterministic")
    base = MappingRequest(
        route=route,
        inputs={"firing_rate": np.full(4, 0.001)},
        input_forms={"firing_rate": "x"},
        region_start_kb=0,
        region_end_kb=4,
        resolution_x=1,
    )
    prepared = validate_mapping_request(base)
    assert prepared.source.keys() == {"firing_rate"}
    assert "fork_speed" not in prepared.assumptions

    overridden = validate_mapping_request(
        MappingRequest(
            route=route,
            inputs={"firing_rate": np.full(4, 0.001), "fork_speed": 1.8},
            input_forms={"firing_rate": "x", "fork_speed": "scalar"},
            region_start_kb=0,
            region_end_kb=4,
            resolution_x=1,
        )
    )
    assert overridden.source.keys() == {"firing_rate"}
    assert overridden.assumptions["fork_speed"] == 1.8


@pytest.mark.parametrize(
    ("target_shape", "settings"),
    (("weibull", {"temporal_sim_number": 20}), ("weibull", {"spacetime_order": "crossed"})),
)
def test_request_rejects_inactive_settings_left_over_from_another_shape(
    target_shape: str, settings: dict[str, Any]
) -> None:
    with pytest.raises(ValueError, match="does not accept setting"):
        validate_mapping_request(
            MappingRequest(
                route=mapping_route("timing_and_speed_to_time_firing_rate"),
                inputs={"replication_timing": np.arange(4) + 10.0, "fork_speed": 1.4},
                input_forms={"replication_timing": "x", "fork_speed": "scalar"},
                target_shape=target_shape,
                settings=settings,
            )
        )


@pytest.mark.parametrize("speed_form", ["t", "xt"])
def test_request_accepts_weibull_with_time_dependent_speed_even_for_square_grids(
    speed_form: DataForm,
) -> None:
    speed = np.full((1 if speed_form == "t" else 4, 4), 1.4)
    prepared = validate_mapping_request(
        MappingRequest(
            route=mapping_route("timing_and_speed_to_time_firing_rate"),
            inputs={"replication_timing": np.arange(4) + 10.0, "fork_speed": speed},
            input_forms={"replication_timing": "x", "fork_speed": speed_form},
            target_shape="weibull",
        )
    )
    assert prepared.source["fork_speed"].shape == speed.shape


@pytest.mark.parametrize(
    "method",
    [
        "firing_and_speed_to_replicated_fraction_xt_lightcone",
        "firing_and_speed_to_initiation_events_xt_mean_field",
    ],
)
def test_request_accepts_new_periodic_forward_routes(method: str) -> None:
    prepared = validate_mapping_request(
        MappingRequest(
            route=mapping_route(method),
            inputs={"firing_rate": 0.01, "fork_speed": 1.4},
            input_forms={"firing_rate": "scalar", "fork_speed": "scalar"},
            settings={"perQ": True},
            region_end_kb=4,
        )
    )
    assert prepared.assumptions["perQ"] is True


def test_initial_kinetics_text_is_parsed_into_numeric_fields_before_dispatch() -> None:
    prepared = validate_mapping_request(
        MappingRequest(
            route=mapping_route("timing_to_joint_spacetime_kinetic_fields"),
            inputs={"replication_timing": np.arange(4) + 10.0},
            input_forms={"replication_timing": "x"},
            settings={"initial_kinetics": '{"firing_rate": 0.001, "fork_speed": 1.4}'},
        )
    )
    assert prepared.assumptions["initial_kinetics"]["firing_rate"] == 0.001
    assert prepared.assumptions["initial_kinetics"]["fork_speed"] == 1.4


@pytest.mark.parametrize("payload", ["not JSON", "[]", '{"unrelated": 1}', '{"fork_speed": 0}'])
def test_invalid_initial_kinetics_text_fails_before_dispatch(payload: str) -> None:
    with pytest.raises(ValueError, match="Initial"):
        validate_mapping_request(
            MappingRequest(
                route=mapping_route("timing_to_joint_spacetime_kinetic_fields"),
                inputs={"replication_timing": np.arange(4) + 10.0},
                input_forms={"replication_timing": "x"},
                settings={"initial_kinetics": payload},
            )
        )


@pytest.mark.parametrize("whole_region", [False, True])
def test_prepared_bundle_preserves_nonzero_origin_resolution_and_time_axis(
    whole_region: bool,
) -> None:
    axis = 20.0 + np.arange(4) * 2.5
    times = np.asarray([0.0, 1.0, 3.0, 8.0])
    right = np.tile([0.1, 0.2, 0.4, 0.8], (4, 1))
    left = np.full((4, 4), 0.2)
    bundle = {
        "right_fork_density_xt": right,
        "left_fork_density_xt": left,
        "time_grid": times,
        "x_array": axis,
    }
    request = MappingRequest(
        route=mapping_route("fork_density_bundle_to_directionality_time_integrated"),
        inputs={"fork_densities_xt": bundle},
        input_forms={"fork_densities_xt": "bundle"},
        region_start_kb=22.5,
        region_end_kb=27.5,
        whole_genome=whole_region,
        resolution_x=1.0,  # Prior route control must not rescale intrinsic coordinates.
    )
    prepared = validate_mapping_request(request)
    np.testing.assert_array_equal(prepared.assumptions["time_grid"], times)
    assert prepared.resolution_x == 2.5
    retained = prepared.source["fork_densities_xt"]
    np.testing.assert_array_equal(retained["time_grid"], times)
    expected_x = np.arange(4) * 2.5 if whole_region else axis[1:3]
    np.testing.assert_array_equal(retained["x_array"], expected_x)
    assert retained["right_fork_density_xt"].shape == (len(expected_x), 4)

    result = run_mapping(request)
    right_integral = np.trapezoid(right[0], x=times)
    left_integral = np.trapezoid(left[0], x=times)
    expected_rfd = (right_integral - left_integral) / (right_integral + left_integral)
    np.testing.assert_allclose(result.outputs["fork_directionality"], expected_rfd)
    np.testing.assert_array_equal(result.x_axis, expected_x)


def test_request_rejects_an_explicit_time_axis_conflicting_with_a_prepared_bundle() -> None:
    with pytest.raises(ValueError, match="does not match the prepared result"):
        validate_mapping_request(
            MappingRequest(
                route=mapping_route("fork_density_bundle_to_directionality_time_integrated"),
                inputs={
                    "fork_densities_xt": {
                        "right_fork_density_xt": np.ones((3, 3)),
                        "left_fork_density_xt": np.ones((3, 3)),
                        "time_grid": np.asarray([0.0, 1.0, 3.0]),
                    }
                },
                input_forms={"fork_densities_xt": "bundle"},
                settings={"time_grid": [0.0, 5.0, 10.0]},
            )
        )


def test_shorter_prepared_bundle_starts_at_first_bin_with_a_longer_global_speed_profile() -> None:
    from dnascape.mapping import map_coalescence_events_xt

    times = np.asarray([0.0, 2.0, 5.0])
    right = np.repeat((0.001 + np.arange(8) * 0.001)[:, None], 3, axis=1)
    left = np.repeat((0.002 + np.arange(8) * 0.002)[:, None], 3, axis=1)
    fraction = np.full((8, 3), 0.5)
    speed = 1.0 + np.arange(40) * 0.01
    request = MappingRequest(
        route=mapping_route("fork_density_bundle_to_coalescence_events_xt_mean_field"),
        inputs={
            "fork_densities_xt": {
                "right_fork_density_xt": right,
                "left_fork_density_xt": left,
                "replicated_fraction_xt": fraction,
                "time_grid": times,
                "x_array": 20.0 + np.arange(8),
            },
            "fork_speed": speed,
        },
        input_forms={"fork_densities_xt": "bundle", "fork_speed": "x"},
        region_start_kb=22.0,
        region_end_kb=26.0,
    )
    prepared = validate_mapping_request(request)
    bundle = prepared.source["fork_densities_xt"]
    np.testing.assert_array_equal(bundle["x_array"], np.arange(22.0, 26.0))
    np.testing.assert_array_equal(bundle["right_fork_density_xt"], right[:4])
    np.testing.assert_array_equal(prepared.source["fork_speed"], speed[22:26])

    result = run_mapping(request)
    expected = map_coalescence_events_xt(
        speed[22:26], fraction[:4], right[:4], left[:4], time_grid=times
    )
    np.testing.assert_allclose(result.outputs["coalescence_events_xt"], expected)
    np.testing.assert_array_equal(result.x_axis, np.arange(22.0, 26.0))
    np.testing.assert_array_equal(result.time_axis, times)


def test_free_spacetime_firing_is_unavailable_before_dispatch() -> None:
    with pytest.raises(
        ValueError, match="Target shape must be one of: weibull"
    ):
        validate_mapping_request(
            MappingRequest(
                route=mapping_route("timing_and_speed_to_spacetime_firing_rate"),
                inputs={"replication_timing": np.arange(4) + 10.0, "fork_speed": 1.4},
                input_forms={"replication_timing": "x", "fork_speed": "scalar"},
                target_shape="free",
                settings={"spacetime_order": "crossed"},
            )
        )


def test_periodic_setting_is_dispatched_to_rmap_as_perq(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_rmap(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"values": np.arange(4, dtype=float)}

    monkeypatch.setattr(dnascape, "rmap", fake_rmap)
    result = run_mapping(_deterministic_request(settings={"perQ": True}))

    assert captured["assumptions"]["perQ"] is True
    assert result.assumptions["perQ"] is True


def test_periodic_geometry_is_normalized_to_rmap_perq(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_rmap(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"values": np.arange(4, dtype=float)}

    monkeypatch.setattr(dnascape, "rmap", fake_rmap)
    route = mapping_route("firing_and_speed_to_timing_simulation")
    result = run_mapping(
        MappingRequest(
            route=route,
            inputs={
                "firing_rate": np.full(4, 0.001),
                "fork_speed": 1.4,
            },
            input_forms={"firing_rate": "x", "fork_speed": "scalar"},
            settings={"geometry": "periodic"},
            region_start_kb=0,
            region_end_kb=4,
            resolution_x=1,
        )
    )

    assert captured["assumptions"]["geometry"] == "periodic"
    assert captured["assumptions"]["perQ"] is True
    assert result.assumptions["perQ"] is True


def test_prevalidated_read_only_upload_is_not_copied_during_preparation() -> None:
    firing_rate = np.linspace(0.001, 0.004, 4)
    firing_rate.setflags(write=False)
    prepared = validate_mapping_request(
        _deterministic_request(
            inputs={"firing_rate": firing_rate, "fork_speed": 1.4},
            input_forms={"firing_rate": "x", "fork_speed": "scalar"},
            region_start_kb=0,
            region_end_kb=4,
        )
    )

    normalized = prepared.source["firing_rate"]
    assert np.shares_memory(normalized, firing_rate)
    assert normalized.flags.writeable is False


def test_selected_region_is_cropped_before_rmap(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_rmap(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        firing_rate = np.asarray(kwargs["source"]["firing_rate"], dtype=float)
        return {"values": np.full(firing_rate.size, 7.0)}

    monkeypatch.setattr(dnascape, "rmap", fake_rmap)
    source = np.arange(20, dtype=float) + 0.001
    result = run_mapping(
        _deterministic_request(
            inputs={"firing_rate": source, "fork_speed": 1.4},
            input_forms={"firing_rate": "x", "fork_speed": "scalar"},
            region_start_kb=5,
            region_end_kb=9,
        )
    )

    np.testing.assert_array_equal(captured["source"]["firing_rate"], source[5:9])
    assert captured["assumptions"]["n_bins"] == 4
    assert captured["assumptions"]["interval_kb"] == (5.0, 9.0)
    np.testing.assert_array_equal(result.x_axis, [5.0, 6.0, 7.0, 8.0])


def test_whole_genome_maps_the_full_array_and_ignores_region_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_rmap(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        firing_rate = np.asarray(kwargs["source"]["firing_rate"], dtype=float)
        return {"values": np.full(firing_rate.size, 7.0)}

    monkeypatch.setattr(dnascape, "rmap", fake_rmap)
    source = np.arange(20, dtype=float) + 0.001
    result = run_mapping(
        _deterministic_request(
            inputs={"firing_rate": source, "fork_speed": 1.4},
            input_forms={"firing_rate": "x", "fork_speed": "scalar"},
            region_start_kb=50,
            region_end_kb=10,
            whole_genome=True,
        )
    )

    np.testing.assert_array_equal(captured["source"]["firing_rate"], source)
    assert captured["assumptions"]["n_bins"] == 20
    np.testing.assert_array_equal(result.x_axis, np.arange(20, dtype=float))


def test_whole_genome_requires_a_spatial_input_to_define_its_length() -> None:
    with pytest.raises(ValueError, match="uploaded spatial profile"):
        validate_mapping_request(_deterministic_request(whole_genome=True))


def test_real_constant_firing_fit_sets_exact_target_form() -> None:
    route = mapping_route("timing_and_speed_to_constant_firing_rate")
    result = run_mapping(
        MappingRequest(
            route=route,
            inputs={
                "replication_timing": np.asarray([8.0, 9.0, 10.0, 9.0]),
                "fork_speed": 1.4,
            },
            input_forms={"replication_timing": "x", "fork_speed": "scalar"},
            settings={"maxiter": 1},
            # The page starts with 0--1000. The uploaded four-bin profile is
            # authoritative and must prevent a spurious n_bins=1000 request.
            region_start_kb=0,
            region_end_kb=1_000,
            resolution_x=1,
        )
    )

    assert result.output_form == "scalar"
    assert result.output_dependence == "constant"
    assert result.assumptions["target_form"] == "constant"
    assert result.assumptions["n_bins"] == 4
    assert result.outputs["firing_rate"].shape == (1,)
    assert np.isfinite(result.outputs["firing_rate"][0])
    comparison = result.comparisons["replication_timing_fit"]
    np.testing.assert_array_equal(comparison.observed, [8.0, 9.0, 10.0, 9.0])
    np.testing.assert_array_equal(
        comparison.fitted,
        result.outputs["replication_timing_fit"],
    )


def test_real_spatial_firing_fit_exposes_the_observed_profile_comparison() -> None:
    route = mapping_route("timing_and_speed_to_firing_rate")
    observed = np.asarray([8.0, 9.0, 10.0, 9.0])
    result = run_mapping(
        MappingRequest(
            route=route,
            inputs={"replication_timing": observed, "fork_speed": 1.4},
            input_forms={"replication_timing": "x", "fork_speed": "scalar"},
            settings={"maxiter": 1, "sim_number": 1},
            region_start_kb=0,
            region_end_kb=4,
            resolution_x=1,
        )
    )

    comparison = result.comparisons["replication_timing_fit"]
    np.testing.assert_array_equal(comparison.observed, observed)
    assert comparison.fitted.shape == observed.shape
    assert np.isfinite(comparison.fitted).all()
    np.testing.assert_allclose(
        result.outputs["residual"],
        comparison.fitted - comparison.observed,
    )
    assert result.output_forms["residual"] == "x"


def _input_value(key: str, form: DataForm) -> Any:
    if form == "scalar":
        return 1.4 if key == "fork_speed" else 0.01
    if form == "x":
        return np.linspace(1.0, 2.0, 4)
    if form == "t":
        return np.linspace(1.0, 2.0, 3)
    if form == "xt":
        return np.ones((4, 3), dtype=float)
    if form == "regions":
        return {"region_1": (0.0, 2.0)}
    if form == "bundle":
        return {"replicated_fraction_xt": np.ones((4, 3), dtype=float)}
    return np.linspace(1.0, 2.0, 4)


def _fake_values(route: MappingRoute) -> Any:
    if route.target == "kinetic_fields":
        return {
            "firing_rate": np.ones((4, 3)),
            "fork_speed": np.full((4, 3), 1.4),
            "replication_timing_fit": np.arange(4, dtype=float),
            "fit_history": np.asarray([2.0, 1.0]),
            "time_grid": np.asarray([0.0, 5.0, 10.0]),
        }
    if route.target == "fork_densities_xt":
        return {
            "replicated_fraction_xt": np.ones((4, 3)),
            "right_fork_density_xt": np.ones((4, 3)),
            "left_fork_density_xt": np.ones((4, 3)),
            "fork_density_xt": np.ones((4, 3)),
        }
    if route.output_form == "scalar":
        return 1.0
    if route.output_form == "x":
        return np.linspace(1.0, 2.0, 4)
    if route.output_form == "t":
        return np.linspace(0.0, 1.0, 3)
    if route.output_form == "xt":
        return np.ones((4, 3), dtype=float)
    if route.output_form == "regions":
        return {
            "names": np.asarray(["region_1"]),
            "regions": np.asarray([[0.0, 2.0]]),
            "expected_events": np.asarray([0.4]),
            "efficiency": np.asarray([0.4]),
        }
    return np.asarray([1.0, 2.0])


def test_every_public_exact_route_is_generically_dispatched_with_role_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    by_method = {route.method: route for route in mapping_routes()}

    def fake_rmap(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        route = by_method[str(kwargs["method"])]
        return {"values": _fake_values(route), "time_grid": np.asarray([0.0, 5.0, 10.0])}

    monkeypatch.setattr(dnascape, "rmap", fake_rmap)
    for route in by_method.values():
        inputs: dict[str, Any] = {}
        forms: dict[str, DataForm] = {}
        for input_spec in route.inputs:
            if not input_spec.required:
                continue
            form = input_spec.forms[0]
            inputs[input_spec.key] = _input_value(input_spec.key, form)
            forms[input_spec.key] = form
        result = run_mapping(
            MappingRequest(
                route=route,
                inputs=inputs,
                input_forms=forms,
                region_start_kb=0,
                region_end_kb=4,
                resolution_x=1,
                resolution_t=5,
            )
        )
        assert result.method == route.method
        call = calls[-1]
        expected_source = {
            item.key for item in route.inputs if item.required and item.role == "source"
        }
        assert set(call["source"]) == expected_source
        for item in route.inputs:
            if item.required and item.role != "source":
                assert item.key in call["assumptions"]
        accepted_assumptions = (
            set(route.required_assumptions)
            | set(route.optional_assumptions)
            | {key for key, _value in route.defaults}
        )
        if "target_form" in accepted_assumptions:
            assert call["assumptions"]["target_form"] == route.output_dependence

    assert len(calls) == len(by_method)
    assert len(by_method) >= 79


def test_bundle_and_fit_metadata_are_exposed_with_member_forms_and_axes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route = mapping_route("timing_to_joint_spacetime_kinetic_fields")

    def fake_rmap(**_kwargs: Any) -> dict[str, Any]:
        return {
            "values": _fake_values(route),
            "residual": np.arange(4, dtype=float),
            "fit_mse": 0.125,
        }

    monkeypatch.setattr(dnascape, "rmap", fake_rmap)
    result = run_mapping(
        MappingRequest(
            route=route,
            inputs={"replication_timing": np.arange(14, dtype=float) + 5},
            input_forms={"replication_timing": "x"},
            region_start_kb=10,
            region_end_kb=14,
            resolution_x=1,
            resolution_t=5,
        )
    )

    assert result.primary_output == "firing_rate"
    assert result.output_forms["firing_rate"] == "xt"
    assert result.output_forms["replication_timing_fit"] == "x"
    assert result.output_forms["fit_history"] == "supplied"
    assert result.output_forms["residual"] == "x"
    assert result.fit_mse == pytest.approx(0.125)
    np.testing.assert_array_equal(result.time_axis, [0.0, 5.0, 10.0])
    np.testing.assert_array_equal(result.x_axis, [10.0, 11.0, 12.0, 13.0])
    comparison = result.comparisons["replication_timing_fit"]
    assert comparison.source_key == "replication_timing"
    assert comparison.form == "x"
    np.testing.assert_array_equal(comparison.observed, [15.0, 16.0, 17.0, 18.0])
    np.testing.assert_array_equal(comparison.fitted, [0.0, 1.0, 2.0, 3.0])


def test_progress_is_monotonic_and_cancel_is_checked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dnascape.utils import _progress_update

    checks = 0
    progress: list[tuple[float, str]] = []

    def cancel() -> None:
        nonlocal checks
        checks += 1

    def fake_rmap(**_kwargs: Any) -> dict[str, Any]:
        _progress_update(5, 10, 1.0, label="fit")
        _progress_update(1, 10, 1.0, label="fit")
        return {"values": np.arange(4, dtype=float)}

    monkeypatch.setattr(dnascape, "rmap", fake_rmap)
    run_mapping(
        _deterministic_request(),
        lambda value, message: progress.append((value, message)),
        cancel,
    )
    assert [value for value, _message in progress] == sorted(value for value, _message in progress)
    assert checks >= 4  # before, during both updates, and after rmap


def test_spatial_speed_fit_leaves_preparation_and_reports_optimizer_progress(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from dnascape.utils import etaferr

    route = mapping_route("timing_and_firing_to_spatial_fork_speed")
    progress: list[tuple[float, str]] = []

    def fake_rmap(**kwargs: Any) -> dict[str, Any]:
        # A slow first evaluation must already be described as mapping, and
        # the route's quiet Python default must not disable browser updates.
        assert progress[-1] == (0.04, "Mapping fork speed...")
        assert kwargs["assumptions"]["verbose"] is True
        etaferr(1, 20, mapping_service.time.time(), 0.25, label="rmap constrained kinetics")
        return {"values": np.ones(4), "fit_mse": 0.25}

    monkeypatch.setattr(dnascape, "rmap", fake_rmap)
    result = run_mapping(
        MappingRequest(
            route=route,
            inputs={"replication_timing": np.arange(10.0, 14.0), "firing_rate": 0.001},
            input_forms={"replication_timing": "x", "firing_rate": "scalar"},
            resolution_x=1.0,
        ),
        lambda value, message: progress.append((value, message)),
    )

    assert any("1/20" in message and "MSE" in message for _value, message in progress)
    assert [value for value, _message in progress] == sorted(value for value, _message in progress)
    assert "verbose" not in result.assumptions
    np.testing.assert_array_equal(result.outputs["fit_history"], [0.25])
    assert capsys.readouterr().out == ""


def test_progress_does_not_pass_verbose_to_routes_without_that_option(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    progress: list[tuple[float, str]] = []

    def fake_rmap(**kwargs: Any) -> dict[str, Any]:
        assert progress[-1][1] == "Mapping fork directionality..."
        assert "verbose" not in kwargs["assumptions"]
        return {"values": np.zeros(4)}

    monkeypatch.setattr(dnascape, "rmap", fake_rmap)
    run_mapping(
        MappingRequest(
            route=mapping_route("timing_to_directionality"),
            inputs={"replication_timing": np.arange(10.0, 14.0)},
            input_forms={"replication_timing": "x"},
            resolution_x=1.0,
        ),
        lambda value, message: progress.append((value, message)),
    )


def test_rmap_progress_prefix_is_presented_as_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dnascape.utils import _progress_update

    messages: list[str] = []

    def fake_rmap(**_kwargs: Any) -> dict[str, Any]:
        _progress_update(1, 3, 2.0, label="rmap", err=0.25)
        return {"values": np.arange(4, dtype=float)}

    monkeypatch.setattr(dnascape, "rmap", fake_rmap)
    result = run_mapping(
        _deterministic_request(),
        lambda _value, message: messages.append(message),
    )

    assert any(message.startswith("Mapping...") for message in messages)
    assert not any(message.startswith("Rmap...") for message in messages)
    assert any("MSE" in message for message in messages)
    np.testing.assert_array_equal(result.outputs["fit_history"], [0.25])
    assert result.output_forms["fit_history"] == "supplied"
    assert result.fit_mse == pytest.approx(0.25)


def test_final_fit_mse_becomes_exportable_history_when_route_has_no_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_rmap(**_kwargs: Any) -> dict[str, Any]:
        return {
            "values": np.arange(4, dtype=float),
            "fit_mse": 0.125,
        }

    monkeypatch.setattr(dnascape, "rmap", fake_rmap)
    result = run_mapping(_deterministic_request())

    np.testing.assert_array_equal(result.outputs["fit_history"], [0.125])
    assert result.output_forms["fit_history"] == "supplied"
    assert result.fit_mse == pytest.approx(0.125)


def test_structured_fit_history_is_normalized_to_mse_per_iteration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_rmap(**_kwargs: Any) -> dict[str, Any]:
        return {
            "values": np.arange(4, dtype=float),
            "fit_history": [
                {"iteration": 1, "phase": "space", "mse": 0.5},
                {"iteration": 2, "phase": "time", "mse": None},
                {"iteration": 3, "phase": "space", "mse": 0.125},
            ],
            "fit_mse": 0.125,
        }

    monkeypatch.setattr(dnascape, "rmap", fake_rmap)
    result = run_mapping(_deterministic_request())

    np.testing.assert_allclose(
        result.outputs["fit_history"],
        [0.5, np.nan, 0.125],
        equal_nan=True,
    )
    assert result.output_forms["fit_history"] == "supplied"


def test_oversized_input_rejects_before_float_copy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mapping_service, "MAX_INPUT_VALUES", 2)
    with pytest.raises(ValueError, match="interactive limits"):
        validate_mapping_request(
            _deterministic_request(
                inputs={"firing_rate": np.ones(3), "fork_speed": 1.4},
                input_forms={"firing_rate": "x", "fork_speed": "scalar"},
                region_start_kb=0,
                region_end_kb=3,
            )
        )


def test_mapping_accepts_fifty_thousand_simulations_but_not_more() -> None:
    route = mapping_route("firing_and_speed_to_timing_simulation")

    def request(simulations: int) -> MappingRequest:
        return MappingRequest(
            route=route,
            inputs={"firing_rate": 0.001, "fork_speed": 1.4},
            input_forms={"firing_rate": "scalar", "fork_speed": "scalar"},
            settings={"sim_number": simulations},
            region_start_kb=0,
            region_end_kb=4,
            resolution_x=1,
        )

    assert validate_mapping_request(request(50_000)).assumptions["sim_number"] == 50_000
    with pytest.raises(ValueError, match="at most 50000"):
        validate_mapping_request(request(50_001))


def test_mapping_fit_iterations_are_not_arbitrarily_capped() -> None:
    route = mapping_route("timing_and_speed_to_constant_firing_rate")
    prepared = validate_mapping_request(
        MappingRequest(
            route=route,
            inputs={
                "replication_timing": np.asarray([7.0, 8.0, 9.0, 8.0]),
                "fork_speed": 1.4,
            },
            input_forms={"replication_timing": "x", "fork_speed": "scalar"},
            settings={"maxiter": 50_000},
            region_start_kb=0,
            region_end_kb=4,
            resolution_x=1,
        )
    )

    assert prepared.assumptions["maxiter"] == 50_000


def test_iod_mapping_is_standard_output_with_bounded_app_retention() -> None:
    route = mapping_route("firing_and_speed_to_inter_origin_distance_simulation")
    prepared = validate_mapping_request(
        MappingRequest(
            route=route,
            inputs={"firing_rate": 0.001, "fork_speed": 1.4},
            input_forms={"firing_rate": "scalar", "fork_speed": "scalar"},
            settings={"sim_number": 10},
            region_start_kb=0,
            region_end_kb=4,
            resolution_x=1,
        )
    )

    assert (
        prepared.assumptions["retain_inter_origin_distances_limit"]
        == mapping_service.MAX_MAPPING_IOD_SAMPLES
    )


def test_iod_mapping_preserves_total_count_and_truncation_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route = mapping_route("firing_and_speed_to_inter_origin_distance_simulation")

    def fake_rmap(**_kwargs: Any) -> dict[str, Any]:
        return {
            "values": np.asarray([1.0, 2.0, 3.0]),
            "inter_origin_distance_count": 12,
            "inter_origin_distances_truncated": True,
        }

    monkeypatch.setattr(dnascape, "rmap", fake_rmap)
    result = run_mapping(
        MappingRequest(
            route=route,
            inputs={"firing_rate": 0.001, "fork_speed": 1.4},
            input_forms={"firing_rate": "scalar", "fork_speed": "scalar"},
            settings={"sim_number": 10},
            region_start_kb=0,
            region_end_kb=4,
            resolution_x=1,
        )
    )

    assert result.inter_origin_distance_count == 12
    assert result.inter_origin_distances_truncated is True
    np.testing.assert_array_equal(result.outputs[result.primary_output], [1.0, 2.0, 3.0])


def test_mixed_length_whole_region_uses_the_shortest_length_from_zero() -> None:
    route = mapping_route("timing_and_firing_to_spatial_fork_speed")
    timing = np.arange(30, dtype=float)
    local_firing = np.arange(10, dtype=float) + 100

    prepared = validate_mapping_request(
        MappingRequest(
            route=route,
            # Deliberately supply the shorter input first: mapping coordinates
            # must be chosen by length rather than dictionary order.
            inputs={"firing_rate": local_firing, "replication_timing": timing},
            input_forms={"firing_rate": "x", "replication_timing": "x"},
            region_start_kb=12,
            region_end_kb=28,
            resolution_x=1,
            whole_genome=True,
        )
    )

    np.testing.assert_array_equal(prepared.source["replication_timing"], timing[:10])
    np.testing.assert_array_equal(prepared.source["firing_rate"], local_firing)
    assert prepared.start_kb == 0
    assert prepared.spatial_size == 10


def test_mixed_length_inputs_keep_a_narrower_requested_window() -> None:
    route = mapping_route("timing_and_firing_to_spatial_fork_speed")
    timing = np.arange(30, dtype=float)
    local_firing = np.arange(10, dtype=float) + 100

    prepared = validate_mapping_request(
        MappingRequest(
            route=route,
            inputs={"replication_timing": timing, "firing_rate": local_firing},
            input_forms={"replication_timing": "x", "firing_rate": "x"},
            region_start_kb=12,
            region_end_kb=18,
            resolution_x=1,
        )
    )

    np.testing.assert_array_equal(prepared.source["replication_timing"], timing[12:18])
    np.testing.assert_array_equal(prepared.source["firing_rate"], local_firing[:6])
    assert prepared.start_kb == 12
    assert prepared.spatial_size == 6


def test_mixed_length_alignment_reaches_rmap_at_the_reference_location(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route = mapping_route("timing_and_firing_to_spatial_fork_speed")
    timing = np.arange(30, dtype=float)
    local_firing = np.arange(10, dtype=float) + 100
    captured: dict[str, Any] = {}

    def fake_rmap(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"values": np.full(10, 1.4)}

    monkeypatch.setattr(dnascape, "rmap", fake_rmap)
    result = run_mapping(
        MappingRequest(
            route=route,
            inputs={"replication_timing": timing, "firing_rate": local_firing},
            input_forms={"replication_timing": "x", "firing_rate": "x"},
            region_start_kb=12,
            region_end_kb=22,
            resolution_x=1,
        )
    )

    np.testing.assert_array_equal(captured["source"]["replication_timing"], timing[12:22])
    np.testing.assert_array_equal(captured["source"]["firing_rate"], local_firing)
    assert len(captured["source"]["replication_timing"]) == 10
    np.testing.assert_array_equal(result.x_axis, np.arange(12, 22, dtype=float))


def test_region_errors_name_the_reference_profile_and_entered_bounds() -> None:
    route = mapping_route("timing_and_firing_to_spatial_fork_speed")
    inputs = {"replication_timing": np.ones(30), "firing_rate": np.ones(10)}
    forms: dict[str, DataForm] = {"replication_timing": "x", "firing_rate": "x"}

    with pytest.raises(
        ValueError,
        match=r"Replication timing, the longest uploaded spatial input.*Start below 30 kb",
    ):
        validate_mapping_request(
            MappingRequest(
                route=route,
                inputs=inputs,
                input_forms=forms,
                region_start_kb=30,
                region_end_kb=35,
                resolution_x=1,
            )
        )

    with pytest.raises(
        ValueError,
        match=r"received Start 12 kb and End 10 kb",
    ):
        validate_mapping_request(
            MappingRequest(
                route=route,
                inputs=inputs,
                input_forms=forms,
                region_start_kb=12,
                region_end_kb=10,
                resolution_x=1,
            )
        )


def test_overlapping_mapping_runs_fail_fast() -> None:
    assert mapping_service._RUN_LOCK.acquire(blocking=False)
    try:
        with pytest.raises(RuntimeError, match="Another mapping operation"):
            run_mapping(_deterministic_request())
    finally:
        mapping_service._RUN_LOCK.release()


def test_unknown_setting_and_wrong_exact_form_fail_before_rmap() -> None:
    with pytest.raises(ValueError, match="does not accept setting"):
        validate_mapping_request(_deterministic_request(settings={"not_an_option": 1}))
    with pytest.raises(ValueError, match="does not accept 't'"):
        validate_mapping_request(
            _deterministic_request(
                inputs={"firing_rate": np.ones(3), "fork_speed": 1.4},
                input_forms={"firing_rate": "t", "fork_speed": "scalar"},
            )
        )


def test_bedgraph_materialization_uses_rtrack_and_supports_bounded_gzip() -> None:
    content = b"chr1 0 1000 1.0\nchr1 1000 2000 3.0\n"
    options = {
        "chromosome": "chr1",
        "start": 0,
        "end": 2_000,
        "bin_size": 1_000,
        "value_column": "value",
    }
    plain = materialize_track_upload(_TrackUpload("signal.bedgraph", content), **options)
    compressed = materialize_track_upload(
        _TrackUpload("signal.bedgraph.gz", gzip.compress(content)),
        **options,
    )

    np.testing.assert_allclose(plain, [1.0, 3.0])
    np.testing.assert_array_equal(compressed, plain)
    assert plain.flags.writeable is False


def test_multiple_tracks_stack_in_time_order_and_validate_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _TrackUpload("first.bedgraph", b"chr1 0 1000 1\nchr1 1000 2000 2\n")
    second = _TrackUpload("second.bedgraph", b"chr1 0 1000 3\nchr1 1000 2000 4\n")
    values = materialize_many_tracks(
        (first, second),
        chromosome="chr1",
        start=0,
        end=2_000,
        bin_size=1_000,
        value_column="value",
    )
    np.testing.assert_array_equal(values, [[1.0, 3.0], [2.0, 4.0]])
    assert values.flags.writeable is False

    monkeypatch.setattr(mapping_service, "MAX_TRACK_FILES", 1)
    with pytest.raises(ValueError, match="At most 1 genomic tracks"):
        materialize_many_tracks((first, second), chromosome="chr1")


def test_result_mapping_protocol_is_available_for_ui_type_checking() -> None:
    result = run_mapping(_deterministic_request())
    assert isinstance(result.outputs, Mapping)
