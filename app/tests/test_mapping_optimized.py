"""Mapping integration for constrained kinetics and partially identified inverses."""

import numpy as np
import pytest

from app.mapping_catalog import mapping_route
from app.mapping_service import MappingRequest, run_mapping
from app.pages.mapping import _setting_value


def test_inverse_keeps_unidentifiable_mask_instead_of_fabricating_values():
    result = run_mapping(MappingRequest(
        route=mapping_route("initiation_events_and_fraction_to_initiation_rate_local_inverse"),
        inputs={"initiation_events_xt": np.zeros((4, 5)), "replicated_fraction_xt": np.ones((4, 5))},
        input_forms={"initiation_events_xt": "xt", "replicated_fraction_xt": "xt"},
    ))
    assert np.isnan(result.outputs["initiation_rate"]).all()
    assert result.diagnostics["unidentifiable_count"] == 20
    assert not np.any(result.diagnostics["valid_mask"])
    assert not result.diagnostics["valid_mask"].flags.writeable


def test_active_solver_defaults_are_displayed_without_a_global_iteration_cap():
    for dependence, method in (
        ("constant", "timing_and_firing_to_constant_fork_speed"),
        ("space", "timing_and_time_firing_to_spatial_fork_speed"),
    ):
        settings = {item.key for item in mapping_route(method).active_settings()}
        assert {"maxiter", "tolerance", "perQ"} <= settings
        assert not {"fit_step", "sim_number", "refine_seed"} & settings
        assert ("spatial_knots" in settings) == (dependence in {"space", "spacetime"})
        assert ("temporal_knots" in settings) == (dependence in {"time", "spacetime"})

    for dependence, method in (
        ("time", "timing_and_firing_to_time_fork_speed"),
        ("time", "timing_and_time_firing_to_time_fork_speed"),
        ("spacetime", "timing_and_firing_to_spacetime_fork_speed"),
        ("spacetime", "timing_and_general_firing_to_spacetime_fork_speed"),
    ):
        settings = {item.key: _setting_value(item) for item in mapping_route(method).active_settings()}
        assert settings["temporal_fit_step"] == .003
        assert settings["temporal_maxiter"] == 10
        assert settings["initial_fork_speed"] == 1.4
        assert "perQ" in settings
        assert not {
            "maxiter", "fit_step", "sim_number", "temporal_sim_number", "refine_seed",
            "tolerance", "regularization", "spatial_knots", "temporal_knots",
        } & settings.keys()
        if dependence == "spacetime":
            assert settings["space_fit_step"] == .4
            assert settings["space_maxiter"] == 10
            assert settings["spacetime_order"] == "crossed"

    spatial_speed = mapping_route("timing_and_firing_to_spatial_fork_speed")
    spatial_speed_defaults = {item.key: _setting_value(item) for item in spatial_speed.active_settings()}
    assert spatial_speed_defaults["maxiter"] == 10
    assert spatial_speed_defaults["fit_step"] == .4
    assert spatial_speed_defaults["smoothw"] == 100
    assert spatial_speed_defaults["initial_fork_speed"] == 1.4
    assert not {"time_grid", "resolution_time", "tolerance", "spatial_knots", "regularization"} & spatial_speed_defaults.keys()

    for method in ("timing_and_speed_to_initiation_rate", "timing_to_firing_constant_fork_speed"):
        spatial_firing = mapping_route(method)
        spatial_defaults = {item.key: _setting_value(item) for item in spatial_firing.active_settings()}
        assert spatial_defaults["maxiter"] == 10
        assert spatial_defaults["fit_step"] == 2.0
        assert spatial_firing.setting("space_maxiter").default is None
    for dependence in ("time", "spacetime"):
        firing = mapping_route(f"timing_and_speed_to_{dependence}_initiation_rate")
        free = {item.key: _setting_value(item) for item in firing.active_settings("free")}
        assert free["temporal_maxiter"] == 30
        if dependence == "spacetime":
            assert free["space_maxiter"] == 30
        assert "maxiter" not in free


@pytest.mark.parametrize("dependence", ("time", "spacetime"))
def test_weibull_fit_exposes_actual_timing_prediction_and_convergence(dependence):
    result = run_mapping(MappingRequest(
        route=mapping_route(f"timing_and_speed_to_{dependence}_initiation_rate"),
        inputs={"replication_timing": np.full(8, 10.), "fork_speed": 1.4},
        input_forms={"replication_timing": "x", "fork_speed": "scalar"},
        settings={"time_grid": np.arange(0., 61., 3.), "maxiter": 2, "refine_edge_kb": 0.},
        target_shape="weibull",
    ))
    assert result.outputs["replication_timing_fit"].shape == (8,)
    assert "replication_timing_fit" in result.comparisons
    assert result.fit_mse == pytest.approx(np.mean((result.outputs["replication_timing_fit"] - 10.)**2))
    assert isinstance(result.diagnostics["converged"], bool)
