"""Public route dispatch and settings for timing-to-speed fitters."""

from itertools import product

import numpy as np
import pytest

from dnascape.fitting import map_timing_firing_spatial_speed
from dnascape.inverse import map_timing_and_firing_to_speed
from dnascape.mapping import map_fork_densities_xt
from dnascape.routing import (
    map_timing_and_firing_to_general_speed,
    map_timing_and_firing_to_spatial_speed,
    map_timing_and_general_firing_to_spacetime_speed,
    map_timing_and_time_firing_to_temporal_speed,
    rmap,
)


@pytest.mark.parametrize(
    "firing_form,speed_form,periodic",
    [
        case for case in product(
            ("scalar", "x", "t", "xt"), ("constant", "space"), (False, True)
        )
        if not (case[0] in {"scalar", "x"} and case[1] == "space")
    ],
)
def test_mean_field_speed_routes_recover_forward_timing(
    firing_form, speed_form, periodic,
):
    n = 9
    grid = np.arange(0., 63., 3.)
    firing = {
        "scalar": .008,
        "x": np.full(n, .008),
        "t": np.full((1, len(grid)), .008),
        "xt": np.full((n, len(grid)), .008),
    }[firing_form]
    forward = map_fork_densities_xt(
        firing, 1.8, n_bins=n, resolution=4., time_grid=grid, perQ=periodic,
    )
    timing = np.trapezoid(1. - forward["replicated_fraction_xt"], grid, axis=1)
    result = rmap(
        source={"replication_timing": timing, "firing_rate": firing},
        target="fork_speed", target_form=speed_form, resolution=4.,
        time_grid=grid, perQ=periodic, spatial_knots=3, temporal_knots=2,
        maxiter=16, tolerance=1e-6,
    )
    expected_shape = {
        "constant": (), "space": (n,), "time": (1, len(grid)),
        "spacetime": (n, len(grid)),
    }[speed_form]
    assert np.asarray(result).shape == expected_shape
    assert np.all(np.isfinite(result))
    assert np.all((np.asarray(result) >= .2) & (np.asarray(result) <= 6.))
    info = result._rmap_info
    assert info["fit_mse"] < .005


@pytest.mark.parametrize("firing_form,speed_form,periodic", tuple(product(
    ("scalar", "x", "t", "xt"), ("time", "spacetime"), (False, True),
)))
def test_restored_speed_routes_preserve_shape_bounds_and_supplied_firing(
    firing_form, speed_form, periodic,
):
    n = 9
    grid = np.arange(0., 63., 3.)
    firing = {
        "scalar": .002,
        "x": np.linspace(.001, .003, n),
        "t": np.linspace(.001, .003, len(grid))[None, :],
        "xt": np.outer(np.linspace(.001, .003, n), np.linspace(.8, 1.2, len(grid))),
    }[firing_form]
    source = {"replication_timing": np.linspace(20., 35., n), "firing_rate": firing}
    before = np.asarray(firing).copy()
    result = rmap(
        source=source, target="fork_speed", target_form=speed_form,
        resolution=4., time_grid=grid, T0=5., perQ=periodic,
        speed_bounds=(.5, 2.), initial_fork_speed=1., maxiter=2,
        refine_edge_kb=0., verbose=False,
    )
    expected_shape = (1, len(grid)) if speed_form == "time" else (n, len(grid))
    assert np.asarray(result).shape == expected_shape
    assert np.all(np.isfinite(result))
    assert np.all((np.asarray(result) >= .5) & (np.asarray(result) <= 2.))
    np.testing.assert_array_equal(source["firing_rate"], before)
    assert np.isfinite(result._rmap_info["fit_mse"])


def test_speed_route_settings_match_consumed_inverse_controls_and_dimensions():
    specs = [
        spec for spec in rmap._specs
        if set(spec.source) == {"replication_timing", "firing_rate"}
        and spec.target == "fork_speed"
    ]
    assert len(specs) == 8
    for spec in specs:
        settings = set(spec.required_assumptions + spec.optional_assumptions)
        if spec.method == "timing_and_firing_to_spatial_fork_speed":
            assert spec.func is map_timing_and_firing_to_spatial_speed
            assert {"fit_step", "smoothw", "neighbourhood_range", "T0", "refine_edge_kb"} <= settings
            assert not {"time_grid", "resolution_time", "spatial_knots", "regularization", "tolerance"} & settings
            continue
        form = dict(spec.assumption_values)["target_form"][0]
        if form in {"time", "spacetime"}:
            expected_functions = {
                "timing_and_firing_to_time_fork_speed": map_timing_and_firing_to_general_speed,
                "timing_and_time_firing_to_time_fork_speed": map_timing_and_time_firing_to_temporal_speed,
                "timing_and_firing_to_spacetime_fork_speed": map_timing_and_firing_to_general_speed,
                "timing_and_general_firing_to_spacetime_fork_speed": map_timing_and_general_firing_to_spacetime_speed,
            }
            assert spec.func is expected_functions[spec.method]
            assert {"temporal_fit_step", "temporal_maxiter", "speed_bounds", "T0"} <= settings
            assert not settings & {
                "tolerance", "regularization", "spatial_knots", "temporal_knots",
                "sim_number", "temporal_sim_number", "spacetime_sim_number",
                "spacetime_fit_step", "spacetime_space_fit_step", "spacetime_time_fit_step",
                "spacetime_update", "spacetime_kernel_t", "temporal_smoothw", "refine_seed",
            }
            defaults = dict(spec.default_assumptions)
            assert defaults["temporal_fit_step"] == .003
            assert defaults["temporal_maxiter"] == 10
            if form == "spacetime":
                assert {"space_fit_step", "space_maxiter", "smoothw", "spacetime_order"} <= settings
                assert defaults["space_fit_step"] == .4
                assert defaults["space_maxiter"] == 10
                assert defaults["spacetime_order"] == "crossed"
                assert dict(spec.assumption_values)["spacetime_order"] == (
                    "crossed", "space_then_time", "time_then_space",
                )
            else:
                assert not {"space_fit_step", "space_maxiter", "spacetime_order", "smoothw"} & settings
            continue
        assert spec.func is map_timing_and_firing_to_speed
        assert not settings & {
            "fit_step", "smoothw", "sim_number", "space_maxiter",
            "temporal_sim_number", "spacetime_order", "temporal_smoothw",
        }
        assert ("spatial_knots" in settings) == (form in {"space", "spacetime"})
        assert ("temporal_knots" in settings) == (form in {"time", "spacetime"})
        assert ("regularization" in settings) == (form != "constant")


@pytest.mark.parametrize("resolution", [.5, 4.])
@pytest.mark.parametrize("spatial_firing", [False, True])
def test_static_spatial_route_adapts_public_firing_density_without_mutating_source(resolution, spatial_firing):
    firing = np.linspace(.001, .003, 12) if spatial_firing else .002
    source = {"replication_timing": np.linspace(15., 30., 12), "firing_rate": firing}
    before = np.asarray(firing).copy()
    options = dict(resolution=resolution, maxiter=2, smoothw=None, verbose=False)
    actual = rmap(source=source, target="fork_speed", target_form="space", **options)
    expected = map_timing_firing_spatial_speed(
        {**source, "firing_rate": np.asarray(firing) * resolution}, **options,
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-13)
    np.testing.assert_allclose(actual._rmap_info["replication_timing_fit"], expected._rmap_info["replication_timing_fit"])
    np.testing.assert_array_equal(source["firing_rate"], before)


@pytest.mark.parametrize("anchor", [0., 20.])
def test_automatic_spatial_speed_initialization_uses_timing_only_for_bound_dominated_data(anchor):
    relative = np.array([10., 20., 100., 120., 140., 160., 180., 200.])
    options = dict(
        target_form="space", time_grid=[0., 2., 5.],
        spatial_knots=8, maxiter=0, T0=anchor,
    )
    source = {"replication_timing": relative + anchor, "firing_rate": .001}
    result = map_timing_and_firing_to_speed(source=source, initial_fork_speed=None, **options)
    expected = np.clip(np.pi / (4 * .001 * relative**2), .2, 6.)
    np.testing.assert_allclose(result, expected, rtol=1e-13)
    assert result._rmap_info["optimizer_evaluations"] == 1
    explicit = map_timing_and_firing_to_speed(source=source, initial_fork_speed=1.4, **options)
    np.testing.assert_allclose(explicit, 1.4, rtol=1e-13)


@pytest.mark.parametrize("firing", [.001, np.full(8, .001), 0.])
def test_automatic_spatial_speed_initialization_retains_established_start_when_appropriate(firing):
    # The scalar case implies a plausible local speed inside the bounds;
    # supplied spatial firing and zero firing also retain their prior start.
    result = map_timing_and_firing_to_speed(
        source={"replication_timing": np.full(8, 20.), "firing_rate": firing},
        target_form="space", time_grid=[0., 2., 5.], initial_fork_speed=None,
        spatial_knots=8, maxiter=0,
    )
    np.testing.assert_allclose(result, 1.4, rtol=1e-13)


def test_automatic_spatial_speed_avoids_high_cfl_seed_for_partially_bound_limited_tracks():
    timing = np.array([100., 120., 150., 200., 220., 250., 300., 500.])
    result = map_timing_and_firing_to_speed(
        source={"replication_timing": timing, "firing_rate": 1e-6},
        target_form="space", time_grid=[0., 2., 5.], initial_fork_speed=None,
        spatial_knots=8, maxiter=0,
    )
    np.testing.assert_allclose(result, 1.4, rtol=1e-13)
    result = map_timing_and_firing_to_speed(
        source={"replication_timing": timing[:7], "firing_rate": 1e-6},
        target_form="space", time_grid=[0., 2., 5.], initial_fork_speed=None,
        spatial_knots=7, maxiter=0,
    )
    np.testing.assert_allclose(result, 6., rtol=1e-13)
