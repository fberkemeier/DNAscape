"""Physical region and per-bin units when uploads change spatial resolution."""

import io
from dataclasses import replace

import numpy as np
import pytest

from dnascape.mapping import map_firing_timing
from dnascape.workflows.mapping import MappingRequest, run_mapping, validate_mapping_request
from dnascape.workflows.mapping_catalog import mapping_route
from dnascape.workflows.simulation import (
    ConstantSimulationRequest,
    run_constant_simulation,
    validate_constant_simulation,
)
from dnascape.workflows.spatial_grid import resample_spatial


def mapping_request(**changes):
    request = MappingRequest(
        route=mapping_route("timing_and_speed_to_initiation_rate"),
        inputs={"replication_timing": np.arange(1000.), "fork_speed": 1.4},
        input_forms={"replication_timing": "x", "fork_speed": "scalar"},
        region_start_kb=100., region_end_kb=200., resolution_x=10.,
        source_resolution_x=1., initiation_rate_per_bin=True,
    )
    return replace(request, **changes)


def test_coarser_mapping_averages_the_selected_native_bins():
    prepared = validate_mapping_request(mapping_request())
    np.testing.assert_allclose(prepared.source["replication_timing"], np.arange(104.5, 200., 10.))
    assert prepared.spatial_size == 10
    assert prepared.start_kb == 100.
    assert prepared.resolution_x == 10.
    assert prepared.source["fork_speed"] == 1.4


@pytest.mark.parametrize("dx,count", [(10., 100), (.1, 10000), (1., 1000)])
def test_whole_region_never_stretches_with_resolution(dx, count):
    prepared = validate_mapping_request(mapping_request(whole_genome=True, resolution_x=dx))
    assert prepared.start_kb == 0.
    assert prepared.spatial_size == count
    assert prepared.spatial_size * prepared.resolution_x == 1000.


def test_finer_mapping_interpolates_instead_of_reinterpreting_coordinates():
    prepared = validate_mapping_request(mapping_request(region_start_kb=10., region_end_kb=20., resolution_x=.1))
    np.testing.assert_allclose(prepared.source["replication_timing"], np.arange(100) / 10. + 10.)
    assert prepared.spatial_size == 100
    assert prepared.start_kb == 10.


def test_non_aligned_region_uses_partial_bins_without_moving_start_or_end():
    values = np.arange(20.)
    actual = resample_spatial(values, source_resolution=1., start=2.5, end=17., resolution=6.)
    np.testing.assert_allclose(actual, [(2*.5 + sum(range(3, 8)) + 8*.5)/6,
                                      (8*.5 + sum(range(9, 14)) + 14*.5)/6,
                                      (14*.5 + 15 + 16)/2.5])
    prepared = validate_mapping_request(mapping_request(region_start_kb=103., region_end_kb=197.))
    assert prepared.start_kb == 103.
    assert prepared.spatial_size == 10
    np.testing.assert_allclose(prepared.source["replication_timing"][-1], 194.5)


def test_space_time_profiles_rebin_only_space_and_preserve_missing_values():
    values = np.arange(20.)[:, None] + np.array([[0., 100., 200.]])
    coarse = resample_spatial(values, source_resolution=1., start=0., end=20., resolution=5.)
    np.testing.assert_allclose(coarse, np.arange(2., 20., 5.)[:, None] + np.array([[0., 100., 200.]]))
    values[3, 1] = np.nan
    coarse = resample_spatial(values, source_resolution=1., start=0., end=20., resolution=5.)
    assert np.isnan(coarse[0, 1])
    assert np.isfinite(coarse[1:]).all()
    fine = resample_spatial(values, source_resolution=1., start=0., end=20., resolution=.5)
    assert np.isnan(fine[5:8, 1]).all()


def test_prepared_bundles_resample_intrinsic_coordinates_and_keep_the_clock():
    time = np.arange(3.)
    x = np.arange(20., 60., 2.)
    bundle = {"x_array": x, "time_grid": time, "replicated_fraction_xt": np.zeros((20, 3)),
              "right_fork_density_xt": np.arange(20.)[:, None] + np.ones((20, 3)),
              "left_fork_density_xt": np.ones((20, 3))}
    request = MappingRequest(
        route=mapping_route("fork_density_bundle_to_replicated_fraction_xt_extraction"),
        inputs={"fork_densities_xt": bundle}, input_forms={"fork_densities_xt": "bundle"},
        whole_genome=True, resolution_x=10., source_resolution_x=1.,
    )
    prepared = validate_mapping_request(request)
    result = next(iter(prepared.source.values()))
    np.testing.assert_array_equal(result["x_array"], [20., 30., 40., 50.])
    np.testing.assert_array_equal(result["time_grid"], time)
    np.testing.assert_allclose(result["right_fork_density_xt"][:, 0], [3., 8., 13., 18.])
    assert prepared.resolution_x == 10.


@pytest.mark.parametrize("dx,start,end", [(10., 100., 200.), (.1, 10., 20.)])
def test_real_mapping_uses_per_bin_firing_and_physical_fork_speed(dx, start, end):
    firing = np.linspace(.001, .002, 1000)
    rate = resample_spatial(firing, source_resolution=1., start=start, end=end, resolution=dx)
    request = MappingRequest(
        route=mapping_route("firing_and_speed_to_timing_deterministic"),
        inputs={"initiation_rate": firing, "fork_speed": 1.4},
        input_forms={"initiation_rate": "x", "fork_speed": "scalar"},
        region_start_kb=start, region_end_kb=end, resolution_x=dx,
        source_resolution_x=1., initiation_rate_per_bin=True,
    )
    result = run_mapping(request)
    expected = map_firing_timing(rate, fork_speed=1.4, resolution=dx)
    np.testing.assert_allclose(result.outputs["replication_timing"], expected)
    np.testing.assert_allclose(result.x_axis, start + np.arange(rate.size)*dx)
    np.testing.assert_allclose(result.outputs["input:initiation_rate"], rate)


@pytest.mark.parametrize("dx", [10., .1])
def test_simulation_matches_manually_prepared_bins_including_input_previews(dx):
    start, end = 10., 30.
    firing = np.linspace(.01, .1, 100)
    speed = np.linspace(1., 2., 100)
    request = ConstantSimulationRequest(
        initiation_rate=firing, fork_speed=speed, resolution_x=dx, resolution_t=1.,
        region_start_kb=start, region_end_kb=end, replicates=3, geometry="periodic",
        time_window=20., max_replication_time=100., source_resolution_x=1., seed=123,
    )
    expected_rate = resample_spatial(firing, source_resolution=1., start=start, end=end, resolution=dx)
    expected_speed = resample_spatial(speed, source_resolution=1., start=start, end=end, resolution=dx)
    manual = replace(request, initiation_rate=expected_rate, fork_speed=expected_speed, source_resolution_x=None)
    result = run_constant_simulation(request)
    expected = run_constant_simulation(manual)
    np.testing.assert_array_equal(result.replication_timing, expected.replication_timing)
    np.testing.assert_array_equal(result.outputs["input:initiation_rate"], expected_rate)
    np.testing.assert_array_equal(result.outputs["input:fork_speed"], expected_speed)
    np.testing.assert_allclose(result.position_kb, start + np.arange(expected_rate.size)*dx)
    assert result.position_kb[-1] < end


def test_simulation_time_profiles_and_constants_are_not_resampled():
    request = ConstantSimulationRequest(
        initiation_rate=np.array([[.1, .2, .3]]), fork_speed=1.4,
        resolution_x=10., resolution_t=1., region_start_kb=0., region_end_kb=100.,
        replicates=1, geometry="linear", time_window=10., max_replication_time=100.,
        source_resolution_x=1.,
    )
    checked = validate_constant_simulation(request)
    np.testing.assert_array_equal(checked.request.initiation_rate, request.initiation_rate)
    assert checked.request.fork_speed == 1.4
    assert checked.n_bins == 10


def test_refinement_limit_checked_before_allocating():
    with pytest.raises(ValueError, match="limit"):
        resample_spatial(np.ones(100), source_resolution=1., start=0., end=100., resolution=1e-9)


@pytest.mark.parametrize("dx", [10., .1])
@pytest.mark.parametrize("shape", ["constant", "space"])
def test_real_inverse_recovers_per_bin_rate_and_preserves_observed_timing(dx, shape):
    rate = .005
    timing = map_firing_timing(rate, fork_speed=1.4, resolution=dx, n_bins=20, perQ=True)
    native = np.full(round(20*dx), timing[0])
    request = mapping_request(
        route=mapping_route("timing_and_speed_to_constant_initiation_rate" if shape == "constant" else "timing_and_speed_to_initiation_rate"),
        inputs={"replication_timing": native, "fork_speed": 1.4},
        region_start_kb=0., region_end_kb=20*dx, resolution_x=dx,
        settings={"perQ": True, "maxiter": 300},
    )
    result = run_mapping(request)
    np.testing.assert_allclose(result.outputs["initiation_rate"], rate, rtol=3e-4)
    comparison = result.comparisons["replication_timing_fit"]
    np.testing.assert_allclose(comparison.observed, timing)
    np.testing.assert_allclose(comparison.fitted, timing, rtol=3e-4)
    assert result.x_axis.size == 20


def test_recipe_and_npz_export_retain_resampling_and_physical_coordinates(tmp_path):
    from dnascape.workflows.mapping_recipe import run_mapping_recipe
    from dnascape.workflows.mapping_exports import build_mapping_raw_export

    rates = np.linspace(.001, .002, 1000)
    np.savez(tmp_path / "rates.npz", values=rates)
    recipe = {
        "method": "firing_and_speed_to_timing_deterministic",
        "settings": {"resolution": 10.},
        "region": {"start_kb": 100., "end_kb": 200., "source_resolution_x": 1., "initiation_rate_per_bin": True},
        "inputs": {
            "initiation_rate": {"form": "x", "mode": "upload", "files": [{"path": "rates.npz", "array_key": "values"}]},
            "fork_speed": {"form": "scalar", "mode": "constant", "values": 1.4},
        },
    }
    result = run_mapping_recipe(recipe, base_path=tmp_path)[0].result
    expected = map_firing_timing(rates[100:200].reshape(10, 10).mean(axis=1), fork_speed=1.4, resolution=10.)
    np.testing.assert_allclose(result.outputs["replication_timing"], expected)
    artifact = build_mapping_raw_export(result, "replication_timing", "NPZ")
    with np.load(io.BytesIO(artifact.data)) as output:
        np.testing.assert_allclose(output["values"], expected)
        np.testing.assert_array_equal(output["position_kb"], np.arange(100., 200., 10.))


def test_bundle_recipe_without_resolution_control_keeps_the_native_grid(tmp_path):
    from dnascape.workflows.mapping_recipe import prepare_mapping_recipe, run_mapping_recipe

    axis = np.arange(20., 60., 2.)
    np.savez(tmp_path / "prepared.npz", x_array=axis, time_grid=np.arange(3.), replicated_fraction_xt=np.zeros((20, 3)),
             right_fork_density_xt=np.ones((20, 3)), left_fork_density_xt=np.ones((20, 3)))
    recipe = {
        "method": "fork_density_bundle_to_replicated_fraction_xt_extraction",
        "settings": {},
        "region": {"start_kb": 20., "end_kb": 60., "source_resolution_x": 1., "initiation_rate_per_bin": True},
        "inputs": {"fork_densities_xt": {"form": "bundle", "mode": "prepared", "prepared_path": "prepared.npz"}},
    }
    (_, request), = prepare_mapping_recipe(recipe, base_path=tmp_path)
    assert request.resolution_x == 2.
    result = run_mapping_recipe(recipe, base_path=tmp_path)[0].result
    np.testing.assert_array_equal(result.x_axis, axis)
    assert result.outputs["replicated_fraction_xt"].shape == (axis.size, 3)


@pytest.mark.parametrize("form", ["x", "t", "xt"])
@pytest.mark.parametrize("periodic", [False, True])
def test_deterministic_mapping_keeps_units_for_every_rate_dependence(form, periodic):
    from dnascape import rmap

    base = np.linspace(.01, .02, 100)
    native = base if form == "x" else np.linspace(.01, .02, 11)[None, :] if form == "t" else base[:, None] * np.ones((1, 11))
    request = MappingRequest(
        route=mapping_route("firing_and_speed_to_timing_deterministic"),
        inputs={"initiation_rate": native, "fork_speed": 1.4},
        input_forms={"initiation_rate": form, "fork_speed": "scalar"},
        settings={"perQ": periodic, "time_grid": np.arange(11.), "max_rep_time": 100.},
        region_start_kb=20., region_end_kb=60., resolution_x=10., resolution_t=1.,
        source_resolution_x=1., initiation_rate_per_bin=True,
    )
    result = run_mapping(request)
    rate = native if form == "t" else resample_spatial(native, source_resolution=1., start=20., end=60., resolution=10.)
    expected = rmap(source={"initiation_rate": rate / 10., "fork_speed": 1.4}, target="replication_timing",
                    method="firing_and_speed_to_timing_deterministic", resolution=10., resolution_time=1.,
                    n_bins=4, perQ=periodic, time_grid=np.arange(11.))
    np.testing.assert_array_equal(result.outputs["replication_timing"], expected)
    np.testing.assert_array_equal(result.x_axis, [20., 30., 40., 50.])
