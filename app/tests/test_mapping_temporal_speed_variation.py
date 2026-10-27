"""Spatial firing inputs can recover a varying temporal speed without a flat tail bias."""

import io
from unittest.mock import patch

import numpy as np
import pytest

from dnascape import fitting
from dnascape.mapping import map_fork_densities_xt, map_replicated_fraction_xt
from dnascape.workflows.mapping import MappingRequest, run_mapping
from dnascape.workflows.mapping_catalog import mapping_route
from dnascape.workflows.mapping_exports import build_mapping_raw_export
from dnascape.workflows.plotting import mapping_plot_options


def _request(timing, rates, *, periodic=False, axis=None):
    settings = {"perQ": periodic, "refine_edge_kb": 0., "speed_bounds": (.2, 6.),
                "temporal_maxiter": 10}
    if axis is not None:
        settings["time_grid"] = axis
    return MappingRequest(mapping_route("timing_and_firing_to_time_fork_speed"),
        {"replication_timing": timing, "initiation_rate": rates * 5.},
        {"replication_timing": "x", "initiation_rate": "x"}, settings=settings,
        resolution_x=5., resolution_t=5., whole_genome=True, initiation_rate_per_bin=True)


@pytest.mark.parametrize("automatic", [False, True])
def test_temporal_speed_variation_survives_bounds_preview_and_export(automatic):
    n = 64
    axis = np.arange(0., 181., 5.)
    rates = np.geomspace(.0001, .003, n)
    truth = np.minimum(.7 * np.exp(axis / 30.), 6.)[None, :]
    fraction = map_replicated_fraction_xt(rates, truth, resolution=5., time_grid=axis, n_bins=n)
    observed = np.trapezoid(1. - fraction, axis, axis=1)
    original = rates.copy()
    with patch("dnascape.fitting.rsim", side_effect=AssertionError("unexpected ensemble")):
        result = run_mapping(_request(observed, rates, axis=None if automatic else axis))
    values = result.outputs["fork_speed"]
    active = result.time_axis <= observed.max()
    assert np.ptp(values[active]) > 2.
    assert values[0] == pytest.approx(.7, rel=.02)
    assert np.all((values >= .2) & (values <= 6.))
    assert result.fit_mse < 1e-3
    np.testing.assert_array_equal(rates, original)
    if automatic:
        assert result.time_axis[-1] < 400.
    else:
        np.testing.assert_array_equal(result.time_axis, axis)
    options = mapping_plot_options(result, output_key="fork_speed", plot_type="Line",
        normalization="None", title="", x_label="Time (min)", y_label="Fork speed",
        color="#b03412", palette="viridis", histogram_bins=20, line_width=1.5,
        show_markers=False, show_colorbar=False, log_y=False, invert_y=False)
    data = np.asarray(options["series"][0]["data"], dtype=float)
    np.testing.assert_allclose(data[:, 0], result.time_axis)
    np.testing.assert_allclose(data[:, 1], values)
    artifact = build_mapping_raw_export(result, "fork_speed", "npz")
    assert artifact.filename == "rmap_fs_t.npz"
    with np.load(io.BytesIO(artifact.data), allow_pickle=False) as exported:
        np.testing.assert_array_equal(exported["values"].reshape(-1), values)


def test_periodic_temporal_speed_is_not_smoothed_to_an_almost_constant_fit():
    n = 64
    axis = np.arange(0., 181., 5.)
    rates = np.geomspace(.0001, .003, n)
    truth = np.minimum(.7 * np.exp(axis / 70.), 6.)[None, :]
    fraction = map_fork_densities_xt(rates, truth, resolution=5., time_grid=axis,
                                     n_bins=n, perQ=True)["replicated_fraction_xt"]
    observed = np.trapezoid(1. - fraction, axis, axis=1)
    result = run_mapping(_request(observed, rates, periodic=True, axis=axis))
    assert np.ptp(result.outputs["fork_speed"][axis <= observed.max()]) > .6
    assert result.fit_mse < .1


@pytest.mark.parametrize("periodic", [False, True])
def test_a_constant_temporal_speed_remains_constant_when_supported_by_the_data(periodic):
    n = 24
    axis = np.arange(0., 301., 5.)
    rates = np.geomspace(.001, .004, n)
    speed = np.full((1, axis.size), 1.4)
    if periodic:
        fraction = map_fork_densities_xt(rates, speed, resolution=5., time_grid=axis,
                                         n_bins=n, perQ=True)["replicated_fraction_xt"]
    else:
        fraction = map_replicated_fraction_xt(rates, speed, resolution=5., time_grid=axis, n_bins=n)
    observed = np.trapezoid(1. - fraction, axis, axis=1)
    result = run_mapping(_request(observed, rates, periodic=periodic, axis=axis))
    np.testing.assert_allclose(result.outputs["fork_speed"], 1.4, rtol=1e-8, atol=1e-8)


def test_automatic_time_grid_extends_until_the_survival_tail_is_negligible():
    n = 4
    axis = np.arange(0., 601., 5.)
    rates = np.full(n, .002)
    truth = np.full((1, axis.size), 1.4)
    fraction = map_replicated_fraction_xt(rates, truth, resolution=10., time_grid=axis, n_bins=n)
    observed = np.trapezoid(1. - fraction, axis, axis=1)
    result = fitting._fit_exponential_speed_shape(observed, rates, "time", resolution=10.,
        resolution_time=5., speed_bounds=(.2, 6.), refine_edge_kb=0., maxiter=0, refine_verbose=False)
    fitted_axis = result._rmap_info["time_grid"]
    assert fitted_axis[-1] > 4. * observed.max()
    assert fitted_axis[-1] < 1200.
    np.testing.assert_allclose(result._rmap_info["replication_timing_fit"], observed, atol=1e-7)


@pytest.mark.parametrize("periodic", [False, True])
def test_zero_firing_does_not_silently_return_the_initial_speed(periodic):
    with pytest.raises(ValueError, match="positive initiation rate"):
        run_mapping(_request(np.full(24, 20.), np.zeros(24), periodic=periodic))


def test_large_spatial_profile_keeps_origins_between_sampled_observation_sites():
    n = 3001
    axis = np.arange(0., 101., 5.)
    rates = np.zeros(n)
    rates[[1, 1000, 1999, 2998]] = 2.
    assert not np.any(rates[::3])  # The former fit grid discarded every origin.
    truth = np.minimum(np.exp(axis / 50.), 6.)[None, :]
    fraction = map_replicated_fraction_xt(rates, truth, resolution=.1, time_grid=axis, n_bins=n)
    observed = np.trapezoid(1. - fraction, axis, axis=1)
    result = run_mapping(MappingRequest(mapping_route("timing_and_firing_to_time_fork_speed"),
        {"replication_timing": observed, "initiation_rate": rates * .1},
        {"replication_timing": "x", "initiation_rate": "x"},
        settings={"time_grid": axis, "refine_edge_kb": 0., "speed_bounds": (.2, 6.), "temporal_maxiter": 10},
        resolution_x=.1, resolution_t=5., whole_genome=True, initiation_rate_per_bin=True))
    assert np.ptp(result.outputs["fork_speed"]) > 4.5
    assert result.outputs["fork_speed"][0] == pytest.approx(1., rel=.01)
    assert result.fit_mse < 1e-3
