"""Real app execution of the restored temporal and space-time speed fits."""

from itertools import product

import numpy as np
import pytest

from app.mapping_catalog import mapping_route
from app.mapping_service import MappingRequest, run_mapping
from dnascape.mapping import map_replicated_fraction_xt


@pytest.mark.parametrize("dependence,firing_form,periodic", tuple(product(
    ("time", "spacetime"), ("scalar", "x", "t", "xt"), (False, True),
)))
def test_restored_speed_routes_keep_selected_region_time_axis_and_firing_forms(
    monkeypatch, dependence, firing_form, periodic,
):
    if not periodic:
        def forbidden(*args, **kwargs):
            raise AssertionError("Restored linear fit must not call the general PDE inverse")

        monkeypatch.setattr("dnascape.inverse.map_timing_and_firing_to_speed", forbidden)
        monkeypatch.setattr("dnascape.inverse._mean_field_timing", forbidden)

    n = 16
    dx = 4.
    anchor = 20.
    grid = np.arange(0., 85., 5.)
    fraction = map_replicated_fraction_xt(
        .006, 1.7, n_bins=n, resolution=dx, time_grid=grid, perQ=periodic,
    )
    observed = np.trapezoid(1 - fraction, grid, axis=1) + anchor
    full_timing = np.full(64, 999.)
    full_timing[24:40] = observed
    firing = {
        "scalar": .006,
        "x": np.full(64, .006),
        "t": np.full(grid.size, .006),
        "xt": np.full((64, grid.size), .006),
    }[firing_form]
    if dependence == "time":
        method = ("timing_and_firing_to_time_fork_speed" if firing_form in {"scalar", "x"}
                  else "timing_and_time_firing_to_time_fork_speed")
    else:
        method = ("timing_and_firing_to_spacetime_fork_speed" if firing_form == "xt"
                  else "timing_and_general_firing_to_spacetime_fork_speed")
    options = {"time_grid": grid, "T0": anchor, "temporal_maxiter": 2,
               "refine_edge_kb": 0., "speed_bounds": (.2, 6.), "perQ": periodic,
               "refine_verbose": False}
    if dependence == "spacetime":
        options.update(space_maxiter=2, smoothw=None)
    messages = []
    result = run_mapping(MappingRequest(
        route=mapping_route(method),
        inputs={"replication_timing": full_timing, "initiation_rate": firing},
        input_forms={"replication_timing": "x", "initiation_rate": firing_form},
        settings=options, resolution_x=dx, resolution_t=5.,
        region_start_kb=96., region_end_kb=160.,
    ), report=lambda _, message: messages.append(message))
    speed = result.outputs["fork_speed"]
    assert speed.shape == ((grid.size,) if dependence == "time" else (n, grid.size))
    assert np.all(np.isfinite(speed))
    assert np.all((speed >= .2) & (speed <= 6.))
    np.testing.assert_array_equal(result.x_axis, 96. + dx * np.arange(n))
    np.testing.assert_array_equal(result.time_axis, grid)
    np.testing.assert_allclose(result.comparisons["replication_timing_fit"].observed, observed)
    np.testing.assert_allclose(
        result.outputs["residual"], result.outputs["replication_timing_fit"] - observed,
    )
    assert np.isfinite(result.fit_mse)
    assert any("MSE" in message for message in messages)
