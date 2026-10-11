"""Independent forward, derivative, and constrained inverse verification."""

from itertools import product

import numpy as np
import pytest

from dnascape.inverse import (
    _make_field,
    _mean_field_timing,
    fit_timing_fork_speed,
)
from dnascape.mapping import map_fork_densities_xt


@pytest.mark.parametrize("periodic", [False, True])
def test_mean_field_prediction_and_analytic_gradient_match_independent_forward(periodic):
    n = 9
    t = np.arange(0.0, 36.0, 3.0)
    firing = _make_field("spacetime", None, 0.009, (1e-6, 0.1), n, t, 3, 3, 2, 0, "firing_rate")
    speed = _make_field("space", None, 1.4, (0.2, 5), n, t, 3, 3, 2, firing.count, "fork_speed")
    params = np.concatenate([firing.initial, speed.initial]) + np.linspace(
        -0.07, 0.11, firing.count + speed.count
    )
    i, v = firing.evaluate(params), speed.evaluate(params)
    prediction, jacobian, completion = _mean_field_timing(
        i, v, t, 4.0, periodic, (firing, speed), params
    )
    forward = map_fork_densities_xt(i.T, v.T, resolution=4, time_grid=t, perQ=periodic)
    expected = np.trapezoid(1 - forward["replicated_fraction_xt"], x=t, axis=1)
    np.testing.assert_allclose(prediction, expected, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(completion, forward["replicated_fraction_xt"][:, -1])
    for column in (0, firing.count - 1, len(params) - 1):
        delta = np.zeros_like(params)
        delta[column] = 1e-6
        plus = _mean_field_timing(
            firing.evaluate(params + delta), speed.evaluate(params + delta), t, 4, periodic
        )[0]
        minus = _mean_field_timing(
            firing.evaluate(params - delta), speed.evaluate(params - delta), t, 4, periodic
        )[0]
        np.testing.assert_allclose(jacobian[:, column], (plus - minus) / 2e-6, rtol=2e-4, atol=2e-5)


@pytest.mark.parametrize("speed_form", ["constant", "space", "time", "spacetime"])
def test_periodic_speed_inverse_fits_all_output_forms_against_independent_model(speed_form):
    t = np.arange(0.0, 63.0, 3.0)
    forward = map_fork_densities_xt(0.008, 1.8, n_bins=12, resolution=4, time_grid=t, perQ=True)
    observed = np.trapezoid(1 - forward["replicated_fraction_xt"], x=t, axis=1)
    result = fit_timing_fork_speed(
        observed,
        firing_rate=0.008,
        speed_form=speed_form,
        resolution=4,
        time_grid=t,
        perQ=True,
        spatial_knots=3,
        temporal_knots=2,
        maxiter=16,
    )
    assert result["fit_mse"] < 0.005
    assert np.all(
        (np.asarray(result["fork_speed"]) >= 0.2) & (np.asarray(result["fork_speed"]) <= 6)
    )


def test_inverse_validates_positive_bounds_and_reports_truncated_time_horizon():
    with pytest.raises(ValueError, match="positive increasing"):
        fit_timing_fork_speed(np.full(8, 10.0), firing_rate=0.008, speed_bounds=(-1, 4))
    result = fit_timing_fork_speed(
        np.full(8, 10.0),
        firing_rate=1e-5,
        time_grid=np.arange(4.0),
        maxiter=2,
    )
    assert result["unreplicated_at_end"] > 0.9


@pytest.mark.parametrize(
    "function,expected_shape",
    [
        ("map_timing_firing_constant_speed", ()),
        ("map_timing_firing_constant_speed_simulated", ()),
        ("map_timing_firing_spatial_speed_simulated", (12,)),
        ("map_timing_firing_temporal_speed", (1, 21)),
        ("map_timing_firing_temporal_speed_simulated", (1, 21)),
        ("map_timing_firing_spacetime_speed", (12, 21)),
        ("map_timing_firing_spacetime_speed_simulated", (12, 21)),
    ],
)
def test_existing_speed_entry_points_execute_periodic_inverse(function, expected_shape):
    from dnascape import fitting

    t = np.arange(0.0, 63.0, 3.0)
    forward = map_fork_densities_xt(0.008, 1.8, n_bins=12, resolution=4, time_grid=t, perQ=True)
    observed = np.trapezoid(1 - forward["replicated_fraction_xt"], x=t, axis=1)
    result = getattr(fitting, function)(
        {"replication_timing": observed, "firing_rate": 0.008},
        resolution=4,
        time_grid=t,
        perQ=True,
        maxiter=16,
        verbose=False,
    )
    assert np.asarray(result).shape == expected_shape
    assert result._rmap_info["fit_mse"] < 0.01


def test_uniform_mean_field_timing_converges_to_closed_form_gaussian_survival():
    """An independent continuum limit checks the shared forward model itself."""
    rate, speed = 0.008, 1.8
    exact = np.sqrt(np.pi) / (2 * np.sqrt(rate * speed))
    errors = []
    for step in (0.4, 0.1):
        t = np.arange(0.0, 40.0 + step, step)
        prediction, _, _ = _mean_field_timing(
            np.full((len(t), 128), rate), np.full((len(t), 128), speed), t, 4.0, True
        )
        errors.append(abs(float(prediction.mean()) - exact))
    assert errors[1] < errors[0] / 2
    assert errors[1] / exact < 0.005


@pytest.mark.parametrize(
    "source_form,target_form", tuple(product(("constant", "space", "time", "spacetime"), repeat=2))
)
def test_public_rmap_speed_inverse_supports_every_source_and_target_form_periodically(
    source_form,
    target_form,
):
    from dnascape import rmap

    t = np.arange(0.0, 63.0, 3.0)
    n = 12
    source = {
        "constant": 0.008,
        "space": np.full(n, 0.008),
        "time": np.full((1, len(t)), 0.008),
        "spacetime": np.full((n, len(t)), 0.008),
    }[source_form]
    forward = map_fork_densities_xt(0.008, 1.8, n_bins=n, resolution=4, time_grid=t, perQ=True)
    timing = np.trapezoid(1 - forward["replicated_fraction_xt"], x=t, axis=1)
    result = rmap(
        {"replication_timing": timing, "firing_rate": source},
        target="fork_speed",
        assumptions={
            "target_form": target_form,
            "resolution": 4,
            "time_grid": t,
            "perQ": True,
            "maxiter": 16,
        },
        return_info=True,
    )
    assert result["fit_mse"] < 0.01
    expected = {"constant": (), "space": (n,), "time": (1, len(t)), "spacetime": (n, len(t))}
    assert np.asarray(result["values"]).shape == expected[target_form]
