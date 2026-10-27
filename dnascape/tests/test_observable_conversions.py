"""Numerical identities and validation for supplied-observable conversions."""

import itertools

import numpy as np
import pytest

from dnascape import rmap
from dnascape.conversions import (
    map_density_to_spatial,
    map_density_to_temporal,
    map_events_and_fraction_to_firing,
    map_fraction_and_density_to_speed,
    map_fraction_to_mean_fraction,
    map_fraction_to_timing,
    map_spatial_events_to_pdf,
)
from dnascape.mapping import map_fork_densities_xt, map_initiation_events_xt


def test_fraction_spatial_means_handle_irregular_positions_and_single_bin():
    x = np.array([2., 3., 7.])
    fraction = np.array([[0., .1, .2], [0., .3, .7], [0., .5, 1.]])
    np.testing.assert_allclose(map_fraction_to_mean_fraction(fraction), fraction.mean(axis=0))
    np.testing.assert_allclose(
        map_fraction_to_mean_fraction(fraction, x_array=x, spatial_average="trapezoid"),
        np.trapezoid(fraction, x, axis=0) / 5.,
    )
    np.testing.assert_allclose(
        rmap(source="replicated_fraction_xt", data=fraction[:1], target="replicated_fraction"),
        fraction[0],
    )


def test_survival_timing_integral_respects_offset_and_completion():
    time = np.array([20., 21., 24., 30.])
    fraction = np.tile((time - 20.) / 10., (3, 1))
    np.testing.assert_allclose(
        rmap(source="replicated_fraction_xt", data=fraction, target="replication_timing", time_grid=time),
        [25.] * 3,
    )
    with pytest.raises(ValueError, match="completion"):
        map_fraction_to_timing(fraction[:, :-1], time_grid=time[:-1])


def test_survival_tail_extrapolation_and_restricted_mean_are_explicit():
    time = np.linspace(0., 4., 401)
    survival = np.exp(-time / 2.)
    fraction = np.tile(1. - survival, (2, 1))
    restricted = map_fraction_to_timing(fraction, time_grid=time, tail_policy="truncate")
    extrapolated = map_fraction_to_timing(
        fraction, time_grid=time, tail_policy="exponential", tail_fit_points=8,
    )
    np.testing.assert_allclose(restricted, np.trapezoid(survival, time))
    np.testing.assert_allclose(extrapolated, 2., atol=5e-6)
    assert restricted._rmap_info["timing_is_restricted_mean"]
    assert np.all(extrapolated._rmap_info["extrapolated_tail"] > 0)


def test_initially_replicated_timing_requires_conditional_interpretation():
    fraction = np.array([[.5, .75, 1.]])
    with pytest.raises(ValueError, match="first fraction"):
        map_fraction_to_timing(fraction, time_grid=[3., 5., 7.])
    np.testing.assert_allclose(
        map_fraction_to_timing(fraction, time_grid=[3., 5., 7.], initial_policy="conditional"),
        [5.],
    )
    with pytest.raises(ValueError, match="undefined"):
        map_fraction_to_timing(np.ones((2, 3)), initial_policy="conditional")
    with pytest.raises(ValueError, match="decreasing positive"):
        map_fraction_to_timing(np.zeros((2, 3)), tail_policy="exponential")


@pytest.mark.parametrize("options", [
    {"time_grid": [0., 2., 1.]}, {"time_grid": [0., 0., 1.]},
    {"time_grid": [0., np.nan, 1.]}, {"resolution_time": 0},
    {"tail_policy": "guess"}, {"tail_tolerance": -1},
    {"tail_tolerance": 1}, {"initial_policy": "guess"},
])
def test_timing_rejects_invalid_axes_and_policies(options):
    with pytest.raises(ValueError):
        map_fraction_to_timing(np.array([[0., .5, 1.]]), **options)


def test_noisy_fraction_projection_is_opt_in():
    fraction = np.array([[0., .4, .3, 1.]])
    with pytest.raises(ValueError, match="decreases"):
        map_fraction_to_timing(fraction)
    projected = np.maximum.accumulate(fraction, axis=1)
    np.testing.assert_allclose(
        map_fraction_to_timing(fraction, monotonic_policy="project"),
        np.trapezoid(1. - projected, dx=5., axis=1),
    )


def test_initiation_inverse_recovers_hazard_and_masks_completed_dna():
    fraction = np.tile(np.linspace(0., 1., 7), (4, 1))
    firing = np.linspace(.01, .07, fraction.size).reshape(fraction.shape)
    events = map_initiation_events_xt(firing, fraction)
    result = rmap(
        source={"initiation_events_xt": events, "replicated_fraction_xt": fraction},
        target="initiation_rate", return_info=True,
    )
    np.testing.assert_allclose(result["values"][:, :-1], firing[:, :-1])
    assert np.isnan(result["values"][:, -1]).all()
    assert result["unidentifiable_count"] == 4
    np.testing.assert_array_equal(result["valid_mask"], fraction < 1.)
    assert np.all(events[:, -1] == 0)


@pytest.mark.parametrize("policy", ["mask", "zero", "raise"])
def test_inverse_denominator_policy_is_explicit(policy):
    source = {"initiation_events_xt": np.zeros((2, 3)), "replicated_fraction_xt": np.ones((2, 3))}
    if policy == "raise":
        with pytest.raises(ValueError, match="unidentifiable"):
            map_events_and_fraction_to_firing(source, invalid_policy=policy)
    else:
        result = map_events_and_fraction_to_firing(source, invalid_policy=policy)
        assert result._rmap_info["unidentifiable_count"] == 6
        assert not result._rmap_info["valid_mask"].any()
        assert np.isnan(result).all() if policy == "mask" else np.all(result == 0.)


def test_gradient_inverse_recovers_quadratic_flux_on_nonuniform_grid():
    time = np.array([0., .5, 1.5, 3.])
    scale = np.array([.8, 1.2])[:, None]
    fraction = scale * (.02 * time + .03 * time ** 2)
    velocity = np.array([1.2, 2.4])[:, None]
    density = scale * (.02 + .06 * time) / velocity
    result = rmap(
        source={"replicated_fraction_xt": fraction, "fork_density_xt": density},
        target="fork_speed", time_grid=time, return_info=True,
    )
    np.testing.assert_allclose(result["values"], np.broadcast_to(velocity, fraction.shape), atol=1e-13)
    assert result["unidentifiable_count"] == 0
    assert result["clipped_derivative_count"] == 0


@pytest.mark.parametrize("method", ["gradient", "forward"])
def test_speed_inverse_identifies_only_populated_fork_density(method):
    time = np.array([0., 1., 3., 5.])
    fraction = np.tile(time * .1, (2, 1))
    density = np.full_like(fraction, .05)
    density[0, 1] = 0.
    output = map_fraction_and_density_to_speed(
        {"replicated_fraction_xt": fraction, "fork_density_xt": density},
        time_grid=time, derivative_method=method,
    )
    assert np.isnan(output[0, 1])
    np.testing.assert_allclose(output[output._rmap_info["valid_mask"]], 2.)
    assert output._rmap_info["unidentifiable_count"] == 1


@pytest.mark.parametrize("policy", ["mask", "zero", "raise"])
def test_speed_inverse_masks_nearly_complete_dna_with_explicit_policies(policy):
    fraction = np.tile([0., .5, .998, .9995, 1.], (2, 1))
    density = np.ones_like(fraction)
    density[0, 1] = 0.
    source = {"replicated_fraction_xt": fraction, "fork_density_xt": density}
    if policy == "raise":
        with pytest.raises(ValueError, match="unidentifiable"):
            map_fraction_and_density_to_speed(source, invalid_policy=policy)
        return
    result = map_fraction_and_density_to_speed(source, invalid_policy=policy)
    expected_mask = np.asarray([[True, False, True, False, False], [True, True, True, False, False]])
    np.testing.assert_array_equal(result._rmap_info["valid_mask"], expected_mask)
    assert result._rmap_info["unidentifiable_count"] == 5
    assert result._rmap_info["unreplicated_floor"] == .001
    assert np.isfinite(result[expected_mask]).all()
    if policy == "mask":
        assert np.isnan(result[~expected_mask]).all()
    else:
        assert np.all(result[~expected_mask] == 0.)


def test_speed_inverse_zero_survival_floor_keeps_incomplete_tails_only():
    fraction = np.tile([0., .998, .9995, 1.], (2, 1))
    source = {"replicated_fraction_xt": fraction, "fork_density_xt": np.ones_like(fraction)}
    restricted = map_fraction_and_density_to_speed(source)
    permissive = map_fraction_and_density_to_speed(source, unreplicated_floor=0.)
    assert np.isnan(restricted[:, 2:]).all()
    assert np.isfinite(permissive[:, :-1]).all()
    assert np.isnan(permissive[:, -1]).all()
    np.testing.assert_allclose(restricted[:, :2], permissive[:, :2])


def test_speed_inverse_survival_floor_is_strict_at_its_boundary():
    fraction = np.tile([0., .75, .875, .9375], (2, 1))
    result = map_fraction_and_density_to_speed(
        {"replicated_fraction_xt": fraction, "fork_density_xt": np.ones_like(fraction)},
        unreplicated_floor=.125,
    )
    np.testing.assert_array_equal(result._rmap_info["valid_mask"], [[True, True, False, False]] * 2)


def test_speed_inverse_survival_floor_preserves_weak_incomplete_kinetics():
    # Every density is below 1e-8: raising the absolute density floor would
    # erase this exactly identifiable profile, even though it is incomplete.
    time = np.asarray([0., 1., 3., 7.])
    speed = np.asarray([1.2, 2.4])[:, None]
    fraction = np.broadcast_to(1e-9 * time, (2, time.size)).copy()
    density = np.broadcast_to(1e-9 / speed, fraction.shape).copy()
    result = map_fraction_and_density_to_speed(
        {"replicated_fraction_xt": fraction, "fork_density_xt": density}, time_grid=time,
    )
    np.testing.assert_allclose(result, np.broadcast_to(speed, fraction.shape), rtol=1e-13)
    assert result._rmap_info["valid_mask"].all()


def test_speed_inverse_smoothing_cannot_restore_completed_raw_samples():
    fraction = np.tile([0., .2, .5, .9999, 1.], (2, 1))
    result = map_fraction_and_density_to_speed(
        {"replicated_fraction_xt": fraction, "fork_density_xt": np.ones_like(fraction)},
        temporal_smoothw=3,
    )
    assert np.isfinite(result[:, :3]).all()
    assert np.isnan(result[:, 3:]).all()
    np.testing.assert_array_equal(fraction[0], [0., .2, .5, .9999, 1.])


def test_speed_inverse_reports_projection_and_boundary_derivative_clipping():
    fraction = np.array([[0., .2, .1, .8], [0., .8, .9, 1.]])
    source = {"replicated_fraction_xt": fraction, "fork_density_xt": np.ones_like(fraction)}
    with pytest.raises(ValueError, match="decreases"):
        map_fraction_and_density_to_speed(source)
    output = map_fraction_and_density_to_speed(source, monotonic_policy="project", temporal_smoothw=3)
    assert np.all(output[np.isfinite(output)] >= 0)
    assert np.isnan(output[1, -1])
    assert output._rmap_info["monotonic_policy"] == "project"
    assert output._rmap_info["temporal_smoothw"] == 3
    np.testing.assert_array_equal(source["replicated_fraction_xt"], fraction)


@pytest.mark.parametrize("options", [
    {"density_floor": -1}, {"temporal_smoothw": 0}, {"temporal_smoothw": 1.5},
    {"temporal_smoothw": 10}, {"monotonic_policy": "guess"},
    {"derivative_method": "guess"}, {"invalid_policy": "guess"},
    {"time_grid": [0., 0., 2.]},
    {"unreplicated_floor": -1}, {"unreplicated_floor": 1},
    {"unreplicated_floor": 1.1}, {"unreplicated_floor": np.nan},
    {"unreplicated_floor": np.inf},
])
def test_speed_inverse_rejects_invalid_numerical_controls(options):
    source = {"replicated_fraction_xt": np.tile([0., .3, 1.], (2, 1)), "fork_density_xt": np.ones((2, 3))}
    with pytest.raises(ValueError):
        map_fraction_and_density_to_speed(source, **options)


@pytest.mark.parametrize("bad", [np.ones(3), np.empty((0, 3)), np.array([[0., np.nan]]), np.array([[0., -.1]]), np.array([[0., 1.1]])])
def test_fraction_validation_rejects_nonfields_missing_and_nonphysical_values(bad):
    with pytest.raises(ValueError):
        map_fraction_to_mean_fraction(bad)


@pytest.mark.parametrize("member,dependence,source_kind", list(itertools.product(
    ("right_fork_density_xt", "left_fork_density_xt"),
    ("time", "space"), ("field", "bundle", "forward"),
)))
def test_density_summaries_match_field_integrals(member, dependence, source_kind):
    time = np.array([0., 1., 3., 6., 10.])
    source = {"initiation_rate": np.linspace(.01, .04, 6), "fork_speed": 1.4}
    bundle = map_fork_densities_xt(**source, time_grid=time)
    field = bundle[member]
    payload = ({member: field} if source_kind == "field" else
               {"fork_densities_xt": {**bundle, "time_grid": time}} if source_kind == "bundle" else source)
    output = rmap(
        source=payload, target=member.removesuffix("_xt"), target_form=dependence,
        time_grid=time,
    )
    expected = (field.mean(axis=0, keepdims=True) if dependence == "time"
                else np.trapezoid(field, time, axis=1))
    np.testing.assert_allclose(output, expected)
    assert output.shape == expected.shape


def test_bundle_extraction_and_summaries_preserve_original_arrays():
    right = np.arange(12., dtype=float).reshape(3, 4)
    bundle = {"right_fork_density_xt": right, "left_fork_density_xt": right / 2., "replicated_fraction_xt": np.zeros_like(right)}
    extracted = rmap(source={"fork_densities_xt": bundle}, target="right_fork_density_xt")
    extracted[:] = 0.
    assert bundle["right_fork_density_xt"][2, 3] == 11.
    with pytest.raises(ValueError, match="missing"):
        rmap(source={"fork_densities_xt": {}}, target="replicated_fraction_xt")


def test_density_single_time_sample_does_not_invent_an_integration_width():
    with pytest.raises(ValueError, match="at least 2"):
        map_density_to_spatial(np.ones((4, 1)))
    np.testing.assert_allclose(map_density_to_temporal(np.ones((4, 1))), [[1.]])


@pytest.mark.parametrize("source,target", [("initiation_events", "initiation_pdf"), ("coalescence_events", "coalescence_pdf")])
def test_spatial_pdf_routes_are_normalized_on_irregular_positions(source, target):
    x = np.array([2., 3., 7., 10.])
    events = np.array([.1, .4, .3, .2])
    result = rmap(source={source: events}, target=target, x_array=x)
    assert np.trapezoid(result, x) == pytest.approx(1.)
    assert np.all(result >= 0)
    with pytest.raises(ValueError, match="positive"):
        map_spatial_events_to_pdf(np.zeros(4))
    np.testing.assert_allclose(map_spatial_events_to_pdf(np.array([2.]), resolution=3.), [1./3.])


def test_coalescence_bundle_and_split_density_pdf_routes_agree():
    time = np.array([0., 1., 3., 6., 10.])
    x = np.arange(6.) + 20.
    bundle = map_fork_densities_xt(np.linspace(.01, .04, 6), fork_speed=1.4, time_grid=time)
    bundled = rmap(
        source={"fork_densities_xt": {**bundle, "time_grid": time, "x_array": x}, "fork_speed": 1.4},
        target="coalescence_pdf",
    )
    split = rmap(
        source={key: bundle[key] for key in ("replicated_fraction_xt", "right_fork_density_xt", "left_fork_density_xt")} | {"fork_speed": 1.4},
        target="coalescence_pdf", time_grid=time, x_array=x,
    )
    np.testing.assert_allclose(bundled, split)
    assert np.trapezoid(split, x) == pytest.approx(1.)
