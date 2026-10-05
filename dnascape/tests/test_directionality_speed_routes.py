"""Regression coverage for timing/directionality fork-speed reconstruction."""

from __future__ import annotations

import numpy as np
import pytest

from dnascape import rmap
from dnascape.mapping import map_directionality_timing, map_timing_directionality


def test_constant_speed_route_round_trips_timing_and_directionality() -> None:
    timing = 8.0 + np.cumsum(np.r_[0.0, np.linspace(-0.08, 0.11, 30)])
    directionality = map_timing_directionality(
        timing,
        fork_speed=1.7,
        resolution=5.0,
    )

    result = rmap(
        source={
            "replication_timing": timing,
            "fork_directionality": directionality,
        },
        target="fork_speed",
        assumptions={"target_form": "constant", "resolution": 5.0},
        return_info=True,
    )

    assert float(result["values"]) == pytest.approx(1.7)
    assert result["method"] == "timing_and_directionality_to_constant_fork_speed"
    assert result["assumptions"]["target_form"] == "constant"
    assert result["assumptions"]["T0"] is None
    np.testing.assert_allclose(result["replication_timing_fit"], timing, atol=1e-12)
    assert result["fit_mse"] == pytest.approx(0.0, abs=1e-24)


def test_spatial_speed_route_uses_regularized_site_reconstruction() -> None:
    count = 41
    axis = np.linspace(0.0, 1.0, count)
    timing = 6.0 + np.cumsum(np.linspace(0.04, 0.12, count))
    speed = 1.2 + 0.3 * np.sin(np.pi * axis) ** 2
    directionality = map_timing_directionality(
        timing,
        fork_speed=speed,
        resolution=5.0,
        T0=5.0,
    )

    result = rmap(
        source={
            "replication_timing": timing,
            "fork_directionality": directionality,
        },
        target="fork_speed",
        assumptions={
            "target_form": "space",
            "resolution": 5.0,
            "T0": 5.0,
            "speed_regularization": 0.05,
        },
        return_info=True,
    )

    np.testing.assert_allclose(result["values"], speed, atol=3e-4)
    np.testing.assert_allclose(result["replication_timing_fit"], timing, atol=4e-5)
    assert np.isfinite(result["fit_mse"])


def test_near_flat_gradients_do_not_create_spatial_speed_spikes() -> None:
    timing = np.asarray([8.0, 8.0 + 1e-12, 8.2, 8.2 + 1e-12, 8.4])
    directionality = map_timing_directionality(
        timing,
        fork_speed=1.4,
        resolution=1.0,
    )

    result = rmap(
        source={
            "replication_timing": timing,
            "fork_directionality": directionality,
        },
        target="fork_speed",
        assumptions={"target_form": "space", "resolution": 1.0},
        return_info=True,
    )

    np.testing.assert_allclose(result["values"], 1.4, atol=1e-8)
    assert np.isfinite(result["values"]).all()
    assert np.all(np.asarray(result["values"]) > 0.0)


def test_default_regularization_stabilizes_noisy_periodic_speed_reconstruction() -> None:
    count = 100
    axis = np.linspace(0.0, 4.0 * np.pi, count, endpoint=False)
    timing = 10.0 + 2.0 * np.sin(axis)
    expected_speed = 1.4 + 0.2 * np.sin(axis / 2.0)
    directionality = map_timing_directionality(
        timing,
        fork_speed=expected_speed,
        resolution=5.0,
        perQ=True,
    )
    noisy_directionality = np.clip(
        directionality + np.random.default_rng(0).normal(0.0, 0.005, count),
        -1.0,
        1.0,
    )

    result = rmap(
        source={
            "replication_timing": timing,
            "fork_directionality": noisy_directionality,
        },
        target="fork_speed",
        assumptions={"target_form": "space", "resolution": 5.0, "perQ": True},
        return_info=True,
    )

    inferred_speed = np.asarray(result["values"], dtype=float)
    assert np.isfinite(inferred_speed).all()
    assert np.all(inferred_speed > 0.0)
    assert float(np.sqrt(np.mean(np.square(inferred_speed - expected_speed)))) < 0.1
    assert result["assumptions"]["speed_regularization"] == pytest.approx(4.0)


@pytest.mark.parametrize("regularization", (0.0, -1.0, np.inf, np.nan))
def test_spatial_speed_regularization_must_be_positive(regularization: float) -> None:
    timing = np.asarray([8.0, 8.2, 8.5, 8.9])
    directionality = map_timing_directionality(timing, fork_speed=1.4)

    with pytest.raises(ValueError, match="speed_regularization must be finite and > 0"):
        rmap(
            source={
                "replication_timing": timing,
                "fork_directionality": directionality,
            },
            target="fork_speed",
            assumptions={
                "target_form": "space",
                "speed_regularization": regularization,
            },
        )


def test_flat_timing_fails_with_an_identifiability_message() -> None:
    with pytest.raises(ValueError, match="too flat to identify"):
        rmap(
            source={
                "replication_timing": np.ones(5),
                "fork_directionality": np.zeros(5),
            },
            target="fork_speed",
            assumptions={"target_form": "space"},
        )


@pytest.mark.parametrize(
    ("timing", "directionality", "message"),
    (
        (np.ones(4), np.ones(3), "same length"),
        (np.arange(4.0), np.asarray([0.0, 0.2, 1.2, 0.3]), "between -1 and 1"),
        (
            np.asarray([0.0, 1.0, 2.0, 3.0]),
            np.asarray([0.0, -0.2, -0.2, -0.2]),
            "positive fork speed",
        ),
    ),
)
def test_invalid_directionality_speed_inputs_fail_clearly(
    timing: np.ndarray,
    directionality: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        rmap(
            source={
                "replication_timing": timing,
                "fork_directionality": directionality,
            },
            target="fork_speed",
            assumptions={"target_form": "space"},
        )


def test_periodic_directionality_integration_preserves_exact_circular_gradients() -> None:
    count = 32
    phase = np.arange(count, dtype=float) * (2.0 * np.pi / count)
    timing = 10.0 + 2.0 * np.sin(phase) + 0.3 * np.cos(2.0 * phase)
    directionality = map_timing_directionality(
        timing,
        fork_speed=1.4,
        resolution=5.0,
        perQ=True,
    )

    reconstructed = map_directionality_timing(
        directionality,
        fork_speed=1.4,
        resolution=5.0,
        perQ=True,
    )
    reconstructed += float(np.mean(timing - reconstructed))
    np.testing.assert_allclose(reconstructed, timing, atol=1e-12)

    result = rmap(
        source={
            "replication_timing": timing,
            "fork_directionality": directionality,
        },
        target="fork_speed",
        assumptions={"target_form": "space", "resolution": 5.0, "perQ": True},
        return_info=True,
    )
    np.testing.assert_allclose(result["values"], 1.4, atol=1e-9)
    np.testing.assert_allclose(result["replication_timing_fit"], timing, atol=1e-9)


@pytest.mark.parametrize("periodic", (False, True))
@pytest.mark.parametrize("noise", (0.0, 0.02, 0.05))
def test_constant_speed_uses_timing_observations_without_gradient_noise_bias(periodic, noise):
    phase = np.arange(128) / 128
    clean = 10 + 2 * np.sin(2 * np.pi * phase)
    rfd = map_timing_directionality(clean, fork_speed=1.7, resolution=5, perQ=periodic)
    timing = clean + noise * np.mean(clean) * np.random.default_rng(1729).normal(size=clean.size)
    result = rmap(
        {"replication_timing": timing, "fork_directionality": rfd},
        "fork_speed",
        assumptions={"target_form": "constant", "resolution": 5, "perQ": periodic},
        return_info=True,
    )
    assert float(result["values"]) == pytest.approx(1.7, abs=0.06 if noise else 1e-12)
    assert abs(np.mean(result["residual"])) < 1e-12
    assert np.sqrt(np.mean((result["replication_timing_fit"] - clean) ** 2)) < 0.07


@pytest.mark.parametrize("periodic", (False, True))
@pytest.mark.parametrize("sharp", (False, True))
def test_spatial_speed_stays_bounded_and_recovers_noisy_smooth_and_sharp_timing(periodic, sharp):
    phase = np.arange(128) / 128
    clean = 10 + 4 * np.abs(phase - 0.5) if sharp else 10 + 2 * np.sin(2 * np.pi * phase)
    speed = 1.4 + (
        0.35 * np.where(phase < 0.5, -1, 1) if sharp else 0.35 * np.sin(4 * np.pi * phase)
    )
    rfd = map_timing_directionality(clean, fork_speed=speed, resolution=5, perQ=periodic)
    timing = clean + 0.05 * np.mean(clean) * np.random.default_rng(1729).normal(size=clean.size)
    result = rmap(
        {"replication_timing": timing, "fork_directionality": rfd},
        "fork_speed",
        assumptions={
            "target_form": "space",
            "resolution": 5,
            "perQ": periodic,
            "speed_bounds": (0.6, 2.5),
        },
        return_info=True,
    )
    assert np.all((result["values"] >= 0.6) & (result["values"] <= 2.5))
    assert np.sqrt(np.mean((result["replication_timing_fit"] - clean) ** 2)) / np.std(clean) < 0.22


@pytest.mark.parametrize(("periodic", "reference"), ((False, None), (False, 9.0), (True, 9.0)))
def test_integrated_spatial_jacobian_and_adjoint_match_independent_differences(
    monkeypatch, periodic, reference
):
    import dnascape.directionality_inverse as inverse

    original = inverse.least_squares
    checked = []

    def verify(fun, initial, *, jac, **kwargs):
        rng = np.random.default_rng(99)
        perturbation = rng.normal(size=initial.size)
        operator = jac(initial)
        step = 1e-6
        finite_difference = (
            fun(initial + step * perturbation) - fun(initial - step * perturbation)
        ) / (2 * step)
        np.testing.assert_allclose(operator @ perturbation, finite_difference, atol=1e-8, rtol=1e-7)
        dual = rng.normal(size=operator.shape[0])
        assert np.dot(operator @ perturbation, dual) == pytest.approx(
            np.dot(perturbation, operator.rmatvec(dual)), abs=1e-10
        )
        checked.append(True)
        return original(fun, initial, jac=jac, **kwargs)

    monkeypatch.setattr(inverse, "least_squares", verify)
    phase = np.arange(13) / 13
    timing = 10 + np.sin(2 * np.pi * phase)
    rfd = map_timing_directionality(
        timing, fork_speed=1.4, T0=reference, resolution=5, perQ=periodic
    )
    result = rmap(
        {"replication_timing": timing, "fork_directionality": rfd},
        "fork_speed",
        assumptions={"target_form": "space", "resolution": 5, "perQ": periodic, "T0": reference},
        return_info=True,
    )
    assert checked
    np.testing.assert_allclose(result["values"], 1.4, atol=1e-8)
    np.testing.assert_allclose(result["replication_timing_fit"], timing, atol=1e-8)
