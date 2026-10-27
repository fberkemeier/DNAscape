"""Compact inverse fields retain dense projection and interpolation semantics."""

from itertools import product

import numpy as np
import pytest

from dnascape.inverse import _compact_space_time_field, _make_field
from dnascape.mapping import _expand_space_time_field


def _source(kind, n):
    return {
        "constant": 1.4,
        "space": np.linspace(0.8, 2.0, n),
        "column": np.linspace(0.8, 2.0, n)[:, None],
        "time": np.array([[0.8, 1.3, 1.7, 2.0]]),
        "spacetime": np.linspace(0.8, 2.0, n * 4).reshape(n, 4),
    }[kind]


@pytest.mark.parametrize("kind", ["constant", "space", "column", "time", "spacetime"])
@pytest.mark.parametrize("resample", [False, True])
def test_compact_fixed_fields_match_expansion_with_time_interpolation(kind, resample):
    n = 9
    t = np.array([0.0, 1.0, 3.0, 8.0, 11.0]) if resample else np.arange(4) * 2.0
    source = _source(kind, n)
    expected = _expand_space_time_field(source, len(t), n, t, 2.0, "fork_speed", True)
    actual = _compact_space_time_field(source, len(t), n, t, 2.0, "fork_speed", True)
    np.testing.assert_array_equal(actual, expected)
    if kind in {"constant", "space", "column"}:
        assert actual.strides[0] == 0
    if kind in {"constant", "time"}:
        assert actual.strides[1] == 0


@pytest.mark.parametrize(
    "form,kind",
    tuple(product(
        ("constant", "space", "time", "spacetime"),
        ("constant", "space", "column", "time", "spacetime"),
    )),
)
def test_compact_initialization_and_evaluation_match_dense_projection(form, kind):
    n = 9
    t = np.arange(0.0, 12.0, 2.0)
    source = _source(kind, n)
    field = _make_field(form, None, source, (0.2, 6.0), n, t, 2.0, 3, 2, 0, "fork_speed")
    dense = _expand_space_time_field(source, len(t), n, t, 2.0, "fork_speed", True)
    expected = np.linalg.lstsq(field.bt, np.log(dense), rcond=None)[0]
    expected = np.linalg.lstsq(field.bx, expected.T, rcond=None)[0].T
    np.testing.assert_allclose(field.initial, expected.ravel(), rtol=1e-13, atol=1e-14)
    parameters = field.initial + np.linspace(-0.1, 0.1, field.count)
    dense_value = np.exp(
        field.bt @ parameters.reshape(field.bt.shape[1], field.bx.shape[1]) @ field.bx.T
    )
    np.testing.assert_allclose(field.evaluate(parameters), dense_value, rtol=1e-14)
    if form in {"constant", "space"}:
        assert field.evaluate(parameters).strides[0] == 0


def test_chromosome_static_initialization_never_expands_time_axis(monkeypatch):
    # This grid would need 2.4 GB for each materialized time-space field.
    # Fail before that allocation if initialization regresses to dense least
    # squares; verify storage is proportional to the spatial profile instead.
    def unexpected_lstsq(*args, **kwargs):
        raise AssertionError("Scalar initialization must not call dense least squares.")

    monkeypatch.setattr(np.linalg, "lstsq", unexpected_lstsq)
    n = 249_251
    t = np.arange(0.0, 2402.0, 2.0)
    firing = _make_field("space", 1e-5, 1e-5, (1e-9, 1), n, t, 2, 12, 4, 0, "initiation_rate")
    speed = _make_field("space", None, 1.4, (0.2, 6), n, t, 2, 12, 4, 0, "fork_speed")
    values = speed.evaluate(speed.initial)
    assert firing.fixed.strides == (0, 0)
    assert firing.fixed.base.nbytes == 8
    assert values.shape == (len(t), n)
    assert values.strides[0] == 0
    assert values.base.nbytes == n * 8
    np.testing.assert_allclose(values[0, [0, n // 2, -1]], 1.4)


@pytest.mark.parametrize("source", [np.nan, np.inf, -1.0, np.array([1.0, -2.0, 1.0])])
def test_compact_fields_reject_invalid_values(source):
    with pytest.raises(ValueError, match="finite"):
        _compact_space_time_field(source, 4, 3, np.arange(4.0), 1.0, "initiation_rate")


def test_compact_firing_allows_zero_but_speed_does_not():
    field = _compact_space_time_field(0.0, 4, 3, np.arange(4.0), 1.0, "initiation_rate")
    assert not np.any(field)
    with pytest.raises(ValueError, match="positive"):
        _compact_space_time_field(0.0, 4, 3, np.arange(4.0), 1.0, "fork_speed", True)
