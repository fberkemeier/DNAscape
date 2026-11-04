"""Large forward fractions retain the native grid, source halos and boundaries."""

import numpy as np
import pytest

from dnascape import _forward_fraction as storage
from dnascape.mapping import map_replicated_fraction_xt


@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("i_form", ["c", "x", "t", "xt"])
@pytest.mark.parametrize("v_form", ["c", "x", "t", "xt"])
def test_mapped_fraction_matches_dense_on_multiple_tiles(monkeypatch, periodic, i_form, v_form):
    monkeypatch.delenv("RENDER", raising=False)
    n, clock = 3073, np.array([0., .4, 1.2, 2.8, 4., 6., 8.])
    x = np.arange(n)

    def field(base, form):
        space = base * (1 + .2 * np.sin(x / 19))
        temporal = 1 + .02 * clock
        return {"c": base, "x": space, "t": (base * temporal)[None, :],
                "xt": space[:, None] * temporal[None, :]}[form]

    rate, speed = field(.005, i_form), field(1.2, v_form)
    options = dict(time_grid=clock, n_bins=n, perQ=periodic, neighbourhood_range=5.3)
    monkeypatch.setattr(storage, "DENSE_FRACTION_VALUES", 10**9)
    expected = map_replicated_fraction_xt(rate, speed, **options)
    monkeypatch.setattr(storage, "DENSE_FRACTION_VALUES", 1)
    monkeypatch.setattr(storage, "WORK_VALUES", 8192)
    monkeypatch.setattr(storage, "STATIC_CORE_POINTS", 1024)
    actual = map_replicated_fraction_xt(rate, speed, **options)
    assert storage.disk_backed(actual)
    np.testing.assert_allclose(actual, expected, atol=2e-11, rtol=2e-10)
    assert np.isfinite(actual).all()


@pytest.mark.parametrize("periodic", [False, True])
def test_leading_clock_uniform_convolution_and_periodic_join(monkeypatch, periodic):
    monkeypatch.delenv("RENDER", raising=False)
    n, clock = 4099, np.arange(3., 34., 3.)
    rate = np.full((n, len(clock)), .001)
    rate[[0, 1023, 2050, -1]] *= 15 * (1 + clock / 30)
    options = dict(time_grid=clock, perQ=periodic, neighbourhood_range=12.7)
    expected = map_replicated_fraction_xt(rate, 1.4, **options)
    monkeypatch.setattr(storage, "DENSE_FRACTION_VALUES", 1)
    monkeypatch.setattr(storage, "WORK_VALUES", 8192)
    actual = map_replicated_fraction_xt(rate, 1.4, **options)
    np.testing.assert_allclose(actual, expected, atol=2e-11, rtol=2e-10)


def test_native_workflow_accepts_large_mapped_fraction_without_output_copy(monkeypatch):
    from dnascape.workflows import mapping
    from dnascape.workflows.mapping_catalog import mapping_route

    monkeypatch.delenv("RENDER", raising=False)
    monkeypatch.setattr(storage, "DENSE_FRACTION_VALUES", 1)
    monkeypatch.setattr(mapping, "MAX_GRID_CELLS", 10)
    monkeypatch.setattr(mapping, "MAX_OUTPUT_VALUES", 10)
    monkeypatch.setattr(mapping, "MAX_OUTPUT_BYTES", 80)
    rate = np.linspace(.001, .002, 101)
    result = mapping.run_mapping(mapping.MappingRequest(
        mapping_route("firing_and_speed_to_replicated_fraction_xt_lightcone"),
        {"initiation_rate": rate, "fork_speed": 1.4},
        {"initiation_rate": "x", "fork_speed": "scalar"},
        whole_genome=True, settings={"max_time": 60.}, resolution_x=1., resolution_t=5.))
    field = result.outputs[result.primary_output]
    assert field.shape == (101, 13)
    assert storage.disk_backed(field)
    assert not field.flags.writeable
    np.testing.assert_array_equal(result.x_axis, np.arange(101))


def test_blocked_survival_integral_matches_dense_and_checks_completion(monkeypatch):
    from dnascape._forward_mapping import timing_from_fraction

    clock = np.array([0., .4, 1., 3., 8.])
    fraction = -np.expm1(-.1 * np.arange(1, 9)[:, None] * clock**2)
    expected = np.trapezoid(1 - fraction, clock, axis=1)
    monkeypatch.setattr(storage, "DENSE_FRACTION_VALUES", 1)
    monkeypatch.setattr(storage, "WORK_VALUES", 6)
    np.testing.assert_allclose(timing_from_fraction(clock, fraction, require_complete=False), expected)
    with pytest.raises(ValueError, match="unreplicated"):
        timing_from_fraction(clock, fraction)


def test_whole_profile_axis_limit_changes_only_for_bounded_local_forward(monkeypatch):
    from dnascape.workflows import mapping
    from dnascape.workflows.mapping_catalog import mapping_route

    route = mapping_route("firing_and_speed_to_replicated_fraction_xt_lightcone")
    monkeypatch.setattr(mapping, "MAX_SPATIAL_POINTS", 10)
    inputs = {"initiation_rate": np.ones(101), "fork_speed": 1.4}
    forms = {"initiation_rate": "x", "fork_speed": "scalar"}
    monkeypatch.delenv("RENDER", raising=False)
    assert mapping._validate_input_shapes(route, inputs, forms) == (101, 0)
    monkeypatch.setenv("RENDER", "true")
    with pytest.raises(ValueError, match="axis exceeds"):
        mapping._validate_input_shapes(route, inputs, forms)


@pytest.mark.parametrize("periodic", [False, True])
def test_general_transport_keeps_one_global_substep_plan(monkeypatch, periodic):
    monkeypatch.delenv("RENDER", raising=False)
    n, clock = 4099, np.array([0., 2., 4., 8.])
    x = np.arange(n)
    rate = (.004 * (1 + .2 * np.sin(x / 13)))[:, None] * (1 + clock / 10)
    velocity = np.full((n, len(clock)), 1.2)
    velocity[1300:1600, 1:] = np.array([9., 1., 4.])
    options = dict(time_grid=clock, perQ=periodic, neighbourhood_range=4.7)
    expected = map_replicated_fraction_xt(rate, velocity, **options)
    monkeypatch.setattr(storage, "DENSE_FRACTION_VALUES", 1)
    monkeypatch.setattr(storage, "WORK_VALUES", 8192)
    actual = map_replicated_fraction_xt(rate, velocity, **options)
    np.testing.assert_allclose(actual, expected, atol=3e-11, rtol=3e-9)


def test_extended_irregular_input_clock_retains_values_in_mapped_batches(monkeypatch):
    from dnascape._forward_mapping import fraction_field

    monkeypatch.delenv("RENDER", raising=False)
    native = np.array([0., .7, 2.3])
    spatial = 1 + .1 * np.sin(np.arange(67) / 5)
    rate = .03 * spatial[:, None] * np.array([1., 1.2, 1.1])
    options = dict(time_grid=native, resolution_time=1.3, perQ=True, _complete=True)
    clock, expected = fraction_field(rate, 1.4, **options)
    assert clock[-1] > native[-1]
    monkeypatch.setattr(storage, "DENSE_FRACTION_VALUES", 1)
    monkeypatch.setattr(storage, "WORK_VALUES", 64)
    actual_clock, actual = fraction_field(rate, 1.4, **options)
    np.testing.assert_array_equal(actual_clock, clock)
    np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=2e-10)
    assert storage.disk_backed(actual)


def test_hosted_forward_does_not_select_local_mapped_storage(monkeypatch):
    from dnascape import _resources

    monkeypatch.setenv("RENDER", "true")
    monkeypatch.setattr(_resources, "memory_status", lambda: None)
    monkeypatch.setattr(storage, "DENSE_FRACTION_VALUES", 1)
    actual = map_replicated_fraction_xt(np.full(17, .01), 1.4, time_grid=np.arange(5.))
    assert not storage.disk_backed(actual)


def test_mapped_result_views_count_their_full_storage_in_session_retention():
    from app.session_cache import estimate_retained_bytes
    from dnascape._kinetic_tiled import disk_array

    result = disk_array((37, 11))
    view = np.asarray(result)[2:4]
    assert estimate_retained_bytes(view) == result.nbytes


def test_large_prepared_xt_input_is_reused_locally_above_input_caps(monkeypatch):
    from dnascape import _resources
    from dnascape._kinetic_tiled import disk_array
    from dnascape.workflows import mapping
    from dnascape.workflows.mapping_catalog import mapping_route

    monkeypatch.delenv("RENDER", raising=False)
    rate = disk_array((101, 13), order="F")
    rate[:] = np.linspace(.001, .002, 101)[:, None] * (1 + np.arange(13)[None, :] / 20)
    rate.setflags(write=False)
    request = mapping.MappingRequest(
        mapping_route("firing_and_speed_to_replicated_fraction_xt_lightcone"),
        {"initiation_rate": rate, "fork_speed": 1.4},
        {"initiation_rate": "xt", "fork_speed": "scalar"},
        whole_genome=True, settings={"max_time": 60.}, resolution_x=1., resolution_t=5.)
    monkeypatch.setattr(mapping, "MAX_INPUT_VALUES", 10)
    monkeypatch.setattr(mapping, "MAX_INPUT_BYTES", 80)
    prepared = mapping.validate_mapping_request(request)
    assert np.shares_memory(prepared.source["initiation_rate"], rate)
    result = mapping.run_mapping(request)
    assert result.outputs[result.primary_output].shape == rate.shape
    monkeypatch.setenv("RENDER", "true")
    monkeypatch.setattr(_resources, "memory_status", lambda: None)
    with pytest.raises(ValueError, match="Mapped inputs contain"):
        mapping.validate_mapping_request(request)


def test_large_mutable_input_gets_an_independent_mapped_snapshot(monkeypatch):
    from dnascape.workflows.mapping import _numeric_array

    monkeypatch.setattr(storage, "DENSE_FRACTION_VALUES", 1)
    original = np.arange(30., dtype=float).reshape(6, 5)
    captured = _numeric_array(original, "Initiation rate", allow_large=True)
    assert storage.disk_backed(captured) and not captured.flags.writeable
    original[0, 0] = 99.
    assert captured[0, 0] == 0.
