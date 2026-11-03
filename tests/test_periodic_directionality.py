"""A circular model must not depend on the position of its stored seam."""

import asyncio
from dataclasses import replace

import numpy as np
import pytest

from dnascape import rmap
from dnascape.mapping import map_directionality_timing, map_timing_directionality
from dnascape.simulation import rsim, simulate_replication
from dnascape.utils import smoothf
from dnascape.workflows.mapping import MappingRequest, run_mapping
from dnascape.workflows.mapping_catalog import mapping_route, mapping_route_families
from dnascape.workflows.plotting import mapping_plot_options
from dnascape.workflows.simulation import ConstantSimulationResult


PLOT = dict(
    title="", x_label="Position (kb)", y_label="Directionality", color="#b03412",
    palette="viridis", histogram_bins=10, line_width=1.8, show_markers=False,
    show_colorbar=True, log_y=False, invert_y=False, normalization="None",
)


@pytest.mark.parametrize("count", [31, 32])
@pytest.mark.parametrize("window", [2, 9, 32, 100])
def test_circular_smoothing_matches_wrapped_neighbours_and_preserves_mean(count, window):
    values = np.random.default_rng(12).normal(size=count)
    width = min(window, count)
    expected = np.mean([np.roll(values, k - width // 2) for k in range(width)], axis=0)
    actual = smoothf(values, window, perQ=True)
    np.testing.assert_allclose(actual, expected, atol=1e-15)
    assert actual.mean() == pytest.approx(values.mean())
    np.testing.assert_allclose(smoothf(np.ones(count), window, perQ=True), 1.)
    np.testing.assert_allclose(
        smoothf(values, window),
        np.convolve(values, np.ones(width) / width)[width // 2:width // 2 + count],
        atol=1e-15,
    )


@pytest.mark.parametrize("method", [
    "timing_and_speed_to_directionality", "timing_and_spatial_speed_to_directionality",
    "firing_and_speed_to_directionality_deterministic",
])
@pytest.mark.parametrize("count", [31, 32])
def test_both_mapping_routes_rotate_with_the_ring_even_when_smoothed(method, count):
    phase = 2 * np.pi * np.arange(count) / count
    first_key = "replication_timing" if method.startswith("timing") else "initiation_rate"
    first = 10 + np.sin(phase) if first_key == "replication_timing" else .02 * (1 + .6 * np.sin(phase))
    speed = 1.4 + .2 * np.cos(phase) if "spatial" in method else 1.4
    options = dict(method=method, assumptions={"resolution": 2., "perQ": True, "smoothw": 9})
    reference = rmap({first_key: first, "fork_speed": speed}, "fork_directionality", **options)
    shifted = rmap(
        {first_key: np.roll(first, 7), "fork_speed": np.roll(speed, 7) if np.ndim(speed) else speed},
        "fork_directionality", **options,
    )
    np.testing.assert_allclose(shifted, np.roll(reference, 7), atol=1e-12)


def test_temporal_speed_directionality_wraps_its_smoothing_and_spatial_gradient():
    count = 24
    phase = 2 * np.pi * np.arange(count) / count
    clock = np.arange(81.)
    firing = .02 * (1 + .6 * np.sin(phase))
    speed = (1.4 + .2 * np.cos(phase))[:, None] * (1 + .1 * clock / clock[-1])
    options = dict(method="firing_and_speed_to_directionality_deterministic",
                   assumptions={"resolution": 2., "perQ": True, "smoothw": 9, "time_grid": clock})
    reference = rmap({"initiation_rate": firing, "fork_speed": speed}, "fork_directionality", **options)
    shifted = rmap({"initiation_rate": np.roll(firing, 7), "fork_speed": np.roll(speed, 7, axis=0)},
                   "fork_directionality", **options)
    np.testing.assert_allclose(shifted, np.roll(reference, 7), atol=1e-12)


def test_periodic_directionality_integral_uses_the_same_wrapped_smoothing():
    phase = 2 * np.pi * np.arange(32) / 32
    timing = 10 + np.sin(phase)
    raw = map_timing_directionality(timing, fork_speed=1.4, perQ=True)
    recovered = map_directionality_timing(raw, fork_speed=1.4, perQ=True, smoothw=9)
    expected = smoothf(timing, 9, perQ=True)
    np.testing.assert_allclose(recovered - recovered.mean(), expected - expected.mean(), atol=1e-13)


@pytest.mark.parametrize("count", [1, 2, 8, 9, 16])
def test_simulation_wraps_and_balances_exact_opposing_arrivals_without_a_preferred_seam(count):
    firing = np.r_[1., np.zeros(count - 1)]

    def run(shift):
        return simulate_replication(np.roll(firing, shift), fork_speed=1., n_bins=count,
                                   perQ=True, rng=np.random.default_rng(3), max_rep_time=None)

    reference = run(0)
    expected = np.sign(np.sin(2 * np.pi * np.arange(count) / count))
    expected[0] = 0
    if count % 2 == 0:
        expected[count // 2] = 0
    np.testing.assert_array_equal(reference[6], expected)
    for shift in (1, count // 2, count - 1):
        shifted = run(shift)
        np.testing.assert_allclose(shifted[3], np.roll(reference[3], shift))
        np.testing.assert_array_equal(shifted[6], np.roll(reference[6], shift))


def test_simulation_tied_directional_densities_retain_total_passages(tmp_path):
    result = rsim(initiation_rate=np.r_[1., np.zeros(7)], fork_speed=1., perQ=True,
                  sim_number=3, seed=3, verbose=False, time_stats_xtQ=True,
                  time_grid=np.arange(0., 20., .1), max_rep_time=None, dens_memmap_dir=tmp_path)
    right, left = result["right_fork_density_xt"], result["left_fork_density_xt"]
    assert np.any(right[4] > 0)
    np.testing.assert_array_equal(right[4], left[4])
    np.testing.assert_array_equal(result["fork_density_xt"], right + left)
    assert result["fork_directionality"][4] == 0


@pytest.mark.parametrize("method,input_key", [
    ("timing_and_speed_to_directionality", "replication_timing"),
    ("firing_and_speed_to_directionality_deterministic", "initiation_rate"),
])
def test_mapping_periodic_plot_closes_at_true_region_end_without_changing_raw_bins(method, input_key):
    phase = 2 * np.pi * np.arange(32) / 32
    values = 10 + np.sin(phase) if input_key == "replication_timing" else .02 * (1 + .6 * np.sin(phase))
    result = run_mapping(MappingRequest(
        route=mapping_route(method), inputs={input_key: values, "fork_speed": 1.4},
        input_forms={input_key: "x", "fork_speed": "scalar"},
        settings={"perQ": True}, resolution_x=2.,
    ))
    raw = result.outputs["fork_directionality"].copy()
    plot = mapping_plot_options(result, output_key="fork_directionality", plot_type="Line", **PLOT)
    points = np.asarray(plot["series"][0]["data"])
    assert points.shape == (33, 2)
    assert points[-1, 0] == 64
    assert points[-1, 1] == raw[0]
    np.testing.assert_array_equal(points[:-1, 1], raw)
    np.testing.assert_array_equal(result.outputs["fork_directionality"], raw)
    linear = replace(result, assumptions={"perQ": False})
    assert len(mapping_plot_options(linear, output_key="fork_directionality", plot_type="Line", **PLOT)["series"][0]["data"]) == 32


@pytest.mark.parametrize("geometry", ["linear", "periodic"])
@pytest.mark.parametrize("view", ["Default", "Histogram"])
def test_simulation_periodic_profile_closes_only_the_line_preview(geometry, view):
    from app.pages.simulation import _result_plot_options

    raw = np.array([.2, .5, -.4])
    result = ConstantSimulationResult(
        position_kb=np.array([40., 42., 44.]), replication_timing=np.ones(3),
        elapsed_seconds=0., replicates=1, geometry=geometry, time_grid=np.array([]),
        outputs={"fork_directionality": raw},
    )
    plot = _result_plot_options(
        result, output_label="Fork directionality", title="", series_label="",
        x_axis_label="", y_axis_label="", color="#b03412", palette="viridis",
        view_mode=view, histogram_bins=3, density=False, show_legend=False,
        show_colorbar=False, log_y=False, invert_y=False, y_limits=None,
    )
    data = plot["series"][0]["data"]
    if view == "Default":
        assert len(data) == (4 if geometry == "periodic" else 3)
        if geometry == "periodic":
            assert data[-1] == [46., .2]
    else:
        assert plot["dnascapePlotType"] == "Histogram"
        # Histogram bins retain the three raw samples, without a closure copy.
        assert sum(point[2] for point in data) == 3
    np.testing.assert_array_equal(raw, [.2, .5, -.4])


@pytest.mark.parametrize("target", ["initiation_events", "coalescence_events"])
@pytest.mark.parametrize("dependence", ["space", "spacetime"])
def test_event_targets_offer_only_initiation_rate_and_speed(target, dependence):
    families = mapping_route_families(target, dependence)
    assert len(families) == 1
    assert families[0].label == "Initiation rate + Fork speed"
    assert set(families[0].factor_keys) == {"initiation_rate", "fork_speed"}


def test_event_route_picker_stays_valid_when_changing_target_and_quantity():
    from nicegui.testing import user_simulation
    from app.main import index

    async def scenario():
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-mapping").click()
            await user.should_see(marker="mapping-target", retries=60)

            def control(marker):
                return next(e for e in user.find(marker=marker).elements if not e.is_deleted)

            for target in ("initiation_events", "coalescence_events"):
                with user.client:
                    control("mapping-target").set_value(target)
                for dependence in ("space", "spacetime"):
                    with user.client:
                        control("mapping-target-form").set_value(dependence)
                    for _ in range(5):
                        await asyncio.sleep(0)
                    route = control("mapping-route")
                    assert list(route.options.values()) == ["Initiation rate + Fork speed"]
                    assert route.value in route.options
                    assert control("mapping-input-initiation-rate-type")
                    assert control("mapping-input-fork-speed-type")

    asyncio.run(scenario())
