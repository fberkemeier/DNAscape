"""Spatial firing matches the published fitter and retains its MSE and ETA."""

import numpy as np
import pytest

from app.mapping_catalog import mapping_route
from app.mapping_service import MappingRequest, run_mapping
from app.pages.mapping import _setting_value

# Recorded from published DNAscape 187e89d at 50 kb/bin, v=1.4 kb/min,
# fit_step=0.6, neighbourhood_range=2000 kb, and ten iterations.
OBSERVED = np.array([80, 75, 60, 55, 50, 58, 70, 100, 130, 140, 130, 100], dtype=float)
HISTORICAL_RATES_PER_BIN = np.array([
    .00403355201007147, .00360836808502536, .00775928049605051,
    .00825857959959521, .01155443264663463, .00766053544496224,
    .00563063543682130, .00164554786461452, .00067402821796497,
    .00065534492788078, .00128301554911849, .00388337196598087,
])
HISTORICAL_TIMING = np.array([
    77.91622490697891, 73.22540441697170, 60.49645648957600,
    54.62110735020376, 50.208610636946965, 57.70937791851602,
    70.68167244376801, 95.30409017542310, 117.95896491121373,
    128.50324789886724, 127.12781683280377, 123.40302954987650,
])


@pytest.mark.parametrize("method", [
    "timing_and_speed_to_firing_rate", "timing_to_firing_constant_fork_speed",
])
def test_original_spatial_firing_fit_reaches_mapping_with_ten_iteration_mse(method):
    route = mapping_route(method)
    forms = {"replication_timing": "x", "fork_speed": "scalar"}
    settings = {
        setting.key: _setting_value(setting)
        for setting in route.active_settings(input_forms=forms)
        if setting.location != "derived" and setting.has_default and setting.default is not None
    }
    assert settings["maxiter"] == 10
    assert settings["fit_step"] == 2.0
    # The original update/model remains reproducible with its historical step.
    settings["fit_step"] = .6
    messages = []
    result = run_mapping(
        MappingRequest(
            route=route,
            inputs={"replication_timing": OBSERVED, "fork_speed": 1.4},
            input_forms=forms, settings=settings, resolution_x=50.,
        ),
        report=lambda _, message: messages.append(message),
    )
    # Mapping uses density units; the historical solver's per-bin values
    # therefore cross the public boundary divided by the bin width.
    np.testing.assert_allclose(result.outputs["firing_rate"], HISTORICAL_RATES_PER_BIN / 50., rtol=0, atol=2e-16)
    np.testing.assert_allclose(result.outputs["replication_timing_fit"], HISTORICAL_TIMING, rtol=0, atol=2e-12)
    assert result.fit_mse == pytest.approx(37.52451119545481, abs=1e-10)
    assert result.fit_mse == pytest.approx(np.mean((HISTORICAL_TIMING[2:-2] - OBSERVED[2:-2])**2))
    assert result.outputs["fit_history"].size == 10
    assert sum("/10" in message and "MSE" in message and "ETA" in message for message in messages) == 10
    np.testing.assert_array_equal(result.comparisons["replication_timing_fit"].observed, OBSERVED)


@pytest.mark.parametrize("method", [
    "timing_and_speed_to_firing_rate", "timing_to_firing_constant_fork_speed",
])
def test_mapping_safeguards_worsening_fit_and_reports_the_actual_returned_rates(method):
    from dnascape.fitting import _map_spatial_firing_timing_with_context

    observed = np.full(120, 450.)
    observed[55:59] = 160.
    route = mapping_route(method)
    forms = {"replication_timing": "x", "fork_speed": "scalar"}
    settings = {
        item.key: _setting_value(item)
        for item in route.active_settings(input_forms=forms)
        if item.location != "derived" and item.has_default and item.default is not None
    }
    messages = []
    result = run_mapping(
        MappingRequest(route=route, inputs={"replication_timing": observed, "fork_speed": 1.4},
                       input_forms=forms, settings=settings, resolution_x=10.),
        report=lambda _, message: messages.append(message),
    )
    assert result.diagnostics["fit_used_descent"]
    history = result.outputs["fit_history"]
    assert len(history) == 10 and np.all(np.diff(history) <= 0)
    assert result.fit_mse == pytest.approx(history[-1])
    assert sum("/10" in m and "MSE" in m and "ETA" in m for m in messages) == 10
    # The page returns rates per kb, while the preserved forward kernel takes
    # rates per bin. A round trip checks both the units and the plotted fit.
    prediction = _map_spatial_firing_timing_with_context(
        result.outputs["firing_rate"] * 10., fork_speed=1.4, resolution=10.,
        neighbourhood_range=2000., perQ=False, edge_bins=12, historical_linear=True,
    )
    np.testing.assert_allclose(prediction, result.outputs["replication_timing_fit"], atol=1e-8)
    assert result.fit_mse == pytest.approx(np.mean((prediction[12:-12] - observed[12:-12])**2))


@pytest.mark.parametrize("method", [
    "timing_and_speed_to_firing_rate", "timing_to_firing_constant_fork_speed",
])
def test_paper_step_default_fits_resolvable_origin_peaks_with_ten_iterations(method):
    from dnascape import rmap
    from dnascape.mapping import map_firing_timing

    positions = np.arange(240, dtype=float)
    rates_per_bin = (
        .00001 + .002 * np.exp(-((positions - 60.) / 5.)**2)
        + .001 * np.exp(-((positions - 170.) / 8.)**2)
    )
    timing = map_firing_timing(rates_per_bin, fork_speed=1.4, resolution=10.)
    source = {"replication_timing": timing}
    if method == "timing_and_speed_to_firing_rate":
        source["fork_speed"] = 1.4
    options = dict(source=source, target="firing_rate", method=method,
                   resolution=10., refine_edge_kb=0., maxiter=10, verbose=False)
    historical_step = rmap(fit_step=.6, **options)
    improved = rmap(**options)
    info = improved._rmap_info
    assert info["fit_mse"] < .1 * historical_step._rmap_info["fit_mse"]
    assert info["fit_mse"] < 25.
    assert len(info["fit_history"]) == 10
    assert np.all(np.diff(info["fit_history"]) <= 0)
    predicted = map_firing_timing(np.asarray(improved) * 10., fork_speed=1.4, resolution=10.)
    np.testing.assert_allclose(predicted, info["replication_timing_fit"], atol=1e-8)


@pytest.mark.parametrize("method", [
    "timing_and_speed_to_firing_rate", "timing_to_firing_constant_fork_speed",
])
@pytest.mark.parametrize("stall_options,expected_step", [
    ({}, 2.),
    ({"stall_rate": .01}, .6),
    ({"stall_rate_per_kb": .01}, .6),
    ({"stall_rate_per_kb": 0.}, 2.),
    ({"stall_rate_per_kb": .01, "fit_step": 1.}, 1.),
])
def test_public_spatial_firing_default_respects_stalling_and_explicit_step(
    monkeypatch, method, stall_options, expected_step,
):
    from dnascape import rmap, routing

    recorded = {}

    def capture(timing, **kwargs):
        recorded.update(kwargs)
        return np.ones_like(timing)

    monkeypatch.setattr(routing, "map_timing_firing", capture)
    source = {"replication_timing": OBSERVED}
    if method == "timing_and_speed_to_firing_rate":
        source["fork_speed"] = 1.4
    rmap(source=source, target="firing_rate", method=method, resolution=10., **stall_options)
    assert recorded["fit_step"] == expected_step
