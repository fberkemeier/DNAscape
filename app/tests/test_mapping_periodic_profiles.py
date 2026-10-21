"""Periodic model previews preserve all original observations and raw bins."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from dnascape.workflows.mapping import MappingComparison, MappingRequest, MappingResult, run_mapping
from dnascape.workflows.mapping_catalog import mapping_route
from dnascape.workflows.plotting import combined_mapping_plot_options, mapping_plot_options


PLOT_OPTIONS = dict(
    plot_type="Line", normalization="None", title="", x_label="Position (kb)",
    y_label="", color="#b03412", palette="viridis", histogram_bins=20,
    line_width=1.8, show_markers=False, show_colorbar=True, log_y=False, invert_y=False,
)


def _result(*, key="firing_rate", form="x", periodic=True, values=None):
    return MappingResult(
        target=key, target_key=key, target_label=key, method="test",
        output_form=form, output_dependence={"x": "space", "t": "time", "xt": "spacetime"}[form],
        primary_output=key, outputs={key: np.asarray([0.1, 0.3, 0.2] if values is None else values)},
        output_forms={key: form}, x_axis=np.asarray([40000., 40001., 40002.]),
        time_axis=np.asarray([0., 5., 10.]), region_labels=(), elapsed_seconds=0.,
        assumptions={"perQ": periodic},
    )


@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("key", ["firing_rate", "fork_speed", "replication_timing", "input:replication_timing"])
def test_spatial_previews_keep_exact_samples_and_coordinates(periodic, key):
    result = _result(key=key, periodic=periodic)
    original = result.outputs[key].copy()
    options = mapping_plot_options(result, output_key=key, **PLOT_OPTIONS)
    assert options["series"][0]["data"] == np.column_stack((result.x_axis, original)).tolist()
    np.testing.assert_array_equal(result.outputs[key], original)


@pytest.mark.parametrize("form", ["x", "t"])
@pytest.mark.parametrize("normalization", ["None", "Z-score", "Min-max [0,1]"])
def test_periodic_comparison_keeps_observed_and_fitted_samples(form, normalization):
    observed = np.asarray([117., 130., 139.])
    fitted = np.asarray([118., 129., 120.])
    observed.setflags(write=False)
    fitted.setflags(write=False)
    comparison = MappingComparison("replication_timing", "replication_timing_fit", form, observed, fitted)
    result = replace(_result(form=form), comparisons={"replication_timing_fit": comparison})
    kwargs = {**PLOT_OPTIONS, "normalization": normalization, "output_key": "comparison:replication_timing_fit"}
    periodic = mapping_plot_options(result, **kwargs)
    linear = mapping_plot_options(replace(result, assumptions={"perQ": False}), **kwargs)
    assert periodic["series"] == linear["series"]
    axis = result.x_axis if form == "x" else result.time_axis
    for series in periodic["series"]:
        assert len(series["data"]) == observed.size
        np.testing.assert_array_equal(np.asarray(series["data"])[:, 0], axis)
    if normalization == "None":
        np.testing.assert_array_equal(np.asarray(periodic["series"][0]["data"])[:, 1], observed)
        np.testing.assert_array_equal(np.asarray(periodic["series"][1]["data"])[:, 1], fitted)
    np.testing.assert_array_equal(observed, [117., 130., 139.])
    np.testing.assert_array_equal(fitted, [118., 129., 120.])


@pytest.mark.parametrize("form", ["t", "xt"])
def test_periodic_heatmaps_keep_the_same_extent_and_pixels(form):
    values = np.asarray([0.1, 0.5, 0.9]) if form == "t" else np.arange(9.).reshape(3, 3)
    result = _result(key="replicated_fraction_xt", form=form, values=values)
    kwargs = {**PLOT_OPTIONS, "output_key": result.primary_output, "plot_type": "Heatmap"}
    periodic = mapping_plot_options(result, **kwargs)
    linear = mapping_plot_options(replace(result, assumptions={"perQ": False}), **kwargs)
    assert periodic == linear
    np.testing.assert_array_equal(result.outputs[result.primary_output], values)


def test_combined_periodic_preview_does_not_append_wraparound_bins():
    results = [SimpleNamespace(key=str(i), name=f"Dataset {i}", result=_result()) for i in range(2)]
    options = combined_mapping_plot_options(results, output_key="firing_rate", **PLOT_OPTIONS)
    for series in options["series"]:
        assert series["data"] == [[40000., 0.1], [40001., 0.3], [40002., 0.2]]


def test_periodic_geometry_does_not_shift_downsampled_observations():
    observed = np.linspace(117., 139., 10000)
    fitted = observed + np.sin(np.arange(observed.size) / 100.)
    result = replace(
        _result(key="replication_timing", values=observed),
        x_axis=40000. + np.arange(observed.size),
        comparisons={"replication_timing_fit": MappingComparison(
            "replication_timing", "replication_timing_fit", "x", observed, fitted,
        )},
    )
    for output_key in ("replication_timing", "comparison:replication_timing_fit"):
        periodic = mapping_plot_options(result, output_key=output_key, **PLOT_OPTIONS)
        linear = mapping_plot_options(replace(result, assumptions={"perQ": False}), output_key=output_key, **PLOT_OPTIONS)
        assert periodic["series"] == linear["series"]
        assert periodic["series"][0]["data"][-1] == [49999., 139.]


@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("method,speed_form", [
    ("timing_and_speed_to_firing_rate", "scalar"),
    ("timing_to_firing_constant_fork_speed", "scalar"),
    ("timing_and_variable_speed_to_spatial_firing_rate", "x"),
])
def test_real_firing_fit_keeps_observed_timing_and_plot_coordinates(periodic, method, speed_form):
    observed = 110. + 15. * np.sin(np.arange(32.) / 4.) + np.arange(32.) / 2.
    original = observed.copy()
    observed.setflags(write=False)
    speed = 1.4 if speed_form == "scalar" else np.linspace(1.2, 1.6, observed.size)
    settings = {"perQ": periodic, "maxiter": 2}
    if speed_form == "x":
        settings.update(sim_number=8, refine_seed=17)
    result = run_mapping(MappingRequest(
        route=mapping_route(method), inputs={"replication_timing": observed, "fork_speed": speed},
        input_forms={"replication_timing": "x", "fork_speed": speed_form},
        settings=settings,
        resolution_x=1.,
    ))
    comparison = result.comparisons["replication_timing_fit"]
    np.testing.assert_array_equal(observed, original)
    np.testing.assert_array_equal(comparison.observed, original)
    assert np.all(np.isfinite(result.outputs["firing_rate"]))
    options = mapping_plot_options(result, output_key="comparison:replication_timing_fit", **PLOT_OPTIONS)
    np.testing.assert_array_equal(np.asarray(options["series"][0]["data"])[:, 1], original)
    np.testing.assert_array_equal(np.asarray(options["series"][0]["data"])[:, 0], result.x_axis)


@pytest.mark.parametrize("count", [31, 32])
def test_periodic_inverse_and_forward_do_not_depend_on_where_the_ring_starts(count):
    from dnascape.fitting import map_timing_firing
    from dnascape.mapping import map_firing_timing

    phase = np.arange(count) / count
    firing = 4e-5 + 3e-5 * np.sin(2 * np.pi * phase) ** 2 + 1e-5 * phase
    observed = map_firing_timing(firing, fork_speed=1.4, perQ=True)
    shifted = map_firing_timing(np.roll(firing, 7), fork_speed=1.4, perQ=True)
    np.testing.assert_allclose(shifted, np.roll(observed, 7), atol=1e-10, rtol=1e-12)
    options = dict(fork_speed=1.4, maxiter=10, fit_step=2., perQ=True, verbose=False)
    fitted = map_timing_firing(observed, **options)
    shifted_fit = map_timing_firing(shifted, **options)
    np.testing.assert_allclose(shifted_fit, np.roll(fitted, 7), atol=1e-14, rtol=1e-10)
    prediction = map_firing_timing(fitted, fork_speed=1.4, perQ=True)
    np.testing.assert_allclose(prediction, fitted._rmap_info["replication_timing_fit"], atol=1e-10)


def test_smooth_periodic_firing_model_closes_without_extending_observed_timing():
    observed = np.linspace(110., 130., 64)
    result = run_mapping(MappingRequest(
        route=mapping_route("timing_and_speed_to_firing_rate"),
        inputs={"replication_timing": observed, "fork_speed": 1.4},
        input_forms={"replication_timing": "x", "fork_speed": "scalar"},
        settings={"perQ": True, "maxiter": 2}, resolution_x=2.,
    ))
    assert result.diagnostics["fit_periodic_smoothing_kb"] > 0
    assert "fit_periodic_smoothing_kb" not in result.outputs
    rates = result.outputs["firing_rate"]
    assert rates.size == observed.size
    rate_plot = mapping_plot_options(result, output_key="firing_rate", **PLOT_OPTIONS)
    points = rate_plot["series"][0]["data"]
    assert len(points) == rates.size + 1
    assert points[-1] == [128., rates[0]]
    np.testing.assert_array_equal(np.asarray(points[:-1])[:, 1], rates)
    comparison = mapping_plot_options(result, output_key="comparison:replication_timing_fit", **PLOT_OPTIONS)
    np.testing.assert_array_equal(np.asarray(comparison["series"][0]["data"])[:, 1], observed)
    assert len(comparison["series"][0]["data"]) == observed.size
