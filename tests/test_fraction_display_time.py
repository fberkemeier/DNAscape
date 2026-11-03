"""Fraction previews use local CDF medians without trimming scientific data."""

import asyncio
from dataclasses import replace

import numpy as np
import pytest

from dnascape.workflows.mapping import MappingRequest, MappingResult, run_mapping
from dnascape.workflows.mapping_catalog import mapping_route
from dnascape.workflows.plotting import mapping_automatic_y_limits, mapping_plot_options


PLOT = dict(
    output_key="replicated_fraction_xt", plot_type="Heatmap", normalization="None",
    title="", x_label="Position (kb)", y_label="Time (min)", color="#b03412",
    palette="viridis", histogram_bins=10, line_width=1.8, show_markers=False,
    show_colorbar=True, log_y=False, invert_y=False,
)


def fraction_result(values=None, clock=None):
    # Local medians are 4 and 10 minutes, interpolated on an irregular clock.
    values = np.array([[0., .25, .75, 1., 1.], [0., .1, .3, 1., 1.]]) if values is None else np.asarray(values)
    clock = np.array([0., 2., 6., 20., 100.]) if clock is None else np.asarray(clock)
    values.setflags(write=False)
    clock.setflags(write=False)
    return MappingResult(
        target="replicated_fraction_xt", target_key="replicated_fraction", target_label="Replicated fraction",
        method="test", output_form="xt", output_dependence="spacetime",
        primary_output="replicated_fraction_xt", outputs={"replicated_fraction_xt": values},
        output_forms={"replicated_fraction_xt": "xt"}, x_axis=np.arange(values.shape[0], dtype=float),
        time_axis=clock, region_labels=(), elapsed_seconds=0., assumptions={},
    )


def automatic(result, **kwargs):
    return mapping_automatic_y_limits(result, output_key="replicated_fraction_xt",
        plot_type="Heatmap", histogram_bins=10, density=False, log_y=kwargs.get("log_y", False))


@pytest.mark.parametrize("normalization", ["None", "Z-score", "Min-max [0,1]"])
def test_fraction_default_uses_latest_local_median_before_display_normalization(normalization):
    result = fraction_result()
    raw = result.outputs[result.primary_output].copy()
    options = mapping_plot_options(result, **{**PLOT, "normalization": normalization})
    assert (options["yAxis"]["min"], options["yAxis"]["max"]) == (0., 15.)
    assert automatic(result) == (0., 15.)
    # Full coordinates remain available for navigation and raw-data export.
    assert options["dnascapeDefaultAxes"]["y"] == [0., 100.]
    np.testing.assert_array_equal(result.time_axis, [0., 2., 6., 20., 100.])
    np.testing.assert_array_equal(result.outputs[result.primary_output], raw)


def test_manual_plot_limits_override_the_median_default():
    options = mapping_plot_options(fraction_result(), **PLOT, y_limits=(1., 70.))
    assert (options["yAxis"]["min"], options["yAxis"]["max"]) == (1., 70.)


@pytest.mark.parametrize("values,expected", [
    ([[0., .1, .2, .4, .51]], 100.),  # Median near the final sample: cap at available time.
    ([[0., .1, .2, .3, .49]], 100.),  # No finite median within the available window.
    ([[0., .25, .75, 1., 1.], [0., .1, .2, .3, .49]], 100.),
    ([[0., 0., 0., 0., 0.]], 100.),
    ([[0., np.nan, .75, 1., 1.]], 100.),
    ([[1., 1., 1., 1., 1.]], 100.),  # All replication precedes the first sample.
    ([[0., .5, .5, 1., 1.]], 3.),   # First crossing of a plateau, not its last sample.
])
def test_partial_degenerate_and_plateau_fraction_curves(values, expected):
    result = fraction_result(values=values)
    assert automatic(result)[1] == pytest.approx(expected)
    assert mapping_plot_options(result, **PLOT)["yAxis"]["max"] == pytest.approx(expected)


def test_logarithmic_time_keeps_a_positive_lower_bound():
    result = fraction_result()
    options = mapping_plot_options(result, **{**PLOT, "log_y": True})
    assert (options["yAxis"]["min"], options["yAxis"]["max"]) == (2., 15.)
    assert automatic(result, log_y=True) == (2., 15.)


def test_other_quantities_and_fraction_inputs_keep_their_original_time_window():
    result = fraction_result()
    for key in ("initiation_rate", "input:replicated_fraction_xt"):
        other = replace(result, primary_output=key, outputs={key: result.outputs[result.primary_output]},
                        output_forms={key: "xt"})
        options = mapping_plot_options(other, **{**PLOT, "output_key": key})
        assert options["yAxis"]["max"] == 100.


def test_real_homogeneous_fraction_has_the_expected_analytic_median_window():
    rate, speed = .01, 1.4
    result = run_mapping(MappingRequest(
        route=mapping_route("firing_and_speed_to_replicated_fraction_xt_lightcone"),
        inputs={"initiation_rate": rate, "fork_speed": speed},
        input_forms={"initiation_rate": "scalar", "fork_speed": "scalar"},
        region_start_kb=0., region_end_kb=200., resolution_x=2., resolution_t=.1,
        settings={"perQ": True},
    ))
    # Before cones reach the opposite side, f(t)=1-exp(-I*v*t^2).
    expected = 1.5 * np.sqrt(np.log(2.) / (rate * speed))
    assert automatic(result)[1] == pytest.approx(expected, abs=.002)
    assert result.time_axis[-1] > 3 * automatic(result)[1]
    assert np.min(result.outputs[result.primary_output][:, -1]) > 1 - 1e-9


def test_mapping_fraction_axis_fields_default_customise_and_reset(monkeypatch):
    from nicegui.testing import user_simulation
    from app.main import index
    from app.pages import mapping

    monkeypatch.setattr(mapping, "run_mapping", lambda *args, **kwargs: fraction_result())

    async def scenario():
        async with user_simulation(index) as user:
            def control(marker):
                return next(e for e in user.find(marker=marker).elements if not e.is_deleted)

            async def value(marker, selected):
                with user.client:
                    control(marker).set_value(selected)
                for _ in range(5):
                    await asyncio.sleep(0)

            await user.open("/")
            user.find(marker="nav-mapping").click()
            await value("mapping-target", "replicated_fraction")
            await value("mapping-input-initiation-rate-type", "constant")
            await value("mapping-input-initiation-rate-constant", .01)
            await value("mapping-region-start", 0.)
            await value("mapping-region-end", 2.)
            user.find(marker="action-map").click()
            await user.should_see(marker="mapping-result-plot", retries=100)
            assert control("mapping-y-minimum").value == 0.
            assert control("mapping-y-maximum").value == 15.
            assert control("mapping-result-plot").options["yAxis"]["max"] == 15.
            await value("mapping-y-maximum", 70.)
            assert control("mapping-result-plot").options["yAxis"]["max"] == 70.
            user.find(marker="mapping-reset-y-limits").click()
            for _ in range(5):
                await asyncio.sleep(0)
            assert control("mapping-y-maximum").value == 15.
            assert control("mapping-result-plot").options["yAxis"]["max"] == 15.

    asyncio.run(scenario())
