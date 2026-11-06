"""Empty histogram windows and output changes retain valid preview domains."""

import asyncio

import numpy as np
import pytest

from dnascape.workflows.preview import PreviewLine, build_preview_plot_options


def histogram_options(values, **overrides):
    settings = dict(
        plot_type="Histogram", title="Samples", x_label="Length (kb)", y_label="Count",
        color="#b03412", palette="viridis", histogram_bins=5, density=False,
        show_legend=False, show_colorbar=False, log_y=False, invert_y=False,
        y_limits=None, series_label="Samples", values=values,
    )
    settings.update(overrides)
    return build_preview_plot_options(**settings)


@pytest.mark.parametrize("limits", [(3., 4.), (20., 30.)])
@pytest.mark.parametrize("width", [None, .3])
@pytest.mark.parametrize("boundary", ["left", "right"])
@pytest.mark.parametrize("log_x", [False, True])
def test_empty_histogram_windows_have_zero_counts_and_keep_full_reset_domain(limits, width, boundary, log_x):
    values = np.array([1., 2., 10., 11., np.nan, np.inf])
    original = values.copy()
    options = histogram_options(values, histogram_x_limits=limits,
                                histogram_bin_size=width, histogram_boundary=boundary,
                                log_x=log_x, show_histogram_statistics=True)
    rows = options["series"][0]["data"]
    assert rows[0][0] == limits[0] and rows[-1][1] == limits[1]
    assert all(row[2] == 0 for row in rows)
    assert options["dnascapeDefaultAxes"]["x"] == [1., 11.]
    assert options["dnascapeDefaultAxes"]["y"][1] > 0
    assert "No finite values" in options["graphic"][0]["style"]["text"]
    np.testing.assert_equal(values, original)


def test_cropped_histograms_reset_to_all_data_and_keep_visible_counts():
    options = histogram_options([1., 2., 10., 11.], histogram_x_limits=(1., 2.), histogram_bins=2)
    assert [row[2] for row in options["series"][0]["data"]] == [1., 1.]
    assert options["dnascapeDefaultAxes"]["x"] == [1., 11.]


def test_combined_histogram_allows_one_dataset_without_samples_in_shared_window():
    lines = (PreviewLine("First", np.arange(2.), np.array([1., 2.]), "#b03412"),
             PreviewLine("Second", np.arange(2.), np.array([10., 11.]), "#2596be"))
    options = histogram_options([], lines=lines, histogram_x_limits=(1., 2.), histogram_grouped=True)
    assert sum(row[2] for row in options["series"][0]["data"]) == 2
    assert all(row[2] == 0 for row in options["series"][1]["data"])


def test_all_missing_histogram_input_still_has_a_clear_validation_error():
    with pytest.raises(ValueError, match="no finite values"):
        histogram_options([np.nan, np.inf])


def test_simulation_output_and_view_changes_render_after_settings_are_synchronized(monkeypatch):
    from nicegui.testing import user_simulation
    from app.main import index
    from app.pages import simulation as page
    from dnascape.workflows.simulation import ConstantSimulationResult

    timing = np.arange(10., 16.)
    result = ConstantSimulationResult(
        position_kb=np.arange(6.), replication_timing=timing, elapsed_seconds=.01,
        replicates=4, geometry="linear", time_grid=np.array([]),
        outputs={"replication_timing": timing, "replicon_lengths": np.arange(180., 186.),
                 "s_phase_duration": np.array([50., 60., np.inf, np.inf])},
        incomplete_simulations=2,
    )
    renders, messages = [], []
    original_render = page._render_result_plot

    def render(*args, **kwargs):
        renders.append(kwargs)
        return original_render(*args, **kwargs)

    monkeypatch.setattr(page, "run_constant_simulation", lambda *_args: result)
    monkeypatch.setattr(page, "_render_result_plot", render)
    monkeypatch.setattr(page, "notify", lambda message, **kwargs: messages.append((message, kwargs)))

    async def scenario():
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-simulation").click()
            await user.should_see(marker="action-simulate", retries=60)

            def control(marker):
                return next(item for item in user.client.elements.values()
                            if marker in item._markers and not item.is_deleted)

            async def value(marker, selected):
                with user.client:
                    control(marker).set_value(selected)
                for _ in range(4):
                    await asyncio.sleep(0)

            await value("source-initiation-rate-mode", "scalar")
            await value("simulation-region-end", 6.)
            user.find(marker="action-simulate").click()
            await user.should_see(marker="simulation-result-plot", retries=120)
            for label, bounds in (("Replicon length", (180., 185.)),
                                  ("S-phase duration", (50., 60.)),
                                  ("Replicon length", (180., 185.))):
                renders.clear()
                await value("simulation-output", label)
                assert len(renders) == 1, [(r["output_label"], r["histogram_x_limits"]) for r in renders]
                assert renders[0]["histogram_x_limits"] == bounds
                assert control("simulation-result-plot").options["dnascapePlotType"] == "Histogram"
            for view in ("Boxplot", "Histogram"):
                renders.clear()
                await value("simulation-view", view)
                assert len(renders) == 1
                assert control("simulation-result-plot").options["dnascapePlotType"] == view
            await value("simulation-histogram-x-minimum", 300.)
            await value("simulation-histogram-x-maximum", 310.)
            options = control("simulation-result-plot").options
            assert all(row[2] == 0 for row in options["series"][0]["data"])
            assert not [message for message, settings in messages if settings.get("type") == "negative"], messages
    asyncio.run(scenario())
