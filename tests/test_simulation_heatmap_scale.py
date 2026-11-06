"""Simulation density colour ranges follow the pixels actually displayed."""
import asyncio

import numpy as np
import pytest
from nicegui.testing import user_simulation

from app.main import index
from app.pages.simulation import _result_plot_options
from dnascape.workflows.preview import _sample_heatmap
from dnascape.workflows.simulation import ConstantSimulationResult


def density_result():
    n = 5760
    density = np.zeros((n, 4))
    density[0, 1] = .1
    return ConstantSimulationResult(
        position_kb=np.arange(n, dtype=float), replication_timing=np.full(n, 10.),
        elapsed_seconds=.01, replicates=20, geometry="linear", time_grid=np.arange(4.),
        outputs={"replication_timing": np.full(n, 10.), "fork_density_xt": density,
                 "left_moving_fork_density_xt": np.full((n, 4), 2e-6),
                 "replicated_fraction_xt": np.ones((n, 4))}, detailed_outputs=True,
    )


def options(result, **overrides):
    settings = dict(output_label="Total fork density (space-time)", title="", series_label="",
                    x_axis_label="Position (kb)", y_axis_label="Time (min)", color="#b03412",
                    palette="viridis", view_mode="Default", histogram_bins=20, density=False,
                    show_legend=False, show_colorbar=True, log_y=False, invert_y=False, y_limits=None)
    settings.update(overrides)
    return _result_plot_options(result, **settings)


def test_overview_density_range_does_not_use_isolated_native_event_peaks():
    result = density_result()
    plotted = options(result)
    assert plotted["visualMap"]["min"] == 0.
    assert .0125 <= plotted["visualMap"]["max"] <= .015
    manual = options(result, z_limits=(0., .1))
    assert manual["visualMap"]["max"] == .1
    fraction = options(result, output_label="Replicated fraction (space-time)")
    assert fraction["visualMap"]["max"] >= 1.


@pytest.mark.parametrize("value", [0., 2e-9, .0032, 4.])
@pytest.mark.parametrize("log_z", [False, True])
def test_sparse_constant_missing_and_log_density_ranges_remain_valid(value, log_z):
    field = np.full((7, 9), value)
    field[0, 0] = np.nan
    if log_z and value == 0:
        with pytest.raises(ValueError, match="positive heatmap value"):
            _sample_heatmap(field, np.arange(9.), np.arange(7.), log_z=True, scale_to_preview=True)
        return
    _, _, pixels, lower, upper = _sample_heatmap(
        field, np.arange(9.), np.arange(7.), log_z=log_z, scale_to_preview=True)
    assert np.isnan(pixels[0, 0])
    assert np.isfinite([lower, upper]).all()
    assert upper > lower
    if log_z:
        assert lower < np.log10(value) <= upper
    elif value:
        assert lower == 0
        assert value <= upper <= value * 1.25
    else:
        assert (lower, upper) == (0., 1.)


def test_simulation_density_z_controls_follow_data_and_keep_manual_override(monkeypatch):
    from app.pages import simulation as page
    monkeypatch.setattr(page, "run_constant_simulation", lambda *_args: density_result())

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
            await value("simulation-region-end", 5760.)
            user.find(marker="action-simulate").click()
            await user.should_see(marker="simulation-result-plot", retries=120)
            await value("simulation-output", "Total fork density (space-time)")
            automatic_maximum = control("simulation-z-maximum").value
            assert .0125 <= automatic_maximum <= .015
            await value("simulation-z-maximum", .2)
            assert control("simulation-result-plot").options["visualMap"]["max"] == .2
            user.find(marker="simulation-reset-z-limits").click()
            for _ in range(4):
                await asyncio.sleep(0)
            assert control("simulation-z-maximum").value == pytest.approx(automatic_maximum)
            await value("simulation-output", "Left fork density (space-time)")
            assert control("simulation-z-maximum").value == pytest.approx(2e-6)
            assert not user.notify.messages, user.notify.messages

    asyncio.run(scenario())
