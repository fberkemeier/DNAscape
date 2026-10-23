"""Optional simulation observables use the shared preview and raw-data paths."""

import asyncio
import io

import numpy as np
from nicegui.testing import user_simulation

from app.main import index
from app.pages import simulation as page
from app.simulation_exports import build_raw_export, select_result_output


def test_tracking_options_gate_new_outputs_and_real_results_plot_and_export(monkeypatch):
    results, messages = [], []
    original = page.run_constant_simulation

    def run(*args, **kwargs):
        result = original(*args, **kwargs)
        results.append(result)
        return result

    monkeypatch.setattr(page, "run_constant_simulation", run)
    monkeypatch.setattr(page, "notify", lambda message, **kwargs: messages.append((message, kwargs)))

    async def scenario():
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-simulation").click()

            def control(marker):
                return next(item for item in user.client.elements.values() if marker in item._markers and not item.is_deleted)

            user.find("Simulation settings").click()
            output = control("simulation-output")
            new_labels = {"Origin firing (time)", "Replicon length", "Total fork density (space-time)"}
            assert new_labels.isdisjoint(output.options)
            user.find(marker="simulation-track-time-outputs").click()
            assert {"Origin firing (time)", "Replicon length"} <= set(output.options)
            assert "Total fork density (space-time)" not in output.options
            user.find(marker="simulation-track-time-outputs").click()
            user.find(marker="simulation-track-detailed-outputs").click()
            assert "Total fork density (space-time)" in output.options
            assert {"Origin firing (time)", "Replicon length"}.isdisjoint(output.options)
            user.find(marker="simulation-track-time-outputs").click()
            user.find(marker="simulation-fix-seed").click()
            with user.client:
                control("source-firing-rate-mode").set_value("scalar")
                control("source-firing-rate-constant").set_value(.1)
                control("simulation-region-end").set_value(12)
                control("simulation-replicates").set_value(6)
            user.find(marker="action-simulate").click()
            await user.should_see(marker="simulation-result-plot", retries=120)
            await user.should_see("Complete in", retries=60)
            assert len(results) == 1, messages
            assert new_labels <= set(output.options)
            assert all(label.endswith("(time)") for label in page._TIME_OUTPUTS)

            for label, key, plot_type, axis_kind in (
                ("Origin firing (time)", "origin_firing", "Line", "time"),
                ("Replicon length", "replicon_lengths", "Histogram", "sample"),
                ("Total fork density (space-time)", "fork_density_xt", "Heatmap", "xt"),
            ):
                with user.client:
                    output.set_value(label)
                await asyncio.sleep(.04)
                options = control("simulation-result-plot").options
                assert options["dnascapePlotType"] == plot_type, messages
                selected = select_result_output(results[0], key)
                assert selected.axis_kind == axis_kind
                artifact = build_raw_export(results[0], key, "npz")
                with np.load(io.BytesIO(artifact.data)) as archive:
                    expected = results[0].outputs[key]
                    if axis_kind == "time":
                        expected = expected.reshape(1, -1)
                        np.testing.assert_equal(archive["time_min"], results[0].time_grid)
                    if axis_kind == "xt":
                        np.testing.assert_equal(archive["position_kb"], results[0].position_kb)
                        np.testing.assert_equal(archive["time_min"], results[0].time_grid)
                    np.testing.assert_equal(archive["values"], expected)
                for file_type in ("tsv", "txt", "csv"):
                    assert build_raw_export(results[0], key, file_type).data
            assert not [message for message, kwargs in messages if kwargs.get("type") == "negative"], messages

    asyncio.run(scenario())
