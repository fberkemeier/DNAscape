"""Actual app dispatch for timing plus constant firing to spatial speed."""

import asyncio

import numpy as np
import pytest
from nicegui.testing import user_simulation

from app.main import index
from app.mapping_catalog import mapping_route
from app.mapping_service import MappingRequest, run_mapping


def _request():
    return MappingRequest(
        route=mapping_route("timing_and_firing_to_spatial_fork_speed"),
        inputs={"replication_timing": np.linspace(20., 50., 96), "firing_rate": .001},
        input_forms={"replication_timing": "x", "firing_rate": "scalar"},
        settings={"maxiter": 2, "smoothw": None, "T0": 10.},
        region_start_kb=32., region_end_kb=64., resolution_x=1.,
    )


def test_actual_route_reports_iterations_and_preserves_requested_region(capsys):
    progress = []
    result = run_mapping(_request(), report=lambda value, message: progress.append((value, message)))
    np.testing.assert_array_equal(result.x_axis, np.arange(32., 64.))
    np.testing.assert_array_equal(
        result.comparisons["replication_timing_fit"].observed,
        _request().inputs["replication_timing"][32:64],
    )
    messages = [message for _, message in progress]
    assert "Fitting fork speed... iteration 0/2" in messages
    assert "Fitting fork speed... iteration 2/2" in messages
    assert messages[-1] == "Mapping complete"
    values = [value for value, _ in progress]
    assert values == sorted(values)
    assert result.fit_mse <= result.outputs["fit_history"][0]
    assert result.outputs["fork_speed"].shape == (32,)
    assert capsys.readouterr().out == ""


def test_cancelling_at_first_iteration_releases_the_app_for_another_mapping():
    class Stopped(Exception):
        pass

    stop = False

    def report(value, message):
        nonlocal stop
        if message == "Fitting fork speed... iteration 0/2":
            stop = True

    def cancel():
        if stop:
            raise Stopped

    with pytest.raises(Stopped):
        run_mapping(_request(), report=report, cancel=cancel)
    result = run_mapping(_request())
    assert result.outputs["fork_speed"].shape == (32,)


def test_mapping_button_completes_the_real_restored_route_with_page_defaults(monkeypatch):
    from nicegui.elements.upload import Upload

    from app.pages import mapping as mapping_page

    results = []
    requests = []

    def recording_run(request, report, cancel):
        requests.append(request)
        result = run_mapping(request, report, cancel)
        results.append(result)
        return result

    monkeypatch.setattr(mapping_page, "run_mapping", recording_run)

    async def scenario():
        async with user_simulation(index) as user:
            def control(marker):
                return next(iter(user.find(marker=marker).elements))

            async def set_value(marker, value):
                with user.client:
                    control(marker).set_value(value)
                await asyncio.sleep(0)
                await asyncio.sleep(0)

            await user.open("/")
            user.find(marker="nav-mapping").click()
            await user.should_see(marker="mapping-target", retries=60)
            await set_value("mapping-target", "fork_speed")
            await set_value("mapping-target-form", "space")
            await set_value("mapping-region-start", 2.)
            await set_value("mapping-region-end", 10.)
            await set_value("mapping-input-firing-rate-constant", .001)
            content = "value\n" + "\n".join(str(x) for x in np.linspace(20., 40., 16)) + "\n"
            await control("source-replication-timing-upload").handle_uploads([
                Upload.SmallFileUpload("timing.tsv", "text/tab-separated-values", content.encode()),
            ])
            await user.should_see(marker="mapping-input-replication-timing-detected-data", retries=40)
            assert control("mapping-setting-fit-step").value == .4
            assert control("mapping-setting-maxiter").value == 10
            user.find(marker="action-map").click()
            await user.should_see(marker="mapping-result-plot", retries=60)
            await user.should_not_see(marker="mapping-error")
            assert len(results) == 1
            result = results[0]
            assert requests[0].route.method == "timing_and_firing_to_spatial_fork_speed"
            assert not {"spatial_knots", "regularization", "time_grid"} & requests[0].settings.keys()
            np.testing.assert_array_equal(result.x_axis, np.arange(2., 10.))
            assert result.fit_mse < result.outputs["fit_history"][0]

    asyncio.run(scenario())
