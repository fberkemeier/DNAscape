"""Movement switches remain independent; uploads and code exports share a grid."""
import asyncio
import io
import runpy

import numpy as np
import pytest
from nicegui.elements.upload import Upload
from nicegui.testing import user_simulation

from app.main import index
from dnascape.workflows.simulation import ConstantSimulationResult, run_constant_simulation, validate_constant_simulation


def element(user, marker):
    return next(item for item in user.client.elements.values()
                if marker in item._markers and not item.is_deleted)


async def set_value(user, marker, value):
    with user.client:
        element(user, marker).set_value(value)
    for _ in range(3):
        await asyncio.sleep(0)


async def open_simulation(user):
    await user.open("/")
    user.find(marker="nav-simulation").click()
    await user.should_see(marker="source-initiation-rate-mode", retries=60)
    await set_value(user, "source-initiation-rate-mode", "scalar")
    await set_value(user, "simulation-region-end", 8.)
    await set_value(user, "simulation-replicates", 3)
    user.find("Simulation settings").click()


def fake_result(request):
    prepared = validate_constant_simulation(request).request
    n = int(round((prepared.region_end_kb - prepared.region_start_kb) / prepared.resolution_x))
    values = np.ones(n)
    return ConstantSimulationResult(position_kb=prepared.region_start_kb + np.arange(n) * prepared.resolution_x,
        replication_timing=values, elapsed_seconds=.01, replicates=prepared.replicates,
        geometry=prepared.geometry, time_grid=np.array([]), outputs={"replication_timing": values})


@pytest.mark.parametrize("stalling, stochastic", [(False, False), (True, False), (False, True), (True, True)])
def test_four_independent_movement_configurations(monkeypatch, stalling, stochastic):
    from app.pages import simulation as page
    captured = []

    def run(request, report, cancel):
        captured.append(request)
        return fake_result(request)

    monkeypatch.setattr(page, "run_constant_simulation", run)

    async def scenario():
        async with user_simulation(index) as user:
            await open_simulation(user)
            await user.should_see("Origin dynamics")
            await user.should_see("Fork dynamics")
            await user.should_not_see("Origin & fork behaviour")
            await user.should_not_see("Recovery specification")
            await set_value(user, "simulation-fork-stalling", stalling)
            await set_value(user, "simulation-stochastic-movement", stochastic)
            if stalling:
                await set_value(user, "simulation-stall-rate-per-kb", .2)
                await set_value(user, "simulation-stall-recovery", True)
                assert element(user, "simulation-stall-recovery-rate").value == pytest.approx(.1)
                await set_value(user, "simulation-stall-recovery-rate", .25)
            else:
                await user.should_not_see(marker="source-stalling-density-mode")
                await user.should_not_see(marker="simulation-stall-recovery")
            if stochastic:
                await user.should_see(marker="simulation-fork-speed-std")
                await set_value(user, "simulation-fork-speed-std", .3)
            else:
                await user.should_not_see(marker="simulation-fork-speed-std")
            user.find(marker="action-simulate").click()
            await user.should_see(marker="simulation-result-plot", retries=120)
            assert len(captured) == 1, user.notify.messages
            request = captured[0]
            assert request.fork_speed_std == (.3 if stochastic else 0.)
            assert request.stall_rate_per_kb == (.2 if stalling else 0.)
            if stalling:
                assert request.stall_duration is None
                assert request.stall_recovery_rate == .25
            else:
                assert request.stall_duration == np.inf

    asyncio.run(scenario())


def test_spatial_uploads_selection_alignment_and_developer_script(monkeypatch, tmp_path):
    from app import code_export
    from app.export_settings import get_export_settings
    from app.pages import simulation as page
    from dnascape.workflows.simulation_recipe import prepare_simulation_recipe
    snapshots, requests = [], []
    original = code_export.build_python
    monkeypatch.setattr(get_export_settings(), "dev_mode", True)

    def capture(snapshot):
        snapshots.append(snapshot)
        return original(snapshot)

    def run(request, report, cancel):
        requests.append(request)
        return run_constant_simulation(request)

    monkeypatch.setattr(code_export, "build_python", capture)
    monkeypatch.setattr(page, "run_constant_simulation", run)
    density = b"chr1\tchr2\n0.01\t0.1\n0.02\t0.2\n0.03\t0.3\n0.04\t0.4\n0.05\t0.5\n0.06\t0.6\n0.07\t0.7\n0.08\t0.8\n"
    rate = io.BytesIO()
    np.savez(rate, values=np.arange(1., 9.) / 10, data_form="space")

    async def scenario():
        async with user_simulation(index) as user:
            await open_simulation(user)
            with user.client:
                get_export_settings().dev_mode = True
            await user.should_see(marker="action-code", retries=60)
            await set_value(user, "simulation-fork-stalling", True)
            await set_value(user, "source-stalling-density-mode", "profile")
            await element(user, "source-stalling-density-upload").handle_uploads([
                Upload.SmallFileUpload("density.tsv", "text/tab-separated-values", density)])
            await user.should_see(marker="source-stalling-density-value-column", retries=60)
            await set_value(user, "source-stalling-density-value-column", "chr2")
            await set_value(user, "simulation-stall-recovery", True)
            await set_value(user, "source-recovery-mode", "profile")
            await element(user, "source-recovery-upload").handle_uploads([
                Upload.SmallFileUpload("recovery.npz", "application/octet-stream", rate.getvalue())])
            await user.should_see(marker="source-recovery-detected-data", retries=60)
            await set_value(user, "simulation-resolution-x", 2.)
            await set_value(user, "simulation-region-start", 2.)
            await set_value(user, "simulation-region-end", 6.)
            await set_value(user, "simulation-fix-seed", True)
            await set_value(user, "simulation-seed", 7)
            await set_value(user, "simulation-stochastic-movement", True)
            await set_value(user, "simulation-fork-speed-std", .2)
            user.find(marker="action-code").click()
            download = await user.download.next(timeout=15.)
            assert snapshots, user.notify.messages
            (tmp_path / "simulation.py").write_bytes(download.content)
            user.find(marker="action-simulate").click()
            await user.should_see(marker="simulation-result-plot", retries=120)
            assert len(requests) == 1, user.notify.messages

    asyncio.run(scenario())
    snapshot = snapshots[0]
    for name, data in snapshot.files:
        path = tmp_path / "inputs" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    prepared = prepare_simulation_recipe(snapshot.recipe, input_directory=tmp_path / "inputs")[0][1]
    expected = validate_constant_simulation(requests[0]).request
    actual = validate_constant_simulation(prepared).request
    np.testing.assert_array_equal(actual.stall_rate_per_kb, [.35, .55])
    for key in ("initiation_rate", "fork_speed", "stall_rate_per_kb", "stall_recovery_rate"):
        np.testing.assert_array_equal(getattr(actual, key), getattr(expected, key))
    namespace = runpy.run_path(str(tmp_path / "simulation.py"))
    reproduced = namespace["main"]()[0][1]
    original_result = run_constant_simulation(requests[0])
    np.testing.assert_array_equal(reproduced.replication_timing, original_result.replication_timing)
    np.testing.assert_array_equal(reproduced.outputs["completion_time"], original_result.outputs["completion_time"])
    assert tuple((tmp_path / "outputs").glob("*"))


def test_zero_recovery_rate_means_no_restart(monkeypatch):
    from app.pages import simulation as page
    captured = []
    def run(request, report, cancel):
        captured.append(request)
        return fake_result(request)
    monkeypatch.setattr(page, "run_constant_simulation", run)
    async def scenario():
        async with user_simulation(index) as user:
            await open_simulation(user)
            await set_value(user, "simulation-fork-stalling", True)
            await set_value(user, "simulation-stall-rate-per-kb", .2)
            await set_value(user, "simulation-stall-recovery", True)
            await set_value(user, "simulation-stall-recovery-rate", 0.)
            user.find(marker="action-simulate").click()
            await user.should_see(marker="simulation-result-plot", retries=100)
            assert captured[0].stall_duration is None
            assert captured[0].stall_recovery_rate == 0.
            assert element(user, "simulation-stall-recovery-rate").value == 0.
    asyncio.run(scenario())


def test_data_manager_transfer_targets_use_recovery_rates():
    from app.session_cache import current_session_cache
    from dnascape.workflows.exports import ExportArtifact
    async def scenario():
        async with user_simulation(index) as user:
            await open_simulation(user)
            with user.client:
                targets = {target.key: target for target in current_session_cache().input_targets("simulation")}
                assert {"stall_rate_per_kb", "stall_recovery_rate"} <= targets.keys()
                assert "stall_duration" not in targets
                await targets["stall_recovery_rate"].receive((ExportArtifact(b"rate\n0.1\n0\n0.2\n0.5\n", "recovery.tsv", "text/tab-separated-values"),))
            await user.should_see(marker="source-recovery-detected-data", retries=60)
            assert element(user, "simulation-fork-stalling").value
            assert element(user, "simulation-stall-recovery").value
            await user.should_not_see("Recovery specification")
            assert element(user, "source-recovery-mode").value == "profile"
    asyncio.run(scenario())
