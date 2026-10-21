"""Overlay downloads are bounded and individual simulations never enter cache."""

import asyncio
import io
import zipfile

import numpy as np
import pytest
from nicegui.elements.upload import Upload
from nicegui.testing import user_simulation

from app.main import index
from app.pages import simulation as page
from app.session_cache import current_session_cache
from app.simulation_service import ConstantSimulationResult


async def wait_for(predicate, messages=None):
    for _ in range(150):
        if predicate():
            return
        await asyncio.sleep(.02)
    raise AssertionError(f"Simulation or download did not finish: {messages}")


@pytest.mark.parametrize("datasets", (1, 2))
def test_overlay_downloads_and_profile_retention_do_not_pollute_cache(monkeypatch, datasets):
    requests = []
    downloads = []
    messages = []

    def run(request, report, cancel):
        cancel()
        requests.append(request)
        profiles = np.arange(request.replicates * 4.0).reshape(request.replicates, 4)
        mean = profiles.mean(axis=0)
        return ConstantSimulationResult(
            position_kb=np.arange(4.0), replication_timing=mean, elapsed_seconds=.01,
            replicates=request.replicates, geometry="linear", time_grid=np.asarray([]),
            outputs={"replication_timing": mean, "fork_directionality": np.zeros(4)},
            replication_time_profiles=profiles[:request.retained_profile_count].copy(),
            replication_timing_std=profiles.std(axis=0),
        )

    monkeypatch.setattr(page, "run_constant_simulation", run)
    monkeypatch.setattr(page, "notify", lambda message, **kwargs: messages.append(message))

    async def scenario():
        async with user_simulation(index) as user:
            monkeypatch.setattr(
                type(user.download), "__call__",
                lambda _self, data, filename, media_type: downloads.append((data, filename, media_type)),
            )
            await user.open("/")
            user.find(marker="nav-simulation").click()
            await user.should_see(marker="simulation-output", retries=60)

            def control(marker):
                return next(element for element in user.client.elements.values() if marker in element._markers)

            user.find("Plot settings").click()
            user.find(marker="simulation-show-individual-profiles").click()
            user.find(marker="simulation-show-standard-deviation").click()
            assert user.client is not None
            with user.client:
                control("simulation-individual-profile-count").set_value(3)
                control("source-firing-rate-mode").set_value("profile")
            await user.should_see(marker="source-firing-rate-upload", retries=30)
            await control("source-firing-rate-upload").handle_uploads([
                Upload.SmallFileUpload(f"firing_{i}.tsv", "text/tab-separated-values", b"signal\n0.01\n0.02\n0.01\n0.02\n")
                for i in range(datasets)
            ])
            await user.should_see("Array size: 4", retries=60)
            user.find(marker="action-simulate").click()
            await user.should_see(marker="simulation-result-plot", retries=80)
            await wait_for(lambda: len(requests) == datasets and control("simulation-individual-profile-count").enabled)
            assert all(request.retained_profile_count == 3 for request in requests)
            await user.should_see(marker="simulation-profiles-1-row")
            await user.should_see(marker="simulation-standard-deviation-1-row")
            assert control("simulation-profiles-1-compression").value is True
            assert control("simulation-profiles-1-compression").enabled is False
            assert control("simulation-profiles-1-file-type").value == "TSV"
            assert control("simulation-profiles-1-file-type").options == ["TSV", "TXT", "CSV", "NPZ"]

            with user.client:
                cache = current_session_cache()
                workflows = cache.list_entries("workflows")
            assert len(workflows) == 1
            cached_results = (
                (workflows[0].payload,) if datasets == 1
                else tuple(item.result for item in workflows[0].payload.items)
            )
            assert len(cached_results) == datasets
            assert all(result.replication_time_profiles is None for result in cached_results)
            assert all(result.replication_timing_std is not None for result in cached_results)

            before = cache.count()
            user.find("Raw data").click()
            download_button = control("simulation-profiles-1-download")
            assert download_button.enabled
            with user.client:
                cache.set_tracking("uploads", True)
            user.find(marker="simulation-profiles-1-row-open").click()
            user.find(marker="simulation-profiles-1-row-open-plotting").click()
            await user.should_see(marker="plotting-data-upload", retries=80)
            from app.pages.common import _DATA_UPLOAD_STATES
            await wait_for(lambda: len(_DATA_UPLOAD_STATES[control("plotting-data-upload")].uploaded_names) == 3)
            assert _DATA_UPLOAD_STATES[control("plotting-data-upload")].uploaded_names == [
                "replicate_001.tsv", "replicate_002.tsv", "replicate_003.tsv",
            ]
            assert cache.count() == before
            user.find(marker="nav-simulation").click()
            user.find(marker="simulation-profiles-1-download").click()
            await wait_for(lambda: len(downloads) == 1, messages)
            assert cache.count() == before
            assert downloads[-1][1].endswith("_3_replicates.zip")
            with zipfile.ZipFile(io.BytesIO(downloads[-1][0])) as archive:
                assert archive.namelist() == [f"replicate_{i:03d}.tsv" for i in range(1, 4)]
                np.testing.assert_array_equal(np.loadtxt(io.BytesIO(archive.read(archive.namelist()[1])), skiprows=1), [4, 5, 6, 7])

            with user.client:
                control("simulation-standard-deviation-1-file-type").set_value("NPZ")
            user.find(marker="simulation-standard-deviation-1-download").click()
            await wait_for(lambda: len(downloads) == 2)
            with np.load(io.BytesIO(downloads[-1][0]), allow_pickle=False) as stored:
                np.testing.assert_allclose(stored["values"].reshape(-1), cached_results[0].replication_timing_std)

            if datasets > 1:
                with user.client:
                    control("simulation-presentation").set_value("Individual")
            with user.client:
                control("simulation-individual-profile-count").set_value(2)
            await wait_for(lambda: control("simulation-profiles-1-file-name").value.endswith("_2_replicates"), messages)
            with user.client:
                control("simulation-profiles-1-file-type").set_value("CSV")
            before = cache.count()
            user.find(marker="simulation-profiles-1-download").click()
            await wait_for(lambda: len(downloads) == 3)
            assert cache.count() == before
            with zipfile.ZipFile(io.BytesIO(downloads[-1][0])) as archive:
                assert archive.namelist() == ["replicate_001.csv", "replicate_002.csv"]

            with user.client:
                control("simulation-individual-profile-count").set_value(4)
            user.find(marker="action-simulate").click()
            await wait_for(lambda: len(requests) == datasets * 2 and control("simulation-individual-profile-count").enabled)
            assert all(request.retained_profile_count == 4 for request in requests[datasets:])
            await wait_for(lambda: control("simulation-profiles-1-file-name").value.endswith("_4_replicates"), messages)
            assert control("simulation-profiles-1-file-name").value.endswith("_4_replicates")
            assert control("simulation-profiles-1-file-type").value == "CSV"
            with user.client:
                control("simulation-show-individual-profiles").set_value(False)
                control("simulation-show-standard-deviation").set_value(False)
            await user.should_not_see(marker="simulation-profiles-1-row")
            await user.should_not_see(marker="simulation-standard-deviation-1-row")

    asyncio.run(scenario())
