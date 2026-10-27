"""Statistics exports follow each workflow's histogram and raw-file controls."""

import asyncio
import csv
import gzip
import io
import zipfile

import numpy as np
import pytest
from nicegui.elements.upload import Upload
from nicegui.elements.checkbox import Checkbox
from nicegui.testing import user_simulation

from app.export_settings import get_export_settings
from app.main import index


def element(user, marker):
    return next(item for item in user.client.elements.values() if marker in item._markers and not item.is_deleted)


async def wait_for(predicate):
    for _ in range(150):
        if predicate():
            return
        await asyncio.sleep(.02)
    raise AssertionError("The statistics export did not update")


@pytest.mark.parametrize(("page", "prefix"), [
    ("data-management", "data-manager-plot"), ("mapping", "mapping"),
    ("simulation", "simulation"), ("plotting", "plotting"),
])
def test_histogram_statistics_have_their_own_row_below_guides(page, prefix):
    async def scenario():
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker=f"nav-{page}").click()
            control = lambda name: element(user, f"{prefix}-{name}")
            assert not control("export-histogram-statistics").value
            with user.client:
                control("view").set_value("Histogram")
            row = control("histogram-statistics-options")
            assert row.visible
            assert [item for item in row.default_slot.children if isinstance(item, Checkbox)] == [
                control("show-histogram-statistics"), control("export-histogram-statistics"),
            ]
            legend_row = control("show-legend").parent_slot.parent
            guides = row.parent_slot.parent
            assert legend_row.parent_slot.parent is guides
            assert guides.default_slot.children.index(row) == guides.default_slot.children.index(legend_row) + 1
            assert control("show-hover").parent_slot.parent.parent_slot.parent is legend_row
            with user.client:
                control("view").set_value("Default")
            assert not row.visible
            assert not control("export-histogram-statistics").visible
    asyncio.run(scenario())


@pytest.mark.parametrize("page_name", ["mapping", "simulation"])
@pytest.mark.parametrize("datasets", [1, 2])
def test_histogram_statistics_download_on_scientific_workflows(monkeypatch, page_name, datasets):
    downloads = []
    if page_name == "mapping":
        from app.pages import mapping as page
        from app.tests.test_mapping_constant_ui import _mapped_result, _open_route
        monkeypatch.setattr(page, "run_mapping", lambda request, _report, _cancel: _mapped_result(request))
    else:
        from app.pages import simulation as page
        from app.simulation_service import ConstantSimulationResult
        def simulate(request, _report, _cancel):
            values = np.asarray([1., 2., 3., 4.])
            return ConstantSimulationResult(
                position_kb=np.arange(4.), replication_timing=values, elapsed_seconds=.01,
                replicates=request.replicates, geometry="linear", time_grid=np.asarray([]),
                outputs={"replication_timing": values, "inter_origin_distances": values},
                inter_origin_distance_count=4,
            )
        monkeypatch.setattr(page, "run_constant_simulation", simulate)

    async def scenario():
        async with user_simulation(index) as user:
            await user.open("/")
            monkeypatch.setattr(type(user.download), "__call__",
                lambda _self, data, filename=None, media_type=None: downloads.append((data, filename)))
            if page_name == "mapping":
                await _open_route(user)
                await element(user, "source-replication-timing-upload").handle_uploads([
                    Upload.SmallFileUpload(f"timing_{i}.tsv", "text/plain", b"signal\n1\n2\n3\n4\n")
                    for i in range(datasets)
                ])
                await user.should_see("Array size: 4", retries=60)
                action = "action-map"
            else:
                user.find(marker="nav-simulation").click()
                if datasets > 1:
                    with user.client:
                        element(user, "source-initiation-rate-mode").set_value("profile")
                    await element(user, "source-initiation-rate-upload").handle_uploads([
                        Upload.SmallFileUpload(f"firing_{i}.tsv", "text/plain", b"signal\n0.01\n0.02\n0.03\n0.04\n")
                        for i in range(datasets)
                    ])
                    await wait_for(lambda: any(
                        item._props.get("data-upload-name") == "firing_1.tsv"
                        and item._props.get("data-upload-status") == "complete"
                        for item in user.client.elements.values() if not item.is_deleted
                    ))
                action = "action-simulate"
            user.find(marker=action).click()
            await user.should_see(marker=f"{page_name}-result-plot", retries=80)
            await wait_for(lambda: element(user, action).enabled)
            with user.client:
                element(user, f"{page_name}-view").set_value("Histogram")
                element(user, f"{page_name}-show-histogram-statistics").set_value(False)
                element(user, f"{page_name}-export-histogram-statistics").set_value(True)
            await user.should_see(marker=f"{page_name}-statistics", retries=60)
            user.find(marker=f"{page_name}-statistics-download").click()
            await wait_for(lambda: bool(downloads))
            assert downloads[-1][1] == f"{page_name}_statistics.tsv"
            rows = list(csv.DictReader(io.StringIO(downloads[-1][0].decode()), delimiter="\t"))
            # Simulation's histogram layout displays the active batch case.
            expected_rows = datasets if page_name == "mapping" else 1
            assert len(rows) == expected_rows
            assert all(float(row["mean"]) == 2.5 for row in rows)
            assert all(float(row["standard_deviation"]) == np.std([1, 2, 3, 4]) for row in rows)
            expected_downloads = 1
            if page_name == "simulation":
                with user.client:
                    element(user, "simulation-view").set_value("Default")
                    element(user, "simulation-output").set_value("Inter-origin distance")
                await user.should_see(marker="simulation-statistics")
                assert element(user, "simulation-view").value == "Default"
                user.find(marker="simulation-statistics-download").click()
                expected_downloads += 1
                await wait_for(lambda: len(downloads) == expected_downloads)
                rows = list(csv.DictReader(io.StringIO(downloads[-1][0].decode()), delimiter="\t"))
                assert len(rows) == expected_rows
                assert all(float(row["mean"]) == 2.5 for row in rows)
            if datasets > 1:
                marker = "mapping-download-all-raw" if page_name == "mapping" else "action-download-all-simulation-raw"
                user.find(marker=marker).click()
                await wait_for(lambda: len(downloads) == expected_downloads + 1)
                with zipfile.ZipFile(io.BytesIO(downloads[-1][0])) as archive:
                    assert len(archive.namelist()) == 3
                    assert f"{page_name}_statistics.tsv" in archive.namelist()
    asyncio.run(scenario())


def test_analysis_combined_and_separate_statistics_preserve_export_preferences(monkeypatch):
    downloads = []
    async def scenario():
        async with user_simulation(index) as user:
            await user.open("/")
            monkeypatch.setattr(type(user.download), "__call__",
                lambda _self, data, filename=None, media_type=None: downloads.append((data, filename)))
            user.find(marker="nav-plotting").click()
            await element(user, "plotting-data-upload").handle_uploads([
                Upload.SmallFileUpload("one.tsv", "text/plain", b"signal\n1\n2\n3\n4\n"),
                Upload.SmallFileUpload("two.tsv", "text/plain", b"signal\n10\n20\n30\n40\n"),
            ])
            await user.should_see(marker="plotting-profile-source-2", retries=60)
            user.find(marker="action-plot").click()
            await user.should_see(marker="plotting-result-plot", retries=60)
            with user.client:
                element(user, "plotting-view").set_value("Histogram")
                element(user, "plotting-show-histogram-statistics").set_value(False)
                element(user, "plotting-export-histogram-statistics").set_value(True)
                element(user, "analysis-statistics-type").set_value("CSV")
            await user.should_see(marker="analysis-statistics")
            user.find(marker="analysis-statistics-download").click()
            await wait_for(lambda: len(downloads) == 1)
            rows = list(csv.DictReader(io.StringIO(downloads[-1][0].decode())))
            assert [float(row["mean"]) for row in rows] == [2.5, 25.]
            with user.client:
                element(user, "plotting-presentation").set_value("Separate")
            await user.should_see(marker="plotting-view-separate-1", retries=60)
            await user.should_not_see(marker="analysis-statistics")
            with user.client:
                element(user, "plotting-view-separate-1").set_value("Histogram")
                element(user, "plotting-export-histogram-statistics-separate-1").set_value(True)
                element(user, "analysis-statistics-1-name").set_value("first_histogram")
                element(user, "analysis-statistics-1-type").set_value("TXT")
            await user.should_see(marker="analysis-statistics-1")
            user.find(marker="analysis-statistics-1-download").click()
            await wait_for(lambda: len(downloads) == 2)
            assert downloads[-1][1] == "first_histogram.txt"
            row, = list(csv.DictReader(io.StringIO(downloads[-1][0].decode()), delimiter="\t"))
            assert float(row["mean"]) == 2.5
            with user.client:
                element(user, "plotting-presentation").set_value("Combined")
            await user.should_see(marker="analysis-statistics")
            assert element(user, "analysis-statistics-type").value == "CSV"
            with user.client:
                element(user, "plotting-presentation").set_value("Separate")
            await user.should_see(marker="analysis-statistics-1", retries=60)
            assert element(user, "analysis-statistics-1-name").value == "first_histogram"
            assert element(user, "analysis-statistics-1-type").value == "TXT"
    asyncio.run(scenario())


def test_data_manager_statistics_are_independent_and_follow_the_selected_file(monkeypatch):
    downloads = []
    monkeypatch.setattr(get_export_settings(), "raw_compression", "gzip")

    async def scenario():
        async with user_simulation(index) as user:
            await user.open("/")
            monkeypatch.setattr(type(user.download), "__call__",
                lambda _self, data, filename=None, media_type=None: downloads.append((data, filename)))
            user.find(marker="nav-data-management").click()
            control = lambda name: element(user, f"data-manager-{name}")
            await control("upload").handle_uploads([
                Upload.SmallFileUpload("one.tsv", "text/plain", b"signal\n1\n2\n3\n4\n"),
                Upload.SmallFileUpload("two.tsv", "text/plain", b"signal\n10\n20\n30\n40\n"),
            ])
            await user.should_see(marker="data-manager-upload-source-label-2", retries=60)
            user.find(marker="action-convert").click()
            await user.should_see(marker="data-manager-result-name-2", retries=80)
            with user.client:
                control("plot-view").set_value("Histogram")
                control("plot-show-histogram-statistics").set_value(False)
                control("plot-export-histogram-statistics").set_value(True)
            await user.should_see(marker="data-manager-statistics-2", retries=60)
            with user.client:
                control("statistics-2-name").set_value("second_statistics")
                control("statistics-2-type").set_value("CSV")
            user.find(marker="data-manager-statistics-2-download").click()
            await wait_for(lambda: len(downloads) == 1)
            payload, filename = downloads[-1]
            assert filename == "second_statistics.csv"
            row, = list(csv.DictReader(io.StringIO(payload.decode())))
            assert float(row["mean"]) == 25
            assert float(row["median"]) == 25
            assert float(row["standard_deviation"]) == np.std([10, 20, 30, 40])
            user.find(marker="data-manager-select-result-1").click()
            await user.should_see(marker="data-manager-statistics-1", retries=60)
            with user.client:
                control("statistics-1-name").set_value("first_statistics")
                control("statistics-1-type").set_value("NPZ")
                control("statistics-1-compression").set_value(True)
            user.find(marker="data-manager-statistics-1-download").click()
            await wait_for(lambda: len(downloads) == 2)
            assert downloads[-1][1] == "first_statistics.npz.gz"
            with np.load(io.BytesIO(gzip.decompress(downloads[-1][0])), allow_pickle=False) as archive:
                np.testing.assert_equal(archive["mean"], [2.5])
                np.testing.assert_equal(archive["standard_deviation"], [np.std([1, 2, 3, 4])])
            user.find(marker="download-all-converted-data").click()
            await wait_for(lambda: len(downloads) == 3)
            with zipfile.ZipFile(io.BytesIO(downloads[-1][0])) as archive:
                assert len(archive.namelist()) == 3
                assert "first_statistics.npz.gz" in archive.namelist()
            with user.client:
                control("plot-view").set_value("Default")
            await user.should_not_see(marker="data-manager-statistics-1")
            with user.client:
                control("plot-view").set_value("Histogram")
            await user.should_see(marker="data-manager-statistics-1")
            user.find(marker="data-manager-select-result-2").click()
            await user.should_see(marker="data-manager-statistics-2")
            assert control("statistics-2-name").value == "second_statistics"
            assert control("statistics-2-type").value == "CSV"
            assert not control("statistics-2-compression").value
            assert not control("plot-show-histogram-statistics").value
            with user.client:
                control("plot-export-histogram-statistics").set_value(False)
            await user.should_not_see(marker="data-manager-statistics-2")
            user.find(marker="delete-all-converted-data").click()
            await user.should_not_see(marker="data-manager-result-name-2")
    asyncio.run(scenario())
