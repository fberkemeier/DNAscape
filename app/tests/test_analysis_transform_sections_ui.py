"""The Analysis controls, mask-only statistics and .py exports agree."""
import asyncio
import runpy

import numpy as np
from nicegui.elements.switch import Switch
from nicegui.elements.upload import Upload
from nicegui.testing import user_simulation

from app import code_export
from app.export_settings import get_export_settings
from app.main import index
from app.pages import plotting as page


def test_dataset_transforms_mask_only_and_exported_script(monkeypatch, tmp_path):
    monkeypatch.setattr(get_export_settings(), "dev_mode", True)
    snapshots, messages, downloads = [], [], []
    original = code_export.build_python
    def capture(snapshot):
        snapshots.append(snapshot)
        return original(snapshot)
    monkeypatch.setattr(code_export, "build_python", capture)
    monkeypatch.setattr(page, "notify", lambda message, **kwargs: messages.append(message))

    async def scenario():
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-plotting").click()
            def control(marker):
                return next(item for item in user.client.elements.values() if marker in item._markers and not item.is_deleted)
            async def settle():
                for _ in range(8):
                    await asyncio.sleep(0)
            async def set_value(marker, value):
                with user.client:
                    control(marker).set_value(value)
                await settle()
            async def commit():
                previous = control("plotting-result-plot").id if any("plotting-result-plot" in item._markers and not item.is_deleted for item in user.client.elements.values()) else None
                user.find(marker="action-plot").click()
                for _ in range(150):
                    await asyncio.sleep(.02)
                    if control("analysis-action-label").text == "Analyse" and any("plotting-result-plot" in item._markers and item.id != previous and not item.is_deleted for item in user.client.elements.values()):
                        return
                raise AssertionError(messages)
            assert control("analysis-data-transform-formula-1").value == "source"
            assert not control("analysis-data-transform-enabled").value
            assert not control("analysis-transform-enabled").value
            assert isinstance(control("analysis-transform-mask-only"), Switch)
            assert not control("analysis-transform-mask-only").value
            mask_only = control("analysis-transform-mask-only")
            assert mask_only.parent_slot.parent.default_slot.children.index(mask_only) > mask_only.parent_slot.parent.default_slot.children.index(control("analysis-transform-enabled").parent_slot.parent)
            await control("plotting-data-upload").handle_uploads([
                Upload.SmallFileUpload("a.csv", "text/csv", b"A\n-2\n5\n-1\n8\n"),
                Upload.SmallFileUpload("b.csv", "text/csv", b"B\n4\n1\n2\n0\n"),
            ])
            await user.should_see(marker="plotting-detected-data-2", retries=80)
            user.find(marker="analysis-data-transform").click()
            user.find(marker="analysis-transform").click()
            await set_value("analysis-data-transform-target", "1:A")
            await set_value("analysis-data-transform-formula-1", "source * 2")
            await set_value("analysis-data-transform-enabled", True)
            await set_value("analysis-value-scaling-1", "multiplier")
            await set_value("analysis-scale-multiplier-1", 2.)
            await set_value("analysis-invert-1", True)
            await set_value("analysis-data-transform-target", "2:B")
            assert control("analysis-data-transform-formula-1").value == "source"
            assert not control("analysis-invert-1").value
            await set_value("analysis-data-transform-target", "1:A")
            assert control("analysis-data-transform-formula-1").value == "source * 2"
            assert control("analysis-invert-1").value
            await set_value("analysis-transform-target", "1:A")
            await set_value("analysis-transform-comparison", "above")
            await set_value("analysis-transform-threshold", 10.)
            await set_value("analysis-transform-action", "multiply")
            await set_value("analysis-transform-factor", .5)
            await set_value("analysis-transform-export", True)
            await set_value("analysis-transform-mask-only", True)
            await set_value("analysis-transform-enabled", True)
            await set_value("analysis-enabled", True)
            await commit()
            plotted = control("plotting-result-plot").options
            np.testing.assert_equal(np.asarray(plotted["series"][0]["data"])[:, 1], [16, 4, 14, -8])
            np.testing.assert_equal(np.asarray(plotted["series"][1]["data"])[:, 1], [4, 1, 2, 0])
            await user.should_see(marker="analysis-report-table", retries=80)
            assert control("analysis-report-table").rows[0]["counts"]["pearson_1"] == 2
            assert control("analysis-report-table").rows[0]["pearson_1"] == "1.000"
            await user.should_see(marker="analysis-mask-result-1")
            monkeypatch.setattr(type(user.download), "__call__", lambda _self, data, filename=None, media_type=None: downloads.append((data, filename)))
            user.find(marker="analysis-mask-download-1").click()
            for _ in range(100):
                if downloads:break
                await asyncio.sleep(.02)
            assert downloads[-1][0] == b"0\n2\n"
            await set_value("plotting-view", "Histogram")
            for _ in range(120):
                chart = control("plotting-result-plot").options
                if chart.get("dnascapePlotType") == "Histogram":break
                await asyncio.sleep(.02)
            rows = control("plotting-result-plot").options["dnascapeHistogramStatistics"]
            assert [row["mean"] for row in rows] == [15., 1.75]
            user.find(marker="action-code").click()
            for _ in range(150):
                if snapshots and len(downloads)>1:break
                await asyncio.sleep(.02)
            assert snapshots, messages

    asyncio.run(scenario())
    snapshot = snapshots[-1]
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    for name, content in snapshot.files:
        path = inputs / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    script = tmp_path / "plotting.py"
    script.write_bytes(original(snapshot).data)
    runpy.run_path(str(script), run_name="__main__")
    assert next((tmp_path / "outputs").glob("*_mask_positions.tsv")).read_bytes() == b"0\n2\n"
    from dnascape.workflows.analysis_recipe import prepare_analysis_recipe
    prepared, report = prepare_analysis_recipe(snapshot.recipe, input_directory=inputs)
    np.testing.assert_equal(prepared[0]["values"], [16, 4, 14, -8])
    assert report.matrices[0].sample_counts[0, 1] == 2


def test_combine_updates_region_dataset_choices_and_applies_one_transform(monkeypatch):
    messages = []
    monkeypatch.setattr(page, "notify", lambda message, **kwargs: messages.append(message))
    async def scenario():
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-plotting").click()
            def control(marker):
                return next(item for item in user.client.elements.values() if marker in item._markers and not item.is_deleted)
            async def set_value(marker, value):
                with user.client:
                    control(marker).set_value(value)
                for _ in range(8):await asyncio.sleep(0)
            await control("plotting-data-upload").handle_uploads([
                Upload.SmallFileUpload("a.csv", "text/csv", b"A\n1\n2\n3\n"),
                Upload.SmallFileUpload("b.csv", "text/csv", b"B\n4\n5\n6\n"),
            ])
            await user.should_see(marker="plotting-detected-data-2", retries=80)
            await set_value("analysis-data-transform-enabled", True)
            await set_value("analysis-data-transform-combine", True)
            assert set(control("analysis-transform-target").options) == {"all", "1:__combined__"}
            assert control("analysis-data-transform-combined-formula").label == "Combined transform formula"
            await set_value("analysis-data-transform-combined-formula", "source1 + source2")
            await set_value("analysis-transform-target", "1:__combined__")
            await set_value("analysis-transform-enabled", True)
            await set_value("analysis-transform-selection", "all")
            await set_value("analysis-transform-action", "multiply")
            await set_value("analysis-transform-factor", 2.)
            user.find(marker="action-plot").click()
            await user.should_see(marker="plotting-result-plot", retries=80)
            chart = control("plotting-result-plot").options
            assert len(chart["series"]) == 1, messages
            np.testing.assert_equal(np.asarray(chart["series"][0]["data"])[:, 1], [10, 14, 18])
            await set_value("analysis-data-transform-enabled", False)
            assert set(control("analysis-transform-target").options) == {"all", "1:A", "2:B"}
            assert control("analysis-transform-target").value == "all"
    asyncio.run(scenario())
