"""Raw files reopen through real page uploads without replacing or caching them."""

import asyncio
import io
import zipfile
from dataclasses import replace

import numpy as np
import pytest
from nicegui.elements.upload import Upload
from nicegui.elements.dialog import Dialog
from nicegui.testing import user_simulation

from app.export_settings import get_export_settings
from app.main import index
from app.mapping_service import MappingResult
from app.mapping_uploads import materialize_upload
from app.pages.common import _DATA_UPLOAD_STATES
from app.raw_data_transfer import unpack_raw_artifact
from app.session_cache import current_session_cache
from app.simulation_exports import ExportArtifact


def element(user, marker):
    matches = [item for item in user.client.elements.values() if marker in item._markers and not item.is_deleted]
    assert len(matches) == 1, (marker, len(matches))
    return matches[0]


async def settle():
    for _ in range(5):
        await asyncio.sleep(0)


def test_zip_transfer_expands_numeric_members_and_keeps_npz_intact():
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr("folder/one.tsv", "signal\n1\n2\n")
        archive.writestr("two.csv", "signal\n3\n4\n")
        archive.writestr("manifest.json", "{}")
    files = unpack_raw_artifact(ExportArtifact(payload.getvalue(), "profiles.zip", "application/zip"))
    assert [item.filename for item in files] == ["one.tsv", "two.csv"]
    assert files[0].data == b"signal\n1\n2\n"
    npz = ExportArtifact(payload.getvalue(), "profiles.npz", "application/octet-stream")
    assert unpack_raw_artifact(npz) == (npz,)


@pytest.mark.parametrize("destination,input_key,upload_marker,preserve", [
    ("data-management", None, "data-manager-upload", True),
    ("mapping", "replication_timing", "source-replication-timing-upload", True),
    ("simulation", "firing_rate", "source-firing-rate-upload", True),
    ("plotting", None, "plotting-data-upload", True),
    ("plotting", None, "plotting-data-upload", False),
])
def test_converted_raw_file_opens_in_each_page_and_preserves_existing_uploads(monkeypatch, destination, input_key, upload_marker, preserve):
    messages = []
    monkeypatch.setattr("app.pages.data_management.notify", lambda message, **kwargs: messages.append(message))
    monkeypatch.setattr("app.raw_data_transfer.notify", lambda message, **kwargs: messages.append(message))
    async def scenario():
        async with user_simulation(index) as user:
            await user.open("/")
            settings = get_export_settings()
            previous = settings.preserve_page_state
            settings.preserve_page_state = preserve
            try:
                user.find(marker=f"nav-{destination}").click()
                if destination == "simulation":
                    with user.client:
                        element(user, "source-firing-rate-mode").set_value("profile")
                    await settle()
                await user.should_see(marker=upload_marker, retries=60)
                await element(user, upload_marker).handle_uploads([
                    Upload.SmallFileUpload("existing.tsv", "text/tab-separated-values", b"position\tsignal\n0\t10\n1\t20\n2\t30\n3\t40\n"),
                ])
                expected_count = (3 if destination == "data-management" else 2) if preserve else 1
                for _ in range(150):
                    await asyncio.sleep(.02)
                    if "existing.tsv" in _DATA_UPLOAD_STATES[element(user, upload_marker)].uploaded_names:
                        break
                user.find(marker="nav-data-management").click()
                await user.should_see(marker="data-manager-upload", retries=60)
                await element(user, "data-manager-upload").handle_uploads([
                    Upload.SmallFileUpload("source.tsv", "text/tab-separated-values", b"position\tsignal\n0\t1\n1\t2\n2\t3\n3\t4\n"),
                ])
                for _ in range(150):
                    await asyncio.sleep(.02)
                    if "source.tsv" in _DATA_UPLOAD_STATES[element(user, "data-manager-upload")].uploaded_names and element(user, "action-convert").enabled:
                        break
                assert element(user, "action-convert").enabled
                user.find(marker="action-convert").click()
                for _ in range(150):
                    await asyncio.sleep(.02)
                    if any("data-manager-result-row-1" in item._markers for item in user.client.elements.values()):
                        break
                assert any("data-manager-result-row-1" in item._markers for item in user.client.elements.values()), messages
                with user.client:
                    cache = current_session_cache()
                    cache.set_tracking("uploads", True)
                before = cache.count()
                prefix = "data-manager-result-row-1-open"
                user.find(marker=prefix).click()
                user.find(marker=f"{prefix}-{destination}").click()
                if input_key is not None:
                    await user.should_see(marker=f"raw-open-input-{input_key}", retries=100)
                    user.find(marker=f"raw-open-input-{input_key}").click()
                try:
                    await user.should_see(marker=upload_marker, retries=100)
                except AssertionError:
                    raise AssertionError(messages) from None
                for _ in range(150):
                    await asyncio.sleep(.02)
                    uploader = element(user, upload_marker)
                    names = _DATA_UPLOAD_STATES[uploader].uploaded_names
                    if len(names) >= expected_count:
                        break
                if preserve:
                    assert "existing.tsv" in names
                assert len(names) == expected_count
                assert cache.count() == before
                if not preserve:
                    return
                # Reopening once more appends a unique name rather than replacing.
                user.find(marker="nav-data-management").click()
                user.find(marker=prefix).click()
                user.find(marker=f"{prefix}-{destination}").click()
                if input_key is not None:
                    await user.should_see(marker=f"raw-open-input-{input_key}", retries=100)
                    user.find(marker=f"raw-open-input-{input_key}").click()
                for _ in range(150):
                    await asyncio.sleep(.02)
                    names = _DATA_UPLOAD_STATES[element(user, upload_marker)].uploaded_names
                    if len(names) >= (4 if destination == "data-management" else 3):
                        break
                assert len(names) == len(set(names)) == (4 if destination == "data-management" else 3)
                assert any("(2)" in name for name in names)
                assert cache.count() == before
            finally:
                settings.preserve_page_state = previous
    asyncio.run(scenario())


@pytest.mark.parametrize("datasets,destination", [(1, "mapping"), (2, "simulation")])
def test_mapping_raw_actions_use_the_chosen_row_and_match_or_choose_an_input(monkeypatch, datasets, destination):
    from app.pages import mapping, simulation
    received = []
    target_page = mapping if destination == "mapping" else simulation
    original_inspect = target_page.inspect_upload_event

    async def inspect(event, **kwargs):
        result = await original_inspect(event, **kwargs)
        if result.name.startswith("chosen-file"):
            received.append(result)
        return result

    def mapped(request, *_args):
        timing = np.asarray(request.inputs["replication_timing"], dtype=float)
        return MappingResult(
            target=request.route.target, target_key=request.route.target_key, target_label="Firing rate",
            method=request.route.method, output_form="x", output_dependence="space", primary_output="firing_rate",
            outputs={"firing_rate": timing / 10, "replication_timing_fit": timing},
            output_forms={"firing_rate": "x", "replication_timing_fit": "x"},
            x_axis=np.arange(timing.size, dtype=float), time_axis=np.asarray([]), region_labels=(),
            elapsed_seconds=.01, assumptions={},
        )

    monkeypatch.setattr(mapping, "run_mapping", mapped)
    monkeypatch.setattr(target_page, "inspect_upload_event", inspect)

    async def scenario():
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-mapping").click()
            await user.should_see(marker="source-replication-timing-upload", retries=60)
            await element(user, "source-replication-timing-upload").handle_uploads([
                Upload.SmallFileUpload(f"timing-{i}.tsv", "text/tab-separated-values", f"timing\n{10+i}\n{20+i}\n{30+i}\n".encode())
                for i in range(datasets)
            ])
            for _ in range(150):
                await asyncio.sleep(.02)
                if len(_DATA_UPLOAD_STATES[element(user, "source-replication-timing-upload")].uploaded_names) == datasets:
                    break
            user.find(marker="action-map").click()
            await user.should_see(marker="mapping-result-plot", retries=100)
            for _ in range(150):
                await asyncio.sleep(.02)
                rows = [item for item in user.client.elements.values() if "dn-raw-output-row" in item._classes and not item.is_deleted and item.visible]
                if datasets == 2:
                    rows = [row for row in rows if "dn-mapping-batch-raw-row" in row._classes]
                if len(rows) >= datasets:
                    break
            assert len(rows) == datasets
            row = rows[-1]
            fields = next(child for child in row.default_slot.children if "dn-raw-output-fields" in child._classes)
            with user.client:
                fields.default_slot.children[0].set_value("chosen-file")
                fields.default_slot.children[1].set_value("NPZ")
                cache = current_session_cache()
                cache.set_tracking("uploads", True)
            before = cache.count()
            button = next(child for child in row.default_slot.children if "dn-raw-output-open" in child._classes)
            marker = next(iter(button._markers))
            user.find(marker=marker).click()
            user.find(marker=f"{marker}-{destination}").click()
            if destination == "mapping":
                await user.should_see(marker="raw-open-input-replication_timing", retries=100)
                user.find(marker="raw-open-input-replication_timing").click()
            for _ in range(150):
                await asyncio.sleep(.02)
                if received:
                    break
            assert len(received) == 1
            values = materialize_upload(replace(received[0], expected_form="x"), missing_policy="error")
            np.testing.assert_array_equal(values, np.asarray([10, 20, 30]) / 10 + (datasets - 1) / 10)
            if destination == "simulation":
                assert element(user, "source-firing-rate-mode").value == "profile"
                assert not any("raw-open-input-firing_rate" in item._markers for item in user.client.elements.values())
            assert cache.count() == before
    asyncio.run(scenario())


def test_cached_raw_file_uses_the_same_destination_menu():
    async def scenario():
        async with user_simulation(index) as user:
            await user.open("/")
            with user.client:
                cache = current_session_cache()
                entry = cache.add_artifact(
                    ExportArtifact(b"value\n1\n2\n3\n", "cached.tsv", "text/tab-separated-values"),
                    source="Mapping", kind="raw",
                )
            user.find(marker="nav-cached-files").click()
            await user.should_see(marker=f"cache-open-{entry.id}", retries=60)
            user.find(marker=f"cache-open-{entry.id}").click()
            user.find(marker=f"cache-open-{entry.id}-plotting").click()
            await user.should_see(marker="plotting-data-upload", retries=60)
            for _ in range(150):
                await asyncio.sleep(.02)
                if _DATA_UPLOAD_STATES[element(user, "plotting-data-upload")].uploaded_names:
                    break
            assert _DATA_UPLOAD_STATES[element(user, "plotting-data-upload")].uploaded_names == ["cached.tsv"]
            assert cache.count() == 1
            # Closing the input chooser via the dialog's outside-click value
            # change cancels the import and releases its pending callback.
            user.find(marker="nav-cached-files").click()
            user.find(marker=f"cache-open-{entry.id}").click()
            user.find(marker=f"cache-open-{entry.id}-mapping").click()
            await user.should_see(marker="raw-open-input-replication_timing", retries=60)
            dialog = element(user, "raw-open-input-replication_timing")
            while not isinstance(dialog, Dialog):
                dialog = dialog.parent_slot.parent
            with user.client:
                dialog.set_value(False)
            for _ in range(150):
                await asyncio.sleep(.02)
                if dialog.is_deleted:
                    break
            assert dialog.is_deleted
            assert not _DATA_UPLOAD_STATES[element(user, "source-replication-timing-upload")].uploaded_names
            assert cache.count() == 1
    asyncio.run(scenario())

