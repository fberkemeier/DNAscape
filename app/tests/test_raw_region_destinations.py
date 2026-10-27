"""Masks reopen as masks; Mapping imports can choose a different target/route."""
import asyncio
import io

import numpy as np
import pytest
from nicegui.elements.upload import Upload
from nicegui.testing import user_simulation

from app.data_filter_controls import DataFilterControls
from app.main import index
from app.pages.common import _DATA_UPLOAD_STATES
from app.raw_data_transfer import open_raw_artifact
from app.session_cache import current_session_cache
from app.simulation_exports import ExportArtifact


def control(user, marker):
    return next(item for item in user.client.elements.values() if marker in item._markers and not item.is_deleted)


async def set_value(user, marker, value):
    with user.client:
        control(user, marker).set_value(value)
    for _ in range(8):await asyncio.sleep(0)


@pytest.mark.parametrize("destination,prefix,choice,layout,file_type,compress", [
    ("data-management", "data-manager-filter", None, "positions", "TSV", False),
    ("mapping", "mapping-filter", None, "intervals", "TSV", False),
    ("simulation", "simulation-filter-initiation-rate", "initiation_rate_mask", "positions", "NPZ", True),
    ("simulation", "simulation-filter-fork-speed", "fork_speed_mask", "intervals", "NPZ", False),
    ("plotting", "analysis-transform", None, "positions", "CSV", True),
])
def test_data_manager_mask_opens_in_region_upload_without_adding_source_data(monkeypatch, destination, prefix, choice, layout, file_type, compress):
    sections, messages = [], []
    original = DataFilterControls.__init__
    def capture(self, *args, **kwargs):
        original(self, *args, **kwargs)
        sections.append(self)
    monkeypatch.setattr(DataFilterControls, "__init__", capture)
    monkeypatch.setattr("app.raw_data_transfer.notify", lambda message, **kwargs: messages.append(message))

    async def scenario():
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-data-management").click()
            await control(user,"data-manager-upload").handle_uploads([Upload.SmallFileUpload("source.tsv", "text/plain", b"signal\n1\n-2\n-3\n4\n")])
            await user.should_see("Array size: 4", retries=60)
            await set_value(user,"data-manager-filter-enabled",True)
            await set_value(user,"data-manager-filter-export",True)
            user.find(marker="action-convert").click()
            await user.should_see(marker="data-manager-filter-result-1", retries=80)
            await set_value(user,"data-manager-filter-result-name-1","selection")
            await set_value(user,"data-manager-filter-result-layout-1",layout)
            await set_value(user,"data-manager-filter-result-type-1",file_type)
            await set_value(user,"data-manager-filter-result-compression-1",compress)
            with user.client:
                cache=current_session_cache()
                cache.set_tracking("uploads",True)
            before=cache.count()
            button="data-manager-filter-result-row-1-open"
            user.find(marker=button).click()
            user.find(marker=f"{button}-{destination}").click()
            if choice:
                await user.should_see(marker=f"raw-open-input-{choice}",retries=80)
                user.find(marker=f"raw-open-input-{choice}").click()
            await user.should_see(marker=f"{prefix}-upload-file",retries=80)
            section=next(item for item in reversed(sections) if item.marker_prefix==prefix)
            assert section.controls["selection"].value=="profile",messages
            assert section.controls["profile_units"].value=="bp"
            assert section.profile.layout==layout
            np.testing.assert_equal(section.profile.values, [1000,2000] if layout=="positions" else [[1000,3000]])
            assert section.section.visible
            if destination=="data-management":
                assert _DATA_UPLOAD_STATES[control(user,"data-manager-upload")].uploaded_names==["source.tsv"]
            else:
                assert not section.controls["enabled"].value
                if destination=="mapping":
                    assert not _DATA_UPLOAD_STATES[control(user,"source-replication-timing-upload")].uploaded_names
                    assert not any("raw-open-mapping-target" in item._markers and not item.is_deleted for item in user.client.elements.values())
                elif destination=="simulation":
                    assert control(user,"simulation-filter-relative-to").value==("fork_speed" if choice=="fork_speed_mask" else "initiation_rate")
                    assert control(user,"source-initiation-rate-mode").value=="scalar"
                else:
                    assert not _DATA_UPLOAD_STATES[control(user,"plotting-data-upload")].uploaded_names
            assert cache.count()==before
    asyncio.run(scenario())


def test_cancelling_mapping_destination_keeps_current_route_and_uploads():
    async def scenario():
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-mapping").click()
            await user.should_see(marker="source-replication-timing-upload",retries=60)
            await control(user,"source-replication-timing-upload").handle_uploads([
                Upload.SmallFileUpload("existing.tsv","text/plain",b"timing\n10\n20\n30\n"),
            ])
            await user.should_see(marker="source-replication-timing-upload-file",retries=60)
            original=tuple(control(user,marker).value for marker in ("mapping-target","mapping-target-form","mapping-route"))
            async def transfer():
                with user.client:
                    await open_raw_artifact(ExportArtifact(b"value\n1\n2\n3\n","new.tsv","text/plain"),"mapping",lambda _slug:None)
            task=asyncio.create_task(transfer())
            await user.should_see(marker="raw-open-mapping-target",retries=60)
            await set_value(user,"raw-open-mapping-target","replication_timing")
            await set_value(user,"raw-open-mapping-route","replication_timing--space--replicated_fraction")
            from nicegui.elements.dialog import Dialog
            dialog=control(user,"raw-open-mapping-target")
            while not isinstance(dialog,Dialog):dialog=dialog.parent_slot.parent
            with user.client:dialog.set_value(False)
            await task
            assert tuple(control(user,marker).value for marker in ("mapping-target","mapping-target-form","mapping-route"))==original
            assert _DATA_UPLOAD_STATES[control(user,"source-replication-timing-upload")].uploaded_names==["existing.tsv"]
            assert dialog.is_deleted
    asyncio.run(scenario())


@pytest.mark.parametrize("target,dependence,factors,key,form", [
    ("replication_timing","space","replicated_fraction","replicated_fraction","xt"),
    ("fork_speed","time","replication_timing-and-initiation_rate","initiation_rate","x"),
    ("fork_directionality","space","initiation_rate-and-fork_speed","fork_speed","t"),
    ("initiation_rate","spacetime","replication_timing-and-fork_speed","fork_speed","xt"),
])
def test_mapping_destination_chooses_target_dependence_route_and_exact_input(monkeypatch,target,dependence,factors,key,form):
    messages=[]
    monkeypatch.setattr("app.raw_data_transfer.notify",lambda message,**kwargs:messages.append(message))
    values={"x":np.arange(3.)+1,"t":np.arange(3.)[None,:]+1,"xt":np.arange(9.).reshape(3,3)+1}[form]
    if key=="replicated_fraction":
        values=values/10
    stream=io.BytesIO()
    np.savez(stream,values=values,data_form={"x":"space","t":"time","xt":"spacetime"}[form])
    artifact=ExportArtifact(stream.getvalue(),"new-profile.npz","application/octet-stream")
    async def scenario():
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-mapping").click()
            await user.should_see(marker="source-replication-timing-upload",retries=60)
            with user.client:
                cache=current_session_cache()
                cache.set_tracking("uploads",True)
                before=cache.count()
            async def transfer():
                with user.client:
                    await open_raw_artifact(artifact,"mapping",lambda _slug:None)
            task=asyncio.create_task(transfer())
            await user.should_see(marker="raw-open-mapping-target",retries=60)
            original_target=control(user,"mapping-target").value
            await set_value(user,"raw-open-mapping-target",target)
            await set_value(user,"raw-open-mapping-dependence",dependence)
            route=f"{target}--{dependence}--{factors}"
            await set_value(user,"raw-open-mapping-route",route)
            assert control(user,"mapping-target").value==original_target
            await user.should_see(marker=f"raw-open-input-{key}",retries=60)
            user.find(marker=f"raw-open-input-{key}").click()
            await task
            assert not messages, messages
            assert not user.notify.messages, user.notify.messages
            await user.should_see(marker=f"source-{key.replace('_','-')}-upload-file",retries=60)
            assert control(user,"mapping-target").value==target,messages
            assert control(user,"mapping-target-form").value==dependence
            assert control(user,"mapping-route").value==route
            assert control(user,f"mapping-input-{key.replace('_','-')}-type").value=="profile"
            assert control(user,f"mapping-input-{key.replace('_','-')}-form").value==form
            assert cache.count()==before
            assert not messages
    asyncio.run(scenario())
