"""Hosted retention and single-input controls, including their local boundary."""

import asyncio
import gc
import weakref
from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import pytest
from nicegui import ui
from nicegui.elements.upload import Upload
from nicegui.testing import user_simulation

from app import session_cache as cache_module
from app.main import index
from app.mapping_uploads import inspect_upload
from app.pages import mapping, simulation
from app.pages.common import _DATA_UPLOAD_STATES, data_upload
from app.session_cache import SessionCache, estimate_retained_bytes
from app.simulation_exports import ExportArtifact
from dnascape import _resources as resources
from dnascape.workflows.batch import (
    BatchSourceSummary,
    plan_batch_sources,
    plan_combination_indices,
)


@pytest.fixture
def render(monkeypatch):
    monkeypatch.setenv("RENDER", "true")
    monkeypatch.setattr(resources, "memory_status", lambda: resources.MemoryStatus(
        512 * resources.MIB, 160 * resources.MIB, 85 * resources.MIB))
    monkeypatch.setattr(cache_module, "_SESSION_CACHES", OrderedDict())


def test_repeated_downloads_and_visitors_stay_within_shared_budget(render):
    caches = cache_module._SESSION_CACHES
    caches.update((str(i), SessionCache()) for i in range(3))
    references = []
    for i in range(60):
        array = np.full(resources.MIB // 8, i, dtype=float)
        references.append(weakref.ref(array))
        caches[str(i % 3)].add_reference(
            name=f"Result {i}", source="Mapping", detail="", payload=array,
            navigate_to="mapping", keep=True)
    del array
    gc.collect()
    assert sum(cache.size() for cache in caches.values()) <= 8 * resources.MIB
    assert sum(ref() is not None for ref in references) <= 8
    assert all(cache.size() <= 4 * resources.MIB for cache in caches.values())
    assert all(cache.max_total_items == 10 and cache.max_items == 5 for cache in caches.values())


def test_oversized_item_does_not_evict_small_download_or_hash_large_bytes(render, monkeypatch):
    cache = SessionCache()
    small = cache.add_artifact(ExportArtifact(b"ok", "small.txt", "text/plain"),
                               source="Data Manager", kind="raw")
    monkeypatch.setattr(cache_module, "_artifact_digest", lambda _: pytest.fail("hashed oversized artifact"))
    result = cache.add_artifact(ExportArtifact(b"x" * (5 * resources.MIB), "large.txt", "text/plain"),
                                source="Data Manager", kind="raw")
    assert result is None
    assert cache.get(small.id) is small


def test_zero_budget_disables_retention_and_reset_restores_hosted_defaults(render):
    cache = SessionCache()
    assert not cache.is_tracking("workflows")
    cache.set_byte_limit(0)
    assert cache.add_reference(name="empty", source="Mapping", detail="", payload={},
                               navigate_to="mapping") is None
    cache.set_tracking("workflows", True)
    cache.reset()
    assert cache.max_total_bytes == 4 * resources.MIB
    assert not cache.is_tracking("workflows")
    cache.set_byte_limit(256 * resources.MIB)
    assert cache.max_total_bytes == 4 * resources.MIB


def test_pressure_reduces_retention_without_changing_user_budget(render, monkeypatch):
    cache = SessionCache()
    cache.add_reference(name="old", source="Mapping", detail="", payload=None,
                        size_bytes=4 * resources.MIB, navigate_to="mapping")
    monkeypatch.setattr(resources, "memory_status", lambda: resources.MemoryStatus(
        512 * resources.MIB, 423 * resources.MIB, 85 * resources.MIB))
    cache.add_reference(name="new", source="Mapping", detail="", payload=None,
                        size_bytes=resources.MIB, navigate_to="mapping")
    assert cache.size() == cache.byte_limit == resources.MIB
    assert cache.max_total_bytes == 4 * resources.MIB


def test_slices_account_for_the_allocation_they_retain():
    array = np.ones(1000)
    assert estimate_retained_bytes((array[:1], array[1:2])) == array.nbytes


def test_upload_registry_does_not_keep_deleted_pages_alive():
    references = []

    def page():
        payload = np.ones(100_000)
        references.append(weakref.ref(payload))

        def uploaded(_event):
            return payload.size > 0

        uploader = data_upload(accept=".csv", accepted_formats="CSV", on_upload=uploaded)
        references.append(weakref.ref(uploader))

    async def scenario():
        async with user_simulation(page) as user:
            await user.open("/")
            assert references[-1]() in _DATA_UPLOAD_STATES
            # NiceGUI keeps the last refresh target globally until the next
            # refresh. Move it off this uploader to isolate our registry.
            with user.client:
                ui.refreshable(lambda: None)()
            user.client.delete()
            gc.collect()
            assert references[0]() is None
            assert references[-1]() not in _DATA_UPLOAD_STATES

    asyncio.run(scenario())


@pytest.mark.parametrize("mode", ["pairwise", "all"])
def test_render_rejects_combinations_before_constructing_them(render, mode):
    with pytest.raises(resources.HostedResourceError, match="Online input limit"):
        plan_combination_indices((100_000, 100_000), mode=mode)
    with pytest.raises(resources.HostedResourceError, match="Timing"):
        plan_batch_sources((BatchSourceSummary("Timing", ("a.csv", "b.csv")),))
    assert plan_combination_indices((1, 1), mode=mode) == ((0, 0),)


def test_local_cache_defaults_and_combinations_remain_available(monkeypatch):
    monkeypatch.delenv("RENDER", raising=False)
    cache = SessionCache()
    assert cache.max_total_bytes == 256 * resources.MIB
    assert cache.is_tracking("workflows")
    assert len(plan_combination_indices((2, 3), mode="all")) == 6


def test_hosted_simulation_rejects_batch_before_materialization(render, monkeypatch):
    controls = SimpleNamespace(mode=SimpleNamespace(value="profile"), inspections=[object(), object()])
    monkeypatch.setattr(simulation, "_materialize_simulation_source", lambda *a, **k: pytest.fail("materialized batch"))
    with pytest.raises(resources.HostedResourceError, match="Fork speed"):
        simulation._materialize_simulation_sources(controls, label="Fork speed", missing_policy="zero")


def test_hosted_mapping_rejects_columns_before_materialization(render, monkeypatch):
    inspection = SimpleNamespace(name="two.csv", numeric_columns=("one", "two"))
    controls = SimpleNamespace(override=None, form=SimpleNamespace(value="x"),
                               spec=SimpleNamespace(key="replication_timing", label="Timing"),
                               surface={"inspections": [inspection],
                                        "selection_values": {"two.csv": {"column": "__all__"}}})
    with pytest.raises(resources.HostedResourceError, match="Timing"):
        asyncio.run(mapping._read_mapping_input_batch_async("replication_timing", controls))


@pytest.mark.parametrize("form", ["x", "xt"])
def test_render_column_selector_keeps_single_matrix_without_profile_expansion(render, form):
    inspection = inspect_upload("two.csv", b"one,two\n1,2\n3,4\n")
    retained = mapping._MappingUploadState(inspections=[inspection])
    spec = SimpleNamespace(formats_for=lambda _: ("csv",), accept_for=lambda _: ".csv")

    def page():
        mapping._render_upload_input(spec, (form,), "test", {}, retained,
                                     lambda: None, lambda _: None, lambda _: None, lambda: None)

    async def scenario():
        async with user_simulation(page) as user:
            await user.open("/")
            selector = next(e for e in user.client.elements.values()
                            if "mapping-input-test-value-column" in e._markers)
            assert ("__all__" in selector.options) == (form == "xt")
            assert selector.value in selector.options
            if form == "xt":
                assert selector.value == "__all__"

    asyncio.run(scenario())


@pytest.mark.parametrize("online", [False, True])
def test_live_controls_and_profile_replacement(monkeypatch, online):
    if online:
        monkeypatch.setenv("RENDER", "true")
    else:
        monkeypatch.delenv("RENDER", raising=False)
    monkeypatch.setattr(cache_module, "_SESSION_CACHES", OrderedDict())

    async def scenario():
        async with user_simulation(index) as user:
            def element(marker):
                return next(e for e in user.client.elements.values()
                            if marker in e._markers and not e.is_deleted)

            await user.open("/")
            for page, uploader_marker, mode_marker in (
                ("mapping", "source-replication-timing-upload", "mapping-input-fork-speed-type"),
                ("simulation", "source-initiation-rate-upload", "source-initiation-rate-mode"),
            ):
                user.find(marker=f"nav-{page}").click()
                mode = element(mode_marker)
                assert ("constants" in mode.options) is not online
                if page == "simulation":
                    with user.client:
                        mode.set_value("profile")
                    await asyncio.sleep(0)
                uploader = element(uploader_marker)
                assert bool(uploader._props.get("multiple")) is not online
                assert uploader._props.get("max-files") == (1 if online else None)
                for name in ("one.csv", "two.csv"):
                    await uploader.handle_uploads([Upload.SmallFileUpload(name, "text/csv", b"value\n1\n2\n3\n")])
                    await asyncio.sleep(.05)
                assert user.find("two.csv").elements
                if online:
                    assert not any(getattr(e, "text", None) == "one.csv" and not e.is_deleted
                                   for e in user.client.elements.values())
                else:
                    assert user.find("one.csv").elements
            user.find(marker="nav-cached-files").click()
            if online:
                with user.client:
                    cache = cache_module.current_session_cache()
                    cache.add_reference(name="small", source="Mapping", detail="", payload=np.ones(2),
                                        navigate_to="mapping")
                user.find(marker="cache-settings-button").click()
                with user.client:
                    element("cache-memory-limit").set_value(0)
                await asyncio.sleep(0)
                assert cache.max_total_bytes == 0 and cache.count() == 0
                assert not cache.is_tracking("workflows")
                assert cache._page_refreshers
                # NiceGUI calls these after its reconnect grace period.
                for callback in tuple(user.client.delete_handlers):
                    callback()
                assert not cache._page_refreshers and not cache._output_clearers
                assert cache not in cache_module._SESSION_CACHES.values()
            else:
                assert not any("cache-memory-limit" in e._markers and not e.is_deleted
                               for e in user.client.elements.values())

    asyncio.run(scenario())
