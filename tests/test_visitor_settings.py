"""Preferences belong to clients and are detached at every worker boundary."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest
from nicegui.testing import User, user_simulation

from app import background
from app.main import index
from app.page_activity import current_page_activity
from app.pages.cached_files import _download_artifact
from app.session_cache import current_session_cache
from app.simulation_service import ConstantSimulationResult
from dnascape.workflows import analysis_recipe, code_recipes
from dnascape.workflows.export_settings import (
    ExportSettings,
    get_export_settings,
    load_export_settings,
    preview_point_limit,
    reset_export_settings,
    settings_scope,
    snapshot_export_settings,
)


def hydrate(user, **values):
    with user.client:
        bridge = user.client.storage["dnascape.preferences_bridge"]
        bridge.receive(SimpleNamespace(args={"preferences": {"version": 1, "settings": values}}))


def test_two_clients_apply_reset_and_workspaces_are_independent():
    headless = snapshot_export_settings()

    async def scenario():
        async with user_simulation(index) as alice:
            bob = User(alice.http_client)  # separate tab/client, same host and IP
            await alice.open("/")
            await bob.open("/")
            hydrate(alice, theme_color="#2563eb", preview_points=200, control_font_size_px=16,
                    preserve_page_state=False)
            hydrate(bob, theme_color="#0f766e", preview_points=700, raw_compression="xz")
            for visitor, end in ((alice, 123), (bob, 456)):
                visitor.find(marker="nav-simulation").click()
                with visitor.client:
                    next(iter(visitor.find(marker="simulation-region-end").elements)).set_value(end)
                visitor.find(marker="nav-overview").click()
                visitor.find(marker="nav-simulation").click()
                value = next(iter(visitor.find(marker="simulation-region-end").elements)).value
                assert value == (1000 if visitor is alice else 456)
            cancel_a, cancel_b = Event(), Event()
            with alice.client:
                first = get_export_settings()
                cache_a = current_session_cache()
                activity_a = current_page_activity()
                activity_a.begin("mapping", cancel=cancel_a.set)
                cache_a.add_reference(name="Alice result", source="Mapping", detail="", payload=np.ones(2), navigate_to="mapping")
                assert preview_point_limit() == 200
            with bob.client:
                second = get_export_settings()
                cache_b = current_session_cache()
                activity_b = current_page_activity()
                assert preview_point_limit() == 700
                assert not cache_b.count() and activity_b.indicator("mapping") is None
                activity_b.begin("simulation", cancel=cancel_b.set)
            assert first is not second and cache_a is not cache_b and activity_a is not activity_b
            alice.find(marker="app-menu-toggle").click()
            alice.find(marker="menu-settings").click()
            with alice.client:
                control = next(iter(alice.find(marker="settings-theme-color").elements))
                control.set_value("#be185d")
            alice.find(marker="settings-apply").click()
            assert first.theme_color == "#be185d" and second.theme_color == "#0f766e"
            alice.find(marker="app-menu-toggle").click()
            alice.find(marker="menu-reset").click()
            alice.find(marker="confirm-reset-everything").click()
            assert first == ExportSettings()
            assert second.theme_color == "#0f766e" and second.raw_compression == "xz"
            assert cancel_a.is_set() and not cancel_b.is_set()
            assert activity_b.indicator("simulation") == "running"
            assert not cache_a.count()
            with bob.client:
                cache_b.add_reference(name="Bob result", source="Simulation", detail="", payload=np.ones(2), navigate_to="simulation")
            with alice.client:
                assert not current_session_cache().count()
            assert cache_b.count() == 1

    asyncio.run(scenario())
    assert get_export_settings() == headless


def test_delayed_worker_export_uses_initiating_snapshot(monkeypatch):
    started, resume = Event(), Event()
    values = np.asarray([3., 2., 4.])
    result = ConstantSimulationResult(position_kb=np.arange(3.), replication_timing=values,
                                     elapsed_seconds=0., replicates=2, geometry="linear",
                                     time_grid=np.array([]), outputs={"replication_timing": values})

    def delayed(entry):
        started.set()
        assert resume.wait(5)
        return _download_artifact(entry), preview_point_limit()

    async def scenario():
        async with user_simulation(index) as alice:
            bob = User(alice.http_client)
            await alice.open("/")
            await bob.open("/")
            hydrate(alice, cached_file_format="csv", cached_compression="gzip", preview_points=200)
            hydrate(bob, cached_file_format="npz", cached_compression="none", preview_points=900)
            with alice.client:
                entry = current_session_cache().add_reference(name="Simulation result", source="Simulation",
                    detail="", payload=result, navigate_to="simulation")
                work = asyncio.create_task(background.io_bound(delayed, entry))
            assert await asyncio.to_thread(started.wait, 5)
            with alice.client:
                get_export_settings().cached_file_format = "txt"
                get_export_settings().preview_points = 5000
            with bob.client:
                reset_export_settings()
            resume.set()
            artifact, points = await work
            assert artifact.filename.endswith(".csv.gz") and points == 200
            with alice.client:
                assert get_export_settings().cached_file_format == "txt"
                assert preview_point_limit() == 5000
    asyncio.run(scenario())


@pytest.mark.parametrize("module,outer,inner", [
    (code_recipes, "run_data_manager_recipe", "_run_data_manager_recipe"),
    (analysis_recipe, "run_plotting_recipe", "_run_plotting_recipe"),
])
def test_parallel_headless_recipes_do_not_temporarily_mutate_shared_defaults(monkeypatch, module, outer, inner):
    before = snapshot_export_settings()
    barrier = __import__("threading").Barrier(2)

    def recipe(*args, **kwargs):
        barrier.wait(timeout=5)
        return preview_point_limit()

    monkeypatch.setattr(module, inner, recipe)
    with ThreadPoolExecutor(2) as pool:
        futures = [pool.submit(getattr(module, outer), {"preview_points": n}) for n in (200, 800)]
        assert [f.result() for f in futures] == [200, 800]
    assert get_export_settings() == before


def test_corrupt_or_untrusted_browser_values_cannot_bypass_limits():
    settings = load_export_settings({"plot_dpi": 9999, "preview_points": -1,
        "conversion_memory_limit_mb": 100000, "control_font_size_px": 99,
        "theme_color": "url(https://example.org)", "theme_mode": "invalid",
        "dev_mode": "false", "raw_compression": ["zip"], "unknown": True})
    assert settings == ExportSettings()
    assert load_export_settings(None) == ExportSettings()
    assert load_export_settings({"theme_color": "#ABCDEF"}).theme_color == "#abcdef"
    with settings_scope(replace(settings, preview_points=700)):
        assert preview_point_limit() == 700
    assert asdict(settings) == asdict(ExportSettings())
