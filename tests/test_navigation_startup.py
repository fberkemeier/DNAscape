"""Shared warm imports remove cold navigation without mounting visitor pages."""

import asyncio
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from nicegui.testing import User, user_simulation

from app import pages
from app.main import index
from dnascape import _resources as resources


def test_hosted_preload_is_shared_idempotent_and_budgeted(monkeypatch):
    monkeypatch.setenv("RENDER", "true")
    loaded = {}
    monkeypatch.setattr(pages, "sys", SimpleNamespace(modules=loaded))
    calls, budgets = [], []

    def load(name):
        calls.append(name)
        loaded[name] = object()

    monkeypatch.setattr(pages, "import_module", load)
    monkeypatch.setattr(resources, "ensure_capacity", lambda amount, label: budgets.append(amount))
    assert pages.preload_workflow_pages()
    assert calls == ["app.pages." + name for name in (
        "mapping", "data_management", "simulation", "plotting", "cached_files", "resources")]
    assert budgets == [128 * resources.MIB]
    assert pages.preload_workflow_pages()
    assert len(calls) == 6 and len(budgets) == 1


def test_low_memory_keeps_render_lazy_without_loading_more_modules(monkeypatch):
    monkeypatch.setenv("RENDER", "true")
    monkeypatch.setattr(pages, "sys", SimpleNamespace(modules={}))
    monkeypatch.setattr(resources, "memory_status", lambda: resources.MemoryStatus(
        512 * resources.MIB, 400 * resources.MIB, 85 * resources.MIB))
    monkeypatch.setattr(pages, "import_module", lambda name: pytest.fail("Imported despite low memory"))
    assert not pages.preload_workflow_pages()


def test_local_startup_keeps_selective_preloading(monkeypatch):
    monkeypatch.delenv("RENDER", raising=False)
    monkeypatch.setattr(pages, "sys", SimpleNamespace(modules={}))
    calls = []
    monkeypatch.setattr(pages, "import_module", calls.append)
    assert pages.preload_workflow_pages()
    assert calls == ["app.pages.mapping", "app.pages.data_management"]


def test_real_render_startup_warms_modules_without_creating_ui():
    program = '''
from unittest.mock import patch
import sys
from nicegui import Client
from nicegui.slot import Slot
import app.main as shell
from dnascape import _resources as resources
from dnascape.workflows.export_settings import get_export_settings
# Known headroom, independent of the test runner's own memory use.
resources.memory_status = lambda: resources.MemoryStatus(
    512 * resources.MIB, 128 * resources.MIB, 85 * resources.MIB)
assert not Slot.get_stack()
before = {key: tuple(client.elements) for key, client in Client.instances.items()}
get_export_settings()
assert not Slot.get_stack()
with patch.object(shell.ui, 'run') as run:
    shell.main()
assert all('app.pages.' + name in sys.modules for name in
           ('mapping','data_management','simulation','plotting','cached_files','resources'))
assert {key: tuple(client.elements) for key, client in Client.instances.items()} == before
assert not Slot.get_stack()
run.assert_called_once()
'''
    env = {**os.environ, "RENDER": "true"}
    result = subprocess.run([sys.executable, "-c", program], env=env,
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def test_warm_modules_keep_pages_lazy_and_visitors_separate(monkeypatch):
    monkeypatch.setenv("RENDER", "true")
    pages.preload_workflow_pages()

    async def scenario():
        async with user_simulation(index) as first:
            second = User(first.http_client)
            await first.open("/")
            await second.open("/")
            for visitor in (first, second):
                hosts = [element for element in visitor.client.elements.values()
                         if "page-host-mapping" in element._markers]
                assert not hosts[0].default_slot.children
            first.find(marker="nav-simulation").click()
            with first.client:
                control = next(iter(first.find(marker="simulation-region-end").elements))
                control.set_value(123)
            first.find(marker="nav-overview").click()
            first.find(marker="nav-simulation").click()
            assert next(iter(first.find(marker="simulation-region-end").elements)) is control
            second.find(marker="nav-simulation").click()
            assert next(iter(second.find(marker="simulation-region-end").elements)).value == 1000
    asyncio.run(scenario())
