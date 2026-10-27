"""Selective startup imports keep expensive work away from first navigation."""

import subprocess
import sys
from pathlib import Path
from typing import cast


def test_workflow_registry_does_not_import_scientific_dependencies() -> None:
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import app.pages; "
            "loaded = set(sys.modules); "
            "assert not loaded.intersection({'numpy', 'pandas', 'scipy', 'matplotlib'}); "
            "assert not any(name.startswith('app.pages.') for name in loaded)",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_startup_preloads_expensive_modules_before_serving_without_creating_ui() -> None:
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "-c", """
import sys
from unittest.mock import patch

import app.main as shell
from nicegui import Client
from dnascape.workflows.mapping_catalog import get_mapping_catalog

assert 'app.pages.mapping' not in sys.modules
assert 'app.pages.data_management' not in sys.modules
before = {key: tuple(client.elements) for key, client in Client.instances.items()}

def start_server(**kwargs):
    assert 'app.pages.mapping' in sys.modules
    assert 'app.pages.data_management' in sys.modules
    assert get_mapping_catalog.cache_info().currsize == 1
    assert not {'app.pages.simulation', 'app.pages.plotting'} & sys.modules.keys()
    assert {key: tuple(client.elements) for key, client in Client.instances.items()} == before

with patch.object(shell.ui, 'run', side_effect=start_server) as run:
    shell.main()
    run.assert_called_once()

catalog = get_mapping_catalog()
shell.preload_workflow_pages()
assert get_mapping_catalog() is catalog
assert {key: tuple(client.elements) for key, client in Client.instances.items()} == before
"""],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_preloaded_pages_still_build_on_demand_with_independent_client_controls() -> None:
    import asyncio

    from nicegui.elements.select import Select
    from nicegui.testing import user_simulation

    from app.main import index
    from app.pages import preload_workflow_pages

    preload_workflow_pages()

    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            assert user.client is not None
            for slug in ("mapping", "data-management"):
                host = next(
                    element for element in user.client.elements.values()
                    if f"page-host-{slug}" in element._markers
                )
                assert not host.default_slot.children

            user.find(marker="nav-mapping").click()
            target = cast(Select, next(iter(user.find(marker="mapping-target").elements)))
            assert target.value == "initiation_rate"
            assert user.client is not None
            first_client = user.client
            with first_client:
                target.set_value("fork_speed")
            await asyncio.sleep(0)
            await asyncio.sleep(0)

            # Rewarming must not rebuild a page or reset its controls.
            preload_workflow_pages()
            assert target.value == "fork_speed"
            user.find(marker="nav-data-management").click()
            await user.should_see(marker="data-manager-upload")

            await user.open("/")
            assert user.client is not None
            assert user.client is not first_client
            for slug in ("mapping", "data-management"):
                host = next(
                    element for element in user.client.elements.values()
                    if f"page-host-{slug}" in element._markers
                )
                assert not host.default_slot.children
            user.find(marker="nav-mapping").click()
            fresh_target = cast(Select, next(iter(user.find(marker="mapping-target").elements)))
            assert fresh_target is not target
            assert fresh_target.value == "initiation_rate"
            assert target.value == "fork_speed"

    asyncio.run(scenario())
