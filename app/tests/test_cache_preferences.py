"""Session-local automatic cache tracking preferences."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
from nicegui.elements.checkbox import Checkbox
from nicegui.elements.dialog import Dialog
from nicegui.testing import user_simulation

from app.main import index
from app.session_cache import (
    SessionCache,
    cache_artifact,
    cache_dataset,
    cache_upload,
    cache_workflow,
    current_session_cache,
)
from app.simulation_exports import ExportArtifact


def _artifact(name: str, data: bytes = b"data") -> ExportArtifact:
    return ExportArtifact(data, name, "application/octet-stream")


def test_high_level_recording_respects_preferences_without_gating_explicit_adds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.session_cache as cache_module

    cache = SessionCache()
    monkeypatch.setattr(cache_module, "current_session_cache", lambda: cache)

    assert cache.tracked_categories() == ("generated", "datasets", "workflows")
    assert not cache.is_tracking("uploads")

    # Explicit storage remains available to internal callers and restores; the
    # preference gates only automatic high-level recording.
    existing_upload = cache.add_artifact(
        _artifact("restored.csv"),
        category="uploads",
        source="Test",
        kind="upload",
    )
    upload = SimpleNamespace(
        content=b"position,value\n0,1\n",
        name="new.csv",
        content_type="text/csv",
        shape=(1,),
        inferred_form="x",
        source_format="csv",
    )
    assert cache_upload(upload, source="Mapping", navigate_to="mapping") is None
    assert cache.count("uploads") == 1
    assert cache.get(existing_upload.id) is existing_upload

    cache.set_tracking("uploads", True)
    recorded_upload = cache_upload(upload, source="Mapping", navigate_to="mapping")
    assert recorded_upload is not None
    assert cache.count("uploads") == 2

    cache.set_tracking("generated", False)
    assert cache_artifact(_artifact("plot.png"), source="Plotting", kind="plot") is None
    assert cache.count("generated") == 0

    dataset = SimpleNamespace(
        artifact=_artifact("prepared.tsv"),
        shape=(3,),
        data_form="x",
    )
    recorded_dataset = cache_dataset(dataset)
    assert recorded_dataset is not None
    cache.set_tracking("datasets", False)
    assert cache_dataset(dataset) is None
    assert cache.get(recorded_dataset.id) is recorded_dataset

    result = SimpleNamespace(values=np.arange(3))
    recorded_workflow = cache_workflow(
        result,
        name="Simulation result",
        source="Simulation",
        detail="3 points",
        navigate_to="simulation",
    )
    assert recorded_workflow is not None
    cache.set_tracking("workflows", False)
    assert (
        cache_workflow(
            result,
            name="Another simulation",
            source="Simulation",
            detail="3 points",
            navigate_to="simulation",
        )
        is None
    )
    assert cache.get(recorded_workflow.id) is recorded_workflow


def test_cached_files_exposes_session_local_tracking_controls() -> None:
    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-cached-files").click()
            await user.should_see(marker="cache-settings-button")
            dialog = cast(
                Dialog,
                next(iter(user.find(marker="cache-settings-dialog").elements)),
            )
            assert dialog.value is False
            user.find(marker="cache-settings-button").click()
            await asyncio.sleep(0)
            assert dialog.value is True
            await user.should_see(marker="cache-tracking-settings")
            assert user.client is not None

            controls = {
                category: cast(
                    Checkbox,
                    next(iter(user.find(marker=f"cache-track-{category}").elements)),
                )
                for category in ("generated", "uploads", "datasets", "workflows")
            }
            assert controls["generated"].value is True
            assert controls["uploads"].value is False
            assert controls["datasets"].value is True
            assert controls["workflows"].value is True

            with user.client:
                cache = current_session_cache()
                controls["uploads"].set_value(True)
                controls["generated"].set_value(False)
            await asyncio.sleep(0)

            assert cache.is_tracking("uploads")
            assert not cache.is_tracking("generated")
            assert cache.is_tracking("datasets")
            assert cache.is_tracking("workflows")

            user.find(marker="cache-settings-close").click()
            await asyncio.sleep(0)
            assert dialog.value is False
            user.find(marker="cache-settings-button").click()
            await asyncio.sleep(0)
            assert dialog.value is True
            assert controls["uploads"].value is True
            assert controls["generated"].value is False

    asyncio.run(scenario())
