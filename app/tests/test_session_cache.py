"""Coverage for the bounded browser-session cache model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.session_cache import SessionCache, estimate_retained_bytes
from app.simulation_exports import ExportArtifact


def _artifact(name: str, content: bytes) -> ExportArtifact:
    return ExportArtifact(content, name, "application/octet-stream")


def test_identical_artifacts_are_refreshed_without_duplicate_memory(
    monkeypatch,
) -> None:
    import app.session_cache as cache_module

    clock = iter((10.0, 20.0))
    monkeypatch.setattr(cache_module.time, "time", lambda: next(clock))
    cache = SessionCache()

    first = cache.add_artifact(
        _artifact("profile.csv.gz", b"values"),
        source="Mapping",
        kind="raw",
    )
    second = cache.add_artifact(
        _artifact("profile.csv.gz", b"values"),
        source="Simulation",
        kind="raw",
    )

    assert first.id == second.id
    assert second.created_at == 20.0
    assert second.source == "Simulation"
    assert second.format_label == "CSV + GZ"
    assert cache.count("generated") == 1
    assert cache.size() == len(b"values")


def test_item_limit_is_enforced_independently_for_each_group() -> None:
    cache = SessionCache(max_items=2)
    for index in range(3):
        cache.add_artifact(
            _artifact(f"plot-{index}.png", bytes([index])),
            source="Plotting",
            kind="plot",
        )
        cache.add_artifact(
            _artifact(f"upload-{index}.csv", bytes([index, index])),
            category="uploads",
            source="Data Manager",
            kind="upload",
        )

    assert [entry.name for entry in cache.list_entries("generated", sort="oldest")] == [
        "plot-1.png",
        "plot-2.png",
    ]
    assert [entry.name for entry in cache.list_entries("uploads", sort="oldest")] == [
        "upload-1.csv",
        "upload-2.csv",
    ]


def test_total_item_and_memory_limits_evict_the_oldest_entries() -> None:
    cache = SessionCache(max_items=10, max_total_items=3, max_total_bytes=8)
    for index in range(4):
        cache.add_artifact(
            _artifact(f"item-{index}.bin", bytes([index, index])),
            source="Plotting",
            kind="raw",
        )

    assert [entry.name for entry in cache.list_entries(None, sort="oldest")] == [
        "item-1.bin",
        "item-2.bin",
        "item-3.bin",
    ]

    cache.add_artifact(
        _artifact("larger.bin", b"12345"),
        category="datasets",
        source="Data Manager",
        kind="dataset",
    )
    assert [entry.name for entry in cache.list_entries(None, sort="oldest")] == [
        "item-3.bin",
        "larger.bin",
    ]
    assert cache.size() == 7


def test_artifact_memory_includes_payload_without_double_counting_shared_bytes() -> None:
    content = b"profile"
    values = np.arange(4, dtype=np.float64)
    cache = SessionCache()
    entry = cache.add_artifact(
        _artifact("profile.npy", content),
        category="uploads",
        source="Mapping",
        kind="upload",
        payload={"content": content, "values": values},
    )

    assert entry.size_bytes == len(content) + values.nbytes


def test_filter_sort_remove_and_clear_are_functional(monkeypatch) -> None:
    import app.session_cache as cache_module

    clock = iter((10.0, 20.0, 30.0))
    monkeypatch.setattr(cache_module.time, "time", lambda: next(clock))
    cache = SessionCache()
    raw = cache.add_artifact(
        _artifact("small.npy", b"1"),
        source="Simulation",
        kind="raw",
        detail="Replication timing",
    )
    plot = cache.add_artifact(
        _artifact("large.png", b"12345"),
        source="Mapping",
        kind="plot",
        detail="Firing rate",
    )
    cache.add_artifact(
        _artifact("medium.csv", b"123"),
        source="Mapping",
        kind="raw",
        detail="Fork speed",
    )

    assert cache.list_entries("generated", query="firing") == [plot]
    assert cache.list_entries("generated", kind="plot") == [plot]
    assert [entry.name for entry in cache.list_entries("generated", sort="largest")] == [
        "large.png",
        "medium.csv",
        "small.npy",
    ]
    assert cache.remove(raw.id) is True
    assert cache.remove(raw.id) is False
    assert cache.clear("generated") == 2
    assert cache.count("generated") == 0


@dataclass(frozen=True)
class _Result:
    position: np.ndarray
    outputs: dict[str, np.ndarray]


def test_workflow_references_are_measured_and_consumed_once() -> None:
    result = _Result(
        position=np.arange(5, dtype=np.float64),
        outputs={"profile": np.arange(3, dtype=np.float32)},
    )
    assert estimate_retained_bytes(result) == 5 * 8 + 3 * 4

    cache = SessionCache()
    entry = cache.add_reference(
        name="Simulation result",
        source="Simulation",
        detail="10 replicates",
        payload=result,
        navigate_to="simulation",
    )

    assert entry.size_bytes == 52
    assert entry.format_label == "RESULT"
    assert cache.activate(entry.id) is entry
    assert cache.consume_active("simulation") is entry
    assert cache.consume_active("simulation") is None
