"""Coverage for the bounded browser-session cache model."""

from __future__ import annotations

import gzip
from dataclasses import dataclass

import numpy as np

from app.export_settings import ExportSettings, get_export_settings
from app.mapping_service import MappingResult
from app.pages.cached_files import _download_artifact
from app.session_cache import (
    DEFAULT_MAX_ITEMS,
    DEFAULT_MAX_TOTAL_BYTES,
    DEFAULT_MAX_TOTAL_ITEMS,
    DEFAULT_TRACKED_CATEGORIES,
    SessionCache,
    estimate_retained_bytes,
)
from app.simulation_exports import ExportArtifact
from app.simulation_service import ConstantSimulationResult


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


def test_cached_names_are_unique_and_keep_compound_extensions() -> None:
    cache = SessionCache()
    first = cache.add_artifact(
        _artifact("profile.tsv.gz", b"first"),
        source="Mapping",
        kind="raw",
    )
    second = cache.add_artifact(
        _artifact("profile.tsv.gz", b"second"),
        source="Simulation",
        kind="raw",
    )
    workflow = cache.add_reference(
        name="profile.tsv.gz",
        source="Simulation",
        detail="Result",
        payload={"values": np.arange(2)},
        navigate_to="simulation",
    )

    assert first.name == "profile.tsv.gz"
    assert second.name == "profile (2).tsv.gz"
    assert workflow.name == "profile (3).tsv.gz"
    assert len({entry.name.casefold() for entry in cache.list_entries(None)}) == 3


def test_unkept_items_are_evicted_before_items_marked_to_keep() -> None:
    cache = SessionCache(max_items=2)
    kept = cache.add_artifact(
        _artifact("kept.bin", b"1"),
        source="Mapping",
        kind="raw",
    )
    disposable = cache.add_artifact(
        _artifact("disposable.bin", b"2"),
        source="Mapping",
        kind="raw",
        keep=False,
    )
    newest = cache.add_artifact(
        _artifact("newest.bin", b"3"),
        source="Mapping",
        kind="raw",
        keep=False,
    )

    assert cache.get(kept.id) is not None
    assert cache.get(disposable.id) is None
    assert cache.get(newest.id) is not None
    cache.set_keep(newest.id, True)
    retained = cache.get(newest.id)
    assert retained is not None
    assert retained.keep is True


def test_cached_workflow_download_defaults_to_compressed_tsv() -> None:
    defaults = ExportSettings()
    assert defaults.cached_file_format == "tsv"
    assert defaults.cached_compression == "gzip"

    result = ConstantSimulationResult(
        position_kb=np.asarray([0.0, 1.0, 2.0]),
        replication_timing=np.asarray([3.0, 2.0, 4.0]),
        elapsed_seconds=0.1,
        replicates=2,
        geometry="linear",
        time_grid=np.asarray([0.0, 1.0]),
        outputs={"replication_timing": np.asarray([3.0, 2.0, 4.0])},
    )
    entry = SessionCache().add_reference(
        name="Simulation result",
        source="Simulation",
        detail="2 replicates",
        payload=result,
        navigate_to="simulation",
    )
    settings = get_export_settings()
    previous = (settings.cached_file_format, settings.cached_compression)
    try:
        settings.cached_file_format = "tsv"
        settings.cached_compression = "gzip"
        artifact = _download_artifact(entry)
    finally:
        settings.cached_file_format, settings.cached_compression = previous

    assert artifact.filename == "rsim_rt.tsv.gz"
    text = gzip.decompress(artifact.data).decode("utf-8")
    assert text.splitlines() == ["replication_timing", "3", "2", "4"]


def test_cached_mapping_download_keeps_compact_target_and_dataset_identity() -> None:
    values = np.asarray([0.2, 0.3, 0.4])
    result = MappingResult(
        target="initiation_rate",
        target_key="initiation_rate",
        target_label="Initiation rate",
        method="test",
        output_form="x",
        output_dependence="space",
        primary_output="initiation_rate",
        outputs={"initiation_rate": values},
        output_forms={"initiation_rate": "x"},
        x_axis=np.arange(values.size, dtype=float),
        time_axis=np.asarray([], dtype=float),
        region_labels=(),
        elapsed_seconds=0.1,
        assumptions={},
    )
    entry = SessionCache().add_reference(
        name="Initiation rate Â· H1",
        source="Mapping",
        detail="3 points",
        payload=result,
        navigate_to="mapping",
    )
    settings = get_export_settings()
    previous = (settings.cached_file_format, settings.cached_compression)
    try:
        settings.cached_file_format = "tsv"
        settings.cached_compression = "gzip"
        artifact = _download_artifact(entry)
    finally:
        settings.cached_file_format, settings.cached_compression = previous

    assert artifact.filename == "rmap_ir-h1_x.tsv.gz"


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
        detail="Initiation rate",
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


def test_clear_all_releases_every_category_and_invokes_output_clearers() -> None:
    cache = SessionCache()
    cache.add_artifact(_artifact("raw.npy", b"1"), source="Mapping", kind="raw")
    cache.add_artifact(
        _artifact("upload.csv", b"2"),
        category="uploads",
        source="Mapping",
        kind="upload",
    )
    cleared: list[str] = []
    cache.register_output_clearer("mapping", lambda: cleared.append("mapping"))

    assert cache.clear_all() == 2
    cache.clear_page_outputs()
    assert cache.count() == 0
    assert cleared == ["mapping"]


def test_reset_clears_entries_and_restores_cache_preferences() -> None:
    cache = SessionCache(max_items=4, max_total_items=5, max_total_bytes=64)
    cache.set_tracking("uploads", True)
    cache.set_tracking("workflows", False)
    cache.add_artifact(_artifact("raw.npy", b"1"), source="Mapping", kind="raw")

    assert cache.reset() == 1
    assert cache.count() == 0
    assert cache.max_items == DEFAULT_MAX_ITEMS
    assert cache.max_total_items == DEFAULT_MAX_TOTAL_ITEMS
    assert cache.max_total_bytes == DEFAULT_MAX_TOTAL_BYTES
    assert set(cache.tracked_categories()) == set(DEFAULT_TRACKED_CATEGORIES)


def test_page_refreshers_are_replaced_and_can_be_unregistered() -> None:
    cache = SessionCache()
    refreshed: list[str] = []
    cache.register_page_refresher("cached-files", lambda: refreshed.append("first"))
    cache.register_page_refresher("cached-files", lambda: refreshed.append("latest"))

    cache.refresh_page("cached-files")
    cache.unregister_page_refresher("cached-files")
    cache.refresh_page("cached-files")

    assert refreshed == ["latest"]


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
