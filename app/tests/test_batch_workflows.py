from __future__ import annotations

import pytest

from app.artifact_archives import build_artifact_archive
from app.batch_workflows import (
    BatchSourceSummary,
    CombinationSource,
    batch_item,
    plan_batch_sources,
    plan_combination_indices,
    target_first_combination_names,
    unique_dataset_names,
)
from app.simulation_exports import ExportArtifact


def test_batch_plan_broadcasts_singletons_and_uses_multi_source_names() -> None:
    plan = plan_batch_sources(
        (
            BatchSourceSummary(
                "Replication timing",
                ("cell_a.tsv", "cell_b.tsv"),
                ("x", "x"),
                ((3,), (3,)),
            ),
            BatchSourceSummary("Fork speed", ("shared.tsv",), ("x",), ((3,),)),
        )
    )

    assert plan.size == 2
    assert plan.names == ("cell_a", "cell_b")
    assert batch_item((1,), 1) == 1
    assert batch_item((1, 2), 1) == 2


@pytest.mark.parametrize(
    "source, message",
    (
        (
            BatchSourceSummary(
                "Input",
                ("a.tsv", "b.tsv"),
                ("x", "t"),
                ((3,), (3,)),
            ),
            "consistent data type",
        ),
        (
            BatchSourceSummary(
                "Input",
                ("a.tsv", "b.tsv"),
                ("x", "x"),
                ((3,), (4,)),
            ),
            "inconsistent array sizes",
        ),
    ),
)
def test_batch_plan_rejects_inconsistent_files(
    source: BatchSourceSummary,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        plan_batch_sources((source,))


def test_batch_plan_rejects_unpaired_multi_sources() -> None:
    with pytest.raises(ValueError, match="one shared file or 3 paired files"):
        plan_batch_sources(
            (
                BatchSourceSummary("A", ("a", "b", "c")),
                BatchSourceSummary("B", ("a", "b")),
            )
        )


def test_mapping_style_batch_can_allow_independent_dataset_lengths() -> None:
    plan = plan_batch_sources(
        (
            BatchSourceSummary(
                "Input",
                ("short.tsv", "long.tsv"),
                ("x", "x"),
                ((3,), (5,)),
            ),
        ),
        require_matching_shapes=False,
    )

    assert plan.names == ("short", "long")


def test_dataset_names_remove_double_extensions_and_are_unique() -> None:
    assert unique_dataset_names(("sample.tsv.gz", "sample.csv")) == (
        "sample",
        "sample_2",
    )


def test_pairwise_combinations_broadcast_singletons_in_source_order() -> None:
    assert plan_combination_indices((3, 1, 3)) == (
        (0, 0, 0),
        (1, 0, 1),
        (2, 0, 2),
    )


def test_pairwise_combinations_reject_incompatible_counts() -> None:
    with pytest.raises(ValueError, match="one shared item or the same number"):
        plan_combination_indices((3, 2), mode="pairwise")


def test_all_combinations_return_cartesian_indices() -> None:
    assert plan_combination_indices((2, 3, 1), mode="all") == (
        (0, 0, 0),
        (0, 1, 0),
        (0, 2, 0),
        (1, 0, 0),
        (1, 1, 0),
        (1, 2, 0),
    )


def test_target_first_names_include_each_varying_file_or_value() -> None:
    combinations = plan_combination_indices((2, 2, 1))
    names = target_first_combination_names(
        "Firing rate",
        (
            CombinationSource(
                "Replication timing",
                ("chr5_rt_70-130", "chr3_rt_70-130"),
            ),
            CombinationSource("Fork speed", ("1.4", "1.8")),
            CombinationSource("Shared setting", ("linear",)),
        ),
        combinations,
    )

    assert names == (
        "Firing rate · Replication timing: chr5_rt_70-130 · Fork speed: 1.4",
        "Firing rate · Replication timing: chr3_rt_70-130 · Fork speed: 1.8",
    )


def test_target_first_names_can_include_shared_inputs_and_remain_unique() -> None:
    source = CombinationSource("Fork speed", ("1.4", "1.4"))
    names = target_first_combination_names(
        "Firing rate",
        (source,),
        ((0,), (1,)),
        include_singletons=True,
    )

    assert names == (
        "Firing rate · Fork speed: 1.4",
        "Firing rate · Fork speed: 1.4_2",
    )


@pytest.mark.parametrize("mode", ("pairwise", "all"))
def test_combination_planner_rejects_empty_sources(mode: str) -> None:
    with pytest.raises(ValueError, match="at least one"):
        plan_combination_indices((2, 0), mode=mode)  # type: ignore[arg-type]


def test_archive_consumes_generators_and_bounds_bulk_source_bytes() -> None:
    artifacts = (
        ExportArtifact(b"1234", f"result_{index}.tsv", "text/tab-separated-values")
        for index in range(2)
    )
    archive = build_artifact_archive(artifacts, max_source_bytes=8)
    assert archive.filename == "dnascape_generated_files.zip"

    oversized = (
        ExportArtifact(b"12345", f"result_{index}.tsv", "text/tab-separated-values")
        for index in range(2)
    )
    with pytest.raises(ValueError, match="Download the files individually"):
        build_artifact_archive(oversized, max_source_bytes=8)
