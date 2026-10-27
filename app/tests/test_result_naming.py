"""Focused contracts for shared result labels and portable filename stems."""

from __future__ import annotations

import re

from app.batch_workflows import CombinationSource, combination_names, plan_combination_indices
from app.result_naming import (
    combination_result_names,
    compact_export_stem,
    output_file_token,
    safe_filename_stem,
    unique_compact_export_stems,
    unique_filename_stems,
)


def test_combination_names_offer_concise_and_fully_descriptive_forms() -> None:
    names = combination_result_names(
        "Simulation",
        (
            ("Initiation rate", ("sample_a", "sample_b")),
            ("Fork speed", ("1.4 kb/min",)),
        ),
        ((0, 0), (1, 0)),
    )

    assert tuple(name.short for name in names) == (
        "Simulation · sample_a",
        "Simulation · sample_b",
    )
    assert tuple(name.full for name in names) == (
        "Simulation · Initiation rate: sample_a · Fork speed: 1.4 kb/min",
        "Simulation · Initiation rate: sample_b · Fork speed: 1.4 kb/min",
    )
    assert names[0].display("short") == names[0].short
    assert names[0].display("FULL") == names[0].full


def test_short_names_omit_shared_singletons_but_remain_unique() -> None:
    names = combination_result_names(
        "Initiation rate",
        (
            ("Replication timing", ("cell_a", "cell_b")),
            ("Fork speed", ("1.4 kb/min",)),
            ("Geometry", ("linear",)),
        ),
        ((0, 0, 0), (1, 0, 0)),
    )

    assert tuple(name.short for name in names) == (
        "Initiation rate · cell_a",
        "Initiation rate · cell_b",
    )
    assert all("1.4" not in name.short and "linear" not in name.short for name in names)
    assert all("Fork speed: 1.4 kb/min" in name.full for name in names)
    assert all("Geometry: linear" in name.full for name in names)


def test_short_names_keep_a_single_uploaded_profile_but_omit_fixed_scalars() -> None:
    names = combination_names(
        "Initiation rate",
        (
            CombinationSource(
                "Replication timing",
                ("sample_a",),
                include_singleton_in_short=True,
            ),
            CombinationSource("Fork speed", ("1.4 kb/min",)),
        ),
        ((0, 0),),
        include_singletons=True,
    )

    assert names[0].short == "Initiation rate · sample_a"
    assert names[0].full.endswith("Fork speed: 1.4 kb/min")


def test_cartesian_combination_names_follow_planner_order_deterministically() -> None:
    combinations = plan_combination_indices((2, 3), mode="all")
    first = combination_result_names(
        "Initiation rate",
        (
            ("Replication timing", ("timing_a", "timing_b")),
            ("Fork speed", ("1.0 kb/min", "1.5 kb/min", "2.0 kb/min")),
        ),
        combinations,
    )
    second = combination_result_names(
        "Initiation rate",
        (
            ("Replication timing", ("timing_a", "timing_b")),
            ("Fork speed", ("1.0 kb/min", "1.5 kb/min", "2.0 kb/min")),
        ),
        combinations,
    )

    assert first == second
    assert tuple(name.short for name in first) == (
        "Initiation rate · timing_a + 1.0",
        "Initiation rate · timing_a + 1.5",
        "Initiation rate · timing_a + 2.0",
        "Initiation rate · timing_b + 1.0",
        "Initiation rate · timing_b + 1.5",
        "Initiation rate · timing_b + 2.0",
    )


def test_filename_stems_are_collision_safe_and_preserve_decimals() -> None:
    stems = unique_filename_stems(
        (
            "Fork speed: 1.4 kb/min",
            "Fork speed: 1.4 kb/min",
            "Förk speed: 1.4 kb/min",
        )
    )

    assert stems == (
        "fork-speed-1p4-kb-min",
        "fork-speed-1p4-kb-min-2",
        "fork-speed-1p4-kb-min-3",
    )
    assert all(re.fullmatch(r"[a-z0-9-]+", stem) for stem in stems)
    assert all("1p4" in stem and "." not in stem for stem in stems)


def test_filename_stems_fold_unicode_and_use_a_readable_fallback() -> None:
    assert safe_filename_stem("Réplication Ånalysis") == "replication-analysis"
    assert safe_filename_stem("数据", fallback="Mapping result") == "mapping-result"


def test_export_stems_use_shared_scientific_acronyms() -> None:
    assert output_file_token("replication_timing") == "rt"
    assert output_file_token("initiation_rate") == "ir"
    assert output_file_token("fork_speed") == "fs"
    assert output_file_token("inter_origin_distances") == "iod"
    assert output_file_token("replicated_fraction_xt") == "rf-xt"
    assert compact_export_stem("Initiation rate / Replication timing H1") == "ir-rt-h1"
    assert compact_export_stem("Initiation rate Â· H1") == "ir-h1"


def test_compact_export_stems_remain_unique_after_abbreviation() -> None:
    assert unique_compact_export_stems(
        ("Replication timing", "rt", "Replication timing"),
    ) == ("rt", "rt-2", "rt-3")


def test_long_filename_stems_are_bounded_hashed_and_still_unique() -> None:
    shared_prefix = "replication-profile-" * 20
    first = safe_filename_stem(f"{shared_prefix}alpha", maximum=48)
    repeated = safe_filename_stem(f"{shared_prefix}alpha", maximum=48)
    second = safe_filename_stem(f"{shared_prefix}beta", maximum=48)
    duplicates = unique_filename_stems(
        (f"{shared_prefix}alpha", f"{shared_prefix}alpha"),
        maximum=48,
    )

    assert first == repeated
    assert first != second
    assert len(first) <= 48
    assert re.search(r"-[0-9a-f]{8}$", first)
    assert duplicates[0] == first
    assert duplicates[1].endswith("-2")
    assert len(duplicates[1]) <= 48
