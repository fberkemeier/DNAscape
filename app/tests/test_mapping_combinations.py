from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest

from app.pages import mapping as mapping_page
from app.pages.mapping import (
    _plan_mapping_cases,
    _PreparedBatchInput,
    _validate_mapping_case_axes,
)


def _prepared(
    key: str,
    label: str,
    values: Sequence[Any],
    *,
    form: str = "scalar",
    components: Sequence[str] = (),
) -> _PreparedBatchInput:
    materialized = tuple(values)
    return _PreparedBatchInput(
        key=key,
        label=label,
        form=form,  # type: ignore[arg-type]
        values=materialized,
        prewindowed=(False,) * len(materialized),
        components=tuple(components),
    )


def test_pairwise_cases_broadcast_singleton_and_use_target_first_names() -> None:
    timing = _prepared(
        "replication_timing",
        "Replication timing",
        (np.arange(3.0), np.arange(5.0)),
        form="x",
        components=("sample_a", "sample_b"),
    )
    speed = _prepared(
        "fork_speed",
        "Fork speed",
        (1.4,),
        components=("1.4 kb/min",),
    )

    cases = _plan_mapping_cases(
        (timing, speed),
        target_label="Firing rate",
        combination_mode="pairwise",
    )

    assert tuple(case.item_indices for case in cases) == ((0, 0), (1, 0))
    assert tuple(case.name for case in cases) == (
        "Firing rate · Replication timing: sample_a · Fork speed: 1.4 kb/min",
        "Firing rate · Replication timing: sample_b · Fork speed: 1.4 kb/min",
    )


def test_all_combinations_have_deterministic_cartesian_order() -> None:
    timing = _prepared(
        "replication_timing",
        "Replication timing",
        (np.arange(2.0), np.arange(3.0)),
        form="x",
        components=("timing_a", "timing_b"),
    )
    speed = _prepared(
        "fork_speed",
        "Fork speed",
        (1.0, 1.5, 2.0),
        components=("1.0", "1.5", "2.0"),
    )

    cases = _plan_mapping_cases(
        (timing, speed),
        target_label="Firing rate",
        combination_mode="all",
    )

    assert tuple(case.item_indices for case in cases) == (
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 1),
        (1, 2),
    )
    assert tuple(
        ("timing_b" in case.name, value in case.name)
        for case, value in zip(cases, ("1.0", "1.5", "2.0", "1.0", "1.5", "2.0"), strict=True)
    ) == (
        (False, True),
        (False, True),
        (False, True),
        (True, True),
        (True, True),
        (True, True),
    )


def test_case_export_stems_are_distinct_safe_and_preserve_decimals() -> None:
    repeated = _prepared(
        "fork_speed",
        "Fork speed",
        (1.4, 1.4),
        components=("1.4", "1.4"),
    )

    cases = _plan_mapping_cases((repeated,), target_label="Firing rate")
    stems = tuple(case.export_stem for case in cases)

    assert len(set(stems)) == 2
    assert all(stem.startswith("firing-rate-") for stem in stems)
    assert all("1p4" in stem for stem in stems)
    assert all(re.fullmatch(r"[a-z0-9-]+", stem) for stem in stems)
    assert all("." not in stem for stem in stems)


def test_pairwise_cases_reject_unequal_non_singleton_counts() -> None:
    timing = _prepared(
        "replication_timing",
        "Replication timing",
        (np.arange(2.0), np.arange(3.0)),
        form="x",
        components=("timing_a", "timing_b"),
    )
    speed = _prepared(
        "fork_speed",
        "Fork speed",
        (1.0, 1.5, 2.0),
        components=("1.0", "1.5", "2.0"),
    )

    with pytest.raises(ValueError, match="one shared item or the same number"):
        _plan_mapping_cases(
            (timing, speed),
            target_label="Firing rate",
            combination_mode="pairwise",
        )


def test_all_combinations_enforce_cap_before_materializing_product(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = tuple(float(index) for index in range(224))
    first = _prepared("first", "First", values)
    second = _prepared("second", "Second", values)

    def should_not_materialize(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("Cartesian indices were materialized before the cap check")

    monkeypatch.setattr(mapping_page, "plan_combination_indices", should_not_materialize)

    with pytest.raises(ValueError, match=r"50,176 mapping runs.*50,000 combinations"):
        _plan_mapping_cases(
            (first, second),
            target_label="Target",
            combination_mode="all",
        )


def test_axis_validation_uses_each_resolved_case_not_whole_batch() -> None:
    timing = _prepared(
        "replication_timing",
        "Replication timing",
        (np.arange(3.0), np.arange(5.0)),
        form="x",
    )
    speed = _prepared(
        "fork_speed",
        "Fork speed",
        (np.arange(3.0), np.arange(5.0)),
        form="x",
    )

    _validate_mapping_case_axes((timing, speed), (0, 0), "short")
    _validate_mapping_case_axes((timing, speed), (1, 1), "long")

    with pytest.raises(ValueError, match=r"crossed.*inconsistent spatial bins"):
        _validate_mapping_case_axes((timing, speed), (0, 1), "crossed")
