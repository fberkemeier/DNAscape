"""Small framework-independent helpers shared by batch scientific workflows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from pathlib import PurePath
from typing import Literal, TypeVar

from app.result_naming import ResultName, combination_result_names

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class BatchSourceSummary:
    """Inspectable metadata for one logical input across several datasets."""

    label: str
    filenames: tuple[str, ...]
    forms: tuple[str | None, ...] = ()
    shapes: tuple[tuple[int, ...] | None, ...] = ()


@dataclass(frozen=True, slots=True)
class BatchPlan:
    """Validated dataset count and stable display names for one shared run."""

    size: int
    names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CombinationSource:
    """Presentation metadata for one input participating in combinations.

    ``components`` contains already formatted file stems or scalar values.  A
    singleton is a shared input, so it is omitted from result names by default.
    """

    label: str
    components: tuple[str, ...]
    include_singleton_in_short: bool = False


CombinationMode = Literal["pairwise", "all"]


def dataset_stem(filename: str) -> str:
    """Return a concise, filesystem-independent dataset label."""

    path = PurePath(str(filename or "dataset").replace("\\", "/"))
    if path.suffix.casefold() == ".gz":
        path = path.with_suffix("")
    return path.stem.strip() or "dataset"


def unique_dataset_names(filenames: Sequence[str]) -> tuple[str, ...]:
    """Create stable, case-insensitively unique labels from source filenames."""

    used: set[str] = set()
    labels: list[str] = []
    for index, filename in enumerate(filenames, start=1):
        base = dataset_stem(filename) or f"dataset_{index}"
        candidate = base
        duplicate = 2
        while candidate.casefold() in used:
            candidate = f"{base}_{duplicate}"
            duplicate += 1
        used.add(candidate.casefold())
        labels.append(candidate)
    return tuple(labels)


def plan_combination_indices(
    source_sizes: Sequence[int],
    *,
    mode: CombinationMode = "pairwise",
) -> tuple[tuple[int, ...], ...]:
    """Return deterministic source indices for pairwise or Cartesian mapping.

    Pairwise mode broadcasts singleton sources and otherwise requires every
    source to have the same cardinality.  ``all`` mode returns the Cartesian
    product in source order, with the rightmost source changing fastest.
    """

    sizes = tuple(int(size) for size in source_sizes)
    if any(size < 1 for size in sizes):
        raise ValueError("Combination sources must each contain at least one value or file.")
    if mode not in {"pairwise", "all"}:
        raise ValueError(f"Unknown combination mode {mode!r}; use 'pairwise' or 'all'.")
    if not sizes:
        return ((),)

    if mode == "all":
        return tuple(product(*(range(size) for size in sizes)))

    result_size = max(sizes)
    incompatible = tuple(size for size in sizes if size not in {1, result_size})
    if incompatible:
        readable = ", ".join(str(size) for size in sizes)
        raise ValueError(
            "Pairwise inputs must contain one shared item or the same number "
            f"of items (received {readable})."
        )
    return tuple(
        tuple(0 if size == 1 else result_index for size in sizes)
        for result_index in range(result_size)
    )


def target_first_combination_names(
    target_label: str,
    sources: Sequence[CombinationSource],
    combinations: Sequence[Sequence[int]],
    *,
    include_singletons: bool = False,
) -> tuple[str, ...]:
    """Build stable, unique names beginning with the mapped target.

    File/value components which actually distinguish combinations are rendered
    as ``"Source: component"`` after the target.  Shared singleton inputs can
    optionally be included for callers that need fully descriptive one-off
    output names.
    """

    return tuple(
        item.full
        for item in combination_names(
            target_label,
            sources,
            combinations,
            include_singletons=include_singletons,
        )
    )


def combination_names(
    target_label: str,
    sources: Sequence[CombinationSource],
    combinations: Sequence[Sequence[int]],
    *,
    include_singletons: bool = False,
) -> tuple[ResultName, ...]:
    """Return both concise and fully descriptive combination labels."""

    return combination_result_names(
        target_label,
        tuple((source.label, source.components) for source in sources),
        combinations,
        include_singletons_in_full=include_singletons,
        short_singleton_indices=tuple(
            index
            for index, source in enumerate(sources)
            if source.include_singleton_in_short
        ),
    )


def plan_batch_sources(
    sources: Sequence[BatchSourceSummary],
    *,
    fallback_name: str = "result",
    require_matching_shapes: bool = True,
) -> BatchPlan:
    """Validate batch cardinality/form/shape and choose dataset names.

    A one-file source is a shared profile and is broadcast across the batch;
    constants never appear here and are naturally shared. Every other uploaded
    source must provide the full batch. Files within one source must represent
    the same dependence. Callers can require one common array shape, or defer
    shape checks until factors have been paired so independent datasets may use
    different spatial extents.
    """

    populated = tuple(source for source in sources if source.filenames)
    if not populated:
        return BatchPlan(size=1, names=(fallback_name,))

    size = max(len(source.filenames) for source in populated)
    for source in populated:
        count = len(source.filenames)
        if count not in {1, size}:
            raise ValueError(
                f"{source.label} has {count} files, but this batch needs either "
                f"one shared file or {size} paired files."
            )
        if source.forms and len(source.forms) != count:
            raise ValueError(f"{source.label} inspection metadata are incomplete.")
        known_forms = {
            str(form)
            for form in source.forms
            if form is not None and str(form) not in {"", "auto", "unknown"}
        }
        if len(known_forms) > 1:
            readable = ", ".join(sorted(known_forms))
            raise ValueError(
                f"{source.label} files do not have one consistent data type ({readable})."
            )
        if source.shapes and len(source.shapes) != count:
            raise ValueError(f"{source.label} shape metadata are incomplete.")
        known_shapes = {tuple(shape) for shape in source.shapes if shape is not None}
        if require_matching_shapes and len(known_shapes) > 1:
            readable = ", ".join(" x ".join(map(str, shape)) or "scalar" for shape in known_shapes)
            raise ValueError(f"{source.label} files have inconsistent array sizes ({readable}).")

    naming_source = next(
        (source for source in populated if len(source.filenames) == size),
        populated[0],
    )
    if size == 1:
        return BatchPlan(size=1, names=unique_dataset_names(naming_source.filenames))
    return BatchPlan(size=size, names=unique_dataset_names(naming_source.filenames))


def batch_item(values: Sequence[T], index: int) -> T:
    """Return the paired item, broadcasting a singleton sequence."""

    if not values:
        raise ValueError("A batch source cannot be empty.")
    if len(values) == 1:
        return values[0]
    try:
        return values[index]
    except IndexError as exc:
        raise ValueError("Batch sources do not contain the same number of datasets.") from exc


def named_batch(values: Sequence[T], plan: BatchPlan) -> Mapping[str, T]:
    """Pair a validated result sequence with its stable dataset labels."""

    if len(values) != plan.size:
        raise ValueError(f"Expected {plan.size} batch results, received {len(values)}.")
    return dict(zip(plan.names, values, strict=True))
