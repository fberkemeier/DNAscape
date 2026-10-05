"""Shared concise display names and safe filenames for app workflows."""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

NameStyle = Literal["short", "full"]

# Automatic downloads identify the workflow which produced them without
# repeating the DNAscape product name in every filename.
RDATA_FILE_PREFIX = "rdata"
RMAP_FILE_PREFIX = "rmap"
RSIM_FILE_PREFIX = "rsim"
PLOTTING_FILE_PREFIX = "plot"

# Stable, human-readable abbreviations shared by raw data, plot captures and
# batch archives. Unknown package outputs remain readable through the normal
# portable-stem fallback instead of being silently discarded.
OUTPUT_FILE_TOKENS: dict[str, str] = {
    "replication-timing": "rt",
    "replication-timing-fit": "rt-fit",
    "firing-rate": "fr",
    "fork-speed": "fs",
    "fork-directionality": "fd",
    "inter-origin-distance": "iod",
    "inter-origin-distances": "iod",
    "origin-efficiency": "oe",
    "completion-time": "ct",
    "s-phase-duration": "spd",
    "num-origins": "nori",
    "number-of-origins": "nori",
    "origins-per-replicate": "nori",
    "replicated-fraction": "rf",
    "replicated-fraction-xt": "rf-xt",
    "total-forks": "tf",
    "active-forks": "af",
    "stalled-forks": "sf",
    "firing-factors": "ff",
    "initiation-events": "init",
    "initiation-events-xt": "init-xt",
    "initiation-pdf": "ipdf",
    "initiation-pdf-xt": "ipdf-xt",
    "initiation-probability-density": "ipdf",
    "coalescence-events": "coal",
    "coalescence-events-xt": "coal-xt",
    "coalescence-pdf": "cpdf",
    "coalescence-pdf-xt": "cpdf-xt",
    "coalescence-probability-density": "cpdf",
    "right-moving-fork-density": "rfd",
    "right-moving-fork-density-xt": "rfd-xt",
    "right-fork-density": "rfd",
    "right-fork-density-xt": "rfd-xt",
    "left-moving-fork-density": "lfd",
    "left-moving-fork-density-xt": "lfd-xt",
    "left-fork-density": "lfd",
    "left-fork-density-xt": "lfd-xt",
    "fork-density": "fdens",
    "fork-density-xt": "fdens-xt",
    "fork-densities": "fdens",
    "fork-densities-xt": "fdens-xt",
    "directional-fork-densities": "fdens",
    "kinetic-fields": "kin",
    "joint-kinetic-fields": "kin",
    "fit-history": "mse",
    "fit-diagnostic": "res",
    "mse-diagnostic": "mse",
    "residual": "res",
}

NAME_STYLE_OPTIONS: dict[str, str] = {
    "short": "Short",
    "full": "Full",
}

_SEPARATOR = " · "


@dataclass(frozen=True, slots=True)
class ResultName:
    """The compact and fully descriptive labels for one resolved result."""

    short: str
    full: str

    def display(self, style: str | None = None) -> str:
        """Return the requested presentation, defaulting safely to compact."""

        return self.full if normalize_name_style(style) == "full" else self.short


def normalize_name_style(value: str | None) -> NameStyle:
    """Normalize a settings value to one supported naming presentation."""

    return "full" if str(value or "short").strip().casefold() == "full" else "short"


def _unique_labels(values: Sequence[str]) -> tuple[str, ...]:
    """Make labels unique without making the common case verbose."""

    used: set[str] = set()
    result: list[str] = []
    for raw_value in values:
        base = str(raw_value).strip() or "Result"
        candidate = base
        duplicate = 2
        while candidate.casefold() in used:
            candidate = f"{base}_{duplicate}"
            duplicate += 1
        used.add(candidate.casefold())
        result.append(candidate)
    return tuple(result)


def compact_component(value: str, *, maximum: int = 36) -> str:
    """Reduce one file/value component while retaining its distinguishing part."""

    component = " ".join(str(value).strip().split()) or "Dataset"
    first, _separator, _rest = component.partition(" ")
    try:
        numeric = float(first)
    except ValueError:
        numeric = math.nan
    if math.isfinite(numeric):
        component = first
    if len(component) <= maximum:
        return component
    # Keep both the descriptive prefix and chromosome/replicate suffix.  Those
    # suffixes are often the only difference between related genomic files.
    tail_length = max(8, maximum // 3)
    head_length = max(1, maximum - tail_length - 1)
    return f"{component[:head_length].rstrip()}…{component[-tail_length:].lstrip()}"


def combination_result_names(
    target_label: str,
    sources: Sequence[tuple[str, Sequence[str]]],
    combinations: Sequence[Sequence[int]],
    *,
    include_singletons_in_full: bool = True,
    short_singleton_indices: Sequence[int] = (),
) -> tuple[ResultName, ...]:
    """Build concise and full target-first names for resolved combinations.

    Short labels include only inputs which distinguish results. Full labels
    retain source labels and, by default, shared singleton assumptions.
    """

    target = str(target_label).strip() or "Result"
    short_singletons = {int(index) for index in short_singleton_indices}
    normalized: list[tuple[str, tuple[str, ...]]] = []
    for source_index, (source_label, components) in enumerate(sources, start=1):
        label = str(source_label).strip() or f"Input {source_index}"
        values = tuple(str(component).strip() for component in components)
        if not values:
            raise ValueError(f"{label} must contain at least one naming component.")
        if any(not component for component in values):
            raise ValueError(f"{label} contains an empty naming component.")
        normalized.append((label, values))

    short_candidates: list[str] = []
    full_candidates: list[str] = []
    for combination in combinations:
        indices = tuple(int(index) for index in combination)
        if len(indices) != len(normalized):
            raise ValueError(
                "A combination index tuple must contain one index for every source."
            )
        short_parts: list[str] = []
        full_parts = [target]
        for source_index, (item_index, (label, components)) in enumerate(
            zip(indices, normalized, strict=True)
        ):
            if item_index < 0 or item_index >= len(components):
                raise ValueError(
                    f"Combination index {item_index} is out of range for {label}."
                )
            component = components[item_index]
            if len(components) > 1 or source_index in short_singletons:
                compact = compact_component(component)
                if compact.casefold() not in {
                    existing.casefold() for existing in short_parts
                }:
                    short_parts.append(compact)
            if include_singletons_in_full or len(components) > 1:
                full_parts.append(f"{label}: {component}")
        short_candidates.append(
            target
            if not short_parts
            else f"{target}{_SEPARATOR}{' + '.join(short_parts)}"
        )
        full_candidates.append(_SEPARATOR.join(full_parts))

    short_names = _unique_labels(short_candidates)
    full_names = _unique_labels(full_candidates)
    return tuple(
        ResultName(short=short, full=full)
        for short, full in zip(short_names, full_names, strict=True)
    )


def safe_filename_stem(
    value: str,
    *,
    fallback: str = "result",
    maximum: int = 120,
) -> str:
    """Return a bounded portable stem, preserving readable decimal values."""

    normalized = unicodedata.normalize("NFKD", str(value))
    normalized = normalized.encode("ascii", "ignore").decode("ascii")
    normalized = re.sub(r"(?<=\d)\.(?=\d)", "p", normalized)
    normalized = re.sub(r"[^A-Za-z0-9]+", "-", normalized).strip("-").casefold()
    fallback_stem = re.sub(
        r"[^A-Za-z0-9]+",
        "-",
        str(fallback),
    ).strip("-").casefold() or "result"
    if not normalized:
        normalized = fallback_stem
    if len(normalized) <= maximum:
        return normalized
    digest = hashlib.sha256(normalized.encode("ascii")).hexdigest()[:8]
    prefix_length = max(1, maximum - len(digest) - 1)
    return f"{normalized[:prefix_length].rstrip('-')}-{digest}"


def output_file_token(output_key: str, *, fallback: str = "data") -> str:
    """Return the shared concise token for one scientific output key."""

    stem = safe_filename_stem(output_key, fallback=fallback)
    return OUTPUT_FILE_TOKENS.get(stem, stem)


def compact_export_stem(
    value: str,
    *,
    fallback: str = "result",
    maximum: int = 120,
) -> str:
    """Build a portable stem and abbreviate known scientific terms within it."""

    # Treat both a real middle dot and the historical mojibake spelling as
    # separators; neither should leak an artificial ``a`` into downloads.
    cleaned = str(value).replace("Â·", " ").replace("·", " ")
    stem = safe_filename_stem(cleaned, fallback=fallback, maximum=max(256, maximum * 2))
    for phrase, token in sorted(
        OUTPUT_FILE_TOKENS.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        stem = re.sub(
            rf"(?:^|-)({re.escape(phrase)})(?=-|$)",
            lambda match, replacement=token: (
                replacement if match.start() == 0 else f"-{replacement}"
            ),
            stem,
        )
    return safe_filename_stem(stem, fallback=fallback, maximum=maximum)


def unique_compact_export_stems(
    values: Sequence[str],
    *,
    fallback: str = "result",
    maximum: int = 120,
) -> tuple[str, ...]:
    """Create collision-safe compact scientific stems in source order."""

    return unique_filename_stems(
        tuple(
            compact_export_stem(
                value,
                fallback=f"{fallback}-{index}",
                maximum=maximum,
            )
            for index, value in enumerate(values, start=1)
        ),
        fallback=fallback,
        maximum=maximum,
    )


def unique_filename_stems(
    values: Sequence[str],
    *,
    fallback: str = "result",
    maximum: int = 120,
) -> tuple[str, ...]:
    """Create collision-safe stems in source order."""

    used: set[str] = set()
    stems: list[str] = []
    for index, value in enumerate(values, start=1):
        base = safe_filename_stem(
            value,
            fallback=f"{fallback}-{index}",
            maximum=maximum,
        )
        candidate = base
        duplicate = 2
        while candidate.casefold() in used:
            suffix = f"-{duplicate}"
            candidate = f"{base[: maximum - len(suffix)].rstrip('-')}{suffix}"
            duplicate += 1
        used.add(candidate.casefold())
        stems.append(candidate)
    return tuple(stems)


__all__ = [
    "NAME_STYLE_OPTIONS",
    "OUTPUT_FILE_TOKENS",
    "PLOTTING_FILE_PREFIX",
    "RDATA_FILE_PREFIX",
    "RMAP_FILE_PREFIX",
    "RSIM_FILE_PREFIX",
    "NameStyle",
    "ResultName",
    "combination_result_names",
    "compact_component",
    "compact_export_stem",
    "normalize_name_style",
    "output_file_token",
    "safe_filename_stem",
    "unique_compact_export_stems",
    "unique_filename_stems",
]
