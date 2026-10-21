"""Bounded, on-demand downloads for simulation ensemble profiles."""

from __future__ import annotations

import re
from dataclasses import replace

import numpy as np

from app.artifact_archives import DEFAULT_BATCH_ARCHIVE_SOURCE_BYTES, build_artifact_archive
from app.simulation_exports import ExportArtifact, build_raw_export, safe_export_filename
from app.simulation_service import ConstantSimulationResult


def limit_replication_profiles(
    result: ConstantSimulationResult, count: int,
) -> ConstantSimulationResult:
    """Release omitted profiles, including their original backing allocation."""

    profiles = result.replication_time_profiles
    if profiles is None:
        return result
    if count <= 0:
        return replace(result, replication_time_profiles=None)
    if profiles.shape[0] <= count:
        return result
    retained = np.array(profiles[:count], copy=True)
    retained.setflags(write=False)
    return replace(result, replication_time_profiles=retained)


def individual_profile_count(result: ConstantSimulationResult, count: int) -> int:
    profiles = result.replication_time_profiles
    if profiles is None:
        return 0
    if profiles.ndim != 2 or profiles.shape[1] != result.position_kb.size:
        raise ValueError("Individual profiles do not match the spatial grid.")
    return min(max(0, int(count)), profiles.shape[0])


def profile_archive_stem(filename: str, count: int) -> str:
    """Keep the actual replicate count in automatic and edited ZIP names."""

    stem = safe_export_filename(filename, "tsv", default_stem="individual_profiles")[:-4]
    stem = re.sub(r"_\d+_replicates?$", "", stem, flags=re.IGNORECASE)
    return f"{stem[:140]}_{count}_replicates"


def build_standard_deviation_export(
    result: ConstantSimulationResult,
    file_format: str = "TSV",
    *,
    filename: str = "rsim_rt_standard_deviation",
    compression: str = "none",
) -> ExportArtifact:
    if result.replication_timing_std is None:
        raise ValueError("Replication timing standard deviation is not available.")
    selected = replace(result, outputs={"replication_timing_std": result.replication_timing_std})
    return build_raw_export(
        selected, "replication_timing_std", file_format, filename=filename,
        compression=compression, axis_kind="space",
    )


def build_individual_profile_archive(
    result: ConstantSimulationResult,
    count: int,
    file_format: str = "TSV",
    *,
    filename: str = "rsim_rt_individual_profiles",
) -> ExportArtifact:
    """Serialize one profile at a time directly into a compressed ZIP."""

    count = individual_profile_count(result, count)
    if count == 0:
        raise ValueError("Individual replication timing profiles are not available. Run a simulation first.")
    profiles = result.replication_time_profiles
    assert profiles is not None
    stem = profile_archive_stem(filename, count)

    def artifacts():
        for index in range(count):
            profile = profiles[index]
            selected = replace(
                result, replication_timing=profile,
                outputs={"replication_timing": profile}, replication_time_profiles=None,
            )
            yield build_raw_export(
                selected, "replication_timing", file_format,
                filename=f"replicate_{index + 1:03d}", compression="none", axis_kind="space",
            )

    return build_artifact_archive(
        artifacts(), filename=f"{stem}.zip",
        max_source_bytes=DEFAULT_BATCH_ARCHIVE_SOURCE_BYTES,
        compress_all=True,
    )
