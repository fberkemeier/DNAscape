"""Ensemble downloads preserve values and release omitted profile arrays."""

import io
import zipfile

import numpy as np
import pytest

from app.simulation_profile_exports import (
    build_individual_profile_archive,
    build_standard_deviation_export,
    limit_replication_profiles,
)
from app.simulation_service import ConstantSimulationResult


def result():
    profiles = np.arange(24.0).reshape(6, 4)
    mean = profiles.mean(axis=0)
    return ConstantSimulationResult(
        position_kb=100 + np.arange(4) * .5, replication_timing=mean,
        elapsed_seconds=.1, replicates=6, geometry="linear", time_grid=np.asarray([]),
        outputs={"replication_timing": mean}, replication_time_profiles=profiles,
        replication_timing_std=profiles.std(axis=0),
    )


@pytest.mark.parametrize("file_type", ("TSV", "TXT", "CSV", "NPZ"))
def test_archive_contains_only_requested_profiles_in_the_selected_format(file_type):
    value = result()
    artifact = build_individual_profile_archive(
        value, 3, file_type, filename="../../my profiles_99_replicates.zip",
    )
    assert artifact.filename == "my_profiles_3_replicates.zip"
    assert artifact.media_type == "application/zip"
    with zipfile.ZipFile(io.BytesIO(artifact.data)) as archive:
        assert archive.namelist() == [f"replicate_{index:03d}.{file_type.lower()}" for index in range(1, 4)]
        for index, member in enumerate(archive.infolist()):
            assert member.compress_type == zipfile.ZIP_DEFLATED
            data = archive.read(member)
            if file_type == "NPZ":
                with np.load(io.BytesIO(data), allow_pickle=False) as stored:
                    np.testing.assert_array_equal(stored["position_kb"], value.position_kb)
                    values = stored["values"].reshape(-1)
            else:
                values = np.loadtxt(
                    io.BytesIO(data), skiprows=0 if file_type == "TXT" else 1,
                    delimiter="," if file_type == "CSV" else None,
                )
            np.testing.assert_array_equal(values, value.replication_time_profiles[index])


def test_archive_name_uses_available_count_and_rejects_missing_profiles():
    value = result()
    artifact = build_individual_profile_archive(value, 50)
    assert artifact.filename.endswith("_6_replicates.zip")
    with zipfile.ZipFile(io.BytesIO(artifact.data)) as archive:
        assert len(archive.namelist()) == 6
    with pytest.raises(ValueError, match="not available"):
        build_individual_profile_archive(limit_replication_profiles(value, 0), 10)


def test_standard_deviation_is_exported_with_spatial_metadata():
    value = result()
    artifact = build_standard_deviation_export(value, "NPZ")
    with np.load(io.BytesIO(artifact.data), allow_pickle=False) as stored:
        np.testing.assert_array_equal(stored["values"].reshape(-1), value.replication_timing_std)
        np.testing.assert_array_equal(stored["position_kb"], value.position_kb)


def test_trimming_releases_backing_memory_and_keeps_full_ensemble_statistics():
    value = result()
    retained = limit_replication_profiles(value, 2)
    assert retained.replication_time_profiles.shape == (2, 4)
    assert not retained.replication_time_profiles.flags.writeable
    assert not np.shares_memory(retained.replication_time_profiles, value.replication_time_profiles)
    assert retained.replication_timing_std is value.replication_timing_std
    assert retained.outputs is value.outputs
    cached = limit_replication_profiles(retained, 0)
    assert cached.replication_time_profiles is None
    assert cached.replication_timing_std is value.replication_timing_std
    assert value.replication_time_profiles.shape == (6, 4)
