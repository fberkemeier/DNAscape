"""Mapping downloads preserve native values while formatting bounded blocks."""

import asyncio
import csv
import gzip
import io
import zipfile
from types import MappingProxyType

import numpy as np
import pytest

from dnascape.workflows import mapping_exports as exports
from dnascape.workflows.mapping import MappingResult
from dnascape.workflows.mapping_catalog import mapping_route, mapping_route_families


def result_for(values):
    return MappingResult(
        target="replicated_fraction_xt", target_key="replicated_fraction",
        target_label="Replicated fraction", method="test", output_form="xt",
        output_dependence="spacetime", primary_output="replicated_fraction_xt",
        outputs=MappingProxyType({"replicated_fraction_xt": values}),
        output_forms=MappingProxyType({"replicated_fraction_xt": "xt"}),
        x_axis=np.arange(values.shape[0], dtype=float),
        time_axis=np.arange(values.shape[1], dtype=float), region_labels=(),
        elapsed_seconds=0., assumptions=MappingProxyType({}),
    )


def test_fraction_route_uses_initiation_and_speed_only():
    families = mapping_route_families("replicated_fraction", "spacetime")
    assert len(families) == 1
    assert families[0].factor_keys == ("initiation_rate", "fork_speed")
    with pytest.raises(KeyError):
        mapping_route("fork_density_bundle_to_replicated_fraction_xt_extraction")
    # Directional densities use the same deterministic initiation-and-speed inputs.
    mapping_route("firing_and_speed_to_right_fork_density_xt_mean_field")


def test_fraction_route_selector_excludes_directional_densities():
    from nicegui.testing import user_simulation
    from app.main import index

    async def scenario():
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-mapping").click()
            await user.should_see(marker="mapping-target", retries=60)
            with user.client:
                next(iter(user.find(marker="mapping-target").elements)).set_value("replicated_fraction")
            await user.should_see(marker="mapping-route", retries=60)
            await asyncio.sleep(0)
            routes = next(iter(user.find(marker="mapping-route").elements))
            assert list(routes.options.values()) == ["Initiation rate + Fork speed"]

    asyncio.run(scenario())


@pytest.mark.parametrize("file_format, delimiter", [("csv", ","), ("tsv", "\t")])
def test_numeric_spellings_and_signed_zero_remain_compatible(file_format, delimiter):
    values = np.array([
        [0., -0., 1., -2., .1, 1e-7],
        [1e16, 1e15, 1e-4, 5e-324, np.inf, -np.inf],
        [np.nan, 9999999999999998., -9999999999999998., 1e20, -1e20, .9999999999999999],
    ])
    artifact = exports.build_mapping_raw_export(result_for(values), "replicated_fraction_xt", file_format)
    rows = list(csv.reader(io.StringIO(artifact.data.decode()), delimiter=delimiter))
    assert rows == [
        ["t0", "t1", "t2", "t3", "t4", "t5"],
        ["0", "-0", "1", "-2", "0.1", "1e-07"],
        ["1e+16", "1000000000000000", "0.0001", "5e-324", "inf", "-inf"],
        ["nan", "9999999999999998", "-9999999999999998", "1e+20", "-1e+20", "0.9999999999999999"],
    ]


@pytest.mark.parametrize("file_format", ["csv", "tsv", "txt", "npz"])
def test_mapped_native_grid_round_trips_exactly_across_blocks(tmp_path, monkeypatch, file_format):
    # A Fortran-order mapped grid exercises the storage used for whole chromosomes.
    values = np.memmap(tmp_path / "fraction.dat", dtype=float, mode="w+", shape=(37, 17), order="F")
    values[:] = np.random.default_rng(19).random(values.shape)
    values[:, 0] = -0.
    values[:, -1] = 1.
    values.flush()
    values.flags.writeable = False
    original = np.array(values)
    result = result_for(values)
    monkeypatch.setattr(exports, "_TEXT_BLOCK_VALUES", 53)
    selected = exports.select_mapping_output(result, "replicated_fraction_xt")
    assert np.shares_memory(selected.values, values)
    assert not selected.values.flags.writeable
    artifact = exports.build_mapping_raw_export(result, "replicated_fraction_xt", file_format)
    if file_format == "npz":
        with np.load(io.BytesIO(artifact.data), allow_pickle=False) as payload:
            actual = payload["values"]
            np.testing.assert_array_equal(payload["position_kb"], result.x_axis)
            np.testing.assert_array_equal(payload["time_min"], result.time_axis)
    else:
        actual = np.loadtxt(io.BytesIO(artifact.data),
                            delimiter="," if file_format == "csv" else None,
                            skiprows=0 if file_format == "txt" else 1)
    np.testing.assert_array_equal(actual.view(np.uint64), original.view(np.uint64))
    np.testing.assert_array_equal(values.view(np.uint64), original.view(np.uint64))


def test_mutable_input_still_has_an_independent_download_snapshot():
    values = np.array([[.1, .5, 1.]])
    selected = exports.select_mapping_output(result_for(values), "replicated_fraction_xt")
    assert not np.shares_memory(selected.values, values)
    values[:] = 0.
    np.testing.assert_array_equal(selected.values, [[.1, .5, 1.]])
    assert not selected.values.flags.writeable


def test_compression_and_size_guards_remain_intact(monkeypatch):
    values = np.array([[.1, .5, 1.], [.2, .7, 1.]])
    result = result_for(values)
    plain = exports.build_mapping_raw_export(result, "replicated_fraction_xt", "tsv")
    compressed = exports.build_mapping_raw_export(result, "replicated_fraction_xt", "tsv", compression="gzip")
    assert gzip.decompress(compressed.data) == plain.data
    monkeypatch.setattr(exports, "MAX_TEXT_EXPORT_BYTES", len(plain.data) - 1)
    with pytest.raises(ValueError, match="Use NPZ"):
        exports.build_mapping_raw_export(result, "replicated_fraction_xt", "tsv")


@pytest.mark.parametrize("compression", ["gzip", "zip"])
@pytest.mark.parametrize("file_format", ["tsv", "npz"])
def test_large_fraction_fast_compression_preserves_all_values(compression, file_format):
    values = np.broadcast_to(np.array([0., .1, 1.]), (334_000, 3))
    result = result_for(values)
    plain = exports.build_mapping_raw_export(result, "replicated_fraction_xt", file_format)
    compressed = exports.build_mapping_raw_export(
        result, "replicated_fraction_xt", file_format, compression=compression,
    )
    if compression == "gzip":
        content = gzip.decompress(compressed.data)
    else:
        with zipfile.ZipFile(io.BytesIO(compressed.data)) as archive:
            assert archive.namelist() == [plain.filename]
            content = archive.read(plain.filename)
    assert content == plain.data
