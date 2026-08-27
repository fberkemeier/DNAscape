"""Raw mapping downloads retain rdata's scalar/X/T/XT conventions."""

from __future__ import annotations

import gzip
import io
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

import app.mapping_exports as mapping_exports
from app.mapping_exports import (
    build_mapping_comparison_raw_export,
    build_mapping_raw_export,
    select_mapping_output,
)
from app.mapping_service import MappingComparison, MappingResult
from app.mapping_uploads import inspect_upload, materialize_upload
from dnascape.io import rload


def _result(
    values: np.ndarray,
    form: str,
    *,
    key: str = "mapped_output",
    x_axis: np.ndarray | None = None,
    time_axis: np.ndarray | None = None,
    region_labels: tuple[str, ...] = (),
) -> MappingResult:
    array = np.asarray(values, dtype=float)
    array.setflags(write=False)
    x = np.asarray([] if x_axis is None else x_axis, dtype=float)
    time = np.asarray([] if time_axis is None else time_axis, dtype=float)
    x.setflags(write=False)
    time.setflags(write=False)
    return MappingResult(
        target=key,
        target_key=key,
        target_label="Mapped output",
        method="test_method",
        output_form=form,  # type: ignore[arg-type]
        output_dependence="space",
        primary_output=key,
        outputs=MappingProxyType({key: array}),
        output_forms=MappingProxyType({key: form}),  # type: ignore[arg-type]
        x_axis=x,
        time_axis=time,
        region_labels=region_labels,
        elapsed_seconds=0.1,
        assumptions=MappingProxyType({}),
    )


@pytest.mark.parametrize(
    ("form", "values", "x_axis", "time_axis", "expected_shape", "expected_form"),
    (
        ("scalar", np.asarray([1.5]), None, None, (), "scalar"),
        ("x", np.asarray([1.0, 2.0, 3.0]), np.asarray([10.0, 11.0, 12.0]), None, (3,), "space"),
        ("t", np.asarray([0.1, 0.5, 1.0]), None, np.asarray([0.0, 5.0, 10.0]), (1, 3), "time"),
        (
            "xt",
            np.arange(6, dtype=float).reshape(2, 3),
            np.asarray([10.0, 11.0]),
            np.asarray([0.0, 5.0, 10.0]),
            (2, 3),
            "spacetime",
        ),
    ),
)
def test_npz_exports_round_trip_through_rdata_with_canonical_forms(
    tmp_path: Path,
    form: str,
    values: np.ndarray,
    x_axis: np.ndarray | None,
    time_axis: np.ndarray | None,
    expected_shape: tuple[int, ...],
    expected_form: str,
) -> None:
    result = _result(values, form, x_axis=x_axis, time_axis=time_axis)
    artifact = build_mapping_raw_export(result, "mapped_output", "npz")
    path = tmp_path / artifact.filename
    path.write_bytes(artifact.data)

    loaded = rload(path, return_metadata=True)
    assert loaded["data_form"] == expected_form
    assert np.asarray(loaded["values"]).shape == expected_shape
    np.testing.assert_array_equal(np.asarray(loaded["values"]), values.reshape(expected_shape))


def test_csv_profile_has_detectable_header_and_round_trips(tmp_path: Path) -> None:
    result = _result(
        np.asarray([8.0, 7.0, 9.0]),
        "x",
        key="replication_timing",
        x_axis=np.asarray([0.0, 1.0, 2.0]),
    )
    artifact = build_mapping_raw_export(result, "replication_timing", "csv")
    assert artifact.data.decode("utf-8").splitlines() == [
        "replication_timing",
        "8",
        "7",
        "9",
    ]

    path = tmp_path / artifact.filename
    path.write_bytes(artifact.data)
    loaded = rload(path, value_column="replication_timing")
    np.testing.assert_array_equal(loaded, [8.0, 7.0, 9.0])


def test_scalar_npy_payload_auto_detects_as_scalar(tmp_path: Path) -> None:
    artifact = build_mapping_raw_export(
        _result(np.asarray([1.5]), "scalar"),
        "mapped_output",
        "npy",
    )
    path = tmp_path / artifact.filename
    path.write_bytes(artifact.data)
    loaded = rload(path, return_metadata=True)
    assert loaded["data_form"] == "scalar"
    assert np.asarray(loaded["values"]).shape == ()
    assert float(np.asarray(loaded["values"]).reshape(()).item()) == 1.5


@pytest.mark.parametrize("file_format", ("csv", "tsv", "txt"))
def test_scalar_text_exports_are_one_visible_value(
    tmp_path: Path,
    file_format: str,
) -> None:
    artifact = build_mapping_raw_export(
        _result(np.asarray([1.5]), "scalar"),
        "mapped_output",
        file_format,
    )
    path = tmp_path / artifact.filename
    path.write_bytes(artifact.data)

    if file_format in {"csv", "tsv"}:
        assert artifact.data.decode().splitlines() == ["mapped_output", "1.5"]
        loaded = rload(path, value_column="mapped_output")
    else:
        loaded = rload(path)
    assert float(np.asarray(loaded).reshape(()).item()) == 1.5


def test_time_and_spacetime_text_payloads_use_rows_and_time_headers() -> None:
    temporal = build_mapping_raw_export(
        _result(
            np.asarray([0.1, 0.5, 1.0]),
            "t",
            time_axis=np.asarray([0.0, 5.0, 10.0]),
        ),
        "mapped_output",
        "csv",
    )
    assert temporal.data.decode().splitlines() == ["t0,t1,t2", "0.1,0.5,1"]

    spacetime = build_mapping_raw_export(
        _result(
            np.arange(6, dtype=float).reshape(2, 3),
            "xt",
            x_axis=np.asarray([0.0, 1.0]),
            time_axis=np.asarray([0.0, 5.0, 10.0]),
        ),
        "mapped_output",
        "tsv",
    )
    assert spacetime.data.decode().splitlines() == [
        "t0\tt1\tt2",
        "0\t1\t2",
        "3\t4\t5",
    ]


def test_region_export_preserves_labels_and_round_trips_through_rdata(tmp_path: Path) -> None:
    result = _result(
        np.asarray([0.2, 0.8]),
        "regions",
        key="origin_efficiency",
        region_labels=("early", "late"),
    )
    artifact = build_mapping_raw_export(result, "origin_efficiency", "csv")
    assert artifact.data.decode().splitlines() == [
        "region,origin_efficiency",
        "early,0.2",
        "late,0.8",
    ]

    npz = build_mapping_raw_export(result, "origin_efficiency", "npz")
    with np.load(io.BytesIO(npz.data), allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["region_names"], ["early", "late"])
        assert str(archive["data_form"]) == "space"
        assert str(archive["semantic_form"]) == "regions"

    path = tmp_path / npz.filename
    path.write_bytes(npz.data)
    loaded = rload(path, return_metadata=True)
    assert loaded["data_form"] == "space"
    np.testing.assert_array_equal(loaded["values"], [0.2, 0.8])

    # A raw export must also be accepted by the app's own upload inspector;
    # semantic string arrays are metadata rather than numeric mapping members.
    inspection = inspect_upload(npz.filename, npz.data, expected_form="auto")
    assert inspection.default_array_key == "values"
    assert inspection.semantic_form == "regions"
    assert inspection.region_names == ("early", "late")
    np.testing.assert_array_equal(materialize_upload(inspection), [0.2, 0.8])


@pytest.mark.parametrize("semantic_form", ("ensemble", "supplied"))
def test_non_rdata_vector_semantics_use_space_storage_metadata(
    tmp_path: Path,
    semantic_form: str,
) -> None:
    result = _result(np.asarray([0.4, 0.2, 0.1]), semantic_form)
    artifact = build_mapping_raw_export(result, "mapped_output", "npz")
    path = tmp_path / artifact.filename
    path.write_bytes(artifact.data)

    loaded = rload(path, return_metadata=True)
    assert loaded["data_form"] == "space"
    np.testing.assert_array_equal(loaded["values"], [0.4, 0.2, 0.1])
    with np.load(io.BytesIO(artifact.data), allow_pickle=False) as archive:
        assert str(archive["data_form"]) == "space"
        assert str(archive["semantic_form"]) == semantic_form


def test_bundle_member_selection_and_dynamic_filename() -> None:
    firing = np.ones((2, 3), dtype=float)
    speed = np.full((2, 3), 1.4)
    result = MappingResult(
        target="kinetic_fields",
        target_key="kinetic_fields",
        target_label="Joint kinetic fields",
        method="joint",
        output_form="bundle",
        output_dependence="spacetime",
        primary_output="firing_rate",
        outputs=MappingProxyType({"firing_rate": firing, "fork_speed": speed}),
        output_forms=MappingProxyType({"firing_rate": "xt", "fork_speed": "xt"}),
        x_axis=np.asarray([0.0, 1.0]),
        time_axis=np.asarray([0.0, 5.0, 10.0]),
        region_labels=(),
        elapsed_seconds=0.1,
        assumptions=MappingProxyType({}),
    )

    selected = select_mapping_output(result, "fork_speed")
    np.testing.assert_array_equal(selected.values, speed)
    artifact = build_mapping_raw_export(result, "fork_speed", "npy")
    assert artifact.filename == "dnascape_mapping_fork_speed.npy"
    np.testing.assert_array_equal(np.load(io.BytesIO(artifact.data), allow_pickle=False), speed)


def test_gzip_format_and_export_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    result = _result(
        np.asarray([1.0, 2.0]),
        "x",
        x_axis=np.asarray([0.0, 1.0]),
    )
    plain = build_mapping_raw_export(result, "mapped_output", "csv", filename="mapped")
    compressed = build_mapping_raw_export(
        result,
        "mapped_output",
        "csv",
        filename="mapped.csv",
        gzip_output=True,
    )
    assert gzip.decompress(compressed.data) == plain.data
    assert compressed.filename == "mapped.csv.gz"
    assert compressed.media_type == "application/gzip"

    monkeypatch.setattr(mapping_exports, "MAX_TEXT_EXPORT_VALUES", 1)
    with pytest.raises(ValueError, match="Use NPY or NPZ"):
        build_mapping_raw_export(result, "mapped_output", "csv")


def test_missing_member_and_axis_mismatch_fail_clearly() -> None:
    result = _result(
        np.asarray([1.0, 2.0]),
        "x",
        x_axis=np.asarray([0.0]),
    )
    with pytest.raises(ValueError, match="is not available"):
        build_mapping_raw_export(result, "missing", "csv")
    with pytest.raises(ValueError, match="length must match"):
        build_mapping_raw_export(result, "mapped_output", "csv")


def test_observed_and_fitted_profiles_export_with_one_shared_axis() -> None:
    result = _result(
        np.asarray([8.2, 7.1, 9.1]),
        "x",
        key="replication_timing_fit",
        x_axis=np.asarray([10.0, 11.0, 12.0]),
    )
    observed = np.asarray([8.0, 7.0, 9.0])
    fitted = np.asarray([8.2, 7.1, 9.1])
    result = replace(
        result,
        comparisons=MappingProxyType(
            {
                "replication_timing_fit": MappingComparison(
                    source_key="replication_timing",
                    fitted_key="replication_timing_fit",
                    form="x",
                    observed=observed,
                    fitted=fitted,
                )
            }
        ),
    )

    artifact = build_mapping_comparison_raw_export(
        result,
        "replication_timing_fit",
        "csv",
    )
    assert artifact.filename == "dnascape_mapping_replication_timing_comparison.csv"
    assert artifact.data.decode().splitlines() == [
        "position_kb,observed,fitted",
        "10,8,8.2",
        "11,7,7.1",
        "12,9,9.1",
    ]

    archive = build_mapping_comparison_raw_export(
        result,
        "replication_timing_fit",
        "npz",
    )
    with np.load(io.BytesIO(archive.data), allow_pickle=False) as values:
        np.testing.assert_array_equal(values["axis"], [10.0, 11.0, 12.0])
        np.testing.assert_array_equal(values["observed"], observed)
        np.testing.assert_array_equal(values["fitted"], fitted)
