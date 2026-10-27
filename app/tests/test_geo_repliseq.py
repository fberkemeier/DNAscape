"""GEO Repli-seq matrices retain coordinates and chronological fractions."""

from __future__ import annotations

import asyncio
import gzip
import io
from dataclasses import replace
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
from nicegui.elements.upload import Upload
from nicegui.testing import user_simulation

import app.mapping_uploads as uploads
from app.data_conversion_service import DataConversionRequest, convert_data, prepare_data
from app.main import index
from app.mapping_uploads import UploadInspectionError, UploadTooLargeError, inspect_upload, materialize_upload
from app.remote_sources import RemoteSource


def _matrix(bins: int = 3) -> tuple[bytes, np.ndarray]:
    values = (np.arange(bins * 16, dtype=float).reshape(bins, 16) + 1) / 8
    chromosomes = ["chr1"] * (bins - 1) + ["chr2"]
    starts = np.arange(bins) * 50_000
    starts[-1] = 0
    rows = [chromosomes, starts, starts + 50_000, *values.T]
    payload = "\n".join("\t".join(str(value) for value in row) for row in rows) + "\n"
    return payload.encode(), values


@pytest.mark.parametrize("extension", [".mat", ".mat.gz", ".map", ".map.gz"])
def test_geo_matrix_is_transposed_and_preserves_the_original_upload(extension: str) -> None:
    payload, expected = _matrix()
    content = gzip.compress(payload) if extension.endswith(".gz") else payload
    inspection = inspect_upload("GSE137764_H1" + extension, content)
    assert inspection.content == content
    assert inspection.compressed is extension.endswith(".gz")
    assert inspection.source_layout == "geo_repliseq"
    assert inspection.source_format == "tsv"
    assert inspection.shape == (3, 19)
    assert inspection.columns == ("chrom", "start", "end", *(f"S{i}" for i in range(1, 17)))
    assert inspection.numeric_columns == tuple(f"S{i}" for i in range(1, 17))
    assert inspection.inferred_form == "xt"
    np.testing.assert_array_equal(
        materialize_upload(replace(inspection, expected_form="xt")), expected,
    )
    np.testing.assert_array_equal(materialize_upload(inspection, column="S16"), expected[:, -1])
    # Rebuilding numeric storage must use the normalized table, not reparse the
    # original wide matrix after its cached signal arrays have been released.
    np.testing.assert_array_equal(
        materialize_upload(replace(inspection, expected_form="xt", _table_data={})), expected,
    )


def test_geo_matrix_with_more_than_ten_thousand_bins_is_accepted() -> None:
    payload, expected = _matrix(10_003)
    inspection = inspect_upload("H1.mat.gz", gzip.compress(payload))
    assert inspection.shape == (10_003, 19)
    np.testing.assert_array_equal(
        prepare_data(DataConversionRequest(inspection=inspection, input_form="xt")).values,
        expected,
    )


@pytest.mark.parametrize("target", ["csv", "tsv"])
@pytest.mark.parametrize("compressed", [False, True])
def test_geo_full_table_export_preserves_coordinates_and_all_fraction_values(
    target: str, compressed: bool,
) -> None:
    payload, expected = _matrix()
    inspection = inspect_upload("H1.mat.gz", gzip.compress(payload))
    result = convert_data(DataConversionRequest(
        inspection=inspection, input_form="table", target_format=target, gzip_output=compressed,
        missing_policy="ignore",
    ))
    data = gzip.decompress(result.artifact.data) if compressed else result.artifact.data
    table = pd.read_csv(io.BytesIO(data), sep="," if target == "csv" else "\t")
    assert table.columns.tolist() == ["chrom", "start", "end", *(f"S{i}" for i in range(1, 17))]
    assert table["chrom"].tolist() == ["chr1", "chr1", "chr2"]
    assert table["start"].tolist() == [0, 50_000, 0]
    assert table["end"].tolist() == [50_000, 100_000, 50_000]
    np.testing.assert_array_equal(table.iloc[:, 3:].to_numpy(), expected)
    assert result.preview is not None
    assert result.preview.value_label == "S1"


@pytest.mark.parametrize("missing", ["nan", "", "NA", "."])
def test_geo_matrix_signal_export_preserves_missing_values_and_fraction_order(missing: str) -> None:
    payload, expected = _matrix()
    lines = payload.decode().splitlines()
    cells = lines[8].split("\t")
    cells[1] = missing
    cells[-1] = missing
    lines[8] = "\t".join(cells)
    expected[1, 5] = np.nan
    expected[-1, 5] = np.nan
    inspection = inspect_upload("H1.mat", ("\n".join(lines) + "\n").encode())
    result = convert_data(DataConversionRequest(
        inspection=inspection, input_form="xt", target_format="npz", missing_policy="ignore",
    ))
    with np.load(io.BytesIO(result.artifact.data), allow_pickle=False) as archive:
        np.testing.assert_allclose(archive["values"], expected, equal_nan=True)
        assert archive["columns"].tolist() == [f"S{i}" for i in range(1, 17)]
    assert inspection.missing_columns == ("S6",)


@pytest.mark.parametrize("damage", ["rows", "columns", "numeric", "coordinates"])
def test_malformed_geo_matrices_fail_with_a_clear_error(damage: str) -> None:
    payload, _ = _matrix()
    lines = payload.decode().splitlines()
    if damage == "rows":
        lines.pop()
        message = "16 S-phase rows"
    elif damage == "columns":
        lines[3] += "\t1"
        message = "same number of genomic bins"
    elif damage == "numeric":
        lines[3] = "1\tbroken\t3"
        message = "must be numeric"
    else:
        lines[1] = "-1\t50000\t0"
        message = "non-negative integer starts"
    with pytest.raises(UploadInspectionError, match=message):
        inspect_upload("bad.mat.gz", gzip.compress(("\n".join(lines) + "\n").encode()))


def test_geo_matrix_keeps_the_total_inspection_allocation_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    payload, _ = _matrix()
    monkeypatch.setattr(uploads, "MAX_TABLE_CELLS", 38)
    with pytest.raises(UploadTooLargeError, match="local inspection limit"):
        inspect_upload("too_large.mat", payload)


@pytest.mark.parametrize("mode", ["upload", "url"])
def test_geo_matrix_upload_and_public_link_can_be_converted_in_data_manager(
    monkeypatch: pytest.MonkeyPatch, mode: str,
) -> None:
    from app.pages import data_management as page

    payload, expected = _matrix()
    content = gzip.compress(payload)
    results = []
    real_convert = convert_data

    def record_conversion(request: DataConversionRequest, *args: Any, **kwargs: Any):
        result = real_convert(request, *args, **kwargs)
        results.append(result)
        return result

    def fetch(url: str) -> RemoteSource:
        return RemoteSource(
            requested_url=url, final_url=url, filename="GSE137764_H1.mat.gz", content=content,
            content_type="application/gzip", etag=None, last_modified=None,
            accept_ranges=False, redirects=(),
        )

    monkeypatch.setattr(page, "fetch_remote_source", fetch)
    monkeypatch.setattr(page, "convert_data", record_conversion)

    async def scenario() -> None:
        async with user_simulation(index) as user:
            def control(marker: str):
                return next(iter(user.find(marker=marker).elements))

            await user.open("/")
            user.find(marker="nav-data-management").click()
            if mode == "upload":
                uploader = cast(Upload, control("data-manager-upload"))
                assert ".mat" in uploader._props["accept"]
                await uploader.handle_uploads([
                    Upload.SmallFileUpload("GSE137764_H1.mat.gz", "application/gzip", content),
                ])
            else:
                control("data-manager-source-tabs").set_value("url")
                await user.should_see(marker="data-manager-source-url")
                control("data-manager-source-url").set_value("https://ftp.ncbi.nlm.nih.gov/example.mat.gz")
                user.find(marker="action-inspect-data-link-1").click()
            await user.should_see("Format: GEO Repli-seq MAT", retries=60)
            selection = control("data-manager-source-selection")
            assert selection.value == "__table__"
            assert control("data-manager-profile-missing").value == "zero"
            user.find(marker="action-convert").click()
            await user.should_see(marker="data-manager-result-1", retries=60)
            table = pd.read_csv(io.BytesIO(results[0].artifact.data), sep="\t")
            assert table["chrom"].tolist() == ["chr1", "chr1", "chr2"]
            np.testing.assert_array_equal(table.iloc[:, 3:].to_numpy(), np.nan_to_num(expected))
            selection.set_value("__matrix__")
            await user.should_see("Dependence: Space-time", retries=60)
            user.find(marker="action-convert").click()
            for _attempt in range(60):
                if len(results) == 2:
                    break
                await asyncio.sleep(0.05)
            assert len(results) == 2
            await user.should_see("Complete in", retries=60)
            # Configuration defaults to the selected chromosome, chr1.
            assert results[1].shape == (2, 16)
            matrix = pd.read_csv(io.BytesIO(results[1].artifact.data), sep="\t")
            np.testing.assert_allclose(matrix.to_numpy(), np.nan_to_num(expected[:2]))

    asyncio.run(scenario())
