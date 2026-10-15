"""Chromosome selection and H1 example links use the timing workflow."""

import asyncio
import gzip
import io
from typing import Any

import numpy as np
import pytest
from nicegui.elements.upload import Upload
from nicegui.testing import user_simulation

from app.data_conversion_service import DataConversionRequest, convert_data, inspect_repliseq, prepare_data
from app.main import index
from app.mapping_uploads import inspect_upload
from app.pages import data_management as page
from app.remote_sources import RemoteSource


def source_payload() -> bytes:
    values = np.zeros((4, 16))
    values[np.arange(4), [2, 14, 6, 10]] = 100
    rows = [["chr1", "chr1", "chr2", "chr2"], [0, 100000, 0, 50000],
            [50000, 150000, 50000, 100000], *values.T]
    return gzip.compress(("\n".join("\t".join(map(str, row)) for row in rows) + "\n").encode())


def test_timing_conversion_preserves_chromosome_grid_and_export_coordinates() -> None:
    inspection = inspect_upload("H1.mat.gz", source_payload())
    summary = inspect_repliseq(inspection)
    assert [region.chromosome for region in summary.regions] == ["chr1", "chr2"]
    request = DataConversionRequest(
        inspection=inspection, input_form="x", chromosome="chr1", target_format="npy",
        repliseq_timing_method="sigmoid", value_name="replication_timing", missing_policy="ignore",
    )
    prepared = prepare_data(request)
    np.testing.assert_allclose(prepared.values, [93.75, np.nan, 543.75], atol=0.01, equal_nan=True)
    np.testing.assert_array_equal(prepared.coordinates["position_kb"], [0, 50, 100])
    result = convert_data(request)
    np.testing.assert_allclose(np.load(io.BytesIO(result.artifact.data)), prepared.values, equal_nan=True)
    assert dict(result.metadata)["units"] == "min"
    assert dict(result.metadata)["invalid_timing_bins"] == "1"
    assert result.preview is not None and result.preview.value_label == "replication_timing"
    np.testing.assert_array_equal(result.preview.x, [0, 50, 100])
    with pytest.raises(ValueError, match="does not contain chromosome"):
        prepare_data(DataConversionRequest(inspection=inspection, input_form="x",
                     chromosome="chr3", repliseq_timing_method="sigmoid"))


@pytest.mark.parametrize("policy", ["ignore", "zero", "fill", "min", "max", "error"])
def test_timing_missing_options_apply_after_timing_extraction(policy):
    request = DataConversionRequest(
        inspection=inspect_upload("H1.mat.gz", source_payload()), input_form="x", chromosome="chr1",
        repliseq_timing_method="mean", missing_policy=policy, fill_value=7,
    )
    if policy == "error":
        with pytest.raises(ValueError, match="contain missing values"):
            convert_data(request)
        return
    middle = {"ignore": np.nan, "zero": 0, "fill": 7, "min": 93.75, "max": 543.75}[policy]
    np.testing.assert_allclose(prepare_data(request).values, [93.75, middle, 543.75], equal_nan=True)


@pytest.mark.parametrize("mode", ["upload", "example"])
def test_choose_timing_chromosome_method_and_duration_in_data_manager(monkeypatch, mode: str) -> None:
    results = []
    requests = []
    content = source_payload()

    def record(request, *args, **kwargs):
        requests.append(request)
        result = convert_data(request, *args, **kwargs)
        results.append(result)
        return result

    def fetch(url):
        assert url == page._H1_REPLISEQ_EXAMPLE_URL
        return RemoteSource(url, url, "H1.mat.gz", content, "application/gzip", None, None, False, ())

    monkeypatch.setattr(page, "convert_data", record)
    monkeypatch.setattr(page, "fetch_remote_source", fetch)

    async def scenario() -> None:
        async with user_simulation(index) as user:
            await user.open("/")
            user.find(marker="nav-data-management").click()

            def control(marker: str) -> Any:
                return next(iter(user.find(marker=f"data-manager-{marker}").elements))

            if mode == "upload":
                await control("upload").handle_uploads([
                    Upload.SmallFileUpload("H1.mat.gz", "application/gzip", content),
                ])
            else:
                control("source-tabs").set_value("url")
                await user.should_see(marker="use-h1-repliseq-data-example")
                user.find(marker="use-h1-repliseq-data-example").click()
                await user.should_see(page._H1_REPLISEQ_EXAMPLE_URL)
                user.find(marker="action-inspect-data-link-1").click()
            await user.should_see("Format: GEO Repli-seq MAT", retries=60)
            assert "__repliseq_timing__" in control("source-selection").options
            if mode == "example":
                assert control("source-selection").value == "__repliseq_timing__"
            assert user.client is not None
            with user.client:
                control("source-selection").set_value("__repliseq_timing__")
            await user.should_see(marker="data-manager-chromosome", retries=60)
            assert control("repliseq-timing-method").value == "sigmoid"
            assert control("s-phase-duration").value == 600
            assert control("chromosome").options == ["chr1", "chr2"]
            assert control("profile-missing").value == "zero"
            assert list(control("profile-missing").options.values()) == [
                "Keep empty values", "Fill with zero", "Fill with a value",
                "Fill with minimum", "Fill with maximum", "Stop if detected",
            ]
            with user.client:
                control("preview-toggle").set_value(True)
            user.find(marker="action-convert").click()
            await user.should_see(marker="data-manager-result-1", retries=60)
            assert results[0].shape == (3,)
            assert results[0].labels == ("chr1",)
            assert "chr1" in results[0].artifact.filename
            await user.should_see(marker="data-manager-preview-chart")
            assert requests[0].biological_role == "replication_timing"
            assert requests[0].missing_policy == "zero"
            assert results[0].preview.values[1] == 0
            with user.client:
                control("profile-missing").set_value("fill")
            await user.should_see(marker="data-manager-fill-value")
            with user.client:
                control("fill-value").set_value(7)
            with user.client:
                control("chromosome").set_value("chr2")
            await user.should_see("Array size: 2", retries=60)
            with user.client:
                control("repliseq-timing-method").set_value("mean")
                control("s-phase-duration").set_value(320)
            user.find(marker="action-convert").click()
            await user.should_see(marker="data-manager-result-2", retries=60)
            assert results[1].shape == (2,)
            assert results[1].labels == ("chr2",)
            assert dict(results[1].metadata)["s_phase_duration_min"] == "320.0"
            np.testing.assert_allclose(results[1].preview.values, [130, 210])
            assert requests[1].repliseq_timing_method == "mean"
            assert requests[1].missing_policy == "fill" and requests[1].fill_value == 7
            assert "chr2" in results[1].artifact.filename

    asyncio.run(scenario())
