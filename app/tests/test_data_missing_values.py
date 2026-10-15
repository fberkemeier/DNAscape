"""Data Manager uses the same row-preserving missing-value choices for every source."""

import gzip
import io

import numpy as np
import pandas as pd
import pytest

from dnascape.workflows.conversion import DataConversionRequest, convert_data, extract_values, prepare_data
from dnascape.workflows.uploads import inspect_upload


POLICIES = ["ignore", "zero", "fill", "min", "max", "error"]


@pytest.mark.parametrize("form", ["x", "t", "xt"])
@pytest.mark.parametrize("policy", POLICIES)
def test_profile_choices_preserve_shape_and_match_converted_values(form, policy):
    values = np.array([10., np.nan, 20.])
    if form == "t":
        values = values.reshape(1, -1)
    elif form == "xt":
        values = np.column_stack((values, values * 10))
    buffer = io.BytesIO()
    np.save(buffer, values)
    inspection = inspect_upload("signal.npy", buffer.getvalue())
    request = DataConversionRequest(
        inspection=inspection, input_form=form, target_format="npy", missing_policy=policy, fill_value=7,
    )
    if policy == "error":
        with pytest.raises(ValueError, match="contain missing values"):
            prepare_data(request)
        with pytest.raises(ValueError, match="contain missing values"):
            convert_data(request)
        return
    expected = values.copy()
    if policy != "ignore":
        fill = {"zero": 0, "fill": 7, "min": 10, "max": 20}[policy]
        expected[np.isnan(expected)] = fill
        if form == "xt" and policy in {"min", "max"}:
            expected[1, 1] = fill * 10
    np.testing.assert_allclose(prepare_data(request).values, expected, equal_nan=True)
    np.testing.assert_allclose(np.load(io.BytesIO(convert_data(request).artifact.data)), expected, equal_nan=True)
    if policy in {"min", "max", "fill"}:
        np.testing.assert_allclose(
            extract_values(inspection, data_form=form, missing_policy=policy, fill_value=7), expected,
        )
        np.testing.assert_allclose(extract_values(inspection, missing_policy=policy, fill_value=7), expected)


@pytest.mark.parametrize("source", ["csv", "mat.gz"])
@pytest.mark.parametrize("policy", POLICIES)
def test_full_table_choices_fill_only_signals_and_preserve_all_coordinates(source, policy):
    signals = np.tile([10., np.nan, 20.], (16, 1))
    if source == "csv":
        payload = b"position,A,B\n100,10,10\n200,,\n300,20,20\n"
        labels = ["A", "B"]
    else:
        rows = [["chr1"] * 3, [0, 50_000, 100_000], [50_000, 100_000, 150_000], *signals]
        payload = gzip.compress(("\n".join("\t".join(map(str, row)) for row in rows) + "\n").encode())
        labels = [f"S{i}" for i in range(1, 17)]
    inspection = inspect_upload(f"signal.{source}", payload)
    request = DataConversionRequest(
        inspection=inspection, input_form="table", target_format="tsv", missing_policy=policy, fill_value=7,
    )
    if policy == "error":
        with pytest.raises(ValueError, match="contain missing values"):
            convert_data(request)
        return
    table = pd.read_csv(io.BytesIO(convert_data(request).artifact.data), sep="\t")
    expected = np.array([10., {"ignore": np.nan, "zero": 0, "fill": 7, "min": 10, "max": 20}[policy], 20.])
    np.testing.assert_allclose(table[labels], np.tile(expected, (len(labels), 1)).T, equal_nan=True)
    if source == "csv":
        np.testing.assert_array_equal(table["position"], [100, 200, 300])
    else:
        assert table["chrom"].tolist() == ["chr1"] * 3
        np.testing.assert_array_equal(table["start"], [0, 50_000, 100_000])
        np.testing.assert_array_equal(table["end"], [50_000, 100_000, 150_000])


def test_default_zero_fill_and_stop_before_track_interpolation():
    inspection = inspect_upload("profile.csv", b"position,value\n0,10\n1,\n2,20\n")
    request = DataConversionRequest(inspection=inspection, input_form="x")
    assert request.missing_policy == "zero"
    np.testing.assert_array_equal(prepare_data(request).values, [10, 0, 20])
    track = inspect_upload("signal.bedgraph", b"chr1\t0\t10\t10\nchr1\t20\t30\t20\n")
    with pytest.raises(ValueError, match="contain missing values"):
        convert_data(DataConversionRequest(
            inspection=track, input_form="track", bin_size=10, track_missing="error", interpolation="linear",
        ))


@pytest.mark.parametrize("policy", ["min", "max", "error"])
def test_missing_policy_applies_to_thresholded_samples(policy):
    request = DataConversionRequest(
        inspection=inspect_upload("profile.csv", b"value\n10\n100\n20\n"), input_form="x",
        missing_policy=policy, threshold_enabled=True, threshold_max=50,
        scaling_mode="interval", rescale_min=2, rescale_max=4,
    )
    if policy == "error":
        with pytest.raises(ValueError, match="contain missing values"):
            prepare_data(request)
    else:
        np.testing.assert_allclose(prepare_data(request).values, [2, 2 if policy == "min" else 4, 4])
