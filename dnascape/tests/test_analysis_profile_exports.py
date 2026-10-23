"""Raw transformed profiles preserve shape, precision and missing values."""

import gzip
import io
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from dnascape.workflows.analysis import export_profile_data
from dnascape.workflows.uploads import inspect_upload, materialize_upload


@pytest.mark.parametrize("form,shape", [("x", (3,)), ("t", (1, 3)), ("xt", (3, 2))])
@pytest.mark.parametrize("file_format", ["tsv", "txt", "csv", "npz"])
def test_transformed_profile_export_preserves_native_values(form, shape, file_format):
    values = np.arange(np.prod(shape), dtype=float).reshape(shape) / 7
    values.flat[1] = np.nan
    artifact = export_profile_data(values, dependence=form, file_format=file_format, filename="changed", compress=True)
    assert artifact.filename == f"changed.{file_format}.gz"
    raw = gzip.decompress(artifact.data)
    inspection = inspect_upload(f"changed.{file_format}", raw, expected_form="auto")
    assert inspection.inferred_form == form
    restored = materialize_upload(replace(inspection, expected_form=form), missing_policy="ignore")
    np.testing.assert_allclose(restored.reshape(-1), values.reshape(-1), rtol=1e-15, equal_nan=True)
    if file_format == "npz":
        with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
            actual = archive["values"]
            assert actual.shape == shape
    elif file_format == "txt":
        actual = np.loadtxt(io.BytesIO(raw))
    else:
        actual = pd.read_csv(io.BytesIO(raw), sep="," if file_format == "csv" else "\t").to_numpy()
    np.testing.assert_allclose(actual.reshape(-1), values.reshape(-1), rtol=1e-15, equal_nan=True)
