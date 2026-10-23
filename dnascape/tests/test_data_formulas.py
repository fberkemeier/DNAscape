"""Numeric formulas retain grids and source data throughout the conversion pipeline."""

import io
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from dnascape.workflows.conversion import DataConversionRequest, convert_data, convert_combined_data, prepare_data
from dnascape.workflows.filtering import DataFilterSettings, apply_data_filter
from dnascape.workflows.formulas import evaluate_formula
from dnascape.workflows.modulation import ProfileModulationSettings
from dnascape.workflows.uploads import inspect_upload


def request(values, **kwargs):
    payload = ("signal\n" + "\n".join(map(str, values)) + "\n").encode()
    return DataConversionRequest(inspect_upload("source.tsv", payload), "x", **kwargs)


def test_python_numpy_expressions_and_inputs_remain_unchanged():
    source = np.array([-2., 0., 2., 4.])
    np.testing.assert_equal(evaluate_formula("np.where(source > 0, source**2, -source)", {"source": source}), [2, 0, 4, 16])
    np.testing.assert_equal(evaluate_formula("source - source.mean()", {"source": source}), [-3, -1, 1, 3])
    np.testing.assert_equal(evaluate_formula("3", {"source": source}), [3] * 4)
    np.testing.assert_equal(evaluate_formula("source if source > 0 else 0", {"source": source}), [0, 0, 2, 4])
    np.testing.assert_equal(source, [-2, 0, 2, 4])
    assert source.flags.writeable
    np.testing.assert_allclose(evaluate_formula("smooth(source, 1)", {"source": np.ones(9)}), 1)
    np.testing.assert_allclose(evaluate_formula("convolve(source, [0, 1, 0])", {"source": source}), source)


@pytest.mark.parametrize("formula", ["__import__('os')", "source.__class__", "np.load('file')", "open('file')", "[x for x in source]", "(lambda: 1)()", "source.mean(out=source)"])
def test_formula_does_not_execute_statements_access_files_or_mutate_inputs(formula):
    with pytest.raises(ValueError):
        evaluate_formula(formula, {"source": np.arange(4.)})


def test_invalid_formula_shapes_names_and_infinities_are_actionable():
    for formula, message in [("source +", "Python syntax"), ("missing * source", "Unknown formula name"),
                             ("source[1:]", "source shape"), ("source / 0", "infinite")]:
        with pytest.raises(ValueError, match=message):
            evaluate_formula(formula, {"source": np.arange(4.)})


def test_transform_is_after_configure_before_filter_in_preview_and_raw_data():
    req = request([1, 2, 3], scaling_mode="multiplier", scale=2,
                  transform_formula="source - source.mean()",
                  data_filter=DataFilterSettings(enabled=True, selection="threshold"))
    np.testing.assert_equal(prepare_data(req).values, [0, 0, 2])
    result = convert_data(req)
    np.testing.assert_equal(result.preview.values, [0, 0, 2])
    np.testing.assert_equal(pd.read_csv(io.BytesIO(result.artifact.data), sep="\t").iloc[:, -1].to_numpy(), [0, 0, 2])
    assert dict(result.metadata)["transform_formula"] == req.transform_formula


def test_modulation_uses_full_profile_context_and_only_applies_selected_regions():
    settings = DataFilterSettings(enabled=True, action="modulate",
        modulation=ProfileModulationSettings(operation="expression", values=np.array([0., 2., 0., 4.]),
                                            expression="source / profile"))
    np.testing.assert_equal(apply_data_filter([1, -4, 3, -8], settings), [1, -2, 3, -2])
    settings = replace(settings, modulation=replace(settings.modulation, expression="source + np.mean(profile)"))
    np.testing.assert_equal(apply_data_filter([1, -4, 3, -8], settings), [1, -2.5, 3, -6.5])


def test_combination_configures_each_source_and_filters_once():
    first = request([1, 2, 3], scaling_mode="multiplier", scale=2,
                    data_filter=DataFilterSettings(enabled=True))
    second = request([3, 1, 2], scaling_mode="multiplier", scale=3)
    result = convert_combined_data((first, second), "source1 - source2")
    np.testing.assert_equal(result.preview.values, [0, 1, 0])
    np.testing.assert_equal(pd.read_csv(io.BytesIO(result.artifact.data), sep="\t").iloc[:, -1].to_numpy(), [0, 1, 0])
    assert dict(result.metadata)["combined_sources"] == "2"
    with pytest.raises(ValueError, match="matching post-Configure"):
        convert_combined_data((first, request([1, 2])), "source1 + source2")


def test_combination_requires_matching_coordinates_and_supports_chromosome_tracks():
    def track(payload):
        return DataConversionRequest(inspect_upload("track.bedgraph", payload), "track", chromosome_scope="all", bin_size=1000)
    first = track(b"chr1\t0\t2000\t2\nchr2\t0\t1000\t3\n")
    second = track(b"chr1\t0\t2000\t4\nchr2\t0\t1000\t5\n")
    result = convert_combined_data((first, second), "source1 * source2")
    frame = pd.read_csv(io.BytesIO(result.artifact.data), sep="\t")
    np.testing.assert_equal(frame.chr1.to_numpy(), [8, 8])
    np.testing.assert_equal(frame.chr2.to_numpy(), [15, np.nan])
    left = DataConversionRequest(inspect_upload("one.csv", b"position,signal\n0,1\n1,2\n"), "x")
    right = DataConversionRequest(inspect_upload("two.csv", b"position,signal\n0,1\n2,2\n"), "x")
    with pytest.raises(ValueError, match="matching regions"):
        convert_combined_data((left, right), "source1 + source2")


@pytest.mark.parametrize("form,values", [("t", [[1., 2., 3.]]), ("xt", [[1., 2.], [3., 4.], [5., 6.]])])
def test_combined_time_and_spacetime_exports_preserve_dimensions(form, values):
    stream = io.BytesIO()
    np.savez(stream, values=values)
    req = DataConversionRequest(inspect_upload("signal.npz", stream.getvalue()), form, target_format="npz")
    result = convert_combined_data((req, req), "source1**2 + source2")
    with np.load(io.BytesIO(result.artifact.data), allow_pickle=False) as archive:
        np.testing.assert_equal(archive["values"], np.square(values) + values)


def test_full_tables_do_not_silently_ignore_an_enabled_transform():
    req = request([1, 2, 3], transform_formula="source * 2")
    with pytest.raises(ValueError, match="Choose a numeric profile"):
        convert_data(replace(req, input_form="table"))
