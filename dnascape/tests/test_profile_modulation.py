"""Aligned assay profiles modulate converted data with explicit numeric policies."""

import gzip
import io
from concurrent.futures import CancelledError
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from dnascape.workflows import DataFilterSettings, ProfileModulationSettings
from dnascape.workflows.conversion import DataConversionRequest, convert_data, prepare_data
from dnascape.workflows.filtering import apply_data_filter
from dnascape.workflows.modulation import ModulationProfile
from dnascape.workflows.uploads import inspect_upload


def config(values=None, *, selection="all", **kwargs):
    return DataFilterSettings(enabled=True, selection=selection, action="modulate",
                              modulation=ProfileModulationSettings(values=np.asarray(values) if values is not None else None, **kwargs))


@pytest.mark.parametrize("operation,expected", [
    ("multiply", [8, 60, 160]), ("add", [6, 16, 28]), ("subtract", [2, 4, 12]),
    ("divide", [2, 10/6, 2.5]), ("blend", [3, 8, 14]),
    ("minimum", [2, 6, 8]), ("maximum", [4, 10, 20]),
])
def test_pointwise_operations_have_explicit_signal_profile_direction(operation, expected):
    signal = np.array([4., 10., 20.])
    profile = np.array([2., 6., 8.])
    np.testing.assert_allclose(apply_data_filter(signal, config(profile, operation=operation)), expected)
    np.testing.assert_equal(signal, [4, 10, 20])
    np.testing.assert_equal(profile, [2, 6, 8])


@pytest.mark.parametrize("options,expected", [
    (dict(combination="sum", signal_multiplier=.75, profile_multiplier=.25), [3.5, 9, 17]),
    (dict(combination="sum", profile_multiplier=-2), [0, -2, 4]),
    (dict(combination="multiply", signal_exponent=2, profile_multiplier=.5), [16, 300, 1600]),
    (dict(combination="divide", signal_multiplier=3, profile_exponent=2), [3, 30/36, 60/64]),
    (dict(combination="sum", signal_function="log2", profile_function="ln"), np.log2([4, 10, 20]) + np.log([2, 6, 8])),
    (dict(combination="multiply", signal_exponent=.5, profile_function="log10"), np.sqrt([4, 10, 20]) * np.log10([2, 6, 8])),
])
def test_formula_supports_signed_weights_functions_and_powers(options, expected):
    np.testing.assert_allclose(apply_data_filter([4, 10, 20], config([2, 6, 8], operation="formula", **options)), expected)


def test_formula_domains_zero_divisors_unused_terms_and_selection_are_explicit():
    with pytest.raises(ValueError, match="logarithms require positive"):
        apply_data_filter([1, -2], config([2, 3], operation="formula", signal_function="ln"))
    with pytest.raises(ValueError, match="exponent is undefined"):
        apply_data_filter([-1, 2], config([2, 3], operation="formula", signal_exponent=.5))
    with pytest.raises(ValueError, match="infinite"):
        apply_data_filter([1000], config([2], operation="formula", signal_function="exp"))
    options = dict(operation="formula", combination="divide", profile_function="ln")
    with pytest.raises(ValueError, match="zero divisors"):
        apply_data_filter([4, 8], config([1, np.e], **options))
    np.testing.assert_allclose(apply_data_filter([4, 8], config([1, np.e], zero_policy="keep", **options)), [4, 8])
    np.testing.assert_equal(apply_data_filter([1, 2], config([np.nan, np.nan], operation="formula", combination="sum", profile_multiplier=0, scaling="mean")), [1, 2])
    selected = replace(config([1, 2], operation="formula", signal_function="ln", selection="threshold"), comparison="above", threshold=0)
    np.testing.assert_allclose(apply_data_filter([-2, 4], selected), [-2, np.log(4)*2])
    for key in ("signal_multiplier", "profile_exponent"):
        with pytest.raises(ValueError, match="must be finite"):
            apply_data_filter([1], config([2], operation="formula", **{key: np.nan}))


def test_selection_limits_changes_but_size_check_always_uses_the_full_converted_profile():
    np.testing.assert_equal(apply_data_filter([4, -2, 8, -3], config([2, 3, 4, 5], selection="threshold")), [4, -6, 8, -15])
    # A selection with no affected bins must not hide a shape mismatch.
    with pytest.raises(ValueError, match="does not match the post-Configure data shape"):
        apply_data_filter([1, 2, 3], config([10, 20], selection="threshold"))


@pytest.mark.parametrize("shape", [(3,), (1, 3), (3, 1)])
def test_vector_orientation_is_compatible_but_does_not_broadcast(shape):
    signal = np.array([2., 4., 8.]).reshape(shape)
    result = apply_data_filter(signal, config([1., 2., 4.], operation="divide"))
    assert result.shape == signal.shape
    np.testing.assert_equal(result.reshape(-1), [2, 2, 2])
    with pytest.raises(ValueError, match="does not match"):
        apply_data_filter(signal, config([2]))


def test_matrices_require_matching_dimensions_and_apply_each_time_slice_independently():
    signal = np.arange(1., 9.).reshape(4, 2)
    profile = np.arange(2., 10.).reshape(4, 2)
    np.testing.assert_equal(apply_data_filter(signal, config(profile)), signal * profile)
    for wrong in (profile.T, profile.reshape(-1), profile[:, 0]):
        with pytest.raises(ValueError, match="does not match"):
            apply_data_filter(signal, config(wrong))


def test_blend_endpoints_do_not_mix_in_missing_values_from_an_unused_input():
    np.testing.assert_equal(apply_data_filter([1, 2, 3], config([np.nan, 4, 6], operation="blend", weight=0)), [1, 2, 3])
    np.testing.assert_equal(apply_data_filter([1, np.nan, 3], config([2, 4, 6], operation="blend", weight=1)), [2, 4, 6])


@pytest.mark.parametrize("scaling,expected", [
    ("none", [2, 4, 6]), ("mean", [.5, 1, 1.5]), ("maxabs", [1/3, 2/3, 1]), ("interval", [0, .5, 1]),
])
def test_profile_scaling_precedes_combining_and_uses_the_whole_auxiliary_profile(scaling, expected):
    np.testing.assert_allclose(apply_data_filter([1, 1, 1], config([2, 4, 6], scaling=scaling)), expected)


@pytest.mark.parametrize("policy,expected", [("keep", [2, 4, 2]), ("nan", [2, np.nan, 2])])
def test_zero_divisors_have_no_implicit_infinities(policy, expected):
    np.testing.assert_equal(apply_data_filter([2, 4, 8], config([1, 0, 4], operation="divide", zero_policy=policy)), expected)
    with pytest.raises(ValueError, match="zero divisors"):
        apply_data_filter([2, 4, 8], config([1, 0, 4], operation="divide"))
    # A zero outside the selected region is not a division error.
    np.testing.assert_equal(apply_data_filter([2, -4, 8], config([0, 2, 0], operation="divide", selection="threshold")), [2, -2, 8])


@pytest.mark.parametrize("policy,expected", [("zero", [2, 0, 12]), ("nan", [2, np.nan, 12])])
def test_missing_modulation_values_follow_the_selected_policy(policy, expected):
    np.testing.assert_equal(apply_data_filter([2, 4, 6], config([1, np.nan, 2], missing_policy=policy)), expected)
    with pytest.raises(ValueError, match="missing values in the selected region"):
        apply_data_filter([2, 4, 6], config([1, np.nan, 2]))


@pytest.mark.parametrize("boundary", ["reflect", "wrap", "edge"])
@pytest.mark.parametrize("size", [1, 4, 5])
def test_normalized_convolution_preserves_constant_signals_and_output_size(boundary, size):
    signal = np.full(size, 7.)
    kernel = np.arange(1., size + 1)
    result = apply_data_filter(signal, config(kernel, operation="convolve", boundary=boundary))
    assert result.shape == signal.shape
    np.testing.assert_allclose(result, signal)


@pytest.mark.parametrize("size", [4, 5])
def test_zero_padded_convolution_matches_standard_same_mode_for_odd_and_even_sizes(size):
    signal = np.arange(1., size+1)
    kernel = np.arange(size, 0, -1.)
    result = apply_data_filter(signal, config(kernel, operation="convolve", boundary="zero", normalize_kernel=False))
    np.testing.assert_allclose(result, np.convolve(signal, kernel, mode="same"))


def test_periodic_convolution_has_no_privileged_end_and_preserves_the_unselected_bins():
    signal = np.array([1., 7., 3., 2., 9., 4.])
    settings = config([1., 2., 0., 3., 0., 1.], operation="convolve", boundary="wrap")
    expected = apply_data_filter(signal, settings)
    np.testing.assert_allclose(apply_data_filter(np.roll(signal, 2), settings), np.roll(expected, 2))
    selected_settings = replace(settings, selection="threshold", comparison="above", threshold=5)
    actual = apply_data_filter(signal, selected_settings)
    np.testing.assert_allclose(actual, np.where(signal > 5, expected, signal))


def test_convolution_missing_values_zero_sum_and_irregular_coordinates_are_explicit():
    with pytest.raises(ValueError, match="finite values in both profiles"):
        apply_data_filter([1, np.nan, 3], config([1, 1, 1], operation="convolve"))
    result = apply_data_filter([1, np.nan, 3], config([1, np.nan, 1], operation="convolve", missing_policy="zero", boundary="zero"))
    np.testing.assert_allclose(result, np.convolve([1, 0, 3], [.5, 0, .5], "same"))
    with pytest.raises(ValueError, match="non-zero sum"):
        apply_data_filter([1, 2, 3], config([1, -1, 0], operation="convolve"))
    with pytest.raises(ValueError, match="uniformly spaced"):
        apply_data_filter([1, 2, 3], config([1, 1, 1], operation="convolve"), positions=[0, 1, 3])


@pytest.mark.parametrize("format_name", ["tsv", "csv", "txt", "npy", "npz", "tsv.gz"])
def test_uploaded_assay_profiles_modulate_after_configure_in_preview_prepared_data_and_export(format_name):
    stream = io.BytesIO()
    if format_name == "npy":
        np.save(stream, np.array([2., 4., 8.]))
        payload = stream.getvalue()
    elif format_name == "npz":
        np.savez(stream, values=[2., 4., 8.])
        payload = stream.getvalue()
    else:
        payload = b"value\n2\n4\n8\n"
        if format_name.endswith(".gz"):
            payload = gzip.compress(payload)
    request = DataConversionRequest(
        inspection=inspect_upload("source.csv", b"position,signal\n0,1\n1,2\n2,3\n"), input_form="x",
        scaling_mode="multiplier", scale=3,
        data_filter=config(upload=inspect_upload(f"assay.{format_name}", payload), operation="add"),
    )
    prepared = prepare_data(request)
    np.testing.assert_equal(prepared.values, [5, 10, 17])
    result = convert_data(request)
    np.testing.assert_equal(result.preview.values, [5, 10, 17])
    np.testing.assert_equal(pd.read_csv(io.BytesIO(result.artifact.data), sep="\t").iloc[:, -1].to_numpy(), [5, 10, 17])
    metadata = dict(result.metadata)
    assert metadata["modulation_operation"] == "add" and len(metadata["modulation_source_sha256"]) == 64


def test_matching_shape_does_not_hide_misaligned_explicit_coordinates():
    upload = inspect_upload("atac.tsv", b"position_kb\tvalue\n10\t2\n11\t3\n12\t4\n")
    with pytest.raises(ValueError, match="coordinates do not match"):
        apply_data_filter([1, 2, 3], config(upload=upload), positions=[0, 1, 2])
    np.testing.assert_equal(apply_data_filter([1, 2, 3], config(upload=upload), positions=[10, 11, 12]), [2, 6, 12])


@pytest.mark.parametrize("format_name", ["npz", "tsv", "long.csv"])
def test_chromosomes_match_by_name_and_keep_individual_lengths(format_name):
    stream = io.BytesIO()
    if format_name == "npz":
        np.savez(stream, chr1=[2., 3., 4.], chr2=[10., 20.])
        payload = stream.getvalue()
    elif format_name == "tsv":
        payload = b"chr1\tchr2\n2\t10\n3\t20\n4\t\n"
    else:
        payload = b"chrom,value\nchr1,2\nchr1,3\nchr1,4\nchr2,10\nchr2,20\n"
    request = DataConversionRequest(
        inspection=inspect_upload("source.bedgraph", b"chr1\t0\t3000\t1\nchr2\t0\t2000\t2\n"),
        input_form="track", bin_size=1000, chromosome_scope="all", target_format="npz",
        data_filter=config(upload=inspect_upload(f"assay.{format_name}", payload)),
    )
    with np.load(io.BytesIO(convert_data(request).artifact.data)) as archive:
        np.testing.assert_equal(archive["chr1"], [2, 3, 4])
        np.testing.assert_equal(archive["chr2"], [20, 40])
    with pytest.raises(ValueError, match="every converted chromosome"):
        apply_data_filter([1, 2], config(upload=inspect_upload(f"assay.{format_name}", payload)), chromosome="chr3")


def test_mismatched_chromosome_and_matrix_sizes_are_rejected_before_writing():
    stream = io.BytesIO()
    np.savez(stream, values=np.ones((3, 2)))
    request = DataConversionRequest(inspection=inspect_upload("source.npz", stream.getvalue()), input_form="xt", data_filter=config([1, 2, 3]))
    with pytest.raises(ValueError, match="post-Configure data shape"):
        convert_data(request)


def test_advanced_settings_are_inert_for_other_filter_operations():
    request = DataFilterSettings(enabled=True, action="replace", modulation=ProfileModulationSettings(operation="unfinished"))
    np.testing.assert_equal(apply_data_filter([1, -2], request), [1, 0])
    with pytest.raises(ValueError, match="Upload a modulation profile"):
        apply_data_filter([1, -2], config())
    with pytest.raises(ValueError, match="between zero and one"):
        apply_data_filter([1, 2], config([1, 2], operation="blend", weight=2))
    with pytest.raises(ValueError, match="infinite"):
        apply_data_filter([1e300], config([1e300]))
    def cancelled():
        raise CancelledError()
    with pytest.raises(CancelledError):
        apply_data_filter([1, 2], config([1, 2], operation="convolve"), cancel=cancelled)
