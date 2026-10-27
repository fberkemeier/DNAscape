"""Typed genomic intervals use the same masks as uploaded interval profiles."""

from dataclasses import replace

import numpy as np
import pytest

from dnascape.workflows.filtering import (
    DataFilterSettings, FilterProfile, apply_data_filter, parse_position_ranges, resolve_filter,
)


@pytest.mark.parametrize("text,expected", [
    ("(1000,2000)", [[1000., 2000.]]),
    (" (1000,2000),\n (4000,5000), ", [[1000., 2000.], [4000., 5000.]]),
    ("(.5,1.5), (2e3,3.5e3)", [[.5, 1.5], [2000., 3500.]]),
    ("(4000,5000),(1000,3000),(2000,4000)", [[4000., 5000.], [1000., 3000.], [2000., 4000.]]),
])
def test_parse_ranges_preserves_numeric_coordinates(text, expected):
    profile = parse_position_ranges(text)
    assert profile.layout == "intervals" and profile.units == "bp"
    np.testing.assert_array_equal(profile.values, expected)


@pytest.mark.parametrize("text", [
    "(1,2)(3,4)", "1,2", "[(1,2)]", "(1,2),unfinished", "(1,2,3)",
    "(True,2)", "(nan,2)", "(1,inf)", "(1,1e309)", "(-1,2)", "(2,2)",
    "(3,2)", "(__import__('os'),2)", "(1+1,3)", pytest.param("x" * 65537, id="oversized"),
])
def test_invalid_or_executable_ranges_are_rejected(text):
    with pytest.raises(ValueError):
        parse_position_ranges(text)


@pytest.mark.parametrize("units,text", [("bp", "(11000,13000),(14000,15000)"),
                                        ("kb", "(11,13),(14,15)")])
@pytest.mark.parametrize("complement", [False, True])
def test_ranges_respect_units_nonzero_start_end_and_complement(units, text, complement):
    values = np.arange(6.) + 1
    positions = np.arange(10., 16.)
    config = DataFilterSettings(enabled=True, selection="ranges", position_ranges=text,
                                profile_units=units, value=99., complement=complement)
    original = values.copy()
    actual = apply_data_filter(values, config, positions=positions)
    selected = np.isin(positions, [11., 12., 14.])
    expected = np.where(~selected if complement else selected, 99., values)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(values, original)
    normalized = resolve_filter(config)
    assert normalized.selection == "profile" and normalized.profile.layout == "intervals"
    np.testing.assert_array_equal(actual, apply_data_filter(values, normalized, positions=positions))


def test_empty_and_disabled_ranges_are_inert_even_with_complement():
    assert parse_position_ranges("  \n ") is None
    values = np.array([1., -2., np.nan])
    for config in (DataFilterSettings(enabled=True, selection="ranges", complement=True),
                   DataFilterSettings(selection="ranges", position_ranges="unfinished", complement=True)):
        np.testing.assert_equal(apply_data_filter(values, config), values)


def test_ranges_reuse_native_bin_overlap_and_do_not_use_a_stale_upload():
    config = DataFilterSettings(enabled=True, selection="ranges", position_ranges="(500,1500)", value=9,
                                profile=FilterProfile(np.array([2.]), "positions", "kb"))
    positions = np.arange(3.) + .5
    bins = np.column_stack((np.arange(3.), np.arange(1., 4.)))
    np.testing.assert_equal(apply_data_filter([1, 2, 3], config, positions=positions, intervals=bins), [9, 9, 3])


def test_mapping_promotes_time_target_for_typed_spatial_ranges():
    from dnascape.workflows.mapping import MappingRequest, validate_mapping_request
    from dnascape.workflows.mapping_catalog import mapping_route
    from dnascape.workflows.mapping_filter import MappingFilterSettings, compatible_filter_dependence

    assert compatible_filter_dependence("time", "t", "ranges") == "spacetime"
    request = MappingRequest(route=mapping_route("timing_and_speed_to_time_initiation_rate"),
        inputs={"replication_timing": np.arange(20., 36.), "fork_speed": 1.4},
        input_forms={"replication_timing": "x", "fork_speed": "scalar"},
        fit_filter=MappingFilterSettings(DataFilterSettings(enabled=True, selection="ranges",
            position_ranges="(1000,3000)"), relative_to="unused", export=True))
    prepared = validate_mapping_request(request)
    assert prepared.route.output_dependence == "spacetime"
    assert prepared.fit_filter.regions.profile.units == "bp"


def test_simulation_ranges_expand_temporal_inputs_and_export_each_mask():
    from dnascape.workflows.simulation_filters import apply_simulation_filters

    config = DataFilterSettings(enabled=True, selection="ranges", position_ranges="(11000,13000)", value=.5)
    values, state = apply_simulation_filters(
        {"initiation_rate": .1, "fork_speed": np.array([[1., 2., 3.]])},
        {"initiation_rate": replace(config, value=0.), "fork_speed": config},
        start=10., end=15., resolution_x=1., resolution_t=5., n_bins=5,
        exports=frozenset({"initiation_rate", "fork_speed"}))
    np.testing.assert_equal(values["initiation_rate"], [.1, 0, 0, .1, .1])
    assert values["fork_speed"].shape == (5, 3)
    np.testing.assert_equal(values["fork_speed"][1:3], .5)
    for selection in state.selections.values():
        np.testing.assert_array_equal(selection.positions(), [11000, 12000])


def test_mapping_applies_and_exports_the_typed_mask():
    from dnascape.workflows.mapping import MappingRequest, run_mapping
    from dnascape.workflows.mapping_catalog import mapping_route
    from dnascape.workflows.mapping_filter import MappingFilterSettings

    timing = np.arange(20., 36.)
    request = MappingRequest(route=mapping_route("timing_and_speed_to_initiation_rate"),
        inputs={"replication_timing": timing, "fork_speed": 1.4},
        input_forms={"replication_timing": "x", "fork_speed": "scalar"},
        settings={"maxiter": 1, "initial_initiation_rate": .003},
        initiation_rate_per_bin=True,
        fit_filter=MappingFilterSettings(DataFilterSettings(enabled=True, selection="ranges",
            position_ranges="(1000,3000)", value=0.), export=True))
    result = run_mapping(request)
    np.testing.assert_array_equal(result.outputs["initiation_rate"][1:3], 0.)
    np.testing.assert_array_equal(result.outputs["filter_positions"], [1000, 2000])
    np.testing.assert_array_equal(timing, np.arange(20., 36.))
