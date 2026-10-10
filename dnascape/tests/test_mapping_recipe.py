"""Mapping recipes retain the scientific and import behaviour of the page."""

from __future__ import annotations

import asyncio
import copy
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from dnascape.workflows import mapping_recipe
from dnascape.workflows.mapping import MappingRequest, run_mapping
from dnascape.workflows.mapping_catalog import mapping_route
from dnascape.workflows.mapping_recipe import prepare_mapping_recipe, run_mapping_recipe
from dnascape.workflows.uploads import inspect_upload


def _recipe() -> dict[str, Any]:
    return {
        "method": "firing_and_speed_to_timing_deterministic",
        "target_shape": None,
        "settings": {"resolution": 1.0, "neighbourhood_range": 10.0, "perQ": False},
        "region": {"start_kb": 20.0, "end_kb": 24.0, "whole_genome": False, "missing_policy": "error"},
        "combination_mode": "pairwise",
        "inputs": {
            "firing_rate": {"enabled": True, "form": "scalar", "mode": "constant", "values": 0.001},
            "fork_speed": {"enabled": True, "form": "scalar", "mode": "constant", "values": 1.4},
        },
    }


def _assert_same_results(actual: Any, expected: Any) -> None:
    assert actual.method == expected.method
    assert actual.primary_output == expected.primary_output
    assert actual.output_forms == expected.output_forms
    assert actual.assumptions == expected.assumptions
    np.testing.assert_array_equal(actual.x_axis, expected.x_axis)
    np.testing.assert_array_equal(actual.time_axis, expected.time_axis)
    assert actual.outputs.keys() == expected.outputs.keys()
    for key in actual.outputs:
        np.testing.assert_array_equal(actual.outputs[key], expected.outputs[key])


def test_recipe_matches_the_exact_page_mapping_request() -> None:
    recipe = _recipe()
    expected = run_mapping(MappingRequest(
        route=mapping_route(recipe["method"]),
        inputs={"firing_rate": 0.001, "fork_speed": 1.4},
        input_forms={"firing_rate": "scalar", "fork_speed": "scalar"},
        settings=recipe["settings"],
        region_start_kb=20.0,
        region_end_kb=24.0,
        resolution_x=1.0,
    ))
    progress: list[float] = []
    completed = run_mapping_recipe(recipe, report=lambda value, _message: progress.append(value))

    assert len(completed) == 1
    assert completed[0].key == "mapping-case-1"
    _assert_same_results(completed[0].result, expected)
    assert progress[-1] == 1.0
    assert progress == sorted(progress)


@pytest.mark.parametrize("whole_region", [False, True])
def test_recipe_matches_shorter_firing_profile_at_chromosome_coordinates(
    tmp_path: Path, whole_region: bool,
) -> None:
    timing = np.linspace(20., 60., 40)
    firing = np.linspace(.001, .002, 10)
    np.save(tmp_path / "timing.npy", timing)
    np.save(tmp_path / "firing.npy", firing)
    recipe = _recipe()
    recipe["method"] = "timing_and_firing_to_spatial_fork_speed"
    recipe["target_shape"] = "best"
    recipe["settings"] = {"resolution": 1., "maxiter": 2, "smoothw": None, "refine_edge_kb": 3.}
    recipe["region"].update(start_kb=24., end_kb=30., whole_genome=whole_region)
    recipe["inputs"] = {
        "replication_timing": {"form": "x", "mode": "upload", "files": [{"path": "timing.npy"}]},
        "firing_rate": {"form": "x", "mode": "upload", "files": [{"path": "firing.npy"}]},
    }
    (_, request), = prepare_mapping_recipe(recipe, base_path=tmp_path)
    assert request.settings["refine_edge_kb"] == 3.
    expected = run_mapping(MappingRequest(
        route=mapping_route(recipe["method"]),
        inputs={"replication_timing": timing, "firing_rate": firing},
        input_forms={"replication_timing": "x", "firing_rate": "x"},
        settings=recipe["settings"], target_shape="best",
        region_start_kb=24., region_end_kb=30., resolution_x=1.,
        whole_genome=whole_region,
    ))
    (completed,) = run_mapping_recipe(recipe, base_path=tmp_path)
    _assert_same_results(completed.result, expected)
    expected_axis = np.arange(10.) if whole_region else np.arange(24., 30.)
    expected_timing = timing[:10] if whole_region else timing[24:30]
    np.testing.assert_array_equal(completed.result.x_axis, expected_axis)
    np.testing.assert_array_equal(completed.result.comparisons["replication_timing_fit"].observed, expected_timing)

    recipe["region"].update(whole_genome=False, end_kb=35.)
    with pytest.raises(ValueError, match="choose a width of at most 10 kb"):
        run_mapping_recipe(recipe, base_path=tmp_path)


@pytest.mark.parametrize("mode", ("pairwise", "all"))
def test_constants_match_page_combination_order_names_and_override_controls(mode: str) -> None:
    from app.pages.mapping import _plan_mapping_cases, _PreparedBatchInput

    recipe = _recipe()
    recipe["combination_mode"] = mode
    recipe["inputs"]["firing_rate"].update(mode="constants", values=[0.001, 0.002])
    recipe["inputs"]["fork_speed"].update(mode="constants", values=[1.0, 1.5])
    recipe["setting_overrides"] = {
        "resolution": {"enabled": True, "value": 2.0},
        "perQ": {"enabled": False, "value": True},
    }
    recipe["inputs"]["unused"] = {"enabled": False, "form": "x", "mode": "upload", "files": []}
    expected_cases = _plan_mapping_cases((
        _PreparedBatchInput("firing_rate", "Firing rate", "scalar", (0.001, 0.002), (False, False)),
        _PreparedBatchInput("fork_speed", "Fork speed", "scalar", (1.0, 1.5), (False, False)),
    ), target_label="Replication timing", combination_mode=mode)
    before = copy.deepcopy(recipe)
    prepared = prepare_mapping_recipe(recipe)

    assert recipe == before
    assert len(prepared) == (4 if mode == "all" else 2)
    for (case, request), expected in zip(prepared, expected_cases, strict=True):
        assert case.key == expected.key
        assert case.name == expected.name
        assert case.short_name == expected.short_name
        assert case.curve_label == expected.curve_label
        assert case.export_stem == expected.export_stem
        assert case.short_export_stem == expected.short_export_stem
        assert case.item_indices == expected.item_indices
        assert request.resolution_x == 2.0
        assert request.settings["resolution"] == 2.0
        assert "perQ" not in request.settings
        assert "unused" not in request.inputs


def _page_import(
    recipe: dict[str, Any],
    key: str,
    base_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Any:
    """Use the existing live import reader with lightweight control values."""
    from app.pages import mapping as page

    async def io_bound(function: Any, *args: Any, **kwargs: Any) -> Any:
        return function(*args, **kwargs)

    monkeypatch.setattr(page.run, "io_bound", io_bound)
    source = recipe["inputs"][key]
    route = mapping_route(recipe["method"])
    spec = next(item for item in route.inputs if item.key == key)
    inspections = [
        inspect_upload(file["name"], (base_path / file["path"]).read_bytes(), expected_form=source["form"])
        for file in source["files"]
    ]
    surface = {
        "inspections": inspections,
        "selection_values": {
            file["name"]: {"column": file.get("column"), "array_key": file.get("array_key")}
            for file in source["files"]
        },
    }
    if source.get("track") is not None:
        surface["track"] = {key: SimpleNamespace(value=value) for key, value in source["track"].items()}
    live = page._MappingInputControl(
        spec, SimpleNamespace(value=source["form"]), lambda: None, None, surface,
    )
    return asyncio.run(page._read_mapping_input_batch_async(
        key, live, missing_policy=recipe["region"]["missing_policy"],
        region_start_kb=recipe["region"]["start_kb"],
        region_end_kb=recipe["region"]["end_kb"],
        resolution_x=recipe["settings"]["resolution"],
        whole_genome=recipe["region"]["whole_genome"],
    ))


def test_table_columns_missing_values_and_file_names_match_live_page(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.pages.mapping import _plan_mapping_cases, _PreparedBatchInput

    (tmp_path / "profile.csv").write_text("position_kb,a,b\n0,0.001,0.002\n1,,0.003\n2,0.003,0.004\n3,0.004,0.005\n")
    recipe = _recipe()
    recipe["region"].update(start_kb=0.0, end_kb=4.0, missing_policy="interpolate")
    recipe["inputs"]["firing_rate"] = {
        "form": "x", "mode": "upload",
        "files": [{"path": "profile.csv", "name": "source.csv", "column": "__all__", "array_key": None}],
    }
    page_input = _page_import(recipe, "firing_rate", tmp_path, monkeypatch)
    page_cases = _plan_mapping_cases((
        page_input, _PreparedBatchInput("fork_speed", "Fork speed", "scalar", (1.4,), (False,)),
    ), target_label="Replication timing")
    prepared = prepare_mapping_recipe(recipe, base_path=tmp_path)

    assert len(prepared) == len(page_input.values) == 2
    for (case, request), expected_case, values in zip(prepared, page_cases, page_input.values, strict=True):
        assert case.name == expected_case.name
        assert case.export_stem == expected_case.export_stem
        assert case.short_export_stem == expected_case.short_export_stem
        np.testing.assert_array_equal(request.inputs["firing_rate"], values)
        assert request.prewindowed_inputs == frozenset()
        expected_result = run_mapping(MappingRequest(
            route=request.route, inputs={"firing_rate": values, "fork_speed": 1.4},
            input_forms=request.input_forms, settings=request.settings,
            region_start_kb=0, region_end_kb=4, resolution_x=1,
        ))
        _assert_same_results(run_mapping(request), expected_result)


def test_numpy_array_selection_can_be_changed_without_reexporting(tmp_path: Path) -> None:
    np.savez(tmp_path / "profiles.npz", first=np.full(4, 0.001), second=np.full(4, 0.003))
    recipe = _recipe()
    recipe["region"].update(start_kb=0.0, end_kb=4.0)
    file = {"path": "profiles.npz", "name": "profiles.npz", "array_key": "first"}
    recipe["inputs"]["firing_rate"] = {"form": "x", "mode": "upload", "files": [file]}
    first = run_mapping_recipe(recipe, base_path=tmp_path)[0].result
    file["array_key"] = "second"
    second = run_mapping_recipe(recipe, base_path=tmp_path)[0].result

    assert np.all(second.outputs["replication_timing"] < first.outputs["replication_timing"])
    np.testing.assert_array_equal(second.x_axis, first.x_axis)


@pytest.mark.parametrize("whole_genome", (False, True))
def test_track_region_binning_and_prewindowed_flag_match_live_page(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    whole_genome: bool,
) -> None:
    (tmp_path / "profile.bedgraph").write_text(
        "chr1\t0\t2000\t0.001\nchr1\t2000\t4000\t0.002\nchr1\t4000\t6000\t0.003\n"
    )
    recipe = _recipe()
    recipe["region"].update(start_kb=2.0, end_kb=6.0, whole_genome=whole_genome)
    recipe["inputs"]["firing_rate"] = {
        "form": "x", "mode": "upload",
        "files": [{"path": "profile.bedgraph", "name": "profile.bedgraph"}],
        "track": {
            "chromosome": "chr1", "start": 0, "end": 2000, "bin_size": 2000,
            "aggregation": "mean", "value_column": None, "missing": "zero",
            "fill_value": 0.0, "interpolation": "none", "smooth_window": 1, "normalize": "none",
        },
    }
    page_input = _page_import(recipe, "firing_rate", tmp_path, monkeypatch)
    (_case, request), = prepare_mapping_recipe(recipe, base_path=tmp_path)

    np.testing.assert_array_equal(request.inputs["firing_rate"], page_input.values[0])
    assert ("firing_rate" in request.prewindowed_inputs) is page_input.prewindowed[0]
    assert page_input.values[0].size == (6 if whole_genome else 4)
    result = run_mapping_recipe(recipe, base_path=tmp_path)[0].result
    assert result.x_axis[0] == (0.0 if whole_genome else 2.0)


def test_prepared_bundle_retains_intrinsic_grid_for_accompanying_tracks(
    tmp_path: Path,
) -> None:
    recipe = _recipe()
    recipe["method"] = "fork_density_bundle_to_coalescence_events_xt_mean_field"
    bundle_path = tmp_path / "prepared.npz"
    np.savez(bundle_path, x_array=np.asarray([10.0, 12.0, 14.0]), time_grid=np.asarray([0.0, 1.0]),
             right_fork_density_xt=np.ones((3, 2)), left_fork_density_xt=np.ones((3, 2)),
             replicated_fraction_xt=np.ones((3, 2)))
    route = mapping_route(recipe["method"])
    bundle_spec = next(item for item in route.inputs if item.prepared_result_only)
    recipe["inputs"] = {bundle_spec.key: {"form": "bundle", "mode": "prepared", "prepared_path": "prepared.npz"}}
    (tmp_path / "speed.bedgraph").write_text("chr1\t0\t16000\t1.4\n")
    recipe["inputs"]["fork_speed"] = {
        "form": "x", "mode": "upload",
        "files": [{"path": "speed.bedgraph", "name": "speed.bedgraph", "source_format": "bedgraph"}],
        "track": {"chromosome": "chr1", "missing": "zero"},
    }
    recipe["region"].update(start_kb=10.0, end_kb=16.0)
    (_case, request), = prepare_mapping_recipe(recipe, base_path=tmp_path)

    assert request.resolution_x == 2.0
    np.testing.assert_array_equal(request.inputs[bundle_spec.key]["x_array"], [10.0, 12.0, 14.0])
    np.testing.assert_array_equal(request.inputs[bundle_spec.key]["time_grid"], [0.0, 1.0])
    np.testing.assert_array_equal(request.inputs["fork_speed"], [1.4, 1.4, 1.4])
    assert request.prewindowed_inputs == frozenset({"fork_speed"})


def test_track_column_choices_require_a_common_value_column_like_the_page(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "profile.bedgraph").write_text("chr1\t0\t4000\t0.001\n")
    recipe = _recipe()
    recipe["region"].update(start_kb=0.0, end_kb=4.0)
    recipe["inputs"]["firing_rate"] = {
        "form": "x", "mode": "upload",
        "files": [
            {"path": "profile.bedgraph", "name": "first.bedgraph", "column": "value"},
            {"path": "profile.bedgraph", "name": "second.bedgraph", "column": "other"},
        ],
        "track": {
            "chromosome": "chr1", "start": None, "end": None, "bin_size": 1000,
            "aggregation": "mean", "value_column": None, "missing": "zero",
            "fill_value": 0.0, "interpolation": "none", "smooth_window": 1, "normalize": "none",
        },
    }
    with pytest.raises(ValueError, match="same bedGraph value column"):
        _page_import(recipe, "firing_rate", tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="same bedGraph value column"):
        prepare_mapping_recipe(recipe, base_path=tmp_path)


def test_invalid_batch_is_validated_before_any_mapping_starts(monkeypatch: pytest.MonkeyPatch) -> None:
    recipe = _recipe()
    recipe["inputs"]["fork_speed"].update(mode="constants", values=[1.4, 1.5])
    calls: list[MappingRequest] = []
    validated: list[MappingRequest] = []

    def validate_second_fails(request: MappingRequest, *_args: Any) -> None:
        validated.append(request)
        if len(validated) == 2:
            raise ValueError("Second request cannot be mapped.")

    def forbidden_run(request: MappingRequest, *_args: Any) -> Any:
        calls.append(request)
        raise AssertionError("Mapping began before the batch was fully validated.")

    monkeypatch.setattr(mapping_recipe, "run_mapping", forbidden_run)
    monkeypatch.setattr(mapping_recipe, "validate_mapping_request", validate_second_fails)
    with pytest.raises(ValueError, match="Second request"):
        run_mapping_recipe(recipe)
    assert len(validated) == 2
    assert calls == []


def test_recipe_can_run_without_app_or_nicegui_imports(tmp_path: Path) -> None:
    recipe_path = tmp_path / "recipe.json"
    recipe_path.write_text(json.dumps(_recipe()), encoding="utf-8")
    script = """
import importlib.abc
import json
import sys
class RejectWebImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'app', 'nicegui'}:
            raise AssertionError('Unexpected web dependency: ' + fullname)
sys.meta_path.insert(0, RejectWebImports())
from dnascape.workflows.mapping_recipe import run_mapping_recipe
with open(sys.argv[1], encoding='utf-8') as source:
    recipe = json.load(source)
assert run_mapping_recipe(recipe)[0].result.outputs['replication_timing'].size == 4
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(recipe_path)], capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
