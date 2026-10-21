"""Editable Mapping downloads run the package without the local web app."""

from __future__ import annotations

import asyncio
import io
import json
import lzma
import re
import subprocess
import sys
import zipfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from app.mapping_code_export import (
    MappingCodeSnapshot,
    build_mapping_code_bundle,
    build_mapping_python,
    snapshot_mapping_code,
)
from dnascape.workflows.mapping_recipe import run_mapping_recipe
from dnascape.workflows.plotting import mapping_plot_options
from dnascape.workflows.recipe_exports import plot_mapping_outputs, save_mapping_outputs
from dnascape.workflows.uploads import inspect_upload


def _plot_settings() -> dict[str, Any]:
    return {
        "plot_type": "Line", "normalization": "None", "title": "Edited title",
        "x_label": "Genomic position", "y_label": "Replication time",
        "color": "#2596be", "palette": "viridis", "histogram_bins": 12,
        "line_width": 3.0, "show_markers": False, "show_colorbar": False,
        "log_y": False, "invert_y": True, "show_grid": False,
        "show_legend": False, "y_limits": None, "aspect_ratio": 1.5, "line_type": "dotted",
    }


def _snapshot() -> MappingCodeSnapshot:
    payload = b"position_kb,first,second\n0,0.001,0.003\n1,0.001,0.003\n2,0.001,0.003\n3,0.001,0.003\n"
    inspection = inspect_upload("profiles.csv", payload, expected_form="x")
    return snapshot_mapping_code(
        method="firing_and_speed_to_timing_deterministic", target_shape=None,
        settings={"resolution": 1.0, "neighbourhood_range": 10.0, "perQ": False},
        setting_controls={
            "n_bins": SimpleNamespace(active=True, override=SimpleNamespace(value=False), control=SimpleNamespace(value=8)),
        },
        inputs={
            "firing_rate": SimpleNamespace(form=SimpleNamespace(value="x"), override=None, surface={
                "inspections": [inspection],
                "selection_values": {"profiles.csv": {"column": "first", "array_key": None}},
            }),
            "fork_speed": SimpleNamespace(form=SimpleNamespace(value="scalar"), override=None, surface={
                "scalar": SimpleNamespace(value=1.4),
            }),
        },
        read_input=lambda _live: pytest.fail("Code materialized a scientific input while snapshotting."),
        region={"start_kb": 0.0, "end_kb": 4.0, "whole_genome": False, "missing_policy": "error"},
        combination_mode="pairwise",
        raw_export={"output": "primary", "filename": "my_result", "file_format": "NPZ",
                    "compress": True, "compression": "xz", "name_style": "short", "per_dataset": {}},
        plotting={"output": "primary", "layout": "Individual", "settings": _plot_settings(), "colors": []},
    )


def test_snapshot_keeps_original_inputs_selections_and_disabled_overrides() -> None:
    snapshot = _snapshot()
    source = snapshot.recipe["inputs"]["firing_rate"]
    assert source["files"][0]["column"] == "first"
    assert source["files"][0]["source_format"] == "csv"
    assert snapshot.files == ((source["files"][0]["path"], inspect_upload(
        "profiles.csv", snapshot.files[0][1], expected_form="x",
    ).content),)
    assert snapshot.recipe["setting_overrides"]["n_bins"] == {"enabled": False, "value": 8}
    assert snapshot.raw_export["file_format"] == "NPZ"
    assert snapshot.raw_export["compression"] == "xz"
    script = build_mapping_python(snapshot).data.decode("utf-8")
    assert "PLOT = False" in script
    assert "EXPORT_RAW = True" in script
    assert "from dnascape.workflows.mapping_recipe import run_mapping_recipe" in script
    assert "def rmap" not in script


def test_snapshot_keeps_automatic_advanced_fields_editable_without_enabling_them() -> None:
    snapshot = snapshot_mapping_code(
        method="timing_and_speed_to_firing_rate", target_shape=None,
        settings={"maxiter": 25, "fit_step": 1.0, "error": .02},
        setting_controls={
            key: SimpleNamespace(
                active=True, override=None, automatic=True, control=SimpleNamespace(value=value),
            ) for key, value in {"space_maxiter": 25, "space_fit_step": 1.0, "error": .02}.items()
        },
        inputs={}, read_input=lambda _: None, region={}, combination_mode="pairwise",
        raw_export={}, plotting={},
    )
    assert snapshot.recipe["settings"] == {"maxiter": 25, "fit_step": 1.0, "error": .02}
    assert snapshot.recipe["setting_overrides"] == {
        "space_maxiter": {"enabled": False, "value": 25},
        "space_fit_step": {"enabled": False, "value": 1.0},
        "error": {"enabled": True, "value": .02},
    }


def test_runnable_bundle_executes_its_own_package_and_inputs_without_web_imports(tmp_path: Path) -> None:
    snapshot = _snapshot()
    artifact = build_mapping_code_bundle(snapshot)
    with zipfile.ZipFile(io.BytesIO(artifact.data)) as archive:
        assert "dnascape/workflows/mapping_recipe.py" in archive.namelist()
        assert not any(name.startswith("app/") for name in archive.namelist())
        archive.extractall(tmp_path)
    runner = """
import importlib.abc
import pathlib
import runpy
import sys
class RejectWebImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'app', 'nicegui'}:
            raise AssertionError('Unexpected web dependency: ' + fullname)
sys.meta_path.insert(0, RejectWebImports())
folder = pathlib.Path(sys.argv[1])
sys.path.insert(0, str(folder))
runpy.run_path(str(folder / 'mapping.py'), run_name='__main__')
import dnascape
assert pathlib.Path(dnascape.__file__).is_relative_to(folder)
"""
    completed = subprocess.run([sys.executable, "-c", runner, str(tmp_path)], cwd=tmp_path,
                               capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    generated = tmp_path / "outputs" / "my_result.npz.xz"
    assert generated.is_file()
    assert not tuple((tmp_path / "outputs").glob("*.html"))
    expected = run_mapping_recipe(snapshot.recipe, base_path=tmp_path)[0].result
    with np.load(io.BytesIO(lzma.decompress(generated.read_bytes())), allow_pickle=False) as data:
        np.testing.assert_array_equal(data["values"], expected.outputs["replication_timing"])
        np.testing.assert_array_equal(data["position_kb"], expected.x_axis)


def test_raw_batch_settings_remain_independent_and_optional_plot_matches_page(tmp_path: Path) -> None:
    snapshot = _snapshot()
    for path, content in snapshot.files:
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    source = snapshot.recipe["inputs"]["fork_speed"]
    source.update(mode="constants", values=[1.0, 2.0])
    results = run_mapping_recipe(snapshot.recipe, base_path=tmp_path)
    options = {"output": "primary", "per_dataset": {
        "mapping-case-1": {"filename": "first", "file_format": "NPZ", "compress": False},
        "mapping-case-2": {"filename": "second", "file_format": "CSV", "compress": True, "compression": "xz"},
    }}
    paths = save_mapping_outputs(results, options, directory=tmp_path / "outputs")
    assert [path.name for path in paths] == ["first.npz", "second.csv.xz"]
    with np.load(paths[0], allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["values"], results[0].result.outputs["replication_timing"])
    assert lzma.decompress(paths[1].read_bytes()).decode("utf-8").startswith("replication_timing\n")

    # Clearing a filename should use the same automatic name as the page.
    from app.export_settings import get_export_settings
    from app.pages.mapping import _mapping_raw_default_filename
    from dnascape.workflows.mapping import MappingComparison

    automatic = save_mapping_outputs(results[:1], {
        "output": "primary", "filename": "  ", "file_format": "NPZ",
        "name_style": get_export_settings().result_name_style,
    }, directory=tmp_path / "automatic")
    assert automatic[0].stem == _mapping_raw_default_filename(results[0], results[0].result.primary_output)

    values = results[0].result.outputs["replication_timing"]
    comparison = replace(results[0], result=replace(results[0].result, comparisons={
        "replication_timing_fit": MappingComparison(
            source_key="replication_timing", fitted_key="replication_timing_fit",
            form="x", observed=values, fitted=values,
        ),
    }))
    automatic = save_mapping_outputs((comparison,), {
        "output": "comparison:replication_timing_fit", "filename": "", "file_format": "CSV",
        "name_style": get_export_settings().result_name_style,
    }, directory=tmp_path / "automatic")
    assert automatic[0].stem == _mapping_raw_default_filename(comparison, "comparison:replication_timing_fit")

    plots = plot_mapping_outputs(results[:1], snapshot.plotting, directory=tmp_path / "outputs")
    expected_specification = mapping_plot_options(
        results[0].result, output_key=results[0].result.primary_output, **snapshot.plotting["settings"],
    )
    html = plots[0].read_text(encoding="utf-8")
    specification = re.search(r"const specification = (.*);\n", html)
    assert specification is not None
    assert json.loads(specification[1]) == expected_specification
    assert '"color": "#2596be"' in specification[1]


@pytest.mark.parametrize("target", ["firing_rate", "fork_speed"])
def test_mapping_code_menu_downloads_current_controls_without_running_map(
    monkeypatch: pytest.MonkeyPatch, target: str,
) -> None:
    from nicegui.elements.upload import Upload
    from nicegui.testing import user_simulation

    from app.export_settings import get_export_settings
    from app.main import index
    from app.pages import mapping as page

    monkeypatch.setattr(get_export_settings(), "dev_mode", True)
    monkeypatch.setattr(get_export_settings(), "raw_compression", "xz")
    snapshots: list[MappingCodeSnapshot] = []

    def record_python(snapshot: MappingCodeSnapshot) -> Any:
        snapshots.append(snapshot)
        return build_mapping_python(snapshot)

    def forbid_mapping(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("Downloading Code started a Mapping run.")

    monkeypatch.setattr(page, "build_mapping_python", record_python)
    monkeypatch.setattr(page, "run_mapping", forbid_mapping)

    async def scenario() -> None:
        async with user_simulation(index) as user:
            def control(marker: str) -> Any:
                assert user.client is not None
                return next(element for element in user.client.elements.values()
                            if marker in element._markers and not element.is_deleted)

            async def set_value(marker: str, value: Any) -> None:
                assert user.client is not None
                with user.client:
                    control(marker).set_value(value)
                await asyncio.sleep(0)
                await asyncio.sleep(0)

            await user.open("/")
            user.find(marker="nav-mapping").click()
            await user.should_see(marker="source-replication-timing-upload", retries=60)
            await set_value("mapping-target", target)
            await set_value("mapping-target-form", "space")
            await control("source-replication-timing-upload").handle_uploads([
                Upload.SmallFileUpload("timing.csv", "text/csv", b"first,second\n8,9\n7,8\n9,10\n")
            ])
            await user.should_see(marker="mapping-input-replication-timing-value-column", retries=40)
            await set_value("mapping-input-replication-timing-value-column", "second")
            await set_value("mapping-region-start", 1.0)
            await set_value("mapping-region-end", 3.0)
            await set_value("mapping-setting-resolution", 0.5)
            constant_key = "fork_speed" if target == "firing_rate" else "firing_rate"
            constant_value = 1.8 if target == "firing_rate" else .0018
            await set_value(f"mapping-input-{constant_key.replace('_', '-')}-constant", constant_value)
            await set_value("mapping-setting-refine-edge-kb", 23.)
            await set_value("mapping-raw-filename", "user_chosen_name")
            await set_value("mapping-raw-file-type", "CSV")
            await set_value("raw-output-compression", False)
            await set_value("mapping-preview-x-label", "Genomic position")
            status = control("mapping-status").text
            button = control("action-map")
            user.find(marker="action-code").click()
            user.find(marker="mapping-code-download-python").click()
            download = await user.download.next(timeout=10.0)
            assert len(user.download.http_responses) == len(snapshots) == 1, user.notify.messages
            snapshot = snapshots[0]
            assert snapshot.recipe["region"]["start_kb"] == 1.0
            assert snapshot.recipe["region"]["end_kb"] == 3.0
            assert "match_smallest_profile" not in snapshot.recipe["region"]
            assert snapshot.recipe["settings"]["resolution"] == 0.5
            assert snapshot.recipe["settings"]["refine_edge_kb"] == 23.
            assert snapshot.recipe["inputs"]["replication_timing"]["files"][0]["column"] == "second"
            assert snapshot.recipe["inputs"][constant_key]["values"] == constant_value
            assert snapshot.raw_export["filename"] == "user_chosen_name"
            assert snapshot.raw_export["file_format"] == "CSV"
            assert snapshot.raw_export["compress"] is False
            assert snapshot.raw_export["compression"] == "xz"
            assert snapshot.plotting["settings"]["x_label"] == "Genomic position"
            assert "PLOT = False" in download.content.decode("utf-8")
            assert control("mapping-status").text == status
            assert control("action-map") is button
            assert button.text == "Map" and button.enabled
            await user.should_not_see(marker="mapping-result-plot")

    asyncio.run(scenario())
