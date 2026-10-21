"""Capture Mapping controls and build portable, editable Python downloads."""

from __future__ import annotations

import io
import math
import pprint
import zipfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path, PurePath
from typing import Any

import numpy as np

from app.simulation_exports import ExportArtifact

MAX_CODE_INPUT_BYTES = 256 * 1024 * 1024


@dataclass(frozen=True)
class MappingCodeSnapshot:
    recipe: dict[str, Any]
    raw_export: dict[str, Any]
    plotting: dict[str, Any]
    files: tuple[tuple[str, bytes | Mapping[str, np.ndarray]], ...]


def _value(control: Any, default: Any = None) -> Any:
    return getattr(control, "value", default)


def _literal(value: Any) -> Any:
    """Keep generated values portable and readable without executable reprs."""
    if isinstance(value, np.ndarray):
        return _literal(value.tolist())
    if isinstance(value, np.generic):
        return _literal(value.item())
    if isinstance(value, Mapping):
        return {str(key): _literal(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_literal(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        # Scientific nullable settings use None; infinity remains explicit in
        # generated Python through the `inf` binding below.
        if math.isnan(value):
            raise ValueError("Mapping settings cannot contain NaN.")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Cannot export a Mapping setting of type {type(value).__name__}.")


def snapshot_mapping_code(
    *,
    method: str,
    target_shape: str | None,
    settings: Mapping[str, Any],
    setting_controls: Mapping[str, Any],
    inputs: Mapping[str, Any],
    read_input: Callable[[Any], Any],
    region: Mapping[str, Any],
    combination_mode: str,
    raw_export: Mapping[str, Any],
    plotting: Mapping[str, Any],
) -> MappingCodeSnapshot:
    """Read widgets synchronously, before archive work yields to the UI.

    Original uploads are retained so changing regions, columns or track
    processing in the downloaded script actually rereads the original data.
    """
    members: list[tuple[str, bytes | Mapping[str, np.ndarray]]] = []
    input_recipes: dict[str, Any] = {}
    for index, (key, live) in enumerate(inputs.items(), start=1):
        form = str(_value(live.form))
        surface = live.surface
        enabled = live.override is None or bool(_value(live.override))
        item: dict[str, Any] = {"enabled": enabled, "form": form}
        if surface.get("prepared_only"):
            item["mode"] = "prepared"
            item["prepared_path"] = None
            if enabled:
                bundle = read_input(live)
                path = f"inputs/{index:02d}-prepared.npz"
                # Completed workflow arrays are immutable. Retain their
                # references here and compress them in the export worker.
                members.append((path, dict(bundle)))
                item["prepared_path"] = path
        elif form == "scalar":
            multiple = str(_value(surface.get("mode"), "")) == "constants"
            item["mode"] = "constants" if multiple else "constant"
            controls = surface.get("constant_controls", surface.get("constants", ()))
            if multiple:
                item["values"] = [_value(control) for control in controls]
            else:
                item["values"] = _value(surface.get("scalar"))
        elif form == "regions" and str(_value(surface.get("source"), "")) != "upload":
            item["mode"] = "regions"
            item["values"] = read_input(live).tolist() if enabled else _value(surface.get("editor"), "")
        else:
            item["mode"] = "upload"
            selections = surface.get("selection_values", {})
            files: list[dict[str, Any]] = []
            for file_index, inspection in enumerate(surface.get("inspections", ()), start=1):
                name = PurePath(inspection.name.replace("\\", "/")).name
                path = f"inputs/{index:02d}-{file_index:02d}-{name}"
                members.append((path, bytes(inspection.content)))
                selected = selections.get(inspection.name, {})
                files.append({
                    "path": path,
                    "name": inspection.name,
                    "column": selected.get("column"),
                    "array_key": selected.get("array_key"),
                    "source_format": inspection.source_format,
                })
            item["files"] = files
            track = surface.get("track")
            if track is not None:
                item["track"] = {key: _value(control) for key, control in track.items()}
        input_recipes[key] = item

    overrides = {
        key: {
            "enabled": bool(_value(live.override)) if live.override is not None else key in settings,
            "value": _value(live.control),
        }
        for key, live in setting_controls.items()
        if live.active and (live.override is not None or getattr(live, "automatic", False))
    }
    recipe = _literal({
        "method": method,
        "target_shape": target_shape,
        "settings": dict(settings),
        "setting_overrides": overrides,
        "region": dict(region),
        "combination_mode": combination_mode,
        "inputs": input_recipes,
    })
    input_bytes = sum(
        len(data) if isinstance(data, bytes) else sum(array.nbytes for array in data.values())
        for _path, data in members
    )
    if input_bytes > MAX_CODE_INPUT_BYTES:
        raise ValueError("Mapping code inputs exceed the 256 MB download limit.")
    return MappingCodeSnapshot(recipe, _literal(raw_export), _literal(plotting), tuple(members))


def _requirements(snapshot: MappingCodeSnapshot) -> tuple[str, ...]:
    names = ["numpy", "pandas", "scipy", "matplotlib"]
    if any(
        file.get("source_format") == "bigwig"
        for item in snapshot.recipe["inputs"].values()
        for file in item.get("files", ())
    ):
        # The app prefers pybigtools, with pyBigWig as a compatible fallback.
        for name in ("pybigtools", "pyBigWig"):
            try:
                metadata.version(name)
            except metadata.PackageNotFoundError:
                continue
            names.append(name)
            break
    exports = [snapshot.raw_export, *snapshot.raw_export.get("per_dataset", {}).values()]
    if any(item.get("compression") == "7z" for item in exports):
        names.append("py7zr")
    requirements = []
    for name in names:
        try:
            requirements.append(f"{name}=={metadata.version(name)}")
        except metadata.PackageNotFoundError:
            requirements.append(name)
    return tuple(requirements)


def build_mapping_python(snapshot: MappingCodeSnapshot) -> ExportArtifact:
    """Generate Python with explicit editable controls and optional plotting."""
    requirements = _requirements(snapshot)
    dependency_lines = "\n".join(f"#     {requirement!r}," for requirement in requirements)
    def format_value(value: Any) -> str:
        return pprint.pformat(value, width=95, sort_dicts=False)
    script = f'''# DNAscape Mapping: edit the controls below, then run this file.
# Extract the runnable bundle first; it contains the inputs and DNAscape source.
# Run: python mapping.py   (after installing requirements.txt)
# Or:  uv run mapping.py
# /// script
# requires-python = ">=3.10"
# dependencies = [
{dependency_lines}
# ]
# ///

from pathlib import Path
from math import inf

from dnascape.workflows.mapping_recipe import run_mapping_recipe
from dnascape.workflows.recipe_exports import save_mapping_outputs, plot_mapping_outputs

HERE = Path(__file__).resolve().parent
OUTPUT_DIRECTORY = HERE / "outputs"

# Mapping controls. Positions/resolution are in kb; times are in minutes.
# Files are relative to this script. Edit paths, selected columns/array keys,
# input forms, constants, regions, solver settings and enabled overrides here.
MAPPING = {format_value(snapshot.recipe)}

# Raw data controls: output, filename, format, compression and batch overrides.
# Use output="primary" to select the route's primary output. Formats:
# TSV, TXT, CSV, NPZ. Compression: none, gzip, zip, 7z, xz.
EXPORT_RAW = True
RAW_EXPORT = {format_value(snapshot.raw_export)}

# Plotting is optional. The HTML preserves the app's ECharts plot settings.
# It works without NiceGUI. Change output/view, axes, colours and layout here.
PLOT = False
PLOTTING = {format_value(snapshot.plotting)}

def main():
    results = run_mapping_recipe(MAPPING, base_path=HERE)
    if EXPORT_RAW:
        for path in save_mapping_outputs(results, RAW_EXPORT, directory=OUTPUT_DIRECTORY):
            print(f"Saved {{path}}")
    if PLOT:
        for path in plot_mapping_outputs(results, PLOTTING, directory=OUTPUT_DIRECTORY,
                                         echarts_path=HERE / "echarts.min.js"):
            print(f"Saved {{path}}")

if __name__ == "__main__":
    main()
'''
    return ExportArtifact(script.encode("utf-8"), "mapping.py", "text/x-python; charset=utf-8")


def build_mapping_code_bundle(snapshot: MappingCodeSnapshot) -> ExportArtifact:
    """Bundle the actual local package, original inputs and a short script."""
    import dnascape.workflows as workflows

    package_root = Path(workflows.__file__).resolve().parent.parent
    script = build_mapping_python(snapshot)
    requirements = _requirements(snapshot)
    readme = """DNAscape Mapping code

Extract this entire archive, then run:
    python -m pip install -r requirements.txt
    python mapping.py

Alternatively, with uv installed:
    uv run mapping.py

Edit MAPPING, RAW_EXPORT and PLOTTING in mapping.py. Raw export is enabled;
plotting is optional (set PLOT = True). Outputs go into outputs/ by default.
The inputs/ folder contains the original uploads or a prepared result bundle.
Changing input paths, columns, regions or processing rereads those inputs.
The dnascape/ folder contains the actual package source used to create this
download, including local changes. It does not require the NiceGUI app.
Dependencies are pinned to the exporting environment. Installing them needs
internet access initially. The package source and inputs are already included.
Optional plots embed the supplied ECharts runtime; if no compatible runtime
was available when exporting, their HTML loads ECharts from a pinned CDN URL.

Stochastic routes with seed=None produce a new realization on each run. Set a
fixed seed in the route settings to make repeated runs reproducible.
"""
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(script.filename, script.data)
        archive.writestr("requirements.txt", "\n".join(requirements) + "\n")
        archive.writestr("README.txt", readme)
        for path, data in snapshot.files:
            if isinstance(data, bytes):
                archive.writestr(path, data)
            else:
                prepared = io.BytesIO()
                np.savez_compressed(prepared, **data)
                archive.writestr(path, prepared.getvalue())
        for path in sorted(package_root.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts or "tests" in path.relative_to(package_root).parts:
                continue
            if path.suffix.casefold() in {".py", ".c", ".so", ".pyd", ".dll"}:
                archive.write(path, "dnascape/" + path.relative_to(package_root).as_posix())
        license_path = package_root.parent / "LICENSE"
        if license_path.is_file():
            archive.write(license_path, "LICENSE")
        # Reuse the exact runtime already installed with the app for offline
        # optional plots. Generation itself never downloads JavaScript.
        try:
            import nicegui
            from dnascape.workflows.plotting import prepare_echarts_runtime
            runtime = Path(nicegui.__file__).resolve().parent / "elements" / "echart" / "dist" / "index.js"
            if runtime.is_file():
                archive.writestr("echarts.min.js", prepare_echarts_runtime(runtime))
        except (ImportError, OSError, ValueError):
            pass
    return ExportArtifact(stream.getvalue(), "mapping_code.zip", "application/zip")
