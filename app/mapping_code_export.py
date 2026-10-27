"""Capture Mapping controls and build portable, editable Python downloads."""

from __future__ import annotations

import math
import pprint
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from app.simulation_exports import ExportArtifact

MAX_CODE_INPUT_BYTES = 256 * 1024 * 1024


@dataclass(frozen=True)
class MappingCodeSnapshot:
    recipe: dict[str, Any]
    raw_export: dict[str, Any]
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
        return {None if key is None else str(key): _literal(item) for key, item in value.items()}
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
    fit_filter=None,
) -> MappingCodeSnapshot:
    """Read widgets synchronously, before background export yields to the UI.

    Original uploads are retained so changing regions, columns or track
    processing in the downloaded script actually rereads the original data.
    """
    members: list[tuple[str, bytes | Mapping[str, np.ndarray]]] = []
    from app.code_export import CodeInputs
    retained_inputs = CodeInputs()
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
                path = f"prepared-{index:02d}.npz"
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
            if enabled and not surface.get("inspections"):
                raise ValueError(f"Upload {key.replace('_', ' ')} data before exporting code.")
            for inspection in surface.get("inspections", ()):
                path = retained_inputs.upload(inspection)
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

    def setting_value(key, live):
        if key in settings:
            return settings[key]
        value = _value(live.control)
        if key == "err_threshold" and value not in (None, ""):
            return -math.log10(float(value))
        return value

    overrides = {
        key: {
            "enabled": bool(_value(live.override)) if live.override is not None else key in settings,
            "value": setting_value(key, live),
        }
        for key, live in setting_controls.items()
        if live.active and (live.override is not None or getattr(live, "automatic", False))
    }
    from dnascape.workflows.code_recipes import encode_settings
    filter_recipe = (encode_settings(fit_filter, upload=retained_inputs.upload)
                     if fit_filter is not None and fit_filter.regions.enabled else None)
    known_paths = {path for path, _data in members}
    members.extend((path, data) for path, data in retained_inputs.files.items() if path not in known_paths)
    recipe = _literal({
        "method": method,
        "target_shape": target_shape,
        "settings": dict(settings),
        "setting_overrides": overrides,
        "region": dict(region),
        "combination_mode": combination_mode,
        "inputs": input_recipes,
        **({'filter': filter_recipe} if filter_recipe is not None else {}),
    })
    input_bytes = sum(
        0 if isinstance(data, bytes) else sum(array.nbytes for array in data.values())
        for _path, data in members
    )
    if input_bytes > MAX_CODE_INPUT_BYTES:
        raise ValueError("Mapping code inputs exceed the 256 MB download limit.")
    return MappingCodeSnapshot(recipe, _literal(raw_export), tuple(members))


def build_mapping_python(snapshot: MappingCodeSnapshot) -> ExportArtifact:
    """One concise script with editable source and destination directories."""
    from copy import deepcopy
    from dnascape.workflows.code_recipes import encode_settings
    recipe = deepcopy(snapshot.recipe)
    prepared = {path: data for path, data in snapshot.files if not isinstance(data, bytes)}
    for source in recipe["inputs"].values():
        if source.get("mode") == "prepared" and source.get("prepared_path") in prepared:
            source["prepared_values"] = encode_settings(prepared[source["prepared_path"]], upload=lambda _source: None)
    inputs = "\n".join(f"#   {path!r}" for path, data in snapshot.files if isinstance(data, bytes))
    source = f'''# DNAscape Mapping: run with this version of DNAscape installed.
# Run: python mapping.py
# Place the original files under INPUT_DIRECTORY:
{inputs or "#   No external files: constants/prepared data are included below."}

from pathlib import Path
from math import inf, nan

from dnascape.workflows.mapping_recipe import run_mapping_recipe
from dnascape.workflows.recipe_exports import save_mapping_outputs

HERE = Path(__file__).resolve().parent
INPUT_DIRECTORY = HERE / "inputs"
OUTPUT_DIRECTORY = HERE / "outputs"

# Positions and resolution are in kb; time is in minutes. Edit input paths,
# columns/arrays, constants, region, fitting settings and region transforms here.
MAPPING = {pprint.pformat(recipe, width=95, sort_dicts=False)}

# Select the output, filename, TSV/TXT/CSV/NPZ format and compression.
RAW_EXPORT = {pprint.pformat(snapshot.raw_export, width=95, sort_dicts=False)}


def main():
    results = run_mapping_recipe(MAPPING, base_path=INPUT_DIRECTORY)
    for path in save_mapping_outputs(results, RAW_EXPORT, directory=OUTPUT_DIRECTORY):
        print(f"Saved {{path}}")
    return results


if __name__ == "__main__":
    main()
'''
    return ExportArtifact(source.encode("utf-8"), "mapping.py", "text/x-python; charset=utf-8")
