"""Editable Python recipes shared by developer downloads and headless runs.

Only explicit DNAscape data classes can be restored; recipes never import the
application or evaluate Python representations of uploaded data.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import MISSING, fields, is_dataclass
from importlib import import_module
from pathlib import Path

import numpy as np

_TYPES = {
    "DataConversionRequest": "conversion", "DataFilterSettings": "filtering",
    "FilterProfile": "filtering", "ProfileModulationSettings": "modulation",
    "ModulationProfile": "modulation", "MappingFilterSettings": "mapping_filter",
    "SmoothingSettings": "analysis", "ConstantSimulationRequest": "simulation",
}


def encode_settings(value, *, upload):
    """Keep active configuration readable and preserve arrays without pickle."""
    from .uploads import UploadInspection

    if isinstance(value, UploadInspection):
        result = {"__upload__": upload(value), "form": value.expected_form}
        if value.source_format in {"bigwig", "bedgraph"}:
            result["format"] = value.source_format
        return result
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise ValueError("Code export requires numeric or string arrays, not object arrays.")
        if value.size > 64 or (np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all()):
            import base64
            import io
            import zlib
            buffer = io.BytesIO()
            np.save(buffer, value, allow_pickle=False)
            return {"__npy_z85__": base64.b85encode(zlib.compress(buffer.getvalue())).decode("ascii")}
        return {"__array__": value.tolist(), "dtype": value.dtype.str}
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        name = type(value).__name__
        if name not in _TYPES:
            raise TypeError(f"Unsupported recipe settings: {name}.")
        result = {"__type__": name}
        for item in fields(value):
            if item.name.startswith("_"):
                continue
            current = getattr(value, item.name)
            default = item.default
            if default is MISSING and item.default_factory is not MISSING:
                default = item.default_factory()
            if default is not MISSING and not is_dataclass(current) and not isinstance(current, np.ndarray):
                try:
                    if bool(current == default):
                        continue
                except (TypeError, ValueError):
                    pass
            result[item.name] = encode_settings(current, upload=upload)
        return result
    if isinstance(value, Mapping):
        return {key: encode_settings(item, upload=upload) for key, item in value.items()}
    if isinstance(value, frozenset):
        return {"__frozenset__": sorted(value)}
    if isinstance(value, (tuple, list)):
        return [encode_settings(item, upload=upload) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported recipe value: {type(value).__name__}.")


def restore_settings(value, *, input_directory="inputs", _uploads=None):
    """Load original files and restore the recipe's declared DNAscape settings."""
    if _uploads is None:
        _uploads = {}
    if isinstance(value, list):
        return [restore_settings(item, input_directory=input_directory, _uploads=_uploads) for item in value]
    if not isinstance(value, dict):
        return value
    if "__upload__" in value:
        from .uploads import inspect_upload
        path = Path(input_directory) / value["__upload__"]
        if not path.is_file():
            raise FileNotFoundError(f"Place the source file at {path}, or edit INPUT_DIRECTORY/the recipe path.")
        key = (str(path.resolve()), value.get("form"), value.get("format", "auto"))
        if key not in _uploads:
            _uploads[key] = inspect_upload(path.name, path.read_bytes(), expected_form=value.get("form"),
                                          source_format=value.get("format", "auto"))
        return _uploads[key]
    if "__array__" in value:
        return np.asarray(value["__array__"], dtype=value["dtype"])
    if "__npy_z85__" in value:
        import base64
        import io
        import zlib
        return np.load(io.BytesIO(zlib.decompress(base64.b85decode(value["__npy_z85__"]))), allow_pickle=False)
    if "__frozenset__" in value:
        return frozenset(value["__frozenset__"])
    values = {key: restore_settings(item, input_directory=input_directory, _uploads=_uploads)
              for key, item in value.items() if key != "__type__"}
    if "__type__" in value:
        name = value["__type__"]
        if name not in _TYPES:
            raise ValueError(f"Unknown DNAscape recipe settings: {name}.")
        cls = getattr(import_module(f"dnascape.workflows.{_TYPES[name]}"), name)
        return cls(**values)
    return values


def write_artifact(artifact, directory, paths):
    """Write one safe, unique output and return its path."""
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / Path(artifact.filename).name
    if path in paths:
        raise ValueError(f"Several outputs use the same filename: {path.name}.")
    path.write_bytes(artifact.data)
    paths.append(path)
    return path


def save_recipe_statistics(datasets, settings, directory, paths):
    """Export full-precision histogram summaries without generating a plot."""
    if not settings:
        return
    from .exports import build_histogram_statistics_export
    rows = []
    for index, (name, values) in enumerate(datasets):
        finite = np.asarray(values, dtype=float).reshape(-1)
        finite = finite[np.isfinite(finite)]
        if settings.get("log_x"):
            finite = finite[finite > 0]
        if settings.get("limits") is not None and not settings.get("density"):
            lower, upper = settings["limits"]
            finite = finite[(finite >= lower) & (finite <= upper)]
        labels = settings.get("labels", [])
        rows.append(dict(dataset=labels[index] if index < len(labels) else str(name),
                         mean=float(np.mean(finite)) if finite.size else None,
                         median=float(np.median(finite)) if finite.size else None,
                         standard_deviation=float(np.std(finite)) if finite.size else None))
    options = {key: settings[key] for key in ("filename", "file_format", "compression") if key in settings}
    write_artifact(build_histogram_statistics_export(statistics=rows, **options), directory, paths)


def run_data_manager_recipe(recipe, *, input_directory="inputs", output_directory="outputs"):
    """Configure → data transforms → region transforms → export, using the live page workflows."""
    from .export_settings import settings_scope, snapshot_export_settings
    settings = snapshot_export_settings()
    settings.preview_points = int(recipe.get("preview_points", settings.preview_points))
    with settings_scope(settings):
        return _run_data_manager_recipe(recipe, input_directory=input_directory, output_directory=output_directory)


def _run_data_manager_recipe(recipe, *, input_directory="inputs", output_directory="outputs"):
    from .conversion import (
        convert_combined_data, convert_data, filter_positions_artifact,
        transcode_conversion_artifact,
    )

    requests = [restore_settings(item, input_directory=input_directory) for item in recipe["sources"]]
    combined = recipe.get("combined_formula")
    results = ([convert_combined_data(tuple(requests), combined, output_name=recipe.get("combined_name", "rdata_combined"))]
               if combined is not None else [convert_data(request) for request in requests])
    paths = []
    for index, result in enumerate(results):
        exports = recipe.get("exports", [])
        options = dict(exports[index]) if index < len(exports) else {}
        name = options.get("output_name") or Path(result.artifact.filename).stem
        artifact = transcode_conversion_artifact(
            result, output_name=name, target_format=options.get("target_format", result.target_format),
            gzip_output=bool(options.get("gzip_output", False)),
        )
        write_artifact(artifact, output_directory, paths)
        filter_options = options.get("filter")
        if filter_options and result.filter_positions:
            write_artifact(filter_positions_artifact(result, **filter_options), output_directory, paths)
    if recipe.get("statistics"):
        settings = recipe["statistics"]
        index = min(int(settings.get("source_index", 0)), len(results) - 1)
        result = results[index]
        if result.preview is not None:
            save_recipe_statistics([(result.artifact.filename, result.preview.values)], settings, output_directory, paths)
    return tuple(results), tuple(paths)
