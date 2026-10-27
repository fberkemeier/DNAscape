"""Shared dataset and region transforms for Analysis and its exported scripts."""
from __future__ import annotations

from dataclasses import replace
import gzip
from types import SimpleNamespace

import numpy as np

from .analysis import SmoothingSettings, smooth_profile
from .filtering import apply_data_filter, profile_coordinates, resolve_filter
from .formulas import evaluate_formula


def profile_key(identity):
    return f"{identity[0]}:{identity[1]}"


def scale_analysis_values(values, settings):
    """Use Data Manager's scaling conventions, independently per XT column."""
    mode = settings.get("scaling", "none")
    if mode not in {"none", "interval", "multiplier", "center", "zscore", "maxabs"}:
        raise ValueError("Choose an available value scaling method.")
    lower, upper = settings.get("minimum", 0.), settings.get("maximum", 1.)
    multiplier = settings.get("multiplier", 1.)
    if mode == "interval" and (lower is None or upper is None or not np.isfinite([lower, upper]).all() or lower >= upper):
        raise ValueError("Maximum must be greater than minimum for the target interval.")
    if mode == "multiplier" and (multiplier is None or not np.isfinite(multiplier)):
        raise ValueError("Enter a finite multiplier.")
    if mode == "none" and not settings.get("invert"):
        return np.asarray(values)
    from .conversion import _scale_series
    import pandas as pd

    # Scaling reads only these request attributes; no upload is materialized.
    request = SimpleNamespace(scaling_mode=mode, rescale_min=lower, rescale_max=upper,
                              scale=multiplier, offset=0., rescale_invert=bool(settings.get("invert")))
    data = np.asarray(values, dtype=float)
    columns = data.reshape(-1, 1) if data.ndim < 2 or data.shape[0] == 1 else data
    result = np.column_stack([_scale_series(pd.Series(column), request).to_numpy() for column in columns.T])
    if np.isinf(result).any():
        raise ValueError("Value scaling produced infinite values. Reduce the multiplier or interval.")
    return result.reshape(data.shape)


def transform_profiles(profiles, *, data_transforms=None, region_transforms=None, cancel=lambda: None):
    """Return full profiles plus masks; never compact genomic coordinates."""
    data = data_transforms or {}
    region = region_transforms or {}
    selected = [dict(item, id=tuple(item["id"]), transformed=item.get("transformed", False)) for item in profiles]
    target = data.get("target")
    target = tuple(target) if target is not None else None
    candidates = [item for item in selected if target is None or item["id"] == target]
    if data.get("enabled"):
        if not candidates:
            raise ValueError("Choose an available dataset to transform.")
        if data.get("combine") and len(candidates) > 1:
            first = candidates[0]
            axes = [profile_coordinates(item["source"], np.asarray(item["values"]).shape[0] if item["form"] == "xt" else np.asarray(item["values"]).size) for item in candidates]
            for item, axis in zip(candidates[1:], axes[1:]):
                if item["form"] != first["form"] or np.shape(item["values"]) != np.shape(first["values"]):
                    raise ValueError("Combined datasets must have matching shapes and dependence.")
                if any((a is None) != (b is None) or (a is not None and not np.array_equal(a, b))
                       for a, b in zip(axes[0], axis)):
                    raise ValueError("Combined datasets must have matching coordinates.")
            formula = data.get("combined_formula") or " + ".join(f"source{i}" for i in range(1, len(candidates) + 1))
            values = evaluate_formula(formula, {f"source{i}": item["values"] for i, item in enumerate(candidates, 1)})
            selected = [dict(first, id=(first["id"][0], "__combined__"), name="Combined", values=values, transformed=True)]
        for item in selected:
            cancel()
            if target is not None and item["id"] != target:
                continue
            combined = item["id"][1] == "__combined__"
            key = "combined" if combined else profile_key(item["id"])
            values = item["values"] if combined else evaluate_formula(data.get("formulas", {}).get(key, "source"), {"source": item["values"]})
            settings = data.get("values", {}).get(key, {})
            smoothing = settings.get("smoothing", SmoothingSettings())
            if isinstance(smoothing, dict):
                smoothing = SmoothingSettings(**smoothing)
            values = smooth_profile(values, smoothing, dependence=item["form"])
            item.update(values=scale_analysis_values(values, settings), transformed=True)
    settings = region.get("settings")
    settings = resolve_filter(settings) if settings is not None and settings.enabled else None
    target = region.get("target")
    target = tuple(target) if target is not None else None
    if settings is not None and target is not None and not any(item["id"] == target for item in selected):
        raise ValueError("Choose an available dataset to transform.")
    result = []
    for item in selected:
        cancel()
        if settings is None or not settings.enabled or (target is not None and item["id"] != target):
            result.append(item)
            continue
        values = np.asarray(item["values"])
        length = values.shape[0] if item["form"] == "xt" else values.size
        coordinates, intervals = profile_coordinates(item["source"], length)
        mask = np.zeros(values.shape, dtype=bool)
        columns = mask.reshape(-1, 1) if mask.ndim < 2 or mask.shape[0] == 1 else mask

        def capture(selected, *_args, **kwargs):
            column = _args[2] if len(_args) > 2 and _args[2] is not None else 0
            columns[:, column] = selected

        warnings = []
        values = apply_data_filter(values, replace(settings, _position_sink=capture, _warning_sink=warnings.append), positions=coordinates, intervals=intervals, cancel=cancel)
        mask_only = bool(region.get("analysis_on_mask_only"))
        if region.get("keep_originals"):
            result.append(dict(item, mask=mask, analysis_on_mask_only=mask_only))
        result.append(dict(item, values=values, mask=mask, analysis_on_mask_only=mask_only, warnings=tuple(warnings), transformed=True,
                           id=(-item["id"][0], item["id"][1]) if region.get("keep_originals") else item["id"],
                           name=item["name"] + " (transformed)" if region.get("keep_originals") else item["name"]))
    cancel()
    return tuple(result)


def analysis_values(profile):
    """Keep unselected samples out of statistics without shifting pair indices."""
    values = np.asarray(profile["values"])
    return np.where(profile["mask"], values, np.nan) if profile.get("analysis_on_mask_only") and profile.get("mask") is not None else values


def analysis_mask_artifact(profile, *, filename, file_format="TSV", layout="positions", compress=False):
    from .conversion import FilterPositionGroup, filter_positions_artifact

    mask = profile.get("mask")
    if mask is None:
        raise ValueError("Apply a region transform before exporting its mask.")
    mask = np.asarray(mask, dtype=bool)
    if profile["form"] == "xt":
        mask = mask.any(axis=1)
    mask = mask.reshape(-1)
    coordinates, intervals = profile_coordinates(profile["source"], mask.size)
    axis = np.arange(mask.size, dtype=float) if coordinates is None else np.asarray(coordinates)
    if intervals is None:
        step = axis[-1] - axis[-2] if axis.size > 1 else (0.001 if coordinates is not None else 1.)
        intervals = np.column_stack((axis, np.r_[axis[1:], axis[-1:] + step]))
    group = FilterPositionGroup(
        gzip.compress(np.asarray(axis[mask], dtype="<f8").tobytes(), mtime=0),
        "position_kb" if coordinates is not None and profile["form"] != "t" else "bin_index",
        interval_data=gzip.compress(np.asarray(intervals[mask], dtype="<f8").tobytes(), mtime=0),
    )
    return filter_positions_artifact(SimpleNamespace(filter_positions=(group,)), output_name=filename,
                                     target_format=file_format, layout=layout, gzip_output=compress)
