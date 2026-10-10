"""Retain the profiles used by a run for plotting and optional export."""

from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType
from typing import Any

import numpy as np

INPUT_PREFIX = "input:"


def input_profile_label(key: str) -> str:
    name = key.removeprefix(INPUT_PREFIX).removesuffix("_xt").replace("_", " ")
    return f"{name.capitalize()} (input)"


def _readonly(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=float).view()
    array.setflags(write=False)
    return array


def _time_axis(count: int, resolution: float, explicit: Any = None) -> np.ndarray:
    if explicit is not None:
        axis = np.asarray(explicit, dtype=float).reshape(-1)
        if axis.size == count:
            return _readonly(axis)
    return _readonly(np.arange(count, dtype=float) * resolution)


def add_mapping_input_previews(result: Any, prepared: Any) -> Any:
    """Keep aligned, nonconstant inputs without copying their numeric data."""

    outputs = dict(result.outputs)
    forms = dict(result.output_forms)
    axes = dict(result.input_axes)
    profiles = []
    for source_key, form in prepared.input_forms.items():
        if source_key in prepared.source:
            source = prepared.source[source_key]
        elif source_key in prepared.assumptions:
            source = prepared.assumptions[source_key]
        else:
            continue
        if form == "bundle":
            profiles.extend(
                (member, "xt", values, source.get("time_grid"))
                for member, values in source.items()
                if member.endswith("_xt") and isinstance(values, np.ndarray)
            )
        elif form in {"x", "t", "xt"}:
            profiles.append((source_key, form, source, prepared.assumptions.get("time_grid")))
    for source_key, form, source, explicit_time in profiles:
        key = INPUT_PREFIX + source_key
        if key in outputs:
            continue
        values = _readonly(source)
        if form in {"x", "t"}:
            values = values.reshape(-1)
        position_count = (
            values.shape[0] if form in {"x", "xt"}
            else max(1, prepared.spatial_size, result.x_axis.size)
        )
        time_count = values.shape[1] if form == "xt" else values.size if form == "t" else 0
        position = _readonly(
            prepared.start_kb + np.arange(position_count, dtype=float) * prepared.resolution_x
        )
        time = _time_axis(time_count, prepared.resolution_t, explicit_time)
        outputs[key] = values
        forms[key] = form
        axes[key] = (position, time)
    return replace(
        result, outputs=MappingProxyType(outputs), output_forms=MappingProxyType(forms),
        input_axes=MappingProxyType(axes),
    )


def add_simulation_input_previews(result: Any, request: Any) -> Any:
    """Retain uploaded kinetic fields in their original position/time order."""

    outputs = dict(result.outputs)
    forms = dict(result.input_forms)
    axes = dict(result.input_axes)
    for source_key in ("firing_rate", "fork_speed"):
        key = INPUT_PREFIX + source_key
        values = _readonly(getattr(request, source_key))
        if values.ndim == 0 or values.size == 1 or key in outputs:
            continue
        form = "x" if values.ndim == 1 or values.shape[1] == 1 else "t" if values.shape[0] == 1 else "xt"
        if form in {"x", "t"}:
            values = values.reshape(-1)
        position_count = values.shape[0] if form in {"x", "xt"} else result.position_kb.size
        time_count = values.shape[1] if form == "xt" else values.size if form == "t" else 0
        position = _readonly(
            request.region_start_kb + np.arange(position_count, dtype=float) * request.resolution_x
        )
        time = _time_axis(time_count, request.resolution_t)
        outputs[key] = values
        forms[key] = form
        axes[key] = (position, time)
    return replace(
        result, outputs=MappingProxyType(outputs), input_forms=MappingProxyType(forms),
        input_axes=MappingProxyType(axes),
    )


def input_preview_result(result: Any, output_key: str) -> Any:
    """Use an input's own axes rather than the mapped or simulated time grid."""

    axes = getattr(result, "input_axes", {}).get(output_key)
    if axes is None:
        return result
    position, time = axes
    if hasattr(result, "x_axis"):
        form = result.output_forms[output_key]
        return replace(
            result, x_axis=position, time_axis=time, output_form=form,
            output_dependence={"x": "space", "t": "time", "xt": "spacetime"}[form],
        )
    return replace(result, position_kb=position, time_grid=time)
