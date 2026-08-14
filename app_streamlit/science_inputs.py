"""Pure preparation of canonical dataset views for scientific workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from app_streamlit.workflow_models import half_open_positions


@dataclass(frozen=True)
class PreparedField:
    """A validated kinetic/observed field and its optional spatial axis."""

    values: np.ndarray
    field_kind: str
    position_kb: np.ndarray | None = None


def _fill_vector(values: np.ndarray, policy: str, name: str) -> np.ndarray:
    finite = np.isfinite(values)
    if finite.all():
        return values
    policy = str(policy).strip().lower()
    missing = int(values.size - np.count_nonzero(finite))
    if policy == "error":
        raise ValueError(
            f"{name} contains {missing:,} non-finite value(s). Choose interpolation "
            "or zero fill before running the workflow."
        )
    if policy == "zero":
        return np.nan_to_num(values, copy=True, nan=0.0, posinf=0.0, neginf=0.0)
    if policy != "interpolate":
        raise ValueError("Missing-value policy must be error, interpolate, or zero.")
    if not finite.any():
        raise ValueError(f"{name} contains no finite values to interpolate.")
    index = np.arange(values.size, dtype=float)
    result = values.copy()
    result[~finite] = np.interp(index[~finite], index[finite], values[finite])
    return result


def _fill_field(values: np.ndarray, policy: str, name: str) -> np.ndarray:
    if values.ndim == 1:
        return _fill_vector(values, policy, name)
    if np.isfinite(values).all():
        return values
    columns = [
        _fill_vector(values[:, index], policy, f"{name}, time column {index + 1}")
        for index in range(values.shape[1])
    ]
    result = np.column_stack(columns)
    result.setflags(write=False)
    return result


def _readonly(values: np.ndarray) -> np.ndarray:
    """Return a non-owning read-only view when possible."""

    result = np.asarray(values).view()
    result.setflags(write=False)
    return result


def _validated_spatial_axis(
    position_kb: Any | None,
    n_bins: int,
    resolution_kb: float,
) -> np.ndarray:
    if position_kb is None:
        return half_open_positions(n_bins, resolution_kb)
    axis = np.asarray(position_kb, dtype=float).reshape(-1)
    if axis.size != n_bins:
        raise ValueError("The spatial coordinate axis does not match the profile length.")
    if not np.all(np.isfinite(axis)):
        raise ValueError("Spatial coordinates must be finite.")
    if axis.size > 1:
        steps = np.diff(axis)
        tolerance = max(1e-9, abs(float(resolution_kb)) * 1e-5)
        if np.any(steps <= 0) or not np.allclose(
            steps, float(resolution_kb), rtol=1e-5, atol=tolerance
        ):
            raise ValueError(
                "The selected profile must use a strictly increasing uniform spatial grid "
                "matching the configured resolution. Resample it on Data Management first."
            )
    return _readonly(axis)


def prepare_profile_field(
    values: Any,
    data_form: str,
    *,
    resolution_kb: float,
    region_start_kb: float,
    region_end_kb: float,
    position_kb: Any | None = None,
    missing: str = "error",
    name: str = "Profile",
    expected_time_points: int | None = None,
) -> PreparedField:
    """Validate and subset a space, time, or space-time field.

    Genomic intervals are always half open. Missing spatial bins are filled in
    place on the regular grid and are never removed, because removing them
    would silently alter fork-travel distances.
    """

    resolution_kb = float(resolution_kb)
    region_start_kb = float(region_start_kb)
    region_end_kb = float(region_end_kb)
    if not np.isfinite(resolution_kb) or resolution_kb <= 0:
        raise ValueError("Spatial resolution must be finite and greater than zero.")
    if not np.isfinite(region_start_kb) or region_start_kb < 0:
        raise ValueError("Region start must be finite and non-negative.")
    if not np.isfinite(region_end_kb) or region_end_kb <= region_start_kb:
        raise ValueError("Region end must be finite and greater than region start.")
    if expected_time_points is not None and int(expected_time_points) <= 0:
        raise ValueError("Expected time-point count must be positive.")

    array = np.asarray(values, dtype=float)
    if array.size == 0:
        raise ValueError(f"{name} cannot be empty.")
    form = str(data_form).strip().lower().replace("-", "")
    if form == "table":
        raise ValueError(
            f"{name} is a table. Prepare it as a space, time, or space-time dataset first."
        )
    if form in {"space", "auto"}:
        array = array.reshape(-1)
        axis = _validated_spatial_axis(position_kb, array.size, resolution_kb)
        mask = (axis >= float(region_start_kb)) & (axis < float(region_end_kb))
        if not np.any(mask):
            raise ValueError(f"The selected half-open interval contains no {name} bins.")
        selected_axis = axis[mask]
        tolerance = max(1e-9, abs(float(resolution_kb)) * 1e-5)
        if not (
            np.isclose(selected_axis[0], float(region_start_kb), atol=tolerance, rtol=1e-5)
            and np.isclose(
                selected_axis[-1] + float(resolution_kb),
                float(region_end_kb),
                atol=tolerance,
                rtol=1e-5,
            )
        ):
            raise ValueError(
                f"The {name} interval must align to its {resolution_kb:g} kb source grid. "
                f"The selected bins span [{selected_axis[0]:g}, "
                f"{selected_axis[-1] + float(resolution_kb):g}) kb."
            )
        selected = _fill_field(array[mask], missing, name)
        return PreparedField(_readonly(selected), "space", _readonly(selected_axis))
    if form == "time":
        array = array.reshape(-1)
        if expected_time_points is not None and array.size != int(expected_time_points):
            raise ValueError(
                f"{name} has {array.size:,} time values; the configured time grid has "
                f"{int(expected_time_points):,}."
            )
        # DNAscape distinguishes time-only fields from spatial vectors by the
        # explicit singleton spatial axis: (1, T), never bare (T,).
        selected = _fill_field(array, missing, name).reshape(1, -1)
        return PreparedField(_readonly(selected), "time", None)
    if form in {"spacetime", "xt"}:
        if array.ndim != 2:
            raise ValueError(f"{name} must be a two-dimensional space-time matrix.")
        axis = _validated_spatial_axis(position_kb, array.shape[0], resolution_kb)
        mask = (axis >= float(region_start_kb)) & (axis < float(region_end_kb))
        if not np.any(mask):
            raise ValueError(f"The selected half-open interval contains no {name} bins.")
        selected_axis = axis[mask]
        tolerance = max(1e-9, abs(float(resolution_kb)) * 1e-5)
        if not (
            np.isclose(selected_axis[0], float(region_start_kb), atol=tolerance, rtol=1e-5)
            and np.isclose(
                selected_axis[-1] + float(resolution_kb),
                float(region_end_kb),
                atol=tolerance,
                rtol=1e-5,
            )
        ):
            raise ValueError(
                f"The {name} interval must align to its {resolution_kb:g} kb source grid. "
                f"The selected bins span [{selected_axis[0]:g}, "
                f"{selected_axis[-1] + float(resolution_kb):g}) kb."
            )
        if expected_time_points is not None and array.shape[1] != int(expected_time_points):
            raise ValueError(
                f"{name} has {array.shape[1]:,} time columns; the configured time grid has "
                f"{int(expected_time_points):,}."
            )
        selected = _fill_field(array[mask, :], missing, name)
        return PreparedField(
            _readonly(selected), "spacetime", _readonly(selected_axis)
        )
    raise ValueError(f"Unsupported data form for {name}: {data_form!r}.")


def validate_aligned_positions(*fields: PreparedField) -> np.ndarray | None:
    """Return the shared spatial axis or reject misaligned profile inputs."""

    axes = [field.position_kb for field in fields if field.position_kb is not None]
    if not axes:
        return None
    reference = axes[0]
    for axis in axes[1:]:
        if axis.shape != reference.shape or not np.allclose(axis, reference):
            raise ValueError(
                "Spatial input profiles do not share the same bins. Align or resample them "
                "on Data Management before running this workflow."
            )
    return reference
