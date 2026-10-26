"""Combine assay profiles on a strict grid or their common extent from the start."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType

import numpy as np

from dnascape.workflows.uploads import UploadInspection, materialize_upload
from dnascape.workflows.formulas import evaluate_formula


MODULATION_OPERATIONS = {
    "formula": "Formula", "minimum": "Minimum", "maximum": "Maximum",
    "convolve": "Convolution",
}
MODULATION_COMBINATIONS = {"sum": "Weighted sum", "multiply": "Multiply", "divide": "Divide"}
MODULATION_FUNCTIONS = {"identity": "None", "ln": "Natural log", "log2": "Log₂", "log10": "Log₁₀", "exp": "Exponential"}
MODULATION_SCALING = {
    "none": "Original values", "mean": "Mean = 1", "maxabs": "Maximum magnitude = 1",
    "interval": "0–1 interval",
}
MODULATION_BOUNDARIES = {"reflect": "Reflect", "wrap": "Periodic", "edge": "Nearest value", "zero": "Zero padding"}


@dataclass(frozen=True, slots=True)
class ModulationProfile:
    values: np.ndarray = field(repr=False, compare=False)
    coordinates: np.ndarray | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class ProfileModulationSettings:
    operation: str = "multiply"
    scaling: str = "none"
    weight: float = .5
    zero_policy: str = "error"
    missing_policy: str = "error"
    normalize_kernel: bool = True
    boundary: str = "reflect"
    upload: UploadInspection | None = field(default=None, repr=False, compare=False)
    column: str | None = None
    array_key: str | None = None
    values: np.ndarray | None = field(default=None, repr=False, compare=False)
    profiles: Mapping[str | None, ModulationProfile] = field(default_factory=dict, repr=False, compare=False)
    source_sha256: str = ""
    combination: str = "multiply"
    signal_multiplier: float = 1.
    profile_multiplier: float = 1.
    signal_exponent: float = 1.
    profile_exponent: float = 1.
    signal_function: str = "identity"
    profile_function: str = "identity"
    expression: str = "source * profile"
    align_from_start: bool = False


def _require_choice(value, choices, name):
    if value not in choices:
        raise ValueError(f"Choose a valid modulation {name}.")


def _is_chromosome(name: str) -> bool:
    return re.fullmatch(r"(?:chr(?:omosome)?(?:[0-9IVXLCDMXYZW]+[A-Za-z0-9_.-]*|Un[A-Za-z0-9_.-]*|EBV)|[0-9]+|X|Y|M|MT)", name, re.IGNORECASE) is not None


def resolve_modulation(settings: ProfileModulationSettings) -> ProfileModulationSettings:
    # Keep older Python requests valid while presenting one formula editor in the UI.
    if not isinstance(settings.align_from_start, (bool, np.bool_)):
        raise ValueError("Modulation alignment must be on or off.")
    _require_choice(settings.operation, {*MODULATION_OPERATIONS, "expression", "multiply", "add", "subtract", "divide", "blend"}, "operation")
    _require_choice(settings.missing_policy, {"error", "zero"} if settings.operation == "convolve" else {"error", "zero", "nan"}, "missing-value policy")
    changes = {}
    if settings.operation == "convolve":
        _require_choice(settings.boundary, MODULATION_BOUNDARIES, "boundary")
        if not isinstance(settings.normalize_kernel, (bool, np.bool_)):
            raise ValueError("Kernel normalization must be on or off.")
    else:
        _require_choice(settings.scaling, MODULATION_SCALING, "scaling")
    if settings.operation == "formula":
        _require_choice(settings.combination, MODULATION_COMBINATIONS, "operator")
        for operand in ("signal", "profile"):
            _require_choice(getattr(settings, f"{operand}_function"), MODULATION_FUNCTIONS, "function")
            for parameter in ("multiplier", "exponent"):
                key = f"{operand}_{parameter}"
                try:
                    number = float(getattr(settings, key))
                except (TypeError, ValueError):
                    raise ValueError(f"{operand.capitalize()} {parameter} must be finite.") from None
                if not np.isfinite(number):
                    raise ValueError(f"{operand.capitalize()} {parameter} must be finite.")
                changes[key] = number
    if settings.operation == "divide" or (settings.operation == "formula" and settings.combination == "divide"):
        _require_choice(settings.zero_policy, {"error", "keep", "nan"}, "division-by-zero policy")
    if settings.operation == "blend":
        try:
            weight = float(settings.weight)
        except (ValueError, TypeError):
            raise ValueError("Profile weight must be between zero and one.") from None
        if not np.isfinite(weight) or not 0 <= weight <= 1:
            raise ValueError("Profile weight must be between zero and one.")
        changes["weight"] = weight
    if settings.upload is not None:
        profiles = read_modulation_profiles(settings.upload, column=settings.column, array_key=settings.array_key)
        changes.update(upload=None, source_sha256=hashlib.sha256(settings.upload.content).hexdigest())
    elif settings.values is not None:
        profiles = {None: ModulationProfile(np.asarray(settings.values, dtype=float))}
        changes["values"] = None
    else:
        profiles = dict(settings.profiles)
    if not profiles:
        raise ValueError("Upload a modulation profile before using profile modulation.")
    for profile in profiles.values():
        values = np.asarray(profile.values, dtype=float)
        if values.ndim > 2 or not values.size:
            raise ValueError("The modulation profile must be a non-empty vector or position-by-time matrix.")
        if np.isinf(values).any():
            raise ValueError("The modulation profile contains infinite values.")
    changes["profiles"] = MappingProxyType(profiles)
    return replace(settings, **changes)


def read_modulation_profiles(inspection: UploadInspection, *, column=None, array_key=None):
    # Imported here because these coordinate readers are shared with Filter.
    from dnascape.workflows.filtering import _upload_table, profile_coordinates

    if inspection.source_format not in {"csv", "tsv", "txt", "npy", "npz"}:
        raise ValueError("Prepare the modulation data as a numeric profile before uploading it here.")
    if column not in (None, "__matrix__") and column not in inspection.numeric_columns:
        raise ValueError("Choose a numeric data column for the modulation profile.")
    if array_key is None and column is None:
        if inspection.source_format == "npz" and inspection.array_keys and all(_is_chromosome(key) for key in inspection.array_keys):
            return {key: ModulationProfile(np.asarray(inspection._array_data[key], dtype=float)) for key in inspection.array_keys}
        if inspection.numeric_columns and all(_is_chromosome(key) for key in inspection.numeric_columns):
            profiles = {}
            for key in inspection.numeric_columns:
                values = materialize_upload(replace(inspection, expected_form="x"), column=key, missing_policy="ignore")
                if len(inspection.numeric_columns) > 1:
                    present = np.flatnonzero(~np.isnan(values))
                    if present.size:
                        values = values[:present[-1] + 1]
                profiles[key] = ModulationProfile(values)
            return profiles

    matrix = column == "__matrix__"
    selected_column = None if matrix else column
    if selected_column is None and not matrix and len(inspection.numeric_columns) == 1:
        selected_column = inspection.numeric_columns[0]
    if inspection.source_format in {"csv", "tsv", "txt"}:
        chromosome_column = next((key for key in inspection.columns if str(key).lower() in {"chrom", "chromosome", "chr"}), None)
        if chromosome_column is not None:
            signal_columns = list(inspection.numeric_columns) if selected_column is None else [selected_column]
            table = _upload_table(inspection)
            if table[chromosome_column].isna().any():
                raise ValueError("Every modulation profile row must have a chromosome label.")
            profiles = {}
            all_coordinates, _ = profile_coordinates(inspection, len(table))
            for chromosome, rows in table.groupby(chromosome_column, sort=False):
                values = rows[signal_columns].to_numpy(dtype=float)
                if values.shape[1] == 1 and not matrix:
                    values = values[:, 0]
                coords = all_coordinates[rows.index] if all_coordinates is not None else None
                profiles[str(chromosome)] = ModulationProfile(values, coords)
            return profiles
        form = "x" if selected_column is not None else "xt"
    else:
        selected = inspection._array_data.get(array_key or inspection.default_array_key or "values")
        shape = np.asarray(selected).shape if selected is not None else inspection.shape or ()
        form = "x" if selected_column is not None or len(shape) < 2 or 1 in shape else "xt"
    values = materialize_upload(replace(inspection, expected_form=form), column=selected_column,
                                array_key=array_key, missing_policy="ignore")
    length = values.size if values.ndim < 2 or 1 in values.shape else values.shape[0]
    coordinates, _ = profile_coordinates(inspection, length)
    label = array_key if array_key is not None and _is_chromosome(array_key) else selected_column if selected_column is not None and _is_chromosome(selected_column) else None
    return {label: ModulationProfile(values, coordinates)}


def aligned_modulation(source, settings: ProfileModulationSettings, *, coordinates=None, chromosome=None, warning=None):
    """Match grids or align leading samples without broadcasting or resampling."""
    profile = settings.profiles.get(chromosome, settings.profiles.get(None))
    chromosome_mismatch = False
    if profile is None and len(settings.profiles) == 1 and (chromosome is None or settings.align_from_start):
        profile = next(iter(settings.profiles.values()))
        chromosome_mismatch = chromosome is not None
    if profile is None:
        raise ValueError("The modulation profile must contain every converted chromosome. Use matching chromosome columns or NPZ arrays.")
    values = np.asarray(profile.values, dtype=float)
    vector = source.ndim < 2 or 1 in source.shape
    profile_vector = values.ndim < 2 or 1 in values.shape
    if settings.align_from_start and profile_vector:
        values = values.reshape(-1, 1)
    if vector and profile_vector:
        values = values.reshape(-1, 1)
        source_shape = (source.size, 1)
    else:
        source_shape = source.shape
    mismatch = values.shape != source_shape
    coordinate_mismatch = False
    if profile.coordinates is not None and coordinates is not None:
        left, right = np.asarray(profile.coordinates, dtype=float), np.asarray(coordinates, dtype=float)
        coordinate_mismatch = left.shape != right.shape or not np.allclose(left, right, rtol=1e-10, atol=1e-9, equal_nan=False)
    if settings.align_from_start:
        if values.ndim != len(source_shape):
            raise ValueError("Modulation requires two profiles or two space-time arrays.")
        if mismatch or coordinate_mismatch or chromosome_mismatch:
            if warning is not None:
                warning("The modulation profile differs in size or coordinates. Arrays were aligned from the first sample; only overlapping samples were combined. Unmatched source values were kept.")
            overlap = tuple(slice(0, min(a, b)) for a, b in zip(source_shape, values.shape))
            aligned = np.ma.masked_all(source_shape, dtype=float)
            aligned[overlap] = values[overlap]
            return aligned
        return values.reshape(-1, 1) if vector else values
    if vector and profile_vector and source.size == values.size:
        values = values.reshape(source.shape)
    if values.shape != source.shape:
        raise ValueError(f"The modulation profile shape {values.shape} does not match the post-Configure data shape {source.shape}. Use the same bin count and matrix dimensions.")
    if coordinate_mismatch:
        raise ValueError("The modulation profile coordinates do not match the converted region. Use the same region, resolution and bin order.")
    return values.reshape(-1, 1) if vector else values


def _scaled_profile(profile, settings):
    values = np.array(profile, dtype=float, copy=True)
    if settings.missing_policy == "zero":
        values[np.isnan(values)] = 0
    if settings.operation == "convolve" or settings.scaling == "none":
        return values
    finite = values[np.isfinite(values)]
    if not finite.size:
        raise ValueError("Modulation scaling requires finite profile values.")
    if settings.scaling == "interval":
        lower, upper = finite.min(), finite.max()
        if lower == upper:
            raise ValueError("0–1 scaling requires distinct modulation profile values.")
        # Rescale first to avoid overflow in upper - lower for signed profiles.
        magnitude = max(abs(lower), abs(upper))
        values = (values / magnitude - lower / magnitude) / (upper / magnitude - lower / magnitude)
    else:
        magnitude = np.max(np.abs(finite))
        if magnitude == 0:
            raise ValueError("Modulation scaling cannot normalize an all-zero profile. Choose Original values.")
        divisor = np.mean(finite / magnitude) if settings.scaling == "mean" else 1.
        if divisor == 0:
            raise ValueError("Mean = 1 scaling requires a non-zero modulation profile mean.")
        with np.errstate(over="ignore", invalid="ignore"):
            values = (values / magnitude) / divisor
    if np.isinf(values).any():
        raise ValueError("Modulation scaling produced infinite values. Choose another scaling option.")
    return values


def modulate_region(original, profile, selected, settings: ProfileModulationSettings, *, coordinates=None, cancel=None):
    if settings.operation == "expression":
        if cancel is not None:
            cancel()
        result = evaluate_formula(settings.expression, {"source": original, "profile": profile}, check_mask=selected)
        if cancel is not None:
            cancel()
        return result[selected]
    if settings.operation == "blend" and settings.weight == 0:
        return original[selected].copy()
    unused_profile = settings.operation == "formula" and settings.combination == "sum" and settings.profile_multiplier == 0
    values = profile if unused_profile else _scaled_profile(profile, settings)
    if settings.operation == "convolve":
        if coordinates is not None and len(coordinates) > 2:
            steps = np.diff(coordinates)
            if not np.isfinite(steps).all() or np.any(steps <= 0) or not np.allclose(steps, steps[0], rtol=1e-7, atol=1e-10):
                raise ValueError("Convolution requires uniformly spaced converted bins.")
        signal = original.copy()
        if settings.missing_policy == "zero":
            signal[np.isnan(signal)] = 0
        if not np.isfinite(signal).all() or not np.isfinite(values).all():
            raise ValueError("Convolution needs finite values in both profiles. Fill missing values or choose Use zero.")
        if settings.normalize_kernel:
            magnitude = np.max(np.abs(values))
            total = np.sum(values / magnitude) if magnitude else 0.
            if total == 0:
                raise ValueError("Kernel normalization requires a non-zero sum. Change the profile or turn normalization off.")
            values = (values / magnitude) / total
        from scipy.signal import convolve
        padded = np.pad(signal, (len(values) // 2, (len(values) - 1) // 2),
                        mode="constant" if settings.boundary == "zero" else settings.boundary)
        if cancel is not None:
            cancel()
        result = convolve(padded, values, mode="valid", method="auto")
        if not np.isfinite(result).all():
            raise ValueError("Convolution produced non-finite values. Normalize the kernel or reduce the profile values.")
        if cancel is not None:
            cancel()
        return result[selected]
    modifier = values[selected]
    signal = original[selected]
    if settings.missing_policy == "error" and not unused_profile and np.isnan(modifier).any():
        raise ValueError("The modulation profile has missing values in the selected region. Choose a missing-value option or prepare the profile first.")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        if settings.operation == "formula":
            numerator = _formula_operand(signal, settings, "signal")
            denominator = _formula_operand(modifier, settings, "profile")
            if settings.combination == "sum":
                result = numerator + denominator
            elif settings.combination == "multiply":
                result = numerator * denominator
            else:
                zero = denominator == 0
                if zero.any() and settings.zero_policy == "error":
                    raise ValueError("The formula contains zero divisors in the selected region.")
                result = np.divide(numerator, denominator, out=np.full_like(signal, np.nan), where=~zero)
                if settings.zero_policy == "keep":
                    result[zero] = signal[zero]
        elif settings.operation == "multiply":
            result = signal * modifier
        elif settings.operation == "add":
            result = signal + modifier
        elif settings.operation == "subtract":
            result = signal - modifier
        elif settings.operation == "divide":
            zero = modifier == 0
            if zero.any() and settings.zero_policy == "error":
                raise ValueError("The modulation profile contains zero divisors in the selected region. Choose how to handle division by zero.")
            result = np.divide(signal, modifier, out=np.full_like(signal, np.nan), where=~zero)
            if settings.zero_policy == "keep":
                result[zero] = signal[zero]
        elif settings.operation == "blend":
            result = modifier.copy() if settings.weight == 1 else (1 - settings.weight) * signal + settings.weight * modifier
        elif settings.operation == "minimum":
            result = np.minimum(signal, modifier)
        else:
            result = np.maximum(signal, modifier)
    if np.isinf(result).any():
        raise ValueError("Profile modulation produced infinite values. Scale the modulation profile or reduce its values.")
    return result


def _formula_operand(values, settings, operand):
    multiplier = getattr(settings, f"{operand}_multiplier")
    if multiplier == 0:
        return np.zeros_like(values)
    function = getattr(settings, f"{operand}_function")
    exponent = getattr(settings, f"{operand}_exponent")
    if function in {"ln", "log2", "log10"} and np.any(values <= 0):
        raise ValueError(f"{operand.capitalize()} logarithms require positive values in the selected regions.")
    transformed = {"identity": lambda v: v, "ln": np.log, "log2": np.log2, "log10": np.log10, "exp": np.exp}[function](values)
    powered = np.power(transformed, exponent)
    if np.any(np.isnan(powered) & np.isfinite(values)):
        raise ValueError(f"{operand.capitalize()} exponent is undefined for some selected values.")
    result = multiplier * powered
    if np.isinf(result).any():
        raise ValueError("The formula produced infinite values. Reduce its multipliers or exponents.")
    return result


def modulation_metadata(settings: ProfileModulationSettings):
    metadata = {"modulation_operation": settings.operation, "modulation_missing": settings.missing_policy}
    if settings.align_from_start:
        metadata["modulation_alignment"] = "start_overlap"
    if settings.operation == "expression":
        metadata["modulation_formula"] = settings.expression
    if settings.operation == "convolve":
        metadata.update(modulation_boundary=settings.boundary, modulation_normalize_kernel=str(settings.normalize_kernel).lower())
    else:
        metadata["modulation_scaling"] = settings.scaling
    if settings.operation == "formula":
        metadata.update({f"modulation_{key}": str(getattr(settings, key)) for key in (
            "combination", "signal_multiplier", "profile_multiplier", "signal_exponent", "profile_exponent", "signal_function", "profile_function",
        )})
    if settings.operation == "divide" or (settings.operation == "formula" and settings.combination == "divide"):
        metadata["modulation_zero_policy"] = settings.zero_policy
    elif settings.operation == "blend":
        metadata["modulation_weight"] = str(settings.weight)
    if settings.source_sha256:
        metadata["modulation_source_sha256"] = settings.source_sha256
    return tuple(metadata.items())
