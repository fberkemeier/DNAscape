"""Region selection and edits on converted profiles, without changing their grid."""

from __future__ import annotations

import gzip
import io
import re
import zlib
from collections.abc import Callable
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from dnascape.workflows.uploads import UploadInspection, materialize_upload
from dnascape.workflows.modulation import (
    ProfileModulationSettings, aligned_modulation, modulate_region, modulation_metadata, resolve_modulation,
)


FILTER_SELECTIONS = {
    "threshold": "Signal threshold",
    "between": "Between thresholds",
    "minimum": "Threshold from minimum",
    "maximum": "Threshold from maximum",
    "nan": "Missing values (NaN)",
    "ranges": "Position ranges",
    "profile": "Upload profile",
    "all": "Whole region",
}
FILTER_ACTIONS = {
    "replace": "Replace with value",
    "mean": "Mean of each region",
    "interpolate": "Interpolate across regions",
    "noise": "Add Gaussian noise",
    "minimum": "Match minimum",
    "maximum": "Match maximum",
    "multiply": "Multiply by value",
    "add": "Add value",
    "nan": "Replace with NaN",
    "modulate": "Modulate with profile",
}
FILTER_INTERPOLATIONS = {
    "linear": "Linear", "pchip": "Smooth (PCHIP)", "nearest": "Nearest value",
    "previous": "Previous value", "next": "Next value",
}
FILTER_PROFILE_LAYOUTS = {
    "mask": "Mask profile", "positions": "Position list", "intervals": "Genomic intervals",
}


@dataclass(frozen=True, slots=True)
class FilterProfile:
    """A bin-aligned mask, a position list, or half-open intervals."""

    values: np.ndarray = field(repr=False, compare=False)
    layout: str = "mask"
    units: str = "bp"
    chromosomes: np.ndarray | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class DataFilterSettings:
    enabled: bool = False
    selection: str = "threshold"
    comparison: str = "below"
    threshold: float = 0.0
    lower: float = 0.0
    upper: float = 1.0
    action: str = "replace"
    value: float = 0.0
    standard_deviation: float = 1.0
    seed: int | None = None
    interpolation: str = "linear"
    reference: str = "outside"
    profile: FilterProfile | None = field(default=None, repr=False, compare=False)
    upload: UploadInspection | None = field(default=None, repr=False, compare=False)
    profile_layout: str = "mask"
    profile_units: str = "bp"
    position_ranges: str = ""
    profile_column: str | None = None
    profile_array_key: str | None = None
    modulation: ProfileModulationSettings = field(default_factory=ProfileModulationSettings)
    between_mode: str = "within"
    complement: bool = False
    _position_sink: Callable | None = field(default=None, repr=False, compare=False)
    _warning_sink: Callable | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class FilterSelection:
    """Packed native-bin selection for the displayed profile, before editing."""

    packed: bytes = field(repr=False)
    length: int
    step: float = 1.0
    end: float = 0.0

    @classmethod
    def from_mask(cls, mask, display_positions=None):
        mask = np.asarray(mask, dtype=bool).reshape(-1)
        positions = None if display_positions is None else np.asarray(display_positions)
        step = float(positions[1] - positions[0]) if positions is not None and len(positions) > 1 else 1.
        end = float(positions[-1] - positions[0]) if positions is not None and len(positions) else max(0, mask.size - 1)
        return cls(np.packbits(mask).tobytes(), mask.size, step, end)

    def ranges(self, x_limits=None, *, max_regions: int = 750) -> list[tuple[float, float]]:
        """Bound shading to screen detail; recover exact bins when zooming in."""
        if not self.length or self.step <= 0:
            return []
        first, stop = 0, self.length
        if x_limits is not None:
            first = max(0, int(np.floor(min(x_limits) / self.step)) - 2)
            stop = min(self.length, int(np.ceil(max(x_limits) / self.step)) + 3)
        if first >= stop:
            return []
        byte_start, byte_stop = first // 8, (stop + 7) // 8
        mask = np.unpackbits(np.frombuffer(self.packed[byte_start:byte_stop], dtype=np.uint8))
        mask = mask[first - byte_start * 8:stop - byte_start * 8]
        stride = max(1, int(np.ceil(mask.size / (2 * max_regions))))
        mask = np.maximum.reduceat(mask, np.arange(0, mask.size, stride))
        edges = np.flatnonzero(np.diff(np.r_[0, mask, 0]))

        def boundary(index):
            if self.length == 1:
                return -.5 if index == 0 else .5
            if index == self.length:
                return self.end + (self.end - (self.length - 2) * self.step) / 2
            if index == self.length - 1:
                return ((self.length - 2) * self.step + self.end) / 2
            return (index - .5) * self.step

        return [(boundary(first + int(start) * stride), boundary(min(stop, first + int(end) * stride)))
                for start, end in edges.reshape(-1, 2)]


def capture_filter_selection(values, settings, *, positions=None, intervals=None,
                             chromosome=None, display_positions=None) -> FilterSelection | None:
    """Inspect a selection independently of Apply and of operation parameters."""
    try:
        selection = resolve_filter(replace(settings, enabled=True, action="replace", value=0.))
        if not selection.enabled:
            return None
        array = np.asarray(values, dtype=float)
        series = array.reshape(-1) if array.ndim < 2 or array.shape[0] == 1 else array[:, 0]
        mask = _selected(series, selection, positions, intervals, chromosome)
        return FilterSelection.from_mask(mask, display_positions)
    except (ValueError, TypeError):
        # Inactive, unfinished controls must never prevent an unfiltered conversion.
        if settings.enabled:
            raise
        return None


def _choice(value: str, choices, label: str) -> str:
    if value not in choices:
        raise ValueError(f"Choose a valid {label}.")
    return value


def _finite(value: float, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be a finite number.") from None
    if not np.isfinite(result):
        raise ValueError(f"{label} must be a finite number.")
    return result


def parse_position_ranges(text: str, *, units: str = "bp") -> FilterProfile | None:
    """Read comma-separated (start, end) intervals without evaluating code."""
    _choice(units, {"bp", "kb", "index"}, "mask coordinates")
    if not isinstance(text, str):
        raise ValueError("Enter position ranges as (1000,2000), (4000,5000).")
    if len(text) > 65536:
        raise ValueError("Position ranges are too long; use Upload profile instead.")
    text = text.strip()
    if not text:
        return None
    number = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
    pair = re.compile(rf"\(\s*({number})\s*,\s*({number})\s*\)")
    ranges, end = [], 0
    for match in pair.finditer(text):
        separator = text[end:match.start()].strip()
        if separator != ("," if ranges else ""):
            raise ValueError("Enter position ranges as (1000,2000), (4000,5000).")
        ranges.append((float(match[1]), float(match[2])))
        end = match.end()
    if not ranges or text[end:].strip() not in {"", ","}:
        raise ValueError("Enter position ranges as (1000,2000), (4000,5000).")
    profile = FilterProfile(np.asarray(ranges, dtype=float), layout="intervals", units=units)
    _validate_profile(profile)
    return profile


def resolve_filter(settings: DataFilterSettings) -> DataFilterSettings:
    """Validate active fields only and materialize an optional uploaded mask once."""
    if not settings.enabled:
        return settings
    _choice(settings.selection, FILTER_SELECTIONS, "mask selection")
    if settings.selection == "ranges":
        settings = replace(settings, selection="profile", upload=None, profile_layout="intervals",
                           profile=parse_position_ranges(settings.position_ranges, units=settings.profile_units))
    if settings.selection == "profile" and settings.profile is None and settings.upload is None:
        return replace(settings, enabled=False)
    _choice(settings.action, FILTER_ACTIONS, "region transform operation")
    changes = {}
    if settings.action == "modulate":
        changes["modulation"] = resolve_modulation(settings.modulation)
    if settings.selection in {"threshold", "minimum", "maximum"}:
        _choice(settings.comparison, {"below", "above"}, "threshold comparison")
        changes["threshold"] = _finite(settings.threshold, "Mask threshold")
        if settings.selection != "threshold" and changes["threshold"] < 0:
            raise ValueError("Distance from the minimum or maximum must be non-negative.")
    if settings.selection == "between":
        _choice(settings.between_mode, {"within", "outside"}, "threshold range")
        changes.update(lower=_finite(settings.lower, "Lower mask threshold"),
                       upper=_finite(settings.upper, "Upper mask threshold"))
        if changes["lower"] > changes["upper"]:
            raise ValueError("Lower mask threshold cannot exceed the upper threshold.")
    if settings.action in {"replace", "multiply", "add"}:
        changes["value"] = _finite(settings.value, "Transform value")
    if settings.action == "noise":
        changes["standard_deviation"] = _finite(settings.standard_deviation, "Noise standard deviation")
        if changes["standard_deviation"] < 0:
            raise ValueError("Noise standard deviation must be non-negative.")
        if settings.seed is not None:
            try:
                seed = int(settings.seed)
            except (ValueError, TypeError, OverflowError):
                raise ValueError("Random seed must be a non-negative integer.") from None
            if seed < 0 or seed != settings.seed:
                raise ValueError("Random seed must be a non-negative integer.")
            changes["seed"] = seed
    if settings.action == "interpolate":
        _choice(settings.interpolation, FILTER_INTERPOLATIONS, "region transform interpolation")
    if settings.action in {"minimum", "maximum"}:
        _choice(settings.reference, {"outside", "all"}, "reference region")
    if settings.selection == "profile":
        _choice(settings.profile_layout, FILTER_PROFILE_LAYOUTS, "mask profile format")
        _choice(settings.profile_units, {"bp", "kb", "index"}, "mask coordinates")
        if settings.upload is not None:
            changes.update(profile=read_filter_profile(
                settings.upload, layout=settings.profile_layout, units=settings.profile_units,
                column=settings.profile_column, array_key=settings.profile_array_key,
            ), upload=None)
        profile = changes.get("profile", settings.profile)
        if profile is not None:
            _validate_profile(profile)
    return replace(settings, **changes)


def _validate_profile(profile: FilterProfile) -> None:
    _choice(profile.layout, FILTER_PROFILE_LAYOUTS, "mask profile format")
    _choice(profile.units, {"bp", "kb", "index"}, "mask coordinates")
    values = np.asarray(profile.values, dtype=float)
    if profile.layout == "intervals":
        if values.ndim != 2 or values.shape[1] != 2:
            raise ValueError("Mask intervals require start and end columns.")
        if not np.isfinite(values).all() or np.any(values[:, 0] >= values[:, 1]) or np.any(values < 0):
            raise ValueError("Mask intervals need finite, non-negative starts and larger ends.")
    elif values.ndim != 1:
        raise ValueError("A mask profile or position list must be one-dimensional.")
    elif np.isinf(values).any() or (profile.layout == "positions" and (np.isnan(values).any() or np.any(values < 0))):
        raise ValueError("Mask positions must be finite and non-negative; mask values cannot be infinite.")
    if profile.chromosomes is not None and len(profile.chromosomes) != len(values):
        raise ValueError("Mask chromosome labels must match the number of positions.")


def read_filter_profile(
    inspection: UploadInspection, *, layout: str = "mask", units: str = "bp",
    column: str | None = None, array_key: str | None = None,
) -> FilterProfile:
    """Reuse upload bounds and numeric validation; never cache the auxiliary file."""
    chromosomes = None
    if layout == "mask":
        values = materialize_upload(replace(inspection, expected_form="x"),
                                    column=column, array_key=array_key, missing_policy="ignore")
    elif inspection.source_format in {"npy", "npv", "npz"}:
        values = materialize_upload(replace(inspection, expected_form="auto"),
                                    array_key=array_key, missing_policy="ignore")
        if layout == "positions":
            values = np.asarray(values).reshape(-1) if np.asarray(values).ndim < 2 or 1 in np.asarray(values).shape else values
    else:
        table = _upload_table(inspection)
        names = {str(name).strip().lower().replace(" ", "_"): name for name in table.columns}
        chrom = next((names[name] for name in ("chrom", "chromosome", "chr") if name in names), None)
        if chrom is not None:
            chromosomes = table[chrom].astype(str).to_numpy()
        numeric = [name for name in table.columns if name != chrom]
        if layout == "intervals":
            start = next((names[name] for name in ("start", "start_bp", "start_kb") if name in names), numeric[0] if numeric else None)
            end = next((names[name] for name in ("end", "end_bp", "end_kb") if name in names), numeric[1] if len(numeric) > 1 else None)
            if start is None or end is None:
                raise ValueError("Mask intervals require start and end columns.")
            values = table[[start, end]].to_numpy(dtype=float)
        else:
            position = next((names[name] for name in ("position_bp", "position_kb", "position", "pos", "start") if name in names), numeric[0] if numeric else None)
            if position is None:
                raise ValueError("The mask needs a numeric position column.")
            values = table[position].to_numpy(dtype=float)
    profile = FilterProfile(np.asarray(values, dtype=float), layout, units, chromosomes)
    _validate_profile(profile)
    return profile


def _upload_table(inspection: UploadInspection, columns=None) -> pd.DataFrame:
    # Inspection already checks compressed/expanded sizes and the table shape.
    payload = inspection._normalized_payload or inspection.content
    if inspection.compressed and inspection._normalized_payload is None:
        payload = gzip.decompress(payload)
    payload = b"\n".join(line for line in payload.splitlines()
                          if not line.lstrip().startswith((b"track ", b"browser ")))
    return pd.read_csv(io.BytesIO(payload), sep=inspection.delimiter or r"\s+",
                       header=0 if inspection.has_header else None, names=inspection.columns,
                       usecols=columns, comment="#")


def detect_filter_profile(inspection: UploadInspection, *, units="kb") -> FilterProfile:
    """Read one position column or two half-open interval columns automatically."""
    chromosomes = None
    if inspection.source_format == "npz":
        if len(inspection.array_keys) != 1:
            raise ValueError("Upload one array containing positions or start/end pairs.")
        values = np.asarray(materialize_upload(
            replace(inspection, expected_form="auto"),
            array_key=inspection.array_keys[0], missing_policy="ignore",
        ), dtype=float)
    elif inspection.source_format in {"csv", "tsv", "txt"}:
        table = _upload_table(inspection)
        chrom = next(
            (name for name in table.columns if str(name).strip().lower() in {"chrom", "chromosome", "chr"}),
            None,
        )
        if chrom is not None:
            chromosomes = table[chrom].astype(str).to_numpy()
            table = table.drop(columns=[chrom])
        values = table.to_numpy(dtype=float)
    else:
        raise ValueError("Upload positions or start/end pairs as TSV, TXT, CSV or NPZ.")
    if values.ndim == 1 or (values.ndim == 2 and values.shape[1] == 1):
        layout, values = "positions", values.reshape(-1)
    elif values.ndim == 2 and values.shape[1] == 2:
        layout = "intervals"
    elif values.ndim == 2 and values.shape[0] == 1 and chromosomes is None:
        layout, values = "positions", values.reshape(-1)
    else:
        raise ValueError("Use one column of positions or two columns of interval starts and ends.")
    profile = FilterProfile(values, layout, units, chromosomes)
    _validate_profile(profile)
    return profile


def profile_coordinates(inspection: UploadInspection, length: int):
    """Coordinate metadata retained by the upload inspector, in kilobases."""
    names = {str(key).lower().replace(" ", "_"): value for key, value in inspection._table_data.items()}
    if inspection.source_format in {"csv", "tsv", "txt"}:
        # The inspector intentionally caches signal columns only. Read just the
        # coordinate columns when a filter needs the original genomic grid.
        keys = {name: str(name).lower().replace(" ", "_") for name in inspection.columns}
        columns = [name for name, key in keys.items() if key in {
            "start", "end", "start_bp", "end_bp", "position_kb", "position_bp",
            "start_kb", "end_kb", "position", "pos", "x", "time", "time_min",
        }]
        if columns:
            table = _upload_table(inspection, columns)
            names.update({keys[name]: table[name].to_numpy(dtype=float) for name in columns})
    starts = next((names[name] for name in ("start", "start_bp") if name in names), None)
    ends = next((names[name] for name in ("end", "end_bp") if name in names), None)
    if starts is not None and ends is not None and len(starts) == length:
        return (starts + ends) / 2000., np.column_stack((starts, ends)) / 1000.
    if "start_kb" in names and "end_kb" in names and len(names["start_kb"]) == length:
        return (names["start_kb"] + names["end_kb"]) / 2, np.column_stack((names["start_kb"], names["end_kb"]))
    for name, divisor in (("position_kb", 1), ("position_bp", 1000), ("position", 1), ("pos", 1), ("x", 1), ("time", 1), ("time_min", 1)):
        if name in names and len(names[name]) == length:
            return np.asarray(names[name], dtype=float) / divisor, None
    if inspection.source_format == "npz":
        payload = gzip.decompress(inspection.content) if inspection.compressed else inspection.content
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            if "position_kb" in archive and archive["position_kb"].shape == (length,):
                return np.asarray(archive["position_kb"], dtype=float), None
    return None, None


def profile_chromosome(inspection: UploadInspection) -> str | None:
    columns = [name for name in inspection.columns if str(name).lower() in {"chrom", "chromosome", "chr"}]
    if not columns or inspection.source_format not in {"csv", "tsv", "txt"}:
        return None
    values = _upload_table(inspection, [columns[0]])[columns[0]].astype(str).unique()
    if len(values) != 1:
        raise ValueError("Use a genomic track or a chromosome-specific profile for a chromosome-labelled mask.")
    return str(values[0])


def _profile_mask(profile: FilterProfile | None, length: int, positions, intervals, chromosome) -> np.ndarray:
    mask = np.zeros(length, dtype=bool)
    if profile is None:
        return mask
    values = np.asarray(profile.values, dtype=float)
    if profile.chromosomes is not None:
        if chromosome is None:
            raise ValueError("A chromosome-labelled mask requires a chromosome-specific converted profile.")
        values = values[np.asarray(profile.chromosomes) == chromosome]
    if profile.layout == "mask":
        if values.size != length:
            raise ValueError("The mask must have one value for each converted bin. Use a position list for a different grid.")
        return np.isfinite(values) & (values != 0)
    if profile.units == "index":
        coordinates = np.arange(length, dtype=float)
        bins = np.column_stack((coordinates, coordinates + 1))
    else:
        if positions is None:
            raise ValueError("This converted profile has no genomic coordinates. Use a bin-aligned mask or Bin index coordinates.")
        coordinates = np.asarray(positions, dtype=float)
        bins = intervals
        values = values / (1000. if profile.units == "bp" else 1.)
    if coordinates.shape != (length,) or not np.isfinite(coordinates).all() or np.any(np.diff(coordinates) <= 0):
        raise ValueError("Coordinate masks require finite, increasing positions in the converted profile.")
    if bins is not None:
        bins = np.asarray(bins, dtype=float)
        if (bins.shape != (length, 2) or not np.isfinite(bins).all() or np.any(bins[:, 0] >= bins[:, 1])
                or np.any(bins[1:, 0] < bins[:-1, 1])):
            raise ValueError("Coordinate masks require ordered, non-overlapping converted bins.")
    if profile.layout == "positions":
        if bins is not None:
            indices = np.searchsorted(bins[:, 0], values, side="right") - 1
            valid = (indices >= 0) & (indices < length)
            valid &= values < bins[np.clip(indices, 0, length - 1), 1]
        else:
            indices = np.searchsorted(coordinates, values)
            valid = indices < length
            valid &= np.isclose(coordinates[np.clip(indices, 0, length - 1)], values, rtol=0, atol=1e-9)
        mask[indices[valid]] = True
    else:
        # Merge via a difference array: O((bins + intervals) log bins), never
        # a potentially chromosome-sized bins-by-intervals matrix.
        starts = bins[:, 0] if bins is not None else coordinates
        ends = bins[:, 1] if bins is not None else coordinates
        left = np.searchsorted(ends, values[:, 0], side="right" if bins is not None else "left")
        right = np.searchsorted(starts, values[:, 1], side="left")
        delta = np.zeros(length + 1, dtype=np.int64)
        np.add.at(delta, left, 1)
        np.add.at(delta, right, -1)
        mask = np.cumsum(delta[:-1]) > 0
    return mask


def _selection_mask(values: np.ndarray, settings: DataFilterSettings, positions, intervals, chromosome) -> np.ndarray:
    if settings.selection == "profile":
        return _profile_mask(settings.profile, len(values), positions, intervals, chromosome)
    if settings.selection == "all":
        return np.ones(values.size, dtype=bool)
    if settings.selection == "nan":
        return np.isnan(values)
    if settings.selection == "between":
        if settings.between_mode == "outside":
            return (values < settings.lower) | (values > settings.upper)
        return (values >= settings.lower) & (values <= settings.upper)
    threshold = settings.threshold
    if settings.selection in {"minimum", "maximum"}:
        finite = values[np.isfinite(values)]
        if not finite.size:
            raise ValueError("A threshold from minimum or maximum requires finite converted values.")
        threshold = float(finite.min() + threshold if settings.selection == "minimum" else finite.max() - threshold)
    return values < threshold if settings.comparison == "below" else values > threshold


def _selected(values: np.ndarray, settings: DataFilterSettings, positions, intervals, chromosome) -> np.ndarray:
    # Values already cover the post-Configure Start/End region. Invert this
    # bounded selection once, so operations, shading and exports stay aligned.
    mask = _selection_mask(values, settings, positions, intervals, chromosome)
    return ~mask if settings.complement else mask


def apply_data_filter(
    values, settings: DataFilterSettings, *, positions=None, intervals=None,
    chromosome: str | None = None, cancel: Callable[[], None] | None = None,
    selection_values=None,
) -> np.ndarray:
    """Edit each ordered profile independently, using a fixed pre-edit selection.

    XT matrices are (space, time). End regions interpolate to the nearest
    available endpoint; a wholly selected profile has no interpolation anchors.
    """
    if not settings.enabled:
        return np.asarray(values)
    settings = resolve_filter(settings)
    if not settings.enabled:
        return np.asarray(values)
    source = np.asarray(values, dtype=float)
    result = source.copy()
    columns = result.reshape(-1, 1) if result.ndim < 2 or result.shape[0] == 1 else result
    reference_columns = columns
    if selection_values is not None:
        reference_values = np.asarray(selection_values, dtype=float)
        if reference_values.ndim == 1:
            reference_values = reference_values[:, None]
        try:
            reference_columns = np.broadcast_to(reference_values, columns.shape)
        except ValueError as exc:
            raise ValueError("The selection input must match the fitted profile's coordinates.") from exc
    modulation = (aligned_modulation(source, settings.modulation, coordinates=positions, chromosome=chromosome, warning=settings._warning_sink)
                  if settings.action == "modulate" else None)
    # Reproducible, distinct noise across chromosome outputs and time slices.
    seed = settings.seed
    rng = np.random.default_rng(None if seed is None else np.random.SeedSequence([seed, zlib.crc32((chromosome or "").encode())]))
    for column in range(columns.shape[1]):
        if cancel is not None:
            cancel()
        original = columns[:, column].copy()
        selected = _selected(reference_columns[:, column], settings, positions, intervals, chromosome)
        if settings._position_sink is not None:
            settings._position_sink(
                selected, positions, chromosome, column if columns.shape[1] > 1 else None,
                intervals=intervals,
            )
        if not selected.any():
            continue
        output = columns[:, column]
        action = settings.action
        if action in {"replace", "nan"}:
            output[selected] = np.nan if action == "nan" else settings.value
        elif action == "mean":
            edges = np.flatnonzero(np.diff(np.r_[False, selected, False]))
            for first, last in zip(edges[::2], edges[1::2]):
                if cancel is not None:
                    cancel()
                finite = original[first:last][np.isfinite(original[first:last])]
                if not finite.size:
                    raise ValueError("A selected region has no finite values to average. Choose interpolation or a replacement value.")
                output[first:last] = finite.mean()
        elif action in {"minimum", "maximum"}:
            reference = original[np.isfinite(original) & (~selected if settings.reference == "outside" else True)]
            if not reference.size:
                raise ValueError("No finite reference values remain. Include selected regions or choose a replacement value.")
            output[selected] = reference.min() if action == "minimum" else reference.max()
        elif action == "interpolate":
            known = np.flatnonzero(~selected & np.isfinite(original))
            if not known.size:
                raise ValueError("Interpolation needs at least one finite value outside the selected regions.")
            x = np.arange(original.size, dtype=float) if positions is None else np.asarray(positions, dtype=float)
            if x.shape != original.shape or not np.isfinite(x).all() or np.any(np.diff(x) <= 0):
                raise ValueError("Interpolation requires finite, increasing converted positions.")
            target = x[selected]
            if settings.interpolation == "linear" or known.size == 1:
                output[selected] = np.interp(target, x[known], original[known])
            elif settings.interpolation == "pchip":
                from scipy.interpolate import PchipInterpolator
                output[selected] = PchipInterpolator(x[known], original[known])(np.clip(target, x[known[0]], x[known[-1]]))
            else:
                right = np.clip(np.searchsorted(x[known], target), 0, len(known) - 1)
                left = np.maximum(right - 1, 0)
                if settings.interpolation == "previous":
                    indices = np.clip(np.searchsorted(x[known], target, side="right") - 1, 0, len(known) - 1)
                elif settings.interpolation == "next":
                    indices = right
                else:
                    indices = np.where(target - x[known[left]] <= x[known[right]] - target, left, right)
                output[selected] = original[known[indices]]
        elif action == "modulate":
            available = ~np.ma.getmaskarray(modulation)[:, column]
            selected &= available
            if not selected.any():
                continue
            length = int(np.count_nonzero(available))
            output[selected] = modulate_region(
                original[:length], np.asarray(modulation[:length, column]), selected[:length], settings.modulation,
                coordinates=None if positions is None else np.asarray(positions)[:length], cancel=cancel,
            )
        else:
            with np.errstate(over="ignore", invalid="ignore"):
                if action == "noise":
                    output[selected] += rng.normal(0, settings.standard_deviation, selected.sum())
                elif action == "multiply":
                    output[selected] *= settings.value
                else:
                    output[selected] += settings.value
        if np.isinf(output).any():
            raise ValueError("The region transform produced infinite values. Reduce its value or noise standard deviation.")
    if cancel is not None:
        cancel()
    return result


def filter_metadata(settings: DataFilterSettings) -> tuple[tuple[str, str], ...]:
    if not settings.enabled:
        return ()
    metadata = {"filter_selection": settings.selection, "filter_action": settings.action}
    if settings.complement:
        metadata["filter_complement"] = "true"
    if settings.selection in {"threshold", "minimum", "maximum"}:
        metadata.update(filter_comparison=settings.comparison, filter_threshold=str(settings.threshold))
    elif settings.selection == "between":
        metadata.update(filter_lower=str(settings.lower), filter_upper=str(settings.upper),
                        filter_between_mode=settings.between_mode)
    elif settings.selection == "profile":
        metadata.update(filter_profile_layout=settings.profile_layout, filter_profile_units=settings.profile_units,
                        filter_profile_present=str(settings.profile is not None or settings.upload is not None).lower())
    if settings.action in {"replace", "add", "multiply"}:
        metadata["filter_value"] = str(settings.value)
    elif settings.action == "noise":
        metadata.update(filter_noise_stdev=str(settings.standard_deviation), filter_seed=str(settings.seed))
    elif settings.action == "interpolate":
        metadata["filter_interpolation"] = settings.interpolation
    elif settings.action in {"minimum", "maximum"}:
        metadata["filter_reference"] = settings.reference
    return tuple(metadata.items()) + (modulation_metadata(settings.modulation) if settings.action == "modulate" else ())
