"""Bounded data preparation and ``rdata`` conversion for the Data Manager."""

from __future__ import annotations

import gzip
import io
import math
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePath
from types import MappingProxyType
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import numpy as np
import pandas as pd

from dnascape.workflows.bigwig import inspect_bigwig, read_bigwig_bins
from dnascape.workflows.export_settings import (
    DATA_FILE_FORMAT_OPTIONS,
    DEFAULT_CONVERSION_MEMORY_LIMIT_MB,
    MAX_CONVERSION_MEMORY_LIMIT_MB,
)
from dnascape.workflows.uploads import (
    MAX_ARCHIVE_MEMBERS,
    MAX_ARRAY_ELEMENTS,
    MAX_DECOMPRESSED_BYTES,
    UploadInspection,
    materialize_upload,
)
from dnascape.workflows.result_naming import RDATA_FILE_PREFIX
from dnascape.workflows.repliseq import replication_timing_from_fractions
from dnascape.workflows.filtering import (
    DataFilterSettings, apply_data_filter, filter_metadata, profile_chromosome, profile_coordinates, resolve_filter,
)
from dnascape.workflows.exports import ExportArtifact, safe_export_filename
from dnascape.io import inspect_genomic_track, rdata, rload, rsave, rtrack

# Public UI order: interoperable text first, NumPy-specific formats last.
DATA_EXPORT_FORMATS: tuple[tuple[str, str], ...] = tuple(
    DATA_FILE_FORMAT_OPTIONS.items()
)
DEFAULT_TARGET_FORMAT = "tsv"

_TARGET_FORMATS = frozenset(key for key, _label in DATA_EXPORT_FORMATS)
_INPUT_FORMS = frozenset({"scalar", "x", "t", "xt", "table", "track"})
_FORM_TO_RDATA = {
    "scalar": "scalar",
    "x": "space",
    "t": "time",
    "xt": "spacetime",
    "table": "table",
    "track": "space",
}
_MEDIA_TYPES = {
    "npz": "application/zip",
    "txt": "text/plain; charset=utf-8",
    "tsv": "text/tab-separated-values; charset=utf-8",
    "csv": "text/csv; charset=utf-8",
}
_AGGREGATIONS = frozenset({"mean", "sum", "min", "max", "coverage", "std"})
_TRACK_MISSING = frozenset({"nan", "zero", "fill", "min", "max", "error"})
_INTERPOLATIONS = frozenset({"none", "linear", "ffill", "bfill"})
_LEGACY_NORMALIZATIONS = frozenset(
    {"none", "minmax", "zscore", "center", "maxabs", "rescale"}
)
_SCALING_MODES = frozenset({"none", "multiplier", "interval", "center", "zscore", "maxabs"})
_SIGNAL_TRANSFORMS = frozenset({"keep", "invert"})
_PLAIN_MISSING = frozenset({"error", "zero", "fill", "min", "max", "interpolate", "ignore"})
_CHROMOSOME_SCOPES = frozenset({"single", "all"})
_ALL_CHROMOSOME_FORMATS = frozenset({"tsv", "csv", "npz"})
_MAX_CONVERTED_BYTES = 128 * 1024 * 1024
_MAX_TRACK_BINS = 1_000_000
_MAX_SMOOTH_WINDOW = 100_001
_TRACK_ESTIMATED_BYTES_PER_BIN = 96

_CancelChecker = Callable[[], None]


def _check_cancelled(cancel: _CancelChecker | None) -> None:
    """Run the caller's worker-local cancellation check, when provided."""

    if cancel is not None:
        cancel()


@dataclass(frozen=True, slots=True)
class TrackRegion:
    """One chromosome and its available half-open genomic bounds."""

    chromosome: str
    start: int
    end: int
    interval_count: int | None = None


@dataclass(frozen=True, slots=True)
class _TrackSlice:
    chromosome: str
    start: int
    end: int
    bins: int


@dataclass(frozen=True, slots=True)
class _TrackLimits:
    max_bytes: int
    max_values: int
    max_chromosome_bins: int


@dataclass(frozen=True, slots=True)
class TrackInspection:
    """Small genomic-track metadata suitable for populating controls."""

    source_format: str
    regions: tuple[TrackRegion, ...]
    value_columns: tuple[str, ...] = ("value",)
    size_bytes: int = 0
    backend: str | None = None
    metadata: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class DataValueArray:
    """One selectable numeric array discovered in an uploaded source."""

    key: str
    label: str
    data_form: str
    shape: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class DataSourceInspection:
    """Unified, immutable metadata for table, NumPy and genomic-track inputs."""

    name: str
    source_format: str
    inferred_form: str
    shape: tuple[int, ...]
    values: tuple[DataValueArray, ...]
    columns: tuple[str, ...]
    chromosomes: tuple[TrackRegion, ...]
    missing_count: int
    warnings: tuple[str, ...]
    metadata: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class DataConversionRequest:
    """A fully resolved, local Data Manager conversion request."""

    inspection: UploadInspection
    input_form: str
    target_format: str = DEFAULT_TARGET_FORMAT
    output_name: str = RDATA_FILE_PREFIX
    gzip_output: bool = False
    value_column: str | None = None
    array_key: str | None = None
    missing_policy: str = "zero"
    decimal_places: int | None = None
    value_name: str = "value"
    chromosome: str | None = None
    chromosome_scope: str = "single"
    start: int | None = None
    end: int | None = None
    bin_size: int = 1_000
    aggregation: str = "mean"
    track_missing: str = "zero"
    fill_value: float = 0.0
    interpolation: str = "none"
    smooth_window: int = 1
    scaling_mode: str | None = None
    # ``normalize``, ``offset`` and ``signal_transform`` remain accepted for
    # requests created by older callers. New callers should use exactly one
    # ``scaling_mode`` and, where relevant, ``scale`` or the target interval.
    normalize: str = "none"
    rescale_min: float = 0.0
    rescale_max: float = 1.0
    rescale_invert: bool = False
    threshold_enabled: bool = False
    threshold_min: float | None = None
    threshold_max: float | None = None
    signal_transform: str = "keep"
    scale: float = 1.0
    offset: float = 0.0
    genome_build: str | None = None
    units: str | None = None
    source_url: str | None = None
    biological_role: str | None = None
    semantic_note: str | None = None
    memory_limit_mb: int = DEFAULT_CONVERSION_MEMORY_LIMIT_MB
    repliseq_timing_method: str | None = None
    s_phase_duration_min: float = 600.0
    data_filter: DataFilterSettings = field(default_factory=DataFilterSettings)


def _track_limits(request: DataConversionRequest) -> _TrackLimits:
    """Scale track allocation guards with this conversion's memory budget."""

    return _TrackLimits(
        max_bytes=request.memory_limit_mb * 1024**2,
        max_values=(
            MAX_ARRAY_ELEMENTS * request.memory_limit_mb // DEFAULT_CONVERSION_MEMORY_LIMIT_MB
        ),
        max_chromosome_bins=(
            _MAX_TRACK_BINS * request.memory_limit_mb // DEFAULT_CONVERSION_MEMORY_LIMIT_MB
        ),
    )


@dataclass(frozen=True, slots=True)
class PreparedData:
    """A bounded DNAscape-ready value array with coordinate metadata."""

    values: np.ndarray = field(repr=False, compare=False)
    source_format: str
    data_form: str
    labels: tuple[str, ...]
    coordinates: Mapping[str, np.ndarray] = field(
        default_factory=lambda: MappingProxyType({}),
        repr=False,
        compare=False,
    )
    profile_values: Mapping[str, np.ndarray] = field(
        default_factory=lambda: MappingProxyType({}),
        repr=False,
        compare=False,
    )
    profile_coordinates: Mapping[str, np.ndarray] = field(
        default_factory=lambda: MappingProxyType({}),
        repr=False,
        compare=False,
    )
    metadata: tuple[tuple[str, str], ...] = ()

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.values.shape)


@dataclass(frozen=True, slots=True)
class DataPreview:
    """At most 500 display points and full-series summary statistics."""

    x: np.ndarray = field(repr=False, compare=False)
    values: np.ndarray = field(repr=False, compare=False)
    x_label: str
    value_label: str
    source_shape: tuple[int, ...]
    total_count: int
    finite_count: int
    missing_count: int
    minimum: float | None
    maximum: float | None


@dataclass(frozen=True, slots=True)
class DataConversionResult:
    """Converted bytes and concise output metadata."""

    artifact: ExportArtifact
    source_format: str
    target_format: str
    data_form: str
    shape: tuple[int, ...]
    labels: tuple[str, ...]
    metadata: tuple[tuple[str, str], ...] = ()
    preview: DataPreview | None = field(default=None, repr=False, compare=False)


def transcode_conversion_artifact(
    result: DataConversionResult,
    *,
    output_name: str,
    target_format: str | None = None,
    gzip_output: bool = False,
) -> ExportArtifact:
    """Re-encode a retained conversion result without repeating data preparation."""

    resolved_target = str(target_format or result.target_format).strip().casefold().lstrip(".")
    if resolved_target not in _TARGET_FORMATS:
        raise ValueError(f"Unsupported generated file type {target_format!r}.")

    source_format = result.target_format.casefold()
    payload = result.artifact.data
    if result.artifact.filename.casefold().endswith(".gz"):
        try:
            payload = gzip.decompress(payload)
        except (EOFError, OSError) as exc:
            raise ValueError("The converted gzip file could not be decompressed.") from exc

    if resolved_target != source_format:
        data_form = {
            "x": "space",
            "t": "time",
            "xt": "spacetime",
        }.get(result.data_form, result.data_form)
        with tempfile.TemporaryDirectory(prefix="dnascape-generated-") as temporary:
            directory = Path(temporary)
            source_path = directory / f"source.{source_format}"
            output_path = directory / f"output.{resolved_target}"
            source_path.write_bytes(payload)
            rdata(
                source_path,
                target=resolved_target,
                output_path=output_path,
                source_format=source_format,
                data_form=data_form,
                value_column=(
                    result.labels[0]
                    if source_format in {"csv", "tsv"}
                    and len(result.labels) == 1
                    and data_form != "table"
                    else None
                ),
                labels=result.labels or None,
            )
            payload = _read_output(output_path)

    if gzip_output:
        payload = gzip.compress(payload, mtime=0)
    filename = safe_export_filename(
        output_name,
        resolved_target,
        default_stem=RDATA_FILE_PREFIX,
        gzip_output=gzip_output,
    )
    return ExportArtifact(
        data=payload,
        filename=filename,
        media_type=("application/gzip" if gzip_output else _MEDIA_TYPES[resolved_target]),
    )


def _safe_source_name(name: str, source_format: str, compressed: bool) -> str:
    safe_name = PurePath(str(name or "upload").replace("\\", "/")).name or "upload"
    if compressed and safe_name.casefold().endswith(".gz"):
        safe_name = safe_name[:-3]
    suffixes = {
        "bigwig": ".bw",
        "bedgraph": ".bedgraph",
        "csv": ".csv",
        "tsv": ".tsv",
        "txt": ".txt",
        "npy": ".npy",
        "npv": ".npv",
        "npz": ".npz",
    }
    expected = suffixes.get(source_format)
    if expected and not PurePath(safe_name).suffix:
        safe_name += expected
    return safe_name


def _source_payload(
    inspection: UploadInspection,
    cancel: _CancelChecker | None = None,
) -> bytes:
    _check_cancelled(cancel)
    if inspection._normalized_payload is not None:
        return inspection._normalized_payload
    if not inspection.compressed:
        if len(inspection.content) > MAX_DECOMPRESSED_BYTES:
            raise ValueError("The uploaded source exceeds the local conversion limit.")
        _check_cancelled(cancel)
        return inspection.content

    chunks: list[bytes] = []
    total = 0
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(inspection.content)) as source:
            while True:
                _check_cancelled(cancel)
                chunk = source.read(min(1024 * 1024, MAX_DECOMPRESSED_BYTES - total + 1))
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_DECOMPRESSED_BYTES:
                    raise ValueError("The compressed source expands beyond the 64 MB limit.")
                chunks.append(chunk)
    except ValueError:
        raise
    except (EOFError, OSError) as exc:
        raise ValueError("The compressed upload is incomplete or invalid.") from exc
    _check_cancelled(cancel)
    return b"".join(chunks)


def _write_source(
    directory: Path,
    inspection: UploadInspection,
    cancel: _CancelChecker | None = None,
) -> Path:
    _check_cancelled(cancel)
    source_directory = directory / "source"
    source_directory.mkdir(exist_ok=True)
    source_path = source_directory / _safe_source_name(
        inspection.name,
        inspection.source_format,
        inspection.compressed,
    )
    source_path.write_bytes(_source_payload(inspection, cancel))
    _check_cancelled(cancel)
    return source_path


def _integer(value: Any, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a whole number.")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a whole number.") from exc
    if not np.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"{name} must be a whole number.")
    result = int(numeric)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return result


def _boolean(value: Any, name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    raise ValueError(f"{name} must be true or false.")


def _optional_finite_number(value: Any, name: str) -> float | None:
    if value in (None, ""):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number.") from exc
    if not np.isfinite(numeric):
        raise ValueError(f"{name} must be a finite number.")
    return numeric


def _choice(value: Any, name: str, allowed: frozenset[str]) -> str:
    normalized = str(value).strip().casefold().replace("-", "_").replace(" ", "_")
    if normalized not in allowed:
        raise ValueError(f"Unsupported {name} {value!r}.")
    return normalized


def _optional_label(value: Any, name: str, *, maximum: int) -> str | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    if not text:
        return None
    if len(text) > maximum or any(ord(character) < 32 for character in text):
        raise ValueError(f"{name} must be a short, printable value.")
    return text


def _public_provenance_url(value: Any) -> str | None:
    text = _optional_label(value, "Source URL", maximum=2_048)
    if text is None:
        return None
    parsed = urlsplit(text)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Source URL must be an absolute HTTP or HTTPS URL.")
    if parsed.username or parsed.password:
        raise ValueError("Source URL cannot contain a username or password.")
    # Queries often contain temporary access tokens. Provenance records the
    # public resource location without retaining credentials or fragments.
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))


def _polishing_active(
    *,
    smooth_window: int,
    scaling_mode: str,
    threshold_enabled: bool,
    signal_transform: str,
    scale: float,
    offset: float,
) -> bool:
    return (
        smooth_window != 1
        or scaling_mode != "none"
        or threshold_enabled
        or signal_transform != "keep"
        or scale != 1.0
        or offset != 0.0
    )


def _normalized_request(request: DataConversionRequest) -> DataConversionRequest:
    if request.repliseq_timing_method is not None:
        if request.inspection.source_layout != "geo_repliseq" or request.input_form != "x":
            raise ValueError("Repli-seq timing extraction requires a GEO S1–S16 matrix and a space profile.")
        if request.repliseq_timing_method not in {"sigmoid", "mean"}:
            raise ValueError("Choose sigmoid or mean replication timing.")
        duration = float(request.s_phase_duration_min)
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError("S-phase duration must be a positive finite number of minutes.")
    memory_limit_mb = _integer(
        request.memory_limit_mb,
        "Conversion memory limit (MB)",
        minimum=DEFAULT_CONVERSION_MEMORY_LIMIT_MB,
    )
    if memory_limit_mb > MAX_CONVERSION_MEMORY_LIMIT_MB:
        raise ValueError(
            f"Conversion memory limit (MB) cannot exceed {MAX_CONVERSION_MEMORY_LIMIT_MB:,}."
        )
    input_form = _choice(request.input_form, "input dependence", _INPUT_FORMS)
    data_filter = resolve_filter(request.data_filter)
    if data_filter.enabled and input_form in {"scalar", "table"}:
        raise ValueError("Filtering requires a profile. Choose a numeric profile rather than a constant or full table.")
    chromosome_scope = _choice(
        request.chromosome_scope,
        "chromosome scope",
        _CHROMOSOME_SCOPES,
    )
    target_format = str(request.target_format).strip().casefold().lstrip(".")
    if target_format not in _TARGET_FORMATS:
        raise ValueError(f"Unsupported output format {request.target_format!r}.")
    missing_policy = _choice(request.missing_policy, "missing-value policy", _PLAIN_MISSING)
    aggregation = _choice(request.aggregation, "track aggregation", _AGGREGATIONS)
    track_missing = _choice(request.track_missing, "track missing-value policy", _TRACK_MISSING)
    interpolation = _choice(request.interpolation, "interpolation", _INTERPOLATIONS)
    legacy_normalize = _choice(
        request.normalize,
        "normalization",
        _LEGACY_NORMALIZATIONS,
    )
    decimal_places = (
        None
        if request.decimal_places is None
        else _integer(request.decimal_places, "Decimal places", minimum=0)
    )
    if decimal_places is not None and decimal_places > 15:
        raise ValueError("Decimal places cannot exceed 15.")
    bin_size = _integer(request.bin_size, "Bin size", minimum=1)
    smooth_window = _integer(request.smooth_window, "Smoothing window", minimum=1)
    if smooth_window > _MAX_SMOOTH_WINDOW:
        raise ValueError(f"Smoothing window cannot exceed {_MAX_SMOOTH_WINDOW:,} bins.")
    threshold_enabled = _boolean(request.threshold_enabled, "Threshold filtering")
    if threshold_enabled:
        threshold_min = _optional_finite_number(request.threshold_min, "Lower threshold")
        threshold_max = _optional_finite_number(request.threshold_max, "Upper threshold")
        if threshold_min is None and threshold_max is None:
            raise ValueError("Threshold filtering requires a lower or upper threshold.")
        if (
            threshold_min is not None
            and threshold_max is not None
            and threshold_min > threshold_max
        ):
            raise ValueError("Lower threshold cannot exceed upper threshold.")
    else:
        # Hidden UI fields can retain stale text while the operation is off.
        # Disabled thresholds are deliberately inert and omitted from provenance.
        threshold_min = None
        threshold_max = None
    legacy_scaling = request.scaling_mode in (None, "")
    if legacy_scaling:
        signal_transform = _choice(
            request.signal_transform,
            "signal transform",
            _SIGNAL_TRANSFORMS,
        )
        scaling_mode = {
            "none": "multiplier" if request.scale != 1.0 or request.offset != 0.0 else "none",
            "minmax": "interval",
            "rescale": "interval",
            "center": "center",
            "zscore": "zscore",
            "maxabs": "maxabs",
        }[legacy_normalize]
    else:
        # Explicit value scaling is intentionally self-contained. Legacy signal
        # inversion is inert; target-interval inversion is the sole current UI
        # mechanism for reversing a scale.
        signal_transform = "keep"
        scaling_mode = _choice(request.scaling_mode, "value scaling", _SCALING_MODES)
        if legacy_normalize != "none":
            raise ValueError(
                "Choose value scaling with scaling_mode; it cannot be combined with normalize."
            )

    if legacy_scaling and legacy_normalize == "minmax":
        # Preserve the historical fixed 0..1 min-max operation; target-range
        # fields were inactive for this legacy mode.
        rescale_min = 0.0
        rescale_max = 1.0
        rescale_invert = False
    elif scaling_mode == "interval":
        rescale_invert = _boolean(request.rescale_invert, "Range inversion")
        rescale_min = _optional_finite_number(request.rescale_min, "Target minimum")
        rescale_max = _optional_finite_number(request.rescale_max, "Target maximum")
        if rescale_min is None or rescale_max is None:
            raise ValueError("Target-interval scaling requires a minimum and maximum.")
        if rescale_max <= rescale_min:
            raise ValueError("Target maximum must be greater than target minimum.")
    else:
        # Hidden controls are inert when target-interval scaling is not selected.
        rescale_min = 0.0
        rescale_max = 1.0
        rescale_invert = False
    fill_value = float(request.fill_value)
    if not np.isfinite(fill_value):
        raise ValueError("Missing-value fill must be finite.")
    if scaling_mode == "multiplier" or legacy_scaling:
        scale = float(request.scale)
        if not np.isfinite(scale) or abs(scale) > 1e12:
            raise ValueError("Multiplier must be finite and no larger than 1e12 in magnitude.")
    else:
        scale = 1.0
    if legacy_scaling:
        offset = float(request.offset)
        if not np.isfinite(offset) or abs(offset) > 1e12:
            raise ValueError("Legacy offset must be finite and no larger than 1e12 in magnitude.")
    else:
        # Offset is deliberately not part of the current scaling API. Reset a
        # stale value when an explicit mode is supplied rather than activating it.
        offset = 0.0
    value_name = str(request.value_name or "").strip()
    if not value_name or len(value_name) > 80 or any(ord(char) < 32 for char in value_name):
        raise ValueError("Value name must be a short, printable label.")
    if input_form == "track" and request.inspection.source_format not in {"bigwig", "bedgraph"}:
        raise ValueError("Track conversion requires a BigWig or bedGraph upload.")
    if input_form != "track" and request.inspection.source_format in {"bigwig", "bedgraph"}:
        raise ValueError("BigWig and bedGraph uploads must use track conversion.")
    if input_form in {"scalar", "table"} and _polishing_active(
        smooth_window=smooth_window,
        scaling_mode=scaling_mode,
        threshold_enabled=threshold_enabled,
        signal_transform=signal_transform,
        scale=scale,
        offset=offset,
    ):
        raise ValueError(
            "Data polishing is available for space, time, space-time, and genomic-track "
            "profiles; scalar and full-table conversions are preserved unchanged."
        )
    if input_form != "track" and chromosome_scope != "single":
        raise ValueError("All-chromosome scope is available only for genomic tracks.")
    if input_form == "track" and chromosome_scope == "all":
        if request.chromosome not in (None, "", "all", "*"):
            raise ValueError("All-chromosome scope cannot also select one chromosome.")
        if request.start is not None or request.end is not None:
            raise ValueError(
                "All-chromosome scope uses each chromosome's full range; clear Start and End."
            )
        if target_format not in _ALL_CHROMOSOME_FORMATS:
            allowed = ", ".join(format_name.upper() for format_name in ("tsv", "csv", "npz"))
            raise ValueError(f"All-chromosome export requires a labelled format: {allowed}.")
    genome_build = _optional_label(request.genome_build, "Genome build", maximum=80)
    units = _optional_label(request.units, "Units", maximum=80)
    biological_role = _optional_label(
        request.biological_role,
        "Biological role",
        maximum=120,
    )
    semantic_note = _optional_label(request.semantic_note, "Semantic note", maximum=500)
    source_url = _public_provenance_url(request.source_url)
    return replace(
        request,
        input_form=input_form,
        chromosome_scope=chromosome_scope,
        target_format=target_format,
        missing_policy=missing_policy,
        decimal_places=decimal_places,
        value_name=value_name,
        bin_size=bin_size,
        aggregation=aggregation,
        track_missing=track_missing,
        fill_value=fill_value,
        interpolation=interpolation,
        smooth_window=smooth_window,
        scaling_mode=scaling_mode,
        normalize="none",
        rescale_min=rescale_min,
        rescale_max=rescale_max,
        rescale_invert=rescale_invert,
        threshold_enabled=threshold_enabled,
        threshold_min=threshold_min,
        threshold_max=threshold_max,
        signal_transform=signal_transform,
        scale=scale,
        offset=offset,
        genome_build=genome_build,
        units=units,
        source_url=source_url,
        biological_role=biological_role,
        semantic_note=semantic_note,
        memory_limit_mb=memory_limit_mb,
        data_filter=data_filter,
    )


def _readonly(values: Any, *, limits: _TrackLimits | None = None) -> np.ndarray:
    array = np.asarray(values)
    max_values = MAX_ARRAY_ELEMENTS if limits is None else limits.max_values
    max_bytes = MAX_DECOMPRESSED_BYTES if limits is None else limits.max_bytes
    if array.size > max_values or array.nbytes > max_bytes:
        raise ValueError(
            f"Prepared data exceed the local limit of {max_values:,} values / "
            f"{max_bytes / 1024**2:.0f} MB."
        )
    if np.issubdtype(array.dtype, np.number) and np.any(np.isinf(array)):
        raise ValueError("Prepared data contain infinite values.")
    array.setflags(write=False)
    return array


def _readonly_coordinates(values: Mapping[str, Any]) -> Mapping[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for key, raw in values.items():
        array = np.asarray(raw)
        array.setflags(write=False)
        result[str(key)] = array
    return MappingProxyType(result)


def _readonly_profiles(
    values: Mapping[str, Any], *, limits: _TrackLimits
) -> Mapping[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    total_values = 0
    total_bytes = 0
    for key, raw in values.items():
        array = np.asarray(raw, dtype=float).reshape(-1)
        if np.any(np.isinf(array)):
            raise ValueError(f"Prepared chromosome {key!r} contains infinite values.")
        array.setflags(write=False)
        result[str(key)] = array
        total_values += int(array.size)
        total_bytes += int(array.nbytes)
    if total_values > limits.max_values or total_bytes > limits.max_bytes:
        raise ValueError(
            f"Prepared chromosome profiles exceed {limits.max_values:,} values / "
            f"{limits.max_bytes / 1024**2:.0f} MB. Increase the bin size."
        )
    return MappingProxyType(result)


def _peak_preview_indices(series: np.ndarray, max_points: int) -> np.ndarray:
    """Keep local maxima and background at their real positions in a small preview."""

    if series.size <= max_points or max_points < 4:
        return np.linspace(0, series.size - 1, min(series.size, max_points), dtype=np.int64)
    edges = np.linspace(1, series.size - 1, (max_points - 2) // 2 + 1, dtype=np.int64)
    selected = {0, series.size - 1}
    for start, stop in zip(edges[:-1], edges[1:]):
        segment = series[start:stop]
        finite = np.flatnonzero(np.isfinite(segment))
        missing = np.flatnonzero(~np.isfinite(segment))
        if finite.size:
            selected.add(int(start + finite[np.argmax(segment[finite])]))
            if not missing.size:
                selected.add(int(start + finite[np.argmin(segment[finite])]))
        if missing.size:
            selected.add(int(start + missing[0]))
    return np.asarray(sorted(selected), dtype=np.int64)


def _preview(
    values: Any,
    *,
    labels: tuple[str, ...] = (),
    coordinates: Mapping[str, np.ndarray] | None = None,
    table: Any = None,
    max_points: int = 500,
    preserve_peaks: bool = False,
) -> DataPreview | None:
    """Build a bounded display series without retaining prepared full arrays."""

    value_label = labels[0] if labels else "value"
    x_values: np.ndarray | None = None
    x_label = "Index"
    if isinstance(table, pd.DataFrame):
        numeric = table.select_dtypes(include=["number"])
        coordinate_names = {
            "bin",
            "end",
            "end_bp",
            "index",
            "position",
            "position_bp",
            "position_kb",
            "start",
            "start_bp",
            "time",
            "time_min",
            "x",
        }
        signal_columns = [
            column
            for column in numeric.columns
            if str(column).strip().casefold() not in coordinate_names
        ]
        if signal_columns:
            value_label = str(signal_columns[0])
            values = numeric[signal_columns[0]].to_numpy(dtype=float)
            for coordinate in ("position_kb", "position_bp", "position", "time_min", "time"):
                if coordinate in numeric and len(numeric[coordinate]) == len(values):
                    x_values = numeric[coordinate].to_numpy(dtype=float)
                    x_label = coordinate.replace("_", " ").title()
                    break
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        return None
    if array.ndim == 0:
        series = array.reshape(1)
    elif array.ndim == 1:
        series = array
    elif array.ndim == 2 and array.shape[0] == 1:
        series = array[0]
    elif array.ndim == 2 and array.shape[1] >= 1:
        series = array[:, 0]
    else:
        return None
    series = np.asarray(series, dtype=float).reshape(-1)
    if not series.size:
        return None

    if coordinates:
        for name in ("position_kb", "position_bp", "time_min", "time"):
            candidate = coordinates.get(name)
            if candidate is not None and np.asarray(candidate).size == series.size:
                x_values = np.asarray(candidate, dtype=float).reshape(-1)
                if name == "position_kb" and x_values.size:
                    # Preview converted profiles from their first bin start.
                    x_values = x_values - x_values[0]
                x_label = name.replace("_", " ").title()
                break
    if x_values is None:
        x_values = np.arange(series.size, dtype=float)

    finite_mask = np.isfinite(series)
    finite = series[finite_mask]
    sample_count = min(max(1, int(max_points)), series.size)
    indices = (
        _peak_preview_indices(series, sample_count)
        if preserve_peaks else np.linspace(0, series.size - 1, sample_count, dtype=np.int64)
    )
    sampled_x = np.array(x_values[indices], dtype=float, copy=True)
    sampled_values = np.array(series[indices], dtype=float, copy=True)
    sampled_x.setflags(write=False)
    sampled_values.setflags(write=False)
    return DataPreview(
        x=sampled_x,
        values=sampled_values,
        x_label=x_label,
        value_label=value_label,
        source_shape=tuple(int(size) for size in array.shape),
        total_count=int(series.size),
        finite_count=int(finite.size),
        missing_count=int(series.size - finite.size),
        minimum=float(finite.min()) if finite.size else None,
        maximum=float(finite.max()) if finite.size else None,
    )


def _infer_form(values: Any) -> str:
    array = np.asarray(values)
    if array.ndim == 0 or array.size == 1:
        return "scalar"
    if array.ndim == 1:
        return "x"
    if array.ndim == 2 and array.shape[0] == 1:
        return "t"
    if array.ndim == 2:
        return "xt"
    return "unknown"


def extract_values(
    inspection: UploadInspection,
    *,
    data_form: str = "auto",
    value_column: str | None = None,
    array_key: str | None = None,
    missing_policy: str = "ignore",
    fill_value: float = 0.0,
    cancel: _CancelChecker | None = None,
) -> np.ndarray:
    """Materialize one bounded numeric selection as a read-only array."""

    _check_cancelled(cancel)
    if inspection.source_format in {"bigwig", "bedgraph"}:
        raise ValueError("Use prepare_data with genomic region settings for track values.")
    form = str(data_form).strip().casefold()
    if form != "auto" and form not in _INPUT_FORMS - {"track", "table"}:
        raise ValueError(f"Unsupported data dependence {data_form!r}.")
    policy = _choice(missing_policy, "missing-value policy", _PLAIN_MISSING)
    if policy == "fill" and not np.isfinite(float(fill_value)):
        raise ValueError("Fill value must be finite.")
    values = materialize_upload(
        replace(inspection, expected_form=form),
        column=value_column,
        array_key=array_key,
        missing_policy="ignore" if policy in {"fill", "min", "max"} else policy,
    )
    if policy in {"fill", "min", "max"}:
        array = np.array(values, dtype=float, copy=True)
        columns = array.reshape(-1, 1) if array.ndim < 2 or array.shape[0] == 1 else array
        for column in range(columns.shape[1]):
            _check_cancelled(cancel)
            columns[:, column] = _fill_missing(
                pd.Series(columns[:, column]), policy, float(fill_value),
            ).to_numpy(dtype=float)
        values = array
    _check_cancelled(cancel)
    return _readonly(values)


def _bedgraph_value_column(
    inspection: UploadInspection,
    selected: str | None,
) -> str | int | None:
    """Translate an inspector label to rtrack's headerless column index."""

    column = selected or inspection.default_column
    if column is not None and column not in inspection.numeric_columns:
        raise ValueError(f"Column {column!r} is not a selectable numeric signal column.")
    if column in inspection.columns:
        return inspection.columns.index(column)
    return column


def inspect_track(
    inspection: UploadInspection,
    cancel: _CancelChecker | None = None,
) -> TrackInspection:
    """Read bounded chromosome metadata from one retained genomic track."""

    _check_cancelled(cancel)
    if inspection.source_format not in {"bigwig", "bedgraph"}:
        raise ValueError("Chromosome inspection requires a BigWig or bedGraph source.")
    with tempfile.TemporaryDirectory(prefix="dnascape-track-") as temporary:
        source_path = _write_source(Path(temporary), inspection, cancel)
        _check_cancelled(cancel)
        if inspection.source_format == "bigwig":
            summary = inspect_bigwig(source_path)
            _check_cancelled(cancel)
            regions = tuple(
                TrackRegion(chromosome=name, start=0, end=size)
                for name, size in summary.chromosomes
            )
            backend = summary.backend
            metadata = summary.file_info
        else:
            raw = inspect_genomic_track(
                source_path,
                source_format="bedgraph",
                value_column=_bedgraph_value_column(inspection, None),
            )
            _check_cancelled(cancel)
            interval_counts = {
                str(row.chrom): int(row.intervals)
                for row in raw["table"][["chrom", "intervals"]].itertuples(index=False)
            }
            regions = tuple(
                TrackRegion(
                    chromosome=str(chromosome),
                    # DNAscape-ready vectors have an implicit origin, so tracks
                    # always expose the full chromosome-aligned range from bp 0.
                    start=0,
                    end=int(raw["chromosomes"][chromosome]),
                    interval_count=interval_counts.get(str(chromosome)),
                )
                for chromosome in raw["chromosome_names"]
            )
            backend = "pandas"
            metadata = ()
    _check_cancelled(cancel)
    if not regions:
        raise ValueError("The genomic track contains no chromosomes.")
    return TrackInspection(
        source_format=inspection.source_format,
        regions=regions,
        value_columns=tuple(inspection.numeric_columns) or ("value",),
        size_bytes=len(inspection.content),
        backend=backend,
        metadata=metadata,
    )


def _repliseq_table(inspection: UploadInspection, *, coordinates_only: bool = False) -> pd.DataFrame:
    if inspection.source_layout != "geo_repliseq":
        raise ValueError("Choose a GEO Repli-seq matrix containing S1–S16.")
    columns = ["chrom", "start", "end"]
    if not coordinates_only:
        columns += [f"S{i}" for i in range(1, 17)]
    return pd.read_csv(io.BytesIO(_source_payload(inspection)), sep="\t", usecols=columns)


def inspect_repliseq(inspection: UploadInspection) -> TrackInspection:
    """Expose chromosomes and native resolution without calculating timing."""

    table = _repliseq_table(inspection, coordinates_only=True)
    widths = table["end"] - table["start"]
    bin_size = int(widths.mode().iloc[0])
    regions = tuple(
        TrackRegion(str(chrom), 0, int(group["end"].max()), len(group))
        for chrom, group in table.groupby("chrom", sort=False)
    )
    return TrackInspection(
        source_format=inspection.source_format, regions=regions,
        value_columns=("replication_timing",), size_bytes=len(inspection.content),
        backend="repliseq", metadata=(("native_bin_size_bp", str(bin_size)),),
    )


def _prepare_repliseq_timing(
    request: DataConversionRequest,
    *,
    progress: Callable[[float], None] | None = None,
    cancel: _CancelChecker | None = None,
) -> PreparedData:
    _check_cancelled(cancel)
    table = _repliseq_table(request.inspection)
    chromosome = str(request.chromosome or table["chrom"].iloc[0])
    selected = table.loc[table["chrom"] == chromosome].sort_values("start")
    if selected.empty:
        raise ValueError(f"The Repli-seq matrix does not contain chromosome {chromosome}.")
    bin_size = int((table["end"] - table["start"]).mode().iloc[0])
    starts = selected["start"].to_numpy(dtype=np.int64)
    ends = selected["end"].to_numpy(dtype=np.int64)
    if np.any(starts % bin_size) or np.any(ends > starts + bin_size) or len(np.unique(starts)) != len(starts):
        raise ValueError("The selected chromosome must contain non-overlapping bins on a regular genomic grid.")
    end = int(ends.max())
    length = math.ceil(end / bin_size)
    limits = _track_limits(request)
    if length > limits.max_chromosome_bins or length * 16 > limits.max_values:
        raise ValueError("The selected Repli-seq chromosome exceeds the conversion memory limit.")
    signals = selected[[f"S{i}" for i in range(1, 17)]].to_numpy(dtype=float)
    timings = replication_timing_from_fractions(
        signals, method=request.repliseq_timing_method or "sigmoid",
        s_phase_duration_min=request.s_phase_duration_min, progress=progress, cancel=cancel,
    )
    if not np.isfinite(timings).any():
        raise ValueError("The selected chromosome contains no usable complete, non-zero S1–S16 bins for timing extraction.")
    values = np.full(length, np.nan)
    values[starts // bin_size] = timings
    invalid_bins = int(np.isnan(values).sum())
    values = _postprocess_profile_values(values, request, cancel)
    coordinates = np.arange(length, dtype=float) * (bin_size / 1000)
    values = apply_data_filter(
        values, request.data_filter, positions=coordinates,
        intervals=(np.column_stack((coordinates, np.minimum(coordinates + bin_size / 1000, end / 1000)))
                   if request.data_filter.enabled and request.data_filter.selection == "profile" else None),
        chromosome=chromosome, cancel=cancel,
    )
    metadata = dict(_profile_metadata(request))
    metadata.update(
        source_layout="geo_repliseq", chromosome=chromosome,
        bin_size_bp=str(bin_size), start_bp="0", end_bp=str(end),
        units="min", biological_role="replication_timing",
        timing_method=request.repliseq_timing_method or "sigmoid",
        timing_statistic="T50" if request.repliseq_timing_method == "sigmoid" else "mean",
        s_phase_duration_min=str(float(request.s_phase_duration_min)),
        fraction_count="16", fraction_spacing="equal", invalid_timing_bins=str(invalid_bins),
    )
    _check_cancelled(cancel)
    return PreparedData(
        values=_readonly(values, limits=limits), source_format=request.inspection.source_format,
        data_form="space", labels=(chromosome,),
        coordinates=_readonly_coordinates({"position_kb": coordinates}),
        metadata=tuple(metadata.items()),
    )


def inspect_data_source(inspection: UploadInspection) -> DataSourceInspection:
    """Return a single auto-detected view of a supported local upload."""

    track = (
        inspect_track(inspection) if inspection.source_format in {"bigwig", "bedgraph"} else None
    )
    selections: list[DataValueArray] = []
    if track is not None:
        selections.extend(
            DataValueArray(column, column, "track", tuple(inspection.shape or ()))
            for column in track.value_columns
        )
    elif inspection.array_keys:
        for key in inspection.array_keys:
            values = extract_values(inspection, array_key=key)
            selections.append(DataValueArray(key, key, _infer_form(values), values.shape))
    elif inspection.numeric_columns:
        row_count = int(inspection.shape[0]) if inspection.shape else 0
        selections.extend(
            DataValueArray(column, column, "x", (row_count,))
            for column in inspection.numeric_columns
        )
        if len(inspection.numeric_columns) > 1:
            selections.insert(
                0,
                DataValueArray(
                    "__matrix__",
                    "All numeric profiles",
                    "xt",
                    (row_count, len(inspection.numeric_columns)),
                ),
            )
    else:
        selections.append(
            DataValueArray("values", "Values", inspection.inferred_form, inspection.shape or ())
        )

    metadata = [
        ("compressed", str(bool(inspection.compressed)).lower()),
        ("size_bytes", str(len(inspection.content))),
    ]
    if inspection.semantic_form:
        metadata.append(("semantic_form", inspection.semantic_form))
    if inspection.source_layout:
        metadata.append(("source_layout", inspection.source_layout))
    if inspection.region_names:
        metadata.append(("regions", str(len(inspection.region_names))))
    if track is not None:
        metadata.append(("backend", track.backend or "unknown"))
        metadata.extend(track.metadata)
    return DataSourceInspection(
        name=inspection.name,
        source_format=inspection.source_format,
        inferred_form="track" if track is not None else inspection.inferred_form,
        shape=tuple(inspection.shape or ()),
        values=tuple(selections),
        columns=tuple(inspection.columns),
        chromosomes=(track.regions if track is not None else
                     inspect_repliseq(inspection).regions if inspection.source_layout == "geo_repliseq" else ()),
        missing_count=int(inspection.missing_count),
        warnings=tuple(inspection.warnings),
        metadata=tuple(metadata),
    )


def _resolved_track_request(
    request: DataConversionRequest,
    cancel: _CancelChecker | None = None,
) -> tuple[DataConversionRequest, TrackInspection, tuple[_TrackSlice, ...]]:
    _check_cancelled(cancel)
    summary = inspect_track(request.inspection, cancel)
    _check_cancelled(cancel)
    regions = {region.chromosome: region for region in summary.regions}
    if request.chromosome_scope == "all":
        slices = tuple(
            _TrackSlice(
                chromosome=region.chromosome,
                start=0,
                end=region.end,
                bins=math.ceil(region.end / request.bin_size),
            )
            for region in summary.regions
        )
        resolved = request
    else:
        chromosome = str(request.chromosome or summary.regions[0].chromosome)
        if chromosome not in regions:
            raise KeyError(f"Chromosome {chromosome!r} was not found in the genomic track.")
        region = regions[chromosome]
        # A coordinate-free DNAscape profile is aligned to chromosome position
        # zero unless the caller explicitly requests a cropped region.
        start = 0 if request.start is None else _integer(request.start, "Start", minimum=0)
        end = region.end if request.end is None else _integer(request.end, "End", minimum=1)
        if end <= start:
            raise ValueError("End must be greater than start for the genomic region.")
        if end > region.end:
            raise ValueError(f"End exceeds {chromosome}'s available bound of {region.end:,} bp.")
        slices = (
            _TrackSlice(
                chromosome=chromosome,
                start=start,
                end=end,
                bins=math.ceil((end - start) / request.bin_size),
            ),
        )
        resolved = replace(request, chromosome=chromosome, start=start, end=end)

    limits = _track_limits(request)
    largest = max(item.bins for item in slices)
    if largest > limits.max_chromosome_bins:
        raise ValueError(
            f"The selected region creates {largest:,} bins; the local limit is "
            f"{limits.max_chromosome_bins:,}. Increase Conversion memory limit (MB) in "
            "Settings, increase the bin size, or choose a smaller region."
        )
    total_bins = sum(item.bins for item in slices)
    if total_bins > limits.max_values:
        raise ValueError(
            f"The selected chromosomes create {total_bins:,} bins; increase the bin size "
            f"to stay below {limits.max_values:,} values, or increase Conversion memory "
            "limit (MB) in Settings."
        )
    padded_cells = largest * len(slices)
    if request.chromosome_scope == "all" and padded_cells > limits.max_values:
        raise ValueError(
            f"The wide chromosome view would contain {padded_cells:,} padded cells; "
            "increase Conversion memory limit (MB) in Settings, increase the bin size, "
            "or export fewer chromosomes."
        )
    if request.chromosome_scope == "all" and request.target_format == "npz":
        max_profiles = MAX_ARCHIVE_MEMBERS - 3
        if len(slices) > max_profiles:
            raise ValueError(
                f"The track contains {len(slices):,} sequences; labelled NPZ export supports "
                f"at most {max_profiles:,}. Export TSV/CSV or reduce the sequence set."
            )
    estimated_bytes = total_bins * (_TRACK_ESTIMATED_BYTES_PER_BIN + (64 if request.data_filter.enabled else 0))
    if request.chromosome_scope == "all":
        estimated_bytes += padded_cells * np.dtype(float).itemsize
    if estimated_bytes > limits.max_bytes:
        raise ValueError(
            f"Preparing {total_bins:,} chromosome bins would require about "
            f"{estimated_bytes / 1024**2:.0f} MB; the conversion memory limit is "
            f"{request.memory_limit_mb:,} MB. Increase Conversion memory limit (MB) "
            f"in Settings to at least {math.ceil(estimated_bytes / 1024**2):,} MB, "
            "increase the bin size, or choose a smaller region."
        )
    _check_cancelled(cancel)
    return resolved, summary, slices


def _transform_signal(series: pd.Series, request: DataConversionRequest) -> pd.Series:
    """Apply legacy direction/affine transforms without changing the sample grid."""

    if request.signal_transform == "invert":
        numeric = series.to_numpy(dtype=float)
        finite_values = numeric[np.isfinite(numeric)]
        if finite_values.size:
            # Reflect around the finite range midpoint. This reverses relative
            # signal direction while preserving the original units and range.
            series = float(finite_values.min() + finite_values.max()) - series
    # Older requests could combine an affine transform with normalization.
    # Preserve that behavior only when legacy fields survived normalization;
    # explicit scaling modes reset ``offset`` and use ``scale`` below.
    if request.scaling_mode != "multiplier" and (
        request.scale != 1.0 or request.offset != 0.0
    ):
        series = series * request.scale + request.offset
    return series


def _fill_missing(series: pd.Series, policy: str, fill_value: float = 0.0) -> pd.Series:
    """Apply a row-preserving policy to one numeric signal, never coordinates."""

    if not series.isna().any():
        return series
    if policy == "error":
        raise ValueError(
            "The selected data contain missing values. Choose another missing-value "
            "option or correct the source file."
        )
    if policy in {"min", "max"}:
        finite_values = series[np.isfinite(series)]
        if finite_values.empty:
            name = "minimum" if policy == "min" else "maximum"
            raise ValueError(
                f"Cannot fill empty values with the {name}: no finite values remain "
                "after thresholding and smoothing."
            )
        fill_value = float(finite_values.min() if policy == "min" else finite_values.max())
    elif policy == "zero":
        fill_value = 0.0
    elif policy != "fill":
        return series
    return series.fillna(fill_value)


def _polish_series(series: pd.Series, request: DataConversionRequest) -> pd.Series:
    """Threshold, smooth and scale one ordered profile without dropping samples."""

    series = pd.Series(series, index=series.index, dtype=float, copy=True)
    if request.threshold_enabled:
        outside = pd.Series(False, index=series.index, dtype=bool)
        if request.threshold_min is not None:
            outside |= series < request.threshold_min
        if request.threshold_max is not None:
            outside |= series > request.threshold_max
        series = series.mask(outside)

    policy = request.track_missing if request.input_form == "track" else request.missing_policy
    if policy == "error":
        _fill_missing(series, policy)

    # Rolling means ignore missing neighbours, but an explicitly missing sample
    # must remain missing unless the caller selected interpolation/filling. This
    # also ensures threshold filtering never shifts or silently restores a bin.
    missing = series.isna()
    if request.smooth_window > 1:
        smoothed = series.rolling(
            window=request.smooth_window,
            center=True,
            min_periods=1,
        ).mean()
        series = pd.Series(smoothed, index=series.index, dtype=float)
        series = series.mask(missing)

    if policy in {"min", "max"}:
        series = _fill_missing(series, policy)

    series = _scale_series(series, request)
    if policy in {"zero", "fill"}:
        series = _fill_missing(series, policy, request.fill_value)
    return series


def _scale_series(series: pd.Series, request: DataConversionRequest) -> pd.Series:
    """Scale values including minimum/maximum fills, before zero/custom fills."""

    numeric = series.to_numpy(dtype=float)
    finite_values = numeric[np.isfinite(numeric)]
    if request.scaling_mode == "none" or not finite_values.size:
        return series
    if request.scaling_mode == "multiplier":
        return series * request.scale + request.offset
    if request.scaling_mode == "center":
        return series - float(finite_values.mean())
    if request.scaling_mode == "zscore":
        divisor = float(finite_values.std(ddof=0))
        return series * 0.0 if divisor == 0 else (series - float(finite_values.mean())) / divisor
    if request.scaling_mode == "maxabs":
        divisor = float(np.max(np.abs(finite_values)))
        return series * 0.0 if divisor == 0 else series / divisor
    if request.scaling_mode == "interval":
        source_minimum = float(finite_values.min())
        source_maximum = float(finite_values.max())
        target_minimum = float(request.rescale_min)
        target_maximum = float(request.rescale_max)
        source_span = source_maximum - source_minimum
        if source_span == 0:
            return series * 0.0 + (target_minimum + target_maximum) / 2.0
        fraction = (series - source_minimum) / source_span
        if request.rescale_invert:
            fraction = 1.0 - fraction
        return target_minimum + fraction * (target_maximum - target_minimum)
    raise AssertionError(f"Unhandled value scaling {request.scaling_mode!r}.")


def _postprocess_profile_values(
    values: Any,
    request: DataConversionRequest,
    cancel: _CancelChecker | None = None,
) -> np.ndarray:
    """Polish an x/t/xt DNAscape profile while preserving its exact shape."""

    _check_cancelled(cancel)
    if request.missing_policy in {"ignore", "interpolate"} and not _polishing_active(
        smooth_window=request.smooth_window,
        scaling_mode=str(request.scaling_mode),
        threshold_enabled=request.threshold_enabled,
        signal_transform=request.signal_transform,
        scale=request.scale,
        offset=request.offset,
    ):
        _check_cancelled(cancel)
        return _readonly(values)
    array = np.array(values, dtype=float, copy=True)
    _check_cancelled(cancel)
    if request.input_form in {"scalar", "x", "t", "table"}:
        series = pd.Series(array.reshape(-1), dtype=float)
        result = _polish_series(_transform_signal(series, request), request).to_numpy(dtype=float)
        _check_cancelled(cancel)
        return _readonly(result.reshape(array.shape))
    if request.input_form == "xt":
        if array.ndim != 2:
            raise ValueError("Space-time polishing requires a two-dimensional profile.")
        result = np.empty_like(array, dtype=float)
        # Each column is one spatial profile/time slice, matching DNAscape's
        # documented (space, time) matrix orientation.
        for column in range(array.shape[1]):
            _check_cancelled(cancel)
            series = pd.Series(array[:, column], dtype=float)
            result[:, column] = _polish_series(
                _transform_signal(series, request),
                request,
            ).to_numpy(dtype=float)
        _check_cancelled(cancel)
        return _readonly(result)
    _check_cancelled(cancel)
    return _readonly(array)


def _postprocess_track_table(
    table: pd.DataFrame,
    request: DataConversionRequest,
    cancel: _CancelChecker | None = None,
) -> pd.DataFrame:
    _check_cancelled(cancel)
    result = table.copy()
    series = pd.Series(
        pd.to_numeric(result[request.value_name], errors="coerce"),
        index=result.index,
        dtype=float,
    )
    series = _transform_signal(series, request)
    if request.track_missing == "error":
        _fill_missing(series, "error")
    if request.interpolation == "linear":
        series = series.interpolate(method="linear", limit_direction="both")
    elif request.interpolation == "ffill":
        series = series.ffill()
    elif request.interpolation == "bfill":
        series = series.bfill()
    _check_cancelled(cancel)
    result[request.value_name] = _polish_series(series, request)
    _check_cancelled(cancel)
    return result


def _provenance_metadata(request: DataConversionRequest) -> tuple[tuple[str, str], ...]:
    values = (
        ("genome_build", request.genome_build),
        ("units", request.units),
        ("source_url", request.source_url),
        ("biological_role", request.biological_role),
        ("semantic_note", request.semantic_note),
    )
    return tuple((name, value) for name, value in values if value is not None)


def _polishing_metadata(request: DataConversionRequest) -> tuple[tuple[str, str], ...]:
    metadata: list[tuple[str, str]] = [
        ("smooth_window", str(request.smooth_window)),
        ("value_scaling", str(request.scaling_mode)),
        ("threshold_enabled", str(request.threshold_enabled).lower()),
    ]
    if request.scaling_mode == "multiplier":
        metadata.append(("scale_multiplier", str(request.scale)))
    if request.signal_transform != "keep":
        metadata.append(("legacy_signal_transform", request.signal_transform))
    if request.scaling_mode != "multiplier" and request.scale != 1.0:
        metadata.append(("legacy_scale_multiplier", str(request.scale)))
    if request.offset != 0.0:
        metadata.append(("legacy_offset", str(request.offset)))
    if request.threshold_enabled:
        metadata.extend(
            (
                ("threshold_min", "none" if request.threshold_min is None else str(request.threshold_min)),
                ("threshold_max", "none" if request.threshold_max is None else str(request.threshold_max)),
                ("threshold_action", "mask"),
            )
        )
    if request.scaling_mode == "interval":
        metadata.extend(
            (
                ("target_minimum", str(request.rescale_min)),
                ("target_maximum", str(request.rescale_max)),
                ("interval_inverted", str(request.rescale_invert).lower()),
            )
        )
    return tuple(metadata)


def _profile_metadata(request: DataConversionRequest) -> tuple[tuple[str, str], ...]:
    return (
        (
            ("source_name", request.inspection.name),
            ("source_format", request.inspection.source_format),
            ("missing", request.missing_policy),
        )
        + _polishing_metadata(request)
        + _provenance_metadata(request)
        + filter_metadata(request.data_filter)
    )


def _track_metadata(
    request: DataConversionRequest,
    slices: tuple[_TrackSlice, ...],
) -> tuple[tuple[str, str], ...]:
    scope_metadata: tuple[tuple[str, str], ...]
    if request.chromosome_scope == "single":
        item = slices[0]
        scope_metadata = (
            ("chromosome", item.chromosome),
            ("start", str(item.start)),
            ("end", str(item.end)),
        )
    else:
        scope_metadata = (
            ("chromosome_count", str(len(slices))),
            ("chromosomes", ",".join(item.chromosome for item in slices)),
            ("bounds", ",".join(f"{item.chromosome}:0-{item.end}" for item in slices)),
        )
    return (
        (
            ("source_name", request.inspection.name),
            ("source_format", request.inspection.source_format),
            ("chromosome_scope", request.chromosome_scope),
            ("bin_size", str(request.bin_size)),
            ("aggregation", request.aggregation),
            ("missing", request.track_missing),
            ("interpolation", request.interpolation),
        )
        + _polishing_metadata(request)
        + scope_metadata
        + _provenance_metadata(request)
        + filter_metadata(request.data_filter)
    )


def _slice_coordinates(item: _TrackSlice, bin_size: int) -> np.ndarray:
    starts = np.arange(item.start, item.end, bin_size, dtype=np.int64)
    ends = np.minimum(starts + bin_size, item.end).astype(np.int64)
    return (starts + ends) / 2_000.0


def _processed_profile(
    raw_values: Any,
    item: _TrackSlice,
    request: DataConversionRequest,
    cancel: _CancelChecker | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    _check_cancelled(cancel)
    values = np.asarray(raw_values, dtype=float).reshape(-1)
    if values.size != item.bins:
        raise RuntimeError(
            f"Track preparation returned {values.size:,} bins for {item.chromosome}, "
            f"but {item.bins:,} were expected."
        )
    table = pd.DataFrame({request.value_name: values})
    table = _postprocess_track_table(
        table,
        replace(
            request,
            chromosome=item.chromosome,
            chromosome_scope="single",
            start=item.start,
            end=item.end,
        ),
        cancel,
    )
    _check_cancelled(cancel)
    coordinates = _slice_coordinates(item, request.bin_size)
    intervals = None
    if request.data_filter.enabled and request.data_filter.selection == "profile":
        starts = np.arange(item.start, item.end, request.bin_size, dtype=np.int64)
        intervals = np.column_stack((starts, np.minimum(starts + request.bin_size, item.end))) / 1000.
    filtered = apply_data_filter(
        np.asarray(table[request.value_name], dtype=float), request.data_filter,
        positions=coordinates, intervals=intervals,
        chromosome=item.chromosome, cancel=cancel,
    )
    return (
        filtered,
        coordinates,
    )


def _combine_track_profiles(
    request: DataConversionRequest,
    slices: tuple[_TrackSlice, ...],
    profile_values: Mapping[str, Any],
    profile_coordinates: Mapping[str, Any],
    cancel: _CancelChecker | None = None,
) -> PreparedData:
    _check_cancelled(cancel)
    labels = tuple(item.chromosome for item in slices)
    limits = _track_limits(request)
    profiles = _readonly_profiles(profile_values, limits=limits)
    profile_x = _readonly_profiles(profile_coordinates, limits=limits)
    if request.chromosome_scope == "single":
        values = profiles[labels[0]]
        data_form = "space"
    else:
        rows = max(profile.size for profile in profiles.values())
        matrix = np.full((rows, len(labels)), np.nan, dtype=float)
        for column, label in enumerate(labels):
            _check_cancelled(cancel)
            profile = profiles[label]
            matrix[: profile.size, column] = profile
        values = _readonly(matrix, limits=limits)
        data_form = "table"
    first_label = labels[0]
    result = PreparedData(
        values=values,
        source_format=request.inspection.source_format,
        data_form=data_form,
        labels=labels,
        coordinates=_readonly_coordinates({"position_kb": profile_x[first_label]}),
        profile_values=profiles,
        profile_coordinates=profile_x,
        metadata=_track_metadata(request, slices),
    )
    _check_cancelled(cancel)
    return result


def _prepare_bigwig(
    request: DataConversionRequest,
    slices: tuple[_TrackSlice, ...],
    *,
    progress: Callable[[float], None] | None = None,
    cancel: _CancelChecker | None = None,
) -> PreparedData:
    _check_cancelled(cancel)
    profiles: dict[str, np.ndarray] = {}
    coordinates: dict[str, np.ndarray] = {}
    total_bins = sum(item.bins for item in slices)
    completed_bins = 0
    with tempfile.TemporaryDirectory(prefix="dnascape-bigwig-") as temporary:
        source_path = _write_source(Path(temporary), request.inspection, cancel)
        for item in slices:
            _check_cancelled(cancel)

            def report_fraction(
                fraction: float,
                *,
                done: int = completed_bins,
                count: int = item.bins,
            ) -> None:
                _check_cancelled(cancel)
                if progress is not None:
                    progress((done + count * fraction) / total_bins)

            raw_values = read_bigwig_bins(
                source_path,
                chromosome=item.chromosome,
                start=item.start,
                end=item.end,
                bin_size=request.bin_size,
                aggregation=request.aggregation,
                # A progress hook is also installed for cancellation-only runs,
                # allowing chunked BigWig backends to stop between reads.
                progress=report_fraction if progress is not None or cancel is not None else None,
            )
            _check_cancelled(cancel)
            profiles[item.chromosome], coordinates[item.chromosome] = _processed_profile(
                raw_values,
                item,
                request,
                cancel,
            )
            completed_bins += item.bins
            if progress is not None:
                progress(completed_bins / total_bins)
    _check_cancelled(cancel)
    return _combine_track_profiles(request, slices, profiles, coordinates, cancel)


def _prepare_bedgraph(
    request: DataConversionRequest,
    slices: tuple[_TrackSlice, ...],
    *,
    progress: Callable[[float], None] | None = None,
    cancel: _CancelChecker | None = None,
) -> PreparedData:
    _check_cancelled(cancel)
    profiles: dict[str, np.ndarray] = {}
    coordinates: dict[str, np.ndarray] = {}
    with tempfile.TemporaryDirectory(prefix="dnascape-bedgraph-") as temporary:
        source_path = _write_source(Path(temporary), request.inspection, cancel)
        _check_cancelled(cancel)
        raw = rtrack(
            source_path,
            source_format="bedgraph",
            chromosome=("all" if request.chromosome_scope == "all" else slices[0].chromosome),
            start=(0 if request.chromosome_scope == "all" else slices[0].start),
            end=(None if request.chromosome_scope == "all" else slices[0].end),
            bin_size=request.bin_size,
            aggregation=request.aggregation,
            value_column=_bedgraph_value_column(request.inspection, request.value_column),
            missing="nan",
            interpolation="none",
            smooth_window=1,
            normalize="none",
            data_form="space",
            as_table=True,
            return_metadata=True,
            value_name=request.value_name,
        )
    _check_cancelled(cancel)
    if not isinstance(raw, Mapping):
        raise RuntimeError("Track preparation stopped before returning metadata.")
    table = raw.get("table")
    if not isinstance(table, pd.DataFrame):
        raise RuntimeError("Track preparation did not return a coordinate table.")
    if progress is not None:
        progress(0.6)
    for index, item in enumerate(slices):
        _check_cancelled(cancel)
        selected = table.loc[table["chrom"].astype(str) == item.chromosome]
        profiles[item.chromosome], coordinates[item.chromosome] = _processed_profile(
            selected[request.value_name].to_numpy(dtype=float),
            item,
            request,
            cancel,
        )
        if progress is not None:
            progress(0.6 + 0.4 * (index + 1) / len(slices))
    _check_cancelled(cancel)
    return _combine_track_profiles(request, slices, profiles, coordinates, cancel)


def _prepare_track(
    request: DataConversionRequest,
    slices: tuple[_TrackSlice, ...],
    progress: Callable[[float], None] | None = None,
    cancel: _CancelChecker | None = None,
) -> PreparedData:
    return (
        _prepare_bigwig(request, slices, progress=progress, cancel=cancel)
        if request.inspection.source_format == "bigwig"
        else _prepare_bedgraph(request, slices, progress=progress, cancel=cancel)
    )


def prepare_data(request: DataConversionRequest) -> PreparedData:
    """Prepare values for preview/mapping without writing a download file."""

    request = _normalized_request(request)
    if request.repliseq_timing_method is not None:
        return _prepare_repliseq_timing(request)
    if request.input_form == "track":
        request, _summary, slices = _resolved_track_request(request)
        return _prepare_track(request, slices)

    if request.input_form == "table" and request.value_column is None:
        raise ValueError("Choose a numeric value column before preparing a full table preview.")
    values = extract_values(
        request.inspection,
        data_form="x" if request.input_form == "table" else request.input_form,
        value_column=request.value_column,
        array_key=request.array_key,
        missing_policy="interpolate" if request.missing_policy == "interpolate" else "ignore",
    )
    values = _postprocess_profile_values(values, request)
    values = _filter_plain_values(values, request)
    labels: tuple[str, ...] = ()
    if request.value_column:
        labels = (request.value_column,)
    elif request.array_key:
        labels = (request.array_key,)
    elif request.input_form == "xt" and len(request.inspection.numeric_columns) == values.shape[1]:
        labels = tuple(request.inspection.numeric_columns)
    return PreparedData(
        values=values,
        source_format=request.inspection.source_format,
        data_form=_FORM_TO_RDATA[request.input_form],
        labels=labels,
        metadata=(
            _profile_metadata(request)
            if request.input_form in {"x", "t", "xt"}
            else _provenance_metadata(request)
        ),
    )


def _source_labels(request: DataConversionRequest, values: np.ndarray) -> list[str] | None:
    inspection = request.inspection
    if request.value_column:
        return [request.value_column]
    if request.input_form == "xt":
        columns = list(inspection.numeric_columns)
        if values.ndim == 2 and len(columns) == values.shape[1]:
            return columns
        if values.ndim == 2 and inspection.columns and len(inspection.columns) == values.shape[1]:
            return list(inspection.columns)
    if request.input_form == "t":
        columns = list(inspection.numeric_columns)
        width = values.shape[1] if values.ndim == 2 else values.size
        if len(columns) == width:
            return columns
    if request.input_form == "x" and len(inspection.numeric_columns) == 1:
        return [inspection.numeric_columns[0]]
    if request.array_key and _FORM_TO_RDATA[request.input_form] == "space":
        return [request.array_key]
    return None


def _plain_source(
    request: DataConversionRequest,
    cancel: _CancelChecker | None = None,
) -> tuple[Any, list[str] | None]:
    _check_cancelled(cancel)
    values = extract_values(
        request.inspection,
        data_form=request.input_form,
        value_column=request.value_column,
        array_key=request.array_key,
        missing_policy="interpolate" if request.missing_policy == "interpolate" else "ignore",
        cancel=cancel,
    )
    labels = _source_labels(request, values)
    result = _postprocess_profile_values(values, request, cancel)
    result = _filter_plain_values(result, request, cancel)
    _check_cancelled(cancel)
    return result, labels


def _filter_plain_values(
    values: np.ndarray, request: DataConversionRequest, cancel: _CancelChecker | None = None,
) -> np.ndarray:
    if not request.data_filter.enabled:
        return values
    length = values.size if values.ndim < 2 or values.shape[0] == 1 else values.shape[0]
    positions, intervals = (profile_coordinates(request.inspection, length)
                            if request.data_filter.selection == "profile" or request.data_filter.action == "interpolate"
                            else (None, None))
    chromosome = request.chromosome
    if chromosome is None and request.data_filter.profile is not None and request.data_filter.profile.chromosomes is not None:
        chromosome = profile_chromosome(request.inspection)
    return _readonly(apply_data_filter(
        values, request.data_filter, positions=positions, intervals=intervals,
        chromosome=chromosome, cancel=cancel,
    ))


def _table_source(
    directory: Path,
    request: DataConversionRequest,
    cancel: _CancelChecker | None = None,
) -> tuple[Any, str]:
    """Fill signal columns while retaining table labels and genomic coordinates."""

    source = _write_source(directory, request.inspection, cancel)
    if request.missing_policy == "ignore" or not request.inspection.missing_count:
        return source, request.inspection.source_format
    table = rload(source, source_format=request.inspection.source_format, data_form="table", as_table=True)
    _check_cancelled(cancel)
    for column in request.inspection.numeric_columns:
        _check_cancelled(cancel)
        series = pd.to_numeric(table[column], errors="raise")
        table[column] = (
            series.interpolate(limit_direction="both") if request.missing_policy == "interpolate"
            else _fill_missing(series, request.missing_policy, request.fill_value)
        )
    return table, "auto"


def _read_output(
    path: Path,
    cancel: _CancelChecker | None = None,
    *,
    max_bytes: int | None = None,
) -> bytes:
    _check_cancelled(cancel)
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise RuntimeError("rdata did not create the requested output file.") from exc
    if size <= 0:
        raise RuntimeError("rdata created an empty output file.")
    if max_bytes is None:
        max_bytes = _MAX_CONVERTED_BYTES
    if size > max_bytes:
        raise ValueError(
            f"The converted file exceeds the {max_bytes / 1024**2:g} MB download limit."
        )
    chunks: list[bytes] = []
    with path.open("rb") as source:
        while True:
            _check_cancelled(cancel)
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
    _check_cancelled(cancel)
    return b"".join(chunks)


def _progress_callback(
    progress: Callable[[dict[str, Any]], None] | bool,
) -> Callable[[dict[str, Any]], None] | None:
    if progress in (None, False):
        return None
    if callable(progress):
        return progress
    if progress is True:
        return lambda event: print(
            f"rdata {event['route']}: {event['stage']} {event['status']} "
            f"({int(100 * event['progress'])}%)"
        )
    raise TypeError("progress must be False, True, or a callable.")


def _rdata_progress(
    progress: Callable[[dict[str, Any]], None] | bool,
    cancel: _CancelChecker | None,
) -> Callable[[dict[str, Any]], None] | bool:
    """Combine public progress reporting with cancellation at rdata stages."""

    callback = _progress_callback(progress)
    if callback is None and cancel is None:
        return False

    def report(event: dict[str, Any]) -> None:
        _check_cancelled(cancel)
        if callback is not None:
            callback(event)
        _check_cancelled(cancel)

    return report


def _emit_track_progress(
    callback: Callable[[dict[str, Any]], None] | None,
    request: DataConversionRequest,
    stage: str,
    status: str,
    value: float,
    output_path: Path | None = None,
) -> None:
    if callback is None:
        return
    event: dict[str, Any] = {
        "route": f"{request.inspection.source_format}->{request.target_format}",
        "source_format": request.inspection.source_format,
        "target_format": request.target_format,
        "stage": stage,
        "status": status,
        "progress": value,
        "elapsed": 0.0,
        "eta": None,
    }
    if output_path is not None:
        event["output_path"] = str(output_path)
    callback(event)


def _write_all_chromosome_npz(
    prepared: PreparedData,
    output_path: Path,
    cancel: _CancelChecker | None = None,
) -> None:
    _check_cancelled(cancel)
    reserved = {"data_form", "semantic_form", "region_names"}
    archive: dict[str, np.ndarray] = {
        "data_form": np.asarray("space"),
        "semantic_form": np.asarray("chromosome_profiles"),
        "region_names": np.asarray(prepared.labels),
    }
    for label in prepared.labels:
        _check_cancelled(cancel)
        if (
            not label
            or label in reserved
            or not all(
                character.isascii() and (character.isalnum() or character in "._-")
                for character in label
            )
            or label.startswith(".")
        ):
            raise ValueError(
                f"Chromosome name {label!r} cannot be used safely as an NPZ array key. "
                "Export TSV or CSV instead."
            )
        archive[label] = prepared.profile_values[label]
    _check_cancelled(cancel)
    stream = io.BytesIO()
    savez: Any = np.savez
    savez(stream, **archive)
    payload = stream.getvalue()
    if output_path.suffix.casefold() == ".gz":
        payload = gzip.compress(payload)
    _check_cancelled(cancel)
    output_path.write_bytes(payload)
    _check_cancelled(cancel)


def _convert_track(
    request: DataConversionRequest,
    slices: tuple[_TrackSlice, ...],
    output_path: Path,
    progress: Callable[[dict[str, Any]], None] | bool,
    cancel: _CancelChecker | None = None,
) -> tuple[PreparedData, bytes]:
    callback = _progress_callback(progress)
    _check_cancelled(cancel)
    _emit_track_progress(callback, request, "load", "start", 0.0)
    _check_cancelled(cancel)
    prepared = _prepare_track(
        request,
        slices,
        progress=lambda fraction: _emit_track_progress(
            callback,
            request,
            "load",
            "running",
            0.4 * fraction,
        ),
        cancel=cancel,
    )
    _check_cancelled(cancel)
    _emit_track_progress(callback, request, "load", "done", 0.4)
    _check_cancelled(cancel)
    _emit_track_progress(callback, request, "prepare", "start", 0.4)
    source: Any
    form: str
    labels: list[str] | None
    if request.target_format in {"csv", "tsv"}:
        source = pd.DataFrame(prepared.values, columns=prepared.labels)
        form, labels = "table", None
    else:
        source, form, labels = prepared.values, "space", list(prepared.labels)
    _check_cancelled(cancel)
    _emit_track_progress(callback, request, "prepare", "done", 0.5, output_path)
    _check_cancelled(cancel)
    _emit_track_progress(callback, request, "write", "start", 0.5, output_path)
    _check_cancelled(cancel)
    if request.chromosome_scope == "all" and request.target_format == "npz":
        _write_all_chromosome_npz(prepared, output_path, cancel)
    else:
        rsave(
            source,
            output_path,
            target_format=request.target_format,
            data_form=form,
            labels=labels,
            decimal_places=request.decimal_places,
            value_name=request.value_name,
        )
    _check_cancelled(cancel)
    _emit_track_progress(callback, request, "write", "done", 1.0, output_path)
    _check_cancelled(cancel)
    _emit_track_progress(callback, request, "complete", "done", 1.0, output_path)
    _check_cancelled(cancel)
    return prepared, _read_output(
        output_path,
        cancel,
        max_bytes=max(_MAX_CONVERTED_BYTES, _track_limits(request).max_bytes),
    )


def convert_data(
    request: DataConversionRequest,
    progress: Callable[[dict[str, Any]], None] | bool = False,
    cancel: _CancelChecker | None = None,
) -> DataConversionResult:
    """Run one bounded local conversion and return a downloadable artifact.

    ``cancel`` is an optional worker-local checker. It should return while the
    conversion may continue and raise ``concurrent.futures.CancelledError``
    when a caller requests a stop. Existing two-argument calls remain valid.
    """

    _check_cancelled(cancel)
    request = _normalized_request(request)
    _check_cancelled(cancel)
    if len(request.inspection.content) > MAX_DECOMPRESSED_BYTES:
        raise ValueError("The uploaded source exceeds the local conversion limit.")
    track_summary: TrackInspection | None = None
    track_slices: tuple[_TrackSlice, ...] = ()
    if request.input_form == "track":
        request, track_summary, track_slices = _resolved_track_request(request, cancel)
    _check_cancelled(cancel)

    filename = safe_export_filename(
        request.output_name,
        request.target_format,
        default_stem=RDATA_FILE_PREFIX,
        gzip_output=request.gzip_output,
    )
    with tempfile.TemporaryDirectory(prefix="dnascape-data-") as temporary:
        _check_cancelled(cancel)
        directory = Path(temporary)
        output_path = directory / filename
        labels: list[str] | None = None

        if request.input_form == "track":
            prepared, artifact_bytes = _convert_track(
                request,
                track_slices,
                output_path,
                progress,
                cancel,
            )
            _check_cancelled(cancel)
            data_form = prepared.data_form
            shape = prepared.shape
            result_labels = prepared.labels
            source_format = prepared.source_format
            result_metadata = prepared.metadata
            preview_label = prepared.labels[0]
            preview = _preview(
                prepared.profile_values[preview_label],
                labels=(request.value_name,),
                coordinates={"position_kb": prepared.profile_coordinates[preview_label]},
                preserve_peaks=(
                    request.inspection.track_kind in {"peaks", "intervals"}
                    or (
                        request.inspection.source_format == "bedgraph"
                        and request.biological_role == "firing_rate"
                    )
                ),
            )
            _check_cancelled(cancel)
        else:
            repliseq_prepared = None
            if request.repliseq_timing_method is not None:
                callback = _progress_callback(progress)
                repliseq_prepared = _prepare_repliseq_timing(
                    request, cancel=cancel,
                    progress=lambda fraction: _emit_track_progress(callback, request, "timing", "running", 0.8 * fraction),
                )
                source = repliseq_prepared.values
                labels = list(repliseq_prepared.labels)
                source_format = "auto"
                if callback is not None:
                    progress = lambda event: callback({**event, "progress": 0.8 + 0.2 * event["progress"]})
            elif request.input_form == "table":
                source, source_format = _table_source(directory, request, cancel)
            else:
                source, labels = _plain_source(request, cancel)
                source_format = "auto"
            _check_cancelled(cancel)

            kwargs: dict[str, Any] = {
                "source_format": source_format,
                "data_form": _FORM_TO_RDATA[request.input_form],
                "labels": labels,
                "decimal_places": request.decimal_places,
                "value_name": request.value_name,
                "return_data": True,
                "progress": _rdata_progress(progress, cancel),
            }
            _check_cancelled(cancel)
            raw_result = rdata(
                source,
                target=request.target_format,
                output_path=output_path,
                **kwargs,
            )
            _check_cancelled(cancel)
            if not isinstance(raw_result, Mapping):
                raise RuntimeError("rdata stopped before returning conversion metadata.")
            artifact_bytes = _read_output(output_path, cancel)
            _check_cancelled(cancel)
            raw_values = _readonly(np.asarray(raw_result["values"]))
            data_form = str(raw_result["data_form"])
            shape = tuple(int(value) for value in raw_values.shape)
            result_labels = tuple(str(value) for value in (raw_result.get("labels") or ()))
            source_format = str(raw_result.get("source_format") or request.inspection.source_format)
            result_metadata = (
                repliseq_prepared.metadata if repliseq_prepared is not None else _profile_metadata(request)
                if request.input_form in {"x", "t", "xt"}
                else _provenance_metadata(request)
            )
            preview = _preview(
                raw_result["values"],
                labels=(request.value_name,) if repliseq_prepared is not None else result_labels,
                table=None if repliseq_prepared is not None else raw_result.get("table"),
                coordinates=repliseq_prepared.coordinates if repliseq_prepared is not None else None,
            )
            _check_cancelled(cancel)

    media_type = "application/gzip" if request.gzip_output else _MEDIA_TYPES[request.target_format]
    if track_summary is not None and track_summary.backend:
        result_metadata += (("backend", track_summary.backend),)
    result = DataConversionResult(
        artifact=ExportArtifact(data=artifact_bytes, filename=filename, media_type=media_type),
        source_format=source_format,
        target_format=request.target_format,
        data_form=data_form,
        shape=shape,
        labels=result_labels,
        metadata=result_metadata,
        preview=preview,
    )
    _check_cancelled(cancel)
    return result


__all__ = [
    "DATA_EXPORT_FORMATS",
    "DEFAULT_TARGET_FORMAT",
    "DataConversionRequest",
    "DataConversionResult",
    "DataFilterSettings",
    "DataPreview",
    "DataSourceInspection",
    "DataValueArray",
    "PreparedData",
    "TrackInspection",
    "TrackRegion",
    "convert_data",
    "extract_values",
    "inspect_data_source",
    "inspect_track",
    "inspect_repliseq",
    "prepare_data",
    "transcode_conversion_artifact",
]
