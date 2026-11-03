"""Interactive data conversion through :func:`dnascape.rdata`."""

from __future__ import annotations

import asyncio
import csv
import gzip
import io
import math
import time
import zipfile
from array import array
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import CancelledError
from dataclasses import dataclass, field, replace
from pathlib import PurePath
from threading import Event, Lock
from types import SimpleNamespace
from typing import Any

import numpy as np
from nicegui import run, ui

from app.artifact_archives import (
    DEFAULT_BATCH_ARCHIVE_SOURCE_BYTES,
    build_artifact_archive,
)
from app.data_conversion_service import (
    DATA_EXPORT_FORMATS,
    DataConversionRequest,
    DataConversionResult,
    DataFilterSettings,
    TrackInspection,
    convert_data,
    inspect_track,
    inspect_repliseq,
    transcode_conversion_artifact,
)
from app.export_settings import get_export_settings, preview_point_limit
from app.data_filter_controls import DataFilterControls
from app.data_transform_controls import DataTransformControls
from app.icons import RESET_ICON
from app.field_hints import field_hint
from app.mapping_catalog import RDATA_UPLOAD_ACCEPT
from app.notifications import notification_message, notify
from app.statistics_exports import HistogramStatisticsExport, StatisticsExportPreference
from app.raw_data_transfer import RawInputTarget, open_raw_artifact
from app.mapping_uploads import (
    UploadInspection,
    inspect_upload,
    inspect_upload_event,
)
from app.page_activity import current_page_activity
from app.pages.common import (
    DATA_ICON,
    action_dock,
    workflow_action_button,
    workflow_action_icon,
    data_upload,
    expansion,
    page_intro,
    panel,
    panel_title,
    raw_output_file_row,
    append_raw_uploads,
    set_data_upload_locked,
)
from app.preview_plotting import (
    HistogramBinSynchronizer,
    PreviewSettingsControls,
    axis_limits_from_controls,
    bind_preview_settings_updates,
    build_preview_plot_options,
    histogram_x_limits_from_controls,
    nice_linear_limits,
    _logarithmic_data_limits,
    plotting_settings_ready,
    preview_settings_controls,
    preview_settings_markers,
    render_preview_chart,
    sync_preview_settings_visibility,
)
from dnascape.workflows.preview import DetailedPreviewData, buffered_preview_limits, retain_curve_overview
from dnascape.workflows.conversion import filter_positions_artifact, convert_combined_data
from dnascape.workflows.filtering import FilterSelection
from dnascape.workflows.profile_grid import spatial_profile_grid, spatial_profile_rows
from app.remote_sources import RemoteSourceError, fetch_remote_source
from app.result_naming import (
    RDATA_FILE_PREFIX,
    compact_export_stem,
    output_file_token,
)
from app.session_cache import cache_artifact, cache_dataset, cache_upload, current_session_cache
from app.simulation_exports import (
    ExportArtifact,
    build_chart_capture_export,
    safe_export_filename,
)

_NATURAL = "__natural__"
_MATRIX = "__matrix__"
_TABLE = "__table__"
_REPLISEQ_TIMING = "__repliseq_timing__"
_COLUMN_PREFIX = "column:"
_ARRAY_PREFIX = "array:"
_FORM_LABELS = {
    "scalar": "Constant",
    "x": "Space",
    "t": "Time",
    "xt": "Space-time",
    "table": "Table",
    "track": "Space",
    "space": "Space",
    "time": "Time",
    "spacetime": "Space-time",
}
_GENERATED_FILE_TYPES = tuple(label for _key, label in DATA_EXPORT_FORMATS)
_SOURCE_FORMATS = {
    "bigwig": "BigWig",
    "bedgraph": "bedGraph",
    "tsv": "TSV",
    "csv": "CSV",
    "txt": "TXT",
    "npy": "NPY",
    "npv": "NPV",
    "npz": "NPZ",
}
_TRACK_AGGREGATIONS = {
    "mean": "Mean",
    "sum": "Sum",
    "min": "Minimum",
    "max": "Maximum",
    "coverage": "Coverage",
    "std": "Standard deviation",
}
_PROFILE_MISSING = {
    "ignore": "Keep empty values",
    "zero": "Fill with zero",
    "fill": "Fill with a value",
    "min": "Fill with minimum",
    "max": "Fill with maximum",
    "error": "Report error",
}
_TRACK_MISSING = {"nan" if key == "ignore" else key: label for key, label in _PROFILE_MISSING.items()}
_TRACK_INTERPOLATION = {
    "none": "None",
    "linear": "Linear",
    "ffill": "Previous value",
    "bfill": "Next value",
}
_VALUE_SCALING = {
    "none": "None",
    "interval": "Target interval",
    "multiplier": "Multiplier",
    "center": "Centre",
    "zscore": "Z-score",
    "maxabs": "Maximum absolute",
}
_SMOOTHING_MODES = {
    "none": "Off",
    "moving_average": "Moving average",
}
_POLISHABLE_FORMS = frozenset({"x", "t", "xt", "track"})
_FEATURES = {
    "generic": "Generic profile",
    "replication_timing": "Replication timing",
    "initiation_rate": "Initiation rate",
    "fork_speed": "Fork speed",
    "fork_directionality": "Fork directionality",
    "replicated_fraction": "Replicated fraction",
}
_FEATURE_VALUE_NAMES = {
    "generic": "value",
    "replication_timing": "replication_timing",
    "initiation_rate": "initiation_rate",
    "fork_speed": "fork_speed",
    "fork_directionality": "fork_directionality",
    "replicated_fraction": "replicated_fraction",
}
_FEATURE_HINTS = {
    "generic": "Values are preserved without assigning biological meaning.",
    "replication_timing": (
        "ENCODE wavelet Repli-seq is relative timing (earlier = larger), not absolute "
        "DNAscape time in minutes."
    ),
    "initiation_rate": "Initiation-rate profiles should be finite and non-negative.",
    "fork_speed": "Fork-speed profiles should be positive and use consistent units.",
    "fork_directionality": "Fork directionality is normally bounded between -1 and 1.",
    "replicated_fraction": "Replicated fraction is normally bounded between 0 and 1.",
}
_FEATURE_THRESHOLD_DEFAULTS: Mapping[str, tuple[bool, float | None, float | None]] = {
    "generic": (False, None, None),
    "replication_timing": (True, 0.0, None),
    "initiation_rate": (True, 0.0, None),
    "fork_speed": (True, 0.0, None),
    "fork_directionality": (True, -1.0, 1.0),
    "replicated_fraction": (True, 0.0, 1.0),
}
_ENCODE_EXAMPLE_URL = (
    "https://hgdownload.gi.ucsc.edu/goldenPath/hg19/encodeDCC/"
    "wgEncodeUwRepliSeq/wgEncodeUwRepliSeqK562WaveSignalRep1.bigWig"
)
_SNS_EXAMPLE_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM716nnn/GSM716435/"
    "suppl/GSM716435_K562_Peaks_FDR15.bed.gz"
)
_OKSEQ_EXAMPLE_URL = (
    "https://raw.githubusercontent.com/CL-CHEN-Lab/OK-Seq/master/"
    "published_results/K562/K562_merged_hmm_RFD_cutoff10_bs1kb_sorted.bw"
)
_H1_REPLISEQ_EXAMPLE_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE137nnn/GSE137764/suppl/"
    "GSE137764%5FH1%5FGaussiansGSE137764%5Fmooth%5Fscaled%5Fautosome%2Emat%2Egz"
)
_MAX_RETAINED_SOURCE_COUNT = 128
_MAX_RETAINED_SOURCE_BYTES = DEFAULT_BATCH_ARCHIVE_SOURCE_BYTES
_MAX_RETAINED_OUTPUT_COUNT = 256
_MAX_RETAINED_OUTPUT_BYTES = 512 * 1024 * 1024
_DATA_FILE_STEM = RDATA_FILE_PREFIX


@dataclass(frozen=True)
class _SelectedData:
    form: str
    shape: tuple[int, ...]
    value_column: str | None = None
    array_key: str | None = None
    conversion_form: str | None = None

    @property
    def request_form(self) -> str:
        """Return the storage form used by ``rdata`` for this selection."""

        return self.conversion_form or self.form


@dataclass(slots=True)
class _DataManagerSource:
    """One retained local upload or public-link source."""

    source_id: str
    mode: str
    inspection: UploadInspection
    selection: str
    output_name: str
    track_inspection: TrackInspection | None = None
    remote_metadata: dict[str, Any] = field(default_factory=dict)
    output_name_customized: bool = False
    configuration: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class _RemoteLinkInput:
    """One editable public-link row and its last accepted source."""

    row_id: int
    url: str = ""
    source_id: str | None = None


@dataclass(slots=True)
class _GeneratedFilePreference:
    """Editable presentation values for one retained converted artifact."""

    file_name: str
    file_type: str
    compress: bool
    automatic_name: str
    customized: bool = False
    filter_layout: str = "positions"


class _ConversionProgressState:
    """Thread-safe hand-off from ``rdata`` progress events to NiceGUI."""

    _STAGE_LABELS = {
        "load": "Loading data",
        "prepare": "Preparing output",
        "write": "Writing file",
        "complete": "Complete",
    }

    def __init__(self) -> None:
        self._lock = Lock()
        self._value = 0.0
        self._message = "Waiting to convert"

    def report(self, event: dict[str, Any]) -> None:
        stage = str(event.get("stage") or "prepare")
        status = str(event.get("status") or "start")
        label = self._STAGE_LABELS.get(stage, stage.replace("_", " ").title())
        message = "Complete" if stage == "complete" else f"{label}..."
        if status == "done" and stage != "complete":
            message = f"{label} complete"
        with self._lock:
            self._value = max(0.0, min(1.0, float(event.get("progress", 0.0))))
            self._message = message

    def snapshot(self) -> tuple[float, str]:
        with self._lock:
            return self._value, self._message


def _infer_form(values: Any) -> str:
    array = np.asarray(values)
    if array.size <= 1:
        return "scalar"
    if array.ndim == 1:
        return "x"
    if array.ndim == 2 and array.shape[0] == 1:
        return "t"
    if array.ndim == 2:
        return "xt"
    return "array"


def _array_form(inspection: UploadInspection, key: str, values: Any) -> str:
    """Prefer DNAscape archive metadata for the primary array when available."""

    if key == inspection.default_array_key and inspection.inferred_form in {
        "scalar",
        "x",
        "t",
        "xt",
        "table",
    }:
        return inspection.inferred_form
    return _infer_form(values)


def _source_stem(name: str) -> str:
    path = PurePath(name)
    if path.suffix.casefold() == ".gz":
        path = path.with_suffix("")
    return compact_export_stem(
        path.stem,
        fallback="data",
        maximum=96,
    )


def _automatic_output_name(source_name: str, feature: str) -> str:
    """Use the shared workflow/output/source order for converted data."""

    source_stem = _source_stem(source_name)
    if feature == "generic":
        return f"{_DATA_FILE_STEM}_{source_stem}"
    output = _FEATURE_VALUE_NAMES.get(feature, feature)
    output_stem = output_file_token(output)
    if source_stem == output_stem:
        source_stem = "profile"
    elif source_stem.startswith(f"{output_stem}-"):
        source_stem = source_stem.removeprefix(f"{output_stem}-")
    return f"{_DATA_FILE_STEM}_{output_stem}_{source_stem}"


def _source_format_label(source_format: str) -> str:
    return _SOURCE_FORMATS.get(source_format, source_format.upper())


def _shape_text(shape: tuple[int, ...]) -> str:
    return " x ".join(str(value) for value in shape) if shape else "1"


def _file_size_text(size: int) -> str:
    value = float(max(0, size))
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"


def _generated_file_stem(result: DataConversionResult) -> str:
    """Return the editable stem from a converted artifact's safe filename."""

    filename = PurePath(result.artifact.filename).name
    if filename.casefold().endswith(".gz"):
        filename = filename[:-3]
    suffix = f".{result.target_format.casefold()}"
    if filename.casefold().endswith(suffix):
        filename = filename[: -len(suffix)]
    return filename or _DATA_FILE_STEM


def _generated_download_artifact(
    result: DataConversionResult,
    *,
    file_name: str,
    file_type: str | None = None,
    compress: bool,
) -> ExportArtifact:
    """Apply editable format, naming, and gzip presentation to a retained result."""

    return transcode_conversion_artifact(
        result,
        output_name=file_name,
        target_format=file_type,
        gzip_output=compress,
    )


def _build_generated_files_archive(
    artifacts: Sequence[ExportArtifact],
) -> ExportArtifact:
    """Backward-compatible Data Manager wrapper around the shared helper."""

    return build_artifact_archive(
        artifacts,
        filename=f"{_DATA_FILE_STEM}_outputs.zip",
        max_source_bytes=DEFAULT_BATCH_ARCHIVE_SOURCE_BYTES,
    )


def _unique_conversion_output_names(
    values: Sequence[str],
    *,
    target_format: str,
    gzip_output: bool,
) -> tuple[str, ...]:
    """Return stems that stay unique after final export-name sanitization."""

    suffix = f".{target_format}{'.gz' if gzip_output else ''}"
    maximum_stem_length = max(1, 180 - len(suffix))
    used_filenames: set[str] = set()
    result: list[str] = []
    for index, value in enumerate(values, start=1):
        filename = safe_export_filename(
            value,
            target_format,
            default_stem=f"{_DATA_FILE_STEM}_{index}",
            gzip_output=gzip_output,
        )
        stem = filename[: -len(suffix)]
        candidate_stem = stem
        candidate_filename = filename
        duplicate = 2
        while candidate_filename.casefold() in used_filenames:
            duplicate_suffix = f"_{duplicate}"
            candidate_stem = (
                f"{stem[: maximum_stem_length - len(duplicate_suffix)].rstrip(' .')}"
                f"{duplicate_suffix}"
            )
            candidate_filename = safe_export_filename(
                candidate_stem,
                target_format,
                default_stem=f"{_DATA_FILE_STEM}_{index}",
                gzip_output=gzip_output,
            )
            duplicate += 1
        used_filenames.add(candidate_filename.casefold())
        result.append(candidate_stem)
    return tuple(result)


def _selection_options(inspection: UploadInspection) -> Mapping[str, str]:
    if inspection.source_format in {"bigwig", "bedgraph"}:
        if inspection.numeric_columns:
            return {f"{_COLUMN_PREFIX}{column}": column for column in inspection.numeric_columns}
        return {_NATURAL: "Binned profile"}

    if inspection.array_keys:
        options: dict[str, str] = {}
        for key in inspection.array_keys:
            values = inspection._array_data.get(key)
            if values is not None and _array_form(inspection, key, values) in {
                "scalar",
                "x",
                "t",
                "xt",
                "table",
            }:
                options[f"{_ARRAY_PREFIX}{key}"] = key
        return options

    options = {}
    row_time_profile = bool(
        inspection.inferred_form == "t"
        and inspection.shape
        and inspection.shape[0] == 1
        and len(inspection.numeric_columns) > 1
    )
    if row_time_profile:
        options[_NATURAL] = "Time profile"
    else:
        if len(inspection.numeric_columns) > 1:
            options[_MATRIX] = "All"
        options.update(
            {f"{_COLUMN_PREFIX}{column}": column for column in inspection.numeric_columns}
        )
    if inspection.source_format in {"csv", "tsv"} and len(inspection.numeric_columns) > 1 and not row_time_profile:
        options[_TABLE] = "Full table"
    if inspection.source_layout == "geo_repliseq":
        options[_REPLISEQ_TIMING] = "Replication timing (S1–S16)"
    if not options:
        options[_NATURAL] = "Profile"
    return options


def _default_selection(inspection: UploadInspection) -> str:
    options = _selection_options(inspection)
    if inspection.source_layout == "geo_repliseq" and _TABLE in options:
        return _TABLE
    if inspection.source_format in {"bigwig", "bedgraph"}:
        preferred = (
            f"{_COLUMN_PREFIX}{inspection.default_column}"
            if inspection.default_column is not None
            else _NATURAL
        )
    elif inspection.array_keys:
        preferred = f"{_ARRAY_PREFIX}{inspection.default_array_key}"
    elif inspection.inferred_form == "t" and _NATURAL in options:
        preferred = _NATURAL
    elif inspection.inferred_form == "xt" and _MATRIX in options:
        preferred = _MATRIX
    elif inspection.default_column is not None:
        preferred = f"{_COLUMN_PREFIX}{inspection.default_column}"
    else:
        preferred = _NATURAL
    return preferred if preferred in options else next(iter(options))


def _selected_data(inspection: UploadInspection, token: str) -> _SelectedData:
    if token == _REPLISEQ_TIMING and inspection.source_layout == "geo_repliseq":
        return _SelectedData("x", (int(inspection.shape[0]),))
    if inspection.source_format in {"bigwig", "bedgraph"}:
        column = token.removeprefix(_COLUMN_PREFIX) if token.startswith(_COLUMN_PREFIX) else None
        return _SelectedData("track", tuple(inspection.shape or ()), value_column=column)
    if token == _TABLE:
        return _SelectedData("table", tuple(inspection.shape or ()))
    if token == _MATRIX:
        rows = int(inspection.shape[0]) if inspection.shape else 0
        # A wide chromosome table is a collection of independent spatial
        # profiles. Selecting every header must not relabel the biological
        # dependence as space-time merely because its storage is a matrix.
        semantic_form = "x" if inspection.inferred_form == "x" else "xt"
        return _SelectedData(
            semantic_form,
            (rows, len(inspection.numeric_columns)),
            conversion_form="xt",
        )
    if token.startswith(_COLUMN_PREFIX):
        column = token.removeprefix(_COLUMN_PREFIX)
        values = inspection._table_data.get(column)
        rows = (
            int(values.shape[0])
            if values is not None
            else (int(inspection.shape[0]) if inspection.shape else 0)
        )
        return _SelectedData("x", (rows,), value_column=column)
    if token.startswith(_ARRAY_PREFIX):
        array_key = token.removeprefix(_ARRAY_PREFIX)
        values = inspection._array_data.get(array_key)
        if values is None:
            return _SelectedData("array", (), array_key=array_key)
        return _SelectedData(
            _array_form(inspection, array_key, values),
            tuple(values.shape),
            array_key=array_key,
        )
    form = str(inspection.inferred_form)
    return _SelectedData(form, tuple(inspection.shape or ()))


def _apply_feature_defaults(
    settings: dict[str, Any],
    feature: str,
    inspection: UploadInspection | None = None,
    backend: str | None = None,
) -> None:
    """Apply the value-domain defaults implied by a biological profile type."""

    selected = feature if feature in _FEATURES else "generic"
    enabled, minimum, maximum = _FEATURE_THRESHOLD_DEFAULTS[selected]
    settings.update(
        feature=selected,
        threshold_enabled=enabled,
        threshold_min=minimum,
        threshold_max=maximum,
    )
    _apply_track_defaults(settings, selected, inspection, backend)


def _apply_track_defaults(
    settings: dict[str, Any],
    selected: str,
    inspection: UploadInspection | None,
    backend: str | None,
) -> None:
    """Set source-aware binning without resetting user-edited timing bounds."""

    if inspection is not None and inspection.source_format in {"bigwig", "bedgraph"}:
        settings["aggregation"] = _default_track_aggregation(inspection, selected, backend)
        if selected == "initiation_rate" and inspection.source_format == "bedgraph":
            settings.update(
                bin_size=1_000,
                track_missing="zero",
                interpolation="none",
                smoothing_mode="none",
                smooth_window=1,
                value_scaling="none",
                scale=1.0,
                rescale_invert=False,
            )


def _resolved_feature_bounds(
    feature: str,
    minimum: float | None,
    maximum: float | None,
) -> tuple[bool, float | None, float | None]:
    """Resolve blank limits against the selected biological profile defaults."""

    selected = feature if feature in _FEATURES else "generic"
    _enabled, default_minimum, default_maximum = _FEATURE_THRESHOLD_DEFAULTS[selected]
    resolved_minimum = default_minimum if minimum is None else minimum
    resolved_maximum = default_maximum if maximum is None else maximum
    return (
        resolved_minimum is not None or resolved_maximum is not None,
        resolved_minimum,
        resolved_maximum,
    )


def _feature_settings_compatible(
    feature: str, settings: Mapping[str, Any], *, missing_key: str = "missing_policy",
) -> bool:
    """Keep biological names only while the configured value domain fits them."""

    if feature not in _FEATURE_THRESHOLD_DEFAULTS or feature == "generic":
        return True
    _enabled, domain_min, domain_max = _FEATURE_THRESHOLD_DEFAULTS[feature]
    minimum = -math.inf if domain_min is None else domain_min
    maximum = math.inf if domain_max is None else domain_max
    try:
        _enabled, lower, upper = _resolved_feature_bounds(
            feature, settings.get("threshold_min"), settings.get("threshold_max"),
        )
        lower = -math.inf if lower is None else float(lower)
        upper = math.inf if upper is None else float(upper)
        if math.isnan(lower) or math.isnan(upper) or lower > upper:
            return True  # Incomplete edits are validated when Convert is clicked.
        if lower < minimum or upper > maximum:
            return False
        scaling = settings.get("value_scaling", "none")
        if scaling in {"center", "zscore"}:
            return False  # Centred scores no longer retain the physical zero.
        if scaling == "interval":
            lower = float(settings["rescale_min"])
            upper = float(settings["rescale_max"])
            if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
                return True
        elif scaling == "multiplier":
            multiplier = float(settings.get("scale", 1.0))
            if not math.isfinite(multiplier):
                return True
            lower, upper = (
                sorted((lower * multiplier, upper * multiplier)) if multiplier else (0.0, 0.0)
            )
        elif scaling == "maxabs":
            lower, upper = (-1.0 if lower < 0 else 0.0), (1.0 if upper > 0 else 0.0)
        if lower < minimum or upper > maximum:
            return False
        if settings.get(missing_key) == "fill":
            fill_value = float(settings.get("fill_value", 0.0))
            if math.isfinite(fill_value) and not minimum <= fill_value <= maximum:
                return False
    except (KeyError, TypeError, ValueError):
        return True
    return True


def _selected_value_name(
    source: _DataManagerSource,
    details: _SelectedData,
    feature: str,
) -> str:
    """Keep a source label when one exists and otherwise use a useful fallback."""

    if source.selection == _REPLISEQ_TIMING:
        return "replication_timing"

    candidates = (
        details.value_column,
        (
            source.inspection.numeric_columns[0]
            if len(source.inspection.numeric_columns) == 1
            else None
        ),
        details.array_key,
        _FEATURE_VALUE_NAMES.get(feature),
        "value",
    )
    for candidate in candidates:
        label = str(candidate or "").strip()
        if label and len(label) <= 80 and not any(ord(char) < 32 for char in label):
            return label
    return "value"


def _track_aggregation_options(
    backend: str | None,
    inspection: UploadInspection | None = None,
) -> dict[str, str]:
    options = dict(_TRACK_AGGREGATIONS)
    if inspection is not None and inspection.track_kind == "intervals":
        options = {"max": "Presence", "sum": "Covered bases", "coverage": "Coverage"}
    elif inspection is not None and inspection.track_kind == "peaks":
        options = {
            "max": "Maximum peak signal",
            "mean": "Mean peak signal",
            "sum": "Integrated signal (signal × bp)",
            "coverage": "Coverage",
        }
    if backend == "gwseq_io_pp":
        options.pop("min", None)
        options.pop("max", None)
    return options


def _default_track_aggregation(
    inspection: UploadInspection,
    feature: str,
    backend: str | None,
) -> str:
    options = _track_aggregation_options(backend, inspection)
    preferred = "max" if (
        inspection.track_kind in {"peaks", "intervals"}
        or (feature == "initiation_rate" and inspection.source_format == "bedgraph")
    ) else "mean"
    if feature == "replication_timing" and "mean" in options:
        preferred = "mean"
    return preferred if preferred in options else next(iter(options))


def _track_interpolation_options(inspection: UploadInspection) -> dict[str, str]:
    # Empty space between called origins is background, not a missing segment
    # of a continuous signal that can be reconstructed by interpolation.
    if inspection.track_kind in {"peaks", "intervals"}:
        return {"none": "None"}
    return dict(_TRACK_INTERPOLATION)


def _value_scaling_options(feature: str) -> dict[str, str]:
    options = dict(_VALUE_SCALING)
    if feature == "initiation_rate":
        options.pop("center")
        options.pop("zscore")
    return options


def _preview_options(
    preview: Any,
    controls: PreviewSettingsControls | None = None,
    *,
    automatic_y_limits: bool = True,
) -> dict[str, Any]:
    """Build the small, bounded visual check shown after conversion."""

    values = np.asarray(preview.values).reshape(-1)
    x_axis = np.asarray(preview.x).reshape(-1)
    point_count = min(values.size, x_axis.size)
    values = values[:point_count]
    x_axis = x_axis[:point_count]
    plot_type = "Histogram" if controls is not None and controls.view.value == "Histogram" else "Line"
    if plot_type == "Line" and point_count > preview_point_limit():
        sample = np.linspace(0, point_count - 1, preview_point_limit(), dtype=int)
        values = values[sample]
        x_axis = x_axis[sample]
    value_label = str(preview.value_label).replace("_", " ").strip().capitalize()
    series_label = (
        str(controls.legend_name.value or "").strip() or value_label
        if controls is not None and controls.legend_name is not None else value_label
    )

    options: dict[str, Any] = {}
    if controls is not None:
        options = dict(
            log_x=bool(controls.log_x.value),
            invert_x=bool(controls.invert_x.value),
            x_limits=axis_limits_from_controls(controls, "x"),
            histogram_bin_size=controls.histogram_bin_size.value,
            histogram_x_limits=histogram_x_limits_from_controls(
                controls, log_x=bool(controls.log_x.value),
            ),
            histogram_touching=bool(controls.histogram_touching.value),
            kde_bandwidth=float(controls.kde_bandwidth.value or 1.0),
            histogram_grouped=bool(controls.histogram_grouped.value),
            histogram_boundary=str(controls.histogram_boundary.value),
            show_histogram_statistics=bool(controls.show_histogram_statistics.value),
            histogram_statistics_position=str(controls.histogram_statistics_position.value),
            legend_position=str(controls.legend_position.value),
            line_type=str(controls.line_type.value),
            aspect_ratio=controls.aspect_ratio.value,
        )
    result = build_preview_plot_options(
        plot_type=plot_type,
        title="Prepared profile",
        x_label=str(controls.x_axis_label.value or "") if controls is not None else "Position (kb)",
        y_label=str(controls.y_axis_label.value or "") if controls is not None else value_label,
        color=str(controls.primary_color.value) if controls is not None else "#b03412",
        palette="viridis",
        histogram_bins=int(controls.histogram_bins.value) if controls is not None else 40,
        density=bool(controls.density.value) if controls is not None else False,
        show_legend=bool(controls.show_legend.value) if controls is not None else False,
        show_colorbar=False,
        log_y=bool(controls.log_y.value) if controls is not None else False,
        invert_y=bool(controls.invert_y.value) if controls is not None else False,
        y_limits=(
            (float(controls.y_minimum.value), float(controls.y_maximum.value))
            if controls is not None and not automatic_y_limits
            and controls.y_minimum.value is not None and controls.y_maximum.value is not None
            else None
        ),
        series_label=series_label,
        values=values,
        x_axis=x_axis,
        show_grid=bool(controls.show_grid.value) if controls is not None else False,
        interactive=True,
        line_width=float(controls.line_width.value) if controls is not None else 1.8,
        **options,
    )
    if plot_type == "Line" and preview.minimum is not None and preview.maximum is not None:
        global_y = (
            _logarithmic_data_limits(preview.minimum_positive or preview.minimum, preview.maximum)
            if controls is not None and controls.log_y.value else nice_linear_limits(preview.minimum, preview.maximum)
        )
        result["dnascapeDefaultAxes"]["y"] = list(global_y)
        if automatic_y_limits:
            result["yAxis"].update(min=global_y[0], max=global_y[1])
    return result


class _FilterPreviewRegions(list):
    """Serializable ECharts areas with native-bin detail on viewport changes."""

    def __init__(self, selection: FilterSelection, x_limits=None):
        super().__init__([[{"xAxis": start}, {"xAxis": end}] for start, end in selection.ranges(x_limits)])
        self.selection = selection
        self.window = x_limits

    def refine(self, x_limits):
        return _FilterPreviewRegions(self.selection, x_limits)


def _converted_preview_data(result: DataConversionResult, *, log_x: bool = False) -> DetailedPreviewData:
    """Refine converted profiles by streaming retained bytes, without another full array."""
    preview = result.preview
    initial = [[float(x), float(y) if np.isfinite(y) else None] for x, y in zip(preview.x, preview.values, strict=True)]
    coordinate_names = ("position_kb", "position_bp", "position", "time_min", "time")
    sparse_rows = None
    index_lock = Lock()
    first_end = result.artifact.data.find(b"\n")
    first_line = result.artifact.data[:first_end if first_end >= 0 else 1024]
    plain_profile = (
        result.target_format in {"tsv", "txt", "csv"}
        and not result.artifact.data.startswith(b"\x1f\x8b")
        and len(result.labels) == 1
        and (len(preview.source_shape) == 1 or (len(preview.source_shape) == 2 and preview.source_shape[1] == 1))
        and preview.total_count > 1
        and (
            (result.target_format == "tsv" and b"\t" not in first_line)
            or (result.target_format == "csv" and b"," not in first_line)
            or (result.target_format == "txt" and len(first_line.split()) == 1)
        )
    )

    def indexed_text_detail(view):
        """Seek native samples in retained bytes using a small, lazy row index."""
        nonlocal sparse_rows
        raw = result.artifact.data
        stride = 1024
        with index_lock:
            if sparse_rows is None:
                offsets = array("Q")
                position = 0 if result.target_format == "txt" else raw.find(b"\n") + 1
                row = 0
                while position < len(raw):
                    if row % stride == 0:
                        offsets.append(position)
                    end = raw.find(b"\n", position)
                    if end < 0:
                        break
                    position, row = end + 1, row + 1
                sparse_rows = offsets
        step = (preview.x[-1] - preview.x[0]) / (preview.total_count - 1)
        bounds = (min(preview.x[0], preview.x[-1]), max(preview.x[0], preview.x[-1]))
        buffer = buffered_preview_limits(view, bounds, log=log_x)
        def selection(limits):
            indices = sorted((value - preview.x[0]) / step for value in limits)
            return max(0, math.floor(indices[0]) - 1), min(preview.total_count, math.ceil(indices[1]) + 2)
        core_start, core_stop = selection(view)
        start, stop = selection(buffer)
        limit = preview_point_limit()
        def sampled(first, last):
            size = max(0, last - first)
            return first + (np.arange(size) if size <= limit else np.linspace(0, size - 1, limit, dtype=int))
        indices = np.unique(np.concatenate((sampled(start, core_start), sampled(core_start, core_stop), sampled(core_stop, stop))))
        points, current_row, position = [], -1, 0
        for index in indices:
            index = int(index)
            checkpoint = index // stride
            if current_row < checkpoint * stride:
                current_row, position = checkpoint * stride, sparse_rows[checkpoint]
            while current_row < index:
                position = raw.find(b"\n", position) + 1
                current_row += 1
            end = raw.find(b"\n", position)
            cell = raw[position:end if end >= 0 else len(raw)].strip()
            value = float(cell if cell not in {b"", b'""'} else b"nan")
            points.append([float(preview.x[0] + index * step), value if math.isfinite(value) else None])
        overview = initial if len(initial) <= limit else [initial[index] for index in np.linspace(0, len(initial) - 1, limit, dtype=int)]
        return {"data": DetailedPreviewData(retain_curve_overview(overview, points), refine, coverage={"x": buffer})}

    def columns(header: Sequence[str]) -> tuple[int, int | None]:
        labels = [str(label).strip().casefold() for label in header]
        excluded = {*coordinate_names, "bin", "end", "end_bp", "start", "start_bp", "x", "index", "chromosome", "chrom", "chr"}
        value = labels.index(str(preview.value_label).casefold()) if str(preview.value_label).casefold() in labels else next((i for i, name in enumerate(labels) if name not in excluded), 0)
        coordinate = next((labels.index(name) for name in coordinate_names if name in labels), None)
        return value, coordinate

    def numeric_rows(stream: Any):
        """Read small blocks of the first displayed NPZ series, including Fortran arrays."""
        with zipfile.ZipFile(stream) as archive:
            names = archive.namelist()
            name = "values.npy" if "values.npy" in names else next((f"{label}.npy" for label in result.labels if f"{label}.npy" in names), None)
            if name is None:
                return
            header = list(result.labels)
            if "columns.npy" in names:
                with archive.open("columns.npy") as metadata:
                    header = np.lib.format.read_array(metadata, allow_pickle=False).tolist()
            with archive.open(name) as data:
                version = np.lib.format.read_magic(data)
                shape, fortran, dtype = (np.lib.format.read_array_header_1_0(data) if version == (1, 0)
                                         else np.lib.format.read_array_header_2_0(data))
                if dtype.hasobject or len(shape) > 2:
                    return
                offset = data.tell()
                row_count = shape[0] if shape else 1
                column_count = shape[1] if len(shape) == 2 else 1
                if row_count == 1:
                    row_count, column_count = column_count, 1
                value, coordinate = columns(header) if len(header) == column_count else (0, None)
                if fortran and column_count > 1:
                    with archive.open(name) as coords:
                        data.seek(offset + value * row_count * dtype.itemsize)
                        if coordinate is not None:
                            coords.seek(offset + coordinate * row_count * dtype.itemsize)
                        for start in range(0, row_count, 8192):
                            size = min(8192, row_count - start)
                            values = np.frombuffer(data.read(size * dtype.itemsize), dtype=dtype)
                            axis = np.frombuffer(coords.read(size * dtype.itemsize), dtype=dtype) if coordinate is not None else None
                            for i, y in enumerate(values):
                                yield start + i, float(axis[i]) if axis is not None else None, float(y)
                else:
                    block_size = max(1, 65536 // column_count)
                    for start in range(0, row_count, block_size):
                        size = min(block_size, row_count - start)
                        block = np.frombuffer(data.read(size * column_count * dtype.itemsize), dtype=dtype).reshape(-1, column_count)
                        for i, row in enumerate(block):
                            yield start + i, float(row[coordinate]) if coordinate is not None else None, float(row[value])

    def source_rows():
        raw = io.BytesIO(result.artifact.data)
        stream = gzip.GzipFile(fileobj=raw) if result.artifact.data.startswith(b"\x1f\x8b") else raw
        if result.target_format == "npz":
            yield from numeric_rows(stream)
            return
        with io.TextIOWrapper(stream, encoding="utf-8-sig") as text:
            if result.target_format == "txt":
                value, coordinate = columns(result.labels)
                rows = (line.split() for line in text)
            else:
                rows = csv.reader(text, delimiter="," if result.target_format == "csv" else "\t")
                value, coordinate = columns(next(rows, []))
            if len(preview.source_shape) == 2 and preview.source_shape[0] == 1 and preview.total_count > 1:
                for index, value_text in enumerate(cell for row in rows for cell in row):
                    yield index, None, float(value_text) if value_text.strip() else float("nan")
                return
            for index, row in enumerate(rows):
                try:
                    yield index, float(row[coordinate]) if coordinate is not None else None, float(row[value]) if row[value].strip() else float("nan")
                except (ValueError, IndexError):
                    continue

    def refine(x_view: Any, _y_view: Any) -> dict[str, Any]:
        limit = preview_point_limit()
        if x_view is None and len(initial) == min(limit, preview.total_count):
            return {"data": DetailedPreviewData(initial, refine)}
        if plain_profile:
            return indexed_text_detail(x_view or (min(preview.x[0], preview.x[-1]), max(preview.x[0], preview.x[-1])))
        if x_view is None:
            x_view = (min(preview.x[0], preview.x[-1]), max(preview.x[0], preview.x[-1]))
        original_view = x_view
        x_view = buffered_preview_limits(
            x_view, (min(preview.x[0], preview.x[-1]), max(preview.x[0], preview.x[-1])), log=log_x,
        )
        transform = math.log if log_x and original_view[0] > 0 else float
        span = transform(original_view[1]) - transform(original_view[0])
        if span > 0:
            limit = min(limit * 3, max(limit, math.ceil(limit * (transform(x_view[1]) - transform(x_view[0])) / span)))
        step = (preview.x[-1] - preview.x[0]) / max(1, preview.total_count - 1)
        ascending = step >= 0

        def window_rows():
            first_coordinate, previous = None, None
            started = False
            for row_index, coordinate, y in source_rows():
                if coordinate is None:
                    x = float(preview.x[0] + row_index * step)
                else:
                    if first_coordinate is None:
                        first_coordinate = coordinate
                    x = coordinate - first_coordinate + float(preview.x[0])
                point = [x, y if np.isfinite(y) else None]
                before = x < x_view[0] if ascending else x > x_view[1]
                after = x > x_view[1] if ascending else x < x_view[0]
                if before:
                    previous = point
                    continue
                if not started:
                    if previous is not None:
                        yield previous
                    started = True
                yield point
                if after:
                    break

        points = []
        count = 0
        for point in window_rows():
            count += 1
            if count <= limit:
                points.append(point)
        if count > limit:
            # Count first, then select evenly spaced native rows. Memory
            # stays bounded even for a broad viewport over a large text file.
            selected_indices = iter(np.linspace(0, count - 1, limit, dtype=int))
            selected_index = next(selected_indices)
            points = []
            for index, point in enumerate(window_rows()):
                if index == selected_index:
                    points.append(point)
                    selected_index = next(selected_indices, None)
                    if selected_index is None:
                        break
        overview_limit = preview_point_limit()
        overview = initial if len(initial) <= overview_limit else [initial[index] for index in np.linspace(0, len(initial) - 1, overview_limit, dtype=int)]
        return {"data": DetailedPreviewData(retain_curve_overview(overview, points), refine, coverage={"x": x_view})}

    return refine(None, None)["data"]


def _number(control: Any, label: str) -> float:
    try:
        value = float(control.value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a number.") from exc
    if not np.isfinite(value):
        raise ValueError(f"{label} must be finite.")
    return value


def _optional_number(control: Any, label: str) -> float | None:
    raw_value = control.value
    if raw_value is None or (isinstance(raw_value, str) and not raw_value.strip()):
        return None
    return _number(control, label)


def _integer(control: Any, label: str, *, minimum: int | None = None) -> int:
    value = _number(control, label)
    if not value.is_integer():
        raise ValueError(f"{label} must be a whole number.")
    result = int(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{label} must be at least {minimum}.")
    return result


def render(_navigate: Callable[[str], None]) -> None:
    """Render a live, local data conversion workspace."""

    retained_sources: list[_DataManagerSource] = []
    active_source_ids: dict[str, str | None] = {"upload": None, "url": None}
    remote_link_inputs: list[_RemoteLinkInput] = [_RemoteLinkInput(row_id=1)]
    remote_inspections: set[int] = set()
    next_remote_link_id = 1
    converted_results: list[tuple[int, DataConversionResult]] = []
    selected_result_id: int | None = None
    generated_file_preferences: dict[int, _GeneratedFilePreference] = {}
    next_result_id = 0
    conversion_running = False
    page_activity = current_page_activity()
    active_cancel: Event | None = None
    upload_inspection_running = False
    result_chart: Any | None = None
    controls: dict[str, Any] = {}
    filter_controls: DataFilterControls | None = None
    transform_controls: DataTransformControls | None = None
    source_controls: dict[str, Any] = {}
    source_metadata_slots: dict[str, Any] = {}
    source_control_states: dict[str, bool] = {}
    request_control_states: dict[tuple[str, str], bool] = {}
    settings: dict[str, Any] = {
        "apply_configuration": True,
        "source_mode": "upload",
        "source_url": "",
        "feature": "generic",
        "target_format": "tsv",
        "output_name": _DATA_FILE_STEM,
        "decimal_places": 6,
        "missing_policy": "zero",
        "chromosome_scope": "single",
        "chromosome": None,
        "start": 0,
        "end": 1,
        "bin_size": 1_000,
        "source_bin_size": 1_000,
        "_profile_grid_key": None,
        "profile_region_custom": False,
        "profile_bin_custom": False,
        "aggregation": "mean",
        "track_missing": "zero",
        "fill_value": 0.0,
        "interpolation": "none",
        "smoothing_mode": "none",
        "smooth_window": 1,
        "value_scaling": "none",
        "rescale_min": 0.0,
        "rescale_max": 1.0,
        "rescale_invert": False,
        "threshold_enabled": False,
        "threshold_min": None,
        "threshold_max": None,
        "scale": 1.0,
        "show_preview": True,
        "repliseq_timing_method": "sigmoid",
        "s_phase_duration_min": 600.0,
    }
    configuration_defaults = {
        key: value for key, value in settings.items()
        if key not in {"source_mode", "source_url", "output_name", "show_preview"}
    }
    displayed_source_id: str | None = None
    apply_configuration_to_all = {"upload": True, "url": True}

    def save_source_configuration() -> None:
        source = next((item for item in retained_sources if item.source_id == displayed_source_id), None)
        if source is not None:
            targets = (
                sources_for_mode(source.mode) if apply_configuration_to_all[source.mode] else [source]
            )
            for target in targets:
                target.configuration.update({key: settings[key] for key in configuration_defaults})

    def load_source_configuration(source: _DataManagerSource) -> None:
        nonlocal displayed_source_id
        settings.update(configuration_defaults)
        settings.update(source.configuration)
        settings["output_name"] = source.output_name
        displayed_source_id = source.source_id

    def configuration_for(source: _DataManagerSource) -> dict[str, Any]:
        displayed = next(
            (item for item in retained_sources if item.source_id == displayed_source_id), None,
        )
        use_displayed = source.source_id == displayed_source_id or (
            apply_configuration_to_all[source.mode] and displayed is not None and displayed.mode == source.mode
        )
        return {
            **configuration_defaults,
            **source.configuration,
            **({key: settings[key] for key in configuration_defaults} if use_displayed else {}),
        }

    def selected_generated_result() -> DataConversionResult | None:
        nonlocal selected_result_id
        if not converted_results:
            selected_result_id = None
            return None
        for result_id, result in converted_results:
            if result_id == selected_result_id:
                return result
        selected_result_id, result = converted_results[-1]
        return result

    def source_mode() -> str:
        return str(settings.get("source_mode") or "upload")

    def sources_for_mode(mode: str | None = None) -> list[_DataManagerSource]:
        selected_mode = str(mode or source_mode())
        candidates = [
            source for source in retained_sources if source.mode == selected_mode
        ]
        if selected_mode == "url":
            row_order = {
                row.source_id or f"url:{row.row_id}": index
                for index, row in enumerate(remote_link_inputs)
            }
            candidates.sort(
                key=lambda source: row_order.get(source.source_id, len(row_order))
            )
        return candidates

    def active_source(mode: str | None = None) -> _DataManagerSource | None:
        selected_mode = str(mode or source_mode())
        candidates = sources_for_mode(selected_mode)
        if not candidates:
            active_source_ids[selected_mode] = None
            return None
        selected_id = active_source_ids.get(selected_mode)
        selected = next(
            (source for source in candidates if source.source_id == selected_id),
            candidates[-1],
        )
        active_source_ids[selected_mode] = selected.source_id
        return selected

    def selected_data(source: _DataManagerSource) -> _SelectedData:
        details = _selected_data(source.inspection, source.selection)
        if source.selection == _REPLISEQ_TIMING and source.track_inspection is not None:
            regions = {region.chromosome: region for region in source.track_inspection.regions}
            region = regions.get(str(configuration_for(source)["chromosome"]), next(iter(regions.values())))
            bin_size = int(dict(source.track_inspection.metadata)["native_bin_size_bp"])
            return _SelectedData("x", (math.ceil(region.end / bin_size),))
        return details

    def render_source_metadata(source_id: str, slot: Any) -> None:
        source = next(
            (source for source in retained_sources if source.source_id == source_id), None,
        )
        slot.clear()
        slot.set_visibility(source is not None)
        if source is None:
            return
        details = selected_data(source)
        format_label = (
            "GEO Repli-seq MAT" if source.inspection.source_layout == "geo_repliseq"
            else _source_format_label(source.inspection.source_format)
        )
        with slot:
            ui.label(f"Size: {_file_size_text(len(source.inspection.content))}").classes(
                "dn-mapping-detected-summary"
            )
            ui.label(f"Format: {format_label}").classes("dn-mapping-detected-summary")
            if source.track_inspection is not None and details.form == "track":
                region = source.track_inspection.regions[0]
                try:
                    bin_size = int(configuration_for(source)["bin_size"])
                    estimated_bins = max(1, (region.end - region.start + bin_size - 1) // bin_size)
                    bin_text = f"{estimated_bins:,}" if bin_size > 0 else "—"
                except (TypeError, ValueError, OverflowError, ZeroDivisionError):
                    bin_text = "—"
                ui.label(f"Estimated bins: {bin_text}").classes("dn-mapping-detected-summary")
            else:
                ui.label(f"Array size: {_shape_text(details.shape)}").classes(
                    "dn-mapping-detected-summary"
                )
            ui.label(f"Dependence: {_FORM_LABELS.get(details.form, 'Unknown')}").classes(
                "dn-mapping-detected-summary dn-mapping-detected-dependence"
            )
            if source.track_inspection is not None:
                chromosomes = len(source.track_inspection.regions)
                chromosome_details = f"Chromosomes: {chromosomes}"
                if source.inspection.source_layout == "geo_repliseq":
                    native_bin = int(dict(source.track_inspection.metadata)["native_bin_size_bp"])
                    chromosome_details += f"; resolution: {native_bin / 1000:g} kb/bin"
                ui.label(chromosome_details).classes("dn-mapping-detected-summary")

    def source_metadata_slot(source_id: str) -> None:
        slot = ui.element("div").classes(
            "dn-mapping-detected-data dn-profile-metadata dn-data-manager-detected"
        ).mark("data-manager-detected-data", f"data-manager-source-details-{source_id}")
        slot._props["data-source-id"] = source_id
        source_metadata_slots[source_id] = slot
        render_source_metadata(source_id, slot)

    def refresh_source_metadata() -> None:
        for source_id, slot in tuple(source_metadata_slots.items()):
            if slot.is_deleted:
                source_metadata_slots.pop(source_id, None)
            else:
                render_source_metadata(source_id, slot)

    def unresolved_link_rows() -> tuple[_RemoteLinkInput, ...]:
        if source_mode() != "url":
            return ()
        return tuple(
            row
            for row in remote_link_inputs
            if row.url.strip() and row.source_id is None
        )

    def set_convert_enabled() -> None:
        if conversion_running:
            if active_cancel is not None and not active_cancel.is_set():
                convert_button.enable()
            else:
                convert_button.disable()
            return
        if (
            upload_inspection_running
            or (filter_controls is not None and filter_controls.loading)
            or remote_inspections
            or unresolved_link_rows()
            or not sources_for_mode()
        ):
            convert_button.disable()
        else:
            convert_button.enable()

    def lock_source_controls(locked: bool) -> None:
        for name, control in tuple(source_controls.items()):
            if getattr(control, "is_deleted", False):
                continue
            if locked:
                source_control_states[name] = bool(getattr(control, "enabled", True))
            if name == "uploader":
                set_data_upload_locked(control, locked)
                if not locked and not source_control_states.pop(name, True):
                    control.disable()
            elif locked:
                control.disable()
            elif not source_control_states.pop(name, True):
                control.disable()
            else:
                control.enable()

    def lock_request_controls(locked: bool) -> None:
        lock_source_controls(locked)
        if filter_controls is not None:
            filter_controls.set_locked(locked)
        if transform_controls is not None:
            transform_controls.set_locked(locked)
        for name, control in tuple(controls.items()):
            if getattr(control, "is_deleted", False):
                continue
            key = ("configure", name)
            if locked:
                request_control_states[key] = bool(getattr(control, "enabled", True))
                control.disable()
            elif request_control_states.pop(key, True):
                control.enable()
            else:
                control.disable()

    page_intro(
        "Data Manager",
        "Convert genomic data files",
        "Import, prepare, and export genomic profiles from local files or public data sources.",
    )
    with (
        ui.element("div")
        .classes("dn-workspace dn-data-manager-workspace")
        .mark("data-manager-workspace")
    ):
        with ui.column().classes("dn-control-stack"):
            with panel(extra_classes="dn-control-card"):
                panel_title("Controls")
                with expansion("Source", icon=DATA_ICON, expanded=True):
                    source_controls.clear()

                    async def refresh_active_source_views() -> None:
                        current = active_source()
                        if current is not None and current.source_id != displayed_source_id:
                            save_source_configuration()
                            load_source_configuration(current)
                        else:
                            save_source_configuration()
                        await render_source_details.refresh()
                        await render_conversion_settings.refresh()
                        save_source_configuration()
                        if transform_controls is not None:
                            transform_controls.refresh()
                        set_convert_enabled()

                    async def adopt_candidate(
                        candidate: UploadInspection,
                        *,
                        mode: str,
                        source_id: str,
                        metadata: Mapping[str, Any] | None = None,
                    ) -> bool:
                        if conversion_running:
                            notify(
                                "Wait for the current conversion to finish before changing "
                                "the sources.",
                                type="warning",
                            )
                            return False
                        existing_index = next(
                            (
                                index
                                for index, source in enumerate(retained_sources)
                                if source.source_id == source_id
                            ),
                            None,
                        )
                        existing = (
                            retained_sources[existing_index]
                            if existing_index is not None
                            else None
                        )
                        if (
                            existing is None
                            and len(retained_sources) >= _MAX_RETAINED_SOURCE_COUNT
                        ):
                            notify(
                                "Remove a retained source before adding another one.",
                                type="warning",
                            )
                            return False
                        retained_bytes = sum(
                            len(source.inspection.content)
                            for source in retained_sources
                            if source.source_id != source_id
                        )
                        if (
                            retained_bytes + len(candidate.content)
                            > _MAX_RETAINED_SOURCE_BYTES
                        ):
                            maximum_mib = _MAX_RETAINED_SOURCE_BYTES / 1024**2
                            notify(
                                "Retained source files are limited to "
                                f"{maximum_mib:g} MiB in one local workspace. "
                                "Remove files you no longer need and try again.",
                                type="warning",
                            )
                            return False
                        convert_button.disable()
                        try:
                            candidate_track = None
                            if candidate.source_layout == "geo_repliseq":
                                candidate_track = await run.io_bound(inspect_repliseq, candidate)
                            if candidate.source_format in {"bigwig", "bedgraph"}:
                                candidate_track = await run.io_bound(inspect_track, candidate)
                                if candidate_track is None:
                                    raise RuntimeError(
                                        "Track inspection stopped before returning chromosome data."
                                    )
                            candidate_selection = _default_selection(candidate)
                            if candidate.source_layout == "geo_repliseq" and settings["feature"] == "replication_timing":
                                candidate_selection = _REPLISEQ_TIMING
                            _selected_data(candidate, candidate_selection)
                        except (OSError, RuntimeError, TypeError, ValueError) as exc:
                            notify(str(exc), type="negative")
                            set_convert_enabled()
                            return False

                        if (
                            existing is not None
                            and existing.selection in _selection_options(candidate)
                        ):
                            candidate_selection = existing.selection
                        save_source_configuration()
                        source_configuration = dict(existing.configuration) if existing is not None else {
                            key: settings[key] for key in configuration_defaults
                        }
                        if existing is None and retained_sources and not apply_configuration_to_all[mode]:
                            source_configuration["apply_configuration"] = True
                        automatic_name = _automatic_output_name(
                            candidate.name,
                            str(source_configuration["feature"]),
                        )
                        adopted = _DataManagerSource(
                            source_id=source_id,
                            mode=mode,
                            inspection=candidate,
                            selection=candidate_selection,
                            output_name=(
                                existing.output_name
                                if existing is not None
                                and existing.output_name_customized
                                else automatic_name
                            ),
                            track_inspection=candidate_track,
                            remote_metadata=dict(metadata or {}),
                            output_name_customized=bool(
                                existing is not None and existing.output_name_customized
                            ),
                            configuration=source_configuration,
                        )
                        if existing_index is None:
                            retained_sources.append(adopted)
                        else:
                            retained_sources[existing_index] = adopted
                        active_source_ids[mode] = source_id
                        load_source_configuration(adopted)
                        cache_upload(
                            candidate,
                            source="Data Manager",
                            navigate_to="data-management",
                        )
                        if candidate_track is not None:
                            region = candidate_track.regions[0]
                            settings.update(
                                chromosome=region.chromosome,
                                start=region.start,
                                end=region.end,
                            )
                            if candidate.source_layout == "geo_repliseq":
                                settings["bin_size"] = int(dict(candidate_track.metadata)["native_bin_size_bp"])
                            _apply_track_defaults(
                                settings, str(settings["feature"]), candidate,
                                candidate_track.backend,
                            )
                        await refresh_active_source_views()
                        return True

                    async def uploaded(event: Any) -> bool:
                        nonlocal upload_inspection_running
                        if conversion_running:
                            notify("A conversion is already running.", type="warning")
                            return False
                        upload_inspection_running = True
                        convert_button.disable()
                        try:
                            candidate = await inspect_upload_event(
                                event,
                                expected_form="auto",
                            )
                            return await adopt_candidate(
                                candidate,
                                mode="upload",
                                source_id=f"upload:{candidate.name}",
                            )
                        except (OSError, RuntimeError, TypeError, ValueError) as exc:
                            notify(str(exc), type="negative")
                            return False
                        finally:
                            upload_inspection_running = False
                            set_convert_enabled()

                    async def cleared(event: Any) -> None:
                        if conversion_running:
                            notify("A conversion is already running.", type="warning")
                            return
                        name = str(getattr(event, "name", ""))
                        if name:
                            removed_names = {name}
                        else:
                            removed_names = {
                                str(value)
                                for value in getattr(event, "names", ())
                                if str(value)
                            }
                        if removed_names:
                            retained_sources[:] = [
                                source
                                for source in retained_sources
                                if not (
                                    source.mode == "upload"
                                    and source.inspection.name in removed_names
                                )
                            ]
                        else:
                            retained_sources[:] = [
                                source for source in retained_sources if source.mode != "upload"
                            ]
                        active_source("upload")
                        await refresh_active_source_views()

                    with (
                        ui.tabs(value=settings["source_mode"])
                        .props("dense no-caps inline-label align=left breakpoint=0")
                        .classes("dn-data-manager-source-tabs")
                        .mark("data-manager-source-tabs")
                    ) as source_tabs:
                        ui.tab("upload", label="Upload", icon="upload_file")
                        ui.tab("url", label="Link", icon="link")

                    async def source_mode_changed(event: Any) -> None:
                        requested_mode = str(event.value)
                        if conversion_running or upload_inspection_running or remote_inspections:
                            source_tabs.set_value(source_mode())
                            notify(
                                "Wait for the current source operation to finish before changing "
                                "the source type.",
                                type="warning",
                            )
                            return
                        if requested_mode not in active_source_ids:
                            return
                        settings["source_mode"] = requested_mode
                        current = active_source()
                        if current is not None:
                            settings["output_name"] = current.output_name
                        await refresh_active_source_views()

                    source_tabs.on_value_change(source_mode_changed)
                    with (
                        ui.tab_panels(
                            source_tabs,
                            value=settings["source_mode"],
                        )
                        .classes("dn-data-manager-source-panels")
                        .mark("data-manager-source-panels")
                    ):
                        with ui.tab_panel("upload"):
                            source_controls["uploader"] = data_upload(
                                accept=f"{RDATA_UPLOAD_ACCEPT},.bed,.mat,.map",
                                accepted_formats=(
                                    "TSV, TXT, CSV, NPZ, bigWig, bedGraph"
                                ),
                                marker="data-manager-upload",
                                show_accepted_formats=True,
                                on_upload=uploaded,
                                on_clear=cleared,
                                multiple=True,
                                max_files=None,
                                hide_when_loaded=False,
                                number_sources=True,
                                file_details=lambda name: source_metadata_slot(f"upload:{name}"),
                                initial_names=tuple(
                                    source.inspection.name
                                    for source in sources_for_mode("upload")
                                ),
                            )

                        with ui.tab_panel("url"):

                            @ui.refreshable
                            def render_remote_links() -> None:
                                for key in tuple(source_controls):
                                    if key.startswith("remote-link-"):
                                        source_controls.pop(key, None)

                                for row_index, row_state in enumerate(remote_link_inputs):
                                    with (
                                        ui.element("div")
                                        .classes("dn-data-manager-url-item")
                                        .mark(f"data-manager-url-item-{row_state.row_id}")
                                    ):
                                        with ui.element("div").classes(
                                            "dn-data-manager-url-grid"
                                        ):
                                            url_input = (
                                                ui.input(
                                                    label=(
                                                        f"Source {row_index + 1} (public file link)"
                                                        if len(remote_link_inputs) > 1
                                                        else "Source (public file link)"
                                                    ),
                                                    value=row_state.url,
                                                    placeholder="https://.../signal.bigWig",
                                                )
                                                .props("dense clearable")
                                                .classes(
                                                    "dn-field dn-data-manager-url-input"
                                                )
                                                .mark(
                                                    "data-manager-source-url",
                                                    f"data-manager-source-url-{row_state.row_id}",
                                                )
                                            )
                                            inspect_button = (
                                                ui.button("✓" if row_state.source_id else "Inspect")
                                                .props(
                                                    'flat dense no-caps aria-label="Inspect public link"'
                                                )
                                                .classes(
                                                    "dn-action-secondary "
                                                    "dn-data-manager-link-action"
                                                )
                                                .mark(
                                                    "action-inspect-data-link",
                                                    f"action-inspect-data-link-{row_state.row_id}",
                                                )
                                            )
                                            if row_state.source_id:
                                                inspect_button.props(
                                                    'aria-label="Source inspected"'
                                                )
                                                inspect_button.disable()
                                            remove_button = (
                                                ui.button(icon="close")
                                                .props(
                                                    'flat round dense aria-label="Remove public link"'
                                                )
                                                .classes("dn-data-manager-link-remove")
                                                .mark(
                                                    f"remove-data-link-{row_state.row_id}"
                                                )
                                            )
                                        url_progress = (
                                            ui.linear_progress(
                                                value=0.0,
                                                show_value=False,
                                            )
                                            .props("indeterminate rounded")
                                            .classes("dn-data-manager-url-progress")
                                            .mark(
                                                f"data-manager-url-progress-{row_state.row_id}"
                                            )
                                        )
                                        url_progress.visible = False
                                        url_status = (
                                            ui.label("Downloading and inspecting...")
                                            .classes("dn-data-manager-url-status")
                                            .mark(
                                                f"data-manager-url-status-{row_state.row_id}"
                                            )
                                        )
                                        url_status.visible = False
                                        source_metadata_slot(f"url:{row_state.row_id}")

                                    async def inspect_url(
                                        row: _RemoteLinkInput = row_state,
                                        input_control: Any = url_input,
                                        action: Any = inspect_button,
                                        progress: Any = url_progress,
                                        status: Any = url_status,
                                        first_row: bool = row_index == 0,
                                        remove: Any = remove_button,
                                    ) -> None:
                                        if conversion_running:
                                            notify(
                                                "A conversion is already running.",
                                                type="warning",
                                            )
                                            return
                                        if row.row_id in remote_inspections or row.source_id is not None:
                                            return
                                        url = str(input_control.value or "").strip()
                                        row.url = url
                                        if first_row:
                                            settings["source_url"] = url
                                        if not url:
                                            notify(
                                                "Paste a public file link first.",
                                                type="warning",
                                            )
                                            return
                                        action.disable()
                                        input_control.disable()
                                        remove.disable()
                                        convert_button.disable()
                                        remote_inspections.add(row.row_id)
                                        progress.visible = True
                                        status.visible = True
                                        status.text = "Downloading and inspecting..."
                                        try:
                                            remote = await run.io_bound(
                                                fetch_remote_source,
                                                url,
                                            )
                                            if remote is None:
                                                raise RuntimeError(
                                                    "The link stopped loading before returning data."
                                                )
                                            candidate = await run.io_bound(
                                                inspect_upload,
                                                remote.filename,
                                                remote.content,
                                                content_type=remote.content_type or "",
                                                expected_form="auto",
                                            )
                                            if candidate is None:
                                                raise RuntimeError(
                                                    "Link inspection stopped before returning data."
                                                )
                                            source_id = f"url:{row.row_id}"
                                            adopted = await adopt_candidate(
                                                candidate,
                                                mode="url",
                                                source_id=source_id,
                                                metadata={
                                                    "requested_url": url,
                                                    "final_url": remote.final_url,
                                                    "size": remote.size,
                                                    "accept_ranges": remote.accept_ranges,
                                                    "etag": remote.etag,
                                                    "last_modified": remote.last_modified,
                                                },
                                            )
                                            if adopted:
                                                row.source_id = source_id
                                                action.set_text("✓")
                                                action.props(
                                                    'aria-label="Source inspected"'
                                                )
                                                notify(
                                                    "The source is ready.",
                                                    type="positive",
                                                )
                                        except (
                                            OSError,
                                            RemoteSourceError,
                                            RuntimeError,
                                            TypeError,
                                            ValueError,
                                        ) as exc:
                                            notify(str(exc), type="negative")
                                        finally:
                                            remote_inspections.discard(row.row_id)
                                            if not progress.is_deleted:
                                                progress.visible = False
                                            if not status.is_deleted:
                                                status.visible = False
                                            if not action.is_deleted:
                                                action.set_enabled(row.source_id is None and not conversion_running)
                                            if (
                                                not input_control.is_deleted
                                                and not conversion_running
                                            ):
                                                input_control.enable()
                                            if not remove.is_deleted and not conversion_running:
                                                remove.enable()
                                            set_convert_enabled()

                                    async def remove_link(
                                        row: _RemoteLinkInput = row_state,
                                    ) -> None:
                                        nonlocal next_remote_link_id
                                        if conversion_running or remote_inspections:
                                            notify(
                                                "Wait for the current link operation to finish.",
                                                type="warning",
                                            )
                                            return
                                        if row.source_id is not None:
                                            retained_sources[:] = [
                                                source
                                                for source in retained_sources
                                                if source.source_id != row.source_id
                                            ]
                                        if len(remote_link_inputs) == 1:
                                            row.url = ""
                                            row.source_id = None
                                        else:
                                            remote_link_inputs.remove(row)
                                        if not remote_link_inputs:
                                            next_remote_link_id += 1
                                            remote_link_inputs.append(
                                                _RemoteLinkInput(next_remote_link_id)
                                            )
                                        active_source("url")
                                        render_remote_links.refresh()
                                        await refresh_active_source_views()

                                    async def url_changed(
                                        row: _RemoteLinkInput = row_state,
                                        input_control: Any = url_input,
                                        action: Any = inspect_button,
                                    ) -> None:
                                        action.set_text("Inspect")
                                        action.props('aria-label="Inspect public link"')
                                        action.set_enabled(not conversion_running and row.row_id not in remote_inspections)
                                        new_url = str(input_control.value or "").strip()
                                        row.url = new_url
                                        if row.source_id is None:
                                            set_convert_enabled()
                                            return
                                        loaded = next(
                                            (
                                                source
                                                for source in retained_sources
                                                if source.source_id == row.source_id
                                            ),
                                            None,
                                        )
                                        if loaded is None:
                                            row.source_id = None
                                            set_convert_enabled()
                                            return
                                        retained_sources.remove(loaded)
                                        row.source_id = None
                                        active_source("url")
                                        await refresh_active_source_views()

                                    inspect_button.on_click(inspect_url)
                                    remove_button.on_click(remove_link)
                                    url_input.on_value_change(url_changed)
                                    url_input.on("keydown.enter", inspect_url)
                                    source_controls[
                                        f"remote-link-input-{row_state.row_id}"
                                    ] = url_input
                                    source_controls[
                                        f"remote-link-action-{row_state.row_id}"
                                    ] = inspect_button
                                    source_controls[
                                        f"remote-link-remove-{row_state.row_id}"
                                    ] = remove_button

                                def add_remote_link() -> None:
                                    nonlocal next_remote_link_id
                                    if conversion_running or remote_inspections:
                                        notify(
                                            "Wait for the current link operation to finish.",
                                            type="warning",
                                        )
                                        return
                                    if len(remote_link_inputs) >= _MAX_RETAINED_SOURCE_COUNT:
                                        notify(
                                            "Remove a link row before adding another one.",
                                            type="warning",
                                        )
                                        return
                                    next_remote_link_id += 1
                                    remote_link_inputs.append(
                                        _RemoteLinkInput(next_remote_link_id)
                                    )
                                    render_remote_links.refresh()

                                add_link_button = (
                                    ui.button(icon="add", on_click=add_remote_link)
                                    .props(
                                        'flat round dense aria-label="Add another public link"'
                                    )
                                    .classes("dn-data-manager-add-link")
                                    .mark("data-manager-add-link")
                                    .tooltip("Add another link")
                                )
                                source_controls["remote-link-add"] = add_link_button

                            render_remote_links()
                            ui.label(
                                "Paste public HTTPS links to genomic tracks. "
                                "Private-network and sign-in links are not supported."
                            ).classes("dn-data-manager-url-help")

                            async def use_link_example(
                                *,
                                url: str,
                                feature: str,
                                aggregation: str | None = None,
                                track_missing: str | None = None,
                            ) -> None:
                                if (
                                    conversion_running
                                    or upload_inspection_running
                                    or remote_inspections
                                ):
                                    notify(
                                        "Wait for the current source operation to finish.",
                                        type="warning",
                                    )
                                    return
                                first = remote_link_inputs[0]
                                if first.source_id is not None:
                                    retained_sources[:] = [
                                        source
                                        for source in retained_sources
                                        if source.source_id != first.source_id
                                    ]
                                    first.source_id = None
                                first.url = url
                                settings["source_url"] = url
                                _apply_feature_defaults(settings, feature)
                                if aggregation is not None:
                                    settings["aggregation"] = aggregation
                                if track_missing is not None:
                                    settings["track_missing"] = track_missing
                                feature_control = controls.get("feature")
                                if feature_control is not None:
                                    feature_control.value = feature
                                    feature_control.update()
                                for source in retained_sources:
                                    if not source.output_name_customized:
                                        source.output_name = _automatic_output_name(
                                            source.inspection.name,
                                            feature,
                                        )
                                first_input = source_controls.get(
                                    f"remote-link-input-{first.row_id}"
                                )
                                if first_input is not None:
                                    first_input.value = url
                                    first_input.update()
                                current = active_source("url")
                                if current is not None:
                                    settings["output_name"] = current.output_name
                                await refresh_active_source_views()

                            async def use_encode_example() -> None:
                                await use_link_example(
                                    url=_ENCODE_EXAMPLE_URL,
                                    feature="replication_timing",
                                    aggregation="mean",
                                    track_missing="zero",
                                )

                            async def use_bedgraph_example() -> None:
                                await use_link_example(
                                    url=_SNS_EXAMPLE_URL,
                                    feature="initiation_rate",
                                    aggregation="max",
                                    track_missing="zero",
                                )

                            async def use_okseq_example() -> None:
                                await use_link_example(
                                    url=_OKSEQ_EXAMPLE_URL,
                                    feature="fork_directionality",
                                    aggregation="mean",
                                    track_missing="zero",
                                )

                            async def use_h1_repliseq_example() -> None:
                                settings["repliseq_timing_method"] = "sigmoid"
                                settings["s_phase_duration_min"] = 600.0
                                await use_link_example(
                                    url=_H1_REPLISEQ_EXAMPLE_URL,
                                    feature="replication_timing",
                                )

                            with (
                                ui.column()
                                .classes("dn-data-manager-example-links")
                                .mark("data-manager-link-examples")
                            ) as link_examples:
                                encode_example_button = (
                                    ui.button(
                                        "Example: ENCODE Repli-seq (bigWig) - K562",
                                        icon="science",
                                        on_click=use_encode_example,
                                    )
                                    .props("flat dense no-caps")
                                    .classes("dn-data-manager-example-link")
                                    .mark("use-encode-data-example")
                                    .tooltip(
                                        "UCSC ENCODE DCC K562 hg19 Repli-seq wavelet timing track"
                                    )
                                )
                                bedgraph_example_button = (
                                    ui.button(
                                        "Example: NCBI GEO SNS-seq (BED) - K562",
                                        icon="science",
                                        on_click=use_bedgraph_example,
                                    )
                                    .props("flat dense no-caps")
                                    .classes("dn-data-manager-example-link")
                                    .mark("use-bedgraph-data-example")
                                    .tooltip(
                                        "GEO GSE28911 K562 hg19 SNS/control enrichment peaks"
                                    )
                                )
                                okseq_example_button = (
                                    ui.button(
                                        "Example: OKseqHMM OK-seq (bigWig) - K562",
                                        icon="science",
                                        on_click=use_okseq_example,
                                    )
                                    .props("flat dense no-caps")
                                    .classes("dn-data-manager-example-link")
                                    .mark("use-okseq-data-example")
                                    .tooltip(
                                        "OKseqHMM K562 hg19 fork directionality, 1 kb bins, "
                                        "read-count cutoff 10; uncovered bins remain missing"
                                    )
                                )
                                h1_example_button = (
                                    ui.button(
                                        "Example: GEO 16-fraction Repli-seq (MAT) - H1",
                                        icon="science", on_click=use_h1_repliseq_example,
                                    )
                                    .props("flat dense no-caps")
                                    .classes("dn-data-manager-example-link")
                                    .mark("use-h1-repliseq-data-example")
                                    .tooltip("GSE137764 H1 hESCs: S1–S16, hg38, 50 kb bins; extract chromosome-specific timing")
                                )
                            link_examples.visible = get_export_settings().show_examples
                            source_controls["encode-example"] = encode_example_button
                            source_controls["bedgraph-example"] = bedgraph_example_button
                            source_controls["okseq-example"] = okseq_example_button
                            source_controls["h1-repliseq-example"] = h1_example_button

                    @ui.refreshable
                    def render_source_details() -> None:
                        refresh_source_metadata()
                        for key in tuple(source_controls):
                            if key.startswith("source-detail-"):
                                source_controls.pop(key, None)
                        current_source = active_source()
                        if current_source is None:
                            return

                        options = _selection_options(current_source.inspection)
                        if (
                            len(options) > 1
                            and current_source.inspection.source_format != "bedgraph"
                        ):
                            profile_select = (
                                ui.select(
                                    dict(options),
                                    value=current_source.selection,
                                    label="Profile",
                                )
                                .props("dense options-dense")
                                .classes("dn-field dn-data-manager-source-select")
                                .mark(
                                    "data-manager-source-selection",
                                    f"data-manager-source-selection-{current_source.source_id}",
                                )
                            )
                            profile_select._props[
                                "data-source-id"
                            ] = current_source.source_id

                            async def profile_changed() -> None:
                                if profile_select.value is None:
                                    return
                                current_source.selection = str(profile_select.value)
                                if current_source.selection == _REPLISEQ_TIMING:
                                    _apply_feature_defaults(settings, "replication_timing")
                                    if not current_source.output_name_customized:
                                        current_source.output_name = _automatic_output_name(
                                            current_source.inspection.name, "replication_timing",
                                        )
                                    settings["output_name"] = current_source.output_name
                                await refresh_active_source_views()

                            profile_select.on_value_change(profile_changed)
                            source_controls["source-detail-profile"] = profile_select

                        if conversion_running:
                            lock_source_controls(True)

                    render_source_details()

                # Keep this slot outside Configure's refreshable subtree; it is
                # mounted below the formulas and rebuilt for each source.
                value_transform_controls = ui.element("div").classes("w-full")

                with expansion("Configure", icon="tune", expanded=True):

                    @ui.refreshable
                    def render_conversion_settings() -> None:
                        value_transform_controls.clear()
                        controls.clear()
                        candidates = sources_for_mode()
                        current_source = active_source()

                        with ui.row().classes("dn-inline-actions").mark("data-manager-configuration-actions"):
                            controls["apply_configuration"] = (
                                ui.switch("Apply configuration", value=bool(settings["apply_configuration"]))
                                .mark("data-manager-apply-configuration")
                            )
                            controls["apply_configuration"].on_value_change(
                                lambda: store_control("apply_configuration")
                            )
                            controls["show_preview"] = (
                                ui.switch("Show plot", value=bool(settings["show_preview"]))
                                .classes("dn-data-manager-preview-toggle")
                                .mark("data-manager-preview-toggle")
                            )
                            if len(candidates) > 1:
                                controls["apply_configuration_to_all"] = (
                                    ui.switch("Apply to all", value=apply_configuration_to_all[source_mode()])
                                    .mark("data-manager-apply-configuration-to-all")
                                )

                                async def configuration_scope_changed() -> None:
                                    save_source_configuration()
                                    apply_configuration_to_all[source_mode()] = bool(
                                        controls["apply_configuration_to_all"].value
                                    )
                                    save_source_configuration()
                                    await refresh_active_source_views()

                                controls["apply_configuration_to_all"].on_value_change(
                                    configuration_scope_changed
                                )

                        if len(candidates) > 1 and not apply_configuration_to_all[source_mode()]:
                            controls["configuration_source"] = (
                                ui.select(
                                    {
                                        source.source_id: f"Source {index}"
                                        for index, source in enumerate(candidates, 1)
                                    },
                                    value=current_source.source_id if current_source else None,
                                    label="Source",
                                ).props("dense options-dense").classes("dn-field")
                                .mark("data-manager-source-dataset", "data-manager-configuration-source")
                            )

                            async def configuration_source_changed(event: Any) -> None:
                                save_source_configuration()
                                active_source_ids[source_mode()] = str(event.value)
                                await refresh_active_source_views()

                            controls["configuration_source"].on_value_change(configuration_source_changed)

                        def store_control(name: str) -> None:
                            settings[name] = controls[name].value
                            if name in {"start", "end", "bin_size"}:
                                source = active_source()
                                if source is not None and selected_data(source).form in {"x", "xt"}:
                                    if name in {"start", "end"}:
                                        settings["profile_region_custom"] = True
                                    else:
                                        settings["profile_bin_custom"] = True
                            if name in {"bin_size", "chromosome"}:
                                refresh_source_metadata()
                            if name in {
                                "threshold_min", "threshold_max", "value_scaling", "scale",
                                "rescale_min", "rescale_max", "fill_value", "track_missing", "missing_policy",
                            }:
                                sync_prepared_feature()

                        def update_source_names() -> None:
                            for source in (
                                sources_for_mode() if apply_configuration_to_all[source_mode()] else [active_source()]
                            ):
                                if source is None:
                                    continue
                                if not source.output_name_customized:
                                    source.output_name = _automatic_output_name(
                                        source.inspection.name, str(settings["feature"]),
                                    )
                            current = active_source()
                            if current is not None:
                                settings["output_name"] = current.output_name

                        def sync_prepared_feature() -> None:
                            feature = str(settings["feature"])
                            if _feature_settings_compatible(
                                feature, settings,
                                missing_key="track_missing" if "track_missing" in controls else "missing_policy",
                            ):
                                return
                            # Preserve implicit preset bounds and every edited
                            # setting; changing the label must not reapply defaults.
                            enabled, lower, upper = _resolved_feature_bounds(
                                feature, settings["threshold_min"], settings["threshold_max"],
                            )
                            settings.update(
                                feature="generic", threshold_enabled=enabled,
                                threshold_min=lower, threshold_max=upper,
                            )
                            controls["feature"].set_value("generic")
                            for key in ("threshold_min", "threshold_max"):
                                if key in controls:
                                    controls[key].set_value(settings[key])
                            if "value_scaling" in controls:
                                controls["value_scaling"].set_options(
                                    _value_scaling_options("generic"), value=settings["value_scaling"],
                                )
                            update_source_names()

                        def feature_changed() -> None:
                            feature = str(controls["feature"].value or "generic")
                            if feature == settings["feature"]:
                                return
                            current = active_source()
                            _apply_feature_defaults(
                                settings, feature,
                                current.inspection if current else None,
                                current.track_inspection.backend
                                if current and current.track_inspection else None,
                            )
                            update_source_names()
                            render_conversion_settings.refresh()

                        with (
                            ui.element("div")
                            .classes("dn-data-manager-config-grid")
                            .mark("data-manager-config-grid")
                        ):
                            controls["feature"] = (
                                ui.select(
                                    _FEATURES,
                                    value=settings["feature"],
                                    label="Prepare as",
                                )
                                .props("dense options-dense")
                                .classes("dn-field dn-data-manager-feature-select")
                                .mark("data-manager-feature")
                            )
                        controls["feature"].on_value_change(feature_changed)

                        def preview_changed() -> None:
                            store_control("show_preview")
                            render_result.refresh()

                        controls["show_preview"].on_value_change(preview_changed)

                        current_source = active_source()
                        if current_source is None:
                            ui.label(
                                "Add a source to reveal only the controls that apply to it."
                            ).classes("dn-data-manager-section-guidance")
                            return
                        details = selected_data(current_source)

                        def render_decimal_places() -> None:
                            controls["decimal_places"] = (
                                ui.number(
                                    label="Decimal places",
                                    value=settings["decimal_places"],
                                    min=0,
                                    max=15,
                                    step=1,
                                )
                                .props("dense")
                                .classes("dn-field")
                                .mark("data-manager-decimal-places")
                            )
                            controls["decimal_places"].on_value_change(
                                lambda: store_control("decimal_places")
                            )

                        def render_bound_controls() -> None:
                            with (
                                ui.element("div")
                                .classes("dn-data-manager-threshold-grid")
                                .mark("data-manager-threshold-fields")
                            ):
                                for key, label, marker in (
                                    ("threshold_min", "Lower bound", "data-manager-threshold-min"),
                                    ("threshold_max", "Upper bound", "data-manager-threshold-max"),
                                ):
                                    controls[key] = (
                                        ui.number(label=label, value=settings[key])
                                        .props("dense clearable").classes("dn-field").mark(marker)
                                    )

                        def render_value_controls(
                            *,
                            missing_kind: str,
                            polishable: bool = True,
                        ) -> None:
                            with (
                                ui.element("div")
                                .classes(
                                    "dn-data-manager-polishing-grid "
                                    "dn-data-manager-values-grid"
                                    + (" dn-data-manager-values-basic" if not polishable else "")
                                )
                                .mark("data-manager-polishing-grid")
                            ):
                                render_decimal_places()
                                with (
                                    ui.element("div")
                                    .classes("dn-data-manager-polishing-column")
                                    .mark("data-manager-missing-column")
                                ):
                                    missing_key = "track_missing" if missing_kind == "track" else "missing_policy"
                                    controls[missing_key] = (
                                        ui.select(
                                            _TRACK_MISSING if missing_kind == "track" else _PROFILE_MISSING,
                                            value=settings[missing_key], label="Missing values",
                                        )
                                        .props("dense options-dense")
                                        .classes("dn-field")
                                        .mark(f"data-manager-{missing_kind}-missing")
                                    )
                                    controls["fill_value"] = (
                                        ui.number(label="Fill value", value=settings["fill_value"])
                                        .props("dense").classes("dn-field").mark("data-manager-fill-value")
                                    )
                                    controls["fill_value"].set_visibility(settings[missing_key] == "fill")

                                    def missing_changed() -> None:
                                        store_control(missing_key)
                                        controls["fill_value"].set_visibility(controls[missing_key].value == "fill")

                                    controls[missing_key].on_value_change(missing_changed)
                                    controls["fill_value"].on_value_change(lambda: store_control("fill_value"))
                                    if missing_kind == "track":
                                        interpolation_options = _track_interpolation_options(
                                            current_source.inspection
                                        )
                                        if settings["interpolation"] not in interpolation_options:
                                            settings["interpolation"] = "none"
                                        controls["interpolation"] = (
                                            ui.select(
                                                interpolation_options,
                                                value=settings["interpolation"],
                                                label="Interpolation",
                                            )
                                            .props("dense options-dense")
                                            .classes("dn-field")
                                            .mark("data-manager-interpolation")
                                        )
                                        if len(interpolation_options) == 1:
                                            controls["interpolation"].disable()
                                            field_hint(controls["interpolation"], "Keep gaps between called origins as background.")
                                        controls["interpolation"].on_value_change(
                                            lambda: store_control("interpolation")
                                        )

                                if polishable:
                                    render_bound_controls()

                            if not polishable:
                                return

                            with value_transform_controls, (
                                ui.element("div")
                                .classes("dn-data-manager-polishing-grid dn-data-manager-transform-grid")
                                .mark("data-manager-transform-values")
                            ):
                                with (
                                    ui.element("div")
                                    .classes("dn-data-manager-polishing-column")
                                    .mark("data-manager-smoothing-column")
                                ):
                                    controls["smoothing_mode"] = (
                                        ui.select(
                                            _SMOOTHING_MODES,
                                            value=settings["smoothing_mode"],
                                            label="Smoothing",
                                        )
                                        .props("dense options-dense")
                                        .classes("dn-field")
                                        .mark("data-manager-smoothing")
                                    )
                                    controls["smooth_window"] = (
                                        ui.number(
                                            label="Window (bins)",
                                            value=settings["smooth_window"],
                                            min=2,
                                            step=1,
                                        )
                                        .props("dense")
                                        .classes("dn-field")
                                        .mark("data-manager-smooth-window")
                                    )

                                with (
                                    ui.element("div")
                                    .classes("dn-data-manager-polishing-column")
                                    .mark("data-manager-scaling-column")
                                ):
                                    scaling_options = _value_scaling_options(str(settings["feature"]))
                                    if settings["value_scaling"] not in scaling_options:
                                        settings["value_scaling"] = "none"
                                    controls["value_scaling"] = (
                                        ui.select(
                                            scaling_options,
                                            value=settings["value_scaling"],
                                            label="Value scaling",
                                        )
                                        .props("dense options-dense")
                                        .classes("dn-field")
                                        .mark("data-manager-value-scaling")
                                    )

                                    with (
                                        ui.element("div")
                                        .classes("dn-data-manager-multiplier-grid")
                                        .mark("data-manager-multiplier-fields")
                                    ) as multiplier_fields:
                                        controls["scale"] = (
                                            ui.number(
                                                label="Multiplier",
                                                value=settings["scale"],
                                            )
                                            .props("dense")
                                            .classes("dn-field")
                                            .mark("data-manager-scale-multiplier")
                                        )

                                    with (
                                        ui.element("div")
                                        .classes("dn-data-manager-range-grid")
                                        .mark("data-manager-target-interval")
                                    ) as interval_fields:
                                        controls["rescale_min"] = (
                                            ui.number(
                                                label="Minimum",
                                                value=settings["rescale_min"],
                                            )
                                            .props("dense")
                                            .classes("dn-field")
                                            .mark("data-manager-target-minimum")
                                        )
                                        controls["rescale_max"] = (
                                            ui.number(
                                                label="Maximum",
                                                value=settings["rescale_max"],
                                            )
                                            .props("dense")
                                            .classes("dn-field")
                                            .mark("data-manager-target-maximum")
                                        )
                                        field_hint(controls["rescale_min"], "Lowest value after rescaling.")
                                        field_hint(controls["rescale_max"], "Highest value after rescaling.")
                                controls["rescale_invert"] = (
                                    ui.checkbox(
                                        "Invert",
                                        value=bool(settings["rescale_invert"]),
                                    )
                                    .classes("dn-data-manager-inline-toggle")
                                    .mark("data-manager-interval-invert")
                                )
                                field_hint(controls["rescale_invert"], "Reverse high and low signal values.")

                            smoothing_enabled = settings["smoothing_mode"] == "moving_average"
                            if smoothing_enabled and int(settings["smooth_window"] or 1) < 2:
                                settings["smooth_window"] = 3
                                controls["smooth_window"].value = 3
                            controls["smooth_window"].set_visibility(smoothing_enabled)
                            multiplier_fields.set_visibility(
                                settings["value_scaling"] == "multiplier"
                            )
                            interval_fields.set_visibility(
                                settings["value_scaling"] == "interval"
                            )

                            def smoothing_changed() -> None:
                                store_control("smoothing_mode")
                                enabled = (
                                    str(controls["smoothing_mode"].value) == "moving_average"
                                )
                                if enabled and int(controls["smooth_window"].value or 1) < 2:
                                    controls["smooth_window"].value = 3
                                    controls["smooth_window"].update()
                                    store_control("smooth_window")
                                elif not enabled:
                                    controls["smooth_window"].value = 1
                                    controls["smooth_window"].update()
                                    store_control("smooth_window")
                                controls["smooth_window"].set_visibility(enabled)

                            def value_scaling_changed() -> None:
                                store_control("value_scaling")
                                selected = str(controls["value_scaling"].value)
                                multiplier_fields.set_visibility(selected == "multiplier")
                                interval_fields.set_visibility(
                                    selected == "interval"
                                )

                            controls["smoothing_mode"].on_value_change(smoothing_changed)
                            controls["value_scaling"].on_value_change(value_scaling_changed)
                            for name in (
                                "smooth_window",
                                "scale",
                                "rescale_min",
                                "rescale_max",
                                "rescale_invert",
                                "threshold_min",
                                "threshold_max",
                            ):
                                controls[name].on_value_change(
                                    lambda name=name: store_control(name)
                                )

                        if current_source.selection == _REPLISEQ_TIMING:
                            summary = current_source.track_inspection
                            assert summary is not None
                            chromosome_names = [region.chromosome for region in summary.regions]
                            if settings["chromosome"] not in chromosome_names:
                                settings["chromosome"] = chromosome_names[0]
                            ui.label("Timing extraction").classes("dn-setting-group-title")
                            with ui.element("div").classes("dn-data-manager-binning-grid"):
                                controls["chromosome"] = (
                                    ui.select(chromosome_names, value=settings["chromosome"], label="Chromosome")
                                    .props("dense options-dense").classes("dn-field").mark("data-manager-chromosome")
                                )
                                controls["repliseq_timing_method"] = (
                                    ui.select({"sigmoid": "Sigmoid fit (T50)", "mean": "Signal-weighted mean"},
                                              value=settings["repliseq_timing_method"], label="Timing method")
                                    .props("dense options-dense").classes("dn-field").mark("data-manager-repliseq-timing-method")
                                )
                                controls["s_phase_duration_min"] = (
                                    ui.number(label="S-phase duration (min)", value=settings["s_phase_duration_min"], min=1, step=30)
                                    .props("dense").classes("dn-field").mark("data-manager-s-phase-duration")
                                )
                            native_bin = int(dict(summary.metadata)["native_bin_size_bp"])
                            ui.label(
                                f"16 equal S-phase intervals; {native_bin / 1000:g} kb bins. "
                                "T50 fits the cumulative signal and takes 50% replication. "
                                "Incomplete or empty bins remain missing."
                            ).classes("dn-data-manager-section-guidance")
                            async def repliseq_chromosome_changed() -> None:
                                store_control("chromosome")
                                await refresh_active_source_views()

                            controls["chromosome"].on_value_change(repliseq_chromosome_changed)
                            for name in ("repliseq_timing_method", "s_phase_duration_min"):
                                controls[name].on_value_change(lambda name=name: store_control(name))

                        if (
                            details.form != "track"
                            or current_source.track_inspection is None
                        ):
                            if details.form in {"x", "xt"} and current_source.selection != _REPLISEQ_TIMING:
                                length = (details.shape[0] if len(details.shape) == 2 and (details.form == "xt" or 1 not in details.shape)
                                          else int(np.prod(details.shape)))
                                _rows, chromosomes = spatial_profile_rows(current_source.inspection)
                                if chromosomes and settings["chromosome"] not in chromosomes:
                                    settings["chromosome"] = chromosomes[0]
                                positions, intervals, supplied = spatial_profile_grid(
                                    current_source.inspection, length,
                                    source_bin_size=settings["source_bin_size"],
                                    chromosome=settings["chromosome"] if chromosomes else None,
                                )
                                native_bin_size = max(1, int(round(np.median(intervals[:, 1] - intervals[:, 0]) * 1000)))
                                coordinate_names = {str(name).lower().replace(" ", "_") for name in current_source.inspection.columns}
                                intrinsic_resolution = supplied and (
                                    len(positions) > 1
                                    or any({start, end} <= coordinate_names for start, end in (
                                        ("start", "end"), ("start_bp", "end_bp"), ("start_kb", "end_kb"),
                                    ))
                                )
                                if intrinsic_resolution:
                                    settings["source_bin_size"] = native_bin_size
                                full_start = int(round(intervals[0, 0] * 1000))
                                full_end = int(round(intervals[-1, 1] * 1000))
                                grid_key = (current_source.source_id, current_source.selection, settings["chromosome"] if chromosomes else None)
                                if settings["_profile_grid_key"] != grid_key:
                                    previous = settings["_profile_grid_key"]
                                    if previous is not None and previous[0] == current_source.source_id:
                                        settings.update(profile_region_custom=False, profile_bin_custom=False)
                                    settings["_profile_grid_key"] = grid_key
                                if not settings["profile_region_custom"]:
                                    settings.update(start=full_start, end=full_end)
                                if not settings["profile_bin_custom"]:
                                    settings["bin_size"] = native_bin_size

                                def reset_profile_region() -> None:
                                    controls["start"].set_value(full_start)
                                    controls["end"].set_value(full_end)
                                    settings.update(start=full_start, end=full_end, profile_region_custom=False)

                                ui.label("Region & binning").classes("dn-setting-group-title")
                                with ui.element("div").classes("dn-data-filter-grid"):
                                    if len(chromosomes) > 1:
                                        controls["chromosome"] = ui.select(
                                            list(chromosomes), value=settings["chromosome"], label="Chromosome",
                                        ).props("dense options-dense").classes("dn-field").style("grid-column: 1 / -1").mark("data-manager-chromosome")
                                    controls["source_bin_size"] = (
                                        ui.number("Source resolution (bp)", value=settings["source_bin_size"], min=1, step=1)
                                        .props("dense").classes("dn-field").mark("data-manager-source-bin-size")
                                    )
                                    controls["source_bin_size"].set_enabled(not intrinsic_resolution)
                                    controls["bin_size"] = (
                                        ui.number("Bin size (bp)", value=settings["bin_size"], min=1, step=1)
                                        .props("dense").classes("dn-field").mark("data-manager-bin-size")
                                    )
                                    controls["start"] = (
                                        ui.number("Start (bp)", value=settings["start"], min=0, step=1)
                                        .props("dense").classes("dn-field").mark("data-manager-start")
                                    )
                                    with ui.element("div").classes("dn-data-manager-end-field"):
                                        controls["end"] = (
                                            ui.number("End (bp)", value=settings["end"], min=1, step=1)
                                            .props("dense").classes("dn-field").mark("data-manager-end")
                                        )
                                        with controls["end"].add_slot("append"):
                                            controls["reset_region"] = (
                                                ui.button(icon=RESET_ICON, color=None, on_click=reset_profile_region)
                                                .props('flat round dense aria-label="Reset to full region"')
                                                .classes("dn-reset-button dn-data-manager-region-reset")
                                                .mark("data-manager-reset-region")
                                                .tooltip("Use the full source region")
                                            )

                                def source_grid_changed(name: str) -> None:
                                    value = controls[name].value
                                    if value is None or (name != "chromosome" and value < (1 if name == "source_bin_size" else 0)):
                                        return
                                    store_control(name)
                                    settings["_profile_grid_key"] = None
                                    settings.update(profile_region_custom=False, profile_bin_custom=False)
                                    render_conversion_settings.refresh()

                                for name in ("source_bin_size", "chromosome"):
                                    if name in controls:
                                        controls[name].on_value_change(lambda _event, name=name: source_grid_changed(name))
                                for name in ("start", "end", "bin_size"):
                                    controls[name].on_value_change(lambda _event, name=name: store_control(name))
                            ui.label("Values").classes(
                                "dn-setting-group-title dn-data-manager-setting-heading"
                            )
                            render_value_controls(
                                missing_kind="profile", polishable=details.form in _POLISHABLE_FORMS,
                            )
                            if details.form == "table":
                                ui.label("Table columns and labels will be preserved.").classes(
                                    "dn-data-manager-section-guidance"
                                )
                            return

                        if current_source.inspection.source_format == "bedgraph":
                            is_peak_track = current_source.inspection.track_kind in {"peaks", "intervals"}
                            ui.label("BED intervals" if is_peak_track else "bedGraph input").classes(
                                "dn-setting-group-title dn-data-manager-setting-heading"
                            )
                            bedgraph_options = {
                                f"{_COLUMN_PREFIX}{column}": column.replace("_", " ").capitalize()
                                for column in current_source.inspection.numeric_columns
                            }
                            controls["bedgraph_value_column"] = (
                                ui.select(
                                    dict(bedgraph_options),
                                    value=current_source.selection,
                                    label="Signal column",
                                )
                                .props("dense options-dense")
                                .classes("dn-field dn-data-manager-bedgraph-column")
                                .mark("data-manager-bedgraph-value-column")
                            )
                            if len(bedgraph_options) == 1:
                                controls["bedgraph_value_column"].disable()

                            def bedgraph_column_changed() -> None:
                                selected = controls["bedgraph_value_column"].value
                                if selected is not None:
                                    current_source.selection = str(selected)

                            controls["bedgraph_value_column"].on_value_change(
                                bedgraph_column_changed
                            )
                            if current_source.inspection.track_kind != "peaks":
                                ui.label(
                                    "BED3 calls have unit signal. Presence marks bins containing "
                                    "origins; coverage measures the fraction overlapped. "
                                    "Smaller bins separate nearby origins."
                                    if current_source.inspection.track_kind == "intervals" else
                                    "Intervals use standard 0-based, half-open bedGraph "
                                    "coordinates. Bin aggregation below controls how interval "
                                    "coverage is combined."
                                ).classes(
                                    "dn-data-manager-section-guidance "
                                    "dn-data-manager-bedgraph-guidance"
                                )

                        ui.label("Region & binning").classes("dn-setting-group-title")
                        regions = {
                            item.chromosome: item
                            for item in current_source.track_inspection.regions
                        }
                        chromosome = str(settings["chromosome"] or next(iter(regions)))
                        if chromosome not in regions:
                            chromosome = next(iter(regions))
                        region = regions[chromosome]
                        aggregation_options = _track_aggregation_options(
                            current_source.track_inspection.backend, current_source.inspection
                        )
                        if settings["aggregation"] not in aggregation_options:
                            settings["aggregation"] = _default_track_aggregation(
                                current_source.inspection, str(settings["feature"]),
                                current_source.track_inspection.backend,
                            )
                        with ui.element("div").classes("dn-data-manager-binning-grid"):
                            controls["chromosome_scope"] = (
                                ui.select(
                                    {
                                        "single": "One chromosome",
                                        "all": "All chromosomes",
                                    },
                                    value=settings["chromosome_scope"],
                                    label="Scope",
                                )
                                .props("dense options-dense")
                                .classes("dn-field")
                                .mark("data-manager-chromosome-scope")
                            )
                            controls["bin_size"] = (
                                ui.number(
                                    label="Bin size (bp)",
                                    value=settings["bin_size"],
                                    min=1,
                                    step=1_000,
                                )
                                .props("dense")
                                .classes("dn-field")
                                .mark("data-manager-bin-size")
                            )
                            controls["aggregation"] = (
                                ui.select(
                                    aggregation_options,
                                    value=settings["aggregation"],
                                    label="Aggregation",
                                )
                                .props("dense options-dense")
                                .classes("dn-field dn-data-manager-aggregation")
                                .mark("data-manager-aggregation")
                            )

                        with (
                            ui.element("div")
                            .classes("dn-data-manager-single-region")
                            .mark("data-manager-single-region")
                        ) as single_region_fields:
                            controls["chromosome"] = (
                                ui.select(
                                    {name: name for name in regions},
                                    value=chromosome,
                                    label="Chromosome",
                                )
                                .props("dense options-dense")
                                .classes("dn-field")
                                .mark("data-manager-chromosome")
                            )
                            controls["start"] = (
                                ui.number(
                                    label="Start (bp)",
                                    value=settings["start"],
                                    min=region.start,
                                    max=region.end - 1,
                                    step=1_000,
                                )
                                .props("dense")
                                .classes("dn-field")
                                .mark("data-manager-start")
                            )
                            with ui.element("div").classes("dn-data-manager-end-field"):
                                controls["end"] = (
                                    ui.number(
                                        label="End (bp)", value=settings["end"],
                                        min=region.start + 1, max=region.end, step=1_000,
                                    )
                                    .props("dense").classes("dn-field").mark("data-manager-end")
                                )
                                with controls["end"].add_slot("append"):
                                    controls["reset_region"] = (
                                        ui.button(icon=RESET_ICON, color=None, on_click=lambda: reset_region())
                                        .props('flat round dense aria-label="Reset to whole chromosome"')
                                        .classes("dn-reset-button dn-data-manager-region-reset")
                                        .mark("data-manager-reset-region")
                                        .tooltip("Reset Start and End to the whole chromosome")
                                    )

                        single_region_fields.set_visibility(
                            settings["chromosome_scope"] == "single"
                        )
                        all_scope_note = (
                            ui.label("Every chromosome will use its full available extent.")
                            .classes("dn-data-manager-section-guidance dn-data-manager-scope-note")
                            .mark("data-manager-all-scope-note")
                        )
                        all_scope_note.set_visibility(settings["chromosome_scope"] == "all")

                        def scope_changed() -> None:
                            scope = str(controls["chromosome_scope"].value or "single")
                            settings["chromosome_scope"] = scope
                            single_region_fields.set_visibility(scope == "single")
                            all_scope_note.set_visibility(scope == "all")

                        def reset_region() -> None:
                            selected = str(controls["chromosome"].value)
                            selected_region = regions[selected]
                            settings.update(
                                chromosome=selected,
                                start=selected_region.start,
                                end=selected_region.end,
                            )
                            controls["start"].props(
                                f"min={selected_region.start} max={selected_region.end - 1}"
                            )
                            controls["end"].props(
                                f"min={selected_region.start + 1} max={selected_region.end}"
                            )
                            controls["start"].set_value(selected_region.start)
                            controls["end"].set_value(selected_region.end)

                        controls["chromosome_scope"].on_value_change(scope_changed)
                        controls["chromosome"].on_value_change(reset_region)
                        for name in (
                            "start",
                            "end",
                            "bin_size",
                            "aggregation",
                        ):
                            controls[name].on_value_change(lambda name=name: store_control(name))
                        ui.label("Values").classes(
                            "dn-setting-group-title dn-data-manager-setting-heading"
                        )
                        render_value_controls(missing_kind="track")

                    render_conversion_settings()

                transform_controls = DataTransformControls(
                    lambda: [(source.source_id, source.inspection.name) for source in sources_for_mode()],
                    value_controls=value_transform_controls,
                )
                filter_controls = DataFilterControls(
                    set_convert_enabled, allow_export=True, allow_complement=True,
                    profile_units="bp", align_modulation_from_start=True,
                )

                with action_dock(on_code=lambda: download_data_manager_code()):
                    convert_button = (
                        workflow_action_button("Convert", symbol="data")
                        .props("no-caps")
                        .classes("dn-action-primary")
                        .mark("action-convert")
                    )
                    if not sources_for_mode():
                        convert_button.disable()

        async def download_preview_plot(file_format: str) -> None:
            selected = selected_generated_result()
            chart = result_chart
            if selected is None or chart is None:
                notify("Convert data before downloading a plot.", type="warning")
                return
            try:
                dpi = int(get_export_settings().plot_dpi)
                capture_format = "png" if str(file_format).casefold() == "png" else "svg"
                capture_options: dict[str, Any] = {
                    "type": capture_format,
                    "backgroundColor": "#ffffff",
                }
                if capture_format == "png":
                    capture_options["pixelRatio"] = max(1.0, min(4.0, dpi / 96.0))
                data_uri = await chart.run_chart_method(
                    "getDataURL",
                    capture_options,
                    timeout=10,
                )
                artifact = await run.io_bound(
                    build_chart_capture_export,
                    data_uri,
                    file_format,
                    filename=f"{PurePath(selected.artifact.filename).stem}_preview",
                    dpi=dpi,
                )
                if artifact is None or not artifact.data:
                    raise RuntimeError("Plot export stopped before returning data.")
            except (RuntimeError, TimeoutError, TypeError, ValueError) as exc:
                notify(str(exc), type="negative")
                return
            cache_artifact(
                artifact,
                source="Data Manager",
                kind="plot",
                detail="Prepared profile",
                navigate_to="data-management",
            )
            ui.download(artifact.data, artifact.filename, artifact.media_type)

        with ui.column().classes("dn-preview-stack"):
            with panel(extra_classes="dn-preview-card dn-data-manager-output-card"):
                panel_title("Generated files")
                conversion_progress = (
                    ui.linear_progress(value=0.0, show_value=False)
                    .props("rounded instant-feedback")
                    .classes("dn-simulation-progress dn-data-manager-progress")
                    .mark("data-manager-progress")
                )
                conversion_progress.visible = False
                conversion_status = (
                    ui.label("Ready")
                    .classes("dn-muted dn-data-manager-status")
                    .mark("data-manager-status")
                )
                with ui.element("div").classes("w-full").mark("data-manager-result-container"):
                    preview_result_id: int | None = None
                    y_limits_automatic = True
                    updating_plot_settings = False

                    def preview_values() -> Any | None:
                        selected = selected_generated_result()
                        return selected.preview.values if selected is not None and selected.preview is not None else None

                    def filter_applied() -> bool:
                        selected = selected_generated_result()
                        return bool(
                            filter_controls.controls["enabled"].value
                            and selected is not None and "filter_action" in dict(selected.metadata)
                            and selected.preview is not None and selected.preview.filter_selection is not None
                        )

                    def redraw_prepared_preview() -> None:
                        nonlocal updating_plot_settings
                        if updating_plot_settings or result_chart is None:
                            return
                        sync_plot_visibility()
                        selected = selected_generated_result()
                        if selected is None or selected.preview is None:
                            return
                        plot_type = "Histogram" if plot_settings.view.value == "Histogram" else "Line"
                        if not plotting_settings_ready(plot_settings, plot_type=plot_type):
                            return
                        try:
                            options = _preview_options(
                                selected.preview, plot_settings,
                                automatic_y_limits=y_limits_automatic,
                            )
                            if plot_type == "Line":
                                options["series"][0]["data"] = _converted_preview_data(selected, log_x=bool(plot_settings.log_x.value))
                                if plot_settings.show_filter.value and filter_applied():
                                    name = str(plot_settings.filter_name.value or "").strip() or "Mask"
                                    if name == options["series"][0]["name"]:
                                        name += " (mask)"
                                    options["series"].append({
                                        "name": name, "type": "line", "data": [], "silent": True,
                                        "showSymbol": False, "tooltip": {"show": False},
                                        "itemStyle": {"color": "#b5b5b5"}, "lineStyle": {"opacity": 0},
                                        "markArea": {
                                            "silent": True, "label": {"show": False},
                                            "itemStyle": {"color": "rgba(128, 128, 128, 0.16)"},
                                            "emphasis": {"disabled": True},
                                            "data": _FilterPreviewRegions(selected.preview.filter_selection),
                                        },
                                    })
                                    options["legend"]["data"].append({"name": name, "icon": "roundRect"})
                            updating_plot_settings = True
                            try:
                                if y_limits_automatic:
                                    for endpoint in ("minimum", "maximum"):
                                        value = options["yAxis"]["min" if endpoint == "minimum" else "max"]
                                        getattr(plot_settings, f"y_{endpoint}").set_value(float(f"{value:.12g}"))
                                result_chart.options.clear()
                                result_chart.options.update(options)
                                result_chart.update()
                            finally:
                                updating_plot_settings = False
                        except (RuntimeError, TypeError, ValueError) as exc:
                            notify(str(exc), type="negative")

                    def reset_preview_y_limits() -> None:
                        nonlocal y_limits_automatic, updating_plot_settings
                        y_limits_automatic = True
                        updating_plot_settings = True
                        try:
                            plot_settings.y_minimum.set_value(None)
                            plot_settings.y_maximum.set_value(None)
                        finally:
                            updating_plot_settings = False
                        redraw_prepared_preview()

                    def sync_plot_visibility() -> None:
                        sync_preview_settings_visibility(
                            plot_settings,
                            plot_type="Histogram" if plot_settings.view.value == "Histogram" else "Line",
                            filter_available=filter_applied(),
                        )

                    def preview_view_changed() -> None:
                        nonlocal updating_plot_settings
                        selected = selected_generated_result()
                        value_label = (
                            str(selected.preview.value_label).replace("_", " ").strip().capitalize()
                            if selected is not None and selected.preview is not None else "Value"
                        )
                        histogram = plot_settings.view.value == "Histogram"
                        updating_plot_settings = True
                        try:
                            plot_settings.x_axis_label.set_value(value_label if histogram else "Position (kb)")
                            plot_settings.y_axis_label.set_value(
                                "Density" if histogram and plot_settings.density.value else "Count" if histogram else value_label
                            )
                            plot_settings._automatic_axes.add("x")
                        finally:
                            updating_plot_settings = False
                        sync_plot_visibility()
                        histogram_sync.sync_to_data(redraw=False)
                        reset_preview_y_limits()

                    def preview_log_x_changed() -> None:
                        histogram_sync.sync_to_data(redraw=False)
                        reset_preview_y_limits()

                    def preview_y_limits_changed() -> None:
                        nonlocal y_limits_automatic
                        if not updating_plot_settings:
                            y_limits_automatic = (
                                plot_settings.y_minimum.value is None and plot_settings.y_maximum.value is None
                            )
                            redraw_prepared_preview()

                    def preview_range_changed() -> None:
                        if y_limits_automatic:
                            reset_preview_y_limits()
                        else:
                            redraw_prepared_preview()

                    with (
                        ui.element("section")
                        .classes("dn-data-manager-prepared-preview")
                        .mark("data-manager-result-preview")
                    ) as preview_surface:
                        @ui.refreshable
                        def render_prepared_preview() -> None:
                            nonlocal result_chart
                            result_chart = None
                            selected = selected_generated_result()
                            if not settings["show_preview"] or selected is None or selected.preview is None:
                                return
                            selected_preview = selected.preview
                            with ui.element("div").classes("w-full"):
                                with ui.row().classes("dn-data-manager-preview-heading"):
                                    ui.label("Prepared profile").classes("dn-setting-group-title")
                                    ui.space()
                                    sampled = min(selected_preview.total_count, preview_point_limit())
                                    sample_text = f"{selected_preview.total_count:,} values"
                                    if sampled < selected_preview.total_count:
                                        sample_text += f" · {sampled:,} shown"
                                    ui.label(sample_text).classes(
                                        "dn-data-manager-preview-sampling"
                                    ).bind_text_from(
                                        get_export_settings(), "preview_points",
                                        backward=lambda limit, total=selected_preview.total_count: (
                                            f"{total:,} values" + (f" · {min(total, int(limit)):,} shown" if int(limit) < total else "")
                                        ),
                                    )
                                with ui.element("div").classes("dn-data-manager-preview-stats"):
                                    ui.label(f"Finite {selected_preview.finite_count:,}")
                                    ui.label(f"Missing {selected_preview.missing_count:,}")
                                    if (
                                        selected_preview.minimum is not None
                                        and selected_preview.maximum is not None
                                    ):
                                        ui.label(
                                            "Range "
                                            f"{selected_preview.minimum:.4g} to "
                                            f"{selected_preview.maximum:.4g}"
                                        )
                                if selected_preview.finite_count == 0:
                                    ui.label("No finite values to plot. The raw data remain available below.").classes("dn-data-manager-section-guidance")
                                    return
                                result_chart = render_preview_chart(
                                    _preview_options(selected_preview),
                                    title="Prepared profile",
                                    marker="data-manager-preview-chart",
                                    on_download=download_preview_plot,
                                )
                                for chart in (result_chart.canvas, result_chart.vector):
                                    chart.classes("dn-data-manager-profile-chart")
                                result_chart.attach_settings(plot_settings)
                                redraw_prepared_preview()

                        render_prepared_preview()
                        plot_settings = preview_settings_controls(
                            initial_title="Prepared profile",
                            initial_x_label="Position (kb)",
                            initial_y_label="Value",
                            on_reset_y_limits=reset_preview_y_limits,
                            markers=preview_settings_markers("data-manager-plot"),
                            allow_filter=True, allow_legend_name=True,
                        )
                        plot_settings.histogram_bins.set_value(40)
                        plot_settings.line_width.set_value(1.8)
                        plot_settings.show_legend.set_value(False)
                        plot_settings.show_grid.set_value(False)
                        histogram_sync = HistogramBinSynchronizer(
                            plot_settings, values=preview_values,
                            on_valid_change=reset_preview_y_limits,
                            log_x=lambda: bool(plot_settings.log_x.value),
                        )
                        bind_preview_settings_updates(
                            plot_settings,
                            on_redraw=redraw_prepared_preview,
                            on_view_change=preview_view_changed,
                            on_density_change=preview_view_changed,
                            on_histogram_range_change=preview_range_changed,
                            on_log_x_change=preview_log_x_changed,
                            on_y_limits_change=preview_y_limits_changed,
                            on_visibility_change=sync_plot_visibility,
                            default_aspect_ratio=2.5,
                        )
                        sync_plot_visibility()
                        def filter_application_changed() -> None:
                            sync_plot_visibility()
                            redraw_prepared_preview()

                        filter_controls.controls["enabled"].on_value_change(filter_application_changed)
                    preview_surface.visible = False

                    filter_file_preferences: dict[int, _GeneratedFilePreference] = {}
                    statistics_file_preferences: dict[int, StatisticsExportPreference] = {}
                    filter_exports_slot: Any | None = None

                    def render_filter_exports() -> None:
                        if filter_exports_slot is None or filter_exports_slot.is_deleted:
                            return
                        filter_exports_slot.clear()
                        if not filter_controls.controls["export"].value or not filter_controls.controls["enabled"].value:
                            return
                        result = selected_generated_result()
                        result_id = selected_result_id
                        if result is None or not result.filter_positions:
                            return
                        with filter_exports_slot:
                            automatic = _generated_file_stem(result) + "_filter_positions"
                            preference = filter_file_preferences.setdefault(result_id, _GeneratedFilePreference(
                                file_name=automatic, file_type="TSV", compress=False, automatic_name=automatic,
                            ))

                            async def download_filter(controls, retained=result, *, open_in=None):
                                try:
                                    artifact = await run.io_bound(
                                        filter_positions_artifact, retained,
                                        output_name=str(controls.file_name.value or (_generated_file_stem(retained) + "_filter_positions")),
                                        target_format=str(controls.file_type.value).lower(),
                                        gzip_output=bool(controls.compress.value),
                                        layout=preference.filter_layout,
                                    )
                                    if open_in is not None:
                                        await open_raw_artifact(artifact, open_in, _navigate, preferred_input="filter_positions")
                                    else:
                                        ui.download(artifact.data, artifact.filename, artifact.media_type)
                                except (OSError, RuntimeError, ValueError) as exc:
                                    notify(str(exc), type="negative")

                            with ui.column().classes("w-full dn-filter-export-row dn-mapping-batch-raw-item dn-data-manager-file").mark(f"data-manager-filter-result-{result_id}"):
                                ui.label("Mask positions").classes("dn-setting-group-title")
                                row = raw_output_file_row(
                                    file_name=preference.file_name, default_type=preference.file_type,
                                    default_compress=preference.compress,
                                    on_download=download_filter,
                                    on_open=lambda controls, destination, handler=download_filter: handler(controls, open_in=destination),
                                    reserve_remove_space=True,
                                    extra_classes="dn-mapping-batch-raw-row",
                                    markers={"row": f"data-manager-filter-result-row-{result_id}",
                                             "file_name": f"data-manager-filter-result-name-{result_id}",
                                             "file_type": f"data-manager-filter-result-type-{result_id}",
                                             "compression": f"data-manager-filter-result-compression-{result_id}",
                                             "download": f"download-filter-positions-{result_id}"},
                                )
                                row.file_type.set_options(list(_GENERATED_FILE_TYPES), value=preference.file_type)
                                fields = row.file_type.parent_slot.parent
                                fields.classes(add="dn-filter-export-fields")
                                with fields:
                                    layout_control = ui.select(
                                        {"positions": "Positions", "intervals": "Intervals"},
                                        value=preference.filter_layout, label="Export as",
                                    ).props("dense options-dense").classes("dn-field").mark(
                                        f"data-manager-filter-result-layout-{result_id}"
                                    )
                                layout_control.move(fields, target_index=1)

                                def store_filter_name(event) -> None:
                                    preference.file_name = str(event.value)
                                    preference.customized = preference.file_name != preference.automatic_name

                                def filter_layout_changed(event) -> None:
                                    preference.filter_layout = str(event.value)
                                    preference.automatic_name = _generated_file_stem(result) + (
                                        "_filter_intervals" if event.value == "intervals" else "_filter_positions"
                                    )
                                    if not preference.customized:
                                        row.file_name.set_value(preference.automatic_name)

                                row.file_name.on_value_change(store_filter_name)
                                layout_control.on_value_change(filter_layout_changed)
                                row.file_type.on_value_change(lambda event, pref=preference: setattr(pref, "file_type", str(event.value)))
                                row.compress.on_value_change(lambda event, pref=preference: setattr(pref, "compress", bool(event.value)))

                    @ui.refreshable
                    def render_result() -> None:
                        nonlocal preview_result_id, result_chart, filter_exports_slot
                        filter_exports_slot = None
                        object.__setattr__(plot_settings, "_statistics_export", None)
                        result_chart = None
                        selected = selected_generated_result()
                        preview_surface.visible = bool(
                            settings["show_preview"] and selected is not None and selected.preview is not None
                        )
                        if preview_result_id != selected_result_id:
                            histogram_sync._x_limits_automatic = True
                            preview_view_changed()
                        preview_result_id = selected_result_id
                        render_prepared_preview.refresh()
                        if selected is None:
                            with (
                                ui.column()
                                .classes("dn-data-manager-empty")
                                .mark("data-manager-empty-result")
                            ):
                                ui.icon("file_download")
                                ui.label("Converted files will appear here")
                            return

                        def select_result(result_id: int) -> None:
                            nonlocal selected_result_id
                            if selected_result_id != result_id:
                                selected_result_id = result_id
                                render_result.refresh()

                        def remove_result(result_id: int) -> None:
                            converted_results[:] = [
                                item for item in converted_results if item[0] != result_id
                            ]
                            generated_file_preferences.pop(result_id, None)
                            filter_file_preferences.pop(result_id, None)
                            statistics_file_preferences.pop(result_id, None)
                            render_result.refresh()

                        def clear_results() -> None:
                            converted_results.clear()
                            generated_file_preferences.clear()
                            filter_file_preferences.clear()
                            statistics_file_preferences.clear()
                            render_result.refresh()

                        def retained_generated_preference(
                            result_id: int,
                            result: DataConversionResult,
                        ) -> _GeneratedFilePreference:
                            automatic_name = _generated_file_stem(result)
                            preference = generated_file_preferences.setdefault(
                                result_id,
                                _GeneratedFilePreference(
                                    file_name=automatic_name,
                                    file_type=result.target_format.upper(),
                                    compress=result.artifact.filename.casefold().endswith(
                                        ".gz"
                                    ),
                                    automatic_name=automatic_name,
                                ),
                            )
                            if not preference.customized:
                                preference.file_name = automatic_name
                            preference.automatic_name = automatic_name
                            return preference

                        async def download_all_results() -> None:
                            snapshots = tuple(
                                (
                                    result,
                                    preference.file_name,
                                    preference.file_type,
                                    preference.compress,
                                )
                                for result_id, result in converted_results
                                for preference in (
                                    retained_generated_preference(result_id, result),
                                )
                            )
                            if not snapshots:
                                return
                            filter_snapshots = tuple(
                                (result, preference.file_name, preference.file_type, preference.compress, preference.filter_layout)
                                for result_id, result in converted_results
                                if filter_controls.controls["export"].value and filter_controls.controls["enabled"].value and result.filter_positions
                                for preference in (filter_file_preferences.get(result_id) or _GeneratedFilePreference(
                                    file_name=_generated_file_stem(result) + "_filter_positions", file_type="TSV", compress=False,
                                    automatic_name=_generated_file_stem(result) + "_filter_positions",
                                ),)
                            )
                            try:
                                statistics_export = plot_settings._statistics_export
                                statistics_snapshot = (
                                    statistics_export.snapshot()
                                    if statistics_export is not None and statistics_export.surface.visible else None
                                )
                                def prepare_archive() -> ExportArtifact:
                                    artifacts = tuple(
                                        _generated_download_artifact(
                                            result,
                                            file_name=file_name,
                                            file_type=file_type,
                                            compress=compress,
                                        )
                                        for result, file_name, file_type, compress in snapshots
                                    )
                                    artifacts += tuple(filter_positions_artifact(
                                        result, output_name=file_name, target_format=file_type.lower(),
                                        gzip_output=compress, layout=layout,
                                    ) for result, file_name, file_type, compress, layout in filter_snapshots)
                                    if statistics_snapshot is not None:
                                        artifacts += (statistics_export.artifact(statistics_snapshot),)
                                    return _build_generated_files_archive(artifacts)

                                bundled = await run.io_bound(prepare_archive)
                            except (OSError, RuntimeError, ValueError, zipfile.BadZipFile) as exc:
                                notify(str(exc), type="negative")
                                return
                            ui.download(
                                bundled.data,
                                bundled.filename,
                                bundled.media_type,
                            )

                        with expansion(
                            "Raw data", icon=DATA_ICON, expanded=True,
                            extra_classes="dn-output-section",
                        ):
                            with ui.element("div").classes("dn-download-file-list"):
                                for result_id, result in reversed(converted_results):
                                    preference = retained_generated_preference(
                                        result_id,
                                        result,
                                    )

                                    async def download_result(
                                        controls: Any,
                                        retained_result: DataConversionResult = result,
                                        *, open_in: str | None = None,
                                    ) -> None:
                                        try:
                                            artifact = await run.io_bound(
                                                _generated_download_artifact,
                                                retained_result,
                                                file_name=str(
                                                    controls.file_name.value
                                                    or _generated_file_stem(retained_result)
                                                ),
                                                file_type=str(
                                                    controls.file_type.value
                                                    or retained_result.target_format
                                                ),
                                                compress=bool(controls.compress.value),
                                            )
                                        except (OSError, RuntimeError, ValueError) as exc:
                                            notify(str(exc), type="negative")
                                            return
                                        if open_in is not None:
                                            preferred_input = next((
                                                key for key in _FEATURE_VALUE_NAMES.values()
                                                if key != "value" and key in retained_result.labels
                                            ), None)
                                            await open_raw_artifact(artifact, open_in, _navigate, preferred_input=preferred_input)
                                            return
                                        ui.download(
                                            artifact.data,
                                            artifact.filename,
                                            artifact.media_type,
                                        )

                                    with (
                                        ui.element("div")
                                        .classes("dn-mapping-batch-raw-item dn-data-manager-file")
                                        .mark(f"data-manager-result-{result_id}")
                                    ):
                                        row_controls = raw_output_file_row(
                                            file_name=preference.file_name,
                                            default_type=preference.file_type,
                                            default_compress=preference.compress,
                                            on_download=download_result,
                                            on_open=lambda controls, destination, handler=download_result: handler(controls, open_in=destination),
                                            on_select=lambda result_id=result_id: select_result(result_id),
                                            selected=result_id == selected_result_id,
                                            on_remove=(
                                                lambda _controls, retained_id=result_id: (
                                                    remove_result(retained_id)
                                                )
                                            ),
                                            markers={
                                                "row": f"data-manager-result-row-{result_id}",
                                                "select": f"data-manager-select-result-{result_id}",
                                                "file_name": (
                                                    f"data-manager-result-name-{result_id}"
                                                ),
                                                "file_type": (
                                                    f"data-manager-result-type-{result_id}"
                                                ),
                                                "compression": (
                                                    "data-manager-result-compression-"
                                                    f"{result_id}"
                                                ),
                                                "download": "",
                                                "remove": f"remove-converted-data-{result_id}",
                                            },
                                            extra_classes="dn-mapping-batch-raw-row",
                                        )
                                        row_controls.select_button.set_enabled(result.preview is not None)
                                        row_controls.file_type.set_options(
                                            list(_GENERATED_FILE_TYPES),
                                            value=preference.file_type,
                                        )
                                        row_controls.download_button.mark(
                                            "action-download-converted-data",
                                            f"action-download-converted-data-{result_id}"
                                        )

                                        def store_file_name(
                                            _event: Any = None,
                                            *,
                                            control: Any = row_controls.file_name,
                                            retained: _GeneratedFilePreference = preference,
                                        ) -> None:
                                            retained.file_name = str(control.value or "")
                                            retained.customized = bool(
                                                retained.file_name
                                                and retained.file_name
                                                != retained.automatic_name
                                            )

                                        def store_compression(
                                            _event: Any = None,
                                            *,
                                            control: Any = row_controls.compress,
                                            retained: _GeneratedFilePreference = preference,
                                        ) -> None:
                                            retained.compress = bool(control.value)

                                        def store_file_type(
                                            _event: Any = None,
                                            *,
                                            control: Any = row_controls.file_type,
                                            retained: _GeneratedFilePreference = preference,
                                        ) -> None:
                                            retained.file_type = str(control.value or "TSV")

                                        row_controls.file_name.on_value_change(
                                            store_file_name
                                        )
                                        row_controls.compress.on_value_change(
                                            store_compression
                                        )
                                        row_controls.file_type.on_value_change(store_file_type)
                                        if result_id == selected_result_id:
                                            filter_exports_slot = ui.element("div").classes("w-full")
                                            automatic_statistics_name = _generated_file_stem(result) + "_statistics"
                                            HistogramStatisticsExport(
                                                plot_settings, get_chart=lambda: result_chart,
                                                filename=automatic_statistics_name,
                                                marker=f"data-manager-statistics-{result_id}", navigate=_navigate,
                                                preference=statistics_file_preferences.setdefault(
                                                    result_id, StatisticsExportPreference(automatic_statistics_name),
                                                ),
                                            )
                                render_filter_exports()
                            with ui.row().classes("dn-data-manager-delete-all-row"):
                                (
                                    ui.button(
                                        "Download all",
                                        icon="download",
                                        on_click=download_all_results,
                                    )
                                    .props("flat no-caps")
                                    .classes("dn-action-secondary dn-data-manager-download-all")
                                    .mark("download-all-converted-data")
                                )
                                (
                                    ui.button(
                                        "Delete all",
                                        icon="delete_sweep",
                                        on_click=clear_results,
                                    )
                                    .props("flat no-caps")
                                    .classes("dn-action-secondary dn-data-manager-delete-all")
                                    .mark("delete-all-converted-data")
                                )

                    render_result()
                    filter_controls.controls["export"].on_value_change(render_filter_exports)
                    filter_controls.controls["enabled"].on_value_change(render_filter_exports)

    def build_request(
        source: _DataManagerSource | None,
        *,
        output_name: str,
    ) -> DataConversionRequest:
        from app.code_export import placeholder_upload

        configuration = configuration_for(source) if source is not None else dict(settings)
        controls = {key: SimpleNamespace(value=value) for key, value in configuration.items()}
        apply_configuration = bool(configuration["apply_configuration"])
        current = source.inspection if source is not None else placeholder_upload()
        details = selected_data(source) if source is not None else _SelectedData("x", ())
        selection = source.selection if source is not None else ""
        target_format = "tsv"
        decimal_places = _integer(
            controls["decimal_places"],
            "Decimal places",
            minimum=0,
        ) if apply_configuration else None
        missing_policy = str(configuration["missing_policy"]) if apply_configuration else "ignore"
        track_missing = str(configuration["track_missing"]) if apply_configuration else "nan"
        kwargs: dict[str, Any] = {}
        if details.form in {"x", "xt"} and selection != _REPLISEQ_TIMING and apply_configuration:
            chromosome = configuration["chromosome"] if any(str(name).lower() in {"chr", "chrom", "chromosome"} for name in current.columns) else None
            _rows, chromosomes = spatial_profile_rows(current) if source is not None else (None, ())
            if chromosomes and chromosome not in chromosomes:
                chromosome = chromosomes[0]
            if source is not None and (not configuration["profile_region_custom"] or not configuration["profile_bin_custom"]):
                length = (details.shape[0] if len(details.shape) == 2 and (details.form == "xt" or 1 not in details.shape)
                          else int(np.prod(details.shape)))
                _positions, intervals, _supplied = spatial_profile_grid(
                    current, length, source_bin_size=configuration["source_bin_size"], chromosome=chromosome,
                )
                if not configuration["profile_region_custom"]:
                    controls["start"].value = int(round(intervals[0, 0] * 1000))
                    controls["end"].value = int(round(intervals[-1, 1] * 1000))
                if not configuration["profile_bin_custom"]:
                    controls["bin_size"].value = max(1, int(round(np.median(intervals[:, 1] - intervals[:, 0]) * 1000)))
            kwargs.update(
                profile_binning=True,
                source_bin_size=_integer(controls["source_bin_size"], "Source resolution", minimum=1),
                start=_integer(controls["start"], "Start", minimum=0) if source is not None or configuration["profile_region_custom"] else None,
                end=_integer(controls["end"], "End", minimum=1) if source is not None or configuration["profile_region_custom"] else None,
                bin_size=_integer(controls["bin_size"], "Bin size", minimum=1),
                aggregation="mean",
                chromosome=chromosome,
                memory_limit_mb=get_export_settings().conversion_memory_limit_mb,
            )
        if selection == _REPLISEQ_TIMING:
            chromosome = str(controls["chromosome"].value or "")
            if source.track_inspection is None or chromosome not in {
                region.chromosome for region in source.track_inspection.regions
            }:
                raise ValueError(f"The selected data does not contain chromosome {chromosome}.")
            kwargs.update(
                chromosome=chromosome,
                repliseq_timing_method=str(controls["repliseq_timing_method"].value),
                s_phase_duration_min=_number(controls["s_phase_duration_min"], "S-phase duration"),
                memory_limit_mb=get_export_settings().conversion_memory_limit_mb,
            )
        if details.form == "track":
            track_summary = source.track_inspection
            if track_summary is None:
                raise ValueError(
                    "The selected data could not be inspected as a genomic track."
                )
            chromosome_scope = str(controls["chromosome_scope"].value or "single")
            aggregation = str(controls["aggregation"].value)
            if aggregation not in _track_aggregation_options(track_summary.backend, current):
                raise ValueError(
                    f"The selected data does not support the {aggregation} aggregation."
                )
            kwargs.update(
                chromosome_scope=chromosome_scope,
                memory_limit_mb=get_export_settings().conversion_memory_limit_mb,
                bin_size=_integer(controls["bin_size"], "Bin size", minimum=1),
                aggregation=aggregation,
                track_missing=track_missing,
                interpolation=str(controls["interpolation"].value) if apply_configuration else "none",
            )
            if chromosome_scope == "single":
                chromosome = str(controls["chromosome"].value)
                regions = {
                    region.chromosome: region for region in track_summary.regions
                }
                if chromosome not in regions:
                    raise ValueError(
                        f"The selected data does not contain chromosome {chromosome}."
                    )
                available = regions[chromosome]
                start = _integer(controls["start"], "Start", minimum=0)
                end = _integer(controls["end"], "End", minimum=1)
                if end <= start:
                    raise ValueError(
                        "End must be greater than start for the genomic region."
                    )
                if start < available.start or end > available.end:
                    raise ValueError(
                        f"The selected data covers {chromosome}:"
                        f"{available.start:,}-{available.end:,}, which does not include "
                        f"the requested {start:,}-{end:,} interval."
                    )
                kwargs.update(
                    chromosome=chromosome,
                    start=start,
                    end=end,
                )
            else:
                kwargs.update(chromosome=None, start=None, end=None)
        if details.form in _POLISHABLE_FORMS and apply_configuration:
            threshold_enabled, threshold_min, threshold_max = (
                _resolved_feature_bounds(
                    str(configuration["feature"]),
                    _optional_number(controls["threshold_min"], "Lower threshold"),
                    _optional_number(controls["threshold_max"], "Upper threshold"),
                )
            )
            kwargs.update(
                threshold_enabled=threshold_enabled,
                threshold_min=threshold_min,
                threshold_max=threshold_max,
            )
        # Smoothing, scaling and inversion belong to Data transforms, not
        # Configure. Keep them active when configuration processing is bypassed.
        if details.form in _POLISHABLE_FORMS:
            value_scaling = str(controls["value_scaling"].value)
            kwargs.update(
                smooth_window=_integer(
                    controls["smooth_window"],
                    "Smoothing window",
                    minimum=1,
                ) if configuration["smoothing_mode"] == "moving_average" else 1,
                scaling_mode=value_scaling,
                normalize="none",
                rescale_min=(
                    _number(controls["rescale_min"], "Minimum")
                    if value_scaling == "interval"
                    else 0.0
                ),
                rescale_max=(
                    _number(controls["rescale_max"], "Maximum")
                    if value_scaling == "interval"
                    else 1.0
                ),
                rescale_invert=bool(controls["rescale_invert"].value),
                signal_transform="keep",
                scale=(
                    _number(controls["scale"], "Multiplier")
                    if value_scaling == "multiplier"
                    else 1.0
                ),
                offset=0.0,
            )
        return DataConversionRequest(
            inspection=current,
            input_form=details.request_form,
            target_format=target_format,
            output_name=output_name,
            # Compression is an output-presentation choice in Generated Files,
            # so conversions always retain an editable uncompressed artifact.
            gzip_output=False,
            value_column=details.value_column,
            array_key=details.array_key,
            missing_policy=missing_policy,
            fill_value=(
                _number(controls["fill_value"], "Fill value")
                if (track_missing if details.form == "track" else missing_policy) == "fill"
                else 0.0
            ),
            decimal_places=decimal_places,
            data_filter=filter_controls.snapshot() if filter_controls is not None else DataFilterSettings(),
            transform_formula=transform_controls.formula_for(source.source_id if source is not None else None) if transform_controls is not None else None,
            value_name=_selected_value_name(
                source,
                details,
                str(configuration["feature"]),
            ) if source is not None else _FEATURE_VALUE_NAMES[str(configuration["feature"])],
            genome_build=None,
            units="min" if selection == _REPLISEQ_TIMING else None,
            source_url=(str(source.remote_metadata.get("final_url") or "") or None
                        if source is not None else configuration.get("source_url") or None),
            biological_role=str(configuration["feature"]),
            semantic_note=("Timing from sixteen equally spaced S-phase fractions. "
                           "Default T50 is the midpoint of a cumulative sigmoid fit."
                           if selection == _REPLISEQ_TIMING else _FEATURE_HINTS[str(configuration["feature"])]),
            **kwargs,
        )

    def build_requests(*, allow_placeholders: bool = False) -> tuple[
        tuple[_DataManagerSource | None, DataConversionRequest],
        ...,
    ]:
        if not allow_placeholders and (pending_links := unresolved_link_rows()):
            noun = "link" if len(pending_links) == 1 else "links"
            raise ValueError(
                f"Inspect the {len(pending_links)} pending public {noun} before converting."
            )
        candidates = tuple(sources_for_mode())
        if not candidates:
            if allow_placeholders:
                return ((None, build_request(None, output_name=_DATA_FILE_STEM)),)
            raise ValueError("Add at least one data source before converting.")
        save_source_configuration()

        forms = tuple(selected_data(source).form for source in candidates)
        if len(set(forms)) != 1 and not (transform_controls is not None and transform_controls.combining):
            raise ValueError(
                "All sources in one conversion batch must use the same data type."
            )

        requested_output_names: list[str] = []
        for index, source in enumerate(candidates, start=1):
            base = (
                str(source.output_name).strip()
                or _automatic_output_name(
                    source.inspection.name,
                    str(configuration_for(source)["feature"]),
                )
            )
            if source.selection == _REPLISEQ_TIMING:
                base += "_" + str(configuration_for(source)["chromosome"])
            requested_output_names.append(base or f"{_DATA_FILE_STEM}_{index}")

        output_names = _unique_conversion_output_names(
            requested_output_names,
            target_format="tsv",
            gzip_output=False,
        )

        return tuple(
            (
                source,
                build_request(source, output_name=output_name),
            )
            for source, output_name in zip(
                candidates,
                output_names,
                strict=True,
            )
        )

    async def download_data_manager_code() -> None:
        from app.code_export import CodeInputs, CodeSnapshot, build_python, filter_code_settings, statistics_code_settings

        if conversion_running or upload_inspection_running or remote_inspections or filter_controls.loading:
            notify("Wait for the data operation to finish before exporting code.", type="warning")
            return
        try:
            requests = build_requests(allow_placeholders=True)
            inputs = CodeInputs()
            request_options = [inputs.encode(replace(
                request, data_filter=filter_code_settings(filter_controls),
            )) for _source, request in requests]
            combining = transform_controls.combining
            names = ([f"{_DATA_FILE_STEM}_combined"] if combining else
                     [request.output_name for _source, request in requests])
            exports = []
            for name in names:
                retained = next(((key, result) for key, result in converted_results
                                 if _generated_file_stem(result) == name), None)
                preference = generated_file_preferences.get(retained[0]) if retained else None
                options = dict(output_name=preference.file_name if preference else name,
                               target_format=preference.file_type.lower() if preference else "tsv",
                               gzip_output=preference.compress if preference else False)
                if filter_controls.controls["enabled"].value and filter_controls.controls["export"].value:
                    filter_preference = filter_file_preferences.get(retained[0]) if retained else None
                    options["filter"] = dict(
                        output_name=filter_preference.file_name if filter_preference else name + "_filter_positions",
                        target_format=filter_preference.file_type.lower() if filter_preference else "tsv",
                        gzip_output=filter_preference.compress if filter_preference else False,
                        layout=filter_preference.filter_layout if filter_preference else "positions",
                    )
                exports.append(options)
            statistics = statistics_code_settings(plot_settings, filename=names[0] + "_statistics")
            if statistics is not None:
                selected = selected_generated_result()
                statistics["source_index"] = next((index for index, name in enumerate(names)
                    if selected is not None and _generated_file_stem(selected) == name), 0)
            snapshot = CodeSnapshot("data_manager", dict(
                sources=request_options, exports=exports,
                statistics=statistics,
                preview_points=get_export_settings().preview_points,
                combined_formula=(
                    transform_controls.combined_formula if transform_controls.combined_formula is not None
                    else " + ".join(f"source{i}" for i in range(1, len(requests) + 1))
                ) if combining else None,
                combined_name=f"{_DATA_FILE_STEM}_combined",
            ), tuple(inputs.files.items()))
            artifact = await run.io_bound(build_python, snapshot)
            ui.download(artifact.data, artifact.filename, artifact.media_type)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            notify(str(exc), type="negative")

    async def convert() -> None:
        nonlocal active_cancel, conversion_running, next_result_id, selected_result_id
        if conversion_running:
            if active_cancel is not None and not active_cancel.is_set():
                page_activity.stop("data-management")
                active_cancel.set()
                conversion_status.text = "Stopping conversion..."
                convert_button.disable()
            return
        if upload_inspection_running or remote_inspections or (filter_controls is not None and filter_controls.loading):
            notify(
                "Wait for the sources to finish loading before converting.",
                type="warning",
            )
            return
        cancel_event = Event()

        def check_cancelled() -> None:
            if cancel_event.is_set():
                raise CancelledError("Conversion stopped.")

        conversion_running = True
        active_cancel = cancel_event
        page_run = page_activity.begin("data-management", cancel=cancel_event.set)
        completed_successfully = False
        convert_button.set_text("Stop")
        convert_button.set_icon("stop")
        lock_request_controls(True)
        convert_button.enable()
        try:
            requests = build_requests()
            combining = transform_controls is not None and transform_controls.combining
            combined_formula = (
                transform_controls.combined_formula if transform_controls.combined_formula is not None
                else " + ".join(f"source{i}" for i in range(1, len(requests) + 1))
            ) if combining else None
            check_cancelled()
            started = time.perf_counter()
            jobs = requests[:1] if combining else requests
            total = len(jobs)
            completed: list[DataConversionResult] = []
            failures: list[tuple[str, str]] = []
            conversion_progress.visible = True
            conversion_progress.value = 0.01
            for index, (source, request) in enumerate(jobs):
                progress_state = _ConversionProgressState()
                item_number = index + 1
                conversion_status.text = (
                    f"Converting {item_number}/{total} - {source.inspection.name}"
                )
                worker = asyncio.create_task(
                    run.io_bound(
                        convert_combined_data, tuple(item for _source, item in requests),
                        combined_formula, output_name=f"{_DATA_FILE_STEM}_combined",
                        progress=progress_state.report, cancel=check_cancelled,
                    ) if combining else run.io_bound(
                        convert_data,
                        request,
                        progress_state.report,
                        check_cancelled,
                    )
                )
                while not worker.done():
                    if cancel_event.is_set():
                        conversion_status.text = "Stopping conversion..."
                    else:
                        value, message = progress_state.snapshot()
                        conversion_progress.value = max(
                            0.01,
                            min(0.999, (index + value) / total),
                        )
                        conversion_status.text = (
                            f"{item_number}/{total} - {source.inspection.name} - {message}"
                        )
                    await asyncio.sleep(0.04)
                try:
                    result = await worker
                    if result is None:
                        raise RuntimeError(
                            "Conversion stopped before returning data."
                        )
                except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
                    failures.append((source.inspection.name, str(exc)))
                    continue
                completed.append(result)
                for warning in result.warnings:
                    notify(warning, type="warning")
                conversion_progress.value = item_number / total
                check_cancelled()

            check_cancelled()
            if combining and completed:
                keep_names = {result.artifact.filename.casefold() for result in completed}
                converted_results[:] = [
                    (result_id, result) for result_id, result in converted_results
                    if result.artifact.filename.casefold() in keep_names
                ]
                kept_ids = {result_id for result_id, _result in converted_results}
                for result_id in tuple(generated_file_preferences):
                    if result_id not in kept_ids:
                        generated_file_preferences.pop(result_id)
            retained_output_bytes = sum(
                len(result.artifact.data) for _result_id, result in converted_results
            )
            evicted_results = 0
            for result in completed:
                matching_index = next(
                    (
                        index
                        for index, (_result_id, retained) in enumerate(converted_results)
                        if retained.artifact.filename.casefold()
                        == result.artifact.filename.casefold()
                    ),
                    None,
                )
                if matching_index is None:
                    next_result_id += 1
                    result_id = next_result_id
                else:
                    result_id, previous = converted_results.pop(matching_index)
                    retained_output_bytes -= len(previous.artifact.data)

                result_bytes = len(result.artifact.data)
                while converted_results and (
                    len(converted_results) >= _MAX_RETAINED_OUTPUT_COUNT
                    or retained_output_bytes + result_bytes
                    > _MAX_RETAINED_OUTPUT_BYTES
                ):
                    removed_id, removed = converted_results.pop(0)
                    generated_file_preferences.pop(removed_id, None)
                    retained_output_bytes -= len(removed.artifact.data)
                    evicted_results += 1
                converted_results.append((result_id, result))
                retained_output_bytes += result_bytes
                cache_dataset(result)
            if completed:
                selected_result_id = converted_results[-1][0]
                render_result.refresh()

            elapsed = time.perf_counter() - started
            conversion_progress.value = 1.0
            if failures:
                conversion_status.text = (
                    f"Completed {len(completed)}/{total} in {elapsed:.2f} s"
                )
                detail = "; ".join(dict.fromkeys(
                    notification_message(message) for _name, message in failures
                ))
                notify(
                    f"Converted {len(completed)} of {total}. {detail}",
                    type="warning" if completed else "negative",
                )
            else:
                conversion_status.text = f"Complete in {elapsed:.2f} s"
            if evicted_results:
                notify(
                    f"Removed {evicted_results} oldest generated "
                    f"{'file' if evicted_results == 1 else 'files'} to keep the local "
                    "workspace responsive.",
                    type="warning",
                )
            completed_successfully = bool(completed) and not failures
        except CancelledError:
            conversion_progress.visible = False
            conversion_status.text = "Conversion stopped"
        except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
            conversion_progress.visible = False
            conversion_status.text = "Conversion could not be completed"
            notify(str(exc), type="negative")
        finally:
            conversion_running = False
            active_cancel = None
            convert_button.set_text("Convert")
            convert_button.set_icon(workflow_action_icon("data"))
            lock_request_controls(False)
            set_convert_enabled()
            page_activity.finish("data-management", page_run, success=completed_successfully)

    convert_button.on_click(convert)

    def clear_conversion_outputs() -> None:
        converted_results.clear()
        generated_file_preferences.clear()
        conversion_progress.visible = False
        conversion_status.text = "Ready"
        render_result.refresh()

    current_session_cache().register_output_clearer(
        "data-management",
        clear_conversion_outputs,
    )

    async def receive_raw_files(files) -> None:
        if conversion_running or upload_inspection_running or remote_inspections:
            raise ValueError("Wait for the current source operation to finish before adding data.")
        source_tabs.set_value("upload")
        for _ in range(4):
            await asyncio.sleep(0)
        await append_raw_uploads(lambda: source_controls["uploader"], files)

    current_session_cache().register_input_targets(
        "data-management", lambda: (
            RawInputTarget("data", "Data", receive_raw_files),
            RawInputTarget("filter_positions", "Region transforms", filter_controls.receive_mask, kind="mask"),
        ),
    )

    def refresh_data_manager_page() -> None:
        """Apply global presentation preferences without rebuilding inputs."""

        link_examples.set_visibility(get_export_settings().show_examples)

    current_session_cache().register_page_refresher(
        "data-management",
        refresh_data_manager_page,
    )
