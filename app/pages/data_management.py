"""Interactive data conversion through :func:`dnascape.rdata`."""

from __future__ import annotations

import asyncio
import time
import zipfile
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import CancelledError
from dataclasses import dataclass, field
from pathlib import PurePath
from threading import Event, Lock
from typing import Any

import numpy as np
from nicegui import run, ui

from app.artifact_archives import (
    DEFAULT_BATCH_ARCHIVE_SOURCE_BYTES,
    build_artifact_archive,
)
from app.batch_workflows import unique_dataset_names
from app.data_conversion_service import (
    DATA_EXPORT_FORMATS,
    DataConversionRequest,
    DataConversionResult,
    TrackInspection,
    convert_data,
    inspect_track,
    transcode_conversion_artifact,
)
from app.export_settings import get_export_settings
from app.mapping_catalog import RDATA_UPLOAD_ACCEPT
from app.mapping_uploads import (
    UploadInspection,
    inspect_upload,
    inspect_upload_event,
)
from app.pages.common import (
    DATA_ICON,
    action_dock,
    data_upload,
    expansion,
    page_intro,
    panel,
    panel_title,
    raw_output_file_row,
    set_data_upload_locked,
)
from app.preview_plotting import build_preview_plot_options, render_preview_chart
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
_TRACK_MISSING = {
    "nan": "Keep empty bins",
    "zero": "Fill with zero",
    "fill": "Fill with a value",
    "min": "Fill with minimum",
    "max": "Fill with maximum",
}
_TRACK_INTERPOLATION = {
    "none": "None",
    "linear": "Linear",
    "ffill": "Previous value",
    "bfill": "Next value",
}
_VALUE_SCALING = {
    "none": "None",
    "multiplier": "Multiplier",
    "interval": "Target interval",
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
    "firing_rate": "Firing rate",
    "fork_speed": "Fork speed",
    "fork_directionality": "Fork directionality",
    "replicated_fraction": "Replicated fraction",
}
_FEATURE_VALUE_NAMES = {
    "generic": "value",
    "replication_timing": "replication_timing",
    "firing_rate": "firing_rate",
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
    "firing_rate": "Firing-rate profiles should be finite and non-negative.",
    "fork_speed": "Fork-speed profiles should be positive and use consistent units.",
    "fork_directionality": "Fork directionality is normally bounded between -1 and 1.",
    "replicated_fraction": "Replicated fraction is normally bounded between 0 and 1.",
}
_FEATURE_THRESHOLD_DEFAULTS: Mapping[str, tuple[bool, float | None, float | None]] = {
    "generic": (False, None, None),
    "replication_timing": (True, 0.0, None),
    "firing_rate": (True, 0.0, None),
    "fork_speed": (True, 0.0, None),
    "fork_directionality": (True, -1.0, 1.0),
    "replicated_fraction": (True, 0.0, 1.0),
}
_ENCODE_EXAMPLE_URL = (
    "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/encodeDCC/"
    "wgEncodeUwRepliSeq/wgEncodeUwRepliSeqK562WaveSignalRep1.bigWig"
)
_SNS_EXAMPLE_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM716nnn/GSM716435/"
    "suppl/GSM716435_K562_Peaks_FDR15.bed.gz"
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
    if inspection.source_format in {"csv", "tsv"}:
        options[_TABLE] = "Full table"
    if not options:
        options[_NATURAL] = "Profile"
    return options


def _default_selection(inspection: UploadInspection) -> str:
    options = _selection_options(inspection)
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
        if selected == "firing_rate" and inspection.source_format == "bedgraph":
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


def _selected_value_name(
    source: _DataManagerSource,
    details: _SelectedData,
    feature: str,
) -> str:
    """Keep a source label when one exists and otherwise use a useful fallback."""

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
        or (feature == "firing_rate" and inspection.source_format == "bedgraph")
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
    if feature == "firing_rate":
        options.pop("center")
        options.pop("zscore")
    return options


def _preview_options(preview: Any) -> dict[str, Any]:
    """Build the small, bounded visual check shown after conversion."""

    values = np.asarray(preview.values).reshape(-1)
    x_axis = np.asarray(preview.x).reshape(-1)
    point_count = min(values.size, x_axis.size)
    values = values[:point_count]
    x_axis = x_axis[:point_count]
    if point_count > 500:
        sample = np.linspace(0, point_count - 1, 500, dtype=int)
        values = values[sample]
        x_axis = x_axis[sample]
    value_label = str(preview.value_label).replace("_", " ").strip().capitalize()

    return build_preview_plot_options(
        plot_type="Line",
        title="Prepared profile",
        x_label=str(preview.x_label),
        y_label=value_label,
        color="#2c7f77",
        palette="viridis",
        histogram_bins=40,
        density=False,
        show_legend=False,
        show_colorbar=False,
        log_y=False,
        invert_y=False,
        y_limits=None,
        series_label=value_label,
        values=values,
        x_axis=x_axis,
        show_grid=False,
        interactive=True,
        line_width=1.8,
    )


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
    generated_file_preferences: dict[int, _GeneratedFilePreference] = {}
    next_result_id = 0
    conversion_running = False
    active_cancel: Event | None = None
    upload_inspection_running = False
    result_chart: Any | None = None
    controls: dict[str, Any] = {}
    source_controls: dict[str, Any] = {}
    source_control_states: dict[str, bool] = {}
    request_control_states: dict[tuple[str, str], bool] = {}
    settings: dict[str, Any] = {
        "source_mode": "upload",
        "source_url": "",
        "feature": "generic",
        "target_format": "tsv",
        "output_name": _DATA_FILE_STEM,
        "decimal_places": 6,
        "missing_policy": "ignore",
        "chromosome_scope": "single",
        "chromosome": None,
        "start": 0,
        "end": 1,
        "bin_size": 10_000,
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
        "show_preview": False,
    }

    def source_mode() -> str:
        return str(settings.get("source_mode") or "upload")

    def sources_for_mode(mode: str | None = None) -> list[_DataManagerSource]:
        selected_mode = str(mode or source_mode())
        candidates = [
            source for source in retained_sources if source.mode == selected_mode
        ]
        if selected_mode == "url":
            row_order = {
                row.source_id: index
                for index, row in enumerate(remote_link_inputs)
                if row.source_id is not None
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
        return _selected_data(source.inspection, source.selection)

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
        "Detect, clean, bin, and export profiles from local files or public genomic tracks.",
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
                        await render_source_details.refresh()
                        await render_conversion_settings.refresh()
                        set_convert_enabled()

                    async def adopt_candidate(
                        candidate: UploadInspection,
                        *,
                        mode: str,
                        source_id: str,
                        metadata: Mapping[str, Any] | None = None,
                    ) -> bool:
                        if conversion_running:
                            ui.notify(
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
                            ui.notify(
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
                            ui.notify(
                                "Retained source files are limited to "
                                f"{maximum_mib:g} MiB in one local workspace. "
                                "Remove files you no longer need and try again.",
                                type="warning",
                            )
                            return False
                        convert_button.disable()
                        try:
                            candidate_track = None
                            if candidate.source_format in {"bigwig", "bedgraph"}:
                                candidate_track = await run.io_bound(inspect_track, candidate)
                                if candidate_track is None:
                                    raise RuntimeError(
                                        "Track inspection stopped before returning chromosome data."
                                    )
                            candidate_selection = _default_selection(candidate)
                            _selected_data(candidate, candidate_selection)
                        except (OSError, RuntimeError, TypeError, ValueError) as exc:
                            ui.notify(str(exc), type="negative")
                            set_convert_enabled()
                            return False

                        if (
                            existing is not None
                            and existing.selection in _selection_options(candidate)
                        ):
                            candidate_selection = existing.selection
                        automatic_name = _automatic_output_name(
                            candidate.name,
                            str(settings["feature"]),
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
                        )
                        if existing_index is None:
                            retained_sources.append(adopted)
                        else:
                            retained_sources[existing_index] = adopted
                        active_source_ids[mode] = source_id
                        settings["output_name"] = adopted.output_name
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
                            _apply_track_defaults(
                                settings, str(settings["feature"]), candidate,
                                candidate_track.backend,
                            )
                        await refresh_active_source_views()
                        return True

                    async def uploaded(event: Any) -> bool:
                        nonlocal upload_inspection_running
                        if conversion_running:
                            ui.notify("A conversion is already running.", type="warning")
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
                            ui.notify(str(exc), type="negative")
                            return False
                        finally:
                            upload_inspection_running = False
                            set_convert_enabled()

                    async def cleared(event: Any) -> None:
                        if conversion_running:
                            ui.notify("A conversion is already running.", type="warning")
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
                            ui.notify(
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
                                accept=f"{RDATA_UPLOAD_ACCEPT},.bed",
                                accepted_formats=(
                                    "CSV, TSV, TXT, NPY, NPV, NPZ, BED, bedGraph, BigWig, GZ"
                                ),
                                marker="data-manager-upload",
                                show_accepted_formats=False,
                                on_upload=uploaded,
                                on_clear=cleared,
                                multiple=True,
                                max_files=None,
                                hide_when_loaded=False,
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
                                                    label="Public file link",
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
                                                ui.button("Inspect")
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
                                            ui.notify(
                                                "A conversion is already running.",
                                                type="warning",
                                            )
                                            return
                                        if row.row_id in remote_inspections:
                                            return
                                        url = str(input_control.value or "").strip()
                                        row.url = url
                                        if first_row:
                                            settings["source_url"] = url
                                        if not url:
                                            ui.notify(
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
                                                ui.notify(
                                                    f"{candidate.name} is ready.",
                                                    type="positive",
                                                )
                                        except (
                                            OSError,
                                            RemoteSourceError,
                                            RuntimeError,
                                            TypeError,
                                            ValueError,
                                        ) as exc:
                                            ui.notify(str(exc), type="negative")
                                        finally:
                                            remote_inspections.discard(row.row_id)
                                            if not progress.is_deleted:
                                                progress.visible = False
                                            if not status.is_deleted:
                                                status.visible = False
                                            if not action.is_deleted:
                                                action.enable()
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
                                            ui.notify(
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
                                    ) -> None:
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
                                        loaded_url = str(
                                            loaded.remote_metadata.get("requested_url") or ""
                                        ).strip()
                                        if new_url == loaded_url:
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
                                        ui.notify(
                                            "Wait for the current link operation to finish.",
                                            type="warning",
                                        )
                                        return
                                    if len(remote_link_inputs) >= _MAX_RETAINED_SOURCE_COUNT:
                                        ui.notify(
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
                                "Paste public HTTPS links to BigWig, BED, or bedGraph files. "
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
                                    ui.notify(
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
                                    feature="firing_rate",
                                    aggregation="max",
                                    track_missing="zero",
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
                            link_examples.visible = get_export_settings().show_examples
                            source_controls["encode-example"] = encode_example_button
                            source_controls["bedgraph-example"] = bedgraph_example_button

                    @ui.refreshable
                    def render_source_details() -> None:
                        for key in tuple(source_controls):
                            if key.startswith("source-detail-"):
                                source_controls.pop(key, None)
                        candidates = sources_for_mode()
                        current_source = active_source()
                        if current_source is None:
                            return

                        if len(candidates) > 1:
                            labels = unique_dataset_names(
                                tuple(source.inspection.name for source in candidates)
                            )
                            dataset_select = (
                                ui.select(
                                    {
                                        source.source_id: label
                                        for source, label in zip(
                                            candidates,
                                            labels,
                                            strict=True,
                                        )
                                    },
                                    value=current_source.source_id,
                                    label="Dataset",
                                )
                                .props("dense options-dense")
                                .classes("dn-field dn-data-manager-source-select")
                                .mark("data-manager-source-dataset")
                            )

                            async def dataset_changed() -> None:
                                if dataset_select.value is None:
                                    return
                                active_source_ids[source_mode()] = str(
                                    dataset_select.value
                                )
                                selected = active_source()
                                if selected is not None:
                                    settings["output_name"] = selected.output_name
                                await refresh_active_source_views()

                            dataset_select.on_value_change(dataset_changed)
                            source_controls["source-detail-dataset"] = dataset_select

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
                                await refresh_active_source_views()

                            profile_select.on_value_change(profile_changed)
                            source_controls["source-detail-profile"] = profile_select

                        details = selected_data(current_source)
                        with (
                            ui.element("div")
                            .classes(
                                "dn-mapping-detected-data dn-profile-metadata "
                                "dn-data-manager-detected"
                            )
                            .mark("data-manager-detected-data")
                        ):
                            if len(candidates) > 1:
                                ui.label(f"Datasets: {len(candidates)}").classes(
                                    "dn-mapping-detected-summary"
                                )
                            ui.label(
                                "Format: "
                                f"{_source_format_label(current_source.inspection.source_format)}"
                            ).classes("dn-mapping-detected-summary")
                            ui.label(
                                "Size: "
                                f"{_file_size_text(len(current_source.inspection.content))}"
                            ).classes("dn-mapping-detected-summary")
                            ui.label(
                                f"Dependence: {_FORM_LABELS.get(details.form, 'Unknown')}"
                            ).classes(
                                "dn-mapping-detected-summary "
                                "dn-mapping-detected-dependence"
                            )
                            if current_source.track_inspection is not None:
                                region = current_source.track_inspection.regions[0]
                                estimated_bins = max(
                                    1,
                                    (
                                        region.end
                                        - region.start
                                        + int(settings["bin_size"])
                                        - 1
                                    )
                                    // int(settings["bin_size"]),
                                )
                                ui.label(f"Estimated bins: {estimated_bins:,}").classes(
                                    "dn-mapping-detected-summary"
                                )
                                ui.label(
                                    "Chromosomes: "
                                    f"{len(current_source.track_inspection.regions)}"
                                ).classes("dn-mapping-detected-summary")
                            else:
                                ui.label(
                                    f"Array size: {_shape_text(details.shape)}"
                                ).classes("dn-mapping-detected-summary")
                            if current_source.remote_metadata:
                                ui.label("Source: Public link").classes(
                                    "dn-mapping-detected-summary"
                                )
                        if conversion_running:
                            lock_source_controls(True)

                    render_source_details()

                with expansion("Configure", icon="tune", expanded=True):

                    @ui.refreshable
                    def render_conversion_settings() -> None:
                        controls.clear()

                        def store_control(name: str) -> None:
                            settings[name] = controls[name].value

                        def feature_changed() -> None:
                            feature = str(controls["feature"].value or "generic")
                            current = active_source()
                            _apply_feature_defaults(
                                settings, feature,
                                current.inspection if current else None,
                                current.track_inspection.backend
                                if current and current.track_inspection else None,
                            )
                            for source in retained_sources:
                                if not source.output_name_customized:
                                    source.output_name = _automatic_output_name(
                                        source.inspection.name,
                                        feature,
                                    )
                            current = active_source()
                            if current is not None:
                                settings["output_name"] = current.output_name
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

                        current_source = active_source()
                        if current_source is None:
                            ui.label(
                                "Add a source to reveal only the controls that apply to it."
                            ).classes("dn-data-manager-section-guidance")
                            return
                        details = selected_data(current_source)

                        def render_output_controls() -> None:
                            ui.label("Output").classes(
                                "dn-setting-group-title dn-data-manager-setting-heading"
                            )
                            with ui.element("div").classes(
                                "dn-data-manager-destination-grid"
                            ):
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
                                controls["show_preview"] = (
                                    ui.checkbox(
                                        "Show plot preview after conversion",
                                        value=bool(settings["show_preview"]),
                                    )
                                    .classes("dn-data-manager-preview-toggle")
                                    .mark("data-manager-preview-toggle")
                                )

                            def preview_changed() -> None:
                                store_control("show_preview")
                                render_result.refresh()

                            controls["decimal_places"].on_value_change(
                                lambda: store_control("decimal_places")
                            )
                            controls["show_preview"].on_value_change(preview_changed)

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
                                )
                                .mark("data-manager-polishing-grid")
                            ):
                                with (
                                    ui.element("div")
                                    .classes("dn-data-manager-polishing-column")
                                    .mark("data-manager-missing-column")
                                ):
                                    if missing_kind == "track":
                                        controls["track_missing"] = (
                                            ui.select(
                                                _TRACK_MISSING,
                                                value=settings["track_missing"],
                                                label="Empty bins",
                                            )
                                            .props("dense options-dense")
                                            .classes("dn-field")
                                            .mark("data-manager-track-missing")
                                        )
                                        controls["track_missing"].tooltip(
                                            "Minimum and maximum use each chromosome's values "
                                            "after thresholding and smoothing, before value scaling. "
                                            "Zero and custom-value fills are applied after scaling."
                                        )
                                        controls["fill_value"] = (
                                            ui.number(
                                                label="Fill value",
                                                value=settings["fill_value"],
                                            )
                                            .props("dense")
                                            .classes("dn-field")
                                            .mark("data-manager-fill-value")
                                        )
                                        controls["fill_value"].set_visibility(
                                            settings["track_missing"] == "fill"
                                        )

                                        def missing_changed() -> None:
                                            store_control("track_missing")
                                            controls["fill_value"].set_visibility(
                                                str(controls["track_missing"].value)
                                                == "fill"
                                            )

                                        controls["track_missing"].on_value_change(
                                            missing_changed
                                        )
                                        controls["fill_value"].on_value_change(
                                            lambda: store_control("fill_value")
                                        )
                                    else:
                                        controls["missing_policy"] = (
                                            ui.select(
                                                {
                                                    "ignore": "Keep missing values",
                                                    "error": "Stop if values are missing",
                                                    "interpolate": "Interpolate",
                                                    "zero": "Fill with zero",
                                                },
                                                value=settings["missing_policy"],
                                                label="Missing values",
                                            )
                                            .props("dense options-dense")
                                            .classes("dn-field")
                                            .mark("data-manager-profile-missing")
                                        )
                                        controls["missing_policy"].on_value_change(
                                            lambda: store_control("missing_policy")
                                        )

                                if not polishable:
                                    return

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
                                            controls["interpolation"].tooltip(
                                                "Gaps between called peaks are background. "
                                                "Interpolation would create activity between origins."
                                            )
                                        controls["interpolation"].on_value_change(
                                            lambda: store_control("interpolation")
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
                                                label="Target minimum",
                                                value=settings["rescale_min"],
                                            )
                                            .props("dense")
                                            .classes("dn-field")
                                            .mark("data-manager-target-minimum")
                                        )
                                        controls["rescale_max"] = (
                                            ui.number(
                                                label="Target maximum",
                                                value=settings["rescale_max"],
                                            )
                                            .props("dense")
                                            .classes("dn-field")
                                            .mark("data-manager-target-maximum")
                                        )
                                        controls["rescale_invert"] = (
                                            ui.checkbox(
                                                "Invert",
                                                value=bool(settings["rescale_invert"]),
                                            )
                                            .classes("dn-data-manager-inline-toggle")
                                            .mark("data-manager-interval-invert")
                                        )
                                        controls["rescale_invert"].tooltip(
                                            "Map the lowest input value to the target maximum and "
                                            "the highest to the target minimum."
                                        )
                                    with (
                                        ui.element("div")
                                        .classes("dn-data-manager-threshold-grid")
                                        .mark("data-manager-threshold-fields")
                                    ):
                                        controls["threshold_min"] = (
                                            ui.number(
                                                label="Minimum bound (optional)",
                                                value=settings["threshold_min"],
                                            )
                                            .props("dense clearable")
                                            .classes("dn-field")
                                            .mark("data-manager-threshold-min")
                                        )
                                        controls["threshold_max"] = (
                                            ui.number(
                                                label="Maximum bound (optional)",
                                                value=settings["threshold_max"],
                                            )
                                            .props("dense clearable")
                                            .classes("dn-field")
                                            .mark("data-manager-threshold-max")
                                        )
                                    controls["threshold_min"].tooltip(
                                        "Values below this bound become missing. Leave blank "
                                        "to use the Prepare as default or the data minimum."
                                    )
                                    controls["threshold_max"].tooltip(
                                        "Values above this bound become missing. Leave blank "
                                        "to use the Prepare as default or the data maximum."
                                    )

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

                        if (
                            details.form != "track"
                            or current_source.track_inspection is None
                        ):
                            if details.form != "table":
                                ui.label("Values").classes(
                                    "dn-setting-group-title dn-data-manager-setting-heading"
                                )
                                render_value_controls(
                                    missing_kind="profile",
                                    polishable=details.form in _POLISHABLE_FORMS,
                                )
                            else:
                                ui.label("Table columns and labels will be preserved.").classes(
                                    "dn-data-manager-section-guidance"
                                )
                            render_output_controls()
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
                            controls["bedgraph_value_column"].tooltip(
                                "Choose the signal column; chromosome, start, and end "
                                "coordinates are detected automatically."
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
                            ui.label(
                                "BED3 calls have unit signal. Presence marks bins containing "
                                "origins; coverage measures the fraction overlapped. "
                                "Smaller bins separate nearby origins."
                                if current_source.inspection.track_kind == "intervals" else
                                "Use enrichment to retain peak heights. Maximum keeps the "
                                "strongest peak in each bin; coverage measures peak width. "
                                "Smaller bins separate nearby origins. SNS-seq enrichment "
                                "is relative activity; calibrate its scale for firing-rate units."
                                if is_peak_track else
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
                            controls["end"] = (
                                ui.number(
                                    label="End (bp)",
                                    value=settings["end"],
                                    min=region.start + 1,
                                    max=region.end,
                                    step=1_000,
                                )
                                .props("dense")
                                .classes("dn-field")
                                .mark("data-manager-end")
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

                        def chromosome_changed() -> None:
                            selected = str(controls["chromosome"].value)
                            selected_region = regions[selected]
                            settings.update(
                                chromosome=selected,
                                start=selected_region.start,
                                end=selected_region.end,
                            )
                            controls["start"].value = selected_region.start
                            controls["end"].value = selected_region.end
                            controls["start"].update()
                            controls["end"].update()

                        controls["chromosome_scope"].on_value_change(scope_changed)
                        controls["chromosome"].on_value_change(chromosome_changed)
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
                        render_output_controls()

                    render_conversion_settings()

                with action_dock():
                    convert_button = (
                        ui.button("Convert", icon="sync_alt")
                        .props("no-caps")
                        .classes("dn-action-primary")
                        .mark("action-convert")
                    )
                    if not sources_for_mode():
                        convert_button.disable()

        async def download_preview_plot(file_format: str) -> None:
            if not converted_results or result_chart is None:
                ui.notify("Convert data before downloading a plot.", type="warning")
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
                data_uri = await result_chart.run_chart_method(
                    "getDataURL",
                    capture_options,
                    timeout=10,
                )
                latest = converted_results[-1][1]
                artifact = await run.io_bound(
                    build_chart_capture_export,
                    data_uri,
                    file_format,
                    filename=f"{PurePath(latest.artifact.filename).stem}_preview",
                    dpi=dpi,
                )
                if artifact is None or not artifact.data:
                    raise RuntimeError("Plot export stopped before returning data.")
            except (RuntimeError, TimeoutError, TypeError, ValueError) as exc:
                ui.notify(str(exc), type="negative")
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

                    @ui.refreshable
                    def render_result() -> None:
                        nonlocal result_chart
                        result_chart = None
                        if not converted_results:
                            with (
                                ui.column()
                                .classes("dn-data-manager-empty")
                                .mark("data-manager-empty-result")
                            ):
                                ui.icon("file_download")
                                ui.label("Converted files will appear here")
                            return

                        latest_preview = converted_results[-1][1].preview
                        if settings["show_preview"] and latest_preview is not None:
                            with (
                                ui.element("section")
                                .classes("dn-data-manager-prepared-preview")
                                .mark("data-manager-result-preview")
                            ):
                                with ui.row().classes("dn-data-manager-preview-heading"):
                                    ui.label("Prepared profile").classes("dn-setting-group-title")
                                    ui.space()
                                    sampled = len(latest_preview.values)
                                    sample_text = f"{latest_preview.total_count:,} values"
                                    if sampled < latest_preview.total_count:
                                        sample_text += f" · {sampled:,} shown"
                                    ui.label(sample_text).classes(
                                        "dn-data-manager-preview-sampling"
                                    )
                                with ui.element("div").classes("dn-data-manager-preview-stats"):
                                    ui.label(f"Finite {latest_preview.finite_count:,}")
                                    ui.label(f"Missing {latest_preview.missing_count:,}")
                                    if (
                                        latest_preview.minimum is not None
                                        and latest_preview.maximum is not None
                                    ):
                                        ui.label(
                                            "Range "
                                            f"{latest_preview.minimum:.4g} to "
                                            f"{latest_preview.maximum:.4g}"
                                        )
                                result_chart = render_preview_chart(
                                    _preview_options(latest_preview),
                                    title="Prepared profile",
                                    marker="data-manager-preview-chart",
                                    on_download=download_preview_plot,
                                )

                        def remove_result(result_id: int) -> None:
                            converted_results[:] = [
                                item for item in converted_results if item[0] != result_id
                            ]
                            generated_file_preferences.pop(result_id, None)
                            render_result.refresh()

                        def clear_results() -> None:
                            converted_results.clear()
                            generated_file_preferences.clear()
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
                            try:
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
                                    return _build_generated_files_archive(artifacts)

                                bundled = await run.io_bound(prepare_archive)
                            except (OSError, RuntimeError, ValueError, zipfile.BadZipFile) as exc:
                                ui.notify(str(exc), type="negative")
                                return
                            ui.download(
                                bundled.data,
                                bundled.filename,
                                bundled.media_type,
                            )

                        with ui.element("div").classes("dn-download-file-list"):
                            for result_id, result in converted_results:
                                preference = retained_generated_preference(
                                    result_id,
                                    result,
                                )

                                async def download_result(
                                    controls: Any,
                                    retained_result: DataConversionResult = result,
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
                                        ui.notify(str(exc), type="negative")
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
                                        on_remove=(
                                            lambda _controls, retained_id=result_id: (
                                                remove_result(retained_id)
                                            )
                                        ),
                                        markers={
                                            "row": f"data-manager-result-row-{result_id}",
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

    def build_request(
        source: _DataManagerSource,
        *,
        output_name: str,
    ) -> DataConversionRequest:
        current = source.inspection
        details = selected_data(source)
        target_format = "tsv"
        decimal_places = _integer(
            controls["decimal_places"],
            "Decimal places",
            minimum=0,
        )
        kwargs: dict[str, Any] = {}
        if details.form == "track":
            track_summary = source.track_inspection
            if track_summary is None:
                raise ValueError(
                    f"{current.name} could not be inspected as a genomic track."
                )
            chromosome_scope = str(controls["chromosome_scope"].value or "single")
            aggregation = str(controls["aggregation"].value)
            if aggregation not in _track_aggregation_options(track_summary.backend, current):
                raise ValueError(
                    f"{current.name} does not support the {aggregation} aggregation."
                )
            kwargs.update(
                chromosome_scope=chromosome_scope,
                memory_limit_mb=get_export_settings().conversion_memory_limit_mb,
                bin_size=_integer(controls["bin_size"], "Bin size", minimum=1),
                aggregation=aggregation,
                track_missing=str(controls["track_missing"].value),
                fill_value=_number(controls["fill_value"], "Fill value"),
                interpolation=str(controls["interpolation"].value),
            )
            if chromosome_scope == "single":
                chromosome = str(controls["chromosome"].value)
                regions = {
                    region.chromosome: region for region in track_summary.regions
                }
                if chromosome not in regions:
                    raise ValueError(
                        f"{current.name} does not contain chromosome {chromosome}."
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
                        f"{current.name} covers {chromosome}:"
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
        if details.form in _POLISHABLE_FORMS:
            value_scaling = str(controls["value_scaling"].value)
            threshold_enabled, threshold_min, threshold_max = (
                _resolved_feature_bounds(
                    str(settings["feature"]),
                    _optional_number(controls["threshold_min"], "Lower threshold"),
                    _optional_number(controls["threshold_max"], "Upper threshold"),
                )
            )
            kwargs.update(
                smooth_window=_integer(
                    controls["smooth_window"],
                    "Smoothing window",
                    minimum=1,
                ),
                scaling_mode=value_scaling,
                normalize="none",
                rescale_min=(
                    _number(controls["rescale_min"], "Target minimum")
                    if value_scaling == "interval"
                    else 0.0
                ),
                rescale_max=(
                    _number(controls["rescale_max"], "Target maximum")
                    if value_scaling == "interval"
                    else 1.0
                ),
                rescale_invert=(
                    bool(controls["rescale_invert"].value)
                    if value_scaling == "interval"
                    else False
                ),
                threshold_enabled=threshold_enabled,
                threshold_min=threshold_min,
                threshold_max=threshold_max,
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
            missing_policy=(
                str(controls["missing_policy"].value)
                if "missing_policy" in controls
                else "ignore"
            ),
            decimal_places=decimal_places,
            value_name=_selected_value_name(
                source,
                details,
                str(settings["feature"]),
            ),
            genome_build=None,
            units=None,
            source_url=str(source.remote_metadata.get("final_url") or "") or None,
            biological_role=str(settings["feature"]),
            semantic_note=_FEATURE_HINTS[str(settings["feature"])],
            **kwargs,
        )

    def build_requests() -> tuple[
        tuple[_DataManagerSource, DataConversionRequest],
        ...,
    ]:
        if pending_links := unresolved_link_rows():
            noun = "link" if len(pending_links) == 1 else "links"
            raise ValueError(
                f"Inspect the {len(pending_links)} pending public {noun} before converting."
            )
        candidates = tuple(sources_for_mode())
        if not candidates:
            raise ValueError("Add at least one data source before converting.")

        forms = tuple(selected_data(source).form for source in candidates)
        if len(set(forms)) != 1:
            readable = ", ".join(
                f"{source.inspection.name}: "
                f"{_FORM_LABELS.get(form, form.replace('_', ' ').title())}"
                for source, form in zip(candidates, forms, strict=True)
            )
            raise ValueError(
                "All sources in one conversion batch must use the same data type. "
                f"Choose compatible profiles ({readable})."
            )

        requested_output_names: list[str] = []
        for index, source in enumerate(candidates, start=1):
            base = (
                str(source.output_name).strip()
                or _automatic_output_name(
                    source.inspection.name,
                    str(settings["feature"]),
                )
            )
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

    async def convert() -> None:
        nonlocal active_cancel, conversion_running, next_result_id
        if conversion_running:
            if active_cancel is not None and not active_cancel.is_set():
                active_cancel.set()
                conversion_status.text = "Stopping conversion..."
                convert_button.disable()
            return
        if upload_inspection_running or remote_inspections:
            ui.notify(
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
        convert_button.set_text("Stop")
        convert_button.set_icon("stop")
        lock_request_controls(True)
        convert_button.enable()
        try:
            requests = build_requests()
            check_cancelled()
            started = time.perf_counter()
            total = len(requests)
            completed: list[DataConversionResult] = []
            failures: list[tuple[str, str]] = []
            conversion_progress.visible = True
            conversion_progress.value = 0.01
            for index, (source, request) in enumerate(requests):
                progress_state = _ConversionProgressState()
                item_number = index + 1
                conversion_status.text = (
                    f"Converting {item_number}/{total} - {source.inspection.name}"
                )
                worker = asyncio.create_task(
                    run.io_bound(
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
                conversion_progress.value = item_number / total
                check_cancelled()

            check_cancelled()
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
                render_result.refresh()

            elapsed = time.perf_counter() - started
            conversion_progress.value = 1.0
            if failures:
                conversion_status.text = (
                    f"Completed {len(completed)}/{total} in {elapsed:.2f} s"
                )
                failed_names = ", ".join(name for name, _message in failures)
                detail = "; ".join(
                    f"{name}: {message}" for name, message in failures
                )
                ui.notify(
                    f"Converted {len(completed)} of {total}. "
                    f"Could not convert {failed_names}. {detail}",
                    type="warning" if completed else "negative",
                )
            else:
                conversion_status.text = f"Complete in {elapsed:.2f} s"
                if total == 1:
                    ui.notify(
                        f"Converted to {completed[0].artifact.filename}.",
                        type="positive",
                    )
                else:
                    ui.notify(f"Converted {total} files.", type="positive")
            if evicted_results:
                ui.notify(
                    f"Removed {evicted_results} oldest generated "
                    f"{'file' if evicted_results == 1 else 'files'} to keep the local "
                    "workspace responsive.",
                    type="warning",
                )
        except CancelledError:
            conversion_progress.visible = False
            conversion_status.text = "Conversion stopped"
        except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
            conversion_progress.visible = False
            conversion_status.text = "Conversion could not be completed"
            ui.notify(str(exc), type="negative")
        finally:
            conversion_running = False
            active_cancel = None
            convert_button.set_text("Convert")
            convert_button.set_icon("sync_alt")
            lock_request_controls(False)
            set_convert_enabled()

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

    def refresh_data_manager_page() -> None:
        """Apply global presentation preferences without rebuilding inputs."""

        link_examples.set_visibility(get_export_settings().show_examples)

    current_session_cache().register_page_refresher(
        "data-management",
        refresh_data_manager_page,
    )
