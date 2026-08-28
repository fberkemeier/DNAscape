"""Interactive data conversion through :func:`dnascape.rdata`."""

from __future__ import annotations

import asyncio
import time
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePath
from threading import Lock
from typing import Any

import numpy as np
from nicegui import run, ui

from app.artifact_archives import build_artifact_archive
from app.data_conversion_service import (
    DATA_EXPORT_FORMATS,
    DataConversionRequest,
    DataConversionResult,
    TrackInspection,
    convert_data,
    inspect_track,
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
    set_data_upload_locked,
    two_column_fields,
)
from app.preview_plotting import build_preview_plot_options, render_preview_chart
from app.remote_sources import RemoteSourceError, fetch_remote_source
from app.result_naming import safe_filename_stem
from app.session_cache import cache_artifact, cache_dataset, cache_upload, current_session_cache
from app.simulation_exports import ExportArtifact, build_chart_capture_export

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
_TARGET_FORMATS = dict(DATA_EXPORT_FORMATS)
_TEXT_TARGETS = {"txt", "tsv", "csv"}
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
_ENCODE_EXAMPLE_URL = (
    "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/encodeDCC/"
    "wgEncodeUwRepliSeq/wgEncodeUwRepliSeqK562WaveSignalRep1.bigWig"
)


@dataclass(frozen=True)
class _SelectedData:
    form: str
    shape: tuple[int, ...]
    value_column: str | None = None
    array_key: str | None = None


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
    return safe_filename_stem(
        path.stem,
        fallback="data",
        maximum=96,
    )


def _automatic_output_name(source_name: str, feature: str) -> str:
    """Use the shared workflow/output/source order for converted data."""

    output = (
        "converted"
        if feature == "generic"
        else _FEATURE_VALUE_NAMES.get(feature, feature)
    )
    output_stem = safe_filename_stem(output, fallback="converted")
    source_stem = _source_stem(source_name)
    if source_stem == output_stem:
        source_stem = "profile"
    elif source_stem.startswith(f"{output_stem}-"):
        source_stem = source_stem.removeprefix(f"{output_stem}-")
    return f"dnascape_data_{output_stem}_{source_stem}"


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


def _build_generated_files_archive(
    artifacts: Sequence[ExportArtifact],
) -> ExportArtifact:
    """Backward-compatible Data Manager wrapper around the shared helper."""

    return build_artifact_archive(
        artifacts,
        filename="dnascape_data_outputs.zip",
    )


def _selection_options(inspection: UploadInspection) -> Mapping[str, str]:
    if inspection.source_format in {"bigwig", "bedgraph"}:
        if len(inspection.numeric_columns) > 1:
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
            options[_MATRIX] = "All profiles (space-time)"
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
        return _SelectedData(
            "xt",
            (rows, len(inspection.numeric_columns)),
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


def _target_formats(
    input_form: str,
    *,
    chromosome_scope: str = "single",
) -> Mapping[str, str]:
    if input_form == "table":
        return {key: _TARGET_FORMATS[key] for key in ("tsv", "csv")}
    if input_form == "track" and chromosome_scope == "all":
        return {key: _TARGET_FORMATS[key] for key in ("tsv", "csv", "npz")}
    return dict(_TARGET_FORMATS)


def _track_aggregation_options(backend: str | None) -> dict[str, str]:
    options = dict(_TRACK_AGGREGATIONS)
    if backend == "gwseq_io_pp":
        options.pop("min", None)
        options.pop("max", None)
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

    inspection: UploadInspection | None = None
    track_inspection: TrackInspection | None = None
    remote_metadata: dict[str, Any] = {}
    selection = _NATURAL
    converted_results: list[tuple[int, DataConversionResult]] = []
    next_result_id = 0
    conversion_running = False
    result_chart: Any | None = None
    controls: dict[str, Any] = {}
    export_controls: dict[str, Any] = {}
    source_controls: dict[str, Any] = {}
    source_control_states: dict[str, bool] = {}
    request_control_states: dict[tuple[str, str], bool] = {}
    settings: dict[str, Any] = {
        "source_mode": "upload",
        "source_url": "",
        "feature": "generic",
        "target_format": "tsv",
        "output_name": "dnascape_data",
        "gzip_output": False,
        "decimal_places": 6,
        "missing_policy": "ignore",
        "chromosome_scope": "single",
        "chromosome": None,
        "start": 0,
        "end": 1,
        "bin_size": 10_000,
        "aggregation": "mean",
        "track_missing": "nan",
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
        "value_name": "value",
        "genome_build": "",
        "units": "",
        "show_preview": False,
    }

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
        for group_name, group in (("configure", controls), ("export", export_controls)):
            for name, control in tuple(group.items()):
                if getattr(control, "is_deleted", False):
                    continue
                key = (group_name, name)
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

                    @ui.refreshable
                    def render_data_source() -> None:
                        current = inspection
                        source_controls.clear()

                        async def adopt_candidate(candidate: UploadInspection) -> bool:
                            nonlocal inspection, selection, track_inspection
                            if conversion_running:
                                ui.notify(
                                    "Wait for the current conversion to finish before changing "
                                    "the source.",
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
                                if inspection is not None and not conversion_running:
                                    convert_button.enable()
                                return False
                            inspection = candidate
                            track_inspection = candidate_track
                            selection = candidate_selection
                            cache_upload(
                                candidate,
                                source="Data Manager",
                                navigate_to="data-management",
                            )
                            feature = str(settings["feature"])
                            settings["output_name"] = _automatic_output_name(
                                candidate.name,
                                feature,
                            )
                            if candidate_track is not None:
                                region = candidate_track.regions[0]
                                settings.update(
                                    chromosome=region.chromosome,
                                    start=region.start,
                                    end=region.end,
                                )
                            await render_data_source.refresh()
                            await render_conversion_settings.refresh()
                            await render_export_settings.refresh()
                            convert_button.enable()
                            return True

                        async def uploaded(event: Any) -> bool:
                            if conversion_running:
                                ui.notify("A conversion is already running.", type="warning")
                                return False
                            try:
                                candidate = await inspect_upload_event(
                                    event,
                                    expected_form="auto",
                                )
                            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                                ui.notify(str(exc), type="negative")
                                return False
                            remote_metadata.clear()
                            return await adopt_candidate(candidate)

                        async def cleared(_event: Any) -> None:
                            nonlocal inspection, selection, track_inspection
                            if conversion_running:
                                ui.notify("A conversion is already running.", type="warning")
                                return
                            inspection = None
                            track_inspection = None
                            selection = _NATURAL
                            remote_metadata.clear()
                            convert_button.disable()
                            await render_data_source.refresh()
                            await render_conversion_settings.refresh()
                            await render_export_settings.refresh()

                        def feature_changed() -> None:
                            feature = str(feature_select.value or "generic")
                            settings["feature"] = feature
                            settings["value_name"] = _FEATURE_VALUE_NAMES[feature]
                            if inspection is not None:
                                settings["output_name"] = _automatic_output_name(
                                    inspection.name,
                                    feature,
                                )
                                render_conversion_settings.refresh()
                                render_export_settings.refresh()

                        feature_select = (
                            ui.select(
                                _FEATURES,
                                value=settings["feature"],
                                label="Prepare as",
                            )
                            .props("dense options-dense")
                            .classes("dn-field dn-data-manager-feature-select")
                            .mark("data-manager-feature")
                        )
                        feature_select.on_value_change(feature_changed)
                        source_controls["feature"] = feature_select

                        with (
                            ui.tabs(value=settings["source_mode"])
                            .classes("dn-data-manager-source-tabs")
                            .mark("data-manager-source-tabs")
                        ) as source_tabs:
                            ui.tab("upload", label="Upload", icon="upload_file")
                            ui.tab("url", label="Link", icon="link")

                        def source_mode_changed(event: Any) -> None:
                            settings["source_mode"] = str(event.value)

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
                                    accept=RDATA_UPLOAD_ACCEPT,
                                    accepted_formats=(
                                        "CSV, TSV, TXT, NPY, NPV, NPZ, bedGraph, BigWig, GZ"
                                    ),
                                    marker="data-manager-upload",
                                    show_accepted_formats=False,
                                    on_upload=uploaded,
                                    on_clear=cleared,
                                    max_files=1,
                                    hide_when_loaded=True,
                                    initial_names=(
                                        (current.name,)
                                        if current is not None
                                        and settings["source_mode"] == "upload"
                                        else ()
                                    ),
                                )
                            with ui.tab_panel("url"):
                                with ui.element("div").classes("dn-data-manager-url-grid"):
                                    url_input = (
                                        ui.input(
                                            label="Public file link",
                                            value=settings["source_url"],
                                            placeholder="https://.../signal.bigWig",
                                        )
                                        .props("dense clearable")
                                        .classes("dn-field")
                                        .mark("data-manager-source-url")
                                    )
                                    source_controls["url"] = url_input

                                    async def inspect_url() -> None:
                                        if conversion_running:
                                            ui.notify(
                                                "A conversion is already running.",
                                                type="warning",
                                            )
                                            return
                                        url = str(url_input.value or "").strip()
                                        settings["source_url"] = url
                                        if not url:
                                            ui.notify(
                                                "Paste a public file link first.", type="warning"
                                            )
                                            return
                                        link_button.disable()
                                        convert_button.disable()
                                        url_progress.visible = True
                                        url_status.visible = True
                                        url_status.text = "Downloading and inspecting..."
                                        try:
                                            remote = await run.io_bound(fetch_remote_source, url)
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
                                            adopted = await adopt_candidate(candidate)
                                            if not adopted:
                                                remote_metadata.clear()
                                                return
                                            remote_metadata.update(
                                                final_url=remote.final_url,
                                                size=remote.size,
                                                accept_ranges=remote.accept_ranges,
                                                etag=remote.etag,
                                                last_modified=remote.last_modified,
                                            )
                                            await render_data_source.refresh()
                                            ui.notify("Linked data are ready.", type="positive")
                                        except (
                                            OSError,
                                            RemoteSourceError,
                                            RuntimeError,
                                            TypeError,
                                            ValueError,
                                        ) as exc:
                                            remote_metadata.clear()
                                            ui.notify(str(exc), type="negative")
                                        finally:
                                            if not url_progress.is_deleted:
                                                url_progress.visible = False
                                            if not url_status.is_deleted:
                                                url_status.visible = False
                                            if not link_button.is_deleted:
                                                link_button.enable()
                                            if inspection is not None:
                                                convert_button.enable()

                                    link_button = (
                                        ui.button(
                                            "Inspect", icon="arrow_forward", on_click=inspect_url
                                        )
                                        .props("no-caps")
                                        .classes("dn-action-secondary dn-data-manager-link-action")
                                        .mark("action-inspect-data-link")
                                    )
                                    source_controls["link"] = link_button
                                    url_input.on("keydown.enter", inspect_url)
                                ui.label(
                                    "Paste a public HTTPS link to a BigWig or bedGraph file. "
                                    "Private-network and sign-in links are not supported."
                                ).classes("dn-data-manager-url-help")

                                def use_encode_example() -> None:
                                    settings["source_url"] = _ENCODE_EXAMPLE_URL
                                    settings["feature"] = "replication_timing"
                                    settings["value_name"] = _FEATURE_VALUE_NAMES[
                                        "replication_timing"
                                    ]
                                    settings["genome_build"] = "hg19"
                                    settings["units"] = "relative wavelet signal"
                                    url_input.value = _ENCODE_EXAMPLE_URL
                                    url_input.update()
                                    feature_select.value = "replication_timing"
                                    feature_select.update()

                                (
                                    ui.button(
                                        "Example: ENCODE Repli-seq",
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
                                url_progress = (
                                    ui.linear_progress(value=0.0)
                                    .props("indeterminate rounded")
                                    .classes("dn-data-manager-url-progress")
                                    .mark("data-manager-url-progress")
                                )
                                url_progress.visible = False
                                url_status = (
                                    ui.label("Downloading and inspecting...")
                                    .classes("dn-data-manager-url-status")
                                    .mark("data-manager-url-status")
                                )
                                url_status.visible = False
                        if current is None:
                            return

                        options = _selection_options(current)
                        source_select = (
                            ui.select(
                                dict(options),
                                value=selection,
                                label="Profile",
                            )
                            .props("dense options-dense")
                            .classes("dn-field dn-data-manager-source-select")
                            .mark("data-manager-source-selection")
                        )
                        if len(options) == 1:
                            source_select.disable()

                        def source_changed() -> None:
                            nonlocal selection
                            if source_select.value is None:
                                return
                            selection = str(source_select.value)
                            render_data_source.refresh()
                            render_conversion_settings.refresh()
                            render_export_settings.refresh()

                        source_select.on_value_change(source_changed)
                        source_controls["selection"] = source_select
                        details = _selected_data(current, selection)
                        with (
                            ui.element("div")
                            .classes(
                                "dn-mapping-detected-data dn-profile-metadata "
                                "dn-data-manager-detected"
                            )
                            .mark("data-manager-detected-data")
                        ):
                            ui.label(
                                f"Format: {_source_format_label(current.source_format)}"
                            ).classes("dn-mapping-detected-summary")
                            ui.label(f"Size: {_file_size_text(len(current.content))}").classes(
                                "dn-mapping-detected-summary"
                            )
                            ui.label(
                                f"Dependence: {_FORM_LABELS.get(details.form, 'Unknown')}"
                            ).classes("dn-mapping-detected-summary dn-mapping-detected-dependence")
                            if track_inspection is not None:
                                region = track_inspection.regions[0]
                                estimated_bins = max(
                                    1,
                                    (region.end - region.start + int(settings["bin_size"]) - 1)
                                    // int(settings["bin_size"]),
                                )
                                ui.label(f"Estimated bins: {estimated_bins:,}").classes(
                                    "dn-mapping-detected-summary"
                                )
                                ui.label(f"Chromosomes: {len(track_inspection.regions)}").classes(
                                    "dn-mapping-detected-summary"
                                )
                            else:
                                ui.label(f"Array size: {_shape_text(details.shape)}").classes(
                                    "dn-mapping-detected-summary"
                                )
                            if remote_metadata:
                                ui.label("Source: Public link").classes(
                                    "dn-mapping-detected-summary"
                                )
                        if conversion_running:
                            lock_source_controls(True)

                    render_data_source()

                with expansion("Configure", icon="tune", expanded=True):

                    @ui.refreshable
                    def render_conversion_settings() -> None:
                        controls.clear()
                        current = inspection
                        if current is None:
                            ui.label(
                                "Add a source to reveal only the controls that apply to it."
                            ).classes("dn-data-manager-section-guidance")
                            return
                        details = _selected_data(current, selection)

                        def store_control(name: str) -> None:
                            settings[name] = controls[name].value

                        def render_metadata_fields() -> None:
                            ui.label("Metadata").classes(
                                "dn-setting-group-title dn-data-manager-setting-heading"
                            )
                            with ui.element("div").classes("dn-data-manager-metadata-grid"):
                                controls["genome_build"] = (
                                    ui.input(
                                        label="Genome build (optional)",
                                        value=settings["genome_build"],
                                        placeholder="e.g. hg38",
                                    )
                                    .props("dense")
                                    .classes("dn-field")
                                    .mark("data-manager-genome-build")
                                )
                                controls["units"] = (
                                    ui.input(
                                        label="Units (optional)",
                                        value=settings["units"],
                                    )
                                    .props("dense")
                                    .classes("dn-field")
                                    .mark("data-manager-units")
                                )
                            for name in ("genome_build", "units"):
                                controls[name].on_value_change(
                                    lambda name=name: store_control(name)
                                )

                        def render_polishing_controls() -> None:
                            ui.label("Polish values").classes(
                                "dn-setting-group-title dn-data-manager-setting-heading"
                            )
                            with ui.element("div").classes("dn-data-manager-polishing-grid"):
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
                                controls["value_scaling"] = (
                                    ui.select(
                                        _VALUE_SCALING,
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

                            controls["threshold_enabled"] = (
                                ui.checkbox(
                                    "Set values outside thresholds to missing",
                                    value=bool(settings["threshold_enabled"]),
                                )
                                .classes("dn-data-manager-threshold-toggle")
                                .mark("data-manager-threshold-enabled")
                            )
                            controls["threshold_enabled"].tooltip(
                                "Positions are preserved; only values outside the inclusive "
                                "bounds become missing."
                            )
                            with (
                                ui.element("div")
                                .classes("dn-data-manager-threshold-grid")
                                .mark("data-manager-threshold-fields")
                            ) as threshold_fields:
                                controls["threshold_min"] = (
                                    ui.number(
                                        label="Lower bound (optional)",
                                        value=settings["threshold_min"],
                                    )
                                    .props("dense clearable")
                                    .classes("dn-field")
                                    .mark("data-manager-threshold-min")
                                )
                                controls["threshold_max"] = (
                                    ui.number(
                                        label="Upper bound (optional)",
                                        value=settings["threshold_max"],
                                    )
                                    .props("dense clearable")
                                    .classes("dn-field")
                                    .mark("data-manager-threshold-max")
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
                            threshold_fields.set_visibility(bool(settings["threshold_enabled"]))

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

                            def threshold_changed() -> None:
                                store_control("threshold_enabled")
                                threshold_fields.set_visibility(
                                    bool(controls["threshold_enabled"].value)
                                )

                            controls["smoothing_mode"].on_value_change(smoothing_changed)
                            controls["value_scaling"].on_value_change(value_scaling_changed)
                            controls["threshold_enabled"].on_value_change(threshold_changed)
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

                        if details.form != "track" or track_inspection is None:
                            if details.form != "table":
                                ui.label("Values").classes("dn-setting-group-title")
                                with two_column_fields():
                                    controls["value_name"] = (
                                        ui.input(
                                            label="Value name",
                                            value=settings["value_name"],
                                        )
                                        .props("dense")
                                        .classes("dn-field")
                                        .mark("data-manager-value-name")
                                    )
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
                                for name in ("value_name", "missing_policy"):
                                    controls[name].on_value_change(
                                        lambda name=name: store_control(name)
                                    )
                                if details.form in _POLISHABLE_FORMS:
                                    render_polishing_controls()
                            else:
                                ui.label("Table columns and labels will be preserved.").classes(
                                    "dn-data-manager-section-guidance"
                                )
                            return

                        ui.label("Region & binning").classes("dn-setting-group-title")
                        regions = {item.chromosome: item for item in track_inspection.regions}
                        chromosome = str(settings["chromosome"] or next(iter(regions)))
                        if chromosome not in regions:
                            chromosome = next(iter(regions))
                        region = regions[chromosome]
                        aggregation_options = _track_aggregation_options(track_inspection.backend)
                        if settings["aggregation"] not in aggregation_options:
                            settings["aggregation"] = "mean"
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
                            render_export_settings.refresh()

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
                            render_export_settings.refresh()

                        controls["chromosome_scope"].on_value_change(scope_changed)
                        controls["chromosome"].on_value_change(chromosome_changed)
                        for name in (
                            "start",
                            "end",
                            "bin_size",
                            "aggregation",
                        ):
                            controls[name].on_value_change(lambda name=name: store_control(name))
                        render_metadata_fields()

                        ui.label("Missing values").classes(
                            "dn-setting-group-title dn-data-manager-setting-heading"
                        )
                        with ui.element("div").classes("dn-data-manager-preprocessing-grid"):
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
                            controls["fill_value"] = (
                                ui.number(
                                    label="Fill value",
                                    value=settings["fill_value"],
                                )
                                .props("dense")
                                .classes("dn-field")
                                .mark("data-manager-fill-value")
                            )
                            controls["interpolation"] = (
                                ui.select(
                                    _TRACK_INTERPOLATION,
                                    value=settings["interpolation"],
                                    label="Interpolation",
                                )
                                .props("dense options-dense")
                                    .classes("dn-field")
                                    .mark("data-manager-interpolation")
                                )
                        controls["fill_value"].set_visibility(settings["track_missing"] == "fill")

                        def missing_changed() -> None:
                            store_control("track_missing")
                            controls["fill_value"].set_visibility(
                                str(controls["track_missing"].value) == "fill"
                            )

                        controls["track_missing"].on_value_change(missing_changed)
                        for name in ("fill_value", "interpolation"):
                            controls[name].on_value_change(lambda name=name: store_control(name))
                        render_polishing_controls()

                    render_conversion_settings()

                with expansion("Export settings", icon="file_download", expanded=False):

                    @ui.refreshable
                    def render_export_settings() -> None:
                        export_controls.clear()
                        current = inspection
                        if current is None:
                            ui.label("Add a source to choose the output file settings.").classes(
                                "dn-data-manager-section-guidance"
                            ).mark("data-manager-export-guidance")
                            return

                        details = _selected_data(current, selection)
                        target_options = _target_formats(
                            details.form,
                            chromosome_scope=str(settings["chromosome_scope"]),
                        )
                        if settings["target_format"] not in target_options:
                            settings["target_format"] = "tsv"

                        with ui.element("div").classes("dn-data-manager-destination-grid"):
                            export_controls["target_format"] = (
                                ui.select(
                                    dict(target_options),
                                    value=settings["target_format"],
                                    label="File type",
                                )
                                .props("dense options-dense")
                                .classes("dn-field")
                                .mark("data-manager-output-format")
                            )
                            export_controls["output_name"] = (
                                ui.input(
                                    label="File name",
                                    value=settings["output_name"],
                                )
                                .props("dense")
                                .classes("dn-field dn-data-manager-output-name")
                                .mark("data-manager-output-name")
                            )
                            export_controls["decimal_places"] = (
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
                            export_controls["gzip_output"] = (
                                ui.checkbox(
                                    "Compress (.gz)",
                                    value=bool(settings["gzip_output"]),
                                )
                                .classes("dn-data-manager-compression")
                                .mark("data-manager-gzip")
                            )

                        export_controls["decimal_places"].set_visibility(
                            settings["target_format"] in _TEXT_TARGETS
                        )
                        export_controls["show_preview"] = (
                            ui.checkbox(
                                "Show plot preview after conversion",
                                value=bool(settings["show_preview"]),
                            )
                            .classes("dn-data-manager-preview-toggle")
                            .mark("data-manager-preview-toggle")
                        )

                        def store_export(name: str) -> None:
                            settings[name] = export_controls[name].value

                        def target_changed() -> None:
                            store_export("target_format")
                            export_controls["decimal_places"].set_visibility(
                                str(export_controls["target_format"].value) in _TEXT_TARGETS
                            )

                        def preview_changed() -> None:
                            store_export("show_preview")
                            render_result.refresh()

                        export_controls["target_format"].on_value_change(target_changed)
                        export_controls["show_preview"].on_value_change(preview_changed)
                        for name in ("output_name", "decimal_places", "gzip_output"):
                            export_controls[name].on_value_change(
                                lambda name=name: store_export(name)
                            )

                    render_export_settings()

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

                with action_dock():
                    convert_button = (
                        ui.button("Convert", icon="sync_alt")
                        .props("no-caps")
                        .classes("dn-action-primary")
                        .mark("action-convert")
                    )
                    if inspection is None:
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
                            render_result.refresh()

                        def clear_results() -> None:
                            converted_results.clear()
                            render_result.refresh()

                        async def download_all_results() -> None:
                            artifacts = tuple(
                                result.artifact for _result_id, result in converted_results
                            )
                            if not artifacts:
                                return
                            try:
                                bundled = await run.io_bound(
                                    _build_generated_files_archive,
                                    artifacts,
                                )
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

                                def download_result(result: DataConversionResult = result) -> None:
                                    artifact = result.artifact
                                    ui.download(
                                        artifact.data,
                                        artifact.filename,
                                        artifact.media_type,
                                    )

                                with (
                                    ui.element("div")
                                    .classes("dn-download-file-row dn-data-manager-file")
                                    .mark(f"data-manager-result-{result_id}")
                                ):
                                    ui.icon("description").classes(
                                        "dn-download-file-icon dn-data-manager-file-icon"
                                    )
                                    with ui.column().classes(
                                        "dn-download-file-copy dn-data-manager-file-copy"
                                    ):
                                        ui.label(result.artifact.filename).classes(
                                            "dn-download-file-name"
                                        )
                                        ui.label(
                                            " | ".join(
                                                (
                                                    _FORM_LABELS.get(
                                                        result.data_form,
                                                        result.data_form.title(),
                                                    ),
                                                    _shape_text(result.shape),
                                                    result.target_format.upper(),
                                                    _file_size_text(len(result.artifact.data)),
                                                )
                                            )
                                        ).classes("dn-data-manager-file-detail")
                                    with ui.row().classes("dn-download-file-actions"):
                                        download_button = (
                                            ui.button(
                                                icon="download",
                                                on_click=download_result,
                                            )
                                            .props(
                                                'flat round dense aria-label="Download converted file"'
                                            )
                                            .classes("dn-download-file-action")
                                            .mark(
                                                "action-download-converted-data",
                                                f"action-download-converted-data-{result_id}",
                                            )
                                        )
                                        download_button.tooltip(
                                            f"Download {result.artifact.filename}"
                                        )
                                        remove_button = (
                                            ui.button(
                                                icon="close",
                                                on_click=lambda _event, result_id=result_id: (
                                                    remove_result(result_id)
                                                ),
                                            )
                                            .props(
                                                'flat round dense aria-label="Remove converted file"'
                                            )
                                            .classes("dn-download-file-action")
                                            .mark(f"remove-converted-data-{result_id}")
                                        )
                                        remove_button.tooltip("Remove")
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

    def build_request() -> DataConversionRequest:
        current = inspection
        if current is None:
            raise ValueError("Upload a data file before converting.")
        details = _selected_data(current, selection)
        if not export_controls:
            raise ValueError("Choose export settings.")
        target_format = str(export_controls["target_format"].value)
        decimal_places = (
            _integer(export_controls["decimal_places"], "Decimal places", minimum=0)
            if target_format in _TEXT_TARGETS
            else None
        )
        kwargs: dict[str, Any] = {}
        if details.form == "track":
            chromosome_scope = str(controls["chromosome_scope"].value or "single")
            kwargs.update(
                chromosome_scope=chromosome_scope,
                bin_size=_integer(controls["bin_size"], "Bin size", minimum=1),
                aggregation=str(controls["aggregation"].value),
                track_missing=str(controls["track_missing"].value),
                fill_value=_number(controls["fill_value"], "Fill value"),
                interpolation=str(controls["interpolation"].value),
            )
            if chromosome_scope == "single":
                start = _integer(controls["start"], "Start", minimum=0)
                end = _integer(controls["end"], "End", minimum=1)
                if end <= start:
                    raise ValueError("End must be greater than start for the genomic region.")
                kwargs.update(
                    chromosome=str(controls["chromosome"].value),
                    start=start,
                    end=end,
                )
            else:
                kwargs.update(chromosome=None, start=None, end=None)
        if details.form in _POLISHABLE_FORMS:
            threshold_enabled = bool(controls["threshold_enabled"].value)
            value_scaling = str(controls["value_scaling"].value)
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
                threshold_min=(
                    _optional_number(controls["threshold_min"], "Lower threshold")
                    if threshold_enabled
                    else None
                ),
                threshold_max=(
                    _optional_number(controls["threshold_max"], "Upper threshold")
                    if threshold_enabled
                    else None
                ),
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
            input_form=details.form,
            target_format=target_format,
            output_name=str(export_controls["output_name"].value or ""),
            gzip_output=bool(export_controls["gzip_output"].value),
            value_column=details.value_column,
            array_key=details.array_key,
            missing_policy=(
                str(controls["missing_policy"].value) if "missing_policy" in controls else "ignore"
            ),
            decimal_places=decimal_places,
            value_name=(
                str(controls["value_name"].value or "value")
                if "value_name" in controls
                else "value"
            ),
            genome_build=(
                str(controls["genome_build"].value or "") or None
                if "genome_build" in controls
                else None
            ),
            units=(str(controls["units"].value or "") or None if "units" in controls else None),
            source_url=str(remote_metadata.get("final_url") or "") or None,
            biological_role=str(settings["feature"]),
            semantic_note=_FEATURE_HINTS[str(settings["feature"])],
            **kwargs,
        )

    async def convert() -> None:
        nonlocal conversion_running, next_result_id
        if conversion_running:
            ui.notify("A conversion is already running.", type="warning")
            return
        conversion_running = True
        convert_button.disable()
        lock_request_controls(True)
        try:
            request = build_request()
            started = time.perf_counter()
            progress_state = _ConversionProgressState()
            conversion_progress.visible = True
            conversion_progress.value = 0.01
            conversion_status.text = "Loading data..."
            worker = asyncio.create_task(run.io_bound(convert_data, request, progress_state.report))
            while not worker.done():
                value, message = progress_state.snapshot()
                conversion_progress.value = max(0.01, value)
                conversion_status.text = message
                await asyncio.sleep(0.04)
            result = await worker
            if result is None:
                raise RuntimeError("Conversion stopped before returning data.")
            conversion_progress.value = 1.0
            conversion_status.text = f"Complete in {time.perf_counter() - started:.2f} s"
            next_result_id += 1
            converted_results.append((next_result_id, result))
            cache_dataset(result)
            render_result.refresh()
            ui.notify(f"Converted to {result.artifact.filename}.", type="positive")
        except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
            conversion_progress.visible = False
            conversion_status.text = "Conversion could not be completed"
            ui.notify(str(exc), type="negative")
        finally:
            conversion_running = False
            lock_request_controls(False)
            if inspection is None:
                convert_button.disable()
            else:
                convert_button.enable()

    convert_button.on_click(convert)

    def clear_conversion_outputs() -> None:
        converted_results.clear()
        conversion_progress.visible = False
        conversion_status.text = "Ready"
        render_result.refresh()

    current_session_cache().register_output_clearer(
        "data-management",
        clear_conversion_outputs,
    )
