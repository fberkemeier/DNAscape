"""Interactive data conversion through :func:`dnascape.rdata`."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import PurePath
from typing import Any

import numpy as np
from nicegui import run, ui

from app.data_conversion_service import (
    DataConversionRequest,
    DataConversionResult,
    TrackInspection,
    convert_data,
    inspect_track,
)
from app.mapping_catalog import RDATA_UPLOAD_ACCEPT
from app.mapping_uploads import UploadInspection, inspect_upload_event, materialize_upload
from app.pages.common import (
    DATA_ICON,
    action_dock,
    data_upload,
    expansion,
    page_intro,
    panel,
    panel_title,
    two_column_fields,
)
from app.session_cache import cache_dataset, cache_upload

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
_OUTPUT_FORM_LABELS = {
    "scalar": "Constant",
    "space": "Space",
    "time": "Time",
    "spacetime": "Space-time",
    "table": "Table",
}
_TARGET_FORMATS = {
    "npy": "NPY",
    "npv": "NPV",
    "npz": "NPZ",
    "txt": "TXT",
    "tsv": "TSV",
    "csv": "CSV",
}
_TEXT_TARGETS = {"txt", "tsv", "csv"}


@dataclass(frozen=True)
class _SelectedData:
    form: str
    shape: tuple[int, ...]
    value_column: str | None = None
    array_key: str | None = None


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
    return path.stem or "dnascape_data"


def _shape_text(shape: tuple[int, ...]) -> str:
    return " x ".join(str(value) for value in shape) if shape else "1"


def _file_size_text(size: int) -> str:
    value = float(max(0, size))
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"


def _selection_options(inspection: UploadInspection) -> Mapping[str, str]:
    if inspection.source_format in {"bigwig", "bedgraph"}:
        if len(inspection.numeric_columns) > 1:
            return {
                f"{_COLUMN_PREFIX}{column}": column
                for column in inspection.numeric_columns
            }
        return {_NATURAL: "Binned profile"}

    if inspection.array_keys:
        options: dict[str, str] = {}
        for key in inspection.array_keys:
            values = materialize_upload(
                replace(inspection, expected_form="auto"),
                array_key=key,
                missing_policy="zero",
            )
            if _array_form(inspection, key, values) in {
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
            {
                f"{_COLUMN_PREFIX}{column}": column
                for column in inspection.numeric_columns
            }
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
        values = materialize_upload(
            replace(inspection, expected_form="xt"),
            missing_policy="zero",
        )
        return _SelectedData("xt", tuple(values.shape))
    if token.startswith(_COLUMN_PREFIX):
        column = token.removeprefix(_COLUMN_PREFIX)
        values = materialize_upload(
            replace(inspection, expected_form="x"),
            column=column,
            missing_policy="zero",
        )
        return _SelectedData("x", tuple(values.shape), value_column=column)
    if token.startswith(_ARRAY_PREFIX):
        array_key = token.removeprefix(_ARRAY_PREFIX)
        values = materialize_upload(
            replace(inspection, expected_form="auto"),
            array_key=array_key,
            missing_policy="zero",
        )
        return _SelectedData(
            _array_form(inspection, array_key, values),
            tuple(values.shape),
            array_key=array_key,
        )
    form = str(inspection.inferred_form)
    values = materialize_upload(
        replace(inspection, expected_form=form),
        missing_policy="zero",
    )
    return _SelectedData(form, tuple(values.shape))


def _output_forms(input_form: str) -> Mapping[str, str]:
    if input_form == "table":
        keys = ("table",)
    elif input_form == "track":
        keys = ("space",)
    elif input_form == "scalar":
        keys = ("scalar", "space", "time", "table")
    elif input_form in {"x", "t"}:
        keys = ("space", "time", "table")
    elif input_form == "xt":
        keys = ("spacetime", "table")
    else:
        keys = ()
    return {key: _OUTPUT_FORM_LABELS[key] for key in keys}


def _detected_output_form(input_form: str) -> str:
    return {
        "scalar": "scalar",
        "x": "space",
        "t": "time",
        "xt": "spacetime",
        "table": "table",
        "track": "space",
    }.get(input_form, "space")


def _target_formats(input_form: str) -> Mapping[str, str]:
    if input_form == "table":
        return {key: _TARGET_FORMATS[key] for key in ("tsv", "csv")}
    return dict(_TARGET_FORMATS)


def _number(control: Any, label: str) -> float:
    try:
        value = float(control.value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a number.") from exc
    if not np.isfinite(value):
        raise ValueError(f"{label} must be finite.")
    return value


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
    selection = _NATURAL
    latest_result: DataConversionResult | None = None
    controls: dict[str, Any] = {}
    settings: dict[str, Any] = {
        "output_form": "space",
        "target_format": "npy",
        "output_name": "dnascape_data",
        "gzip_output": True,
        "value_name": "value",
        "missing_policy": "error",
        "decimal_places": 6,
        "chromosome": None,
        "start": 0,
        "end": 1,
        "bin_size": 1_000,
        "aggregation": "mean",
        "track_missing": "nan",
        "fill_value": 0.0,
        "interpolation": "none",
        "smooth_window": 1,
        "normalize": "none",
    }

    page_intro(
        "Data Manager",
        "Convert genomic data files",
        "Move profiles and tables between DNAscape data forms and file formats.",
    )
    with (
        ui.element("div")
        .classes("dn-workspace dn-data-manager-workspace")
        .mark("data-manager-workspace")
    ):
        with ui.column().classes("dn-control-stack"):
            with panel(extra_classes="dn-control-card"):
                panel_title("Controls")
                with expansion("Data", icon=DATA_ICON, expanded=True):

                    @ui.refreshable
                    def render_data_source() -> None:
                        current = inspection

                        async def uploaded(event: Any) -> bool:
                            nonlocal inspection, selection, track_inspection
                            try:
                                candidate = await inspect_upload_event(event, expected_form="auto")
                                candidate_track = None
                                if candidate.source_format in {"bigwig", "bedgraph"}:
                                    candidate_track = await run.io_bound(inspect_track, candidate)
                                    if candidate_track is None:
                                        raise RuntimeError(
                                            "Track inspection stopped before returning chromosome data."
                                        )
                                candidate_selection = _default_selection(candidate)
                                candidate_details = _selected_data(candidate, candidate_selection)
                            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                                ui.notify(str(exc), type="negative")
                                return False
                            inspection = candidate
                            track_inspection = candidate_track
                            selection = candidate_selection
                            settings["output_form"] = _detected_output_form(
                                candidate_details.form
                            )
                            cache_upload(
                                candidate,
                                source="Data Manager",
                                navigate_to="data-management",
                            )
                            settings["output_name"] = f"{_source_stem(candidate.name)}_converted"
                            if candidate_track is not None:
                                region = candidate_track.regions[0]
                                settings.update(
                                    chromosome=region.chromosome,
                                    start=region.start,
                                    end=region.end,
                                )
                            render_data_source.refresh()
                            render_conversion_settings.refresh()
                            return True

                        def cleared(_event: Any) -> None:
                            nonlocal inspection, selection, track_inspection
                            inspection = None
                            track_inspection = None
                            selection = _NATURAL
                            render_data_source.refresh()
                            render_conversion_settings.refresh()

                        data_upload(
                            accept=RDATA_UPLOAD_ACCEPT,
                            accepted_formats="CSV, TSV, TXT, NPY, NPV, NPZ, bedGraph, BigWig, GZ",
                            marker="data-manager-upload",
                            show_accepted_formats=False,
                            on_upload=uploaded,
                            on_clear=cleared,
                            max_files=1,
                            hide_when_loaded=True,
                            initial_names=((current.name,) if current is not None else ()),
                        )
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
                            details = _selected_data(current, selection)
                            settings["output_form"] = _detected_output_form(details.form)
                            render_data_source.refresh()
                            render_conversion_settings.refresh()

                        source_select.on_value_change(source_changed)
                        details = _selected_data(current, selection)
                        with ui.element("div").mark("data-manager-detected-data"):
                            ui.label(
                                f"Dependence: {_FORM_LABELS.get(details.form, 'Unknown')}"
                            ).classes(
                                "dn-mapping-detected-summary dn-mapping-detected-dependence"
                            )
                            ui.label(f"Array size: {_shape_text(details.shape)}").classes(
                                "dn-mapping-detected-summary"
                            )

                    render_data_source()

                with expansion("Conversion", icon="sync_alt", expanded=True):

                    @ui.refreshable
                    def render_conversion_settings() -> None:
                        controls.clear()
                        current = inspection
                        if current is None:
                            return
                        details = _selected_data(current, selection)
                        form_options = _output_forms(details.form)
                        if not form_options:
                            return
                        if settings["output_form"] not in form_options:
                            settings["output_form"] = next(iter(form_options))
                        target_options = _target_formats(details.form)
                        if settings["target_format"] not in target_options:
                            settings["target_format"] = next(iter(target_options))

                        with ui.element("div").classes("dn-data-manager-conversion-grid"):
                            with ui.column().classes("dn-data-manager-setting-group"):
                                ui.label("Destination").classes("dn-setting-group-title")
                                controls["output_form"] = (
                                    ui.select(
                                        dict(form_options),
                                        value=settings["output_form"],
                                        label="Output dependence",
                                    )
                                    .props("dense options-dense")
                                    .classes("dn-field")
                                    .mark("data-manager-output-form")
                                )
                                controls["target_format"] = (
                                    ui.select(
                                        dict(target_options),
                                        value=settings["target_format"],
                                        label="File type",
                                    )
                                    .props("dense options-dense")
                                    .classes("dn-field")
                                    .mark("data-manager-output-format")
                                )
                                controls["output_name"] = (
                                    ui.input(
                                        label="File name",
                                        value=settings["output_name"],
                                    )
                                    .props("dense")
                                    .classes("dn-field")
                                    .mark("data-manager-output-name")
                                )
                                controls["gzip_output"] = (
                                    ui.checkbox(
                                        "Compress (.gz)",
                                        value=bool(settings["gzip_output"]),
                                    )
                                    .classes("dn-data-manager-compression")
                                    .mark("data-manager-gzip")
                                )
                            with ui.column().classes("dn-data-manager-setting-group"):
                                ui.label("Values").classes("dn-setting-group-title")
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
                                            "error": "Error",
                                            "ignore": "Ignore",
                                            "interpolate": "Interpolate",
                                            "zero": "Fill with zero",
                                        },
                                        value=settings["missing_policy"],
                                        label="Missing values",
                                    )
                                    .props("dense options-dense")
                                    .classes("dn-field")
                                    .mark("data-manager-missing-values")
                                )
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

                        controls["missing_policy"].set_visibility(
                            details.form not in {"table", "track"}
                        )
                        controls["decimal_places"].set_visibility(
                            settings["target_format"] in _TEXT_TARGETS
                        )

                        def store_control(name: str) -> None:
                            settings[name] = controls[name].value

                        def target_changed() -> None:
                            store_control("target_format")
                            controls["decimal_places"].set_visibility(
                                str(controls["target_format"].value) in _TEXT_TARGETS
                            )

                        controls["output_form"].on_value_change(
                            lambda: store_control("output_form")
                        )
                        controls["target_format"].on_value_change(target_changed)
                        for name in (
                            "output_name",
                            "value_name",
                            "missing_policy",
                            "decimal_places",
                            "gzip_output",
                        ):
                            controls[name].on_value_change(
                                lambda name=name: store_control(name)
                            )

                        if details.form != "track" or track_inspection is None:
                            return

                        ui.label("Genomic region").classes(
                            "dn-field-heading dn-data-manager-setting-heading"
                        )
                        regions = {item.chromosome: item for item in track_inspection.regions}
                        chromosome = str(settings["chromosome"] or next(iter(regions)))
                        if chromosome not in regions:
                            chromosome = next(iter(regions))
                        region = regions[chromosome]
                        with two_column_fields():
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
                            controls["bin_size"] = (
                                ui.number(
                                    label="Bin size (bp)",
                                    value=settings["bin_size"],
                                    min=1,
                                    step=100,
                                )
                                .props("dense")
                                .classes("dn-field")
                                .mark("data-manager-bin-size")
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

                        ui.label("Track processing").classes(
                            "dn-field-heading dn-data-manager-setting-heading"
                        )
                        with two_column_fields():
                            controls["aggregation"] = (
                                ui.select(
                                    {
                                        "mean": "Mean",
                                        "sum": "Sum",
                                        "min": "Minimum",
                                        "max": "Maximum",
                                        "coverage": "Coverage",
                                        "std": "Standard deviation",
                                    },
                                    value=settings["aggregation"],
                                    label="Aggregation",
                                )
                                .props("dense options-dense")
                                .classes("dn-field")
                                .mark("data-manager-aggregation")
                            )
                            controls["track_missing"] = (
                                ui.select(
                                    {
                                        "nan": "Keep missing",
                                        "zero": "Fill with zero",
                                        "fill": "Custom fill",
                                        "drop": "Drop bins",
                                    },
                                    value=settings["track_missing"],
                                    label="Missing bins",
                                )
                                .props("dense options-dense")
                                .classes("dn-field")
                                .mark("data-manager-track-missing")
                            )
                            controls["fill_value"] = (
                                ui.number(
                                    label="Fill value",
                                    value=settings["fill_value"],
                                    step=0.1,
                                )
                                .props("dense")
                                .classes("dn-field")
                                .mark("data-manager-fill-value")
                            )
                            controls["interpolation"] = (
                                ui.select(
                                    {
                                        "none": "None",
                                        "linear": "Linear",
                                        "ffill": "Forward fill",
                                        "bfill": "Backward fill",
                                    },
                                    value=settings["interpolation"],
                                    label="Interpolation",
                                )
                                .props("dense options-dense")
                                .classes("dn-field")
                                .mark("data-manager-interpolation")
                            )
                            controls["smooth_window"] = (
                                ui.number(
                                    label="Smoothing window",
                                    value=settings["smooth_window"],
                                    min=1,
                                    step=1,
                                )
                                .props("dense")
                                .classes("dn-field")
                                .mark("data-manager-smooth-window")
                            )
                            controls["normalize"] = (
                                ui.select(
                                    {
                                        "none": "None",
                                        "center": "Center",
                                        "zscore": "Z-score",
                                        "minmax": "Min-max",
                                        "maxabs": "Maximum absolute",
                                    },
                                    value=settings["normalize"],
                                    label="Normalization",
                                )
                                .props("dense options-dense")
                                .classes("dn-field")
                                .mark("data-manager-normalization")
                            )
                        controls["fill_value"].set_visibility(
                            settings["track_missing"] == "fill"
                        )

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

                        def track_missing_changed() -> None:
                            store_control("track_missing")
                            controls["fill_value"].set_visibility(
                                str(controls["track_missing"].value) == "fill"
                            )

                        controls["chromosome"].on_value_change(chromosome_changed)
                        controls["track_missing"].on_value_change(track_missing_changed)
                        for name in (
                            "start",
                            "end",
                            "bin_size",
                            "aggregation",
                            "fill_value",
                            "interpolation",
                            "smooth_window",
                            "normalize",
                        ):
                            controls[name].on_value_change(
                                lambda name=name: store_control(name)
                            )

                    render_conversion_settings()

                with action_dock():
                    convert_button = (
                        ui.button("Convert", icon="sync_alt")
                        .props("no-caps")
                        .classes("dn-action-primary")
                        .mark("action-convert")
                    )

        with ui.column().classes("dn-preview-stack"):
            with panel(extra_classes="dn-preview-card dn-data-manager-output-card"):
                panel_title("Download")
                with ui.element("div").classes("w-full").mark(
                    "data-manager-result-container"
                ):

                    @ui.refreshable
                    def render_result() -> None:
                        result = latest_result
                        if result is None:
                            with ui.column().classes("dn-data-manager-empty").mark(
                                "data-manager-empty-result"
                            ):
                                ui.icon("file_download")
                                ui.label("No converted file")
                            return

                        def download_result() -> None:
                            assert latest_result is not None
                            artifact = latest_result.artifact
                            ui.download(artifact.data, artifact.filename, artifact.media_type)

                        with ui.element("div").classes("dn-data-manager-file"):
                            ui.icon("description").classes("dn-data-manager-file-icon")
                            with ui.column().classes("dn-data-manager-file-copy"):
                                download_link = (
                                    ui.button(
                                        result.artifact.filename,
                                        icon="download",
                                        on_click=download_result,
                                    )
                                    .props(
                                        'flat no-caps align=left aria-label="Download converted file"'
                                    )
                                    .classes("dn-data-manager-download")
                                    .mark("action-download-converted-data")
                                )
                                download_link.tooltip(
                                    f"Download {result.artifact.filename}"
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

                    render_result()

    def build_request() -> DataConversionRequest:
        current = inspection
        if current is None:
            raise ValueError("Upload a data file before converting.")
        details = _selected_data(current, selection)
        if not controls:
            raise ValueError("Choose conversion settings.")
        target_format = str(controls["target_format"].value)
        decimal_places = (
            _integer(controls["decimal_places"], "Decimal places", minimum=0)
            if target_format in _TEXT_TARGETS
            else None
        )
        kwargs: dict[str, Any] = {}
        if details.form == "track":
            start = _integer(controls["start"], "Start", minimum=0)
            end = _integer(controls["end"], "End", minimum=1)
            if end <= start:
                raise ValueError("End must be greater than start for the genomic region.")
            kwargs.update(
                chromosome=str(controls["chromosome"].value),
                start=start,
                end=end,
                bin_size=_integer(controls["bin_size"], "Bin size", minimum=1),
                aggregation=str(controls["aggregation"].value),
                track_missing=str(controls["track_missing"].value),
                fill_value=_number(controls["fill_value"], "Fill value"),
                interpolation=str(controls["interpolation"].value),
                smooth_window=_integer(
                    controls["smooth_window"],
                    "Smoothing window",
                    minimum=1,
                ),
                normalize=str(controls["normalize"].value),
            )
        return DataConversionRequest(
            inspection=current,
            input_form=details.form,
            output_form=str(controls["output_form"].value),
            target_format=target_format,
            output_name=str(controls["output_name"].value or ""),
            gzip_output=bool(controls["gzip_output"].value),
            value_column=details.value_column,
            array_key=details.array_key,
            missing_policy=str(controls["missing_policy"].value or "error"),
            decimal_places=decimal_places,
            value_name=str(controls["value_name"].value or ""),
            **kwargs,
        )

    async def convert() -> None:
        nonlocal latest_result
        convert_button.disable()
        try:
            request = build_request()
            result = await run.io_bound(convert_data, request)
            if result is None:
                raise RuntimeError("Conversion stopped before returning data.")
            latest_result = result
            cache_dataset(result)
            render_result.refresh()
            ui.notify(f"Converted to {result.artifact.filename}.", type="positive")
        except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
            ui.notify(str(exc), type="negative")
        finally:
            convert_button.enable()

    convert_button.on_click(convert)
