"""Shared controls for optional region selection and profile transformations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any

from nicegui import ui

from app.mapping_uploads import inspect_upload_event
from app.notifications import notify
from app.pages.common import data_upload, set_data_upload_locked
from dnascape.workflows.filtering import (
    FILTER_ACTIONS, FILTER_INTERPOLATIONS, FILTER_SELECTIONS,
    DataFilterSettings, detect_filter_profile,
)
from dnascape.workflows.modulation import ProfileModulationSettings


class DataFilterControls:
    def __init__(
        self, on_loading_change: Callable[[], None], *, title="Filter",
        marker_prefix="data-manager-filter", toggle_label="Apply filter",
        default_selection="threshold", default_action="replace", header: Callable[[], None] | None = None,
        allow_export: bool = False, allow_complement: bool = False,
        profile_units: str = "kb", align_modulation_from_start: bool = False,
        relative_options: dict[str, str] | None = None,
        on_change: Callable[[], None] | None = None,
        embedded: bool = False,
    ) -> None:
        self.controls: dict[str, Any] = {}
        self.marker_prefix = marker_prefix
        self.align_modulation_from_start = align_modulation_from_start
        self.upload = None
        self.profile = None
        self.modulation_upload = None
        self.loading = False
        self._loading_count = 0
        self._locked_states: dict[str, bool] = {}
        self._on_loading_change = on_loading_change
        section = (ui.column().classes('dn-data-filter-embedded') if embedded else
                   ui.expansion(title, icon="filter_alt", value=False).props("dense").classes("dn-expansion"))
        with section.mark(marker_prefix):
            if embedded:
                ui.label(title).classes('dn-setting-group-title')
            with ui.column().classes("dn-data-filter" + ("" if embedded else " dn-expansion-content")):
                if relative_options is not None:
                    self._select('relative_to', 'Relative to', relative_options, next(iter(relative_options), None))
                with ui.row().classes("dn-inline-actions"):
                    self._mark("enabled", ui.switch(toggle_label, value=False))
                    if allow_export:
                        self._mark("export", ui.switch("Export filter", value=False))
                        self.controls["export"].bind_enabled_from(self.controls["enabled"], "value")
                    if allow_complement:
                        self._mark("complement", ui.switch("Complement region", value=False))
                with ui.column().classes("dn-data-filter-options"):
                    if header is not None:
                        header()
                    ui.label("Select regions").classes("dn-setting-group-title")
                    self._select("selection", "Selection", FILTER_SELECTIONS, default_selection)
                    with ui.element("div").classes("dn-data-filter-grid") as threshold:
                        self._select("comparison", "Values", {"below": "Below", "above": "Above"}, "below")
                        self._number("threshold", "Threshold", 0.)
                    with ui.element("div").classes("dn-data-filter-grid dn-data-filter-between") as between:
                        self._select("between_mode", "Values", {"within": "Within", "outside": "Outside"}, "within")
                        self._number("lower", "Minimum", 0.)
                        self._number("upper", "Maximum", 1.)
                    with ui.column().classes("dn-data-filter-profile") as profile:
                        with ui.element("div").classes("dn-data-filter-profile-input"):
                            self.controls["uploader"] = data_upload(
                                accept=".tsv,.csv,.txt,.npz,.gz",
                                accepted_formats="TSV, TXT, CSV, NPZ",
                                marker=f"{marker_prefix}-upload", on_upload=self._uploaded,
                                on_clear=self._clear_upload, hide_when_loaded=True,
                            )
                            self._select("profile_units", "Coordinates", {"kb": "kb", "bp": "bp"}, profile_units)
                    ui.label("Transform").classes("dn-setting-group-title dn-data-filter-operation-heading")
                    self._select("action", "Operation", FILTER_ACTIONS, default_action)
                    self._number("replace_value", "Replacement value", 0.)
                    self._number("factor", "Factor", 1.)
                    self._number("offset", "Value to add", 0.)
                    self._select("interpolation", "Method", FILTER_INTERPOLATIONS, "linear")
                    with ui.element("div").classes("dn-data-filter-grid") as noise:
                        self._number("standard_deviation", "Standard deviation", 1., min=0)
                        self._number("seed", "Random seed (optional)", None, min=0, step=1)
                    self._select("reference", "Reference values", {"outside": "Outside selected regions", "all": "Whole profile"}, "outside")
                    with ui.column().classes("dn-data-filter-modulation") as modulation:
                        ui.label("Modulation profile").classes("dn-setting-group-title")
                        ui.label("Combine aligned signals, gene annotations or assays such as RNA-seq, ATAC-seq and ChIP-seq.").classes("dn-upload-formats")
                        self.controls["modulation_uploader"] = data_upload(
                            accept=".tsv,.txt,.csv,.npz", accepted_formats="TSV, TXT, CSV, NPZ",
                            marker=f"{marker_prefix}-modulation-upload", on_upload=self._modulation_uploaded,
                            on_clear=self._clear_modulation, hide_when_loaded=True,
                        )
                        ui.label(
                            "Arrays align from the first sample; only their overlap is combined."
                            if align_modulation_from_start else "Array sizes must match the source data."
                        ).classes("dn-upload-formats")
                        self.modulation_summary = ui.label().classes("dn-data-manager-section-guidance")
                        self._select("modulation_column", "Data", {"__auto__": "Automatic"}, "__auto__")
                        self._select("modulation_array", "Array", {"__auto__": "Automatic"}, "__auto__")
                        self._mark("modulation_expression", ui.textarea(
                            "Formula", value="source * profile",
                        ).props("dense autogrow rows=2 spellcheck=false").classes("dn-field dn-formula-input"))
                        ui.label(
                            "Python / NumPy: source is the current signal; profile is the uploaded data. "
                            "Only selected regions change."
                        ).classes("dn-upload-formats")
        self._threshold_group, self._between_group = threshold, between
        self._profile_group, self._noise_group = profile, noise
        self._modulation_group = modulation
        for key in ("selection", "action", "modulation_column", "modulation_array"):
            self.controls[key].on_value_change(self._sync)
        self._sync()
        if on_change is not None:
            for control in self.controls.values():
                if hasattr(control, 'on_value_change'):
                    control.on_value_change(on_change)

    def _mark(self, name: str, control):
        self.controls[name] = control.mark(f"{self.marker_prefix}-{name.replace('_', '-')}")
        return control

    def _select(self, name, label, options, value):
        return self._mark(name, ui.select(options, value=value, label=label).props("dense options-dense").classes("dn-field"))

    def _number(self, name, label, value, **kwargs):
        return self._mark(name, ui.number(label=label, value=value, **kwargs).props("dense").classes("dn-field"))

    def _sync(self) -> None:
        selection, action = self.controls["selection"].value, self.controls["action"].value
        if 'relative_to' in self.controls:
            self.controls['relative_to'].set_visibility(selection not in {'profile', 'all'})
        self._threshold_group.set_visibility(selection in {"threshold", "minimum", "maximum"})
        self._between_group.set_visibility(selection == "between")
        self._profile_group.set_visibility(selection == "profile")
        self.controls["threshold"].props(f'label="{"Distance" if selection in {"minimum", "maximum"} else "Threshold"}"')
        for name, operation in (("replace_value", "replace"), ("factor", "multiply"), ("offset", "add"), ("interpolation", "interpolate")):
            self.controls[name].set_visibility(action == operation)
        self._noise_group.set_visibility(action == "noise")
        self.controls["reference"].set_visibility(action in {"minimum", "maximum"})
        self._sync_modulation()

    def _sync_modulation(self) -> None:
        self._modulation_group.set_visibility(self.controls["action"].value == "modulate")
        upload = self.modulation_upload
        self.controls["modulation_column"].set_visibility(bool(upload and len(upload.numeric_columns) > 1))
        self.controls["modulation_array"].set_visibility(bool(upload and len(upload.array_keys) > 1))
        shape = upload.shape if upload is not None else None
        if upload is not None and upload.array_keys:
            key = self.controls["modulation_array"].value
            array = upload._array_data.get(upload.default_array_key if key == "__auto__" else key)
            shape = array.shape if array is not None else shape
        if upload is not None and shape:
            column = self.controls["modulation_column"].value
            if column not in {"__auto__", "__matrix__"} or len(upload.numeric_columns) == 1:
                shape = (shape[0],)
            elif upload.source_format in {"csv", "tsv", "txt"}:
                shape = (shape[0], len(upload.numeric_columns))
        self.modulation_summary.set_text("Profile shape: " + " × ".join(f"{n:,}" for n in shape) if shape else "")
        self.modulation_summary.set_visibility(bool(shape))

    def _set_loading(self, change: int):
        self._loading_count += change
        self.loading = self._loading_count > 0
        self._on_loading_change()

    async def _uploaded(self, event):
        self._set_loading(1)
        try:
            candidate = await inspect_upload_event(event, expected_form="regions")
            detected = detect_filter_profile(candidate, units=self.controls["profile_units"].value)
            self.upload = candidate
            self.profile = detected
            names = {name.lower().replace(" ", "_") for name in candidate.columns}
            if names & {"position_bp", "start_bp", "end_bp"}:
                self.controls["profile_units"].set_value("bp")
            elif names & {"position_kb", "start_kb", "end_kb"}:
                self.controls["profile_units"].set_value("kb")
            self._sync()
            return True
        except (OSError, RuntimeError, ValueError, TypeError) as exc:
            notify(str(exc), type="negative")
            return False
        finally:
            self._set_loading(-1)

    async def _modulation_uploaded(self, event):
        self._set_loading(1)
        try:
            candidate = await inspect_upload_event(event)
            if candidate.source_format not in {"csv", "tsv", "txt", "npz"}:
                raise ValueError("Prepare the modulation data as a numeric profile before uploading it here.")
            self.modulation_upload = candidate
            columns = {"__auto__": "Automatic"}
            if len(candidate.numeric_columns) > 1:
                columns["__matrix__"] = "All value columns"
            columns.update({name: name for name in candidate.numeric_columns})
            self.controls["modulation_column"].set_options(columns, value="__auto__")
            self.controls["modulation_array"].set_options({"__auto__": "Automatic", **{key: key for key in candidate.array_keys}}, value="__auto__")
            self._sync_modulation()
            return True
        except (OSError, RuntimeError, ValueError, TypeError) as exc:
            notify(str(exc), type="negative")
            return False
        finally:
            self._set_loading(-1)

    def _clear_modulation(self):
        self.modulation_upload = None
        self._sync_modulation()

    def _clear_upload(self):
        self.upload = None
        self.profile = None
        self._sync()

    def set_locked(self, locked: bool):
        if locked:
            self._locked_states = {name: control.enabled for name, control in self.controls.items()}
        for name, control in self.controls.items():
            if name in {"uploader", "modulation_uploader"}:
                set_data_upload_locked(control, locked)
            elif locked:
                control.disable()
            elif self._locked_states.pop(name, True):
                control.enable()

    def snapshot(self) -> DataFilterSettings:
        value = lambda name: self.controls[name].value
        action = value("action")
        return DataFilterSettings(
            enabled=bool(value("enabled")), selection=value("selection"), comparison=value("comparison"),
            threshold=value("threshold"), lower=value("lower"), upper=value("upper"), action=action,
            between_mode=value("between_mode"),
            complement=bool(value("complement")) if "complement" in self.controls else False,
            value=value({"multiply": "factor", "add": "offset"}.get(action, "replace_value")),
            standard_deviation=value("standard_deviation"), seed=value("seed"),
            interpolation=value("interpolation"), reference=value("reference"),
            profile=(
                replace(self.profile, units=value("profile_units")) if self.profile is not None else None
            ),
            profile_layout=self.profile.layout if self.profile is not None else "positions",
            profile_units=value("profile_units"),
            modulation=ProfileModulationSettings(
                operation="expression", expression=str(value("modulation_expression") or ""),
                align_from_start=self.align_modulation_from_start,
                upload=self.modulation_upload,
                column=None if value("modulation_column") == "__auto__" else value("modulation_column"),
                array_key=None if value("modulation_array") == "__auto__" else value("modulation_array"),
            ),
        )
