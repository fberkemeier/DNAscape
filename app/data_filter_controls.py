"""The Data Manager's optional, post-conversion region editor."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from nicegui import ui

from app.mapping_uploads import inspect_upload_event
from app.notifications import notify
from app.pages.common import data_upload, set_data_upload_locked
from dnascape.workflows.filtering import (
    FILTER_ACTIONS, FILTER_INTERPOLATIONS, FILTER_PROFILE_LAYOUTS, FILTER_SELECTIONS,
    DataFilterSettings,
)


class DataFilterControls:
    def __init__(self, on_loading_change: Callable[[], None]) -> None:
        self.controls: dict[str, Any] = {}
        self.upload = None
        self.loading = False
        self._locked_states: dict[str, bool] = {}
        self._on_loading_change = on_loading_change
        with ui.expansion("Filter", icon="filter_alt", value=False).props("dense").classes("dn-expansion").mark("data-manager-filter"):
            with ui.column().classes("dn-expansion-content dn-data-filter"):
                self._mark("enabled", ui.switch("Apply filter", value=False))
                ui.label("Edit selected regions after Configure. The original upload is preserved.").classes("dn-data-manager-section-guidance")
                with ui.column().classes("dn-data-filter-options").bind_visibility_from(self.controls["enabled"], "value"):
                    ui.label("Select regions").classes("dn-setting-group-title")
                    self._select("selection", "Selection", FILTER_SELECTIONS, "threshold")
                    with ui.element("div").classes("dn-data-filter-grid") as threshold:
                        self._select("comparison", "Values", {"below": "Below", "above": "Above"}, "below")
                        self._number("threshold", "Threshold", 0.)
                    with ui.element("div").classes("dn-data-filter-grid") as between:
                        self._number("lower", "Minimum", 0.)
                        self._number("upper", "Maximum", 1.)
                    self.selection_note = ui.label().classes("dn-data-manager-section-guidance")
                    with ui.column().classes("dn-data-filter-profile") as profile:
                        with ui.element("div").classes("dn-data-filter-grid") as profile_grid:
                            self._select("profile_layout", "Profile contents", FILTER_PROFILE_LAYOUTS, "mask")
                            self._select("profile_units", "Coordinates", {"bp": "Base pairs (bp)", "kb": "Kilobases (kb)", "index": "Bin index (from 0)"}, "bp")
                        self.controls["uploader"] = data_upload(
                            accept=".tsv,.csv,.txt,.npz,.npy,.bed,.bedgraph,.gz",
                            accepted_formats="TSV, CSV, TXT, NPZ, NPY, BED, GZ",
                            marker="data-manager-filter-upload", on_upload=self._uploaded,
                            on_clear=self._clear_upload, hide_when_loaded=True,
                        )
                        self._select("profile_column", "Mask values", {}, None)
                        self._select("profile_array_key", "Array", {}, None)
                        ui.label("No profile means no filter. Mask values other than zero select bins; positions and intervals select overlapping bins.").classes("dn-data-manager-section-guidance")
                    ui.label("Apply to selection").classes("dn-setting-group-title dn-data-filter-operation-heading")
                    self._select("action", "Operation", FILTER_ACTIONS, "replace")
                    self._number("replace_value", "Replacement value", 0.)
                    self._number("factor", "Factor", 1.)
                    self._number("offset", "Value to add", 0.)
                    self._select("interpolation", "Method", FILTER_INTERPOLATIONS, "linear")
                    with ui.element("div").classes("dn-data-filter-grid") as noise:
                        self._number("standard_deviation", "Standard deviation", 1., min=0)
                        self._number("seed", "Random seed (optional)", None, min=0, step=1)
                    self._select("reference", "Reference values", {"outside": "Outside selected regions", "all": "Whole converted profile"}, "outside")
                    self.action_note = ui.label().classes("dn-data-manager-section-guidance")
        self._threshold_group, self._between_group = threshold, between
        self._profile_group, self._noise_group = profile, noise
        self._profile_grid = profile_grid
        for key in ("selection", "action", "profile_layout"):
            self.controls[key].on_value_change(self._sync)
        self._sync()

    def _mark(self, name: str, control):
        self.controls[name] = control.mark(f"data-manager-filter-{name.replace('_', '-')}")
        return control

    def _select(self, name, label, options, value):
        return self._mark(name, ui.select(options, value=value, label=label).props("dense options-dense").classes("dn-field"))

    def _number(self, name, label, value, **kwargs):
        return self._mark(name, ui.number(label=label, value=value, **kwargs).props("dense").classes("dn-field"))

    def _sync(self) -> None:
        selection, action = self.controls["selection"].value, self.controls["action"].value
        self._threshold_group.set_visibility(selection in {"threshold", "minimum", "maximum"})
        self._between_group.set_visibility(selection == "between")
        self._profile_group.set_visibility(selection == "profile")
        self.controls["threshold"].props(f'label="{"Distance" if selection in {"minimum", "maximum"} else "Threshold"}"')
        self.selection_note.set_text({
            "threshold": "Select values strictly above or below the threshold.",
            "between": "Select values between the minimum and maximum, including both limits.",
            "minimum": "Threshold = converted minimum + distance.",
            "maximum": "Threshold = converted maximum − distance.",
            "nan": "Select values still missing after Configure. Use Keep empty values to retain NaN regions.",
            "all": "Select every bin in the converted region.",
        }.get(selection, ""))
        self.selection_note.set_visibility(selection != "profile")
        self.controls["profile_units"].set_visibility(self.controls["profile_layout"].value != "mask")
        self._profile_grid.classes(
            add="dn-data-filter-single-column" if self.controls["profile_layout"].value == "mask" else None,
            remove=None if self.controls["profile_layout"].value == "mask" else "dn-data-filter-single-column",
        )
        self.controls["profile_column"].set_visibility(bool(self.upload and len(self.upload.numeric_columns) > 1 and self.controls["profile_layout"].value == "mask"))
        self.controls["profile_array_key"].set_visibility(bool(self.upload and len(self.upload.array_keys) > 1))
        for name, operation in (("replace_value", "replace"), ("factor", "multiply"), ("offset", "add"), ("interpolation", "interpolate")):
            self.controls[name].set_visibility(action == operation)
        self._noise_group.set_visibility(action == "noise")
        self.controls["reference"].set_visibility(action in {"minimum", "maximum"})
        self.action_note.set_text({
            "mean": "Each contiguous region is replaced by its own mean, ignoring missing values.",
            "interpolate": "Bridge each region using the surrounding values. Edge regions use the nearest available endpoint.",
            "noise": "Add zero-mean noise to selected values. Set a seed for reproducible results.",
            "minimum": "Use the minimum before filtering, from the reference region selected above.",
            "maximum": "Use the maximum before filtering, from the reference region selected above.",
            "nan": "Selected bins remain in place with missing values (NaN).",
        }.get(action, ""))
        self.action_note.set_visibility(action in {"mean", "interpolate", "noise", "minimum", "maximum", "nan"})

    async def _uploaded(self, event):
        self.loading = True
        self._on_loading_change()
        try:
            candidate = await inspect_upload_event(event, expected_form="regions")
            if candidate.source_format not in {"csv", "tsv", "txt", "npz", "npy", "bedgraph"}:
                raise ValueError("Choose a mask profile, numeric position list, or interval table.")
            self.upload = candidate
            self.controls["profile_column"].set_options(list(candidate.numeric_columns), value=candidate.default_column)
            self.controls["profile_array_key"].set_options(list(candidate.array_keys), value=candidate.default_array_key)
            if candidate.source_format == "bedgraph":
                self.controls["profile_layout"].set_value("intervals")
                self.controls["profile_units"].set_value("bp")
            else:
                names = {name.lower().replace(" ", "_") for name in candidate.columns}
                if any(bounds <= names for bounds in ({"start", "end"}, {"start_bp", "end_bp"}, {"start_kb", "end_kb"})):
                    self.controls["profile_layout"].set_value("intervals")
                    self.controls["profile_units"].set_value("kb" if "start_kb" in names else "bp")
                elif names & {"position", "position_bp", "position_kb", "pos"} and len(candidate.numeric_columns) == 1:
                    self.controls["profile_layout"].set_value("positions")
                    self.controls["profile_units"].set_value("kb" if "position_kb" in names else "bp")
                elif names & {"mask", "filter", "selected"}:
                    self.controls["profile_layout"].set_value("mask")
            self._sync()
            return True
        except (OSError, RuntimeError, ValueError, TypeError) as exc:
            notify(str(exc), type="negative")
            return False
        finally:
            self.loading = False
            self._on_loading_change()

    def _clear_upload(self):
        self.upload = None
        self._sync()

    def set_locked(self, locked: bool):
        for name, control in self.controls.items():
            if locked:
                self._locked_states[name] = control.enabled
            if name == "uploader":
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
            value=value({"multiply": "factor", "add": "offset"}.get(action, "replace_value")),
            standard_deviation=value("standard_deviation"), seed=value("seed"),
            interpolation=value("interpolation"), reference=value("reference"), upload=self.upload,
            profile_layout=value("profile_layout"), profile_units=value("profile_units"),
            profile_column=value("profile_column"), profile_array_key=value("profile_array_key"),
        )
