"""Grouped application preferences, edited as a draft until Apply."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from nicegui import ui

from app.export_settings import (
    CACHE_FILE_FORMAT_OPTIONS,
    COMPRESSION_OPTIONS,
    DEFAULT_CONVERSION_MEMORY_LIMIT_MB,
    MAX_CONVERSION_MEMORY_LIMIT_MB,
    ExportSettings,
    get_export_settings,
)
from app.result_naming import NAME_STYLE_OPTIONS


@dataclass
class SettingsDialog:
    dialog: Any
    controls: dict[str, Any]
    apply_button: Any = None
    _applied_values: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def has_changes(self) -> bool:
        return any(
            control.value != self._applied_values.get(key)
            for key, control in self.controls.items()
        )

    def refresh_apply(self) -> None:
        if self.apply_button is not None:
            self.apply_button.enable() if self.has_changes else self.apply_button.disable()

    def load(self, settings: ExportSettings) -> None:
        self._applied_values = {
            key: getattr(settings, key) for key in self.controls
        }
        for key, control in self.controls.items():
            control.set_value(getattr(settings, key))
        self.refresh_apply()

    def restore_defaults(self) -> None:
        defaults = ExportSettings()
        for key, control in self.controls.items():
            control.set_value(getattr(defaults, key))
        self.refresh_apply()

    def open(self) -> None:
        self.load(get_export_settings())
        self.dialog.open()


def settings_dialog(*, on_apply: Callable[[], None]) -> SettingsDialog:
    settings = get_export_settings()
    controls: dict[str, Any] = {}

    def request_close() -> None:
        if editor.has_changes:
            unsaved_dialog.open()
        else:
            dialog.close()

    def switch(key: str, label: str, marker: str) -> None:
        controls[key] = (
            ui.switch(label, value=getattr(settings, key))
            .props("dense left-label")
            .classes("dn-settings-switch")
            .mark(marker)
        )

    def select(key: str, label: str, options: dict[str, str], marker: str) -> None:
        controls[key] = (
            ui.select(options, value=getattr(settings, key), label=label)
            .props("dense options-dense")
            .classes("dn-field")
            .mark(marker)
        )

    with (
        ui.dialog().props("persistent").mark("settings-dialog") as dialog,
        ui.card().classes("dn-settings-dialog"),
    ):
        with ui.row().classes("dn-settings-dialog-head"):
            ui.label("Settings").classes("dn-panel-title")
            ui.space()
            ui.button(icon="close", on_click=request_close).props(
                'flat round dense aria-label="Close settings"'
            ).mark("settings-close")
        with ui.column().classes("dn-settings-section"):
            ui.label("Interface").classes("dn-setting-group-title")
            switch("preserve_page_state", "Keep page controls and outputs", "settings-preserve-page-state")
            switch("show_examples", "Show example links", "settings-show-examples")
            switch("show_help", "Show field help", "settings-show-help")
            switch("dev_mode", "Developed mode", "settings-dev-mode")
        with ui.column().classes("dn-settings-section"):
            ui.label("Files and exports").classes("dn-setting-group-title")
            with ui.element("div").classes("dn-settings-grid"):
                controls["plot_dpi"] = (
                    ui.number(label="Plot export DPI", value=settings.plot_dpi, min=72, max=384, step=24)
                    .props("dense").classes("dn-field").mark("settings-plot-dpi")
                )
                select("raw_compression", "Raw data compression", COMPRESSION_OPTIONS, "settings-raw-compression")
                select("cached_file_format", "Cached file type", CACHE_FILE_FORMAT_OPTIONS, "settings-cached-file-format")
                select("cached_compression", "Cached compression", COMPRESSION_OPTIONS, "settings-cached-compression")
                select("result_name_style", "Result and file names", NAME_STYLE_OPTIONS, "settings-result-name-style")
                controls["result_name_style"].classes("dn-settings-wide-control")
        with ui.column().classes("dn-settings-section"):
            ui.label("Processing").classes("dn-setting-group-title")
            controls["conversion_memory_limit_mb"] = (
                ui.number(
                    label="Conversion memory limit (MB)",
                    value=settings.conversion_memory_limit_mb,
                    min=DEFAULT_CONVERSION_MEMORY_LIMIT_MB,
                    max=MAX_CONVERSION_MEMORY_LIMIT_MB,
                    step=64,
                    validation={
                        f"Enter a whole number from {DEFAULT_CONVERSION_MEMORY_LIMIT_MB} "
                        f"to {MAX_CONVERSION_MEMORY_LIMIT_MB:,}.": lambda value: (
                            value is not None
                            and DEFAULT_CONVERSION_MEMORY_LIMIT_MB <= value <= MAX_CONVERSION_MEMORY_LIMIT_MB
                            and float(value).is_integer()
                        ),
                    },
                ).props("dense").classes("dn-field").mark("settings-conversion-memory-limit")
            )
            controls["conversion_memory_limit_mb"].tooltip(
                "Estimated working-memory budget for genomic tracks; does not reserve RAM. "
                "All K562 chromosomes at 1 kb need 512 MB."
            )
        editor = SettingsDialog(dialog, controls)

        def apply() -> None:
            if not editor.has_changes:
                return
            dpi = controls["plot_dpi"].value
            if dpi is None or not 72 <= float(dpi) <= 384 or not float(dpi).is_integer():
                ui.notify("Enter a whole DPI value from 72 to 384.", type="negative")
                return
            if not controls["conversion_memory_limit_mb"].validate():
                ui.notify("Enter a valid conversion memory limit.", type="negative")
                return
            for key, control in controls.items():
                value = int(control.value) if key in {"plot_dpi", "conversion_memory_limit_mb"} else control.value
                setattr(settings, key, value)
            editor.load(settings)
            on_apply()
            dialog.close()

        with ui.row().classes("dn-dialog-actions dn-settings-actions"):
            ui.button("Restore defaults", icon="restart_alt", on_click=editor.restore_defaults).props(
                "flat no-caps"
            ).classes("dn-action-compact").mark("settings-restore-defaults")
            editor.apply_button = (
                ui.button("Apply", on_click=apply).props("no-caps")
                .classes("dn-action-primary dn-action-compact dn-settings-apply")
                .mark("settings-apply")
            )

    with (
        ui.dialog().mark("settings-unsaved-dialog") as unsaved_dialog,
        ui.card().classes("dn-settings-dialog dn-reset-dialog"),
    ):
        ui.label("Unapplied changes").classes("dn-panel-title")
        ui.label("Changes have been made but not applied.")

        def discard_changes() -> None:
            editor.load(get_export_settings())
            unsaved_dialog.close()
            dialog.close()

        with ui.row().classes("dn-dialog-actions"):
            ui.button("Keep editing", on_click=unsaved_dialog.close).props(
                "flat no-caps"
            ).mark("settings-keep-editing")
            ui.button("Discard changes", on_click=discard_changes).props(
                "no-caps"
            ).classes("dn-action-primary dn-action-compact").mark("settings-discard-changes")

    # A persistent Quasar dialog emits shake for backdrop clicks and Escape,
    # allowing the draft to be checked before the dialog is dismissed.
    dialog.on("shake", request_close)
    for control in controls.values():
        control.on_value_change(editor.refresh_apply)
    editor.load(settings)
    return editor
