"""Shared raw-data controls for the statistics of a histogram preview."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from nicegui import run, ui

from app.export_settings import get_export_settings
from app.notifications import notify
from app.pages.common import raw_output_file_row
from app.raw_data_transfer import open_raw_artifact
from dnascape.workflows.exports import ExportArtifact, build_histogram_statistics_export


@dataclass
class StatisticsExportPreference:
    file_name: str
    file_type: str = "TSV"
    compress: bool = False


class HistogramStatisticsExport:
    """One supplemental file, with the same controls as an exported filter."""

    def __init__(
        self, settings: Any, *, get_chart: Callable[[], Any], filename: str,
        marker: str, navigate: Callable, preference: StatisticsExportPreference | None = None,
        section: Any = None, has_other_rows: Callable[[], bool] = lambda: False,
    ) -> None:
        self.settings = settings
        self.get_chart = get_chart
        self.preference = preference or StatisticsExportPreference(filename)
        self.navigate = navigate
        self.section = section
        self.has_other_rows = has_other_rows
        with ui.column().classes("w-full dn-statistics-export-row dn-mapping-batch-raw-item").mark(marker) as self.surface:
            ui.label("Statistics").classes("dn-setting-group-title")
            self.controls = raw_output_file_row(
                file_name=self.preference.file_name, default_type=self.preference.file_type,
                default_compress=self.preference.compress, reserve_remove_space=True,
                on_download=self.download,
                on_open=lambda _controls, destination: self.download(open_in=destination),
                extra_classes="dn-mapping-batch-raw-row",
                markers={
                    "row": f"{marker}-row", "file_name": f"{marker}-name",
                    "file_type": f"{marker}-type", "compression": f"{marker}-compression",
                    "download": f"{marker}-download", "open": f"{marker}-open",
                },
            )
        for control, field in (
            (self.controls.file_name, "file_name"), (self.controls.file_type, "file_type"),
            (self.controls.compress, "compress"),
        ):
            control.on_value_change(lambda event, key=field: setattr(self.preference, key, event.value))
        object.__setattr__(settings, "_statistics_export", self)
        self.refresh()

    def rows(self) -> list[dict]:
        chart = self.get_chart()
        if chart is None or getattr(chart.canvas, "is_deleted", False):
            return []
        if chart.options.get("dnascapePlotType") != "Histogram":
            return []
        return chart.options.get("dnascapeHistogramStatistics", [])

    def refresh(self) -> None:
        if not self.surface.is_deleted:
            visible = (
                self.settings.export_histogram_statistics.visible
                and bool(self.settings.export_histogram_statistics.value) and bool(self.rows())
            )
            self.surface.set_visibility(visible)
            if self.section is not None and not self.section.is_deleted:
                self.section.set_visibility(visible or self.has_other_rows())

    def snapshot(self) -> dict[str, Any]:
        return dict(
            statistics=[dict(row) for row in self.rows()],
            file_format=str(self.controls.file_type.value),
            filename=str(self.controls.file_name.value or "histogram_statistics"),
            compression=get_export_settings().raw_compression if self.controls.compress.value else "none",
        )

    def artifact(self, snapshot: dict[str, Any]) -> ExportArtifact:
        return build_histogram_statistics_export(**snapshot)

    async def download(self, _controls=None, *, open_in: str | None = None) -> None:
        try:
            artifact = await run.io_bound(self.artifact, self.snapshot())
            if open_in is not None:
                await open_raw_artifact(artifact, open_in, self.navigate)
            else:
                ui.download(artifact.data, artifact.filename, artifact.media_type)
        except (OSError, RuntimeError, ValueError) as exc:
            notify(str(exc), type="negative")
