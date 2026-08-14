"""Data-management interface prototype."""

from __future__ import annotations

from collections.abc import Callable

from nicegui import ui

from app.pages.common import (
    expansion,
    inert_button,
    mini_table,
    muted,
    page_intro,
    panel,
    panel_title,
    two_column_fields,
)


def render(_navigate: Callable[[str], None]) -> None:
    """Render all conversion controls without loading or transforming data."""

    page_intro(
        "Data Manager",
        "Convert genomic data files",
        "Upload a supported file and convert it between DNAscape-compatible formats.",
    )
    with ui.element("div").classes("dn-workspace"):
        with ui.column().classes("dn-control-stack"):
            with panel(extra_classes="dn-control-card"):
                panel_title("Input")
                ui.upload(label="Upload", auto_upload=False).props(
                    "flat bordered accept=.csv,.tsv,.txt,.npy,.npz,.bw,.bigwig,.bedgraph,.gz"
                ).classes("dn-upload-compact")
                ui.select(
                    ["No retained uploads"],
                    value="No retained uploads",
                    label="Retained source",
                ).classes("dn-field")
                muted("Supported formats: CSV, TSV, TXT, NumPy, BigWig, bedGraph, and gzip variants.")

                with expansion("Source inspection", expanded=True, icon="manage_search"):
                    with two_column_fields():
                        ui.select(
                            ["Auto-detect", "CSV", "TSV", "NumPy", "BigWig", "bedGraph"],
                            value="Auto-detect",
                            label="Source format",
                        ).classes("dn-field")
                        ui.select(
                            ["Auto-detect", "Table", "Space profile", "Time profile", "Space-time matrix"],
                            value="Auto-detect",
                            label="DNAscape data form",
                        ).classes("dn-field")
                        ui.select(
                            ["Auto-detect", "value", "replication_timing", "firing_rate"],
                            value="Auto-detect",
                            label="Value column",
                        ).classes("dn-field")
                        ui.select(
                            ["Auto-detect", "arr_0", "profile", "matrix"],
                            value="Auto-detect",
                            label="NPZ array",
                        ).classes("dn-field")
                    inert_button("Inspect source", icon="search", compact=True)

                with expansion("Genomic track region", icon="location_on"):
                    with two_column_fields():
                        ui.select(["chr1", "chr2", "chrX"], value="chr1", label="Chromosome").classes(
                            "dn-field"
                        )
                        ui.number(label="Start (bp, inclusive)", value=0, min=0, step=1000).classes(
                            "dn-field"
                        )
                        ui.number(
                            label="End (bp, exclusive)", value=1_000_000, min=1, step=1000
                        ).classes("dn-field")
                        ui.number(label="Custom fill value", value=0.0, step=0.1).classes("dn-field")

                with expansion("Advanced preprocessing", icon="tune"):
                    with two_column_fields():
                        ui.select(
                            ["mean", "sum", "min", "max", "coverage", "std"],
                            value="mean",
                            label="Aggregation",
                        ).classes("dn-field")
                        ui.select(
                            ["nan", "zero", "fill", "drop"],
                            value="nan",
                            label="Missing values",
                        ).classes("dn-field")
                        ui.select(
                            ["none", "linear", "ffill", "bfill"],
                            value="none",
                            label="Interpolation",
                        ).classes("dn-field")
                        ui.select(
                            ["none", "center", "zscore", "minmax", "maxabs"],
                            value="none",
                            label="Normalization",
                        ).classes("dn-field")
                        ui.number(label="Bin size (bp)", value=1000, min=1, step=100).classes("dn-field")
                        ui.number(
                            label="Centered smoothing window (bins)", value=1, min=1, step=1
                        ).classes("dn-field")
                ui.input(label="Output feature name", value="value").classes("dn-field")
            inert_button("Convert", icon="sync_alt", primary=True)

        with ui.column().classes("dn-preview-stack"):
            with panel(extra_classes="dn-preview-card"):
                panel_title("Conversion")
                mini_table(
                    (("position", "Position (kb)"), ("value", "Value")),
                    (
                        {"position": "0", "value": "0.124"},
                        {"position": "1", "value": "0.137"},
                        {"position": "2", "value": "0.151"},
                        {"position": "3", "value": "0.146"},
                    ),
                )
                with expansion("Preview", icon="preview"):
                    ui.switch("Render plot preview", value=False)
                ui.separator().classes("dn-section-separator")
                ui.label("Export").classes("dn-field-heading")
                with ui.element("div").classes("dn-field-grid dn-field-grid-three"):
                    ui.input(label="File name", placeholder="Name (auto)").classes("dn-field")
                    ui.select(
                        ["CSV", "TSV", "TXT", "NPY", "NPV", "NPZ"],
                        value="CSV",
                        label="File type",
                    ).classes("dn-field")
                    ui.checkbox("GZ", value=False)
                inert_button("Download", icon="download", primary=True)
