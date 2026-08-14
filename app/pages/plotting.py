"""Plotting workspace interface prototype."""

from __future__ import annotations

from collections.abc import Callable

from nicegui import ui

from app.pages.common import (
    DATA_ICON,
    expansion,
    inert_button,
    muted,
    page_intro,
    panel,
    panel_title,
    preview_plot,
    raw_output_controls,
    stat_strip,
    two_column_fields,
)


def render(_navigate: Callable[[str], None]) -> None:
    """Render a complete but computation-free plotting workspace."""

    page_intro(
        "Plotting",
        "Visualize DNAscape outputs",
        "Upload kinetic profiles, choose a genomic window, and preview DNAscape outputs.",
    )
    with ui.element("div").classes("dn-workspace"):
        with ui.column().classes("dn-control-stack"):
            with panel(extra_classes="dn-control-card"):
                panel_title("Controls")
                with expansion(
                    "Data",
                    icon=DATA_ICON,
                    expanded=True,
                    extra_classes="dn-data-section",
                ):
                    with ui.element("div").classes("dn-source-block dn-plot-data-source"):
                        ui.radio(
                            ["Prepared data or result", "Upload a profile"],
                            value="Prepared data or result",
                        ).props("inline dense")
                        ui.select(
                            [
                                "Replication profile",
                                "Mapping result",
                                "Simulation result",
                            ],
                            value="Replication profile",
                            label="Data source",
                        ).classes("dn-field")
                        ui.upload(label="Upload a profile", auto_upload=False).props(
                            "flat bordered accept=.csv,.tsv,.txt,.npy,.npz,.gz"
                        ).classes("dn-upload-compact")
                        ui.select(
                            ["value", "replication_timing", "firing_rate"],
                            value="value",
                            label="Profile column",
                        ).classes("dn-field")
                        muted("Replication profile · 1,000 values · space")

                with expansion("Region", icon="straighten"):
                    with two_column_fields():
                        ui.number(label="Start index (inclusive)", value=0, min=0, step=1).classes(
                            "dn-field"
                        )
                        ui.number(label="End index (exclusive)", value=1000, min=1, step=1).classes(
                            "dn-field"
                        )
                        ui.number(label="First time / feature index", value=0, min=0, step=1).classes(
                            "dn-field"
                        )
                        ui.number(
                            label="Last time / feature index", value=60, min=1, step=1
                        ).classes("dn-field")
                        ui.number(label="Resolution (kb/bin)", value=1.0, min=0.000001, step=0.1).classes(
                            "dn-field"
                        )
                        ui.number(label="Time step (min)", value=1.0, min=0.000001, step=0.1).classes(
                            "dn-field"
                        )

                with expansion("Plotting settings", icon="palette"):
                    with two_column_fields():
                        ui.select(
                            ["value", "replication_timing", "firing_rate"],
                            value="value",
                            label="Output",
                        ).classes("dn-field")
                        ui.select(
                            ["Line", "Heatmap", "Histogram"], value="Line", label="Plot type"
                        ).classes("dn-field")
                        ui.select(
                            ["None", "Simulation result", "Mapping result"],
                            value="None",
                            label="Optional comparison line",
                        ).classes("dn-field")
                        ui.input(label="Comparison color", value="#d97757").props("type=color").classes(
                            "dn-field"
                        )
                    ui.input(label="Title", value="Replication profile").classes("dn-field")
                    with two_column_fields():
                        ui.input(label="X-axis label", value="Position (kb)").classes("dn-field")
                        ui.input(label="Y-axis label", value="Value").classes("dn-field")
                        ui.select(
                            ["None", "Z-score", "Min-max [0,1]"],
                            value="None",
                            label="Presentation normalization",
                        ).classes("dn-field")
                        ui.input(label="Line / bar color", value="#2c7f77").props("type=color").classes(
                            "dn-field"
                        )
                        ui.select(
                            ["viridis", "magma", "plasma", "cividis", "coolwarm"],
                            value="viridis",
                            label="Heatmap palette",
                        ).classes("dn-field")
                        ui.number(label="Line width", value=1.8, min=0.2, max=8.0, step=0.1).classes(
                            "dn-field"
                        )
                        ui.number(label="Histogram bins", value=60, min=1, max=2000, step=1).classes(
                            "dn-field"
                        )
                        ui.input(label="Colorbar label", value="Value").classes("dn-field")
                    with ui.element("div").classes("dn-inline-actions"):
                        ui.checkbox("Grid", value=True)
                        ui.checkbox("Colorbar", value=True)
                        ui.checkbox("Log Y", value=False)
                        ui.checkbox("Invert Y", value=False)
                    with two_column_fields():
                        ui.input(label="X limits (min,max; optional)").classes("dn-field")
                        ui.input(label="Y limits (min,max; optional)").classes("dn-field")
                        ui.input(label="Heatmap minimum (optional)").classes("dn-field")
                        ui.input(label="Heatmap maximum (optional)").classes("dn-field")
            inert_button("Plot", icon="show_chart", primary=True)

        with ui.column().classes("dn-preview-stack"):
            with panel(extra_classes="dn-preview-card"):
                panel_title("Preview")
                stat_strip(
                    (
                        ("Source", "Profile"),
                        ("Plot", "Line"),
                        ("Points", "1,000"),
                        ("Status", "Ready"),
                    )
                )
                preview_plot("Replication profile")
                raw_output_controls(file_name="dnascape_plot_data", default_type="CSV")
