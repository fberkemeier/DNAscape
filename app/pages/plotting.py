"""Plotting workspace interface prototype."""

from __future__ import annotations

from collections.abc import Callable

from nicegui import ui

from app.pages.common import (
    DATA_ICON,
    action_dock,
    data_upload,
    expansion,
    muted,
    page_intro,
    panel,
    panel_title,
    plot_action_button,
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
                    with ui.element("div").classes("dn-source-compact dn-plot-data-source"):
                        with ui.element("div").classes("dn-source-head"):
                            ui.label("Plot data").classes("dn-field-heading")
                            source_mode = (
                                ui.select(
                                    ["Prepared data", "Upload file"],
                                    value="Prepared data",
                                    label="Input type",
                                )
                                .props("dense options-dense")
                                .classes("dn-source-mode")
                                .mark("plotting-source-mode")
                            )
                        with (
                            ui.element("div")
                            .classes("dn-source-fields")
                            .mark("plotting-prepared-fields")
                        ) as prepared_fields:
                            ui.select(
                                [
                                    "Replication profile",
                                    "Mapping result",
                                    "Simulation result",
                                ],
                                value="Replication profile",
                                label="Data source",
                            ).props("dense options-dense").classes("dn-field")
                        with (
                            ui.element("div")
                            .classes("dn-source-fields dn-source-profile-fields")
                            .mark("plotting-upload-fields")
                        ) as upload_fields:
                            data_upload(
                                accept=".csv,.tsv,.txt,.npy,.npz,.gz",
                                accepted_formats="CSV, TSV, TXT, NPY, NPZ, GZ",
                                marker="plotting-data-upload",
                            )
                        ui.select(
                            ["value", "replication_timing", "firing_rate"],
                            value="value",
                            label="Profile column",
                        ).props("dense options-dense").classes("dn-field")
                        muted("Replication profile · 1,000 values · space")
                    prepared_fields.bind_visibility_from(
                        source_mode,
                        "value",
                        backward=lambda value: value == "Prepared data",
                    )
                    upload_fields.bind_visibility_from(
                        source_mode,
                        "value",
                        backward=lambda value: value == "Upload file",
                    )

                with expansion("Region", icon="straighten"):
                    with two_column_fields():
                        ui.number(label="Start index (inclusive)", value=0, min=0, step=1).props(
                            "dense"
                        ).classes("dn-field")
                        ui.number(label="End index (exclusive)", value=1000, min=1, step=1).props(
                            "dense"
                        ).classes("dn-field")
                        ui.number(label="First time / feature index", value=0, min=0, step=1).props(
                            "dense"
                        ).classes("dn-field")
                        ui.number(label="Last time / feature index", value=60, min=1, step=1).props(
                            "dense"
                        ).classes("dn-field")
                        ui.number(
                            label="Resolution (kb/bin)", value=1.0, min=0.000001, step=0.1
                        ).props("dense").classes("dn-field")
                        ui.number(label="Time step (min)", value=1.0, min=0.000001, step=0.1).props(
                            "dense"
                        ).classes("dn-field")

                with expansion("Plotting settings", icon="palette"):
                    with two_column_fields():
                        ui.select(
                            ["value", "replication_timing", "firing_rate"],
                            value="value",
                            label="Output",
                        ).props("dense options-dense").classes("dn-field")
                        plot_type = (
                            ui.select(
                                ["Line", "Heatmap", "Histogram"], value="Line", label="Plot type"
                            )
                            .props("dense options-dense")
                            .classes("dn-field")
                            .mark("plotting-plot-type")
                        )
                        comparison_source = (
                            ui.select(
                                ["None", "Simulation result", "Mapping result"],
                                value="None",
                                label="Optional comparison line",
                            )
                            .props("dense options-dense")
                            .classes("dn-field")
                            .mark("plotting-comparison-source")
                        )
                        comparison_color = (
                            ui.input(label="Comparison color", value="#d97757")
                            .props("type=color dense")
                            .classes("dn-field")
                            .mark("plotting-comparison-color")
                        )
                    ui.input(label="Title", value="Replication profile").props("dense").classes(
                        "dn-field"
                    )
                    with two_column_fields():
                        ui.input(label="X-axis label", value="Position (kb)").props(
                            "dense"
                        ).classes("dn-field")
                        ui.input(label="Y-axis label", value="Value").props("dense").classes(
                            "dn-field"
                        )
                        ui.select(
                            ["None", "Z-score", "Min-max [0,1]"],
                            value="None",
                            label="Presentation normalization",
                        ).props("dense options-dense").classes("dn-field")
                        line_or_bar_color = (
                            ui.input(label="Line / bar color", value="#2c7f77")
                            .props("type=color dense")
                            .classes("dn-field")
                            .mark("plotting-line-or-bar-color")
                        )
                        heatmap_palette = (
                            ui.select(
                                ["viridis", "magma", "plasma", "cividis", "coolwarm"],
                                value="viridis",
                                label="Heatmap palette",
                            )
                            .props("dense options-dense")
                            .classes("dn-field")
                            .mark("plotting-heatmap-palette")
                        )
                        line_width = (
                            ui.number(label="Line width", value=1.8, min=0.2, max=8.0, step=0.1)
                            .props("dense")
                            .classes("dn-field")
                            .mark("plotting-line-width")
                        )
                        histogram_bins = (
                            ui.number(label="Histogram bins", value=60, min=1, max=2000, step=1)
                            .props("dense")
                            .classes("dn-field")
                            .mark("plotting-histogram-bins")
                        )
                        colorbar_label = (
                            ui.input(label="Colorbar label", value="Value")
                            .props("dense")
                            .classes("dn-field")
                            .mark("plotting-colorbar-label")
                        )
                    with ui.element("div").classes("dn-inline-actions"):
                        colorbar = ui.checkbox("Colorbar", value=True).mark("plotting-colorbar")
                        ui.checkbox("Log Y", value=False)
                        ui.checkbox("Invert Y", value=False)
                    with two_column_fields():
                        ui.input(label="X limits (min,max; optional)").props("dense").classes(
                            "dn-field"
                        )
                        ui.input(label="Y limits (min,max; optional)").props("dense").classes(
                            "dn-field"
                        )
                        heatmap_minimum = (
                            ui.input(label="Heatmap minimum (optional)")
                            .props("dense")
                            .classes("dn-field")
                            .mark("plotting-heatmap-minimum")
                        )
                        heatmap_maximum = (
                            ui.input(label="Heatmap maximum (optional)")
                            .props("dense")
                            .classes("dn-field")
                            .mark("plotting-heatmap-maximum")
                        )
                    comparison_color.bind_visibility_from(
                        comparison_source, "value", backward=lambda value: value != "None"
                    )
                    line_width.bind_visibility_from(
                        plot_type, "value", backward=lambda value: value == "Line"
                    )
                    line_or_bar_color.bind_visibility_from(
                        plot_type, "value", backward=lambda value: value != "Heatmap"
                    )
                    histogram_bins.bind_visibility_from(
                        plot_type, "value", backward=lambda value: value == "Histogram"
                    )
                    for heatmap_control in (
                        heatmap_palette,
                        colorbar_label,
                        colorbar,
                        heatmap_minimum,
                        heatmap_maximum,
                    ):
                        heatmap_control.bind_visibility_from(
                            plot_type, "value", backward=lambda value: value == "Heatmap"
                        )
                with action_dock():
                    plot_action_button()

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
