"""Mapping workflow interface."""

from __future__ import annotations

from collections.abc import Callable

from nicegui import ui

from app.pages.common import (
    DATA_ICON,
    action_dock,
    expansion,
    inert_button,
    muted,
    page_intro,
    panel,
    panel_title,
    preview_plot,
    raw_output_controls,
    source_input,
    stat_strip,
    three_column_fields,
    two_column_fields,
)

MAPPING_OPERATIONS = (
    "Infer firing rate from timing",
    "Infer fork speed from timing",
    "Predict replication timing",
    "Calculate fork directionality",
    "Reconstruct timing from directionality",
    "Map replicated fraction",
    "Map replicated fraction in space-time",
    "Map initiation events",
    "Map initiation events in space-time",
    "Map coalescence events",
    "Map coalescence events in space-time",
    "Map right-moving fork density",
    "Map left-moving fork density",
    "Map total fork density",
)


def render(_navigate: Callable[[str], None]) -> None:
    """Render mapping routes, inputs, configuration, and output placeholders."""

    page_intro(
        "Mapping",
        "Map replication kinetic landscapes",
        "Choose a target, set a precise mapping route, and preview mapped DNAscape outputs.",
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
                    ui.select(
                        list(MAPPING_OPERATIONS),
                        value=MAPPING_OPERATIONS[0],
                        label="Mapping operation",
                    ).props("dense options-dense").classes("dn-field")
                    muted("Select how each required input is supplied.")
                    with ui.element("div").classes("dn-source-grid dn-data-fields"):
                        source_input(
                            "Replication timing", default_mode="Profile", constant=30.0, units="min"
                        )
                        source_input(
                            "Firing rate",
                            default_mode="Profile",
                            constant=0.001,
                            units="min⁻¹ per bin",
                        )
                        source_input(
                            "Fork speed", default_mode="Constant", constant=1.4, units="kb/min"
                        )
                        source_input(
                            "Fork directionality",
                            default_mode="Profile",
                            constant=0.0,
                            units="fraction",
                        )

                with expansion("Region", icon="straighten"):
                    with two_column_fields():
                        ui.number(
                            label="Spatial resolution (kb)", value=1.0, min=0.000001, step=0.1
                        ).props("dense").classes("dn-field")
                        ui.select(
                            ["error", "interpolate", "zero"], value="error", label="Missing values"
                        ).props("dense options-dense").classes("dn-field")
                        ui.number(label="Region start (kb)", value=0.0, min=0.0, step=1.0).props(
                            "dense"
                        ).classes("dn-field")
                        ui.number(
                            label="Region end (kb, exclusive)", value=1000.0, min=0.000001, step=1.0
                        ).props("dense").classes("dn-field")
                        ui.number(label="Time step (min)", value=5.0, min=0.000001, step=1.0).props(
                            "dense"
                        ).classes("dn-field")
                        ui.number(
                            label="Time window (min)", value=300.0, min=0.000001, step=10.0
                        ).props("dense").classes("dn-field")
                    ui.checkbox("Override the package-derived fit time grid", value=False)

                with expansion("Mapping settings", icon="tune"):
                    with two_column_fields():
                        ui.select(
                            ["space", "time", "spacetime"], value="space", label="Target form"
                        ).props("dense options-dense").classes("dn-field")
                        ui.select(
                            ["weibull", "free", "best"],
                            value="weibull",
                            label="Temporal target shape",
                        ).props("dense options-dense").classes("dn-field")
                        ui.number(label="Fit iterations", value=10, min=0, step=1).props(
                            "dense"
                        ).classes("dn-field")
                    override_fit_step = ui.checkbox("Override route-specific fit step", value=False)
                    fit_step = (
                        ui.number(label="Fit step", value=0.1, min=0.000000001, step=0.01)
                        .props("dense")
                        .classes("dn-field")
                    )
                    fit_step.bind_visibility_from(override_fit_step, "value")
                    with ui.element("div").classes("dn-inline-actions"):
                        ui.checkbox("Periodic / circular boundary", value=False)
                        use_mse_threshold = ui.checkbox("MSE early stop", value=False)
                        override_neighbourhood = ui.checkbox("Fit neighbourhood", value=False)
                    with two_column_fields():
                        mse_threshold = (
                            ui.number(label="MSE threshold", value=0.001, min=0.0, step=0.001)
                            .props("dense")
                            .classes("dn-field")
                        )
                        neighbourhood_range = (
                            ui.number(
                                label="Neighbourhood range (kb)", value=2000.0, min=0.0, step=100.0
                            )
                            .props("dense")
                            .classes("dn-field")
                        )
                        ui.number(
                            label="Firing floor exponent", value=15.0, min=0.0, step=1.0
                        ).props("dense").classes("dn-field")
                        ui.number(
                            label="Initial fork speed (kb/min)", value=1.4, min=0.000001, step=0.1
                        ).props("dense").classes("dn-field")
                        ui.number(label="Temporal smoothness", value=0.0, min=0.0, step=0.01).props(
                            "dense"
                        ).classes("dn-field")
                        ui.number(label="Temporal prior", value=0.0, min=0.0, step=0.01).props(
                            "dense"
                        ).classes("dn-field")
                    mse_threshold.bind_visibility_from(use_mse_threshold, "value")
                    neighbourhood_range.bind_visibility_from(override_neighbourhood, "value")
                    with three_column_fields():
                        ui.number(label="Weibull shape", value=2.0, min=0.000001, step=0.1).props(
                            "dense"
                        ).classes("dn-field")
                        ui.number(label="Weibull delay (min)", value=0.0, min=0.0, step=1.0).props(
                            "dense"
                        ).classes("dn-field")
                        ui.checkbox("Fit shape", value=False)
                    with two_column_fields():
                        ui.select(
                            ["factorized", "mixed"], value="factorized", label="Space-time update"
                        ).props("dense options-dense").classes("dn-field")
                        ui.select(
                            ["space_then_time", "time_then_space", "crossed"],
                            value="space_then_time",
                            label="Factorized phase order",
                        ).props("dense options-dense").classes("dn-field")
                    with ui.element("div").classes("dn-inline-actions"):
                        set_timing_anchor = ui.checkbox("Timing anchor T0", value=False)
                        smooth_timing = ui.checkbox("Smooth timing", value=False)
                    with two_column_fields():
                        timing_anchor = (
                            ui.number(label="Timing anchor T0 (min)", value=0.0, step=1.0)
                            .props("dense")
                            .classes("dn-field")
                        )
                        ui.number(
                            label="CFL factor", value=0.9, min=0.01, max=1.0, step=0.01
                        ).props("dense").classes("dn-field")
                    timing_anchor.bind_visibility_from(set_timing_anchor, "value")
                    smoothing_window = (
                        ui.number(label="Smoothing window (bins)", value=5, min=1, step=1)
                        .props("dense")
                        .classes("dn-field")
                    )
                    smoothing_window.bind_visibility_from(smooth_timing, "value")

                with expansion("Plotting settings", icon="palette"):
                    with two_column_fields():
                        ui.select(
                            [
                                "Mapped firing rate",
                                "Predicted timing",
                                "Fork directionality",
                                "Replicated fraction",
                                "Initiation events",
                                "Coalescence events",
                            ],
                            value="Mapped firing rate",
                            label="Output",
                        ).props("dense options-dense").classes("dn-field")
                        plot_type = (
                            ui.select(
                                ["Line", "Heatmap", "Histogram"],
                                value="Line",
                                label="Plot type",
                            )
                            .props("dense options-dense")
                            .classes("dn-field")
                            .mark("mapping-plot-type")
                        )
                        ui.input(label="Title", value="Mapped output").props("dense").classes(
                            "dn-field"
                        )
                        ui.input(label="X-axis label", value="Position (kb)").props(
                            "dense"
                        ).classes("dn-field")
                        ui.input(label="Y-axis label", value="Value").props("dense").classes(
                            "dn-field"
                        )
                        line_or_bar_color = (
                            ui.input(label="Line / bar color", value="#2c7f77")
                            .props("type=color dense")
                            .classes("dn-field")
                            .mark("mapping-line-or-bar-color")
                        )
                        heatmap_palette = (
                            ui.select(
                                ["viridis", "magma", "plasma", "cividis", "coolwarm"],
                                value="viridis",
                                label="Heatmap palette",
                            )
                            .props("dense options-dense")
                            .classes("dn-field")
                            .mark("mapping-heatmap-palette")
                        )
                        histogram_bins = (
                            ui.number(label="Histogram bins", value=60, min=1, max=2000, step=1)
                            .props("dense")
                            .classes("dn-field")
                            .mark("mapping-histogram-bins")
                        )
                    with ui.element("div").classes("dn-inline-actions"):
                        ui.checkbox("Overlay compatible input", value=False)
                        colorbar = ui.checkbox("Colorbar", value=True).mark("mapping-colorbar")
                        ui.checkbox("Log Y", value=False)
                        ui.checkbox("Invert Y", value=False)
                    heatmap_palette.bind_visibility_from(
                        plot_type, "value", backward=lambda value: value == "Heatmap"
                    )
                    colorbar.bind_visibility_from(
                        plot_type, "value", backward=lambda value: value == "Heatmap"
                    )
                    histogram_bins.bind_visibility_from(
                        plot_type, "value", backward=lambda value: value == "Histogram"
                    )
                    line_or_bar_color.bind_visibility_from(
                        plot_type, "value", backward=lambda value: value != "Heatmap"
                    )

                with action_dock():
                    inert_button("Map", icon="map", primary=True)

        with ui.column().classes("dn-preview-stack"):
            with panel(extra_classes="dn-preview-card"):
                panel_title("Preview")
                stat_strip(
                    (
                        ("Target", "Firing rate"),
                        ("Form", "Space"),
                        ("Bins", "1,000"),
                        ("Status", "Not run"),
                    )
                )
                preview_plot("Mapped firing landscape", kind="mapping")
                raw_output_controls(file_name="dnascape_mapping", default_type="CSV")
