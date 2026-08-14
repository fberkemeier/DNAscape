"""Simulation workflow interface."""

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
    source_input,
    stat_strip,
    two_column_fields,
)


def render(_navigate: Callable[[str], None]) -> None:
    """Render simulation inputs and outputs without executing an ensemble."""

    page_intro(
        "Simulation",
        "Simulate DNA replication",
        "Simulate stochastic replication dynamics, vary kinetic assumptions, and preview DNAscape-ready outputs.",
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
                    with ui.element("div").classes("dn-source-grid"):
                        source_input(
                            "Firing rate",
                            default_mode="Profile",
                            constant=0.01,
                            units="min⁻¹ per bin",
                        )
                        source_input(
                            "Fork speed",
                            default_mode="Constant",
                            constant=1.4,
                            units="kb/min",
                        )

                with expansion("Region", icon="straighten"):
                    with two_column_fields():
                        ui.number(label="Resolution (kb/bin)", value=1.0, min=0.000001, step=0.1).classes(
                            "dn-field"
                        )
                        ui.number(label="Time step (min/bin)", value=5.0, min=0.000001, step=1.0).classes(
                            "dn-field"
                        )
                        ui.number(label="Start (kb)", value=0.0, min=0.0, step=1.0).classes("dn-field")
                        ui.number(label="End (kb, exclusive)", value=1000.0, min=0.000001, step=1.0).classes(
                            "dn-field"
                        )

                with expansion("Simulation settings", icon="tune"):
                    with two_column_fields():
                        ui.number(label="Replicates", value=20, min=1, step=1).classes("dn-field")
                        ui.select(
                            ["linear", "periodic"], value="linear", label="Geometry"
                        ).classes("dn-field")
                        ui.number(label="Time window (min)", value=300.0, min=0.000001, step=10.0).classes(
                            "dn-field"
                        )
                        ui.number(
                            label="Max replication time (min)", value=1200.0, min=0.000001, step=50.0
                        ).classes("dn-field")
                        ui.select(
                            ["error", "interpolate", "zero"], value="error", label="Missing values"
                        ).classes("dn-field")
                    fixed_seed = ui.checkbox("Fixed seed", value=True)
                    seed = (
                        ui.number(label="Seed", value=1, min=0, step=1)
                        .classes("dn-field")
                        .mark("simulation-seed")
                    )
                    seed.bind_visibility_from(fixed_seed, "value")
                    with two_column_fields():
                        ui.number(label="Stall rate", value=0.0, min=0.0, max=1.0, step=0.001).classes(
                            "dn-field"
                        )
                    non_recovering = ui.checkbox("Non-recovering stalls", value=True)
                    stall_duration = ui.number(
                        label="Stall duration (min)", value=10.0, min=0.000001, step=1.0
                    ).classes("dn-field").mark("simulation-stall-duration")
                    stall_duration.bind_visibility_from(
                        non_recovering,
                        "value",
                        backward=lambda value: not value,
                    )
                    finite_pool = ui.checkbox("Finite firing-factor pool", value=False)
                    with ui.column().classes("w-full gap-2").mark("simulation-pool-fields") as pool_fields:
                        with two_column_fields():
                            ui.number(label="Pool size", value=100, min=2, step=1).classes("dn-field")
                            ui.number(label="Recycle rate", value=0.0, min=0.0, step=0.001).classes(
                                "dn-field"
                            )
                        ui.checkbox("Couple firing factors to active forks", value=False)
                    pool_fields.bind_visibility_from(finite_pool, "value")

                with expansion("Plotting settings", icon="palette"):
                    with two_column_fields():
                        ui.select(
                            [
                                "Replication timing",
                                "Origin efficiency",
                                "Fork directionality",
                                "Replicated fraction",
                                "Total forks",
                                "Active forks",
                                "Stalled forks",
                                "Firing factors",
                                "Replicated fraction (space-time)",
                                "Initiation events (space-time)",
                                "Coalescence events (space-time)",
                                "Right fork density (space-time)",
                                "Left fork density (space-time)",
                            ],
                            value="Replication timing",
                            label="Output",
                        ).classes("dn-field")
                        ui.select(
                            ["Single view", "Multiple view", "Histogram", "Time array"],
                            value="Single view",
                            label="View",
                        ).classes("dn-field")
                        ui.input(label="Title", value="Simulation output").classes("dn-field")
                        ui.input(label="Label", value="Mean").classes("dn-field")
                        ui.input(label="X-axis label", value="Position (kb)").classes("dn-field")
                        ui.input(label="Y-axis label", value="Value").classes("dn-field")
                        ui.input(label="Line / bar color", value="#2c7f77").props("type=color").classes(
                            "dn-field"
                        )
                        ui.select(
                            ["viridis", "magma", "plasma", "cividis", "coolwarm"],
                            value="viridis",
                            label="Heatmap palette",
                        ).classes("dn-field")
                        ui.number(label="Histogram bins", value=60, min=1, max=2000, step=1).classes(
                            "dn-field"
                        )
                        ui.number(label="Width (in)", value=8.0, min=1.0, step=0.5).classes("dn-field")
                        ui.number(label="Height (in)", value=4.5, min=1.0, step=0.5).classes("dn-field")
                    show_legend = ui.checkbox("Legend", value=True)
                    legend_position = ui.select(
                        ["best", "upper right", "upper left", "lower right", "lower left"],
                        value="best",
                        label="Legend position",
                    ).classes("dn-field")
                    legend_position.bind_visibility_from(show_legend, "value")
                    ui.checkbox("Density", value=False)
                    ui.checkbox("Grid", value=True)
                    ui.checkbox("Colorbar", value=True)
                    ui.checkbox("Log Y", value=False)
                    ui.checkbox("Invert Y", value=False)
                    set_y_limits = ui.checkbox("Y limits", value=False)
                    with two_column_fields() as y_limit_fields:
                        ui.number(label="Y minimum", value=0.0, step=0.1).classes("dn-field")
                        ui.number(label="Y maximum", value=1.0, step=0.1).classes("dn-field")
                    y_limit_fields.bind_visibility_from(set_y_limits, "value")

                muted("Ensemble size, memory estimate, and progress will appear here after integration.")
            inert_button("Simulate", icon="play_arrow", primary=True)

        with ui.column().classes("dn-preview-stack"):
            with panel(extra_classes="dn-preview-card"):
                panel_title("Preview")
                stat_strip(
                    (
                        ("Replicates", "20"),
                        ("Region", "0–1,000 kb"),
                        ("Geometry", "Linear"),
                        ("Status", "Not run"),
                    )
                )
                preview_plot("Mean replication timing", kind="simulation")
                raw_output_controls(file_name="dnascape_simulation", default_type="NPZ")
