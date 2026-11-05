"""Consistent colour-picker defaults for settings and plot controls."""

from typing import Any

from nicegui.element import Element


DNASCAPE_PALETTE = (
    "#b03412", "#d08427", "#b79b32",
    "#5f873f", "#267d74", "#2596be",
    "#7460a6", "#ad557e", "#586978",
)


class _ColorPicker(Element, component="static/color_picker.js"):
    """Native colour views with a shared DNAscape palette tab."""


def configure_color_picker(control: Any) -> None:
    """Use the same three views and colour choices in every colour control."""
    control.picker.q_color.delete()
    with control.picker:
        picker = _ColorPicker().props("format-model=hex default-view=palette")
    picker._props.update({
        "model-value": control.value,
        "brand-palette": DNASCAPE_PALETTE,
        "brand-icon": "/dnascape-assets/dnascape_logo_tab.svg",
    })
    picker.on("change", lambda event: control.set_value(event.args))
    control.picker.q_color = picker

    def sync_color() -> None:
        picker._props["model-value"] = control.value
        picker.update()

    control.on_value_change(sync_color)
