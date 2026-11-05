"""Consistent colour-picker defaults for settings and plot controls."""

from typing import Any

from nicegui.element import Element


_PALETTE_BASES = ("#b03412", "#c47a1b", "#2c7f77", "#2596be", "#6f5a9c", "#6b6661")
DNASCAPE_PALETTE = tuple(
    "#" + "".join(
        f"{round(channel + (255 - channel) * shade if shade >= 0 else channel * (1 + shade)):02x}"
        for channel in (int(base[index:index + 2], 16) for index in (1, 3, 5))
    )
    for shade in (.75, .5, .25, 0., -.2, -.4)
    for base in _PALETTE_BASES
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
