"""Application accents and recoloured versions of the original SVG artwork."""

from __future__ import annotations

import base64
import colorsys
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from nicegui import ui

from app.export_settings import get_export_settings

DEFAULT_THEME_COLOR = "#b03412"
THEME_SWATCHES = (DEFAULT_THEME_COLOR, "#2563eb", "#0f766e", "#6d28d9", "#be185d", "#b7791f")
_ASSETS = Path(__file__).resolve().parents[1] / "docs" / "source" / "_static"
_LOGOS = {"dnascape_logo.svg", "dnascape_logo_small.svg", "dnascape_logo_tab.svg"}


def normalize_theme_color(value: Any) -> str:
    color = str(value or "").strip().lower()
    if re.fullmatch(r"#[0-9a-f]{3}", color):
        color = "#" + "".join(character * 2 for character in color[1:])
    if not re.fullmatch(r"#[0-9a-f]{6}", color):
        raise ValueError("Choose a valid theme colour, such as #2563eb.")
    return color


def _rgb(color: str) -> tuple[float, float, float]:
    return (int(color[1:3], 16) / 255, int(color[3:5], 16) / 255, int(color[5:7], 16) / 255)


def _hex(values: tuple[float, float, float]) -> str:
    return "#" + "".join(f"{round(value * 255):02x}" for value in values)


def contrast_ratio(first: str, second: str) -> float:
    def luminance(color: str) -> float:
        channels = [value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4 for value in _rgb(color)]
        return sum(weight * channel for weight, channel in zip((0.2126, 0.7152, 0.0722), channels))

    low, high = sorted((luminance(first), luminance(second)))
    return (high + 0.05) / (low + 0.05)


def _readable(color: str, background: str, *, lighten: bool) -> str:
    hue, lightness, saturation = colorsys.rgb_to_hls(*_rgb(color))
    for step in range(101):
        amount = step / 100
        level = lightness + (1 - lightness) * amount if lighten else lightness * (1 - amount)
        candidate = _hex(colorsys.hls_to_rgb(hue, level, saturation))
        if contrast_ratio(candidate, background) >= 4.5:
            return candidate
    return "#ffffff" if lighten else "#000000"


def _foreground(color: str) -> str:
    return max(("#ffffff", "#000000"), key=lambda text: contrast_ratio(text, color))


@lru_cache(maxsize=128)
def theme_css(color: str) -> dict[str, str]:
    base = normalize_theme_color(color)
    light = _readable(base, "#ffffff", lighten=False)
    if base == DEFAULT_THEME_COLOR:
        dark = "#da8068"
    else:
        hue, lightness, saturation = colorsys.rgb_to_hls(*_rgb(base))
        dark = _readable(_hex(colorsys.hls_to_rgb(hue, lightness + (1 - lightness) * 0.28, saturation)), "#211f1d", lighten=True)
    return {
        "--dn-theme-base": base,
        "--dn-theme-light": light,
        "--dn-theme-dark": dark,
        "--dn-theme-text-light": _foreground(light),
        "--dn-theme-text-dark": _foreground(dark),
        "--dn-action-text": _foreground(base),
        "--q-primary": "var(--dn-accent)",
    }


@lru_cache(maxsize=256)
def logo_sources(asset: str, color: str) -> tuple[str, str]:
    if asset not in _LOGOS:
        raise ValueError("Unknown DNAscape logo.")
    source = (_ASSETS / asset).read_text(encoding="utf-8")
    palette = theme_css(color)
    images = []
    for accent, neutral in ((palette["--dn-theme-light"], "#403932"), (palette["--dn-theme-dark"], "#f3eee8")):
        svg = re.sub(r"#b03412\b", accent, source, flags=re.IGNORECASE)
        svg = re.sub(r"#403932\b", neutral, svg, flags=re.IGNORECASE)
        images.append("data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("ascii"))
    return images[0], images[1]


def update_logo(element: Any, asset: str, color: str) -> None:
    light, dark = logo_sources(asset, color)
    element._props.update({"data-dn-logo": asset, "data-dn-logo-light": light, "data-dn-logo-dark": dark})
    element.update()


def brand_logo(asset: str, *, alt: str = "DNAscape") -> Any:
    light, _dark = logo_sources(asset, get_export_settings().theme_color)
    logo = ui.element("img").props("draggable=false")
    logo._props.update({"src": light, "alt": alt})
    update_logo(logo, asset, get_export_settings().theme_color)
    return logo
