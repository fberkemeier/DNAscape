"""Theme colours stay readable and preserve the original logo geometry."""

import base64
from xml.etree import ElementTree

import pytest

from app.theme import DEFAULT_THEME_COLOR, contrast_ratio, logo_sources, normalize_theme_color, theme_css


@pytest.mark.parametrize(("value", "expected"), (("#2563EB", "#2563eb"), (" #abc ", "#aabbcc"), ("#000000", "#000000")))
def test_theme_color_normalizes_hex(value: str, expected: str) -> None:
    assert normalize_theme_color(value) == expected


@pytest.mark.parametrize("value", (None, "", "red", "#12345", "#12345678", "#gggggg", "url(example)"))
def test_invalid_theme_colors_are_rejected(value) -> None:
    with pytest.raises(ValueError, match="valid theme colour"):
        normalize_theme_color(value)


@pytest.mark.parametrize("color", (DEFAULT_THEME_COLOR, "#2563eb", "#0f766e", "#ffffff", "#000000", "#ffff00", "#be185d"))
def test_theme_accents_and_action_labels_remain_readable(color: str) -> None:
    values = theme_css(color)
    assert values["--dn-theme-base"] == color
    assert contrast_ratio(values["--dn-theme-light"], "#ffffff") >= 4.5
    assert contrast_ratio(values["--dn-theme-dark"], "#211f1d") >= 4.5
    assert contrast_ratio(values["--dn-action-text"], color) >= 4.5
    assert contrast_ratio(values["--dn-theme-text-light"], values["--dn-theme-light"]) >= 4.5
    assert contrast_ratio(values["--dn-theme-text-dark"], values["--dn-theme-dark"]) >= 4.5


@pytest.mark.parametrize("asset", ("dnascape_logo.svg", "dnascape_logo_small.svg", "dnascape_logo_tab.svg"))
def test_logos_recolour_without_changing_paths_or_dimensions(asset: str) -> None:
    original = logo_sources(asset, DEFAULT_THEME_COLOR)
    themed = logo_sources(asset, "#2563eb")
    palette = theme_css("#2563eb")
    for index, (before, after) in enumerate(zip(original, themed)):
        old_svg = base64.b64decode(before.split(",", 1)[1]).decode()
        new_svg = base64.b64decode(after.split(",", 1)[1]).decode()
        old_root, new_root = ElementTree.fromstring(old_svg), ElementTree.fromstring(new_svg)
        assert old_root.attrib == new_root.attrib
        assert [element.attrib.get("d") for element in old_root.iter()] == [element.attrib.get("d") for element in new_root.iter()]
        assert palette["--dn-theme-light" if index == 0 else "--dn-theme-dark"] in new_svg
        assert ("#403932" if index == 0 else "#f3eee8") in new_svg
        assert "#b03412" not in new_svg


def test_default_theme_preserves_both_brand_accents() -> None:
    palette = theme_css(DEFAULT_THEME_COLOR)
    assert palette["--dn-theme-light"] == "#b03412"
    assert palette["--dn-theme-dark"] == "#da8068"
